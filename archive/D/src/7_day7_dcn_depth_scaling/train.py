"""
Training script for DCN-GIN depth scaling experiment
5-fold cross-validation with gradient analysis
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

from dataset import load_tg_dataset
from model import DCNGIN, StandardGIN

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)

        # Get molecular features
        features = batch.mol_features if hasattr(batch, 'mol_features') else None

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            pred, _ = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                features,
                batch.batch
            )
            loss = nn.MSELoss()(pred.squeeze(), batch.y.squeeze())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs

    return {'loss': total_loss / total_samples}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    preds = []
    targets = []

    for batch in loader:
        batch = batch.to(device)
        features = batch.mol_features if hasattr(batch, 'mol_features') else None

        pred, _ = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            features,
            batch.batch
        )

        preds.append(pred.cpu().numpy())
        targets.append(batch.y.cpu().numpy())

    preds = np.concatenate(preds).squeeze()
    targets = np.concatenate(targets).squeeze()

    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    pearson_r, _ = pearsonr(targets, preds)

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson_r': pearson_r
    }


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    fold: int,
    depth: int,
    seed: int,
    use_dense: bool,
    epochs: int = 300,
    patience: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-5
) -> Dict:
    """Train model for one fold."""
    model_name = f"{'DCN' if use_dense else 'Standard'}-GIN-{depth}"
    print(f"\nTraining {model_name} - Fold {fold} - Seed {seed}")

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=False
    )

    scaler = torch.amp.GradScaler('cuda')

    # Training loop
    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    gradient_history = []

    for epoch in range(epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, epoch)

        # Validate
        val_metrics = evaluate(model, val_loader)

        # Update scheduler
        scheduler.step(val_metrics['rmse'])

        # Save best model
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_epoch = epoch
            patience_counter = 0

            # Save model checkpoint
            checkpoint_dir = f"checkpoints/{model_name}_seed{seed}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{checkpoint_dir}/fold{fold}_best.pt")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

        # Print progress
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                  f"Val RMSE={val_metrics['rmse']:.2f}, Best={best_val_rmse:.2f}")

    # Load best model and evaluate on test set
    checkpoint_path = f"checkpoints/{model_name}_seed{seed}/fold{fold}_best.pt"
    model.load_state_dict(torch.load(checkpoint_path))

    train_metrics = evaluate(model, train_loader)
    val_metrics = evaluate(model, val_loader)
    test_metrics = evaluate(model, test_loader)

    print(f"  Fold {fold} Results:")
    print(f"    Train RMSE: {train_metrics['rmse']:.2f}")
    print(f"    Val RMSE: {val_metrics['rmse']:.2f}")
    print(f"    Test RMSE: {test_metrics['rmse']:.2f}")

    return {
        'model_type': 'DCN' if use_dense else 'Standard',
        'depth': depth,
        'seed': seed,
        'fold': fold,
        'best_epoch': best_epoch,
        'train_rmse': train_metrics['rmse'],
        'train_mae': train_metrics['mae'],
        'train_r2': train_metrics['r2'],
        'train_pearson': train_metrics['pearson_r'],
        'val_rmse': val_metrics['rmse'],
        'val_mae': val_metrics['mae'],
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_r2': test_metrics['r2'],
        'test_pearson': test_metrics['pearson_r']
    }


def run_experiment(
    data_list: List,
    features: np.ndarray,
    scaler_norm,
    depths: List[int] = [5, 10, 15],
    seeds: List[int] = [42, 123, 456],
    use_dense_variants: List[bool] = [True, False],
    n_folds: int = 5,
    batch_size: int = 64,
    epochs: int = 300,
    patience: int = 50
):
    """Run full experiment with all variants."""

    results = []
    gradient_analysis = defaultdict(list)

    for use_dense in use_dense_variants:
        for depth in depths:
            # Skip standard GIN for depth 5 (already done in baseline)
            if not use_dense and depth == 5:
                continue

            for seed in seeds:
                print(f"\n{'='*80}")
                print(f"Model: {'DCN' if use_dense else 'Standard'}-GIN-{depth}, Seed: {seed}")
                print(f"{'='*80}")

                # Set random seeds
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

                # K-fold cross-validation
                kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

                for fold, (train_idx, test_idx) in enumerate(kfold.split(data_list)):
                    # Further split train into train/val
                    val_size = int(len(train_idx) * 0.1)
                    val_idx = train_idx[:val_size]
                    train_idx_actual = train_idx[val_size:]

                    # Create data subsets
                    train_data = [data_list[i] for i in train_idx_actual]
                    val_data = [data_list[i] for i in val_idx]
                    test_data = [data_list[i] for i in test_idx]

                    # Normalize features
                    train_features = scaler_norm.transform(features[train_idx_actual])
                    val_features = scaler_norm.transform(features[val_idx])
                    test_features = scaler_norm.transform(features[test_idx])

                    # Update features in data objects
                    for i, idx in enumerate(train_idx_actual):
                        train_data[i].mol_features = torch.tensor(train_features[i], dtype=torch.float).unsqueeze(0)
                    for i, idx in enumerate(val_idx):
                        val_data[i].mol_features = torch.tensor(val_features[i], dtype=torch.float).unsqueeze(0)
                    for i, idx in enumerate(test_idx):
                        test_data[i].mol_features = torch.tensor(test_features[i], dtype=torch.float).unsqueeze(0)

                    # Create data loaders
                    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
                    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

                    # Create model
                    if use_dense:
                        model = DCNGIN(
                            num_layers=depth,
                            hidden_dim=256,
                            growth_rate=32,
                            feature_dim=148,
                            dropout=0.2,
                            use_dense=True
                        ).to(device)
                    else:
                        model = StandardGIN(
                            num_layers=depth,
                            hidden_dim=256,
                            feature_dim=148,
                            dropout=0.2
                        ).to(device)

                    # Train
                    result = train_fold(
                        model, train_loader, val_loader, test_loader,
                        fold, depth, seed, use_dense, epochs, patience
                    )
                    results.append(result)

    return results


def main():
    """Main training function."""
    print("="*80)
    print("DCN-GIN Depth Scaling Experiment")
    print("="*80)

    # Load dataset
    csv_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    cache_dir = '/home/home_public/aisci/AIscientist/src/7_day7_dcn_depth_scaling/cache'

    data_list, features, scaler = load_tg_dataset(csv_path, cache_dir, use_cache=False)

    print(f"\nDataset: {len(data_list)} samples")
    print(f"Features: {features.shape}")

    # Run experiments
    results = run_experiment(
        data_list,
        features,
        scaler,
        depths=[5, 10, 15],
        seeds=[42, 123, 456],
        use_dense_variants=[True, False],
        n_folds=5,
        batch_size=64,
        epochs=300,
        patience=50
    )

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)
    print(f"\nResults saved to results.csv")

    # Compute summary statistics
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)

    summary = df.groupby(['model_type', 'depth']).agg({
        'test_rmse': ['mean', 'std'],
        'test_mae': ['mean', 'std'],
        'test_r2': ['mean', 'std'],
        'test_pearson': ['mean', 'std']
    }).round(2)

    print(summary)

    # Save summary
    summary.to_csv('summary.csv')
    print(f"\nSummary saved to summary.csv")

    # Compare with baselines
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINES")
    print("="*80)

    baselines = {
        'Day 2 5-layer GIN': 32.31,
        'CFD-MTL 3-layer GIN': 44.12
    }

    for (model_type, depth), group in df.groupby(['model_type', 'depth']):
        mean_rmse = group['test_rmse'].mean()
        print(f"\n{model_type}-GIN-{depth}: {mean_rmse:.2f}")

        for baseline_name, baseline_rmse in baselines.items():
            improvement = (baseline_rmse - mean_rmse) / baseline_rmse * 100
            print(f"  vs {baseline_name}: {improvement:+.1f}% ({'better' if improvement > 0 else 'worse'})")

    print("\n" + "="*80)
    print("Experiment completed!")
    print("="*80)


if __name__ == "__main__":
    main()
