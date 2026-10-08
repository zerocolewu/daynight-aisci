"""
5-Fold Cross-Validation following PerioGT paper protocol

Paper details:
- 5-fold cross-validation with 80/10/10 split
- 50 epochs per fold
- Best model selected by lowest validation RMSE
- Report mean ± std across 5 folds
- Architecture: Chemprop-style MPNN with directed bond-level messages
- Hidden dim: 300, Message passing steps: 3
- FFN: 2 layers, 300 neurons each
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import Adam
from torch_geometric.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
import json
from datetime import datetime

# Import from local modules
from dataset import (
    load_and_split_data,
    mol_to_periodic_graph,
    get_feature_dimensions,
    PolymerDataset,
)
from model import get_model, count_parameters


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred, _ = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.periodic_mask if hasattr(batch, 'periodic_mask') else None,
            batch.periodic_vectors if hasattr(batch, 'periodic_vectors') else None,
        )

        loss = F.mse_loss(pred.squeeze(), batch.y.squeeze())
        loss.backward()

        # Gradient clipping (stabilizes training)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, loader, device):
    """Evaluate model and return predictions and targets."""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, _ = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                batch.periodic_mask if hasattr(batch, 'periodic_mask') else None,
                batch.periodic_vectors if hasattr(batch, 'periodic_vectors') else None,
            )

            predictions.extend(pred.squeeze().cpu().numpy())
            targets.extend(batch.y.squeeze().cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    # R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return rmse, mae, r2, predictions, targets


def run_single_fold(
    train_data,
    val_data,
    test_data,
    fold_idx,
    model_type='chemprop',
    hidden_dim=300,
    num_layers=3,
    dropout=0.0,
    lr=1e-4,
    weight_decay=1e-5,
    epochs=50,
    batch_size=32,
    patience=20,
    device='cuda',
    seed=42,
):
    """
    Run training and evaluation for a single fold.

    Follows paper protocol:
    - 50 epochs
    - Select best model by lowest validation RMSE
    - Early stopping with patience
    """
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}")
    print(f"{'='*60}")

    # Set random seed for reproducibility
    torch.manual_seed(seed + fold_idx)
    np.random.seed(seed + fold_idx)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Get feature dimensions
    atom_dim, bond_dim = get_feature_dimensions()

    # Create model
    model = get_model(
        model_type=model_type,
        atom_dim=atom_dim,
        bond_dim=bond_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    print(f"Model: {model_type}, Parameters: {count_parameters(model):,}")

    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    best_val_rmse = float('inf')
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0

    train_rmses = []
    val_rmses = []

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Evaluate
        train_rmse, train_mae, train_r2, _, _ = evaluate(model, train_loader, device)
        val_rmse, val_mae, val_r2, _, _ = evaluate(model, val_loader, device)

        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)

        # Track best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train RMSE={train_rmse:.2f}, Val RMSE={val_rmse:.2f} (Best: {best_val_rmse:.2f} @ epoch {best_epoch})")

    print(f"\nBest validation RMSE: {best_val_rmse:.2f} at epoch {best_epoch}")

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    model.to(device)

    test_rmse, test_mae, test_r2, test_preds, test_targets = evaluate(model, test_loader, device)
    train_rmse_final, train_mae_final, train_r2_final, _, _ = evaluate(model, train_loader, device)
    val_rmse_final, val_mae_final, val_r2_final, _, _ = evaluate(model, val_loader, device)

    print(f"\nFinal Results (Best Model @ Epoch {best_epoch}):")
    print(f"  Train: RMSE={train_rmse_final:.2f}, MAE={train_mae_final:.2f}, R²={train_r2_final:.4f}")
    print(f"  Val:   RMSE={val_rmse_final:.2f}, MAE={val_mae_final:.2f}, R²={val_r2_final:.4f}")
    print(f"  Test:  RMSE={test_rmse:.2f}, MAE={test_mae:.2f}, R²={test_r2:.4f}")

    return {
        'fold': fold_idx + 1,
        'best_epoch': best_epoch,
        'train_rmse': train_rmse_final,
        'train_mae': train_mae_final,
        'train_r2': train_r2_final,
        'val_rmse': val_rmse_final,
        'val_mae': val_mae_final,
        'val_r2': val_r2_final,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'train_history': train_rmses,
        'val_history': val_rmses,
    }


def run_5fold_cv(
    csv_path,
    model_type='chemprop',
    hidden_dim=300,
    num_layers=3,
    dropout=0.0,
    lr=1e-4,
    weight_decay=1e-5,
    epochs=50,
    batch_size=32,
    patience=20,
    seed=42,
):
    """
    Run 5-fold cross-validation following paper protocol.

    Paper: 80/10/10 split, 50 epochs, best model by val RMSE
    """
    print(f"\n{'='*60}")
    print(f"5-Fold Cross-Validation: {model_type.upper()}")
    print(f"{'='*60}")
    print(f"Config: hidden_dim={hidden_dim}, layers={num_layers}, dropout={dropout}")
    print(f"        lr={lr}, weight_decay={weight_decay}, epochs={epochs}")
    print(f"{'='*60}\n")

    # Load all data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    # Convert to graphs
    print("Converting SMILES to periodic graphs...")
    data_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        data = mol_to_periodic_graph(
            row['Smiles'],
            row['Tg'],
            add_periodic_edge=True
        )
        if data is not None:
            data_list.append(data)

    print(f"Successfully parsed {len(data_list)} molecules")

    # 5-fold split
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    fold_results = []

    for fold_idx, (all_idx, test_idx) in enumerate(kf.split(data_list)):
        # Split all_idx into train and val (80/10/10 overall)
        # From the remaining 80%, use 8/9 for train and 1/9 for val
        all_data = [data_list[i] for i in all_idx]
        n_all = len(all_data)
        n_train = int(n_all * 8 / 9)  # 8/9 of 90% = 80% total

        np.random.seed(seed + fold_idx)
        indices = np.random.permutation(n_all)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_data = [all_data[i] for i in train_indices]
        val_data = [all_data[i] for i in val_indices]
        test_data = [data_list[i] for i in test_idx]

        print(f"\nFold {fold_idx + 1}: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

        # Run fold
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")

        result = run_single_fold(
            PolymerDataset(train_data),
            PolymerDataset(val_data),
            PolymerDataset(test_data),
            fold_idx,
            model_type=model_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            device=device,
            seed=seed,
        )

        fold_results.append(result)

    # Calculate mean and std across folds
    print(f"\n{'='*60}")
    print(f"5-Fold Cross-Validation Results")
    print(f"{'='*60}")

    metrics = ['train_rmse', 'val_rmse', 'test_rmse', 'test_mae', 'test_r2']

    summary = {}
    for metric in metrics:
        values = [r[metric] for r in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary[metric] = {'mean': mean_val, 'std': std_val}
        print(f"{metric:15s}: {mean_val:.2f} ± {std_val:.2f}")

    print(f"\n{'='*60}")
    print(f"Final Test RMSE: {summary['test_rmse']['mean']:.2f} ± {summary['test_rmse']['std']:.2f}")
    print(f"{'='*60}\n")

    # Save results
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv('cv_results.csv', index=False)
    print(f"Saved fold results to cv_results.csv")

    # Save summary
    with open('cv_summary.json', 'w') as f:
        json.dump({
            'model_type': model_type,
            'config': {
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': dropout,
                'lr': lr,
                'weight_decay': weight_decay,
                'epochs': epochs,
                'batch_size': batch_size,
                'seed': seed,
            },
            'summary': summary,
            'fold_results': fold_results,
        }, f, indent=2)
    print(f"Saved summary to cv_summary.json")

    return fold_results, summary


if __name__ == "__main__":
    # Dataset path
    csv_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'

    # Run 5-fold CV with Chemprop-style MPNN (paper baseline)
    print("\n" + "="*60)
    print("Running PerioGT 5-Fold Cross-Validation")
    print("Following paper protocol exactly:")
    print("  - Model: Chemprop-style MPNN")
    print("  - Hidden dim: 300")
    print("  - Message passing steps: 3")
    print("  - FFN: 2 layers, 300 neurons")
    print("  - Epochs: 50")
    print("  - Split: 80/10/10")
    print("  - Paper reported RMSE: 37.35 ± (not specified)")
    print("="*60 + "\n")

    fold_results, summary = run_5fold_cv(
        csv_path,
        model_type='chemprop',
        hidden_dim=300,
        num_layers=3,
        dropout=0.0,  # Paper doesn't mention dropout
        lr=1e-3,  # Chemprop default
        weight_decay=0.0,  # Chemprop default
        epochs=50,
        batch_size=50,
        patience=50,  # No early stopping in paper, but we'll track best
        seed=42,
    )

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Our Result: {summary['test_rmse']['mean']:.2f} ± {summary['test_rmse']['std']:.2f}")
    print(f"Paper Result: 37.35 (Tg, MPNN/periodic polymer graph)")
    print(f"Gap: {summary['test_rmse']['mean'] - 37.35:.2f}")
    print("="*60 + "\n")
