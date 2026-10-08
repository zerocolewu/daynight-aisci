"""
Simplified DCN-GIN experiment - Quick validation
Runs DCN-GIN at different depths to test the hypothesis
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Import our modules
from dataset import load_tg_dataset
from model import DCNGIN, StandardGIN

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def quick_train(model, train_loader, val_loader, epochs=100, patience=20):
    """Quick training loop."""
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    scaler = torch.amp.GradScaler('cuda')

    best_val_rmse = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            features = batch.mol_features if hasattr(batch, 'mol_features') else None

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, features, batch.batch)
                loss = nn.MSELoss()(pred.squeeze(), batch.y.squeeze())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validate
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                features = batch.mol_features if hasattr(batch, 'mol_features') else None
                pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, features, batch.batch)
                preds.append(pred.cpu().numpy())
                targets.append(batch.y.cpu().numpy())

        preds = np.concatenate(preds).squeeze()
        targets = np.concatenate(targets).squeeze()
        val_rmse = np.sqrt(mean_squared_error(targets, preds))

        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: Val RMSE={val_rmse:.2f}")

    return best_val_rmse


def main():
    print("="*80)
    print("DCN-GIN Depth Scaling - Quick Validation")
    print("="*80)

    # Load dataset
    csv_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    cache_dir = '/home/home_public/aisci/AIscientist/src/7_day7_dcn_depth_scaling/cache'

    data_list, features, scaler_norm = load_tg_dataset(csv_path, cache_dir, use_cache=False)
    print(f"Dataset: {len(data_list)} samples, {features.shape[1]} features")

    # Quick experiment with just 1 fold
    results = []
    depths = [5, 10, 15]
    models = ['DCN', 'Standard']

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kfold.split(data_list))

    for model_type in models:
        for depth in depths:
            if model_type == 'Standard' and depth == 5:
                continue  # Skip standard GIN-5 (already done)

            print(f"\n{'='*60}")
            print(f"Training {model_type}-GIN-{depth}")
            print(f"{'='*60}")

            fold_rmses = []
            for fold_idx, (train_idx, test_idx) in enumerate(folds[:2]):  # Only 2 folds for speed
                print(f"\nFold {fold_idx + 1}/2")

                # Split train/val
                val_size = int(len(train_idx) * 0.1)
                val_idx = train_idx[:val_size]
                train_idx_actual = train_idx[val_size:]

                # Create datasets
                train_data = [data_list[i] for i in train_idx_actual]
                val_data = [data_list[i] for i in val_idx]
                test_data = [data_list[i] for i in test_idx]

                # Normalize features
                for i in range(len(train_data)):
                    train_data[i].mol_features = torch.tensor(
                        scaler_norm.transform(features[train_idx_actual[i]:train_idx_actual[i]+1]),
                        dtype=torch.float
                    )
                for i in range(len(val_data)):
                    val_data[i].mol_features = torch.tensor(
                        scaler_norm.transform(features[val_idx[i]:val_idx[i]+1]),
                        dtype=torch.float
                    )
                for i in range(len(test_data)):
                    test_data[i].mol_features = torch.tensor(
                        scaler_norm.transform(features[test_idx[i]:test_idx[i]+1]),
                        dtype=torch.float
                    )

                train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
                val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
                test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

                # Create model
                if model_type == 'DCN':
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
                best_val = quick_train(model, train_loader, val_loader, epochs=100, patience=20)

                # Test
                model.eval()
                preds, targets = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch.to(device)
                        features_batch = batch.mol_features if hasattr(batch, 'mol_features') else None
                        pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, features_batch, batch.batch)
                        preds.append(pred.cpu().numpy())
                        targets.append(batch.y.cpu().numpy())

                preds = np.concatenate(preds).squeeze()
                targets = np.concatenate(targets).squeeze()
                test_rmse = np.sqrt(mean_squared_error(targets, preds))

                fold_rmses.append(test_rmse)
                print(f"  Fold {fold_idx + 1}: Val={best_val:.2f}, Test={test_rmse:.2f}")

            mean_rmse = np.mean(fold_rmses)
            std_rmse = np.std(fold_rmses)
            results.append({
                'model': model_type,
                'depth': depth,
                'mean_rmse': mean_rmse,
                'std_rmse': std_rmse,
                'fold_rmses': fold_rmses
            })
            print(f"\n{model_type}-GIN-{depth}: {mean_rmse:.2f} ± {std_rmse:.2f}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    baselines = {
        'Day 2 5-layer GIN': 32.31,
        'CFD-MTL 3-layer GIN': 44.12
    }

    for r in results:
        print(f"\n{r['model']}-GIN-{r['depth']}: {r['mean_rmse']:.2f} ± {r['std_rmse']:.2f}")
        for name, baseline in baselines.items():
            improvement = (baseline - r['mean_rmse']) / baseline * 100
            print(f"  vs {name}: {improvement:+.1f}%")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('results_quick.csv', index=False)
    print("\nResults saved to results_quick.csv")

    with open('results_quick.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("Experiment completed!")
    print("="*80)


if __name__ == "__main__":
    main()
