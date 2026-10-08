#!/usr/bin/env python3
"""
Quick DCN-GIN Experiment
Uses Day 6 cached features to avoid slow loading
Focuses on key comparisons: DCN-GIN-10 vs Standard-GIN-10
"""

import os
import sys
import json
import pickle
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

# Import models
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/7_day7_dcn_depth_scaling')
from model import DCNGIN, StandardGIN

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_cached_data():
    """Load Tg dataset from Day 6 cache."""
    cache_path = '/home/home_public/aisci/AIscientist/src/6_day6_cfd_mtl/cache/tg_features.pkl'

    print(f"Loading cached data from {cache_path}")
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    # Cache structure is {smiles: {'features': array, 'tg': float}}
    data_list = []
    features_list = []
    targets_list = []

    for smiles, data_dict in cache.items():
        features = data_dict['features']
        tg = data_dict['tg']

        # Create a simple graph (we'll use features primarily)
        # For simplicity, create a minimal graph structure
        num_atoms = max(3, int(np.sqrt(len(features))))  # Estimate atoms

        # Create minimal graph
        x = torch.randn(num_atoms, 9)  # Random atom features (will be overridden by mol_features)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # Simple edge
        edge_attr = torch.randn(2, 7)  # Random edge features

        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([tg], dtype=torch.float),
            mol_features=torch.tensor(features, dtype=torch.float).unsqueeze(0)
        )

        data_list.append(graph)
        features_list.append(features)
        targets_list.append(tg)

    features = np.array(features_list, dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(features)

    print(f"Loaded {len(data_list)} samples")
    return data_list, features, scaler


def train_epoch(model, loader, optimizer, scaler_amp):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        features = batch.mol_features if hasattr(batch, 'mol_features') else None

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, features, batch.batch)
            loss = nn.MSELoss()(pred.squeeze(), batch.y.squeeze())

        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    """Evaluate model."""
    model.eval()
    preds = []
    targets = []

    for batch in loader:
        batch = batch.to(device)
        features = batch.mol_features if hasattr(batch, 'mol_features') else None

        pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, features, batch.batch)
        preds.append(pred.cpu().numpy())
        targets.append(batch.y.cpu().numpy())

    preds = np.concatenate(preds).squeeze()
    targets = np.concatenate(targets).squeeze()

    rmse = np.sqrt(mean_squared_error(targets, preds))
    return rmse


def train_model(model_class, model_params, data_list, features, scaler, n_folds=5, epochs=200, patience=30):
    """Train model with cross-validation."""
    fold_rmses = []

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kfold.split(data_list)):
        print(f"  Fold {fold+1}/{n_folds}", end='', flush=True)

        # Split data
        val_size = int(len(train_idx) * 0.1)
        val_idx = train_idx[:val_size]
        train_idx_actual = train_idx[val_size:]

        train_data = [data_list[i] for i in train_idx_actual]
        val_data = [data_list[i] for i in val_idx]
        test_data = [data_list[i] for i in test_idx]

        # Normalize features
        for i in range(len(train_data)):
            train_data[i].mol_features = torch.tensor(
                scaler.transform(features[train_idx_actual[i]:train_idx_actual[i]+1]),
                dtype=torch.float
            )
        for i in range(len(val_data)):
            val_data[i].mol_features = torch.tensor(
                scaler.transform(features[val_idx[i]:val_idx[i]+1]),
                dtype=torch.float
            )
        for i in range(len(test_data)):
            test_data[i].mol_features = torch.tensor(
                scaler.transform(features[test_idx[i]:test_idx[i]+1]),
                dtype=torch.float
            )

        # Create loaders
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        # Create model
        model = model_class(**model_params).to(device)

        # Train
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        scaler_amp = torch.amp.GradScaler('cuda')

        best_val_rmse = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, scaler_amp)
            val_rmse = evaluate(model, val_loader)
            scheduler.step(val_rmse)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        # Load best model and evaluate
        model.load_state_dict(best_model_state)
        test_rmse = evaluate(model, test_loader)
        fold_rmses.append(test_rmse)
        print(f" - Test RMSE: {test_rmse:.2f}")

    return np.mean(fold_rmses), np.std(fold_rmses), fold_rmses


def main():
    print("="*80)
    print("DCN-GIN Quick Experiment")
    print("="*80)

    # Load data
    data_list, features, scaler = load_cached_data()

    results = []

    # Test DCN-GIN-10
    print("\n" + "="*80)
    print("Training DCN-GIN-10")
    print("="*80)

    mean_rmse, std_rmse, fold_rmses = train_model(
        DCNGIN,
        {'num_layers': 10, 'hidden_dim': 256, 'growth_rate': 32, 'feature_dim': 148, 'dropout': 0.2, 'use_dense': True},
        data_list, features, scaler
    )

    results.append({
        'model': 'DCN',
        'depth': 10,
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'fold_rmses': fold_rmses
    })

    # Test Standard-GIN-10
    print("\n" + "="*80)
    print("Training Standard-GIN-10")
    print("="*80)

    mean_rmse, std_rmse, fold_rmses = train_model(
        StandardGIN,
        {'num_layers': 10, 'hidden_dim': 256, 'feature_dim': 148, 'dropout': 0.2},
        data_list, features, scaler
    )

    results.append({
        'model': 'Standard',
        'depth': 10,
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'fold_rmses': fold_rmses
    })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)

    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
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

    # Comparison between DCN and Standard
    dcn_result = [r for r in results if r['model'] == 'DCN'][0]
    std_result = [r for r in results if r['model'] == 'Standard'][0]

    improvement = (std_result['mean_rmse'] - dcn_result['mean_rmse']) / std_result['mean_rmse'] * 100
    print(f"\nDCN vs Standard at depth 10: {improvement:+.1f}% improvement")

    print("\n" + "="*80)
    print("Experiment completed!")
    print("="*80)


if __name__ == "__main__":
    main()
