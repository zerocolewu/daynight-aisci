"""
Fast ablation study script - optimized for quick experiments
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# Add local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import load_and_split_data, get_feature_dimensions, PolymerDataset
from model import get_model, count_parameters


def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, _ = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                batch.periodic_mask, batch.periodic_vectors
            )
            predictions.extend(pred.cpu().numpy().flatten())
            targets.extend(batch.y.cpu().numpy().flatten())

    predictions = np.array(predictions)
    targets = np.array(targets)

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'predictions': predictions, 'targets': targets}


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred, _ = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch,
            batch.periodic_mask, batch.periodic_vectors
        )

        loss = nn.functional.mse_loss(pred.squeeze(), batch.y.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def run_experiment(config, train_ds, val_ds, test_ds, device):
    """Run a single experiment with given config."""
    atom_dim, bond_dim = get_feature_dimensions()

    model = get_model(
        model_type=config.get('model_type', 'chemprop'),
        atom_dim=atom_dim,
        bond_dim=bond_dim,
        hidden_dim=config.get('hidden_dim', 300),
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.0),
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=config.get('batch_size', 64), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.get('batch_size', 64), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.get('batch_size', 64), shuffle=False, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-3), weight_decay=config.get('weight_decay', 0.0))

    best_val_rmse = float('inf')
    best_model_state = None
    patience_counter = 0
    epochs = config.get('epochs', 50)
    patience = config.get('patience', 20)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {train_loss:.2f} | Val RMSE: {val_metrics['rmse']:.2f}")

        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best model and evaluate
    model.load_state_dict(best_model_state)
    model.to(device)
    test_metrics = evaluate(model, test_loader, device)

    return {
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_r2': test_metrics['r2'],
        'best_val_rmse': best_val_rmse,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv')
    parser.add_argument('--output_dir', type=str, default='/home/home_public/aisci/AIscientist/src/2_02_PerioGT_Ablation')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data once
    print("Loading data...")
    train_ds, val_ds, test_ds, df, norm_params = load_and_split_data(
        args.data_path, seed=42, add_periodic_edge=True
    )

    # Define ablation configurations
    configs = [
        # Paper baseline
        {'name': 'paper_baseline', 'model_type': 'chemprop', 'hidden_dim': 300, 'num_layers': 3,
         'epochs': 50, 'lr': 1e-3, 'patience': 20},

        # Different hidden dims
        {'name': 'chemprop_h256', 'model_type': 'chemprop', 'hidden_dim': 256, 'num_layers': 3,
         'epochs': 50, 'lr': 1e-3, 'patience': 20},

        # More epochs
        {'name': 'chemprop_e100', 'model_type': 'chemprop', 'hidden_dim': 300, 'num_layers': 3,
         'epochs': 100, 'lr': 1e-3, 'patience': 30},

        # Different learning rates
        {'name': 'chemprop_lr1e4', 'model_type': 'chemprop', 'hidden_dim': 300, 'num_layers': 3,
         'epochs': 50, 'lr': 1e-4, 'patience': 20},

        # GraphFormer comparison
        {'name': 'graphformer_h256', 'model_type': 'graphformer', 'hidden_dim': 256, 'num_layers': 4,
         'epochs': 50, 'lr': 1e-3, 'patience': 20},
    ]

    results = []

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Config: {config['name']}")
        print(f"{'='*60}")

        start_time = time.time()
        result = run_experiment(config, train_ds, val_ds, test_ds, device)
        elapsed = time.time() - start_time

        result['name'] = config['name']
        result['time_sec'] = elapsed
        results.append(result)

        print(f"\n  Test RMSE: {result['test_rmse']:.2f}")
        print(f"  Test MAE: {result['test_mae']:.2f}")
        print(f"  Time: {elapsed:.1f}s")

    # Save results
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('test_rmse')
    df_results.to_csv(os.path.join(args.output_dir, 'ablation_results.csv'), index=False)

    print("\n" + "="*60)
    print("ABLATION RESULTS SUMMARY")
    print("="*60)
    print(df_results[['name', 'test_rmse', 'test_mae', 'test_r2', 'time_sec']].to_string(index=False))

    # Save best config
    best = df_results.iloc[0]
    best_config = {'name': best['name'], 'test_rmse': float(best['test_rmse'])}
    with open(os.path.join(args.output_dir, 'best_config.json'), 'w') as f:
        json.dump(best_config, f, indent=2)

    print(f"\nBest config: {best['name']} with RMSE: {best['test_rmse']:.2f}")


if __name__ == "__main__":
    main()
