#!/usr/bin/env python
"""
Fast ablation study with proper output flushing
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import load_and_split_data, get_feature_dimensions
from model import get_model, count_parameters


def log(msg):
    """Print with flush."""
    print(msg, flush=True)


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds.extend(pred.cpu().numpy().flatten())
            targets.extend(batch.y.cpu().numpy().flatten())
    preds, targets = np.array(preds), np.array(targets)
    return {
        'rmse': np.sqrt(mean_squared_error(targets, preds)),
        'mae': mean_absolute_error(targets, preds),
        'r2': r2_score(targets, preds)
    }


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = nn.functional.mse_loss(pred.squeeze(), batch.y.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def run_config(name, config, train_ds, val_ds, test_ds, device, atom_dim, bond_dim):
    log(f"\n{'='*50}")
    log(f"Config: {name}")
    log(f"{'='*50}")

    model = get_model(
        model_type=config.get('model_type', 'chemprop'),
        atom_dim=atom_dim,
        bond_dim=bond_dim,
        hidden_dim=config.get('hidden_dim', 300),
        num_layers=config.get('num_layers', 3),
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))

    epochs = config.get('epochs', 50)
    patience = config.get('patience', 20)
    best_val_rmse = float('inf')
    patience_counter = 0
    best_state = None

    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        if epoch % 10 == 0 or epoch == 1:
            log(f"Epoch {epoch:3d} | Loss: {train_loss:.2f} | Val RMSE: {val_metrics['rmse']:.2f}")

        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(f"Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = evaluate(model, test_loader, device)
    elapsed = time.time() - start

    log(f"Test RMSE: {test_metrics['rmse']:.2f} | MAE: {test_metrics['mae']:.2f} | Time: {elapsed:.1f}s")

    return {
        'name': name,
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_r2': test_metrics['r2'],
        'best_val_rmse': best_val_rmse,
        'time_sec': elapsed
    }


def main():
    output_dir = '/home/home_public/aisci/AIscientist/src/2_02_PerioGT_Ablation'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")

    log("Loading data...")
    train_ds, val_ds, test_ds, df, norm_params = load_and_split_data(
        '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv',
        seed=42, add_periodic_edge=True
    )
    atom_dim, bond_dim = get_feature_dimensions()
    log(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Key ablation configs
    configs = {
        'paper_baseline': {'model_type': 'chemprop', 'hidden_dim': 300, 'num_layers': 3, 'epochs': 50, 'lr': 1e-3, 'patience': 20},
        'chemprop_h256': {'model_type': 'chemprop', 'hidden_dim': 256, 'num_layers': 3, 'epochs': 50, 'lr': 1e-3, 'patience': 20},
        'chemprop_e100': {'model_type': 'chemprop', 'hidden_dim': 300, 'num_layers': 3, 'epochs': 100, 'lr': 1e-3, 'patience': 30},
        'chemprop_lr1e4': {'model_type': 'chemprop', 'hidden_dim': 300, 'num_layers': 3, 'epochs': 50, 'lr': 1e-4, 'patience': 20},
        'graphformer': {'model_type': 'graphformer', 'hidden_dim': 256, 'num_layers': 4, 'epochs': 50, 'lr': 1e-3, 'patience': 20},
    }

    results = []
    for name, config in configs.items():
        result = run_config(name, config, train_ds, val_ds, test_ds, device, atom_dim, bond_dim)
        results.append(result)

    # Summary
    df_results = pd.DataFrame(results).sort_values('test_rmse')
    df_results.to_csv(os.path.join(output_dir, 'ablation_results.csv'), index=False)

    log("\n" + "="*60)
    log("ABLATION RESULTS SUMMARY")
    log("="*60)
    log(df_results[['name', 'test_rmse', 'test_mae', 'test_r2', 'time_sec']].to_string(index=False))

    # Save best
    best = df_results.iloc[0]
    with open(os.path.join(output_dir, 'best_config.json'), 'w') as f:
        json.dump(best.to_dict(), f, indent=2)
    log(f"\nBest: {best['name']} with RMSE {best['test_rmse']:.2f}")


if __name__ == "__main__":
    main()
