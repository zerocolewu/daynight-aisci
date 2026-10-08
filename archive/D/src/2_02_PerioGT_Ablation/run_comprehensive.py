#!/usr/bin/env python
"""
Comprehensive ablation study - multiple seeds and configurations
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
from model import get_model


def log(msg):
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
    return np.sqrt(mean_squared_error(targets, preds))


def run_experiment(config, seed, device, train_ds, val_ds, test_ds, atom_dim, bond_dim):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = get_model(
        model_type=config.get('model_type', 'chemprop'),
        atom_dim=atom_dim,
        bond_dim=bond_dim,
        hidden_dim=config.get('hidden_dim', 300),
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.0),
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=config.get('batch_size', 64), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 0.0)
    )

    epochs = config.get('epochs', 300)
    patience = config.get('patience', 50)
    best_val = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = nn.functional.mse_loss(pred.squeeze(), batch.y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_rmse = evaluate(model, val_loader, device)
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.to(device)
    test_rmse = evaluate(model, test_loader, device)
    return test_rmse


def main():
    output_dir = '/home/home_public/aisci/AIscientist/src/2_02_PerioGT_Ablation'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")

    # Load data
    log("Loading data...")
    train_ds, val_ds, test_ds, df, _ = load_and_split_data(
        '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv',
        seed=42, add_periodic_edge=True
    )
    atom_dim, bond_dim = get_feature_dimensions()

    # Key configurations to test
    configs = {
        'graphformer_best': {
            'model_type': 'graphformer', 'hidden_dim': 256, 'num_layers': 4,
            'lr': 1e-4, 'weight_decay': 1e-5, 'dropout': 0.1,
            'epochs': 300, 'patience': 50
        },
        'chemprop_wd': {
            'model_type': 'chemprop', 'hidden_dim': 300, 'num_layers': 3,
            'lr': 1e-3, 'weight_decay': 1e-5, 'dropout': 0.0,
            'epochs': 300, 'patience': 50
        },
        'chemprop_dropout': {
            'model_type': 'chemprop', 'hidden_dim': 300, 'num_layers': 3,
            'lr': 1e-3, 'weight_decay': 1e-5, 'dropout': 0.1,
            'epochs': 300, 'patience': 50
        },
    }

    seeds = [42, 123, 456]
    results = []

    for name, config in configs.items():
        log(f"\n{'='*50}")
        log(f"Config: {name}")
        log(f"{'='*50}")

        rmses = []
        for seed in seeds:
            log(f"  Seed {seed}...")
            rmse = run_experiment(config, seed, device, train_ds, val_ds, test_ds, atom_dim, bond_dim)
            rmses.append(rmse)
            log(f"    RMSE: {rmse:.2f}")

        mean_rmse = np.mean(rmses)
        std_rmse = np.std(rmses)
        log(f"  Mean: {mean_rmse:.2f} ± {std_rmse:.2f}")

        results.append({
            'name': name,
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'rmses': rmses,
        })

    # Summary
    df_results = pd.DataFrame([
        {'name': r['name'], 'mean_rmse': r['mean_rmse'], 'std_rmse': r['std_rmse'],
         'seed42': r['rmses'][0], 'seed123': r['rmses'][1], 'seed456': r['rmses'][2]}
        for r in results
    ]).sort_values('mean_rmse')

    df_results.to_csv(os.path.join(output_dir, 'comprehensive_results.csv'), index=False)

    log("\n" + "="*60)
    log("COMPREHENSIVE RESULTS")
    log("="*60)
    log(df_results.to_string(index=False))

    # Save best
    best = df_results.iloc[0]
    with open(os.path.join(output_dir, 'best_config.json'), 'w') as f:
        json.dump(best.to_dict(), f, indent=2)
    log(f"\nBest: {best['name']} with RMSE {best['mean_rmse']:.2f} ± {best['std_rmse']:.2f}")


if __name__ == "__main__":
    main()
