"""
PerioGT Ablation Study - Main Training Script

Systematic ablation to identify sources of the 5.6 RMSE gap.
"""
import os
import sys
import json
import time
import argparse
import itertools
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
import matplotlib.pyplot as plt

from dataset import load_and_split_data, get_feature_dimensions, denormalize_predictions
from model import get_model, count_parameters


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    norm_params: Optional[Dict] = None,
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
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
                batch.periodic_mask,
                batch.periodic_vectors,
            )

            predictions.extend(pred.cpu().numpy().flatten())
            targets.extend(batch.y.cpu().numpy().flatten())

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Denormalize if needed
    if norm_params is not None:
        predictions = denormalize_predictions(predictions, norm_params)
        targets = denormalize_predictions(targets, norm_params)

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    pearson_r, _ = pearsonr(targets, predictions)

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson_r': pearson_r,
        'predictions': predictions,
        'targets': targets,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred, _ = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.periodic_mask,
            batch.periodic_vectors,
        )

        loss = nn.functional.mse_loss(pred.squeeze(), batch.y.squeeze())
        loss.backward()

        # Gradient clipping (helps stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def train_model(
    train_dataset,
    val_dataset,
    test_dataset,
    args,
    norm_params: Optional[Dict] = None,
) -> Dict:
    """Full training loop."""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    atom_dim, bond_dim = get_feature_dimensions()
    model = get_model(
        model_type=args.model_type,
        atom_dim=atom_dim,
        bond_dim=bond_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    num_params = count_parameters(model)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    elif args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    else:
        scheduler = None

    # Training tracking
    history = {
        'train_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'lr': [],
    }

    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_metrics = evaluate(model, val_loader, device, norm_params)

        # Update scheduler
        if scheduler is not None:
            if args.lr_scheduler == 'plateau':
                scheduler.step(val_metrics['rmse'])
            else:
                scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = args.lr

        # Record history
        history['train_loss'].append(train_loss)
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_mae'].append(val_metrics['mae'])
        history['lr'].append(current_lr)

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                  f"Val RMSE: {val_metrics['rmse']:.2f} | LR: {current_lr:.2e}")

        # Early stopping
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_epoch = epoch
            patience_counter = 0
            # Save best model state in memory
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    training_time = time.time() - start_time

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    model.to(device)
    test_metrics = evaluate(model, test_loader, device, norm_params)

    results = {
        'best_epoch': best_epoch,
        'best_val_rmse': best_val_rmse,
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_r2': test_metrics['r2'],
        'test_pearson_r': test_metrics['pearson_r'],
        'training_time': training_time,
        'num_params': num_params,
        'history': history,
        'test_predictions': test_metrics['predictions'].tolist(),
        'test_targets': test_metrics['targets'].tolist(),
    }

    return results


def run_single_experiment(args, seed: int) -> Dict:
    """Run a single experiment with given configuration."""
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load data
    train_ds, val_ds, test_ds, df, norm_params = load_and_split_data(
        args.data_path,
        seed=seed,
        add_periodic_edge=args.add_periodic_edge,
        normalize_targets=args.normalize_targets,
    )

    # Train model
    results = train_model(train_ds, val_ds, test_ds, args, norm_params)

    return results


def run_ablation_grid(args):
    """Run grid search over ablation parameters."""

    # Define ablation configurations
    ablation_configs = []

    # Ablation 1: Model architecture
    for model_type in ['chemprop', 'graphformer']:
        for hidden_dim in [256, 300]:
            for num_layers in [3, 4]:
                ablation_configs.append({
                    'name': f'arch_{model_type}_h{hidden_dim}_l{num_layers}',
                    'model_type': model_type,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                })

    # Ablation 2: Training protocol
    for epochs in [50, 100, 300]:
        for lr in [1e-3, 1e-4, 1e-5]:
            for lr_scheduler in ['cosine', 'none']:
                ablation_configs.append({
                    'name': f'train_e{epochs}_lr{lr:.0e}_sch{lr_scheduler}',
                    'epochs': epochs,
                    'lr': lr,
                    'lr_scheduler': lr_scheduler,
                })

    # Ablation 3: Regularization
    for dropout in [0.0, 0.1, 0.2]:
        for weight_decay in [0.0, 1e-5, 1e-4]:
            ablation_configs.append({
                'name': f'reg_drop{dropout}_wd{weight_decay:.0e}',
                'dropout': dropout,
                'weight_decay': weight_decay,
            })

    print(f"Total ablation configurations: {len(ablation_configs)}")

    # Run experiments
    all_results = []

    for config in ablation_configs:
        print(f"\n{'='*60}")
        print(f"Running: {config['name']}")
        print(f"{'='*60}")

        # Merge config with args
        exp_args = argparse.Namespace(**vars(args))
        for key, value in config.items():
            if key != 'name':
                setattr(exp_args, key, value)

        # Run with multiple seeds
        seed_results = []
        for seed in [42, 123, 456]:
            print(f"\nSeed {seed}...")
            result = run_single_experiment(exp_args, seed)
            seed_results.append(result)

        # Aggregate results
        test_rmses = [r['test_rmse'] for r in seed_results]
        aggregated = {
            'config': config,
            'test_rmse_mean': np.mean(test_rmses),
            'test_rmse_std': np.std(test_rmses),
            'best_seed_rmse': min(test_rmses),
            'all_test_rmses': test_rmses,
        }

        all_results.append(aggregated)

        print(f"\nResults: {aggregated['test_rmse_mean']:.2f} +/- {aggregated['test_rmse_std']:.2f}")

    # Save all results
    results_df = pd.DataFrame([
        {
            'config_name': r['config']['name'],
            'test_rmse_mean': r['test_rmse_mean'],
            'test_rmse_std': r['test_rmse_std'],
            'best_rmse': r['best_seed_rmse'],
            **{k: v for k, v in r['config'].items() if k != 'name'}
        }
        for r in all_results
    ])

    results_df = results_df.sort_values('test_rmse_mean')
    results_df.to_csv(os.path.join(args.output_dir, 'ablation_results.csv'), index=False)

    # Print top 10
    print("\n" + "="*60)
    print("TOP 10 CONFIGURATIONS BY TEST RMSE")
    print("="*60)
    print(results_df.head(10).to_string())

    return all_results


def main():
    parser = argparse.ArgumentParser(description='PerioGT Ablation Study')

    # Data
    parser.add_argument('--data_path', type=str,
                        default='/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv')
    parser.add_argument('--output_dir', type=str,
                        default='/home/home_public/aisci/AIscientist/src/2_02_PerioGT_Ablation')
    parser.add_argument('--seed', type=int, default=42)

    # Model
    parser.add_argument('--model_type', type=str, default='chemprop', choices=['chemprop', 'graphformer'])
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)

    # Data options
    parser.add_argument('--add_periodic_edge', action='store_true', default=True)
    parser.add_argument('--no_periodic_edge', dest='add_periodic_edge', action='store_false')
    parser.add_argument('--normalize_targets', action='store_true', default=False)

    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr_scheduler', type=str, default='none', choices=['none', 'cosine', 'plateau'])

    # Run mode
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'grid', 'quick'])
    parser.add_argument('--num_seeds', type=int, default=1, help='Number of random seeds for single mode')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("="*60)
    print("PerioGT Ablation Study")
    print("="*60)
    print(f"Model type: {args.model_type}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Num layers: {args.num_layers}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Add periodic edge: {args.add_periodic_edge}")
    print(f"Output: {args.output_dir}")
    print("="*60)

    if args.mode == 'grid':
        # Full grid search
        results = run_ablation_grid(args)
    elif args.mode == 'quick':
        # Quick ablation with fewer configurations
        run_quick_ablation(args)
    else:
        # Single experiment
        results = []
        for seed in [42 + i * 100 for i in range(args.num_seeds)]:
            print(f"\n{'='*60}")
            print(f"Running with seed {seed}")
            print(f"{'='*60}")
            result = run_single_experiment(args, seed)
            results.append(result)

            print(f"\n=== Test Results (seed {seed}) ===")
            print(f"RMSE: {result['test_rmse']:.2f}")
            print(f"MAE: {result['test_mae']:.2f}")
            print(f"R2: {result['test_r2']:.4f}")
            print(f"Pearson R: {result['test_pearson_r']:.4f}")
            print(f"Best epoch: {result['best_epoch']}")

        # Aggregate if multiple seeds
        if len(results) > 1:
            rmses = [r['test_rmse'] for r in results]
            print(f"\n=== Aggregated Results ({len(results)} seeds) ===")
            print(f"RMSE: {np.mean(rmses):.2f} +/- {np.std(rmses):.2f}")

        # Save results
        results_to_save = {
            'args': vars(args),
            'results': [{
                'seed': 42 + i * 100,
                'test_rmse': r['test_rmse'],
                'test_mae': r['test_mae'],
                'test_r2': r['test_r2'],
                'test_pearson_r': r['test_pearson_r'],
                'best_epoch': r['best_epoch'],
                'training_time': r['training_time'],
            } for i, r in enumerate(results)]
        }

        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(results_to_save, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


def run_quick_ablation(args):
    """Run quick ablation focusing on key factors."""
    print("\nRunning Quick Ablation Study...")

    quick_configs = [
        # Paper baseline
        {'name': 'paper_baseline', 'model_type': 'chemprop', 'hidden_dim': 300,
         'num_layers': 3, 'epochs': 50, 'lr': 1e-3, 'lr_scheduler': 'none'},

        # Architecture variations
        {'name': 'chemprop_h256', 'model_type': 'chemprop', 'hidden_dim': 256,
         'num_layers': 3, 'epochs': 50, 'lr': 1e-3, 'lr_scheduler': 'none'},

        # Training variations
        {'name': 'chemprop_e100', 'model_type': 'chemprop', 'hidden_dim': 300,
         'num_layers': 3, 'epochs': 100, 'lr': 1e-3, 'lr_scheduler': 'none'},

        {'name': 'chemprop_cosine', 'model_type': 'chemprop', 'hidden_dim': 300,
         'num_layers': 3, 'epochs': 50, 'lr': 1e-3, 'lr_scheduler': 'cosine'},

        # No periodic edge
        {'name': 'chemprop_no_periodic', 'model_type': 'chemprop', 'hidden_dim': 300,
         'num_layers': 3, 'epochs': 50, 'lr': 1e-3, 'lr_scheduler': 'none',
         'add_periodic_edge': False},

        # GraphFormer comparison
        {'name': 'graphformer', 'model_type': 'graphformer', 'hidden_dim': 256,
         'num_layers': 4, 'epochs': 50, 'lr': 1e-3, 'lr_scheduler': 'none'},
    ]

    all_results = []

    for config in quick_configs:
        print(f"\n{'='*60}")
        print(f"Quick Ablation: {config['name']}")
        print(f"{'='*60}")

        # Create experiment args
        exp_args = argparse.Namespace(**vars(args))
        for key, value in config.items():
            if key != 'name':
                setattr(exp_args, key, value)

        # Run single seed for quick test
        result = run_single_experiment(exp_args, 42)

        all_results.append({
            'name': config['name'],
            'test_rmse': result['test_rmse'],
            'test_mae': result['test_mae'],
            'test_r2': result['test_r2'],
            'best_epoch': result['best_epoch'],
        })

        print(f"RMSE: {result['test_rmse']:.2f}")

    # Print summary
    print("\n" + "="*60)
    print("QUICK ABLATION SUMMARY")
    print("="*60)
    df = pd.DataFrame(all_results)
    df = df.sort_values('test_rmse')
    print(df.to_string(index=False))

    # Save
    df.to_csv(os.path.join(args.output_dir, 'quick_ablation_results.csv'), index=False)


if __name__ == "__main__":
    main()
