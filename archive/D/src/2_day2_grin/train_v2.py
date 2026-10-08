"""
Improved training script for GRIN experiments on Tg dataset.
Key improvements:
1. Use rich atom/bond features
2. Proper target normalization
3. Longer training with early stopping
4. Better hyperparameters
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from data_utils_v2 import get_dataloaders, TOTAL_ATOM_FEATURES
from models import GINMaxAggregation, count_parameters


def train_epoch(model, loader, optimizer, device, use_l1=False, l1_lambda=0.001):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        pred = model(batch.x, batch.edge_index, batch.batch)
        target = batch.y.view(-1, 1)

        # MSE loss
        mse_loss = nn.functional.mse_loss(pred, target)

        # L1 regularization
        if use_l1:
            l1_penalty = model.get_l1_penalty()
            loss = mse_loss + l1_lambda * l1_penalty
        else:
            loss = mse_loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs

    return total_loss / total_samples


def evaluate(model, loader, device, target_scaler=None):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    total_samples = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            pred = model(batch.x, batch.edge_index, batch.batch)
            target = batch.y.view(-1, 1)

            loss = nn.functional.mse_loss(pred, target)

            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs

            # Inverse transform predictions and targets
            if target_scaler is not None:
                pred_original = target_scaler.inverse_transform(pred.cpu().numpy())
                target_original = target_scaler.inverse_transform(target.cpu().numpy())
                predictions.append(pred_original)
                targets.append(target_original)
            else:
                predictions.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Calculate metrics in original scale
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'predictions': predictions,
        'targets': targets
    }


def train_model(config, save_dir):
    """
    Train a model with given configuration.

    Args:
        config: Dictionary with training configuration
        save_dir: Directory to save results

    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*80}")
    print(f"Experiment: {config['name']}")
    print(f"{'='*80}")
    print(f"Config: {json.dumps({k: v for k, v in config.items() if k not in ['csv_path']}, indent=2)}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader, target_scaler = get_dataloaders(
        csv_path=config['csv_path'],
        batch_size=config['batch_size'],
        augmentation=config['augmentation'],
        num_workers=0
    )

    # Get input dimension
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch.x.shape[1]
    print(f"Input dimension: {input_dim}")

    # Create model
    model = GINMaxAggregation(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_max_aggregation=config['use_max_aggregation'],
        readout=config['readout']
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['max_epochs'], eta_min=1e-6)

    # Training loop
    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_rmse': [],
        'val_mae': []
    }

    print(f"\nStarting training for {config['max_epochs']} epochs...")

    for epoch in range(config['max_epochs']):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            use_l1=config['use_l1'],
            l1_lambda=config['l1_lambda']
        )

        # Validate
        val_metrics = evaluate(model, val_loader, device, target_scaler)

        # Update history
        history['train_loss'].append(train_loss)
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_mae'].append(val_metrics['mae'])

        # Update scheduler
        scheduler.step()

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config['max_epochs']}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val RMSE: {val_metrics['rmse']:.2f}, "
                  f"Val MAE: {val_metrics['mae']:.2f}", flush=True)

        # Early stopping
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_epoch = epoch + 1
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nBest validation RMSE: {best_val_rmse:.2f} at epoch {best_epoch}")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))
    test_metrics = evaluate(model, test_loader, device, target_scaler)

    print(f"Test RMSE: {test_metrics['rmse']:.2f}")
    print(f"Test MAE: {test_metrics['mae']:.2f}")

    # Save results
    results = {
        'config': {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict, type(None))) else v
                   for k, v in config.items()},
        'best_epoch': int(best_epoch),
        'best_val_rmse': float(best_val_rmse),
        'test_rmse': float(test_metrics['rmse']),
        'test_mae': float(test_metrics['mae']),
        'history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_rmse': [float(x) for x in history['val_rmse']],
            'val_mae': [float(x) for x in history['val_mae']]
        },
        'predictions': [float(x) for x in test_metrics['predictions'].flatten()],
        'targets': [float(x) for x in test_metrics['targets'].flatten()]
    }

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['val_rmse'], label='Val RMSE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Validation RMSE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()

    # Plot predictions vs targets
    plt.figure(figsize=(6, 6))
    plt.scatter(test_metrics['targets'], test_metrics['predictions'], alpha=0.5, s=10)
    plt.plot([test_metrics['targets'].min(), test_metrics['targets'].max()],
             [test_metrics['targets'].min(), test_metrics['targets'].max()],
             'r--', lw=2)
    plt.xlabel('True Tg')
    plt.ylabel('Predicted Tg')
    plt.title(f'Predictions vs Targets (RMSE: {test_metrics["rmse"]:.2f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictions.png'), dpi=150)
    plt.close()

    return results


def main():
    parser = argparse.ArgumentParser(description='GRIN experiments on Tg dataset (v2)')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'baseline', 'max_only', 'max_l1', 'full_grin', 'mean_aug'],
                        help='Which experiment to run')
    args = parser.parse_args()

    # Base configuration - improved hyperparameters
    base_config = {
        'csv_path': '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv',
        'batch_size': 64,
        'hidden_dim': 256,
        'num_layers': 5,
        'dropout': 0.15,
        'lr': 5e-4,
        'weight_decay': 1e-5,
        'max_epochs': 300,
        'patience': 40,
        'readout': 'mean'
    }

    # Define experiments
    experiments = {
        'baseline': {
            **base_config,
            'name': 'Baseline: GIN with mean aggregation, no augmentation',
            'use_max_aggregation': False,
            'use_l1': False,
            'l1_lambda': 0.0,
            'augmentation': None
        },
        'max_only': {
            **base_config,
            'name': 'Ablation 1: GIN with max aggregation only',
            'use_max_aggregation': True,
            'use_l1': False,
            'l1_lambda': 0.0,
            'augmentation': None
        },
        'max_l1': {
            **base_config,
            'name': 'Ablation 2: GIN with max aggregation + L1 sparsity',
            'use_max_aggregation': True,
            'use_l1': True,
            'l1_lambda': 0.0005,
            'augmentation': None
        },
        'full_grin': {
            **base_config,
            'name': 'Full GRIN: Max aggregation + L1 + {1,3}-RU augmentation',
            'use_max_aggregation': True,
            'use_l1': True,
            'l1_lambda': 0.0005,
            'augmentation': 'both'
        },
        'mean_aug': {
            **base_config,
            'name': 'Additional: Mean aggregation + {1,3}-RU augmentation',
            'use_max_aggregation': False,
            'use_l1': False,
            'l1_lambda': 0.0,
            'augmentation': 'both'
        }
    }

    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'/home/home_public/aisci/AIscientist/src/2_day2_grin/results_v2_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)

    print(f"Results directory: {results_dir}")

    # Run experiments
    all_results = {}

    if args.experiment == 'all':
        exp_list = list(experiments.keys())
    else:
        exp_list = [args.experiment]

    for exp_name in exp_list:
        config = experiments[exp_name]
        exp_dir = os.path.join(results_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        try:
            results = train_model(config, exp_dir)
            all_results[exp_name] = {
                'test_rmse': results['test_rmse'],
                'test_mae': results['test_mae'],
                'best_val_rmse': results['best_val_rmse'],
                'best_epoch': results['best_epoch']
            }
        except Exception as e:
            print(f"ERROR in experiment {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp_name] = {
                'test_rmse': float('inf'),
                'test_mae': float('inf'),
                'best_val_rmse': float('inf'),
                'best_epoch': 0,
                'error': str(e)
            }

    # Create comparison summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")

    summary_df = pd.DataFrame(all_results).T
    summary_df = summary_df.round(2)
    print(summary_df)

    # Save summary
    summary_df.to_csv(os.path.join(results_dir, 'summary.csv'))

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(all_results))
    rmses = [all_results[exp]['test_rmse'] for exp in exp_list]

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax.bar(x, rmses, color=colors[:len(exp_list)])
    ax.set_xticks(x)
    ax.set_xticklabels([exp.replace('_', '\n') for exp in exp_list], fontsize=10)
    ax.set_ylabel('Test RMSE', fontsize=12)
    ax.set_title('Comparison of GRIN Ablation Experiments (v2)', fontsize=14)
    ax.axhline(y=37.0, color='red', linestyle='--', label='GraphFormer baseline (37.0)')
    ax.axhline(y=35.0, color='green', linestyle='--', label='Target (35.0)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, rmse in zip(bars, rmses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comparison.png'), dpi=150)
    plt.close()

    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    main()
