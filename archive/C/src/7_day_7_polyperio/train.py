"""
Main training script for PolyPerio-DenseGNN ablation study.
Runs all 4 ablation variants across 5 seeds with proper train/val/test splits.
"""

import os
import sys
import json
import time
import random
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import PolymerTgDataset, mol_to_graph
from models.polyperio_densegnn import build_model, count_parameters

warnings.filterwarnings('ignore')

# Reproducibility
SEEDS = [42, 123, 456, 789, 2024]

# Variants for ablation study
VARIANTS = ['baseline', 'periodic', 'angular', 'full']

# Variant-specific dataset settings
VARIANT_CONFIG = {
    'baseline': {'use_trimer': False, 'use_periodic': False, 'use_angular': False, 'use_3d': False},
    'periodic': {'use_trimer': True, 'use_periodic': True, 'use_angular': False, 'use_3d': False},
    'angular': {'use_trimer': True, 'use_periodic': True, 'use_angular': True, 'use_3d': False},
    'full': {'use_trimer': True, 'use_periodic': True, 'use_angular': True, 'use_3d': False},
}


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_dataset(dataset, seed, train_ratio=0.72, val_ratio=0.08, test_ratio=0.20):
    """Split dataset into train/val/test with fixed test set."""
    n = len(dataset)
    indices = list(range(n))

    # Use seed for reproducible split
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)
        target = batch.y.squeeze()

        loss = criterion(pred, target)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

    return total_loss / num_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    num_samples = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        target = batch.y.squeeze()

        loss = criterion(pred, target)

        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    mse = torch.mean((all_preds - all_targets) ** 2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(all_preds - all_targets)).item()
    r2 = 1 - torch.sum((all_preds - all_targets) ** 2) / torch.sum((all_targets - all_targets.mean()) ** 2)
    r2 = r2.item()

    return {
        'loss': total_loss / num_samples,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
    }


def run_experiment(variant, seed, args):
    """Run a single experiment (one variant, one seed)."""
    print(f"\n{'='*60}")
    print(f"Running: variant={variant}, seed={seed}")
    print(f"{'='*60}")

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get variant config
    config = VARIANT_CONFIG[variant]

    # Load dataset
    print(f"Loading dataset with config: {config}")
    dataset = PolymerTgDataset(
        csv_path=args.data_path,
        use_trimer=config['use_trimer'],
        use_periodic=config['use_periodic'],
        use_angular=config['use_angular'],
        use_3d=config['use_3d'],
    )

    if len(dataset) == 0:
        print("ERROR: No valid molecules in dataset!")
        return None

    # Split dataset
    train_idx, val_idx, test_idx = split_dataset(dataset, seed)

    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]
    test_dataset = [dataset[i] for i in test_idx]

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Get feature dimensions from first graph
    sample_graph = dataset[0]
    node_in_dim = sample_graph.x.size(1)
    edge_in_dim = sample_graph.edge_attr.size(1)
    print(f"Node features: {node_in_dim}, Edge features: {edge_in_dim}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Build model
    model = build_model(
        variant=variant,
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    )
    model = model.to(device)

    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )

    # Loss function - use Huber loss for robustness
    criterion = nn.HuberLoss(delta=1.0)

    # Training loop
    best_val_rmse = float('inf')
    best_epoch = 0
    best_model_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_rmses = []
    val_rmses = []

    start_time = time.time()

    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_metrics = evaluate(model, train_loader, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics['rmse'])

        train_losses.append(train_loss)
        val_losses.append(val_metrics['loss'])
        train_rmses.append(train_metrics['rmse'])
        val_rmses.append(val_metrics['rmse'])

        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 20 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:4d} | "
                  f"Train RMSE: {train_metrics['rmse']:.4f} | "
                  f"Val RMSE: {val_metrics['rmse']:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {elapsed:.1f}s")

        # Early stopping
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s. Best val RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    model = model.to(device)

    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Results ({variant}, seed={seed}):")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  R2:   {test_metrics['r2']:.4f}")

    return {
        'variant': variant,
        'seed': seed,
        'num_params': num_params,
        'best_epoch': best_epoch,
        'best_val_rmse': best_val_rmse,
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_r2': test_metrics['r2'],
        'test_mse': test_metrics['mse'],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_rmses': train_rmses,
        'val_rmses': val_rmses,
    }


def main():
    parser = argparse.ArgumentParser(description='PolyPerio-DenseGNN Training')
    parser.add_argument('--data_path', type=str,
                       default='/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv',
                       help='Path to Tg dataset')
    parser.add_argument('--hidden_dim', type=int, default=96,
                       help='Hidden dimension')
    parser.add_argument('--num_blocks', type=int, default=5,
                       help='Number of MPNN blocks')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=500,
                       help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience')
    parser.add_argument('--variants', nargs='+', default=VARIANTS,
                       help='Which variants to run')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS,
                       help='Random seeds')
    parser.add_argument('--output_dir', type=str,
                       default='/home/home_public/aisci/AIscientist/src/7_day_7_polyperio/results',
                       help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    for variant in args.variants:
        variant_results = []
        for seed in args.seeds:
            result = run_experiment(variant, seed, args)
            if result is not None:
                variant_results.append(result)
                # Save intermediate results
                result_save = {k: v for k, v in result.items()
                             if k not in ['train_losses', 'val_losses', 'train_rmses', 'val_rmses']}
                result_save['train_rmses_final'] = result['train_rmses'][-10:]
                result_save['val_rmses_final'] = result['val_rmses'][-10:]
                temp_path = os.path.join(args.output_dir, f'{variant}_seed{seed}_temp.json')
                with open(temp_path, 'w') as f:
                    json.dump(result_save, f, indent=2)

        if variant_results:
            # Aggregate results
            test_rmses = [r['test_rmse'] for r in variant_results]
            test_maes = [r['test_mae'] for r in variant_results]
            test_r2s = [r['test_r2'] for r in variant_results]
            val_rmses_best = [r['best_val_rmse'] for r in variant_results]
            num_params = variant_results[0]['num_params']

            mean_rmse = np.mean(test_rmses)
            std_rmse = np.std(test_rmses)
            mean_mae = np.mean(test_maes)
            std_mae = np.std(test_maes)
            mean_r2 = np.mean(test_r2s)
            std_r2 = np.std(test_r2s)

            summary = {
                'variant': variant,
                'num_params': num_params,
                'num_seeds': len(variant_results),
                'test_rmse_mean': mean_rmse,
                'test_rmse_std': std_rmse,
                'test_rmse_all': test_rmses,
                'test_mae_mean': mean_mae,
                'test_mae_std': std_mae,
                'test_r2_mean': mean_r2,
                'test_r2_std': std_r2,
                'best_val_rmse_mean': np.mean(val_rmses_best),
                'best_val_rmse_std': np.std(val_rmses_best),
                'config': VARIANT_CONFIG[variant],
            }

            all_results[variant] = summary

            print(f"\n{'*'*60}")
            print(f"Summary for {variant}:")
            print(f"  Parameters: {num_params:,}")
            print(f"  Test RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
            print(f"  Test MAE:  {mean_mae:.4f} ± {std_mae:.4f}")
            print(f"  Test R2:   {mean_r2:.4f} ± {std_r2:.4f}")
            print(f"  Individual RMSEs: {[f'{r:.4f}' for r in test_rmses]}")
            print(f"{'*'*60}")

    # Save all results
    results_path = os.path.join(args.output_dir, 'ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"{'Variant':<15} {'Params':>10} {'RMSE':>15} {'MAE':>15} {'R2':>15}")
    print(f"{'-'*80}")
    for variant in args.variants:
        if variant in all_results:
            r = all_results[variant]
            print(f"{variant:<15} {r['num_params']:>10,d} "
                  f"{r['test_rmse_mean']:>7.4f}±{r['test_rmse_std']:<6.4f} "
                  f"{r['test_mae_mean']:>7.4f}±{r['test_mae_std']:<6.4f} "
                  f"{r['test_r2_mean']:>7.4f}±{r['test_r2_std']:<6.4f}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
