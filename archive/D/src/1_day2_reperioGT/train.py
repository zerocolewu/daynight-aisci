"""
Training script for RE-PerioGT
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from dataset import load_and_split_data, PolymerDataset, get_feature_dimensions
from model import RE_PerioGT, ContrastiveLoss, count_parameters


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()
    predictions = []
    targets = []
    rigidities = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            pred, _ = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                batch.rigidity,
                batch.periodic_mask,
                batch.periodic_vectors,
            )

            predictions.extend(pred.cpu().numpy().flatten())
            targets.extend(batch.y.cpu().numpy().flatten())
            rigidities.extend(batch.rigidity.cpu().numpy().flatten())

    predictions = np.array(predictions)
    targets = np.array(targets)
    rigidities = np.array(rigidities)

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
        'rigidities': rigidities,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    contrastive_loss_fn: Optional[ContrastiveLoss] = None,
    lambda_cl: float = 0.1,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_cl_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        pred, embedding = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.rigidity,
            batch.periodic_mask,
            batch.periodic_vectors,
        )

        # Prediction loss
        pred_loss = nn.functional.mse_loss(pred.squeeze(), batch.y.squeeze())

        # Contrastive loss
        cl_loss = torch.tensor(0.0, device=device)
        if contrastive_loss_fn is not None and model.use_contrastive:
            projection = model.get_projection(embedding)
            cl_loss = contrastive_loss_fn(projection, batch.rigidity)

        # Total loss
        loss = pred_loss + lambda_cl * cl_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_pred_loss += pred_loss.item()
        total_cl_loss += cl_loss.item()
        num_batches += 1

    return total_loss / num_batches, total_pred_loss / num_batches, total_cl_loss / num_batches


def train_model(
    train_dataset: PolymerDataset,
    val_dataset: PolymerDataset,
    test_dataset: PolymerDataset,
    args: argparse.Namespace,
) -> Dict:
    """Full training loop."""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    atom_dim, bond_dim = get_feature_dimensions()
    model = RE_PerioGT(
        atom_dim=atom_dim,
        bond_dim=bond_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_rigidity_feature=args.use_rigidity_feature,
        use_contrastive=args.use_contrastive,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Use rigidity feature: {args.use_rigidity_feature}")
    print(f"Use contrastive: {args.use_contrastive}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Contrastive loss
    contrastive_loss_fn = None
    if args.use_contrastive:
        contrastive_loss_fn = ContrastiveLoss(
            temperature=args.cl_temperature,
            pos_threshold=args.cl_pos_threshold,
            neg_threshold=args.cl_neg_threshold,
        ).to(device)

    # Training tracking
    history = {
        'train_loss': [],
        'train_pred_loss': [],
        'train_cl_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'val_r2': [],
        'lr': [],
    }

    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, pred_loss, cl_loss = train_epoch(
            model, train_loader, optimizer, device,
            contrastive_loss_fn, args.lambda_cl
        )

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Record history
        history['train_loss'].append(train_loss)
        history['train_pred_loss'].append(pred_loss)
        history['train_cl_loss'].append(cl_loss)
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_r2'].append(val_metrics['r2'])
        history['lr'].append(current_lr)

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} (pred: {pred_loss:.4f}, cl: {cl_loss:.4f}) | "
                  f"Val RMSE: {val_metrics['rmse']:.2f} | MAE: {val_metrics['mae']:.2f} | "
                  f"R2: {val_metrics['r2']:.4f} | LR: {current_lr:.2e}")

        # Early stopping
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    print(f"Best validation RMSE: {best_val_rmse:.2f} at epoch {best_epoch}")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    test_metrics = evaluate(model, test_loader, device)

    print(f"\n=== Test Set Results ===")
    print(f"RMSE: {test_metrics['rmse']:.2f}")
    print(f"MAE: {test_metrics['mae']:.2f}")
    print(f"R2: {test_metrics['r2']:.4f}")
    print(f"Pearson R: {test_metrics['pearson_r']:.4f}")

    results = {
        'best_epoch': best_epoch,
        'best_val_rmse': best_val_rmse,
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_r2': test_metrics['r2'],
        'test_pearson_r': test_metrics['pearson_r'],
        'training_time': training_time,
        'history': history,
        'test_predictions': test_metrics['predictions'].tolist(),
        'test_targets': test_metrics['targets'].tolist(),
        'test_rigidities': test_metrics['rigidities'].tolist(),
    }

    return results


def plot_results(results: Dict, output_dir: str):
    """Generate result plots."""
    history = results['history']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curve
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['train_pred_loss'], label='Pred Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation RMSE
    ax = axes[0, 1]
    ax.plot(history['val_rmse'], label='Val RMSE')
    ax.axhline(y=results['best_val_rmse'], color='r', linestyle='--', label=f'Best: {results["best_val_rmse"]:.2f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('Validation RMSE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Predicted vs Actual
    ax = axes[1, 0]
    predictions = np.array(results['test_predictions'])
    targets = np.array(results['test_targets'])
    ax.scatter(targets, predictions, alpha=0.5, s=20)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    ax.set_xlabel('Actual Tg')
    ax.set_ylabel('Predicted Tg')
    ax.set_title(f'Test: RMSE={results["test_rmse"]:.2f}, R²={results["test_r2"]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residuals
    ax = axes[1, 1]
    residuals = predictions - targets
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--')
    ax.set_xlabel('Residual (Predicted - Actual)')
    ax.set_ylabel('Count')
    ax.set_title(f'Residual Distribution (mean={residuals.mean():.2f})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_results.png'), dpi=150)
    plt.close()

    # Additional plot: Residuals vs Rigidity
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    rigidities = np.array(results['test_rigidities'])
    ax.scatter(rigidities, residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Rotatable Bond Fraction')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Rigidity')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'residuals_vs_rigidity.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train RE-PerioGT')

    # Data
    parser.add_argument('--data_path', type=str,
                        default='/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv')
    parser.add_argument('--output_dir', type=str,
                        default='/home/home_public/aisci/AIscientist/src/1_day2_reperioGT')
    parser.add_argument('--seed', type=int, default=42)

    # Model
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_rigidity_feature', action='store_true', default=True)
    parser.add_argument('--use_contrastive', action='store_true', default=True)
    parser.add_argument('--no_rigidity_feature', dest='use_rigidity_feature', action='store_false')
    parser.add_argument('--no_contrastive', dest='use_contrastive', action='store_false')

    # Contrastive learning
    parser.add_argument('--lambda_cl', type=float, default=0.1)
    parser.add_argument('--cl_temperature', type=float, default=0.1)
    parser.add_argument('--cl_pos_threshold', type=float, default=0.1)
    parser.add_argument('--cl_neg_threshold', type=float, default=0.3)

    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=50)

    # Experiment mode
    parser.add_argument('--mode', type=str, default='full', choices=['baseline', 'feature_only', 'full'],
                        help='baseline: no rigidity, feature_only: rigidity as feature, full: with contrastive')

    args = parser.parse_args()

    # Set mode-specific settings
    if args.mode == 'baseline':
        args.use_rigidity_feature = False
        args.use_contrastive = False
        args.output_dir = os.path.join(args.output_dir, 'results_baseline')
    elif args.mode == 'feature_only':
        args.use_rigidity_feature = True
        args.use_contrastive = False
        args.output_dir = os.path.join(args.output_dir, 'results_feature_only')
    else:  # full
        args.use_rigidity_feature = True
        args.use_contrastive = True
        args.output_dir = os.path.join(args.output_dir, 'results_full')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 60)
    print(f"RE-PerioGT Training - Mode: {args.mode}")
    print("=" * 60)
    print(f"Use rigidity feature: {args.use_rigidity_feature}")
    print(f"Use contrastive: {args.use_contrastive}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_ds, val_ds, test_ds, df = load_and_split_data(
        args.data_path,
        seed=args.seed
    )

    # Train model
    results = train_model(train_ds, val_ds, test_ds, args)

    # Save results
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        # Convert numpy arrays and non-serializable objects
        results_to_save = {}
        for k, v in results.items():
            if k == 'history':
                results_to_save['history'] = {kk: [float(x) for x in vv] for kk, vv in v.items()}
            elif k in ['test_predictions', 'test_targets', 'test_rigidities']:
                results_to_save[k] = [float(x) for x in v]
            elif isinstance(v, (np.floating, np.integer)):
                results_to_save[k] = float(v)
            else:
                results_to_save[k] = v
        json.dump(results_to_save, f, indent=2)

    # Generate plots
    plot_results(results, args.output_dir)

    print(f"\nResults saved to {args.output_dir}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
