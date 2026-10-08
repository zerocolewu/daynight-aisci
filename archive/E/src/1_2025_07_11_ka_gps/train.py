"""
Training script for KA-GPS and MLP-GPS models on Tg dataset.

Includes training loop, evaluation, visualization, and result reporting.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.data import Batch

from dataset import (
    get_data_loaders, ATOM_FEATURE_DIM, EDGE_FEATURE_DIM, RWSE_DIM,
    psmiles_to_graph, TgDataset, collate_fn
)
from model import KAGPS, MLPGPS, count_parameters, FourierKANLayer

# ── Reproducibility ──
SEED = 71
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

CSV_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
WORK_DIR = '/home/home_public/aisci/AIscientist/src/1_2025_07_11_ka_gps'

# ── Hyperparameters ──
HIDDEN_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 4
N_FREQ = 5
DROPOUT = 0.1
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 500
PATIENCE = 30


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)
        target = batch.y.view(-1)

        # Handle potential NaN in targets
        valid_mask = ~torch.isnan(target)
        if valid_mask.sum() == 0:
            continue

        loss = criterion(pred[valid_mask], target[valid_mask])
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * valid_mask.sum().item()
        total_samples += valid_mask.sum().item()

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        target = batch.y.view(-1)

        valid_mask = ~torch.isnan(target)
        if valid_mask.sum() == 0:
            continue

        loss = criterion(pred[valid_mask], target[valid_mask])
        total_loss += loss.item() * valid_mask.sum().item()
        total_samples += valid_mask.sum().item()

        all_preds.append(pred[valid_mask].cpu())
        all_targets.append(target[valid_mask].cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    mse = total_loss / max(total_samples, 1)
    rmse = np.sqrt(mse)

    # Also compute MAE
    mae = (all_preds - all_targets).abs().mean().item()

    return rmse, mae, all_preds.numpy(), all_targets.numpy()


def train_model(model_type='kan', hidden_dim=128, num_layers=4,
                num_heads=4, n_freq=5, dropout=0.1,
                lr=1e-3, weight_decay=1e-5, batch_size=32,
                max_epochs=500, patience=30, seed=71):
    """Train a model and return results."""

    node_dim = ATOM_FEATURE_DIM + RWSE_DIM
    edge_dim = EDGE_FEATURE_DIM

    # Create model
    if model_type == 'kan':
        model = KAGPS(node_dim, edge_dim, hidden_dim, num_layers,
                       num_heads, n_freq, dropout).to(DEVICE)
        model_name = 'KA-GPS'
    else:
        model = MLPGPS(node_dim, edge_dim, hidden_dim, num_layers,
                        num_heads, dropout).to(DEVICE)
        model_name = 'MLP-GPS'

    n_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Training {model_name} | Parameters: {n_params:,}")
    print(f"{'='*60}")

    # Data
    train_loader, val_loader, test_loader, train_idx, val_idx, test_idx = \
        get_data_loaders(CSV_PATH, seed=seed, batch_size=batch_size)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )

    criterion = nn.MSELoss()

    # Training loop
    best_val_rmse = float('inf')
    best_epoch = 0
    train_rmses = []
    val_rmses = []
    epochs_no_improve = 0

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        train_rmse = np.sqrt(train_loss)

        val_rmse, val_mae, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_rmse)

        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            epochs_no_improve = 0
            # Save best model
            save_path = os.path.join(WORK_DIR, f'best_model_{model_type}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse,
            }, save_path)
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:4d} | Train RMSE: {train_rmse:.4f} | "
                  f"Val RMSE: {val_rmse:.4f} | Best Val: {best_val_rmse:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed:.1f}s")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Load best model and evaluate on test set
    save_path = os.path.join(WORK_DIR, f'best_model_{model_type}.pt')
    checkpoint = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_rmse, test_mae, test_preds, test_targets = evaluate(
        model, test_loader, criterion, DEVICE
    )

    total_time = time.time() - start_time
    print(f"\n{model_name} Final Results:")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Val RMSE: {best_val_rmse:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  Total Time: {total_time:.1f}s")
    print(f"  Parameters: {n_params:,}")

    return {
        'model_name': model_name,
        'model_type': model_type,
        'n_params': n_params,
        'best_epoch': best_epoch,
        'best_val_rmse': float(best_val_rmse),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'train_rmses': [float(x) for x in train_rmses],
        'val_rmses': [float(x) for x in val_rmses],
        'test_preds': test_preds.tolist(),
        'test_targets': test_targets.tolist(),
        'total_time': total_time,
    }


def plot_training_curves(kan_results, mlp_results):
    """Plot training curves for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # KA-GPS
    ax = axes[0]
    epochs_kan = range(1, len(kan_results['train_rmses']) + 1)
    ax.plot(epochs_kan, kan_results['train_rmses'], label='Train RMSE', alpha=0.8)
    ax.plot(epochs_kan, kan_results['val_rmses'], label='Val RMSE', alpha=0.8)
    ax.axhline(y=kan_results['best_val_rmse'], color='r', linestyle='--',
               alpha=0.5, label=f'Best Val: {kan_results["best_val_rmse"]:.2f}')
    ax.axhline(y=30.8, color='g', linestyle=':', alpha=0.5, label='PerioGT SOTA: 30.8')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title(f"KA-GPS Training (Test RMSE: {kan_results['test_rmse']:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MLP-GPS
    ax = axes[1]
    epochs_mlp = range(1, len(mlp_results['train_rmses']) + 1)
    ax.plot(epochs_mlp, mlp_results['train_rmses'], label='Train RMSE', alpha=0.8)
    ax.plot(epochs_mlp, mlp_results['val_rmses'], label='Val RMSE', alpha=0.8)
    ax.axhline(y=mlp_results['best_val_rmse'], color='r', linestyle='--',
               alpha=0.5, label=f'Best Val: {mlp_results["best_val_rmse"]:.2f}')
    ax.axhline(y=30.8, color='g', linestyle=':', alpha=0.5, label='PerioGT SOTA: 30.8')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title(f"MLP-GPS Training (Test RMSE: {mlp_results['test_rmse']:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(WORK_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_scatter(kan_results, mlp_results):
    """Plot prediction scatter plots for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, results, color in zip(axes,
                                   [kan_results, mlp_results],
                                   ['steelblue', 'coral']):
        preds = np.array(results['test_preds'])
        targets = np.array(results['test_targets'])

        ax.scatter(targets, preds, alpha=0.5, s=20, c=color, edgecolors='k',
                   linewidths=0.3)

        # Perfect prediction line
        min_val = min(targets.min(), preds.min())
        max_val = max(targets.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8,
                label='Perfect prediction')

        rmse = results['test_rmse']
        ax.set_xlabel('True Tg (°C)', fontsize=12)
        ax.set_ylabel('Predicted Tg (°C)', fontsize=12)
        ax.set_title(f"{results['model_name']} (RMSE: {rmse:.2f})", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add text box with stats
        textstr = f'RMSE = {rmse:.2f}\nMAE = {results["test_mae"]:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

    plt.tight_layout()
    save_path = os.path.join(WORK_DIR, 'scatter_plots.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scatter plots saved to {save_path}")


def visualize_kan_functions(model_path, n_freq=5):
    """Visualize learned KAN functions from a trained KA-GPS model."""
    node_dim = ATOM_FEATURE_DIM + RWSE_DIM
    edge_dim = EDGE_FEATURE_DIM

    model = KAGPS(node_dim, edge_dim, HIDDEN_DIM, NUM_LAYERS,
                   NUM_HEADS, N_FREQ, DROPOUT).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Extract KAN layers from the model
    kan_layers = []
    for layer in model.layers:
        kan_layers.append(layer.local_nn.block.kan1)
        kan_layers.append(layer.local_nn.block.kan2)

    # Visualize learned functions from first few KAN layers
    x_range = torch.linspace(-2, 2, 200).to(DEVICE)

    for idx, kan in enumerate(kan_layers[:6]):
        ax = axes[idx]

        # Pick a few (input_dim, output_dim) pairs to visualize
        d_in = min(kan.d_in, 3)
        d_out = min(kan.d_out, 3)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        func_count = 0
        for i in range(d_in):
            for j in range(d_out):
                if func_count >= 5:
                    break
                x_input = torch.zeros(200, kan.d_in).to(DEVICE)
                x_input[:, i] = x_range

                with torch.no_grad():
                    out = kan(x_input)

                ax.plot(x_range.cpu().numpy(), out[:, j].cpu().numpy(),
                        color=colors[func_count % len(colors)],
                        label=f'φ_{{{i},{j}}}', alpha=0.8)
                func_count += 1
            if func_count >= 5:
                break

        ax.set_title(f'KAN Layer {idx + 1} Functions')
        ax.set_xlabel('x')
        ax.set_ylabel('φ(x)')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Learned Fourier KAN Functions in KA-GPS', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(WORK_DIR, 'kan_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"KAN visualization saved to {save_path}")


def main():
    print("=" * 60)
    print("KA-GPS: Kolmogorov-Arnold Enhanced Graph Transformer")
    print("for Polymer Tg Prediction")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Hidden dim: {HIDDEN_DIM}")
    print(f"Num layers: {NUM_LAYERS}")
    print(f"Num heads: {NUM_HEADS}")
    print(f"N_freq (KAN): {N_FREQ}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Max epochs: {MAX_EPOCHS}")
    print(f"Early stopping patience: {PATIENCE}")
    print(f"Seed: {SEED}")

    # ── Train MLP-GPS baseline ──
    print("\n\n" + "=" * 60)
    print("Phase 1: Training MLP-GPS Baseline")
    print("=" * 60)
    mlp_results = train_model(
        model_type='mlp',
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        seed=SEED,
    )

    # ── Train KA-GPS ──
    print("\n\n" + "=" * 60)
    print("Phase 2: Training KA-GPS")
    print("=" * 60)
    kan_results = train_model(
        model_type='kan',
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        n_freq=N_FREQ,
        dropout=DROPOUT,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        seed=SEED,
    )

    # ── Generate visualizations ──
    print("\n\n" + "=" * 60)
    print("Phase 3: Generating Visualizations")
    print("=" * 60)
    plot_training_curves(kan_results, mlp_results)
    plot_scatter(kan_results, mlp_results)

    # KAN function visualization
    kan_model_path = os.path.join(WORK_DIR, 'best_model_kan.pt')
    if os.path.exists(kan_model_path):
        visualize_kan_functions(kan_model_path, N_FREQ)

    # ── Save results ──
    results = {
        'ka_gps': {
            'test_rmse': kan_results['test_rmse'],
            'test_mae': kan_results['test_mae'],
            'best_val_rmse': kan_results['best_val_rmse'],
            'best_epoch': kan_results['best_epoch'],
            'n_params': kan_results['n_params'],
        },
        'mlp_gps': {
            'test_rmse': mlp_results['test_rmse'],
            'test_mae': mlp_results['test_mae'],
            'best_val_rmse': mlp_results['best_val_rmse'],
            'best_epoch': mlp_results['best_epoch'],
            'n_params': mlp_results['n_params'],
        },
        'baselines': {
            'gat_rmse': 50.5,
            'periogt_rmse': 30.8,
        },
        'config': {
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'num_heads': NUM_HEADS,
            'n_freq': N_FREQ,
            'dropout': DROPOUT,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
            'max_epochs': MAX_EPOCHS,
            'patience': PATIENCE,
            'seed': SEED,
        }
    }

    results_path = os.path.join(WORK_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── Print summary ──
    print("\n\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'Params':>10} {'Val RMSE':>10} {'Test RMSE':>10} {'Test MAE':>10}")
    print("-" * 55)
    print(f"{'KA-GPS':<15} {kan_results['n_params']:>10,} "
          f"{kan_results['best_val_rmse']:>10.2f} "
          f"{kan_results['test_rmse']:>10.2f} "
          f"{kan_results['test_mae']:>10.2f}")
    print(f"{'MLP-GPS':<15} {mlp_results['n_params']:>10,} "
          f"{mlp_results['best_val_rmse']:>10.2f} "
          f"{mlp_results['test_rmse']:>10.2f} "
          f"{mlp_results['test_mae']:>10.2f}")
    print("-" * 55)
    print(f"{'GAT baseline':<15} {'':>10} {'':>10} {'50.5':>10}")
    print(f"{'PerioGT SOTA':<15} {'':>10} {'':>10} {'30.8':>10}")
    print("=" * 60)

    if kan_results['test_rmse'] < mlp_results['test_rmse']:
        print("KA-GPS outperforms MLP-GPS! KAN layers provide benefit.")
    else:
        print("MLP-GPS outperforms KA-GPS. KAN layers did not help in this case.")

    print("\nDone!")


if __name__ == '__main__':
    main()
