"""
Training pipeline for GPS-Lite experiments.

Three variants:
  1. baseline: GAT+RWPE+PSPE (no global attention)
  2. no_bias:  GAT+RWPE+PSPE + global attention WITHOUT SPD bias
  3. spd:      GAT+RWPE+PSPE + global attention WITH SPD bias (primary)
"""

import os
import sys
import json
import time
import random
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_utils import (
    build_datasets, load_rwpe, assign_rwpe_to_graphs,
    assemble_features, get_input_dim, EDGE_DIM_RICH,
    create_dataloaders,
)
from model import create_model

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIG = {
    'csv_path': os.path.join(BASE_DIR, '..', 'alldataset', 'Tg.csv'),
    'rwpe_path': os.path.join(BASE_DIR, '..', '2_day_2_lspe_gat', 'rwpe_k32.pkl'),
    'batch_size': 64,
    'max_epochs': 300,
    'patience': 50,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'hidden_dim': 256,
    'heads': 4,
    'head_dim': 64,
    'dropout': 0.1,
    'max_grad_norm': 1.0,
    'use_amp': True,
    'seed': 71,
    'split_seed': 71,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

print(f"Device: {CONFIG['device']}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def extract_spd_list(batch, spd_map):
    """Extract per-graph SPD matrices using graph_id lookup.

    Since SPD matrices [N_i, N_i] can't be collated by PyG,
    we look them up from a pre-built dictionary using graph_id.
    """
    spd_list = []
    for i in range(batch.num_graphs):
        gid = batch.graph_id[i].item()
        spd_list.append(spd_map[gid].to(batch.x.device))
    return spd_list


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scaler_amp, criterion, device, variant, spd_map):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        x = assemble_features(batch)

        optimizer.zero_grad()

        with autocast(enabled=CONFIG['use_amp']):
            if variant == 'baseline':
                out = model(x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
            else:
                spd_list = extract_spd_list(batch, spd_map)
                out, _ = model(x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, spd_list=spd_list)

            loss = criterion(out, batch.y.view(-1))

        # Skip NaN/Inf gradients
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  WARNING: NaN/Inf loss in training — skipping batch")
            optimizer.zero_grad()
            continue

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
        scaler_amp.step(optimizer)
        scaler_amp.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, scaler_obj, variant, spd_map, collect_attn=False):
    """Evaluate model, return scaled loss + unscaled metrics. Optionally collect attention weights."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    n_batches = 0
    all_attn_weights = [] if collect_attn else None

    for batch in loader:
        batch = batch.to(device)
        x = assemble_features(batch)

        if variant == 'baseline':
            out = model(x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
        else:
            spd_list = extract_spd_list(batch, spd_map)
            out, attn_w = model(x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, spd_list=spd_list)
            if collect_attn:
                all_attn_weights.append((attn_w, spd_list))

        loss = criterion(out, batch.y.view(-1))
        total_loss += loss.item()
        n_batches += 1

        all_preds.append(out.cpu())
        all_targets.append(batch.y.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy().flatten()

    # NaN safety check
    if np.isnan(preds).any() or np.isinf(preds).any():
        print(f"  WARNING: NaN/Inf in predictions — returning sentinel values")
        result = (float('nan'), float('inf'), float('inf'), float('nan'),
                  preds, targets)
        if collect_attn:
            result = result + (all_attn_weights,)
        return result

    # Unscale
    preds_unscaled = scaler_obj.inverse_transform(preds.reshape(-1, 1)).flatten()
    targets_unscaled = scaler_obj.inverse_transform(targets.reshape(-1, 1)).flatten()

    rmse = np.sqrt(np.mean((preds_unscaled - targets_unscaled) ** 2))
    mae = mean_absolute_error(targets_unscaled, preds_unscaled)
    r2 = r2_score(targets_unscaled, preds_unscaled)

    result = (total_loss / max(n_batches, 1), rmse, mae, r2, preds_unscaled, targets_unscaled)
    if collect_attn:
        result = result + (all_attn_weights,)
    return result


# ---------------------------------------------------------------------------
# Main training run
# ---------------------------------------------------------------------------

def run_training(variant, train_loader, val_loader, test_loader, scaler, spd_maps):
    """Run full training pipeline for one variant."""
    print(f"\n{'='*60}")
    print(f"Training variant: {variant}")
    print(f"{'='*60}")

    set_seed(CONFIG['seed'])

    in_dim = get_input_dim()
    device = torch.device(CONFIG['device'])

    model = create_model(
        variant=variant,
        in_dim=in_dim,
        edge_dim=EDGE_DIM_RICH,
        hidden_dim=CONFIG['hidden_dim'],
        heads=CONFIG['heads'],
        head_dim=CONFIG['head_dim'],
        dropout=CONFIG['dropout'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} parameters, input_dim={in_dim}")

    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['max_epochs'], eta_min=1e-6)
    scaler_amp = GradScaler(enabled=CONFIG['use_amp'])
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_epoch = 0
    best_state = None
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'val_r2': [],
        'lr': [],
    }

    start_time = time.time()

    for epoch in range(1, CONFIG['max_epochs'] + 1):
        epoch_start = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, scaler_amp, criterion, device, variant, spd_maps['train'],
        )
        val_loss, val_rmse, val_mae, val_r2, _, _ = evaluate(
            model, val_loader, criterion, device, scaler, variant, spd_maps['val'],
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_rmse'].append(float(val_rmse))
        history['val_mae'].append(float(val_mae))
        history['val_r2'].append(float(val_r2))
        history['lr'].append(float(current_lr))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 20 == 0 or patience_counter == 0:
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val RMSE: {val_rmse:.2f}K | Val MAE: {val_mae:.2f}K | "
                  f"Val R²: {val_r2:.4f} | LR: {current_lr:.2e} | "
                  f"Time: {time.time()-epoch_start:.1f}s")

        if patience_counter >= CONFIG['patience']:
            print(f"  Early stopping at epoch {epoch}")
            break

    train_time = time.time() - start_time

    # Load best model for test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_rmse, test_mae, test_r2, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device, scaler, variant, spd_maps['test'],
    )

    print(f"\n  Best epoch: {best_epoch}")
    print(f"  Test  RMSE: {test_rmse:.2f}K")
    print(f"  Test  MAE:  {test_mae:.2f}K")
    print(f"  Test  R²:   {test_r2:.4f}")
    print(f"  Time: {train_time:.1f}s")

    # Collect attention weights for diagnostic (test set, best model)
    attn_data = None
    if variant in ('no_bias', 'spd'):
        _, _, _, _, _, _, all_attn = evaluate(
            model, test_loader, criterion, device, scaler, variant, spd_maps['test'],
            collect_attn=True,
        )
        attn_data = _process_attention_data(all_attn)

    # Save model
    model_path = os.path.join(RESULTS_DIR, f"model_{variant}.pt")
    torch.save({
        'model_state_dict': {k: v.cpu().clone() for k, v in best_state.items()} if best_state else {},
        'variant': variant,
        'in_dim': in_dim,
        'best_val_rmse': best_val_rmse,
        'best_epoch': best_epoch,
    }, model_path)

    results = {
        'variant': variant,
        'seed': CONFIG['seed'],
        'split_seed': CONFIG['split_seed'],
        'in_dim': in_dim,
        'n_params': n_params,
        'best_epoch': best_epoch,
        'best_val_rmse': float(best_val_rmse),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'train_time': train_time,
        'history': history,
        'test_targets': test_targets.tolist(),
        'test_predictions': test_preds.tolist(),
    }

    results_path = os.path.join(RESULTS_DIR, f"results_{variant}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results, attn_data, history


def _process_attention_data(all_attn):
    """Process collected attention weights: bin by SPD distance."""
    spd_bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 1000)]
    bin_labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30+']
    bin_attn = {label: [] for label in bin_labels}

    all_spd_values = []

    for batch_attn_weights, batch_spd_list in all_attn:
        for attn_w, spd in zip(batch_attn_weights, batch_spd_list):
            # attn_w: [H, N, N] average over heads
            avg_attn = attn_w.mean(dim=0).cpu().numpy()  # [N, N]
            spd_np = spd.cpu().numpy()  # [N, N]

            all_spd_values.extend(spd_np[spd_np < np.inf].flatten().tolist())

            for bin_idx, (lo, hi) in enumerate(spd_bins):
                mask = (spd_np >= lo) & (spd_np < hi)
                if mask.sum() > 0:
                    bin_attn[bin_labels[bin_idx]].extend(avg_attn[mask].tolist())

    # Compute mean attention per bin
    bin_means = {label: float(np.mean(vals)) if vals else 0.0 for label, vals in bin_attn.items()}
    bin_stds = {label: float(np.std(vals)) if vals else 0.0 for label, vals in bin_attn.items()}

    return {
        'bin_attention_mean': bin_means,
        'bin_attention_std': bin_stds,
        'all_spd_values': all_spd_values,
    }


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_attention_analysis(attn_data_spd, attn_data_nobias, save_path):
    """Plot attention weights binned by SPD distance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bin_labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30+']

    for ax, (attn_data, title) in zip(
        axes,
        [(attn_data_spd, 'GPS-Lite-SPD (with SPD bias)'),
         (attn_data_nobias, 'GPS-Lite-noBias (no SPD bias)')]
    ):
        if attn_data is None:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(title)
            continue

        means = [attn_data['bin_attention_mean'].get(l, 0) for l in bin_labels]
        stds = [attn_data['bin_attention_std'].get(l, 0) for l in bin_labels]
        x = np.arange(len(bin_labels))

        ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=45)
        ax.set_xlabel('SPD Distance')
        ax.set_ylabel('Mean Attention Weight')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved attention analysis: {save_path}")


def plot_spd_histogram(all_spd_values, save_path):
    """Plot SPD distribution."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    spd_vals = np.array(all_spd_values)
    spd_vals = spd_vals[spd_vals < np.inf]

    ax.hist(spd_vals, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Shortest Path Distance')
    ax.set_ylabel('Count')
    ax.set_title('SPD Distribution Across Test Set')
    ax.axvline(x=np.mean(spd_vals), color='red', linestyle='--',
               label=f'Mean: {np.mean(spd_vals):.1f}')
    ax.axvline(x=np.median(spd_vals), color='green', linestyle='--',
               label=f'Median: {np.median(spd_vals):.1f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved SPD histogram: {save_path}")


def plot_training_curves(histories, save_path):
    """Plot training curves for all variants."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'baseline': 'blue', 'no_bias': 'orange', 'spd': 'green'}
    labels = {'baseline': 'Baseline (GAT+RWPE+PSPE)',
              'no_bias': 'GPS-Lite-noBias',
              'spd': 'GPS-Lite-SPD'}

    for variant, history in histories.items():
        color = colors.get(variant, 'gray')
        label = labels.get(variant, variant)
        epochs = range(1, len(history['train_loss']) + 1)

        axes[0, 0].plot(epochs, history['train_loss'], color=color, label=label, alpha=0.7)
        axes[0, 1].plot(epochs, history['val_loss'], color=color, label=label)
        axes[1, 0].plot(epochs, history['val_rmse'], color=color, label=label)
        if 'lr' in history:
            axes[1, 1].plot(epochs, history['lr'], color=color, label=label)

    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Train Loss'); axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Val Loss'); axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Val RMSE (K)'); axes[1, 0].set_title('Validation RMSE')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('LR'); axes[1, 1].set_title('Learning Rate')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("GPS-Lite: Single Global Attention Layer atop GAT Backbone")
    print("=" * 60)

    # ---- Data Preparation ----
    print("\n[1] Building datasets...")
    train_graphs, val_graphs, test_graphs, scaler, metadata = build_datasets(
        CONFIG['csv_path'], seed=CONFIG['split_seed'],
    )
    print(f"  Train: {metadata['num_train']}, Val: {metadata['num_val']}, Test: {metadata['num_test']}")

    # Load and assign RWPE
    print("\n[2] Loading RWPE...")
    rwpe_data = load_rwpe(CONFIG['rwpe_path'])
    n_rwpe_train, n_fb_train = assign_rwpe_to_graphs(train_graphs, rwpe_data, offset=0)
    offset_val = metadata['num_train']
    n_rwpe_val, n_fb_val = assign_rwpe_to_graphs(val_graphs, rwpe_data, offset=offset_val)
    offset_test = metadata['num_train'] + metadata['num_val']
    n_rwpe_test, n_fb_test = assign_rwpe_to_graphs(test_graphs, rwpe_data, offset=offset_test)
    print(f"  RWPE assigned: train={n_rwpe_train}/{n_fb_train} fallbacks, "
          f"val={n_rwpe_val}/{n_fb_val}, test={n_rwpe_test}/{n_fb_test}")

    # Create dataloaders (now returns spd_maps)
    train_loader, val_loader, test_loader, spd_maps = create_dataloaders(
        train_graphs, val_graphs, test_graphs, scaler, batch_size=CONFIG['batch_size'],
    )
    print(f"  Dataloaders created: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)} batches")

    # ---- Training ----
    # Skip baseline if already completed (load from saved results)
    baseline_path = os.path.join(RESULTS_DIR, 'results_baseline.json')
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            all_results = {'baseline': json.load(f)}
            all_histories = {'baseline': all_results['baseline'].get('history', {})}
        variants = ['no_bias', 'spd']
        print(f"\n  Loaded baseline results, training remaining variants: {variants}")
    else:
        all_results = {}
        all_histories = {}
        variants = ['baseline', 'no_bias', 'spd']
    attn_data_spd = None
    attn_data_nobias = None

    for variant in variants:
        try:
            results, attn_data, history = run_training(
                variant, train_loader, val_loader, test_loader, scaler, spd_maps,
            )
            all_results[variant] = results
            all_histories[variant] = history

            if variant == 'spd':
                attn_data_spd = attn_data
            elif variant == 'no_bias':
                attn_data_nobias = attn_data

        except Exception as e:
            print(f"\n  ERROR training {variant}: {e}")
            import traceback
            traceback.print_exc()
            all_results[variant] = {'variant': variant, 'error': str(e)}

    # ---- Summary ----
    print(f"\n\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    summary = {}
    for variant in variants:
        r = all_results.get(variant, {})
        rmse = r.get('test_rmse', float('nan'))
        mae = r.get('test_mae', float('nan'))
        r2 = r.get('test_r2', float('nan'))
        print(f"\n  {variant}:")
        if 'error' in r:
            print(f"    ERROR: {r['error']}")
        else:
            print(f"    Test RMSE: {rmse:.2f} K")
            print(f"    Test MAE:  {mae:.2f} K")
            print(f"    Test R²:   {r2:.4f}")
            print(f"    Best Epoch: {r.get('best_epoch', 'N/A')}")
            print(f"    Params: {r.get('n_params', 'N/A'):,}")
        summary[variant] = {'test_rmse': rmse, 'test_mae': mae, 'test_r2': r2}

    # Compute delta RMSE
    baseline_rmse = summary.get('baseline', {}).get('test_rmse', float('nan'))
    if not np.isnan(baseline_rmse):
        for v in ['no_bias', 'spd']:
            if v in summary and not np.isnan(summary[v].get('test_rmse', float('nan'))):
                delta = baseline_rmse - summary[v]['test_rmse']
                summary[v]['delta_rmse_vs_baseline'] = float(delta)
                print(f"\n  ΔRMSE ({v} - baseline): {delta:+.2f} K "
                      f"({'improvement' if delta > 0 else 'degradation'})")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, 'results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    # ---- Diagnostic Plots ----
    print(f"\n[3] Generating diagnostic plots...")

    # SPD histogram from test set
    all_spd_test = []
    for g in test_graphs:
        spd_np = g.spd.numpy().flatten()
        all_spd_test.extend(spd_np[(spd_np < np.inf) & (spd_np > 0)].tolist())
    plot_spd_histogram(all_spd_test, os.path.join(RESULTS_DIR, 'spd_histogram.png'))

    # Attention analysis
    plot_attention_analysis(attn_data_spd, attn_data_nobias,
                           os.path.join(RESULTS_DIR, 'attention_analysis.png'))

    # Training curves
    plot_training_curves(all_histories, os.path.join(RESULTS_DIR, 'training_curves.png'))

    # ---- Final Report ----
    print(f"\n{'='*60}")
    print("FINAL REPORT")
    print(f"{'='*60}")
    for variant in variants:
        r = all_results.get(variant, {})
        if 'error' in r:
            print(f"  {variant}: FAILED - {r['error']}")
        else:
            print(f"  {variant}: RMSE={r['test_rmse']:.2f}K, MAE={r['test_mae']:.2f}K, R²={r['test_r2']:.4f}")

    if not np.isnan(baseline_rmse):
        spd_rmse = summary.get('spd', {}).get('test_rmse', float('nan'))
        nobias_rmse = summary.get('no_bias', {}).get('test_rmse', float('nan'))
        print(f"\n  ΔRMSE (noBias vs baseline): {baseline_rmse - nobias_rmse:+.2f} K")
        print(f"  ΔRMSE (SPD vs baseline): {baseline_rmse - spd_rmse:+.2f} K")

    return summary


if __name__ == '__main__':
    main()
