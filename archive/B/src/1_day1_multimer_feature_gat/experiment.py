#!/usr/bin/env python
"""
Data Centric Foundation Experiment: Multimer Construction x Node Feature Richness
for Tg Prediction using a fixed GAT architecture.

12 configurations (4 multimer x 3 feature) x 3 seeds = 36 runs.
"""

from __future__ import annotations

import os
import sys
import time
import random
import json
import itertools
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# Ensure we can import local modules
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

from data_utils import (
    build_datasets, create_dataloaders, get_feature_dims,
)
from model import GATModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DGX_CSV = os.path.join(os.path.dirname(WORK_DIR), 'alldataset', 'Tg.csv')

MULTIMERS = ['1-RU', '2-RU', '3-RU', '{1,3}-RU mixed']
FEATURE_LEVELS = ['basic', 'standard', 'rich']
SEEDS = [71, 72, 73]

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_EPOCHS = 300
PATIENCE = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 64
AMP_ENABLED = True  # mixed precision

LOG_FILE = os.path.join(WORK_DIR, 'experiment_log.txt')
RESULTS_CSV = os.path.join(WORK_DIR, 'results_summary.csv')
HEATMAP_PNG = os.path.join(WORK_DIR, 'rmse_heatmap.png')
CURVES_PNG = os.path.join(WORK_DIR, 'learning_curves.png')
SCATTER_PNG = os.path.join(WORK_DIR, 'best_model_scatter.png')
REPORT_TXT = os.path.join(WORK_DIR, 'analysis_report.txt')
CURVES_JSON = os.path.join(WORK_DIR, 'learning_curves.json')


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        if AMP_ENABLED:
            with autocast('cuda'):
                out = model(batch.x, batch.edge_index, batch.batch,
                            edge_attr=getattr(batch, 'edge_attr', None))
                loss = criterion(out, batch.y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(batch.x, batch.edge_index, batch.batch,
                        edge_attr=getattr(batch, 'edge_attr', None))
            loss = criterion(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch,
                    edge_attr=getattr(batch, 'edge_attr', None))
        loss = criterion(out, batch.y)
        total_loss += loss.item() * batch.num_graphs
        all_preds.append(out.cpu())
        all_targets.append(batch.y.cpu())

    mse = total_loss / len(loader.dataset)
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    rmse = np.sqrt(mse)
    return mse, rmse, preds, targets


def train_one_config(
    multimer: str,
    feature_level: str,
    seed: int,
    train_graphs, val_graphs, test_graphs, scaler,
    device, log_fp,
) -> dict:
    """Train a single (multimer, feature, seed) configuration."""
    set_seed(seed)

    msg = f"[{datetime.now()}] Starting: multimer={multimer}, feature={feature_level}, seed={seed}"
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.flush()

    train_loader, val_loader, test_loader = create_dataloaders(
        train_graphs, val_graphs, test_graphs, scaler, BATCH_SIZE
    )

    node_dim, edge_dim = get_feature_dims(feature_level)
    model = GATModel(node_dim=node_dim, edge_dim=edge_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scaler_amp = GradScaler('cuda') if AMP_ENABLED else None

    best_val_rmse = float('inf')
    best_epoch = 0
    best_state = None
    train_rmse_hist = []
    val_rmse_hist = []

    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_mse = train_epoch(model, train_loader, optimizer, scaler_amp, criterion, device)
        val_mse, val_rmse_raw, _, _ = evaluate(model, val_loader, criterion, device)

        train_rmse = np.sqrt(train_mse)
        train_rmse_hist.append(train_rmse)
        val_rmse_hist.append(val_rmse_raw)

        if val_rmse_raw < best_val_rmse:
            best_val_rmse = val_rmse_raw
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            waited = 0
        else:
            waited = epoch - best_epoch

        if epoch % 10 == 0 or waited >= PATIENCE:
            elapsed = time.time() - t0
            info = (f"  epoch {epoch:3d}/{MAX_EPOCHS} | "
                    f"train_rmse={train_rmse:.4f} | val_rmse={val_rmse_raw:.4f} | "
                    f"best={best_val_rmse:.4f}@{best_epoch} | waited={waited} | "
                    f"time={elapsed:.0f}s")
            print(info)
            log_fp.write(info + '\n')
            log_fp.flush()

        if waited >= PATIENCE:
            break

    # Restore best model and evaluate on test
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    train_mse, train_rmse, _, _ = evaluate(model, train_loader, criterion, device)
    val_mse, val_rmse, _, _ = evaluate(model, val_loader, criterion, device)
    test_mse, test_rmse, test_preds, test_targets = evaluate(model, test_loader, criterion, device)

    # Inverse-transform to original scale
    train_rmse_orig = float(train_rmse * scaler.scale_[0])
    val_rmse_orig = float(val_rmse * scaler.scale_[0])
    test_rmse_orig = float(test_rmse * scaler.scale_[0])
    test_preds_orig = test_preds * scaler.scale_[0] + scaler.mean_[0]
    test_targets_orig = test_targets * scaler.scale_[0] + scaler.mean_[0]

    # Also compute R² and MAE on original scale
    test_r2 = float(r2_score(test_targets_orig, test_preds_orig))
    test_mae = float(mean_absolute_error(test_targets_orig, test_preds_orig))

    elapsed = time.time() - t0
    msg = (f"[{datetime.now()}] Finished: multimer={multimer}, feature={feature_level}, "
           f"seed={seed} | train_rmse={train_rmse_orig:.3f}, val_rmse={val_rmse_orig:.3f}, "
           f"test_rmse={test_rmse_orig:.3f} | r2={test_r2:.4f} | mae={test_mae:.3f} | "
           f"epochs={epoch} | time={elapsed:.0f}s")
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.flush()

    return {
        'multimer': multimer,
        'feature': feature_level,
        'seed': seed,
        'train_rmse': train_rmse_orig,
        'val_rmse': val_rmse_orig,
        'test_rmse': test_rmse_orig,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'best_epoch': best_epoch,
        'total_epochs': epoch,
        'train_rmse_hist': train_rmse_hist,
        'val_rmse_hist': val_rmse_hist,
        'test_preds_orig': test_preds_orig.tolist(),
        'test_targets_orig': test_targets_orig.tolist(),
        'time_s': elapsed,
    }


# ---------------------------------------------------------------------------
# Caching: build datasets once per (multimer, feature) pair
# ---------------------------------------------------------------------------

_dataset_cache = {}
_cached_scaler = {}

def get_or_build_datasets(multimer, feature_level):
    """Build fresh datasets for each seed to avoid y-value corruption.
    Caches the scaler (fit on first build) for consistency across seeds.
    """
    # Use a composite key that includes seed to force rebuild each time
    key = (multimer, feature_level)
    train_g, val_g, test_g, scaler = build_datasets(
        DGX_CSV, multimer, feature_level
    )
    if key in _cached_scaler:
        # Use the cached scaler from first build for reproducibility
        scaler = _cached_scaler[key]
    else:
        _cached_scaler[key] = scaler
    if key not in _dataset_cache:
        _dataset_cache[key] = True  # just mark as seen
        print(f"  Built dataset: multimer={multimer}, feature={feature_level}, "
              f"n_train={len(train_g)}, n_val={len(val_g)}, n_test={len(test_g)}")
    return train_g, val_g, test_g, scaler


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_all_experiments():
    os.makedirs(WORK_DIR, exist_ok=True)

    with open(LOG_FILE, 'w') as log_fp:
        log_fp.write(f"Experiment started: {datetime.now()}\n")
        log_fp.write(f"Device: {DEVICE}\n")
        log_fp.write(f"Configurations: {len(MULTIMERS)} multimer x {len(FEATURE_LEVELS)} feature x {len(SEEDS)} seeds = {len(MULTIMERS)*len(FEATURE_LEVELS)*len(SEEDS)} runs\n")
        log_fp.write(f"Max epochs: {MAX_EPOCHS}, patience: {PATIENCE}, lr: {LR}, batch: {BATCH_SIZE}\n")
        log_fp.write("=" * 80 + "\n")
        log_fp.flush()

        all_results = []

        for multimer in MULTIMERS:
            for feature_level in FEATURE_LEVELS:

                for seed in SEEDS:
                    # Build fresh datasets for each seed to avoid y-value corruption
                    train_graphs, val_graphs, test_graphs, scaler = get_or_build_datasets(
                        multimer, feature_level
                    )
                    result = train_one_config(
                        multimer, feature_level, seed,
                        train_graphs, val_graphs, test_graphs, scaler,
                        DEVICE, log_fp,
                    )
                    all_results.append(result)

                    # Save intermediate results
                    pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)

        log_fp.write("=" * 80 + "\n")
        log_fp.write(f"All experiments finished: {datetime.now()}\n")

    return all_results


# ---------------------------------------------------------------------------
# Analysis and visualization
# ---------------------------------------------------------------------------

def generate_analysis(all_results):
    """Generate all required outputs."""
    df = pd.DataFrame(all_results)

    # --- 1. Summary table ---
    # Aggregate mean±std across seeds
    agg = df.groupby(['multimer', 'feature']).agg(
        train_rmse_mean=('train_rmse', 'mean'),
        train_rmse_std=('train_rmse', 'std'),
        val_rmse_mean=('val_rmse', 'mean'),
        val_rmse_std=('val_rmse', 'std'),
        test_rmse_mean=('test_rmse', 'mean'),
        test_rmse_std=('test_rmse', 'std'),
        test_r2_mean=('test_r2', 'mean'),
        test_mae_mean=('test_mae', 'mean'),
    ).reset_index()

    agg.to_csv(RESULTS_CSV.replace('.csv', '_aggregated.csv'), index=False)
    print("\n=== Results Summary ===")
    print(agg.to_string(index=False))

    # --- 2. Heatmap ---
    pivot_test = df.pivot_table(values='test_rmse', index='multimer', columns='feature', aggfunc='mean')
    pivot_std = df.pivot_table(values='test_rmse', index='multimer', columns='feature', aggfunc='std')

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot_test.values, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(FEATURE_LEVELS)))
    ax.set_xticklabels(FEATURE_LEVELS, fontsize=11)
    ax.set_yticks(range(len(MULTIMERS)))
    ax.set_yticklabels(MULTIMERS, fontsize=11)
    for i in range(len(MULTIMERS)):
        for j in range(len(FEATURE_LEVELS)):
            val = pivot_test.values[i, j]
            std = pivot_std.values[i, j]
            text = f'{val:.1f}\n±{std:.1f}'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white' if val > np.nanmean(pivot_test.values) else 'black')
    plt.colorbar(im, ax=ax, label='Test RMSE (K)')
    ax.set_title('Test RMSE Heatmap: Multimer × Feature\n(mean ± std over 3 seeds)')
    plt.tight_layout()
    plt.savefig(HEATMAP_PNG, dpi=150)
    plt.close()

    # --- 3. Learning curves ---
    n_rows, n_cols = len(MULTIMERS), len(FEATURE_LEVELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for ri, multimer in enumerate(MULTIMERS):
        for ci, feature_level in enumerate(FEATURE_LEVELS):
            ax = axes[ri][ci]
            subset = [r for r in all_results
                      if r['multimer'] == multimer and r['feature'] == feature_level]
            for s in subset:
                epochs = range(1, len(s['train_rmse_hist']) + 1)
                ax.plot(epochs, s['train_rmse_hist'], alpha=0.3, color='blue')
                ax.plot(epochs, s['val_rmse_hist'], alpha=0.3, color='orange')
            ax.set_title(f'{multimer} × {feature_level}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('RMSE')
            ax.legend(['Train', 'Val'], loc='upper right', fontsize=8)
    plt.suptitle('Learning Curves (all configurations, 3 seeds each)')
    plt.tight_layout()
    plt.savefig(CURVES_PNG, dpi=150)
    plt.close()

    # Save learning curves data
    curves_data = {}
    for r in all_results:
        key = f"{r['multimer']}_{r['feature']}_seed{r['seed']}"
        curves_data[key] = {
            'train_rmse': r['train_rmse_hist'],
            'val_rmse': r['val_rmse_hist'],
        }
    with open(CURVES_JSON, 'w') as f:
        json.dump(curves_data, f)

    # --- 4. Best model scatter ---
    best_row = df.loc[df['test_rmse'].idxmin()]
    print(f"\nBest config: {best_row['multimer']} × {best_row['feature']} seed={best_row['seed']}")
    print(f"  Test RMSE: {best_row['test_rmse']:.3f} | R²: {best_row['test_r2']:.4f} | MAE: {best_row['test_mae']:.3f}")

    fig, ax = plt.subplots(figsize=(7, 7))
    preds = np.array(best_row['test_preds_orig'])
    targets = np.array(best_row['test_targets_orig'])
    ax.scatter(targets, preds, alpha=0.5, s=20)
    lim_min = min(targets.min(), preds.min())
    lim_max = max(targets.max(), preds.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.5)
    ax.set_xlabel('Actual Tg (K)')
    ax.set_ylabel('Predicted Tg (K)')
    ax.set_title(f"Best Model: {best_row['multimer']} × {best_row['feature']}\n"
                 f"Test RMSE={best_row['test_rmse']:.2f}K, R²={best_row['test_r2']:.4f}, MAE={best_row['test_mae']:.2f}K")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    plt.tight_layout()
    plt.savefig(SCATTER_PNG, dpi=150)
    plt.close()

    # --- 5. Analysis report ---
    with open(REPORT_TXT, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DIAGNOSTIC ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. MAIN RESULTS TABLE\n")
        f.write("-" * 40 + "\n")
        f.write("Format: Test RMSE (K), mean ± std across 3 seeds\n\n")
        f.write(f"{'Multimer':<20} {'Basic':>12} {'Standard':>12} {'Rich':>12} {'Avg':>10}\n")
        f.write("-" * 70 + "\n")
        row_avgs = {}
        col_sums = defaultdict(float)
        col_counts = defaultdict(int)
        for multimer in MULTIMERS:
            vals = []
            line = f"{multimer:<20}"
            for feat in FEATURE_LEVELS:
                subset = (agg[(agg['multimer'] == multimer) & (agg['feature'] == feat)])
                if len(subset) > 0:
                    val_mean = subset['test_rmse_mean'].values[0]
                    val_std = subset['test_rmse_std'].values[0]
                    line += f" {val_mean:7.1f}±{val_std:.1f}"
                    vals.append(val_mean)
                    col_sums[feat] += val_mean
                    col_counts[feat] += 1
                else:
                    line += f" {'N/A':>10}"
            avg = np.mean(vals) if vals else float('nan')
            row_avgs[multimer] = avg
            line += f" {avg:10.1f}"
            f.write(line + "\n")

        f.write("-" * 70 + "\n")
        avg_line = f"{'Avg':<20}"
        for feat in FEATURE_LEVELS:
            avg_feat = col_sums[feat] / col_counts[feat] if col_counts[feat] > 0 else float('nan')
            avg_line += f" {avg_feat:10.1f}"
        f.write(avg_line + "\n\n")

        # 2. Now the detailed analysis
        f.write("2. FEATURE ABLATION ANALYSIS (for {1,3}-RU mixed)\n")
        f.write("-" * 40 + "\n")
        f.write("Marginal gains from richer features:\n")
        mixed_subset = agg[agg['multimer'] == '{1,3}-RU mixed']
        prev_val = None
        for feat in ['basic', 'standard', 'rich']:
            row = mixed_subset[mixed_subset['feature'] == feat]
            if len(row) > 0:
                val = row['test_rmse_mean'].values[0]
                if prev_val is not None:
                    diff = prev_val - val
                    direction = "improvement" if diff > 0 else "degradation"
                    f.write(f"  {prev_feat} → {feat}: {prev_val:.1f} → {val:.1f} "
                            f"({diff:+.1f} K, {direction})\n")
                prev_val = val
                prev_feat = feat
        f.write("\n")

        f.write("3. MULTIMER ABLATION ANALYSIS (for rich features)\n")
        f.write("-" * 40 + "\n")
        f.write("Marginal gains from multimer augmentation:\n")
        rich_subset = agg[agg['feature'] == 'rich']
        prev_val = None
        for multimer in ['1-RU', '2-RU', '3-RU', '{1,3}-RU mixed']:
            row = rich_subset[rich_subset['multimer'] == multimer]
            if len(row) > 0:
                val = row['test_rmse_mean'].values[0]
                if prev_val is not None:
                    diff = prev_val - val
                    direction = "improvement" if diff > 0 else "degradation"
                    f.write(f"  {prev_multimer} → {multimer}: {prev_val:.1f} → {val:.1f} "
                            f"({diff:+.1f} K, {direction})\n")
                prev_val = val
                prev_multimer = multimer
        f.write("\n")

        f.write("4. GRIN HYPOTHESIS TEST\n")
        f.write("-" * 40 + "\n")
        f.write("Hypothesis: {1,3}-RU > 3-RU > 2-RU > 1-RU in test RMSE\n")
        rich_vals = []
        for multimer in ['1-RU', '2-RU', '3-RU', '{1,3}-RU mixed']:
            row = rich_subset[rich_subset['multimer'] == multimer]
            if len(row) > 0:
                rich_vals.append((multimer, row['test_rmse_mean'].values[0],
                                  row['test_rmse_std'].values[0]))
        f.write("  Rich feature results:\n")
        last_val = float('inf')
        hypothesis_holds = True
        for name, val, std in rich_vals:
            marker = "✓" if val < last_val else "✗"
            f.write(f"  {marker} {name}: {val:.1f} ± {std:.1f} K\n")
            if val >= last_val:
                hypothesis_holds = False
            last_val = val
        f.write(f"  Hypothesis holds: {'Yes' if hypothesis_holds else 'No'}\n\n")

        f.write("5. BEST OVERALL CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        best_agg = agg.loc[agg['test_rmse_mean'].idxmin()]
        f.write(f"  Configuration: {best_agg['multimer']} × {best_agg['feature']}\n")
        f.write(f"  Test RMSE: {best_agg['test_rmse_mean']:.2f} ± {best_agg['test_rmse_std']:.2f} K\n")
        f.write(f"  Test R²: {best_agg['test_r2_mean']:.4f}\n")
        f.write(f"  Test MAE: {best_agg['test_mae_mean']:.2f} K\n\n")
        f.write(f"  vs GAT baseline RMSE=50.5: ")
        diff = 50.5 - best_agg['test_rmse_mean']
        f.write(f"{'better' if diff > 0 else 'worse'} by {abs(diff):.1f} K\n\n")

        f.write("6. SUMMARY OF ALL CONFIGURATIONS\n")
        f.write("-" * 40 + "\n")
        for _, row in agg.iterrows():
            f.write(f"  {row['multimer']:>18} × {row['feature']:<8}: "
                    f"test_rmse={row['test_rmse_mean']:.1f}±{row['test_rmse_std']:.1f} K, "
                    f"R²={row['test_r2_mean']:.4f}\n")

    print(f"\nAnalysis report saved to {REPORT_TXT}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"Working dir: {WORK_DIR}")
    print(f"Data CSV: {DGX_CSV}")
    print(f"Configs: {len(MULTIMERS)} multimer × {len(FEATURE_LEVELS)} feature × {len(SEEDS)} seeds = {len(MULTIMERS)*len(FEATURE_LEVELS)*len(SEEDS)} runs")
    print()

    results = run_all_experiments()
    generate_analysis(results)
    print("\n=== Done ===")
