#!/usr/bin/env python
"""
GRIN + ELENE Readout Variants Experiment on {1,3}-RU Mixed Data.

5 conditions (A-E) x 3 seeds = 15 runs.
All conditions use Rich node features and {1,3}-RU mixed augmentation.

Condition A: GAT + mean readout (baseline)
Condition B: GAT + max readout (no L1)
Condition C: GAT + max readout + L1 sparsity (GRIN-style)
Condition D: GAT + mean readout + ELENE edge encodings
Condition E: GAT + max readout + L1 sparsity + ELENE edge encodings
"""

from __future__ import annotations

import os
import sys
import time
import random
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

# Local imports
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

from data_utils import build_datasets, create_dataloaders, get_feature_dims
from models import create_model


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DGX_CSV = os.path.join(os.path.dirname(WORK_DIR), 'alldataset', 'Tg.csv')

CONDITIONS = ['A', 'B', 'C', 'D', 'E']
SEEDS = [71, 72, 73]
MULTIMER = '{1,3}-RU mixed'
FEATURE_LEVEL = 'rich'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_EPOCHS = 300
PATIENCE = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
AMP_ENABLED = True
L1_LAMBDA = 1e-3
L1_WARMUP_EPOCHS = 50

LOG_FILE = os.path.join(WORK_DIR, 'experiment_log.txt')
RESULTS_CSV = os.path.join(WORK_DIR, 'results.csv')
CURVES_PNG = os.path.join(WORK_DIR, 'training_curves.png')
BAR_PNG = os.path.join(WORK_DIR, 'rmse_comparison.png')
SUMMARY_FILE = os.path.join(WORK_DIR, 'summary.txt')

CONDITION_LABELS = {
    'A': 'GAT+mean',
    'B': 'GAT+max',
    'C': 'GAT+max+L1',
    'D': 'GAT+mean+ELENE',
    'E': 'GAT+max+L1+ELENE',
}


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

def train_epoch(model, loader, optimizer, scaler_amp, criterion, device,
                epoch: int, use_l1: bool = False):
    model.train()
    total_loss = 0.0
    total_l1 = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        if AMP_ENABLED:
            with autocast('cuda'):
                out = model(batch.x, batch.edge_index, batch.batch,
                            edge_attr=getattr(batch, 'edge_attr', None))
                mse_loss = criterion(out, batch.y)
                loss = mse_loss
                if use_l1 and epoch > L1_WARMUP_EPOCHS:
                    l1_penalty = model.get_l1_penalty()
                    loss = loss + L1_LAMBDA * l1_penalty
                    total_l1 += l1_penalty.item() * batch.num_graphs
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            out = model(batch.x, batch.edge_index, batch.batch,
                        edge_attr=getattr(batch, 'edge_attr', None))
            mse_loss = criterion(out, batch.y)
            loss = mse_loss
            if use_l1 and epoch > L1_WARMUP_EPOCHS:
                l1_penalty = model.get_l1_penalty()
                loss = loss + L1_LAMBDA * l1_penalty
                total_l1 += l1_penalty.item() * batch.num_graphs
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    avg_loss = total_loss / len(loader.dataset)
    avg_l1 = total_l1 / len(loader.dataset) if total_l1 > 0 else 0.0
    return avg_loss, avg_l1


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
    condition: str,
    seed: int,
    device,
    log_fp,
    dataset_cache: dict,
) -> dict:
    """Train a single (condition, seed) configuration."""
    set_seed(seed)

    label = CONDITION_LABELS[condition]
    use_elene = (condition in ('D', 'E'))
    use_l1 = (condition in ('C', 'E'))

    msg = f"\n{'='*60}\n[{datetime.now()}] Starting: {label} (cond={condition}), seed={seed}\n{'='*60}"
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.flush()

    # Build datasets (cached per condition + seed)
    cache_key = (condition, seed)
    if cache_key in dataset_cache:
        train_graphs, val_graphs, test_graphs, scaler = dataset_cache[cache_key]
    else:
        train_graphs, val_graphs, test_graphs, scaler = build_datasets(
            DGX_CSV, MULTIMER, FEATURE_LEVEL, seed=seed, elene=use_elene
        )
        dataset_cache[cache_key] = (train_graphs, val_graphs, test_graphs, scaler)
        print(f"  Built dataset: cond={condition}, seed={seed}, "
              f"elene={use_elene}, n_train={len(train_graphs)}, "
              f"n_val={len(val_graphs)}, n_test={len(test_graphs)}")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_graphs, val_graphs, test_graphs, scaler, BATCH_SIZE
    )

    node_dim, edge_dim = get_feature_dims(FEATURE_LEVEL, elene=use_elene)
    model = create_model(condition, node_dim=node_dim, edge_dim=edge_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scaler_amp = GradScaler('cuda') if AMP_ENABLED else None

    best_val_rmse = float('inf')
    best_epoch = 0
    best_state = None
    train_rmse_hist = []
    val_rmse_hist = []
    l1_penalty_hist = []
    sparsity_hist = []

    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, l1_val = train_epoch(
            model, train_loader, optimizer, scaler_amp, criterion,
            device, epoch=epoch, use_l1=use_l1
        )
        val_mse, val_rmse_raw, _, _ = evaluate(model, val_loader, criterion, device)

        train_rmse = np.sqrt(train_loss)
        train_rmse_hist.append(train_rmse)
        val_rmse_hist.append(val_rmse_raw)

        if use_l1:
            l1_penalty_hist.append(l1_val)
            sparsity_hist.append(model.get_sparsity_fraction())

        if val_rmse_raw < best_val_rmse:
            best_val_rmse = val_rmse_raw
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            waited = 0
        else:
            waited = epoch - best_epoch

        if epoch % 20 == 0 or waited >= PATIENCE:
            elapsed = time.time() - t0
            parts = [f"epoch {epoch:3d}/{MAX_EPOCHS}",
                     f"train_rmse={train_rmse:.4f}",
                     f"val_rmse={val_rmse_raw:.4f}",
                     f"best={best_val_rmse:.4f}@{best_epoch}",
                     f"waited={waited}",
                     f"time={elapsed:.0f}s"]
            if use_l1 and epoch > L1_WARMUP_EPOCHS:
                parts.append(f"L1={l1_val:.6f}")
                parts.append(f"sparsity={sparsity_hist[-1]:.4f}")
            info = "  " + " | ".join(parts)
            print(info)
            log_fp.write(info + '\n')
            log_fp.flush()

        if waited >= PATIENCE:
            break

    # Restore best model
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Final evaluation
    train_mse, train_rmse, _, _ = evaluate(model, train_loader, criterion, device)
    val_mse, val_rmse, _, _ = evaluate(model, val_loader, criterion, device)
    test_mse, test_rmse, test_preds, test_targets = evaluate(model, test_loader, criterion, device)

    # Inverse-transform to original scale
    train_rmse_orig = float(train_rmse * scaler.scale_[0])
    val_rmse_orig = float(val_rmse * scaler.scale_[0])
    test_rmse_orig = float(test_rmse * scaler.scale_[0])
    test_preds_orig = test_preds * scaler.scale_[0] + scaler.mean_[0]
    test_targets_orig = test_targets * scaler.scale_[0] + scaler.mean_[0]

    test_r2 = float(r2_score(test_targets_orig, test_preds_orig))
    test_mae = float(mean_absolute_error(test_targets_orig, test_preds_orig))

    # Final sparsity
    final_sparsity = model.get_sparsity_fraction() if use_l1 else None

    elapsed = time.time() - t0
    msg = (f"[{datetime.now()}] Finished: {label} (cond={condition}), seed={seed} | "
           f"test_rmse={test_rmse_orig:.3f} | test_mae={test_mae:.3f} | "
           f"r2={test_r2:.4f} | best_epoch={best_epoch} | "
           f"total_epochs={epoch} | time={elapsed:.0f}s")
    if use_l1:
        msg += f" | final_sparsity={final_sparsity:.4f}"
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.flush()

    result = {
        'condition': condition,
        'condition_label': label,
        'seed': seed,
        'train_rmse': train_rmse_orig,
        'val_rmse': val_rmse_orig,
        'test_rmse': test_rmse_orig,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'best_epoch': best_epoch,
        'total_epochs': epoch,
        'training_time_s': elapsed,
        'train_rmse_hist': train_rmse_hist,
        'val_rmse_hist': val_rmse_hist,
        'l1_penalty_hist': l1_penalty_hist,
        'sparsity_hist': sparsity_hist,
        'final_sparsity': final_sparsity,
        'test_preds_orig': test_preds_orig.tolist(),
        'test_targets_orig': test_targets_orig.tolist(),
    }
    return result


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_all_experiments():
    os.makedirs(WORK_DIR, exist_ok=True)
    dataset_cache = {}

    with open(LOG_FILE, 'w') as log_fp:
        log_fp.write(f"Experiment started: {datetime.now()}\n")
        log_fp.write(f"Device: {DEVICE}\n")
        log_fp.write(f"Configurations: {len(CONDITIONS)} conditions x {len(SEEDS)} seeds = {len(CONDITIONS)*len(SEEDS)} runs\n")
        log_fp.write(f"Multimer: {MULTIMER}, Feature: {FEATURE_LEVEL}\n")
        log_fp.write(f"Max epochs: {MAX_EPOCHS}, patience: {PATIENCE}, lr: {LR}\n")
        log_fp.write(f"Batch size: {BATCH_SIZE}, L1 lambda: {L1_LAMBDA}, L1 warmup: {L1_WARMUP_EPOCHS}\n")
        log_fp.write("=" * 80 + "\n")
        log_fp.flush()

        all_results = []

        for condition in CONDITIONS:
            for seed in SEEDS:
                result = train_one_config(
                    condition, seed, DEVICE, log_fp, dataset_cache
                )
                all_results.append(result)

                # Save intermediate results
                save_df = pd.DataFrame([{
                    'condition': r['condition'],
                    'condition_label': r['condition_label'],
                    'seed': r['seed'],
                    'test_rmse': r['test_rmse'],
                    'test_mae': r['test_mae'],
                    'test_r2': r['test_r2'],
                    'best_epoch': r['best_epoch'],
                    'training_time_s': r['training_time_s'],
                    'final_sparsity': r.get('final_sparsity', None),
                } for r in all_results])
                save_df.to_csv(RESULTS_CSV, index=False)

        log_fp.write("=" * 80 + "\n")
        log_fp.write(f"All experiments finished: {datetime.now()}\n")

    return all_results


# ---------------------------------------------------------------------------
# Analysis and visualization
# ---------------------------------------------------------------------------

def generate_analysis(all_results):
    """Generate all required outputs: CSV, summary, plots."""
    os.makedirs(WORK_DIR, exist_ok=True)

    # Save final CSV
    save_df = pd.DataFrame([{
        'condition': r['condition'],
        'condition_label': r['condition_label'],
        'seed': r['seed'],
        'test_rmse': r['test_rmse'],
        'test_mae': r['test_mae'],
        'test_r2': r['test_r2'],
        'best_epoch': r['best_epoch'],
        'training_time_s': r['training_time_s'],
        'final_sparsity': r.get('final_sparsity', None),
    } for r in all_results])
    save_df.to_csv(RESULTS_CSV, index=False)

    # Compute aggregated stats
    agg = save_df.groupby(['condition', 'condition_label']).agg(
        test_rmse_mean=('test_rmse', 'mean'),
        test_rmse_std=('test_rmse', 'std'),
        test_mae_mean=('test_mae', 'mean'),
        test_mae_std=('test_mae', 'std'),
        test_r2_mean=('test_r2', 'mean'),
        test_r2_std=('test_r2', 'std'),
        best_epoch_mean=('best_epoch', 'mean'),
        training_time_mean=('training_time_s', 'mean'),
    ).reset_index()

    # --- Generate summary.txt ---
    with open(SUMMARY_FILE, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GRIN + ELENE Readout Variants Experiment - SUMMARY\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Experiment Configuration:\n")
        f.write(f"  Multimer: {MULTIMER} (1:1 mixed augmentation)\n")
        f.write(f"  Feature level: {FEATURE_LEVEL} (Rich: 47 node + 10 edge)\n")
        f.write(f"  Backbone: 3-layer GAT, 128 hidden, 4 heads, dropout=0.1\n")
        f.write(f"  Train/Val/Test split: 72/8/20, seed=71\n")
        f.write(f"  Optimizer: Adam, lr={LR}, weight_decay={WEIGHT_DECAY}\n")
        f.write(f"  Batch size: {BATCH_SIZE}, Max epochs: {MAX_EPOCHS}, Patience: {PATIENCE}\n")
        f.write(f"  L1 lambda: {L1_LAMBDA}, L1 warmup: {L1_WARMUP_EPOCHS} epochs\n")
        f.write(f"  Seeds: {SEEDS} (3 per condition)\n")
        f.write(f"  Test: 1-RU graphs (standard test)\n\n")

        f.write("=" * 80 + "\n")
        f.write("1. RESULTS TABLE (mean +/- std over 3 seeds)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Condition':<25} {'Test RMSE':>14} {'Test MAE':>12} {'Test R2':>10} {'Best Ep':>8} {'Time(s)':>10}\n")
        f.write("-" * 80 + "\n")
        for _, row in agg.iterrows():
            rmse = f"{row['test_rmse_mean']:.2f}±{row['test_rmse_std']:.2f}"
            mae = f"{row['test_mae_mean']:.2f}±{row['test_mae_std']:.2f}"
            f.write(f"{row['condition_label']:<25} {rmse:>14} {mae:>12} "
                    f"{row['test_r2_mean']:>8.4f}±{row['test_r2_std']:.4f} "
                    f"{row['best_epoch_mean']:>7.0f} {row['training_time_mean']:>9.0f}\n")

        # Sparsity info for L1 conditions
        f.write("\nL1 Sparsity Analysis:\n")
        f.write("-" * 40 + "\n")
        for cond in ['C', 'E']:
            subset = [r for r in all_results if r['condition'] == cond]
            if subset and subset[0]['final_sparsity'] is not None:
                sparsities = [r['final_sparsity'] for r in subset]
                f.write(f"  {CONDITION_LABELS[cond]}: final sparsity = "
                        f"{np.mean(sparsities):.4f} ± {np.std(sparsities):.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("2. HYPOTHESIS TESTS\n")
        f.write("=" * 80 + "\n\n")

        # Get mean values
        means = {row['condition']: row['test_rmse_mean'] for _, row in agg.iterrows()}

        # 1. B vs A: max alone
        f.write("a) B vs A: Does max readout alone help with {1,3}-RU mixed?\n")
        if 'A' in means and 'B' in means:
            diff_BA = means['B'] - means['A']
            verdict = "YES (max better)" if diff_BA < 0 else "NO (max worse/same)"
            f.write(f"   {CONDITION_LABELS['A']}: {means['A']:.2f}\n")
            f.write(f"   {CONDITION_LABELS['B']}: {means['B']:.2f}\n")
            f.write(f"   Difference: {diff_BA:+.2f} K\n")
            f.write(f"   Verdict: {verdict}\n\n")

        # 2. C vs B: L1 on top of max
        f.write("b) C vs B: Does L1 sparsity on top of max provide additional benefit?\n")
        if 'B' in means and 'C' in means:
            diff_CB = means['C'] - means['B']
            verdict = "YES (L1 helps)" if diff_CB < 0 else "NO (L1 doesn't help)"
            f.write(f"   {CONDITION_LABELS['B']}: {means['B']:.2f}\n")
            f.write(f"   {CONDITION_LABELS['C']}: {means['C']:.2f}\n")
            f.write(f"   Difference: {diff_CB:+.2f} K\n")
            f.write(f"   Verdict: {verdict}\n\n")

        # 3. C vs A: GRIN rescue vs mean
        f.write("c) C vs A: Does GRIN-style (max+L1) rescue {1,3}-RU mixed vs mean?\n")
        if 'A' in means and 'C' in means:
            diff_CA = means['C'] - means['A']
            verdict = "YES (GRIN rescues)" if diff_CA < 0 else "NO (GRIN doesn't rescue)"
            f.write(f"   {CONDITION_LABELS['A']}: {means['A']:.2f}\n")
            f.write(f"   {CONDITION_LABELS['C']}: {means['C']:.2f}\n")
            f.write(f"   Difference: {diff_CA:+.2f} K\n")
            f.write(f"   Verdict: {verdict}\n\n")

        # 4. D vs A: ELENE with mean
        f.write("d) D vs A: Do ELENE edge encodings improve mean readout?\n")
        if 'A' in means and 'D' in means:
            diff_DA = means['D'] - means['A']
            verdict = "YES (ELENE helps)" if diff_DA < 0 else "NO (ELENE doesn't help)"
            f.write(f"   {CONDITION_LABELS['A']}: {means['A']:.2f}\n")
            f.write(f"   {CONDITION_LABELS['D']}: {means['D']:.2f}\n")
            f.write(f"   Difference: {diff_DA:+.2f} K\n")
            f.write(f"   Verdict: {verdict}\n\n")

        # 5. E vs C: ELENE beyond GRIN
        f.write("e) E vs C: Do ELENE edge encodings add value beyond GRIN-style?\n")
        if 'C' in means and 'E' in means:
            diff_EC = means['E'] - means['C']
            verdict = "YES (ELENE adds value)" if diff_EC < 0 else "NO (ELENE doesn't add value)"
            f.write(f"   {CONDITION_LABELS['C']}: {means['C']:.2f}\n")
            f.write(f"   {CONDITION_LABELS['E']}: {means['E']:.2f}\n")
            f.write(f"   Difference: {diff_EC:+.2f} K\n")
            f.write(f"   Verdict: {verdict}\n\n")

        # 6. Comparison against baselines
        f.write("=" * 80 + "\n")
        f.write("3. COMPARISON AGAINST DAY 1 BASELINES\n")
        f.write("=" * 80 + "\n\n")
        f.write("  Pure 3-RU baseline (Day 1, GAT+mean+rich): 41.6 RMSE\n")
        f.write("  {1,3}-RU mixed baseline (Day 1, GAT+mean+rich): 43.6 RMSE\n\n")

        best_cond = min(means, key=means.get)
        f.write(f"  Best condition: {CONDITION_LABELS[best_cond]} = {means[best_cond]:.2f} RMSE\n")
        f.write(f"  vs pure 3-RU (41.6): {means[best_cond] - 41.6:+.2f} K\n")
        f.write(f"  vs mixed mean baseline (43.6): {means[best_cond] - 43.6:+.2f} K\n\n")

        # Overall verdict
        f.write("=" * 80 + "\n")
        f.write("4. OVERALL VERDICT\n")
        f.write("=" * 80 + "\n\n")
        verdicts = []
        if 'B' in means and 'A' in means and means['B'] < means['A']:
            verdicts.append("Max readout improves over mean readout on {1,3}-RU mixed data")
        if 'C' in means and 'B' in means and means['C'] < means['B']:
            verdicts.append("L1 sparsity provides additional benefit on top of max readout")
        if 'C' in means and 'A' in means and means['C'] < means['A']:
            verdicts.append("GRIN-style (max+L1) successfully rescues {1,3}-RU mixed augmentation")
        if 'D' in means and 'A' in means and means['D'] < means['A']:
            verdicts.append("ELENE edge encodings improve performance with mean readout")
        if 'E' in means and 'C' in means and means['E'] < means['C']:
            verdicts.append("ELENE edge encodings provide complementary gains beyond GRIN-style")

        if not verdicts:
            verdicts.append("None of the tested hypotheses were supported by the data")
        for v in verdicts:
            f.write(f"  * {v}\n")

    print(f"\nSummary saved to {SUMMARY_FILE}")

    # --- Training curves ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    axes_flat = axes.flatten()
    for i, condition in enumerate(CONDITIONS):
        ax = axes_flat[i]
        subset = [r for r in all_results if r['condition'] == condition]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for j, r in enumerate(subset):
            epochs = range(1, len(r['train_rmse_hist']) + 1)
            ax.plot(epochs, r['train_rmse_hist'], alpha=0.4, color=colors[j], linestyle='--')
            ax.plot(epochs, r['val_rmse_hist'], alpha=0.4, color=colors[j], linestyle='-')
        ax.set_title(CONDITION_LABELS[condition], fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE (scaled)')
        ax.grid(True, alpha=0.3)
    # Hide extra subplot if any
    if len(CONDITIONS) < len(axes_flat):
        axes_flat[-1].set_visible(False)
    plt.suptitle('Training Curves: GRIN + ELENE Readout Variants\n(all on {1,3}-RU mixed, Rich features)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CURVES_PNG, dpi=150)
    plt.close()
    print(f"Training curves saved to {CURVES_PNG}")

    # --- RMSE bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    order = CONDITION_LABELS
    labels = [CONDITION_LABELS[c] for c in CONDITIONS]
    means_list = [agg[agg['condition'] == c]['test_rmse_mean'].values[0] for c in CONDITIONS]
    stds_list = [agg[agg['condition'] == c]['test_rmse_std'].values[0] for c in CONDITIONS]

    bars = ax.bar(range(len(CONDITIONS)), means_list, yerr=stds_list, capsize=8,
                  color=['#3498db', '#e74c3c', '#e67e22', '#2ecc71', '#9b59b6'],
                  edgecolor='black', linewidth=1)

    # Add baselines
    ax.axhline(y=43.6, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='{1,3}-RU mixed mean baseline (43.6)')
    ax.axhline(y=41.6, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Pure 3-RU baseline (41.6)')

    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels(labels, fontsize=11, rotation=15)
    ax.set_ylabel('Test RMSE (K)', fontsize=12)
    ax.set_title('Test RMSE Comparison: GRIN + ELENE Readout Variants\n'
                 '({1,3}-RU mixed training, Rich features, mean±std over 3 seeds)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, mean_val in zip(bars, means_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{mean_val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(BAR_PNG, dpi=150)
    plt.close()
    print(f"RMSE comparison bar chart saved to {BAR_PNG}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"Working dir: {WORK_DIR}")
    print(f"Data CSV: {DGX_CSV}")
    print(f"Conditions: {len(CONDITIONS)} x {len(SEEDS)} seeds = {len(CONDITIONS)*len(SEEDS)} runs")
    print(f"Condition labels: {[CONDITION_LABELS[c] for c in CONDITIONS]}")
    print()

    results = run_all_experiments()
    generate_analysis(results)
    print("\n=== Done ===")
