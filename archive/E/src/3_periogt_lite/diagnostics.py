"""
Diagnostics and analysis of PerioGT-Lite results.
Generates plots and summary statistics.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_pretrain_log():
    """Plot pre-training loss curve and cosine similarity evolution."""
    log_path = os.path.join(OUTPUT_DIR, 'pretrain_log.json')
    if not os.path.exists(log_path):
        print("No pretrain log found, skipping pretrain plots")
        return

    with open(log_path) as f:
        log = json.load(f)

    metrics = log['epoch_metrics']
    epochs = [m['epoch'] for m in metrics]
    losses = [m['avg_loss'] for m in metrics]
    cos_sims = [m['avg_cosine_sim'] for m in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('InfoNCE Loss')
    ax1.set_title('Pre-training Loss Curve')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, cos_sims, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Avg Cosine Similarity (positive pairs)')
    ax2.set_title('Positive Pair Cosine Similarity')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diagnostic_pretrain.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved pretrain diagnostic plot")


def plot_finetune_comparison():
    """Plot learning curves for baseline vs pre-trained models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, color, ls in [('Baseline', 'blue', '-'), ('Pre-trained', 'red', '--')]:
        fname = f'results_{name.lower().replace("-", "")}.json'
        fpath = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(fpath):
            continue

        with open(fpath) as f:
            data = json.load(f)

        metrics = data['epoch_metrics']
        epochs = [m['epoch'] for m in metrics]
        train_rmses = [m['train_rmse'] for m in metrics]
        val_rmses = [m['val_rmse'] for m in metrics]

        ax1.plot(epochs, train_rmses, color=color, linestyle=ls, linewidth=1.5,
                label=f'{name} (train)')
        ax1.plot(epochs, val_rmses, color=color, linestyle='-', linewidth=1.5,
                alpha=0.6, label=f'{name} (val)')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Learning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bar chart comparison
    baseline_test = None
    pretrained_test = None
    for name in ['Baseline', 'Pretrained']:
        fname = f'results_{name.lower()}.json'
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            r = data['results']
            if name == 'Baseline':
                baseline_test = r['test_rmse']
            else:
                pretrained_test = r['test_rmse']

    if baseline_test is not None and pretrained_test is not None:
        bars = ax2.bar(['Baseline', 'PerioGT-Lite'], [baseline_test, pretrained_test],
                      color=['blue', 'red'], alpha=0.7)
        ax2.set_ylabel('Test RMSE')
        ax2.set_title('Test RMSE Comparison')
        for bar, val in zip(bars, [baseline_test, pretrained_test]):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diagnostic_finetune.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved finetune diagnostic plot")


def plot_predictions():
    """Plot predicted vs actual Tg values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, name in zip(axes, ['Baseline', 'Pretrained']):
        fname = f'results_{name.lower()}.json'
        fpath = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(fpath):
            continue

        with open(fpath) as f:
            data = json.load(f)

        preds = np.array(data['test_preds'])
        targets = np.array(data['test_targets'])

        ax.scatter(targets, preds, alpha=0.3, s=10)
        lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
        ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect')
        ax.set_xlabel('Actual Tg')
        ax.set_ylabel('Predicted Tg')
        ax.set_title(f'{name} (Test Set)')

        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        ax.text(0.05, 0.95, f'RMSE: {rmse:.2f}', transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diagnostic_predictions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved predictions diagnostic plot")


def generate_report():
    """Generate final text report."""
    report = []
    report.append("=" * 60)
    report.append("PerioGT-Lite Experiment Report")
    report.append("=" * 60)

    # Pre-training summary
    pretrain_log_path = os.path.join(OUTPUT_DIR, 'pretrain_log.json')
    if os.path.exists(pretrain_log_path):
        with open(pretrain_log_path) as f:
            pt_log = json.load(f)
        report.append(f"\n--- Pre-training ---")
        report.append(f"  Best epoch: {pt_log['best_epoch']}")
        report.append(f"  Best InfoNCE loss: {pt_log['best_loss']:.4f}")
        report.append(f"  Dataset size: {pt_log['config']['subset_size']}")
        report.append(f"  Effective batch size: {pt_log['config']['effective_batch_size']}")

        metrics = pt_log['epoch_metrics']
        first_cos = metrics[0]['avg_cosine_sim']
        last_cos = metrics[-1]['avg_cosine_sim']
        report.append(f"  Initial cosine sim: {first_cos:.4f}")
        report.append(f"  Final cosine sim: {last_cos:.4f}")

    # Baseline results
    baseline_path = os.path.join(OUTPUT_DIR, 'results_baseline.json')
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            bl = json.load(f)
        r = bl['results']
        report.append(f"\n--- Baseline (no pre-training) ---")
        report.append(f"  Train RMSE: {r['train_rmse']:.4f}")
        report.append(f"  Val RMSE:   {r['val_rmse']:.4f}")
        report.append(f"  Test RMSE:  {r['test_rmse']:.4f}")
        report.append(f"  Best epoch: {r['best_epoch']}")
        report.append(f"  Overfit gap: {r['overfitting_gap']:.4f}")

    # Pre-trained results
    pretrained_path = os.path.join(OUTPUT_DIR, 'results_pretrained.json')
    if os.path.exists(pretrained_path):
        with open(pretrained_path) as f:
            pt = json.load(f)
        r = pt['results']
        report.append(f"\n--- PerioGT-Lite (pre-trained) ---")
        report.append(f"  Train RMSE: {r['train_rmse']:.4f}")
        report.append(f"  Val RMSE:   {r['val_rmse']:.4f}")
        report.append(f"  Test RMSE:  {r['test_rmse']:.4f}")
        report.append(f"  Best epoch: {r['best_epoch']}")
        report.append(f"  Overfit gap: {r['overfitting_gap']:.4f}")

    # Comparison
    if os.path.exists(baseline_path) and os.path.exists(pretrained_path):
        bl_rmse = json.load(open(baseline_path))['results']['test_rmse']
        pt_rmse = json.load(open(pretrained_path))['results']['test_rmse']
        delta = pt_rmse - bl_rmse
        report.append(f"\n--- Comparison ---")
        report.append(f"  Delta (pretrained - baseline): {delta:+.4f}")
        if delta < 0:
            report.append(f"  Pre-training IMPROVED test RMSE by {-delta:.4f}")
        else:
            report.append(f"  Pre-training HURT test RMSE by {delta:.4f}")

    report_text = "\n".join(report)
    print(report_text)

    with open(os.path.join(OUTPUT_DIR, 'report.txt'), 'w') as f:
        f.write(report_text)


if __name__ == '__main__':
    print("Generating diagnostics...")
    plot_pretrain_log()
    plot_finetune_comparison()
    plot_predictions()
    generate_report()
    print("Done!")
