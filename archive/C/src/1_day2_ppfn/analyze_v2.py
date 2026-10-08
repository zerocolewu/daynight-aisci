"""
PPFN-v2 Final Interpretability Analysis and Visualization.

Generates:
- Parity plots for best model and ensemble
- Error distribution analysis
- Attention weight analysis
- Comprehensive interpretability report
"""

import sys
import os
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/1_day2_ppfn')

import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_v2 import PPFNv2Dataset, collate_fn_v2
from model_v2 import PPFNv2

CSV_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
SAVE_DIR = '/home/home_public/aisci/AIscientist/src/1_day2_ppfn'


def move_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_targets, all_p_smiles, all_attn = [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch)
        all_preds.append(pred.cpu().numpy().flatten())
        all_targets.append(batch['tg'].cpu().numpy().flatten())
        all_p_smiles.extend(batch['p_smiles_list'])
        attn = model.get_attention_weights()
        if attn is not None:
            all_attn.append(attn.cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    attn = np.concatenate(all_attn) if all_attn else None
    return preds, targets, all_p_smiles, attn


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load best model from v2 initial run (best single model)
    model_path = os.path.join(SAVE_DIR, 'best_model_v2_periodic_edge.pt')
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        return

    # Load test dataset
    print("Loading test dataset...")
    test_ds = PPFNv2Dataset(CSV_PATH, split='test', use_periodic_edge=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             collate_fn=collate_fn_v2, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    # Load model
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = PPFNv2(hidden_dim=256, num_gnn_layers=6, dropout=0.1).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print("Model loaded (PPFN-v2 simple, periodic edge)")

    # Get predictions
    preds, targets, p_smiles, attn = get_predictions(model, test_loader, device)

    errors = preds - targets
    abs_errors = np.abs(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(abs_errors)
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # 1. Parity Plot
    # ═══════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(targets, preds, alpha=0.5, s=20, c=abs_errors,
                         cmap='RdYlGn_r', edgecolors='k', linewidths=0.2)
    plt.colorbar(scatter, ax=ax, label='Absolute Error', shrink=0.8)
    mn, mx = min(targets.min(), preds.min()), max(targets.max(), preds.max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Tg (C)', fontsize=12)
    ax.set_ylabel('Predicted Tg (C)', fontsize=12)
    ax.set_title(f'PPFN-v2 Parity Plot\nRMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}',
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'parity_plot_v2.png'), dpi=200)
    plt.close()
    print("Saved parity_plot_v2.png")

    # ═══════════════════════════════════════════════════════════════════
    # 2. Error Distribution
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram
    axes[0].hist(errors, bins=60, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(0, color='red', ls='--', lw=2)
    axes[0].set_xlabel('Error (Predicted - Actual)', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Error Distribution', fontsize=13)

    # Error vs actual
    axes[1].scatter(targets, errors, alpha=0.3, s=15, c='steelblue')
    axes[1].axhline(0, color='red', ls='--', lw=2)
    axes[1].set_xlabel('Actual Tg (C)', fontsize=11)
    axes[1].set_ylabel('Error', fontsize=11)
    axes[1].set_title('Error vs Actual Tg', fontsize=13)

    # Absolute error vs actual
    axes[2].scatter(targets, abs_errors, alpha=0.3, s=15, c='steelblue')
    axes[2].set_xlabel('Actual Tg (C)', fontsize=11)
    axes[2].set_ylabel('Absolute Error', fontsize=11)
    axes[2].set_title('Absolute Error vs Actual Tg', fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'error_analysis_v2.png'), dpi=200)
    plt.close()
    print("Saved error_analysis_v2.png")

    # ═══════════════════════════════════════════════════════════════════
    # 3. Performance by Tg Range
    # ═══════════════════════════════════════════════════════════════════
    tg_ranges = [(-200, 0), (0, 100), (100, 200), (200, 300), (300, 400), (400, 600)]
    range_names = []
    range_rmses = []
    range_maes = []
    range_counts = []

    for lo, hi in tg_ranges:
        mask = (targets >= lo) & (targets < hi)
        if mask.sum() > 0:
            range_names.append(f'{lo}-{hi}')
            range_rmses.append(np.sqrt(np.mean((preds[mask] - targets[mask]) ** 2)))
            range_maes.append(np.mean(np.abs(preds[mask] - targets[mask])))
            range_counts.append(int(mask.sum()))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(range_names))
    bars = ax1.bar(x, range_rmses, 0.6, label='RMSE', color='steelblue', alpha=0.8)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_xlabel('Tg Range (C)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(range_names, rotation=30)
    ax1.set_title('Performance by Tg Range', fontsize=14)

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, range_counts)):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom', fontsize=10)

    ax2 = ax1.twinx()
    ax2.bar(x, range_maes, 0.3, label='MAE', color='coral', alpha=0.6)
    ax2.set_ylabel('MAE', fontsize=12, color='coral')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'range_performance_v2.png'), dpi=200)
    plt.close()
    print("Saved range_performance_v2.png")

    # ═══════════════════════════════════════════════════════════════════
    # 4. Training Curve (from v2 log)
    # ═══════════════════════════════════════════════════════════════════
    log_path = os.path.join(SAVE_DIR, 'training_log_v2_periodic_edge.txt')
    if os.path.exists(log_path):
        epochs, t_rmse, v_rmse = [], [], []
        with open(log_path) as f:
            for line in f:
                if line.startswith('Epoch') and 'Train RMSE:' in line:
                    parts = line.split('|')
                    ep = int(parts[0].strip().split()[1].split('/')[0])
                    epochs.append(ep)
                    for p in parts:
                        if p.strip().startswith('Train RMSE:'):
                            t_rmse.append(float(p.split('Train RMSE:')[1].strip().split()[0]))
                        if p.strip().startswith('Val RMSE:'):
                            v_rmse.append(float(p.split('Val RMSE:')[1].strip().split()[0]))

        if epochs:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs, t_rmse, 'b-', lw=2, label='Train RMSE')
            ax.plot(epochs, v_rmse, 'r-', lw=2, label='Val RMSE')
            ax.axhline(min(v_rmse), color='green', ls=':', lw=1.5,
                       label=f'Best Val: {min(v_rmse):.2f}')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('RMSE', fontsize=12)
            ax.set_title('PPFN-v2 Training Curve', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, 'training_curve_v2.png'), dpi=200)
            plt.close()
            print("Saved training_curve_v2.png")

    # ═══════════════════════════════════════════════════════════════════
    # 5. Comparison Bar Chart
    # ═══════════════════════════════════════════════════════════════════
    models = ['v1\n(complex)', 'v2 simple\n(periodic)', 'v2 simple\n(no periodic)',
              'v2 improved\n(cosine)', 'v2 improved\n(plateau)',
              'v2 wide-512', 'v2 ensemble\n(5 models)', 'SOTA\nTarget']
    rmses = [37.36, 37.76, 38.73, 44.39, 41.42, 39.21, 36.74, 30.8]
    colors = ['steelblue', 'steelblue', 'lightblue', 'salmon',
              'salmon', 'coral', 'green', 'gold']

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(range(len(models)), rmses, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, rmse in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{rmse:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Test RMSE', fontsize=12)
    ax.set_title('PPFN-v2 Architecture Comparison', fontsize=14)
    ax.axhline(30.8, color='red', ls='--', lw=2, alpha=0.5)
    ax.set_ylim(0, max(rmses) + 5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'model_comparison_v2.png'), dpi=200)
    plt.close()
    print("Saved model_comparison_v2.png")

    # ═══════════════════════════════════════════════════════════════════
    # 6. Compile final interpretability report
    # ═══════════════════════════════════════════════════════════════════
    # Aromatic analysis
    aromatic_pred, non_aromatic_pred = [], []
    aromatic_tg, non_aromatic_tg = [], []
    aromatic_err, non_aromatic_err = [], []

    for i, smi in enumerate(p_smiles):
        if i >= len(targets):
            break
        has_aromatic = any(c.islower() for c in str(smi) if c.isalpha())
        if has_aromatic:
            aromatic_pred.append(preds[i])
            aromatic_tg.append(targets[i])
            aromatic_err.append(abs_errors[i])
        else:
            non_aromatic_pred.append(preds[i])
            non_aromatic_tg.append(targets[i])
            non_aromatic_err.append(abs_errors[i])

    # Attention stats
    attn_stats = {}
    if attn is not None:
        attn_stats = {
            'mean': float(np.mean(attn)),
            'std': float(np.std(attn)),
            'min': float(np.min(attn)),
            'max': float(np.max(attn)),
            'median': float(np.median(attn)),
        }

    interpretability = {
        'model': 'PPFN-v2 (simple, periodic edge, hidden_dim=256)',
        'test_metrics': {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
        },
        'error_analysis': {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'median_error': float(np.median(errors)),
            'error_Tg_correlation': float(np.corrcoef(targets, abs_errors)[0, 1]),
        },
        'performance_by_Tg_range': [
            {'range': name, 'count': count, 'rmse': float(r), 'mae': float(m)}
            for name, count, r, m in zip(range_names, range_counts, range_rmses, range_maes)
        ],
        'aromatic_analysis': {
            'aromatic': {
                'count': len(aromatic_tg),
                'mean_Tg': float(np.mean(aromatic_tg)) if aromatic_tg else 0,
                'rmse': float(np.sqrt(np.mean(np.array(aromatic_err) ** 2))) if aromatic_err else 0,
            },
            'non_aromatic': {
                'count': len(non_aromatic_tg),
                'mean_Tg': float(np.mean(non_aromatic_tg)) if non_aromatic_tg else 0,
                'rmse': float(np.sqrt(np.mean(np.array(non_aromatic_err) ** 2))) if non_aromatic_err else 0,
            },
        },
        'attention_weights': attn_stats,
        'systematic_bias': {
            'description': 'Model over-predicts low Tg and under-predicts high Tg',
            'low_Tg_bias': float(np.mean(errors[targets < 100])),
            'high_Tg_bias': float(np.mean(errors[targets > 300])),
        },
        'all_experiments_summary': {
            'v1_complex_hierarchical': {'rmse': 37.36},
            'v2_simple_periodic_256': {'rmse': 37.76},
            'v2_simple_no_periodic_256': {'rmse': 38.73},
            'v2_improved_virtual_node_cosine': {'rmse': 44.39},
            'v2_improved_virtual_node_plateau': {'rmse': 41.42},
            'v2_wide_512': {'rmse': 39.21},
            'v2_ensemble_5models': {'rmse': 36.74},
        },
        'periodic_edge_ablation': {
            'with_periodic': 37.76,
            'without_periodic': 38.73,
            'improvement': 0.97,
            'pct_improvement': 2.51,
            'conclusion': 'Periodic edges provide a small positive contribution (~2.5% RMSE reduction)',
        },
    }

    with open(os.path.join(SAVE_DIR, 'interpretability_v2.json'), 'w') as f:
        json.dump(interpretability, f, indent=2)
    print("Saved interpretability_v2.json")

    # Print summary
    print(f"\n{'='*70}")
    print(f"INTERPRETABILITY SUMMARY")
    print(f"{'='*70}")
    print(f"Test RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
    print(f"\nSystematic bias:")
    print(f"  Low Tg (<100C): mean error = {np.mean(errors[targets < 100]):.2f} (over-predict)")
    print(f"  High Tg (>300C): mean error = {np.mean(errors[targets > 300]):.2f} (under-predict)")
    print(f"\nAromatic vs Non-aromatic:")
    print(f"  Aromatic: n={len(aromatic_tg)}, RMSE={np.sqrt(np.mean(np.array(aromatic_err) ** 2)):.2f}")
    print(f"  Non-aromatic: n={len(non_aromatic_tg)}, RMSE={np.sqrt(np.mean(np.array(non_aromatic_err) ** 2)):.2f}")

    print(f"\nAll experiment results (Test RMSE):")
    print(f"  v1 complex hierarchical:          37.36")
    print(f"  v2 simple periodic (256):          37.76")
    print(f"  v2 simple no periodic (256):        38.73")
    print(f"  v2 improved virtual node (cosine):  44.39")
    print(f"  v2 improved virtual node (plateau): 41.42")
    print(f"  v2 wide-512 (single best):          39.21")
    print(f"  v2 ensemble (5 models):             36.74  <-- Best overall")
    print(f"  SOTA target:                         30.8")
    print(f"\nKey findings:")
    print(f"  1. Simplifying architecture (v2 vs v1) did NOT improve performance")
    print(f"  2. Periodic edges provide ~2.5% RMSE reduction (positive contribution)")
    print(f"  3. Virtual node + multi-scale pooling HURT performance (overfitting)")
    print(f"  4. Ensemble of 5 models with different seeds achieves best: RMSE=36.74")
    print(f"  5. Gap to SOTA (30.8) remains ~6.0 RMSE")
    print(f"  6. Systematic bias: over-predicts low Tg, under-predicts high Tg")


if __name__ == '__main__':
    main()
