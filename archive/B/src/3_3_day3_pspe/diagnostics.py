"""
Diagnostic analysis for PSPE experiments.
1. PSPE Feature Statistics
2. PSPE-RWPE Correlation
3. Tg Correlation with PSPE
4. Error Analysis
"""

import json
import os
import sys
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pspe_data import (
    build_datasets_pspe,
    load_rwpe,
    assign_rwpe_to_graphs,
    create_dataloaders,
    assemble_features,
    NODE_DIM_RICH,
    RWPE_K,
)
from model import GATPredictor
from train import evaluate, CONFIG, SEED, set_seed

warnings.filterwarnings("ignore")

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG_DIR = os.path.join(SAVE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Diagnostic 1: PSPE Feature Statistics
# ---------------------------------------------------------------------------

def diagnostic_pspe_stats(train_graphs, val_graphs, test_graphs, metadata):
    print("\n" + "=" * 70)
    print("Diagnostic 1: PSPE Feature Statistics")
    print("=" * 70)

    all_graphs = train_graphs + val_graphs + test_graphs

    all_dce = []
    all_mi = []
    all_dce_nonzero = []
    all_mi_nonzero = []
    backbone_lengths = []
    backbone_ratios = []

    for g in all_graphs:
        dce_np = g.dce.numpy()
        mi_np = g.mi.numpy()
        all_dce.extend(dce_np.tolist())
        all_mi.extend(mi_np.tolist())
        nonzero_mask = dce_np > 0
        all_dce_nonzero.extend(dce_np[nonzero_mask].tolist())
        nonzero_mi = mi_np > 0
        all_mi_nonzero.extend(mi_np[nonzero_mi].tolist())
        backbone_lengths.append(g.backbone_info['backbone_length'])
        backbone_ratios.append(g.backbone_info['backbone_frac'])

    all_dce = np.array(all_dce)
    all_mi = np.array(all_mi)
    all_dce_nonzero = np.array(all_dce_nonzero) if len(all_dce_nonzero) > 0 else np.array([])
    all_mi_nonzero = np.array(all_mi_nonzero) if len(all_mi_nonzero) > 0 else np.array([])

    frac_non_backbone = np.mean(all_dce == 0) * 100
    frac_backbone = 100 - frac_non_backbone

    stats = {
        'dce_all': {'mean': float(np.mean(all_dce)), 'std': float(np.std(all_dce)),
                    'min': float(np.min(all_dce)), 'max': float(np.max(all_dce))},
        'dce_nonzero': {'mean': float(np.mean(all_dce_nonzero)), 'std': float(np.std(all_dce_nonzero))}
            if len(all_dce_nonzero) > 0 else {},
        'mi_all': {'mean': float(np.mean(all_mi)), 'std': float(np.std(all_mi)),
                   'min': float(np.min(all_mi)), 'max': float(np.max(all_mi))},
        'backbone_ratio_percent': float(frac_backbone),
        'non_backbone_percent': float(frac_non_backbone),
        'avg_backbone_length': float(np.mean(backbone_lengths)),
        'std_backbone_length': float(np.std(backbone_lengths)),
    }

    print(f"DCE (all): mean={stats['dce_all']['mean']:.4f}, std={stats['dce_all']['std']:.4f}")
    if stats['dce_nonzero']:
        print(f"DCE (non-zero): mean={stats['dce_nonzero']['mean']:.4f}")
    print(f"MI (all): mean={stats['mi_all']['mean']:.4f}, std={stats['mi_all']['std']:.4f}")
    print(f"Backbone: {frac_backbone:.1f}%, Non-backbone: {frac_non_backbone:.1f}%")
    print(f"Backbone length: {stats['avg_backbone_length']:.1f} +/- {stats['std_backbone_length']:.1f}")

    # Histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if len(all_dce_nonzero) > 0:
        axes[0].hist(all_dce_nonzero, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('DCE value')
    axes[0].set_title('Non-Zero DCE Values')

    if len(all_mi_nonzero) > 0:
        axes[1].hist(all_mi_nonzero, bins=50, color='darkorange', edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('MI value')
    axes[1].set_title('Non-Zero MI Values')

    axes[2].hist(backbone_lengths, bins=50, color='forestgreen', edgecolor='white', alpha=0.8)
    axes[2].set_xlabel('Backbone Length')
    axes[2].set_title('Backbone Lengths')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pspe_stats_histograms.png"), dpi=150)
    plt.close()
    print(f"Saved: pspe_stats_histograms.png")

    return stats


# ---------------------------------------------------------------------------
# Diagnostic 2: PSPE-RWPE Correlation
# ---------------------------------------------------------------------------

def diagnostic_pspe_rwpe_corr(train_graphs):
    print("\n" + "=" * 70)
    print("Diagnostic 2: PSPE-RWPE Correlation")
    print("=" * 70)

    all_dce = []
    all_mi = []
    all_rwpe = []

    for g in train_graphs:
        all_dce.append(g.dce.numpy())
        all_mi.append(g.mi.numpy())
        all_rwpe.append(g.rwpe.numpy())

    all_dce = np.concatenate(all_dce)
    all_mi = np.concatenate(all_mi)
    all_rwpe = np.concatenate(all_rwpe)

    nonzero_mask = all_dce > 0
    dce_rwpe_corr = []
    for k in range(RWPE_K):
        r, p = pearsonr(all_dce[nonzero_mask], all_rwpe[nonzero_mask, k])
        dce_rwpe_corr.append((k, r, p))

    mi_nonzero = all_mi > 0
    mi_rwpe_corr = []
    for k in range(RWPE_K):
        r, p = pearsonr(all_mi[mi_nonzero], all_rwpe[mi_nonzero, k])
        mi_rwpe_corr.append((k, r, p))

    dce_rwpe_corr.sort(key=lambda x: abs(x[1]), reverse=True)
    mi_rwpe_corr.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 DCE-RWPE correlations:")
    for k, r, p in dce_rwpe_corr[:5]:
        print(f"  RWPE dim {k:2d}: r={r:.4f} (p={p:.2e})")

    print("Top 5 MI-RWPE correlations:")
    for k, r, p in mi_rwpe_corr[:5]:
        print(f"  RWPE dim {k:2d}: r={r:.4f} (p={p:.2e})")

    max_dce_corr = max(abs(r) for _, r, _ in dce_rwpe_corr)
    max_mi_corr = max(abs(r) for _, r, _ in mi_rwpe_corr)
    print(f"Max |r| DCE-RWPE: {max_dce_corr:.4f}")
    print(f"Max |r| MI-RWPE:  {max_mi_corr:.4f}")

    is_orthogonal = (max_dce_corr < 0.3) and (max_mi_corr < 0.3)
    print(f"PSPE orthogonal to RWPE (<0.3): {is_orthogonal}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    dce_sorted = sorted(dce_rwpe_corr, key=lambda x: x[0])
    dce_abs_r = [abs(r) for _, r, _ in dce_sorted]
    axes[0].bar(range(RWPE_K), dce_abs_r, color='steelblue', alpha=0.8)
    axes[0].axhline(y=0.3, color='red', linestyle='--', label='|r|=0.3')
    axes[0].set_xlabel('RWPE Dimension')
    axes[0].set_ylabel('|Pearson r|')
    axes[0].set_title('DCE vs RWPE Dimensions')
    axes[0].legend()

    mi_sorted = sorted(mi_rwpe_corr, key=lambda x: x[0])
    mi_abs_r = [abs(r) for _, r, _ in mi_sorted]
    axes[1].bar(range(RWPE_K), mi_abs_r, color='darkorange', alpha=0.8)
    axes[1].axhline(y=0.3, color='red', linestyle='--', label='|r|=0.3')
    axes[1].set_xlabel('RWPE Dimension')
    axes[1].set_ylabel('|Pearson r|')
    axes[1].set_title('MI vs RWPE Dimensions')
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pspe_rwpe_correlation.png"), dpi=150)
    plt.close()
    print(f"Saved: pspe_rwpe_correlation.png")

    return {
        'max_abs_dce_rwpe': max_dce_corr,
        'max_abs_mi_rwpe': max_mi_corr,
        'is_orthogonal': is_orthogonal,
    }


# ---------------------------------------------------------------------------
# Diagnostic 3: PSPE-Tg Correlation
# ---------------------------------------------------------------------------

def diagnostic_tg_correlation(train_graphs):
    print("\n" + "=" * 70)
    print("Diagnostic 3: PSPE-Tg Correlation")
    print("=" * 70)

    tg_values = []
    mean_dce = []
    mean_mi = []

    for g in train_graphs:
        tg_values.append(g.y.item())
        mean_dce.append(g.dce.mean().item())
        mean_mi.append(g.mi.mean().item())

    tg_values = np.array(tg_values)
    mean_dce = np.array(mean_dce)
    mean_mi = np.array(mean_mi)

    r_dce, p_dce = pearsonr(mean_dce, tg_values)
    r_mi, p_mi = pearsonr(mean_mi, tg_values)

    print(f"mean_dce vs Tg: r={r_dce:.4f} (p={p_dce:.2e})")
    print(f"mean_mi vs Tg:  r={r_mi:.4f} (p={p_mi:.2e})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(mean_dce, tg_values, alpha=0.3, s=5, c='steelblue')
    axes[0].set_xlabel('Graph-Mean DCE')
    axes[0].set_ylabel('Tg (K)')
    axes[0].set_title(f'Mean DCE vs Tg (r={r_dce:.3f})')

    axes[1].scatter(mean_mi, tg_values, alpha=0.3, s=5, c='darkorange')
    axes[1].set_xlabel('Graph-Mean MI')
    axes[1].set_ylabel('Tg (K)')
    axes[1].set_title(f'Mean MI vs Tg (r={r_mi:.3f})')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pspe_tg_correlation.png"), dpi=150)
    plt.close()
    print(f"Saved: pspe_tg_correlation.png")

    return {'dce_tg_r': float(r_dce), 'dce_tg_p': float(p_dce),
            'mi_tg_r': float(r_mi), 'mi_tg_p': float(p_mi)}


# ---------------------------------------------------------------------------
# Diagnostic 4: Error Analysis
# ---------------------------------------------------------------------------

def diagnostic_error_analysis(all_results):
    print("\n" + "=" * 70)
    print("Diagnostic 4: Error Analysis")
    print("=" * 70)

    if 'baseline' not in all_results or 'pspe_combined' not in all_results:
        print("Missing results - cannot perform error analysis")
        return {}

    baseline = all_results['baseline']
    pspes = {k: v for k, v in all_results.items() if k != 'baseline'}
    best_pspe = min(pspes.keys(), key=lambda k: pspes[k]['test_rmse'])
    pspe = all_results[best_pspe]

    bl_errors = np.abs(np.array(baseline['test_targets']) - np.array(baseline['test_predictions']))
    pspe_errors = np.abs(np.array(pspe['test_targets']) - np.array(pspe['test_predictions']))
    targets = np.array(baseline['test_targets'])

    min_len = min(len(bl_errors), len(pspe_errors))
    bl_errors = bl_errors[:min_len]
    pspe_errors = pspe_errors[:min_len]
    targets = targets[:min_len]

    delta = pspe_errors - bl_errors
    improved = np.sum(delta < 0)
    degraded = np.sum(delta > 0)
    print(f"Samples improved by {best_pspe}: {improved}/{min_len} ({100*improved/min_len:.1f}%)")
    print(f"Samples degraded by {best_pspe}: {degraded}/{min_len} ({100*degraded/min_len:.1f}%)")
    print(f"Mean error delta: {np.mean(delta):.4f}K")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].scatter(bl_errors, pspe_errors, alpha=0.4, s=8, c='steelblue')
    max_err = max(bl_errors.max(), pspe_errors.max())
    axes[0, 0].plot([0, max_err], [0, max_err], 'k--', alpha=0.5, label='y=x')
    axes[0, 0].set_xlabel('Baseline |Error| (K)')
    axes[0, 0].set_ylabel(f'{best_pspe} |Error| (K)')
    axes[0, 0].set_title(f'Per-Sample Error: Baseline vs {best_pspe}')
    axes[0, 0].legend()

    axes[0, 1].hist(delta, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0, 1].axvline(x=0, color='red', linestyle='--')
    axes[0, 1].set_xlabel(f'Error Delta ({best_pspe} - Baseline) (K)')
    axes[0, 1].set_title('Error Delta Distribution')

    # Error by Tg range
    tg_bins = np.percentile(targets, [0, 25, 50, 75, 100])
    bin_labels = [f'<{tg_bins[1]:.0f}K', f'{tg_bins[1]:.0f}-{tg_bins[2]:.0f}K',
                  f'{tg_bins[2]:.0f}-{tg_bins[3]:.0f}K', f'>{tg_bins[3]:.0f}K']
    bin_idx = np.digitize(targets, tg_bins[1:-1])
    bl_bin_mean = [np.mean(bl_errors[bin_idx == i]) for i in range(len(bin_labels))]
    pspe_bin_mean = [np.mean(pspe_errors[bin_idx == i]) for i in range(len(bin_labels))]

    x = np.arange(len(bin_labels))
    width = 0.35
    axes[1, 0].bar(x - width/2, bl_bin_mean, width, label='Baseline', color='steelblue', alpha=0.8)
    axes[1, 0].bar(x + width/2, pspe_bin_mean, width, label=best_pspe, color='darkorange', alpha=0.8)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(bin_labels)
    axes[1, 0].set_ylabel('Mean |Error| (K)')
    axes[1, 0].set_title('Error by Tg Range')
    axes[1, 0].legend()

    pspe_preds = np.array(pspe['test_predictions'][:min_len])
    pspe_targets = np.array(pspe['test_targets'][:min_len])
    axes[1, 1].scatter(pspe_targets, pspe_preds, alpha=0.3, s=5, c='darkorange')
    axes[1, 1].plot([min(pspe_targets), max(pspe_targets)],
                     [min(pspe_targets), max(pspe_targets)], 'k--')
    axes[1, 1].set_xlabel('True Tg (K)')
    axes[1, 1].set_ylabel('Predicted Tg (K)')
    axes[1, 1].set_title(f'Parity: {best_pspe} (RMSE={pspe["test_rmse"]:.2f}K)')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "error_analysis.png"), dpi=150)
    plt.close()
    print(f"Saved: error_analysis.png")

    return {
        'best_pspe_variant': best_pspe,
        'fraction_improved': float(improved / min_len),
        'fraction_degraded': float(degraded / min_len),
        'mean_delta': float(np.mean(delta)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PSPE Diagnostics v2 (Corrected Architecture)")
    print("=" * 70)

    device = torch.device(CONFIG['device'])
    csv_path = CONFIG['csv_path']
    rwpe_path = CONFIG['rwpe_path']
    batch_size = CONFIG['batch_size']

    print("\nBuilding datasets...")
    train_graphs, val_graphs, test_graphs, scaler, metadata = build_datasets_pspe(
        csv_path=csv_path, seed=71,
    )
    print(f"Graphs: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test")

    print("\nLoading RWPE...")
    rwpe_data = load_rwpe(rwpe_path)
    n1_train, _ = assign_rwpe_to_graphs(train_graphs, rwpe_data, offset=0)
    n1_val, _ = assign_rwpe_to_graphs(val_graphs, rwpe_data, offset=len(train_graphs))
    n1_test, _ = assign_rwpe_to_graphs(
        test_graphs, rwpe_data, offset=len(train_graphs) + len(val_graphs)
    )
    print(f"RWPE assigned: train={n1_train}, val={n1_val}, test={n1_test}")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_graphs, val_graphs, test_graphs, scaler, batch_size=batch_size
    )

    stats1 = diagnostic_pspe_stats(train_graphs, val_graphs, test_graphs, metadata)
    stats2 = diagnostic_pspe_rwpe_corr(train_graphs)
    stats3 = diagnostic_tg_correlation(train_graphs)

    # Load trained results for error analysis
    all_results = {}
    conditions = ['baseline', 'pspe_dce', 'pspe_mi', 'pspe_combined']
    for condition in conditions:
        results_path = os.path.join(SAVE_DIR, f"results_{condition}.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                all_results[condition] = json.load(f)

    if all_results:
        stats4 = diagnostic_error_analysis(all_results)
    else:
        stats4 = {}

    diagnostics = {
        'pspe_stats': stats1,
        'pspe_rwpe_corr': stats2,
        'pspe_tg_corr': stats3,
        'error_analysis': stats4,
    }

    with open(os.path.join(SAVE_DIR, "diagnostics.json"), 'w') as f:
        json.dump(diagnostics, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

    print(f"\nDiagnostics saved to: {SAVE_DIR}/diagnostics.json")
    print(f"Figures saved to: {FIG_DIR}/")


if __name__ == '__main__':
    main()
