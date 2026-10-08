"""
Diagnostic analysis for PSPE experiments.

1. PSPE Feature Statistics
2. PSPE-RWPE Correlation
3. Tg Correlation with PSPE
4. Attention Analysis
5. Error Analysis
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
    assign_rwpe_to_graphs_normalized,
    create_dataloaders,
    assemble_features,
    NODE_DIM_RICH,
    RWPE_K,
)
from model import GATv2Predictor
from train import evaluate, CONFIG, SEED, set_seed

warnings.filterwarnings("ignore")

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG_DIR = os.path.join(SAVE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Diagnostic 1: PSPE Feature Statistics
# ---------------------------------------------------------------------------

def diagnostic_pspe_stats(train_graphs, val_graphs, test_graphs, metadata):
    """Analyze PSPE feature statistics across all graphs."""
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
    all_dce_nonzero = np.array(all_dce_nonzero)
    all_mi_nonzero = np.array(all_mi_nonzero)

    # Fraction of non-backbone atoms (dce=0 or mi=0)
    frac_non_backbone = np.mean(all_dce == 0) * 100
    frac_backbone = 100 - frac_non_backbone

    stats = {
        'dce_all': {
            'mean': float(np.mean(all_dce)), 'std': float(np.std(all_dce)),
            'min': float(np.min(all_dce)), 'max': float(np.max(all_dce)),
        },
        'dce_nonzero': {
            'mean': float(np.mean(all_dce_nonzero)), 'std': float(np.std(all_dce_nonzero)),
            'min': float(np.min(all_dce_nonzero)), 'max': float(np.max(all_dce_nonzero)),
        } if len(all_dce_nonzero) > 0 else {},
        'mi_all': {
            'mean': float(np.mean(all_mi)), 'std': float(np.std(all_mi)),
            'min': float(np.min(all_mi)), 'max': float(np.max(all_mi)),
        },
        'mi_nonzero': {
            'mean': float(np.mean(all_mi_nonzero)), 'std': float(np.std(all_mi_nonzero)),
            'min': float(np.min(all_mi_nonzero)), 'max': float(np.max(all_mi_nonzero)),
        } if len(all_mi_nonzero) > 0 else {},
        'backbone_ratio_percent': float(frac_backbone),
        'non_backbone_percent': float(frac_non_backbone),
        'avg_backbone_length': float(np.mean(backbone_lengths)),
        'std_backbone_length': float(np.std(backbone_lengths)),
    }

    print(f"DCE (all atoms): mean={stats['dce_all']['mean']:.4f}, std={stats['dce_all']['std']:.4f}, "
          f"min={stats['dce_all']['min']:.4f}, max={stats['dce_all']['max']:.4f}")
    if stats['dce_nonzero']:
        print(f"DCE (non-zero): mean={stats['dce_nonzero']['mean']:.4f}, std={stats['dce_nonzero']['std']:.4f}")
    print(f"MI (all atoms): mean={stats['mi_all']['mean']:.4f}, std={stats['mi_all']['std']:.4f}")
    print(f"Backbone atoms: {frac_backbone:.1f}%, Non-backbone: {frac_non_backbone:.1f}%")
    print(f"Avg backbone length: {stats['avg_backbone_length']:.1f} +/- {stats['std_backbone_length']:.1f}")

    # Histogram of non-zero DCE values
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(all_dce_nonzero, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('DCE value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Histogram of Non-Zero DCE Values')
    axes[0].axvline(x=np.mean(all_dce_nonzero), color='red', linestyle='--', label=f'Mean={np.mean(all_dce_nonzero):.3f}')
    axes[0].legend()

    axes[1].hist(all_mi_nonzero, bins=50, color='darkorange', edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('MI value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Histogram of Non-Zero MI Values')

    axes[2].hist(backbone_lengths, bins=50, color='forestgreen', edgecolor='white', alpha=0.8)
    axes[2].set_xlabel('Backbone Length')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Histogram of Backbone Lengths')
    axes[2].axvline(x=np.mean(backbone_lengths), color='red', linestyle='--', label=f'Mean={np.mean(backbone_lengths):.1f}')
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pspe_stats_histograms.png"), dpi=150)
    plt.close()
    print(f"Saved: pspe_stats_histograms.png")

    return stats


# ---------------------------------------------------------------------------
# Diagnostic 2: PSPE-RWPE Correlation
# ---------------------------------------------------------------------------

def diagnostic_pspe_rwpe_corr(train_graphs):
    """Compute correlation between PSPE features and top RWPE dimensions."""
    print("\n" + "=" * 70)
    print("Diagnostic 2: PSPE-RWPE Correlation")
    print("=" * 70)

    # Collect per-node features across all training graphs
    all_dce = []
    all_mi = []
    all_rwpe = []

    for g in train_graphs:
        dce_np = g.dce.numpy()
        mi_np = g.mi.numpy()
        rwpe_np = g.rwpe.numpy()
        all_dce.append(dce_np)
        all_mi.append(mi_np)
        all_rwpe.append(rwpe_np)

    all_dce = np.concatenate(all_dce)
    all_mi = np.concatenate(all_mi)
    all_rwpe = np.concatenate(all_rwpe)  # [N_nodes, 32]

    # Filter non-zero dce/mi for meaningful correlation
    nonzero_mask = all_dce > 0

    # DCE vs each RWPE dimension
    dce_rwpe_corr = []
    for k in range(RWPE_K):
        r, p = pearsonr(all_dce[nonzero_mask], all_rwpe[nonzero_mask, k])
        dce_rwpe_corr.append((k, r, p))

    # MI vs each RWPE dimension
    mi_rwpe_corr = []
    mi_nonzero = all_mi > 0
    for k in range(RWPE_K):
        r, p = pearsonr(all_mi[mi_nonzero], all_rwpe[mi_nonzero, k])
        mi_rwpe_corr.append((k, r, p))

    # Sort by absolute correlation
    dce_rwpe_corr.sort(key=lambda x: abs(x[1]), reverse=True)
    mi_rwpe_corr.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 DCE-RWPE correlations:")
    for k, r, p in dce_rwpe_corr[:5]:
        print(f"  RWPE dim {k:2d}: r={r:.4f} (p={p:.2e})")

    print("Top 5 MI-RWPE correlations:")
    for k, r, p in mi_rwpe_corr[:5]:
        print(f"  RWPE dim {k:2d}: r={r:.4f} (p={p:.2e})")

    # Compute overall orthogonality (max absolute correlation)
    max_dce_corr = max(abs(r) for _, r, _ in dce_rwpe_corr)
    max_mi_corr = max(abs(r) for _, r, _ in mi_rwpe_corr)
    print(f"Max |r| DCE-RWPE: {max_dce_corr:.4f}")
    print(f"Max |r| MI-RWPE:  {max_mi_corr:.4f}")

    is_orthogonal = (max_dce_corr < 0.3) and (max_mi_corr < 0.3)
    print(f"PSPE orthogonal to RWPE: {is_orthogonal} (threshold |r|<0.3)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    dce_abs_r = [abs(r) for _, r, _ in dce_rwpe_corr]
    axes[0].bar(range(RWPE_K), dce_abs_r, color='steelblue', alpha=0.8)
    axes[0].axhline(y=0.3, color='red', linestyle='--', label='|r|=0.3 threshold')
    axes[0].set_xlabel('RWPE Dimension')
    axes[0].set_ylabel('|Pearson r|')
    axes[0].set_title('DCE vs RWPE Dimensions: Absolute Correlation')
    axes[0].legend()

    mi_abs_r = [abs(r) for _, r, _ in mi_rwpe_corr]
    axes[1].bar(range(RWPE_K), mi_abs_r, color='darkorange', alpha=0.8)
    axes[1].axhline(y=0.3, color='red', linestyle='--', label='|r|=0.3 threshold')
    axes[1].set_xlabel('RWPE Dimension')
    axes[1].set_ylabel('|Pearson r|')
    axes[1].set_title('MI vs RWPE Dimensions: Absolute Correlation')
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pspe_rwpe_correlation.png"), dpi=150)
    plt.close()
    print(f"Saved: pspe_rwpe_correlation.png")

    return {
        'max_abs_dce_rwpe': max_dce_corr,
        'max_abs_mi_rwpe': max_mi_corr,
        'is_orthogonal': is_orthogonal,
        'top_dce_rwpe': [(int(k), float(r), float(p)) for k, r, p in dce_rwpe_corr[:10]],
        'top_mi_rwpe': [(int(k), float(r), float(p)) for k, r, p in mi_rwpe_corr[:10]],
    }


# ---------------------------------------------------------------------------
# Diagnostic 3: PSPE-Tg Correlation
# ---------------------------------------------------------------------------

def diagnostic_tg_correlation(train_graphs):
    """Correlate graph-mean PSPE features with Tg values."""
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
    print(f"mean_mi stats: mean={np.mean(mean_mi):.4f}, std={np.std(mean_mi):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(mean_dce, tg_values, alpha=0.3, s=5, c='steelblue')
    axes[0].set_xlabel('Graph-Mean DCE')
    axes[0].set_ylabel('Tg (K)')
    axes[0].set_title(f'Mean DCE vs Tg (r={r_dce:.3f}, p={p_dce:.1e})')

    axes[1].scatter(mean_mi, tg_values, alpha=0.3, s=5, c='darkorange')
    axes[1].set_xlabel('Graph-Mean MI')
    axes[1].set_ylabel('Tg (K)')
    axes[1].set_title(f'Mean MI vs Tg (r={r_mi:.3f}, p={p_mi:.1e})')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pspe_tg_correlation.png"), dpi=150)
    plt.close()
    print(f"Saved: pspe_tg_correlation.png")

    return {
        'dce_tg_r': float(r_dce), 'dce_tg_p': float(p_dce),
        'mi_tg_r': float(r_mi), 'mi_tg_p': float(p_mi),
        'mean_mi_mean': float(np.mean(mean_mi)),
        'mean_mi_std': float(np.std(mean_mi)),
    }


# ---------------------------------------------------------------------------
# Diagnostic 4: Attention Analysis
# ---------------------------------------------------------------------------

def diagnostic_attention(model, test_loader, scaler, condition, device):
    """Analyze attention patterns vs PSPE feature differences."""
    print("\n" + "=" * 70)
    print("Diagnostic 4: Attention Analysis")
    print("=" * 70)

    model.eval()
    all_pspe_diffs = []  # |dce_i - dce_j| for each edge
    all_att_means = []   # mean attention across heads for each edge

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            x = assemble_features(batch, condition)

            _, (att_edge_index, att_weights) = model(
                x, batch.edge_index, batch.batch,
                edge_attr=batch.edge_attr, return_attention=True
            )

            # att_weights shape: [num_edges, heads] or [heads, num_edges]
            if att_weights.dim() == 1:
                # Single head output
                att_mean = att_weights.cpu().numpy()
            elif att_weights.shape[0] == 4:  # heads x edges
                att_mean = att_weights.mean(dim=0).cpu().numpy()
            else:
                att_mean = att_weights.mean(dim=-1).cpu().numpy()

            # Compute PSPE feature differences per edge
            dce = batch.dce
            mi = batch.mi

            att_ei = att_edge_index.cpu().numpy()

            for e in range(att_ei.shape[1]):
                src, dst = int(att_ei[0, e]), int(att_ei[1, e])
                if condition == 'pspe_dce':
                    pspe_diff = abs(dce[src].item() - dce[dst].item())
                elif condition == 'pspe_mi':
                    pspe_diff = abs(mi[src].item() - mi[dst].item())
                elif condition == 'pspe_combined':
                    pspe_diff = abs(dce[src].item() - dce[dst].item())
                else:
                    pspe_diff = 0.0

                all_pspe_diffs.append(pspe_diff)
                all_att_means.append(float(att_mean[e]))

    all_pspe_diffs = np.array(all_pspe_diffs)
    all_att_means = np.array(all_att_means)

    # Bin by PSPE diff
    if len(all_pspe_diffs) > 0 and np.std(all_pspe_diffs) > 0:
        bins = np.linspace(0, 1, 11)
        bin_centers = []
        bin_att_means = []
        bin_att_stds = []

        for i in range(len(bins) - 1):
            mask = (all_pspe_diffs >= bins[i]) & (all_pspe_diffs < bins[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_att_means.append(np.mean(all_att_means[mask]))
                bin_att_stds.append(np.std(all_att_means[mask]))

        bin_centers = np.array(bin_centers)
        bin_att_means = np.array(bin_att_means)
        bin_att_stds = np.array(bin_att_stds)

        # Correlation
        if len(all_pspe_diffs) > 100:
            r_att, p_att = pearsonr(all_pspe_diffs[:10000], all_att_means[:10000])
            print(f"PSPE diff vs attention, r={r_att:.4f} (p={p_att:.2e})")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(bin_centers, bin_att_means, yerr=bin_att_stds,
                     fmt='o-', capsize=3, color='steelblue')
        ax.set_xlabel('|PSPE difference| (between node pairs)')
        ax.set_ylabel('Mean Attention Weight')
        ax.set_title(f'Attention vs PSPE Feature Difference ({condition})')
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"attention_vs_pspe_{condition}.png"), dpi=150)
        plt.close()
        print(f"Saved: attention_vs_pspe_{condition}.png")
    else:
        print("  No PSPE features for attention analysis (baseline or insufficient data)")
        r_att = 0.0

    return {'attention_pspe_corr': float(r_att) if 'r_att' in dir() else 0.0}


# ---------------------------------------------------------------------------
# Diagnostic 5: Error Analysis
# ---------------------------------------------------------------------------

def diagnostic_error_analysis(all_results):
    """Compare per-sample errors between baseline and best PSPE variant."""
    print("\n" + "=" * 70)
    print("Diagnostic 5: Error Analysis")
    print("=" * 70)

    if 'baseline' not in all_results or 'pspe_combined' not in all_results:
        print("Missing results - cannot perform error analysis")
        return {}

    baseline = all_results['baseline']
    # Use combined as the 'best PSPE' (or whichever had lowest RMSE)
    pspes = {k: v for k, v in all_results.items() if k != 'baseline'}
    best_pspe = min(pspes.keys(), key=lambda k: pspes[k]['test_rmse'])
    pspe = all_results[best_pspe]

    bl_errors = np.abs(np.array(baseline['test_targets']) - np.array(baseline['test_predictions']))
    pspe_errors = np.abs(np.array(pspe['test_targets']) - np.array(pspe['test_predictions']))
    targets = np.array(baseline['test_targets'])

    # Ensure same length
    min_len = min(len(bl_errors), len(pspe_errors))
    bl_errors = bl_errors[:min_len]
    pspe_errors = pspe_errors[:min_len]
    targets = targets[:min_len]

    delta = pspe_errors - bl_errors
    improved = np.sum(delta < 0)
    degraded = np.sum(delta > 0)
    print(f"Samples improved by {best_pspe}: {improved} / {min_len} ({100*improved/min_len:.1f}%)")
    print(f"Samples degraded by {best_pspe}: {degraded} / {min_len} ({100*degraded/min_len:.1f}%)")
    print(f"Mean error delta ({best_pspe} - baseline): {np.mean(delta):.4f}K")

    # Error by Tg range
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Scatter: baseline error vs PSPE error
    axes[0, 0].scatter(bl_errors, pspe_errors, alpha=0.4, s=8, c='steelblue')
    max_err = max(bl_errors.max(), pspe_errors.max())
    axes[0, 0].plot([0, max_err], [0, max_err], 'k--', alpha=0.5, label='y=x')
    axes[0, 0].set_xlabel('Baseline |Error| (K)')
    axes[0, 0].set_ylabel(f'{best_pspe} |Error| (K)')
    axes[0, 0].set_title(f'Per-Sample Error: Baseline vs {best_pspe}')
    axes[0, 0].legend()

    # Histogram of error deltas
    axes[0, 1].hist(delta, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0, 1].axvline(x=0, color='red', linestyle='--')
    axes[0, 1].set_xlabel(f'Error Delta ({best_pspe} - Baseline) (K)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Error Deltas')

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

    # Parity plot for best PSPE
    pspe_preds = np.array(pspe['test_predictions'][:min_len])
    pspe_targets = np.array(pspe['test_targets'][:min_len])
    axes[1, 1].scatter(pspe_targets, pspe_preds, alpha=0.3, s=5, c='darkorange')
    axes[1, 1].plot([min(pspe_targets), max(pspe_targets)],
                     [min(pspe_targets), max(pspe_targets)], 'k--')
    axes[1, 1].set_xlabel('True Tg (K)')
    axes[1, 1].set_ylabel('Predicted Tg (K)')
    axes[1, 1].set_title(f'Parity Plot: {best_pspe} (RMSE={pspe["test_rmse"]:.2f}K)')

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
    print("PSPE Diagnostics")
    print("=" * 70)

    device = torch.device(CONFIG['device'])
    csv_path = CONFIG['csv_path']
    rwpe_path = CONFIG['rwpe_path']
    batch_size = CONFIG['batch_size']

    # Build datasets
    print("\nBuilding datasets...")
    train_graphs, val_graphs, test_graphs, scaler, metadata = build_datasets_pspe(
        csv_path=csv_path, seed=71,
    )
    print(f"Graphs: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test")

    # Load RWPE
    print("\nLoading RWPE...")
    rwpe_data = load_rwpe(rwpe_path)
    # Assign RWPE by list position
    n1_train, _, _, _ = assign_rwpe_to_graphs_normalized(train_graphs, rwpe_data, offset=0)
    n1_val, _, _, _ = assign_rwpe_to_graphs_normalized(val_graphs, rwpe_data, offset=len(train_graphs))
    n1_test, _, _, _ = assign_rwpe_to_graphs_normalized(
        test_graphs, rwpe_data, offset=len(train_graphs) + len(val_graphs)
    )
    print(f"RWPE assigned: train={n1_train}, val={n1_val}, test={n1_test}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_graphs, val_graphs, test_graphs, scaler, batch_size=batch_size
    )

    # Run diagnostics
    stats1 = diagnostic_pspe_stats(train_graphs, val_graphs, test_graphs, metadata)
    stats2 = diagnostic_pspe_rwpe_corr(train_graphs)
    stats3 = diagnostic_tg_correlation(train_graphs)

    # Load trained models for attention analysis
    all_results = {}
    conditions = ['baseline', 'pspe_dce', 'pspe_mi', 'pspe_combined']
    for condition in conditions:
        results_path = os.path.join(SAVE_DIR, f"results_{condition}.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                all_results[condition] = json.load(f)

    if all_results:
        # Attention analysis on best PSPE model
        pspes = {k: v for k, v in all_results.items() if k != 'baseline'}
        if pspes:
            best_pspe = min(pspes.keys(), key=lambda k: pspes[k]['test_rmse'])
            model_path = os.path.join(SAVE_DIR, f"model_{best_pspe}.pt")
            if os.path.exists(model_path):
                set_seed(SEED)
                in_dim_input = pspes[best_pspe]['in_dim']
                from pspe_data import get_input_dim
                model = GATv2Predictor(
                    in_dim=in_dim_input,
                    edge_dim=10,
                    hidden_dim=CONFIG['hidden_dim'],
                    heads=CONFIG['heads'],
                    dropout=CONFIG['dropout'],
                ).to(device)
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                stats4 = diagnostic_attention(model, test_loader, scaler, best_pspe, device)
            else:
                stats4 = {'attention_pspe_corr': 0.0}
                print(f"Model file not found for attention analysis")
        else:
            stats4 = {'attention_pspe_corr': 0.0}

        stats5 = diagnostic_error_analysis(all_results)
    else:
        stats4 = {'attention_pspe_corr': 0.0}
        stats5 = {}

    # Save all diagnostics
    diagnostics = {
        'pspe_stats': stats1,
        'pspe_rwpe_corr': stats2,
        'pspe_tg_corr': stats3,
        'attention': stats4,
        'error_analysis': stats5,
    }

    with open(os.path.join(SAVE_DIR, "diagnostics.json"), 'w') as f:
        json.dump(diagnostics, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

    print(f"\nDiagnostics saved to: {SAVE_DIR}/diagnostics.json")
    print(f"Figures saved to: {FIG_DIR}/")


if __name__ == '__main__':
    main()
