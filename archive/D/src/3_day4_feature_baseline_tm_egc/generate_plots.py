#!/usr/bin/env python3
"""
Generate diagnostic plots for Tm and Egc baseline experiments.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_results():
    """Load all results files."""
    base_dir = Path('/home/home_public/aisci/AIscientist/src/3_day4_feature_baseline_tm_egc')

    with open(base_dir / 'tm_baseline_results_v2.json', 'r') as f:
        tm_results = json.load(f)

    with open(base_dir / 'egc_baseline_results_v2.json', 'r') as f:
        egc_results = json.load(f)

    with open(base_dir / 'tm_feature_importance.json', 'r') as f:
        tm_importance = json.load(f)

    with open(base_dir / 'egc_feature_importance.json', 'r') as f:
        egc_importance = json.load(f)

    with open(base_dir / 'cross_property_comparison_v2.json', 'r') as f:
        comparison = json.load(f)

    return tm_results, egc_results, tm_importance, egc_importance, comparison


def plot_results(tm_results, egc_results, tm_importance, egc_importance, comparison):
    """Generate comprehensive diagnostic plots."""

    fig = plt.figure(figsize=(20, 12))

    # 1. Cross-validation performance comparison
    ax1 = fig.add_subplot(2, 3, 1)
    tm_folds = [r['rmse'] for r in tm_results['fold_results']]
    egc_folds = [r['rmse'] for r in egc_results['fold_results']]
    x = range(1, 6)

    ax1.bar([i - 0.2 for i in x], tm_folds, width=0.4, label='Tm', color='steelblue', alpha=0.8)
    ax1.bar([i + 0.2 for i in x], egc_folds, width=0.4, label='Egc', color='coral', alpha=0.8)
    ax1.axhline(y=60.0, color='blue', linestyle='--', alpha=0.5, label='Tm target (60.0)')
    ax1.axhline(y=0.50, color='red', linestyle='--', alpha=0.5, label='Egc target (0.50)')
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('5-Fold Cross-Validation Performance', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. R² comparison
    ax2 = fig.add_subplot(2, 3, 2)
    tm_r2 = [r['r2'] for r in tm_results['fold_results']]
    egc_r2 = [r['r2'] for r in egc_results['fold_results']]

    ax2.bar([i - 0.2 for i in x], tm_r2, width=0.4, label='Tm', color='steelblue', alpha=0.8)
    ax2.bar([i + 0.2 for i in x], egc_r2, width=0.4, label='Egc', color='coral', alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Fold', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('R² Score Across Folds', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. Feature importance - Tm
    ax3 = fig.add_subplot(2, 3, 3)
    tm_features = tm_importance['top_features'][:10]
    names = [f['feature'] for f in tm_features]
    importances = [f['importance'] for f in tm_features]

    y_pos = range(len(names))
    ax3.barh(y_pos, importances, color='steelblue', alpha=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names)
    ax3.invert_yaxis()
    ax3.set_xlabel('Importance', fontsize=12)
    ax3.set_title('Tm Top 10 Features', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

    # 4. Feature importance - Egc
    ax4 = fig.add_subplot(2, 3, 4)
    egc_features = egc_importance['top_features'][:10]
    names = [f['feature'] for f in egc_features]
    importances = [f['importance'] for f in egc_features]

    y_pos = range(len(names))
    ax4.barh(y_pos, importances, color='coral', alpha=0.8)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(names)
    ax4.invert_yaxis()
    ax4.set_xlabel('Importance', fontsize=12)
    ax4.set_title('Egc Top 10 Features', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)

    # 5. Summary statistics
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')

    summary_text = f"""
    PERFORMANCE SUMMARY
    ==================

    Tm Prediction:
    • Samples: {tm_results['n_samples']:,}
    • Mean RMSE: {tm_results['mean_rmse']:.2f} ± {tm_results['std_rmse']:.2f}
    • Best RMSE: {tm_results['best_rmse']:.2f} (Fold {tm_results['best_fold']})
    • Target: < 60.0 | Status: {'✓ ACHIEVED' if tm_results['mean_rmse'] < 60.0 else '✗ NOT ACHIEVED'}
    • GAT Baseline: 79.3 | Improvement: {(79.3 - tm_results['mean_rmse'])/79.3*100:.1f}%

    Egc Prediction:
    • Samples: {egc_results['n_samples']:,}
    • Mean RMSE: {egc_results['mean_rmse']:.2f} ± {egc_results['std_rmse']:.2f}
    • Best RMSE: {egc_results['best_rmse']:.2f} (Fold {egc_results['best_fold']})
    • Target: < 0.50 | Status: {'✓ ACHIEVED' if egc_results['mean_rmse'] < 0.50 else '✗ NOT ACHIEVED'}
    • GAT Baseline: 0.63 | Improvement: {(0.63 - egc_results['mean_rmse'])/0.63*100:.1f}%

    Common Features: {comparison['n_common']}
    """

    ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 6. Comparison with baselines
    ax6 = fig.add_subplot(2, 3, 6)

    # Tm comparison
    categories = ['Tm', 'Egc']
    gat_baselines = [79.3, 0.63]
    our_results = [tm_results['mean_rmse'], egc_results['mean_rmse']]
    targets = [60.0, 0.50]
    best_folds = [tm_results['best_rmse'], egc_results['best_rmse']]

    x = np.arange(len(categories))
    width = 0.25

    ax6.bar(x - width, gat_baselines, width, label='GAT Baseline', color='gray', alpha=0.7)
    ax6.bar(x, our_results, width, label='Our Mean', color='steelblue', alpha=0.8)
    ax6.bar(x + width, best_folds, width, label='Our Best', color='green', alpha=0.8)

    # Add target lines (scaled for visibility)
    ax6.plot([-0.5, 0.5], [targets[0], targets[0]], 'b--', linewidth=2, label='Target' if x[0] == 0 else '')
    ax6.plot([0.5, 1.5], [targets[1]*100, targets[1]*100], 'r--', linewidth=2)

    # Add secondary y-axis for Egc scaling
    ax6_twin = ax6.twinx()
    ax6_twin.set_ylim(ax6.get_ylim())

    ax6.set_xlabel('Dataset', fontsize=12)
    ax6.set_ylabel('RMSE', fontsize=12)
    ax6.set_title('Comparison with Baselines', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(categories)
    ax6.legend(loc='upper right')
    ax6.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = Path('/home/home_public/aisci/AIscientist/src/3_day4_feature_baseline_tm_egc/prediction_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved diagnostic plots to: {output_path}")

    plt.close()


def main():
    print("Loading results...")
    tm_results, egc_results, tm_importance, egc_importance, comparison = load_results()

    print("Generating diagnostic plots...")
    plot_results(tm_results, egc_results, tm_importance, egc_importance, comparison)

    print("Done!")


if __name__ == '__main__':
    main()
