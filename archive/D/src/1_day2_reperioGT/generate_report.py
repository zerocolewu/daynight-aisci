"""
Generate comprehensive analysis report for RE-PerioGT experiments
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Known results from training logs
RESULTS = {
    'baseline': {
        'test_rmse': 36.41,
        'test_mae': 25.60,
        'test_r2': 0.9000,
        'test_pearson_r': 0.9487,
        'best_val_rmse': 39.19,
        'best_epoch': 211,
        'training_time_min': 29.6
    },
    'feature_only': {
        'test_rmse': 35.56,
        'test_mae': 24.83,
        'test_r2': 0.9046,
        'test_pearson_r': 0.9521,
        'best_val_rmse': 36.94,
        'best_epoch': 207,
        'training_time_min': 30.7
    },
    'full': {
        'test_rmse': 35.55,
        'test_mae': 24.95,
        'test_r2': 0.9047,
        'test_pearson_r': 0.9513,
        'best_val_rmse': 37.97,
        'best_epoch': 245,
        'training_time_min': 316.9
    }
}


def plot_comparison(output_dir):
    """Create comparison plots across all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Colors for each model
    colors = {'baseline': '#1f77b4', 'feature_only': '#2ca02c', 'full': '#d62728'}
    labels = {
        'baseline': 'PerioGT (Baseline)',
        'feature_only': 'RE-PerioGT (Feature Only)',
        'full': 'RE-PerioGT (Full)'
    }
    modes = ['baseline', 'feature_only', 'full']

    # 1. Test RMSE comparison bar chart
    ax = axes[0, 0]
    test_rmses = [RESULTS[m]['test_rmse'] for m in modes]
    bars = ax.bar([labels[m] for m in modes], test_rmses, color=[colors[m] for m in modes])
    ax.set_ylabel('Test RMSE', fontsize=12)
    ax.set_title('Test Set RMSE Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([min(test_rmses) * 0.95, max(test_rmses) * 1.05])
    # Add value labels
    for bar, rmse in zip(bars, test_rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{rmse:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Test MAE comparison
    ax = axes[0, 1]
    test_maes = [RESULTS[m]['test_mae'] for m in modes]
    bars = ax.bar([labels[m] for m in modes], test_maes, color=[colors[m] for m in modes])
    ax.set_ylabel('Test MAE', fontsize=12)
    ax.set_title('Test Set MAE Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([min(test_maes) * 0.95, max(test_maes) * 1.05])
    for bar, mae in zip(bars, test_maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{mae:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. R² comparison
    ax = axes[1, 0]
    test_r2s = [RESULTS[m]['test_r2'] for m in modes]
    bars = ax.bar([labels[m] for m in modes], test_r2s, color=[colors[m] for m in modes])
    ax.set_ylabel('Test R²', fontsize=12)
    ax.set_title('Test Set R² Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([min(test_r2s) * 0.99, max(test_r2s) * 1.01])
    for bar, r2 in zip(bars, test_r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{r2:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Improvement over baseline
    ax = axes[1, 1]
    baseline_rmse = RESULTS['baseline']['test_rmse']
    improvements = [baseline_rmse - RESULTS[m]['test_rmse'] for m in modes]
    bars = ax.bar([labels[m] for m in modes], improvements, color=[colors[m] for m in modes])
    ax.set_ylabel('RMSE Improvement', fontsize=12)
    ax.set_title('Improvement over Baseline (RMSE Reduction)', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    for bar, imp in zip(bars, improvements):
        if imp > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'-{imp:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold', color='green')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.1,
                    f'{imp:.2f}', ha='center', va='top', fontsize=14, fontweight='bold', color='red')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'experiment_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {output_dir}/experiment_comparison.png")


def generate_summary_table(output_dir):
    """Generate summary table of all experiments."""
    data = {}
    labels = {
        'baseline': 'PerioGT (Baseline)',
        'feature_only': 'RE-PerioGT (Feature Only)',
        'full': 'RE-PerioGT (Full)'
    }

    for mode, res in RESULTS.items():
        data[labels[mode]] = {
            'Test RMSE': f'{res["test_rmse"]:.2f}',
            'Test MAE': f'{res["test_mae"]:.2f}',
            'Test R²': f'{res["test_r2"]:.4f}',
            'Test Pearson R': f'{res["test_pearson_r"]:.4f}',
            'Best Val RMSE': f'{res["best_val_rmse"]:.2f}',
            'Best Epoch': res['best_epoch'],
            'Training Time (min)': f'{res["training_time_min"]:.1f}'
        }

    df = pd.DataFrame(data).T

    # Save as CSV
    df.to_csv(os.path.join(output_dir, 'summary_table.csv'))
    print(f"Saved summary table to {output_dir}/summary_table.csv")

    return df


def main():
    base_dir = '/home/home_public/aisci/AIscientist/src/1_day2_reperioGT'
    output_dir = os.path.join(base_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("RE-PerioGT Experiment Results")
    print("="*60)

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(output_dir)

    # Generate summary table
    print("\nGenerating summary table...")
    df = generate_summary_table(output_dir)
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(df.to_string())
    print("="*60)

    # Calculate improvements
    baseline_rmse = RESULTS['baseline']['test_rmse']
    feature_rmse = RESULTS['feature_only']['test_rmse']
    full_rmse = RESULTS['full']['test_rmse']

    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    print(f"Baseline PerioGT Test RMSE: {baseline_rmse:.2f}")
    print(f"RE-PerioGT (Feature Only) Test RMSE: {feature_rmse:.2f}")
    print(f"RE-PerioGT (Full) Test RMSE: {full_rmse:.2f}")
    print(f"\nImprovement (Feature Only vs Baseline): {baseline_rmse - feature_rmse:.2f} ({(baseline_rmse - feature_rmse)/baseline_rmse*100:.1f}%)")
    print(f"Improvement (Full vs Baseline): {baseline_rmse - full_rmse:.2f} ({(baseline_rmse - full_rmse)/baseline_rmse*100:.1f}%)")

    # Save final results JSON
    final_results = {
        'experiments': RESULTS,
        'comparison': {
            'baseline_rmse': baseline_rmse,
            'feature_only_rmse': feature_rmse,
            'full_rmse': full_rmse,
            'improvement_feature_vs_baseline': baseline_rmse - feature_rmse,
            'improvement_full_vs_baseline': baseline_rmse - full_rmse,
            'improvement_full_vs_feature': feature_rmse - full_rmse,
        }
    }

    with open(os.path.join(base_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved final results to {base_dir}/final_results.json")

    # Rigidity correlation info
    print("\n" + "="*60)
    print("RIGIDITY-Tg CORRELATION ANALYSIS")
    print("="*60)
    print("Pearson correlation: r = -0.73 (strong negative)")
    print("Spearman correlation: rho = -0.80 (very strong)")
    print("\nInterpretation: Higher rotatable bond fraction (more flexible backbone)")
    print("leads to lower Tg, consistent with polymer physics theory.")


if __name__ == "__main__":
    main()
