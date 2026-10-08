#!/usr/bin/env python3
"""
Generate analysis plots for multi-seed ensemble with uncertainty quantification.
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_squared_error

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results():
    """Load results from JSON file."""
    base_dir = Path(__file__).parent
    results_path = base_dir / "results.json"

    with open(results_path, 'r') as f:
        results = json.load(f)

    return results


def load_predictions(property_name):
    """Load predictions from pickle file."""
    base_dir = Path(__file__).parent
    predictions_path = base_dir / "checkpoints" / f"{property_name}_predictions.pkl"

    with open(predictions_path, 'rb') as f:
        predictions = pickle.load(f)

    return predictions


def plot_fold_performance(results, property_name, output_dir):
    """Plot fold-level performance comparison."""
    prop_results = results[property_name.lower()]
    fold_results = prop_results['fold_results']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Individual seeds vs ensemble RMSE
    folds = [f['fold'] for f in fold_results]
    individual_means = [f['individual_seed_rmse_mean'] for f in fold_results]
    individual_stds = [f['individual_seed_rmse_std'] for f in fold_results]
    ensemble_rmses = [f['ensemble_rmse'] for f in fold_results]

    # Plot 1: RMSE comparison
    ax = axes[0]
    x = np.arange(len(folds))
    width = 0.35

    bars1 = ax.bar(x - width/2, individual_means, width, yerr=individual_stds,
                   label='Individual Seeds', capsize=5, alpha=0.7)
    bars2 = ax.bar(x + width/2, ensemble_rmses, width,
                   label='Ensemble', alpha=0.7)

    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(f'{property_name}: Individual Seeds vs Ensemble RMSE', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Variance across seeds
    ax = axes[1]
    seed_variances = [f['individual_seed_rmse_std']**2 for f in fold_results]
    ax.bar(folds, seed_variances, alpha=0.7, color='coral')
    ax.axhline(y=np.mean(seed_variances), color='red', linestyle='--',
               label=f'Mean: {np.mean(seed_variances):.2f}')
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Variance', fontsize=12)
    ax.set_title(f'{property_name}: Variance Across Seeds', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Improvement from ensemble
    ax = axes[2]
    improvements = [(ind - ens) / ind * 100
                   for ind, ens in zip(individual_means, ensemble_rmses)]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax.bar(folds, improvements, alpha=0.7, color=colors)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title(f'{property_name}: Ensemble Improvement', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{property_name.lower()}_fold_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {property_name.lower()}_fold_performance.png")


def plot_uncertainty_vs_error(results, property_name, predictions, output_dir):
    """Plot uncertainty vs prediction error scatter plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (fold_idx, fold_data) in enumerate(predictions.items()):
        if idx >= 6:
            break

        ax = axes[idx]
        ensemble_pred = fold_data['ensemble_pred']
        ensemble_std = fold_data['ensemble_std']
        targets = fold_data['targets']

        abs_errors = np.abs(targets - ensemble_pred)

        # Scatter plot
        ax.scatter(ensemble_std, abs_errors, alpha=0.5, s=10)

        # Calculate correlation
        correlation, p_value = stats.pearsonr(ensemble_std, abs_errors)

        # Add trend line
        z = np.polyfit(ensemble_std, abs_errors, 1)
        p = np.poly1d(z)
        x_line = np.linspace(ensemble_std.min(), ensemble_std.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2,
               label=f'r={correlation:.3f}')

        ax.set_xlabel('Uncertainty (std)', fontsize=10)
        ax.set_ylabel('Absolute Error', fontsize=10)
        ax.set_title(f'Fold {fold_idx}', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(predictions), 6):
        axes[idx].axis('off')

    plt.suptitle(f'{property_name}: Uncertainty vs Error', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / f'{property_name.lower()}_uncertainty_vs_error.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {property_name.lower()}_uncertainty_vs_error.png")


def plot_high_uncertainty_analysis(results, property_name, predictions, output_dir):
    """Analyze high vs low uncertainty samples."""
    prop_results = results[property_name.lower()]
    fold_results = prop_results['fold_results']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: RMSE comparison for high vs low uncertainty
    ax = axes[0]
    folds = [f['fold'] for f in fold_results]
    high_unc_rmse = [f['high_uncertainty_rmse'] for f in fold_results]
    low_unc_rmse = [f['low_uncertainty_rmse'] for f in fold_results]

    x = np.arange(len(folds))
    width = 0.35

    ax.bar(x - width/2, low_unc_rmse, width, label='Low Uncertainty (80%)',
           alpha=0.7, color='green')
    ax.bar(x + width/2, high_unc_rmse, width, label='High Uncertainty (20%)',
           alpha=0.7, color='red')

    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(f'{property_name}: RMSE by Uncertainty Level', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Uncertainty-error correlation
    ax = axes[1]
    correlations = [f['uncertainty_error_correlation'] for f in fold_results]
    p_values = [f['uncertainty_error_pvalue'] for f in fold_results]

    colors = ['green' if p < 0.05 else 'gray' for p in p_values]
    bars = ax.bar(folds, correlations, alpha=0.7, color=colors)

    ax.axhline(y=0.3, color='blue', linestyle='--', alpha=0.5,
               label='Target: r=0.3')
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Correlation (r)', fontsize=12)
    ax.set_title(f'{property_name}: Uncertainty-Error Correlation', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{property_name.lower()}_uncertainty_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {property_name.lower()}_uncertainty_analysis.png")


def plot_prediction_distribution(results, property_name, predictions, output_dir):
    """Plot prediction and uncertainty distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    all_predictions = []
    all_uncertainties = []
    all_targets = []

    for idx, (fold_idx, fold_data) in enumerate(predictions.items()):
        if idx >= 6:
            break

        ax = axes[idx]
        ensemble_pred = fold_data['ensemble_pred']
        ensemble_std = fold_data['ensemble_std']
        targets = fold_data['targets']

        all_predictions.extend(ensemble_pred)
        all_uncertainties.extend(ensemble_std)
        all_targets.extend(targets)

        # Scatter plot of predictions vs targets
        ax.scatter(targets, ensemble_pred, alpha=0.5, s=10, c=ensemble_std,
                  cmap='viridis')

        # Perfect prediction line
        min_val = min(targets.min(), ensemble_pred.min())
        max_val = max(targets.max(), ensemble_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(targets, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(targets, ensemble_pred))

        ax.set_xlabel('Target', fontsize=10)
        ax.set_ylabel('Prediction', fontsize=10)
        ax.set_title(f'Fold {fold_idx}: R²={r2:.3f}, RMSE={rmse:.2f}', fontsize=12)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(predictions), 6):
        axes[idx].axis('off')

    plt.suptitle(f'{property_name}: Predictions vs Targets', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / f'{property_name.lower()}_prediction_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {property_name.lower()}_prediction_distribution.png")


def plot_cross_property_comparison(results, output_dir):
    """Compare results across Tm and Egc."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    properties = ['Tm', 'Egc']
    colors = {'Tm': 'steelblue', 'Egc': 'coral'}

    # Plot 1: Mean RMSE comparison
    ax = axes[0]
    mean_rmses = [results[p.lower()]['cross_fold_ensemble']['mean_rmse'] for p in properties]
    std_rmses = [results[p.lower()]['cross_fold_ensemble']['std_rmse'] for p in properties]

    x = np.arange(len(properties))
    bars = ax.bar(x, mean_rmses, yerr=std_rmses, capsize=10,
                  color=[colors[p] for p in properties], alpha=0.7)

    ax.set_ylabel('Mean RMSE', fontsize=12)
    ax.set_title('Cross-Fold Ensemble Performance', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(properties)
    ax.grid(True, alpha=0.3)

    # Add value labels
    for bar, mean, std in zip(bars, mean_rmses, std_rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01*max(mean_rmses),
               f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10)

    # Plot 2: Variance reduction
    ax = axes[1]
    var_reductions = [results[p.lower()]['cross_fold_ensemble']['variance_reduction'] for p in properties]

    bar_colors = ['green' if vr > 0 else 'red' for vr in var_reductions]
    bars = ax.bar(x, var_reductions, color=bar_colors, alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Variance Reduction (%)', fontsize=12)
    ax.set_title('Ensemble Variance Reduction', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(properties)
    ax.grid(True, alpha=0.3)

    # Plot 3: Fold-wise RMSE distribution
    ax = axes[2]

    for i, prop in enumerate(properties):
        fold_rmses = [f['ensemble_rmse'] for f in results[prop.lower()]['fold_results']]
        ax.boxplot([fold_rmses], positions=[i], widths=0.4, patch_artist=True,
                  boxprops=dict(facecolor=colors[prop], alpha=0.7))

    ax.set_ylabel('Ensemble RMSE', fontsize=12)
    ax.set_title('Fold-wise RMSE Distribution', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(properties)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'cross_property_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: cross_property_comparison.png")


def plot_individual_seed_variability(results, property_name, output_dir):
    """Plot individual seed performance variability."""
    prop_results = results[property_name.lower()]
    fold_results = prop_results['fold_results']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect all seed results
    all_seeds = []
    for fold_result in fold_results:
        for seed_result in fold_result['seed_results']:
            all_seeds.append({
                'fold': fold_result['fold'],
                'seed': seed_result['seed'],
                'rmse': seed_result['test_rmse']
            })

    # Group by fold
    folds = sorted(set(s['fold'] for s in all_seeds))
    seeds = sorted(set(s['seed'] for s in all_seeds))

    x = np.arange(len(folds))
    width = 0.15

    for i, seed in enumerate(seeds):
        seed_rmses = [s['rmse'] for s in all_seeds if s['seed'] == seed]
        ax.bar(x + i*width, seed_rmses, width, label=f'Seed {seed}', alpha=0.7)

    # Add ensemble line
    ensemble_rmses = [f['ensemble_rmse'] for f in fold_results]
    ax.plot(x + width*2, ensemble_rmses, 'k-', linewidth=2, marker='o',
           markersize=8, label='Ensemble')

    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(f'{property_name}: Individual Seed Performance', fontsize=14)
    ax.set_xticks(x + width*2)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.legend(loc='best', ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{property_name.lower()}_seed_variability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {property_name.lower()}_seed_variability.png")


def main():
    """Main function to generate all plots."""
    base_dir = Path(__file__).parent
    output_dir = base_dir / "plots"
    output_dir.mkdir(exist_ok=True)

    print("Loading results...")
    results = load_results()

    print("\nGenerating plots for Tm...")
    tm_predictions = load_predictions("Tm")
    plot_fold_performance(results, "Tm", output_dir)
    plot_uncertainty_vs_error(results, "Tm", tm_predictions, output_dir)
    plot_high_uncertainty_analysis(results, "Tm", tm_predictions, output_dir)
    plot_prediction_distribution(results, "Tm", tm_predictions, output_dir)
    plot_individual_seed_variability(results, "Tm", output_dir)

    print("\nGenerating plots for Egc...")
    egc_predictions = load_predictions("Egc")
    plot_fold_performance(results, "Egc", output_dir)
    plot_uncertainty_vs_error(results, "Egc", egc_predictions, output_dir)
    plot_high_uncertainty_analysis(results, "Egc", egc_predictions, output_dir)
    plot_prediction_distribution(results, "Egc", egc_predictions, output_dir)
    plot_individual_seed_variability(results, "Egc", output_dir)

    print("\nGenerating cross-property comparison...")
    plot_cross_property_comparison(results, output_dir)

    print(f"\nAll plots saved to: {output_dir}")
    print("\nPlot files:")
    for plot_file in sorted(output_dir.glob("*.png")):
        print(f"  - {plot_file.name}")


if __name__ == "__main__":
    main()
