"""
Analysis script for PolyPerio-DenseGNN ablation study.
Generates visualizations and detailed analysis report.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

RESULTS_PATH = '/home/home_public/aisci/AIscientist/src/7_day_7_polyperio/results/ablation_results.json'
OUTPUT_DIR = '/home/home_public/aisci/AIscientist/src/7_day_7_polyperio/results'

with open(RESULTS_PATH) as f:
    results = json.load(f)

# ============================================================
# 1. Bar chart: RMSE comparison across variants
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

variants = ['baseline', 'periodic', 'angular', 'full']
labels = ['Baseline\n(Monomer)', '+ Periodic\n(Trimer)', '+ Angular\nEncoding', 'Full Model\n(+ Dense)']
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

# RMSE
rmse_means = [results[v]['test_rmse_mean'] for v in variants]
rmse_stds = [results[v]['test_rmse_std'] for v in variants]
bars = axes[0].bar(labels, rmse_means, yerr=rmse_stds, color=colors, capsize=5, edgecolor='black', linewidth=0.5)
axes[0].set_ylabel('Test RMSE', fontsize=12)
axes[0].set_title('RMSE Comparison', fontsize=13, fontweight='bold')
axes[0].axhline(y=30.8, color='red', linestyle='--', alpha=0.7, label='PerioGT SOTA (30.8)')
axes[0].legend(fontsize=9)
for bar, mean in zip(bars, rmse_means):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# MAE
mae_means = [results[v]['test_mae_mean'] for v in variants]
mae_stds = [results[v]['test_mae_std'] for v in variants]
bars = axes[1].bar(labels, mae_means, yerr=mae_stds, color=colors, capsize=5, edgecolor='black', linewidth=0.5)
axes[1].set_ylabel('Test MAE', fontsize=12)
axes[1].set_title('MAE Comparison', fontsize=13, fontweight='bold')
for bar, mean in zip(bars, mae_means):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# R2
r2_means = [results[v]['test_r2_mean'] for v in variants]
r2_stds = [results[v]['test_r2_std'] for v in variants]
bars = axes[2].bar(labels, r2_means, yerr=r2_stds, color=colors, capsize=5, edgecolor='black', linewidth=0.5)
axes[2].set_ylabel('Test R2', fontsize=12)
axes[2].set_title('R2 Comparison', fontsize=13, fontweight='bold')
for bar, mean in zip(bars, r2_means):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_comparison.png'), dpi=150, bbox_inches='tight')
print("Saved: ablation_comparison.png")

# ============================================================
# 2. Box plot: Individual seed RMSEs
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
data = [results[v]['test_rmse_all'] for v in variants]
bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(y=30.8, color='red', linestyle='--', alpha=0.7, label='PerioGT SOTA (30.8)')
ax.set_ylabel('Test RMSE', fontsize=12)
ax.set_title('RMSE Distribution Across Seeds', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)

# Add individual points
for i, (d, color) in enumerate(zip(data, colors)):
    x = np.random.normal(i+1, 0.04, size=len(d))
    ax.scatter(x, d, alpha=0.7, color=color, edgecolor='black', s=50, zorder=3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rmse_boxplot.png'), dpi=150, bbox_inches='tight')
print("Saved: rmse_boxplot.png")

# ============================================================
# 3. Improvement over baseline
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
baseline_rmse = results['baseline']['test_rmse_mean']
improvements = [(baseline_rmse - results[v]['test_rmse_mean']) / baseline_rmse * 100 for v in variants]
bars = ax.bar(labels, improvements, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('RMSE Improvement over Baseline (%)', fontsize=12)
ax.set_title('Relative Improvement from Each Component', fontsize=13, fontweight='bold')
for bar, imp in zip(bars, improvements):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
            f'{imp:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'improvement_over_baseline.png'), dpi=150, bbox_inches='tight')
print("Saved: improvement_over_baseline.png")

# ============================================================
# 4. Print detailed analysis
# ============================================================
print("\n" + "="*80)
print("DETAILED ANALYSIS REPORT")
print("="*80)

print("\n1. ABLATION STUDY RESULTS")
print("-"*80)
print(f"{'Variant':<15} {'Params':>10} {'RMSE':>18} {'MAE':>18} {'R2':>18}")
print("-"*80)
for v in variants:
    r = results[v]
    print(f"{v:<15} {r['num_params']:>10,d} "
          f"{r['test_rmse_mean']:>7.2f} +/- {r['test_rmse_std']:<7.2f} "
          f"{r['test_mae_mean']:>7.2f} +/- {r['test_mae_std']:<7.2f} "
          f"{r['test_r2_mean']:>7.4f} +/- {r['test_r2_std']:<7.4f}")

print("\n2. COMPONENT CONTRIBUTIONS")
print("-"*80)
print(f"Baseline (monomer, no periodic, no angular): RMSE = {results['baseline']['test_rmse_mean']:.2f}")
print(f"+ Periodic edge (trimer + periodic):        RMSE = {results['periodic']['test_rmse_mean']:.2f}  "
      f"(improvement: {results['baseline']['test_rmse_mean'] - results['periodic']['test_rmse_mean']:.2f})")
print(f"+ Angular encoding (directional info):       RMSE = {results['angular']['test_rmse_mean']:.2f}  "
      f"(improvement: {results['periodic']['test_rmse_mean'] - results['angular']['test_rmse_mean']:.2f})")
print(f"+ Dense connectivity (deep supervision):     RMSE = {results['full']['test_rmse_mean']:.2f}  "
      f"(change: {results['angular']['test_rmse_mean'] - results['full']['test_rmse_mean']:+.2f})")

print("\n3. KEY FINDINGS")
print("-"*80)
print("a) Periodic graph (trimer + periodic edge) provides the LARGEST improvement:")
print(f"   {results['baseline']['test_rmse_mean'] - results['periodic']['test_rmse_mean']:.2f} RMSE reduction "
      f"({(results['baseline']['test_rmse_mean'] - results['periodic']['test_rmse_mean'])/results['baseline']['test_rmse_mean']*100:.1f}%)")
print(f"   This confirms that polymer periodicity representation is crucial for Tg prediction.")

print(f"\nb) Angular encoding provides ADDITIONAL improvement:")
print(f"   {results['periodic']['test_rmse_mean'] - results['angular']['test_rmse_mean']:.2f} RMSE reduction "
      f"({(results['periodic']['test_rmse_mean'] - results['angular']['test_rmse_mean'])/results['periodic']['test_rmse_mean']*100:.1f}%)")
print(f"   Directional information complements distance-based features.")

print(f"\nc) Dense connectivity HURTS performance:")
print(f"   {results['full']['test_rmse_mean'] - results['angular']['test_rmse_mean']:.2f} RMSE increase")
print(f"   Likely adds too many parameters ({results['full']['num_params']:,} vs {results['angular']['num_params']:,})")
print(f"   for this dataset size, causing mild overfitting despite implicit regularization claims.")

print("\n4. COMPARISON TO SOTA")
print("-"*80)
print(f"PerioGT SOTA: RMSE = 30.8")
print(f"Our best (angular): RMSE = {results['angular']['test_rmse_mean']:.2f} +/- {results['angular']['test_rmse_std']:.2f}")
print(f"Gap: {results['angular']['test_rmse_mean'] - 30.8:.2f} RMSE")
print(f"Note: Different data splits may contribute to this gap.")

print("\n5. PARAMETER COUNT ANALYSIS")
print("-"*80)
for v in variants:
    r = results[v]
    print(f"  {v}: {r['num_params']:>10,d} params")

print("\n6. INDIVIDUAL SEED RESULTS")
print("-"*80)
seeds = [42, 123, 456, 789, 2024]
header = f"{'Variant':<12}" + "".join([f"{'seed='+str(s):>10}" for s in seeds])
print(header)
for v in variants:
    rmses = results[v]['test_rmse_all']
    row = f"{v:<12}" + "".join([f"{r:>10.2f}" for r in rmses])
    print(row)

print("\n" + "="*80)
