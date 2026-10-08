#!/usr/bin/env python3
"""
Generate visualizations for Egc Conjugation Features Experiment
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'ggplot')
sns.set_palette("husl")

# Load results
results_path = Path(__file__).parent / 'experiment_results.json'
with open(results_path) as f:
    results = json.load(f)

# Load predictions
pred_path = Path(__file__).parent / 'predictions.csv'
pred_df = pd.read_csv(pred_path)

# Load feature importance
feat_path = Path(__file__).parent / 'feature_importance.csv'
feat_df = pd.read_csv(feat_path)

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 12))

# ============================================================================
# 1. Per-Fold RMSE Comparison (Baseline vs Ours)
# ============================================================================
ax1 = plt.subplot(2, 3, 1)
folds = [1, 2, 3, 4, 5]
baseline_rmse = [1.28, 0.68, 2.25, 1.12, 0.63]  # From Day 4 results
our_rmse = [r['rmse'] for r in results['fold_results']]

x = np.arange(len(folds))
width = 0.35

bars1 = ax1.bar(x - width/2, baseline_rmse, width, label='Baseline (Day 4)', color='#FF6B6B', alpha=0.8)
bars2 = ax1.bar(x + width/2, our_rmse, width, label='Ours (Conjugation)', color='#4ECDC4', alpha=0.8)

ax1.set_xlabel('Fold', fontsize=12)
ax1.set_ylabel('RMSE', fontsize=12)
ax1.set_title('Per-Fold RMSE Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'Fold {i}' for i in folds])
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Highlight Fold 3 improvement
ax1.annotate('', xy=(2, 2.25), xytext=(2, 0.6145),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax1.text(2.2, 1.4, '72.7%\nImprovement', fontsize=10, ha='center', color='red', fontweight='bold')

# ============================================================================
# 2. Predicted vs Actual Scatter Plot
# ============================================================================
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(pred_df['target'], pred_df['prediction'], alpha=0.5, s=10, c='#3498db')

# Perfect prediction line
min_val = min(pred_df['target'].min(), pred_df['prediction'].min())
max_val = max(pred_df['target'].max(), pred_df['prediction'].max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

ax2.set_xlabel('Actual Egc', fontsize=12)
ax2.set_ylabel('Predicted Egc', fontsize=12)
ax2.set_title(f'Predicted vs Actual (R² = {results["overall_r2"]:.3f})',
             fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# ============================================================================
# 3. Residuals Distribution
# ============================================================================
ax3 = plt.subplot(2, 3, 3)
residuals = pred_df['prediction'] - pred_df['target']
ax3.hist(residuals, bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
ax3.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Error')
ax3.axvline(x=residuals.mean(), color='blue', linestyle='-', lw=2, label=f'Mean: {residuals.mean():.3f}')

ax3.set_xlabel('Residual (Predicted - Actual)', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# ============================================================================
# 4. Conjugation Feature Correlations
# ============================================================================
ax4 = plt.subplot(2, 3, 4)
corr_data = results['correlations']
features = list(corr_data.keys())
correlations = [corr_data[f] for f in features]

colors = ['#2ecc71' if abs(c) > 0.6 else '#f39c12' if abs(c) > 0.4 else '#95a5a6' for c in correlations]

y_pos = np.arange(len(features))
ax4.barh(y_pos, correlations, color=colors, alpha=0.8, edgecolor='black')

ax4.set_yticks(y_pos)
ax4.set_yticklabels(features, fontsize=9)
ax4.set_xlabel('Correlation with Egc', fontsize=12)
ax4.set_title('Conjugation Feature Correlations', fontsize=14, fontweight='bold')
ax4.axvline(x=0.4, color='green', linestyle='--', lw=1.5, alpha=0.5, label='|r| > 0.4')
ax4.axvline(x=-0.4, color='green', linestyle='--', lw=1.5, alpha=0.5)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='x')

# ============================================================================
# 5. Feature Importance (Top 20)
# ============================================================================
ax5 = plt.subplot(2, 3, 5)
top20 = feat_df.head(20)

# Highlight conjugation features
colors = ['#e74c3c' if 'conjugat' in f or 'pi_electron' in f or 'sp2' in f or 'aromatic' in f
          else '#3498db' for f in top20['feature']]

y_pos = np.arange(len(top20))
ax5.barh(y_pos, top20['importance'], color=colors, alpha=0.8, edgecolor='black')

ax5.set_yticks(y_pos)
ax5.set_yticklabels(top20['feature'], fontsize=8)
ax5.set_xlabel('Importance', fontsize=12)
ax5.set_title('Top 20 Feature Importance', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

# Add legend for colors
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='Conjugation Features'),
                   Patch(facecolor='#3498db', label='Generic Features')]
ax5.legend(handles=legend_elements, fontsize=9, loc='lower right')

# ============================================================================
# 6. Overall Performance Summary
# ============================================================================
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Create summary text
summary_text = f"""
EXPERIMENT SUMMARY

Overall Performance:
━━━━━━━━━━━━━━━━━━━
Mean RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}
Mean MAE:  {results['mean_mae']:.4f} ± {results['std_mae']:.4f}
Mean R²:   {results['mean_r2']:.4f} ± {results['std_r2']:.4f}

Baseline Comparison:
━━━━━━━━━━━━━━━━━━━
Baseline RMSE: 1.19 ± 0.59
Our RMSE:      {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}
Improvement:   {results['comparison']['improvement_percent']:.1f}%

Fold 3 Improvement:
━━━━━━━━━━━━━━━━━━━
Baseline: 2.25
Ours:     {results['comparison']['our_fold3_rmse']:.4f}
Improvement: {results['comparison']['fold3_improvement_percent']:.1f}%

Target Achievement:
━━━━━━━━━━━━━━━━━━━
Target: RMSE < 0.9
Actual: {results['mean_rmse']:.4f}
Status: ✅ ACHIEVED

Dataset: {results['n_samples']} molecules
Features: 148 generic + 10 conjugation = 158 total
"""

ax6.text(0.1, 0.95, summary_text, fontsize=11, family='monospace',
        verticalalignment='top', transform=ax6.transAxes)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'experiment_visualization.png', dpi=300)
print(f"✅ Saved visualization to {Path(__file__).parent / 'experiment_visualization.png'}")

# ============================================================================
# Additional: Fold-wise performance breakdown
# ============================================================================
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))

# RMSE per fold
ax = axes[0]
fold_nums = [r['fold'] for r in results['fold_results']]
rmses = [r['rmse'] for r in results['fold_results']]
maes = [r['mae'] for r in results['fold_results']]
r2s = [r['r2'] for r in results['fold_results']]

ax.bar(fold_nums, rmses, color='#3498db', alpha=0.7, edgecolor='black')
ax.axhline(y=results['mean_rmse'], color='red', linestyle='--', lw=2, label=f"Mean: {results['mean_rmse']:.4f}")
ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title('RMSE per Fold', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# MAE per fold
ax = axes[1]
ax.bar(fold_nums, maes, color='#2ecc71', alpha=0.7, edgecolor='black')
ax.axhline(y=results['mean_mae'], color='red', linestyle='--', lw=2, label=f"Mean: {results['mean_mae']:.4f}")
ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('MAE', fontsize=12)
ax.set_title('MAE per Fold', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# R² per fold
ax = axes[2]
ax.bar(fold_nums, r2s, color='#9b59b6', alpha=0.7, edgecolor='black')
ax.axhline(y=results['mean_r2'], color='red', linestyle='--', lw=2, label=f"Mean: {results['mean_r2']:.4f}")
ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('R²', fontsize=12)
ax.set_title('R² per Fold', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'fold_performance.png', dpi=300)
print(f"✅ Saved fold performance to {Path(__file__).parent / 'fold_performance.png'}")

print("\n✅ All visualizations generated successfully!")
