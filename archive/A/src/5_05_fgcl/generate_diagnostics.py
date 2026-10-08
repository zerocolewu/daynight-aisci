"""
Generate diagnostic visualizations for FGCL results
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create diagnostics directory
os.makedirs('diagnostics', exist_ok=True)

# Load results
results_df = pd.read_csv('results/results.csv')

with open('results/detailed_results.json', 'r') as f:
    detailed_results = json.load(f)

print(f"Loaded {len(results_df)} results")

# 1. Performance Comparison Across Ablation Modes
print("\n1. Creating performance comparison plot...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# RMSE comparison
ax1 = axes[0]
modes = ['supervised_only', 'pretrain_only', 'regularization_only', 'full']
mode_labels = ['Supervised\nOnly', 'Pre-train\nOnly', 'Regularization\nOnly', 'Full FGCL']

data_rmse = [results_df[results_df['ablation_mode'] == mode]['test_rmse'].values for mode in modes]
bp1 = ax1.boxplot(data_rmse, labels=mode_labels, patch_artist=True)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
ax1.set_ylabel('Test RMSE', fontsize=12, fontweight='bold')
ax1.set_title('Test RMSE by Ablation Mode', fontsize=14, fontweight='bold')
ax1.axhline(y=38.78, color='red', linestyle='--', linewidth=2, label='Target: 38.78')
ax1.legend()
ax1.grid(True, alpha=0.3)

# MAE comparison
ax2 = axes[1]
data_mae = [results_df[results_df['ablation_mode'] == mode]['test_mae'].values for mode in modes]
bp2 = ax2.boxplot(data_mae, labels=mode_labels, patch_artist=True)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
ax2.set_ylabel('Test MAE', fontsize=12, fontweight='bold')
ax2.set_title('Test MAE by Ablation Mode', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# R2 comparison
ax3 = axes[2]
data_r2 = [results_df[results_df['ablation_mode'] == mode]['test_r2'].values for mode in modes]
bp3 = ax3.boxplot(data_r2, labels=mode_labels, patch_artist=True)
for patch, color in zip(bp3['boxes'], colors):
    patch.set_facecolor(color)
ax3.set_ylabel('Test R²', fontsize=12, fontweight='bold')
ax3.set_title('Test R² by Ablation Mode', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostics/performance_comparison.png', dpi=300, bbox_inches='tight')
print("   Saved: diagnostics/performance_comparison.png")

# 2. Lambda Ablation Study
print("\n2. Creating lambda ablation plot...")

lambda_modes = ['full_lambda_0.1', 'full_lambda_0.5', 'full_lambda_1.0', 'full_lambda_2.0']
lambda_values = [0.1, 0.5, 1.0, 2.0]

lambda_rmse_mean = []
lambda_rmse_std = []

for mode in lambda_modes:
    rmse_vals = results_df[results_df['ablation_mode'] == mode]['test_rmse'].values
    lambda_rmse_mean.append(np.mean(rmse_vals))
    lambda_rmse_std.append(np.std(rmse_vals))

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(lambda_values, lambda_rmse_mean, yerr=lambda_rmse_std,
            marker='o', linewidth=2, markersize=10, capsize=5, capthick=2, color='#2E86AB')
ax.axhline(y=38.78, color='red', linestyle='--', linewidth=2, label='Target: 38.78')
ax.axhline(y=41.59, color='green', linestyle='--', linewidth=2, label='Supervised Baseline: 41.59')
ax.set_xlabel('Lambda (λ)', fontsize=12, fontweight='bold')
ax.set_ylabel('Test RMSE', fontsize=12, fontweight='bold')
ax.set_title('Lambda Ablation Study: Effect of Contrastive Loss Weight', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostics/lambda_ablation.png', dpi=300, bbox_inches='tight')
print("   Saved: diagnostics/lambda_ablation.png")

# 3. Fold-wise Performance
print("\n3. Creating fold-wise performance plot...")

fig, ax = plt.subplots(figsize=(12, 6))

for mode in modes:
    mode_data = results_df[results_df['ablation_mode'] == mode]
    ax.plot(mode_data['fold'], mode_data['test_rmse'], marker='o', linewidth=2,
            label=mode.replace('_', ' ').title(), markersize=8)

ax.axhline(y=38.78, color='red', linestyle='--', linewidth=2, label='Target: 38.78')
ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
ax.set_ylabel('Test RMSE', fontsize=12, fontweight='bold')
ax.set_title('Test RMSE Across 5 Folds', fontsize=14, fontweight='bold')
ax.set_xticks([1, 2, 3, 4, 5])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostics/foldwise_performance.png', dpi=300, bbox_inches='tight')
print("   Saved: diagnostics/foldwise_performance.png")

# 4. Predictions vs Targets (Best Model)
print("\n4. Creating predictions vs targets plot...")

# Find best performing fold
best_idx = results_df['test_rmse'].idxmin()
best_result = results_df.loc[best_idx]
print(f"   Best model: {best_result['ablation_mode']}, Fold {best_result['fold']}, RMSE {best_result['test_rmse']:.2f}")

# Get predictions for this fold
fold_results = detailed_results[best_idx]

predictions = np.array(fold_results['predictions'])
targets = np.array(fold_results['targets'])

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(targets, predictions, alpha=0.5, s=20, color='#2E86AB')
ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()],
        'r--', linewidth=2, label='Perfect Prediction')

ax.set_xlabel('True Tg (°C)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Tg (°C)', fontsize=12, fontweight='bold')
ax.set_title(f'Predictions vs True Values\n{best_result["ablation_mode"].replace("_", " ").title()} - Fold {best_result["fold"]}\nRMSE: {best_result["test_rmse"]:.2f}',
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Add metrics text
textstr = f'RMSE: {best_result["test_rmse"]:.2f}\nMAE: {best_result["test_mae"]:.2f}\nR²: {best_result["test_r2"]:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, fontweight='bold')

plt.tight_layout()
plt.savefig('diagnostics/predictions_vs_targets.png', dpi=300, bbox_inches='tight')
print("   Saved: diagnostics/predictions_vs_targets.png")

# 5. Error Distribution
print("\n5. Creating error distribution plot...")

errors = predictions - targets

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax1 = axes[0]
ax1.hist(errors, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax1.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
ax1.set_xlabel('Prediction Error (°C)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Error vs True Value
ax2 = axes[1]
ax2.scatter(targets, errors, alpha=0.5, s=20, color='#2E86AB')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('True Tg (°C)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Prediction Error (°C)', fontsize=12, fontweight='bold')
ax2.set_title('Error vs True Value', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostics/error_distribution.png', dpi=300, bbox_inches='tight')
print("   Saved: diagnostics/error_distribution.png")

# 6. Training Curves (for best model)
print("\n6. Creating training curves...")

# Get training history for best model
best_detailed = detailed_results[best_idx]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Supervised loss
ax1 = axes[0, 0]
ax1.plot(best_detailed['finetune_supervised_losses'], linewidth=2, color='#2E86AB')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Supervised Loss (MSE)', fontsize=12, fontweight='bold')
ax1.set_title('Supervised Loss During Fine-tuning', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Contrastive loss (if available)
ax2 = axes[0, 1]
if best_detailed['ablation_mode'] not in ['supervised_only', 'pretrain_only']:
    ax2.plot(best_detailed['finetune_contrastive_losses'], linewidth=2, color='#FF6B6B')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Contrastive Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Contrastive Loss During Fine-tuning', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, 'No Contrastive Loss\n(Supervised Only or Pre-train Only)',
             ha='center', va='center', fontsize=14, fontweight='bold')
    ax2.set_title('Contrastive Loss (Not Used)', fontsize=14, fontweight='bold')

# Total loss
ax3 = axes[1, 0]
ax3.plot(best_detailed['finetune_total_losses'], linewidth=2, color='#45B7D1')
ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax3.set_ylabel('Total Loss', fontsize=12, fontweight='bold')
ax3.set_title('Total Loss During Fine-tuning', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Validation RMSE
ax4 = axes[1, 1]
ax4.plot(best_detailed['val_rmses'], linewidth=2, color='#FFA07A')
ax4.axhline(y=best_detailed['best_val_rmse'], color='red', linestyle='--', linewidth=2,
            label=f'Best Val RMSE: {best_detailed["best_val_rmse"]:.2f}')
ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax4.set_ylabel('Validation RMSE', fontsize=12, fontweight='bold')
ax4.set_title('Validation RMSE During Training', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostics/training_curves.png', dpi=300, bbox_inches='tight')
print("   Saved: diagnostics/training_curves.png")

# 7. Summary Statistics Table
print("\n7. Creating summary statistics table...")

summary_stats = []

for mode in modes + lambda_modes:
    mode_data = results_df[results_df['ablation_mode'] == mode]

    summary_stats.append({
        'Ablation Mode': mode.replace('_', ' ').title(),
        'Mean RMSE': f"{mode_data['test_rmse'].mean():.2f} ± {mode_data['test_rmse'].std():.2f}",
        'Mean MAE': f"{mode_data['test_mae'].mean():.2f} ± {mode_data['test_mae'].std():.2f}",
        'Mean R²': f"{mode_data['test_r2'].mean():.3f} ± {mode_data['test_r2'].std():.3f}",
        'Best RMSE': f"{mode_data['test_rmse'].min():.2f}",
        'Worst RMSE': f"{mode_data['test_rmse'].max():.2f}"
    })

summary_df = pd.DataFrame(summary_stats)

# Save as CSV
summary_df.to_csv('diagnostics/summary_statistics.csv', index=False)
print("   Saved: diagnostics/summary_statistics.csv")

# Create visualization table
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=summary_df.values,
                colLabels=summary_df.columns,
                cellLoc='center',
                loc='center')

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for i in range(len(summary_df.columns)):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold')

# Style rows
for i in range(1, len(summary_df) + 1):
    if i <= 4:  # Main ablation modes
        for j in range(len(summary_df.columns)):
            table[(i, j)].set_facecolor('#F0F8FF')
    else:  # Lambda ablation
        for j in range(len(summary_df.columns)):
            table[(i, j)].set_facecolor('#FFF8DC')

plt.title('FGCL Ablation Study - Summary Statistics', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('diagnostics/summary_table.png', dpi=300, bbox_inches='tight')
print("   Saved: diagnostics/summary_table.png")

print("\n" + "="*80)
print("All diagnostics generated successfully!")
print("="*80)
print("\nGenerated files:")
print("  - diagnostics/performance_comparison.png")
print("  - diagnostics/lambda_ablation.png")
print("  - diagnostics/foldwise_performance.png")
print("  - diagnostics/predictions_vs_targets.png")
print("  - diagnostics/error_distribution.png")
print("  - diagnostics/training_curves.png")
print("  - diagnostics/summary_statistics.csv")
print("  - diagnostics/summary_table.png")
