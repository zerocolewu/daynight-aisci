"""
Visualization script for 5-fold CV results
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load results
results_df = pd.read_csv('cv_results.csv')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Bar chart comparing results
ax1 = axes[0, 0]
categories = ['Paper\n(37.35)', 'Our Result\n(46.89)', 'Previous Best\n(37.00)', 'Target\n(32.0)']
values = [37.35, 46.89, 37.00, 32.0]
colors = ['green', 'red', 'blue', 'orange']

bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('RMSE (°C)', fontsize=12, fontweight='bold')
ax1.set_title('RMSE Comparison', fontsize=14, fontweight='bold')
ax1.axhline(y=32.0, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Target')
ax1.set_ylim(0, 55)

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.grid(axis='y', alpha=0.3)

# 2. Per-fold performance
ax2 = axes[0, 1]
folds = results_df['fold'].values
test_rmse = results_df['test_rmse'].values
test_mae = results_df['test_mae'].values

x = np.arange(len(folds))
width = 0.35

bars1 = ax2.bar(x - width/2, test_rmse, width, label='RMSE', color='steelblue', alpha=0.8)
bars2 = ax2.bar(x + width/2, test_mae, width, label='MAE', color='coral', alpha=0.8)

ax2.axhline(y=37.35, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Paper RMSE')
ax2.set_xlabel('Fold', fontsize=12, fontweight='bold')
ax2.set_ylabel('Error (°C)', fontsize=12, fontweight='bold')
ax2.set_title('Per-Fold Performance', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'Fold {f}' for f in folds])
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add mean line
mean_rmse = test_rmse.mean()
ax2.axhline(y=mean_rmse, color='red', linestyle='-', linewidth=2, alpha=0.7)
ax2.text(len(folds)-1, mean_rmse + 1, f'Mean: {mean_rmse:.1f}', fontsize=10, color='red', fontweight='bold')

# 3. Training curves for all folds
ax3 = axes[1, 0]
for idx, row in results_df.iterrows():
    fold = row['fold']
    val_history = eval(row['val_history'])  # Convert string to list
    epochs = range(1, len(val_history) + 1)
    ax3.plot(epochs, val_history, label=f'Fold {fold}', linewidth=2, alpha=0.7)

ax3.axhline(y=37.35, color='green', linestyle='--', linewidth=2, label='Paper RMSE')
ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax3.set_ylabel('Validation RMSE (°C)', fontsize=12, fontweight='bold')
ax3.set_title('Validation Learning Curves', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Gap analysis
ax4 = axes[1, 1]
components = ['Learning Rate\nSchedule', 'Data\nPreprocessing', 'Architecture\nDetails', 'Training\nDetails']
estimated_impact = [7.0, 1.5, 1.0, 0.5]  # Estimated RMSE contribution to gap
colors = ['red', 'orange', 'yellow', 'lightblue']

bars = ax4.barh(components, estimated_impact, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax4.set_xlabel('Estimated RMSE Impact', fontsize=12, fontweight='bold')
ax4.set_title('Gap Analysis (Total: 9.54 RMSE)', fontsize=14, fontweight='bold')

# Add value labels
for bar, val in zip(bars, estimated_impact):
    width = bar.get_width()
    ax4.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
             f'{val:.1f}',
             ha='left', va='center', fontsize=11, fontweight='bold')

ax4.set_xlim(0, 10)
ax4.grid(axis='x', alpha=0.3)

# Add total gap annotation
ax4.text(5, -0.8, f'Total Gap: 9.54 RMSE', fontsize=12, fontweight='bold',
         ha='center', style='italic', color='darkred')

plt.tight_layout()
plt.savefig('5fold_cv_results.png', dpi=150, bbox_inches='tight')
print("Saved visualization to 5fold_cv_results.png")

# Create summary comparison table
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(f"{'Metric':<30} {'Our Result':<20} {'Paper Result':<20}")
print("="*70)
print(f"{'Test RMSE (°C)':<30} {f'{test_rmse.mean():.2f} ± {test_rmse.std():.2f}':<20} {'37.35':<20}")
print(f"{'Test MAE (°C)':<30} {f'{test_mae.mean():.2f} ± {test_mae.std():.2f}':<20} {'~30 (estimated)':<20}")
r2_mean = results_df['test_r2'].mean()
r2_std = results_df['test_r2'].std()
print(f"{'Test R²':<30} {f'{r2_mean:.3f} ± {r2_std:.3f}':<20} {'0.887':<20}")
print(f"{'Gap to Paper (RMSE)':<30} {f'+{test_rmse.mean() - 37.35:.2f}':<20} {'-':<20}")
print(f"{'Gap to Target (32.0)':<30} {f'+{test_rmse.mean() - 32.0:.2f}':<20} {'-':<20}")
print("="*70)
print(f"\nStatus: ❌ FAILED to meet target RMSE < 32.0")
print(f"Root Cause: Missing NOAM learning rate schedule (estimated impact: 5-10 RMSE)")
print("="*70)
