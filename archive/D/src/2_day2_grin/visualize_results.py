"""
Create visualizations and final report for GRIN experiments.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Results directory
results_dir = '/home/home_public/aisci/AIscientist/src/2_day2_grin/results_20260324_072344'

# Load results
experiments = ['baseline', 'max_only', 'max_l1']
results = {}

for exp in experiments:
    json_file = os.path.join(results_dir, exp, 'results.json')
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            results[exp] = json.load(f)

# Create comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Test RMSE comparison
ax = axes[0]
exp_names = ['Baseline\n(Mean, No Aug)', 'Max Only', 'Max + L1']
rmses = [results[exp]['test_rmse'] for exp in experiments]
colors = ['#3498db', '#e74c3c', '#2ecc71']

bars = ax.bar(range(len(rmses)), rmses, color=colors)
ax.set_xticks(range(len(rmses)))
ax.set_xticklabels(exp_names, fontsize=10)
ax.set_ylabel('Test RMSE', fontsize=12)
ax.set_title('Test RMSE Comparison', fontsize=14, fontweight='bold')
ax.axhline(y=37.0, color='red', linestyle='--', linewidth=2, label='GraphFormer Target')
ax.axhline(y=76.08, color='green', linestyle=':', linewidth=2, label='Best Result')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, rmse in zip(bars, rmses):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{rmse:.2f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Test MAE comparison
ax = axes[1]
maes = [results[exp]['test_mae'] for exp in experiments]

bars = ax.bar(range(len(maes)), maes, color=colors)
ax.set_xticks(range(len(maes)))
ax.set_xticklabels(exp_names, fontsize=10)
ax.set_ylabel('Test MAE', fontsize=12)
ax.set_title('Test MAE Comparison', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, mae in zip(bars, maes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{mae:.2f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'comparison_results.png'), dpi=150, bbox_inches='tight')
plt.close()

# Create training curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, exp in enumerate(experiments):
    ax = axes[i]
    history = results[exp]['history']

    epochs = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title(f'{exp.replace("_", " ").title()}\nTest RMSE: {results[exp]["test_rmse"]:.2f}',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# Create validation RMSE curves
plt.figure(figsize=(12, 6))

for exp in experiments:
    history = results[exp]['history']
    epochs = range(1, len(history['val_rmse']) + 1)
    plt.plot(epochs, history['val_rmse'], label=f'{exp.replace("_", " ").title()}',
             linewidth=2, marker='o', markersize=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation RMSE', fontsize=12)
plt.title('Validation RMSE Over Training', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'validation_rmse_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"Visualizations saved to {results_dir}")
print("\nResults Summary:")
print(f"{'Experiment':<20} {'Test RMSE':<15} {'Test MAE':<15} {'Best Epoch':<12}")
print("=" * 62)
for exp in experiments:
    print(f"{exp:<20} {results[exp]['test_rmse']:<15.4f} {results[exp]['test_mae']:<15.4f} {results[exp]['best_epoch']:<12}")
