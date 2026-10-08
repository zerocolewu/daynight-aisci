#!/usr/bin/env python
"""
Generate visualizations for the ablation study
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = '/home/home_public/aisci/AIscientist/src/2_02_PerioGT_Ablation'
os.makedirs(os.path.join(output_dir, 'diagnostics'), exist_ok=True)

# Load results
comprehensive_df = pd.read_csv(os.path.join(output_dir, 'comprehensive_results.csv'))
quick_df = pd.read_csv(os.path.join(output_dir, 'ablation_results.csv'))

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Comparison bar chart
fig, ax = plt.subplots(figsize=(10, 6))

configs = comprehensive_df['name'].tolist()
means = comprehensive_df['mean_rmse'].tolist()
stds = comprehensive_df['std_rmse'].tolist()

colors = ['#2ecc71', '#3498db', '#e74c3c']
bars = ax.bar(configs, means, yerr=stds, capsize=5, color=colors, alpha=0.8)

# Add baseline and reference lines
ax.axhline(y=36.41, color='orange', linestyle='--', linewidth=2, label='Previous Baseline (36.41)')
ax.axhline(y=30.8, color='red', linestyle='--', linewidth=2, label='Reference (30.8)')

ax.set_ylabel('Test RMSE (°C)', fontsize=12)
ax.set_xlabel('Configuration', fontsize=12)
ax.set_title('PerioGT Ablation Study Results', fontsize=14)
ax.legend(loc='upper right')

# Add value labels
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
            f'{mean:.2f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'diagnostics', 'ablation_comparison.png'), dpi=150)
plt.close()

# 2. Seed variation plot
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(configs))
width = 0.25

seeds = [42, 123, 456]
seed_colors = ['#3498db', '#2ecc71', '#e74c3c']

for i, (seed, color) in enumerate(zip(seeds, seed_colors)):
    values = comprehensive_df[f'seed{seed}'].tolist()
    ax.bar(x + i*width, values, width, label=f'Seed {seed}', color=color, alpha=0.8)

ax.set_ylabel('Test RMSE (°C)', fontsize=12)
ax.set_xlabel('Configuration', fontsize=12)
ax.set_title('RMSE Variation Across Random Seeds', fontsize=14)
ax.set_xticks(x + width)
ax.set_xticklabels(configs)
ax.legend()

# Add mean line for each config
for i, config in enumerate(configs):
    mean = comprehensive_df[comprehensive_df['name']==config]['mean_rmse'].values[0]
    ax.plot([i-0.1, i+0.6], [mean, mean], 'k--', linewidth=1.5, alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'diagnostics', 'seed_variation.png'), dpi=150)
plt.close()

# 3. Quick ablation comparison
fig, ax = plt.subplots(figsize=(12, 6))

quick_sorted = quick_df.sort_values('test_rmse')
configs_quick = quick_sorted['name'].tolist()
rmses_quick = quick_sorted['test_rmse'].tolist()

colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(configs_quick)))
bars = ax.barh(configs_quick, rmses_quick, color=colors)

# Add baseline and reference
ax.axvline(x=36.41, color='orange', linestyle='--', linewidth=2, label='Baseline (36.41)')
ax.axvline(x=30.8, color='red', linestyle='--', linewidth=2, label='Reference (30.8)')

ax.set_xlabel('Test RMSE (°C)', fontsize=12)
ax.set_title('Quick Ablation Results', fontsize=14)
ax.legend(loc='lower right')

# Add value labels
for bar, val in zip(bars, rmses_quick):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'diagnostics', 'quick_ablation.png'), dpi=150)
plt.close()

# 4. Gap analysis visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Gap breakdown
categories = ['Reference\n(30.8)', 'Best\n(37.0)', 'Baseline\n(36.41)', 'Chemprop\n(39.65)']
values = [30.8, 37.0, 36.41, 39.65]

colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c']
bars = ax.bar(categories, values, color=colors, alpha=0.8)

# Add gap annotations
ax.annotate('', xy=(1, 37.0), xytext=(0, 30.8),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(0.5, 33.5, 'Gap: 6.2', ha='center', fontsize=11, color='red', fontweight='bold')

ax.annotate('', xy=(2, 36.41), xytext=(1, 37.0),
            arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
ax.text(1.5, 36.7, 'Gap: 0.59', ha='center', fontsize=11, color='blue', fontweight='bold')

ax.set_ylabel('Test RMSE (°C)', fontsize=12)
ax.set_title('Performance Gap Analysis', fontsize=14)

# Add value labels
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'diagnostics', 'gap_analysis.png'), dpi=150)
plt.close()

# 5. Create summary figure
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: Main comparison
ax = axes[0, 0]
configs = comprehensive_df['name'].tolist()
means = comprehensive_df['mean_rmse'].tolist()
stds = comprehensive_df['std_rmse'].tolist()
colors = ['#2ecc71', '#3498db', '#e74c3c']
bars = ax.bar(configs, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
ax.axhline(y=36.41, color='orange', linestyle='--', linewidth=2, label='Baseline (36.41)')
ax.axhline(y=30.8, color='red', linestyle='--', linewidth=2, label='Reference (30.8)')
ax.set_ylabel('Test RMSE (°C)', fontsize=11)
ax.set_title('(A) Model Architecture Comparison', fontsize=12)
ax.legend(loc='upper right', fontsize=9)

# Panel 2: Seed variation
ax = axes[0, 1]
x = np.arange(len(configs))
width = 0.25
for i, seed in enumerate([42, 123, 456]):
    values = comprehensive_df[f'seed{seed}'].tolist()
    ax.bar(x + i*width, values, width, label=f'Seed {seed}', alpha=0.8)
ax.set_ylabel('Test RMSE (°C)', fontsize=11)
ax.set_title('(B) Random Seed Variation', fontsize=12)
ax.set_xticks(x + width)
ax.set_xticklabels(configs, fontsize=9)
ax.legend(fontsize=9)

# Panel 3: Quick ablation
ax = axes[1, 0]
quick_sorted = quick_df.sort_values('test_rmse')
y_pos = np.arange(len(quick_sorted))
ax.barh(y_pos, quick_sorted['test_rmse'], color=plt.cm.RdYlGn_r(np.linspace(0, 1, len(quick_sorted))))
ax.set_yticks(y_pos)
ax.set_yticklabels(quick_sorted['name'], fontsize=9)
ax.axvline(x=36.41, color='orange', linestyle='--', linewidth=2)
ax.set_xlabel('Test RMSE (°C)', fontsize=11)
ax.set_title('(C) Training Protocol Ablation', fontsize=12)

# Panel 4: Gap breakdown
ax = axes[1, 1]
categories = ['Reference', 'Best Ours', 'Baseline', 'Chemprop']
values = [30.8, 37.0, 36.41, 39.65]
colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c']
ax.bar(categories, values, color=colors, alpha=0.8)
ax.set_ylabel('Test RMSE (°C)', fontsize=11)
ax.set_title('(D) Performance Summary', fontsize=12)
for i, v in enumerate(values):
    ax.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'diagnostics', 'training_curves.png'), dpi=150)
plt.close()

print("Visualizations saved to diagnostics/")
print("Generated files:")
for f in os.listdir(os.path.join(output_dir, 'diagnostics')):
    print(f"  - {f}")
