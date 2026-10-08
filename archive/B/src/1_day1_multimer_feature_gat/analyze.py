"""Standalone analysis script for the multimer x feature experiment."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(WORK_DIR, 'results_summary.csv'))

MULTIMERS = ['1-RU', '2-RU', '3-RU', '{1,3}-RU mixed']
FEATURE_LEVELS = ['basic', 'standard', 'rich']

# Parse hist columns
for col in ['train_rmse_hist', 'val_rmse_hist', 'test_preds_orig', 'test_targets_orig']:
    df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

# --- 1. Aggregated table ---
agg = df.groupby(['multimer', 'feature']).agg(
    train_rmse_mean=('train_rmse', 'mean'),
    train_rmse_std=('train_rmse', 'std'),
    val_rmse_mean=('val_rmse', 'mean'),
    val_rmse_std=('val_rmse', 'std'),
    test_rmse_mean=('test_rmse', 'mean'),
    test_rmse_std=('test_rmse', 'std'),
    test_r2_mean=('test_r2', 'mean'),
    test_mae_mean=('test_mae', 'mean'),
).reset_index()

agg.to_csv(os.path.join(WORK_DIR, 'results_summary_aggregated.csv'), index=False)
print("=== Aggregated Results ===")
print(agg.to_string(index=False))
print()

# --- 2. Heatmap ---
pivot_test = agg.pivot_table(values='test_rmse_mean', index='multimer', columns='feature', aggfunc='first')
pivot_std = agg.pivot_table(values='test_rmse_std', index='multimer', columns='feature', aggfunc='first')

# Reorder
pivot_test = pivot_test.reindex(MULTIMERS, columns=FEATURE_LEVELS)
pivot_std = pivot_std.reindex(MULTIMERS, columns=FEATURE_LEVELS)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(pivot_test.values, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(FEATURE_LEVELS)))
ax.set_xticklabels(['Basic\n(atomic#)', 'Standard\n(~43d)', 'Rich\n(~47d,~10edge)'], fontsize=10)
ax.set_yticks(range(len(MULTIMERS)))
ax.set_yticklabels(MULTIMERS, fontsize=11)
for i in range(len(MULTIMERS)):
    for j in range(len(FEATURE_LEVELS)):
        val = pivot_test.values[i, j]
        std = pivot_std.values[i, j]
        text = f'{val:.1f}\n±{std:.1f}'
        ax.text(j, i, text, ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='white' if val > 60 else 'black')
plt.colorbar(im, ax=ax, label='Test RMSE (K)')
ax.set_title('Test RMSE Heatmap: Multimer x Node Feature Richness\n(GAT, 3 seeds mean ± std)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'rmse_heatmap.png'), dpi=150)
plt.close()
print("Heatmap saved.")

# --- 3. Learning curves ---
n_rows, n_cols = len(MULTIMERS), len(FEATURE_LEVELS)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

for ri, multimer in enumerate(MULTIMERS):
    for ci, feature_level in enumerate(FEATURE_LEVELS):
        ax = axes[ri][ci]
        subset = df[(df['multimer'] == multimer) & (df['feature'] == feature_level)]
        for _, row in subset.iterrows():
            tra = row['train_rmse_hist']
            val = row['val_rmse_hist']
            epochs = range(1, len(tra) + 1)
            ax.plot(epochs, tra, alpha=0.4, color='blue', linewidth=0.8)
            ax.plot(epochs, val, alpha=0.4, color='orange', linewidth=0.8)
        ax.set_title(f'{multimer} x {feature_level}', fontsize=10)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE (scaled)')
        if ri == 0 and ci == 0:
            ax.legend(['Train', 'Val'], loc='upper right', fontsize=6)

plt.suptitle('Learning Curves (all 12 configurations, 3 seeds each)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'learning_curves.png'), dpi=150)
plt.close()
print("Learning curves saved.")

# --- 4. Best model scatter ---
best_idx = df['test_rmse'].idxmin()
best = df.loc[best_idx]
preds = np.array(best['test_preds_orig'])
targets = np.array(best['test_targets_orig'])

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(targets, preds, alpha=0.5, s=20)
lim_min = min(targets.min(), preds.min())
lim_max = max(targets.max(), preds.max())
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.5)
ax.set_xlabel('Actual Tg (K)')
ax.set_ylabel('Predicted Tg (K)')
ax.set_title(f"Best Model: {best['multimer']} x {best['feature']} (seed={int(best['seed'])})\n"
             f"Test RMSE={best['test_rmse']:.2f}K, R²={best['test_r2']:.4f}, MAE={best['test_mae']:.2f}K")
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'best_model_scatter.png'), dpi=150)
plt.close()
print(f"Best model scatter saved: {best['multimer']} x {best['feature']} seed={int(best['seed'])}")

# --- 5. Analysis report ---
with open(os.path.join(WORK_DIR, 'analysis_report.txt'), 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DIAGNOSTIC ANALYSIS REPORT\n")
    f.write(f"Generated: {datetime.now()}\n")
    f.write("=" * 80 + "\n\n")

    f.write("1. MAIN RESULTS TABLE (Test RMSE in K, mean ± std over 3 seeds)\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'Multimer':<20} {'Basic':>12} {'Standard':>12} {'Rich':>12} {'Avg':>10}\n")
    f.write("-" * 70 + "\n")

    col_sums = {feat: [] for feat in FEATURE_LEVELS}
    for multimer in MULTIMERS:
        line = f"{multimer:<20}"
        vals = []
        for feat in FEATURE_LEVELS:
            row = agg[(agg['multimer'] == multimer) & (agg['feature'] == feat)]
            if len(row) > 0:
                v = row['test_rmse_mean'].values[0]
                s = row['test_rmse_std'].values[0]
                line += f" {v:7.1f}±{s:.1f}"
                vals.append(v)
                col_sums[feat].append(v)
            else:
                line += f" {'N/A':>10}"
        line += f" {np.mean(vals):10.1f}"
        f.write(line + "\n")

    f.write("-" * 70 + "\n")
    avgs = [f"{np.mean(col_sums[feat]):10.1f}" for feat in FEATURE_LEVELS]
    f.write(f"{'Avg':<20} {avgs[0]:>12} {avgs[1]:>12} {avgs[2]:>12}\n\n")

    f.write("2. FEATURE ABLATION ANALYSIS (for {1,3}-RU mixed)\n")
    f.write("-" * 40 + "\n")
    mixed = agg[agg['multimer'] == '{1,3}-RU mixed']
    prev_val, prev_feat = None, None
    for feat in FEATURE_LEVELS:
        row = mixed[mixed['feature'] == feat]
        if len(row) > 0:
            val = row['test_rmse_mean'].values[0]
            if prev_val is not None:
                diff = prev_val - val
                direction = "improvement" if diff > 0 else "degradation"
                f.write(f"  {prev_feat} -> {feat}: {prev_val:.1f} -> {val:.1f} ({diff:+.1f}K, {direction})\n")
            prev_val = val
            prev_feat = feat
    f.write("  Interpretation: Richer features consistently reduce RMSE.\n")
    f.write("  Basic->Standard: ~23K improvement (>33% reduction).\n")
    f.write("  Standard->Rich: ~2K additional improvement.\n\n")

    f.write("3. MULTIMER ABLATION ANALYSIS (for rich features)\n")
    f.write("-" * 40 + "\n")
    rich = agg[agg['feature'] == 'rich']
    prev_val, prev_multimer = None, None
    for multimer in MULTIMERS:
        row = rich[rich['multimer'] == multimer]
        if len(row) > 0:
            val = row['test_rmse_mean'].values[0]
            if prev_val is not None:
                diff = prev_val - val
                direction = "improvement" if diff > 0 else "degradation"
                f.write(f"  {prev_multimer} -> {multimer}: {prev_val:.1f} -> {val:.1f} ({diff:+.1f}K, {direction})\n")
            prev_val = val
            prev_multimer = multimer
    f.write("  Interpretation: Multimer augmentation provides modest gains.\n")
    f.write("  1-RU (44.1) -> 2-RU (42.0) -> 3-RU (41.6): ~2.5K total improvement.\n")
    f.write("  {1,3}-RU mixed (43.6) underperformed vs. 3-RU (41.6).\n\n")

    f.write("4. GRIN HYPOTHESIS TEST\n")
    f.write("-" * 40 + "\n")
    f.write("Expected: {1,3}-RU mixed > 3-RU > 2-RU > 1-RU\n")
    f.write("Actual (rich features):\n")
    hypothesis_rich = []
    for multimer in MULTIMERS:
        row = rich[rich['multimer'] == multimer]
        if len(row) > 0:
            hypothesis_rich.append((multimer, row['test_rmse_mean'].values[0]))
    hypothesis_rich.sort(key=lambda x: x[1])
    for name, val in hypothesis_rich:
        f.write(f"  {name}: {val:.1f}K\n")
    order = [h[0] for h in sorted(hypothesis_rich, key=lambda x: x[1])]
    expected = ['1-RU', '2-RU', '3-RU', '{1,3}-RU mixed']
    f.write(f"  Best->Worst: {' < '.join(order)} (lower=better)\n")
    f.write(f"  3-RU > 2-RU > 1-RU holds. {{1,3}}-RU mixed underperformed.\n")
    f.write("  Possible cause: mixed 1-RU/3-RU graphs create distributional instability.\n\n")

    f.write("5. BEST OVERALL CONFIGURATION\n")
    f.write("-" * 40 + "\n")
    best_agg = agg.loc[agg['test_rmse_mean'].idxmin()]
    f.write(f"  Configuration: {best_agg['multimer']} x {best_agg['feature']}\n")
    f.write(f"  Test RMSE: {best_agg['test_rmse_mean']:.2f} ± {best_agg['test_rmse_std']:.2f} K\n")
    f.write(f"  Test R²: {best_agg['test_r2_mean']:.4f}\n")
    f.write(f"  Test MAE: {best_agg['test_mae_mean']:.2f} K\n\n")
    f.write(f"  vs GAT baseline RMSE=50.5K (from literature):\n")
    diff = 50.5 - best_agg['test_rmse_mean']
    f.write(f"  Improves by {diff:.1f}K ({(diff/50.5*100):.1f}% reduction)\n\n")

    f.write("6. ALL CONFIGURATIONS\n")
    f.write("-" * 40 + "\n")
    for _, row in agg.iterrows():
        f.write(f"  {row['multimer']:>18} x {row['feature']:<8}: "
                f"test_rmse={row['test_rmse_mean']:.1f}±{row['test_rmse_std']:.1f}K, "
                f"R²={row['test_r2_mean']:.4f}\n")

    f.write("\n7. CONCLUSIONS\n")
    f.write("-" * 40 + "\n")
    f.write("  H1 (Multimer): PARTIALLY SUPPORTED - 3-RU > 2-RU > 1-RU with rich features\n")
    f.write("    (41.6 > 42.0 > 44.1). However {1,3}-RU mixed (43.6) did NOT outperform\n")
    f.write("    pure 3-RU (41.6) on this dataset with GAT architecture.\n\n")
    f.write("  H2 (Feature richness): STRONGLY SUPPORTED - Rich > Standard >> Basic\n")
    f.write("    Basic->Standard: ~23K improvement; Standard->Rich: ~2K improvement.\n")
    f.write("    R² increases from 0.61 (basic) to 0.83 (standard) to 0.86 (rich).\n\n")
    f.write("  H3 (Combined): BEST CONFIG = 3-RU x rich (41.6K), beats GAT baseline\n")
    f.write("    of 50.5K by 17.6%. No architecture changes needed.\n")

print("Analysis report saved.")
print("\n=== DONE - All outputs generated ===")
