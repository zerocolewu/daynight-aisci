#!/usr/bin/env python
"""
Diagnostic analysis for SPSE-MaxGAT experiment.
Compares Baseline, Treatment A, Treatment B on multiple axes.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
DIAG_DIR = os.path.join(WORK_DIR, 'diagnostics')
os.makedirs(DIAG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results(path):
    with open(path) as f:
        return json.load(f)

baseline = load_results(os.path.join(WORK_DIR, 'baseline', 'results', 'results.json'))
trt_a = load_results(os.path.join(WORK_DIR, 'treatment_a', 'results', 'results.json'))
trt_b = load_results(os.path.join(WORK_DIR, 'treatment_b', 'results', 'results.json'))

all_results = {
    'Baseline (GAT + Rich + 3-RU)': baseline,
    'Treatment A (+SPSE)': trt_a,
    'Treatment B (+SPSE + MaxAggr)': trt_b,
}

# ---------------------------------------------------------------------------
# 1. Ablation Table
# ---------------------------------------------------------------------------

print("=" * 80)
print("1. ABLATION TABLE")
print("=" * 80)

print(f"{'Model':<35} {'Test RMSE':>10} {'Test MAE':>10} {'Test R²':>8} {'Best Epoch':>10} {'Time (s)':>8}")
print("-" * 85)
for name, r in all_results.items():
    print(f"{name:<35} {r['test_rmse']:>8.1f} K {r['test_mae']:>8.1f} K {r['test_r2']:>8.4f} {r['best_epoch']:>8} {r['time_s']:>7.0f}")

# Save ablation table
with open(os.path.join(DIAG_DIR, 'ablation_table.txt'), 'w') as f:
    f.write("SPSE-MaxGAT Ablation Table\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"{'Model':<35} {'Test RMSE':>10} {'Test MAE':>10} {'Test R²':>8} {'Best Epoch':>10} {'Time (s)':>8}\n")
    f.write("-" * 85 + "\n")
    for name, r in all_results.items():
        f.write(f"{name:<35} {r['test_rmse']:>8.1f} K {r['test_mae']:>8.1f} K {r['test_r2']:>8.4f} {r['best_epoch']:>8} {r['time_s']:>7.0f}\n")
    f.write("\n")
    baseline_rmse = baseline['test_rmse']
    f.write(f"Delta vs Baseline:\n")
    for name, r in all_results.items():
        if 'Baseline' not in name:
            delta = r['test_rmse'] - baseline_rmse
            f.write(f"  {name}: {delta:+.1f} K\n")

# ---------------------------------------------------------------------------
# 2. SPSE Analysis
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("2. SPSE ANALYSIS")
print("=" * 80)

# Learned parameters
print("\nLearned SPSE parameters (Treatment A):")
print(f"  f_k(x) = alpha_k * ln(1 + x) + beta_k")
print(f"  {'k':>4} {'alpha':>10} {'beta':>10}")
print(f"  {'-'*26}")
for k in range(4):
    print(f"  {k+1:>4} {trt_a['spse_alpha'][k]:>10.4f} {trt_a['spse_beta'][k]:>10.4f}")

print("\nLearned SPSE parameters (Treatment B):")
print(f"  {'k':>4} {'alpha':>10} {'beta':>10}")
print(f"  {'-'*26}")
for k in range(4):
    print(f"  {k+1:>4} {trt_b['spse_alpha'][k]:>10.4f} {trt_b['spse_beta'][k]:>10.4f}")

with open(os.path.join(DIAG_DIR, 'spse_analysis.txt'), 'w') as f:
    f.write("SPSE Encoding Analysis\n")
    f.write("=" * 60 + "\n\n")
    f.write("Learned parameters (Treatment A):\n")
    f.write(f"  {'k':>4} {'alpha':>10} {'beta':>10}\n")
    for k in range(4):
        f.write(f"  {k+1:>4} {trt_a['spse_alpha'][k]:>10.4f} {trt_a['spse_beta'][k]:>10.4f}\n")
    f.write("\nLearned parameters (Treatment B):\n")
    f.write(f"  {'k':>4} {'alpha':>10} {'beta':>10}\n")
    for k in range(4):
        f.write(f"  {k+1:>4} {trt_b['spse_alpha'][k]:>10.4f} {trt_b['spse_beta'][k]:>10.4f}\n")

# Path count histogram - load from test graphs
print("\nLoading test graphs for SPSE path count analysis...")
sys.path.insert(0, WORK_DIR)
from data_utils import build_datasets

csv_path = os.path.join(os.path.dirname(WORK_DIR), 'alldataset', 'Tg.csv')
_, _, test_graphs, _ = build_datasets(csv_path, seed=71)

# Compute SPSE path counts
from spse_encoding import compute_spse_path_counts

all_path_counts = {1: [], 2: [], 3: [], 4: []}
for g in test_graphs:
    counts = compute_spse_path_counts(g.edge_index, g.x.shape[0], max_k=4)
    for k in range(4):
        all_path_counts[k+1].extend(counts[:, k].tolist())

# Path count statistics
print("\nPath count statistics per edge (test set):")
print(f"  {'k':>4} {'Mean':>8} {'Std':>8} {'Median':>8} {'Max':>8} {'%Zero':>8}")
for k in range(4):
    vals = np.array(all_path_counts[k+1])
    pct_zero = (vals == 0).mean() * 100
    print(f"  {k+1:>4} {vals.mean():>8.2f} {vals.std():>8.2f} {np.median(vals):>8.1f} {vals.max():>8.0f} {pct_zero:>7.1f}%")

# Plot histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, k in enumerate([1, 2, 3, 4]):
    ax = axes[idx // 2][idx % 2]
    vals = np.array(all_path_counts[k])
    ax.hist(vals, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel(f'k={k} path count')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Path Count Distribution (k={k}) — μ={vals.mean():.2f}, max={vals.max():.0f}')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
plt.suptitle('SPSE Path Count Distributions (Test Set)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(DIAG_DIR, 'spse_path_count_hist.png'), dpi=150)
plt.close()

with open(os.path.join(DIAG_DIR, 'spse_analysis.txt'), 'a') as f:
    f.write("\n\nPath count statistics (test set):\n")
    f.write(f"  {'k':>4} {'Mean':>8} {'Std':>8} {'Median':>8} {'Max':>8} {'%Zero':>8}\n")
    for k in range(4):
        vals = np.array(all_path_counts[k+1])
        pct_zero = (vals == 0).mean() * 100
        f.write(f"  {k+1:>4} {vals.mean():>8.2f} {vals.std():>8.2f} {np.median(vals):>8.1f} {vals.max():>8.0f} {pct_zero:>7.1f}%\n")

# ---------------------------------------------------------------------------
# 3. Error Analysis — predicted vs true scatter plots
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("3. ERROR ANALYSIS")
print("=" * 80)

# Per-model error statistics
for name, r in all_results.items():
    preds = np.array(r['test_preds'])
    targets = np.array(r['test_targets'])
    errors = preds - targets
    abs_errors = np.abs(errors)
    print(f"\n{name}:")
    print(f"  Mean Error: {errors.mean():.1f} K")
    print(f"  Std Error:  {errors.std():.1f} K")
    print(f"  Median |Error|: {np.median(abs_errors):.1f} K")

# Combined scatter plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
model_names = list(all_results.keys())
for idx, (name, r) in enumerate(all_results.items()):
    ax = axes[idx]
    preds = np.array(r['test_preds'])
    targets = np.array(r['test_targets'])
    ax.scatter(targets, preds, alpha=0.3, s=10)
    lim_min = min(targets.min(), preds.min())
    lim_max = max(targets.max(), preds.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.5)
    ax.set_xlabel('Actual Tg (K)')
    ax.set_ylabel('Predicted Tg (K)')
    ax.set_title(f'{name}\nRMSE={r["test_rmse"]:.1f}K, R²={r["test_r2"]:.4f}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
plt.suptitle('Predicted vs True Tg — Model Comparison', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(DIAG_DIR, 'scatter_comparison.png'), dpi=150)
plt.close()

# Error histogram comparison
fig, ax = plt.subplots(figsize=(10, 6))
for name, r in all_results.items():
    preds = np.array(r['test_preds'])
    targets = np.array(r['test_targets'])
    errors = preds - targets
    ax.hist(errors, bins=50, alpha=0.5, label=name, density=True)
ax.set_xlabel('Prediction Error (K)')
ax.set_ylabel('Density')
ax.set_title('Error Distribution Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIAG_DIR, 'error_histogram.png'), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. Improvement analysis
# ---------------------------------------------------------------------------

baseline_preds = np.array(baseline['test_preds'])
baseline_targets = np.array(baseline['test_targets'])
baseline_errors = np.abs(baseline_preds - baseline_targets)

trt_a_preds = np.array(trt_a['test_preds'])
trt_a_errors = np.abs(trt_a_preds - baseline_targets)

trt_b_preds = np.array(trt_b['test_preds'])
trt_b_errors = np.abs(trt_b_preds - baseline_targets)

# Which samples improved/worsened with Treatment A vs Baseline?
a_delta = trt_a_errors - baseline_errors
a_improved = a_delta < 0
print(f"\nTreatment A vs Baseline:")
print(f"  Improved: {a_improved.sum()}/{len(a_delta)} ({a_improved.mean()*100:.1f}%)")
print(f"  Mean delta: {a_delta.mean():.1f} K")

b_delta = trt_b_errors - baseline_errors
b_improved = b_delta < 0
print(f"\nTreatment B vs Baseline:")
print(f"  Improved: {b_improved.sum()}/{len(b_delta)} ({b_improved.mean()*100:.1f}%)")
print(f"  Mean delta: {b_delta.mean():.1f} K")

# Learning curves comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (name, r) in enumerate(all_results.items()):
    ax = axes[idx]
    train_hist = r['train_rmse_hist']
    val_hist = r['val_rmse_hist']
    epochs = range(1, len(train_hist) + 1)
    ax.plot(epochs, train_hist, 'b-', alpha=0.7, label='Train')
    ax.plot(epochs, val_hist, 'r-', alpha=0.7, label='Val')
    ax.axvline(x=r['best_epoch'], color='g', linestyle='--', alpha=0.5, label=f'Best@{r["best_epoch"]}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE (normalized)')
    ax.set_title(f'{name}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle('Learning Curves Comparison', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(DIAG_DIR, 'learning_curves_comparison.png'), dpi=150)
plt.close()

# Superimposed val curves
fig, ax = plt.subplots(figsize=(10, 6))
for name, r in all_results.items():
    val_hist = r['val_rmse_hist']
    epochs = range(1, len(val_hist) + 1)
    ax.plot(epochs, val_hist, alpha=0.8, label=name)
ax.set_xlabel('Epoch')
ax.set_ylabel('Val RMSE (normalized)')
ax.set_title('Validation RMSE — Model Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIAG_DIR, 'val_curves_comparison.png'), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. Max Aggregation diagnostic
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("4. MAX AGGREGATION ANALYSIS")
print("=" * 80)

# Since we can't easily hook into the trained models, we do a forward pass
# inspection on the test set using Treatment B model
import torch
from treatment_b.model import GATMaxAggr
from spse_encoding import SPSEEncoder, precompute_spse_for_graphs
from data_utils import NODE_DIM_RICH, EDGE_DIM_RICH, EDGE_DIM_SPSE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load Treatment B model
checkpoint = torch.load(os.path.join(WORK_DIR, 'treatment_b', 'results', 'best_model.pt'), map_location=device)
model_b = GATMaxAggr(node_dim=NODE_DIM_RICH, edge_dim=EDGE_DIM_RICH + EDGE_DIM_SPSE).to(device)
model_b.load_state_dict(checkpoint['model'])
model_b.eval()

# Also load baseline model for mean aggregation comparison
baseline_ckpt_path = os.path.join(WORK_DIR, 'baseline', 'results', 'best_model.pt')
if os.path.exists(baseline_ckpt_path):
    from baseline.model import GATBaseline
    baseline_ckpt = torch.load(baseline_ckpt_path, map_location=device)
    model_base = GATBaseline(node_dim=NODE_DIM_RICH, edge_dim=EDGE_DIM_RICH).to(device)
    model_base.load_state_dict(baseline_ckpt)
    model_base.eval()

# SPSE encoder
spse_encoder = SPSEEncoder(embed_dim=4).to(device)
spse_encoder.load_state_dict(checkpoint['spse_encoder'])

# Precompute SPSE for test graphs
precompute_spse_for_graphs(test_graphs, encoder=None, max_k=4)

# Select a batch of test graphs for analysis
from torch_geometric.loader import DataLoader
import numpy as np

test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)
batch = next(iter(test_loader))
batch = batch.to(device)

edge_attr = batch.edge_attr
spse_raw = batch.spse_raw.to(device)
spse_encoded = spse_encoder(spse_raw)
edge_attr_full = torch.cat([edge_attr, spse_encoded], dim=1)

# Hook-based analysis for max aggregation
# We need to capture intermediate outputs
activations = {}

def make_hook(name, layer):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

hooks = []
for i in range(1, 5):
    hook = getattr(model_b, f'conv{i}').register_forward_hook(make_hook(f'conv{i}', None))
    hooks.append(hook)

# For baseline, also register hooks
base_acts = {}
if os.path.exists(baseline_ckpt_path):
    base_hooks = []
    for i in range(1, 5):
        hook = getattr(model_base, f'conv{i}').register_forward_hook(
            lambda module, input, output, idx=i: base_acts.update({f'conv{idx}': output.detach()}),
        )
        base_hooks.append(hook)

with torch.no_grad():
    _ = model_b(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr_full)

for h in hooks:
    h.remove()

if os.path.exists(baseline_ckpt_path):
    with torch.no_grad():
        _ = model_base(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)

# Analyze activation statistics
print("\nActivation statistics (Treatment B - Max Aggregation):")
for name in ['conv1', 'conv2', 'conv3', 'conv4']:
    if name in activations:
        act = activations[name]  # [num_nodes, 256]
        print(f"  {name}: mean={act.mean():.4f}, std={act.std():.4f}, "
              f"min={act.min():.4f}, max={act.max():.4f}")

if os.path.exists(baseline_ckpt_path) and base_acts:
    print("\nActivation statistics (Baseline - Mean Aggregation):")
    for name in ['conv1', 'conv2', 'conv3', 'conv4']:
        if name in base_acts:
            act = base_acts[name]
            print(f"  {name}: mean={act.mean():.4f}, std={act.std():.4f}, "
                  f"min={act.min():.4f}, max={act.max():.4f}")

    # Compare variance
    print("\nVariance ratio (Max / Mean):")
    for name in ['conv1', 'conv2', 'conv3', 'conv4']:
        if name in activations and name in base_acts:
            var_max = activations[name].var().item()
            var_mean = base_acts[name].var().item()
            print(f"  {name}: max_var={var_max:.4f}, mean_var={var_mean:.4f}, ratio={var_max/var_mean:.3f}")

with open(os.path.join(DIAG_DIR, 'max_aggregation_analysis.txt'), 'w') as f:
    f.write("Max Aggregation Analysis\n")
    f.write("=" * 60 + "\n\n")
    f.write("Activation statistics (Treatment B - Max Aggregation):\n")
    for name in ['conv1', 'conv2', 'conv3', 'conv4']:
        if name in activations:
            act = activations[name]
            f.write(f"  {name}: mean={act.mean():.4f}, std={act.std():.4f}, "
                    f"min={act.min():.4f}, max={act.max():.4f}\n")
    if os.path.exists(baseline_ckpt_path) and base_acts:
        f.write("\nActivation statistics (Baseline - Mean Aggregation):\n")
        for name in ['conv1', 'conv2', 'conv3', 'conv4']:
            if name in base_acts:
                act = base_acts[name]
                f.write(f"  {name}: mean={act.mean():.4f}, std={act.std():.4f}, "
                        f"min={act.min():.4f}, max={act.max():.4f}\n")
        f.write("\nVariance ratio (Max / Mean):\n")
        for name in ['conv1', 'conv2', 'conv3', 'conv4']:
            if name in activations and name in base_acts:
                var_max = activations[name].var().item()
                var_mean = base_acts[name].var().item()
                f.write(f"  {name}: max_var={var_max:.4f}, mean_var={var_mean:.4f}, ratio={var_max/var_mean:.3f}\n")

print("\n" + "=" * 80)
print(f"Diagnostics saved to {DIAG_DIR}/")
print("=" * 80)
