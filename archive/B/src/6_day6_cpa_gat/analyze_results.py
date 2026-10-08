"""
Diagnostic analysis for CPA-GAT experiments.
- Cardinality correlation analysis
- Per-sample predictions comparison
- Gate saturation histogram
- Training curves
"""
import os
import sys
import json
import pickle
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_gat import BaselineGAT
from cpa_gat import CPAGAT
from cpa_last_gat import CPALastLayerGAT
from degree_baseline import DegreeBaselineGAT

DATA_PATH = "/home/home_public/aisci/AIscientist/src/4_day4_selfgnn_polymer/data/tg_cache/tg_graphs_3ru_42dim.pkl"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['train_graphs'], data['val_graphs'], data['test_graphs'], data['scaler']


def compute_neighbor_counts(graphs):
    """Compute number of neighbors (including self-loop) per node for each graph."""
    neighbor_counts = []
    for g in graphs:
        n = g.num_nodes
        src, dst = g.edge_index
        deg = torch.zeros(n)
        deg.scatter_add_(0, dst, torch.ones(dst.size(0)))
        deg.scatter_add_(0, torch.arange(n), torch.ones(n))  # self-loop
        neighbor_counts.append(deg.numpy())
    return neighbor_counts


def analyze_cpa_cardinality():
    """Analyze CPA gate and sum channel correlation with neighbor counts."""
    print("\n" + "=" * 60)
    print("CPA Cardinality Correlation Analysis")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_graphs, scaler = load_data()

    # Load best CPA-GAT model (seed 42)
    model = CPAGAT(in_dim=42).to(device)
    state = torch.load(os.path.join(RESULTS_DIR, "cpa_gat_seed42.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Compute neighbor counts
    neighbor_counts = compute_neighbor_counts(test_graphs)

    # Run forward pass and collect gate values
    all_gates = {0: [], 1: [], 2: [], 3: []}
    all_sums = {0: [], 1: [], 2: [], 3: []}
    all_neighbor_counts = []

    loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    for idx, batch in enumerate(loader):
        batch = batch.to(device)
        x, edge_index = batch.x, batch.edge_index
        n = x.size(0)

        x = model.input_proj(x)

        for layer_idx, conv in enumerate(model.convs):
            # Compute CPA components for this single graph
            self_idx = torch.arange(n, device=device)
            all_src = torch.cat([edge_index[0], self_idx])
            all_dst = torch.cat([edge_index[1], self_idx])

            # Per-node gate and sum norms (average across heads for correlation with neighbor count)
            gate_norms_this_layer = torch.zeros(n, device=device)
            sum_norms_this_layer = torch.zeros(n, device=device)

            for h in range(conv.heads):
                # Value projection
                v = conv.gat_heads[h].lin_r(x)

                # CPA sum (with self-loops)
                v_all = v[all_src]
                s = torch.zeros(n, conv.head_dim, device=device)
                s.scatter_add_(0, all_dst.unsqueeze(-1).expand(-1, conv.head_dim), v_all)

                # Gate
                g = torch.sigmoid(F.linear(x, conv.W_g[h], conv.b_g[h]))

                # Accumulate L2 norms (average across heads)
                gate_norms_this_layer += g.norm(dim=1).detach() / conv.heads
                sum_norms_this_layer += s.norm(dim=1).detach() / conv.heads

            all_gates[layer_idx].extend(gate_norms_this_layer.cpu().tolist())
            all_sums[layer_idx].extend(sum_norms_this_layer.cpu().tolist())

            # Forward pass to next layer
            x = conv(x, edge_index)
            x = F.elu(x)

        all_neighbor_counts.extend(neighbor_counts[idx].tolist())

    # Compute Pearson correlations per layer
    all_gates_flat = {l: np.array(all_gates[l]) for l in range(4)}
    all_sums_flat = {l: np.array(all_sums[l]) for l in range(4)}
    all_nc = np.array(all_neighbor_counts)

    print(f"\nTotal nodes analyzed: {len(all_nc)}")
    print(f"\nNeighbor count stats: min={all_nc.min():.0f}, max={all_nc.max():.0f}, "
          f"mean={all_nc.mean():.1f}, std={all_nc.std():.1f}")

    print("\nPearson correlations: |S(i)| vs gate L2 norm")
    for layer_idx in range(4):
        r, p = pearsonr(all_nc, all_gates_flat[layer_idx])
        print(f"  Layer {layer_idx+1}: r={r:.4f}, p={p:.4e}")

    print("\nPearson correlations: |S(i)| vs CPA sum L2 norm")
    for layer_idx in range(4):
        r, p = pearsonr(all_nc, all_sums_flat[layer_idx])
        print(f"  Layer {layer_idx+1}: r={r:.4f}, p={p:.4e}")

    # Plot gate saturation histogram
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for layer_idx in range(4):
        ax = axes[layer_idx // 2, layer_idx % 2]
        # Gate values (element-wise, not L2 norm)
        # We need to recompute individual gate values
        ax.hist(np.clip(all_gates_flat[layer_idx], 0, np.percentile(all_gates_flat[layer_idx], 99)),
                bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'Layer {layer_idx+1} Gate L2 Norm Distribution')
        ax.set_xlabel('Gate L2 Norm')
        ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gate_saturation.png"), dpi=150)
    plt.close()

    # Gate vs neighbor count scatter
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for layer_idx in range(4):
        ax = axes[layer_idx // 2, layer_idx % 2]
        # Subsample for clarity
        mask = np.random.choice(len(all_nc), min(5000, len(all_nc)), replace=False)
        ax.scatter(all_nc[mask], all_gates_flat[layer_idx][mask], alpha=0.3, s=3,
                  c=all_gates_flat[layer_idx][mask], cmap='viridis')
        r, _ = pearsonr(all_nc, all_gates_flat[layer_idx])
        ax.set_title(f'Layer {layer_idx+1}: r={r:.4f}')
        ax.set_xlabel('Neighbor Count |S(i)|')
        ax.set_ylabel('Gate L2 Norm')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cardinality_correlation.png"), dpi=150)
    plt.close()

    # Element-wise gate value histogram (sigmoid outputs directly)
    print("\nComputing element-wise gate saturation...")
    # Re-run with all test data to get raw gate values
    raw_gate_vals = {0: [], 1: [], 2: [], 3: []}
    loader2 = DataLoader(test_graphs, batch_size=32, shuffle=False)
    with torch.no_grad():
        for batch in loader2:
            batch = batch.to(device)
            x, edge_index = batch.x, batch.edge_index
            n = x.size(0)
            x_proj = model.input_proj(x)

            for layer_idx, conv in enumerate(model.convs):
                for h in range(conv.heads):
                    g = torch.sigmoid(F.linear(x_proj, conv.W_g[h], conv.b_g[h]))
                    raw_gate_vals[layer_idx].extend(g.cpu().flatten().tolist())
                x_proj = conv(x_proj, edge_index)
                x_proj = F.elu(x_proj)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for layer_idx in range(4):
        ax = axes[layer_idx // 2, layer_idx % 2]
        vals = np.array(raw_gate_vals[layer_idx])
        ax.hist(vals, bins=50, alpha=0.7, edgecolor='black')
        mean_val = vals.mean()
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        ax.set_title(f'Layer {layer_idx+1} Element-wise Gate Values')
        ax.set_xlabel('Sigmoid Gate Value')
        ax.set_ylabel('Count')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gate_saturation_elementwise.png"), dpi=150)
    plt.close()

    print(f"Element-wise gate stats (per layer):")
    for l in range(4):
        vals = np.array(raw_gate_vals[l])
        print(f"  Layer {l+1}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"min={vals.min():.4f}, max={vals.max():.4f}")
        frac_low = (vals < 0.1).mean()
        frac_high = (vals > 0.9).mean()
        frac_mid = 1 - frac_low - frac_high
        print(f"    <0.1: {frac_low:.3f}, 0.1-0.9: {frac_mid:.3f}, >0.9: {frac_high:.3f}")

    return all_nc, all_gates_flat


def analyze_per_sample():
    """Compare CPA-GAT vs Baseline predictions, color by chain length."""
    print("\n" + "=" * 60)
    print("Per-Sample Analysis: CPA-GAT vs Baseline")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_graphs, scaler = load_data()

    # Load models
    cpa = CPAGAT(in_dim=42).to(device)
    cpa.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "cpa_gat_seed42.pt"), map_location=device))
    cpa.eval()

    baseline = BaselineGAT(in_dim=42).to(device)
    baseline.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "baseline_gat_seed42.pt"), map_location=device))
    baseline.eval()

    loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    cpa_preds, baseline_preds, targets = [], [], []
    num_nodes_list = []

    for batch in loader:
        batch = batch.to(device)
        # Get per-graph node counts from batch
        nn_per_graph = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
        num_nodes_list.extend(nn_per_graph)

        with torch.no_grad():
            p_cpa = cpa(batch)
            p_bl = baseline(batch)

        # Denormalize
        p_cpa_denorm = p_cpa.cpu() * scaler.scale_[0] + scaler.mean_[0]
        p_bl_denorm = p_bl.cpu() * scaler.scale_[0] + scaler.mean_[0]
        t_denorm = batch.y.cpu().view(-1) * scaler.scale_[0] + scaler.mean_[0]

        cpa_preds.append(p_cpa_denorm)
        baseline_preds.append(p_bl_denorm)
        targets.append(t_denorm)

    cpa_preds = torch.cat(cpa_preds).numpy()
    baseline_preds = torch.cat(baseline_preds).numpy()
    targets = torch.cat(targets).numpy()
    num_nodes = np.array(num_nodes_list)

    # Errors
    cpa_errors = np.abs(cpa_preds - targets)
    baseline_errors = np.abs(baseline_preds - targets)

    print(f"\nCPA-GAT MAE: {cpa_errors.mean():.2f} K")
    print(f"Baseline MAE: {baseline_errors.mean():.2f} K")
    print(f"Improvement: {(baseline_errors.mean() - cpa_errors.mean()):.2f} K")

    # Break down by chain length quartiles
    q25, q50, q75 = np.percentile(num_nodes, [25, 50, 75])
    print(f"\nChain length quartiles: Q1={q25:.0f}, Q2={q50:.0f}, Q3={q75:.0f}")

    for label, mask in [("Short (<Q1)", num_nodes < q25),
                        ("Q1-Q2", (num_nodes >= q25) & (num_nodes < q50)),
                        ("Q2-Q3", (num_nodes >= q50) & (num_nodes < q75)),
                        ("Long (>Q3)", num_nodes >= q75)]:
        if mask.sum() > 0:
            print(f"  {label:15s}: n={mask.sum():4d}, "
                  f"CPA MAE={cpa_errors[mask].mean():.2f}, "
                  f"Baseline MAE={baseline_errors[mask].mean():.2f}, "
                  f"Delta={baseline_errors[mask].mean() - cpa_errors[mask].mean():.2f}")

    # Scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # CPA-GAT vs Target
    axes[0].scatter(targets, cpa_preds, alpha=0.5, s=15, c=num_nodes, cmap='plasma')
    axes[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    axes[0].set_xlabel('True Tg (K)')
    axes[0].set_ylabel('CPA-GAT Predicted Tg (K)')
    axes[0].set_title('CPA-GAT Predictions\n(colored by chain length)')
    axes[0].set_xlim(100, 800)
    axes[0].set_ylim(100, 800)

    # Baseline vs Target
    axes[1].scatter(targets, baseline_preds, alpha=0.5, s=15, c=num_nodes, cmap='plasma')
    axes[1].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    axes[1].set_xlabel('True Tg (K)')
    axes[1].set_ylabel('Baseline Predicted Tg (K)')
    axes[1].set_title('Baseline Predictions\n(colored by chain length)')
    axes[1].set_xlim(100, 800)
    axes[1].set_ylim(100, 800)

    # CPA vs Baseline predictions
    c = num_nodes
    error_diff = baseline_errors - cpa_errors
    sc = axes[2].scatter(cpa_preds, baseline_preds, alpha=0.5, s=15, c=error_diff,
                         cmap='RdBu_r', vmin=-abs(error_diff).max(), vmax=abs(error_diff).max())
    axes[2].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    axes[2].set_xlabel('CPA-GAT Prediction (K)')
    axes[2].set_ylabel('Baseline Prediction (K)')
    axes[2].set_title('CPA vs Baseline Predictions\n(red=CPA better, blue=baseline better)')
    axes[2].set_xlim(100, 800)
    axes[2].set_ylim(100, 800)
    plt.colorbar(sc, ax=axes[2], label='|Error Bl| - |Error CPA| (K)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "per_sample_comparison.png"), dpi=150)
    plt.close()

    # Error by chain length
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # CPA error vs chain length
    axes[0].scatter(num_nodes, cpa_errors, alpha=0.4, s=10, c=cpa_errors, cmap='hot')
    axes[0].set_xlabel('Chain Length (nodes)')
    axes[0].set_ylabel('Absolute Error (K)')
    axes[0].set_title('CPA-GAT Error vs Chain Length')

    # Baseline error vs chain length
    axes[1].scatter(num_nodes, baseline_errors, alpha=0.4, s=10, c=baseline_errors, cmap='hot')
    axes[1].set_xlabel('Chain Length (nodes)')
    axes[1].set_ylabel('Absolute Error (K)')
    axes[1].set_title('Baseline Error vs Chain Length')

    # Error difference vs chain length
    axes[2].scatter(num_nodes, error_diff, alpha=0.4, s=10, c=error_diff,
                    cmap='RdBu_r', vmin=-abs(error_diff).max(), vmax=abs(error_diff).max())
    axes[2].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Chain Length (nodes)')
    axes[2].set_ylabel('|Bl Error| - |CPA Error| (K)')
    axes[2].set_title('Error Improvement vs Chain Length\n(positive = CPA better)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "error_vs_chain_length.png"), dpi=150)
    plt.close()

    return cpa_preds, baseline_preds, targets, num_nodes


def plot_training_curves():
    """Overlay training curves for baseline vs CPA-GAT."""
    print("\n" + "=" * 60)
    print("Training Curves")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for seed_idx, seed in enumerate([42, 123, 456]):
        # Load histories
        for model_name, color, ls in [("baseline_gat", "blue", "-"), ("cpa_gat", "red", "-")]:
            history_file = os.path.join(RESULTS_DIR, f"{model_name}_seed{seed}_history.json")
            with open(history_file) as f:
                history = json.load(f)
            epochs = [h['epoch'] for h in history]
            val_rmse = [h['val_rmse'] for h in history]
            alpha = 0.3 if seed_idx > 0 else 1.0
            axes[0].plot(epochs, val_rmse, color=color, linestyle=ls, alpha=alpha,
                        label=f"{model_name} (seed {seed})" if seed_idx == 0 else None)

            train_loss = [h['train_loss'] for h in history]
            axes[1].plot(epochs, train_loss, color=color, linestyle=ls, alpha=alpha,
                        label=f"{model_name} (seed {seed})" if seed_idx == 0 else None)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation RMSE (K)')
    axes[0].set_title('Validation RMSE Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Training Loss (MSE)')
    axes[1].set_title('Training Loss Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close()

    print("Training curves saved.")


def create_summary_table():
    """Create a results summary table."""
    with open(os.path.join(RESULTS_DIR, "summary.json")) as f:
        summary = json.load(f)

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20s} {'Test RMSE (K)':<20s} {'Best Seed':<12s} {'vs Baseline':<15s}")
    print("-" * 80)

    baseline_rmse = summary["baseline_gat"]["test_rmse_mean"]
    baseline_std = summary["baseline_gat"]["test_rmse_std"]

    for model_name, data in summary.items():
        mean = data["test_rmse_mean"]
        std = data["test_rmse_std"]
        best = min(s['test_rmse'] for s in data['seeds'])
        delta = baseline_rmse - mean
        delta_str = f"-{delta:.2f} K" if delta > 0 else f"+{-delta:.2f} K"
        print(f"{model_name:<20s} {mean:.2f} +/- {std:.2f}    {best:.2f}       {delta_str}")


if __name__ == "__main__":
    print("CPA-GAT Diagnostic Analysis")
    print("=" * 60)

    # Cardinality correlation
    nc, gates = analyze_cpa_cardinality()

    # Per-sample analysis
    cpa_preds, bl_preds, targets, num_nodes = analyze_per_sample()

    # Training curves
    plot_training_curves()

    # Summary
    create_summary_table()

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("Analysis complete!")
