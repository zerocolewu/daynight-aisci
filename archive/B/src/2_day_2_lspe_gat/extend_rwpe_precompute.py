"""
Extended RWPE precomputation for k=24 and k=32.

Reuses the graph datasets from the existing rwpe_data.pkl and
extends computation to higher k-steps. Also saves k=16 separately.
Computes and saves:
  - rwpe_k16.pkl
  - rwpe_k24.pkl
  - rwpe_k32.pkl

Includes cross-k diagnostic comparisons.
"""
import os
import sys
import pickle
import time
import warnings

import numpy as np
import torch
from torch_geometric.utils import to_dense_adj

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '1_day1_multimer_feature_gat'))
from data_utils import build_datasets

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'alldataset', 'Tg.csv')
RWPE_ORIGINAL_PATH = os.path.join(OUTPUT_DIR, 'rwpe_data.pkl')

TARGET_K_VALUES = [16, 24, 32]


def compute_rwpe_for_graph(edge_index, num_nodes, k_steps):
    """Compute k-step random walk return probabilities for a single graph."""
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).numpy()
    deg = adj.sum(axis=1)
    deg[deg == 0] = 1
    P = adj / deg[:, np.newaxis]

    rwpe = np.zeros((num_nodes, k_steps), dtype=np.float32)
    P_pow = P.copy()
    for step in range(k_steps):
        diag_vals = np.diag(P_pow)
        diag_vals = np.nan_to_num(diag_vals, nan=0.0)
        rwpe[:, step] = diag_vals
        P_pow = P_pow @ P
        P_pow = np.nan_to_num(P_pow, nan=0.0)

    return rwpe


def compute_rwpe_variance_stats(rwpe_dict, k=None):
    """Compute per-graph variance of RWPE across nodes."""
    variances = []
    for idx, rwpe in rwpe_dict.items():
        if isinstance(rwpe, torch.Tensor):
            rwpe_np = rwpe.numpy()
        else:
            rwpe_np = rwpe
        node_vars = np.var(rwpe_np, axis=0)
        avg_var = np.mean(node_vars)
        variances.append(avg_var)
    return np.array(variances)


def convert_k16_and_compute_higher_k(all_graphs, k16_dict, k_targets):
    """For k=16: just save rwpe_k16.pkl from existing data.
    For k>16: recompute only higher-k values, reusing k=16 as starting point.

    Optimization: instead of recomputing P^k from scratch for each k,
    compute P^16 once (from existing data), then continue to k=24 and k=32.
    """
    results = {}

    # First, save k=16 from existing data
    if 16 in k_targets:
        print(f"  Saving k=16 from existing data ({len(k16_dict)} graphs)...")
        rwpe_tensors = {i: torch.tensor(v, dtype=torch.float) for i, v in k16_dict.items()}
        variances_16 = compute_rwpe_variance_stats(rwpe_tensors, 16)
        results[16] = {'rwpe': rwpe_tensors, 'variances': variances_16}
        k_targets = [k for k in k_targets if k > 16]

    # Compute higher k values
    if k_targets:
        max_k = max(k_targets)
        print(f"\n  Computing RWPE for k variants up to k={max_k} (from scratch)...")
        t0 = time.time()

        for k_steps in k_targets:
            print(f"\n  Computing k={k_steps}...")
            rwpe_dict = {}
            failed = 0

            for i, graph in enumerate(all_graphs):
                try:
                    rwpe = compute_rwpe_for_graph(graph.edge_index, graph.x.shape[0], k_steps)
                    rwpe_dict[i] = torch.tensor(rwpe, dtype=torch.float)
                except Exception as e:
                    failed += 1
                    rwpe_dict[i] = torch.zeros((graph.x.shape[0], k_steps), dtype=torch.float)

                if (i + 1) % 2000 == 0:
                    print(f"    Processed {i + 1}/{len(all_graphs)} graphs...")

            elapsed = time.time() - t0
            print(f"    Completed: {len(rwpe_dict)} graphs, {failed} failed in {elapsed:.1f}s")

            variances = compute_rwpe_variance_stats(rwpe_dict, k_steps)
            results[k_steps] = {'rwpe': rwpe_dict, 'variances': variances}

        print(f"  Total time: {time.time() - t0:.1f}s")

    return results


def main():
    print("=" * 70)
    print("Extended RWPE Precomputation (k=16, 24, 32)")
    print("=" * 70)

    # Step 1: Build datasets (reuse graph structure)
    print("\n[1/4] Building datasets...")
    t0 = time.time()
    train_graphs, val_graphs, test_graphs, scaler = build_datasets(
        csv_path=CSV_PATH,
        multimer='3-RU',
        feature_level='rich',
        seed=71,
    )
    all_graphs = train_graphs + val_graphs + test_graphs
    print(f"  Total graphs: {len(all_graphs)} (train={len(train_graphs)}, "
          f"val={len(val_graphs)}, test={len(test_graphs)})")
    print(f"  Time: {time.time() - t0:.1f}s")

    # Graph size stats
    sizes = [g.x.shape[0] for g in all_graphs]
    print(f"  Node counts: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}, std={np.std(sizes):.1f}")

    # Step 2: Load existing k=16 data, compute higher k values
    print(f"\n[2/4] Computing RWPE...")

    # Load existing k=16 data
    if os.path.exists(RWPE_ORIGINAL_PATH):
        print(f"  Loading existing k=16 data from rwpe_data.pkl...")
        with open(RWPE_ORIGINAL_PATH, 'rb') as f:
            existing_data = pickle.load(f)
        k16_dict = existing_data['rwpe']
        # Convert from torch tensors to numpy if needed
        if isinstance(list(k16_dict.values())[0], torch.Tensor):
            k16_dict = {i: v.numpy() for i, v in k16_dict.items()}
        print(f"  Loaded {len(k16_dict)} graphs with k=16")
    else:
        print(f"  ERROR: rwpe_data.pkl not found!")
        return

    # Compute all needed k values
    results = convert_k16_and_compute_higher_k(all_graphs, k16_dict, TARGET_K_VALUES)
    all_variances = []
    all_k = []

    # Step 3: Save and print stats
    print(f"\n[3/4] Saving RWPE data and diagnostics...")

    for k_steps in sorted(results.keys()):
        r = results[k_steps]
        variances = r['variances']

        zero_var = np.sum(variances < 1e-10)
        print(f"\n  k={k_steps}:")
        print(f"    Variance: mean={variances.mean():.8f}, median={np.median(variances):.8f}")
        print(f"    Zero-variance graphs: {zero_var}/{len(variances)} ({100*zero_var/len(variances):.1f}%)")

        # Save
        save_path = os.path.join(OUTPUT_DIR, f'rwpe_k{k_steps}.pkl')
        save_data = {
            'rwpe': r['rwpe'],
            'k_steps': k_steps,
            'num_graphs': len(all_graphs),
            'train_size': len(train_graphs),
            'val_size': len(val_graphs),
            'test_size': len(test_graphs),
            'variances': variances,
        }
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"    Saved to {save_path} ({file_size_mb:.1f} MB)")

        all_variances.append(variances)
        all_k.append(k_steps)

    # Cross-k comparison
    print(f"\n[4/4] Cross-k comparison...")
    print(f"  {'k':<10} {'Mean Variance':<18} {'Median Variance':<18} {'Nonzero %':<12}")
    print(f"  {'-'*58}")
    for k, vars_ in zip(all_k, all_variances):
        nonzero_pct = 100 * np.mean(vars_ > 1e-10)
        print(f"  k={k:<8} {vars_.mean():<18.8f} {np.median(vars_):<18.8f} {nonzero_pct:<12.1f}%")

    plot_cross_k_comparison(all_variances, all_k)

    print(f"\nFiles created:")
    for k in sorted(results.keys()):
        p = os.path.join(OUTPUT_DIR, f'rwpe_k{k}.pkl')
        if os.path.exists(p):
            print(f"  {p} ({os.path.getsize(p)/(1024*1024):.1f} MB)")

    print("\n" + "=" * 70)
    print("Extended RWPE Precomputation COMPLETE")
    print("=" * 70)


def plot_cross_k_comparison(all_variances, all_k):
    """Plot variance distributions across k values."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ['#2196F3', '#FF9800', '#4CAF50']

    # 1. Histogram overlay
    ax = axes[0]
    for k, vars_ in zip(all_k, all_variances):
        ax.hist(np.log10(vars_[vars_ > 1e-10]), bins=60, alpha=0.5,
                label=f'k={k}', color=colors[len(ax.patches) % 3])
    ax.set_xlabel('log10(Per-graph RWPE variance)')
    ax.set_ylabel('Count')
    ax.set_title('RWPE Variance Distribution by k')
    ax.legend()

    # 2. Mean variance per k-step (average over graphs)
    ax = axes[1]
    for ki, (k, vars_) in enumerate(zip(all_k, all_variances)):
        ax.bar(ki, np.mean(vars_), color=colors[ki], label=f'k={k}')
    ax.set_xticks(range(len(all_k)))
    ax.set_xticklabels([f'k={k}' for k in all_k])
    ax.set_ylabel('Mean RWPE Variance')
    ax.set_title('Mean RWPE Variance vs k')
    ax.legend()

    # 3. Fraction of graphs with non-degenerate RWPE
    ax = axes[2]
    for ki, (k, vars_) in enumerate(zip(all_k, all_variances)):
        frac_nonzero = np.mean(vars_ > 1e-10)
        ax.bar(ki, frac_nonzero * 100, color=colors[ki], label=f'k={k}')
    ax.set_xticks(range(len(all_k)))
    ax.set_xticklabels([f'k={k}' for k in all_k])
    ax.set_ylabel('% Graphs with Nonzero Variance')
    ax.set_title('Fraction of Informative RWPE')
    ax.set_ylim(0, 105)
    for ki, frac in enumerate([np.mean(v > 1e-10) for v in all_variances]):
        ax.text(ki, frac * 100 + 1, f'{frac*100:.1f}%', ha='center')
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'rwpe_cross_k_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-k comparison to {save_path}")


if __name__ == '__main__':
    main()
