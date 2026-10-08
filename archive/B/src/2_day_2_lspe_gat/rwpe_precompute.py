"""
RWPE (Random Walk Positional Encoding) Precomputation for Polymer Graphs.

Computes k-step random walk return probabilities for each polymer graph.
RWPE_i = [RW_ii^1, RW_ii^2, ..., RW_ii^k] where RW_ii^k = (AD^{-1})^k_{ii}.

Includes quality diagnostics: variance distribution, PCA visualization.
"""

import os
import sys
import pickle
import time
import random
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, degree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add day1 data_utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '1_day1_multimer_feature_gat'))
from data_utils import build_datasets, get_feature_dims

warnings.filterwarnings("ignore")

# Configuration
K_STEPS = 16
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RWPE_SAVE_PATH = os.path.join(OUTPUT_DIR, 'rwpe_data.pkl')
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'alldataset', 'Tg.csv')


def compute_rwpe_for_graph(edge_index, num_nodes, k_steps=K_STEPS):
    """Compute k-step random walk return probabilities for a single graph.

    RW = AD^{-1} (row-stochastic random walk matrix)
    RWPE_i = [P^1_ii, P^2_ii, ..., P^k_ii]

    Args:
        edge_index: PyG edge_index tensor [2, E]
        num_nodes: Number of nodes
        k_steps: Number of random walk steps

    Returns:
        rwpe: numpy array of shape [num_nodes, k_steps]
    """
    # Build dense adjacency (undirected, unweighted)
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).numpy()  # [N, N]

    # Degree matrix
    deg = adj.sum(axis=1)
    deg[deg == 0] = 1  # Avoid division by zero for isolated nodes

    # Random walk transition matrix: P = D^(-1) * A
    # Row-stochastic: each row sums to 1
    P = adj / deg[:, np.newaxis]  # [N, N]

    # Compute powers and extract diagonal
    rwpe = np.zeros((num_nodes, k_steps), dtype=np.float32)
    P_pow = P.copy()
    for step in range(k_steps):
        # Handle potential numerical issues: P_pow could become NaN due to absorbing states
        diag_vals = np.diag(P_pow)
        # Replace NaN with 0
        diag_vals = np.nan_to_num(diag_vals, nan=0.0)
        rwpe[:, step] = diag_vals
        P_pow = P_pow @ P
        P_pow = np.nan_to_num(P_pow, nan=0.0)

    return rwpe


def compute_graph_stats(graphs):
    """Compute statistics about graph sizes."""
    sizes = [g.x.shape[0] for g in graphs]
    return {
        'num_graphs': len(graphs),
        'min_nodes': np.min(sizes),
        'max_nodes': np.max(sizes),
        'mean_nodes': np.mean(sizes),
        'std_nodes': np.std(sizes),
    }


def compute_rwpe_variance(rwpe_dict):
    """Compute per-graph variance of RWPE across nodes.

    For each graph, compute variance of RWPE values across nodes
    (per dimension, then average across dimensions). Report distribution.

    Returns:
        variances: list of per-graph variances
    """
    variances = []
    for idx, rwpe in rwpe_dict.items():
        # Variance across nodes, per dimension
        node_vars = np.var(rwpe, axis=0)  # [k_steps]
        avg_var = np.mean(node_vars)
        variances.append(avg_var)

    return np.array(variances)


def plot_rwpe_variance_histogram(variances, save_path):
    """Plot histogram of per-graph RWPE variances."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.hist(variances, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Per-graph RWPE variance (mean across k-steps)')
    ax.set_ylabel('Number of graphs')
    ax.set_title(f'RWPE Variance Distribution\nMean={variances.mean():.6f}, Median={np.median(variances):.6f}')
    ax.axvline(0, color='red', linestyle='--', linewidth=1, label='Zero variance')
    ax.legend()

    # Log-scale histogram
    ax = axes[1]
    positive_vars = variances[variances > 1e-10]
    if len(positive_vars) > 0:
        ax.hist(np.log10(positive_vars), bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('log10(Per-graph RWPE variance)')
        ax.set_ylabel('Number of graphs')
        ax.set_title('RWPE Variance (log-scale, positive only)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved RWPE variance histogram to {save_path}")
    print(f"  Variance stats: mean={variances.mean():.6f}, median={np.median(variances):.6f}, "
          f"min={variances.min():.6f}, max={variances.max():.6f}")
    print(f"  Graphs with zero variance: {np.sum(variances < 1e-10)} / {len(variances)}")


def plot_rwpe_pca(rwpe_dict, graphs_list, train_graphs, num_examples=6, save_path=None):
    """PCA-project RWPE to 2D for example graphs, color by node position type.

    Node types: chain-end (degree=1), interior (degree=2), branch (degree>=3).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Choose diverse example graphs (different sizes, different RWPE variances)
    variances = []
    valid_indices = []
    for i, idx in enumerate(rwpe_dict.keys()):
        rwpe = rwpe_dict[idx]
        var = np.mean(np.var(rwpe, axis=0))
        variances.append(var)
        valid_indices.append(idx)

    variances = np.array(variances)
    sorted_by_var = np.argsort(variances)

    # Pick examples: 2 lowest, 2 median, 2 highest variance
    if len(sorted_by_var) >= 6:
        example_indices = [
            sorted_by_var[0],
            sorted_by_var[1],
            sorted_by_var[len(sorted_by_var) // 2 - 1],
            sorted_by_var[len(sorted_by_var) // 2],
            sorted_by_var[-2],
            sorted_by_var[-1],
        ]
    else:
        example_indices = sorted_by_var[:num_examples]

    # Build the train graphs list index mapping
    # We need to map rwpe_dict keys to graph data
    graph_map = {i: g for i, g in enumerate(graphs_list)}

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Collect all RWPE for PCA
    all_rwpe_list = [rwpe_dict[i] for i in example_indices]
    all_rwpe = np.vstack(all_rwpe_list)
    scaler = StandardScaler()
    all_rwpe_scaled = scaler.fit_transform(all_rwpe)
    pca = PCA(n_components=2)
    pca.fit(all_rwpe_scaled)

    for ax_idx, example_idx in enumerate(example_indices):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        idx = example_idx
        rwpe = rwpe_dict[idx]
        graph = graph_map.get(idx)
        if graph is None:
            continue

        rwpe_scaled = scaler.transform(rwpe)
        rwpe_2d = pca.transform(rwpe_scaled)

        # Classify nodes by degree
        deg = degree(graph.edge_index[0], num_nodes=graph.x.shape[0]).numpy()
        chain_end = deg == 1
        interior = deg == 2
        branch = deg >= 3

        ax.scatter(rwpe_2d[chain_end, 0], rwpe_2d[chain_end, 1],
                   c='red', marker='o', s=80, label='Chain end (deg=1)', edgecolors='black', linewidths=0.5)
        ax.scatter(rwpe_2d[interior, 0], rwpe_2d[interior, 1],
                   c='blue', marker='s', s=60, label='Interior (deg=2)', edgecolors='black', linewidths=0.5)
        ax.scatter(rwpe_2d[branch, 0], rwpe_2d[branch, 1],
                   c='green', marker='^', s=100, label='Branch (deg>=3)', edgecolors='black', linewidths=0.5)

        var_val = variances[np.where(np.array(valid_indices) == idx)[0][0]]
        ax.set_title(f'RWPE PCA (Var={var_val:.4f}, N={graph.x.shape[0]} nodes)')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        if ax_idx == 0:
            ax.legend(fontsize=7, loc='best')

    for ax in axes[len(example_indices):]:
        ax.set_visible(False)

    plt.suptitle('RWPE PCA: Node Positional Encoding (Colored by Structural Role)', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved RWPE PCA to {save_path}")
        print(f"  PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, "
              f"PC2={pca.explained_variance_ratio_[1]:.3f}")


def plot_rwpe_decay(rwpe_dict, train_graphs, save_path=None):
    """Plot RWPE values vs k-step for a few example nodes, showing decay pattern."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Pick diverse examples
    graph_map = {i: g for i, g in enumerate(train_graphs)}

    # Try to pick examples with different graph sizes
    sizes = [g.x.shape[0] for g in train_graphs]
    size_ranks = np.argsort(sizes)
    example_graphs = []
    if len(size_ranks) >= 6:
        for frac in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            idx = size_ranks[int(frac * (len(size_ranks) - 1))]
            example_graphs.append(idx)
    else:
        example_graphs = list(size_ranks[:6])

    for ax_idx, g_idx in enumerate(example_graphs):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        if g_idx not in rwpe_dict:
            continue
        rwpe = rwpe_dict[g_idx]
        graph = graph_map[g_idx]
        deg = degree(graph.edge_index[0], num_nodes=graph.x.shape[0]).numpy()

        # Plot a few representative nodes
        end_nodes = np.where(deg == 1)[0][:2]
        interior_nodes = np.where(deg == 2)[0][:2]
        branch_nodes = np.where(deg >= 3)[0][:2]

        for n in end_nodes:
            ax.plot(range(1, rwpe.shape[1] + 1), rwpe[n], 'o-', label=f'End (n={n})',
                    markersize=4, linewidth=1.5)
        for n in interior_nodes:
            ax.plot(range(1, rwpe.shape[1] + 1), rwpe[n], 's--', label=f'Int (n={n})',
                    markersize=4, linewidth=1.5)
        for n in branch_nodes:
            ax.plot(range(1, rwpe.shape[1] + 1), rwpe[n], '^:', label=f'Brc (n={n})',
                    markersize=4, linewidth=1.5)

        ax.set_xlabel('k-step')
        ax.set_ylabel('RWPE (return prob)')
        ax.set_title(f'RWPE Decay (N={graph.x.shape[0]} nodes)')
        if ax_idx == 0:
            ax.legend(fontsize=6)

    for ax in axes[len(example_graphs):]:
        ax.set_visible(False)

    plt.suptitle('RWPE Return Probability vs. Random Walk Steps', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved RWPE decay plot to {save_path}")


def main():
    print("=" * 70)
    print("RWPE Precomputation for Polymer Tg Graphs")
    print("=" * 70)

    # Step 1: Build datasets
    print("\n[1/5] Building datasets...")
    t0 = time.time()
    train_graphs, val_graphs, test_graphs, scaler = build_datasets(
        csv_path=CSV_PATH,
        multimer='3-RU',
        feature_level='rich',
        seed=71,
    )
    all_graphs = train_graphs + val_graphs + test_graphs
    print(f"  Total graphs: {len(all_graphs)} (train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)})")
    print(f"  Time: {time.time() - t0:.1f}s")

    # Graph statistics
    stats = compute_graph_stats(all_graphs)
    print(f"\n  Graph stats:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    # Step 2: Compute RWPE
    print(f"\n[2/5] Computing RWPE (k={K_STEPS}) for all graphs...")
    t0 = time.time()
    rwpe_dict = {}  # graph_index -> np.array [num_nodes, k_steps]
    failed = 0
    for i, graph in enumerate(all_graphs):
        try:
            rwpe = compute_rwpe_for_graph(graph.edge_index, graph.x.shape[0], K_STEPS)
            rwpe_dict[i] = rwpe
        except Exception as e:
            failed += 1
            print(f"  Warning: Failed for graph {i}: {e}")
            # Use zeros as fallback
            rwpe_dict[i] = np.zeros((graph.x.shape[0], K_STEPS), dtype=np.float32)

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(all_graphs)} graphs...")

    elapsed = time.time() - t0
    print(f"  Completed: {len(rwpe_dict)} graphs, {failed} failed")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(all_graphs):.3f}s per graph)")

    # Step 3: Quality diagnostics
    print(f"\n[3/5] Running quality diagnostics...")

    # Variance analysis
    variances = compute_rwpe_variance(rwpe_dict)
    zero_var = np.sum(variances < 1e-10)
    low_var = np.sum(variances < 1e-6)
    print(f"  RWPE Variance across nodes (per graph):")
    print(f"    Mean:    {variances.mean():.8f}")
    print(f"    Median:  {np.median(variances):.8f}")
    print(f"    Std:     {variances.std():.8f}")
    print(f"    Min:     {variances.min():.8f}")
    print(f"    Max:     {variances.max():.8f}")
    print(f"    Zero var (<1e-10): {zero_var}/{len(variances)} ({100*zero_var/len(variances):.1f}%)")
    print(f"    Low var (<1e-6):   {low_var}/{len(variances)} ({100*low_var/len(variances):.1f}%)")

    if zero_var > len(variances) * 0.5:
        print(f"  WARNING: More than 50% graphs have degenerate (zero-variance) RWPE!")
        print(f"  This means RWPE may not provide useful positional information for most polymers.")
    else:
        print(f"  OK: RWPE is NOT degenerate — most graphs have meaningful positional variance.")

    # Step 4: Diagnostic plots
    print(f"\n[4/5] Generating diagnostic plots...")

    # Variance histogram
    plot_rwpe_variance_histogram(
        variances,
        os.path.join(OUTPUT_DIR, 'rwpe_variance_hist.png'),
    )

    # PCA visualization
    plot_rwpe_pca(
        rwpe_dict, all_graphs, train_graphs,
        num_examples=6,
        save_path=os.path.join(OUTPUT_DIR, 'rwpe_pca.png'),
    )

    # Decay pattern
    plot_rwpe_decay(
        rwpe_dict, train_graphs,
        save_path=os.path.join(OUTPUT_DIR, 'rwpe_decay.png'),
    )

    # Step 5: Save
    print(f"\n[5/5] Saving RWPE data...")

    # Convert to dict of tensors for PyTorch compatibility
    rwpe_tensors = {i: torch.tensor(v, dtype=torch.float) for i, v in rwpe_dict.items()}

    save_data = {
        'rwpe': rwpe_tensors,
        'k_steps': K_STEPS,
        'num_graphs': len(all_graphs),
        'train_size': len(train_graphs),
        'val_size': len(val_graphs),
        'test_size': len(test_graphs),
        'variances': variances,
    }
    with open(RWPE_SAVE_PATH, 'wb') as f:
        pickle.dump(save_data, f)

    file_size_mb = os.path.getsize(RWPE_SAVE_PATH) / (1024 * 1024)
    print(f"  Saved to: {RWPE_SAVE_PATH}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  RWPE shape per graph: [{rwpe_dict[0].shape[0]}, {K_STEPS}] (varies by graph)")

    print("\n" + "=" * 70)
    print("RWPE Precomputation COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
