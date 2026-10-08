#!/usr/bin/env python3
"""
Diagnostic Analysis for MoMuCon-NT-Xent Pretraining.

Includes:
  1. Collapse monitoring (cosine similarity trajectory)
  2. t-SNE visualization of encoder embeddings
  3. Pretraining loss curve plotting
  4. Finetuning comparison plots
"""

import os
import json
import pickle
import random
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from data_utils import (
    prepare_pi1m_pretrain_data,
    prepare_tg_data,
    TOTAL_DIM,
    EDGE_DIM,
)
from model import GATEncoder

warnings.filterwarnings('ignore')

WORK_DIR = "/home/home_public/aisci/AIscientist/src/4_day5_momucon_ntxent"
RESULTS_DIR = os.path.join(WORK_DIR, "results")
MODELS_DIR = os.path.join(WORK_DIR, "models")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# --------------------------------------------------------------------
# Plot pretraining loss
# --------------------------------------------------------------------
def plot_pretrain_loss(loss_csv_path=None):
    """Plot pretraining NT-Xent loss curve."""
    if loss_csv_path is None:
        loss_csv_path = os.path.join(RESULTS_DIR, "pretrain_loss.csv")

    if not os.path.exists(loss_csv_path):
        print(f"Loss CSV not found: {loss_csv_path}")
        return

    import csv
    epochs, losses = [], []
    with open(loss_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            losses.append(float(row['loss']))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(epochs, losses, 'b-', linewidth=1.5, alpha=0.7)
    ax.scatter(epochs, losses, s=10, alpha=0.5)

    # Mark best
    best_idx = np.argmin(losses)
    ax.scatter([epochs[best_idx]], [losses[best_idx]], s=100, c='red', marker='*',
               zorder=5, label=f'Best: epoch {epochs[best_idx]}, loss={losses[best_idx]:.4f}')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('NT-Xent Loss')
    ax.set_title('MoMuCon NT-Xent Pretraining Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(FIGURES_DIR, "pretrain_loss.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss plot to: {save_path}")

    return epochs, losses


# --------------------------------------------------------------------
# Plot collapse diagnostics
# --------------------------------------------------------------------
def plot_collapse_diagnostics(diag_path=None):
    """Plot collapse monitoring metrics."""
    if diag_path is None:
        diag_path = os.path.join(RESULTS_DIR, "pretrain_diagnostics.json")

    if not os.path.exists(diag_path):
        print(f"Diagnostics JSON not found: {diag_path}")
        return

    with open(diag_path, 'r') as f:
        diag = json.load(f)

    epochs = [d['epoch'] for d in diag]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss
    ax = axes[0, 0]
    ax.plot(epochs, [d['loss'] for d in diag], 'b-', marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NT-Xent Loss')
    ax.set_title('Pretraining Loss')
    ax.grid(True, alpha=0.3)

    # 2. Mean pairwise cosine similarity
    ax = axes[0, 1]
    ax.plot(epochs, [d.get('mean_pairwise', 0) for d in diag], 'g-', marker='s', markersize=4)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Collapse threshold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Cos Sim')
    ax.set_title('Collapse Monitor: Mean Pairwise Cosine Similarity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Positive vs negative pair similarity
    ax = axes[1, 0]
    pos_vals = [d.get('pos_sim', 0) for d in diag]
    neg_vals = [d.get('neg_sim', 0) for d in diag]
    ax.plot(epochs, pos_vals, 'b-', marker='o', markersize=4, label='Positive pairs')
    ax.plot(epochs, neg_vals, 'r-', marker='s', markersize=4, label='Negative pairs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Positive vs Negative Pair Similarity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Pos-Neg gap
    ax = axes[1, 1]
    gaps = [d.get('pos_neg_gap', 0) for d in diag]
    ax.plot(epochs, gaps, 'purple', marker='D', markersize=4)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pos-Neg Gap')
    ax.set_title('Positive-Negative Similarity Gap')
    ax.grid(True, alpha=0.3)

    plt.suptitle('MoMuCon NT-Xent Pretraining Diagnostics', fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "pretrain_diagnostics.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved diagnostics plot to: {save_path}")


# --------------------------------------------------------------------
# t-SNE visualization
# --------------------------------------------------------------------
def compute_tsne(encoder_path=None, n_polymers=500):
    """Compute t-SNE of encoder embeddings for random PI1M polymers.

    For each polymer, encode both 1-RU and 3-RU views.
    Plot same-polymer pairs should cluster together.
    """
    if encoder_path is None:
        encoder_path = os.path.join(MODELS_DIR, "pretrained_encoder.pt")

    if not os.path.exists(encoder_path):
        print(f"Encoder not found: {encoder_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== t-SNE Visualization ({n_polymers} polymers) ===")

    # Load PI1M data
    all_pairs = prepare_pi1m_pretrain_data(force_rebuild=False)

    # Sample polymers
    random.seed(42)
    np.random.seed(42)
    sampled = random.sample(all_pairs, min(n_polymers, len(all_pairs)))
    print(f"Sampled {len(sampled)} polymers")

    # Load encoder
    sample_data = sampled[0][0]
    input_dim = sample_data.x.size(1)
    edge_dim = sample_data.edge_attr.size(1) if hasattr(sample_data, 'edge_attr') and sample_data.edge_attr is not None else None

    encoder = GATEncoder(
        input_dim=input_dim, hidden_dim=256, num_heads=4,
        num_layers=4, dropout=0.1, edge_dim=edge_dim
    ).to(device)

    state_dict = torch.load(encoder_path, map_location=device)
    encoder_state = encoder.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in encoder_state}
    encoder_state.update(matched)
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    print(f"Loaded encoder ({len(matched)}/{len(encoder_state)} params matched)")

    # Encode all views
    embeddings = []  # [2 * n_polymers, 256]
    labels = []      # polymer index
    view_type = []    # '1-RU' or '3-RU'

    for i, (g1, g3) in enumerate(tqdm(sampled, desc="Encoding graphs")):
        # 1-RU
        batch_1ru = Batch.from_data_list([g1]).to(device)
        edge_attr_1 = batch_1ru.edge_attr if hasattr(batch_1ru, 'edge_attr') else None
        with torch.no_grad():
            h1 = encoder(batch_1ru.x, batch_1ru.edge_index, edge_attr_1, batch_1ru.batch)
        embeddings.append(h1.cpu().numpy().flatten())
        labels.append(i)
        view_type.append('1-RU')

        # 3-RU
        batch_3ru = Batch.from_data_list([g3]).to(device)
        edge_attr_3 = batch_3ru.edge_attr if hasattr(batch_3ru, 'edge_attr') else None
        with torch.no_grad():
            h3 = encoder(batch_3ru.x, batch_3ru.edge_index, edge_attr_3, batch_3ru.batch)
        embeddings.append(h3.cpu().numpy().flatten())
        labels.append(i)
        view_type.append('3-RU')

    embeddings = np.array(embeddings)  # [2N, 256]
    labels = np.array(labels)

    # t-SNE
    print("Computing t-SNE...")
    # First reduce to 50 dims with PCA for speed
    if embeddings.shape[1] > 50:
        pca = PCA(n_components=50)
        embeddings_pca = pca.fit_transform(embeddings)
        print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        embeddings_pca = embeddings

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sampled) - 1),
                n_iter=1000, verbose=0)
    embeddings_2d = tsne.fit_transform(embeddings_pca)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: color by polymer (each polymer has 2 points connected by line)
    ax = axes[0]
    n_show = min(n_polymers, 200)
    colors = plt.cm.tab20(np.linspace(0, 1, n_show))

    for i in range(n_show):
        idx_1ru = 2 * i
        idx_3ru = 2 * i + 1
        ax.plot([embeddings_2d[idx_1ru, 0], embeddings_2d[idx_3ru, 0]],
                [embeddings_2d[idx_1ru, 1], embeddings_2d[idx_3ru, 1]],
                '-', color=colors[i % 20], alpha=0.3, linewidth=0.5)

    # 1-RU points (circles)
    indices_1ru = [2 * i for i in range(n_show)]
    c_1ru = [colors[i % 20] for i in range(n_show)]
    ax.scatter(embeddings_2d[indices_1ru, 0], embeddings_2d[indices_1ru, 1],
               c=c_1ru, marker='o', s=15, alpha=0.7, label='1-RU')
    # 3-RU points (squares)
    indices_3ru = [2 * i + 1 for i in range(n_show)]
    ax.scatter(embeddings_2d[indices_3ru, 0], embeddings_2d[indices_3ru, 1],
               c=c_1ru, marker='s', s=25, alpha=0.7, label='3-RU')

    ax.set_title(f't-SNE: Polymer Embeddings ({n_show} polymers)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=8)

    # Right: color by view type
    ax = axes[1]
    for i in range(n_show):
        idx_1ru = 2 * i
        idx_3ru = 2 * i + 1
        ax.plot([embeddings_2d[idx_1ru, 0], embeddings_2d[idx_3ru, 0]],
                [embeddings_2d[idx_1ru, 1], embeddings_2d[idx_3ru, 1]],
                '-', color='gray', alpha=0.15, linewidth=0.3)

    ax.scatter(embeddings_2d[indices_1ru, 0], embeddings_2d[indices_1ru, 1],
               c='blue', marker='o', s=10, alpha=0.5, label='1-RU view')
    ax.scatter(embeddings_2d[indices_3ru, 0], embeddings_2d[indices_3ru, 1],
               c='red', marker='s', s=10, alpha=0.5, label='3-RU view')
    ax.set_title('t-SNE: View Type Comparison')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=8)

    plt.suptitle('MoMuCon-NT-Xent: t-SNE of Pretrained Encoder Embeddings', fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "tsne_embeddings.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to: {save_path}")

    # Compute same-polymer vs different-polymer distances in t-SNE space
    same_dist = []
    diff_dist = []
    for i in range(n_show):
        idx_i_1 = 2 * i
        idx_i_3 = 2 * i + 1
        dist_same = np.linalg.norm(embeddings_2d[idx_i_1] - embeddings_2d[idx_i_3])
        same_dist.append(dist_same)

        # Distance to a different polymer's 1-RU
        j = (i + 1) % n_show
        idx_j_1 = 2 * j
        dist_diff = np.linalg.norm(embeddings_2d[idx_i_1] - embeddings_2d[idx_j_1])
        diff_dist.append(dist_diff)

    same_dist = np.array(same_dist)
    diff_dist = np.array(diff_dist)
    print(f"\nt-SNE distance stats:")
    print(f"  Same-polymer pairs: mean={same_dist.mean():.4f}, std={same_dist.std():.4f}")
    print(f"  Different-polymer pairs: mean={diff_dist.mean():.4f}, std={diff_dist.std():.4f}")
    print(f"  Ratio (diff/same): {diff_dist.mean() / max(same_dist.mean(), 1e-8):.2f}")

    return embeddings_2d, labels


# --------------------------------------------------------------------
# Finetuning comparison bar chart
# --------------------------------------------------------------------
def plot_finetune_comparison(results_path=None):
    """Bar chart comparing finetuning results."""
    if results_path is None:
        results_path = os.path.join(RESULTS_DIR, "finetune_results.json")

    if not os.path.exists(results_path):
        print(f"Finetune results not found: {results_path}")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    modes = list(results.keys())
    rmses = [results[m]['test_rmse'] for m in modes]

    # Sort by RMSE
    sorted_idx = np.argsort(rmses)
    modes = [modes[i] for i in sorted_idx]
    rmses = [rmses[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(modes)), rmses, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(modes))),
                  edgecolor='black', linewidth=0.5)

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Test RMSE (K)')
    ax.set_title('Tg Prediction: Finetuning Results Comparison')

    # Add value labels
    for bar, rmse in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{rmse:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add reference lines for historical baselines
    historical = {
        'RWPE-only (Day 2)': 35.94,
        'PSPE-combined (Day 3)': 35.44,
        'SelfGNN (Day 4)': 34.89,  # -0.55K from baseline
    }
    colors = ['red', 'orange', 'green']
    y_pos = max(rmses) * 0.95
    for (label, val), color in zip(historical.items(), colors):
        ax.axhline(y=val, color=color, linestyle='--', alpha=0.5, linewidth=1)
        ax.text(len(modes) - 0.5, val + 0.2, f'{label}: {val:.2f}K',
                fontsize=7, color=color, va='bottom', ha='right')

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "finetune_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved finetune comparison to: {save_path}")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    print("=" * 70)
    print("MoMuCon-NT-Xent: Diagnostic Analysis")
    print("=" * 70)

    # 1. Pretraining loss curve
    print("\n[1/4] Plotting pretraining loss curve...")
    plot_pretrain_loss()

    # 2. Collapse diagnostics
    print("\n[2/4] Plotting collapse diagnostics...")
    plot_collapse_diagnostics()

    # 3. t-SNE
    print("\n[3/4] Computing t-SNE visualization...")
    compute_tsne(n_polymers=300)

    # 4. Finetuning comparison
    print("\n[4/4] Plotting finetuning comparison...")
    plot_finetune_comparison()

    print("\n=== Diagnostics Complete ===")
    print(f"All figures saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
