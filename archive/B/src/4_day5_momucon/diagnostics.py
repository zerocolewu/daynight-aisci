#!/usr/bin/env python3
"""
Diagnostic Analysis for MoMuCon-NT-Xent Pretraining.

Generates:
  1. Pretraining loss curve
  2. Collapse monitoring plots (cosine similarity trajectory)
  3. t-SNE visualization of encoder embeddings (500 polymers)
  4. Finetuning comparison bar chart
"""

import os
import json
import random
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from data_utils import prepare_pi1m_pretrain_data, EDGE_DIM
from model import GATEncoder

warnings.filterwarnings('ignore')

WORK_DIR = "/home/home_public/aisci/AIscientist/src/4_day5_momucon"
RESULTS_DIR = os.path.join(WORK_DIR, "results")
MODELS_DIR = os.path.join(WORK_DIR, "models")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# --------------------------------------------------------------------
# Pretraining loss curve
# --------------------------------------------------------------------
def plot_pretrain_loss(loss_csv_path=None):
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

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, losses, 'b-', linewidth=1.5, alpha=0.7)
    ax.scatter(epochs, losses, s=8, alpha=0.5)
    best_idx = np.argmin(losses)
    ax.scatter([epochs[best_idx]], [losses[best_idx]], s=120, c='red', marker='*',
               zorder=5, label=f'Best: epoch {epochs[best_idx]} (loss={losses[best_idx]:.4f})')
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


# --------------------------------------------------------------------
# Collapse diagnostics
# --------------------------------------------------------------------
def plot_collapse_diagnostics(diag_path=None):
    if diag_path is None:
        diag_path = os.path.join(RESULTS_DIR, "pretrain_diagnostics.json")
    if not os.path.exists(diag_path):
        print(f"Diagnostics JSON not found: {diag_path}")
        return

    with open(diag_path, 'r') as f:
        diag = json.load(f)

    epochs = [d['epoch'] for d in diag]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(epochs, [d['loss'] for d in diag], 'b-', marker='o', markersize=4)
    ax.set_xlabel('Epoch'); ax.set_ylabel('NT-Xent Loss')
    ax.set_title('Pretraining Loss'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, [d.get('mean_pairwise', 0) for d in diag], 'g-', marker='s', markersize=4)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Collapse threshold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Mean Cos Sim')
    ax.set_title('Collapse Monitor: Mean Pairwise Cosine Similarity')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    pos_vals = [d.get('pos_sim', 0) for d in diag]
    neg_vals = [d.get('neg_sim', 0) for d in diag]
    ax.plot(epochs, pos_vals, 'b-', marker='o', markersize=4, label='Positive pairs')
    ax.plot(epochs, neg_vals, 'r-', marker='s', markersize=4, label='Negative pairs')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Cosine Similarity')
    ax.set_title('Positive vs Negative Pair Similarity')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    gaps = [d.get('pos_neg_gap', 0) for d in diag]
    ax.plot(epochs, gaps, 'purple', marker='D', markersize=4)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Pos-Neg Gap')
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
    if encoder_path is None:
        encoder_path = os.path.join(MODELS_DIR, "pretrained_encoder.pt")
    if not os.path.exists(encoder_path):
        print(f"Encoder not found: {encoder_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== t-SNE Visualization ({n_polymers} polymers) ===")

    all_pairs = prepare_pi1m_pretrain_data(force_rebuild=False)
    random.seed(42)
    np.random.seed(42)
    sampled = random.sample(all_pairs, min(n_polymers, len(all_pairs)))
    print(f"Sampled {len(sampled)} polymers")

    sample_data = sampled[0][0]
    input_dim = sample_data.x.size(1)
    edge_dim = sample_data.edge_attr.size(1)

    encoder = GATEncoder(
        input_dim=input_dim, hidden_dim=256, num_heads=4,
        num_layers=4, dropout=0.1, edge_dim=edge_dim, output_dim=256
    ).to(device)

    state_dict = torch.load(encoder_path, map_location=device)
    encoder_state = encoder.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in encoder_state}
    encoder_state.update(matched)
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    print(f"Loaded encoder ({len(matched)}/{len(encoder_state)} params)")

    embeddings = []
    labels = []
    view_type = []

    for i, (g1, g3) in enumerate(tqdm(sampled, desc="Encoding graphs")):
        for g, vt in [(g1, '1-RU'), (g3, '3-RU')]:
            batch = Batch.from_data_list([g]).to(device)
            edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
            with torch.no_grad():
                h = encoder(batch.x, batch.edge_index, edge_attr, batch.batch)
            embeddings.append(h.cpu().numpy().flatten())
            labels.append(i)
            view_type.append(vt)

    embeddings = np.array(embeddings)  # [2N, 256]
    labels = np.array(labels)

    print("Computing t-SNE...")
    if embeddings.shape[1] > 50:
        pca = PCA(n_components=50)
        embeddings_reduced = pca.fit_transform(embeddings)
        print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        embeddings_reduced = embeddings

    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(30, len(sampled) - 1),
                n_iter=1000, verbose=0)
    embeddings_2d = tsne.fit_transform(embeddings_reduced)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: color by polymer ID (show first 200)
    ax = axes[0]
    n_show = min(n_polymers, 200)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_show, 20)))

    for i in range(n_show):
        idx_1ru = 2 * i
        idx_3ru = 2 * i + 1
        if idx_3ru < embeddings_2d.shape[0]:
            ax.plot([embeddings_2d[idx_1ru, 0], embeddings_2d[idx_3ru, 0]],
                    [embeddings_2d[idx_1ru, 1], embeddings_2d[idx_3ru, 1]],
                    '-', color=colors[i % 20], alpha=0.3, linewidth=0.5)

    indices_1ru = [2 * i for i in range(n_show) if 2 * i < embeddings_2d.shape[0]]
    indices_3ru = [2 * i + 1 for i in range(n_show) if 2 * i + 1 < embeddings_2d.shape[0]]
    c_vals = [colors[i % 20] for i in range(len(indices_1ru))]

    ax.scatter(embeddings_2d[indices_1ru, 0], embeddings_2d[indices_1ru, 1],
               c=c_vals, marker='o', s=15, alpha=0.7, label='1-RU')
    ax.scatter(embeddings_2d[indices_3ru, 0], embeddings_2d[indices_3ru, 1],
               c=c_vals, marker='s', s=25, alpha=0.7, label='3-RU')
    ax.set_title(f't-SNE: Polymer Embeddings by ID ({n_show} polymers)')
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=8)

    # Right: color by view type
    ax = axes[1]
    for i in range(min(n_show, embeddings_2d.shape[0] // 2)):
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
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=8)

    plt.suptitle('MoMuCon-NT-Xent: t-SNE of Pretrained Encoder Embeddings', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "tsne_embeddings.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to: {save_path}")

    # Distance stats
    n_dist = min(n_show, embeddings_2d.shape[0] // 2)
    same_dist, diff_dist = [], []
    for i in range(n_dist):
        idx_i_1 = 2 * i; idx_i_3 = 2 * i + 1
        same_dist.append(np.linalg.norm(embeddings_2d[idx_i_1] - embeddings_2d[idx_i_3]))
        j = (i + 1) % n_dist
        idx_j_1 = 2 * j
        diff_dist.append(np.linalg.norm(embeddings_2d[idx_i_1] - embeddings_2d[idx_j_1]))

    same_dist = np.array(same_dist); diff_dist = np.array(diff_dist)
    print(f"t-SNE distances: Same-pair mean={same_dist.mean():.4f}±{same_dist.std():.4f}")
    print(f"  Different-pair mean={diff_dist.mean():.4f}±{diff_dist.std():.4f}")
    print(f"  Ratio (diff/same): {diff_dist.mean() / max(same_dist.mean(), 1e-8):.2f}")


# --------------------------------------------------------------------
# Finetuning comparison bar chart
# --------------------------------------------------------------------
def plot_finetune_comparison(results_path=None):
    if results_path is None:
        results_path = os.path.join(RESULTS_DIR, "finetune_results.json")
    if not os.path.exists(results_path):
        print(f"Finetune results not found: {results_path}")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    modes = list(results.keys())
    rmses = [results[m]['test_rmse'] for m in modes]
    sorted_idx = np.argsort(rmses)
    modes = [modes[i] for i in sorted_idx]
    rmses = [rmses[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(range(len(modes)), rmses,
                  color=plt.cm.viridis(np.linspace(0.2, 0.8, len(modes))),
                  edgecolor='black', linewidth=0.5)

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Test RMSE (K)')
    ax.set_title('Tg Prediction: Finetuning Results Comparison')

    for bar, rmse in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{rmse:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    historical = {
        'RWPE-only (Day 2)': 35.94,
        'PSPE-combined (Day 3)': 35.44,
        'SelfGNN (Day 4)': 34.89,
    }
    hist_colors = ['red', 'orange', 'green']
    for (label, val), color in zip(historical.items(), hist_colors):
        ax.axhline(y=val, color=color, linestyle='--', alpha=0.5, linewidth=1)
        ax.text(len(modes) - 0.5, val + 0.15, f'{label}: {val:.2f}K',
                fontsize=7, color=color, va='bottom', ha='right')

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "finetune_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved finetune comparison to: {save_path}")


def main():
    print("=" * 70)
    print("MoMuCon-NT-Xent: Diagnostic Analysis")
    print("=" * 70)

    print("\n[1/4] Plotting pretraining loss curve...")
    plot_pretrain_loss()

    print("\n[2/4] Plotting collapse diagnostics...")
    plot_collapse_diagnostics()

    print("\n[3/4] Computing t-SNE visualization...")
    compute_tsne(n_polymers=500)

    print("\n[4/4] Plotting finetuning comparison...")
    plot_finetune_comparison()

    print(f"\n=== Diagnostics Complete ===")
    print(f"All figures saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
