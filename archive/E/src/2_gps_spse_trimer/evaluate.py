"""
Evaluation and diagnostic analysis for GPS-SPSE-Trimer model.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import (TgTrimerDataset, get_data_splits, ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
                     compute_shortest_path_distances)
from model import GPSSPSETrimerModel
from train import SPSEDataLoader

WORK_DIR = "/home/home_public/aisci/AIscientist/src/2_gps_spse_trimer"
CSV_PATH = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def plot_training_curves(history_path, work_dir):
    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = [h['epoch'] for h in history]
    train_rmse = [h['train_rmse'] for h in history]
    val_rmse = [h['val_rmse'] for h in history]
    denoise_loss = [h.get('denoise_loss', 0.0) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, train_rmse, label='Train RMSE', color='blue')
    axes[0].plot(epochs, val_rmse, label='Val RMSE', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('Training/Validation RMSE')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, denoise_loss, label='Denoise Loss', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Denoise Loss')
    axes[1].set_title('Denoising Auxiliary Loss Trajectory')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    gap = [t - v for t, v in zip(train_rmse, val_rmse)]
    axes[2].plot(epochs, gap, label='Train-Val Gap', color='orange')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('RMSE Gap')
    axes[2].set_title('Overfitting Gap (Train RMSE - Val RMSE)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, 'training_curves.png'), dpi=150)
    plt.close()
    logger.info("Saved training_curves.png")


def analyze_spse_distribution(dataset, n_samples=500):
    spse_cache = dataset.spse_distances
    all_distances = []
    within_distances = []
    cross_distances = []

    n = min(n_samples, len(spse_cache))
    for i in range(n):
        sp = spse_cache[i].numpy()
        n_atoms = sp.shape[0]

        finite = sp[sp < 1000]
        all_distances.extend(finite.tolist())

        unit_size = n_atoms // 3
        if unit_size > 1:
            cross = sp[:unit_size, 2*unit_size:]
            cross_finite = cross[cross < 1000]
            cross_distances.extend(cross_finite.tolist())

            within = sp[:unit_size, :unit_size]
            within_finite = within[within < 1000]
            within_distances.extend(within_finite.tolist())

    all_dists = np.array(all_distances)
    within_dists = np.array(within_distances) if within_distances else np.array([0])
    cross_dists = np.array(cross_distances) if cross_distances else np.array([0])

    stats = {
        'all_distances': {
            'mean': float(np.mean(all_dists)),
            'std': float(np.std(all_dists)),
            'min': int(np.min(all_dists)),
            'max': int(np.max(all_dists)),
            'median': float(np.median(all_dists)),
        },
        'within_unit_distances': {
            'mean': float(np.mean(within_dists)),
            'std': float(np.std(within_dists)),
            'max': int(np.max(within_dists)),
        },
        'cross_unit_distances': {
            'mean': float(np.mean(cross_dists)),
            'std': float(np.std(cross_dists)),
            'max': int(np.max(cross_dists)),
        },
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(all_dists[all_dists <= 25], bins=25, range=(0, 25), alpha=0.7, label='All pairs')
    if len(within_dists) > 0:
        ax.hist(within_dists[within_dists <= 25], bins=25, range=(0, 25), alpha=0.5, label='Within-unit')
    if len(cross_dists) > 0:
        ax.hist(cross_dists[cross_dists <= 25], bins=25, range=(0, 25), alpha=0.5, label='Cross-unit')
    ax.set_xlabel('Shortest Path Distance')
    ax.set_ylabel('Frequency')
    ax.set_title('SPSE Distance Distribution in Trimer Graphs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(WORK_DIR, 'spse_distance_distribution.png'), dpi=150)
    plt.close()
    logger.info("Saved spse_distance_distribution.png")

    return stats


def analyze_attention_patterns(model, dataset, device, spse_cache, n_samples=20):
    """Analyze attention entropy to check if SPSE provides structured attention."""
    model.eval()
    n_blocks = len(model.blocks)
    layer_entropies = [[] for _ in range(n_blocks)]

    indices = list(range(min(n_samples, len(dataset))))

    with torch.no_grad():
        for idx in indices:
            data = dataset[idx].to(device)
            # Create single-graph batch
            from torch_geometric.data import Batch
            batch = Batch.from_data_list([dataset[idx]]).to(device)

            h = model.input_proj(batch.x)

            # Get precomputed SPSE
            spse_biases = model._precompute_spse(batch.edge_index, batch.batch,
                                                   spse_cache, [idx])

            for layer_idx, block in enumerate(model.blocks):
                h_local = block.gin_conv(h, batch.edge_index, edge_attr=batch.edge_attr)
                h_local = block.mpnn_norm(h_local)
                h_local = block.dropout(h_local)
                h_local = h + h_local

                # Per-graph attention
                for node_indices, spse_enc in spse_biases:
                    n_nodes = len(node_indices)
                    h_graph = h_local[node_indices]

                    Q = block.q_proj(h_graph).view(n_nodes, block.n_heads, block.head_dim).transpose(0, 1)
                    K = block.k_proj(h_graph).view(n_nodes, block.n_heads, block.head_dim).transpose(0, 1)

                    scale = block.head_dim ** -0.5
                    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
                    # Project SPSE encoding through block's trainable projection
                    spse_bias = block.spse_proj(spse_enc).squeeze(-1)  # (n, n)
                    attn_scores = attn_scores + spse_bias.unsqueeze(0)
                    attn_weights = torch.softmax(attn_scores, dim=-1)

                    eps = 1e-10
                    entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)
                    mean_entropy = entropy.mean().item()
                    uniform_entropy = np.log(n_nodes)
                    ratio = mean_entropy / uniform_entropy if uniform_entropy > 0 else 1.0

                    layer_entropies[layer_idx].append({
                        'mean_entropy': mean_entropy,
                        'uniform_entropy': uniform_entropy,
                        'ratio': ratio,
                        'n_nodes': n_nodes,
                    })

                h, _ = block(h, batch.edge_index, batch.edge_attr, batch.batch, spse_biases)

    layer_summaries = []
    for layer_idx in range(n_blocks):
        if not layer_entropies[layer_idx]:
            continue
        ratios = [e['ratio'] for e in layer_entropies[layer_idx]]
        entropies = [e['mean_entropy'] for e in layer_entropies[layer_idx]]
        layer_summaries.append({
            'layer': layer_idx,
            'mean_entropy': float(np.mean(entropies)),
            'std_entropy': float(np.std(entropies)),
            'mean_entropy_ratio': float(np.mean(ratios)),
            'interpretation': 'structured' if np.mean(ratios) < 0.8 else 'near-uniform',
        })

    return {
        'attention_entropy_per_layer': layer_summaries,
        'note': 'Lower entropy ratio = more structured attention. Ratio close to 1.0 = near-uniform.',
    }


def plot_predictions(results_path, work_dir, dataset, model, device, spse_cache, test_idx):
    with open(results_path, 'r') as f:
        results = json.load(f)

    model.eval()
    loader = SPSEDataLoader(dataset, test_idx, spse_cache, batch_size=64, shuffle=False)

    preds = []
    targets = []

    with torch.no_grad():
        for batch, graph_indices in loader:
            batch = batch.to(device)
            pred, _, _ = model(batch, spse_cache=spse_cache,
                               graph_indices=graph_indices, mask_ratio=0.0)
            preds.append(pred.squeeze().cpu().numpy())
            targets.append(batch.y.squeeze().cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(targets, preds, alpha=0.3, s=10)
    min_val = min(targets.min(), preds.min())
    max_val = max(targets.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    ax.set_xlabel('Actual Tg')
    ax.set_ylabel('Predicted Tg')
    ax.set_title(f'GPS-SPSE-Trimer: Predicted vs Actual Tg\nTest RMSE: {results["test_rmse"]:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, 'prediction_scatter.png'), dpi=150)
    plt.close()
    logger.info("Saved prediction_scatter.png")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    results_path = os.path.join(WORK_DIR, 'results.json')
    if not os.path.exists(results_path):
        logger.error("No results.json found. Run training first.")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    config = results['config']

    logger.info("Loading dataset...")
    dataset = TgTrimerDataset(root=WORK_DIR, csv_path=CSV_PATH, spse_dim=config['spse_dim'])
    spse_cache = dataset.spse_distances
    train_idx, val_idx, test_idx = get_data_splits(dataset, seed=71)

    model = GPSSPSETrimerModel(
        node_dim=ATOM_FEATURE_DIM,
        edge_dim=BOND_FEATURE_DIM,
        hidden_dim=config['hidden_dim'],
        transformer_dim=config['transformer_dim'],
        n_heads=config['n_heads'],
        n_blocks=config['n_blocks'],
        spse_dim=config['spse_dim'],
        num_node_features=ATOM_FEATURE_DIM,
    ).to(device)

    checkpoint = torch.load(os.path.join(WORK_DIR, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Loaded best model")

    # Diagnostics
    logger.info("Analyzing SPSE distance distribution...")
    spse_stats = analyze_spse_distribution(dataset, n_samples=500)

    logger.info("Analyzing attention patterns...")
    attention_stats = analyze_attention_patterns(model, dataset, device, spse_cache, n_samples=50)

    history_path = os.path.join(WORK_DIR, 'train_history.json')
    denoise_trajectory = []
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        denoise_trajectory = [h.get('denoise_loss', 0.0) for h in history]
        plot_training_curves(history_path, WORK_DIR)

    plot_predictions(results_path, WORK_DIR, dataset, model, device, spse_cache, test_idx)

    diagnostics = {
        'spse_distance_distribution': spse_stats,
        'attention_patterns': attention_stats,
        'denoise_loss_trajectory': {
            'values': denoise_trajectory,
            'initial': denoise_trajectory[0] if denoise_trajectory else None,
            'final': denoise_trajectory[-1] if denoise_trajectory else None,
            'decreased': denoise_trajectory[-1] < denoise_trajectory[0] if len(denoise_trajectory) > 1 else None,
        },
        'progressive_dropout_rates': [round(0.10 + 0.05 * i, 2) for i in range(config['n_blocks'])],
        'results_summary': {
            'test_rmse': results['test_rmse'],
            'train_rmse': results['train_rmse'],
            'val_rmse': results['val_rmse'],
            'overfitting_gap': results['overfitting_gap'],
            'best_epoch': results['best_epoch'],
        },
    }

    with open(os.path.join(WORK_DIR, 'diagnostics.json'), 'w') as f:
        json.dump(diagnostics, f, indent=2)

    logger.info("=" * 60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)
    logger.info(f"SPSE mean distance: {spse_stats['all_distances']['mean']:.2f}")
    logger.info(f"Within-unit mean distance: {spse_stats['within_unit_distances']['mean']:.2f}")
    logger.info(f"Cross-unit mean distance: {spse_stats['cross_unit_distances']['mean']:.2f}")

    for layer_info in attention_stats['attention_entropy_per_layer']:
        logger.info(f"Layer {layer_info['layer']}: entropy_ratio={layer_info['mean_entropy_ratio']:.3f} ({layer_info['interpretation']})")

    if len(denoise_trajectory) > 1:
        logger.info(f"Denoise loss: {denoise_trajectory[0]:.4f} -> {denoise_trajectory[-1]:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
