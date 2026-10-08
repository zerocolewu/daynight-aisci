"""
Analysis utilities for GRIN experiments.
Includes embedding analysis, visualization, and diagnostic tools.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import torch

from models import GINMaxAggregation
from data_utils import TgDataset
from torch_geometric.loader import DataLoader


def extract_embeddings(model, loader, device):
    """
    Extract graph embeddings from the model.

    Args:
        model: Trained model
        loader: Data loader
        device: Device to use

    Returns:
        embeddings: Numpy array of embeddings
        targets: Numpy array of targets
        smiles: List of SMILES strings
    """
    model.eval()
    embeddings = []
    targets = []
    smiles_list = []

    # Register hook to get embeddings before final head
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # Hook on the layer before the head
    # For GINMaxAggregation, this is after global pooling
    handle = model.head[0].register_forward_hook(get_activation('embed'))

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            _ = model(batch.x, batch.edge_index, batch.batch)

            # Get embeddings
            emb = activation['embed'].cpu().numpy()

            embeddings.append(emb)
            targets.append(batch.y.cpu().numpy())

            # Get SMILES if available
            if hasattr(batch, 'smiles'):
                smiles_list.extend(batch.smiles)

    handle.remove()

    embeddings = np.concatenate(embeddings, axis=0)
    targets = np.concatenate(targets, axis=0)

    return embeddings, targets, smiles_list


def analyze_embedding_similarity(embeddings_1ru, embeddings_3ru):
    """
    Analyze cosine similarity between 1-RU and 3-RU embeddings.

    Args:
        embeddings_1ru: Embeddings from 1-RU models
        embeddings_3ru: Embeddings from 3-RU models

    Returns:
        Dictionary with similarity metrics
    """
    # Compute cosine similarities
    similarities = []
    for i in range(min(len(embeddings_1ru), len(embeddings_3ru))):
        sim = cosine_similarity(
            embeddings_1ru[i:i+1],
            embeddings_3ru[i:i+1]
        )[0, 0]
        similarities.append(sim)

    similarities = np.array(similarities)

    return {
        'mean_similarity': np.mean(similarities),
        'std_similarity': np.std(similarities),
        'median_similarity': np.median(similarities),
        'min_similarity': np.min(similarities),
        'max_similarity': np.max(similarities),
        'similarities': similarities
    }


def visualize_embeddings_tsne(embeddings, targets, title, save_path):
    """
    Create t-SNE visualization of embeddings.

    Args:
        embeddings: Numpy array of embeddings
        targets: Numpy array of targets
        title: Plot title
        save_path: Path to save the plot
    """
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=targets, cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Tg Value')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_embedding_similarity_distribution(similarities, save_path):
    """
    Plot distribution of embedding similarities.

    Args:
        similarities: Array of cosine similarities
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    plt.hist(similarities, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(similarities), color='red', linestyle='--',
                label=f'Mean: {np.mean(similarities):.3f}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.title('Distribution of 1-RU vs 3-RU Embedding Similarities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def compare_experiments(results_dir):
    """
    Create comprehensive comparison of all experiments.

    Args:
        results_dir: Directory containing all experiment results
    """
    experiments = ['baseline', 'max_only', 'max_l1', 'full_grin', 'mean_aug']

    results_summary = []

    for exp_name in experiments:
        exp_dir = os.path.join(results_dir, exp_name)
        results_file = os.path.join(exp_dir, 'results.json')

        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)

            results_summary.append({
                'Experiment': exp_name,
                'Test RMSE': results['test_rmse'],
                'Test MAE': results['test_mae'],
                'Best Val RMSE': results['best_val_rmse'],
                'Best Epoch': results['best_epoch']
            })

    df = pd.DataFrame(results_summary)
    df = df.round(4)

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # RMSE comparison
    ax = axes[0]
    x = range(len(df))
    bars = ax.bar(x, df['Test RMSE'], color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
    ax.set_xticks(x)
    ax.set_xticklabels(df['Experiment'], rotation=45, ha='right')
    ax.set_ylabel('Test RMSE')
    ax.set_title('Test RMSE Comparison')
    ax.axhline(y=37.0, color='red', linestyle='--', label='GraphFormer baseline')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar, rmse in zip(bars, df['Test RMSE']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.2f}',
                ha='center', va='bottom', fontweight='bold')

    # MAE comparison
    ax = axes[1]
    bars = ax.bar(x, df['Test MAE'], color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
    ax.set_xticks(x)
    ax.set_xticklabels(df['Experiment'], rotation=45, ha='right')
    ax.set_ylabel('Test MAE')
    ax.set_title('Test MAE Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, mae in zip(bars, df['Test MAE']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae:.2f}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'detailed_comparison.png'), dpi=150)
    plt.close()

    return df


def analyze_sparsity(model, save_path):
    """
    Analyze weight sparsity in the model.

    Args:
        model: Trained model
        save_path: Path to save analysis
    """
    layer_stats = []

    for name, param in model.named_parameters():
        if 'weight' in name:
            weights = param.detach().cpu().numpy().flatten()
            sparsity = np.sum(np.abs(weights) < 1e-6) / len(weights)
            l1_norm = np.sum(np.abs(weights))
            l2_norm = np.sqrt(np.sum(weights**2))

            layer_stats.append({
                'Layer': name,
                'Sparsity': sparsity,
                'L1 Norm': l1_norm,
                'L2 Norm': l2_norm,
                'Mean |W|': np.mean(np.abs(weights)),
                'Std |W|': np.std(np.abs(weights))
            })

    df = pd.DataFrame(layer_stats)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sparsity per layer
    ax = axes[0]
    ax.barh(range(len(df)), df['Sparsity'])
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([name[:30] for name in df['Layer']], fontsize=8)
    ax.set_xlabel('Sparsity (fraction of weights < 1e-6)')
    ax.set_title('Weight Sparsity per Layer')
    ax.grid(True, alpha=0.3, axis='x')

    # L1 norm per layer
    ax = axes[1]
    ax.barh(range(len(df)), df['L1 Norm'])
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([name[:30] for name in df['Layer']], fontsize=8)
    ax.set_xlabel('L1 Norm')
    ax.set_title('L1 Norm per Layer')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return df


if __name__ == '__main__':
    print("Analysis utilities loaded successfully")
