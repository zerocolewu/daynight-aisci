#!/usr/bin/env python3
"""
Utilities for SelfGNN diagnostics and analysis.

1. Embedding cosine similarity distribution
2. Nearest-neighbor consistency
3. Loss curves and result visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_embedding_stats(embeddings):
    """Compute cosine similarity distribution of random pairs.

    Args:
        embeddings: [N, D] normalized embeddings

    Returns:
        dict with mean, std, min, max of cosine similarities
    """
    N = embeddings.size(0)
    n_pairs = min(500, N * (N - 1) // 2)

    # Random pairs
    idx_i = torch.randint(0, N, (n_pairs,))
    idx_j = torch.randint(0, N, (n_pairs,))
    # Avoid self-pairs
    mask = (idx_i == idx_j)
    idx_j[mask] = (idx_j[mask] + 1) % N

    cos_sim = (embeddings[idx_i] * embeddings[idx_j]).sum(dim=-1).detach().cpu().numpy()

    return {
        'mean': float(np.mean(cos_sim)),
        'std': float(np.std(cos_sim)),
        'min': float(np.min(cos_sim)),
        'max': float(np.max(cos_sim)),
    }


def compute_nearest_neighbor_consistency(z_student, z_teacher):
    """Check if student NN matches teacher embedding of the same polymer.

    Args:
        z_student: [B, D] student embeddings
        z_teacher: [B, D] teacher embeddings

    Returns:
        consistency: Fraction where NN of student_i in student set is teacher_i
    """
    z_s_norm = F.normalize(z_student, dim=-1)
    z_t_norm = F.normalize(z_teacher, dim=-1)

    B = z_s_norm.size(0)
    correct = 0

    # Compute pairwise cosine similarity
    sim_matrix = z_s_norm @ z_t_norm.t()  # [B, B]
    nearest = sim_matrix.argmax(dim=-1)  # [B]

    for i in range(B):
        if nearest[i] == i:
            correct += 1

    return correct / B


def compute_regression_metrics(y_true, y_pred):
    """Compute RMSE, MAE, R2."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
    }


def plot_pretrain_loss(losses, save_path):
    """Plot pretraining loss curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = list(range(1, len(losses) + 1))
    ax.plot(epochs, losses, 'b-', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Cosine Similarity Loss', fontsize=12)
    ax.set_title('SelfGNN Pretraining Loss', fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_embedding_diagnostics(diag_history, save_path):
    """Plot embedding diagnostics over pretraining."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = [d['epoch'] for d in diag_history]

    # Cosine similarity distribution
    ax = axes[0]
    means = [d['cos_sim_mean'] for d in diag_history]
    stds = [d['cos_sim_std'] for d in diag_history]
    mins = [d['cos_sim_min'] for d in diag_history]
    maxs = [d['cos_sim_max'] for d in diag_history]

    ax.plot(epochs, means, 'b-', label='Mean')
    ax.fill_between(epochs,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color='b')
    ax.plot(epochs, mins, 'g--', alpha=0.5, label='Min')
    ax.plot(epochs, maxs, 'r--', alpha=0.5, label='Max')
    ax.axhline(y=1.0, color='red', linestyle=':', label='Collapse')
    ax.axhline(y=0.0, color='gray', linestyle=':', label='Random')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Embedding Cosine Similarity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Nearest-neighbor consistency
    ax = axes[1]
    nn_cons = [d['nn_consistency'] for d in diag_history]
    ax.plot(epochs, nn_cons, 'g-', linewidth=2)
    ax.axhline(y=1.0 / len(epochs), color='gray', linestyle=':', label='Chance')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NN Consistency')
    ax.set_title('Nearest-Neighbor Consistency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_finetune_curves(train_losses, val_losses, train_rmses, val_rmses, save_path):
    """Plot finetuning curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = list(range(1, len(train_losses) + 1))

    # Loss
    ax = axes[0]
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', alpha=0.7)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Finetuning Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE
    ax = axes[1]
    ax.plot(epochs, train_rmses, 'b-', label='Train RMSE', alpha=0.7)
    ax.plot(epochs, val_rmses, 'r-', label='Val RMSE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE (scaled)')
    ax.set_title('Finetuning RMSE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_ablation_comparison(results, save_path):
    """Bar chart comparing ablation results."""
    fig, ax = plt.subplots(figsize=(12, 6))

    names = list(results.keys())
    rmses = [results[n]['test_rmse'] for n in names]
    stds = [results[n].get('test_rmse_std', 0) for n in names]

    colors = ['#3498db'] * len(names)
    # Highlight primary (Split+Standardize)
    if 'SelfGNN Split+Std' in names:
        idx = names.index('SelfGNN Split+Std')
        colors[idx] = '#e74c3c'

    bars = ax.bar(names, rmses, yerr=stds, color=colors, capsize=5)
    ax.set_ylabel('Test RMSE (K)', fontsize=12)
    ax.set_title('SelfGNN Ablation Results: Tg Prediction', fontsize=14)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Baseline line
    if 'No Pretraining' in results:
        baseline_rmse = results['No Pretraining']['test_rmse']
        ax.axhline(y=baseline_rmse, color='gray', linestyle='--',
                   label=f'Baseline: {baseline_rmse:.2f}K')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
