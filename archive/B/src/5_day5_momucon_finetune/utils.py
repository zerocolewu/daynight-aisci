#!/usr/bin/env python3
"""
Utilities for MoMuCon-NT-Xent: NT-Xent loss, metric tracking, plotting.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict


# --------------------------------------------------------------------
# NT-Xent Loss (standard InfoNCE formulation)
# --------------------------------------------------------------------
def nt_xent_loss_with_two_heads(z_mono, z_multi, tau=0.07):
    """NT-Xent loss with two separate projection heads.

    Args:
        z_mono: [B, D] monomer head outputs (NOT L2-normalized)
        z_multi: [B, D] multimer head outputs (NOT L2-normalized)
        tau: temperature

    Returns:
        loss: scalar
        pos_sim: mean cosine similarity of positive pairs
        neg_sim_max: max cosine similarity among negative pairs
        stats: dict with additional diagnostics
    """
    B = z_mono.shape[0]
    device = z_mono.device

    # L2 normalize
    z_mono = F.normalize(z_mono, dim=-1)
    z_multi = F.normalize(z_multi, dim=-1)

    # Stack: [mono_0..mono_{B-1}, multi_0..multi_{B-1}]
    Z = torch.cat([z_mono, z_multi], dim=0)  # [2B, D]

    # Similarity matrix
    S = (Z @ Z.T) / tau  # [2B, 2B]

    # Mask self-similarity
    S_masked = S.clone()
    S_masked.fill_diagonal_(float('-inf'))

    # Labels: for z_mono[i], positive is z_multi[i] (index i+B)
    #         for z_multi[i], positive is z_mono[i] (index i)
    labels = torch.cat([
        torch.arange(B, device=device) + B,  # mono positives -> multi
        torch.arange(B, device=device),       # multi positives -> mono
    ], dim=0)  # [2B]

    loss = F.cross_entropy(S_masked, labels)

    # Compute diagnostics
    # Positive pair similarities
    pos_sims = []
    for i in range(B):
        pos_sims.append(S[i, i + B].item())
    pos_sim_mean = np.mean(pos_sims)

    # Max negative similarity
    neg_sims = []
    for i in range(2 * B):
        for j in range(2 * B):
            if i == j or i + B == j or j + B == i:
                continue
            neg_sims.append(S[i, j].item())
    neg_sim_max = max(neg_sims) if neg_sims else 0.0
    neg_sim_mean = np.mean(neg_sims) if neg_sims else 0.0

    stats = {
        'pos_sim_mean': pos_sim_mean,
        'pos_sim_max': max(pos_sims) if pos_sims else 0.0,
        'neg_sim_max': neg_sim_max,
        'neg_sim_mean': neg_sim_mean,
    }

    return loss, pos_sim_mean, neg_sim_max, stats


# --------------------------------------------------------------------
# Metric Tracker
# --------------------------------------------------------------------
class MetricTracker:
    """Tracks metrics over epochs for logging and plotting."""
    def __init__(self):
        self.metrics = defaultdict(list)

    def add(self, epoch, **kwargs):
        self.metrics['epoch'].append(epoch)
        for k, v in kwargs.items():
            self.metrics[k].append(v)

    def get(self, key):
        return self.metrics.get(key, [])

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)

    def load(self, path):
        with open(path, 'r') as f:
            self.metrics = defaultdict(list, json.load(f))


# --------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------
def plot_pretrain_diagnostics(tracker, save_dir):
    """Plot pretraining diagnostics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = tracker.get('epoch')
    if not epochs:
        print("No metrics to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    ax = axes[0, 0]
    loss = tracker.get('loss')
    if loss:
        ax.plot(epochs, loss, 'b-', linewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('NT-Xent Loss')
        ax.set_title('Pretraining Loss')
        ax.grid(True, alpha=0.3)

    # Positive pair cosine similarity
    ax = axes[0, 1]
    pos_sim = tracker.get('pos_sim_mean')
    if pos_sim:
        ax.plot(epochs, pos_sim, 'g-', linewidth=1)
        ax.axhline(y=0.999, color='r', linestyle='--', alpha=0.5, label='Collapse threshold')
        ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, label='No learning')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Positive Pair Cosine Sim')
        ax.set_title('Positive Pair Similarity (Collapse Check)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)

    # Max negative similarity
    ax = axes[1, 0]
    neg_sim_max = tracker.get('neg_sim_max')
    if neg_sim_max:
        ax.plot(epochs, neg_sim_max, 'r-', linewidth=1)
        ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Warning threshold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Max Negative Cosine Sim')
        ax.set_title('Maximum Negative Similarity')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Gradient norm
    ax = axes[1, 1]
    grad_norm = tracker.get('grad_norm')
    if grad_norm:
        ax.plot(epochs, grad_norm, 'purple', linewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    path = os.path.join(save_dir, 'pretrain_diagnostics.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved pretraining diagnostic plot to {path}")


def plot_finetune_curve(val_rmse_list, save_dir):
    """Plot finetuning validation RMSE curve."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(val_rmse_list, 'b-', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation RMSE (normalized)')
    ax.set_title('Finetuning Validation RMSE')
    ax.grid(True, alpha=0.3)

    if val_rmse_list:
        best_idx = np.argmin(val_rmse_list)
        ax.axvline(x=best_idx, color='r', linestyle='--', alpha=0.5,
                   label=f'Best epoch {best_idx + 1}')
        ax.legend()

    path = os.path.join(save_dir, 'finetune_val_rmse.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved finetuning plot to {path}")


# --------------------------------------------------------------------
# Gradient norm computation
# --------------------------------------------------------------------
def compute_grad_norm(model):
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
