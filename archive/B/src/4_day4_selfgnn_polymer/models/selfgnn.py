#!/usr/bin/env python3
"""
SelfGNN Siamese Architecture for Graph-Level SSL Pretraining.

Key components:
  - Student encoder: GAT backbone + prediction MLP g
  - Teacher encoder: GAT backbone (no MLP), EMA-updated
  - Augmentations: Split + Standardize
  - Loss: MSE cosine similarity = 2 - 2*cos(z_student, z_teacher)
  - BatchNorm provides implicit negatives (no NT-Xent, no negative pairs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from models.gat_encoder import GATEncoder


# --------------------------------------------------------------------
# Prediction MLP (student only)
# --------------------------------------------------------------------
class PredictionMLP(nn.Module):
    """g: Linear(256→128) + BatchNorm1d + ReLU + Linear(128→256)"""

    def __init__(self, input_dim=256, hidden_dim=128, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------
# Augmentations
# --------------------------------------------------------------------
def split_augmentation(x, view=0):
    """Split augmentation: randomly permute feature dimensions, split in half.

    Args:
        x: Node features [N, F]
        view: 0 for first half, 1 for second half

    Returns:
        x_aug: Augmented features [N, F] with unseen half zero-padded
    """
    N, F = x.shape
    device = x.device

    # Random permutation of feature dimensions
    perm = torch.randperm(F, device=device)

    # Split into two halves
    half = F // 2
    if view == 0:
        seen_indices = perm[:half]
    else:
        seen_indices = perm[half:2*half]

    # Create zero-padded output
    x_aug = torch.zeros_like(x)
    x_aug[:, seen_indices] = x[:, seen_indices]

    return x_aug


def standardize_augmentation(x):
    """Per-graph z-score normalization.

    For each feature dimension across all nodes: (x - mean) / (std + eps)

    Args:
        x: Node features [N, F]

    Returns:
        x_std: Standardized features [N, F]
    """
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    eps = 1e-5
    x_std = (x - mean) / (std + eps)
    return x_std


# --------------------------------------------------------------------
# SelfGNN Model
# --------------------------------------------------------------------
class SelfGNN(nn.Module):
    """SelfGNN: Siamese GAT with BatchNorm implicit negatives.

    Training flow:
      1. Create two views of each graph via Split augmentation
      2. Apply Standardize to teacher view only
      3. Student encodes view1 → prediction MLP → z_student
      4. Teacher encodes view2 (EMA, no grad) → z_teacher
      5. Loss = 2 - 2*cos(z_student, z_teacher)
    """

    def __init__(self, input_dim=10, hidden_dim=256, num_heads=4,
                 num_layers=4, dropout=0.1, tau=0.999,
                 use_batchnorm=True, use_standardize=True, use_split=True):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_heads: Attention heads
            num_layers: Number of GAT layers
            dropout: Dropout rate
            tau: EMA decay rate for teacher update
            use_batchnorm: Whether to use BatchNorm (False for Ablation H)
            use_standardize: Whether to apply Standardize augmentation
            use_split: Whether to apply Split augmentation
        """
        super().__init__()
        self.tau = tau
        self.use_standardize = use_standardize
        self.use_split = use_split

        # Student encoder
        self.student_encoder = GATEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_heads=num_heads, num_layers=num_layers,
            dropout=dropout, use_batchnorm=use_batchnorm
        )

        # Prediction MLP (student only)
        self.predictor = PredictionMLP(
            input_dim=hidden_dim, hidden_dim=128, output_dim=hidden_dim
        )

        # Teacher encoder (EMA, no grad)
        self.teacher_encoder = GATEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_heads=num_heads, num_layers=num_layers,
            dropout=dropout, use_batchnorm=use_batchnorm
        )

        # Initialize teacher with student weights
        self._copy_weights(self.teacher_encoder, self.student_encoder)

        # Freeze teacher
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False

    def _copy_weights(self, target, source):
        """Copy weights from source to target."""
        target.load_state_dict(source.state_dict())

    @torch.no_grad()
    def update_teacher(self):
        """EMA update teacher: θ_t = τ·θ_t + (1-τ)·θ_s"""
        for t_param, s_param in zip(
            self.teacher_encoder.parameters(),
            self.student_encoder.parameters()
        ):
            t_param.data = self.tau * t_param.data + (1.0 - self.tau) * s_param.data

    def forward(self, data):
        """Forward pass for a batch of graphs.

        Args:
            data: PyG Batch object with x [N, F], edge_index [2, E], batch [N]

        Returns:
            z_student: Student graph embeddings [B, hidden_dim]
            z_teacher: Teacher graph embeddings [B, hidden_dim]
        """
        batch = data.batch
        edge_index = data.edge_index
        x = data.x
        N = x.size(0)

        # ---- Student view ----
        if self.use_split:
            x_student = split_augmentation(x, view=0)
        else:
            x_student = x.clone()

        z_node = self.student_encoder(x_student, edge_index)  # [N, 256]
        z_student = self.predictor(z_node)  # [N, 256]

        # Global mean pooling (per graph)
        z_student = self._global_mean_pool(z_student, batch)  # [B, 256]

        # ---- Teacher view (no grad) ----
        with torch.no_grad():
            if self.use_split:
                x_teacher = split_augmentation(x, view=1)
            else:
                x_teacher = x.clone()

            if self.use_standardize:
                x_teacher = standardize_augmentation(x_teacher)

            z_teacher = self.teacher_encoder(x_teacher, edge_index)  # [N, 256]
            z_teacher = self._global_mean_pool(z_teacher, batch)  # [B, 256]

        return z_student, z_teacher

    def _global_mean_pool(self, x, batch):
        """Mean pool node embeddings to graph embeddings."""
        unique_batches = torch.unique(batch)
        pooled = []
        for b in unique_batches:
            mask = (batch == b)
            pooled.append(x[mask].mean(dim=0))
        return torch.stack(pooled)

    def compute_loss(self, z_student, z_teacher):
        """MSE cosine similarity loss: 2 - 2*cos(z_s, z_t).

        Equivalent to ||z_s_norm - z_t_norm||^2.
        """
        z_s_norm = F.normalize(z_student, dim=-1)
        z_t_norm = F.normalize(z_teacher, dim=-1)
        cos_sim = (z_s_norm * z_t_norm).sum(dim=-1)
        loss = (2.0 - 2.0 * cos_sim).mean()
        return loss

    @torch.no_grad()
    def get_teacher_embedding(self, data):
        """Get teacher embedding for diagnostics (no augmentations, no grad)."""
        z = self.teacher_encoder(data.x, data.edge_index)
        z = self._global_mean_pool(z, data.batch)
        return F.normalize(z, dim=-1)

    def get_student_encoder(self):
        """Return student GAT encoder (discarding prediction MLP) for finetuning."""
        return self.student_encoder
