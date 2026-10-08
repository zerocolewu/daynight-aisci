#!/usr/bin/env python3
"""
MoMuCon Model Architecture for NT-Xent Contrastive Pretraining.

Components:
  - GATEncoder: 4-layer GATv2Conv with LayerNorm, residuals, mean pooling
  - ProjectionHead: MLP projecting encoder output to NT-Xent space
  - Predictor: MLP for downstream Tg regression
  - MoMuConSiamese: Wraps encoder + projection head; computes NT-Xent loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATEncoder(nn.Module):
    """4-layer GATv2 encoder with LayerNorm and residual connections.

    Architecture (as specified):
      - 4 GATv2Conv layers, hidden_dim=256, 4 attention heads, mean aggregation
      - LayerNorm after each GAT layer (not BatchNorm)
      - Residual connections
      - ReLU activation, dropout=0.1
      - Graph readout: global mean pooling
    """

    def __init__(self, input_dim, hidden_dim=256, num_heads=4,
                 num_layers=4, dropout=0.1, edge_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.output_dim = hidden_dim * num_heads
        self.dropout_rate = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_proj = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim * num_heads
            out_dim = hidden_dim * num_heads

            self.convs.append(
                GATv2Conv(
                    in_dim, hidden_dim, heads=num_heads,
                    concat=True, dropout=dropout,
                    edge_dim=edge_dim
                )
            )
            self.norms.append(nn.LayerNorm(out_dim))

            # Residual projection if dimensions change
            if i > 0 and in_dim != out_dim:
                self.residual_proj.append(nn.Linear(in_dim, out_dim, bias=False))
            else:
                self.residual_proj.append(nn.Identity())

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """Forward pass producing graph-level embeddings.

        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            batch: Batch assignments [N] for pooling

        Returns:
            Graph-level embeddings [B, hidden_dim] if batch provided,
            else node-level embeddings [N, hidden_dim]
        """
        for i in range(self.num_layers):
            identity = x

            if edge_attr is not None:
                x = self.convs[i](x, edge_index, edge_attr)
            else:
                x = self.convs[i](x, edge_index)

            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            # Residual connection
            if i > 0:
                identity = self.residual_proj[i](identity)
                x = x + identity

        if batch is not None:
            x = global_mean_pool(x, batch)

        return x

    def forward_node(self, x, edge_index, edge_attr=None):
        """Forward pass returning node-level embeddings (no pooling)."""
        for i in range(self.num_layers):
            identity = x

            if edge_attr is not None:
                x = self.convs[i](x, edge_index, edge_attr)
            else:
                x = self.convs[i](x, edge_index)

            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            if i > 0:
                identity = self.residual_proj[i](identity)
                x = x + identity

        return x


class ProjectionHead(nn.Module):
    """Projection head mapping encoder embeddings to NT-Xent space.

    Architecture:
      Linear(256→128), BatchNorm1d, ReLU
      Linear(128→128) → output z (L2-normalized)
    """

    def __init__(self, input_dim=256, hidden_dim=128, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=-1)


class Predictor(nn.Module):
    """Predictor head for downstream Tg regression.

    Architecture:
      Linear(256→128) + ReLU + Dropout(0.1) + Linear(128→1)
    """

    def __init__(self, input_dim=256, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MoMuConSiamese(nn.Module):
    """Siamese GAT for NT-Xent contrastive learning.

    Both branches are IDENTICAL (shared weights):
      - NO student/teacher asymmetry
      - NO EMA
      - NO momentum encoder
      - Same encoder + same projection head for both views

    Loss: NT-Xent with temperature τ.
    """

    def __init__(self, input_dim, hidden_dim=256, num_heads=4,
                 num_layers=4, dropout=0.1, proj_hidden_dim=128,
                 proj_output_dim=128, tau=0.1, edge_dim=None):
        super().__init__()
        self.tau = tau

        # Shared encoder
        self.encoder = GATEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_heads=num_heads, num_layers=num_layers,
            dropout=dropout, edge_dim=edge_dim
        )

        # Shared projection head (encoder output is hidden_dim * num_heads)
        self.projection = ProjectionHead(
            input_dim=hidden_dim * num_heads, hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim
        )

    def forward(self, batch_1ru, batch_3ru):
        """Forward pass for both views.

        Args:
            batch_1ru: PyG Batch of 1-RU graphs
            batch_3ru: PyG Batch of 3-RU graphs

        Returns:
            z1: Projected embeddings for 1-RU graphs [B, proj_output_dim]
            z2: Projected embeddings for 3-RU graphs [B, proj_output_dim]
            h1: Encoder embeddings for 1-RU graphs [B, hidden_dim] (for diagnostics)
            h2: Encoder embeddings for 3-RU graphs [B, hidden_dim] (for diagnostics)
        """
        # Encode 1-RU view
        edge_attr_1 = batch_1ru.edge_attr if hasattr(batch_1ru, 'edge_attr') and batch_1ru.edge_attr is not None else None
        h1 = self.encoder(batch_1ru.x, batch_1ru.edge_index, edge_attr_1, batch_1ru.batch)

        # Encode 3-RU view
        edge_attr_3 = batch_3ru.edge_attr if hasattr(batch_3ru, 'edge_attr') and batch_3ru.edge_attr is not None else None
        h2 = self.encoder(batch_3ru.x, batch_3ru.edge_index, edge_attr_3, batch_3ru.batch)

        # Project
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        return z1, z2, h1, h2

    def compute_loss(self, z1, z2):
        """NT-Xent loss.

        Args:
            z1: Normalized embeddings from view 1 [B, D]
            z2: Normalized embeddings from view 2 [B, D]

        Returns:
            loss: scalar NT-Xent loss
        """
        return nt_xent_loss(z1, z2, self.tau)

    def encode(self, data):
        """Encode a single batch of graphs (for finetuning).

        Args:
            data: PyG Batch with x, edge_index, edge_attr (optional), batch

        Returns:
            h: Encoder embeddings [B, hidden_dim]
        """
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        return self.encoder(data.x, data.edge_index, edge_attr, data.batch)


def nt_xent_loss(z1, z2, tau=0.1):
    """NT-Xent loss for a pair of normalized embedding matrices.

    z1: [B, D]  -- embeddings from view 1 (1-RU)
    z2: [B, D]  -- embeddings from view 2 (3-RU)

    For batch of B polymers: concatenate z = [z1; z2] → [2B, D]
    Positive pair: (i, i+B) and (i+B, i) for i ∈ [0, B)
    All other pairs are negatives.

    Loss = mean over all 2B samples of:
      -log( exp(sim[i, pos_i] / τ) / Σ_{j≠i} exp(sim[i, j] / τ) )
    """
    B = z1.shape[0]
    device = z1.device

    # Concatenate views
    z = torch.cat([z1, z2], dim=0)  # [2B, D]

    # Cosine similarity matrix (z already normalized)
    sim = (z @ z.T) / tau  # [2B, 2B]

    # Mask self-similarity
    sim_i_j = sim - torch.eye(2 * B, device=device) * 1e9

    # Positive pairs: (i, i+B) upper band and (i+B, i) lower band
    pos_sim = torch.cat([
        torch.diag(sim, B),      # (0, B), (1, B+1), ..., (B-1, 2B-1): 1RU_i matches 3RU_i
        torch.diag(sim, -B),     # (B, 0), (B+1, 1), ..., (2B-1, B-1): 3RU_i matches 1RU_i
    ], dim=0)  # [2B]

    # NT-Xent: -log( exp(pos) / sum(exp(all_neg + pos)) )
    loss = (-pos_sim + torch.logsumexp(sim_i_j, dim=1)).mean()

    return loss


class FinetuneModel(nn.Module):
    """Encoder + Predictor for downstream Tg regression."""

    def __init__(self, encoder, encoder_output_dim=1024, predictor_hidden=128, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.predictor = Predictor(
            input_dim=encoder_output_dim,
            hidden_dim=predictor_hidden,
            dropout=dropout
        )

    def forward(self, data):
        """Forward pass.

        Args:
            data: PyG Batch with x, edge_index, edge_attr (optional), batch

        Returns:
            pred: Predicted Tg values [B]
        """
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        h = self.encoder(data.x, data.edge_index, edge_attr, data.batch)
        return self.predictor(h)


class LinearProbe(nn.Module):
    """Linear probe: frozen encoder + linear regressor."""

    def __init__(self, encoder, encoder_output_dim=1024):
        super().__init__()
        self.encoder = encoder
        self.linear = nn.Linear(encoder_output_dim, 1)

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, data):
        with torch.no_grad():
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
            h = self.encoder(data.x, data.edge_index, edge_attr, data.batch)
        return self.linear(h).squeeze(-1)
