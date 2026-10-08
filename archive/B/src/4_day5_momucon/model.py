#!/usr/bin/env python3
"""
MoMuCon Model Architecture for NT-Xent Contrastive Pretraining.

Components:
  - GATEncoder: 4-layer GATv2Conv with LayerNorm, residuals, mean pooling (256-dim output)
  - ProjectionHead: MLP projecting encoder output to NT-Xent space
  - Predictor: MLP for downstream Tg regression
  - MoMuConSiamese: Shared-weight encoder + projection head; NT-Xent loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATEncoder(nn.Module):
    """4-layer GATv2 encoder with LayerNorm and residual connections.

    Architecture:
      - 4 GATv2Conv layers, hidden_dim=256, 4 heads, mean aggregation
      - LayerNorm after each GAT layer
      - Residual connections (with projection when dims change)
      - ReLU activation, dropout=0.1
      - Graph readout: global mean pooling -> 256-dim embedding

    Output dimension: hidden_dim * num_heads (256 * 4 = 1024 with concat)
    But task says "Graph readout: global mean pool -> 256-dim embedding h"
    So we use a final projection: 1024 -> 256 to match spec.
    """

    def __init__(self, input_dim, hidden_dim=256, num_heads=4,
                 num_layers=4, dropout=0.1, edge_dim=None,
                 output_dim=256):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
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

            # Residual projection if dimensions change (first layer)
            if i > 0 and in_dim != out_dim:
                self.residual_proj.append(nn.Linear(in_dim, out_dim, bias=False))
            else:
                self.residual_proj.append(nn.Identity())

        # Project encoder output to 256-dim as per spec
        encoder_inner_dim = hidden_dim * num_heads  # 1024
        self.output_proj = nn.Linear(encoder_inner_dim, output_dim)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """Forward pass producing graph-level embeddings.

        Returns:
            h: [B, 256] graph embeddings if batch provided
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

        # Project to 256-dim
        x = self.output_proj(x)

        return x


class ProjectionHead(nn.Module):
    """Projection head mapping encoder embeddings to NT-Xent space.

    Architecture:
      Linear(256->128), BatchNorm1d, ReLU
      Linear(128->128) -> output z (L2-normalized)
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
      Linear(256->128) + ReLU + Dropout(0.1) + Linear(128->1)
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


def nt_xent_loss(z1, z2, tau=0.1):
    """NT-Xent loss for a pair of normalized embedding matrices.

    z1: [B, D] - embeddings from view 1 (1-RU)
    z2: [B, D] - embeddings from view 2 (3-RU)

    Concatenates z = [z1; z2] -> [2B, D]
    Positive pairs: (i, i+B) and (i+B, i) for i in [0, B)
    Loss: -log(exp(sim[pos]/tau) / sum(exp(sim_all/tau)))
    """
    B = z1.shape[0]
    device = z1.device

    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = (z @ z.T) / tau  # [2B, 2B]

    # Mask self-similarity
    sim_i_j = sim - torch.eye(2 * B, device=device) * 1e9

    # Positive pair similarities
    pos_sim = torch.cat([
        torch.diag(sim, B),      # upper: (0,B), (1,B+1), ...
        torch.diag(sim, -B),     # lower: (B,0), (B+1,1), ...
    ], dim=0)  # [2B]

    loss = (-pos_sim + torch.logsumexp(sim_i_j, dim=1)).mean()
    return loss


class MoMuConSiamese(nn.Module):
    """Siamese GAT for NT-Xent contrastive learning.

    SHARED weights (NO EMA, NO asymmetry):
      - Same encoder + same projection head for both views
      - Explicit collapse prevention via NT-Xent negative pairs

    Loss: NT-Xent with temperature tau.
    """

    def __init__(self, input_dim, hidden_dim=256, num_heads=4,
                 num_layers=4, dropout=0.1, proj_hidden_dim=128,
                 proj_output_dim=128, tau=0.1, edge_dim=None):
        super().__init__()
        self.tau = tau

        self.encoder = GATEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_heads=num_heads, num_layers=num_layers,
            dropout=dropout, edge_dim=edge_dim, output_dim=hidden_dim
        )

        self.projection = ProjectionHead(
            input_dim=hidden_dim, hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim
        )

    def forward(self, batch_1ru, batch_3ru):
        """Forward pass for both views.

        Returns:
            z1, z2: Projected embeddings [B, 128] (L2-normalized)
            h1, h2: Encoder embeddings [B, 256] (for diagnostics)
        """
        edge_attr_1 = batch_1ru.edge_attr if hasattr(batch_1ru, 'edge_attr') and batch_1ru.edge_attr is not None else None
        edge_attr_3 = batch_3ru.edge_attr if hasattr(batch_3ru, 'edge_attr') and batch_3ru.edge_attr is not None else None

        h1 = self.encoder(batch_1ru.x, batch_1ru.edge_index, edge_attr_1, batch_1ru.batch)
        h2 = self.encoder(batch_3ru.x, batch_3ru.edge_index, edge_attr_3, batch_3ru.batch)

        z1 = self.projection(h1)
        z2 = self.projection(h2)

        return z1, z2, h1, h2

    def compute_loss(self, z1, z2):
        return nt_xent_loss(z1, z2, self.tau)

    def encode(self, data):
        """Encode a batch of graphs (for finetuning)."""
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        return self.encoder(data.x, data.edge_index, edge_attr, data.batch)


class FinetuneModel(nn.Module):
    """Encoder + Predictor for downstream Tg regression."""

    def __init__(self, encoder, encoder_output_dim=256, predictor_hidden=128, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.predictor = Predictor(
            input_dim=encoder_output_dim,
            hidden_dim=predictor_hidden,
            dropout=dropout
        )

    def forward(self, data):
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        h = self.encoder(data.x, data.edge_index, edge_attr, data.batch)
        return self.predictor(h)


class LinearProbe(nn.Module):
    """Linear probe: frozen encoder + linear regressor."""

    def __init__(self, encoder, encoder_output_dim=256):
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
