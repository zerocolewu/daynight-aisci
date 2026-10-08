#!/usr/bin/env python3
"""
MoMuCon-NT-Xent Model Architecture.

Components:
  - GATEncoder: 4-layer GATConv with BatchNorm1d, residuals, mean pooling
  - ProjectionHead: MLP for NT-Xent projection space
  - Predictor: MLP for downstream Tg regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATEncoder(nn.Module):
    """4-layer GAT encoder with BatchNorm1d and residual connections.

    Architecture:
      Layer 1: GATConv -> BatchNorm1d -> ReLU
      Layers 2-3: GATConv -> BatchNorm1d -> ReLU -> residual
      Layer 4: GATConv -> BatchNorm1d
      Readout: global_mean_pool -> LayerNorm
      Output: 256-dim graph embedding
    """

    def __init__(self, in_dim=42, hidden=256, num_layers=4, heads=4,
                 dropout=0.1, edge_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden = hidden
        self.heads = heads
        self.dropout_rate = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                in_dim_layer = in_dim
            else:
                in_dim_layer = hidden

            self.convs.append(
                GATConv(
                    in_dim_layer, hidden // heads, heads=heads,
                    concat=True, dropout=dropout,
                    edge_dim=edge_dim
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden))

            if i >= 2:  # Residual connections at layers 2 and 3 (0-indexed: 1,2)
                self.residuals.append(nn.Linear(hidden, hidden, bias=False))
            else:
                self.residuals.append(nn.Identity())

        # Final LayerNorm after pooling
        self.final_norm = nn.LayerNorm(hidden)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for i in range(self.num_layers):
            identity = x

            if edge_attr is not None:
                x = self.convs[i](x, edge_index, edge_attr)
            else:
                x = self.convs[i](x, edge_index)

            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            # Residual connection at layers 2-3 (indices 1, 2)
            if i >= 1 and i <= 2:
                identity = self.residuals[i](identity)
                x = x + identity

        # Global mean pooling
        if batch is not None:
            x = global_mean_pool(x, batch)

        # Final LayerNorm
        x = self.final_norm(x)

        return x


class ProjectionHead(nn.Module):
    """Projection head for NT-Xent contrastive space.

    Architecture:
      Linear(256 -> 128) + BatchNorm1d + ReLU
      Linear(128 -> 128)
    """

    def __init__(self, in_dim=256, hidden_dim=128, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    """Predictor head for downstream Tg regression.

    Architecture:
      Linear(256 -> 128) + BatchNorm1d + ReLU + Dropout(0.1) + Linear(128 -> 1)
    """

    def __init__(self, in_dim=256, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
