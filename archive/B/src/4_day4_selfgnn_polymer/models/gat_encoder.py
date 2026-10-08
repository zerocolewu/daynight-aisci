#!/usr/bin/env python3
"""
GAT Encoder Backbone for SelfGNN-Polymer.
Uses GATv2Conv for dynamic attention, BatchNorm1d, ReLU, Dropout.
Mean pooling to produce graph-level embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm, global_mean_pool


class GATEncoder(nn.Module):
    """4-layer GATv2 encoder with BatchNorm and mean pooling.

    Architecture:
      - 4 GATv2Conv layers, hidden_dim=256, heads=4, mean aggregation
      - BatchNorm1d after each GAT layer
      - ReLU activation
      - Dropout 0.1
      - Graph embedding: mean pooling of final layer node embeddings → 256-dim

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension (default 256)
        num_heads: Number of attention heads (default 4)
        num_layers: Number of GAT layers (default 4)
        dropout: Dropout rate (default 0.1)
        use_batchnorm: Whether to use BatchNorm (default True)
                       Setting False is for Ablation H (remove BN)
    """

    def __init__(self, input_dim=42, hidden_dim=256, num_heads=4,
                 num_layers=4, dropout=0.1, use_batchnorm=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.output_dim = hidden_dim
        self.use_batchnorm = use_batchnorm

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim * num_heads
            concat = (i < num_layers - 1)
            out_dim = hidden_dim * num_heads if concat else hidden_dim

            self.convs.append(
                GATv2Conv(in_dim, hidden_dim, heads=num_heads,
                          concat=concat, dropout=dropout)
            )

            if use_batchnorm:
                self.bns.append(BatchNorm(out_dim))
            else:
                self.bns.append(nn.Identity())

        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        """Forward pass.

        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            batch: Batch assignments [N] (for pooling)

        Returns:
            If batch is not None: graph-level embeddings [B, hidden_dim]
            If batch is None: node-level embeddings [N, hidden_dim]
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if batch is not None:
            x = global_mean_pool(x, batch)  # [B, hidden_dim]

        return x

    def forward_node(self, x, edge_index):
        """Forward pass returning node-level embeddings only (for downstream use)."""
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATEncoderWithProjection(nn.Module):
    """GAT encoder variant for ablation compatibility with GATConv.

    Uses GATConv (not GATv2) with residual connections, matching Day 3 baseline.
    """

    def __init__(self, input_dim=42, hidden_dim=256, num_heads=4,
                 num_layers=4, dropout=0.1):
        from torch_geometric.nn import GATConv
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.residual_proj = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim * num_heads
            concat = (i < num_layers - 1)
            out_dim = hidden_dim * num_heads if concat else hidden_dim

            self.convs.append(
                GATConv(in_dim, hidden_dim, heads=num_heads, concat=concat, dropout=dropout)
            )
            self.bns.append(BatchNorm(out_dim))

            if i > 0 and in_dim != out_dim:
                self.residual_proj.append(nn.Linear(in_dim, out_dim, bias=False))
            else:
                self.residual_proj.append(nn.Identity())

        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        for i in range(self.num_layers):
            identity = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i > 0:
                identity = self.residual_proj[i](identity)
                x = x + identity

        if batch is not None:
            x = global_mean_pool(x, batch)
        return x

    def forward_node(self, x, edge_index):
        for i in range(self.num_layers):
            identity = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i > 0:
                identity = self.residual_proj[i](identity)
                x = x + identity
        return x
