"""
GAT model for polymer Tg regression.
3 GATConv layers, global mean pooling, dropout 0.1, no batch norm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATModel(nn.Module):
    """3-layer GAT with global mean pooling for graph regression.

    Args:
        node_dim: Input node feature dimension
        edge_dim: Input edge feature dimension (0 = no edge features)
        hidden_dim: Hidden dimension (default 128)
        heads: Number of attention heads (default 4)
        dropout: Dropout probability (default 0.1)
        out_dim: Output dimension (default 1 for scalar regression)
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
        out_dim: int = 1,
    ):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"

        self.dropout = dropout
        out_per_head = hidden_dim // heads

        use_edge = edge_dim > 0

        self.conv1 = GATConv(
            node_dim, out_per_head, heads=heads,
            concat=True, dropout=dropout,
            edge_dim=edge_dim if use_edge else None,
        )
        self.conv2 = GATConv(
            hidden_dim, out_per_head, heads=heads,
            concat=True, dropout=dropout,
            edge_dim=edge_dim if use_edge else None,
        )
        self.conv3 = GATConv(
            hidden_dim, out_per_head, heads=heads,
            concat=True, dropout=dropout,
            edge_dim=edge_dim if use_edge else None,
        )

        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # Layer 1
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global mean pool
        x = global_mean_pool(x, batch)

        return self.head(x).squeeze(-1)
