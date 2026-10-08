"""
Treatment A model: GAT + Rich features + 3-RU + SPSE edge encodings.
Same as baseline but with extended edge features (10 rich + 4 SPSE = 14 dims).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATSPSE(nn.Module):
    """4-layer GAT with SPSE edge encodings and global mean pooling.

    Args:
        node_dim: Input node feature dimension (47 for rich)
        edge_dim: Input edge feature dimension (14 = 10 rich + 4 SPSE)
        hidden_dim: Hidden dimension (default 256)
        heads: Number of attention heads (default 4)
        dropout: Dropout probability (default 0.1)
        out_dim: Output dimension (default 1)
    """

    def __init__(
        self,
        node_dim: int = 47,
        edge_dim: int = 14,
        hidden_dim: int = 256,
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
        self.conv4 = GATConv(
            hidden_dim, out_per_head, heads=heads,
            concat=True, dropout=dropout,
            edge_dim=edge_dim if use_edge else None,
        )

        # 2-layer MLP predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv4(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.predictor(x).squeeze(-1)
