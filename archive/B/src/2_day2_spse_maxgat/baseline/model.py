"""
Baseline model: GAT with Rich features + 3-RU.
4 GATConv layers, hidden_dim=256, 4 attention heads, mean aggregation.
2-layer MLP predictor (256→128→1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATBaseline(nn.Module):
    """4-layer GAT with global mean pooling for graph regression.

    Args:
        node_dim: Input node feature dimension (47 for rich)
        edge_dim: Input edge feature dimension (10 for rich)
        hidden_dim: Hidden dimension (default 256)
        heads: Number of attention heads (default 4)
        dropout: Dropout probability (default 0.1)
        out_dim: Output dimension (default 1)
    """

    def __init__(
        self,
        node_dim: int = 47,
        edge_dim: int = 10,
        hidden_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
        out_dim: int = 1,
    ):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"

        self.dropout = dropout
        out_per_head = hidden_dim // heads  # 64 when hidden_dim=256, heads=4

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

        # Layer 4
        x = self.conv4(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global mean pool
        x = global_mean_pool(x, batch)

        return self.predictor(x).squeeze(-1)
