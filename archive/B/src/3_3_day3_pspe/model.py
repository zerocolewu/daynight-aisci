"""
GAT model for PSPE experiments — matches Day 2's GATStaticRWPE architecture EXACTLY.

Architecture:
- 4 layers of GATConv (NOT GATv2Conv)
- hidden_dim=256, heads=4 (concat mode)
- ELU activation in encoder layers (NOT ReLU)
- 2-layer MLP predictor: 256->128->1 with ReLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATPredictor(nn.Module):
    """4-layer GATConv with ELU activation + 2-layer predictor.

    Matches Day 2's GATStaticRWPE architecture exactly.

    Args:
        in_dim: Input node feature dimension (79 for baseline, 80 for PSPE-dce/mi, 81 for combined)
        edge_dim: Edge feature dimension (10 for rich features)
        hidden_dim: Hidden dimension (256)
        heads: Number of attention heads (4)
        dropout: Dropout rate (0.1)
    """

    def __init__(
        self,
        in_dim: int = 79,
        edge_dim: int = 10,
        hidden_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % heads == 0
        out_per_head = hidden_dim // heads

        self.dropout = dropout
        self.hidden_dim = hidden_dim

        use_edge = edge_dim > 0

        self.conv1 = GATConv(
            in_dim, out_per_head, heads=heads,
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

        # Predictor: matches Day 2 exactly
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        # ELU activation in encoder (matches Day 2)
        x = F.dropout(F.elu(self.conv1(x, edge_index, edge_attr=edge_attr)), p=self.dropout, training=self.training)
        x = F.dropout(F.elu(self.conv2(x, edge_index, edge_attr=edge_attr)), p=self.dropout, training=self.training)
        x = F.dropout(F.elu(self.conv3(x, edge_index, edge_attr=edge_attr)), p=self.dropout, training=self.training)
        x = F.dropout(F.elu(self.conv4(x, edge_index, edge_attr=edge_attr)), p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.predictor(x).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
