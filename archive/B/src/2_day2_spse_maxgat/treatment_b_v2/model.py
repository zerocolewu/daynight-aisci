"""
Iterated Treatment B v2: Residual MaxGATConv with input gating.

Addresses the variance collapse observed in Treatment B:
- Max aggregation caused ~0.4% variance retention by layer 4
- Residual connections preserve node's own features through each layer
- Learnable gate parameter controls blend of residual vs aggregated signal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import scatter


class ResidualMaxGATConv(nn.Module):
    """GATConv with element-wise max aggregation AND residual connection.

    Wraps GATConv and adds a residual: output = (1-gate)*attended + gate*residual_proj(input)

    Max aggregation is implemented by overriding GATConv's aggregate().
    The residual preserves node identity through layers, preventing variance collapse.
    """

    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0, edge_dim=None, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat

        # Inner GATConv that uses max aggregation
        self.gat = _MaxAggrGATConv(
            in_channels, out_channels, heads=heads, concat=concat,
            negative_slope=negative_slope, dropout=dropout,
            edge_dim=edge_dim, bias=bias,
        )

        # Output dimension
        if concat:
            out_dim_total = heads * out_channels
        else:
            out_dim_total = out_channels

        # Residual projection (for dimension mismatch)
        if in_channels != out_dim_total:
            self.residual_proj = nn.Linear(in_channels, out_dim_total, bias=False)
        else:
            self.residual_proj = nn.Identity()

        # Learnable residual gate
        self.residual_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):
        x_input = x
        out = self.gat(x, edge_index, edge_attr=edge_attr,
                       return_attention_weights=return_attention_weights)

        # Residual connection
        gate = torch.sigmoid(self.residual_gate)
        out = (1 - gate) * out + gate * self.residual_proj(x_input)

        return out


class _MaxAggrGATConv(GATConv):
    """GATConv subclass that uses max aggregation instead of sum."""

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='max')


class GATResidualMax(nn.Module):
    """4-layer GAT with residual max aggregation and SPSE edge encodings.

    Args:
        node_dim: Input node feature dimension (47)
        edge_dim: Edge feature dimension (14 = 10 rich + 4 SPSE)
        hidden_dim: Hidden dimension (256)
        heads: Number of attention heads (4)
        dropout: Dropout probability (0.1)
        out_dim: Output dimension (1)
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

        self.conv1 = ResidualMaxGATConv(
            node_dim, out_per_head, heads=heads,
            concat=True, dropout=dropout,
            edge_dim=edge_dim if use_edge else None,
        )
        self.conv2 = ResidualMaxGATConv(
            hidden_dim, out_per_head, heads=heads,
            concat=True, dropout=dropout,
            edge_dim=edge_dim if use_edge else None,
        )
        self.conv3 = ResidualMaxGATConv(
            hidden_dim, out_per_head, heads=heads,
            concat=True, dropout=dropout,
            edge_dim=edge_dim if use_edge else None,
        )
        self.conv4 = ResidualMaxGATConv(
            hidden_dim, out_per_head, heads=heads,
            concat=True, dropout=dropout,
            edge_dim=edge_dim if use_edge else None,
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.conv2(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.conv3(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.conv4(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.predictor(x).squeeze(-1)
