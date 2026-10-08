"""
GAT model variants for GRIN + ELENE readout experiment.

Condition A: GAT + mean readout (baseline)
Condition B: GAT + max readout (no L1 sparsity)
Condition C: GAT + max readout + L1 sparsity (GRIN-style)
Condition D: GAT + mean readout + ELENE edge encodings
Condition E: GAT + max readout + L1 sparsity + ELENE edge encodings (full combo)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class GATModelBase(nn.Module):
    """3-layer GAT with configurable readout and optional L1 sparsity.

    Args:
        node_dim: Input node feature dimension
        edge_dim: Input edge feature dimension (0 = no edge features)
        hidden_dim: Hidden dimension (default 128)
        heads: Number of attention heads (default 4)
        dropout: Dropout probability (default 0.1)
        out_dim: Output dimension (default 1)
        readout: 'mean' or 'max'
        apply_l1: If True, expose conv3 params for L1 regularization
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
        out_dim: int = 1,
        readout: str = 'mean',
        apply_l1: bool = False,
    ):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        assert readout in ('mean', 'max')

        self.dropout = dropout
        self.readout = readout
        self.apply_l1 = apply_l1
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

        # Global readout
        if self.readout == 'mean':
            x = global_mean_pool(x, batch)
        else:
            x = global_max_pool(x, batch)

        return self.head(x).squeeze(-1)

    def get_l1_penalty(self):
        """Compute L1 norm of conv3 parameters for GRIN-style sparsity."""
        if not self.apply_l1:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        l1 = torch.tensor(0.0, device=next(self.parameters()).device)
        for name, param in self.conv3.named_parameters():
            if param.requires_grad:
                l1 = l1 + param.abs().sum()
        return l1

    def get_sparsity_fraction(self, threshold: float = 1e-5):
        """Fraction of conv3 weights with absolute value < threshold."""
        if not self.apply_l1:
            return 0.0
        total = 0
        near_zero = 0
        for name, param in self.conv3.named_parameters():
            if param.requires_grad:
                total += param.numel()
                near_zero += (param.abs() < threshold).sum().item()
        return near_zero / total if total > 0 else 0.0


def create_model(condition: str, node_dim: int, edge_dim: int, **kwargs) -> GATModelBase:
    """Factory function to create the appropriate model for each condition.

    Args:
        condition: 'A', 'B', 'C', 'D', or 'E'
        node_dim: Input node feature dimension
        edge_dim: Input edge feature dimension
    """
    conditions = {
        'A': {'readout': 'mean', 'apply_l1': False},
        'B': {'readout': 'max', 'apply_l1': False},
        'C': {'readout': 'max', 'apply_l1': True},
        'D': {'readout': 'mean', 'apply_l1': False},
        'E': {'readout': 'max', 'apply_l1': True},
    }
    if condition not in conditions:
        raise ValueError(f"Unknown condition: {condition}. Must be one of {list(conditions.keys())}")

    cfg = conditions[condition]
    return GATModelBase(
        node_dim=node_dim,
        edge_dim=edge_dim,
        readout=cfg['readout'],
        apply_l1=cfg['apply_l1'],
        **kwargs,
    )
