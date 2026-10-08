"""
GRIN model implementation with GIN backbone, max aggregation, and L1 sparsity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import List, Optional, Tuple


class MaxGINConv(MessagePassing):
    """
    GIN convolution with MAX aggregation instead of sum/mean.
    This implements the GRIN insight that max aggregation is size-invariant.
    """

    def __init__(self, nn: torch.nn.Module, **kwargs):
        super().__init__(aggr='max', **kwargs)
        self.nn = nn

    def forward(self, x, edge_index):
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Message passing with max aggregation
        out = self.propagate(edge_index, x=x)

        # Apply neural network
        out = self.nn(out)

        return out

    def message(self, x_j):
        return x_j


class GINMaxAggregation(nn.Module):
    """
    GIN model with max aggregation for polymer property prediction.
    """

    def __init__(self, input_dim, hidden_dim=256, num_layers=5, dropout=0.1,
                 use_max_aggregation=True, readout='mean'):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_max_aggregation = use_max_aggregation
        self.readout = readout

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            # MLP for GIN
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )

            if use_max_aggregation:
                conv = MaxGINConv(mlp)
            else:
                # Standard GIN with sum aggregation
                conv = GINConv(mlp, train_eps=True)

            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),  # Use LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Store parameters for L1 regularization
        self.message_params = []
        self.update_params = []

        for conv in self.convs:
            if isinstance(conv, MaxGINConv):
                # For MaxGIN, the nn contains update parameters
                for param in conv.nn.parameters():
                    self.update_params.append(param)
            else:
                # For standard GIN
                for param in conv.nn.parameters():
                    self.update_params.append(param)

    def forward(self, x, edge_index, batch):
        # Ensure float type
        x = x.float()

        # Input projection
        x = self.input_proj(x)

        # GIN layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            identity = x
            x = conv(x, edge_index)
            x = bn(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Readout
        if self.readout == 'mean':
            x = global_mean_pool(x, batch)
        elif self.readout == 'max':
            x = global_max_pool(x, batch)
        else:
            # Use both
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=-1)
            # Adjust head input size
            if not hasattr(self, '_adjusted_head'):
                self.head[0] = nn.Linear(x.size(-1), self.head[0].out_features).to(x.device)
                self._adjusted_head = True

        # Output
        out = self.head(x)

        return out

    def get_l1_penalty(self):
        """Calculate L1 penalty on message and update parameters."""
        l1_loss = 0.0

        for conv in self.convs:
            if isinstance(conv, MaxGINConv):
                for param in conv.nn.parameters():
                    l1_loss += torch.sum(torch.abs(param))
            else:
                for param in conv.nn.parameters():
                    l1_loss += torch.sum(torch.abs(param))

        return l1_loss


class GINEncoder(nn.Module):
    """
    Alternative: Standard GIN encoder for baseline comparison.
    """

    def __init__(self, input_dim, hidden_dim=256, num_layers=5, dropout=0.1):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GIN layers with sum aggregation
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            conv = GINConv(mlp, train_eps=True)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),  # Use LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.input_proj(x)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        out = self.head(x)

        return out


def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model
    from torch_geometric.data import Data, Batch

    # Create dummy data
    x = torch.randn(100, 9)  # 100 nodes, 9 features
    edge_index = torch.randint(0, 100, (2, 200))
    batch = torch.zeros(100, dtype=torch.long)

    # Test with max aggregation
    model_max = GINMaxAggregation(input_dim=9, hidden_dim=256, num_layers=5, use_max_aggregation=True)
    print(f"Model with max aggregation: {count_parameters(model_max):,} parameters")

    out_max = model_max(x, edge_index, batch)
    print(f"Output shape (max): {out_max.shape}")

    # Test with mean aggregation (baseline)
    model_mean = GINMaxAggregation(input_dim=9, hidden_dim=256, num_layers=5, use_max_aggregation=False)
    print(f"Model with mean aggregation: {count_parameters(model_mean):,} parameters")

    out_mean = model_mean(x, edge_index, batch)
    print(f"Output shape (mean): {out_mean.shape}")

    # Test L1 penalty
    l1 = model_max.get_l1_penalty()
    print(f"L1 penalty: {l1.item():.4f}")
