"""
GIN (Graph Isomorphism Network) model for molecular property prediction.
Fixed architecture: 3 GIN layers, hidden_dim=256, mean pooling + MLP head.

Note: Standard GINConv in PyG does not support edge features directly.
We use NNConv-style edge-aware update: incorporate edge features via a linear
projection that modulates neighbor messages before GIN aggregation.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool, MessagePassing
from torch_geometric.utils import scatter


class EdgeAwareGINConv(MessagePassing):
    """
    A GIN-like convolution that incorporates edge features.
    Applies a linear transform to edge features and adds them to neighbor representations
    before the GIN MLP aggregation.
    """

    def __init__(self, nn_func, edge_dim, hidden_dim, train_eps=True, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.nn = nn_func
        self.edge_proj = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.initial_eps = 0.0
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([0.0]))
        else:
            self.register_buffer('eps', torch.Tensor([0.0]))

    def forward(self, x, edge_index, edge_attr):
        # Message passing: neighbors + edge features
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # GIN update: (1 + eps) * x_i + aggregated_neighbors
        return out + (1 + self.eps) * x

    def message(self, x_j, edge_attr):
        # Project edge features and add to neighbor node features
        if edge_attr is not None:
            edge_feat = self.edge_proj(edge_attr)
            return x_j + edge_feat
        return x_j

    def update(self, aggr_out):
        # Apply the MLP
        return self.nn(aggr_out)


class GINModel(nn.Module):
    """
    GIN model with 3 GINConv layers, global mean pooling, and MLP regression head.
    Uses edge-aware GIN convolution to incorporate bond features.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_layers=3):
        super(GINModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection (node features -> hidden_dim)
        self.node_embedding = nn.Linear(node_dim, hidden_dim)

        # GIN layers with edge awareness
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
            conv = EdgeAwareGINConv(
                nn_func=mlp,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                train_eps=True,
            )
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # MLP prediction head
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch assignment [num_nodes]
        Returns:
            Prediction [batch_size, 1]
        """
        # Project node features
        h = self.node_embedding(x)

        # GIN layers
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_attr)
            h = self.batch_norms[i](h)
            h = torch.relu(h)

        # Global pooling
        h = global_mean_pool(h, batch)

        # MLP head
        out = self.mlp_head(h)

        return out
