"""
Model architectures for Day 1 Experiment 2:
1. EdgeAwareGINConv - baseline GIN with edge awareness (from Experiment 1)
2. EdgeAwareGINAttentionConv - GIN-GAT Hybrid: GIN with GAT-style attention aggregation
3. GINModel - Standard GIN model using EdgeAwareGINConv (for VirtualNode variants)
4. GINGATModel - GIN-GAT Hybrid model using EdgeAwareGINAttentionConv
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.utils import softmax
import math


class EdgeAwareGINConv(MessagePassing):
    """
    GIN convolution with edge feature awareness (from Experiment 1).
    Projects edge features and adds them to neighbor messages before GIN MLP aggregation.
    """

    def __init__(self, nn_func, edge_dim, hidden_dim, train_eps=True, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.nn = nn_func
        self.edge_proj = nn.Linear(edge_dim, hidden_dim, bias=False)
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([0.0]))
        else:
            self.register_buffer('eps', torch.Tensor([0.0]))

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out + (1 + self.eps) * x

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            edge_feat = self.edge_proj(edge_attr)
            return x_j + edge_feat
        return x_j

    def update(self, aggr_out):
        return self.nn(aggr_out)


class EdgeAwareGINAttentionConv(MessagePassing):
    """
    GIN-GAT Hybrid convolution: Combines GIN's MLP expressive power with
    GAT-style attention-weighted aggregation.

    Key idea:
    - Standard GIN: h_v' = MLP((1+eps)*h_v + SUM_{u in N(v)} h_u)
    - This:         h_v' = MLP((1+eps)*h_v + SUM_{u in N(v)} a_uv * h_u)
    where a_uv = softmax(LeakyReLU(W_attn * [h_v || h_u || W_edge * e_uv]))

    Single-head attention to keep it simple and comparable.
    """

    def __init__(self, nn_func, edge_dim, hidden_dim, train_eps=True, negative_slope=0.2, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.nn = nn_func
        self.edge_proj = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.negative_slope = negative_slope

        # Attention parameters: project concatenated [h_v || h_u || edge_feat] -> scalar
        self.attn_linear = nn.Linear(3 * hidden_dim, 1, bias=False)

        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([0.0]))
        else:
            self.register_buffer('eps', torch.Tensor([0.0]))

        # Store attention weights for diagnostic analysis
        self._last_attention_weights = None

    def forward(self, x, edge_index, edge_attr):
        # Compute attention-weighted messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # GIN update
        return out + (1 + self.eps) * x

    def message(self, x_i, x_j, edge_attr, index, ptr):
        # x_i: target node features [num_edges, hidden_dim]
        # x_j: source node features [num_edges, hidden_dim]
        # edge_attr: edge features [num_edges, edge_dim]

        # Project edge features
        if edge_attr is not None:
            edge_feat = self.edge_proj(edge_attr)
        else:
            edge_feat = torch.zeros_like(x_j)

        # Compute attention scores: W_attn * [h_v || h_u || W_edge * e_uv]
        concat_features = torch.cat([x_i, x_j, edge_feat], dim=-1)
        scores = self.attn_linear(concat_features)
        scores = torch.nn.functional.leaky_relu(scores, negative_slope=self.negative_slope)

        # Softmax over neighbors (per target node)
        attn_weights = softmax(scores, index, ptr=ptr)
        self._last_attention_weights = attn_weights.detach()

        # Apply attention weights to neighbor features
        return attn_weights * x_j

    def update(self, aggr_out):
        return self.nn(aggr_out)

    def get_attention_weights(self):
        """Return the last computed attention weights for diagnostic analysis."""
        return self._last_attention_weights


class GINModel(nn.Module):
    """
    Standard GIN model with EdgeAwareGINConv layers.
    Used for Models 2 & 4 (VirtualNode variants).
    Architecture: 3 layers, hidden_dim=256, mean pooling + 2-layer MLP head.
    For virtual node variants, pool excludes virtual node (last node).
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_layers=3,
                 has_virtual_node=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.has_virtual_node = has_virtual_node

        # Input projection
        self.node_embedding = nn.Linear(node_dim, hidden_dim)

        # GIN layers
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

        # MLP prediction head: 256 -> 256 -> 1
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch, num_nodes_per_graph=None):
        """
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch assignment [num_nodes]
            num_nodes_per_graph: optional list of num real nodes per graph (for VN exclusion)
        """
        h = self.node_embedding(x)

        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_attr)
            h = self.batch_norms[i](h)
            h = torch.relu(h)

        if self.has_virtual_node and num_nodes_per_graph is not None:
            # Pool only real nodes, exclude virtual node (last node in each graph)
            real_node_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
            offset = 0
            for n_nodes in num_nodes_per_graph:
                # Virtual node is at index offset + n_nodes (last position)
                real_node_mask[offset + n_nodes] = False
                offset += n_nodes + 1  # +1 for virtual node

            h_real = h[real_node_mask]
            batch_real = batch[real_node_mask]
            h_graph = global_mean_pool(h_real, batch_real)
        else:
            h_graph = global_mean_pool(h, batch)

        out = self.mlp_head(h_graph)
        return out


class GINGATModel(nn.Module):
    """
    GIN-GAT Hybrid model with EdgeAwareGINAttentionConv layers.
    Used for Models 1 & 3 (attention variants on trimer and cyclized trimer).
    Architecture: 3 layers, hidden_dim=256, mean pooling + 2-layer MLP head.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_layers=3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.node_embedding = nn.Linear(node_dim, hidden_dim)

        # GIN-GAT attention layers
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
            conv = EdgeAwareGINAttentionConv(
                nn_func=mlp,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                train_eps=True,
            )
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # MLP prediction head: 256 -> 256 -> 1
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch assignment [num_nodes]
        """
        h = self.node_embedding(x)

        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_attr)
            h = self.batch_norms[i](h)
            h = torch.relu(h)

        h_graph = global_mean_pool(h, batch)
        out = self.mlp_head(h_graph)
        return out

    def get_attention_weights(self):
        """Return attention weights from all layers for diagnostic analysis."""
        return [conv.get_attention_weights() for conv in self.convs]


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
