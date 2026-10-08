"""
Model architectures for BlochSpec experiment.

1. EdgeAwareGINConv - same as Day 1 baseline
2. BlochSpecGIN - GIN on trimer + Bloch spectral features in prediction head
3. PureSpectralMLP - MLP on spectral features only (ablation e)
4. BlochSpecGINRandom - GIN + random features (ablation f)
5. BlochSpecGINUnweighted - GIN + unweighted Laplacian features (ablation g)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, MessagePassing
from torch_geometric.utils import softmax


class EdgeAwareGINConv(MessagePassing):
    """
    GIN convolution with edge feature awareness (same as Day 1 baseline).
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


class BlochSpecGIN(nn.Module):
    """
    BlochSpec: GIN on trimer + Bloch spectral features concatenated in prediction head.

    GIN: 3 layers, hidden_dim=256, global_add_pool → h_gnn ∈ R^256
    Prediction head: MLP(h_gnn ⊕ spectral_summary)
    MLP: Linear(268, 128) → ReLU → Dropout(0.1) → Linear(128, 1)
    """

    def __init__(self, node_dim, edge_dim, n_spectral_features=12,
                 hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_spectral = n_spectral_features

        # Input projection
        self.node_embedding = nn.Linear(node_dim, hidden_dim)

        # GIN layers (same architecture as Day 1 baseline)
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

        # Prediction head: GNN embedding + spectral features
        head_input_dim = hidden_dim + n_spectral_features  # 256 + 12 = 268
        self.pred_head = nn.Sequential(
            nn.Linear(head_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch, spectral_features):
        """
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch assignment [num_nodes]
            spectral_features: Bloch spectral features [batch_size, 12]
        """
        # GNN forward pass
        h = self.node_embedding(x)
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_attr)
            h = self.batch_norms[i](h)
            h = torch.relu(h)

        # Global pooling
        h_gnn = global_add_pool(h, batch)  # [batch_size, hidden_dim]

        # Concatenate with spectral features
        combined = torch.cat([h_gnn, spectral_features], dim=-1)  # [batch_size, 268]

        # Prediction
        out = self.pred_head(combined)
        return out


class PureSpectralMLP(nn.Module):
    """
    Pure spectral MLP (no GNN) for ablation (e).
    Input: only 12 computed Bloch spectral features.
    MLP: Linear(12, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 1)
    """

    def __init__(self, n_features=12):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.mlp(x)


class BaselineGIN(nn.Module):
    """
    Standard GIN on trimer (no spectral features) - baseline for comparison.
    Uses global_add_pool (same as BlochSpecGIN) for fair comparison.
    Prediction head: Linear(256, 256) → ReLU → Linear(256, 1)
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_layers=3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

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

        # Standard MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_embedding(x)
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_attr)
            h = self.batch_norms[i](h)
            h = torch.relu(h)

        h_gnn = global_add_pool(h, batch)
        out = self.mlp_head(h_gnn)
        return out


def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
