"""
Model architectures for BlochSpec Iteration 2.

CRITICAL FIX: All models use global_mean_pool (not global_add_pool).
This matches the Day 1 reference baseline (RMSE ~50.19).

Models:
1. EdgeAwareGINConv - same GIN convolution as Day 1 baseline
2. BaselineGIN - GIN on trimer, global_mean_pool, MLP head 256->256->1
3. BlochSpecGIN - GIN + 12 Bloch spectral features in prediction head
4. WiderMLPGIN - Same GIN but wider MLP head (matching BlochSpec param count), no extra features
5. ReducedGNNBlochSpec - GIN hidden_dim=244 + 12 spectral = 256 total input to head
6. PureSpectralMLP - MLP on spectral features only
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, MessagePassing


class EdgeAwareGINConv(MessagePassing):
    """GIN convolution with edge feature awareness (same as Day 1 baseline)."""

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


class BaselineGIN(nn.Module):
    """
    Standard GIN on trimer (no spectral features) - baseline.
    Uses global_mean_pool (matches Day 1 reference).
    MLP head: 256 -> 256 -> 1
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_layers=3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.node_embedding = nn.Linear(node_dim, hidden_dim)

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

        # Same MLP head as Day 1 reference: 256 -> 256 -> 1
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

        h_gnn = global_mean_pool(h, batch)
        out = self.mlp_head(h_gnn)
        return out


class BlochSpecGIN(nn.Module):
    """
    BlochSpec: GIN + Bloch spectral features concatenated in prediction head.
    Uses global_mean_pool.

    GNN output: hidden_dim (default 256)
    Prediction head: (hidden_dim + 12) -> 128 -> 1
    """

    def __init__(self, node_dim, edge_dim, n_spectral_features=12,
                 hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_spectral = n_spectral_features

        self.node_embedding = nn.Linear(node_dim, hidden_dim)

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

        head_input_dim = hidden_dim + n_spectral_features  # 256 + 12 = 268
        self.pred_head = nn.Sequential(
            nn.Linear(head_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch, spectral_features):
        h = self.node_embedding(x)
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_attr)
            h = self.batch_norms[i](h)
            h = torch.relu(h)

        h_gnn = global_mean_pool(h, batch)
        combined = torch.cat([h_gnn, spectral_features], dim=-1)
        out = self.pred_head(combined)
        return out


class WiderMLPGIN(nn.Module):
    """
    GIN with wider MLP head to match BlochSpec parameter count.
    NO extra features - controls for parameter count increase.

    GNN output: 256 (same as baseline)
    MLP head: 256 -> 256 -> 1  (wider hidden layers to match BlochSpec params)
    Note: Baseline already has 256->256->1, so this is identical architecture-wise.
    The point is to verify that the improvement isn't from parameter count.
    Actually, let's make it 256 -> 512 -> 256 -> 1 to have MORE params than BlochSpec.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_layers=3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.node_embedding = nn.Linear(node_dim, hidden_dim)

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

        # Wider MLP head: 256 -> 512 -> 256 -> 1
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_embedding(x)
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_attr)
            h = self.batch_norms[i](h)
            h = torch.relu(h)

        h_gnn = global_mean_pool(h, batch)
        out = self.mlp_head(h_gnn)
        return out


class ReducedGNNBlochSpec(nn.Module):
    """
    GIN with reduced hidden_dim so that GNN output + spectral features
    equals the same input dimension as baseline MLP head.

    hidden_dim = 244, spectral = 12, total = 256 (same as baseline MLP input)
    Uses global_mean_pool.
    """

    def __init__(self, node_dim, edge_dim, n_spectral_features=12,
                 hidden_dim=244, num_layers=3, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_spectral = n_spectral_features

        self.node_embedding = nn.Linear(node_dim, hidden_dim)

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

        # head_input_dim = 244 + 12 = 256 (same as baseline)
        head_input_dim = hidden_dim + n_spectral_features
        self.pred_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch, spectral_features):
        h = self.node_embedding(x)
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_attr)
            h = self.batch_norms[i](h)
            h = torch.relu(h)

        h_gnn = global_mean_pool(h, batch)
        combined = torch.cat([h_gnn, spectral_features], dim=-1)
        out = self.pred_head(combined)
        return out


class PureSpectralMLP(nn.Module):
    """Pure spectral MLP (no GNN) for ablation."""

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


def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
