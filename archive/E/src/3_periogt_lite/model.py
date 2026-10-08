"""
PerioGT-Lite Iteration 2: Model architecture.
EXACTLY matches Day 1/PCR-v2 baseline architecture:
- EdgeAwareGINConv (3 layers, hidden_dim=256)
- BatchNorm inside MLP (2 per layer) + BatchNorm after conv + ReLU
- train_eps=True
- global_mean_pool
- Supervised head: MLP 256 -> 128 -> 1 (same as Day 1)
- Projector: 256 -> 128 -> 64 -> L2 norm (for contrastive pre-training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, MessagePassing

from feature_utils import ATOM_FEATURE_DIM, BOND_FEATURE_DIM


class EdgeAwareGINConv(MessagePassing):
    """GIN convolution with edge feature awareness (same as Day 1/PCR-v2)."""

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


class GINEncoder(nn.Module):
    """
    EXACT Day 1/PCR-v2 GIN encoder:
    - 3 layers, hidden_dim=256
    - BatchNorm inside MLP (2 per layer)
    - BatchNorm after each conv, then ReLU
    - train_eps=True
    - No dropout
    Returns node-level features (before pooling).
    """

    def __init__(self, node_dim=ATOM_FEATURE_DIM, edge_dim=BOND_FEATURE_DIM,
                 hidden_dim=256, num_layers=3):
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

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_embedding(x)

        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_attr)
            h = self.batch_norms[i](h)
            h = torch.relu(h)

        return h, batch


class ProjectorHead(nn.Module):
    """Projection head for contrastive pre-training.
    Same as PCR-v2: 256 -> 128 -> 64 -> L2 norm.
    Input is L2-normalized before projection (matching PCR-v2).
    """

    def __init__(self, input_dim=256, hidden_dim=128, proj_dim=64):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        z = self.projector(x)
        z = F.normalize(z, p=2, dim=-1)
        return z


class SupervisedHead(nn.Module):
    """MLP prediction head: 256 -> 128 -> 1 (same as Day 1)."""

    def __init__(self, input_dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)


class PerioGTPretrainModel(nn.Module):
    """Full model for contrastive pre-training (encoder + projector)."""

    def __init__(self, node_dim=ATOM_FEATURE_DIM, edge_dim=BOND_FEATURE_DIM,
                 hidden_dim=256, num_layers=3, proj_dim=64):
        super().__init__()
        self.encoder = GINEncoder(node_dim, edge_dim, hidden_dim, num_layers)
        self.projector = ProjectorHead(hidden_dim, 128, proj_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        h_node, batch = self.encoder(x, edge_index, edge_attr, batch)
        h_graph = global_mean_pool(h_node, batch)
        # L2 normalize before projector (matching PCR-v2)
        h_normed = F.normalize(h_graph, p=2, dim=-1)
        z = self.projector(h_normed)
        return z


class PerioGTFineTuneModel(nn.Module):
    """Full model for fine-tuning (encoder + supervised head)."""

    def __init__(self, node_dim=ATOM_FEATURE_DIM, edge_dim=BOND_FEATURE_DIM,
                 hidden_dim=256, num_layers=3):
        super().__init__()
        self.encoder = GINEncoder(node_dim, edge_dim, hidden_dim, num_layers)
        self.sup_head = SupervisedHead(hidden_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        h_node, batch = self.encoder(x, edge_index, edge_attr, batch)
        h_graph = global_mean_pool(h_node, batch)
        pred = self.sup_head(h_graph)
        return pred

    def load_pretrained_encoder(self, checkpoint_path):
        """Load pre-trained encoder weights."""
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.encoder.load_state_dict(state_dict)
        print(f"Loaded pre-trained encoder from {checkpoint_path}")


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick test
    model = PerioGTPretrainModel()
    print(f"Pre-training model params: {count_parameters(model):,}")
    model = PerioGTFineTuneModel()
    print(f"Fine-tuning model params: {count_parameters(model):,}")
    print(f"Encoder params: {count_parameters(model.encoder):,}")
    print(f"Sup head params: {count_parameters(model.sup_head):,}")
