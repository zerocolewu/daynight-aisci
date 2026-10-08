"""
PCR-v2 Iteration 2: Hybrid pooling architecture.

Key fix from iteration 1:
- global_mean_pool degraded baseline from 50.19 (sum_pool) to 52.68
- Solution: HYBRID POOLING
  - sum_pool → supervised head (preserves regression magnitude = ~50.19 baseline)
  - mean_pool → contrastive projector (normalizes by graph size for cross-size comparison)
  - Both poolings operate on the same GIN node features, just different aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, MessagePassing


class EdgeAwareGINConv(MessagePassing):
    """GIN convolution with edge feature awareness."""

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
    GIN encoder with 5 layers, hidden_dim=256, BatchNorm, dropout=0.5.
    Returns NODE-level features (no pooling here — pooling done downstream).
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_layers=5, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

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
        """
        Returns:
            h_node: Node-level embeddings [num_nodes, hidden_dim]
            batch: Batch assignment vector (for downstream pooling)
        """
        h = self.node_embedding(x)

        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_attr)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return h, batch


class ContrastiveProjector(nn.Module):
    """MLP projector: 256 -> 128 -> 64 with ReLU and L2 normalization."""

    def __init__(self, input_dim=256, hidden_dim=128, output_dim=64):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        z = self.projector(x)
        z = F.normalize(z, p=2, dim=-1)
        return z


class SupervisedHead(nn.Module):
    """Linear prediction head: 256 -> 1"""

    def __init__(self, input_dim=256):
        super().__init__()
        self.head = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.head(x).squeeze(-1)


class PCRModel(nn.Module):
    """
    PCR-v2 Iteration 2: Hybrid pooling model.

    Pooling strategy:
    - sum_pool  → supervised head (preserves magnitude for Tg regression)
    - mean_pool → contrastive projector (size-invariant for monomer-trimer comparison)

    Both applied to the same GIN node features.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_layers=5,
                 dropout=0.5, proj_dim=64, mode='baseline'):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim

        self.encoder = GINEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        if mode in ('pcr',):
            self.projector = ContrastiveProjector(
                input_dim=hidden_dim,
                hidden_dim=128,
                output_dim=proj_dim,
            )

        self.sup_head = SupervisedHead(input_dim=hidden_dim)

    def _pool(self, h_node, batch):
        """Apply hybrid pooling: sum for supervised, mean for contrastive."""
        h_sum = global_add_pool(h_node, batch)   # preserves magnitude
        h_mean = global_mean_pool(h_node, batch)  # size-invariant
        return h_sum, h_mean

    def forward_baseline(self, x, edge_index, edge_attr, batch):
        """Baseline: encode trimer, sum_pool, predict Tg."""
        h_node, batch = self.encoder(x, edge_index, edge_attr, batch)
        h_sum, _ = self._pool(h_node, batch)
        pred = self.sup_head(h_sum)
        return pred, h_sum

    def forward_pcr(self, mono_x, mono_edge_index, mono_edge_attr, mono_batch,
                    tri_x, tri_edge_index, tri_edge_attr, tri_batch):
        """PCR: encode both views, hybrid pool, contrastive + supervised."""
        h_mono_node, mono_batch = self.encoder(mono_x, mono_edge_index, mono_edge_attr, mono_batch)
        h_tri_node, tri_batch = self.encoder(tri_x, tri_edge_index, tri_edge_attr, tri_batch)

        # Hybrid pooling for both views
        h_mono_sum, h_mono_mean = self._pool(h_mono_node, mono_batch)
        h_tri_sum, h_tri_mean = self._pool(h_tri_node, tri_batch)

        # Contrastive uses mean-pooled embeddings
        z_mono = self.projector(h_mono_mean)
        z_tri = self.projector(h_tri_mean)

        # Supervised uses sum-pooled embeddings (preserves magnitude)
        pred = self.sup_head(h_tri_sum)

        return pred, z_mono, z_tri, h_tri_sum, h_tri_mean


def info_nce_loss(z1, z2, temperature=0.07):
    """InfoNCE contrastive loss between L2-normalized embeddings."""
    batch_size = z1.size(0)
    if batch_size <= 1:
        return torch.tensor(0.0, device=z1.device, requires_grad=True)

    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature

    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    pos_i = torch.arange(batch_size, device=z.device)
    pos_j = pos_i + batch_size

    loss_fwd = F.cross_entropy(sim[pos_i], pos_j)
    loss_bwd = F.cross_entropy(sim[pos_j], pos_i)

    return (loss_fwd + loss_bwd) / 2


def cosine_similarity_pairs(z1, z2):
    """Mean cosine similarity between matched pairs."""
    cos_sim = F.cosine_similarity(z1, z2, dim=-1)
    return cos_sim.mean().item()


def alignment_metric(z1, z2, alpha=2):
    """Alignment: lower = more aligned."""
    return (z1 - z2).norm(dim=1).pow(alpha).mean().item()


def uniformity_metric(z, t=2):
    """Uniformity: lower = more uniform."""
    pw_distances = torch.pdist(z, p=2)
    return pw_distances.pow(2).mul(-t).exp().mean().item()


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
