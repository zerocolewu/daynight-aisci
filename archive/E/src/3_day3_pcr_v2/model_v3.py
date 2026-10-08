"""
PCR-v2 Iteration 3: EXACT Day 1 baseline architecture + contrastive addition.

Day 1 baseline that achieved 50.19 RMSE:
- EdgeAwareGINConv (3 layers, hidden_dim=256)
- BatchNorm after each conv, then ReLU
- global_mean_pool
- MLP head: 256 -> 128 -> 1
- No dropout
- Adam optimizer, LR=1e-3, batch_size=32, patience=30, seed=71

PCR Addition (minimal change):
- Same encoder + mean_pool for supervised head (unchanged)
- Contrastive projector: mean_pooled -> L2_norm -> MLP(256->128->64) -> L2_norm -> InfoNCE
- L2 norm BEFORE projector normalizes monomer/trimer magnitude difference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, MessagePassing


class EdgeAwareGINConv(MessagePassing):
    """GIN convolution with edge feature awareness (same as Day 1)."""

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
    EXACT Day 1 GIN encoder:
    - 3 layers, hidden_dim=256
    - BatchNorm after each conv, then ReLU
    - No dropout
    Returns node-level features.
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

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_embedding(x)

        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_attr)
            h = self.batch_norms[i](h)
            h = torch.relu(h)

        return h, batch


class ContrastiveProjector(nn.Module):
    """
    MLP projector: 256 -> 128 -> 64 with L2 normalization.
    Input is expected to be L2-normalized already (from upstream).
    """

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


class PCRModel(nn.Module):
    """
    PCR-v2 Iteration 3: EXACT Day 1 architecture + contrastive addition.

    Architecture:
    - Encoder: 3-layer EdgeAwareGINConv (same as Day 1)
    - Supervised: mean_pool -> MLP(256->128->1) (same as Day 1)
    - Contrastive (PCR only): mean_pool -> L2_norm -> MLP(256->128->64) -> L2_norm -> InfoNCE
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_layers=3,
                 proj_dim=64, mode='baseline'):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim

        self.encoder = GINEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        if mode == 'pcr':
            self.projector = ContrastiveProjector(
                input_dim=hidden_dim,
                hidden_dim=128,
                output_dim=proj_dim,
            )

        self.sup_head = SupervisedHead(input_dim=hidden_dim)

    def forward_baseline(self, x, edge_index, edge_attr, batch):
        """Baseline: encode trimer, mean_pool, predict Tg."""
        h_node, batch = self.encoder(x, edge_index, edge_attr, batch)
        h_graph = global_mean_pool(h_node, batch)
        pred = self.sup_head(h_graph)
        return pred, h_graph

    def forward_pcr(self, mono_x, mono_edge_index, mono_edge_attr, mono_batch,
                    tri_x, tri_edge_index, tri_edge_attr, tri_batch):
        """PCR: encode both views, mean_pool, contrastive + supervised."""
        h_mono_node, mono_batch = self.encoder(mono_x, mono_edge_index, mono_edge_attr, mono_batch)
        h_tri_node, tri_batch = self.encoder(tri_x, tri_edge_index, tri_edge_attr, tri_batch)

        # Mean pool both views (same as Day 1 baseline)
        h_mono = global_mean_pool(h_mono_node, mono_batch)
        h_tri = global_mean_pool(h_tri_node, tri_batch)

        # Contrastive: L2 normalize before projector to handle magnitude difference
        h_mono_normed = F.normalize(h_mono, p=2, dim=-1)
        h_tri_normed = F.normalize(h_tri, p=2, dim=-1)

        z_mono = self.projector(h_mono_normed)
        z_tri = self.projector(h_tri_normed)

        # Supervised: raw mean_pooled embedding (same as Day 1)
        pred = self.sup_head(h_tri)

        return pred, z_mono, z_tri, h_tri


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


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
