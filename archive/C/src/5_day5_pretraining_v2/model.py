"""
Model architectures for GraphMAE pretraining and fine-tuning (Iteration v2).
Key improvements:
- GIN encoder (5 layers, 256 hidden) with GINEConv
- GraphMAE with re-mask decoding, 65% mask ratio
- Edge feature masking in pretraining
- Contrastive loss component (InfoNCE)
- Attention pooling for fine-tuning
- Label normalization support
- Dropout 0.3 in fine-tuning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, global_max_pool
from typing import Optional, Tuple
import math


class GINEncoder(nn.Module):
    """
    GIN encoder with edge features using GINEConv.
    5 layers, hidden_dim=256, LayerNorm + ReLU + residual connections.
    """

    def __init__(self, in_dim=27, hidden_dim=256, out_dim=256, num_layers=5, drop_rate=0.1):
        super(GINEncoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # GINE layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    ),
                    eps=0.0,
                    train_eps=False,
                    edge_dim=6
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(drop_rate))

    def forward(self, x, edge_index, edge_attr, batch=None):
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # GINE layers with residual
        for conv, norm, dropout in zip(self.convs, self.norms, self.dropouts):
            identity = x
            x_out = conv(x, edge_index, edge_attr)
            x_out = norm(x_out)
            x_out = F.relu(x_out)
            x_out = dropout(x_out)
            x = identity + x_out

        return x


class GraphMAEDecoder(nn.Module):
    """3-layer GINE decoder for GraphMAE."""

    def __init__(self, in_dim=256, hidden_dim=256, out_dim=27):
        super(GraphMAEDecoder, self).__init__()
        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            eps=0.0, train_eps=False, edge_dim=6
        )
        self.conv2 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            eps=0.0, train_eps=False, edge_dim=6
        )
        self.conv3 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, out_dim)),
            eps=0.0, train_eps=False, edge_dim=6
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        return x


class EdgeDecoder(nn.Module):
    """Decoder for edge feature reconstruction."""

    def __init__(self, hidden_dim=256, edge_dim=6):
        super(EdgeDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )

    def forward(self, z, edge_index):
        src, dst = edge_index
        edge_repr = torch.cat([z[src], z[dst]], dim=-1)
        return self.decoder(edge_repr)


class AttentionPooling(nn.Module):
    """Multi-head attention pooling with per-graph softmax."""

    def __init__(self, hidden_dim=256, num_heads=4):
        super(AttentionPooling, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_heads)
        )
        self.out_proj = nn.Linear(hidden_dim * num_heads, hidden_dim)

    def forward(self, x, batch):
        batch_size = batch.max().item() + 1
        scores = self.gate_nn(x)  # [num_nodes, num_heads]

        graph_repr = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() == 0:
                continue

            nodes = x[mask]
            node_scores = scores[mask]

            attn = F.softmax(node_scores, dim=0)

            head_outputs = []
            for h in range(self.num_heads):
                weights = attn[:, h].unsqueeze(-1)
                head_out = (nodes * weights).sum(dim=0)
                head_outputs.append(head_out)

            concat = torch.cat(head_outputs, dim=-1)
            graph_repr[i] = self.out_proj(concat)

        return graph_repr


class GraphMAE(nn.Module):
    """
    GraphMAE v2: masked graph autoencoder with re-mask decoding.
    65% mask ratio, edge masking, contrastive loss.
    """

    def __init__(self, in_dim=27, hidden_dim=256, out_dim=256, num_layers=5,
                 mask_ratio=0.65, drop_rate=0.1, edge_dim=6):
        super(GraphMAE, self).__init__()
        self.mask_ratio = mask_ratio
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.edge_dim = edge_dim

        # Learnable mask tokens
        self.input_mask_token = nn.Parameter(torch.randn(1, in_dim) * 0.1)
        self.latent_mask_token = nn.Parameter(torch.randn(1, hidden_dim) * 0.1)
        self.edge_mask_token = nn.Parameter(torch.randn(1, edge_dim) * 0.1)

        # Encoder
        self.encoder = GINEncoder(in_dim, hidden_dim, out_dim, num_layers, drop_rate)

        # Node decoder (3 layers)
        self.decoder = GraphMAEDecoder(hidden_dim, hidden_dim, in_dim)

        # Edge decoder
        self.edge_decoder = EdgeDecoder(hidden_dim, edge_dim)

        # Projection head for contrastive learning
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        num_nodes = x.size(0)
        num_mask = max(1, int(num_nodes * self.mask_ratio))

        # Select nodes to mask
        mask_idx = torch.randperm(num_nodes, device=x.device)[:num_mask]

        # Apply mask token in input space
        x_masked = x.clone()
        x_masked[mask_idx] = self.input_mask_token.expand(len(mask_idx), -1)

        # Edge feature masking: mask edges connected to masked nodes
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        node_mask[mask_idx] = True
        src, dst = edge_index
        edge_mask = ~(node_mask[src] | node_mask[dst])

        edge_attr_masked = edge_attr.clone()
        edge_attr_masked[~edge_mask] = self.edge_mask_token.expand((~edge_mask).sum(), -1)

        # Encode masked graph
        z = self.encoder(x_masked, edge_index, edge_attr_masked, batch)

        # Re-mask: replace encoder outputs at masked positions with latent mask token
        z_masked = z.clone()
        z_masked[mask_idx] = self.latent_mask_token.expand(len(mask_idx), -1)

        # Decode node features
        x_recon = self.decoder(z_masked, edge_index, edge_attr_masked)

        # Decode edge features
        edge_recon = self.edge_decoder(z_masked, edge_index)

        # Contrastive projection (for graph-level CL)
        if batch is not None:
            graph_emb = global_mean_pool(z, batch)
            proj = self.proj_head(graph_emb)
        else:
            proj = None

        return x_recon, x, mask_idx, z, edge_recon, edge_attr, proj


class FineTuner(nn.Module):
    """
    Fine-tuning model with attention pooling + MLP.
    Uses dropout 0.3, label normalization.
    """

    def __init__(self, encoder: GINEncoder, hidden_dim=256, num_heads=4, drop_rate=0.3):
        super(FineTuner, self).__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim

        # Multi-head attention pooling
        self.attention_pooling = AttentionPooling(hidden_dim, num_heads)

        # Also use mean/max pooling as auxiliary features
        self.aux_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Combined projection
        self.combine_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )

        # Prediction head (with stronger dropout)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        z = self.encoder(x, edge_index, edge_attr, batch)

        # Attention pooling
        attn_repr = self.attention_pooling(z, batch)

        # Mean + Max pooling
        mean_repr = global_mean_pool(z, batch)
        max_repr = global_max_pool(z, batch)
        aux_repr = self.aux_proj(torch.cat([mean_repr, max_repr], dim=-1))

        # Combine
        combined = torch.cat([attn_repr, aux_repr], dim=-1)
        combined = self.combine_proj(combined)

        pred = self.predictor(combined)
        return pred.squeeze(-1)


class SCELoss(nn.Module):
    """Scaled Cosine Error loss: (1 - cos_sim)^power"""

    def __init__(self, power=5.0):
        super(SCELoss, self).__init__()
        self.power = power

    def forward(self, pred, target, mask_idx=None):
        if mask_idx is not None and len(mask_idx) > 0:
            pred = pred[mask_idx]
            target = target[mask_idx]

        cos_sim = F.cosine_similarity(pred, target, dim=-1)
        loss = ((1 - cos_sim) ** self.power).mean()
        return loss


class EdgeReconLoss(nn.Module):
    """MSE loss for edge feature reconstruction."""

    def forward(self, pred, target):
        return F.mse_loss(pred, target)


class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for graph-level representations.
    """

    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, proj1, proj2):
        batch_size = proj1.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=proj1.device)

        proj1 = F.normalize(proj1, dim=-1)
        proj2 = F.normalize(proj2, dim=-1)

        representations = torch.cat([proj1, proj2], dim=0)
        similarity = torch.matmul(representations, representations.T) / self.temperature

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=proj1.device)
        similarity.masked_fill_(mask, -1e9)

        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(proj1.device)

        loss = F.cross_entropy(similarity, labels)
        return loss
