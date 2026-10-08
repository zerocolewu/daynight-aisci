"""
Model architectures for GraphMAE pretraining and fine-tuning (Iteration 3).
Key improvements:
- MultiScaleEncoder: JK-net style multi-scale feature aggregation
- Deeper decoder (3 layers)
- Edge feature masking in pretraining
- Contrastive loss (MoCL-style)
- Progressive unfreezing support
- Label normalization support
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
    Returns intermediate representations for multi-scale features.
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

    def forward_with_intermediates(self, x, edge_index, edge_attr, batch=None):
        """Forward pass returning intermediate layer outputs for multi-scale features."""
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        intermediates = [x]
        for conv, norm, dropout in zip(self.convs, self.norms, self.dropouts):
            identity = x
            x_out = conv(x, edge_index, edge_attr)
            x_out = norm(x_out)
            x_out = F.relu(x_out)
            x_out = dropout(x_out)
            x = identity + x_out
            intermediates.append(x)

        return x, intermediates


class MultiScaleEncoder(nn.Module):
    """
    Wrapper around GINEncoder that aggregates features from multiple layers
    (JK-net style). This helps the fine-tuning model capture both local
    and global graph structure.
    """

    def __init__(self, in_dim=27, hidden_dim=256, out_dim=256, num_layers=5, drop_rate=0.1):
        super(MultiScaleEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.gine = GINEncoder(in_dim, hidden_dim, out_dim, num_layers, drop_rate)

    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Returns concatenated multi-scale features from layers 2, 3, 4, 5.
        Output dim = hidden_dim * 3 = 768
        """
        _, intermediates = self.gine.forward_with_intermediates(
            x, edge_index, edge_attr, batch
        )

        # Use last 3 intermediate representations (skip first two)
        # This gives local (layer 3), medium (layer 4), and global (layer 5) features
        selected = intermediates[-3:]  # layers 3, 4, 5

        # Concatenate along feature dimension
        return torch.cat(selected, dim=-1)  # [N, 768]


class GraphMAEDecoder(nn.Module):
    """Deeper 3-layer GINE decoder for GraphMAE (improved from 2-layer)."""

    def __init__(self, in_dim=256, hidden_dim=256, out_dim=27):
        super(GraphMAEDecoder, self).__init__()
        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            eps=0.0,
            train_eps=False,
            edge_dim=6
        )
        self.conv2 = GINEConv(
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
        self.conv3 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, out_dim)
            ),
            eps=0.0,
            train_eps=False,
            edge_dim=6
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        return x


class EdgeDecoder(nn.Module):
    """Decoder for edge feature reconstruction (new in iteration 2)."""

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

        # Compute attention scores
        scores = self.gate_nn(x)  # [num_nodes, num_heads]

        graph_repr = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() == 0:
                continue

            nodes = x[mask]  # [n, hidden_dim]
            node_scores = scores[mask]  # [n, num_heads]

            # Per-graph softmax for each head
            attn = F.softmax(node_scores, dim=0)  # [n, num_heads]

            # Weighted sum per head
            head_outputs = []
            for h in range(self.num_heads):
                weights = attn[:, h].unsqueeze(-1)  # [n, 1]
                head_out = (nodes * weights).sum(dim=0)  # [hidden_dim]
                head_outputs.append(head_out)

            # Concatenate heads and project
            concat = torch.cat(head_outputs, dim=-1)  # [hidden_dim * num_heads]
            graph_repr[i] = self.out_proj(concat)

        return graph_repr


class GraphMAE(nn.Module):
    """
    GraphMAE v2: masked graph autoencoder with re-mask decoding.
    Improvements for iteration 3:
    - Higher mask ratio (65%)
    - Edge feature masking + reconstruction
    - Contrastive loss component
    - Deeper decoder (3 layers)
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

        # Node decoder (deeper - 3 layers)
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

        # Edge feature masking: mask edges connected to masked nodes (vectorized)
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
    Fine-tuning model with stronger regularization and multi-scale features.
    Improvements: higher dropout (0.3), weight decay, label normalization,
    progressive unfreezing, multi-scale encoder features.
    """

    def __init__(self, encoder: MultiScaleEncoder, hidden_dim=768, num_heads=4, drop_rate=0.3):
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

    def get_embeddings(self, x, edge_index, edge_attr, batch):
        with torch.no_grad():
            z = self.encoder(x, edge_index, edge_attr, batch)
            attn_repr = self.attention_pooling(z, batch)
            mean_repr = global_mean_pool(z, batch)
            max_repr = global_max_pool(z, batch)
            aux_repr = self.aux_proj(torch.cat([mean_repr, max_repr], dim=-1))
            combined = torch.cat([attn_repr, aux_repr], dim=-1)
            combined = self.combine_proj(combined)
        return combined


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
    Uses two augmented views of the same graph.
    """

    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, proj1, proj2):
        """
        proj1, proj2: [batch_size, proj_dim] - projections from two augmented views
        """
        batch_size = proj1.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=proj1.device)

        # Normalize projections
        proj1 = F.normalize(proj1, dim=-1)
        proj2 = F.normalize(proj2, dim=-1)

        # Concatenate to form [2N, dim]
        representations = torch.cat([proj1, proj2], dim=0)

        # Similarity matrix
        similarity = torch.matmul(representations, representations.T) / self.temperature

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=proj1.device)
        similarity.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(proj1.device)

        loss = F.cross_entropy(similarity, labels)
        return loss


if __name__ == "__main__":
    print("Testing models...")

    num_nodes = 20
    x = torch.randn(num_nodes, 27)
    edge_index = torch.randint(0, num_nodes, (2, 40))
    edge_attr = torch.randn(40, 6)
    batch = torch.cat([torch.full((5,), i) for i in range(4)])

    # Encoder
    encoder = GINEncoder(27, 256, 256, 5)
    z = encoder(x, edge_index, edge_attr, batch)
    print(f"Encoder output: {z.shape}")

    # MultiScaleEncoder
    ms_encoder = MultiScaleEncoder(27, 256, 256, 5)
    z_ms = ms_encoder(x, edge_index, edge_attr, batch)
    print(f"MultiScaleEncoder output: {z_ms.shape}")  # Should be [20, 768]

    # GraphMAE v2
    mae = GraphMAE(27, 256, 256, 5, mask_ratio=0.65)
    x_recon, x_orig, mask_idx, z_out, edge_recon, edge_orig, proj = mae(
        x, edge_index, edge_attr, batch
    )
    print(f"Node recon: {x_recon.shape}, Edge recon: {edge_recon.shape}, Masked: {len(mask_idx)}")
    print(f"Projection: {proj.shape}")

    # Losses
    sce = SCELoss(power=5.0)
    node_loss = sce(x_recon, x_orig, mask_idx)
    edge_loss = EdgeReconLoss()(edge_recon, edge_orig)
    cl_loss = ContrastiveLoss()(proj, proj + torch.randn_like(proj) * 0.1)
    print(f"Node SCE loss: {node_loss.item():.4f}")
    print(f"Edge recon loss: {edge_loss.item():.4f}")
    print(f"Contrastive loss: {cl_loss.item():.4f}")

    # FineTuner with MultiScaleEncoder
    ft = FineTuner(ms_encoder, hidden_dim=768, num_heads=4, drop_rate=0.3)
    pred = ft(x, edge_index, edge_attr, batch)
    print(f"FineTuner prediction: {pred.shape}")  # Should be [4]

    print("All models OK!")
