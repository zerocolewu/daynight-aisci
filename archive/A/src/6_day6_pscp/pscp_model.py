"""
Polymer-Specific Contrastive Pretraining (PSCP) Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np


class GINEncoder(nn.Module):
    """GIN backbone for molecular graph encoding"""

    def __init__(self, num_node_features=159, hidden_dim=256, num_layers=3, edge_dim=10):
        super().__init__()

        self.node_embedding = nn.Linear(num_node_features, hidden_dim)

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
            # Note: edge_dim parameter removed in newer PyG versions
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Node embedding
        x = self.node_embedding(x)

        # GIN layers (note: edge_attr not used in standard GINConv)
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)

        # Global pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        graph_embedding = global_mean_pool(x, batch)

        return graph_embedding, x  # Return both graph and node embeddings


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""

    def __init__(self, input_dim=256, hidden_dim=128, output_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class MaskPredictionHead(nn.Module):
    """Head for predicting masked atom types"""

    def __init__(self, input_dim=256, num_atom_types=159):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_atom_types)
        )

    def forward(self, x):
        return self.net(x)


class PSCPModel(nn.Module):
    """
    Full PSCP Model combining encoder and all pretraining heads
    """

    def __init__(self, config=None):
        super().__init__()

        self.config = config or {}
        num_node_features = self.config.get('num_node_features', 159)
        hidden_dim = self.config.get('hidden_dim', 256)
        num_layers = self.config.get('num_layers', 3)
        edge_dim = self.config.get('edge_dim', 10)
        proj_dim = self.config.get('proj_dim', 128)

        # Encoder
        self.encoder = GINEncoder(
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            edge_dim=edge_dim
        )

        # Projection heads for contrastive learning
        self.projection_head = ProjectionHead(hidden_dim, hidden_dim // 2, proj_dim)

        # Mask prediction head for FGM
        self.mask_head = MaskPredictionHead(hidden_dim, num_node_features)

    def forward(self, data, return_node_embeddings=False):
        graph_emb, node_emb = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)

        if return_node_embeddings:
            return graph_emb, node_emb
        return graph_emb

    def get_projection(self, data):
        graph_emb, _ = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        return self.projection_head(graph_emb)

    def predict_masked_atoms(self, data):
        _, node_emb = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        return self.mask_head(node_emb)


class PredictionModel(nn.Module):
    """
    Fine-tuning model for Tg prediction
    Combines pretrained encoder with ECFP fingerprint
    """

    def __init__(self, encoder, hidden_dim=256, ecfp_dim=2048, dropout=0.1):
        super().__init__()

        self.encoder = encoder

        # Freeze encoder initially (optional)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        # Prediction head: concat GIN output + ECFP
        combined_dim = hidden_dim + ecfp_dim

        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, data, ecfp):
        graph_emb, _ = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)

        # Concatenate with ECFP
        combined = torch.cat([graph_emb, ecfp], dim=-1)

        return self.predictor(combined).squeeze(-1)


def info_nce_loss(anchors, positives, negatives, temperature=0.1):
    """
    InfoNCE contrastive loss

    Args:
        anchors: (batch_size, embed_dim)
        positives: (batch_size, embed_dim)
        negatives: (batch_size, num_negatives, embed_dim)
        temperature: temperature scaling
    """
    batch_size = anchors.size(0)

    # Normalize
    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    # Positive similarity
    pos_sim = torch.sum(anchors * positives, dim=-1) / temperature  # (batch_size,)

    # Negative similarity
    neg_sim = torch.bmm(negatives, anchors.unsqueeze(-1)).squeeze(-1) / temperature  # (batch_size, num_neg)

    # Combine
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (batch_size, num_neg + 1)

    # Labels: positive is always at index 0
    labels = torch.zeros(batch_size, dtype=torch.long, device=anchors.device)

    loss = F.cross_entropy(logits, labels)

    return loss


def triplet_loss(anchor, positive, negative, margin=1.0):
    """Triplet loss for contrastive learning"""
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()
