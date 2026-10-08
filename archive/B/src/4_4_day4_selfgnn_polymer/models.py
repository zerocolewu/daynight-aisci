"""GNN model definitions for polymer self-supervised learning (REWRITE).

Fixes:
  - 42-dim input everywhere (no pretrain→finetune dim mismatch)
  - GATConv (not GATv2) with ELU + BatchNorm1d
  - FinetuneModel: Set2Set readout only (no mean_pool concat hack)
  - Predictor: 512→256→128→1 with BN+ReLU+Dropout(0.1)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Set2Set


class GATEncoder(nn.Module):
    """4-layer GAT encoder with BatchNorm.

    Architecture: 4 × GATConv(hidden_dim, hidden_dim//heads, heads=4, concat=True)
    Each layer followed by BatchNorm1d + ELU + Dropout.

    Args:
        in_dim: input feature dimension (42 = 8 rich + 2 PSPE + 32 RWPE)
        hidden_dim: hidden dimension (256)
        heads: number of attention heads (4)
        use_bn: whether to use BatchNorm1d after each GAT layer
        dropout: dropout rate (0.0 default for pretraining encoder)
    """
    def __init__(self, in_dim=42, hidden_dim=256, heads=4, use_bn=True, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.use_bn = use_bn
        self.dropout_rate = dropout
        self.out_dim = hidden_dim  # Output dimension (all layers concat=True)

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(4):
            in_ch = hidden_dim
            out_ch = hidden_dim // heads  # per-head output dim
            self.convs.append(
                GATConv(in_ch, out_ch, heads=heads, concat=True, dropout=dropout)
            )
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.bns.append(nn.Identity())

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.input_proj(x)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = self.dropout(x)

        return x  # [num_nodes, hidden_dim]


class StudentModel(nn.Module):
    """BYOL Student = GATEncoder + prediction MLP g(256→128→256)."""
    def __init__(self, in_dim=42, hidden_dim=256, heads=4, use_bn=True, dropout=0.0):
        super().__init__()
        self.encoder = GATEncoder(in_dim, hidden_dim, heads, use_bn, dropout)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = self.encoder(x, edge_index, edge_attr, batch)
        z = self.predictor(h)
        return z

    def encode(self, x, edge_index, edge_attr=None, batch=None):
        """Get encoder output only (no predictor)."""
        return self.encoder(x, edge_index, edge_attr, batch)


class TeacherModel(nn.Module):
    """BYOL Teacher = GATEncoder with EMA, no gradients."""
    def __init__(self, in_dim=42, hidden_dim=256, heads=4, use_bn=True, dropout=0.0):
        super().__init__()
        self.encoder = GATEncoder(in_dim, hidden_dim, heads, use_bn, dropout)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return self.encoder(x, edge_index, edge_attr, batch)

    @torch.no_grad()
    def update_ema(self, student_encoder, tau=0.999):
        """Update teacher weights via EMA of student encoder weights."""
        for t_param, s_param in zip(self.encoder.parameters(),
                                     student_encoder.parameters()):
            t_param.data.mul_(tau).add_(s_param.data, alpha=1.0 - tau)


class FinetuneModel(nn.Module):
    """Finetuning model: GATEncoder + Set2Set readout + MLP predictor.

    Architecture (matching Day 3 baseline target):
      GATEncoder (42-dim in, 256-dim out)
      → Set2Set(256, processing_steps=3) → 512-dim
      → Linear(512→256) + BN + ReLU + Dropout(0.1)
      → Linear(256→128) + BN + ReLU + Dropout(0.1)
      → Linear(128→1)

    No dimension mismatch — both pretrain and finetune use 42-dim input.
    """
    def __init__(self, encoder, hidden_dim=256, heads=4, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.hidden_dim = hidden_dim

        # Set2Set readout: 256 → 512
        self.set2set = Set2Set(hidden_dim, processing_steps=3, num_layers=1)
        s2s_out = hidden_dim * 2  # 512

        # Predictor MLP: 512 → 256 → 128 → 1
        self.predictor = nn.Sequential(
            nn.Linear(s2s_out, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Run through GAT encoder
        x = self.encoder(x, edge_index, edge_attr, batch)

        # Set2Set pooling → graph-level representation
        graph_emb = self.set2set(x, batch)  # [B, 512]

        # Predictor
        return self.predictor(graph_emb).squeeze(-1)


class NoPretrainModel(nn.Module):
    """Same architecture as FinetuneModel but random init (no-pretrain baseline)."""
    def __init__(self, in_dim=42, hidden_dim=256, heads=4):
        super().__init__()
        encoder = GATEncoder(in_dim, hidden_dim, heads, use_bn=True)
        self.finetune = FinetuneModel(encoder, hidden_dim, heads, freeze_encoder=False)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return self.finetune(x, edge_index, edge_attr, batch)
