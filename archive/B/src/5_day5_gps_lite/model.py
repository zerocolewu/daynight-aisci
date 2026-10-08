"""
GPS-Lite: Single Global Attention Layer atop GAT Backbone for Polymer Tg Prediction.

Three variants:
  1. GPSLiteBaseline — GAT+RWPE+PSPE without global attention (standard 4-layer GAT)
  2. GPSLiteNoBias — GAT+RWPE+PSPE + global attention WITHOUT SPD bias
  3. GPSLiteSPD — GAT+RWPE+PSPE + global attention WITH SPD bias (primary experiment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


# ==============================================================================
# GAT Encoder (shared backbone)
# ==============================================================================

class GATEncoder(nn.Module):
    """4-layer GATConv encoder (stacked, no residual, no LayerNorm).

    Matches the proven Day 2/3 architecture:
    - Layer 1: GATConv(81, 256, heads=4, concat=True) → ELU → 1024 dim
    - Layer 2: GATConv(1024, 256, heads=4, concat=True) → ELU → 1024 dim
    - Layer 3: GATConv(1024, 256, heads=4, concat=True) → ELU → 1024 dim
    - Layer 4: GATConv(1024, 256, heads=4, concat=False) → ELU → 256 dim

    Args:
        in_dim: Input node feature dimension (81 for rich+RWPE+PSPE)
        edge_dim: Edge feature dimension (10 for rich features)
        hidden_dim: Per-head hidden dimension (256)
        heads: Number of attention heads (4)
        dropout: Dropout rate (0.1)
    """

    def __init__(
        self,
        in_dim: int = 81,
        edge_dim: int = 10,
        hidden_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout_rate = dropout
        self.hidden_dim = hidden_dim
        use_edge = edge_dim > 0
        concat_dim = hidden_dim * heads  # 256 * 4 = 1024

        # Layer 1: in_dim → 1024 (concat)
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads,
                             concat=True, dropout=dropout,
                             edge_dim=edge_dim if use_edge else None)
        # Layer 2: 1024 → 1024 (concat)
        self.conv2 = GATConv(concat_dim, hidden_dim, heads=heads,
                             concat=True, dropout=dropout,
                             edge_dim=edge_dim if use_edge else None)
        # Layer 3: 1024 → 1024 (concat)
        self.conv3 = GATConv(concat_dim, hidden_dim, heads=heads,
                             concat=True, dropout=dropout,
                             edge_dim=edge_dim if use_edge else None)
        # Layer 4: 1024 → 256 (mean, concat=False)
        self.conv4 = GATConv(concat_dim, hidden_dim, heads=heads,
                             concat=False, dropout=dropout,
                             edge_dim=edge_dim if use_edge else None)

    def forward(self, x, edge_index, edge_attr=None):
        """Forward pass — simple stacked GATConv + ELU, no residuals, no LayerNorm.

        Returns:
            h: Node representations [N, hidden_dim] = [N, 256]
        """
        h = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = F.elu(self.conv2(h, edge_index, edge_attr=edge_attr))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = F.elu(self.conv3(h, edge_index, edge_attr=edge_attr))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = F.elu(self.conv4(h, edge_index, edge_attr=edge_attr))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        return h


# ==============================================================================
# Global Attention Layer with SPD Bias
# ==============================================================================

class GlobalAttentionSPD(nn.Module):
    """Multi-head self-attention with Shortest-Path-Distance (SPD) bias.

    For each graph in the batch:
      - Compute Q, K, V from node representations
      - Add SPD-based bias to attention logits
      - Apply softmax and aggregate

    Args:
        hidden_dim: Total dimension (256)
        heads: Number of attention heads (4)
        head_dim: Per-head dimension (64)
        dropout: Dropout rate on attention weights (0.1)
        use_spd_bias: Whether to use SPD bias MLPs
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        heads: int = 4,
        head_dim: int = 64,
        dropout: float = 0.1,
        use_spd_bias: bool = True,
    ):
        super().__init__()
        assert hidden_dim == heads * head_dim

        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.use_spd_bias = use_spd_bias

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

        if use_spd_bias:
            # Per-head MLPs: one_hot(clipped_SPD) -> 32 -> 16 -> 1
            self.spd_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                ) for _ in range(heads)
            ])

    def forward(self, h, spd_list):
        """Global attention per graph.

        Args:
            h: Node representations [N_total, hidden_dim]
            spd_list: List of SPD matrices, one per graph [N_i, N_i]

        Returns:
            out: Attended node representations [N_total, hidden_dim]
            all_attn_weights: List of per-graph attention weights [H, N_i, N_i]
        """
        outputs = []
        all_attn_weights = []

        start_idx = 0
        for spd_i in spd_list:
            n_nodes = spd_i.shape[0]
            h_i = h[start_idx:start_idx + n_nodes]  # [n_nodes, hidden_dim]

            # Linear projections → multi-head
            q = self.W_q(h_i).view(n_nodes, self.heads, self.head_dim)  # [n, H, D]
            k = self.W_k(h_i).view(n_nodes, self.heads, self.head_dim)
            v = self.W_v(h_i).view(n_nodes, self.heads, self.head_dim)

            # Scaled dot-product attention: [H, n, n]
            attn_logits = torch.einsum('nhd,mhd->hnm', q, k) * self.scale

            # SPD bias
            if self.use_spd_bias and spd_i is not None:
                spd_clipped = spd_i.clamp(0, 31).long()
                spd_onehot = F.one_hot(spd_clipped, num_classes=32).float()  # [n, n, 32]

                for h_idx in range(self.heads):
                    bias_flat = self.spd_mlps[h_idx](spd_onehot.view(-1, 32))
                    bias_h = bias_flat.view(n_nodes, n_nodes)
                    attn_logits[h_idx] = attn_logits[h_idx] + bias_h

            # Softmax + dropout
            attn_weights = F.softmax(attn_logits, dim=-1)  # [H, n, n]
            attn_weights = self.attn_dropout(attn_weights)

            # Weighted sum: [H, n, n] @ [n, H, D] -> [n, H, D]
            out_i = torch.einsum('hnm,mhd->nhd', attn_weights, v)
            out_i = out_i.reshape(n_nodes, -1)  # [n, hidden_dim]
            out_i = self.W_o(out_i)

            outputs.append(out_i)
            all_attn_weights.append(attn_weights.detach())

            start_idx += n_nodes

        return torch.cat(outputs, dim=0), all_attn_weights


# ==============================================================================
# Predictor
# ==============================================================================

class Predictor(nn.Module):
    """MLP predictor: mean pool -> 256 -> 128 -> 1."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, h, batch):
        """Mean pool over nodes, then predict."""
        h_graph = global_mean_pool(h, batch)
        return self.mlp(h_graph).squeeze(-1)


# ==============================================================================
# Model Variants
# ==============================================================================

class GPSLiteBaseline(nn.Module):
    """GAT+RWPE+PSPE without global attention (baseline).

    Architecture:
      - GATEncoder (4 stacked layers, ELU, no residual/LayerNorm)
      - Mean pooling + MLP predictor (256 -> 128 -> 1)
    """

    def __init__(
        self,
        in_dim: int = 81,
        edge_dim: int = 10,
        hidden_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GATEncoder(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
        )
        self.predictor = Predictor(hidden_dim=hidden_dim, dropout=0.2)

    def forward(self, x, edge_index, batch, edge_attr=None):
        h = self.encoder(x, edge_index, edge_attr=edge_attr)
        return self.predictor(h, batch)


class GPSLiteNoBias(nn.Module):
    """GAT+RWPE+PSPE + global attention WITHOUT SPD bias.

    Architecture:
      - GATEncoder (4 stacked layers, ELU, no residual/LayerNorm)
      - Global multi-head attention (plain, no SPD bias)
      - Residual + LayerNorm
      - Mean pooling + MLP predictor
    """

    def __init__(
        self,
        in_dim: int = 81,
        edge_dim: int = 10,
        hidden_dim: int = 256,
        heads: int = 4,
        head_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GATEncoder(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
        )
        self.pre_attn_norm = nn.LayerNorm(hidden_dim)
        self.global_attn = GlobalAttentionSPD(
            hidden_dim=hidden_dim,
            heads=heads,
            head_dim=head_dim,
            dropout=dropout,
            use_spd_bias=False,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.predictor = Predictor(hidden_dim=hidden_dim, dropout=0.2)

    def forward(self, x, edge_index, batch, edge_attr=None, spd_list=None):
        h = self.encoder(x, edge_index, edge_attr=edge_attr)
        # Pre-attention LayerNorm for numerical stability
        h_normed = self.pre_attn_norm(h)
        # Global attention
        h_attn, attn_weights = self.global_attn(h_normed, spd_list)
        h = self.attn_norm(h + self.attn_dropout(h_attn))
        return self.predictor(h, batch), attn_weights


class GPSLiteSPD(nn.Module):
    """GAT+RWPE+PSPE + global attention WITH SPD bias (primary experiment).

    Architecture:
      - GATEncoder (4 stacked layers, ELU, no residual/LayerNorm)
      - Global multi-head attention with SPD bias
      - Residual + LayerNorm
      - Mean pooling + MLP predictor
    """

    def __init__(
        self,
        in_dim: int = 81,
        edge_dim: int = 10,
        hidden_dim: int = 256,
        heads: int = 4,
        head_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GATEncoder(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
        )
        self.pre_attn_norm = nn.LayerNorm(hidden_dim)
        self.global_attn = GlobalAttentionSPD(
            hidden_dim=hidden_dim,
            heads=heads,
            head_dim=head_dim,
            dropout=dropout,
            use_spd_bias=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.predictor = Predictor(hidden_dim=hidden_dim, dropout=0.2)

    def forward(self, x, edge_index, batch, edge_attr=None, spd_list=None):
        h = self.encoder(x, edge_index, edge_attr=edge_attr)
        # Pre-attention LayerNorm for numerical stability
        h_normed = self.pre_attn_norm(h)
        # Global attention with SPD bias
        h_attn, attn_weights = self.global_attn(h_normed, spd_list)
        h = self.attn_norm(h + self.attn_dropout(h_attn))
        return self.predictor(h, batch), attn_weights


# ==============================================================================
# Factory
# ==============================================================================

def create_model(variant: str, in_dim: int = 81, edge_dim: int = 10,
                 hidden_dim: int = 256, heads: int = 4, head_dim: int = 64,
                 dropout: float = 0.1):
    """Create model for a given variant."""
    if variant == 'baseline':
        return GPSLiteBaseline(
            in_dim=in_dim, edge_dim=edge_dim,
            hidden_dim=hidden_dim, heads=heads, dropout=dropout,
        )
    elif variant == 'no_bias':
        return GPSLiteNoBias(
            in_dim=in_dim, edge_dim=edge_dim,
            hidden_dim=hidden_dim, heads=heads, head_dim=head_dim, dropout=dropout,
        )
    elif variant == 'spd':
        return GPSLiteSPD(
            in_dim=in_dim, edge_dim=edge_dim,
            hidden_dim=hidden_dim, heads=heads, head_dim=head_dim, dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
