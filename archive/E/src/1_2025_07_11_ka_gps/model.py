"""
Model module for KA-GPS: Kolmogorov-Arnold Enhanced Graph Transformer.

Contains:
- FourierKANLayer: Fourier-series KAN module
- KANBlock: LayerNorm + KAN + residual
- MLPBlock: Standard MLP (for ablation control)
- GINKANLayer: Local MPNN with KAN
- GINMLPLayer: Local MPNN with MLP
- TransformerKANLayer: Global attention with KAN
- TransformerMLPLayer: Global attention with MLP
- KAGPS: Full KA-GPS model
- MLPGPS: Standard GraphGPS with MLP (ablation baseline)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj


# ══════════════════════════════════════════════════════════════════════
# Fourier KAN Layer
# ══════════════════════════════════════════════════════════════════════

class FourierKANLayer(nn.Module):
    """Fourier-series KAN layer.

    For input x of shape (batch, d_in), output of shape (batch, d_out):
        phi_{i,j}(x_i) = sum_{n=1}^{N_freq} [a_{i,j,n} * cos(2*pi*n*x_i) + b_{i,j,n} * sin(2*pi*n*x_i)]
        output_j = sum_i phi_{i,j}(x_i)
    """

    def __init__(self, d_in, d_out, n_freq=5):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_freq = n_freq

        # Learnable Fourier coefficients
        self.a = nn.Parameter(torch.randn(d_in, d_out, n_freq) * 0.01)
        self.b = nn.Parameter(torch.randn(d_in, d_out, n_freq) * 0.01)

        # Frequency indices: 1, 2, ..., n_freq
        self.register_buffer('freqs', torch.arange(1, n_freq + 1, dtype=torch.float32))

    def forward(self, x):
        """x: (batch, d_in) -> (batch, d_out)"""
        batch_size = x.size(0)

        # x: (batch, d_in, 1)
        x_expanded = x.unsqueeze(-1)

        # freqs: (n_freq,) -> scale input
        # x_expanded * freqs: (batch, d_in, n_freq)
        angles = 2 * math.pi * x_expanded * self.freqs  # (batch, d_in, n_freq)

        cos_vals = torch.cos(angles)  # (batch, d_in, n_freq)
        sin_vals = torch.sin(angles)  # (batch, d_in, n_freq)

        # a: (d_in, d_out, n_freq), cos_vals: (batch, d_in, n_freq)
        # einsum: for each (i,j,n), sum over n of a[i,j,n]*cos(batch,i,n)
        # Result: (batch, d_in, d_out)
        cos_part = torch.einsum('bin,ion->bio', cos_vals, self.a)
        sin_part = torch.einsum('bin,ion->bio', sin_vals, self.b)

        phi = cos_part + sin_part  # (batch, d_in, d_out)

        # Sum over input dimension
        output = phi.sum(dim=1)  # (batch, d_out)

        return output


# ══════════════════════════════════════════════════════════════════════
# KAN Block and MLP Block (building blocks for both models)
# ══════════════════════════════════════════════════════════════════════

class KANBlock(nn.Module):
    """Two-layer KAN block with LayerNorm and residual connection.

    Architecture: LayerNorm -> KAN1 -> LayerNorm -> KAN2 + residual
    """

    def __init__(self, d_in, d_hidden, d_out, n_freq=5):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_in)
        self.kan1 = FourierKANLayer(d_in, d_hidden, n_freq)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.kan2 = FourierKANLayer(d_hidden, d_out, n_freq)

        # Residual projection if dimensions don't match
        self.residual_proj = None
        if d_in != d_out:
            self.residual_proj = nn.Linear(d_in, d_out)

    def forward(self, x):
        """x: (batch, d_in) -> (batch, d_out)"""
        residual = x
        out = self.norm1(x)
        out = self.kan1(out)
        out = self.norm2(out)
        out = self.kan2(out)

        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        return out + residual


class MLPBlock(nn.Module):
    """Two-layer MLP block with LayerNorm and residual connection (for ablation)."""

    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_in)
        self.linear1 = nn.Linear(d_in, d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_out)

        self.residual_proj = None
        if d_in != d_out:
            self.residual_proj = nn.Linear(d_in, d_out)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = F.relu(self.linear1(out))
        out = self.norm2(out)
        out = self.linear2(out)

        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        return out + residual


# ══════════════════════════════════════════════════════════════════════
# GIN Layer (Local MPNN) — KAN and MLP variants
# ══════════════════════════════════════════════════════════════════════

class GINKANLayer(nn.Module):
    """GIN message passing with KAN-based update.

    h_v' = KAN_block((1+eps) * h_v + sum_{u in N(v)} h_u)
    """

    def __init__(self, dim, hidden_dim, n_freq=5):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(0.0))
        self.block = KANBlock(dim, hidden_dim, dim, n_freq)

    def forward(self, x, edge_index):
        """x: (N, dim), edge_index: (2, E)"""
        # Message passing: aggregate neighbor features
        row, col = edge_index

        # Sum aggregation of neighbor features
        num_nodes = x.size(0)
        agg = torch.zeros_like(x)
        agg.index_add_(0, col, x[row])

        # GIN update: (1 + eps) * x + aggregated
        out = (1 + self.eps) * x + agg

        # Apply KAN block (node-wise)
        out = self.block(out)

        return out


class GINMLPLayer(nn.Module):
    """GIN message passing with MLP-based update (for ablation)."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(0.0))
        self.block = MLPBlock(dim, hidden_dim, dim)

    def forward(self, x, edge_index):
        row, col = edge_index
        num_nodes = x.size(0)
        agg = torch.zeros_like(x)
        agg.index_add_(0, col, x[row])
        out = (1 + self.eps) * x + agg
        out = self.block(out)
        return out


# ══════════════════════════════════════════════════════════════════════
# Transformer Layer (Global Attention) — KAN and MLP variants
# ══════════════════════════════════════════════════════════════════════

class TransformerKANLayer(nn.Module):
    """Transformer block with multi-head attention and KAN FFN.

    Architecture: MultiHeadAttention -> Add&Norm -> KAN_block -> Add&Norm
    """

    def __init__(self, dim, num_heads=4, n_freq=5, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        # KAN block replaces MLP
        self.ffn = KANBlock(dim, dim * 2, dim, n_freq)

    def forward(self, x, batch, mask=None):
        """x: (N_total, dim), batch: (N_total,) assigning nodes to graphs.

        We convert to dense batch for attention, then back.
        """
        # Convert to dense: (B, max_nodes, dim)
        x_dense, mask_dense = to_dense_batch(x, batch)
        B, N_max, D = x_dense.shape

        if mask is None:
            mask = mask_dense  # (B, N_max) True for real nodes

        # Multi-head self-attention
        residual = x_dense

        Q = self.q_proj(x_dense)  # (B, N, D)
        K = self.k_proj(x_dense)
        V = self.v_proj(x_dense)

        # Reshape to (B, heads, N, head_dim)
        Q = Q.view(B, N_max, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N_max, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N_max, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask
        attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
        attn_weights = attn_weights.masked_fill(~attn_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # (B, heads, N, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N_max, D)
        attn_output = self.out_proj(attn_output)

        # Add & Norm
        x_dense = self.norm1(residual + self.dropout(attn_output))

        # FFN (KAN block)
        residual = x_dense
        # For KAN block, we process each node independently (reshape to 2D)
        x_2d = x_dense.view(B * N_max, D)
        ffn_out = self.ffn(x_2d)
        ffn_out = ffn_out.view(B, N_max, D)

        x_dense = self.norm2(residual + self.dropout(ffn_out))

        # Convert back to sparse
        # Only take real nodes
        out = x_dense[mask_dense]

        return out


class TransformerMLPLayer(nn.Module):
    """Transformer block with MLP FFN (for ablation)."""

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = MLPBlock(dim, dim * 2, dim)

    def forward(self, x, batch, mask=None):
        x_dense, mask_dense = to_dense_batch(x, batch)
        B, N_max, D = x_dense.shape

        if mask is None:
            mask = mask_dense

        residual = x_dense

        Q = self.q_proj(x_dense).view(B, N_max, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_dense).view(B, N_max, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_dense).view(B, N_max, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_mask = mask.unsqueeze(1).unsqueeze(2)
        attn_weights = attn_weights.masked_fill(~attn_mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N_max, D)
        attn_output = self.out_proj(attn_output)

        x_dense = self.norm1(residual + self.dropout(attn_output))

        residual = x_dense
        x_2d = x_dense.view(B * N_max, D)
        ffn_out = self.ffn(x_2d)
        ffn_out = ffn_out.view(B, N_max, D)

        x_dense = self.norm2(residual + self.dropout(ffn_out))

        out = x_dense[mask_dense]
        return out


# ══════════════════════════════════════════════════════════════════════
# GPS Layer: Interleaved Local + Global with Gated Fusion
# ══════════════════════════════════════════════════════════════════════

class GPSKANLayer(nn.Module):
    """One KA-GPS layer: GIN-KAN (local) + Transformer-KAN (global) + gated fusion."""

    def __init__(self, dim, hidden_dim, num_heads=4, n_freq=5, dropout=0.1):
        super().__init__()
        self.local_nn = GINKANLayer(dim, hidden_dim, n_freq)
        self.global_nn = TransformerKANLayer(dim, num_heads, n_freq, dropout)

        # Gated fusion
        self.gate_local = nn.Parameter(torch.tensor(0.5))
        self.gate_global = nn.Parameter(torch.tensor(0.5))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, edge_index, batch):
        # Local MPNN
        h_local = self.local_nn(x, edge_index)

        # Global attention
        h_global = self.global_nn(x, batch)

        # Gated fusion with sigmoid
        alpha = torch.sigmoid(self.gate_local)
        beta = torch.sigmoid(self.gate_global)

        out = alpha * h_local + beta * h_global
        out = self.norm(out)

        # Residual connection
        return out + x


class GPSMLPLayer(nn.Module):
    """One MLP-GPS layer: GIN-MLP (local) + Transformer-MLP (global) + gated fusion."""

    def __init__(self, dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.local_nn = GINMLPLayer(dim, hidden_dim)
        self.global_nn = TransformerMLPLayer(dim, num_heads, dropout)

        self.gate_local = nn.Parameter(torch.tensor(0.5))
        self.gate_global = nn.Parameter(torch.tensor(0.5))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, edge_index, batch):
        h_local = self.local_nn(x, edge_index)
        h_global = self.global_nn(x, batch)

        alpha = torch.sigmoid(self.gate_local)
        beta = torch.sigmoid(self.gate_global)

        out = alpha * h_local + beta * h_global
        out = self.norm(out)

        return out + x


# ══════════════════════════════════════════════════════════════════════
# Full Model: KA-GPS and MLP-GPS
# ══════════════════════════════════════════════════════════════════════

class KAGPS(nn.Module):
    """Kolmogorov-Arnold Enhanced Graph Transformer for Polymer Property Prediction.

    Architecture:
    1. Input projection (atom features + RWSE -> hidden dim)
    2. 4x GPS-KAN layers (interleaved GIN-KAN + Transformer-KAN)
    3. Global pooling (mean + sum + max)
    4. KAN-based prediction head
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=128, num_layers=4,
                 num_heads=4, n_freq=5, dropout=0.1):
        super().__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.node_proj = nn.Linear(node_dim, hidden_dim)

        # GPS-KAN layers
        self.layers = nn.ModuleList([
            GPSKANLayer(hidden_dim, hidden_dim * 2, num_heads, n_freq, dropout)
            for _ in range(num_layers)
        ])

        # Global pooling: mean + sum + max -> 3 * hidden_dim
        self.pool_dim = hidden_dim * 3

        # KAN-based prediction head
        self.head = KANBlock(self.pool_dim, hidden_dim, 1, n_freq)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Input projection
        x = self.node_proj(x)

        # GPS-KAN layers
        for layer in self.layers:
            x = layer(x, edge_index, batch)

        # Multi-pooling
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x_max = global_max_pool(x, batch)

        x_pooled = torch.cat([x_mean, x_sum, x_max], dim=-1)

        # Prediction head
        out = self.head(x_pooled)

        return out.squeeze(-1)


class MLPGPS(nn.Module):
    """Standard GraphGPS with MLPs (ablation baseline).

    Same architecture as KA-GPS but uses MLP instead of KAN everywhere.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=128, num_layers=4,
                 num_heads=4, dropout=0.1):
        super().__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        self.node_proj = nn.Linear(node_dim, hidden_dim)

        self.layers = nn.ModuleList([
            GPSMLPLayer(hidden_dim, hidden_dim * 2, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.pool_dim = hidden_dim * 3

        # MLP-based prediction head
        self.head = MLPBlock(self.pool_dim, hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.node_proj(x)

        for layer in self.layers:
            x = layer(x, edge_index, batch)

        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x_max = global_max_pool(x, batch)

        x_pooled = torch.cat([x_mean, x_sum, x_max], dim=-1)

        out = self.head(x_pooled)

        return out.squeeze(-1)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model creation
    from dataset import ATOM_FEATURE_DIM, EDGE_FEATURE_DIM, RWSE_DIM

    node_dim = ATOM_FEATURE_DIM + RWSE_DIM  # 24 + 6 = 30
    edge_dim = EDGE_FEATURE_DIM  # 6

    print(f"Node feature dim: {node_dim}")
    print(f"Edge feature dim: {edge_dim}")

    model_kan = KAGPS(node_dim, edge_dim, hidden_dim=128, num_layers=4, n_freq=5)
    model_mlp = MLPGPS(node_dim, edge_dim, hidden_dim=128, num_layers=4)

    print(f"\nKA-GPS parameters: {count_parameters(model_kan):,}")
    print(f"MLP-GPS parameters: {count_parameters(model_mlp):,}")
