"""
GPS-SPSE-Trimer Model: Hybrid MPNN+Transformer with SPSE Structural Encoding.
Architecture: interleaved GPS blocks with GIN MPNN + Transformer with SPSE attention bias.
SPSE is precomputed and cached, with trainable per-block linear projections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GINEConv, global_mean_pool


class SinusoidalSPSE(nn.Module):
    """Non-trainable sinusoidal encoding for shortest path distances."""

    def __init__(self, dim, max_dist=1001):
        super().__init__()
        self.dim = dim
        self.max_dist = max_dist

        encodings = torch.zeros(max_dist, dim)
        position = torch.arange(0, max_dist, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) *
                             -(np.log(10000.0) / dim))
        encodings[:, 0::2] = torch.sin(position * div_term)
        encodings[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('encodings', encodings)

    def forward(self, distances):
        """Get SPSE encoding for distance matrix.
        Args:
            distances: (n, n) LongTensor
        Returns:
            encoding: (n, n, dim) FloatTensor
        """
        clamped = distances.clamp(0, self.max_dist - 1)
        return self.encodings[clamped]


class GPSBlock(nn.Module):
    """Single GPS block: GIN MPNN + per-graph Transformer with SPSE bias + residual."""

    def __init__(self, hidden_dim, transformer_dim, edge_dim, n_heads, spse_dim, dropout_rate, block_idx):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transformer_dim = transformer_dim
        self.n_heads = n_heads
        self.head_dim = transformer_dim // n_heads
        self.dropout_rate = dropout_rate
        self.block_idx = block_idx

        # GINE MPNN (supports edge features)
        gin_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gin_conv = GINEConv(gin_mlp, edge_dim=edge_dim, train_eps=True)
        self.mpnn_norm = nn.LayerNorm(hidden_dim)

        # Trainable SPSE projection: spse_dim -> scalar attention bias
        self.spse_proj = nn.Sequential(
            nn.Linear(spse_dim, 1, bias=True),
        )

        # Transformer projections
        self.q_proj = nn.Linear(hidden_dim, transformer_dim)
        self.k_proj = nn.Linear(hidden_dim, transformer_dim)
        self.v_proj = nn.Linear(hidden_dim, transformer_dim)
        self.out_proj = nn.Linear(transformer_dim, hidden_dim)
        self.transformer_norm = nn.LayerNorm(hidden_dim)

        # Residual fusion
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr, batch, spse_biases, return_attention=False):
        """Forward pass with per-graph Transformer attention + SPSE bias."""
        N = x.size(0)
        device = x.device

        # --- Local MPNN step (batched, uses edge features) ---
        h_local = self.gin_conv(x, edge_index, edge_attr=edge_attr)
        h_local = self.mpnn_norm(h_local)
        h_local = self.dropout(h_local)
        h_local = x + h_local

        # --- Global Transformer step (per-graph) ---
        h_global = torch.zeros_like(x)
        attn_list = []

        for node_indices, spse_enc in spse_biases:
            n_nodes = len(node_indices)

            # Extract per-graph features
            h_graph = h_local[node_indices]  # (n_nodes, hidden_dim)

            # Q, K, V projections
            Q = self.q_proj(h_graph).view(n_nodes, self.n_heads, self.head_dim).transpose(0, 1)
            K = self.k_proj(h_graph).view(n_nodes, self.n_heads, self.head_dim).transpose(0, 1)
            V = self.v_proj(h_graph).view(n_nodes, self.n_heads, self.head_dim).transpose(0, 1)

            scale = self.head_dim ** -0.5
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            # Project SPSE encoding to scalar bias using per-block projection
            # spse_enc: (n_nodes, n_nodes, spse_dim)
            spse_bias = self.spse_proj(spse_enc).squeeze(-1)  # (n_nodes, n_nodes)
            attn_scores = attn_scores + spse_bias.unsqueeze(0)

            attn_weights = F.softmax(attn_scores, dim=-1)

            if return_attention:
                attn_list.append(attn_weights.mean(dim=0).detach())

            h_attn = torch.matmul(attn_weights, V)
            h_attn = h_attn.transpose(0, 1).contiguous().view(n_nodes, self.transformer_dim)
            h_attn = self.out_proj(h_attn)
            h_global[node_indices] = h_attn

        h_global = self.dropout(h_global)
        h_global = self.transformer_norm(h_global)

        # Residual fusion
        out = h_local + h_global
        out = self.fusion_norm(out)

        return out, attn_list


class GPSSPSETrimerModel(nn.Module):
    """GPS-SPSE-Trimer: Hybrid GIN+Transformer with SPSE encoding for Tg prediction."""

    def __init__(self, node_dim, edge_dim, hidden_dim=128, transformer_dim=64,
                 n_heads=4, n_blocks=3, spse_dim=64, num_node_features=30):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.transformer_dim = transformer_dim
        self.spse_dim = spse_dim
        self.num_node_features = num_node_features
        self.edge_dim = edge_dim

        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.spse_encoder = SinusoidalSPSE(spse_dim, max_dist=1001)

        # Progressive dropout: [0.10, 0.15, 0.20, 0.25, 0.30] for 5 blocks
        dropout_rates = [0.10 + 0.05 * i for i in range(n_blocks)]
        self.blocks = nn.ModuleList([
            GPSBlock(hidden_dim, transformer_dim, edge_dim, n_heads, spse_dim, dropout_rates[i], i)
            for i in range(n_blocks)
        ])

        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.denoise_head = nn.Linear(hidden_dim, num_node_features)

    def forward(self, data, spse_cache=None, graph_indices=None,
                mask_ratio=0.0, return_attention=False):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        n_nodes = x.size(0)

        # Denoising: mask node features
        masked_indices = None
        original_features = None
        if mask_ratio > 0 and self.training:
            mask = torch.rand(n_nodes) < mask_ratio
            masked_indices = mask.nonzero(as_tuple=True)[0]
            if len(masked_indices) > 0:
                original_features = x[masked_indices].clone()
                x = x.clone()
                x[masked_indices] = 0.0

        h = self.input_proj(x)

        # Precompute SPSE encoding for each graph once (shared across blocks)
        spse_biases = self._precompute_spse(edge_index, batch, spse_cache, graph_indices)

        # GPS blocks
        all_attn = []
        for block in self.blocks:
            h, attn_list = block(h, edge_index, edge_attr, batch, spse_biases,
                                 return_attention=return_attention)
            if return_attention and len(attn_list) > 0:
                all_attn.extend(attn_list)

        pooled = global_mean_pool(h, batch)
        pred = self.pred_head(pooled)

        denoise_loss = torch.tensor(0.0, device=x.device)
        if masked_indices is not None and len(masked_indices) > 0:
            reconstructed = self.denoise_head(h[masked_indices])
            denoise_loss = F.mse_loss(reconstructed, original_features)

        return pred, denoise_loss, all_attn

    def _precompute_spse(self, edge_index, batch, spse_cache, graph_indices):
        """Precompute SPSE sinusoidal encoding for each graph.
        Returns the full (n, n, spse_dim) encoding - projection is done per-block.
        """
        device = edge_index.device
        unique_graphs = torch.unique(batch)
        spse_biases = []

        for i, g_idx in enumerate(unique_graphs):
            node_mask = (batch == g_idx)
            node_indices = node_mask.nonzero(as_tuple=True)[0]

            # Look up precomputed distances
            sp_dist = spse_cache[graph_indices[i]].to(device)
            spse_enc = self.spse_encoder(sp_dist)  # (n, n, spse_dim)
            spse_biases.append((node_indices, spse_enc))

        return spse_biases

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mpnn_params = 0
        transformer_params = 0
        spse_params = 0
        other_params = 0
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'gin_conv' in name or 'mpnn_norm' in name:
                mpnn_params += p.numel()
            elif 'spse_proj' in name:
                spse_params += p.numel()
            elif 'q_proj' in name or 'k_proj' in name or 'v_proj' in name or \
                    'out_proj' in name or 'transformer_norm' in name:
                transformer_params += p.numel()
            else:
                other_params += p.numel()
        return {
            'total': total,
            'mpnn': mpnn_params,
            'transformer': transformer_params,
            'spse': spse_params,
            'other': other_params,
        }
