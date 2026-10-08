"""
PolyKAN-SSM: Integrating Fourier-KAN activations, GNN-SSM layers, and Non-covalent edges
for polymer glass transition temperature (Tg) prediction.

Key innovations:
1. Fourier-KAN activations for better function approximation of periodic polymer structures
2. GNN-SSM layer for gradient flow control to address overfitting
3. Non-covalent edges (~5A cutoff) to capture inter-chain interactions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import global_mean_pool, global_add_pool


class FourierKANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network layer with Fourier basis functions.
    Replaces fixed activation with learnable Fourier series.

    y = c_0 + sum_{k=1}^{K} (a_k * cos(2*pi*k*x / T) + b_k * sin(2*pi*k*x / T))

    The input is first projected linearly, then passed through Fourier series.
    This is more stable than applying Fourier directly to raw features.
    """

    def __init__(self, in_features, out_features, num_frequencies=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies

        # Linear projection (residual-style)
        self.base_linear = nn.Linear(in_features, out_features)

        # Learnable Fourier coefficients - initialize small
        self.cos_weights = nn.Parameter(
            torch.randn(num_frequencies, out_features) * (1.0 / math.sqrt(num_frequencies))
        )
        self.sin_weights = nn.Parameter(
            torch.randn(num_frequencies, out_features) * (1.0 / math.sqrt(num_frequencies))
        )

        # Per-frequency input projection
        self.freq_proj = nn.Linear(in_features, num_frequencies, bias=True)

    def forward(self, x):
        # x: [batch, in_features]
        # Base linear path
        out = self.base_linear(x)

        # Fourier path: project input to frequency domain
        freq_input = self.freq_proj(x)  # [batch, num_frequencies]

        for k in range(self.num_frequencies):
            out = out + torch.cos(freq_input[:, k:k+1]) * self.cos_weights[k:k+1, :]
            out = out + torch.sin(freq_input[:, k:k+1]) * self.sin_weights[k:k+1, :]

        return out


class GNNSSMLayer(nn.Module):
    """
    GNN with State-Space Model for gradient flow control.

    The SSM update: H^(l+1) = Lambda * H^(l) + B * Aggregate(Messages)
    with Lambda having controlled spectral radius to prevent gradient vanishing.
    """

    def __init__(self, hidden_dim, spectral_radius=0.99, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # State transition: use diagonal Lambda for efficiency and stability
        # Initialize log-space for stable training
        log_radius = math.log(spectral_radius)
        self.log_lambda = nn.Parameter(torch.full((hidden_dim,), log_radius))

        # Input transformation B
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Message function
        self.message_fn = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_attr=None):
        """
        h: [num_nodes, hidden_dim]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, hidden_dim] or None
        """
        src, dst = edge_index

        # Build message input
        if edge_attr is not None:
            msg_input = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        else:
            msg_input = torch.cat([h[src], h[dst]], dim=-1)

        # Compute messages
        messages = self.message_fn(msg_input)

        # Aggregate messages (mean aggregation)
        aggr = torch.zeros_like(h)
        count = torch.zeros(h.shape[0], 1, device=h.device)
        aggr.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.hidden_dim), messages)
        count.scatter_add_(0, dst.unsqueeze(-1), torch.ones(dst.shape[0], 1, device=h.device))
        aggr = aggr / count.clamp(min=1)

        # SSM update with diagonal Lambda
        lambda_diag = torch.exp(self.log_lambda)  # ensure positive, controlled magnitude
        h_ssm = h * lambda_diag + self.B(aggr)

        # Residual + norm
        h_new = self.norm(h + self.dropout(h_ssm))
        return h_new


class PolyKANSSM(nn.Module):
    """
    PolyKAN-SSM: Integrating Fourier-KAN, GNN-SSM, and Non-covalent edges.

    Architecture:
    1. Node/Edge embedding with Fourier-KAN
    2. N GNN-SSM layers for message passing with gradient control
    3. Multi-head attention pooling for intensive property (Tg)
    4. MLP head with Fourier-KAN activations
    """

    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim=128,
        num_layers=4,
        dropout=0.15,
        spectral_radius=0.99,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Edge embedding
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Input Fourier-KAN projection for nodes
        self.node_kan = FourierKANLayer(hidden_dim, hidden_dim, num_frequencies=4)

        # GNN-SSM layers
        self.ssm_layers = nn.ModuleList(
            [
                GNNSSMLayer(hidden_dim, spectral_radius, dropout)
                for _ in range(num_layers)
            ]
        )

        # Fourier-KAN layers after each SSM layer (for mixing)
        self.kan_mixers = nn.ModuleList(
            [
                FourierKANLayer(hidden_dim, hidden_dim, num_frequencies=4)
                for _ in range(num_layers)
            ]
        )

        # Attention-based pooling
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Prediction head with Fourier-KAN
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Fourier-KAN for final prediction refinement
        self.kan_head = FourierKANLayer(hidden_dim, hidden_dim // 2, num_frequencies=4)

    def forward(self, x, edge_index, edge_attr, batch, batch_size):
        # Embed nodes and edges
        h = self.node_embed(x)
        h = h + self.node_kan(h)  # Fourier-KAN refinement

        edge_h = self.edge_embed(edge_attr) if edge_attr is not None else None

        # GNN-SSM layers with Fourier-KAN mixing
        for ssm_layer, kan_mixer in zip(self.ssm_layers, self.kan_mixers):
            h = ssm_layer(h, edge_index, edge_h)
            h = h + kan_mixer(h)  # Fourier-KAN post-mixing

        # Attention-based pooling
        attn_logits = self.pool_gate(h)  # [num_nodes, 1]
        attn = torch.sigmoid(attn_logits)  # soft gating

        # Weighted sum pooling
        h_weighted = h * attn
        graph_attn = global_add_pool(h_weighted, batch, batch_size)

        # Also add mean pool for stability
        graph_mean = global_mean_pool(h, batch, batch_size)

        # Combine attention and mean pooling
        graph_repr = graph_attn + graph_mean

        # Fourier-KAN enhanced prediction
        kan_feat = self.kan_head(graph_repr)
        out = self.head(graph_repr) + kan_feat.mean(dim=-1, keepdim=True)

        return out.squeeze(-1)
