"""
PolyKAN-SSM v3: Focus on closing the generalization gap.

Key changes from v1:
- GINE backbone for robust message passing
- Virtual node for global information
- SSM residual with learnable per-layer decay
- Fourier-KAN as activation enhancement
- Stronger regularization (dropout 0.3, weight decay 1e-3)
- Huber loss for robustness
- Cosine annealing with warm restarts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool


class FourierKANLayer(nn.Module):
    """Fourier-KAN layer: linear + learnable Fourier series."""

    def __init__(self, in_features, out_features, num_frequencies=4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.freq_proj = nn.Linear(in_features, num_frequencies)
        self.cos_w = nn.Parameter(torch.randn(num_frequencies, out_features) * 0.05)
        self.sin_w = nn.Parameter(torch.randn(num_frequencies, out_features) * 0.05)

    def forward(self, x):
        out = self.linear(x)
        freq = self.freq_proj(x)
        for k in range(self.cos_w.shape[0]):
            out = out + torch.cos(freq[:, k:k+1]) * self.cos_w[k:k+1, :]
            out = out + torch.sin(freq[:, k:k+1]) * self.sin_w[k:k+1, :]
        return out


class PolyKANSSM_v3(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim=256,
        num_layers=5,
        dropout=0.3,
        spectral_radius=0.99,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Node/edge embeddings
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Virtual node
        self.vn_emb = nn.Parameter(torch.zeros(1, hidden_dim))
        self.vn_mlp = nn.ModuleList()

        # GINE layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(num_layers):
            nn_i = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.LayerNorm(2 * hidden_dim),
                nn.SiLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn_i, edge_dim=hidden_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
            self.vn_mlp.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))

        # SSM decay parameters (per layer)
        self.log_lambdas = nn.ParameterList([
            nn.Parameter(torch.full((hidden_dim,), math.log(spectral_radius)))
            for _ in range(num_layers)
        ])

        # Fourier-KAN mixing after GNN
        self.kan = FourierKANLayer(hidden_dim, hidden_dim, num_frequencies=4)

        # Attention pooling
        self.pool_attn = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        # Prediction head
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

    def forward(self, x, edge_index, edge_attr, batch, batch_size):
        h = self.node_embed(x)
        edge_h = self.edge_embed(edge_attr)

        # Virtual node
        batch_size_val = batch.max().item() + 1
        vn_h = self.vn_emb.expand(batch_size_val, -1)  # [batch_size, hidden]

        for i, (conv, norm, do) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            # Add virtual node info to each node
            vn_per_node = vn_h[batch]
            h_in = h + vn_per_node

            # GINE convolution
            h_msg = conv(h_in, edge_index, edge_h)

            # SSM update: lambda * h + h_msg
            lam = torch.exp(self.log_lambdas[i])
            h_ssm = h * lam + h_msg

            # Residual + norm + dropout
            h = do(norm(h + h_ssm))

            # Update virtual node
            vn_agg = global_mean_pool(h, batch, batch_size_val)
            vn_h = vn_h + self.vn_mlp[i](vn_agg)

        # Fourier-KAN enhancement
        h = h + self.kan(h)

        # Attention pooling (sigmoid gating)
        attn = torch.sigmoid(self.pool_attn(h))
        h_w = h * attn
        graph_attn = global_add_pool(h_w, batch, batch_size_val)
        graph_mean = global_mean_pool(h, batch, batch_size_val)
        graph_repr = graph_attn + graph_mean

        out = self.head(graph_repr)
        return out.squeeze(-1)
