"""
PolyKAN-SSM v2: Improved architecture with GINE backbone, Fourier-KAN activations,
SSM-inspired gradient flow, and stronger regularization.

Key improvements over v1:
1. GINEConv backbone for proven message passing with edge features
2. Virtual node for global graph information
3. Stronger regularization (dropout 0.25, weight decay 5e-4)
4. Better feature engineering (more atom features)
5. Ensemble via test-time prediction averaging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool
from torch_geometric.nn.models import MLP


class FourierKANLayer(nn.Module):
    """
    KAN-style Fourier activation layer.
    Adds learnable Fourier basis functions on top of a linear layer.
    """

    def __init__(self, in_features, out_features, num_frequencies=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies

        # Main linear path
        self.linear = nn.Linear(in_features, out_features)

        # Fourier coefficients
        self.freq_proj = nn.Linear(in_features, num_frequencies)
        self.cos_weights = nn.Parameter(
            torch.randn(num_frequencies, out_features) * 0.1
        )
        self.sin_weights = nn.Parameter(
            torch.randn(num_frequencies, out_features) * 0.1
        )

    def forward(self, x):
        out = self.linear(x)
        freq = self.freq_proj(x)
        for k in range(self.num_frequencies):
            out = out + torch.cos(freq[:, k:k+1]) * self.cos_weights[k:k+1, :]
            out = out + torch.sin(freq[:, k:k+1]) * self.sin_weights[k:k+1, :]
        return out


class PolyKANSSM_v2(nn.Module):
    """
    PolyKAN-SSM v2 with GINE backbone + Fourier-KAN + SSM residual connections.

    Architecture:
    1. Node/Edge embedding
    2. Virtual node for global information
    3. GINE layers with Fourier-KAN internal activations
    4. SSM-inspired exponential decay/growth residual
    5. Multi-head attention + mean pooling
    6. MLP prediction head
    """

    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim=256,
        num_layers=5,
        dropout=0.25,
        spectral_radius=0.99,
        num_frequencies=4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )

        # Edge embedding
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )

        # Virtual node embedding
        self.vn_embed = nn.Embedding(1, hidden_dim)
        nn.init.constant_(self.vn_embed.weight.data, 0)

        # GINE layers with custom neural networks
        self.convs = nn.ModuleList()
        self.vn_update = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for i in range(num_layers):
            # GINE uses an internal NN; we make it a small MLP with SiLU
            nn_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            conv = GINEConv(nn_layer, edge_dim=hidden_dim, train_eps=True)
            self.convs.append(conv)

            # Virtual node update
            self.vn_update.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))

            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout))

        # SSM-inspired per-layer decay parameter (diagonal Lambda)
        self.log_lambda = nn.Parameter(
            torch.full((num_layers, hidden_dim), math.log(spectral_radius))
        )

        # Fourier-KAN post-processing
        self.kan_layer = FourierKANLayer(hidden_dim, hidden_dim, num_frequencies)

        # Attention pooling
        self.attn_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch, batch_size):
        device = x.device

        # Embed nodes and edges
        h = self.node_embed(x)
        edge_h = self.edge_embed(edge_attr)

        # Virtual node
        vn_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        vn_h = self.vn_embed(vn_idx)  # [batch_size, hidden_dim]

        # GINE layers with SSM residual and virtual node
        for i in range(self.num_layers):
            # Add virtual node info to nodes
            vn_expanded = vn_h[batch]  # [num_nodes, hidden_dim]
            h_in = h + vn_expanded

            # GINE message passing
            h_out = self.convs[i](h_in, edge_index, edge_h)

            # SSM-inspired update: lambda * h_old + h_new
            lam = torch.exp(self.log_lambda[i])  # [hidden_dim]
            h_ssm = h * lam + h_out

            # Residual + norm + dropout
            h = self.dropout_layers[i](self.layer_norms[i](h + h_ssm))

            # Update virtual node
            vn_agg = global_mean_pool(h, batch, batch_size)
            vn_h = vn_h + self.vn_update[i](vn_agg)

        # Fourier-KAN enhancement
        h = h + self.kan_layer(h)

        # Attention pooling
        attn = torch.sigmoid(self.attn_nn(h))  # [num_nodes, 1]
        h_weighted = h * attn
        graph_attn = global_add_pool(h_weighted, batch, batch_size)

        # Mean pooling
        graph_mean = global_mean_pool(h, batch, batch_size)

        # Concatenate both representations
        graph_repr = torch.cat([graph_attn, graph_mean], dim=-1)

        # Predict
        out = self.head(graph_repr)
        return out.squeeze(-1)
