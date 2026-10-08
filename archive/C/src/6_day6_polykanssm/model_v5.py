"""
PolyKAN-SSM v5: Balanced architecture combining best insights from v3 and v4.

Key design:
1. GINEConv backbone (proven in v3)
2. Fourier-KAN as auxiliary activation (not dominant)
3. JumpingKnowledge for multi-scale features
4. Virtual node with lightweight updates
5. SSM-inspired skip connections (lightweight, not restrictive)
6. Strong regularization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, JumpingKnowledge


class FourierKANLayer(nn.Module):
    """Lightweight Fourier-KAN: linear + small Fourier perturbation."""
    def __init__(self, in_features, out_features, num_frequencies=4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.num_freq = num_frequencies
        # Learnable frequency and phase
        self.freq_weight = nn.Parameter(torch.randn(in_features, num_frequencies) * 0.1)
        self.cos_coeff = nn.Parameter(torch.randn(num_frequencies, out_features) * 0.01)
        self.sin_coeff = nn.Parameter(torch.randn(num_frequencies, out_features) * 0.01)

    def forward(self, x):
        base = self.linear(x)
        freq = x @ self.freq_weight  # [N, num_freq]
        fourier = (torch.cos(freq) @ self.cos_coeff + torch.sin(freq) @ self.sin_coeff)
        return base + fourier


class PolyKANSSM_v5(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim=256,
        num_layers=4,
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
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Edge embedding
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Virtual node
        self.vn_emb = nn.Parameter(torch.zeros(1, hidden_dim))
        self.vn_updates = nn.ModuleList()

        # GINE layers with LayerNorm
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # SSM-inspired per-layer decay (like v3 but with clamping)
        self.log_lambdas = nn.ParameterList()

        # Fourier-KAN post-processing (optional, controlled by gate)
        self.kan_gate = nn.Parameter(torch.tensor(0.1))  # starts small
        self.kan = FourierKANLayer(hidden_dim, hidden_dim, num_frequencies)

        for i in range(num_layers):
            nn_i = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.LayerNorm(2 * hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn_i, edge_dim=hidden_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
            self.vn_updates.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            ))
            self.log_lambdas.append(
                nn.Parameter(torch.full((hidden_dim,), math.log(spectral_radius)))
            )

        # JumpingKnowledge to combine all layers
        self.jk = JumpingKnowledge(mode='cat', channels=hidden_dim, num_layers=num_layers)
        jk_dim = hidden_dim * num_layers

        # Attention pooling
        self.pool_attn = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch, batch_size):
        h = self.node_embed(x)
        edge_h = self.edge_embed(edge_attr)

        batch_size_val = batch.max().item() + 1
        vn_h = self.vn_emb.expand(batch_size_val, -1)

        layer_outputs = []

        for i, (conv, norm, do) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            # Add virtual node info
            vn_per_node = vn_h[batch]
            h_in = h + vn_per_node

            # GINE convolution
            h_msg = conv(h_in, edge_index, edge_h)

            # SSM-inspired update: decayed skip + new message
            lam = torch.exp(self.log_lambdas[i]).clamp(max=0.999)
            h_ssm = h * lam + h_msg

            # Residual + norm + dropout
            h = do(norm(h + h_ssm))

            # Store for JK
            layer_outputs.append(h)

            # Update virtual node
            vn_agg = global_mean_pool(h, batch, batch_size_val)
            vn_h = vn_h + self.vn_updates[i](vn_agg)

        # Fourier-KAN with gated contribution (applied to last layer)
        kan_out = self.kan(h)
        h_final = h + torch.sigmoid(self.kan_gate) * kan_out

        # Replace last layer output with Fourier-enhanced version
        layer_outputs[-1] = h_final

        # JumpingKnowledge
        h_jk = self.jk(layer_outputs)

        # Attention pooling
        attn = torch.sigmoid(self.pool_attn(h_jk))
        h_w = h_jk * attn
        graph_attn = global_add_pool(h_w, batch, batch_size_val)
        graph_mean = global_mean_pool(h_jk, batch, batch_size_val)
        graph_repr = graph_attn + graph_mean  # Sum, not concatenation

        out = self.head(graph_repr)
        return out.squeeze(-1)
