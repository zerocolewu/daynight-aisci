import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, JumpingKnowledge

class FourierKANLayer(nn.Module):
    def __init__(self, in_features, out_features, num_frequencies=4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.num_freq = num_frequencies
        self.freq_weight = nn.Parameter(torch.randn(in_features, num_frequencies) * 0.1)
        self.cos_coeff = nn.Parameter(torch.randn(num_frequencies, out_features) * 0.01)
        self.sin_coeff = nn.Parameter(torch.randn(num_frequencies, out_features) * 0.01)

    def forward(self, x):
        base = self.linear(x)
        freq = x @ self.freq_weight
        fourier = torch.cos(freq) @ self.cos_coeff + torch.sin(freq) @ self.sin_coeff
        return base + fourier

class PolyKANSSM_v5b(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_layers=4,
                 dropout=0.3, spectral_radius=0.99, num_frequencies=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

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

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.log_lambdas = nn.ParameterList()

        self.kan_gate = nn.Parameter(torch.tensor(0.1))
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
            self.log_lambdas.append(
                nn.Parameter(torch.full((hidden_dim,), math.log(spectral_radius)))
            )

        self.jk = JumpingKnowledge(mode='cat', channels=hidden_dim, num_layers=num_layers)
        jk_dim = hidden_dim * num_layers

        # Attention pooling projects jk_dim to scalar attention weights
        self.pool_attn = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # After pooling: both graph_attn and graph_mean have shape (batch_size, jk_dim)
        # So graph_repr has dimension 2 * jk_dim
        self.head = nn.Sequential(
            nn.Linear(2 * jk_dim, hidden_dim),
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

        layer_outputs = []
        for i, (conv, norm, do) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            h_msg = conv(h, edge_index, edge_h)
            lam = torch.exp(self.log_lambdas[i]).clamp(max=0.999)
            h_ssm = h * lam + h_msg
            h = do(norm(h + h_ssm))
            layer_outputs.append(h)

        kan_out = self.kan(h)
        h = h + torch.sigmoid(self.kan_gate) * kan_out
        # Update last layer output for JK
        layer_outputs[-1] = h

        h_jk = self.jk(layer_outputs)

        attn = torch.sigmoid(self.pool_attn(h_jk))
        h_w = h_jk * attn
        graph_attn = global_add_pool(h_w, batch, batch_size_val)
        graph_mean = global_mean_pool(h_jk, batch, batch_size_val)

        graph_repr = torch.cat([graph_attn, graph_mean], dim=-1)
        out = self.head(graph_repr)
        return out.squeeze(-1)
