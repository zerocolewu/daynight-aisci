"""
Bond CPA-GAT encoder: 3 layers, 128 hidden dim, 4 attention heads.
CPA in ALL 3 layers. No residuals, no LayerNorm, ELU activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from .cpa_gat import CPAGATv2Conv


class BondCPAGATEncoder(nn.Module):
    """CPA-GAT encoder operating on bond-level (line) graphs."""

    def __init__(self, in_dim=14, hidden_dim=128, num_layers=3, heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.head_dim = hidden_dim // heads

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                CPAGATv2Conv(hidden_dim, self.head_dim, heads=heads)
            )

    def forward(self, x, edge_index, batch=None):
        x = self.input_proj(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)

        if batch is not None:
            x = global_mean_pool(x, batch)
        return x

    def forward_with_gates(self, x, edge_index):
        """Forward pass that collects gate values from each layer."""
        x = self.input_proj(x)

        gate_values = []
        for conv in self.convs:
            n = x.size(0)
            device = x.device
            self_idx = torch.arange(n, device=device)
            all_src = torch.cat([edge_index[0], self_idx])
            all_dst = torch.cat([edge_index[1], self_idx])

            head_outputs = []
            layer_gates = []
            for h in range(conv.heads):
                o_std = conv.gat_heads[h](x, edge_index)
                v = conv.gat_heads[h].lin_r(x)
                v_all = v[all_src]
                from torch_scatter import scatter
                s = scatter(v_all, all_dst, dim=0, dim_size=n, reduce='sum')
                g = torch.sigmoid(F.linear(x, conv.W_g[h], conv.b_g[h]))
                o_cpa = g * s
                layer_gates.append(g)
                head_outputs.append(o_std + o_cpa)

            gate_values.append(torch.stack(layer_gates, dim=1))
            x = torch.cat(head_outputs, dim=-1)
            x = F.elu(x)

        return x, gate_values


class BondGATEncoder(nn.Module):
    """Plain GAT encoder for bond graphs (no CPA)."""

    def __init__(self, in_dim=14, hidden_dim=128, num_layers=3, heads=4):
        super().__init__()
        self.num_layers = num_layers

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                          concat=True, add_self_loops=True, dropout=0.0)
            )

    def forward(self, x, edge_index, batch=None):
        x = self.input_proj(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)

        if batch is not None:
            x = global_mean_pool(x, batch)
        return x
