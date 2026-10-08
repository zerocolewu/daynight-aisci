"""
CPA-GATv2Conv: Custom GATv2 layer with Cardinality-Preserving Attention channel.
Reuses the exact implementation pattern from 7_day7_cpa_pspe/model.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter


class CPAGATv2Conv(nn.Module):
    """
    Custom GATv2 layer with Cardinality-Preserving Attention (CPA) channel.
    Uses N_heads separate GATv2Conv sub-layers (heads=1 each) for per-head CPA access.

    Formula per head h:
      o_std = GATv2Conv(x, edge_index)
      v = lin_r(x)                     # shared value projection
      s_i = sum_{j in N(i) U {i}} v_j  # scatter sum
      g_i = sigmoid(W_g @ x_i + b_g)   # gate
      o_cpa = g * s
      output = o_std + o_cpa
    """

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.head_dim = out_dim
        self.heads = heads

        # Per-head GATv2Conv sub-layers
        self.gat_heads = nn.ModuleList([
            GATv2Conv(in_dim, out_dim, heads=1, concat=True,
                      add_self_loops=True, dropout=dropout)
            for _ in range(heads)
        ])

        # Per-head gate parameters
        self.W_g = nn.ParameterList([
            nn.Parameter(torch.Tensor(out_dim, in_dim))
            for _ in range(heads)
        ])
        self.b_g = nn.ParameterList([
            nn.Parameter(torch.Tensor(out_dim))
            for _ in range(heads)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for h in range(self.heads):
            nn.init.xavier_uniform_(self.W_g[h])
            nn.init.zeros_(self.b_g[h])

    def forward(self, x, edge_index):
        n = x.size(0)
        device = x.device

        # Build edge_index with self-loops for CPA sum
        self_idx = torch.arange(n, device=device)
        all_src = torch.cat([edge_index[0], self_idx])
        all_dst = torch.cat([edge_index[1], self_idx])

        outputs = []
        for h in range(self.heads):
            o_std = self.gat_heads[h](x, edge_index)  # [N, head_dim]

            v = self.gat_heads[h].lin_r(x)  # [N, head_dim]
            v_all = v[all_src]
            s = scatter(v_all, all_dst, dim=0, dim_size=n, reduce='sum')  # [N, head_dim]

            g = torch.sigmoid(F.linear(x, self.W_g[h], self.b_g[h]))  # [N, head_dim]

            o_cpa = g * s
            output = o_std + o_cpa
            outputs.append(output)

        x_out = torch.cat(outputs, dim=-1)  # [N, heads * head_dim]
        return x_out
