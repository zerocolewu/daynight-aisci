"""
CPA-GAT: Cardinality-Preserving Attention for polymer Tg prediction.
Adds a sigmoid-gated unnormalized sum channel alongside standard GATv2 attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_scatter import scatter


class CPAGATv2Conv(nn.Module):
    """
    Custom GATv2 layer with Cardinality-Preserving Attention (CPA) channel.
    Uses N_heads separate GATv2Conv sub-layers (heads=1 each) for per-head CPA access.
    """

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.head_dim = out_dim  # dim per head
        self.heads = heads

        # Per-head GATv2Conv sub-layers
        self.gat_heads = nn.ModuleList([
            GATv2Conv(in_dim, out_dim, heads=1, concat=True,
                      add_self_loops=True, dropout=dropout)
            for _ in range(heads)
        ])

        # Per-head gate parameters: W_g^h ∈ R^{d_head × d_in}, b_g^h ∈ R^{d_head}
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
            # Standard GATv2 output (already includes self-loops via add_self_loops=True)
            o_std = self.gat_heads[h](x, edge_index)  # [N, head_dim]

            # CPA sum channel: use SAME value projection (lin_r of GATv2Conv)
            v = self.gat_heads[h].lin_r(x)  # [N, head_dim]
            v_all = v[all_src]  # [E+N, head_dim]
            s = scatter(v_all, all_dst, dim=0, dim_size=n, reduce='sum')  # [N, head_dim]

            # Sigmoid gate: g_i = σ(W_g^h @ h_i + b_g^h)
            g = torch.sigmoid(F.linear(x, self.W_g[h], self.b_g[h]))  # [N, head_dim]

            o_cpa = g * s  # [N, head_dim]

            output = o_std + o_cpa  # [N, head_dim]
            outputs.append(output)

        # Concatenate all heads
        x_out = torch.cat(outputs, dim=-1)  # [N, heads * head_dim]
        return x_out


class CPAGAT(nn.Module):
    """CPA-GAT with CPA channel in every GAT layer."""

    def __init__(self, in_dim=42, hidden_dim=256, out_dim=1,
                 num_layers=4, heads=4, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.head_dim = hidden_dim // heads

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                CPAGATv2Conv(hidden_dim, self.head_dim, heads=heads, dropout=dropout)
            )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.input_proj(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.elu(x)

        x = global_mean_pool(x, batch)
        x = self.predictor(x)
        return x.squeeze(-1)
