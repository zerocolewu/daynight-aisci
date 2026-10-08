"""
Model definitions for CPA-PSE experiment.

Models:
  - CPAGAT: CPA-GAT (from day6 mem_146) - cardinality-preserving attention
  - NonCPAGAT: Standard GATv2Conv (no CPA gate)
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
    EXACT match to mem_146 implementation.
    """

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.head_dim = out_dim
        self.heads = heads

        self.gat_heads = nn.ModuleList([
            GATv2Conv(in_dim, out_dim, heads=1, concat=True,
                      add_self_loops=True, dropout=dropout)
            for _ in range(heads)
        ])

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

        self_idx = torch.arange(n, device=device)
        all_src = torch.cat([edge_index[0], self_idx])
        all_dst = torch.cat([edge_index[1], self_idx])

        outputs = []
        for h in range(self.heads):
            o_std = self.gat_heads[h](x, edge_index)  # [N, head_dim]
            v = self.gat_heads[h].lin_r(x)  # [N, head_dim]
            v_all = v[all_src]  # [E+N, head_dim]
            s = scatter(v_all, all_dst, dim=0, dim_size=n, reduce='sum')  # [N, head_dim]
            g = torch.sigmoid(F.linear(x, self.W_g[h], self.b_g[h]))  # [N, head_dim]
            o_cpa = g * s  # [N, head_dim]
            output = o_std + o_cpa  # [N, head_dim]
            outputs.append(output)

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

    def get_cpa_gate_stats(self, data):
        """Collect CPA gate statistics for diagnostic analysis."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.input_proj(x)

        stats = {}
        gate_values_all = []

        n = x.size(0)
        device = x.device

        for layer_idx, conv in enumerate(self.convs):
            self_idx = torch.arange(n, device=device)
            all_src = torch.cat([edge_index[0], self_idx])
            all_dst = torch.cat([edge_index[1], self_idx])

            gate_vals_layer = []
            sum_norms_layer = []

            for h in range(conv.heads):
                o_std = conv.gat_heads[h](x, edge_index)
                v = conv.gat_heads[h].lin_r(x)
                v_all = v[all_src]
                s = scatter(v_all, all_dst, dim=0, dim_size=n, reduce='sum')
                g = torch.sigmoid(F.linear(x, conv.W_g[h], conv.b_g[h]))

                gate_vals_layer.append(g.detach().cpu())
                sum_norms_layer.append(s.detach().cpu().norm(dim=-1))

                o_cpa = g * s
                output = o_std + o_cpa

            gate_all = torch.cat(gate_vals_layer, dim=-1)
            sum_norms_all = torch.cat(sum_norms_layer, dim=-1)

            stats[f'layer_{layer_idx}'] = {
                'gate_mean': gate_all.mean().item(),
                'gate_std': gate_all.std().item(),
                'gate_frac_near_0': (gate_all < 0.1).float().mean().item(),
                'gate_frac_near_1': (gate_all > 0.9).float().mean().item(),
                'gate_frac_mid': ((gate_all >= 0.1) & (gate_all <= 0.9)).float().mean().item(),
                'sum_l2': sum_norms_all.mean().item(),
                'sum_l2_std': sum_norms_all.std().item(),
            }

            x = output
            x = F.elu(x)

        return stats


class NonCPAGAT(nn.Module):
    """Standard GATv2Conv model without CPA (for ablation H)."""

    def __init__(self, in_dim=42, hidden_dim=256, out_dim=1,
                 num_layers=4, heads=4, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                          concat=True, add_self_loops=True, dropout=dropout)
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

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)

        x = global_mean_pool(x, batch)
        x = self.predictor(x)
        return x.squeeze(-1)
