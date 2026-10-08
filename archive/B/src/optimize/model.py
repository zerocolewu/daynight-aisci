"""
CPA-GAT v2: improved CPA-GAT keeping the Cardinality-Preserving Attention core.

CORE INNOVATION PRESERVED (unchanged):
    o = o_std + o_cpa
    o_cpa = g * s,  s = scatter_sum(v over neighbors+self)
    g = sigmoid(W_g @ h + b_g)

IMPROVEMENTS layered around the core (do not alter the CPA formula):
  * Optional input feature standardization (BatchNorm1d)
  * Edge features via GATv2Conv(edge_dim=...)
  * Residual connections around each CPA layer
  * LayerNorm after each layer
  * Dropout inside layers + on residuals
  * Multi-readout pooling with per-branch LayerNorm (so no branch dominates)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_scatter import scatter


class CPAGATv2Conv(nn.Module):
    """
    Custom GATv2 layer with Cardinality-Preserving Attention (CPA) channel.
    CPA core: o = o_std + g * scatter_sum(v)   with g = sigmoid(W_g h + b_g)
    """

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.0, edge_dim=None):
        super().__init__()
        self.in_dim = in_dim
        self.head_dim = out_dim
        self.heads = heads
        self.edge_dim = edge_dim

        self.gat_heads = nn.ModuleList([
            GATv2Conv(in_dim, out_dim, heads=1, concat=True,
                      add_self_loops=True, dropout=dropout, edge_dim=edge_dim,
                      fill_value='mean')
            for _ in range(heads)
        ])
        self.W_g = nn.ParameterList([
            nn.Parameter(torch.Tensor(out_dim, in_dim)) for _ in range(heads)
        ])
        self.b_g = nn.ParameterList([
            nn.Parameter(torch.Tensor(out_dim)) for _ in range(heads)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for h in range(self.heads):
            nn.init.xavier_uniform_(self.W_g[h])
            nn.init.zeros_(self.b_g[h])
            self.gat_heads[h].reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        n = x.size(0)
        device = x.device
        self_idx = torch.arange(n, device=device)
        all_src = torch.cat([edge_index[0], self_idx])
        all_dst = torch.cat([edge_index[1], self_idx])
        outputs = []
        for h in range(self.heads):
            o_std = self.gat_heads[h](x, edge_index, edge_attr) if self.edge_dim is not None \
                else self.gat_heads[h](x, edge_index)
            v = self.gat_heads[h].lin_r(x)
            v_all = v[all_src]
            s = scatter(v_all, all_dst, dim=0, dim_size=n, reduce='sum')
            g = torch.sigmoid(F.linear(x, self.W_g[h], self.b_g[h]))
            outputs.append(o_std + g * s)
        return torch.cat(outputs, dim=-1)


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, 1)

    def forward(self, x, batch):
        scores = self.gate(x).squeeze(-1)
        maxs = scatter(scores.detach(), batch, dim=0, reduce='max')[batch]
        ex = (scores - maxs).exp()
        denom = scatter(ex, batch, dim=0, reduce='sum')[batch] + 1e-9
        w = ex / denom
        return scatter(x * w.unsqueeze(-1), batch, dim=0, reduce='sum')


def _readout(h, batch, pool):
    """Return list of pool tensors (each [G, hidden_dim])."""
    out = []
    if 'mean' in pool:
        out.append(global_mean_pool(h, batch))
    if 'sum' in pool:
        out.append(scatter(h, batch, dim=0, reduce='sum'))
    if 'att' in pool:
        out.append(None)  # placeholder, filled by caller with att_pool
    return out


class CPAGATv2(nn.Module):
    """Improved CPA-GAT. CPA layer formula is unchanged."""

    def __init__(self, in_dim=42, hidden_dim=256, out_dim=1, num_layers=4,
                 heads=4, dropout=0.1, edge_dim=4, u_dim=0, pool='mean_sum_att',
                 use_residual=True, use_norm=True, use_input_bn=False,
                 predictor_type='deep'):
        super().__init__()
        self.num_layers = num_layers
        self.head_dim = hidden_dim // heads
        self.use_residual = use_residual
        self.use_norm = use_norm
        self.pool = pool
        self.u_dim = u_dim
        self.use_input_bn = use_input_bn

        self.input_bn = nn.BatchNorm1d(in_dim) if use_input_bn else None
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(CPAGATv2Conv(hidden_dim, self.head_dim, heads=heads,
                                           dropout=dropout, edge_dim=edge_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.drop = nn.Dropout(dropout)

        # Determine readout branches
        self.branches = [b for b in ['mean', 'sum', 'att'] if b in pool]
        n_branches = len(self.branches)
        # Per-branch LayerNorm keeps scales comparable (fixes sum domination)
        self.branch_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_branches)])
        self.att_pool = AttentionPooling(hidden_dim) if 'att' in self.branches else None
        ro_dim = hidden_dim * n_branches

        if u_dim > 0:
            ro_dim += u_dim

        if predictor_type == 'simple':
            # Matches original CPAGAT: Linear -> ReLU -> Dropout -> Linear
            self.predictor = nn.Sequential(
                nn.Linear(ro_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, out_dim),
            )
        else:
            self.predictor = nn.Sequential(
                nn.Linear(ro_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, out_dim),
            )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)

        if self.use_input_bn:
            x = self.input_bn(x)
        h = self.input_proj(x)

        for i, conv in enumerate(self.convs):
            h_in = h
            h = conv(h, edge_index, edge_attr)
            h = F.elu(h)
            if self.use_norm:
                h = self.norms[i](h)
            if self.use_residual:
                h = h + h_in
            h = self.drop(h)

        # Readout with per-branch LayerNorm
        feats = []
        for bi, bname in enumerate(self.branches):
            if bname == 'mean':
                t = global_mean_pool(h, batch)
            elif bname == 'sum':
                t = scatter(h, batch, dim=0, reduce='sum')
            elif bname == 'att':
                t = self.att_pool(h, batch)
            feats.append(self.branch_norm[bi](t))
        z = torch.cat(feats, dim=-1) if feats else global_mean_pool(h, batch)

        if self.u_dim > 0 and getattr(data, 'u', None) is not None:
            z = torch.cat([z, data.u.to(z.dtype)], dim=-1)

        return self.predictor(z).squeeze(-1)
