"""
Degree Baseline: Standard GAT with node degree concatenated to input features.
Degree = number of neighbors (from edge_index, includes self-loop in degree count).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class DegreeBaselineGAT(nn.Module):
    def __init__(self, in_dim=43, hidden_dim=256, out_dim=1,
                 num_layers=4, heads=4, dropout=0.0):
        super().__init__()
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

        # Compute degree (including self-loop)
        n = x.size(0)
        device = x.device
        src, dst = edge_index
        self_idx = torch.arange(n, device=device)
        all_dst = torch.cat([dst, self_idx])
        all_src = torch.cat([src, self_idx])
        deg = torch.zeros(n, device=device)
        deg.scatter_add_(0, all_dst, torch.ones_like(all_dst, dtype=torch.float32))

        # Normalize degree and concatenate
        deg_feat = (deg / 100.0).unsqueeze(-1)  # [N, 1]
        x = torch.cat([x, deg_feat], dim=-1)  # [N, 43]

        x = self.input_proj(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)

        x = global_mean_pool(x, batch)
        x = self.predictor(x)
        return x.squeeze(-1)
