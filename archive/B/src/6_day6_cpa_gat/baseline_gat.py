"""
Baseline GAT model for polymer Tg prediction.
4-layer GATv2Conv, 256 hidden dim, 4 heads, concat=True.
No residuals, no LayerNorm, ELU activation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class BaselineGAT(nn.Module):
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

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.elu(x)

        x = global_mean_pool(x, batch)
        x = self.predictor(x)
        return x.squeeze(-1)
