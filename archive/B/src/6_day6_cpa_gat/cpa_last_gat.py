"""
CPA-Last-Layer-Only GAT: CPA channel only in the 4th (final) GAT layer.
Layers 1-3 use standard GATv2Conv.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from cpa_gat import CPAGATv2Conv


class CPALastLayerGAT(nn.Module):
    def __init__(self, in_dim=42, hidden_dim=256, out_dim=1,
                 num_layers=4, heads=4, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.head_dim = hidden_dim // heads

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                # Standard GATv2Conv for layers 1-3
                self.convs.append(
                    GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                              concat=True, add_self_loops=True, dropout=dropout)
                )
            else:
                # CPA-GAT only in the final layer
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
