"""
Standard vanilla baseline models for polymer Tg/Tm regression.
- ECFP+MLP: 3-hidden-layer MLP on Morgan ECFP4 (2048-bit).
- GCN: 3-layer GCNConv (Kipf & Welling 2017).
- GIN: 3-layer GINConv with 2-layer MLP (Xu et al. 2019, canonical).
- GAT: 3-layer GATConv with multi-head attention; hidden layers concatenate
  heads, last layer averages them (Velickovic et al. 2018, canonical).

All GNN models use 35-dim one-hot categorical atomic node features
(no positional encoding), global mean pooling, and a 2-layer regression head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool


class GCN(nn.Module):
    def __init__(self, in_dim, hidden=128, n_layers=3, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.head(x)


class GIN(nn.Module):
    """Vanilla GIN (Xu et al. 2019): 3 GINConv layers with 2-layer MLP
    (Linear -> ReLU -> Linear, eps=0 fixed). Last-layer + global mean pool
    + 2-layer regression head. No external ReLU after the conv (canonical)."""
    def __init__(self, in_dim, hidden=128, n_layers=3, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            d_in = in_dim if i == 0 else hidden
            mlp = nn.Sequential(
                nn.Linear(d_in, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINConv(mlp))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.head(x)


class GAT(nn.Module):
    """Vanilla GAT (Velickovic et al. 2018): 3 GATConv layers with multi-head
    attention. Hidden layers concatenate heads; the last layer averages them
    (canonical design). ELU activation, last-layer + global mean pool
    + 2-layer regression head."""
    def __init__(self, in_dim, hidden=64, heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        assert n_layers >= 2, "GAT requires at least 2 layers (first + last)"
        self.convs = nn.ModuleList()
        # First layer: in_dim -> hidden*heads (concat)
        self.convs.append(GATConv(in_dim, hidden, heads=heads, dropout=dropout))
        # Middle layers (n_layers - 2 of them): hidden*heads -> hidden*heads (concat)
        for _ in range(n_layers - 2):
            self.convs.append(GATConv(hidden * heads, hidden, heads=heads, dropout=dropout))
        # Last layer: hidden*heads -> hidden (average heads, concat=False)
        self.convs.append(GATConv(hidden * heads, hidden, heads=heads,
                                  concat=False, dropout=dropout))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.head(x)


class ECFPMLP(nn.Module):
    def __init__(self, in_dim, hidden=512, n_layers=3, dropout=0.2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(name, node_dim=35, ecfp_bits=2048, device='cuda'):
    if name == 'ecfp_mlp':
        m = ECFPMLP(in_dim=ecfp_bits)
    elif name == 'gcn':
        m = GCN(in_dim=node_dim)
    elif name == 'gin':
        m = GIN(in_dim=node_dim)
    elif name == 'gat':
        m = GAT(in_dim=node_dim)
    else:
        raise ValueError(name)
    return m.to(device)
