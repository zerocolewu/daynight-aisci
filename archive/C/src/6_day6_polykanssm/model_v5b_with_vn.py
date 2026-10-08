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

class PolyKANSSM_v5b_with_vn(nn.Module):
    """v5b ablation: Re-add virtual node"""

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

        # Virtual node
        self.vn_emb = nn.Parameter(torch.randn(hidden_dim) * 0.1)

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

        # Add virtual node
        vn_repr = self.vn_emb.unsqueeze(0).expand(batch_size_val, -1)  # (B, H)
        h_with_vn = torch.cat([h, vn_repr], dim=0)  # (N+B, H)

        # Extend edge_index and edge_attr to connect VN to all nodes
        # Create bidirectional edges between VN and all nodes
        num_nodes = h.shape[0]
        vn_indices = torch.arange(batch_size_val, device=x.device) + num_nodes
        node_indices = torch.arange(num_nodes, device=x.device)

        # Edges from nodes to VN and VN to nodes
        src_to_vn = torch.stack([node_indices, vn_indices[batch]], dim=0)  # (2, N)
        vn_to_src = torch.stack([vn_indices[batch], node_indices], dim=0)  # (2, N)

        edge_index_with_vn = torch.cat([edge_index, src_to_vn, vn_to_src], dim=1)

        # Extend edge_attr with zero edges for VN connections
        # edge_h has shape (num_edges, hidden_dim), so we need to create zero edges of same size
        zero_edge_attr = torch.zeros(self.hidden_dim, device=edge_attr.device)
        edge_attr_vn = zero_edge_attr.unsqueeze(0).expand(num_nodes * 2, -1)
        edge_attr_with_vn = torch.cat([edge_h, edge_attr_vn], dim=0)

        # Extend batch indices for VN
        batch_with_vn = torch.cat([batch, vn_indices - num_nodes], dim=0)

        layer_outputs = []
        for i, (conv, norm, do) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            h_msg = conv(h_with_vn, edge_index_with_vn, edge_attr_with_vn)
            lam = torch.exp(self.log_lambdas[i]).clamp(max=0.999)

            # Extract node and VN representations
            h_nodes = h_msg[:num_nodes]
            h_vn = h_msg[num_nodes:]
            # Aggregate VN info back to nodes (mean of VN per graph)
            vn_per_node = h_vn[batch_with_vn[num_nodes:]]
            h_ssm = h_nodes * lam + h_nodes + vn_per_node

            h_with_vn = torch.cat([do(norm(h_ssm)), h_vn], dim=0)
            layer_outputs.append(h_with_vn[:num_nodes])

        # Apply KAN to node features only
        kan_out = self.kan(h_with_vn[:num_nodes])
        h_final = h_with_vn[:num_nodes] + torch.sigmoid(self.kan_gate) * kan_out
        layer_outputs[-1] = h_final

        h_jk = self.jk(layer_outputs)

        attn = torch.sigmoid(self.pool_attn(h_jk))
        h_w = h_jk * attn
        graph_attn = global_add_pool(h_w, batch, batch_size_val)
        graph_mean = global_mean_pool(h_jk, batch, batch_size_val)

        graph_repr = torch.cat([graph_attn, graph_mean], dim=-1)
        out = self.head(graph_repr)
        return out.squeeze(-1)
