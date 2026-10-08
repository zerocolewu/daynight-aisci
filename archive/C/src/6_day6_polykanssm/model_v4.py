"""
PolyKAN-SSM v4: Improved architecture to reduce overfitting.

Key changes from v3:
- Proper SSM-style GNN layer with multi-head attention message aggregation
- Learnable Fourier frequencies initialized to small values
- No virtual node (was adding parameters and overfitting)
- Smaller hidden_dim (192 vs 256) and fewer layers (4 vs 5)
- Dual pooling (attention-weighted + mean)
- Better regularization throughout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import softmax


class FourierKANLayer(nn.Module):
    """
    Improved Fourier-KAN layer with proper formulation.

    y = base_linear(x) + sum_k (a_k * cos(w_k * x_proj) + b_k * sin(w_k * x_proj))

    Uses learnable frequencies initialized to small values so model starts close to linear.
    """

    def __init__(self, in_features, out_features, num_frequencies=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies

        # Base linear transformation
        self.base_linear = nn.Linear(in_features, out_features)

        # Project input to frequency domain
        self.freq_proj = nn.Linear(in_features, num_frequencies)

        # Learnable Fourier coefficients (initialized small)
        self.a_k = nn.Parameter(torch.randn(num_frequencies, out_features) * 0.01)
        self.b_k = nn.Parameter(torch.randn(num_frequencies, out_features) * 0.01)

        # Learnable frequency multipliers (initialized close to 1)
        self.w_k = nn.Parameter(torch.ones(num_frequencies) * 1.0)

    def forward(self, x):
        # Base linear output
        out = self.base_linear(x)

        # Project to frequency domain
        freq = self.freq_proj(x)  # [batch, num_frequencies]

        # Add Fourier components
        for k in range(self.num_frequencies):
            # Apply learnable frequency multiplier
            freq_k = freq[:, k:k+1] * self.w_k[k]

            # Cosine and sine terms
            out = out + self.a_k[k:k+1, :] * torch.cos(freq_k)
            out = out + self.b_k[k:k+1, :] * torch.sin(freq_k)

        return out


class MultiHeadMessageFunction(nn.Module):
    """
    Multi-head attention-based message function for GNN.

    Instead of simple mean aggregation, uses multi-head attention
    to weight messages from different neighbors.
    """

    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Edge feature integration
        self.edge_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h, edge_index, edge_h):
        """
        Args:
            h: [num_nodes, hidden_dim] - node features
            edge_index: [2, num_edges] - edge indices
            edge_h: [num_edges, hidden_dim] - edge features

        Returns:
            messages: [num_nodes, hidden_dim] - aggregated messages
        """
        num_nodes = h.size(0)
        num_edges = edge_index.size(1)

        # Compute Q, K, V for each node
        q = self.q_proj(h)  # [num_nodes, hidden_dim]
        k = self.k_proj(h)
        v = self.v_proj(h)

        # Reshape for multi-head
        q = q.view(num_nodes, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        k = k.view(num_nodes, self.num_heads, self.head_dim)
        v = v.view(num_nodes, self.num_heads, self.head_dim)

        # Get source and target nodes for each edge
        src_nodes = edge_index[0]  # [num_edges]
        dst_nodes = edge_index[1]  # [num_edges]

        # Get edge features and project
        e = self.edge_proj(edge_h)  # [num_edges, hidden_dim]
        e = e.view(num_edges, self.num_heads, self.head_dim)

        # Compute attention scores
        # For each edge, compute attention from source to target
        # We'll use source's query and target's key
        q_src = q[src_nodes]  # [num_edges, num_heads, head_dim]
        k_dst = k[dst_nodes]

        # Add edge information to attention
        scores = (q_src * k_dst).sum(dim=-1) / math.sqrt(self.head_dim)  # [num_edges, num_heads]
        scores = scores.view(-1, self.num_heads)  # [num_edges, num_heads]

        # Normalize scores per destination node
        # Need to group by destination node
        attn_weights = torch.zeros_like(scores)
        for head in range(self.num_heads):
            attn_weights[:, head] = softmax(
                scores[:, head],
                dst_nodes  # index to group by
            )

        # Get values from source nodes
        v_src = v[src_nodes]  # [num_edges, num_heads, head_dim]

        # Add edge information to values
        v_src = v_src + e

        # Compute weighted sum of values
        messages = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=h.device)
        for head in range(self.num_heads):
            # Aggregate messages for each destination node
            messages_dst = torch.zeros(num_nodes, self.head_dim, device=h.device)
            messages_dst.index_add_(0, dst_nodes, v_src[:, head, :] * attn_weights[:, head:head+1])
            messages[:, head, :] = messages_dst

        # Reshape and project output
        messages = messages.view(num_nodes, self.hidden_dim)
        messages = self.out_proj(messages)

        return messages


class GNNSSMLayer(nn.Module):
    """
    GNN-SSM layer: SSM-style state update with attention-weighted message aggregation.

    H_new = Lambda * H + B * Aggregate(Messages)

    where:
    - Lambda is diagonal with eigenvalues ~0.99 (controlled via log parameter)
    - B is a linear projection
    - Messages use multi-head attention aggregation
    """

    def __init__(self, hidden_dim, num_heads=4, spectral_radius=0.99):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Multi-head message function
        self.message_fn = MultiHeadMessageFunction(hidden_dim, num_heads)

        # SSM state update parameters
        # Lambda: diagonal matrix with values close to spectral_radius
        self.log_lambda = nn.Parameter(
            torch.full((hidden_dim,), math.log(spectral_radius))
        )

        # B: projection for messages
        self.b_proj = nn.Linear(hidden_dim, hidden_dim)

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, h, edge_index, edge_h):
        """
        Args:
            h: [num_nodes, hidden_dim] - current hidden states
            edge_index: [2, num_edges] - edge indices
            edge_h: [num_edges, hidden_dim] - edge features

        Returns:
            h_new: [num_nodes, hidden_dim] - updated hidden states
        """
        # SSM state update: H_new = Lambda * H + B * Messages
        lam = torch.exp(self.log_lambda)  # [hidden_dim]

        # Get messages using multi-head attention
        messages = self.message_fn(h, edge_index, edge_h)

        # Project messages
        b_messages = self.b_proj(messages)

        # SSM update
        h_ssm = h * lam + b_messages

        # Residual connection + norm + dropout
        h_new = self.dropout(self.norm(h + h_ssm))

        return h_new


class PolyKANSSM_v4(nn.Module):
    """
    PolyKAN-SSM v4: Improved architecture for polymer property prediction.

    Key features:
    1. Node embedding: Linear -> LayerNorm -> FourierKAN
    2. Edge embedding: Linear -> LayerNorm -> SiLU
    3. 4 GNN-SSM layers with multi-head message aggregation
    4. No virtual node (removed to reduce overfitting)
    5. Fourier-KAN mixing after GNN layers
    6. Dual pooling: attention-weighted + mean
    7. Smaller prediction head to reduce overfitting
    """

    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim=192,
        num_layers=4,
        dropout=0.2,
        spectral_radius=0.99,
        num_frequencies=4,
        num_heads=4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Node embedding: Linear -> LayerNorm -> FourierKAN
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.node_kan = FourierKANLayer(hidden_dim, hidden_dim, num_frequencies)

        # Edge embedding: Linear -> LayerNorm -> SiLU
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # GNN-SSM layers
        self.gnn_ssm_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_ssm_layers.append(
                GNNSSMLayer(hidden_dim, num_heads, spectral_radius)
            )

        # Fourier-KAN mixing after GNN layers
        self.mixing_kan = FourierKANLayer(hidden_dim, hidden_dim, num_frequencies)

        # Attention pooling weights
        self.pool_attn = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        # Prediction head (smaller than v3 to reduce overfitting)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Dual pooling
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch, batch_size):
        """
        Args:
            x: [num_nodes, node_dim] - node features
            edge_index: [2, num_edges] - edge indices
            edge_attr: [num_edges, edge_dim] - edge features
            batch: [num_nodes] - batch assignment
            batch_size: int - batch size

        Returns:
            out: [batch_size] - predictions
        """
        # Node embedding with Fourier-KAN
        h = self.node_embed(x)
        h = self.node_kan(h)
        h = F.silu(h)  # Activation

        # Edge embedding
        edge_h = self.edge_embed(edge_attr)

        # GNN-SSM layers
        for layer in self.gnn_ssm_layers:
            h = layer(h, edge_index, edge_h)

        # Fourier-KAN mixing
        h = h + self.mixing_kan(h)

        # Dual pooling
        # 1. Attention-weighted pooling
        attn = torch.sigmoid(self.pool_attn(h))  # [num_nodes, 1]
        h_w = h * attn
        graph_attn = global_add_pool(h_w, batch, batch_size)  # [batch_size, hidden_dim]

        # 2. Mean pooling
        graph_mean = global_mean_pool(h, batch, batch_size)  # [batch_size, hidden_dim]

        # Concatenate both pooling representations
        graph_repr = torch.cat([graph_attn, graph_mean], dim=1)  # [batch_size, hidden_dim * 2]

        # Prediction
        out = self.head(graph_repr)
        return out.squeeze(-1)
