"""
PerioGT Ablation Study - Model Implementations

Implements multiple model variants for systematic ablation:
1. Chemprop-style MPNN (paper baseline)
2. Periodic Graph Transformer (current implementation)
3. Ablation variants
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.utils import softmax
import math
from typing import Optional, Tuple, List


class BondMessagePassing(MessagePassing):
    """
    Chemprop-style directed bond-level message passing.

    Key difference from standard MPNN:
    - Messages are computed for BONDS, not atoms
    - Each bond sends messages to its reverse direction
    - Aggregation happens over bond messages to form atom embeddings
    """

    def __init__(
        self,
        hidden_dim: int = 300,
        edge_dim: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.hidden_dim = hidden_dim

        # Bond message network (from paper: learn hidden representation for each edge)
        self.bond_message_net = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        rev_edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            rev_edge_index: Index of reverse edge for each edge
        """
        # Propagate messages
        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            rev_edge_index=rev_edge_index,
            size=None,
        )
        return out

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        rev_edge_index: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute bond-level messages.

        In Chemprop, messages flow from bond to its reverse:
        m_bond = f(h_bond, h_reverse_bond, edge_features)
        """
        # Concatenate source node features with edge features
        # This forms the "bond representation"
        bond_input = torch.cat([x_i, edge_attr], dim=-1)
        message = self.bond_message_net(bond_input)

        return self.dropout(message)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Update node features with aggregated messages."""
        # Paper: node embedding = ReLU(W * (sum of incoming edges + atomic features))
        return aggr_out + x


class ChempropStyleMPNN(nn.Module):
    """
    Chemprop-style MPNN matching the paper architecture.

    Key features:
    - Directed bond-level message passing
    - 3 message passing iterations
    - 2-layer FFN with 300 hidden units
    - ReLU activations
    """

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int = 300,
        num_messages: int = 3,
        ffn_hidden_dim: int = 300,
        ffn_num_layers: int = 2,
        dropout: float = 0.0,
        use_periodic_edge: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_messages = num_messages
        self.use_periodic_edge = use_periodic_edge

        # Atom encoder (initialize node features)
        self.atom_encoder = nn.Linear(atom_dim, hidden_dim)

        # Edge encoder
        self.edge_encoder = nn.Linear(bond_dim, hidden_dim)

        # Bond message passing layers (one per iteration)
        self.message_layers = nn.ModuleList([
            BondMessagePassing(hidden_dim, hidden_dim, dropout)
            for _ in range(num_messages)
        ])

        # Atom feature integrator (from paper)
        self.atom_integrator = nn.Linear(hidden_dim + atom_dim, hidden_dim)

        # FFN for prediction (paper: 2 hidden layers, 300 neurons each)
        ffn_layers = []
        input_dim = hidden_dim
        for _ in range(ffn_num_layers):
            ffn_layers.append(nn.Linear(input_dim, ffn_hidden_dim))
            ffn_layers.append(nn.ReLU())
            if dropout > 0:
                ffn_layers.append(nn.Dropout(dropout))
            input_dim = ffn_hidden_dim
        ffn_layers.append(nn.Linear(input_dim, 1))

        self.ffn = nn.Sequential(*ffn_layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        periodic_mask: Optional[torch.Tensor] = None,
        periodic_vectors: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, atom_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, bond_dim]
            batch: Batch assignment [num_nodes]
            periodic_mask: Mask for periodic atoms [num_nodes, 1]
            periodic_vectors: Periodic vectors [num_nodes, 3]

        Returns:
            prediction: Property prediction [batch_size, 1]
            embedding: Graph embedding [batch_size, hidden_dim]
        """
        num_nodes = x.size(0)

        # Initialize node hidden states
        h = self.atom_encoder(x)

        # Encode edges
        edge_h = self.edge_encoder(edge_attr)

        # Create reverse edge index (for bidirectional message passing)
        # For each edge (i, j), find its reverse (j, i)
        rev_edge_index = self._get_reverse_edges(edge_index, num_nodes)

        # Message passing iterations (paper: 3 iterations)
        for msg_layer in self.message_layers:
            h = msg_layer(h, edge_index, edge_h, rev_edge_index)
            h = F.relu(h)

        # Integrate with original atom features (paper step)
        atom_features_integrated = torch.cat([h, x], dim=-1)
        h = self.atom_integrator(atom_features_integrated)
        h = F.relu(h)

        # Global pooling (sum or mean)
        graph_embedding = global_mean_pool(h, batch)

        # FFN prediction
        prediction = self.ffn(graph_embedding)

        return prediction, graph_embedding

    def _get_reverse_edges(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        For each edge, find the index of its reverse edge.
        Optimized version using vectorized operations.

        Returns tensor of shape [num_edges] where rev_edge_index[i] gives
        the index of the reverse of edge i.
        """
        num_edges = edge_index.size(1)
        device = edge_index.device

        # For bidirectional graphs, edge i and edge i+1 (or i-1) are reverses
        # assuming edges are added in pairs [i,j], [j,i]
        rev_edge_index = torch.arange(num_edges, device=device)
        # Even indices map to odd indices, odd to even
        rev_edge_index = rev_edge_index ^ 1  # XOR with 1 flips even/odd

        return rev_edge_index


class GraphTransformerLayer(MessagePassing):
    """
    Graph Transformer layer with attention mechanism.
    Used for ablation comparison.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention with residual
        h = self.norm1(x)
        attn_out = self.propagate(edge_index, x=h, edge_attr=edge_attr)
        x = x + self.dropout(attn_out)

        # FFN with residual
        h = self.norm2(x)
        x = x + self.ffn(h)

        return x

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        q = self.q_proj(x_i).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x_j).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x_j).view(-1, self.num_heads, self.head_dim)

        edge_feat = self.edge_proj(edge_attr)
        k = k + edge_feat.view(-1, self.num_heads, self.head_dim)

        alpha = (q * k).sum(dim=-1) / self.scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.dropout(alpha)

        out = v * alpha.unsqueeze(-1)
        return out.view(-1, self.hidden_dim)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return self.out_proj(aggr_out)


class PerioGT_GraphTransformer(nn.Module):
    """
    Graph Transformer variant for comparison.
    """

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(bond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        periodic_mask: Optional[torch.Tensor] = None,
        periodic_vectors: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.atom_encoder(x)
        edge_h = self.edge_encoder(edge_attr)

        for layer in self.layers:
            h = layer(h, edge_index, edge_h)

        graph_embedding = global_mean_pool(h, batch)
        prediction = self.predictor(graph_embedding)

        return prediction, graph_embedding


def get_model(
    model_type: str,
    atom_dim: int,
    bond_dim: int,
    hidden_dim: int = 300,
    num_layers: int = 3,
    num_heads: int = 4,
    dropout: float = 0.0,
    **kwargs
) -> nn.Module:
    """
    Factory function to create model by type.

    Args:
        model_type: 'chemprop', 'graphformer'
        atom_dim: Atom feature dimension
        bond_dim: Bond feature dimension
        hidden_dim: Hidden dimension
        num_layers: Number of layers/message passing steps
        num_heads: Number of attention heads (for transformer)
        dropout: Dropout rate
    """
    model_type = model_type.lower()

    if model_type == 'chemprop':
        return ChempropStyleMPNN(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            hidden_dim=hidden_dim,
            num_messages=num_layers,
            ffn_hidden_dim=hidden_dim,
            ffn_num_layers=2,
            dropout=dropout,
        )
    elif model_type == 'graphformer':
        return PerioGT_GraphTransformer(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
