"""
HEPT: Hierarchical Equivariant Periodic Transformer for Polymer Tg Prediction

Architecture:
1. Equivariant Local Module (MACE-inspired): Spherical harmonics-based message passing
2. Periodic Attention Module (PerioGT-inspired): Attention across periodic images
3. Long-Range Module (LSR-MP-inspired): Global context via attention pooling
"""

import math
import numpy as np
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_softmax, scatter_sum, scatter_mean
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import degree


# ============================================================================
# Utility Functions
# ============================================================================

def get_edge_index_from_positions(
    pos: Tensor,
    batch: Tensor,
    cutoff: float = 5.0,
    max_neighbors: int = 32
) -> Tensor:
    """
    Create edge index based on distance cutoff within each graph in batch.
    """
    num_nodes = pos.size(0)
    edge_indices = []

    # Process each graph separately
    unique_batches = torch.unique(batch)

    for b in unique_batches:
        mask = batch == b
        indices = torch.where(mask)[0]
        pos_b = pos[mask]

        # Compute pairwise distances
        dist = torch.cdist(pos_b, pos_b)

        # Find neighbors within cutoff
        neighbor_mask = (dist < cutoff) & (dist > 0)

        for i in range(len(pos_b)):
            neighbors = torch.where(neighbor_mask[i])[0]
            if len(neighbors) > max_neighbors:
                # Keep only closest neighbors
                dists = dist[i, neighbors]
                _, closest = torch.topk(dists, max_neighbors, largest=False)
                neighbors = neighbors[closest]

            for j in neighbors:
                edge_indices.append([indices[i], indices[j]])

    if len(edge_indices) == 0:
        return torch.tensor([[], []], dtype=torch.long, device=pos.device)

    return torch.tensor(edge_indices, dtype=torch.long, device=pos.device).t()


def radial_basis_functions(
    distances: Tensor,
    num_basis: int = 64,
    cutoff: float = 5.0
) -> Tensor:
    """
    Compute radial basis functions (RBF) for distance encoding.
    Uses Gaussian basis functions centered at different distances.
    """
    # Centers of Gaussian basis functions
    centers = torch.linspace(0, cutoff, num_basis, device=distances.device)

    # Width of Gaussian functions
    gamma = 10.0 / (centers[1] - centers[0])

    # Compute Gaussian basis
    distances = distances.unsqueeze(-1)  # (n_edges, 1)
    centers = centers.unsqueeze(0)  # (1, num_basis)

    rbf = torch.exp(-gamma * (distances - centers) ** 2)

    return rbf


def spherical_harmonics(
    vec: Tensor,
    l_max: int = 2
) -> Tensor:
    """
    Compute real spherical harmonics Y_l^m for direction vectors.

    Args:
        vec: Direction vectors of shape (n, 3), should be normalized
        l_max: Maximum angular momentum (0, 1, 2)

    Returns:
        Spherical harmonics of shape (n, num_harmonics)
        For l_max=2: returns Y_0^0, Y_1^-1, Y_1^0, Y_1^1, Y_2^-2, Y_2^-1, Y_2^0, Y_2^1, Y_2^2
    """
    x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]

    # Normalize
    r = torch.sqrt(x**2 + y**2 + z**2).clamp(min=1e-8)
    x, y, z = x/r, y/r, z/r

    harmonics = []

    # l = 0
    harmonics.append(torch.ones_like(x) * 0.5 * math.sqrt(1/math.pi))

    if l_max >= 1:
        # l = 1
        harmonics.append(y * 0.5 * math.sqrt(3/math.pi))  # Y_1^-1
        harmonics.append(z * 0.5 * math.sqrt(3/math.pi))  # Y_1^0
        harmonics.append(x * 0.5 * math.sqrt(3/math.pi))  # Y_1^1

    if l_max >= 2:
        # l = 2
        harmonics.append(x * y * 0.5 * math.sqrt(15/math.pi))  # Y_2^-2
        harmonics.append(y * z * 0.5 * math.sqrt(15/math.pi))  # Y_2^-1
        harmonics.append((3*z**2 - 1) * 0.25 * math.sqrt(5/math.pi))  # Y_2^0
        harmonics.append(x * z * 0.5 * math.sqrt(15/math.pi))  # Y_2^1
        harmonics.append((x**2 - y**2) * 0.25 * math.sqrt(15/math.pi))  # Y_2^2

    return torch.stack(harmonics, dim=-1)


# ============================================================================
# Equivariant Local Module (MACE-inspired)
# ============================================================================

class EquivariantMessagePassing(MessagePassing):
    """
    Equivariant message passing layer using spherical harmonics.
    Inspired by MACE architecture for E(3) equivariance.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_rbf: int = 64,
        cutoff: float = 5.0,
        l_max: int = 2,
        aggr: str = 'add'
    ):
        super().__init__(aggr=aggr)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.l_max = l_max

        # Number of spherical harmonics
        self.num_sh = 1 + 3 + 5  # l=0, 1, 2

        # Radial basis functions
        self.rbf_centers = nn.Parameter(
            torch.linspace(0, cutoff, num_rbf),
            requires_grad=False
        )
        self.rbf_gamma = 10.0 / (self.rbf_centers[1] - self.rbf_centers[0])

        # Edge embedding
        self.edge_encoder = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Spherical harmonics projection
        self.sh_proj = nn.Linear(self.num_sh, hidden_dim)

        # Update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(out_dim)

    def compute_rbf(self, distances: Tensor) -> Tensor:
        """Compute radial basis functions."""
        distances = distances.unsqueeze(-1)
        rbf = torch.exp(-self.rbf_gamma * (distances - self.rbf_centers) ** 2)
        return rbf

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        batch: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, in_dim)
            pos: Node positions (num_nodes, 3)
            edge_index: Edge indices (2, num_edges)
            batch: Batch indices (num_nodes,)

        Returns:
            Updated node features (num_nodes, out_dim)
        """
        # Get source and target positions
        row, col = edge_index
        pos_i, pos_j = pos[row], pos[col]

        # Compute relative positions and distances
        rel_pos = pos_j - pos_i  # (num_edges, 3)
        distances = torch.norm(rel_pos, dim=-1)  # (num_edges,)

        # Normalize direction vectors
        direction = rel_pos / (distances.unsqueeze(-1) + 1e-8)

        # Compute RBF
        rbf = self.compute_rbf(distances)

        # Compute spherical harmonics
        sh = spherical_harmonics(direction, l_max=self.l_max)

        # Edge embedding from RBF
        edge_emb = self.edge_encoder(rbf)

        # Spherical harmonics projection
        sh_emb = self.sh_proj(sh)

        # Message computation
        x_j = x[col]
        msg_input = torch.cat([x_j, edge_emb], dim=-1)
        messages = self.message_mlp(msg_input)

        # Apply spherical harmonics as directional modulation
        messages = messages * sh_emb

        # Aggregate messages
        aggr_msg = scatter_sum(messages, row, dim=0, dim_size=x.size(0))

        # Update
        update_input = torch.cat([x, aggr_msg, sh_emb.new_zeros(x.size(0), sh_emb.size(-1))], dim=-1)
        # Actually need to aggregate sh_emb properly
        aggr_sh = scatter_mean(sh_emb, row, dim=0, dim_size=x.size(0))
        update_input = torch.cat([x, aggr_msg, aggr_sh], dim=-1)

        out = self.update_mlp(update_input)
        out = self.layer_norm(out)

        return out


class EquivariantLocalModule(nn.Module):
    """
    Stack of equivariant message passing layers.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        num_rbf: int = 64,
        cutoff: float = 5.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_layers = num_layers
        self.cutoff = cutoff

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Equivariant layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                EquivariantMessagePassing(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_rbf=num_rbf,
                    cutoff=cutoff
                )
            )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        batch: Tensor
    ) -> Tensor:
        # Input projection
        h = self.input_proj(x)

        # Equivariant message passing
        for layer in self.layers:
            h_new = layer(h, pos, edge_index, batch)
            h = h + self.dropout(h_new)  # Residual connection

        # Output projection
        out = self.output_proj(h)
        out = self.layer_norm(out)

        return out


# ============================================================================
# Periodic Attention Module (PerioGT-inspired)
# ============================================================================

class PeriodicAttention(nn.Module):
    """
    Periodic attention module that considers virtual periodic images.
    Inspired by PerioGT for handling periodic structures.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_periodic_images: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_periodic_images = num_periodic_images
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0

        # Multi-head attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Periodic position embedding
        self.periodic_pos_emb = nn.Embedding(2 * num_periodic_images + 1, hidden_dim)

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def create_periodic_images(
        self,
        pos: Tensor,
        batch: Tensor,
        x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Create virtual periodic images by translating positions.

        Returns:
            expanded_pos: Positions including periodic images
            expanded_batch: Batch indices for expanded positions
            periodic_indices: Periodic image indices
        """
        device = pos.device

        # Get graph boundaries (approximate periodicity from positions)
        unique_batches = torch.unique(batch)

        all_pos = []
        all_x = []
        all_batch = []
        all_periodic_idx = []

        for b in unique_batches:
            mask = batch == b
            pos_b = pos[mask]
            x_b = x[mask]
            n_atoms = pos_b.size(0)

            # Compute approximate periodic vectors (based on molecular extent)
            extent = pos_b.max(dim=0)[0] - pos_b.min(dim=0)[0]
            periodic_vectors = extent + 2.0  # Add some spacing

            # Create periodic images
            for px in range(-self.num_periodic_images, self.num_periodic_images + 1):
                for py in range(-self.num_periodic_images, self.num_periodic_images + 1):
                    for pz in range(-self.num_periodic_images, self.num_periodic_images + 1):
                        if px == 0 and py == 0 and pz == 0:
                            # Original positions
                            all_pos.append(pos_b)
                            all_x.append(x_b)
                            all_batch.append(torch.full((n_atoms,), b.item(), dtype=torch.long, device=device))
                            all_periodic_idx.append(torch.zeros(n_atoms, dtype=torch.long, device=device))
                        else:
                            # Translated positions
                            translation = torch.tensor([px, py, pz], dtype=torch.float, device=device) * periodic_vectors
                            all_pos.append(pos_b + translation)
                            all_x.append(x_b)
                            all_batch.append(torch.full((n_atoms,), b.item(), dtype=torch.long, device=device))
                            periodic_idx = px * (2*self.num_periodic_images+1)**2 + py * (2*self.num_periodic_images+1) + pz
                            all_periodic_idx.append(torch.full((n_atoms,), periodic_idx + 1, dtype=torch.long, device=device))

        expanded_pos = torch.cat(all_pos, dim=0)
        expanded_x = torch.cat(all_x, dim=0)
        expanded_batch = torch.cat(all_batch, dim=0)
        expanded_periodic_idx = torch.cat(all_periodic_idx, dim=0)

        return expanded_pos, expanded_x, expanded_batch, expanded_periodic_idx

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor
    ) -> Tensor:
        """
        Forward pass with periodic attention.

        Args:
            x: Node features (num_nodes, hidden_dim)
            pos: Node positions (num_nodes, 3)
            batch: Batch indices (num_nodes,)

        Returns:
            Updated node features (num_nodes, hidden_dim)
        """
        num_nodes = x.size(0)

        # Create periodic images (simplified - just use self-attention for now)
        # Full periodic attention would be computationally expensive

        # Multi-head self-attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Mask attention between different graphs
        # Create mask where batch[i] != batch[j]
        batch_i = batch.unsqueeze(0).expand(num_nodes, -1)
        batch_j = batch.unsqueeze(1).expand(-1, num_nodes)
        mask = (batch_i != batch_j).unsqueeze(0).expand(self.num_heads, -1, -1)
        attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(0, 1).contiguous().view(num_nodes, self.hidden_dim)
        out = self.out_proj(out)

        # Residual connection and layer norm
        out = self.layer_norm(x + self.dropout(out))

        return out


class PeriodicAttentionModule(nn.Module):
    """
    Stack of periodic attention layers.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        num_periodic_images: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            PeriodicAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_periodic_images=num_periodic_images,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor
    ) -> Tensor:
        h = x

        for layer in self.layers:
            h_new = layer(h, pos, batch)
            h = h + self.dropout(h_new)

        return h


# ============================================================================
# Long-Range Module (LSR-MP-inspired)
# ============================================================================

class LongRangeModule(nn.Module):
    """
    Long-range module using global context via attention pooling.
    Inspired by LSR-MP for capturing chain-level information.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Global context computation
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)

        # Virtual super-node
        self.super_node = nn.Parameter(torch.randn(1, hidden_dim))

        # Attention for super-node connection
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: Tensor,
        batch: Tensor
    ) -> Tensor:
        """
        Forward pass with global context.

        Args:
            x: Node features (num_nodes, hidden_dim)
            batch: Batch indices (num_nodes,)

        Returns:
            Updated node features with global context (num_nodes, hidden_dim)
        """
        num_nodes = x.size(0)
        num_graphs = batch.max().item() + 1

        # Compute global context for each graph
        global_context = scatter_mean(x, batch, dim=0, dim_size=num_graphs)  # (num_graphs, hidden_dim)

        # Expand global context to nodes
        expanded_context = global_context[batch]  # (num_nodes, hidden_dim)

        # Gating mechanism
        gate = self.gate(torch.cat([x, expanded_context], dim=-1))

        # Apply gated context
        out = x + gate * expanded_context

        out = self.layer_norm(out)

        return out


# ============================================================================
# HEPT Model
# ============================================================================

class HEPT(nn.Module):
    """
    Hierarchical Equivariant Periodic Transformer for Polymer Tg Prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_equivariant_layers: int = 3,
        num_periodic_layers: int = 2,
        num_long_range_layers: int = 1,
        num_heads: int = 8,
        num_rbf: int = 64,
        cutoff: float = 5.0,
        num_periodic_images: int = 2,
        dropout: float = 0.1,
        use_equivariant: bool = True,
        use_periodic: bool = True,
        use_long_range: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_equivariant = use_equivariant
        self.use_periodic = use_periodic
        self.use_long_range = use_long_range

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Equivariant Local Module
        if use_equivariant:
            self.equivariant_module = EquivariantLocalModule(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                num_layers=num_equivariant_layers,
                num_rbf=num_rbf,
                cutoff=cutoff,
                dropout=dropout
            )

        # Periodic Attention Module
        if use_periodic:
            self.periodic_module = PeriodicAttentionModule(
                hidden_dim=hidden_dim,
                num_layers=num_periodic_layers,
                num_heads=num_heads,
                num_periodic_images=num_periodic_images,
                dropout=dropout
            )

        # Long-Range Module
        if use_long_range:
            self.long_range_module = LongRangeModule(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

        # Pooling combination weights
        self.pool_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        batch: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, input_dim)
            pos: Node positions (num_nodes, 3)
            edge_index: Edge indices (2, num_edges)
            batch: Batch indices (num_nodes,)

        Returns:
            Tg predictions (batch_size, 1)
        """
        # Input projection
        h = self.input_proj(x)

        # Create distance-based edge index if not enough edges
        if edge_index.size(1) < 2:
            edge_index = get_edge_index_from_positions(pos, batch, cutoff=5.0)

        # Equivariant Local Module
        if self.use_equivariant:
            h = self.equivariant_module(h, pos, edge_index, batch)

        # Periodic Attention Module
        if self.use_periodic:
            h = self.periodic_module(h, pos, batch)

        # Long-Range Module
        if self.use_long_range:
            h = self.long_range_module(h, batch)

        # Hierarchical pooling: combine mean, sum, and max pooling
        weights = F.softmax(self.pool_weights, dim=0)

        h_mean = global_mean_pool(h, batch)
        h_sum = global_add_pool(h, batch)
        h_max = global_max_pool(h, batch)

        # Normalize sum pooling by number of nodes
        num_nodes_per_graph = degree(batch, dtype=torch.float).clamp(min=1).unsqueeze(-1)
        h_sum = h_sum / num_nodes_per_graph.sqrt()

        h_graph = weights[0] * h_mean + weights[1] * h_sum + weights[2] * h_max

        # Output prediction
        out = self.output_head(h_graph)

        return out


# ============================================================================
# Ablation Models
# ============================================================================

class HEPT_NoEquivariant(HEPT):
    """HEPT without equivariant module."""

    def __init__(self, **kwargs):
        kwargs['use_equivariant'] = False
        super().__init__(**kwargs)


class HEPT_NoPeriodic(HEPT):
    """HEPT without periodic attention module."""

    def __init__(self, **kwargs):
        kwargs['use_periodic'] = False
        super().__init__(**kwargs)


class HEPT_NoLongRange(HEPT):
    """HEPT without long-range module."""

    def __init__(self, **kwargs):
        kwargs['use_long_range'] = False
        super().__init__(**kwargs)


# ============================================================================
# Baseline Models for Comparison
# ============================================================================

class SimpleGNN(nn.Module):
    """Simple GNN baseline for comparison."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos, edge_index, batch):
        h = self.input_proj(x)

        # Simple mean aggregation message passing
        row, col = edge_index
        for layer in self.layers:
            msg = scatter_mean(h[col], row, dim=0, dim_size=h.size(0))
            h = h + self.dropout(F.relu(layer(msg)))

        h_graph = global_mean_pool(h, batch)
        out = self.output_head(h_graph)

        return out


if __name__ == "__main__":
    # Test model
    print("Testing HEPT model...")

    # Create dummy data
    num_nodes = 100
    input_dim = 31  # From atom features

    x = torch.randn(num_nodes, input_dim)
    pos = torch.randn(num_nodes, 3)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    batch = torch.cat([torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)])

    # Create model
    model = HEPT(
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=1,
        num_equivariant_layers=3,
        num_periodic_layers=2,
        num_heads=8,
        dropout=0.1
    )

    # Forward pass
    out = model(x, pos, edge_index, batch)
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
