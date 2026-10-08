"""
PPFN Model: Polymer Periodic Fragment Network architecture.

Implements hierarchical graph neural network with:
1. Atom-level GNN on trimer graph with periodic edges
2. Fragment-level attention module
3. Long-range bipartite atom-fragment module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, TransformerConv, global_mean_pool
from torch_geometric.utils import scatter
from typing import Dict, Optional, Tuple


# Feature dimensions from dataset.py
NUM_ATOM_FEATURES = 27  # 18 atom types + 1 degree + 5 hybridization + 1 aromatic + 1 formal_charge + 1 ring
NUM_EDGE_FEATURES = 6   # 4 bond types + 1 conjugated + 1 ring
NUM_FRAG_FEATURES = 31  # NUM_ATOM_FEATURES + 3 (frag_size, has_aromatic, has_oh, has_nh)

HIDDEN_DIM = 256


class MultiHeadAttention(nn.Module):
    """Multi-head attention for fragment-level processing."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: (N_q, embed_dim)
            key: (N_k, embed_dim)
            value: (N_k, embed_dim)
            attention_mask: (N_q, N_k) or None
        Returns:
            output: (N_q, embed_dim)
        """
        batch_size = 1  # We handle batching separately via scatter

        Q = self.q_proj(query)  # (N_q, embed_dim)
        K = self.k_proj(key)    # (N_k, embed_dim)
        V = self.v_proj(value)  # (N_k, embed_dim)

        # Reshape for multi-head attention
        Q = Q.view(-1, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, N_q, head_dim)
        K = K.view(-1, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, N_k, head_dim)
        V = V.view(-1, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, N_k, head_dim)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (num_heads, N_q, N_k)

        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (num_heads, N_q, head_dim)

        # Concatenate heads
        out = out.transpose(0, 1).contiguous().view(-1, self.embed_dim)  # (N_q, embed_dim)

        return self.out_proj(out)


class AtomToFragmentAggregation(nn.Module):
    """Attention-based aggregation from atoms to fragments."""

    def __init__(self, atom_dim: int, frag_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.atom_dim = atom_dim
        self.frag_dim = frag_dim
        self.hidden_dim = hidden_dim

        # Project atoms to query space
        self.atom_to_q = nn.Linear(atom_dim, hidden_dim)

        # Project fragments to key/value space
        self.frag_to_k = nn.Linear(frag_dim, hidden_dim)
        self.frag_to_v = nn.Linear(frag_dim, hidden_dim)

        # Attention
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout=0.1)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, frag_dim)

        self.norm = nn.LayerNorm(frag_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, atom_embeddings: torch.Tensor, frag_embeddings: torch.Tensor,
                atom_frag_edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            atom_embeddings: (N_atoms, atom_dim)
            frag_embeddings: (N_frags, frag_dim)
            atom_frag_edge_index: (2, N_edges) where row 0 = atom indices, row 1 = frag indices
        Returns:
            updated_frag_embeddings: (N_frags, frag_dim)
            attention_weights: (N_frags,) for interpretability
        """
        device = atom_embeddings.device
        n_atoms = atom_embeddings.shape[0]
        n_frags = frag_embeddings.shape[0]

        if atom_frag_edge_index.shape[1] == 0:
            # No atom-fragment edges, return original fragments
            return frag_embeddings, torch.zeros(n_frags, device=device)

        atom_indices = atom_frag_edge_index[0]  # (N_edges,)
        frag_indices = atom_frag_edge_index[1]  # (N_edges,)

        # Create queries from atoms, keys/values from fragments
        atom_q = self.atom_to_q(atom_embeddings[atom_indices])  # (N_edges, hidden_dim)
        frag_k = self.frag_to_k(frag_embeddings[frag_indices])  # (N_edges, hidden_dim)
        frag_v = self.frag_to_v(frag_embeddings[frag_indices])  # (N_edges, hidden_dim)

        # Compute attention per edge
        attn_scores = (atom_q * frag_k).sum(dim=-1) / (self.hidden_dim ** 0.5)  # (N_edges,)

        # Aggregate attention weights per fragment
        attn_weights = torch.zeros(n_frags, device=device)
        attn_weights.scatter_add_(0, frag_indices, attn_scores)

        # Normalize attention weights per fragment
        frag_degrees = torch.zeros(n_frags, device=device)
        frag_degrees.scatter_add_(0, frag_indices, torch.ones_like(attn_scores))
        frag_degrees = torch.clamp(frag_degrees, min=1.0)  # Avoid division by zero
        attn_weights = attn_weights / frag_degrees

        # Aggregate atom information to fragments using attention
        atom_values = atom_embeddings[atom_indices]  # (N_edges, atom_dim)
        weighted_atoms = atom_values * attn_scores.unsqueeze(-1)  # (N_edges, atom_dim)

        # Sum weighted atoms per fragment
        frag_aggregated = torch.zeros(n_frags, self.atom_dim, device=device)
        frag_aggregated.scatter_add_(0, frag_indices.unsqueeze(1).expand(-1, self.atom_dim),
                                     weighted_atoms)

        # Normalize by fragment degree
        frag_aggregated = frag_aggregated / frag_degrees.unsqueeze(-1)

        # Project to fragment dimension
        frag_update = self.out_proj(frag_aggregated)  # (N_frags, frag_dim)

        # Residual connection and normalization
        updated_frags = self.norm(frag_embeddings + self.dropout(frag_update))

        return updated_frags, attn_weights


class LongRangeBipartiteModule(nn.Module):
    """Long-range bipartite attention between all atoms and all fragments."""

    def __init__(self, atom_dim: int, frag_dim: int, hidden_dim: int,
                 num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.atom_dim = atom_dim
        self.frag_dim = frag_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Project atoms and fragments to common dimension
        self.atom_proj = nn.Linear(atom_dim, hidden_dim)
        self.frag_proj = nn.Linear(frag_dim, hidden_dim)

        # Cross-attention layers (fragments query atoms)
        self.cross_attn_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, num_heads, dropout=0.1)
            for _ in range(num_layers)
        ])

        # Feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(0.1)
            )
            for _ in range(num_layers)
        ])

        # Layer norms
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, atom_embeddings: torch.Tensor, frag_embeddings: torch.Tensor,
                batch_atom: torch.Tensor, batch_frag: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atom_embeddings: (N_atoms, atom_dim)
            frag_embeddings: (N_frags, frag_dim)
            batch_atom: (N_atoms,) batch assignment for atoms
            batch_frag: (N_frags,) batch assignment for fragments
        Returns:
            long_range_embeddings: (N_frags, hidden_dim) - fragment representations with long-range info
        """
        device = atom_embeddings.device
        batch_size = batch_atom.max().item() + 1

        # Project to common dimension
        atom_proj = self.atom_proj(atom_embeddings)  # (N_atoms, hidden_dim)
        frag_proj = self.frag_proj(frag_embeddings)  # (N_frags, hidden_dim)

        # Batched cross-attention using scatter operations
        # Create batch-aware attention masks
        h = frag_proj

        for layer_idx in range(self.num_layers):
            # For each layer, process all batches at once
            # by creating a batched attention mechanism
            layer_out = []

            for b in range(batch_size):
                # Get atoms and fragments for this batch
                atom_mask = (batch_atom == b)
                frag_mask = (batch_frag == b)

                batch_atoms = atom_proj[atom_mask]  # (N_batch_atoms, hidden_dim)
                batch_frags = h[frag_mask]  # (N_batch_frags, hidden_dim)

                if batch_atoms.shape[0] == 0 or batch_frags.shape[0] == 0:
                    # No atoms or fragments in this batch, use zeros
                    n_batch_frags = frag_mask.sum().item()
                    layer_out.append(torch.zeros(n_batch_frags, self.hidden_dim, device=device))
                    continue

                # Cross-attention: fragments query atoms
                attn_out = self.cross_attn_layers[layer_idx](
                    query=batch_frags, key=batch_atoms, value=batch_atoms
                )

                # Residual connection and norm
                h_batch = self.layer_norms1[layer_idx](batch_frags + attn_out)

                # Feed-forward
                ffn_out = self.ffns[layer_idx](h_batch)
                h_batch = self.layer_norms2[layer_idx](h_batch + ffn_out)

                layer_out.append(h_batch)

            # Update h for next layer
            h = torch.cat(layer_out, dim=0)

        # Output projection
        h = self.output_proj(h)

        return h


class PPFN(nn.Module):
    """
    Polymer Periodic Fragment Network.

    Hierarchical architecture combining:
    1. Atom-level GNN (GINE) on trimer graph
    2. Fragment-level GNN (TransformerConv with edge features)
    3. Atom-to-Fragment aggregation (attention-based)
    4. Long-range bipartite atom-fragment module
    5. Prediction head with skip connection
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── 1. Embedding layers ────────────────────────────────────────────────
        self.atom_embedding = nn.Linear(NUM_ATOM_FEATURES, hidden_dim)
        self.edge_embedding = nn.Linear(NUM_EDGE_FEATURES, hidden_dim)
        self.frag_embedding = nn.Linear(NUM_FRAG_FEATURES, hidden_dim)

        # ── 2. Atom-level GNN (4 layers of GINE with residual connections) ───────
        self.atom_gnn_layers = nn.ModuleList()
        self.atom_layer_norms = nn.ModuleList()
        for _ in range(4):
            nn_module = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            gine_conv = GINEConv(nn=nn_module, eps=0.0, train_eps=True)
            self.atom_gnn_layers.append(gine_conv)
            self.atom_layer_norms.append(nn.LayerNorm(hidden_dim))

        # ── 3. Fragment-level GNN (2 layers of TransformerConv with edge features) ──
        self.frag_gnn_layers = nn.ModuleList()
        self.frag_layer_norms = nn.ModuleList()
        for _ in range(2):
            # TransformerConv supports edge features
            trans_conv = TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=4,
                concat=False,
                dropout=0.1,
                edge_dim=hidden_dim,  # Use edge features
                bias=True
            )
            self.frag_gnn_layers.append(trans_conv)
            self.frag_layer_norms.append(nn.LayerNorm(hidden_dim))

        # ── 4. Atom-to-Fragment aggregation ───────────────────────────────────────
        self.atom_to_frag = AtomToFragmentAggregation(
            atom_dim=hidden_dim,
            frag_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=4
        )

        # ── 5. Long-range bipartite module ────────────────────────────────────────
        self.long_range_bipartite = LongRangeBipartiteModule(
            atom_dim=hidden_dim,
            frag_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2
        )

        # ── 6. Prediction head with skip connection ─────────────────────────────────
        # Global mean pools: atoms (256) + fragments (256) + long-range (256) = 768
        self.predictor = nn.Sequential(
            nn.Linear(3 * hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        # Skip connection from input to output
        self.skip_proj = nn.Linear(3 * hidden_dim, 1)

        # For storing attention weights (interpretability)
        self.last_frag_attention = None

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of PPFN.

        Args:
            batch: Dictionary with keys:
                - 'atom_features': (N_atoms, 27)
                - 'edge_index': (2, N_edges)
                - 'edge_features': (N_edges, 6)
                - 'frag_features': (N_frags, 31)
                - 'frag_edge_index': (2, N_frag_edges)
                - 'frag_edge_features': (N_frag_edges, 6)
                - 'atom_frag_edge_index': (2, N_atom_frag_edges)
                - 'batch_atom': (N_atoms,)
                - 'batch_frag': (N_frags,)

        Returns:
            predictions: (batch_size, 1)
        """
        device = batch['atom_features'].device

        # ── 1. Embeddings ────────────────────────────────────────────────────────
        atom_h = self.atom_embedding(batch['atom_features'])  # (N_atoms, hidden_dim)
        edge_h = self.edge_embedding(batch['edge_features'])  # (N_edges, hidden_dim)
        frag_h = self.frag_embedding(batch['frag_features'])  # (N_frags, hidden_dim)
        frag_edge_h = self.edge_embedding(batch['frag_edge_features'])  # (N_frag_edges, hidden_dim)

        # ── 2. Atom-level GNN with residual connections ───────────────────────────
        edge_index = batch['edge_index']
        n_atoms = atom_h.shape[0]

        # Handle empty edge_index
        if edge_index.shape[1] > 0:
            for idx, gine_layer in enumerate(self.atom_gnn_layers):
                atom_h_new = gine_layer(atom_h, edge_index, edge_h)
                # Residual connection and layer normalization
                atom_h = self.atom_layer_norms[idx](atom_h + atom_h_new)
        else:
            # No edges, just apply linear transformations
            for idx, layer in enumerate(self.atom_gnn_layers):
                atom_h_new = layer.nn(atom_h)
                atom_h = self.atom_layer_norms[idx](atom_h + atom_h_new)

        # ── 3. Fragment-level GNN with edge features ──────────────────────────────
        frag_edge_index = batch['frag_edge_index']

        if frag_edge_index.shape[1] > 0:
            for idx, trans_conv in enumerate(self.frag_gnn_layers):
                # TransformerConv uses edge features
                frag_h_new = trans_conv(frag_h, frag_edge_index, frag_edge_h)
                # Residual connection and layer normalization
                frag_h = self.frag_layer_norms[idx](frag_h + frag_h_new)
        else:
            # No fragment edges, keep original embeddings
            pass

        # ── 4. Atom-to-Fragment aggregation ───────────────────────────────────────
        atom_frag_edge_index = batch['atom_frag_edge_index']
        frag_h, frag_attn = self.atom_to_frag(atom_h, frag_h, atom_frag_edge_index)

        # Store attention weights for interpretability
        self.last_frag_attention = frag_attn.detach()

        # ── 5. Long-range bipartite module ────────────────────────────────────────
        long_range_h = self.long_range_bipartite(
            atom_h, frag_h,
            batch['batch_atom'], batch['batch_frag']
        )  # (N_frags, hidden_dim)

        # ── 6. Global pooling and prediction ──────────────────────────────────────
        # Global mean pooling for each graph in the batch
        atom_global = global_mean_pool(atom_h, batch['batch_atom'])  # (batch_size, hidden_dim)
        frag_global = global_mean_pool(frag_h, batch['batch_frag'])  # (batch_size, hidden_dim)
        long_range_global = global_mean_pool(long_range_h, batch['batch_frag'])  # (batch_size, hidden_dim)

        # Concatenate and predict with skip connection
        combined = torch.cat([atom_global, frag_global, long_range_global], dim=1)  # (batch_size, 3*hidden_dim)
        predictions = self.predictor(combined) + self.skip_proj(combined)  # (batch_size, 1)

        return predictions

    def interpret(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Return interpretability information including fragment attention weights.

        Args:
            batch: Input batch dictionary

        Returns:
            Dictionary with:
                - 'fragment_attention': (N_frags,) attention weights per fragment
                - 'fragment_importance': (batch_size, max_frags) attention weights per batch
        """
        if self.last_frag_attention is None:
            # Run forward pass if not already done
            with torch.no_grad():
                _ = self.forward(batch)

        frag_attn = self.last_frag_attention  # (N_frags,)
        batch_frag = batch['batch_frag']  # (N_frags,)
        device = frag_attn.device
        batch_size = batch['batch_atom'].max().item() + 1

        # Ensure batch_frag is on same device as frag_attn
        batch_frag = batch_frag.to(device)

        # Organize attention by batch
        max_frags = int(batch_frag.bincount().max().item()) if batch_frag.numel() > 0 else 0
        frag_importance = torch.zeros(batch_size, max_frags, device=device)

        for b in range(batch_size):
            mask = (batch_frag == b)
            batch_attn = frag_attn[mask]
            frag_importance[b, :batch_attn.shape[0]] = batch_attn

        return {
            'fragment_attention': frag_attn,
            'fragment_importance': frag_importance
        }


def test_model():
    """Test the model with dummy data."""
    print("Testing PPFN model...")

    # Create dummy batch with proper division
    batch_size = 4
    atoms_per_batch = 12
    frags_per_batch = 3
    n_atoms = batch_size * atoms_per_batch
    n_frags = batch_size * frags_per_batch
    n_edges = 100
    n_frag_edges = 15
    n_atom_frag_edges = 80

    # Create batch assignments
    batch_atom = torch.cat([torch.full((atoms_per_batch,), i, dtype=torch.long)
                           for i in range(batch_size)])
    batch_frag = torch.cat([torch.full((frags_per_batch,), i, dtype=torch.long)
                           for i in range(batch_size)])

    batch = {
        'atom_features': torch.randn(n_atoms, NUM_ATOM_FEATURES),
        'edge_index': torch.randint(0, n_atoms, (2, n_edges)),
        'edge_features': torch.randn(n_edges, NUM_EDGE_FEATURES),
        'frag_features': torch.randn(n_frags, NUM_FRAG_FEATURES),
        'frag_edge_index': torch.randint(0, n_frags, (2, n_frag_edges)),
        'frag_edge_features': torch.randn(n_frag_edges, NUM_EDGE_FEATURES),
        'atom_frag_edge_index': torch.stack([
            torch.randint(0, n_atoms, (n_atom_frag_edges,)),
            torch.randint(0, n_frags, (n_atom_frag_edges,))
        ]),
        'batch_atom': batch_atom,
        'batch_frag': batch_frag
    }

    # Create model
    model = PPFN(hidden_dim=HIDDEN_DIM)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(batch)
        print(f"Predictions shape: {predictions.shape}")  # Should be (batch_size, 1)

    # Test interpretability
    interp = model.interpret(batch)
    print(f"Fragment attention shape: {interp['fragment_attention'].shape}")
    print(f"Fragment importance shape: {interp['fragment_importance'].shape}")

    print("Model test passed!")


if __name__ == '__main__':
    test_model()
