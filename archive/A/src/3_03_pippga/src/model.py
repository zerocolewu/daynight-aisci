"""
PIPGA (Periodic-Invariant Polymer Graph Attention) Model

Key features:
1. Edge-wise attention without softmax
2. Periodic offset embeddings for multi-edge graphs
3. Self-connecting edges for periodic geometry encoding
4. ECFP4 fingerprint modulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PeriodicOffsetEmbedding(nn.Module):
    """
    Learnable embedding for periodic offset k ∈ {-1, 0, +1}
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # 3 embeddings for offsets: -1, 0, +1
        # Shift by +1 to make indices non-negative
        self.embedding = nn.Embedding(3, embed_dim)

    def forward(self, offset):
        """
        Args:
            offset: Periodic offset tensor [num_edges] with values in {-1, 0, +1}
        Returns:
            Embedding [num_edges, embed_dim]
        """
        # Shift offset from {-1, 0, +1} to {0, 1, 2}
        offset_idx = offset + 1
        return self.embedding(offset_idx)


class SelfLoopEncoder(nn.Module):
    """
    Encode self-loop features for periodic geometry.
    Input: [d_head, d_tail, is_head, is_tail]
    """
    def __init__(self, input_dim=4, embed_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, self_loop_features):
        """
        Args:
            self_loop_features: [num_atoms, 4]
        Returns:
            Embedding [num_atoms, embed_dim]
        """
        return self.mlp(self_loop_features)


class EdgeWiseAttention(nn.Module):
    """
    Edge-wise attention mechanism WITHOUT softmax normalization.

    For each edge h between nodes i and j:
    - Query: q_i = W_q · h_i
    - Key: k_jh = W_k · (h_j || e_ij || periodic_offset_embedding)
    - Value: v_j = W_v · h_j
    - Attention: α_ijh = sigmoid((q_i ∘ k_jh) / √d) where ∘ is Hadamard product
    - Apply LayerNorm to attention coefficients
    """
    def __init__(self, node_dim, edge_dim, num_heads=4, use_softmax=False):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.use_softmax = use_softmax

        assert node_dim % num_heads == 0

        # Query, Key, Value projections
        self.W_q = nn.Linear(node_dim, node_dim)
        self.W_k = nn.Linear(node_dim + edge_dim, node_dim)
        self.W_v = nn.Linear(node_dim, node_dim)

        # LayerNorm for attention coefficients
        self.attn_norm = nn.LayerNorm(num_heads)

        # Output projection
        self.W_out = nn.Linear(node_dim, node_dim)

        self.scale = self.head_dim ** 0.5

    def forward(self, node_features, edge_index, edge_features, return_attention=False):
        """
        Args:
            node_features: [num_nodes, node_dim]
            edge_index: [2, num_edges]
            edge_features: [num_edges, edge_dim]
            return_attention: Whether to return attention weights

        Returns:
            Updated node features [num_nodes, node_dim]
            (Optional) attention weights [num_edges, num_heads]
        """
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[1]

        # Compute queries, keys, values
        q = self.W_q(node_features)  # [num_nodes, node_dim]
        v = self.W_v(node_features)  # [num_nodes, node_dim]

        # Prepare edge-wise features
        source_idx = edge_index[0]  # [num_edges]
        target_idx = edge_index[1]  # [num_edges]

        q_edges = q[source_idx]  # [num_edges, node_dim]
        v_edges = v[target_idx]  # [num_edges, node_dim]

        # Concatenate target node features with edge features for keys
        target_features = node_features[target_idx]  # [num_edges, node_dim]
        k_input = torch.cat([target_features, edge_features], dim=-1)  # [num_edges, node_dim + edge_dim]
        k_edges = self.W_k(k_input)  # [num_edges, node_dim]

        # Multi-head attention
        q_heads = q_edges.view(num_edges, self.num_heads, self.head_dim)  # [num_edges, num_heads, head_dim]
        k_heads = k_edges.view(num_edges, self.num_heads, self.head_dim)  # [num_edges, num_heads, head_dim]
        v_heads = v_edges.view(num_edges, self.num_heads, self.head_dim)  # [num_edges, num_heads, head_dim]

        # Hadamard product for attention
        attn = (q_heads * k_heads).sum(dim=-1) / self.scale  # [num_edges, num_heads]

        if self.use_softmax:
            # Standard softmax normalization (for ablation study)
            # Group by source node
            attn_max = torch.zeros(num_nodes, self.num_heads, device=node_features.device)
            attn_max.scatter_reduce_(0, source_idx.unsqueeze(-1).expand(-1, self.num_heads),
                                     attn, reduce='amax', include_self=False)

            attn_exp = torch.exp(attn - attn_max[source_idx])
            attn_sum = torch.zeros(num_nodes, self.num_heads, device=node_features.device)
            attn_sum.scatter_add_(0, source_idx.unsqueeze(-1).expand(-1, self.num_heads), attn_exp)

            attn_weights = attn_exp / (attn_sum[source_idx] + 1e-8)  # [num_edges, num_heads]
        else:
            # Sigmoid activation (edge-wise, no softmax)
            attn_weights = torch.sigmoid(attn)  # [num_edges, num_heads]

        # Apply LayerNorm to attention (for non-softmax version)
        if not self.use_softmax:
            attn_weights = self.attn_norm(attn_weights)

        # Compute messages
        messages = v_heads * attn_weights.unsqueeze(-1)  # [num_edges, num_heads, head_dim]
        messages = messages.view(num_edges, self.node_dim)  # [num_edges, node_dim]

        # Aggregate messages
        aggregated = torch.zeros(num_nodes, self.node_dim, device=node_features.device)
        source_idx_expanded = source_idx.unsqueeze(-1).expand(-1, self.node_dim)
        aggregated.scatter_add_(0, source_idx_expanded, messages)

        # Output projection
        output = self.W_out(aggregated)

        if return_attention:
            return output, attn_weights
        return output


class PIPGALayer(nn.Module):
    """
    Single PIPGA layer with edge-wise attention and residual connection.
    """
    def __init__(self, node_dim, edge_dim, num_heads=4, use_softmax=False, use_self_loop=True):
        super().__init__()
        self.use_self_loop = use_self_loop

        self.attention = EdgeWiseAttention(node_dim, edge_dim, num_heads, use_softmax)

        # Self-loop encoder
        if use_self_loop:
            self.self_loop_encoder = SelfLoopEncoder(input_dim=4, embed_dim=edge_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.ReLU(),
            nn.Linear(node_dim * 2, node_dim)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)

    def forward(self, node_features, edge_index, edge_features, self_loop_features=None,
                return_attention=False):
        """
        Args:
            node_features: [num_nodes, node_dim]
            edge_index: [2, num_edges]
            edge_features: [num_edges, edge_dim]
            self_loop_features: [num_nodes, 4] (optional)
            return_attention: Whether to return attention weights

        Returns:
            Updated node features [num_nodes, node_dim]
        """
        # Add self-loop edges if enabled
        if self.use_self_loop and self_loop_features is not None:
            num_nodes = node_features.shape[0]
            self_loops = torch.stack([torch.arange(num_nodes, device=node_features.device),
                                      torch.arange(num_nodes, device=node_features.device)])

            edge_index = torch.cat([edge_index, self_loops], dim=1)

            # Encode self-loop features
            self_loop_emb = self.self_loop_encoder(self_loop_features)

            # Append to edge features
            edge_features = torch.cat([edge_features, self_loop_emb], dim=0)

        # Attention with residual
        if return_attention:
            attn_out, attn_weights = self.attention(node_features, edge_index, edge_features,
                                                     return_attention=True)
        else:
            attn_out = self.attention(node_features, edge_index, edge_features)

        node_features = self.norm1(node_features + attn_out)

        # FFN with residual
        ffn_out = self.ffn(node_features)
        node_features = self.norm2(node_features + ffn_out)

        if return_attention:
            return node_features, attn_weights
        return node_features


class FingerprintModulator(nn.Module):
    """
    ECFP4 fingerprint modulation for attention gating.
    """
    def __init__(self, fp_dim=2048, hidden_dim=128):
        super().__init__()
        self.proj = nn.Linear(fp_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, fingerprint, num_edges):
        """
        Args:
            fingerprint: [batch_size, fp_dim]
            num_edges: Number of edges in the batch

        Returns:
            Gating factor [num_edges]
        """
        # Project fingerprint
        fp_emb = self.proj(fingerprint)  # [batch_size, hidden_dim]
        fp_emb = F.relu(fp_emb)

        # Compute gate
        gate = torch.sigmoid(self.gate(fp_emb))  # [batch_size, 1]

        return gate.squeeze(-1)


class PIPGA(nn.Module):
    """
    Full PIPGA model for polymer Tg prediction.
    """
    def __init__(self, node_input_dim=23, node_dim=128, edge_dim=64, num_layers=4,
                 num_heads=4, fp_dim=2048, fp_hidden_dim=128, use_softmax=False,
                 use_fingerprint=True, use_self_loop=True, use_multi_edge=True):
        super().__init__()
        self.use_fingerprint = use_fingerprint
        self.use_self_loop = use_self_loop
        self.use_multi_edge = use_multi_edge

        # Node feature projection
        self.node_proj = nn.Linear(node_input_dim, node_dim)

        # Periodic offset embedding
        if use_multi_edge:
            self.offset_embedding = PeriodicOffsetEmbedding(edge_dim)
        else:
            # Dummy embedding for single-edge case
            self.offset_embedding = nn.Embedding(1, edge_dim)

        # PIPGA layers
        self.layers = nn.ModuleList([
            PIPGALayer(node_dim, edge_dim, num_heads, use_softmax, use_self_loop)
            for _ in range(num_layers)
        ])

        # Fingerprint modulator
        if use_fingerprint:
            self.fp_modulator = FingerprintModulator(fp_dim, fp_hidden_dim)
            self.fp_proj = nn.Linear(fp_dim, fp_hidden_dim)

        # Prediction head
        if use_fingerprint:
            pred_input_dim = node_dim + fp_hidden_dim
        else:
            pred_input_dim = node_dim

        self.predictor = nn.Sequential(
            nn.Linear(pred_input_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1)
        )

    def forward(self, atom_features, edge_index, edge_offset, self_loop_features,
                fingerprint, batch, num_atoms, return_attention=False):
        """
        Args:
            atom_features: [num_atoms, node_input_dim]
            edge_index: [2, num_edges]
            edge_offset: [num_edges] with values in {-1, 0, +1}
            self_loop_features: [num_atoms, 4]
            fingerprint: [batch_size, fp_dim]
            batch: [num_atoms] batch index for each atom
            num_atoms: [batch_size] number of atoms in each graph
            return_attention: Whether to return attention weights

        Returns:
            Predictions [batch_size, 1]
        """
        # Project node features
        node_features = self.node_proj(atom_features)  # [num_atoms, node_dim]

        # Get edge embeddings
        if self.use_multi_edge:
            edge_features = self.offset_embedding(edge_offset)  # [num_edges, edge_dim]
        else:
            # Use only k=0 edges
            mask = edge_offset == 0
            edge_index = edge_index[:, mask]
            edge_features = self.offset_embedding(torch.zeros(edge_index.shape[1],
                                                              dtype=torch.long,
                                                              device=edge_index.device))

        # Fingerprint modulation
        if self.use_fingerprint:
            gate = self.fp_modulator(fingerprint, edge_index.shape[1])  # [batch_size]

        # Apply PIPGA layers
        attention_weights = []
        for layer in self.layers:
            if return_attention:
                node_features, attn = layer(node_features, edge_index, edge_features,
                                           self_loop_features, return_attention=True)
                attention_weights.append(attn)
            else:
                node_features = layer(node_features, edge_index, edge_features,
                                     self_loop_features)

        # Apply fingerprint gating to node features
        if self.use_fingerprint:
            # Expand gate to match number of atoms
            gate_expanded = torch.zeros(node_features.shape[0], device=node_features.device)
            atom_offset = 0
            for i, n in enumerate(num_atoms):
                gate_expanded[atom_offset:atom_offset + n] = gate[i]
                atom_offset += n

            # Modulate node features
            node_features = node_features * gate_expanded.unsqueeze(-1)

        # Mean pooling over atoms in each graph
        graph_emb = torch.zeros(len(num_atoms), node_features.shape[1], device=node_features.device)
        atom_offset = 0
        for i, n in enumerate(num_atoms):
            graph_emb[i] = node_features[atom_offset:atom_offset + n].mean(dim=0)
            atom_offset += n

        # Concatenate with fingerprint embedding
        if self.use_fingerprint:
            fp_emb = self.fp_proj(fingerprint)
            fp_emb = F.relu(fp_emb)
            graph_emb = torch.cat([graph_emb, fp_emb], dim=-1)

        # Prediction
        output = self.predictor(graph_emb)

        if return_attention:
            return output, attention_weights
        return output


def count_parameters(model):
    """Count number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing PIPGA model...")

    # Create model
    model = PIPGA(
        node_input_dim=23,
        node_dim=128,
        edge_dim=64,
        num_layers=4,
        num_heads=4,
        use_softmax=False,
        use_fingerprint=True,
        use_self_loop=True,
        use_multi_edge=True
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Create dummy data
    batch_size = 2
    num_atoms_total = 50
    num_edges = 200

    atom_features = torch.randn(num_atoms_total, 23)
    edge_index = torch.randint(0, num_atoms_total, (2, num_edges))
    edge_offset = torch.randint(-1, 2, (num_edges,))
    self_loop_features = torch.randn(num_atoms_total, 4)
    fingerprint = torch.randn(batch_size, 2048)
    batch = torch.cat([torch.zeros(25, dtype=torch.long), torch.ones(25, dtype=torch.long)])
    num_atoms = torch.tensor([25, 25])

    # Forward pass
    output, attn = model(atom_features, edge_index, edge_offset, self_loop_features,
                        fingerprint, batch, num_atoms, return_attention=True)

    print(f"Output shape: {output.shape}")
    print(f"Number of attention layers: {len(attn)}")
    print(f"Attention shape per layer: {attn[0].shape}")

    # Test ablation variants
    print("\nTesting ablation variants...")

    # No fingerprint
    model_no_fp = PIPGA(use_fingerprint=False)
    output_no_fp = model_no_fp(atom_features, edge_index, edge_offset, self_loop_features,
                               fingerprint, batch, num_atoms)
    print(f"No fingerprint model parameters: {count_parameters(model_no_fp):,}")

    # Softmax attention
    model_softmax = PIPGA(use_softmax=True)
    output_softmax = model_softmax(atom_features, edge_index, edge_offset, self_loop_features,
                                   fingerprint, batch, num_atoms)
    print(f"Softmax attention model parameters: {count_parameters(model_softmax):,}")

    # No self-loop
    model_no_self = PIPGA(use_self_loop=False)
    output_no_self = model_no_self(atom_features, edge_index, edge_offset, self_loop_features,
                                   fingerprint, batch, num_atoms)
    print(f"No self-loop model parameters: {count_parameters(model_no_self):,}")

    # Single-edge
    model_single = PIPGA(use_multi_edge=False)
    output_single = model_single(atom_features, edge_index, edge_offset, self_loop_features,
                                 fingerprint, batch, num_atoms)
    print(f"Single-edge model parameters: {count_parameters(model_single):,}")

    print("\nModel test passed!")
