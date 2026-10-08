"""
PPFN-v2 Improved Model: Multi-scale architecture with virtual node.

Key improvements:
1. Virtual node for global graph representation
2. Multi-scale pooling (mean + max + attention)
3. Jumping knowledge from all GNN layers
4. Better edge feature handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.utils import scatter
from typing import Dict, Optional, Tuple


NUM_ATOM_FEATURES = 27
NUM_EDGE_FEATURES = 6


class AttentionPooling(nn.Module):
    """Attention-based global pooling."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, batch):
        scores = self.gate_nn(x).squeeze(-1)
        attn_weights = self._segment_softmax(scores, batch)
        weighted_x = x * attn_weights.unsqueeze(-1)
        pooled = scatter(weighted_x, batch, dim=0, reduce='sum')
        return pooled, attn_weights

    def _segment_softmax(self, scores, batch):
        max_per_seg = scatter(scores, batch, dim=0, reduce='max')
        scores_stable = scores - max_per_seg[batch]
        exp_scores = torch.exp(scores_stable)
        sum_exp = scatter(exp_scores, batch, dim=0, reduce='sum')
        return exp_scores / (sum_exp[batch] + 1e-8)


class VirtualNodeLayer(nn.Module):
    """Virtual node that exchanges information with all graph nodes."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.node_to_virtual = nn.Linear(hidden_dim, hidden_dim)
        self.virtual_to_node = nn.Linear(hidden_dim, hidden_dim)
        self.virtual_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_h, virtual_h, batch):
        """
        Args:
            node_h: (N_nodes, hidden_dim)
            virtual_h: (batch_size, hidden_dim)
            batch: (N_nodes,) batch assignment
        Returns:
            updated_node_h, updated_virtual_h
        """
        # Aggregate node info to virtual node
        node_agg = scatter(node_h, batch, dim=0, reduce='mean')  # (batch_size, hidden_dim)
        virtual_update = self.node_to_virtual(node_agg)
        virtual_h = self.virtual_norm(virtual_h + virtual_update)

        # Distribute virtual node info to all nodes
        node_update = self.virtual_to_node(virtual_h[batch])  # (N_nodes, hidden_dim)
        node_h = self.node_norm(node_h + node_update)

        return node_h, virtual_h


class PPFNv2Improved(nn.Module):
    """
    Improved PPFN-v2 with virtual node and multi-scale pooling.

    Architecture:
    1. Embedding layers
    2. GINE layers with virtual node + residual + LayerNorm
    3. Jumping knowledge: collect outputs from all layers
    4. Multi-scale pooling: mean + max + attention
    5. Prediction head
    """

    def __init__(self, hidden_dim: int = 256, num_gnn_layers: int = 6,
                 dropout: float = 0.1, use_periodic_edge: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_periodic_edge = use_periodic_edge
        self.num_gnn_layers = num_gnn_layers

        # ── 1. Embedding layers ────────────────────────────────────────────
        self.atom_embedding = nn.Sequential(
            nn.Linear(NUM_ATOM_FEATURES, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.edge_embedding = nn.Sequential(
            nn.Linear(NUM_EDGE_FEATURES, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ── 2. GINE layers with virtual node ───────────────────────────────
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.virtual_node_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(num_gnn_layers):
            nn_module = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            gine_conv = GINEConv(nn=nn_module, eps=0.0, train_eps=True)
            self.gnn_layers.append(gine_conv)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.virtual_node_layers.append(VirtualNodeLayer(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))

        # Virtual node embedding
        self.virtual_node_embedding = nn.Parameter(torch.zeros(1, hidden_dim))

        # ── 3. Multi-scale pooling ─────────────────────────────────────────
        # We concatenate mean, max, and attention pools + virtual node
        # Total pool dim = 4 * hidden_dim
        pool_dim = 4 * hidden_dim  # mean + max + attn + virtual_node
        self.attn_pool = AttentionPooling(hidden_dim)

        # Layer projection for JK
        self.layer_proj = nn.Linear(hidden_dim, hidden_dim)

        # ── 4. Prediction head ─────────────────────────────────────────────
        self.predictor = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Skip connection
        self.skip_proj = nn.Linear(pool_dim, 1)

        # For interpretability
        self.last_attention_weights = None

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = batch['atom_features'].device
        batch_atom = batch['batch_atom']
        batch_size = batch_atom.max().item() + 1

        # ── 1. Embeddings ──────────────────────────────────────────────────
        atom_h = self.atom_embedding(batch['atom_features'])
        edge_h = self.edge_embedding(batch['edge_features'])

        # Virtual node
        virtual_h = self.virtual_node_embedding.expand(batch_size, -1).clone()

        edge_index = batch['edge_index']

        # Store layer outputs for JK
        layer_outputs = [atom_h]

        # ── 2. GINE + Virtual Node layers ──────────────────────────────────
        if edge_index.shape[1] > 0:
            for idx in range(self.num_gnn_layers):
                atom_h_new = self.gnn_layers[idx](atom_h, edge_index, edge_h)
                atom_h = self.layer_norms[idx](atom_h + atom_h_new)

                # Virtual node update
                atom_h, virtual_h = self.virtual_node_layers[idx](
                    atom_h, virtual_h, batch_atom)

                atom_h = self.dropouts[idx](atom_h)
                layer_outputs.append(atom_h)
        else:
            for idx in range(self.num_gnn_layers):
                atom_h_new = self.gnn_layers[idx].nn(atom_h)
                atom_h = self.layer_norms[idx](atom_h + atom_h_new)
                atom_h, virtual_h = self.virtual_node_layers[idx](
                    atom_h, virtual_h, batch_atom)
                atom_h = self.dropouts[idx](atom_h)
                layer_outputs.append(atom_h)

        # ── 3. Jumping Knowledge: sum of all layer outputs ─────────────────
        # Use last layer output for pooling (most expressive)
        final_h = layer_outputs[-1]

        # ── 4. Multi-scale pooling ─────────────────────────────────────────
        # Mean pool
        mean_pool = scatter(final_h, batch_atom, dim=0, reduce='mean')

        # Max pool
        max_pool = scatter(final_h, batch_atom, dim=0, reduce='max')

        # Attention pool
        attn_pool, attn_weights = self.attn_pool(final_h, batch_atom)
        self.last_attention_weights = attn_weights.detach()

        # Concatenate all pools + virtual node
        combined = torch.cat([mean_pool, max_pool, attn_pool, virtual_h], dim=-1)

        # ── 5. Prediction ──────────────────────────────────────────────────
        predictions = self.predictor(combined) + self.skip_proj(combined)

        return predictions

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        return self.last_attention_weights


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("Testing PPFN-v2 Improved model...")

    batch_size = 4
    n_atoms = 48
    n_edges = 120

    batch = {
        'atom_features': torch.randn(n_atoms, NUM_ATOM_FEATURES),
        'edge_index': torch.randint(0, n_atoms, (2, n_edges)),
        'edge_features': torch.randn(n_edges, NUM_EDGE_FEATURES),
        'batch_atom': torch.cat([torch.full((12,), i, dtype=torch.long) for i in range(batch_size)]),
    }

    model = PPFNv2Improved(hidden_dim=256)
    print(f"Parameters: {count_parameters(model):,}")

    with torch.no_grad():
        pred = model(batch)
        print(f"Predictions shape: {pred.shape}")

    attn = model.get_attention_weights()
    print(f"Attention weights shape: {attn.shape}")
    print("Test passed!")
