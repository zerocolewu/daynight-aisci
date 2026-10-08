"""
PPFN-v2 Model: Simplified architecture focusing on periodic edge representation.

Key changes from v1:
- Removed complex fragment attention and long-range modules
- Focus on periodic edge representation (proven innovation)
- 6 layers of GINEConv with edge features
- Attention-based global pooling for interpretability
- Stronger prediction head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.utils import scatter
from typing import Dict, Optional, Tuple


# Feature dimensions from dataset.py
NUM_ATOM_FEATURES = 27  # 18 atom types + 1 degree + 5 hybridization + 1 aromatic + 1 formal_charge + 1 ring
NUM_EDGE_FEATURES = 6   # 4 bond types + 1 conjugated + 1 ring


class AttentionPooling(nn.Module):
    """Attention-based global pooling that provides interpretability."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N_atoms, hidden_dim) node features
            batch: (N_atoms,) batch assignment

        Returns:
            pooled: (batch_size, hidden_dim) attention-weighted pooled features
            attention_weights: (N_atoms,) attention weights for interpretability
        """
        # Compute attention scores
        scores = self.gate_nn(x).squeeze(-1)  # (N_atoms,)

        # Softmax per graph in the batch
        attention_weights = self._segment_softmax(scores, batch)  # (N_atoms,)

        # Weighted sum
        weighted_x = x * attention_weights.unsqueeze(-1)  # (N_atoms, hidden_dim)
        pooled = scatter(weighted_x, batch, dim=0, reduce='sum')  # (batch_size, hidden_dim)

        return pooled, attention_weights

    def _segment_softmax(self, scores: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Compute softmax within each segment defined by batch."""
        # Numerical stability: subtract max per segment
        max_per_segment = scatter(scores, batch, dim=0, reduce='max')
        scores_stable = scores - max_per_segment[batch]

        exp_scores = torch.exp(scores_stable)
        sum_exp_per_segment = scatter(exp_scores, batch, dim=0, reduce='sum')
        softmax_scores = exp_scores / (sum_exp_per_segment[batch] + 1e-8)

        return softmax_scores


class PPFNv2(nn.Module):
    """
    PPFN-v2: Simplified Polymer Periodic Fragment Network.

    Architecture:
    1. Embedding layers for atom and edge features
    2. 6 layers of GINEConv (GINE with edge features) + residual + LayerNorm
    3. Attention-based global pooling
    4. MLP prediction head with skip connection
    """

    def __init__(self, hidden_dim: int = 256, num_gnn_layers: int = 6,
                 dropout: float = 0.1, use_periodic_edge: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_periodic_edge = use_periodic_edge

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

        # ── 2. Core GNN: GINE layers with residual connections ─────────────
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
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
            self.dropouts.append(nn.Dropout(dropout))

        # ── 3. Attention-based global pooling ──────────────────────────────
        self.pooling = AttentionPooling(hidden_dim)

        # ── 4. Prediction head ─────────────────────────────────────────────
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Skip connection
        self.skip_proj = nn.Linear(hidden_dim, 1)

        # For interpretability
        self.last_attention_weights = None

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            batch: Dictionary with keys:
                - 'atom_features': (N_atoms, 27)
                - 'edge_index': (2, N_edges)
                - 'edge_features': (N_edges, 6)
                - 'batch_atom': (N_atoms,)

        Returns:
            predictions: (batch_size, 1)
        """
        # ── 1. Embeddings ──────────────────────────────────────────────────
        atom_h = self.atom_embedding(batch['atom_features'])  # (N_atoms, hidden_dim)
        edge_h = self.edge_embedding(batch['edge_features'])  # (N_edges, hidden_dim)

        # ── 2. GINE layers with residual connections ───────────────────────
        edge_index = batch['edge_index']

        if edge_index.shape[1] > 0:
            for idx, gine_layer in enumerate(self.gnn_layers):
                atom_h_new = gine_layer(atom_h, edge_index, edge_h)
                # Residual + LayerNorm + Dropout
                atom_h = self.layer_norms[idx](atom_h + atom_h_new)
                atom_h = self.dropouts[idx](atom_h)
        else:
            # No edges (rare edge case)
            for idx, layer in enumerate(self.gnn_layers):
                atom_h_new = layer.nn(atom_h)
                atom_h = self.layer_norms[idx](atom_h + atom_h_new)
                atom_h = self.dropouts[idx](atom_h)

        # ── 3. Attention-based global pooling ──────────────────────────────
        pooled, attn_weights = self.pooling(atom_h, batch['batch_atom'])
        self.last_attention_weights = attn_weights.detach()

        # ── 4. Prediction ──────────────────────────────────────────────────
        predictions = self.predictor(pooled) + self.skip_proj(pooled)

        return predictions

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return last computed attention weights for interpretability."""
        return self.last_attention_weights


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick test
    print("Testing PPFN-v2 model...")

    batch_size = 4
    n_atoms = 48
    n_edges = 120

    batch = {
        'atom_features': torch.randn(n_atoms, NUM_ATOM_FEATURES),
        'edge_index': torch.randint(0, n_atoms, (2, n_edges)),
        'edge_features': torch.randn(n_edges, NUM_EDGE_FEATURES),
        'batch_atom': torch.cat([torch.full((12,), i, dtype=torch.long) for i in range(batch_size)]),
    }

    model = PPFNv2(hidden_dim=256)
    print(f"Parameters: {count_parameters(model):,}")

    with torch.no_grad():
        pred = model(batch)
        print(f"Predictions shape: {pred.shape}")

    attn = model.get_attention_weights()
    print(f"Attention weights shape: {attn.shape}")
    print("Test passed!")
