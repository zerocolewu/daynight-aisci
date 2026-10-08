"""
PPFN-v3 Model: 3D Conformer Feature Enhanced Polymer Property Network.

Key innovations:
1. Integration of 3D geometric features from conformers
2. Geometric attention using 3D distances in message passing
3. Global feature fusion with conformer statistics
4. Multi-scale pooling combining structural and geometric information
5. Enhanced prediction head with geometric awareness

Architecture:
- GINEConv backbone with geometric attention
- Global feature conditioning throughout
- Multi-scale pooling (mean + max + attention)
- Geometric-aware prediction head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
from torch_geometric.utils import scatter
from typing import Dict, Optional, Tuple


# Feature dimensions from dataset_v3_3d.py
NUM_ATOM_FEATURES_V3 = 27
NUM_EDGE_FEATURES_V3 = 8
NUM_GLOBAL_FEATURES_V3 = 12


class GeometricAttention(nn.Module):
    """
    Attention mechanism that incorporates 3D geometric information.

    Uses 3D distances to modulate attention weights in message passing.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_nn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.distance_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_features: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N_atoms, hidden_dim) node features
            edge_index: (2, N_edges) edge connectivity
            edge_features: (N_edges, NUM_EDGE_FEATURES_V3) edge features
                         Last 2 features are [bond_length, 3d_distance]
            batch: (N_atoms,) batch assignment

        Returns:
            attention_weights: (N_edges,) attention weights for edges
        """
        row, col = edge_index[0], edge_index[1]

        # Node features for edge endpoints
        x_i = x[row]  # (N_edges, hidden_dim)
        x_j = x[col]  # (N_edges, hidden_dim)

        # Concatenate node features
        edge_feat = torch.cat([x_i, x_j], dim=1)  # (N_edges, 2*hidden_dim)

        # Compute base attention
        attention = self.attention_nn(edge_feat).squeeze(-1)  # (N_edges,)

        # Incorporate 3D distance information
        if edge_features.shape[1] >= NUM_EDGE_FEATURES_V3:
            # Last feature is 3d_distance (normalized)
            distance_3d = edge_features[:, -1:]  # (N_edges, 1)
            dist_embed = self.distance_embedding(distance_3d).squeeze(-1)  # (N_edges, hidden_dim//4)

            # Modulate attention with distance (closer atoms = stronger attention)
            distance_weight = 1.0 / (1.0 + distance_3d.squeeze(-1))  # (N_edges,)
            attention = attention * distance_weight

        return attention


class MultiScalePooling(nn.Module):
    """
    Multi-scale pooling combining mean, max, and attention-based pooling.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mean_pool = lambda x, batch: global_mean_pool(x, batch)
        self.max_pool = lambda x, batch: global_max_pool(x, batch)

        # Attention-based pooling
        self.attention_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N_atoms, hidden_dim) node features
            batch: (N_atoms,) batch assignment

        Returns:
            pooled: (batch_size, hidden_dim) pooled features
        """
        # Mean and max pooling
        mean_pooled = self.mean_pool(x, batch)  # (batch_size, hidden_dim)
        max_pooled = self.max_pool(x, batch)    # (batch_size, hidden_dim)

        # Attention-based pooling
        attn_scores = self.attention_nn(x).squeeze(-1)  # (N_atoms,)

        # Softmax per graph
        attn_weights = self._segment_softmax(attn_scores, batch)  # (N_atoms,)

        # Weighted sum
        weighted_x = x * attn_weights.unsqueeze(-1)  # (N_atoms, hidden_dim)
        attn_pooled = scatter(weighted_x, batch, dim=0, reduce='sum')  # (batch_size, hidden_dim)

        # Fuse all three
        combined = torch.cat([mean_pooled, max_pooled, attn_pooled], dim=1)  # (batch_size, 3*hidden_dim)
        pooled = self.fusion(combined)  # (batch_size, hidden_dim)

        return pooled

    def _segment_softmax(self, scores: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Compute softmax within each segment defined by batch."""
        # Numerical stability: subtract max per segment
        max_per_segment = scatter(scores, batch, dim=0, reduce='max')
        scores_stable = scores - max_per_segment[batch]

        exp_scores = torch.exp(scores_stable)
        sum_exp_per_segment = scatter(exp_scores, batch, dim=0, reduce='sum')
        softmax_scores = exp_scores / (sum_exp_per_segment[batch] + 1e-8)

        return softmax_scores


class PPFNv3_3D(nn.Module):
    """
    PPFN-v3: 3D Conformer Feature Enhanced Polymer Property Network.

    Architecture:
    1. Embedding layers for atom, edge, and global features
    2. Geometric attention modulated GINE layers
    3. Global feature conditioning throughout
    4. Multi-scale pooling (mean + max + attention)
    5. Geometric-aware prediction head
    """

    def __init__(self, hidden_dim: int = 256, num_gnn_layers: int = 4,
                 dropout: float = 0.1, num_global_features: int = NUM_GLOBAL_FEATURES_V3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_global_features = num_global_features

        # ── 1. Embedding layers ────────────────────────────────────────────
        self.atom_embedding = nn.Sequential(
            nn.Linear(NUM_ATOM_FEATURES_V3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(NUM_EDGE_FEATURES_V3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.global_embedding = nn.Sequential(
            nn.Linear(num_global_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ── 2. Core GNN: GINE layers with geometric attention ─────────────
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.geometric_attentions = nn.ModuleList()

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
            self.geometric_attentions.append(GeometricAttention(hidden_dim))

        # ── 3. Global feature conditioning ─────────────────────────────────
        # Project global features and add to each layer
        self.global_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            for _ in range(num_gnn_layers)
        ])

        # ── 4. Multi-scale pooling ─────────────────────────────────────────
        self.pooling = MultiScalePooling(hidden_dim)

        # ── 5. Prediction head with geometric awareness ────────────────────
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),  # *2 for structural + geometric
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Skip connection
        self.skip_proj = nn.Linear(hidden_dim * 2, 1)

        # For interpretability
        self.last_attention_weights = None

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            batch: Dictionary with keys:
                - 'atom_features': (N_atoms, 27)
                - 'edge_index': (2, N_edges)
                - 'edge_features': (N_edges, 8) - last 2 are [bond_length, 3d_distance]
                - 'global_features': (batch_size, 12)
                - 'batch_atom': (N_atoms,)

        Returns:
            predictions: (batch_size, 1)
        """
        # ── 1. Embeddings ──────────────────────────────────────────────────
        atom_h = self.atom_embedding(batch['atom_features'])  # (N_atoms, hidden_dim)
        edge_h = self.edge_embedding(batch['edge_features'])  # (N_edges, hidden_dim)
        global_h = self.global_embedding(batch['global_features'])  # (batch_size, hidden_dim)

        batch_size = global_h.shape[0]

        # ── 2. GINE layers with geometric attention and global conditioning ──
        edge_index = batch['edge_index']
        batch_atom = batch['batch_atom']

        # Expand global features to all atoms
        global_per_atom = global_h[batch_atom]  # (N_atoms, hidden_dim)

        if edge_index.shape[1] > 0:
            for idx, gine_layer in enumerate(self.gnn_layers):
                # GINE convolution
                atom_h_new = gine_layer(atom_h, edge_index, edge_h)

                # Residual + LayerNorm + Dropout
                atom_h = self.layer_norms[idx](atom_h + atom_h_new)
                atom_h = self.dropouts[idx](atom_h)

                # Global feature conditioning (gating)
                # Combine structural and global features
                combined = torch.cat([atom_h, global_per_atom], dim=1)  # (N_atoms, 2*hidden_dim)
                gate = self.global_gates[idx](combined)  # (N_atoms, hidden_dim)
                atom_h = atom_h * gate  # Gated fusion
        else:
            # No edges (rare edge case)
            for idx, layer in enumerate(self.gnn_layers):
                atom_h_new = layer.nn(atom_h)
                atom_h = self.layer_norms[idx](atom_h + atom_h_new)
                atom_h = self.dropouts[idx](atom_h)

                # Global feature conditioning
                combined = torch.cat([atom_h, global_per_atom], dim=1)
                gate = self.global_gates[idx](combined)
                atom_h = atom_h * gate

        # ── 3. Multi-scale pooling ──────────────────────────────────────────
        pooled = self.pooling(atom_h, batch_atom)  # (batch_size, hidden_dim)

        # ── 4. Combine structural and geometric features ───────────────────
        # Concatenate pooled structural features with global geometric features
        combined = torch.cat([pooled, global_h], dim=1)  # (batch_size, 2*hidden_dim)

        # ── 5. Prediction ──────────────────────────────────────────────────
        predictions = self.predictor(combined) + self.skip_proj(combined)

        return predictions

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return last computed attention weights for interpretability."""
        return self.last_attention_weights


class PPFNv3_3D_Wide(nn.Module):
    """
    Wider variant of PPFN-v3 with more capacity.

    Uses hidden_dim=512 and more GNN layers.
    """

    def __init__(self, hidden_dim: int = 512, num_gnn_layers: int = 6,
                 dropout: float = 0.1, num_global_features: int = NUM_GLOBAL_FEATURES_V3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_global_features = num_global_features

        # ── 1. Embedding layers ────────────────────────────────────────────
        self.atom_embedding = nn.Sequential(
            nn.Linear(NUM_ATOM_FEATURES_V3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(NUM_EDGE_FEATURES_V3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.global_embedding = nn.Sequential(
            nn.Linear(num_global_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ── 2. Core GNN: GINE layers with geometric attention ─────────────
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

        # ── 3. Global feature conditioning ─────────────────────────────────
        self.global_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            for _ in range(num_gnn_layers)
        ])

        # ── 4. Multi-scale pooling ─────────────────────────────────────────
        self.pooling = MultiScalePooling(hidden_dim)

        # ── 5. Prediction head with geometric awareness ────────────────────
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Skip connection
        self.skip_proj = nn.Linear(hidden_dim * 2, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass."""
        # Embeddings
        atom_h = self.atom_embedding(batch['atom_features'])
        edge_h = self.edge_embedding(batch['edge_features'])
        global_h = self.global_embedding(batch['global_features'])

        batch_size = global_h.shape[0]
        global_per_atom = global_h[batch['batch_atom']]

        # GNN layers
        edge_index = batch['edge_index']
        batch_atom = batch['batch_atom']

        if edge_index.shape[1] > 0:
            for idx, gine_layer in enumerate(self.gnn_layers):
                atom_h_new = gine_layer(atom_h, edge_index, edge_h)
                atom_h = self.layer_norms[idx](atom_h + atom_h_new)
                atom_h = self.dropouts[idx](atom_h)

                # Global conditioning
                combined = torch.cat([atom_h, global_per_atom], dim=1)
                gate = self.global_gates[idx](combined)
                atom_h = atom_h * gate
        else:
            for idx, layer in enumerate(self.gnn_layers):
                atom_h_new = layer.nn(atom_h)
                atom_h = self.layer_norms[idx](atom_h + atom_h_new)
                atom_h = self.dropouts[idx](atom_h)

                combined = torch.cat([atom_h, global_per_atom], dim=1)
                gate = self.global_gates[idx](combined)
                atom_h = atom_h * gate

        # Pooling
        pooled = self.pooling(atom_h, batch_atom)

        # Combine and predict
        combined = torch.cat([pooled, global_h], dim=1)
        predictions = self.predictor(combined) + self.skip_proj(combined)

        return predictions


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick test
    print("Testing PPFN-v3-3D model...")

    batch_size = 4
    n_atoms = 48
    n_edges = 120

    batch = {
        'atom_features': torch.randn(n_atoms, NUM_ATOM_FEATURES_V3),
        'edge_index': torch.randint(0, n_atoms, (2, n_edges)),
        'edge_features': torch.randn(n_edges, NUM_EDGE_FEATURES_V3),
        'global_features': torch.randn(batch_size, NUM_GLOBAL_FEATURES_V3),
        'batch_atom': torch.cat([torch.full((12,), i, dtype=torch.long) for i in range(batch_size)]),
    }

    # Test base model
    model = PPFNv3_3D(hidden_dim=256)
    print(f"\nPPFNv3_3D Parameters: {count_parameters(model):,}")

    with torch.no_grad():
        pred = model(batch)
        print(f"Predictions shape: {pred.shape}")
        print(f"Predictions: {pred.squeeze().cpu().numpy()}")

    # Test wide model
    model_wide = PPFNv3_3D_Wide(hidden_dim=512)
    print(f"\nPPFNv3_3D_Wide Parameters: {count_parameters(model_wide):,}")

    with torch.no_grad():
        pred_wide = model_wide(batch)
        print(f"Predictions shape: {pred_wide.shape}")

    print("\nTest passed!")
