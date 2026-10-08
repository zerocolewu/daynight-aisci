"""
PolymerBipartite v2 Iteration 2 Model Architecture.

Fixes from iteration 1:
- FIX1: Replace MLP gate with two learnable scalar params (boundary/interior)
  initialized to -1.0/-3.0 so sigmoid gives ~0.27/~0.05
- FIX4: Better RU initialization from post-projection atom features
- FIX5: LayerNorm after inter-RU coupling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import scatter


# ============================================================================
# Phase 1: Standard GIN Layer (exact match with reference baseline)
# ============================================================================

class EdgeAwareGINConv(MessagePassing):
    """GIN convolution with edge feature awareness (same as PCR v2 baseline)."""

    def __init__(self, nn_func, edge_dim, hidden_dim, train_eps=True, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.nn = nn_func
        self.edge_proj = nn.Linear(edge_dim, hidden_dim, bias=False)
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([0.0]))
        else:
            self.register_buffer('eps', torch.Tensor([0.0]))

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out + (1 + self.eps) * x

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            edge_feat = self.edge_proj(edge_attr)
            return x_j + edge_feat
        return x_j

    def update(self, aggr_out):
        return self.nn(aggr_out)


# ============================================================================
# PolymerBipartite v2 Iteration 2
# ============================================================================

class PolymerBipartiteV2(nn.Module):
    """
    PolymerBipartite v2 Iteration 2: Fixed gate mechanism.

    Key changes from iter1:
    - Gate: Two learnable scalar params instead of MLP (FIX1)
    - LayerNorm after inter-RU coupling (FIX5)
    - Better RU init from projected atom features (FIX4)
    """

    def __init__(self, node_dim=30, edge_dim=6, hidden_dim=256, num_layers=5,
                 dropout_atom=0.5, dropout_ru_msg=0.3, dropout_head=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_atom = dropout_atom
        self.dropout_ru_msg = dropout_ru_msg
        self.dropout_head = dropout_head

        # Node embedding: 30-dim atom features -> 256-dim hidden
        self.node_embedding = nn.Linear(node_dim, hidden_dim)

        # FIX4: RU initial projection from post-embedding atom features
        self.ru_init_proj = nn.Linear(hidden_dim, hidden_dim)

        # Phase 1: GIN layers (same structure as baseline — UNCHANGED)
        self.gin_convs = nn.ModuleList()
        self.gin_bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
            conv = EdgeAwareGINConv(
                nn_func=mlp,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                train_eps=True,
            )
            self.gin_convs.append(conv)
            self.gin_bns.append(nn.BatchNorm1d(hidden_dim))

        # Phase 2a: Atom → RU aggregation
        self.ru_agg_layers = nn.ModuleList()
        self.ru_proj_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.ru_agg_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.ru_proj_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Phase 2b: Inter-RU coupling (simple linear) + LayerNorm (FIX5)
        self.couple_layers = nn.ModuleList()
        self.ru_layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.couple_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.ru_layer_norms.append(nn.LayerNorm(hidden_dim))

        # FIX1: Gate mechanism — two learnable scalar parameters
        # sigmoid(-1.0) ≈ 0.27 for boundary, sigmoid(-3.0) ≈ 0.05 for interior
        self.boundary_gate_scalar = nn.Parameter(torch.Tensor([-1.0]))
        self.interior_gate_scalar = nn.Parameter(torch.Tensor([-3.0]))

        # Prediction head: 256 -> 128 -> 1 (UNCHANGED)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(128, 1),
        )

    def get_gate_values(self, is_boundary):
        """
        Compute gate values using learnable scalars.

        Args:
            is_boundary: [N_atoms] float tensor (0.0 or 1.0)

        Returns:
            g: [N_atoms, 1] gate values
        """
        boundary_mask = (is_boundary > 0.5)
        g = torch.zeros(is_boundary.shape[0], 1, device=is_boundary.device)
        g[boundary_mask] = torch.sigmoid(self.boundary_gate_scalar)
        g[~boundary_mask] = torch.sigmoid(self.interior_gate_scalar)
        return g

    def forward(self, data_dict, return_gate_values=False):
        """Forward pass."""
        x = data_dict['x']
        edge_index = data_dict['edge_index']
        edge_attr = data_dict['edge_attr']
        batch = data_dict['batch']
        is_boundary = data_dict['is_boundary']
        atom_to_ru_idx = data_dict['atom_to_ru_idx']
        ru_batch = data_dict['ru_batch']
        n_rus = data_dict['n_rus']

        # FIX4: Initial atom features (after projection)
        h_atom = self.node_embedding(x)  # [N_atoms, 256]

        # FIX4: Initialize RU features from post-projection atom features
        ru_means = scatter(h_atom, atom_to_ru_idx, dim=0, dim_size=n_rus, reduce='mean')
        h_ru = self.ru_init_proj(ru_means)  # [3*B, 256]

        gate_values_all = [] if return_gate_values else None

        for layer_idx in range(self.num_layers):
            # ===== Phase 1: Atom → Atom (GIN) — UNCHANGED =====
            h_atom = self.gin_convs[layer_idx](h_atom, edge_index, edge_attr)
            h_atom = self.gin_bns[layer_idx](h_atom)
            h_atom = F.relu(h_atom)
            h_atom = F.dropout(h_atom, p=self.dropout_atom, training=self.training)

            # ===== Phase 2a: Atom → RU aggregation =====
            ru_agg = scatter(h_atom, atom_to_ru_idx, dim=0, dim_size=n_rus, reduce='mean')
            ru_agg = self.ru_agg_layers[layer_idx](ru_agg)
            ru_proj = self.ru_proj_layers[layer_idx](h_ru)
            h_ru_temp = ru_agg + ru_proj

            # ===== Phase 2b: Inter-RU coupling + LayerNorm (FIX5) =====
            h_ru_new = h_ru_temp.clone()
            h_ru_new[0::3] = h_ru_temp[0::3] + self.couple_layers[layer_idx](h_ru_temp[1::3])
            h_ru_new[1::3] = h_ru_temp[1::3] + self.couple_layers[layer_idx](h_ru_temp[0::3]) + self.couple_layers[layer_idx](h_ru_temp[2::3])
            h_ru_new[2::3] = h_ru_temp[2::3] + self.couple_layers[layer_idx](h_ru_temp[1::3])
            # Apply LayerNorm
            h_ru = self.ru_layer_norms[layer_idx](h_ru_new)

            # ===== Phase 2c: RU → Atom broadcast with scalar gates (FIX1) =====
            g = self.get_gate_values(is_boundary)  # [N_atoms, 1]

            # Broadcast RU features to atoms
            h_broadcast = h_ru[atom_to_ru_idx]  # [N_atoms, 256]

            # Gated update
            h_atom = h_atom + F.dropout(g * h_broadcast, p=self.dropout_ru_msg, training=self.training)

            if return_gate_values:
                gate_values_all.append({
                    'gate': g.detach(),
                    'is_boundary': is_boundary.detach(),
                    'boundary_scalar': self.boundary_gate_scalar.item(),
                    'interior_scalar': self.interior_gate_scalar.item(),
                })

        # ===== Graph-level pooling =====
        h_atom_graph = global_mean_pool(h_atom, batch)
        h_ru_graph = global_mean_pool(h_ru, ru_batch)
        h_graph = h_atom_graph + h_ru_graph

        # ===== Prediction =====
        pred = self.pred_head(h_graph).squeeze(-1)

        if return_gate_values:
            return pred, h_ru, gate_values_all
        return pred, h_ru

    def get_ru_params(self):
        """Get RU-related parameter groups for separate learning rate."""
        ru_param_names = [
            'ru_init_proj', 'ru_agg_layers', 'ru_proj_layers',
            'couple_layers', 'ru_layer_norms',
            'boundary_gate_scalar', 'interior_gate_scalar',
        ]
        ru_params = []
        for name, param in self.named_parameters():
            if any(name.startswith(prefix) for prefix in ru_param_names):
                ru_params.append(param)
        return ru_params

    def get_backbone_params(self):
        """Get GIN backbone parameters."""
        ru_param_names = [
            'ru_init_proj', 'ru_agg_layers', 'ru_proj_layers',
            'couple_layers', 'ru_layer_norms',
            'boundary_gate_scalar', 'interior_gate_scalar',
        ]
        backbone_params = []
        for name, param in self.named_parameters():
            if not any(name.startswith(prefix) for prefix in ru_param_names):
                backbone_params.append(param)
        return backbone_params


# ============================================================================
# Baseline GIN+Trimer Model (for comparison)
# ============================================================================

class GINTrimerBaseline(nn.Module):
    """
    Standard GIN+Trimer baseline.
    5-layer GIN with edge awareness, global mean pooling, MLP head.
    Supports configurable hidden_dim for parameter matching by width.
    """

    def __init__(self, node_dim=30, edge_dim=6, hidden_dim=256, num_layers=5,
                 dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.node_embedding = nn.Linear(node_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
            conv = EdgeAwareGINConv(
                nn_func=mlp,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                train_eps=True,
            )
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_embedding(x)
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_attr)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h_graph = global_mean_pool(h, batch)
        pred = self.pred_head(h_graph).squeeze(-1)
        return pred


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_wide_baseline(target_params, node_dim=30, edge_dim=6):
    """
    Create a GIN baseline with wider hidden_dim to match parameter count.
    This is a better approach than adding extra depth (which broke in iter1).
    """
    # Binary search for the right hidden_dim
    lo, hi = 256, 512
    best_dim = 256
    best_diff = float('inf')

    for dim in range(256, 512, 4):
        model = GINTrimerBaseline(node_dim, edge_dim, hidden_dim=dim)
        params = count_parameters(model)
        diff = abs(params - target_params)
        if diff < best_diff:
            best_diff = diff
            best_dim = dim
        if params >= target_params:
            break

    matched = GINTrimerBaseline(node_dim, edge_dim, hidden_dim=best_dim)
    actual_params = count_parameters(matched)
    print(f"Wide baseline: hidden_dim={best_dim}, params={actual_params:,} (target: {target_params:,})")
    return matched
