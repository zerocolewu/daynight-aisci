"""
PolyPerio-DenseGNN model for polymer Tg prediction.

Architecture:
1. Node/edge embedding layers
2. Multiple MPNN blocks with dense connectivity
3. Attention-based readout
4. MLP prediction head

Ablation variants:
- baseline: Monomer graph, no angular, sequential connections
- periodic: Trimer + periodic edge, no angular, sequential
- angular: Trimer + periodic + angular encoding, sequential
- full: Trimer + periodic + angular + dense connectivity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.utils import softmax


class EdgeConditionedMPNN(MessagePassing):
    """Edge-conditioned message passing neural network layer."""

    def __init__(self, d_node, d_edge, d_hidden, dropout=0.1):
        super().__init__(aggr='add')
        self.d_node = d_node
        self.d_edge = d_edge
        self.d_hidden = d_hidden

        # Message network: takes (node_i, node_j, edge_ij)
        self.message_mlp = nn.Sequential(
            nn.Linear(d_node * 2 + d_edge, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
        )

        # Update network
        self.update_mlp = nn.Sequential(
            nn.Linear(d_node + d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_node),
        )

        # Edge update network
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(d_edge + d_hidden * 2, d_edge),
        )

    def forward(self, x, edge_index, edge_attr):
        # Message passing
        m = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Update node features
        x_new = self.update_mlp(torch.cat([x, m], dim=-1))

        # Update edge features
        src, dst = edge_index
        edge_new = self.edge_update_mlp(
            torch.cat([edge_attr, x[src], x[dst]], dim=-1)
        )

        return x_new, edge_new

    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)


class DenseMPNNBlock(nn.Module):
    """MPNN block with residual connection."""

    def __init__(self, d_node, d_edge, d_hidden, dropout=0.1):
        super().__init__()
        self.mpnn = EdgeConditionedMPNN(d_node, d_edge, d_hidden, dropout)
        self.norm = nn.LayerNorm(d_node)

    def forward(self, x, edge_index, edge_attr):
        x_new, edge_new = self.mpnn(x, edge_index, edge_attr)
        # Residual connection
        x_out = self.norm(x + x_new)
        return x_out, edge_new


class AttentionPooling(nn.Module):
    """Attention-based graph pooling for property prediction."""

    def __init__(self, d_node, d_hidden):
        super().__init__()
        self.gate_nn = nn.Sequential(
            nn.Linear(d_node, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x, batch):
        gate = self.gate_nn(x)  # [N, 1]
        gate = softmax(gate.view(-1), batch).view(-1, 1)
        out = global_add_pool(gate * x, batch)
        return out


class PolyPerioDenseGNN(nn.Module):
    """
    PolyPerio-DenseGNN: Polymer Periodic Dense Graph Neural Network.

    Combines:
    1. Periodic polymer graph representation (trimer with periodic edge)
    2. Angular encoding for directional information
    3. Dense connectivity between MPNN blocks
    """

    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=128,
                 num_blocks=5, dropout=0.1, dense_connectivity=True,
                 use_angular=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dense_connectivity = dense_connectivity
        self.use_angular = use_angular

        # Node embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Edge embedding
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        if dense_connectivity:
            # Dense connectivity: each block has hidden_dim input
            # but receives concatenated features projected back to hidden_dim
            self.blocks = nn.ModuleList([
                DenseMPNNBlock(hidden_dim, hidden_dim, hidden_dim, dropout)
                for _ in range(num_blocks)
            ])

            # Projections to reduce concatenated features back to hidden_dim
            self.projections = nn.ModuleList()
            for i in range(1, num_blocks):
                # After block i, we have i+1 feature maps concatenated
                self.projections.append(
                    nn.Linear(hidden_dim * (i + 1), hidden_dim)
                )
        else:
            # Sequential connectivity
            self.blocks = nn.ModuleList([
                DenseMPNNBlock(hidden_dim, hidden_dim, hidden_dim, dropout)
                for _ in range(num_blocks)
            ])

        # Attention pooling
        self.pool = AttentionPooling(hidden_dim, hidden_dim // 2)

        # Also use mean pooling and combine
        self.combined_dim = hidden_dim * 2

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Embed
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        if self.dense_connectivity:
            # Dense connectivity: collect outputs from all blocks
            all_features = [x]

            for i, block in enumerate(self.blocks):
                if i == 0:
                    x_block, edge_attr = block(x, edge_index, edge_attr)
                else:
                    # Concatenate all previous features and project
                    x_cat = torch.cat(all_features, dim=-1)
                    x_proj = self.projections[i - 1](x_cat)
                    x_block, edge_attr = block(x_proj, edge_index, edge_attr)
                all_features.append(x_block)

            # Use last block output
            x = all_features[-1]
        else:
            # Sequential connectivity
            for block in self.blocks:
                x, edge_attr = block(x, edge_index, edge_attr)

        # Pooling: combine attention and mean pooling
        x_attn = self.pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_graph = torch.cat([x_attn, x_mean], dim=-1)

        # Predict
        out = self.predictor(x_graph)

        return out.squeeze(-1)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(variant='full', node_in_dim=None, edge_in_dim=None,
                hidden_dim=128, num_blocks=5, dropout=0.1):
    """
    Build model variant for ablation study.
    """
    if variant == 'full':
        model = PolyPerioDenseGNN(
            node_in_dim=node_in_dim, edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim, num_blocks=num_blocks,
            dropout=dropout, dense_connectivity=True, use_angular=True,
        )
    elif variant == 'angular':
        model = PolyPerioDenseGNN(
            node_in_dim=node_in_dim, edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim, num_blocks=num_blocks,
            dropout=dropout, dense_connectivity=False, use_angular=True,
        )
    elif variant == 'periodic':
        model = PolyPerioDenseGNN(
            node_in_dim=node_in_dim, edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim, num_blocks=num_blocks,
            dropout=dropout, dense_connectivity=False, use_angular=False,
        )
    elif variant == 'baseline':
        model = PolyPerioDenseGNN(
            node_in_dim=node_in_dim, edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim, num_blocks=num_blocks,
            dropout=dropout, dense_connectivity=False, use_angular=False,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return model
