"""
Repetition-Invariant Transformer (RIT) model for polymer Tg prediction.
Implements GINE encoder, repetition-invariant aggregation, and transformer modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.data import Batch
from typing import Optional
import math


class MultiHeadAttentionPooling(nn.Module):
    """Attention-based global pooling with learnable query vector."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Parameter(torch.randn(1, num_heads, self.head_dim))
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, batch):
        """
        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch vector [num_nodes]
        Returns:
            Graph-level features [batch_size, hidden_dim]
        """
        batch_size = batch.max().item() + 1

        # Project keys and values
        keys = self.key_proj(x)  # [num_nodes, hidden_dim]
        values = self.value_proj(x)  # [num_nodes, hidden_dim]

        # Reshape for multi-head attention
        keys = keys.view(-1, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        values = values.view(-1, self.num_heads, self.head_dim)

        # Compute attention scores per graph
        graph_features = []

        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() == 0:
                # Empty graph, use zero features
                graph_features.append(torch.zeros(1, self.hidden_dim, device=x.device))
                continue

            graph_keys = keys[mask]  # [num_nodes_in_graph, num_heads, head_dim]
            graph_values = values[mask]  # [num_nodes_in_graph, num_heads, head_dim]

            # Expand query for all nodes in this graph
            query = self.query.expand(1, self.num_heads, self.head_dim)  # [1, num_heads, head_dim]

            # Compute attention scores
            scores = torch.sum(query * graph_keys, dim=-1) / math.sqrt(self.head_dim)
            scores = F.softmax(scores, dim=0)  # [num_nodes_in_graph, num_heads]

            # Apply attention to values
            attended = torch.sum(scores.unsqueeze(-1) * graph_values, dim=0)  # [num_heads, head_dim]
            attended = attended.view(1, -1)  # [1, hidden_dim]

            # Output projection
            out = self.out_proj(attended)  # [1, hidden_dim]
            graph_features.append(out)

        # Stack all graph features
        graph_features = torch.cat(graph_features, dim=0)  # [batch_size, hidden_dim]

        return graph_features


class RITModel(nn.Module):
    """
    Repetition-Invariant Transformer for polymer Tg prediction.

    Architecture:
    1. Input projection layers
    2. RU positional embeddings
    3. GINE layers (shared across all RUs)
    4. Repetition-invariant max aggregation
    5. Transformer encoder
    6. Attention pooling
    7. Multi-task heads (Tg regression + range classification)
    8. Bias correction
    """

    def __init__(self, node_dim: int, edge_dim: int, global_dim: int,
                 hidden_dim: int = 256, num_gnn_layers: int = 5,
                 num_transformer_layers: int = 4, num_heads: int = 8,
                 num_ru: int = 3, dropout: float = 0.1, num_classes: int = 3):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        self.num_ru = num_ru
        self.num_classes = num_classes

        # 1. Input projections
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # 2. RU positional embeddings
        self.ru_embeddings = nn.Parameter(torch.randn(num_ru, hidden_dim) * 0.1)

        # 3. GINE layers
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()

        for _ in range(num_gnn_layers):
            nn_layer = GINEConv(
                nn=nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                ),
                eps=0.0,
                train_eps=True
            )
            self.gnn_layers.append(nn_layer)
            self.gnn_norms.append(nn.LayerNorm(hidden_dim))

        # 5. Transformer encoder
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers=num_transformer_layers
        )

        # Positional encoding for transformer
        self.transformer_pe = nn.Parameter(torch.randn(1, 100, hidden_dim) * 0.1)

        # 6. Attention pooling
        self.attention_pooling = MultiHeadAttentionPooling(hidden_dim, num_heads=num_heads)

        # Global features projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # 7. Tg prediction head
        self.tg_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 8. Range classification head
        self.range_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # 9. Bias correction network
        self.bias_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_classes, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, edge_index, edge_attr, conformer_stats, batch=None):
        """
        Forward pass of RIT model.

        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            conformer_stats: Global features [batch_size, global_dim]
            batch: Batch vector [num_nodes]

        Returns:
            tg_pred: Predicted Tg values [batch_size, 1]
            range_logits: Range classification logits [batch_size, num_classes]
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        batch_size = 1 if batch.dim() == 0 else batch.max().item() + 1

        # 1. Project node and edge features
        x = self.node_proj(x)  # [num_nodes, hidden_dim]
        edge_attr = self.edge_proj(edge_attr)  # [num_edges, hidden_dim]

        # Apply GINE to each RU independently
        ru_outputs = []

        for ru_idx in range(self.num_ru):
            # Get nodes for this RU across all batches
            ru_x = x.clone()

            # Add RU positional embedding
            ru_x = ru_x + self.ru_embeddings[ru_idx]

            # Apply GINE layers with residual connections
            for gnn_layer, norm in zip(self.gnn_layers, self.gnn_norms):
                identity = ru_x
                ru_x = gnn_layer(ru_x, edge_index, edge_attr)
                ru_x = norm(ru_x)
                ru_x = F.gelu(ru_x)
                ru_x = self.dropout(ru_x)
                ru_x = ru_x + identity

            ru_outputs.append(ru_x)

        # Stack outputs: [num_ru, num_nodes, hidden_dim]
        ru_outputs = torch.stack(ru_outputs, dim=0)

        # 4. Repetition-invariant aggregation: max pool across RUs
        # Result: [num_nodes, hidden_dim]
        x_agg, _ = torch.max(ru_outputs, dim=0)

        # 5. Transformer encoder - process each graph separately due to variable sizes
        graph_features_transformed = []

        for i in range(batch_size):
            mask = (batch == i)
            graph_x = x_agg[mask]  # [num_nodes_in_graph, hidden_dim]

            num_nodes = graph_x.size(0)

            # Add positional encoding
            max_pe_len = self.transformer_pe.size(1)
            if num_nodes <= max_pe_len:
                pe = self.transformer_pe[0, :num_nodes, :]
            else:
                # Interpolate positional encoding to match number of nodes
                # transformer_pe: [1, max_pe_len, hidden_dim] -> [1, hidden_dim, max_pe_len]
                pe = F.interpolate(
                    self.transformer_pe.permute(0, 2, 1),
                    size=num_nodes,
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1)[0]  # [num_nodes, hidden_dim]

            graph_x = graph_x + pe

            # Transformer with batch_first=True: [batch, seq_len, hidden_dim]
            graph_x = graph_x.unsqueeze(0)  # [1, num_nodes, hidden_dim]
            graph_x = self.transformer(graph_x)  # [1, num_nodes, hidden_dim]
            graph_x = graph_x.squeeze(0)  # [num_nodes, hidden_dim]
            graph_features_transformed.append(graph_x)

        # Concat back with batch info
        x_transformed = torch.cat(graph_features_transformed, dim=0)  # [num_nodes, hidden_dim]

        # 6. Attention pooling to get graph-level features
        graph_pooled = self.attention_pooling(x_transformed, batch)  # [batch_size, hidden_dim]

        # Project global conformer stats
        global_feat = self.global_proj(conformer_stats)  # [batch_size, hidden_dim]

        # Combine pooled node features with global features
        combined = torch.cat([graph_pooled, global_feat], dim=-1)  # [batch_size, hidden_dim * 2]

        # 7. Tg prediction (base)
        tg_base = self.tg_head(combined)  # [batch_size, 1]

        # 8. Range classification
        range_logits = self.range_head(combined)  # [batch_size, num_classes]
        range_probs = F.softmax(range_logits, dim=-1)  # [batch_size, num_classes]

        # 9. Bias correction based on range
        bias_input = torch.cat([combined, range_probs], dim=-1)  # [batch_size, hidden_dim*2 + num_classes]
        bias = self.bias_net(bias_input)  # [batch_size, 1]

        # Final prediction with bias correction
        tg_pred = tg_base + bias

        return tg_pred, range_logits

    def predict(self, x, edge_index, edge_attr, conformer_stats, batch=None):
        """Prediction method (returns only Tg values)."""
        tg_pred, _ = self.forward(x, edge_index, edge_attr, conformer_stats, batch)
        return tg_pred


def create_model(node_dim: int = 30, edge_dim: int = 7, global_dim: int = 16,
                 hidden_dim: int = 256, num_gnn_layers: int = 5,
                 num_transformer_layers: int = 4, num_heads: int = 8,
                 num_ru: int = 3, dropout: float = 0.1, num_classes: int = 3) -> RITModel:
    """
    Factory function to create RIT model.

    Args:
        node_dim: Input node feature dimension
        edge_dim: Input edge feature dimension
        global_dim: Global conformer stats dimension
        hidden_dim: Hidden dimension for all layers
        num_gnn_layers: Number of GINE layers
        num_transformer_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_ru: Number of repeat unit copies
        dropout: Dropout rate
        num_classes: Number of Tg range classes

    Returns:
        Initialized RITModel
    """
    model = RITModel(
        node_dim=node_dim,
        edge_dim=edge_dim,
        global_dim=global_dim,
        hidden_dim=hidden_dim,
        num_gnn_layers=num_gnn_layers,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        num_ru=num_ru,
        dropout=dropout,
        num_classes=num_classes
    )

    return model
