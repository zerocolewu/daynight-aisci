"""
GATv2 model for PSPE experiments.

Architecture (same for all conditions):
- 4 layers of GATv2Conv with 4 heads (concat mode)
- hidden_dim=256, dropout=0.1
- ReLU activation
- Mean readout
- Predictor MLP: 256→128→64→1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATv2Predictor(nn.Module):
    """GATv2 model with configurable input dimension.

    Args:
        in_dim: Input node feature dimension
        edge_dim: Edge feature dimension (10 for rich features)
        hidden_dim: Hidden dimension (256)
        heads: Number of attention heads (4)
        dropout: Dropout rate (0.1)
    """

    def __init__(
        self,
        in_dim: int = 79,
        edge_dim: int = 10,
        hidden_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout

        per_head_dim = hidden_dim // heads  # 64

        # 4-layer GATv2
        self.conv1 = GATv2Conv(
            in_dim, per_head_dim, heads=heads, concat=True,
            edge_dim=edge_dim, dropout=dropout,
        )
        self.conv2 = GATv2Conv(
            hidden_dim, per_head_dim, heads=heads, concat=True,
            edge_dim=edge_dim, dropout=dropout,
        )
        self.conv3 = GATv2Conv(
            hidden_dim, per_head_dim, heads=heads, concat=True,
            edge_dim=edge_dim, dropout=dropout,
        )
        self.conv4 = GATv2Conv(
            hidden_dim, per_head_dim, heads=heads, concat=True,
            edge_dim=edge_dim, dropout=dropout,
        )

        # Predictor MLP
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, GATv2Conv):
                pass  # GATv2Conv handles its own init

    def forward(self, x, edge_index, batch, edge_attr=None, return_attention=False):
        """Forward pass.

        Args:
            x: Node features [N, in_dim]
            edge_index: Edge indices [2, E]
            batch: Batch indices [N]
            edge_attr: Edge attributes [E, edge_dim]
            return_attention: If True, returns attention weights from last layer

        Returns:
            If return_attention=False: predictions [batch_size]
            If return_attention=True: (predictions, attention_weights)
        """
        # Layer 1
        x1 = self.conv1(x, edge_index, edge_attr=edge_attr)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # Layer 2
        x2 = self.conv2(x1, edge_index, edge_attr=edge_attr)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # Layer 3
        x3 = self.conv3(x2, edge_index, edge_attr=edge_attr)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=self.dropout, training=self.training)

        # Layer 4
        if return_attention:
            x4, (edge_index_att, att_weights) = self.conv4(
                x3, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
        else:
            x4 = self.conv4(x3, edge_index, edge_attr=edge_attr)
        x4 = F.relu(x4)
        x4 = F.dropout(x4, p=self.dropout, training=self.training)

        # Global mean readout
        graph_emb = global_mean_pool(x4, batch)

        # Predictor
        out = self.predictor(graph_emb).squeeze(-1)

        if return_attention:
            return out, (edge_index_att, att_weights)
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
