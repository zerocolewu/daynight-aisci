"""
Dense Connectivity Network GIN (DCN-GIN) for polymer property prediction

Key features:
1. Dense connectivity pattern (DenseNet-style)
2. GIN backbone with MLP aggregator
3. Transition layers for channel reduction
4. Gradient flow monitoring
5. Feature reuse analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor

from typing import List, Optional, Tuple


class DenseGINConv(MessagePassing):
    """
    Dense GIN Convolution layer.
    Uses MLP as aggregator and supports dense connectivity.
    """
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int = 7, **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # MLP for GIN
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        # Edge encoder - projects edge features to match in_channels
        self.edge_encoder = nn.Linear(edge_dim, in_channels)

        self.eps = nn.Parameter(torch.Tensor([0.0]))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        nn.init.constant_(self.eps, 0.0)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None) -> Tensor:
        # Encode edge features
        if edge_attr is not None:
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            edge_embedding = None

        # GIN aggregation
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_embedding=edge_embedding))
        return out

    def message(self, x_j: Tensor, edge_embedding: OptTensor) -> Tensor:
        if edge_embedding is not None:
            return x_j + edge_embedding
        return x_j


class TransitionLayer(nn.Module):
    """
    Transition layer for channel reduction.
    Used every 4 layers to compress feature maps.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_channels)
        self.conv = nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = F.relu(x)
        x = self.conv(x)
        return x


class DenseBlock(nn.Module):
    """
    Dense block containing multiple DenseGINConv layers.
    Each layer receives concatenated features from all previous layers.
    """
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        edge_dim: int = 7
    ):
        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        current_channels = in_channels
        for i in range(num_layers):
            # Each layer receives all previous features
            self.layers.append(
                DenseGINConv(current_channels, growth_rate)
            )
            current_channels += growth_rate

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None) -> Tuple[Tensor, List[Tensor]]:
        """
        Forward pass with dense connectivity.

        Returns:
            output: Final concatenated features
            layer_outputs: List of outputs from each layer (for gradient analysis)
        """
        features = [x]
        layer_outputs = []

        for layer in self.layers:
            out = layer(torch.cat(features, dim=-1), edge_index, edge_attr)
            features.append(out)
            layer_outputs.append(out)

        return torch.cat(features, dim=-1), layer_outputs


class DCNGIN(nn.Module):
    """
    Dense Connectivity Network GIN for polymer property prediction.

    Architecture:
    1. Initial projection layer
    2. Multiple dense blocks with transition layers
    3. Global pooling
    4. Feature fusion with 148-dim molecular features
    5. Prediction head
    """
    def __init__(
        self,
        num_layers: int = 5,
        hidden_dim: int = 256,
        growth_rate: int = 32,
        compression_factor: float = 0.5,
        feature_dim: int = 148,
        dropout: float = 0.2,
        use_dense: bool = True
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.growth_rate = growth_rate
        self.use_dense = use_dense

        # Atom feature dimension from from_smiles
        atom_dim = 9  # Default atom features from PyG
        self.atom_encoder = nn.Linear(atom_dim, hidden_dim)

        # Edge encoder
        self.edge_encoder = nn.Linear(7, hidden_dim)

        if use_dense:
            # Dense connectivity architecture
            # Divide layers into blocks of 4
            num_blocks = (num_layers + 3) // 4
            layers_per_block = 4

            self.blocks = nn.ModuleList()
            current_channels = hidden_dim

            for block_idx in range(num_blocks):
                # Determine number of layers in this block
                remaining_layers = num_layers - block_idx * layers_per_block
                block_layers = min(layers_per_block, remaining_layers)

                if block_layers <= 0:
                    break

                # Add dense block
                self.blocks.append(
                    DenseBlock(block_layers, current_channels, growth_rate)
                )

                # Update channel count
                current_channels += block_layers * growth_rate

                # Add transition layer (except after last block)
                if block_idx < num_blocks - 1 and remaining_layers > layers_per_block:
                    out_channels = int(current_channels * compression_factor)
                    self.blocks.append(
                        TransitionLayer(current_channels, out_channels)
                    )
                    current_channels = out_channels

            final_hidden_dim = current_channels

        else:
            # Standard GIN architecture (for comparison)
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()

            for i in range(num_layers):
                self.convs.append(
                    DenseGINConv(hidden_dim, hidden_dim)
                )
                self.bns.append(nn.BatchNorm1d(hidden_dim))

            final_hidden_dim = hidden_dim

        # Feature fusion
        self.feature_proj = nn.Linear(feature_dim, 128)

        # Prediction head
        total_dim = final_hidden_dim + 128
        self.head = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # For storing gradient information
        self.gradient_info = {}
        self.layer_outputs = []

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        features: OptTensor = None,
        batch: OptTensor = None
    ) -> Tuple[Tensor, dict]:
        """
        Forward pass.

        Returns:
            prediction: Tg prediction
            info: Dictionary with gradient and layer information
        """
        # Encode atom features
        x = self.atom_encoder(x)

        # Encode edge features
        if edge_attr is not None:
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            edge_embedding = None

        # Store layer outputs for gradient analysis
        self.layer_outputs = []

        if self.use_dense:
            # Dense connectivity path
            for module in self.blocks:
                if isinstance(module, DenseBlock):
                    x, layer_outs = module(x, edge_index, edge_attr)
                    self.layer_outputs.extend(layer_outs)
                else:  # TransitionLayer
                    x = module(x)

            # Global pooling
            graph_emb = global_mean_pool(x, batch)

        else:
            # Standard GIN path
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index, edge_attr)
                x = bn(x)
                x = F.relu(x)
                self.layer_outputs.append(x)

            # Global pooling
            graph_emb = global_mean_pool(x, batch)

        # Feature fusion
        if features is not None:
            feat_emb = self.feature_proj(features)
            combined = torch.cat([graph_emb, feat_emb], dim=-1)
        else:
            combined = graph_emb

        # Prediction
        prediction = self.head(combined)

        info = {
            'layer_outputs': self.layer_outputs,
            'graph_emb': graph_emb
        }

        return prediction, info

    def get_gradient_info(self):
        """Get gradient information for analysis."""
        return self.gradient_info

    def save_gradients(self):
        """Save gradient magnitudes for each layer."""
        self.gradient_info = {}

        if self.use_dense:
            for i, module in enumerate(self.blocks):
                if isinstance(module, DenseBlock):
                    for j, layer in enumerate(module.layers):
                        key = f'block_{i}_layer_{j}'
                        if hasattr(layer, 'weight') and layer.weight.grad is not None:
                            self.gradient_info[key] = layer.weight.grad.abs().mean().item()
        else:
            for i, conv in enumerate(self.convs):
                key = f'layer_{i}'
                if hasattr(conv, 'weight') and conv.weight.grad is not None:
                    self.gradient_info[key] = conv.weight.grad.abs().mean().item()


class StandardGIN(nn.Module):
    """
    Standard GIN without dense connectivity for comparison.
    Same architecture but with simple sequential layers.
    """
    def __init__(
        self,
        num_layers: int = 5,
        hidden_dim: int = 256,
        feature_dim: int = 148,
        dropout: float = 0.2
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Atom feature dimension
        atom_dim = 9
        self.atom_encoder = nn.Linear(atom_dim, hidden_dim)

        # GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(
                DenseGINConv(hidden_dim, hidden_dim)
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Feature fusion
        self.feature_proj = nn.Linear(feature_dim, 128)

        # Prediction head
        total_dim = hidden_dim + 128
        self.head = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        features: OptTensor = None,
        batch: OptTensor = None
    ) -> Tuple[Tensor, dict]:
        """Forward pass."""
        # Encode atom features
        x = self.atom_encoder(x)

        # GIN layers
        layer_outputs = []
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            layer_outputs.append(x)

        # Global pooling
        graph_emb = global_mean_pool(x, batch)

        # Feature fusion
        if features is not None:
            feat_emb = self.feature_proj(features)
            combined = torch.cat([graph_emb, feat_emb], dim=-1)
        else:
            combined = graph_emb

        # Prediction
        prediction = self.head(combined)

        info = {
            'layer_outputs': layer_outputs,
            'graph_emb': graph_emb
        }

        return prediction, info


if __name__ == "__main__":
    # Test model
    print("Testing DCN-GIN model...")

    # Create model
    model = DCNGIN(num_layers=5, hidden_dim=256, growth_rate=32, use_dense=True)
    print(f"DCN-GIN-5 parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = DCNGIN(num_layers=10, hidden_dim=256, growth_rate=32, use_dense=True)
    print(f"DCN-GIN-10 parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = DCNGIN(num_layers=15, hidden_dim=256, growth_rate=32, use_dense=True)
    print(f"DCN-GIN-15 parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test standard GIN
    model = StandardGIN(num_layers=10, hidden_dim=256)
    print(f"Standard-GIN-10 parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nAll models created successfully!")
