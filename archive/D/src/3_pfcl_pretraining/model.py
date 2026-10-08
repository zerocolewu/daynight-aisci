"""
PF-CL Model Architecture
GIN encoder with contrastive and feature prediction heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Optional


class GINEncoder(nn.Module):
    """GIN encoder for graph representation learning"""

    def __init__(
        self,
        node_dim: int = 37,
        edge_dim: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 5,
        dropout: float = 0.1
    ):
        super(GINEncoder, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            nn_layers = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            conv = GINConv(nn_layers)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Final projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through encoder"""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Project input features
        x = F.relu(self.node_proj(x))
        edge_attr = F.relu(self.edge_proj(edge_attr))

        # Apply GIN layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            identity = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

            # Residual connection (except first layer)
            if i > 0 and x.shape == identity.shape:
                x = x + identity

            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level pooling
        x = global_mean_pool(x, batch)

        # Final projection
        x = self.output_proj(x)

        return x


class ContrastiveProjectionHead(nn.Module):
    """Projection head for contrastive learning"""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.1
    ):
        super(ContrastiveProjectionHead, self).__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class FeaturePredictionHead(nn.Module):
    """Feature prediction head for molecular properties"""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 148,
        dropout: float = 0.1
    ):
        super(FeaturePredictionHead, self).__init__()

        self.prediction = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prediction(x)


class FineTuningHead(nn.Module):
    """Fine-tuning head for property prediction"""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list = [128, 64],
        output_dim: int = 1,
        dropout: float = 0.1
    ):
        super(FineTuningHead, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.prediction = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prediction(x).squeeze(-1)


class PFCLModel(nn.Module):
    """Complete PF-CL model for pre-training"""

    def __init__(
        self,
        node_dim: int = 37,
        edge_dim: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 5,
        proj_dim: int = 64,
        feat_dim: int = 154,
        dropout: float = 0.1,
        use_feature_head: bool = True,
        use_contrastive_head: bool = True
    ):
        super(PFCLModel, self).__init__()

        self.use_feature_head = use_feature_head
        self.use_contrastive_head = use_contrastive_head

        # Encoder
        self.encoder = GINEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # Contrastive projection head
        if use_contrastive_head:
            self.contrastive_head = ContrastiveProjectionHead(
                input_dim=hidden_dim,
                hidden_dim=128,
                output_dim=proj_dim,
                dropout=dropout
            )

        # Feature prediction head
        if use_feature_head:
            self.feature_head = FeaturePredictionHead(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=feat_dim,
                dropout=dropout
            )

    def forward(
        self,
        data: Data,
        return_features: bool = False
    ) -> dict:
        """Forward pass"""
        # Encode
        z = self.encoder(data)

        outputs = {'embedding': z}

        # Contrastive projection
        if self.use_contrastive_head:
            proj = self.contrastive_head(z)
            outputs['projection'] = proj

        # Feature prediction
        if self.use_feature_head:
            features = self.feature_head(z)
            outputs['features'] = features

        if return_features:
            outputs['encoder_features'] = z

        return outputs

    def save_encoder(self, path: str):
        """Save encoder weights"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'config': {
                'node_dim': self.encoder.node_dim,
                'edge_dim': self.encoder.edge_dim,
                'hidden_dim': self.encoder.hidden_dim,
                'num_layers': self.encoder.num_layers,
                'dropout': self.encoder.dropout
            }
        }, path)

    def load_encoder(self, path: str):
        """Load encoder weights"""
        checkpoint = torch.load(path, map_location='cpu')
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])


class FineTuningModel(nn.Module):
    """Model for fine-tuning on property prediction"""

    def __init__(
        self,
        encoder_path: str,
        hidden_dim: int = 256,
        hidden_dims: list = [128, 64],
        dropout: float = 0.1,
        freeze_encoder: bool = False
    ):
        super(FineTuningModel, self).__init__()

        # Load pre-trained encoder
        checkpoint = torch.load(encoder_path, map_location='cpu')
        config = checkpoint['config']

        self.encoder = GINEncoder(
            node_dim=config['node_dim'],
            edge_dim=config['edge_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Fine-tuning head
        self.head = FineTuningHead(
            input_dim=hidden_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=dropout
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass"""
        z = self.encoder(data)
        output = self.head(z)
        return output


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Test model architecture"""
    # Create dummy data
    from torch_geometric.data import Batch

    num_nodes = 20
    num_edges = 40

    data = Data(
        x=torch.randn(num_nodes, 30),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        edge_attr=torch.randn(num_edges, 11),
        batch=torch.zeros(num_nodes, dtype=torch.long)
    )

    # Test PF-CL model
    print("Testing PF-CL Model...")
    model = PFCLModel(
        node_dim=30,
        edge_dim=11,
        hidden_dim=256,
        num_layers=5,
        proj_dim=64,
        feat_dim=148,
        dropout=0.1
    )

    print(f"Total parameters: {count_parameters(model):,}")

    outputs = model(data)
    print(f"Output shapes:")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")

    print("\nModel architecture:")
    print(model)


if __name__ == "__main__":
    main()
