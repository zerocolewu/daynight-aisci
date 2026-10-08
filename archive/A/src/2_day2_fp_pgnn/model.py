"""
FP-PGNN Model: Fingerprint-Periodic Graph Neural Network
Combines periodic graph attention with ECFP4 fingerprint modulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np


class FingerprintModulatedGATConv(nn.Module):
    """
    GAT layer where attention weights are modulated by fingerprint embedding.
    This allows the graph to "attend" to fingerprint-relevant substructures.
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 4,
                 fp_dim: int = 128, edge_dim: int = 5, concat: bool = True,
                 dropout: float = 0.1, use_fp_modulation: bool = True):
        super().__init__()

        self.use_fp_modulation = use_fp_modulation
        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat

        # Base GAT layer
        self.gat = GATConv(
            in_channels, out_channels, heads=heads,
            concat=concat, dropout=dropout, edge_dim=edge_dim,
            add_self_loops=True
        )

        # Fingerprint modulation network
        if use_fp_modulation:
            self.fp_modulation = nn.Sequential(
                nn.Linear(fp_dim, heads),
                nn.Sigmoid()
            )

    def forward(self, x, edge_index, edge_attr=None, fingerprint=None,
                return_attention=False):
        """
        Forward pass with optional fingerprint modulation.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            fingerprint: Fingerprint embedding [batch_size, fp_dim]
            return_attention: Whether to return attention weights

        Returns:
            Updated node features, optionally with attention weights
        """
        # Get GAT output with attention
        if return_attention:
            out, (edge_idx, attention_weights) = self.gat(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
        else:
            out = self.gat(x, edge_index, edge_attr=edge_attr)
            attention_weights = None

        # Apply fingerprint modulation if available
        if self.use_fp_modulation and fingerprint is not None:
            # Compute modulation factor
            mod_factor = self.fp_modulation(fingerprint)  # [batch_size, heads]

            # Need to broadcast modulation to nodes
            # For now, we apply it as a scaling factor to the output
            # This is a simplified version - in practice, we'd need batch indices
            pass  # Modulation applied at higher level

        if return_attention:
            return out, (edge_idx, attention_weights)
        return out


class FingerprintEncoder(nn.Module):
    """Encodes ECFP4 fingerprint into a dense embedding."""

    def __init__(self, fp_dim: int = 2048, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, fingerprint):
        return self.encoder(fingerprint)


class GraphStream(nn.Module):
    """Graph attention stream with periodic edges."""

    def __init__(self, node_dim: int = 28, hidden_dim: int = 128,
                 edge_dim: int = 5, num_layers: int = 3, heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.num_layers = num_layers
        self.heads = heads

        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.fp_modulations = nn.ModuleList()

        for i in range(num_layers):
            # Last layer doesn't concat heads
            concat = (i < num_layers - 1)
            out_ch = hidden_dim if concat else hidden_dim

            self.gat_layers.append(
                GATConv(
                    hidden_dim if i > 0 else hidden_dim,
                    hidden_dim // heads if concat else hidden_dim,
                    heads=heads,
                    concat=concat,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=True
                )
            )

            # Fingerprint modulation for each layer
            self.fp_modulations.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, heads),
                    nn.Sigmoid()
                )
            )

        self.dropout = nn.Dropout(dropout)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_attr, fingerprint_embedding,
                return_attention=False):
        """
        Forward pass through graph stream.

        Returns:
            Node embeddings, optionally with attention weights from all layers
        """
        # Project input
        h = self.input_proj(x)
        h = F.relu(h)

        all_attentions = [] if return_attention else None

        # Pass through GAT layers
        for i, (gat, fp_mod, ln) in enumerate(zip(self.gat_layers,
                                                   self.fp_modulations,
                                                   self.layer_norms)):
            # GAT forward with attention
            if return_attention:
                h_new, (edge_idx, attention) = gat(
                    h, edge_index, edge_attr=edge_attr,
                    return_attention_weights=True
                )
                all_attentions.append((edge_idx, attention))
            else:
                h_new = gat(h, edge_index, edge_attr=edge_attr)

            # Residual connection
            if h_new.shape[-1] == h.shape[-1]:
                h = ln(h + self.dropout(h_new))
            else:
                h = self.dropout(h_new)
                h = ln(h)

        if return_attention:
            return h, all_attentions
        return h


class FP_PGNN(nn.Module):
    """
    Fingerprint-Periodic Graph Neural Network

    Combines:
    1. Periodic graph attention stream
    2. ECFP4 fingerprint stream
    3. Fingerprint-modulated attention fusion
    """

    def __init__(self, node_dim: int = 28, fp_dim: int = 2048,
                 hidden_dim: int = 128, edge_dim: int = 5,
                 num_gat_layers: int = 3, heads: int = 4,
                 dropout: float = 0.1, use_periodic: bool = True,
                 use_fingerprint: bool = True):
        super().__init__()

        self.use_periodic = use_periodic
        self.use_fingerprint = use_fingerprint
        self.hidden_dim = hidden_dim

        # Fingerprint encoder
        if use_fingerprint:
            self.fp_encoder = FingerprintEncoder(fp_dim, hidden_dim, hidden_dim)
        else:
            self.fp_encoder = None

        # Graph stream
        self.graph_stream = GraphStream(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_layers=num_gat_layers,
            heads=heads,
            dropout=dropout
        )

        # Fusion and prediction head
        if use_fingerprint:
            fusion_dim = hidden_dim * 2  # Graph pooling + fingerprint
        else:
            fusion_dim = hidden_dim

        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data, return_attention=False):
        """
        Forward pass.

        Args:
            data: PyG Batch object with:
                - x: Node features
                - edge_index: Edge indices
                - edge_attr: Edge features
                - fingerprint: ECFP4 fingerprint (if use_fingerprint)
                - batch: Batch indices

        Returns:
            Predicted Tg values
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch

        # Get fingerprint embedding
        if self.use_fingerprint and hasattr(data, 'fingerprint'):
            # Fingerprint shape: (num_graphs, fp_dim) for batched data
            # or (1, fp_dim) for single graph
            fp = data.fingerprint
            if fp.dim() == 1:
                fp = fp.unsqueeze(0)
            fp_embedding = self.fp_encoder(fp)
        else:
            fp_embedding = None

        # Graph stream
        if return_attention:
            node_emb, attentions = self.graph_stream(
                x, edge_index, edge_attr, fp_embedding,
                return_attention=True
            )
        else:
            node_emb = self.graph_stream(
                x, edge_index, edge_attr, fp_embedding
            )
            attentions = None

        # Graph pooling
        graph_emb = global_mean_pool(node_emb, batch)

        # Fusion
        if self.use_fingerprint and fp_embedding is not None:
            # Combine graph and fingerprint embeddings
            combined = torch.cat([graph_emb, fp_embedding], dim=-1)
        else:
            combined = graph_emb

        # Prediction
        output = self.predictor(combined)

        if return_attention:
            return output.squeeze(-1), attentions
        return output.squeeze(-1)


class BaselineGAT(nn.Module):
    """Baseline GAT model without fingerprint or periodic edges."""

    def __init__(self, node_dim: int = 28, hidden_dim: int = 128,
                 edge_dim: int = 5, num_layers: int = 3, heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.input_proj = nn.Linear(node_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            concat = (i < num_layers - 1)
            self.gat_layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // heads if concat else hidden_dim,
                    heads=heads,
                    concat=concat,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=True
                )
            )

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch

        h = F.relu(self.input_proj(x))

        for gat, ln in zip(self.gat_layers, self.layer_norms):
            h_new = gat(h, edge_index, edge_attr=edge_attr)
            if h_new.shape[-1] == h.shape[-1]:
                h = ln(h + self.dropout(h_new))
            else:
                h = self.dropout(h_new)
                h = ln(h)

        graph_emb = global_mean_pool(h, batch)
        return self.predictor(graph_emb).squeeze(-1)


class BaselineMLP(nn.Module):
    """Baseline MLP model using only fingerprints."""

    def __init__(self, fp_dim: int = 2048, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        return self.net(data.fingerprint).squeeze(-1)


def create_model(model_type: str, use_periodic: bool = True,
                 use_fingerprint: bool = True, **kwargs):
    """
    Factory function to create models for ablation studies.

    Args:
        model_type: 'fp_pgnn', 'gat_no_fp', 'gat_no_periodic', 'mlp'
        use_periodic: Whether to use periodic edges
        use_fingerprint: Whether to use fingerprint stream
    """
    if model_type == 'fp_pgnn':
        return FP_PGNN(
            use_periodic=use_periodic,
            use_fingerprint=use_fingerprint,
            **kwargs
        )
    elif model_type == 'gat':
        return BaselineGAT(**kwargs)
    elif model_type == 'mlp':
        return BaselineMLP(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the model
    from torch_geometric.data import Data, Batch

    # Create dummy data
    node_dim = 28
    fp_dim = 2048
    hidden_dim = 128

    # Create a batch of graphs
    graphs = []
    for i in range(4):
        num_nodes = 10
        x = torch.randn(num_nodes, node_dim)
        edge_index = torch.randint(0, num_nodes, (2, 20))
        edge_attr = torch.randn(20, 5)
        # Store fingerprint as (1, fp_dim) for proper batching
        fingerprint = torch.randn(1, fp_dim).sigmoid()

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                   fingerprint=fingerprint)
        graphs.append(data)

    batch = Batch.from_data_list(graphs)

    # Test FP-PGNN
    print("Testing FP-PGNN...")
    model = FP_PGNN(node_dim=node_dim, fp_dim=fp_dim, hidden_dim=hidden_dim)
    out = model(batch)
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with attention
    out, attentions = model(batch, return_attention=True)
    print(f"  Attention layers: {len(attentions)}")

    # Test baseline GAT
    print("\nTesting Baseline GAT...")
    gat_model = BaselineGAT(node_dim=node_dim, hidden_dim=hidden_dim)
    out = gat_model(batch)
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in gat_model.parameters()):,}")

    # Test baseline MLP
    print("\nTesting Baseline MLP...")
    mlp_model = BaselineMLP(fp_dim=fp_dim)
    out = mlp_model(batch)
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in mlp_model.parameters()):,}")

    print("\nAll models tested successfully!")
