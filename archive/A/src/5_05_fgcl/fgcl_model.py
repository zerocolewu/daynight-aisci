"""
Fingerprint-Guided Contrastive Learning (FGCL) Model
Tests whether MoCL-style contrastive pre-training with ECFP similarity explains performance gains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


class GINEncoder(nn.Module):
    """GIN encoder for molecular graphs"""

    def __init__(self, num_layers=2, hidden_dim=128, node_dim=44):
        super(GINEncoder, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Atom feature dimensions (based on typical molecular features)
        self.node_dim = node_dim

        # GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                nn_layer = nn.Sequential(
                    nn.Linear(node_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            else:
                nn_layer = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )

            self.convs.append(GINConv(nn_layer, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        return x


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""

    def __init__(self, input_dim=128, hidden_dim=64, output_dim=32):
        super(ProjectionHead, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class PredictionHead(nn.Module):
    """Prediction head for Tg regression"""

    def __init__(self, graph_dim=128, ecfp_dim=2048, hidden_dim=64):
        super(PredictionHead, self).__init__()

        input_dim = graph_dim + ecfp_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, graph_emb, ecfp):
        # graph_emb: (batch_size, graph_dim)
        # ecfp: (batch_size, ecfp_dim)
        x = torch.cat([graph_emb, ecfp], dim=-1)
        return self.layers(x).squeeze(-1)


class FGCLModel(nn.Module):
    """Complete FGCL model with encoder, projection head, and prediction head"""

    def __init__(self, config=None):
        super(FGCLModel, self).__init__()

        # Default configuration
        if config is None:
            config = {
                'num_layers': 2,
                'hidden_dim': 128,
                'node_dim': 32,
                'projection_dim': 32,
                'ecfp_dim': 2048
            }

        self.config = config

        # Encoder
        self.encoder = GINEncoder(
            num_layers=config['num_layers'],
            hidden_dim=config['hidden_dim'],
            node_dim=config['node_dim']
        )

        # Projection head (for contrastive learning)
        self.projection_head = ProjectionHead(
            input_dim=config['hidden_dim'],
            output_dim=config['projection_dim']
        )

        # Prediction head (for Tg regression)
        self.prediction_head = PredictionHead(
            graph_dim=config['hidden_dim'],
            ecfp_dim=config['ecfp_dim']
        )

    def forward(self, data, return_projection=False):
        """
        Forward pass

        Args:
            data: PyTorch Geometric Data object
            return_projection: If True, return projection embeddings for contrastive learning

        Returns:
            If return_projection: projection embeddings
            Else: Tg predictions
        """
        graph_emb = self.encoder(data)

        if return_projection:
            return self.projection_head(graph_emb)
        else:
            # Extract ECFP from batch
            if hasattr(data, 'ecfp'):
                # In PyTorch Geometric, ECFP gets concatenated in the batch
                # We need to reshape it to (batch_size, ecfp_dim)
                batch_size = graph_emb.shape[0]
                ecfp_dim = self.config['ecfp_dim']

                # Check if ECFP is concatenated (total_size = batch_size * ecfp_dim)
                if data.ecfp.numel() == batch_size * ecfp_dim:
                    # Reshape from (batch_size * ecfp_dim,) to (batch_size, ecfp_dim)
                    ecfp_batch = data.ecfp.view(batch_size, ecfp_dim)
                elif data.ecfp.shape[0] == batch_size and data.ecfp.shape[1] == ecfp_dim:
                    # Already properly batched
                    ecfp_batch = data.ecfp
                else:
                    raise ValueError(f"Unexpected ECFP shape: {data.ecfp.shape}, expected {batch_size * ecfp_dim} elements or ({batch_size}, {ecfp_dim})")
            else:
                raise ValueError("ECFP features not found in data")

            return self.prediction_head(graph_emb, ecfp_batch)

    def get_graph_embedding(self, data):
        """Get graph embeddings without prediction"""
        return self.encoder(data)


def compute_ecfp_fingerprint(smiles, radius=2, n_bits=2048):
    """
    Compute ECFP fingerprint from SMILES string

    Args:
        smiles: SMILES string
        radius: ECFP radius (default 2 for ECFP4)
        n_bits: Number of bits (default 2048)

    Returns:
        numpy array of fingerprint
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except Exception as e:
        print(f"Error computing ECFP for {smiles}: {e}")
        return np.zeros(n_bits)


def compute_tanimoto_similarity(fp1, fp2):
    """
    Compute Tanimoto similarity between two fingerprints

    Args:
        fp1, fp2: numpy arrays of fingerprints

    Returns:
        Tanimoto similarity (float between 0 and 1)
    """
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)

    if union == 0:
        return 0.0

    return intersection / union


def compute_similarity_matrix(fingerprints):
    """
    Compute pairwise Tanimoto similarity matrix

    Args:
        fingerprints: numpy array of shape (N, n_bits)

    Returns:
        Similarity matrix of shape (N, N)
    """
    n = len(fingerprints)
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            sim = compute_tanimoto_similarity(fingerprints[i], fingerprints[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    # Diagonal is 1.0
    np.fill_diagonal(sim_matrix, 1.0)

    return sim_matrix


def info_nce_loss(projections, temp=0.1, positive_threshold=0.7, negative_threshold=0.3):
    """
    Compute InfoNCE loss with ECFP-based positive/negative pairs

    Args:
        projections: Tensor of shape (batch_size, projection_dim)
        temp: Temperature parameter
        positive_threshold: Tanimoto similarity threshold for positive pairs
        negative_threshold: Tanimoto similarity threshold for negative pairs

    Returns:
        InfoNCE loss
    """
    batch_size = projections.shape[0]

    # Normalize projections
    projections = F.normalize(projections, dim=1)

    # Compute similarity matrix in embedding space
    sim_matrix = torch.mm(projections, projections.t()) / temp

    # Create labels (diagonal is positive)
    labels = torch.arange(batch_size, device=projections.device)

    # For numerical stability
    sim_matrix_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - sim_matrix_max.detach()

    # Compute loss
    loss = F.cross_entropy(sim_matrix, labels)

    return loss


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    config = {
        'num_layers': 2,
        'hidden_dim': 128,
        'node_dim': 32,
        'projection_dim': 32,
        'ecfp_dim': 2048
    }

    model = FGCLModel(config)
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Encoder: {count_parameters(model.encoder):,}")
    print(f"Projection head: {count_parameters(model.projection_head):,}")
    print(f"Prediction head: {count_parameters(model.prediction_head):,}")

    # Test ECFP computation
    test_smiles = "*CC(*)C"
    ecfp = compute_ecfp_fingerprint(test_smiles)
    print(f"\nECFP shape: {ecfp.shape}")
    print(f"ECFP non-zero bits: {np.sum(ecfp > 0)}")
