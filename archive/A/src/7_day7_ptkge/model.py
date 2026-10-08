"""
PTKGE Model - Periodic Table Knowledge Graph Embeddings for Polymer Property Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import Literal, Optional, Tuple, Dict
import numpy as np

from embeddings import PeriodicEmbedding


class GINEncoder(nn.Module):
    """
    GIN encoder with periodic table embeddings for atom features.
    """

    def __init__(
        self,
        num_layers: int = 5,
        hidden_dim: int = 256,
        embedding_strategy: Literal["physics", "learnable", "random", "onehot"] = "physics",
        embedding_dim: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_strategy = embedding_strategy

        # Periodic table embeddings
        self.atom_embedding = PeriodicEmbedding(
            strategy=embedding_strategy,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        )

        # GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            # GIN uses MLP for message passing
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Atomic numbers [num_nodes]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices [num_nodes]

        Returns:
            Graph embeddings [batch_size, hidden_dim]
        """
        # Get atom embeddings
        x = self.atom_embedding(x.unsqueeze(0)).squeeze(0)  # [num_nodes, hidden_dim]

        # GIN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        return x


class ECFP4Encoder(nn.Module):
    """
    ECFP4 fingerprint encoder using MLP.
    Since we don't have precomputed ECFP4, we'll compute it on-the-fly.
    """

    def __init__(self, fp_dim: int = 2048, hidden_dim: int = 256):
        super().__init__()

        self.fp_dim = fp_dim
        self.encoder = nn.Sequential(
            nn.Linear(fp_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, fingerprints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fingerprints: ECFP4 fingerprints [batch_size, fp_dim]

        Returns:
            Encoded fingerprints [batch_size, hidden_dim]
        """
        return self.encoder(fingerprints)


class PTKGEModel(nn.Module):
    """
    PTKGE Model for polymer property prediction.

    Combines GNN encoder with ECFP4 fingerprint encoder,
    using periodic table embeddings for atom features.
    """

    def __init__(
        self,
        num_layers: int = 5,
        hidden_dim: int = 256,
        embedding_strategy: Literal["physics", "learnable", "random", "onehot"] = "physics",
        embedding_dim: int = 8,
        fp_dim: int = 2048,
        dropout: float = 0.1,
        output_dim: int = 1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_strategy = embedding_strategy

        # GNN encoder
        self.gnn_encoder = GINEncoder(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            embedding_strategy=embedding_strategy,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        # ECFP4 encoder
        self.ecfp_encoder = ECFP4Encoder(fp_dim=fp_dim, hidden_dim=hidden_dim)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        fingerprints: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Atomic numbers [num_nodes]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices [num_nodes]
            fingerprints: ECFP4 fingerprints [batch_size, fp_dim]

        Returns:
            Predictions [batch_size, output_dim]
        """
        # Encode molecular graph
        graph_emb = self.gnn_encoder(x, edge_index, batch)  # [batch, hidden]

        # Encode fingerprints
        fp_emb = self.ecfp_encoder(fingerprints)  # [batch, hidden]

        # Fusion
        combined = torch.cat([graph_emb, fp_emb], dim=1)  # [batch, hidden*2]
        fused = self.fusion(combined)  # [batch, hidden]

        # Predict
        output = self.predictor(fused)  # [batch, output_dim]

        return output

    def get_graph_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        fingerprints: torch.Tensor
    ) -> torch.Tensor:
        """Get graph-level embedding (for contrastive learning)."""
        graph_emb = self.gnn_encoder(x, edge_index, batch)
        fp_emb = self.ecfp_encoder(fingerprints)
        combined = torch.cat([graph_emb, fp_emb], dim=1)
        return self.fusion(combined)

    def get_encoder_params(self):
        """Get parameters for encoder (for different learning rates)."""
        encoder_params = []
        encoder_params.extend(self.gnn_encoder.parameters())
        encoder_params.extend(self.ecfp_encoder.parameters())
        encoder_params.extend(self.fusion.parameters())
        return encoder_params

    def get_predictor_params(self):
        """Get parameters for predictor head."""
        return self.predictor.parameters()


class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss with temperature scaling.
    Uses ECFP4 similarity as positive/negative guidance.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        similarity_matrix: torch.Tensor,
        threshold: float = 0.7
    ) -> torch.Tensor:
        """
        Args:
            embeddings: Graph embeddings [batch_size, hidden_dim]
            similarity_matrix: ECFP4 Tanimoto similarity [batch_size, batch_size]
            threshold: Similarity threshold for positive pairs

        Returns:
            Contrastive loss
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, dim=1)

        # Compute similarity matrix
        sim = torch.mm(embeddings_norm, embeddings_norm.t()) / self.temperature

        # Create positive/negative mask based on ECFP4 similarity
        positive_mask = (similarity_matrix > threshold).float()
        # Remove diagonal
        positive_mask.fill_diagonal_(0)

        # If no positive pairs, use standard contrastive loss with self as positive
        if positive_mask.sum() == 0:
            # Use diagonal as positives (each sample with itself in augmented view)
            labels = torch.arange(batch_size, device=device)
            loss = F.cross_entropy(sim, labels)
            return loss

        # InfoNCE with multiple positives
        # For each anchor, positives are samples with high ECFP4 similarity
        exp_sim = torch.exp(sim)

        # Compute loss
        loss = 0.0
        num_anchors = 0

        for i in range(batch_size):
            positives = positive_mask[i]
            if positives.sum() > 0:
                # Numerator: sum of positive similarities
                pos_sim = (exp_sim[i] * positives).sum()

                # Denominator: all similarities (excluding self)
                neg_mask = torch.ones(batch_size, device=device)
                neg_mask[i] = 0
                all_sim = (exp_sim[i] * neg_mask).sum()

                # Loss for this anchor
                loss += -torch.log(pos_sim / (all_sim + 1e-8) + 1e-8)
                num_anchors += 1

        if num_anchors > 0:
            loss = loss / num_anchors
        else:
            loss = F.cross_entropy(sim, torch.arange(batch_size, device=device))

        return loss


def compute_ecfp4_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute ECFP4 fingerprint from SMILES."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except:
        return np.zeros(n_bits, dtype=np.float32)


def compute_tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute Tanimoto similarity between two fingerprints."""
    # Convert to int for bitwise operations
    fp1_int = fp1.astype(np.int32)
    fp2_int = fp2.astype(np.int32)
    intersection = np.sum(fp1_int & fp2_int)
    union = np.sum(fp1_int | fp2_int)
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_similarity_matrix(fingerprints: np.ndarray) -> np.ndarray:
    """Compute pairwise Tanimoto similarity matrix."""
    n = len(fingerprints)
    similarity = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity[i, j] = 1.0
            else:
                sim = compute_tanimoto_similarity(fingerprints[i], fingerprints[j])
                similarity[i, j] = sim
                similarity[j, i] = sim

    return similarity


if __name__ == "__main__":
    # Test model
    print("Testing PTKGE model...")

    for strategy in ["physics", "learnable", "random", "onehot"]:
        model = PTKGEModel(
            num_layers=3,
            hidden_dim=128,
            embedding_strategy=strategy,
            embedding_dim=8
        )
        print(f"\n{strategy}:")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass with dummy data
        x = torch.tensor([6, 1, 8, 1, 7, 6, 6, 1])  # Atomic numbers
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7],
                                   [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6]])
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        fingerprints = torch.randn(2, 2048)

        output = model(x, edge_index, batch, fingerprints)
        print(f"  Output shape: {output.shape}")
