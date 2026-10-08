"""
RE-PerioGT: Rigidity-Enhanced Periodic Graph Transformer for Polymer Tg Prediction

Architecture:
1. Graph Transformer with periodic invariance (PerioGT base)
2. Rigidity-aware contrastive learning module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)
from torch_geometric.utils import softmax
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head attention with periodicity awareness."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        periodic_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, n_nodes, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, n_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, n_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, n_nodes, self.num_heads, self.head_dim)

        # Transpose for attention: [batch, heads, nodes, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply periodic mask if provided (reduce attention to periodic atoms)
        if periodic_mask is not None:
            # periodic_mask shape: [batch, nodes, 1]
            mask = periodic_mask.squeeze(-1).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, nodes]
            scores = scores * (1 - mask * 0.5)  # Reduce but don't zero out

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, n_nodes, self.hidden_dim)

        return self.out_proj(out)


class PeriodicGraphTransformerLayer(MessagePassing):
    """
    Graph Transformer layer with periodic invariance.
    Handles polymer periodicity through special attention mechanism.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Attention parameters
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Periodicity encoding
        self.periodic_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        periodic_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            periodic_mask: Mask for periodic atoms [num_nodes, 1]
        """
        # Self-attention with residual
        h = self.norm1(x)
        attn_out = self.propagate(
            edge_index,
            x=h,
            edge_attr=edge_attr,
            periodic_mask=periodic_mask,
        )
        x = x + self.dropout(attn_out)

        # FFN with residual
        h = self.norm2(x)
        x = x + self.ffn(h)

        return x

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
        periodic_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention-weighted messages."""

        # Project to Q, K, V
        q = self.q_proj(x_i).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x_j).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x_j).view(-1, self.num_heads, self.head_dim)

        # Edge features
        edge_feat = self.edge_proj(edge_attr)
        k = k + edge_feat.view(-1, self.num_heads, self.head_dim)

        # Attention scores
        alpha = (q * k).sum(dim=-1) / self.scale

        # Apply periodic mask
        if periodic_mask is not None:
            # For source nodes (j), reduce their attention weight if periodic
            # periodic_mask shape: [num_nodes, 1], we need mask for source nodes of each edge
            mask = periodic_mask.view(-1)  # [num_nodes]
            mask_j = mask[index]  # [num_edges] - mask for source nodes
            alpha = alpha * (1 - mask_j.unsqueeze(1) * 0.3)  # [num_edges, num_heads]

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.dropout(alpha)

        # Weighted sum
        out = v * alpha.unsqueeze(-1)
        return out.view(-1, self.hidden_dim)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return self.out_proj(aggr_out)


class RigidityEncoder(nn.Module):
    """Encode rigidity (rotatable bond fraction) into embedding."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, rigidity: torch.Tensor) -> torch.Tensor:
        """rigidity: [batch_size, 1] -> [batch_size, hidden_dim]"""
        return self.encoder(rigidity)


class RE_PerioGT(nn.Module):
    """
    Rigidity-Enhanced Periodic Graph Transformer.

    Features:
    - Graph Transformer with periodic invariance
    - Rigidity-aware contrastive learning
    """

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_rigidity_feature: bool = True,
        use_contrastive: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_rigidity_feature = use_rigidity_feature
        self.use_contrastive = use_contrastive

        # Input projections
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(bond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Periodic position encoding
        self.periodic_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # mask(1) + vector(3)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Graph Transformer layers
        self.layers = nn.ModuleList([
            PeriodicGraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Rigidity encoder
        self.rigidity_encoder = RigidityEncoder(hidden_dim)

        # Output head for Tg prediction
        predictor_input_dim = hidden_dim
        if use_rigidity_feature:
            predictor_input_dim += hidden_dim

        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Projection head for contrastive learning
        if use_contrastive:
            self.projection_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        rigidity: torch.Tensor,
        periodic_mask: torch.Tensor,
        periodic_vectors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, atom_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, bond_dim]
            batch: Batch assignment [num_nodes]
            rigidity: Rigidity values [batch_size, 1]
            periodic_mask: Periodic atom mask [num_nodes, 1]
            periodic_vectors: Periodic vectors [num_nodes, 3]

        Returns:
            prediction: Tg prediction [batch_size, 1]
            embedding: Graph embedding [batch_size, hidden_dim]
        """
        # Encode atoms
        h = self.atom_encoder(x)

        # Add periodic position encoding
        periodic_features = torch.cat([periodic_mask, periodic_vectors], dim=-1)
        h = h + self.periodic_encoder(periodic_features)

        # Encode edges
        edge_h = self.edge_encoder(edge_attr)

        # Apply Graph Transformer layers
        for layer in self.layers:
            h = layer(h, edge_index, edge_h, periodic_mask)

        # Global pooling
        graph_embedding = global_mean_pool(h, batch)

        # Prepare prediction input
        pred_input = graph_embedding
        if self.use_rigidity_feature:
            # Ensure rigidity has correct shape [batch_size, 1]
            if rigidity.dim() == 1:
                rigidity = rigidity.unsqueeze(-1)
            rigidity_emb = self.rigidity_encoder(rigidity)
            pred_input = torch.cat([graph_embedding, rigidity_emb], dim=-1)

        # Predict Tg
        prediction = self.predictor(pred_input)

        return prediction, graph_embedding

    def get_projection(self, embedding: torch.Tensor) -> torch.Tensor:
        """Get projection for contrastive learning."""
        if self.use_contrastive:
            return self.projection_head(embedding)
        return embedding


class ContrastiveLoss(nn.Module):
    """
    Rigidity-aware contrastive loss using InfoNCE.
    Positive pairs: similar rigidity (|Δ| < pos_threshold)
    Negative pairs: different rigidity (|Δ| > neg_threshold)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        pos_threshold: float = 0.1,
        neg_threshold: float = 0.3,
    ):
        super().__init__()
        self.temperature = temperature
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def forward(
        self,
        projections: torch.Tensor,
        rigidities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute rigidity-aware contrastive loss.

        Args:
            projections: Projected embeddings [batch_size, proj_dim]
            rigidities: Rigidity values [batch_size, 1]

        Returns:
            Contrastive loss value
        """
        batch_size = projections.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=projections.device)

        # Ensure rigidity has correct shape [batch_size, 1]
        if rigidities.dim() == 1:
            rigidities = rigidities.unsqueeze(-1)

        # Normalize projections
        projections = F.normalize(projections, dim=1)

        # Compute similarity matrix
        similarity = torch.mm(projections, projections.t()) / self.temperature

        # Create rigidity-based masks
        rigidity_diff = torch.abs(rigidities - rigidities.t())

        # Positive mask: similar rigidity
        pos_mask = (rigidity_diff < self.pos_threshold).float()
        pos_mask.fill_diagonal_(0)  # Exclude self

        # Negative mask: different rigidity
        neg_mask = (rigidity_diff > self.neg_threshold).float()

        # Count positive pairs per sample
        num_positives = pos_mask.sum(dim=1)
        valid_samples = num_positives > 0

        if not valid_samples.any():
            return torch.tensor(0.0, device=projections.device)

        # Compute InfoNCE loss
        # For each anchor, positive samples have similar rigidity
        # Negative samples have different rigidity

        loss = 0.0
        num_valid = 0

        for i in range(batch_size):
            if num_positives[i] == 0:
                continue

            # Positive similarities
            pos_sims = similarity[i][pos_mask[i].bool()]

            # All similarities except self
            all_sims = similarity[i]
            all_sims_exp = torch.exp(all_sims)

            # For each positive, compute NCE loss
            for pos_sim in pos_sims:
                # Denominator: all other samples
                denom = all_sims_exp.sum() - all_sims_exp[i]

                if denom > 0:
                    nce_loss = -pos_sim + torch.log(denom + 1e-8)
                    loss = loss + nce_loss
                    num_valid += 1

        if num_valid > 0:
            loss = loss / num_valid

        return loss


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    from dataset import get_feature_dimensions

    atom_dim, bond_dim = get_feature_dimensions()
    print(f"Atom dim: {atom_dim}, Bond dim: {bond_dim}")

    model = RE_PerioGT(
        atom_dim=atom_dim,
        bond_dim=bond_dim,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        use_rigidity_feature=True,
        use_contrastive=True,
    )

    print(f"\nModel parameters: {count_parameters(model):,}")

    # Test forward pass with dummy data
    num_nodes = 20
    num_edges = 40
    batch_size = 4

    x = torch.randn(num_nodes, atom_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, bond_dim)
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    rigidity = torch.rand(batch_size, 1)
    periodic_mask = torch.zeros(num_nodes, 1)
    periodic_vectors = torch.randn(num_nodes, 3)

    pred, emb = model(x, edge_index, edge_attr, batch, rigidity, periodic_mask, periodic_vectors)
    print(f"\nPrediction shape: {pred.shape}")
    print(f"Embedding shape: {emb.shape}")

    # Test contrastive loss
    proj = model.get_projection(emb)
    print(f"Projection shape: {proj.shape}")

    contrastive_loss = ContrastiveLoss()
    loss = contrastive_loss(proj, rigidity)
    print(f"Contrastive loss: {loss.item():.4f}")
