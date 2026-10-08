"""
LEMP: Long-range Entanglement-aware Message Passing for Polymer Tg Prediction

Model Architecture:
1. Local Branch: ECFP4 fingerprint + GIN for local substructure encoding
2. Long-Range Branch: Voxel grid + Structure factor for entanglement capture
3. Fusion Layer: Combines local and long-range representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import cdist
import math


class ECFPEncoder(nn.Module):
    """ECFP4 fingerprint encoder with learnable projection"""
    def __init__(self, fp_size=2048, hidden_dim=256):
        super().__init__()
        self.fp_size = fp_size
        self.mlp = nn.Sequential(
            nn.Linear(fp_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, fp):
        return self.mlp(fp)


class LocalGIN(nn.Module):
    """Simple 2-layer GIN for local substructure encoding (5-hop neighborhood)"""
    def __init__(self, node_dim=44, hidden_dim=128, num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        # Node features: atom type (one-hot), degree, hybridization, aromatic, etc.
        self.node_encoder = nn.Linear(node_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        # Encode node features
        x = self.node_encoder(x)

        # Message passing
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        # Global pooling
        x = global_mean_pool(x, batch)
        return x


class VoxelGridEncoder(nn.Module):
    """
    Voxel Grid Representation for 3D structure encoding
    Places atoms in 3D voxel grid with rotation-invariant local frames
    """
    def __init__(self, grid_size=16, voxel_channels=8, hidden_dim=128):
        super().__init__()
        self.grid_size = grid_size
        self.voxel_channels = voxel_channels

        # 3D CNN for voxel grid processing
        self.conv3d = nn.Sequential(
            nn.Conv3d(voxel_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )

        self.fc = nn.Linear(128, hidden_dim)

    def create_voxel_grid(self, positions, atom_types, grid_size, voxel_channels):
        """
        Create voxel grid from 3D positions
        Uses rotation-invariant local frame based on principal components
        """
        batch_size = len(positions)
        grids = torch.zeros(batch_size, voxel_channels, grid_size, grid_size, grid_size,
                          device=positions[0].device if positions else 'cpu')

        for b, (pos, types) in enumerate(zip(positions, atom_types)):
            if pos is None or len(pos) == 0:
                continue

            pos = torch.tensor(pos, device=grids.device)
            types = torch.tensor(types, device=grids.device)

            # Center the structure
            centroid = pos.mean(dim=0)
            pos_centered = pos - centroid

            # Compute rotation-invariant frame using PCA
            if len(pos_centered) > 2:
                cov = torch.cov(pos_centered.T)
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                # Sort by eigenvalue (descending)
                idx = torch.argsort(eigenvalues, descending=True)
                eigenvectors = eigenvectors[:, idx]
                # Rotate to principal frame
                pos_rotated = torch.mm(pos_centered, eigenvectors)
            else:
                pos_rotated = pos_centered

            # Normalize to [-1, 1] range
            max_extent = pos_rotated.abs().max() + 1e-6
            pos_normalized = pos_rotated / max_extent

            # Map to voxel indices
            indices = ((pos_normalized + 1) / 2 * (grid_size - 1)).long()
            indices = torch.clamp(indices, 0, grid_size - 1)

            # Fill voxel grid with atom type channels
            for i, (idx, atom_type) in enumerate(zip(indices, types)):
                x, y, z = idx
                atom_channel = min(int(atom_type) % voxel_channels, voxel_channels - 1)
                grids[b, atom_channel, x, y, z] += 1

        return grids

    def forward(self, positions_list, atom_types_list):
        grids = self.create_voxel_grid(positions_list, atom_types_list,
                                       self.grid_size, self.voxel_channels)
        grids = grids.to(next(self.parameters()).device)
        features = self.conv3d(grids)
        features = features.view(features.size(0), -1)
        return self.fc(features)


class StructureFactorEncoder(nn.Module):
    """
    Structure Factor Embedding: S(k) = Σ_i exp(i·k·r_i)
    Computes Fourier-space representation for capturing long-range correlations
    """
    def __init__(self, num_kpoints=32, hidden_dim=128):
        super().__init__()
        self.num_kpoints = num_kpoints

        # Learnable k-point sampling weights
        self.k_encoder = nn.Sequential(
            nn.Linear(num_kpoints * 6, hidden_dim),  # 6 = 2 (real+imag) * 3 (xyz)
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Pre-define k-points on a spherical shell
        self.register_buffer('k_points', self._generate_k_points(num_kpoints))

    def _generate_k_points(self, num_points):
        """Generate k-points uniformly distributed on spherical shells"""
        # Use golden spiral for uniform distribution
        indices = torch.arange(0, num_points, dtype=torch.float32)
        phi = torch.acos(1 - 2 * (indices + 0.5) / num_points)
        theta = math.pi * (1 + 5**0.5) * indices

        k_x = torch.sin(phi) * torch.cos(theta)
        k_y = torch.sin(phi) * torch.sin(theta)
        k_z = torch.cos(phi)

        # Stack into (num_points, 3)
        k_points = torch.stack([k_x, k_y, k_z], dim=1)
        return k_points

    def compute_structure_factor(self, positions, k_points):
        """
        Compute S(k) = Σ_i exp(i·k·r_i) for given positions and k-points
        Returns 6*K features: real/imag parts for x, y, z components
        """
        if positions is None or len(positions) == 0:
            return torch.zeros(k_points.shape[0] * 6, device=k_points.device)

        positions = torch.tensor(positions, device=k_points.device)

        # Center positions
        positions = positions - positions.mean(dim=0)

        # Normalize
        max_dist = positions.norm(dim=1).max() + 1e-6
        positions = positions / max_dist

        # Separate x, y, z components
        pos_x = positions[:, 0:1]  # (N, 1)
        pos_y = positions[:, 1:2]  # (N, 1)
        pos_z = positions[:, 2:3]  # (N, 1)

        # Compute phase: k · r for each component
        # k_points: (K, 3)
        k_x = k_points[:, 0:1].T  # (1, K)
        k_y = k_points[:, 1:2].T  # (1, K)
        k_z = k_points[:, 2:3].T  # (1, K)

        # Phase for each component
        phase_x = torch.mm(pos_x, k_x)  # (N, K)
        phase_y = torch.mm(pos_y, k_y)  # (N, K)
        phase_z = torch.mm(pos_z, k_z)  # (N, K)

        # Compute structure factor for each component
        sf_x_real = torch.cos(phase_x).sum(dim=0)  # (K,)
        sf_x_imag = torch.sin(phase_x).sum(dim=0)  # (K,)
        sf_y_real = torch.cos(phase_y).sum(dim=0)  # (K,)
        sf_y_imag = torch.sin(phase_y).sum(dim=0)  # (K,)
        sf_z_real = torch.cos(phase_z).sum(dim=0)  # (K,)
        sf_z_imag = torch.sin(phase_z).sum(dim=0)  # (K,)

        # Concatenate all: 6 features per k-point
        return torch.cat([sf_x_real, sf_x_imag, sf_y_real, sf_y_imag, sf_z_real, sf_z_imag], dim=0)  # (6K,)

    def forward(self, positions_list):
        batch_size = len(positions_list)
        features = []

        for positions in positions_list:
            sf = self.compute_structure_factor(positions, self.k_points)
            features.append(sf)

        features = torch.stack(features, dim=0)  # (batch, 2*K)
        return self.k_encoder(features)


class EwaldLongRangeEncoder(nn.Module):
    """
    Ewald-Style Filtering for long-range interactions
    Splits interactions into short-range (real-space, <cutoff) and
    long-range (Fourier-space, >cutoff) components
    """
    def __init__(self, cutoff=5.0, hidden_dim=128):
        super().__init__()
        self.cutoff = cutoff

        # Long-range feature encoder
        self.long_range_encoder = nn.Sequential(
            nn.Linear(64, hidden_dim),  # Encoded long-range correlations
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def compute_long_range_correlations(self, positions, cutoff):
        """
        Compute long-range correlation features
        Based on distance histogram beyond cutoff
        """
        if positions is None or len(positions) < 2:
            return torch.zeros(64, device=positions.device if isinstance(positions, torch.Tensor) else 'cpu')

        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, device='cpu')

        # Compute pairwise distances
        dists = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0)).squeeze(0)

        # Create distance histogram for long-range interactions
        max_dist = dists.max() + 1e-6
        bins = torch.linspace(cutoff, max(max_dist, cutoff + 1), 64, device=positions.device)

        # Mask out short-range and diagonal
        mask = (dists > cutoff) & (dists < max_dist)
        long_range_dists = dists[mask]

        if len(long_range_dists) == 0:
            return torch.zeros(64, device=positions.device)

        # Histogram of long-range distances
        hist = torch.histc(long_range_dists, bins=64, min=cutoff, max=max_dist)

        # Normalize
        hist = hist / (hist.sum() + 1e-6)

        return hist

    def forward(self, positions_list):
        batch_size = len(positions_list)
        features = []

        for positions in positions_list:
            lrf = self.compute_long_range_correlations(positions, self.cutoff)
            features.append(lrf)

        features = torch.stack(features, dim=0)
        if features.device != next(self.parameters()).device:
            features = features.to(next(self.parameters()).device)

        return self.long_range_encoder(features)


class LongRangeBranch(nn.Module):
    """
    Complete Long-Range Branch combining:
    - Voxel Grid Encoder
    - Structure Factor Encoder
    - Ewald Long-Range Encoder
    """
    def __init__(self, grid_size=16, voxel_channels=8, num_kpoints=32,
                 cutoff=5.0, hidden_dim=128):
        super().__init__()

        self.voxel_encoder = VoxelGridEncoder(grid_size, voxel_channels, hidden_dim)
        self.sf_encoder = StructureFactorEncoder(num_kpoints, hidden_dim)
        self.ewald_encoder = EwaldLongRangeEncoder(cutoff, hidden_dim)

        # Fusion of long-range features
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, positions_list, atom_types_list):
        h_voxel = self.voxel_encoder(positions_list, atom_types_list)
        h_sf = self.sf_encoder(positions_list)
        h_ewald = self.ewald_encoder(positions_list)

        # Concatenate and fuse
        h_long = torch.cat([h_voxel, h_sf, h_ewald], dim=1)
        return self.fusion(h_long)


class LEMP(nn.Module):
    """
    LEMP: Long-range Entanglement-aware Message Passing

    Architecture:
    1. Local Branch: ECFP4 + GIN
    2. Long-Range Branch: Voxel + Structure Factor + Ewald
    3. Fusion: Combined prediction
    """
    def __init__(self, fp_size=2048, node_dim=44, hidden_dim=256,
                 grid_size=16, voxel_channels=8, num_kpoints=32,
                 cutoff=5.0, num_gin_layers=2, use_long_range=True,
                 use_fingerprint=True, use_gin=True):
        super().__init__()

        self.use_long_range = use_long_range
        self.use_fingerprint = use_fingerprint
        self.use_gin = use_gin

        # Determine input dimension for fusion
        fusion_dim = 0

        # Local Branch
        if use_fingerprint:
            self.fp_encoder = ECFPEncoder(fp_size, hidden_dim)
            fusion_dim += hidden_dim

        if use_gin:
            self.local_gin = LocalGIN(node_dim, hidden_dim, num_gin_layers)
            fusion_dim += hidden_dim

        # Long-Range Branch
        if use_long_range:
            self.long_range_branch = LongRangeBranch(
                grid_size, voxel_channels, num_kpoints, cutoff, hidden_dim
            )
            fusion_dim += hidden_dim

        # Fusion and Prediction
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.predictor = nn.Linear(hidden_dim // 2, 1)

    def forward(self, fp, x, edge_index, batch, positions_list, atom_types_list):
        features = []

        # Local Branch
        if self.use_fingerprint and fp is not None:
            h_fp = self.fp_encoder(fp)
            features.append(h_fp)

        if self.use_gin and x is not None:
            h_gin = self.local_gin(x, edge_index, batch)
            features.append(h_gin)

        # Long-Range Branch
        if self.use_long_range:
            h_long = self.long_range_branch(positions_list, atom_types_list)
            features.append(h_long)

        # Fusion
        h_final = torch.cat(features, dim=1)
        h_fused = self.fusion(h_final)

        # Prediction
        output = self.predictor(h_fused)
        return output.squeeze(-1)
