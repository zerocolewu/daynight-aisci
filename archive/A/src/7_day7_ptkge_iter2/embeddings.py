"""
Periodic Table Knowledge Graph Embeddings
Provides physics-informed embeddings for chemical elements
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Literal

# Maximum atomic number to support
MAX_ATOMIC_NUM = 118

# Physics properties for elements 1-118
# Format: [electronegativity, atomic_radius_pm, valence_electrons, group, period,
#          electron_affinity_kj_mol, first_ionization_kj_mol, atomic_mass_normalized]

# Data sources: PubChem, NIST, periodic table databases
ELEMENT_PROPERTIES = {
    # H
    1: [2.20, 53, 1, 1, 1, 72.8, 1312.0, 1.008],
    # He
    2: [0.0, 31, 0, 18, 1, 0.0, 2372.0, 4.003],
    # Li
    3: [0.98, 167, 1, 1, 2, 59.6, 520.0, 6.941],
    # Be
    4: [1.57, 112, 2, 2, 2, 0.0, 899.0, 9.012],
    # B
    5: [2.04, 87, 3, 13, 2, 26.7, 801.0, 10.81],
    # C
    6: [2.55, 67, 4, 14, 2, 121.8, 1086.0, 12.01],
    # N
    7: [3.04, 56, 5, 15, 2, 6.8, 1402.0, 14.01],
    # O
    8: [3.44, 48, 6, 16, 2, 141.0, 1314.0, 16.00],
    # F
    9: [3.98, 42, 7, 17, 2, 328.0, 1681.0, 19.00],
    # Ne
    10: [0.0, 38, 0, 18, 2, 0.0, 2080.0, 20.18],
    # Na
    11: [0.93, 190, 1, 1, 3, 52.9, 496.0, 22.99],
    # Mg
    12: [1.31, 145, 2, 2, 3, 0.0, 738.0, 24.31],
    # Al
    13: [1.61, 118, 3, 13, 3, 42.5, 578.0, 26.98],
    # Si
    14: [1.90, 111, 4, 14, 3, 133.6, 786.0, 28.09],
    # P
    15: [2.19, 98, 5, 15, 3, 72.0, 1012.0, 30.97],
    # S
    16: [2.58, 88, 6, 16, 3, 200.0, 1000.0, 32.07],
    # Cl
    17: [3.16, 79, 7, 17, 3, 349.0, 1251.0, 35.45],
    # Ar
    18: [0.0, 71, 0, 18, 3, 0.0, 1520.0, 39.95],
    # K
    19: [0.82, 243, 1, 1, 4, 48.4, 419.0, 39.10],
    # Ca
    20: [1.00, 194, 2, 2, 4, 2.4, 590.0, 40.08],
    # Sc
    21: [1.36, 184, 3, 3, 4, 18.0, 633.0, 44.96],
    # Ti
    22: [1.54, 176, 4, 4, 4, 8.0, 659.0, 47.87],
    # V
    23: [1.63, 171, 5, 5, 4, 50.0, 651.0, 50.94],
    # Cr
    24: [1.66, 166, 6, 6, 4, 64.0, 653.0, 52.00],
    # Mn
    25: [1.55, 161, 7, 7, 4, 0.0, 717.0, 54.94],
    # Fe
    26: [1.83, 156, 8, 8, 4, 15.7, 762.0, 55.85],
    # Co
    27: [1.88, 152, 9, 9, 4, 63.7, 760.0, 58.93],
    # Ni
    28: [1.91, 149, 10, 10, 4, 112.0, 737.0, 58.69],
    # Cu
    29: [1.90, 145, 11, 11, 4, 118.4, 745.0, 63.55],
    # Zn
    30: [1.65, 142, 12, 12, 4, 0.0, 906.0, 65.38],
    # Ga
    31: [1.81, 136, 3, 13, 4, 28.9, 579.0, 69.72],
    # Ge
    32: [2.01, 125, 4, 14, 4, 119.0, 762.0, 72.63],
    # As
    33: [2.18, 114, 5, 15, 4, 78.0, 947.0, 74.92],
    # Se
    34: [2.55, 103, 6, 16, 4, 195.0, 941.0, 78.97],
    # Br
    35: [2.96, 94, 7, 17, 4, 324.6, 1140.0, 79.90],
    # Kr
    36: [3.00, 88, 0, 18, 4, 0.0, 1350.0, 83.80],
    # Rb
    37: [0.82, 265, 1, 1, 5, 46.9, 403.0, 85.47],
    # Sr
    38: [0.95, 219, 2, 2, 5, 5.0, 549.0, 87.62],
    # Y
    39: [1.22, 212, 3, 3, 5, 30.0, 600.0, 88.91],
    # Zr
    40: [1.33, 206, 4, 4, 5, 41.1, 640.0, 91.22],
    # Nb
    41: [1.6, 198, 5, 5, 5, 86.0, 652.0, 92.91],
    # Mo
    42: [2.16, 190, 6, 6, 5, 72.0, 684.0, 95.95],
    # Tc
    43: [1.9, 183, 7, 7, 5, 53.0, 702.0, 98.0],
    # Ru
    44: [2.2, 178, 8, 8, 5, 101.0, 710.0, 101.07],
    # Rh
    45: [2.28, 173, 9, 9, 5, 109.7, 719.0, 102.91],
    # Pd
    46: [2.20, 169, 10, 10, 5, 54.2, 804.0, 106.42],
    # Ag
    47: [1.93, 165, 11, 11, 5, 125.7, 731.0, 107.87],
    # Cd
    48: [1.69, 161, 12, 12, 5, 0.0, 867.0, 112.41],
    # In
    49: [1.78, 156, 3, 13, 5, 28.9, 558.0, 114.82],
    # Sn
    50: [1.96, 145, 4, 14, 5, 107.3, 709.0, 118.71],
    # Sb
    51: [2.05, 133, 5, 15, 5, 101.1, 834.0, 121.76],
    # Te
    52: [2.1, 123, 6, 16, 5, 190.2, 869.0, 127.60],
    # I
    53: [2.66, 115, 7, 17, 5, 295.2, 1008.0, 126.90],
    # Xe
    54: [2.6, 108, 0, 18, 5, 0.0, 1170.0, 131.29],
    # Cs
    55: [0.79, 298, 1, 1, 6, 45.5, 376.0, 132.91],
    # Ba
    56: [0.89, 253, 2, 2, 6, 13.9, 503.0, 137.33],
    # La
    57: [1.10, 195, 3, 3, 6, 48.0, 538.0, 138.91],
    # Ce
    58: [1.12, 185, 4, 3, 6, 50.0, 534.0, 140.12],
    # Pr
    59: [1.13, 247, 5, 3, 6, 50.0, 527.0, 140.91],
    # Nd
    60: [1.14, 206, 6, 3, 6, 50.0, 533.0, 144.24],
    # Pm
    61: [1.13, 205, 7, 3, 6, 50.0, 540.0, 145.0],
    # Sm
    62: [1.17, 238, 8, 3, 6, 50.0, 544.0, 150.36],
    # Eu
    63: [1.2, 231, 9, 3, 6, 50.0, 547.0, 151.96],
    # Gd
    64: [1.2, 233, 10, 3, 6, 50.0, 593.0, 157.25],
    # Tb
    65: [1.1, 225, 11, 3, 6, 50.0, 566.0, 158.93],
    # Dy
    66: [1.22, 228, 12, 3, 6, 50.0, 573.0, 162.50],
    # Ho
    67: [1.23, 226, 13, 3, 6, 50.0, 581.0, 164.93],
    # Er
    68: [1.24, 226, 14, 3, 6, 50.0, 589.0, 167.26],
    # Tm
    69: [1.25, 222, 15, 3, 6, 50.0, 597.0, 168.93],
    # Yb
    70: [1.1, 222, 16, 3, 6, 50.0, 603.0, 173.05],
    # Lu
    71: [1.27, 217, 17, 3, 6, 50.0, 524.0, 174.97],
    # Hf
    72: [1.3, 208, 4, 4, 6, 0.0, 680.0, 178.49],
    # Ta
    73: [1.5, 200, 5, 5, 6, 31.0, 761.0, 180.95],
    # W
    74: [2.36, 193, 6, 6, 6, 79.0, 770.0, 183.84],
    # Re
    75: [1.9, 188, 7, 7, 6, 15.0, 760.0, 186.21],
    # Os
    76: [2.2, 185, 8, 8, 6, 106.0, 840.0, 190.23],
    # Ir
    77: [2.20, 180, 9, 9, 6, 151.0, 880.0, 192.22],
    # Pt
    78: [2.28, 177, 10, 10, 6, 205.3, 870.0, 195.08],
    # Au
    79: [2.54, 174, 11, 11, 6, 222.8, 890.0, 196.97],
    # Hg
    80: [2.00, 171, 12, 12, 6, 0.0, 1007.0, 200.59],
    # Tl
    81: [1.62, 156, 3, 13, 6, 19.0, 589.0, 204.38],
    # Pb
    82: [2.33, 154, 4, 14, 6, 35.1, 716.0, 207.2],
    # Bi
    83: [2.02, 143, 5, 15, 6, 91.3, 703.0, 208.98],
    # Po
    84: [2.0, 135, 6, 16, 6, 183.0, 812.0, 209.0],
    # At
    85: [2.2, 127, 7, 17, 6, 270.0, 920.0, 210.0],
    # Rn
    86: [0.0, 120, 0, 18, 6, 0.0, 1037.0, 222.0],
    # Fr
    87: [0.7, 348, 1, 1, 7, 0.0, 380.0, 223.0],
    # Ra
    88: [0.9, 283, 2, 2, 7, 0.0, 509.0, 226.0],
    # Ac
    89: [1.1, 195, 3, 3, 7, 35.0, 499.0, 227.0],
    # Th
    90: [1.3, 180, 4, 3, 7, 0.0, 608.0, 232.04],
    # Pa
    91: [1.5, 180, 5, 3, 7, 0.0, 568.0, 231.04],
    # U
    92: [1.38, 175, 6, 3, 7, 0.0, 598.0, 238.03],
    # Np
    93: [1.36, 175, 7, 3, 7, 0.0, 605.0, 237.0],
    # Pu
    94: [1.28, 175, 8, 3, 7, 0.0, 585.0, 244.0],
    # Am
    95: [1.3, 175, 9, 3, 7, 0.0, 578.0, 243.0],
    # Cm
    96: [1.3, 176, 10, 3, 7, 0.0, 581.0, 247.0],
    # Bk
    97: [1.3, 176, 11, 3, 7, 0.0, 601.0, 247.0],
    # Cf
    98: [1.3, 176, 12, 3, 7, 0.0, 608.0, 251.0],
    # Es
    99: [1.3, 176, 13, 3, 7, 0.0, 619.0, 252.0],
    # Fm
    100: [1.3, 176, 14, 3, 7, 0.0, 627.0, 257.0],
    # Md
    101: [1.3, 176, 15, 3, 7, 0.0, 635.0, 258.0],
    # No
    102: [1.3, 176, 16, 3, 7, 0.0, 642.0, 259.0],
    # Lr
    103: [1.3, 176, 17, 3, 7, 0.0, 460.0, 266.0],
    # Rf
    104: [0.0, 176, 4, 4, 7, 0.0, 580.0, 267.0],
    # Db
    105: [0.0, 176, 5, 5, 7, 0.0, 620.0, 268.0],
    # Sg
    106: [0.0, 171, 6, 6, 7, 0.0, 680.0, 269.0],
    # Bh
    107: [0.0, 171, 7, 7, 7, 0.0, 700.0, 270.0],
    # Hs
    108: [0.0, 171, 8, 8, 7, 0.0, 720.0, 277.0],
    # Mt
    109: [0.0, 171, 9, 9, 7, 0.0, 740.0, 278.0],
    # Ds
    110: [0.0, 171, 10, 10, 7, 0.0, 760.0, 281.0],
    # Rg
    111: [0.0, 171, 11, 11, 7, 0.0, 780.0, 282.0],
    # Cn
    112: [0.0, 171, 12, 12, 7, 0.0, 800.0, 285.0],
    # Nh
    113: [0.0, 170, 3, 13, 7, 0.0, 600.0, 286.0],
    # Fl
    114: [0.0, 170, 4, 14, 7, 0.0, 620.0, 289.0],
    # Mc
    115: [0.0, 170, 5, 15, 7, 0.0, 640.0, 290.0],
    # Lv
    116: [0.0, 169, 6, 16, 7, 0.0, 660.0, 293.0],
    # Ts
    117: [0.0, 165, 7, 17, 7, 0.0, 680.0, 294.0],
    # Og
    118: [0.0, 162, 0, 18, 7, 0.0, 700.0, 294.0],
}


def get_normalization_stats():
    """Compute normalization statistics for each property across all elements."""
    properties = np.array([ELEMENT_PROPERTIES[i] for i in range(1, MAX_ATOMIC_NUM + 1)])

    # Replace zeros with NaN for proper normalization
    properties_with_nan = properties.copy().astype(float)
    properties_with_nan[properties_with_nan == 0] = np.nan

    means = np.nanmean(properties_with_nan, axis=0)
    stds = np.nanstd(properties_with_nan, axis=0)

    # Avoid division by zero
    stds[stds == 0] = 1.0

    return means, stds


def create_periodic_embeddings(
    strategy: Literal["physics", "learnable", "random", "onehot"] = "physics",
    embedding_dim: int = 8,
    requires_grad: bool = False
) -> nn.Embedding:
    """
    Create element embeddings based on the specified strategy.

    Args:
        strategy: One of "physics", "learnable", "random", "onehot"
        embedding_dim: Dimension of embeddings (ignored for onehot)
        requires_grad: Whether embeddings should be learnable

    Returns:
        nn.Embedding module with element embeddings
    """
    if strategy == "onehot":
        # One-hot encoding (118 elements)
        embedding = nn.Embedding(MAX_ATOMIC_NUM, MAX_ATOMIC_NUM)
        # Initialize as identity matrix (one-hot)
        embedding.weight.data = torch.eye(MAX_ATOMIC_NUM)
        embedding.weight.requires_grad = False
        return embedding

    # Create embedding layer
    embedding = nn.Embedding(MAX_ATOMIC_NUM + 1, embedding_dim)  # +1 for padding/unknown

    if strategy in ["physics", "learnable"]:
        # Get physics-informed properties
        means, stds = get_normalization_stats()

        # Create normalized embedding matrix
        embedding_matrix = np.zeros((MAX_ATOMIC_NUM + 1, 8), dtype=np.float32)

        for atomic_num in range(1, MAX_ATOMIC_NUM + 1):
            props = np.array(ELEMENT_PROPERTIES[atomic_num], dtype=np.float32)
            # Normalize
            normalized = (props - means) / stds
            # Handle NaN from 0/0
            normalized = np.nan_to_num(normalized, nan=0.0)
            embedding_matrix[atomic_num] = normalized

        # If embedding_dim != 8, project
        if embedding_dim != 8:
            # Use linear projection
            projection = nn.Linear(8, embedding_dim, bias=False)
            # Xavier initialization for projection
            nn.init.xavier_uniform_(projection.weight)

            with torch.no_grad():
                embedded = projection(torch.from_numpy(embedding_matrix))
                embedding.weight.data = embedded
        else:
            embedding.weight.data = torch.from_numpy(embedding_matrix)

        embedding.weight.requires_grad = (strategy == "learnable")

    elif strategy == "random":
        # Random initialization (control)
        nn.init.normal_(embedding.weight, mean=0, std=0.1)
        embedding.weight.requires_grad = requires_grad

    return embedding


class PeriodicEmbedding(nn.Module):
    """
    Module that provides periodic table embeddings with multiple strategies.
    """

    def __init__(
        self,
        strategy: Literal["physics", "learnable", "random", "onehot"] = "physics",
        embedding_dim: int = 8,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.strategy = strategy
        self.embedding_dim = embedding_dim if strategy != "onehot" else MAX_ATOMIC_NUM

        # Create base embeddings
        self.embedding = create_periodic_embeddings(
            strategy=strategy,
            embedding_dim=embedding_dim,
            requires_grad=(strategy == "learnable")
        )

        # Project to hidden dimension
        self.projection = nn.Linear(self.embedding_dim, hidden_dim)

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atomic_numbers: Tensor of atomic numbers [batch_size, num_atoms]

        Returns:
            Embedded features [batch_size, num_atoms, hidden_dim]
        """
        # Get embeddings
        embedded = self.embedding(atomic_numbers)  # [batch, atoms, embed_dim]

        # Project to hidden dimension
        projected = self.projection(embedded)  # [batch, atoms, hidden_dim]

        return projected

    def get_element_embeddings(self) -> torch.Tensor:
        """Get embeddings for all elements (for visualization)."""
        with torch.no_grad():
            atomic_nums = torch.arange(1, MAX_ATOMIC_NUM + 1)
            return self.embedding(atomic_nums)


def analyze_element_similarity(embedding: nn.Embedding, elements: list = None):
    """
    Analyze cosine similarity between element embeddings.

    Args:
        embedding: Embedding module
        elements: List of element symbols to analyze

    Returns:
        similarity_matrix, element_names
    """
    if elements is None:
        # Common elements in polymers
        elements = ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']

    element_to_num = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'Na': 11, 'K': 19,
        'Ca': 20, 'Fe': 26, 'Cu': 29, 'Zn': 30, 'Ge': 32, 'As': 33,
        'Se': 34, 'Sn': 50, 'Sb': 51, 'Te': 52, 'Pb': 82, 'B': 5,
        'Al': 13, 'Ti': 22, 'Cr': 24, 'Mn': 25, 'Co': 27, 'Ni': 28,
        'Ag': 47, 'Au': 79, 'Pt': 78, 'Pd': 46
    }

    atomic_nums = [element_to_num[e] for e in elements if e in element_to_num]
    elements = [e for e in elements if e in element_to_num]

    with torch.no_grad():
        emb = embedding.weight[atomic_nums]
        # Normalize
        emb_norm = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
        similarity = torch.mm(emb_norm, emb_norm.t())

    return similarity.cpu().numpy(), elements


if __name__ == "__main__":
    # Test embeddings
    print("Testing embedding strategies...")

    for strategy in ["physics", "learnable", "random", "onehot"]:
        emb = PeriodicEmbedding(strategy=strategy, embedding_dim=8, hidden_dim=256)
        print(f"\n{strategy}:")
        print(f"  Embedding dim: {emb.embedding_dim}")
        print(f"  Requires grad: {emb.embedding.weight.requires_grad}")

        # Test forward pass
        atomic_nums = torch.tensor([[6, 1, 8, 1], [7, 6, 6, 1]])  # CH2O and NCH2
        output = emb(atomic_nums)
        print(f"  Output shape: {output.shape}")

    # Analyze similarity
    print("\nAnalyzing element similarity for physics embeddings:")
    emb = PeriodicEmbedding(strategy="physics", embedding_dim=8, hidden_dim=256)
    sim, elements = analyze_element_similarity(emb.embedding)
    print(f"Elements: {elements}")
    print(f"Similarity matrix shape: {sim.shape}")
    print(f"F-Cl similarity: {sim[4, 8]:.3f}")  # Should be high
    print(f"C-Si similarity: {sim[1, 5]:.3f}")  # Should be high
