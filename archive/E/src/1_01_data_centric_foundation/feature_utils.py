"""
Atom and bond feature extraction utilities for molecular graphs.
Standard RDKit features for GNN input.
"""

import torch
import numpy as np
from rdkit import Chem


# Atom feature dimensions
ATOM_TYPES = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'Si', 'B']  # 10
DEGREES = [0, 1, 2, 3, 4, 5]  # 6
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
]  # 4
H_COUNTS = [0, 1, 2, 3, 4, 5]  # 6
CHIRALITY = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
]  # 2

# Total atom features: 10 + 6 + 4 + 1 + 1 + 6 + 2 = 30
ATOM_FEATURE_DIM = len(ATOM_TYPES) + len(DEGREES) + len(HYBRIDIZATIONS) + 1 + 1 + len(H_COUNTS) + len(CHIRALITY)

# Bond features
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]  # 4
# + conjugated (1) + ring (1) = 6
BOND_FEATURE_DIM = len(BOND_TYPES) + 1 + 1  # 6


def one_hot(value, choices):
    """One-hot encode a value given a list of possible choices."""
    vec = [0] * len(choices)
    try:
        idx = choices.index(value)
        vec[idx] = 1
    except ValueError:
        pass
    return vec


def get_atom_features(atom):
    """
    Extract atom features as a list of floats.
    Returns list of length ATOM_FEATURE_DIM (30).
    """
    symbol = atom.GetSymbol()
    atom_type = one_hot(symbol, ATOM_TYPES)

    degree = one_hot(atom.GetTotalDegree(), DEGREES)

    hyb = atom.GetHybridization()
    hybridization = one_hot(hyb, HYBRIDIZATIONS)

    aromatic = [int(atom.GetIsAromatic())]
    ring = [int(atom.IsInRing())]

    h_count = one_hot(atom.GetTotalNumHs(), H_COUNTS)

    chirality = one_hot(atom.GetChiralTag(), CHIRALITY)

    features = atom_type + degree + hybridization + aromatic + ring + h_count + chirality
    return features


def get_bond_features(bond):
    """
    Extract bond features as a list of floats.
    Returns list of length BOND_FEATURE_DIM (6).
    """
    bond_type = one_hot(bond.GetBondType(), BOND_TYPES)
    conjugated = [int(bond.GetIsConjugated())]
    ring = [int(bond.IsInRing())]

    features = bond_type + conjugated + ring
    return features


def mol_to_graph_data(mol, target, repeat_unit_indices=None, periodic_bond_atoms=None):
    """
    Convert an RDKit mol to PyG-compatible tensors.

    Args:
        mol: RDKit Mol object
        target: float, Tg value
        repeat_unit_indices: optional list of unit index per atom (for feature augmentation)
        periodic_bond_atoms: optional tuple (atom_i, atom_j) marking the periodic bond

    Returns:
        dict with 'x', 'edge_index', 'edge_attr', 'y', 'num_nodes'
    """
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return None

    # Atom features
    atom_features = []
    for i, atom in enumerate(mol.GetAtoms()):
        feat = get_atom_features(atom)
        if repeat_unit_indices is not None:
            # One-hot encode repeat unit index (3 dims)
            ru_idx = repeat_unit_indices[i]
            ru_onehot = [0, 0, 0]
            if ru_idx < 3:
                ru_onehot[ru_idx] = 1
            feat = feat + ru_onehot
        atom_features.append(feat)

    x = torch.tensor(atom_features, dtype=torch.float)

    # Bond features and edge index
    edge_indices = []
    edge_attrs = []

    # Track periodic bond
    periodic_set = set()
    if periodic_bond_atoms is not None:
        a, b = periodic_bond_atoms
        periodic_set.add((min(a, b), max(a, b)))

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        feat = get_bond_features(bond)

        # Check if this is the periodic bond
        bond_key = (min(begin, end), max(begin, end))
        if bond_key in periodic_set:
            feat = feat + [1]  # periodic flag
        else:
            feat = feat + [0]  # not periodic

        # Add both directions for undirected graph
        edge_indices.append([begin, end])
        edge_indices.append([end, begin])
        edge_attrs.append(feat)
        edge_attrs.append(feat)

    if len(edge_indices) == 0:
        # Single atom molecule, add self-loop
        edge_indices = [[0, 0]]
        edge_attrs = [[0] * BOND_FEATURE_DIM + [0]]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'y': torch.tensor([target], dtype=torch.float),
        'num_nodes': n_atoms,
    }


# Compute actual feature dims including optional augmentations
BASE_ATOM_FEATURE_DIM = ATOM_FEATURE_DIM  # 30
BASE_BOND_FEATURE_DIM = BOND_FEATURE_DIM + 1  # 6 + 1 (periodic flag) = 7
AUGMENTED_ATOM_FEATURE_DIM = ATOM_FEATURE_DIM + 3  # 30 + 3 (repeat unit one-hot) = 33
