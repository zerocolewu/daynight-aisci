"""
Data loading, p-SMILES processing, multimer generation, graph construction,
and 3-level feature extraction for polymer Tg prediction.
"""

from __future__ import annotations

import random
import warnings
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Multimer generation (adapted from existing multimer.py)
# ---------------------------------------------------------------------------

def generate_multimer(psmiles: str, n_mer: int, cap_with_h: bool = True) -> str:
    """Generate N-mer oligomer SMILES from p-SMILES.

    For n_mer=1: cap * with H, return standard SMILES.
    For n_mer>1: chain n repeat units, remove internal *, cap ends with H.
    """
    if n_mer == 1:
        smiles = psmiles.replace('*', '[H]')
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.RemoveHs(mol)
        return Chem.MolToSmiles(mol)

    monomer = Chem.MolFromSmiles(psmiles)
    if monomer is None:
        return None

    star_info = []
    for atom in monomer.GetAtoms():
        if atom.GetSymbol() == '*':
            neis = atom.GetNeighbors()
            if len(neis) != 1:
                return None
            bond = monomer.GetBondBetweenAtoms(atom.GetIdx(), neis[0].GetIdx())
            star_info.append({
                'star_idx': atom.GetIdx(),
                'neighbor_idx': neis[0].GetIdx(),
                'bond_type': bond.GetBondType()
            })

    if len(star_info) != 2:
        return None
    if star_info[0]['bond_type'] != star_info[1]['bond_type']:
        return None

    num_atoms = monomer.GetNumAtoms()
    oligomer = deepcopy(monomer)
    for _ in range(n_mer - 1):
        oligomer = Chem.CombineMols(oligomer, monomer)

    star_indices = np.zeros((n_mer, 2), dtype=int)
    neighbor_indices = np.zeros((n_mer, 2), dtype=int)
    for i in range(n_mer):
        offset = i * num_atoms
        star_indices[i] = [star_info[0]['star_idx'] + offset,
                           star_info[1]['star_idx'] + offset]
        neighbor_indices[i] = [star_info[0]['neighbor_idx'] + offset,
                               star_info[1]['neighbor_idx'] + offset]

    ed_mol = Chem.EditableMol(oligomer)
    removed_atoms = []

    for i in range(n_mer - 1):
        ed_mol.AddBond(
            int(neighbor_indices[i, 1]),
            int(neighbor_indices[i + 1, 0]),
            order=star_info[0]['bond_type']
        )
        removed_atoms.extend([
            int(star_indices[i, 1]),
            int(star_indices[i + 1, 0])
        ])

    if cap_with_h:
        ed_mol.ReplaceAtom(int(star_indices[0, 0]), Chem.Atom(1))
        ed_mol.ReplaceAtom(int(star_indices[n_mer - 1, 1]), Chem.Atom(1))

    for idx in sorted(removed_atoms, reverse=True):
        ed_mol.RemoveAtom(idx)

    final_mol = ed_mol.GetMol()
    final_mol = Chem.RemoveHs(final_mol)
    return Chem.MolToSmiles(final_mol)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

COMMON_ATOMS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H,B,C,N,O,F,Si,P,S,Cl,Br,I

# Chirality types for one-hot
_CHIRALITY_MAP = {
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 1,
    Chem.rdchem.ChiralType.CHI_OTHER: 2,
}
# Map unspecified to index 3
_CHIRALITY_OTHER = 3

_HYBRID_MAP = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
}
_HYBRID_OTHER = 5

_BOND_TYPE_MAP = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}
_BOND_OTHER = 0

_BOND_STEREO_MAP = {
    Chem.BondStereo.STEREOE: 0,
    Chem.BondStereo.STEREOZ: 1,
    Chem.BondStereo.STEREONONE: 2,
    Chem.BondStereo.STEREOANY: 2,
}
_BOND_STEREO_OTHER = 2


def _onehot(idx: int, n_classes: int) -> list:
    enc = [0.0] * n_classes
    if 0 <= idx < n_classes:
        enc[idx] = 1.0
    return enc


# --- Basic features: atomic number only ---

def atom_features_basic(atom: Chem.Atom) -> list:
    """Basic: atomic number one-hot (1-100, 100 dims)"""
    an = atom.GetAtomicNum()
    an = max(1, min(an, 100))
    return _onehot(an - 1, 100)


def edge_features_none(bond: Chem.Bond) -> list:
    return []


# --- Standard features ---

def atom_features_standard(atom: Chem.Atom) -> list:
    """Standard: atom type, chirality, degree, formal charge, aromaticity,
       hybridization, total valence — ~40 dims"""
    features = []

    # 1. Atom type (element) — 12 common + 1 other = 13 dims
    an = atom.GetAtomicNum()
    if an in COMMON_ATOMS:
        features += _onehot(COMMON_ATOMS.index(an), len(COMMON_ATOMS) + 1)
    else:
        features += _onehot(len(COMMON_ATOMS), len(COMMON_ATOMS) + 1)

    # 2. Chirality — 4 dims (CW, CCW, OTHER, unspecified)
    chiral = _CHIRALITY_MAP.get(atom.GetChiralTag(), _CHIRALITY_OTHER)
    features += _onehot(chiral, 4)

    # 3. Degree — 7 dims (0-6)
    degree = min(atom.GetDegree(), 6)
    features += _onehot(degree, 7)

    # 4. Formal charge — 5 dims (-2 to +2)
    fc = atom.GetFormalCharge()
    fc_idx = max(0, min(fc + 2, 4))
    features += _onehot(fc_idx, 5)

    # 5. Aromaticity — 2 dims
    features.append(1.0 if atom.GetIsAromatic() else 0.0)

    # 6. Hybridization — 6 dims (SP, SP2, SP3, SP3D, SP3D2, other)
    hybrid = _HYBRID_MAP.get(atom.GetHybridization(), _HYBRID_OTHER)
    features += _onehot(hybrid, 6)

    # 7. Total valence — 7 dims (1-7)
    val = atom.GetTotalValence()
    val = max(1, min(val, 7))
    features += _onehot(val - 1, 7)

    return features


def edge_features_standard(bond: Chem.Bond) -> list:
    return []


# --- Rich features ---

def atom_features_rich(atom: Chem.Atom) -> list:
    """Rich: standard + ring membership (0-3)"""
    features = atom_features_standard(atom)

    # Ring membership count (0-3), 4 dims
    ring_count = min(atom.GetOwningMol().GetRingInfo().NumAtomRings(atom.GetIdx()), 3)
    features += _onehot(ring_count, 4)

    return features


def edge_features_rich(bond: Chem.Bond) -> list:
    """Rich edge features: bond type (4), conjugated (1), ring bond (1), stereo (4) = 10 dims"""
    features = []

    # Bond type — 4 dims
    bt_idx = _BOND_TYPE_MAP.get(bond.GetBondType(), _BOND_OTHER)
    features += _onehot(bt_idx, 4)

    # Conjugated — 1 dim
    features.append(1.0 if bond.GetIsConjugated() else 0.0)

    # Ring bond — 1 dim
    features.append(1.0 if bond.IsInRing() else 0.0)

    # Stereochemistry — 4 dims
    stereo_idx = _BOND_STEREO_MAP.get(bond.GetStereo(), _BOND_STEREO_OTHER)
    features += _onehot(stereo_idx, 4)

    return features


# ---------------------------------------------------------------------------
# SMILES → PyG Data
# ---------------------------------------------------------------------------

_FEATURE_DIMS = {
    'basic':    {'node_dim': 100, 'edge_dim': 0},
    'standard': {'node_dim': 43,  'edge_dim': 0},
    'rich':     {'node_dim': 47,  'edge_dim': 10},
}


def smiles_to_graph(smiles: str, feature_level: str) -> Optional[Data]:
    """Convert SMILES to PyG Data object with specified feature level."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    if feature_level == 'basic':
        node_func = atom_features_basic
        edge_func = edge_features_none
    elif feature_level == 'standard':
        node_func = atom_features_standard
        edge_func = edge_features_standard
    elif feature_level == 'rich':
        node_func = atom_features_rich
        edge_func = edge_features_rich
    else:
        raise ValueError(f"Unknown feature_level: {feature_level}")

    x = torch.tensor([node_func(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = edge_func(bond)
        edge_indices += [[i, j], [j, i]]
        if feat:
            edge_attrs += [feat, feat]

    if len(edge_indices) == 0:
        return None

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    if edge_attrs:
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_attr = None

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_datasets(
    csv_path: str,
    multimer: str,
    feature_level: str,
    seed: int = 71,
) -> Tuple[list, list, list, StandardScaler]:
    """Build train/val/test graph lists and return target scaler.

    multimer: '1-RU', '2-RU', '3-RU', or '{1,3}-RU mixed'
    feature_level: 'basic', 'standard', 'rich'
    """
    df = pd.read_csv(csv_path)
    psmiles_list = df['Smiles'].values
    tg_values = df['Tg'].values.astype(np.float32)

    # Split indices: 72/8/20
    indices = np.arange(len(psmiles_list))
    train_idx, temp_idx = train_test_split(indices, test_size=0.28, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.20 / 0.28, random_state=seed)

    # Build graphs for each split
    def _build(indices_list):
        graphs = []
        mixed_rng = random.Random(seed + 1)  # fixed seed for reproducibility
        for idx in indices_list:
            psmiles = psmiles_list[idx]
            tg = tg_values[idx]

            if multimer == '{1,3}-RU mixed':
                n_mer = 3 if mixed_rng.random() < 0.5 else 1
            elif multimer == '1-RU':
                n_mer = 1
            elif multimer == '2-RU':
                n_mer = 2
            elif multimer == '3-RU':
                n_mer = 3
            else:
                raise ValueError(f"Unknown multimer: {multimer}")

            smi = generate_multimer(psmiles, n_mer)
            if smi is None:
                continue
            data = smiles_to_graph(smi, feature_level)
            if data is None or data.x.shape[0] == 0:
                continue
            data.y = torch.tensor([tg], dtype=torch.float)
            graphs.append(data)
        return graphs

    train_graphs = _build(train_idx)
    val_graphs = _build(val_idx)
    test_graphs = _build(test_idx)

    # Fit scaler on training targets only
    scaler = StandardScaler()
    y_train = np.array([g.y.item() for g in train_graphs]).reshape(-1, 1)
    scaler.fit(y_train)

    return train_graphs, val_graphs, test_graphs, scaler


def get_feature_dims(feature_level: str) -> Tuple[int, int]:
    """Return (node_dim, edge_dim) for the given feature level."""
    d = _FEATURE_DIMS[feature_level]
    return d['node_dim'], d['edge_dim']


def create_dataloaders(
    train_graphs, val_graphs, test_graphs, scaler,
    batch_size: int = 64,
):
    """Create DataLoaders and normalize targets."""
    for g in train_graphs:
        g.y = torch.tensor(scaler.transform(g.y.numpy().reshape(-1, 1))[0], dtype=torch.float)
    for g in val_graphs:
        g.y = torch.tensor(scaler.transform(g.y.numpy().reshape(-1, 1))[0], dtype=torch.float)
    for g in test_graphs:
        g.y = torch.tensor(scaler.transform(g.y.numpy().reshape(-1, 1))[0], dtype=torch.float)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=0)
    return train_loader, val_loader, test_loader
