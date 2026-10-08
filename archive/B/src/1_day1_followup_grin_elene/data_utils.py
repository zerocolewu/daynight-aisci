"""
Data loading, p-SMILES processing, multimer generation, graph construction,
and 3-level feature extraction for polymer Tg prediction.
Added: ELENE edge-level degree triplet encodings (k=1).
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

_CHIRALITY_MAP = {
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 1,
    Chem.rdchem.ChiralType.CHI_OTHER: 2,
}
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


# --- Basic features ---

def atom_features_basic(atom: Chem.Atom) -> list:
    an = atom.GetAtomicNum()
    an = max(1, min(an, 100))
    return _onehot(an - 1, 100)


def edge_features_none(bond: Chem.Bond) -> list:
    return []


# --- Standard features ---

def atom_features_standard(atom: Chem.Atom) -> list:
    features = []
    an = atom.GetAtomicNum()
    if an in COMMON_ATOMS:
        features += _onehot(COMMON_ATOMS.index(an), len(COMMON_ATOMS) + 1)
    else:
        features += _onehot(len(COMMON_ATOMS), len(COMMON_ATOMS) + 1)

    chiral = _CHIRALITY_MAP.get(atom.GetChiralTag(), _CHIRALITY_OTHER)
    features += _onehot(chiral, 4)

    degree = min(atom.GetDegree(), 6)
    features += _onehot(degree, 7)

    fc = atom.GetFormalCharge()
    fc_idx = max(0, min(fc + 2, 4))
    features += _onehot(fc_idx, 5)

    features.append(1.0 if atom.GetIsAromatic() else 0.0)

    hybrid = _HYBRID_MAP.get(atom.GetHybridization(), _HYBRID_OTHER)
    features += _onehot(hybrid, 6)

    val = atom.GetTotalValence()
    val = max(1, min(val, 7))
    features += _onehot(val - 1, 7)

    return features


def edge_features_standard(bond: Chem.Bond) -> list:
    return []


# --- Rich features ---

def atom_features_rich(atom: Chem.Atom) -> list:
    features = atom_features_standard(atom)
    ring_count = min(atom.GetOwningMol().GetRingInfo().NumAtomRings(atom.GetIdx()), 3)
    features += _onehot(ring_count, 4)
    return features


def edge_features_rich(bond: Chem.Bond) -> list:
    features = []
    bt_idx = _BOND_TYPE_MAP.get(bond.GetBondType(), _BOND_OTHER)
    features += _onehot(bt_idx, 4)
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    features.append(1.0 if bond.IsInRing() else 0.0)
    stereo_idx = _BOND_STEREO_MAP.get(bond.GetStereo(), _BOND_STEREO_OTHER)
    features += _onehot(stereo_idx, 4)
    return features


# ---------------------------------------------------------------------------
# ELENE Edge Encodings (k=1 ego-network degree triplets)
# ---------------------------------------------------------------------------

def compute_elene_edge_features(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute ELENE edge-level degree triplets at ego-network depth k=1.

    For each edge (u,v), treat v as the root and compute:
      d^(-1)(u|v): edges from u to nodes closer to v (at distance l(u,v)-1)
      d^(0)(u|v):  edges from u to nodes at same distance as v's distance from u
      d^(+1)(u|v): edges from u to nodes farther from v (at distance l(u,v)+1)

    Each triplet is normalized by degree of u.

    Returns:
        Tensor of shape (num_edges, 3) with the ELENE edge features.
    """
    ei = edge_index.numpy()  # [2, num_edges]
    num_edges = ei.shape[1]

    # Build adjacency lists (undirected)
    adj = [[] for _ in range(num_nodes)]
    for e_idx in range(num_edges):
        src, dst = int(ei[0, e_idx]), int(ei[1, e_idx])
        adj[src].append(dst)

    # Compute degree for each node
    degrees = np.array([len(adj[i]) for i in range(num_nodes)], dtype=np.float32)

    elene_features = np.zeros((num_edges, 3), dtype=np.float32)

    for e_idx in range(num_edges):
        u, v = int(ei[0, e_idx]), int(ei[1, e_idx])

        # BFS from root v up to depth 1
        dist_to_root = np.full(num_nodes, -1, dtype=np.int32)
        dist_to_root[v] = 0
        queue = [v]
        for node in queue:
            if dist_to_root[node] >= 1:
                continue
            for neighbor in adj[node]:
                if dist_to_root[neighbor] == -1:
                    dist_to_root[neighbor] = dist_to_root[node] + 1
                    if dist_to_root[neighbor] < 1:
                        queue.append(neighbor)

        # u is directly connected to v, so distance d(u,v) = 1
        du = dist_to_root[u]
        if du < 0:
            # should not happen since u and v are connected
            continue

        d_minus, d_zero, d_plus = 0.0, 0.0, 0.0
        for neighbor in adj[u]:
            dn = dist_to_root[neighbor]
            if dn == du - 1:
                d_minus += 1.0
            elif dn == du:
                d_zero += 1.0
            elif dn == du + 1:
                d_plus += 1.0
            # neighbors beyond distance du+1 are not counted in k=1

        deg_u = degrees[u]
        if deg_u > 0:
            d_minus /= deg_u
            d_zero /= deg_u
            d_plus /= deg_u

        elene_features[e_idx] = [d_minus, d_zero, d_plus]

    return torch.tensor(elene_features, dtype=torch.float)


# ---------------------------------------------------------------------------
# SMILES → PyG Data
# ---------------------------------------------------------------------------

_FEATURE_DIMS = {
    'basic':    {'node_dim': 100, 'edge_dim': 0},
    'standard': {'node_dim': 43,  'edge_dim': 0},
    'rich':     {'node_dim': 47,  'edge_dim': 10},
}


def smiles_to_graph(smiles: str, feature_level: str,
                    add_elene: bool = False) -> Optional[Data]:
    """Convert SMILES to PyG Data object with specified feature level.

    If add_elene=True, computes ELENE edge-level degree triplets (3-dim)
    and concatenates them to the existing edge features.
    """
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

    # Optionally add ELENE edge features
    if add_elene:
        elene_feat = compute_elene_edge_features(edge_index, x.shape[0])
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, elene_feat], dim=-1)
        else:
            edge_attr = elene_feat

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def get_feature_dims(feature_level: str, elene: bool = False) -> Tuple[int, int]:
    """Return (node_dim, edge_dim) for the given feature level.

    If elene=True, adds 3 to the edge dimension (for degree triplets).
    """
    d = _FEATURE_DIMS[feature_level]
    node_dim = d['node_dim']
    edge_dim = d['edge_dim']
    if elene:
        edge_dim += 3  # ELENE adds 3-dimensional degree triplets
    return node_dim, edge_dim


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_datasets(
    csv_path: str,
    multimer: str,
    feature_level: str,
    seed: int = 71,
    elene: bool = False,
) -> Tuple[list, list, list, StandardScaler]:
    """Build train/val/test graph lists and return target scaler.

    multimer: '1-RU', '2-RU', '3-RU', or '{1,3}-RU mixed'
    feature_level: 'basic', 'standard', 'rich'
    elene: if True, add ELENE edge encodings to graphs
    """
    df = pd.read_csv(csv_path)
    psmiles_list = df['Smiles'].values
    tg_values = df['Tg'].values.astype(np.float32)

    # Split indices: 72/8/20
    indices = np.arange(len(psmiles_list))
    train_idx, temp_idx = train_test_split(indices, test_size=0.28, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.20 / 0.28, random_state=seed)

    def _build(indices_list):
        graphs = []
        mixed_rng = random.Random(seed + 1)
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
            data = smiles_to_graph(smi, feature_level, add_elene=elene)
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


def create_dataloaders(
    train_graphs, val_graphs, test_graphs, scaler,
    batch_size: int = 32,
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
