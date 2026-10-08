"""
Data preparation for GPS-Lite experiments.
- Builds 3-RU multimers with rich node features (47 dims)
- Computes PSPE features (distance_to_chain_end + monomer_index)
- Loads precomputed RWPE k=32 from Day 2
- Computes SPD (Shortest Path Distance) matrix per graph
"""

from __future__ import annotations

import pickle
import random
import warnings
from collections import deque
from copy import deepcopy
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Multimer generation (from Day 1 data_utils.py)
# ---------------------------------------------------------------------------

def generate_multimer(psmiles: str, n_mer: int, cap_with_h: bool = True) -> Optional[str]:
    """Generate N-mer oligomer SMILES from p-SMILES."""
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
# Feature extraction (from Day 1/3 data_utils.py)
# ---------------------------------------------------------------------------

COMMON_ATOMS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]

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


NODE_DIM_RICH = 47
EDGE_DIM_RICH = 10
RWPE_K = 32


# ---------------------------------------------------------------------------
# PSPE Computation (from Day 3 pspe_data.py)
# ---------------------------------------------------------------------------

def find_backbone_path(edge_index, num_nodes):
    """Find the longest shortest-path backbone via double-BFS."""
    adj = {i: [] for i in range(num_nodes)}
    ei = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        adj[u].append(v)
        adj[v].append(u)

    def bfs_farthest(start):
        dist = {start: 0}
        q = deque([start])
        visited = {start}
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    dist[v] = dist[u] + 1
                    q.append(v)
        if not dist:
            return start, [start], 0
        farthest = max(dist, key=dist.get)
        return farthest, dist, len(dist)

    def bfs_path(start, target):
        if start == target:
            return [start]
        parent = {start: None}
        q = deque([start])
        visited = {start}
        while q:
            u = q.popleft()
            if u == target:
                break
            for v in adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    q.append(v)
        path = []
        cur = target
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        return path

    node_a, _, _ = bfs_farthest(0)
    node_b, _, _ = bfs_farthest(node_a)
    backbone_path = bfs_path(node_a, node_b)
    backbone_nodes = set(backbone_path)
    backbone_length = len(backbone_path)

    return backbone_nodes, backbone_path, backbone_length


def compute_pspe(edge_index, num_nodes):
    """Compute PSPE-dce and PSPE-mi for a graph."""
    backbone_nodes, backbone_path, backbone_length = find_backbone_path(edge_index, num_nodes)

    dce = np.zeros(num_nodes, dtype=np.float32)
    mi = np.zeros(num_nodes, dtype=np.float32)

    if backbone_length < 3:
        info = {
            'backbone_length': backbone_length,
            'backbone_frac': len(backbone_nodes) / num_nodes if num_nodes > 0 else 0,
            'valid': False,
        }
        return dce, mi, info

    pos_map = {}
    for p, node_idx in enumerate(backbone_path):
        pos_map[node_idx] = p

    # PSPE-dce
    max_possible = (backbone_length - 1) / 2.0
    for node_idx in backbone_nodes:
        p = pos_map[node_idx]
        raw_dce = min(p, backbone_length - 1 - p)
        dce[node_idx] = raw_dce / max_possible if max_possible > 0 else 0.0

    # PSPE-mi
    segment_size = backbone_length / 3.0
    for node_idx in backbone_nodes:
        p = pos_map[node_idx]
        segment_idx = min(int(p / segment_size), 2)
        mi[node_idx] = segment_idx / 2.0

    info = {
        'backbone_length': backbone_length,
        'backbone_frac': len(backbone_nodes) / num_nodes if num_nodes > 0 else 0,
        'valid': True,
    }

    return dce, mi, info


# ---------------------------------------------------------------------------
# SPD (Shortest Path Distance) Matrix Computation
# ---------------------------------------------------------------------------

def compute_spd_matrix(edge_index, num_nodes):
    """Compute the all-pairs shortest-path-distance matrix using BFS from each node.

    Returns:
        spd: numpy array [num_nodes, num_nodes] of integers
    """
    adj = {i: [] for i in range(num_nodes)}
    ei = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        adj[u].append(v)
        adj[v].append(u)

    spd = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)

    for src in range(num_nodes):
        dist = {src: 0}
        q = deque([src])
        visited = {src}
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    dist[v] = dist[u] + 1
                    q.append(v)
        for tgt, d in dist.items():
            spd[src, tgt] = float(d)

    return spd


# ---------------------------------------------------------------------------
# SMILES -> PyG Data
# ---------------------------------------------------------------------------

def smiles_to_graph(smiles: str) -> Optional[Data]:
    """Convert SMILES to PyG Data with rich features, PSPE, and SPD."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features_rich(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = edge_features_rich(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [feat, feat]

    if len(edge_indices) == 0:
        return None

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # Compute PSPE features
    dce, mi, backbone_info = compute_pspe(edge_index, x.shape[0])

    # Compute SPD matrix
    spd = compute_spd_matrix(edge_index, x.shape[0])

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        dce=torch.tensor(dce, dtype=torch.float),
        mi=torch.tensor(mi, dtype=torch.float),
        spd=torch.tensor(spd, dtype=torch.long),
        backbone_info=backbone_info,
    )


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_datasets(
    csv_path: str,
    seed: int = 71,
) -> Tuple[list, list, list, StandardScaler, Dict]:
    """Build train/val/test graph lists with all features including SPD."""
    df = pd.read_csv(csv_path)
    psmiles_list = df['Smiles'].values
    tg_values = df['Tg'].values.astype(np.float32)

    # Split indices: 72/8/20
    indices = np.arange(len(psmiles_list))
    train_idx, temp_idx = train_test_split(indices, test_size=0.28, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.20 / 0.28, random_state=seed)

    def _build(indices_list):
        graphs = []
        for idx in indices_list:
            psmiles = psmiles_list[idx]
            tg = tg_values[idx]
            smi = generate_multimer(psmiles, 3)
            if smi is None:
                continue
            data = smiles_to_graph(smi)
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

    # Gather statistics
    backbone_lengths = []
    backbone_fracs = []
    spd_values = []

    for g in train_graphs + val_graphs + test_graphs:
        backbone_lengths.append(g.backbone_info['backbone_length'])
        backbone_fracs.append(g.backbone_info['backbone_frac'])
        # Collect SPD values (skip diagonal and infinity)
        spd_flat = g.spd.numpy().flatten()
        spd_values.extend(spd_flat[(spd_flat < np.inf) & (spd_flat > 0)].tolist())

    metadata = {
        'backbone_lengths': {
            'mean': float(np.mean(backbone_lengths)),
            'std': float(np.std(backbone_lengths)),
            'min': float(np.min(backbone_lengths)),
            'max': float(np.max(backbone_lengths)),
        },
        'backbone_fracs': {
            'mean': float(np.mean(backbone_fracs)),
            'std': float(np.std(backbone_fracs)),
        },
        'spd': {
            'mean': float(np.mean(spd_values)),
            'std': float(np.std(spd_values)),
            'min': float(np.min(spd_values)),
            'max': float(np.max(spd_values)),
        },
        'num_train': len(train_graphs),
        'num_val': len(val_graphs),
        'num_test': len(test_graphs),
    }

    return train_graphs, val_graphs, test_graphs, scaler, metadata


# ---------------------------------------------------------------------------
# RWPE Loading and Assignment
# ---------------------------------------------------------------------------

def load_rwpe(rwpe_path: str) -> Dict:
    """Load precomputed RWPE from pickle."""
    with open(rwpe_path, 'rb') as f:
        rwpe_data = pickle.load(f)
    print(f"Loaded RWPE data: {len(rwpe_data['rwpe'])} graphs, k={rwpe_data['k_steps']}")
    return rwpe_data


def assign_rwpe_to_graphs(graphs, rwpe_data, offset=0):
    """Assign precomputed RWPE to each graph by combined list position."""
    rwpe_dict = rwpe_data['rwpe']
    num_with_rwpe = 0
    num_fallback = 0
    for i, g in enumerate(graphs):
        rwpe_key = offset + i
        if rwpe_key in rwpe_dict:
            rwpe_tensor = rwpe_dict[rwpe_key]
            expected_nodes = g.x.shape[0]
            if rwpe_tensor.shape[0] == expected_nodes:
                g.rwpe = rwpe_tensor.clone().float()
                num_with_rwpe += 1
            else:
                g.rwpe = torch.zeros((expected_nodes, RWPE_K), dtype=torch.float)
                num_fallback += 1
        else:
            g.rwpe = torch.zeros((g.x.shape[0], RWPE_K), dtype=torch.float)
            num_fallback += 1
    return num_with_rwpe, num_fallback


# ---------------------------------------------------------------------------
# Feature Assembly
# ---------------------------------------------------------------------------

def assemble_features(graph):
    """Assemble node features: rich_features (47) + rwpe (32) + dce (1) + mi (1) = 81 dims."""
    features = [graph.x, graph.rwpe, graph.dce.unsqueeze(-1), graph.mi.unsqueeze(-1)]
    return torch.cat(features, dim=-1)


def get_input_dim():
    """Get total input feature dimension."""
    return NODE_DIM_RICH + RWPE_K + 2  # 47 + 32 + 2 = 81


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def create_dataloaders(train_graphs, val_graphs, test_graphs, scaler, batch_size=64):
    """Create DataLoaders with target normalization.

    Returns:
        train_loader, val_loader, test_loader: PyG DataLoaders
        spd_maps: dict mapping 'train', 'val', 'test' to dict of {graph_id: spd_tensor}
    """
    # Normalize targets
    for g in train_graphs:
        g.y = torch.tensor(scaler.transform(g.y.numpy().reshape(-1, 1))[0], dtype=torch.float)
    for g in val_graphs:
        g.y = torch.tensor(scaler.transform(g.y.numpy().reshape(-1, 1))[0], dtype=torch.float)
    for g in test_graphs:
        g.y = torch.tensor(scaler.transform(g.y.numpy().reshape(-1, 1))[0], dtype=torch.float)

    # Assign unique graph_ids for SPD lookup
    for idx, g in enumerate(train_graphs):
        g.graph_id = torch.tensor(idx, dtype=torch.long)
    for idx, g in enumerate(val_graphs):
        g.graph_id = torch.tensor(idx, dtype=torch.long)
    for idx, g in enumerate(test_graphs):
        g.graph_id = torch.tensor(idx, dtype=torch.long)

    # Build SPD lookup dictionaries
    spd_maps = {
        'train': {idx: g.spd for idx, g in enumerate(train_graphs)},
        'val': {idx: g.spd for idx, g in enumerate(val_graphs)},
        'test': {idx: g.spd for idx, g in enumerate(test_graphs)},
    }

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=0,
                              exclude_keys=['spd', 'backbone_info'])
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=0,
                            exclude_keys=['spd', 'backbone_info'])
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=0,
                             exclude_keys=['spd', 'backbone_info'])
    return train_loader, val_loader, test_loader, spd_maps
