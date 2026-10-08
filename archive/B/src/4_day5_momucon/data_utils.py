#!/usr/bin/env python3
"""
Data utilities for MoMuCon-NT-Xent pretraining and Tg finetuning.

Features:
  - 9-dim numeric rich node features: atomic_number, hybridization, degree,
    formal_charge, aromatic, ring_membership, chirality, num_hydrogens, total_valence
  - 3-dim numeric edge features: bond_type, conjugated, in_ring
  - 32-dim RWPE (Random Walk Positional Encoding)

Generates:
  - PI1M (20K) (1-RU, 3-RU) pairs for pretraining
  - Tg finetuning data: 3-RU graphs, 72/8/20 split
"""

import os
import pickle
import random
import warnings
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from copy import deepcopy
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
WORK_DIR = "/home/home_public/aisci/AIscientist/src/4_day5_momucon"
DATA_DIR = "/home/home_public/aisci/AIscientist/src/alldataset"
CACHE_DIR = os.path.join(WORK_DIR, "cache")
PI1M_CSV = os.path.join(DATA_DIR, "PI1M.csv")
TG_CSV = os.path.join(DATA_DIR, "Tg.csv")

os.makedirs(CACHE_DIR, exist_ok=True)

# --------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------
RWPE_K = 32
PI1M_N_POLYMERS = 100000
PI1M_SEED = 42
TG_SPLIT_SEED = 71

RICH_DIM = 9
EDGE_DIM = 3
TOTAL_NODE_DIM = RICH_DIM + RWPE_K  # 41

# --------------------------------------------------------------------
# Feature maps
# --------------------------------------------------------------------
HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
}
HYBRIDIZATION_OTHER = 5

BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}


# --------------------------------------------------------------------
# Multimer generation
# --------------------------------------------------------------------
def generate_1ru_smiles(psmiles: str):
    """Caps p-SMILES with H to create 1-RU SMILES."""
    smiles = psmiles.replace('*', '[H]')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)


def generate_multimer(psmiles: str, n_mer: int):
    """Generate N-mer oligomer SMILES from p-SMILES."""
    if n_mer == 1:
        return generate_1ru_smiles(psmiles)

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

    ed_mol.ReplaceAtom(int(star_indices[0, 0]), Chem.Atom(1))
    ed_mol.ReplaceAtom(int(star_indices[n_mer - 1, 1]), Chem.Atom(1))

    for idx in sorted(removed_atoms, reverse=True):
        ed_mol.RemoveAtom(idx)

    final_mol = ed_mol.GetMol()
    final_mol = Chem.RemoveHs(final_mol)
    return Chem.MolToSmiles(final_mol)


# --------------------------------------------------------------------
# 9-dim numeric rich node features
# --------------------------------------------------------------------
def extract_rich_features_9dim(mol):
    """
    9-dim numeric rich features:
      [0] atomic_number / 100.0
      [1] hybridization / 5.0 (SP=0, SP2=1, SP3=2, SP3D=3, SP3D2=4, OTHER=5)
      [2] degree / 10.0 (clamped)
      [3] formal_charge / 5.0 (clamped [-1, 1])
      [4] aromatic (0/1)
      [5] ring_membership (IsInRing) (0/1)
      [6] chirality_tag / 10.0
      [7] num_hydrogens / 10.0 (clamped)
      [8] total_valence / 10.0 (clamped)
    """
    features = []
    for atom in mol.GetAtoms():
        feat = [
            # [0] atomic_number
            float(atom.GetAtomicNum()) / 100.0,
            # [1] hybridization
            float(HYBRIDIZATION_MAP.get(atom.GetHybridization(), HYBRIDIZATION_OTHER)) / 5.0,
            # [2] degree
            min(float(atom.GetDegree()) / 10.0, 1.0),
            # [3] formal_charge
            max(-1.0, min(float(atom.GetFormalCharge()) / 5.0, 1.0)),
            # [4] aromatic
            1.0 if atom.GetIsAromatic() else 0.0,
            # [5] ring_membership
            1.0 if atom.IsInRing() else 0.0,
            # [6] chirality
            float(int(atom.GetChiralTag())) / 10.0,
            # [7] num_hydrogens
            min(float(atom.GetTotalNumHs(includeNeighbors=False)) / 10.0, 1.0),
            # [8] total_valence
            min(float(atom.GetTotalValence()) / 10.0, 1.0),
        ]
        features.append(feat)
    return torch.tensor(features, dtype=torch.float32)


# --------------------------------------------------------------------
# 3-dim numeric edge features
# --------------------------------------------------------------------
def extract_edge_features(mol, edge_index):
    """
    3-dim numeric edge features:
      [0] bond_type / 3.0
      [1] conjugated (0/1)
      [2] in_ring (0/1)
    """
    num_edges = edge_index.size(1)
    edge_attr = torch.zeros(num_edges, EDGE_DIM, dtype=torch.float32)

    bond_lookup = {}
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = BOND_TYPE_MAP.get(bond.GetBondType(), 0)
        is_conj = bond.GetIsConjugated()
        is_ring = bond.IsInRing()
        bond_lookup[(i, j)] = (bt, is_conj, is_ring)
        bond_lookup[(j, i)] = (bt, is_conj, is_ring)

    for e in range(num_edges):
        u = int(edge_index[0, e])
        v = int(edge_index[1, e])
        bt, is_conj, is_ring = bond_lookup.get((u, v), (0, False, False))
        edge_attr[e, 0] = float(bt) / 3.0
        edge_attr[e, 1] = 1.0 if is_conj else 0.0
        edge_attr[e, 2] = 1.0 if is_ring else 0.0

    return edge_attr


# --------------------------------------------------------------------
# RWPE computation (diagonal-based)
# --------------------------------------------------------------------
def compute_rwpe(edge_index, num_nodes, k_steps=RWPE_K):
    """Compute k-step random walk return probabilities via diagonal of (AD^{-1})^k."""
    if num_nodes <= 1:
        return np.zeros((num_nodes, k_steps), dtype=np.float32)

    try:
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).numpy()
    except Exception:
        return np.zeros((num_nodes, k_steps), dtype=np.float32)

    deg = adj.sum(axis=1)
    deg[deg == 0] = 1

    P = adj / deg[:, np.newaxis]
    rwpe = np.zeros((num_nodes, k_steps), dtype=np.float32)
    P_pow = P.copy()
    for step in range(k_steps):
        diag_vals = np.diag(P_pow)
        diag_vals = np.nan_to_num(diag_vals, nan=0.0)
        rwpe[:, step] = diag_vals
        P_pow = P_pow @ P
        P_pow = np.nan_to_num(P_pow, nan=0.0)

    return rwpe


# --------------------------------------------------------------------
# SMILES -> PyG Data
# --------------------------------------------------------------------
def smiles_to_graph(smiles, add_rwpe=True):
    """Convert SMILES to PyG Data with 9-dim rich features + optional RWPE."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    n_nodes = mol.GetNumAtoms()

    # 9-dim rich features
    x = extract_rich_features_9dim(mol)

    # Edges
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_DIM), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = extract_edge_features(mol, edge_index)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_nodes)

    # RWPE
    if add_rwpe:
        rwpe = compute_rwpe(edge_index, n_nodes, RWPE_K)
        data.x = torch.cat([data.x, torch.tensor(rwpe, dtype=torch.float32)], dim=-1)

    return data


# --------------------------------------------------------------------
# PI1M Preparation
# --------------------------------------------------------------------
def prepare_pi1m_pretrain_data(force_rebuild=False):
    """Prepare (1-RU, 3-RU) graph pairs from 100K PI1M polymers."""
    cache_path = os.path.join(CACHE_DIR, f"pi1m_{PI1M_N_POLYMERS // 1000}k_pairs_9dim.pkl")

    if os.path.exists(cache_path) and not force_rebuild:
        print(f"Loading cached PI1M pairs from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"\n=== Preparing PI1M Pretraining Data ===")
    print(f"Target: {PI1M_N_POLYMERS} polymers with (1-RU, 3-RU) pairs")
    print(f"Features: {RICH_DIM}-dim rich + {RWPE_K}-dim RWPE = {TOTAL_NODE_DIM}-dim")

    print("\nLoading PI1M.csv...")
    df = pd.read_csv(PI1M_CSV)
    print(f"Total PI1M entries: {len(df)}")

    random.seed(PI1M_SEED)
    np.random.seed(PI1M_SEED)
    indices = list(range(len(df)))
    random.shuffle(indices)
    selected = indices[:PI1M_N_POLYMERS]
    print(f"Selected {len(selected)} indices (seed={PI1M_SEED})")

    pairs = []
    failed = 0
    for idx in tqdm(selected, desc="Building 1-RU + 3-RU pairs"):
        psmiles = df.iloc[idx]['SMILES']
        try:
            smiles_1ru = generate_1ru_smiles(psmiles)
            smiles_3ru = generate_multimer(psmiles, n_mer=3)

            if smiles_1ru is None or smiles_3ru is None:
                failed += 1
                continue

            graph_1ru = smiles_to_graph(smiles_1ru, add_rwpe=True)
            graph_3ru = smiles_to_graph(smiles_3ru, add_rwpe=True)

            if graph_1ru is None or graph_3ru is None:
                failed += 1
                continue

            pairs.append((graph_1ru, graph_3ru))

        except Exception:
            failed += 1
            continue

    print(f"\nSuccessfully built {len(pairs)} pairs ({failed} failed)")
    print(f"Feature dim: {pairs[0][0].x.size(1)} (expected {TOTAL_NODE_DIM})")

    nodes_1ru = [p[0].num_nodes for p in pairs]
    nodes_3ru = [p[1].num_nodes for p in pairs]
    print(f"  1-RU nodes: min={min(nodes_1ru)}, max={max(nodes_1ru)}, mean={np.mean(nodes_1ru):.1f}")
    print(f"  3-RU nodes: min={min(nodes_3ru)}, max={max(nodes_3ru)}, mean={np.mean(nodes_3ru):.1f}")

    with open(cache_path, 'wb') as f:
        pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"Saved to: {cache_path} ({size_mb:.1f} MB)")
    return pairs


# --------------------------------------------------------------------
# Tg Data Preparation
# --------------------------------------------------------------------
def prepare_tg_data(force_rebuild=False):
    """Prepare Tg finetuning data with 3-RU graphs, 9-dim rich + RWPE."""
    cache_path = os.path.join(CACHE_DIR, "tg_graphs_3ru_9dim.pkl")
    if os.path.exists(cache_path) and not force_rebuild:
        print(f"Loading cached Tg data from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"\n=== Preparing Tg Finetuning Data ===")
    print(f"Features: {RICH_DIM}-dim rich + {RWPE_K}-dim RWPE = {TOTAL_NODE_DIM}-dim")

    df = pd.read_csv(TG_CSV)
    psmiles_list = df['Smiles'].values
    tg_values = df['Tg'].values.astype(np.float32)
    print(f"Total Tg entries: {len(df)}")

    indices = np.arange(len(psmiles_list))
    train_idx, temp_idx = train_test_split(indices, test_size=0.28, random_state=TG_SPLIT_SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.20 / 0.28, random_state=TG_SPLIT_SEED)
    print(f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    def build_graphs(idx_list, desc):
        graphs = []
        failed = 0
        for i in tqdm(idx_list, desc=desc):
            psmiles = str(psmiles_list[i])
            tg = float(tg_values[i])
            try:
                smiles = generate_multimer(psmiles, n_mer=3)
                if smiles is None:
                    failed += 1
                    continue
                data = smiles_to_graph(smiles, add_rwpe=True)
                if data is None:
                    failed += 1
                    continue
                data.y = torch.tensor([tg], dtype=torch.float32)
                graphs.append(data)
            except Exception:
                failed += 1
                continue
        print(f"  Built {len(graphs)} graphs ({failed} failed)")
        return graphs

    train_graphs = build_graphs(train_idx, "Building Tg train")
    val_graphs = build_graphs(val_idx, "Building Tg val")
    test_graphs = build_graphs(test_idx, "Building Tg test")

    # Normalize targets
    scaler = StandardScaler()
    y_train = np.array([g.y.item() for g in train_graphs]).reshape(-1, 1)
    scaler.fit(y_train)
    for g in train_graphs + val_graphs + test_graphs:
        g.y = torch.tensor(scaler.transform(g.y.numpy().reshape(-1, 1))[0], dtype=torch.float32)

    result = {
        'train_graphs': train_graphs,
        'val_graphs': val_graphs,
        'test_graphs': test_graphs,
        'scaler': scaler,
    }

    with open(cache_path, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"Saved to: {cache_path} ({size_mb:.1f} MB)")
    return result


# --------------------------------------------------------------------
# Dataset and DataLoader helpers
# --------------------------------------------------------------------
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx][0], self.pairs[idx][1]


class TgDataset(torch.utils.data.Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def collate_pairs(batch):
    """Collate function for (graph_1ru, graph_3ru) pairs."""
    graphs_1ru = [item[0] for item in batch]
    graphs_3ru = [item[1] for item in batch]
    return Batch.from_data_list(graphs_1ru), Batch.from_data_list(graphs_3ru)


# --------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Feature dims: rich={RICH_DIM}, RWPE={RWPE_K}, total={TOTAL_NODE_DIM}, edge={EDGE_DIM}")
    # Quick test
    smiles = 'CCCC'
    g = smiles_to_graph(smiles)
    if g is not None:
        print(f"Test butane: {g.num_nodes} nodes, {g.edge_index.size(1)} edges")
        print(f"  Node features shape: {g.x.shape}")
        print(f"  Edge features shape: {g.edge_attr.shape}")
