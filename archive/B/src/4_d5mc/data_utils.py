#!/usr/bin/env python3
"""
Data utilities for MoMuCon NT-Xent contrastive pretraining and Tg finetuning.

Features: 8-dim numeric rich + 2-dim PSPE + 32-dim RWPE = 42 dims.
Edge features: bond_type (single=0/double=1/triple=2/aromatic=3) normalized by /4
               + conjugated (binary) + in_ring (binary).
"""

import os
import pickle
import random
import warnings
from collections import deque
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

DATA_DIR = "/home/home_public/aisci/AIscientist/src/alldataset"
WORK_DIR = "/home/home_public/aisci/AIscientist/src/4_d5mc"
CACHE_DIR = os.path.join(WORK_DIR, "cache")
PI1M_CSV = os.path.join(DATA_DIR, "PI1M.csv")
TG_CSV = os.path.join(DATA_DIR, "Tg.csv")
RWPE_K = 32
PI1M_SUBSET = 50000
SPLIT_SEED = 71
PI1M_SEED = 42

RICH_DIM = 8
PSPE_DIM = 2
TOTAL_DIM = RICH_DIM + PSPE_DIM + RWPE_K  # 42
EDGE_DIM = 3

os.makedirs(CACHE_DIR, exist_ok=True)

# --------------------------------------------------------------------
# Hybridization map
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
    smiles = psmiles.replace('*', '[H]')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)


def generate_multimer(psmiles: str, n_mer: int):
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
# 8-dim numeric rich features
# --------------------------------------------------------------------
def extract_rich_features_8dim(mol):
    features = []
    for atom in mol.GetAtoms():
        feat = [
            float(atom.GetAtomicNum()) / 100.0,
            float(HYBRIDIZATION_MAP.get(atom.GetHybridization(), HYBRIDIZATION_OTHER)) / 6.0,
            1.0 if atom.GetIsAromatic() else 0.0,
            min(float(atom.GetDegree()) / 10.0, 1.0),
            max(-1.0, min(float(atom.GetFormalCharge()) / 5.0, 1.0)),
            1.0 if atom.IsInRing() else 0.0,
            float(int(atom.GetChiralTag())) / 10.0,
            min(float(atom.GetTotalNumHs(includeNeighbors=False)) / 10.0, 1.0),
        ]
        features.append(feat)
    return torch.tensor(features, dtype=torch.float32)


# --------------------------------------------------------------------
# PSPE computation
# --------------------------------------------------------------------
def find_backbone_path(edge_index, num_nodes):
    adj = {i: [] for i in range(num_nodes)}
    ei = edge_index.numpy()
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
    backbone_length = len(backbone_path)
    return backbone_path, backbone_length


def compute_pspe(edge_index, num_nodes):
    backbone_path, backbone_length = find_backbone_path(edge_index, num_nodes)
    dce = np.zeros(num_nodes, dtype=np.float32)
    mi = np.zeros(num_nodes, dtype=np.float32)
    if backbone_length < 3:
        return dce, mi

    pos_map = {node_idx: p for p, node_idx in enumerate(backbone_path)}
    max_possible = (backbone_length - 1) / 2.0
    for node_idx in backbone_path:
        p = pos_map[node_idx]
        raw_dce = min(p, backbone_length - 1 - p)
        dce[node_idx] = raw_dce / max_possible if max_possible > 0 else 0.0

    segment_size = backbone_length / 3.0
    for node_idx in backbone_path:
        p = pos_map[node_idx]
        segment_idx = min(int(p / segment_size), 2)
        mi[node_idx] = segment_idx / 2.0

    return dce, mi


# --------------------------------------------------------------------
# Edge features
# --------------------------------------------------------------------
def extract_edge_features(mol, edge_index):
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
        edge_attr[e, 0] = float(bt) / 4.0
        edge_attr[e, 1] = 1.0 if is_conj else 0.0
        edge_attr[e, 2] = 1.0 if is_ring else 0.0

    return edge_attr


# --------------------------------------------------------------------
# Fast RWPE computation
# --------------------------------------------------------------------
def compute_rwpe_fast(edge_index, num_nodes, k_steps=RWPE_K):
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
def smiles_to_graph(smiles, add_pspe=True, add_rwpe=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    n_nodes = mol.GetNumAtoms()

    x = extract_rich_features_8dim(mol)

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

    if add_pspe:
        dce, mi = compute_pspe(edge_index, n_nodes)
        dce_t = torch.tensor(dce, dtype=torch.float32).unsqueeze(-1)
        mi_t = torch.tensor(mi, dtype=torch.float32).unsqueeze(-1)
        data.x = torch.cat([data.x, dce_t, mi_t], dim=-1)
    else:
        zero_pspe = torch.zeros(n_nodes, PSPE_DIM, dtype=torch.float32)
        data.x = torch.cat([data.x, zero_pspe], dim=-1)

    if add_rwpe:
        rwpe = compute_rwpe_fast(edge_index, n_nodes, RWPE_K)
        data.x = torch.cat([data.x, torch.tensor(rwpe, dtype=torch.float32)], dim=-1)

    return data


# --------------------------------------------------------------------
# PI1M data loading
# --------------------------------------------------------------------
def prepare_pi1m_pretrain_data(force_rebuild=False):
    cache_path = os.path.join(CACHE_DIR, "pi1m_50k_pairs_42dim.pkl")

    if os.path.exists(cache_path) and not force_rebuild:
        print(f"Loading cached PI1M pairs from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"\n=== Preparing PI1M Pretraining Data ===")
    print(f"Target: {PI1M_SUBSET} polymers with (1-RU, 3-RU) pairs")
    print(f"Features: {RICH_DIM}-dim rich + 2-dim PSPE-pad + {RWPE_K}-dim RWPE = {TOTAL_DIM}-dim")

    print("\nLoading PI1M.csv...")
    df = pd.read_csv(PI1M_CSV)
    print(f"Total PI1M entries: {len(df)}")

    random.seed(PI1M_SEED)
    np.random.seed(PI1M_SEED)
    indices = list(range(len(df)))
    random.shuffle(indices)
    selected = indices[:PI1M_SUBSET]
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

            graph_1ru = smiles_to_graph(smiles_1ru, add_pspe=False, add_rwpe=True)
            graph_3ru = smiles_to_graph(smiles_3ru, add_pspe=False, add_rwpe=True)

            if graph_1ru is None or graph_3ru is None:
                failed += 1
                continue

            pairs.append((graph_1ru, graph_3ru))

        except Exception:
            failed += 1
            continue

    print(f"\nSuccessfully built {len(pairs)} pairs ({failed} failed)")

    nodes_1ru = [p[0].num_nodes for p in pairs]
    nodes_3ru = [p[1].num_nodes for p in pairs]
    print(f"  1-RU nodes: min={min(nodes_1ru)}, max={max(nodes_1ru)}, mean={np.mean(nodes_1ru):.1f}")
    print(f"  3-RU nodes: min={min(nodes_3ru)}, max={max(nodes_3ru)}, mean={np.mean(nodes_3ru):.1f}")
    print(f"  Feature dim: {pairs[0][0].x.size(1)}")

    with open(cache_path, 'wb') as f:
        pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"\nSaved to: {cache_path} ({size_mb:.1f} MB)")

    return pairs


# --------------------------------------------------------------------
# Tg data loading
# --------------------------------------------------------------------
def prepare_tg_data(force_rebuild=False, train_frac=0.72, val_frac=0.08, test_frac=0.20):
    cache_path = os.path.join(CACHE_DIR, "tg_graphs_3ru.pkl")
    if os.path.exists(cache_path) and not force_rebuild:
        print(f"Loading cached Tg data from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"\n=== Preparing Tg Finetuning Data ===")
    print(f"Features: {RICH_DIM}-dim rich + {PSPE_DIM}-dim PSPE + {RWPE_K}-dim RWPE = {TOTAL_DIM}-dim")

    df = pd.read_csv(TG_CSV)
    psmiles_list = df['Smiles'].values
    tg_values = df['Tg'].values.astype(np.float32)

    indices = np.arange(len(psmiles_list))
    train_idx, temp_idx = train_test_split(indices, test_size=(val_frac + test_frac), random_state=SPLIT_SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_frac/(val_frac+test_frac), random_state=SPLIT_SEED)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

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
                data = smiles_to_graph(smiles, add_pspe=True, add_rwpe=True)
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

    scaler = StandardScaler()
    y_train = np.array([g.y.item() for g in train_graphs]).reshape(-1, 1)
    scaler.fit(y_train)
    for g in train_graphs + val_graphs + test_graphs:
        g.y = torch.tensor(scaler.transform(g.y.numpy().reshape(-1, 1))[0], dtype=torch.float32)

    result = {'train_graphs': train_graphs, 'val_graphs': val_graphs,
              'test_graphs': test_graphs, 'scaler': scaler}

    with open(cache_path, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"\nSaved to: {cache_path} ({size_mb:.1f} MB)")

    return result


# --------------------------------------------------------------------
# Dataset wrappers
# --------------------------------------------------------------------
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
