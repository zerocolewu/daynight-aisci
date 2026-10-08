#!/usr/bin/env python3
"""
Prepare PI1M pretraining data: 200,000 polymers, 1-RU graphs.
Features: 8-dim rich + 2-dim PSPE-combined = 10 dims.
"""

import os
import sys
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
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Paths
DATA_DIR = "/home/home_public/aisci/AIscientist/src/alldataset"
WORK_DIR = "/home/home_public/aisci/AIscientist/src/4_day4_selfgnn_polymer"
OUTPUT_DIR = os.path.join(WORK_DIR, "data", "pi1m_pretrain_cache")
PI1M_CSV = os.path.join(DATA_DIR, "PI1M.csv")
OUTPUT_PKL = os.path.join(OUTPUT_DIR, "pi1m_200k_1ru_10dim.pkl")
SUBSET_SIZE = 200000
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------
# 1-RU graph: cap * with H
# --------------------------------------------------------------------
def generate_1ru_smiles(psmiles: str) -> str:
    """Generate 1-RU SMILES by capping * with H."""
    smiles = psmiles.replace('*', '[H]')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)


# --------------------------------------------------------------------
# 8-dim rich features
# --------------------------------------------------------------------
HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
}


def extract_rich_features_8dim(mol):
    """Extract 8-dim rich atom features (numeric, not one-hot).

    Features:
      0: atomic_number / 100
      1: hybridization (enum value / 6.0) -- SP=0, SP2=1, SP3=2, SP3D=3, SP3D2=4, OTHER=5
      2: aromatic (binary)
      3: degree / 10
      4: formal_charge / 5
      5: ring_membership (binary)
      6: chirality (int value / 10)
      7: hydrogen_count / 10
    """
    features = []
    for atom in mol.GetAtoms():
        feat = [
            float(atom.GetAtomicNum()) / 100.0,
            float(HYBRIDIZATION_MAP.get(atom.GetHybridization(), 5)) / 6.0,
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
# PSPE computation (same as Day 3)
# --------------------------------------------------------------------
def find_backbone_path(edge_index, num_nodes):
    """Find the longest shortest-path backbone via double-BFS."""
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
    """Compute PSPE-dce and PSPE-mi."""
    backbone_path, backbone_length = find_backbone_path(edge_index, num_nodes)

    dce = np.zeros(num_nodes, dtype=np.float32)
    mi = np.zeros(num_nodes, dtype=np.float32)

    if backbone_length < 3:
        return dce, mi

    pos_map = {node_idx: p for p, node_idx in enumerate(backbone_path)}

    # PSPE-dce: distance-to-chain-end
    max_possible = (backbone_length - 1) / 2.0
    for node_idx in backbone_path:
        p = pos_map[node_idx]
        raw_dce = min(p, backbone_length - 1 - p)
        dce[node_idx] = raw_dce / max_possible if max_possible > 0 else 0.0

    # PSPE-mi: monomer-index (3 segments: 0, 0.5, 1.0)
    segment_size = backbone_length / 3.0
    for node_idx in backbone_path:
        p = pos_map[node_idx]
        segment_idx = min(int(p / segment_size), 2)
        mi[node_idx] = segment_idx / 2.0

    return dce, mi


# --------------------------------------------------------------------
# SMILES -> PyG Data
# --------------------------------------------------------------------
def smiles_to_pyg_data(smiles: str):
    """Convert SMILES to PyG Data with 8-dim rich features + PSPE."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # 8-dim rich features
    x = extract_rich_features_8dim(mol)

    # Edges
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # PSPE
    dce, mi = compute_pspe(edge_index, x.size(0))

    # Combine: 8 rich + 2 PSPE = 10 dims
    dce_t = torch.tensor(dce, dtype=torch.float32).unsqueeze(-1)
    mi_t = torch.tensor(mi, dtype=torch.float32).unsqueeze(-1)
    x = torch.cat([x, dce_t, mi_t], dim=-1)

    data = Data(x=x, edge_index=edge_index, num_nodes=x.size(0))
    return data


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    print("=== PI1M Pretraining Data Preparation ===")
    print(f"Target: {SUBSET_SIZE} polymers, 1-RU graphs, 10-dim features")
    print(f"Output: {OUTPUT_PKL}")

    # Load PI1M
    print("\nLoading PI1M.csv...")
    df = pd.read_csv(PI1M_CSV)
    print(f"Total PI1M entries: {len(df)}")

    # Random sample 200K
    random.seed(SEED)
    np.random.seed(SEED)
    indices = list(range(len(df)))
    random.shuffle(indices)
    selected = indices[:SUBSET_SIZE]
    print(f"Selected {len(selected)} indices (seed={SEED})")

    # Build graphs
    graphs = []
    failed = 0
    for idx in tqdm(selected, desc="Building 1-RU graphs"):
        psmiles = df.iloc[idx]['SMILES']
        try:
            smiles = generate_1ru_smiles(psmiles)
            if smiles is None:
                failed += 1
                continue
            data = smiles_to_pyg_data(smiles)
            if data is None:
                failed += 1
                continue
            graphs.append(data)
        except Exception:
            failed += 1
            continue

    print(f"\nSuccessfully built {len(graphs)} graphs ({failed} failed)")

    # Save
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Stats
    num_nodes = [g.num_nodes for g in graphs]
    num_edges = [g.edge_index.size(1) for g in graphs]
    print(f"\nGraph statistics:")
    print(f"  Nodes: min={min(num_nodes)}, max={max(num_nodes)}, mean={np.mean(num_nodes):.1f}, median={np.median(num_nodes):.1f}")
    print(f"  Edges: min={min(num_edges)}, max={max(num_edges)}, mean={np.mean(num_edges):.1f}, median={np.median(num_edges):.1f}")
    print(f"  Feature dim: {graphs[0].x.size(1)}")
    print(f"\nSaved to: {OUTPUT_PKL}")

    import os as _os
    size_mb = _os.path.getsize(OUTPUT_PKL) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
