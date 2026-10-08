#!/usr/bin/env python3
"""
Prepare Tg finetuning data: 7170 polymers, 3-RU graphs.
Features: 8-dim rich + 2-dim PSPE + 32-dim RWPE = 42 dims.
Split: 72/8/20, seed=71.
"""

import os
import sys
import pickle
import warnings
from collections import deque
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from copy import deepcopy
from torch_geometric.data import Data
from torch_geometric.transforms import AddRandomWalkPE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')

DATA_DIR = "/home/home_public/aisci/AIscientist/src/alldataset"
WORK_DIR = "/home/home_public/aisci/AIscientist/src/4_day4_selfgnn_polymer"
OUTPUT_DIR = os.path.join(WORK_DIR, "data", "tg_cache")
TG_CSV = os.path.join(DATA_DIR, "Tg.csv")
OUTPUT_PKL = os.path.join(OUTPUT_DIR, "tg_graphs_3ru_42dim.pkl")
SPLIT_SEED = 71
RWPE_K = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------
# 3-RU Multimer
# --------------------------------------------------------------------
def generate_multimer(psmiles: str, n_mer: int) -> str:
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

    ed_mol.ReplaceAtom(int(star_indices[0, 0]), Chem.Atom(1))
    ed_mol.ReplaceAtom(int(star_indices[n_mer - 1, 1]), Chem.Atom(1))

    for idx in sorted(removed_atoms, reverse=True):
        ed_mol.RemoveAtom(idx)

    final_mol = ed_mol.GetMol()
    final_mol = Chem.RemoveHs(final_mol)
    return Chem.MolToSmiles(final_mol)


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
# Main
# --------------------------------------------------------------------
def main():
    print("=== Tg Finetuning Data Preparation ===")
    print(f"3-RU graphs, 8 rich + 2 PSPE + {RWPE_K} RWPE = {8+2+RWPE_K} dims")
    print(f"Split: 72/8/20, seed={SPLIT_SEED}")

    # Load Tg data
    print("\nLoading Tg.csv...")
    df = pd.read_csv(TG_CSV)
    print(f"Total Tg entries: {len(df)}")

    psmiles_list = df['Smiles'].values
    tg_values = df['Tg'].values.astype(np.float32)

    # Split indices
    indices = np.arange(len(psmiles_list))
    train_idx, temp_idx = train_test_split(indices, test_size=0.28, random_state=SPLIT_SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.20/0.28, random_state=SPLIT_SEED)

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # RWPE transform
    rwpe_transform = AddRandomWalkPE(walk_length=RWPE_K)

    def build_graphs(idx_list, desc):
        """Build PyG Data objects for a list of indices."""
        graphs = []
        failed = 0
        for idx in tqdm(idx_list, desc=desc):
            psmiles = str(psmiles_list[idx])
            tg = float(tg_values[idx])
            try:
                smiles = generate_multimer(psmiles, n_mer=3)
                if smiles is None:
                    failed += 1
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is None or mol.GetNumAtoms() == 0:
                    failed += 1
                    continue

                # 8-dim rich features
                x = extract_rich_features_8dim(mol)

                # Edges
                edges = []
                for bond in mol.GetBonds():
                    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    edges.append([i, j])
                    edges.append([j, i])

                if len(edges) == 0:
                    failed += 1
                    continue

                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

                # PSPE
                dce, mi = compute_pspe(edge_index, x.size(0))
                dce_t = torch.tensor(dce, dtype=torch.float32).unsqueeze(-1)
                mi_t = torch.tensor(mi, dtype=torch.float32).unsqueeze(-1)
                x = torch.cat([x, dce_t, mi_t], dim=-1)  # 10 dims

                data = Data(x=x, edge_index=edge_index, num_nodes=x.size(0))

                # RWPE
                try:
                    data = rwpe_transform(data)
                except Exception:
                    data.random_walk_pe = torch.zeros(data.num_nodes, RWPE_K)

                # Combine: 8 rich + 2 PSPE + 32 RWPE = 42 dims
                data.x = torch.cat([data.x, data.random_walk_pe], dim=-1)
                del data.random_walk_pe

                data.y = torch.tensor([tg], dtype=torch.float32)
                graphs.append(data)

            except Exception:
                failed += 1
                continue

        print(f"  Built {len(graphs)} graphs ({failed} failed)")
        return graphs

    train_graphs = build_graphs(train_idx, "Building train graphs")
    val_graphs = build_graphs(val_idx, "Building val graphs")
    test_graphs = build_graphs(test_idx, "Building test graphs")

    # Fit scaler
    scaler = StandardScaler()
    y_train = np.array([g.y.item() for g in train_graphs]).reshape(-1, 1)
    scaler.fit(y_train)

    # Normalize targets
    for g in train_graphs + val_graphs + test_graphs:
        g.y = torch.tensor(scaler.transform(g.y.numpy().reshape(-1, 1))[0], dtype=torch.float32)

    # Save
    output = {
        'train_graphs': train_graphs,
        'val_graphs': val_graphs,
        'test_graphs': test_graphs,
        'scaler': scaler,
    }
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Stats
    all_graphs = train_graphs + val_graphs + test_graphs
    num_nodes = [g.num_nodes for g in all_graphs]
    print(f"\nGraph statistics:")
    print(f"  Nodes: min={min(num_nodes)}, max={max(num_nodes)}, mean={np.mean(num_nodes):.1f}")
    print(f"  Feature dim: {all_graphs[0].x.size(1)}")
    print(f"  Train/Val/Test: {len(train_graphs)}/{len(val_graphs)}/{len(test_graphs)}")
    print(f"  Target mean: {scaler.mean_[0]:.2f}, std: {scaler.scale_[0]:.2f}")

    import os as _os
    size_mb = _os.path.getsize(OUTPUT_PKL) / (1024 * 1024)
    print(f"\nSaved to: {OUTPUT_PKL} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
