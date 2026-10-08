#!/usr/bin/env python3
"""
Dataset and data processing for MoMuCon-NT-Xent pretraining and Tg finetuning.

Features:
  - 8-dim numeric rich node features
  - 6-dim one-hot edge features
  - 32-dim RWPE (Random Walk Positional Encoding)
  - 2-dim PSPE (distance-to-chain-end, monomer-index)
  - Total: 42-dim node features

Generates:
  - (1-RU, 3-RU) graph pairs from PI1M for pretraining
  - 3-RU graphs from Tg for finetuning (72/8/20 split)
"""

import os
import pickle
import random
import warnings
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
from copy import deepcopy
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------
RWPE_K = 32
PI1M_SEED = 42
TG_SPLIT_SEED = 71

RICH_DIM = 8
EDGE_DIM = 6
PSPE_DIM = 2
TOTAL_NODE_DIM = RICH_DIM + PSPE_DIM + RWPE_K  # 42

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
# Multimer generation with PSPE tracking
# --------------------------------------------------------------------
def generate_1ru_smiles(psmiles):
    """Caps p-SMILES with H to create 1-RU SMILES."""
    smiles = psmiles.replace('*', '[H]')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)


def generate_multimer_with_info(psmiles, n_mer=3):
    """Generate N-mer oligomer SMILES + monomer assignment info.

    Returns:
        smiles: str, canonical SMILES of multimer
        monomer_info: dict with 'monomer_indices' (array mapping atom idx -> monomer unit)
            and 'chain_end_neighbors' (list of neighbor atom indices at the two chain ends)
        Or None if generation fails.
    """
    if n_mer == 1:
        smiles = generate_1ru_smiles(psmiles)
        if smiles is None:
            return None
        mol = Chem.MolFromSmiles(smiles)
        return {
            'smiles': smiles,
            'monomer_indices': np.zeros(mol.GetNumAtoms(), dtype=np.float32),
            'chain_end_neighbors': None,
            'n_atoms_per_unit': mol.GetNumAtoms(),
        }

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

    # Terminal chain end neighbors (atoms adjacent to terminal stars)
    chain_end_neighbors = [
        int(neighbor_indices[0, 0]),
        int(neighbor_indices[n_mer - 1, 1]),
    ]

    ed_mol.ReplaceAtom(int(star_indices[0, 0]), Chem.Atom(1))
    ed_mol.ReplaceAtom(int(star_indices[n_mer - 1, 1]), Chem.Atom(1))

    for idx in sorted(removed_atoms, reverse=True):
        ed_mol.RemoveAtom(idx)

    final_mol = ed_mol.GetMol()
    final_mol = Chem.RemoveHs(final_mol)

    # Compute approximate monomer indices for each remaining atom
    # Since we removed internal stars and replaced terminal stars with H (removed by RemoveHs),
    # the atom count per monomer unit is roughly: (num_atoms * n_mer - n_removed) / n_mer
    # where n_removed includes internal stars (2 per connection) and terminal H atoms (2)
    # We approximate by dividing the final atom count evenly among units.
    n_final = final_mol.GetNumAtoms()
    monomer_indices = np.zeros(n_final, dtype=np.float32)
    if n_final > 0:
        atoms_per_unit = n_final / n_mer
        for i in range(n_final):
            monomer_indices[i] = min(float(i) / atoms_per_unit, float(n_mer - 1))

    smiles = Chem.MolToSmiles(final_mol)
    return {
        'smiles': smiles,
        'monomer_indices': monomer_indices,
        'chain_end_neighbors': chain_end_neighbors,
        'n_atoms_per_unit': num_atoms,
    }


# --------------------------------------------------------------------
# 8-dim numeric rich node features
# --------------------------------------------------------------------
def extract_rich_features_8dim(mol):
    """
    8-dim numeric rich features:
      [0] atomic_number / 100.0
      [1] hybridization / 5.0
      [2] degree / 10.0 (clamped to 1.0)
      [3] formal_charge / 5.0 (clamped to [-1, 1])
      [4] aromatic (0/1)
      [5] ring_membership (IsInRing) (0/1)
      [6] chirality_tag / 10.0
      [7] hydrogen_count / 10.0 (clamped to 1.0)
    """
    features = []
    for atom in mol.GetAtoms():
        feat = [
            float(atom.GetAtomicNum()) / 100.0,
            float(HYBRIDIZATION_MAP.get(atom.GetHybridization(), HYBRIDIZATION_OTHER)) / 5.0,
            min(float(atom.GetDegree()) / 10.0, 1.0),
            max(-1.0, min(float(atom.GetFormalCharge()) / 5.0, 1.0)),
            1.0 if atom.GetIsAromatic() else 0.0,
            1.0 if atom.IsInRing() else 0.0,
            float(int(atom.GetChiralTag())) / 10.0,
            min(float(atom.GetTotalNumHs(includeNeighbors=False)) / 10.0, 1.0),
        ]
        features.append(feat)
    return torch.tensor(features, dtype=torch.float32)


# --------------------------------------------------------------------
# 6-dim one-hot edge features
# --------------------------------------------------------------------
def extract_edge_features_6dim(mol, edge_index):
    """
    6-dim edge features:
      [0:4] bond type one-hot (single, double, triple, aromatic)
      [4]   is_conjugated (0/1)
      [5]   is_in_ring (0/1)
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
        edge_attr[e, bt] = 1.0  # one-hot bond type
        edge_attr[e, 4] = 1.0 if is_conj else 0.0
        edge_attr[e, 5] = 1.0 if is_ring else 0.0

    return edge_attr


# --------------------------------------------------------------------
# PSPE computation
# --------------------------------------------------------------------
def compute_pspe(mol, monomer_indices, chain_end_neighbors, num_nodes):
    """Compute 2-dim PSPE features.

    [0] distance_to_chain_end: shortest path distance to nearest chain end, normalized
    [1] monomer_index: normalized monomer unit index
    """
    psp = np.zeros((num_nodes, PSPE_DIM), dtype=np.float32)

    # monomer_index: already computed during multimer generation
    if monomer_indices is not None and len(monomer_indices) == num_nodes:
        max_idx = monomer_indices.max()
        if max_idx > 0:
            psp[:, 1] = monomer_indices / max_idx
        else:
            psp[:, 1] = 0.0
    else:
        psp[:, 1] = 0.0

    # distance_to_chain_end
    if chain_end_neighbors is not None:
        # Map chain_end_neighbors (from pre-stripped indices) to final mol atom indices
        # The chain_end_neighbors are atom indices in the pre-H-stripping multimer
        # After H removal, atom indexing changes. We need to find the correct atoms.
        # Strategy: find atoms with degree 1 (terminal) that are at the farthest distance
        try:
            # Find all terminal atoms (degree 1 in heavy-atom graph)
            terminals = []
            for atom in mol.GetAtoms():
                # Count heavy atom neighbors
                heavy_neighbors = 0
                for nei in atom.GetNeighbors():
                    if nei.GetSymbol() != 'H':
                        heavy_neighbors += 1
                if heavy_neighbors == 1:
                    terminals.append(atom.GetIdx())

            if len(terminals) >= 2:
                # Find the two farthest terminals as chain ends
                max_dist = -1
                best_pair = (terminals[0], terminals[1])
                for i in range(min(len(terminals), 10)):
                    for j in range(i + 1, min(len(terminals), 10)):
                        try:
                            path = rdmolops.GetShortestPath(mol, terminals[i], terminals[j])
                            if path and len(path) > max_dist:
                                max_dist = len(path)
                                best_pair = (terminals[i], terminals[j])
                        except Exception:
                            continue

                end1, end2 = best_pair
                if max_dist > 0:
                    for atom_idx in range(num_nodes):
                        try:
                            if atom_idx == end1:
                                d1 = 0
                            else:
                                d1 = len(rdmolops.GetShortestPath(mol, atom_idx, end1))
                            if atom_idx == end2:
                                d2 = 0
                            else:
                                d2 = len(rdmolops.GetShortestPath(mol, atom_idx, end2))
                            min_dist = min(d1, d2)
                            psp[atom_idx, 0] = min_dist / max_dist
                        except Exception:
                            psp[atom_idx, 0] = 0.0
            else:
                # Single terminal or none: use degree-based heuristic
                pass
        except Exception:
            pass

    return psp


# --------------------------------------------------------------------
# RWPE computation
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
def smiles_to_graph_with_pspe(smiles, monomer_indices=None, chain_end_neighbors=None,
                               add_rwpe=True):
    """Convert SMILES to PyG Data with 8-dim rich + PSPE + RWPE."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    n_nodes = mol.GetNumAtoms()

    # 8-dim rich features
    x = extract_rich_features_8dim(mol)  # [N, 8]

    # PSPE
    psp = compute_pspe(mol, monomer_indices, chain_end_neighbors, n_nodes)
    x = torch.cat([x, torch.tensor(psp, dtype=torch.float32)], dim=-1)  # [N, 10]

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
        edge_attr = extract_edge_features_6dim(mol, edge_index)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_nodes)

    # RWPE
    if add_rwpe:
        rwpe = compute_rwpe(edge_index, n_nodes, RWPE_K)
        data.x = torch.cat([data.x, torch.tensor(rwpe, dtype=torch.float32)], dim=-1)

    return data


def smiles_to_graph_with_pspe_info(info, add_rwpe=True):
    """Wrapper that takes the multimer info dict and builds a graph."""
    if info is None:
        return None
    return smiles_to_graph_with_pspe(
        info['smiles'],
        monomer_indices=info.get('monomer_indices'),
        chain_end_neighbors=info.get('chain_end_neighbors'),
        add_rwpe=add_rwpe,
    )


# --------------------------------------------------------------------
# PI1M Data Preparation
# --------------------------------------------------------------------
def prepare_pi1m_pretrain_data(n_polymers=200000, seed=42, force_rebuild=False,
                                cache_dir=None):
    """Prepare (1-RU, 3-RU) graph pairs from PI1M polymers."""
    if cache_dir is None:
        cache_dir = os.getcwd()
    cache_path = os.path.join(
        cache_dir, f"pi1m_{n_polymers//1000}k_pairs_42dim.pkl")

    if os.path.exists(cache_path) and not force_rebuild:
        print(f"Loading cached PI1M pairs from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    pi1m_csv = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "alldataset", "PI1M.csv"
    )
    if not os.path.exists(pi1m_csv):
        pi1m_csv = "/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv"

    print(f"\n=== Preparing PI1M Pretraining Data ===")
    print(f"Target: {n_polymers} polymers with (1-RU, 3-RU) pairs")
    print(f"Features: {RICH_DIM}-dim rich + {PSPE_DIM}-dim PSPE + {RWPE_K}-dim RWPE = {TOTAL_NODE_DIM}-dim")

    print(f"\nLoading PI1M.csv from {pi1m_csv}...")
    df = pd.read_csv(pi1m_csv)
    print(f"Total PI1M entries: {len(df)}")

    random.seed(seed)
    np.random.seed(seed)
    indices = list(range(len(df)))
    random.shuffle(indices)
    selected = indices[:n_polymers]
    print(f"Selected {len(selected)} indices (seed={seed})")

    pairs = []
    failed = 0
    for idx in tqdm(selected, desc="Building 1-RU + 3-RU pairs"):
        psmiles = str(df.iloc[idx]['SMILES'])
        try:
            # 1-RU
            smiles_1ru = generate_1ru_smiles(psmiles)
            if smiles_1ru is None:
                failed += 1
                continue
            graph_1ru = smiles_to_graph_with_pspe(
                smiles_1ru,
                monomer_indices=None,
                chain_end_neighbors=None,
                add_rwpe=True
            )
            if graph_1ru is None:
                failed += 1
                continue

            # 3-RU
            info_3ru = generate_multimer_with_info(psmiles, n_mer=3)
            if info_3ru is None:
                failed += 1
                continue
            graph_3ru = smiles_to_graph_with_pspe_info(info_3ru, add_rwpe=True)
            if graph_3ru is None:
                failed += 1
                continue

            pairs.append((graph_1ru, graph_3ru))

        except Exception:
            failed += 1
            continue

    print(f"\nSuccessfully built {len(pairs)} pairs ({failed} failed)")
    print(f"Feature dim: {pairs[0][0].x.size(1)} (expected {TOTAL_NODE_DIM})")
    print(f"Edge feature dim: {pairs[0][0].edge_attr.size(1)} (expected {EDGE_DIM})")

    nodes_1ru = [p[0].num_nodes for p in pairs]
    nodes_3ru = [p[1].num_nodes for p in pairs]
    print(f"  1-RU nodes: min={min(nodes_1ru)}, max={max(nodes_1ru)}, mean={np.mean(nodes_1ru):.1f}")
    print(f"  3-RU nodes: min={min(nodes_3ru)}, max={max(nodes_3ru)}, mean={np.mean(nodes_3ru):.1f}")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"Saved to: {cache_path} ({size_mb:.1f} MB)")
    return pairs


# --------------------------------------------------------------------
# Tg Data Preparation
# --------------------------------------------------------------------
def prepare_tg_data(force_rebuild=False, cache_dir=None):
    """Prepare Tg finetuning data with 3-RU graphs, 42-dim features, 72/8/20 split."""
    if cache_dir is None:
        cache_dir = os.getcwd()
    cache_path = os.path.join(cache_dir, "tg_graphs_3ru_42dim.pkl")
    if os.path.exists(cache_path) and not force_rebuild:
        print(f"Loading cached Tg data from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    tg_csv = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "alldataset", "Tg.csv"
    )
    if not os.path.exists(tg_csv):
        tg_csv = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"

    print(f"\n=== Preparing Tg Finetuning Data ===")
    print(f"Features: {RICH_DIM}-dim rich + {PSPE_DIM}-dim PSPE + {RWPE_K}-dim RWPE = {TOTAL_NODE_DIM}-dim")

    df = pd.read_csv(tg_csv)
    psmiles_list = df['Smiles'].values
    tg_values = df['Tg'].values.astype(np.float32)
    print(f"Total Tg entries: {len(df)}")
    print(f"Tg range: {tg_values.min():.1f} to {tg_values.max():.1f}")

    indices = np.arange(len(psmiles_list))
    train_idx, temp_idx = train_test_split(indices, test_size=0.28, random_state=TG_SPLIT_SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.20 / 0.28, random_state=TG_SPLIT_SEED)
    print(f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)} (72/8/20)")

    def build_graphs(idx_list, desc):
        graphs = []
        failed = 0
        for i in tqdm(idx_list, desc=desc):
            psmiles = str(psmiles_list[i])
            tg = float(tg_values[i])
            try:
                info_3ru = generate_multimer_with_info(psmiles, n_mer=3)
                if info_3ru is None:
                    failed += 1
                    continue
                data = smiles_to_graph_with_pspe_info(info_3ru, add_rwpe=True)
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

    # Normalize targets (fit on training only)
    scaler = StandardScaler()
    y_train = np.array([g.y.item() for g in train_graphs]).reshape(-1, 1)
    scaler.fit(y_train)
    for g in train_graphs + val_graphs + test_graphs:
        g.y_original = g.y.clone()
        g.y = torch.tensor(scaler.transform(g.y.numpy().reshape(-1, 1))[0],
                           dtype=torch.float32)

    result = {
        'train_graphs': train_graphs,
        'val_graphs': val_graphs,
        'test_graphs': test_graphs,
        'scaler': scaler,
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"Saved to: {cache_path} ({size_mb:.1f} MB)")
    return result


# --------------------------------------------------------------------
# Dataset and DataLoader helpers
# --------------------------------------------------------------------
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
