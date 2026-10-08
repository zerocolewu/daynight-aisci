"""Data processing utilities for polymer GNN experiments (REWRITE).

p-SMILES → Graph conversion with:
  - 8 rich physical-property atom features
  - 2 PSPE (Polymer Structure Positional Encoding) features
  - 32 RWPE (Random Walk Positional Encoding, k=32)
  - Edge features (6-dim bond features)
  → Total: 42-dim node features

Augmentations (SelfGNN-correct):
  - Split: partition node feature dimensions into two halves
  - Standardize: per-graph z-score normalization
  - Combined: Split + Standardize
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import AddRandomWalkPE
from rdkit import Chem
from rdkit.Chem import GetShortestPath
import networkx as nx
import pandas as pd
from collections import deque

# Fix for PyTorch 2.6+ weights_only default
import torch_geometric.data
import torch.serialization
torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage,
])

# ─── Physical property tables ───────────────────────────────────────────────

ATOM_PROPS = {
    1:  [2.20,  0.31, 1,  1, 1,  0.754,  13.60,   1.008],
    5:  [2.04,  0.82, 3, 13, 2,  0.280,   8.30,  10.81],
    6:  [2.55,  0.76, 4, 14, 2,  1.262,  11.26,  12.011],
    7:  [3.04,  0.71, 5, 15, 2, -0.070,  14.53,  14.007],
    8:  [3.44,  0.66, 6, 16, 2,  1.461,  13.62,  15.999],
    9:  [3.98,  0.57, 7, 17, 2,  3.401,  17.42,  18.998],
    14: [1.90,  1.11, 4, 14, 3,  1.385,   8.15,  28.085],
    15: [2.19,  1.07, 5, 15, 3,  0.746,  10.49,  30.974],
    16: [2.58,  1.05, 6, 16, 3,  2.077,  10.36,  32.06],
    17: [3.16,  1.02, 7, 17, 3,  3.613,  12.97,  35.45],
    35: [2.96,  1.20, 7, 17, 4,  3.365,  11.81,  79.904],
    53: [2.66,  1.39, 7, 17, 5,  3.059,  10.45, 126.90],
}
DEFAULT_ATOM_PROP = [2.50, 0.75, 4.0, 14.0, 2.0, 1.0, 10.0, 30.0]

PROP_MEAN = np.array([2.54, 0.81, 4.68, 14.47, 2.18, 1.59, 12.42, 19.22])
PROP_STD  = np.array([0.50, 0.19, 1.48,  1.23, 0.47, 1.06,  2.66, 23.93])
PROP_STD[PROP_STD < 1e-8] = 1.0


def get_atom_rich_features(atom):
    """Return normalized 8-dim rich physical-property features for one atom."""
    anum = atom.GetAtomicNum()
    raw = np.array(ATOM_PROPS.get(anum, DEFAULT_ATOM_PROP), dtype=np.float32)
    return ((raw - PROP_MEAN) / PROP_STD).tolist()


def bond_features(bond):
    """Return 6-dim edge features: 4 bond-type + conjugated + in_ring."""
    bt_map = {
        Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3,
    }
    bt_onehot = [0.0, 0.0, 0.0, 0.0]
    bt_onehot[bt_map.get(bond.GetBondType(), 0)] = 1.0
    return bt_onehot + [float(bond.GetIsConjugated()), float(bond.IsInRing())]


def psmiles_to_graph(psmiles, add_rwpe=False, rwpe_dim=32):
    """Convert a single p-SMILES to a PyG Data object.

    Args:
        psmiles: p-SMILES string (contains '*' polymerization sites)
        add_rwpe: whether to add RWPE features
        rwpe_dim: RWPE walk length (default 32)

    Returns:
        torch_geometric.data.Data or None on failure
    """
    # 1. Parse p-SMILES with * atoms
    mol = Chem.MolFromSmiles(psmiles)
    if mol is None:
        return None

    # 2. Locate * atoms and their neighbours
    star_idxs = set()
    star_neighbors = []
    for a in mol.GetAtoms():
        if a.GetSymbol() == '*':
            star_idxs.add(a.GetIdx())
            neis = a.GetNeighbors()
            if len(neis) > 0:
                star_neighbors.append(neis[0].GetIdx())

    total_atoms = mol.GetNumAtoms()

    # 3. Build adjacency for BFS
    adj = {i: [] for i in range(total_atoms)}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)

    # 4. Compute PSPE features on the full graph (including * atoms)
    pspe1 = np.ones(total_atoms, dtype=np.float32)
    pspe2 = np.zeros(total_atoms, dtype=np.float32)

    if len(star_neighbors) >= 2:
        sn0, sn1 = star_neighbors[0], star_neighbors[1]
        # BFS distances from each star-neighbour
        all_dists = []
        for src in [sn0, sn1]:
            dist = np.full(total_atoms, -1, dtype=np.float32)
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] < 0:
                        dist[v] = dist[u] + 1
                        q.append(v)
            all_dists.append(dist)

        min_d = np.minimum(all_dists[0], all_dists[1])
        max_d = max(np.max(min_d[min_d >= 0]), 1.0)
        pspe1 = np.where(min_d >= 0, min_d / max_d, 1.0)

        # Shortest-path backbone: nodes on path between sn0 and sn1
        parent = {sn0: None}
        q = deque([sn0])
        while q:
            u = q.popleft()
            if u == sn1:
                break
            for v in adj[u]:
                if v not in parent:
                    parent[v] = u
                    q.append(v)
        if sn1 in parent:
            cur = sn1
            while cur is not None:
                pspe2[cur] = 1.0
                cur = parent[cur]

    elif len(star_neighbors) == 1:
        sn = star_neighbors[0]
        dist = np.full(total_atoms, 1.0, dtype=np.float32)
        dist[sn] = 0
        q = deque([sn])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] >= 1.0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        max_d = max(dist.max(), 1.0)
        pspe1 = dist / max_d

    # 5. Build atom features for ALL atoms (including *)
    all_features = []
    for a in mol.GetAtoms():
        if a.GetSymbol() == '*':
            all_features.append([0.0] * 8)
        else:
            all_features.append(get_atom_rich_features(a))

    # 6. Filter out * atoms
    star_list = sorted(star_idxs, reverse=True)
    keep = np.ones(total_atoms, dtype=bool)
    for s in star_list:
        keep[s] = False

    old2new = {}
    ni = 0
    for i in range(total_atoms):
        if keep[i]:
            old2new[i] = ni
            ni += 1

    # 7. Build edge_index and edge_attr
    edge_idx, edge_attr_list = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if keep[i] and keep[j]:
            ni, nj = old2new[i], old2new[j]
            feat = bond_features(b)
            edge_idx += [[ni, nj], [nj, ni]]
            edge_attr_list += [feat, feat]

    if len(edge_idx) == 0:
        return None

    # 8. Assemble node features: 8 rich + 2 PSPE = 10
    x_list = []
    pspe_list = []
    for i in range(total_atoms):
        if keep[i]:
            x_list.append(all_features[i])
            pspe_list.append([pspe1[i], pspe2[i]])

    x = torch.tensor(x_list, dtype=torch.float)
    p = torch.tensor(pspe_list, dtype=torch.float)
    x_full = torch.cat([x, p], dim=-1)  # [N, 10]

    edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    data = Data(x=x_full, edge_index=edge_index, edge_attr=edge_attr)

    # 9. Optionally add RWPE (makes 10+32=42 dim total)
    if add_rwpe:
        try:
            transform = AddRandomWalkPE(walk_length=rwpe_dim, attr_name='pe')
            data = transform(data)
            data.x = torch.cat([data.x, data.pe], dim=-1)  # [N, 42]
        except Exception:
            # If RWPE fails (e.g., graph too small), pad with zeros
            pe = torch.zeros(data.x.size(0), rwpe_dim)
            data.x = torch.cat([data.x, pe], dim=-1)

    return data


# ─── Dataset classes ─────────────────────────────────────────────────────────

class PI1MDataset(InMemoryDataset):
    """Dataset for PI1M pretraining (1-RU, 42-dim node features with RWPE)."""
    def __init__(self, csv_path, n_samples=200000, seed=42, root=None, transform=None):
        self.csv_path = csv_path
        self.n_samples = n_samples
        self.seed = seed
        super().__init__(root=root, transform=transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'pi1m_{self.n_samples}.pt']

    def process(self):
        import time
        df = pd.read_csv(self.csv_path)
        if 'SMILES' in df.columns:
            col = 'SMILES'
        elif 'Smiles' in df.columns:
            col = 'Smiles'
        else:
            col = df.columns[0]

        smiles_list = df[col].dropna().values
        rng = np.random.RandomState(self.seed)
        indices = rng.choice(len(smiles_list), min(self.n_samples, len(smiles_list)), replace=False)

        data_list = []
        n_total = len(indices)
        n_failed = 0
        t_start = time.time()

        for i, idx in enumerate(indices):
            graph = psmiles_to_graph(smiles_list[idx], add_rwpe=True, rwpe_dim=32)
            if graph is not None and graph.x.size(0) > 1:
                graph.y = torch.tensor([0.0])  # dummy label
                data_list.append(graph)
            else:
                n_failed += 1

            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (n_total - i - 1) / rate if rate > 0 else 0
                print(f"  PI1M processing: {i+1}/{n_total} graphs "
                      f"({100*(i+1)/n_total:.1f}%) | {rate:.1f} graphs/s | ETA: {eta/60:.1f} min")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"PI1M dataset: {len(data_list)}/{n_total} graphs saved "
              f"({n_failed} failed, time={time.time()-t_start:.0f}s)")


class TgDataset(InMemoryDataset):
    """Dataset for Tg finetuning (3-RU, 42-dim node features: 8+2+32)."""
    def __init__(self, csv_path, n_ru=3, add_rwpe=True, rwpe_dim=32,
                 root=None, transform=None):
        self.csv_path = csv_path
        self.n_ru = n_ru
        self.add_rwpe = add_rwpe
        self.rwpe_dim = rwpe_dim
        super().__init__(root=root, transform=transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'tg_{self.n_ru}ru.pt']

    def process(self):
        df = pd.read_csv(self.csv_path)
        if 'Smiles' in df.columns:
            col = 'Smiles'
        else:
            col = df.columns[0]
        tg_col = 'Tg' if 'Tg' in df.columns else df.columns[1]

        data_list = []
        for _, row in df.iterrows():
            psmiles = row[col]
            tg = row[tg_col]
            if pd.isna(psmiles) or pd.isna(tg):
                continue

            # Generate N-RU multimer
            if self.n_ru > 1:
                smiles = generate_multimer(psmiles, self.n_ru)
            else:
                smiles = psmiles.replace('*', '[H]')

            if smiles is None:
                continue

            graph = psmiles_to_graph(smiles, add_rwpe=False)
            if graph is None or graph.x.size(0) < 3:
                continue

            # Add RWPE
            if self.add_rwpe:
                try:
                    transform = AddRandomWalkPE(walk_length=self.rwpe_dim, attr_name='pe')
                    graph = transform(graph)
                    graph.x = torch.cat([graph.x, graph.pe], dim=-1)
                except Exception:
                    continue

            graph.y = torch.tensor([float(tg)], dtype=torch.float)
            data_list.append(graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Tg dataset: {len(data_list)} graphs saved.")


# ─── Multimer generation ────────────────────────────────────────────────────

def generate_multimer(psmiles, n_mer):
    """Generate N-mer oligomer SMILES from p-SMILES."""
    from copy import deepcopy

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
                'bond_type': bond.GetBondType(),
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
            order=star_info[0]['bond_type'],
        )
        removed_atoms.extend([
            int(star_indices[i, 1]),
            int(star_indices[i + 1, 0]),
        ])

    # Cap ends with H
    ed_mol.ReplaceAtom(int(star_indices[0, 0]), Chem.Atom(1))
    ed_mol.ReplaceAtom(int(star_indices[n_mer - 1, 1]), Chem.Atom(1))

    for idx in sorted(removed_atoms, reverse=True):
        ed_mol.RemoveAtom(idx)

    final_mol = ed_mol.GetMol()
    try:
        Chem.SanitizeMol(final_mol)
    except Exception:
        return None
    final_mol = Chem.RemoveHs(final_mol)
    return Chem.MolToSmiles(final_mol)


# ─── Augmentation functions ─────────────────────────────────────────────────

def augment_split(data):
    """SelfGNN Split augmentation: split node features along the feature dimension.

    Randomly permutes feature dimensions and creates two views:
      - View 1: sees first half of permuted features (other half zeroed)
      - View 2: sees second half of permuted features (first half zeroed)

    This is a FEATURE-LEVEL operation (not topological).
    Forces the encoder to reconstruct full representations from partial features.

    Args:
        data: PyG Data object with node features x of shape [N, F]

    Returns:
        view1, view2: two Data objects with same structure but split features
    """
    F = data.x.size(1)
    half = F // 2

    # Random permutation of feature dimensions
    perm = torch.randperm(F)

    # View 1: first half of permuted features
    mask1 = torch.zeros(F, device=data.x.device)
    mask1[perm[:half]] = 1.0
    x1 = data.x * mask1.unsqueeze(0)

    # View 2: second half of permuted features
    mask2 = torch.zeros(F, device=data.x.device)
    mask2[perm[half:]] = 1.0
    x2 = data.x * mask2.unsqueeze(0)

    view1 = Data(
        x=x1,
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
    )
    view2 = Data(
        x=x2,
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
    )
    return view1, view2


def augment_standardize(data):
    """SelfGNN Standardize augmentation: per-graph z-score normalization.

    Normalizes node features to zero mean and unit variance per graph.
    Creates two views:
      - View 1: z-score normalized features
      - View 2: original features (identity)

    This creates asymmetry between views to prevent collapse.

    Args:
        data: PyG Data object with node features

    Returns:
        view1 (standardized), view2 (original)
    """
    x = data.x
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_norm = (x - mean) / std

    view1 = Data(
        x=x_norm,
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
    )
    view2 = Data(
        x=x.clone(),
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
    )
    return view1, view2


def augment_combined(data):
    """Combined SelfGNN Split + Standardize augmentation.

    Applies both augmentations:
      - View 1: Split (first half features) + Standardize (z-score norm)
      - View 2: Split (second half features) + Standardize (z-score norm)
    """
    # First apply split (creates two feature-half views)
    v1, v2 = augment_split(data)

    # Then apply standardize to each split view
    x1_mean = v1.x.mean(dim=0, keepdim=True)
    x1_std = v1.x.std(dim=0, keepdim=True).clamp(min=1e-6)
    v1.x = (v1.x - x1_mean) / x1_std

    x2_mean = v2.x.mean(dim=0, keepdim=True)
    x2_std = v2.x.std(dim=0, keepdim=True).clamp(min=1e-6)
    v2.x = (v2.x - x2_mean) / x2_std

    return v1, v2


def split_dataset(dataset, train_ratio=0.72, val_ratio=0.08, seed=71):
    """Split dataset into train/val/test sets."""
    n = len(dataset)
    indices = list(range(n))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return train_idx, val_idx, test_idx
