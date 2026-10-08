"""
PPFN-v2 Dataset: Simplified dataset focusing on trimer graph with periodic edges.

Removes the complex fragmentation logic from v1.
Only builds trimer graphs with periodic boundary edges.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from typing import Optional

# Add parent data processing code to path
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/alldataset/some_dataprocesscode')
from multimer import generate_multimer_smiles


# ── Atom feature dimensions ──────────────────────────────────────────────
ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I',
    'B', 'Se', 'Te', 'Ge', 'As', 'Sb', 'Bi', '*'
]
ATOM_TYPES_DICT = {t: i for i, t in enumerate(ATOM_TYPES)}

HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
HYBRIDIZATION_DICT = {t: i for i, t in enumerate(HYBRIDIZATION_TYPES)}

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_TYPES_DICT = {t: i for i, t in enumerate(BOND_TYPES)}

NUM_ATOM_FEATURES = len(ATOM_TYPES) + 1 + len(HYBRIDIZATION_TYPES) + 1 + 1 + 1  # 27
NUM_EDGE_FEATURES = len(BOND_TYPES) + 1 + 1  # 6


def atom_features(atom: Chem.Atom) -> np.ndarray:
    """Compute enhanced atom features."""
    features = []
    # Atom type (one-hot)
    atom_type = atom.GetSymbol()
    type_feat = np.zeros(len(ATOM_TYPES))
    if atom_type in ATOM_TYPES_DICT:
        type_feat[ATOM_TYPES_DICT[atom_type]] = 1.0
    features.append(type_feat)
    # Degree (normalized)
    features.append(np.array([float(atom.GetDegree()) / 6.0]))
    # Hybridization (one-hot)
    hybrid = atom.GetHybridization()
    hybrid_feat = np.zeros(len(HYBRIDIZATION_TYPES))
    if hybrid in HYBRIDIZATION_DICT:
        hybrid_feat[HYBRIDIZATION_DICT[hybrid]] = 1.0
    features.append(hybrid_feat)
    # Aromatic
    features.append(np.array([float(atom.GetIsAromatic())]))
    # Formal charge (normalized)
    features.append(np.array([float(atom.GetFormalCharge()) / 4.0]))
    # In ring
    features.append(np.array([float(atom.IsInRing())]))
    return np.concatenate(features)


def bond_features(bond: Chem.Bond) -> np.ndarray:
    """Compute bond features."""
    features = []
    # Bond type (one-hot)
    bond_type = bond.GetBondType()
    type_feat = np.zeros(len(BOND_TYPES))
    if bond_type in BOND_TYPES_DICT:
        type_feat[BOND_TYPES_DICT[bond_type]] = 1.0
    else:
        type_feat[0] = 1.0  # default single
    features.append(type_feat)
    # Conjugated
    features.append(np.array([float(bond.GetIsConjugated())]))
    # In ring
    features.append(np.array([float(bond.IsInRing())]))
    return np.concatenate(features)


def build_trimer_graph(p_smiles: str) -> Optional[Chem.Mol]:
    """Build trimer (3 repeat units) from p-SMILES."""
    try:
        trimer_smiles = generate_multimer_smiles(
            num_repeat_units=3, smiles=p_smiles, replace_dummy_atoms=False
        )
        trimer_mol = Chem.MolFromSmiles(trimer_smiles)
        if trimer_mol is None:
            return None
        return trimer_mol
    except Exception:
        return None


def mol_to_graph_data(mol: Chem.Mol) -> tuple:
    """Convert RDKit mol to atom features, edge index, and edge features."""
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(atom_features(atom))
    atom_feats = np.stack(atom_feats)

    edge_indices = []
    edge_feats = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        bf = bond_features(bond)
        edge_feats.append(bf)
        edge_feats.append(bf)

    if len(edge_indices) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, NUM_EDGE_FEATURES))
    else:
        edge_index = np.array(edge_indices, dtype=np.int64).T
        edge_features = np.stack(edge_feats)

    return atom_feats, edge_index, edge_features


def process_polymer_v2(p_smiles: str, tg: float, use_periodic_edge: bool = True) -> Optional[dict]:
    """
    Process a single polymer into graph data for PPFN-v2.

    Args:
        p_smiles: polymer SMILES
        tg: glass transition temperature
        use_periodic_edge: whether to add periodic edges between star atoms

    Returns:
        Dictionary with graph data or None if processing fails
    """
    # Build trimer
    trimer_mol = build_trimer_graph(p_smiles)
    if trimer_mol is None:
        return None

    # Get atom features and edges
    atom_feats, edge_index, edge_feats = mol_to_graph_data(trimer_mol)

    # Add periodic edges between star (dummy) atoms
    if use_periodic_edge:
        star_atoms = [a.GetIdx() for a in trimer_mol.GetAtoms() if a.GetSymbol() == '*']

        periodic_edges = []
        periodic_edge_feats = []
        if len(star_atoms) >= 2:
            # Connect all pairs of star atoms to create periodic boundary
            for i in range(len(star_atoms)):
                for j in range(i + 1, len(star_atoms)):
                    s1, s2 = star_atoms[i], star_atoms[j]
                    periodic_edges.append([s1, s2])
                    periodic_edges.append([s2, s1])
                    # Mark as single bond + in ring (periodic indicator)
                    pe_feat = np.zeros(NUM_EDGE_FEATURES)
                    pe_feat[0] = 1.0  # single bond type
                    pe_feat[-1] = 1.0  # in ring (periodic)
                    periodic_edge_feats.append(pe_feat)
                    periodic_edge_feats.append(pe_feat)

        if periodic_edges:
            periodic_edge_index = np.array(periodic_edges, dtype=np.int64).T
            periodic_edge_features = np.stack(periodic_edge_feats)
            edge_index = np.concatenate([edge_index, periodic_edge_index], axis=1)
            edge_feats = np.concatenate([edge_feats, periodic_edge_features], axis=0)

    return {
        'atom_features': atom_feats,
        'edge_index': edge_index,
        'edge_features': edge_feats,
        'tg': tg,
        'p_smiles': p_smiles,
        'n_atoms': atom_feats.shape[0],
        'n_edges': edge_index.shape[1],
    }


class PPFNv2Dataset(Dataset):
    """Simplified dataset for PPFN-v2."""

    def __init__(self, csv_path: str, split: str = 'train',
                 train_ratio: float = 0.72, val_ratio: float = 0.08,
                 seed: int = 42, use_periodic_edge: bool = True):
        self.split = split
        self.use_periodic_edge = use_periodic_edge

        df = pd.read_csv(csv_path)

        # Deterministic split
        np.random.seed(seed)
        n = len(df)
        indices = np.random.permutation(n)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if split == 'train':
            sel_indices = indices[:n_train]
        elif split == 'val':
            sel_indices = indices[n_train:n_train + n_val]
        else:  # test
            sel_indices = indices[n_train + n_val:]

        df_split = df.iloc[sel_indices].reset_index(drop=True)

        # Process all polymers
        self.samples = []
        self.failed = 0
        print(f"Processing {len(df_split)} {split} samples (periodic_edge={use_periodic_edge})...")

        for idx, row in df_split.iterrows():
            result = process_polymer_v2(row['Smiles'], row['Tg'], use_periodic_edge)
            if result is not None:
                self.samples.append(result)
            else:
                self.failed += 1

        print(f"  Success: {len(self.samples)}, Failed: {self.failed}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'atom_features': torch.FloatTensor(sample['atom_features']),
            'edge_index': torch.LongTensor(sample['edge_index']),
            'edge_features': torch.FloatTensor(sample['edge_features']),
            'tg': torch.FloatTensor([sample['tg']]),
            'p_smiles': sample['p_smiles'],
            'n_atoms': sample['n_atoms'],
        }


def collate_fn_v2(batch):
    """Custom collate function for PPFN-v2 batching."""
    atom_features_list = []
    edge_index_list = []
    edge_features_list = []
    tg_list = []
    batch_atom_indices = []
    p_smiles_list = []
    n_atoms_offset = 0

    for i, sample in enumerate(batch):
        n_atoms = sample['n_atoms']
        atom_features_list.append(sample['atom_features'])
        edge_index_list.append(sample['edge_index'] + n_atoms_offset)
        edge_features_list.append(sample['edge_features'])
        tg_list.append(sample['tg'])
        batch_atom_indices.append(torch.full((n_atoms,), i, dtype=torch.long))
        p_smiles_list.append(sample['p_smiles'])
        n_atoms_offset += n_atoms

    return {
        'atom_features': torch.cat(atom_features_list, dim=0),
        'edge_index': torch.cat(edge_index_list, dim=1),
        'edge_features': torch.cat(edge_features_list, dim=0),
        'tg': torch.cat(tg_list, dim=0),
        'batch_atom': torch.cat(batch_atom_indices, dim=0),
        'p_smiles_list': p_smiles_list,
    }


if __name__ == '__main__':
    # Quick test
    csv_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    dataset = PPFNv2Dataset(csv_path, split='train')
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Atoms: {sample['n_atoms']}")
        print(f"  Edges: {sample['edge_index'].shape[1]}")
        print(f"  Tg: {sample['tg'].item()}")

    # Test collation
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_v2)
    batch = next(iter(loader))
    print(f"\nBatch:")
    print(f"  Total atoms: {batch['atom_features'].shape[0]}")
    print(f"  Total edges: {batch['edge_index'].shape[1]}")
    print(f"  Batch size: {batch['tg'].shape[0]}")
