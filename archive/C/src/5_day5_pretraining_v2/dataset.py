"""
Dataset loading and feature extraction for polymer property prediction (Iteration v2).
Based on proven approach from previous iteration with optimizations.
"""
import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
import numpy as np
from typing import List, Tuple, Optional
import random

# Pre-computed lookup tables for speed
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B',
              'Se', 'Te', 'Ge', 'As', 'Sb', 'Bi', '*']
ATOM_TYPE_DICT = {t: i for i, t in enumerate(ATOM_TYPES)}
NUM_ATOM_TYPES = len(ATOM_TYPES)

HYBRIDIZATIONS = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
                  Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
                  Chem.HybridizationType.SP3D2]
HYBRID_DICT = {h: i for i, h in enumerate(HYBRIDIZATIONS)}
NUM_HYBRID = len(HYBRIDIZATIONS)

BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
              Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
BOND_TYPE_DICT = {bt: i for i, bt in enumerate(BOND_TYPES)}
NUM_BOND_TYPES = len(BOND_TYPES)

NUM_ATOM_FEATURES = NUM_ATOM_TYPES + 1 + NUM_HYBRID + 1 + 1 + 1  # 27
NUM_EDGE_FEATURES = NUM_BOND_TYPES + 1 + 1  # 6


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """
    Convert SMILES string to PyG Data object.
    No AddHs for speed. Returns None if parsing fails.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        atoms = mol.GetAtoms()
        num_atoms = len(atoms)

        if num_atoms == 0:
            return None

        # Atom features: [27]
        x = np.zeros((num_atoms, NUM_ATOM_FEATURES), dtype=np.float32)
        for i, atom in enumerate(atoms):
            # Atom type one-hot (18 types including *)
            symbol = atom.GetSymbol()
            if symbol in ATOM_TYPE_DICT:
                x[i, ATOM_TYPE_DICT[symbol]] = 1.0

            # Degree
            x[i, NUM_ATOM_TYPES] = float(atom.GetDegree())

            # Hybridization one-hot (5 types)
            hyb = atom.GetHybridization()
            if hyb in HYBRID_DICT:
                x[i, NUM_ATOM_TYPES + 1 + HYBRID_DICT[hyb]] = 1.0

            # Aromatic
            x[i, NUM_ATOM_TYPES + 1 + NUM_HYBRID] = float(atom.GetIsAromatic())

            # Formal charge
            x[i, NUM_ATOM_TYPES + 1 + NUM_HYBRID + 1] = float(atom.GetFormalCharge())

            # In ring
            x[i, NUM_ATOM_TYPES + 1 + NUM_HYBRID + 2] = float(atom.IsInRing())

        # Edge features
        bonds = mol.GetBonds()
        num_bonds = len(bonds)

        if num_bonds == 0:
            return Data(
                x=torch.from_numpy(x),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, NUM_EDGE_FEATURES), dtype=torch.float32)
            )

        edge_index = np.zeros((2, num_bonds * 2), dtype=np.int64)
        edge_attr = np.zeros((num_bonds * 2, NUM_EDGE_FEATURES), dtype=np.float32)

        for i, bond in enumerate(bonds):
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()

            bond_type = bond.GetBondType()
            bt_idx = BOND_TYPE_DICT.get(bond_type, 0)
            conj = float(bond.GetIsConjugated())
            ring = float(bond.IsInRing())

            feat = np.zeros(NUM_EDGE_FEATURES, dtype=np.float32)
            feat[bt_idx] = 1.0
            feat[NUM_BOND_TYPES] = conj
            feat[NUM_BOND_TYPES + 1] = ring

            edge_index[0, 2*i] = a1
            edge_index[1, 2*i] = a2
            edge_index[0, 2*i+1] = a2
            edge_index[1, 2*i+1] = a1
            edge_attr[2*i] = feat
            edge_attr[2*i+1] = feat

        return Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr)
        )

    except Exception:
        return None


class PolymerDataset(Dataset):
    """Dataset for polymer SMILES."""

    def __init__(self, smiles_list: List[str], labels=None, verbose=True):
        super().__init__(None, None, None)
        self.labels = labels
        self.graphs = []
        self.valid_indices = []

        if verbose:
            print(f"Processing {len(smiles_list)} SMILES...", end=" ", flush=True)

        failed = 0
        for i, smiles in enumerate(smiles_list):
            graph = smiles_to_graph(smiles)
            if graph is not None:
                self.graphs.append(graph)
                self.valid_indices.append(i)
            else:
                failed += 1
                self.graphs.append(None)

        if verbose:
            print(f"OK ({len(self.valid_indices)} valid, {failed} failed)")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        graph = self.graphs[idx]
        if graph is None:
            graph = Data(
                x=torch.zeros((1, NUM_ATOM_FEATURES), dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, NUM_EDGE_FEATURES), dtype=torch.float32)
            )
        if self.labels is not None:
            graph.y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return graph


def load_pretrain_dataset(file_path: str, max_samples: int = None, batch_size=256):
    """Load PI1M dataset for pretraining."""
    from torch_geometric.loader import DataLoader as PyGDataLoader

    print(f"Loading pretraining data from {file_path}...")
    df = pd.read_csv(file_path)

    if max_samples is not None and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    smiles_list = df['SMILES'].tolist()
    dataset = PolymerDataset(smiles_list)

    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"Pretraining dataset: {len(dataset)} graphs, {len(dataset.valid_indices)} valid")
    return loader, dataset


def load_finetune_dataset(file_path: str, train_ratio=0.72, val_ratio=0.08):
    """Load Tg dataset for fine-tuning with 72/8/20 split."""
    from torch_geometric.loader import DataLoader as PyGDataLoader

    print(f"Loading fine-tuning data from {file_path}...")
    df = pd.read_csv(file_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_df = df[:n_train]
    val_df = df[n_train:n_train + n_val]
    test_df = df[n_train + n_val:]

    print(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    train_dataset = PolymerDataset(train_df['Smiles'].tolist(), train_df['Tg'].values)
    val_dataset = PolymerDataset(val_df['Smiles'].tolist(), val_df['Tg'].values)
    test_dataset = PolymerDataset(test_df['Smiles'].tolist(), test_df['Tg'].values)

    train_loader = PyGDataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = PyGDataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = PyGDataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
