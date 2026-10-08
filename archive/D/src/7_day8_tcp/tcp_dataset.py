"""
Dataset utilities for TCP experiment.
Reuses graph construction from reperioGT dataset.py
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple, List
from sklearn.model_selection import KFold
import sys
sys.path.append('/home/home_public/aisci/AIscientist/src/1_day2_reperioGT')
from dataset import (
    mol_to_graph, get_feature_dimensions,
    compute_rotatable_bond_fraction, count_wildcard_atoms
)


def collate_multi_target(batch):
    """
    Custom collate function to properly batch multi_target tensors.
    Ensures multi_target is stacked as [batch_size, 3] instead of being concatenated.
    """
    # Extract multi_target from each data object before batching
    multi_targets = []
    for data in batch:
        if hasattr(data, 'multi_target'):
            multi_targets.append(data.multi_target)
            # Remove temporarily to prevent PyG from messing with it
            del data.multi_target

    # Use PyG's default batching for everything else
    batched_data = Batch.from_data_list(batch)

    # Stack multi_targets properly as [batch_size, 3]
    if multi_targets:
        batched_data.multi_target = torch.stack(multi_targets, dim=0)

    return batched_data


class PolymerMultiDataset(Dataset):
    """Dataset for multi-property prediction (Tg, Tm, Egc)."""

    def __init__(self, data_list: List[Data], targets: np.ndarray = None):
        """
        Args:
            data_list: List of PyG Data objects
            targets: Array of shape (N, 3) for [Tg, Tm, Egc] or None
        """
        self.data_list = data_list
        self.targets = targets

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.targets is not None:
            # Add multi-property targets
            data.multi_target = torch.tensor(self.targets[idx], dtype=torch.float)
        return data


def load_pi1m_data(csv_path: str, max_samples: int = None) -> List[Data]:
    """Load PI1M unlabeled polymer data."""
    print(f"Loading PI1M from {csv_path}...")
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)

    print(f"Processing {len(df)} polymers...")

    data_list = []
    valid_smiles = []

    for idx, row in df.iterrows():
        # Handle both 'SMILES' and 'Smiles' column names
        smiles = row.get('SMILES', row.get('Smiles', None))
        if smiles is None:
            continue
        data = mol_to_graph(smiles, target=None)
        if data is not None:
            data_list.append(data)
            valid_smiles.append(smiles)

        if (idx + 1) % 100000 == 0:
            print(f"  Processed {idx + 1}/{len(df)}...")

    print(f"Successfully loaded {len(data_list)} valid polymers")
    return data_list


def load_tg_data(csv_path: str) -> Tuple[List[Data], np.ndarray]:
    """Load Tg labeled data."""
    print(f"Loading Tg data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} samples")

    data_list = []
    targets = []

    for idx, row in df.iterrows():
        data = mol_to_graph(row['Smiles'], row['Tg'])
        if data is not None:
            data_list.append(data)
            targets.append([row['Tg']])

    print(f"Successfully loaded {len(data_list)} valid samples")
    return data_list, np.array(targets, dtype=np.float32)


def create_k_fold_splits(data_list: List[Data], targets: np.ndarray,
                        n_splits: int = 5, seed: int = 42):
    """Create K-fold train/val splits."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(data_list)):
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        train_targets = targets[train_idx]
        val_targets = targets[val_idx]
        splits.append((train_data, val_data, train_targets, val_targets))

    return splits


if __name__ == "__main__":
    # Test data loading
    print("Testing PI1M loading...")
    pi1m_data = load_pi1m_data(
        '/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv',
        max_samples=1000
    )
    print(f"Loaded {len(pi1m_data)} samples")

    print("\nTesting Tg data loading...")
    tg_data, tg_targets = load_tg_data(
        '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    )
    print(f"Loaded {len(tg_data)} samples with targets shape {tg_targets.shape}")

    # Test feature dimensions
    atom_dim, bond_dim = get_feature_dimensions()
    print(f"\nFeature dimensions: atom={atom_dim}, bond={bond_dim}")
