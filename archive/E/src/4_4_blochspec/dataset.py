"""
Dataset module for BlochSpec experiment.
Loads trimer graphs + pre-computed spectral features.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import Subset

# Import from the Day 1 baseline (reuse transform and feature utils)
import sys
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/1_day1_arch_improve')
from transform_utils import smiles_to_multimer
from feature_utils import mol_to_graph_data, ATOM_FEATURE_DIM, BASE_BOND_FEATURE_DIM

import logging
logger = logging.getLogger(__name__)


class BlochSpecDataset(InMemoryDataset):
    """
    Tg dataset with trimer graphs and Bloch spectral features.

    Each data object has:
        x, edge_index, edge_attr, y (from trimer graph)
        spectral_features: pre-computed Bloch features (12-dim)
    """

    def __init__(self, root, csv_path, spectral_features_path, spectral_indices_path,
                 transform=None, pre_transform=None, pre_filter=None):
        self.csv_path = csv_path
        self.spectral_features_path = spectral_features_path
        self.spectral_indices_path = spectral_indices_path
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['Tg.csv']

    @property
    def processed_file_names(self):
        return ['blochspec_trimer.pt']

    def download(self):
        pass

    def process(self):
        # Load spectral features
        spectral_features = np.load(self.spectral_features_path)
        spectral_indices = np.load(self.spectral_indices_path)

        # Create mapping from original index to spectral feature index
        idx_to_feat = {}
        for fi, orig_idx in enumerate(spectral_indices):
            idx_to_feat[orig_idx] = fi

        df = pd.read_csv(self.csv_path)
        smiles_list = df['Smiles'].tolist()
        tg_values = df['Tg'].tolist()

        data_list = []
        n_failed = 0

        for i, (smi, tg) in enumerate(zip(smiles_list, tg_values)):
            if i not in idx_to_feat:
                n_failed += 1
                continue

            # Generate trimer
            mol, head, tail, unit_boundaries = smiles_to_multimer(smi, n_units=3)
            if mol is None:
                n_failed += 1
                continue

            graph_data = mol_to_graph_data(mol, tg, add_virtual_node=False)
            if graph_data is None:
                n_failed += 1
                continue

            feat_idx = idx_to_feat[i]
            spectral_feat = spectral_features[feat_idx]

            data = Data(
                x=graph_data['x'],
                edge_index=graph_data['edge_index'],
                edge_attr=graph_data['edge_attr'],
                y=graph_data['y'],
                num_nodes=graph_data['num_nodes'],
                spectral_features=torch.tensor(spectral_feat, dtype=torch.float),
            )
            data_list.append(data)

        logger.info(f"BlochSpec dataset: {len(data_list)} graphs, {n_failed} failed")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SpectralOnlyDataset(torch.utils.data.Dataset):
    """
    Dataset that returns only spectral features (no GNN graph).
    For Pure Spectral MLP ablation.
    """

    def __init__(self, spectral_features, tgs):
        self.features = torch.tensor(spectral_features, dtype=torch.float)
        self.tgs = torch.tensor(tgs, dtype=torch.float).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.tgs[idx]


def get_data_splits(dataset_len, seed=71, train_ratio=0.72, val_ratio=0.08):
    """
    Split dataset by index with fixed seed.
    Returns train, val, test index lists.
    Same split as Day 1 baseline.
    """
    n = dataset_len
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, n))

    return train_indices, val_indices, test_indices


def prepare_dataset(csv_path, spectral_dir, processed_dir):
    """Prepare the BlochSpec dataset."""
    os.makedirs(processed_dir, exist_ok=True)

    spectral_features_path = os.path.join(spectral_dir, 'spectral_features.npy')
    spectral_indices_path = os.path.join(spectral_dir, 'feature_indices.npy')

    dataset = BlochSpecDataset(
        root=processed_dir,
        csv_path=csv_path,
        spectral_features_path=spectral_features_path,
        spectral_indices_path=spectral_indices_path,
    )

    return dataset


if __name__ == "__main__":
    CSV_PATH = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
    SPECTRAL_DIR = "/home/home_public/aisci/AIscientist/src/4_4_blochspec/spectral_data"
    PROCESSED_DIR = "/home/home_public/aisci/AIscientist/src/4_4_blochspec/data"

    dataset = prepare_dataset(CSV_PATH, SPECTRAL_DIR, PROCESSED_DIR)
    print(f"Dataset: {len(dataset)} graphs")
    print(f"Node features: {dataset.num_node_features}")
    print(f"Edge features: {dataset.num_edge_features}")

    sample = dataset[0]
    print(f"Sample spectral features: {sample.spectral_features}")
    print(f"Sample Tg: {sample.y}")
