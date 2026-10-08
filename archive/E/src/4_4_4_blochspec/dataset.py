"""
Dataset module for BlochSpec Iteration 2.
Loads trimer graphs + pre-computed spectral features.
Reuses spectral data from Iteration 1.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import Subset

import sys
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/1_day1_arch_improve')
from transform_utils import smiles_to_multimer
from feature_utils import mol_to_graph_data

import logging
logger = logging.getLogger(__name__)


class BlochSpecDataset(InMemoryDataset):
    """
    Tg dataset with trimer graphs and Bloch spectral features.
    """

    def __init__(self, root, csv_path, spectral_features, spectral_indices,
                 transform=None, pre_transform=None, pre_filter=None):
        self.csv_path = csv_path
        self._spectral_features = spectral_features
        self._spectral_indices = spectral_indices
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['Tg.csv']

    @property
    def processed_file_names(self):
        return ['blochspec_trimer_v2.pt']

    def download(self):
        pass

    def process(self):
        spectral_features = self._spectral_features
        spectral_indices = self._spectral_indices

        idx_to_feat = {}
        for fi, orig_idx in enumerate(spectral_indices):
            idx_to_feat[int(orig_idx)] = fi

        df = pd.read_csv(self.csv_path)
        smiles_list = df['Smiles'].tolist()
        tg_values = df['Tg'].tolist()

        data_list = []
        n_failed = 0

        for i, (smi, tg) in enumerate(zip(smiles_list, tg_values)):
            if i not in idx_to_feat:
                n_failed += 1
                continue

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

        logger.info(f"BlochSpec v2 dataset: {len(data_list)} graphs, {n_failed} failed")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SpectralOnlyDataset(torch.utils.data.Dataset):
    """Dataset that returns only spectral features (no GNN graph)."""

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
    Same split strategy as Day 1 baseline.
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

    spectral_features = np.load(os.path.join(spectral_dir, 'spectral_features.npy'))
    spectral_indices = np.load(os.path.join(spectral_dir, 'feature_indices.npy'))

    dataset = BlochSpecDataset(
        root=processed_dir,
        csv_path=csv_path,
        spectral_features=spectral_features,
        spectral_indices=spectral_indices,
    )

    return dataset
