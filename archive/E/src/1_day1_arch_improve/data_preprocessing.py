"""
Data preprocessing for Day 1 Experiment 2.
Generates 4 conditions: Trimer, Cyclized Trimer, Trimer+VirtualNode, Cyclized+VirtualNode.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import logging
from torch_geometric.data import Data, InMemoryDataset

from transform_utils import smiles_to_multimer, smiles_to_cyclized_trimer
from feature_utils import mol_to_graph_data, ATOM_FEATURE_DIM, BASE_BOND_FEATURE_DIM

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Dataset modes for this experiment
MODES = ['trimer', 'cyclized', 'trimer_vn', 'cyclized_vn']


class TgDataset(InMemoryDataset):
    """
    Tg dataset with configurable structural transformations.

    Modes:
        'trimer'      - 3 repeat units connected (no virtual node)
        'cyclized'    - 3 repeat units with periodic edge (no virtual node)
        'trimer_vn'   - 3 repeat units + virtual node
        'cyclized_vn' - cyclized 3 units + virtual node
    """

    def __init__(self, root, csv_path, mode='trimer', transform=None,
                 pre_transform=None, pre_filter=None):
        self.mode = mode
        self.csv_path = csv_path
        self.failed_indices = []
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['Tg.csv']

    @property
    def processed_file_names(self):
        return [f'tg_{self.mode}.pt']

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.csv_path)
        smiles_list = df['Smiles'].tolist()
        tg_values = df['Tg'].tolist()

        data_list = []
        self.failed_indices = []

        for i, (smi, tg) in enumerate(zip(smiles_list, tg_values)):
            graph_data = self._process_molecule(smi, tg)

            if graph_data is not None:
                data = Data(
                    x=graph_data['x'],
                    edge_index=graph_data['edge_index'],
                    edge_attr=graph_data['edge_attr'],
                    y=graph_data['y'],
                    num_nodes=graph_data['num_nodes'],
                )
                data_list.append(data)
            else:
                self.failed_indices.append(i)

        if len(self.failed_indices) > 0:
            logger.warning(f"Mode '{self.mode}': {len(self.failed_indices)}/{len(smiles_list)} molecules failed conversion")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"Mode '{self.mode}': processed {len(data_list)} molecules successfully")

    def _process_molecule(self, smi, tg):
        """Process a single molecule based on the mode."""
        add_vn = self.mode.endswith('_vn')
        is_cyclized = self.mode.startswith('cyclized')

        if is_cyclized:
            mol, periodic_bond_atoms, unit_boundaries = smiles_to_cyclized_trimer(smi)
            if mol is None:
                return None
            return mol_to_graph_data(mol, tg, periodic_bond_atoms=periodic_bond_atoms,
                                     add_virtual_node=add_vn)
        else:
            mol, head, tail, unit_boundaries = smiles_to_multimer(smi, n_units=3)
            if mol is None:
                return None
            return mol_to_graph_data(mol, tg, add_virtual_node=add_vn)


def get_data_splits(dataset_len, seed=71, train_ratio=0.72, val_ratio=0.08):
    """
    Split dataset by index with fixed seed.
    Returns train, val, test index lists.
    """
    n = dataset_len
    indices = list(range(n))

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    return train_indices, val_indices, test_indices


def prepare_all_datasets(csv_path, processed_dir):
    """Prepare all 4 dataset modes."""
    datasets = {}
    for mode in MODES:
        logger.info(f"Processing mode: {mode}")
        dataset = TgDataset(
            root=processed_dir,
            csv_path=csv_path,
            mode=mode,
        )
        datasets[mode] = dataset
        logger.info(f"  {mode}: {len(dataset)} graphs, node_dim={dataset.num_node_features}, edge_dim={dataset.num_edge_features}")
    return datasets


if __name__ == "__main__":
    csv_path = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
    processed_dir = "./data"
    datasets = prepare_all_datasets(csv_path, processed_dir)
