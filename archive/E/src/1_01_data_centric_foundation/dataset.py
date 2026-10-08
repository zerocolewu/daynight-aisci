"""
Custom PyG Dataset for Tg prediction with various structural transformations.
"""

import torch
import pandas as pd
import numpy as np
import logging
from torch_geometric.data import Data, Dataset, InMemoryDataset

from transform_utils import (
    smiles_to_monomer,
    smiles_to_multimer,
    smiles_to_cyclized_trimer,
    get_graph_stats,
)
from feature_utils import mol_to_graph_data

logger = logging.getLogger(__name__)


class TgDataset(InMemoryDataset):
    """
    Tg dataset with configurable structural transformations.

    Modes:
        'monomer'  - Remove star atoms, single repeat unit (baseline)
        'dimer'    - 2 repeat units connected
        'trimer'   - 3 repeat units connected
        'cyclized' - 3 repeat units with periodic edge
        'augmented' - 3 repeat units with repeat_unit_index feature
    """

    def __init__(self, root, csv_path, mode='monomer', transform=None,
                 pre_transform=None, pre_filter=None):
        self.mode = mode
        self.csv_path = csv_path
        self.failed_indices = []
        super(TgDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['Tg.csv']

    @property
    def processed_file_names(self):
        return [f'tg_{self.mode}.pt']

    def download(self):
        pass  # Data is already available locally

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
        if self.mode == 'monomer':
            mol, head, tail = smiles_to_monomer(smi)
            if mol is None:
                return None
            return mol_to_graph_data(mol, tg)

        elif self.mode == 'dimer':
            mol, head, tail, unit_boundaries = smiles_to_multimer(smi, n_units=2)
            if mol is None:
                return None
            return mol_to_graph_data(mol, tg)

        elif self.mode == 'trimer':
            mol, head, tail, unit_boundaries = smiles_to_multimer(smi, n_units=3)
            if mol is None:
                return None
            return mol_to_graph_data(mol, tg)

        elif self.mode == 'cyclized':
            mol, periodic_bond_atoms, unit_boundaries = smiles_to_cyclized_trimer(smi)
            if mol is None:
                return None
            return mol_to_graph_data(mol, tg, periodic_bond_atoms=periodic_bond_atoms)

        elif self.mode == 'augmented':
            mol, head, tail, unit_boundaries = smiles_to_multimer(smi, n_units=3)
            if mol is None:
                return None
            # Create repeat unit index for each atom
            n_atoms = mol.GetNumAtoms()
            ru_indices = [0] * n_atoms
            for unit_idx, (start, end) in enumerate(unit_boundaries):
                for atom_idx in range(start, end):
                    ru_indices[atom_idx] = unit_idx
            return mol_to_graph_data(mol, tg, repeat_unit_indices=ru_indices)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def get_data_splits(dataset, seed=71, train_ratio=0.72, val_ratio=0.08):
    """
    Split dataset by index (not random) with fixed seed.
    Returns train, val, test DataLoaders.
    """
    n = len(dataset)
    indices = list(range(n))

    # Deterministic split by index
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # n_test = n - n_train - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    return train_indices, val_indices, test_indices


def compute_graph_stats(dataset, n_samples=100):
    """Compute average graph statistics for a dataset."""
    n_nodes = []
    n_edges = []
    n = min(n_samples, len(dataset))
    for i in range(n):
        data = dataset[i]
        n_nodes.append(data.num_nodes)
        n_edges.append(data.edge_index.size(1) // 2)  # undirected

    return {
        'avg_nodes': np.mean(n_nodes),
        'avg_edges': np.mean(n_edges),
        'max_nodes': max(n_nodes),
        'min_nodes': min(n_nodes),
    }
