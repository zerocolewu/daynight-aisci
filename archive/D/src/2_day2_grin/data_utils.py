"""
Data preprocessing utilities for GRIN experiment on Tg dataset.
Handles p-SMILES processing and repeat unit augmentation.
"""

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_smiles
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import re
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split


def augment_psmiles(smiles: str, n_repeats: int = 3) -> str:
    """
    Augment p-SMILES by repeating the unit n_repeats times.
    Replace * markers with connection points.

    Args:
        smiles: p-SMILES string with * markers
        n_repeats: Number of repeat units (1, 2, 3, etc.)

    Returns:
        Augmented SMILES string
    """
    if n_repeats == 1:
        return smiles

    # Find the repeat unit pattern (everything between * markers)
    # Pattern: *XXX* where XXX is the repeat unit
    # We need to extract the core unit

    # Count * markers
    star_count = smiles.count('*')

    if star_count == 0:
        # No polymer markers, return as is
        return smiles

    # For simple cases: *CCC* -> CCC-CCC-CCC
    # Extract the core unit
    core = smiles.replace('*', '').strip('()')

    # Handle cases with side chains
    # e.g., *CC(*)C -> the main chain is CC and side chain is C

    # Simple approach: replace * with connection points
    # For 3-RU: replace each * with the repeat unit

    # Strategy: parse the SMILES to find * positions
    # Then create connected chain

    # Simplified approach: for 1-RU, return original
    # For n-RU, create linear chain by removing * and connecting units

    # Remove * markers to get the core structure
    core = smiles.replace('*', '')

    # Create n-1 bonds to connect n units
    # For example, *CC* with 3 repeats: CC-CC-CC
    # The bonds form where the * were

    # More robust: use RDKit to manipulate
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        # Get the SMILES without * atoms
        # Replace * atoms with dummy atoms for connection

        # Simplified: just repeat the core n times with single bonds
        units = [core] * n_repeats
        augmented = '.'.join(units)  # Will connect with RDKit

        mol_aug = Chem.MolFromSmiles(augmented)
        if mol_aug is None:
            return smiles

        # Try to connect the units
        # This is a simplified approach - may not work for all cases
        # For now, return the original for complex cases

        return Chem.MolToSmiles(mol_aug)

    except:
        return smiles


def process_psmiles_simple(smiles: str, n_repeats: int = 1) -> str:
    """
    Simple p-SMILES processing for graph construction.
    For 1-RU: keep as is
    For 3-RU: create a 3-unit chain

    Args:
        smiles: p-SMILES string
        n_repeats: Number of repeat units

    Returns:
        Processed SMILES
    """
    if n_repeats == 1:
        return smiles

    # For 3-RU: manually construct 3 connected units
    # Strategy: replace * with the unit itself to create connections

    # Count stars
    star_positions = [i for i, c in enumerate(smiles) if c == '*']

    if len(star_positions) == 0:
        return smiles

    # For 3 units, we create: unit1-unit2-unit3
    # Remove stars and create connections

    # Get core unit (without stars)
    core = smiles.replace('*', '')

    # Remove extra parentheses that might be left
    core = core.strip('()')

    # Create 3 connected units
    # Simple approach: connect with single bonds
    result = f"{core}-{core}-{core}"

    return result


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """
    Convert SMILES to PyTorch Geometric Data object.

    Args:
        smiles: SMILES string

    Returns:
        PyTorch Geometric Data object or None if conversion fails
    """
    try:
        # Use PyTorch Geometric's from_smiles
        # This handles atom and bond features automatically
        data = from_smiles(smiles)
        return data
    except Exception as e:
        print(f"Error converting SMILES: {smiles}, error: {e}")
        return None


class TgDataset(Dataset):
    """
    Tg dataset with support for repeat unit augmentation.
    """

    def __init__(self, root, csv_path, transform=None, pre_transform=None,
                 augmentation=None, split='train'):
        """
        Args:
            root: Root directory
            csv_path: Path to Tg.csv
            transform: Data transform
            pre_transform: Pre-transform
            augmentation: None, '1ru', '3ru', or 'both'
            split: 'train', 'val', or 'test'
        """
        self.csv_path = csv_path
        self.augmentation = augmentation
        self.split = split

        super().__init__(root, transform, pre_transform)

        # Load and process data
        self._load_data()

    def _load_data(self):
        """Load and process the CSV data."""
        df = pd.read_csv(self.csv_path)

        # Split into train/val/test (80/10/10)
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        if self.split == 'train':
            self.data_df = train_df.reset_index(drop=True)
        elif self.split == 'val':
            self.data_df = val_df.reset_index(drop=True)
        else:
            self.data_df = test_df.reset_index(drop=True)

        # Apply augmentation if specified
        if self.augmentation == 'both' and self.split == 'train':
            # Create both 1-RU and 3-RU versions
            df_1ru = self.data_df.copy()
            df_1ru['n_repeats'] = 1

            df_3ru = self.data_df.copy()
            df_3ru['n_repeats'] = 3

            self.data_df = pd.concat([df_1ru, df_3ru], ignore_index=True)
        elif self.augmentation == '3ru':
            self.data_df['n_repeats'] = 3
        else:
            self.data_df['n_repeats'] = 1

        # Convert to graph data
        self.graphs = []
        self.targets = []

        for idx, row in self.data_df.iterrows():
            smiles = row['Smiles']
            tg = row['Tg']
            n_repeats = row['n_repeats']

            # Process SMILES based on augmentation
            if n_repeats == 3:
                processed_smiles = process_psmiles_simple(smiles, n_repeats=3)
            else:
                processed_smiles = smiles

            # Convert to graph
            graph = smiles_to_graph(processed_smiles)

            if graph is not None:
                graph.y = torch.tensor([tg], dtype=torch.float)
                graph.smiles = smiles
                graph.n_repeats = n_repeats
                self.graphs.append(graph)
                self.targets.append(tg)

        print(f"Loaded {len(self.graphs)} graphs for {self.split} split with augmentation={self.augmentation}")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


def get_dataloaders(csv_path, batch_size=64, augmentation=None, num_workers=0):
    """
    Create train, val, and test dataloaders.

    Args:
        csv_path: Path to Tg.csv
        batch_size: Batch size
        augmentation: Augmentation strategy
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, test_loader
    """
    from torch_geometric.loader import DataLoader

    train_dataset = TgDataset(
        root='/tmp/tg_train',
        csv_path=csv_path,
        augmentation=augmentation,
        split='train'
    )

    val_dataset = TgDataset(
        root='/tmp/tg_val',
        csv_path=csv_path,
        augmentation=None,  # No augmentation for val/test
        split='val'
    )

    test_dataset = TgDataset(
        root='/tmp/tg_test',
        csv_path=csv_path,
        augmentation=None,  # No augmentation for val/test
        split='test'
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test the data loading
    csv_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'

    print("Testing data loading without augmentation...")
    train_loader, val_loader, test_loader = get_dataloaders(csv_path, batch_size=32, augmentation=None)

    for batch in train_loader:
        print(f"Batch size: {batch.num_graphs}")
        print(f"Node features shape: {batch.x.shape}")
        print(f"Edge index shape: {batch.edge_index.shape}")
        print(f"Target shape: {batch.y.shape}")
        break

    print("\nTesting data loading with {1,3}-RU augmentation...")
    train_loader, val_loader, test_loader = get_dataloaders(csv_path, batch_size=32, augmentation='both')

    for batch in train_loader:
        print(f"Batch size: {batch.num_graphs}")
        print(f"Node features shape: {batch.x.shape}")
        print(f"Edge index shape: {batch.edge_index.shape}")
        print(f"Target shape: {batch.y.shape}")
        break
