"""
Improved data preprocessing utilities for GRIN experiment on Tg dataset.
Key improvements:
1. Rich atom and bond features
2. Proper p-SMILES augmentation using RDKit
3. Target normalization
"""

import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import re
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Atom feature dimensions
ATOM_FEATURE_DIM = 119  # Atomic number (1-118) + unknown
DEGREE_DIM = 11  # Degree 0-10
HYBRIDIZATION_DIM = 6  # SP, SP2, SP3, SP3D, SP3D2, other
FORMAL_CHARGE_DIM = 5  # -2, -1, 0, 1, 2 (binned)
NUM_HS_DIM = 5  # Number of Hs 0-4
IS_AROMATIC_DIM = 1
IS_IN_RING_DIM = 1

TOTAL_ATOM_FEATURES = (ATOM_FEATURE_DIM + DEGREE_DIM + HYBRIDIZATION_DIM +
                        FORMAL_CHARGE_DIM + NUM_HS_DIM + IS_AROMATIC_DIM + IS_IN_RING_DIM)

# Bond feature dimensions
BOND_TYPE_DIM = 4  # Single, double, triple, aromatic
IS_CONJUGATED_DIM = 1
IS_IN_RING_DIM_BOND = 1

TOTAL_BOND_FEATURES = BOND_TYPE_DIM + IS_CONJUGATED_DIM + IS_IN_RING_DIM_BOND


def get_atom_features(atom):
    """Get rich atom features."""
    features = []

    # Atomic number (one-hot)
    atomic_num = atom.GetAtomicNum()
    atomic_num_vec = [0] * ATOM_FEATURE_DIM
    if 1 <= atomic_num <= 118:
        atomic_num_vec[atomic_num - 1] = 1
    else:
        atomic_num_vec[-1] = 1  # Unknown
    features.extend(atomic_num_vec)

    # Degree (one-hot)
    degree = atom.GetDegree()
    degree_vec = [0] * DEGREE_DIM
    if degree < DEGREE_DIM:
        degree_vec[degree] = 1
    features.extend(degree_vec)

    # Hybridization (one-hot)
    hybridization = atom.GetHybridization()
    hybridization_vec = [0] * HYBRIDIZATION_DIM
    hyb_map = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        Chem.rdchem.HybridizationType.SP3D: 3,
        Chem.rdchem.HybridizationType.SP3D2: 4,
    }
    hyb_idx = hyb_map.get(hybridization, 5)
    hybridization_vec[hyb_idx] = 1
    features.extend(hybridization_vec)

    # Formal charge (binned, one-hot)
    formal_charge = atom.GetFormalCharge()
    charge_vec = [0] * FORMAL_CHARGE_DIM
    if -2 <= formal_charge <= 2:
        charge_vec[formal_charge + 2] = 1
    features.extend(charge_vec)

    # Number of Hs (one-hot, capped at 4)
    num_hs = atom.GetTotalNumHs()
    hs_vec = [0] * NUM_HS_DIM
    if num_hs < NUM_HS_DIM:
        hs_vec[num_hs] = 1
    else:
        hs_vec[-1] = 1
    features.extend(hs_vec)

    # Is aromatic
    features.append(1 if atom.GetIsAromatic() else 0)

    # Is in ring
    features.append(1 if atom.IsInRing() else 0)

    return features


def get_bond_features(bond):
    """Get bond features."""
    features = []

    # Bond type (one-hot)
    bond_type = bond.GetBondType()
    bond_type_vec = [0] * BOND_TYPE_DIM
    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    bond_idx = bond_type_map.get(bond_type, 0)
    bond_type_vec[bond_idx] = 1
    features.extend(bond_type_vec)

    # Is conjugated
    features.append(1 if bond.GetIsConjugated() else 0)

    # Is in ring
    features.append(1 if bond.IsInRing() else 0)

    return features


def mol_to_graph(mol, target=None):
    """
    Convert RDKit molecule to PyTorch Geometric Data object.

    Args:
        mol: RDKit molecule
        target: Target value (optional)

    Returns:
        PyTorch Geometric Data object or None
    """
    if mol is None:
        return None

    try:
        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(get_atom_features(atom))

        x = torch.tensor(atom_features, dtype=torch.float)

        # Get bond information
        if mol.GetNumBonds() == 0:
            # Single atom molecule
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, TOTAL_BOND_FEATURES), dtype=torch.float)
        else:
            edge_indices = []
            edge_features = []

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_indices.append([i, j])
                edge_indices.append([j, i])

                bond_feat = get_bond_features(bond)
                edge_features.append(bond_feat)
                edge_features.append(bond_feat)

            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

        # Create data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        if target is not None:
            data.y = torch.tensor([target], dtype=torch.float)

        return data

    except Exception as e:
        print(f"Error converting molecule to graph: {e}")
        return None


def create_n_ru_molecule(p_smiles, n=3):
    """
    Create a molecule with n repeat units from p-SMILES.
    Uses RDKit to properly handle polymer chemistry.

    Strategy:
    - For n=1: return the original with [*] replaced by dummy atoms
    - For n>1: create a linear chain of n units by connecting at the [*] positions

    Args:
        p_smiles: p-SMILES string with [*] markers
        n: Number of repeat units

    Returns:
        RDKit molecule or None
    """
    try:
        if n == 1:
            # For 1-RU: replace [*] with dummy atoms to mark connection points
            # This preserves the monomer structure
            mol = Chem.MolFromSmiles(p_smiles, sanitize=False)
            if mol is None:
                return None

            # Replace wildcard atoms with hydrogen (simpler approach)
            # Edit: Actually keep the structure as is for 1-RU
            Chem.SanitizeMol(mol)
            return mol

        # For n>1: create connected chain
        # Strategy: find the atoms connected to [*] and create bonds between units

        # Parse the monomer
        mol_mono = Chem.MolFromSmiles(p_smiles, sanitize=False)
        if mol_mono is None:
            return None

        # Find wildcard atoms (atom maps or * atoms)
        # In RDKit, * becomes a dummy atom with atomic number 0
        wildcard_atoms = []
        for atom in mol_mono.GetAtoms():
            if atom.GetAtomicNum() == 0:  # Wildcard atom
                wildcard_atoms.append(atom.GetIdx())

        if len(wildcard_atoms) == 0:
            # No polymer markers, just duplicate n times disconnected
            # Return the monomer as is
            Chem.SanitizeMol(mol_mono)
            return mol_mono

        if len(wildcard_atoms) < 2:
            # Only one connection point, can't form chain
            Chem.SanitizeMol(mol_mono)
            return mol_mono

        # Create n copies and connect them
        # Use RDKit's CombineMols and EditableMol
        from rdkit.Chem import rdchem

        # Start with the first unit
        result_mol = Chem.RWMol(mol_mono)

        # Track atoms that will be connected
        # For linear chain: unit1_wildcard2 -> unit2_wildcard1, unit2_wildcard2 -> unit3_wildcard1, etc.
        # The two wildcard atoms define the connection points

        # Get the neighbors of wildcard atoms (these are the atoms we'll actually bond)
        connection_atoms = []  # [(atom_idx, wildcard_idx), ...]
        for wc_idx in wildcard_atoms[:2]:  # Only use first two wildcards
            wc_atom = mol_mono.GetAtomWithIdx(wc_idx)
            neighbors = [n.GetIdx() for n in wc_atom.GetNeighbors()]
            if len(neighbors) > 0:
                connection_atoms.append((neighbors[0], wc_idx))

        if len(connection_atoms) < 2:
            Chem.SanitizeMol(mol_mono)
            return mol_mono

        # Create additional units
        mono_num_atoms = mol_mono.GetNumAtoms()
        for unit_idx in range(1, n):
            # Add another copy of the monomer
            new_mol = Chem.Mol(mol_mono)
            old_num_atoms = result_mol.GetNumAtoms()

            # Combine molecules
            combined = Chem.CombineMols(result_mol, new_mol)

            # Create editable molecule
            result_mol = Chem.RWMol(combined)

            # Connect: previous unit's second connection point to new unit's first connection point
            prev_conn_atom = old_num_atoms - mono_num_atoms + connection_atoms[1][0]
            new_conn_atom = old_num_atoms + connection_atoms[0][0]

            # Add bond between connection points
            result_mol.AddBond(prev_conn_atom, new_conn_atom, Chem.BondType.SINGLE)

        # Remove all wildcard atoms
        # Go in reverse order to maintain correct indices
        atoms_to_remove = []
        for atom in result_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atoms_to_remove.append(atom.GetIdx())

        for idx in sorted(atoms_to_remove, reverse=True):
            result_mol.RemoveAtom(idx)

        # Convert back to regular mol and sanitize
        final_mol = result_mol.GetMol()
        Chem.SanitizeMol(final_mol)

        return final_mol

    except Exception as e:
        print(f"Error creating {n}-RU molecule: {e}")
        return None


def process_psmiles_simple(smiles, n_repeats=1):
    """
    Simplified p-SMILES processing - just replace * with H for valid SMILES.
    For augmentation, we'll use create_n_ru_molecule instead.

    Args:
        smiles: p-SMILES string
        n_repeats: Number of repeat units (ignored in simple mode)

    Returns:
        Valid SMILES string
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None

        # Replace wildcard atoms with hydrogen
        rw_mol = Chem.RWMol(mol)
        atoms_to_replace = []

        for atom in rw_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atoms_to_replace.append(atom.GetIdx())

        # Replace wildcard atoms with hydrogens
        for idx in sorted(atoms_to_replace, reverse=True):
            rw_mol.GetAtomWithIdx(idx).SetAtomicNum(1)  # Replace with H

        final_mol = rw_mol.GetMol()
        Chem.SanitizeMol(final_mol)

        return Chem.MolToSmiles(final_mol)

    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None


class TgDataset(Dataset):
    """
    Improved Tg dataset with proper features and augmentation.
    """

    def __init__(self, root, csv_path, transform=None, pre_transform=None,
                 augmentation=None, split='train', target_scaler=None):
        """
        Args:
            root: Root directory
            csv_path: Path to Tg.csv
            transform: Data transform
            pre_transform: Pre-transform
            augmentation: None, '1ru', '3ru', or 'both'
            split: 'train', 'val', or 'test'
            target_scaler: StandardScaler for target normalization (fitted on train)
        """
        self.csv_path = csv_path
        self.augmentation = augmentation
        self.split = split
        self.target_scaler = target_scaler

        super().__init__(root, transform, pre_transform)

        # Load and process data
        self._load_data()

    def _load_data(self):
        """Load and process the CSV data."""
        df = pd.read_csv(self.csv_path)

        # Filter out invalid entries
        df = df.dropna(subset=['Smiles', 'Tg'])
        df = df[df['Tg'].notna()]

        print(f"Total samples in dataset: {len(df)}")

        # Split into train/val/test (80/10/10)
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        if self.split == 'train':
            self.data_df = train_df.reset_index(drop=True)
        elif self.split == 'val':
            self.data_df = val_df.reset_index(drop=True)
        else:
            self.data_df = test_df.reset_index(drop=True)

        print(f"Split {self.split}: {len(self.data_df)} samples")

        # Fit target scaler on training data if this is training split
        if self.split == 'train' and self.target_scaler is None:
            self.target_scaler = StandardScaler()
            self.target_scaler.fit(self.data_df[['Tg']])
            print(f"Target scaler fitted: mean={self.target_scaler.mean_[0]:.2f}, std={self.target_scaler.scale_[0]:.2f}")

        # Apply augmentation if specified
        if self.augmentation == 'both' and self.split == 'train':
            # Create both 1-RU and 3-RU versions
            augmented_rows = []

            for idx, row in self.data_df.iterrows():
                # Add 1-RU version
                row_1ru = row.copy()
                row_1ru['n_repeats'] = 1
                augmented_rows.append(row_1ru)

                # Add 3-RU version
                row_3ru = row.copy()
                row_3ru['n_repeats'] = 3
                augmented_rows.append(row_3ru)

            self.data_df = pd.DataFrame(augmented_rows).reset_index(drop=True)
            print(f"After {self.augmentation} augmentation: {len(self.data_df)} samples")

        elif self.augmentation == '3ru':
            self.data_df['n_repeats'] = 3
        else:
            self.data_df['n_repeats'] = 1

        # Convert to graph data
        self.graphs = []
        self.targets = []
        failed = 0

        for idx, row in self.data_df.iterrows():
            smiles = row['Smiles']
            tg = row['Tg']
            n_repeats = int(row['n_repeats'])

            # Create molecule based on augmentation
            if n_repeats == 3:
                mol = create_n_ru_molecule(smiles, n=3)
            else:
                mol = create_n_ru_molecule(smiles, n=1)

            if mol is None:
                failed += 1
                continue

            # Normalize target
            if self.target_scaler is not None:
                tg_normalized = self.target_scaler.transform([[tg]])[0, 0]
            else:
                tg_normalized = tg

            # Convert to graph
            graph = mol_to_graph(mol, target=tg_normalized)

            if graph is not None:
                graph.y_original = torch.tensor([tg], dtype=torch.float)
                graph.smiles = smiles
                graph.n_repeats = n_repeats
                self.graphs.append(graph)
                self.targets.append(tg)
            else:
                failed += 1

        print(f"Successfully loaded {len(self.graphs)} graphs, {failed} failed")

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
        train_loader, val_loader, test_loader, target_scaler
    """
    from torch_geometric.loader import DataLoader

    # First create training dataset to fit the scaler
    train_dataset = TgDataset(
        root='/tmp/tg_train_v2',
        csv_path=csv_path,
        augmentation=augmentation,
        split='train',
        target_scaler=None
    )

    # Get the fitted scaler
    target_scaler = train_dataset.target_scaler

    # Create val and test datasets with the same scaler
    val_dataset = TgDataset(
        root='/tmp/tg_val_v2',
        csv_path=csv_path,
        augmentation=None,
        split='val',
        target_scaler=target_scaler
    )

    test_dataset = TgDataset(
        root='/tmp/tg_test_v2',
        csv_path=csv_path,
        augmentation=None,
        split='test',
        target_scaler=target_scaler
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, target_scaler


if __name__ == '__main__':
    # Test the data loading
    csv_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'

    print("="*80)
    print("Testing improved data loading without augmentation...")
    print("="*80)
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        csv_path, batch_size=32, augmentation=None
    )

    for batch in train_loader:
        print(f"Batch size: {batch.num_graphs}")
        print(f"Node features shape: {batch.x.shape}")
        print(f"Edge index shape: {batch.edge_index.shape}")
        print(f"Edge features shape: {batch.edge_attr.shape if batch.edge_attr is not None else 'None'}")
        print(f"Target shape: {batch.y.shape}")
        print(f"Sample target (normalized): {batch.y[0].item():.4f}")
        print(f"Sample target (original): {batch.y_original[0].item():.2f}")
        break

    print("\n" + "="*80)
    print("Testing data loading with {1,3}-RU augmentation...")
    print("="*80)
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        csv_path, batch_size=32, augmentation='both'
    )

    for batch in train_loader:
        print(f"Batch size: {batch.num_graphs}")
        print(f"Node features shape: {batch.x.shape}")
        print(f"Target shape: {batch.y.shape}")

        # Check n_repeats distribution
        n_repeats_list = []
        for i in range(batch.num_graphs):
            n_repeats_list.append(batch.n_repeats[i].item() if hasattr(batch, 'n_repeats') else 1)
        print(f"N_repeats distribution: 1-RU: {n_repeats_list.count(1)}, 3-RU: {n_repeats_list.count(3)}")
        break
