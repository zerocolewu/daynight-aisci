"""
Dataset module for PerioGT Ablation Study

Handles polymer SMILES with periodic edge construction.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem
import torch_geometric
from torch_geometric.data import Data
from typing import Optional, Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Atom feature dimensions (matching Chemprop-style features)
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),  # 118 elements
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
    'num_hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

# Bond feature dimensions
BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'is_conjugated': [False, True],
    'is_in_ring': [False, True],
}


def one_hot_encoding(value, choices: list) -> List[int]:
    """One-hot encoding with unknown handling."""
    encoding = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1  # Unknown
    return encoding


def get_atom_features(atom) -> List[float]:
    """Extract features for a single atom."""
    features = []
    features.extend(one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']))
    features.extend(one_hot_encoding(atom.GetDegree(), ATOM_FEATURES['degree']))
    features.extend(one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']))
    features.extend(one_hot_encoding(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs']))
    features.extend(one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES['hybridization']))
    features.extend(one_hot_encoding(atom.GetIsAromatic(), ATOM_FEATURES['is_aromatic']))
    features.extend(one_hot_encoding(atom.IsInRing(), ATOM_FEATURES['is_in_ring']))
    return features


def get_bond_features(bond) -> List[float]:
    """Extract features for a single bond."""
    features = []
    features.extend(one_hot_encoding(bond.GetBondType(), BOND_FEATURES['bond_type']))
    features.extend(one_hot_encoding(bond.GetIsConjugated(), BOND_FEATURES['is_conjugated']))
    features.extend(one_hot_encoding(bond.IsInRing(), BOND_FEATURES['is_in_ring']))
    return features


def get_feature_dimensions():
    """Return dimensions of atom and bond features."""
    atom_dim = sum(len(v) + 1 for v in ATOM_FEATURES.values())
    bond_dim = sum(len(v) + 1 for v in BOND_FEATURES.values())
    return atom_dim, bond_dim


def find_wildcard_atoms(mol) -> List[int]:
    """Find indices of wildcard (*) atoms in the molecule."""
    wildcard_indices = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:  # Wildcard atom
            wildcard_indices.append(i)
    return wildcard_indices


def mol_to_periodic_graph(
    smiles: str,
    target: float = None,
    add_periodic_edge: bool = True,
) -> Optional[Data]:
    """
    Convert polymer SMILES to periodic polymer graph.

    Key feature: Adds edge between terminal atoms (wildcards) to create periodicity.

    Args:
        smiles: Polymer SMILES with * notation (e.g., *CCO*)
        target: Target property value
        add_periodic_edge: Whether to add periodic edge between wildcards

    Returns:
        PyG Data object or None if parsing fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None

        # Sanitize allowing wildcards
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                        Chem.SanitizeFlags.SANITIZE_ADJUSTHS)

        # Get wildcard atoms (periodic connection points)
        wildcard_indices = find_wildcard_atoms(mol)

        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(get_atom_features(atom))
        x = torch.tensor(atom_features, dtype=torch.float)

        # Bond features and edge index
        edge_index = []
        edge_attr = []

        # Add regular bonds
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bf = get_bond_features(bond)

            # Bidirectional edges
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append(bf)
            edge_attr.append(bf)

        # Add periodic edge if exactly 2 wildcards exist
        # This is the key innovation from the paper
        periodic_edge_added = False
        if add_periodic_edge and len(wildcard_indices) == 2:
            # Get the atoms connected to wildcards
            # We want to connect the NON-wildcard neighbors of each wildcard
            wc1, wc2 = wildcard_indices

            # Find neighbors of each wildcard
            wc1_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(wc1).GetNeighbors()]
            wc2_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(wc2).GetNeighbors()]

            # If wildcards have neighbors, connect those neighbors (more meaningful)
            # Otherwise, connect wildcards directly
            if wc1_neighbors and wc2_neighbors:
                # Connect the actual polymer backbone atoms
                n1, n2 = wc1_neighbors[0], wc2_neighbors[0]
            else:
                n1, n2 = wc1, wc2

            # Use same bond features as a single bond for periodic edge
            periodic_bond_features = get_bond_features_for_periodic()

            # Add bidirectional periodic edge
            edge_index.append([n1, n2])
            edge_index.append([n2, n1])
            edge_attr.append(periodic_bond_features)
            edge_attr.append(periodic_bond_features)
            periodic_edge_added = True

        if len(edge_index) == 0:
            return None

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Periodic mask (1 for atoms involved in periodic connection)
        periodic_mask = torch.zeros(len(atom_features), 1, dtype=torch.float)
        if len(wildcard_indices) == 2:
            for wc_idx in wildcard_indices:
                periodic_mask[wc_idx] = 1.0

        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([target], dtype=torch.float) if target is not None else None,
            periodic_mask=periodic_mask,
            periodic_vectors=torch.zeros(len(atom_features), 3, dtype=torch.float),
            smiles=smiles,
            num_wildcards=len(wildcard_indices),
            has_periodic_edge=periodic_edge_added,
        )

        return data

    except Exception as e:
        return None


def get_bond_features_for_periodic() -> List[float]:
    """Get bond features for the periodic edge (treated as single bond)."""
    features = []
    # Bond type: SINGLE, DOUBLE, TRIPLE, AROMATIC, unknown
    # one_hot_encoding adds 1 for unknown
    features.extend([1, 0, 0, 0, 0])  # bond_type: SINGLE + unknown slot
    # is_conjugated: False, True, unknown
    features.extend([1, 0, 0])  # not conjugated
    # is_in_ring: False, True, unknown
    features.extend([1, 0, 0])  # not in ring
    return features


class PolymerDataset(Dataset):
    """Dataset for polymer property prediction."""

    def __init__(self, data_list: List[Data]):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def load_and_split_data(
    csv_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    add_periodic_edge: bool = True,
    normalize_targets: bool = False,
) -> Tuple[PolymerDataset, PolymerDataset, PolymerDataset, pd.DataFrame, Optional[Dict]]:
    """
    Load data and split into train/val/test sets.

    Args:
        csv_path: Path to CSV file with 'Smiles' and 'Tg' columns
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        add_periodic_edge: Whether to add periodic edges to graphs
        normalize_targets: Whether to z-score normalize targets

    Returns:
        train_dataset, val_dataset, test_dataset, dataframe, normalization_params
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    # Convert to graphs
    data_list = []
    valid_indices = []
    for idx, row in df.iterrows():
        data = mol_to_periodic_graph(
            row['Smiles'],
            row['Tg'],
            add_periodic_edge=add_periodic_edge
        )
        if data is not None:
            data_list.append(data)
            valid_indices.append(idx)

    print(f"Successfully parsed {len(data_list)} molecules")

    # Filter dataframe
    df_valid = df.iloc[valid_indices].reset_index(drop=True)

    # Collect targets for normalization
    targets = torch.tensor([d.y.item() for d in data_list])
    norm_params = None

    if normalize_targets:
        mean = targets.mean()
        std = targets.std()
        norm_params = {'mean': mean.item(), 'std': std.item()}
        for data in data_list:
            data.y = (data.y - mean) / std
        print(f"Normalized targets: mean={mean:.2f}, std={std:.2f}")

    # Split indices
    n = len(data_list)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Create datasets
    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    test_data = [data_list[i] for i in test_indices]

    print(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    return (
        PolymerDataset(train_data),
        PolymerDataset(val_data),
        PolymerDataset(test_data),
        df_valid,
        norm_params
    )


def denormalize_predictions(predictions: np.ndarray, norm_params: Dict) -> np.ndarray:
    """Denormalize predictions if normalization was used."""
    if norm_params is None:
        return predictions
    return predictions * norm_params['std'] + norm_params['mean']


if __name__ == "__main__":
    # Test dataset loading
    csv_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    train_ds, val_ds, test_ds, df, norm_params = load_and_split_data(
        csv_path,
        add_periodic_edge=True
    )

    # Test a sample
    sample = train_ds[0]
    print(f"\nSample graph:")
    print(f"  Nodes: {sample.x.shape}")
    print(f"  Edges: {sample.edge_index.shape}")
    print(f"  Edge features: {sample.edge_attr.shape}")
    print(f"  Target Tg: {sample.y.item():.2f}")
    print(f"  Has periodic edge: {sample.has_periodic_edge}")
    print(f"  Num wildcards: {sample.num_wildcards}")

    # Feature dimensions
    atom_dim, bond_dim = get_feature_dimensions()
    print(f"\nFeature dimensions:")
    print(f"  Atom features: {atom_dim}")
    print(f"  Bond features: {bond_dim}")
