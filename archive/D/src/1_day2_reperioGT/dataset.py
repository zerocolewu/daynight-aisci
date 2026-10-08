"""
Dataset module for RE-PerioGT: Polymer Tg prediction with rigidity-aware features
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem
import torch_geometric
from torch_geometric.data import Data
from typing import Optional, Tuple, List
from sklearn.model_selection import train_test_split


# Atom feature dimensions
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
    atom_dim = sum(len(v) + 1 for v in ATOM_FEATURES.values())  # +1 for unknown
    bond_dim = sum(len(v) + 1 for v in BOND_FEATURES.values())
    return atom_dim, bond_dim


def compute_rotatable_bond_fraction(mol) -> float:
    """Compute rotatable bond fraction from RDKit mol."""
    try:
        num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
        total_bonds = mol.GetNumBonds()
        if total_bonds == 0:
            return 0.0
        return num_rotatable / total_bonds
    except:
        return 0.0


def count_wildcard_atoms(mol) -> int:
    """Count wildcard atoms (*) in molecule."""
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # Wildcard atom
            count += 1
    return count


def get_periodic_features(mol) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract periodicity-related features for polymer representation.

    Returns:
        periodic_mask: Boolean mask indicating which atoms are periodic connection points
        periodic_vectors: 3D vectors indicating periodicity direction (simplified)
    """
    num_atoms = mol.GetNumAtoms()
    periodic_mask = np.zeros(num_atoms, dtype=np.float32)
    periodic_vectors = np.zeros((num_atoms, 3), dtype=np.float32)

    # Identify wildcard atoms (periodic connection points)
    wildcard_positions = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:  # Wildcard atom
            periodic_mask[i] = 1.0
            pos = mol.GetConformer().GetAtomPosition(i)
            wildcard_positions.append((i, np.array([pos.x, pos.y, pos.z])))

    # For polymers with 2 wildcards, compute periodic vector
    if len(wildcard_positions) == 2:
        idx1, pos1 = wildcard_positions[0]
        idx2, pos2 = wildcard_positions[1]
        periodic_vec = pos2 - pos1
        periodic_vec = periodic_vec / (np.linalg.norm(periodic_vec) + 1e-8)
        periodic_vectors[idx1] = periodic_vec
        periodic_vectors[idx2] = -periodic_vec

    return periodic_mask, periodic_vectors


def mol_to_graph(smiles: str, target: float = None) -> Optional[Data]:
    """
    Convert SMILES to PyTorch Geometric Data object.

    Args:
        smiles: Polymer SMILES with * notation
        target: Target Tg value

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

        # Generate 3D conformer for periodic features
        try:
            AllChem.Compute2DCoords(mol)
        except:
            pass

        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(get_atom_features(atom))
        x = torch.tensor(atom_features, dtype=torch.float)

        # Bond features and edge index
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
            bf = get_bond_features(bond)
            edge_attr.append(bf)
            edge_attr.append(bf)

        if len(edge_index) == 0:
            return None

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Compute rigidity
        rigidity = compute_rotatable_bond_fraction(mol)

        # Periodic features
        periodic_mask, periodic_vectors = get_periodic_features(mol)
        periodic_mask = torch.tensor(periodic_mask, dtype=torch.float).unsqueeze(1)
        periodic_vectors = torch.tensor(periodic_vectors, dtype=torch.float)

        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([target], dtype=torch.float) if target is not None else None,
            rigidity=torch.tensor([rigidity], dtype=torch.float),
            periodic_mask=periodic_mask,
            periodic_vectors=periodic_vectors,
            smiles=smiles,
        )

        return data

    except Exception as e:
        return None


class PolymerDataset(Dataset):
    """Dataset for polymer Tg prediction."""

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
    seed: int = 42
) -> Tuple[PolymerDataset, PolymerDataset, PolymerDataset, pd.DataFrame]:
    """
    Load data and split into train/val/test sets.

    Returns:
        train_dataset, val_dataset, test_dataset, original_dataframe
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    # Convert to graphs
    data_list = []
    valid_indices = []
    for idx, row in df.iterrows():
        data = mol_to_graph(row['Smiles'], row['Tg'])
        if data is not None:
            data_list.append(data)
            valid_indices.append(idx)

    print(f"Successfully parsed {len(data_list)} molecules")

    # Filter dataframe
    df_valid = df.iloc[valid_indices].reset_index(drop=True)

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

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    return (
        PolymerDataset(train_data),
        PolymerDataset(val_data),
        PolymerDataset(test_data),
        df_valid
    )


def get_rigidity_groups(dataset: PolymerDataset, threshold: float = 0.1) -> List[List[int]]:
    """
    Group samples by rigidity for contrastive learning.

    Args:
        dataset: PolymerDataset
        threshold: Rigidity difference threshold for positive pairs

    Returns:
        List of sample groups with similar rigidity
    """
    rigidities = [data.rigidity.item() for data in dataset.data_list]
    n = len(rigidities)

    # Simple grouping by rigidity bins
    groups = {}
    for i, r in enumerate(rigidities):
        bin_idx = int(r / threshold)
        if bin_idx not in groups:
            groups[bin_idx] = []
        groups[bin_idx].append(i)

    return list(groups.values())


if __name__ == "__main__":
    # Test dataset loading
    csv_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    train_ds, val_ds, test_ds, df = load_and_split_data(csv_path)

    # Test a sample
    sample = train_ds[0]
    print(f"\nSample graph:")
    print(f"  Nodes: {sample.x.shape}")
    print(f"  Edges: {sample.edge_index.shape}")
    print(f"  Edge features: {sample.edge_attr.shape}")
    print(f"  Target Tg: {sample.y.item():.2f}")
    print(f"  Rigidity: {sample.rigidity.item():.4f}")
    print(f"  Periodic mask shape: {sample.periodic_mask.shape}")
    print(f"  Periodic vectors shape: {sample.periodic_vectors.shape}")

    # Feature dimensions
    atom_dim, bond_dim = get_feature_dimensions()
    print(f"\nFeature dimensions:")
    print(f"  Atom features: {atom_dim}")
    print(f"  Bond features: {bond_dim}")
