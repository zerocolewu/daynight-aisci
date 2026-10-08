"""
Dataset module for FP-PGNN
Handles periodic graph construction from polymer SMILES
"""

import torch
from torch_geometric.data import Data, Dataset
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import re
from typing import Optional, Tuple, List


# Atom feature dimensions
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'B', 'Na', 'K', 'Ca', 'Mg', 'Zn', 'Fe', 'Cu', 'Other']
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    'Other'
]
NUM_ATOM_FEATURES = len(ATOM_TYPES) + 1 + len(HYBRIDIZATION_TYPES) + 1 + 1  # type + degree + hybrid + aromatic + charge

# Bond feature dimensions
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
NUM_BOND_FEATURES = len(BOND_TYPES) + 1  # bond type + periodic marker


def get_atom_features(atom) -> np.ndarray:
    """Extract atom features as one-hot encoded vector."""
    features = []

    # Atom type (one-hot)
    symbol = atom.GetSymbol()
    atom_type_feat = [0] * len(ATOM_TYPES)
    if symbol in ATOM_TYPES:
        atom_type_feat[ATOM_TYPES.index(symbol)] = 1
    else:
        atom_type_feat[-1] = 1  # Other
    features.extend(atom_type_feat)

    # Degree (normalized)
    degree = atom.GetDegree()
    features.append(min(degree, 6) / 6.0)

    # Hybridization (one-hot)
    hybrid = atom.GetHybridization()
    hybrid_feat = [0] * len(HYBRIDIZATION_TYPES)
    if hybrid in HYBRIDIZATION_TYPES:
        hybrid_feat[HYBRIDIZATION_TYPES.index(hybrid)] = 1
    else:
        hybrid_feat[-1] = 1  # Other
    features.extend(hybrid_feat)

    # Aromaticity
    features.append(1.0 if atom.GetIsAromatic() else 0.0)

    # Formal charge (normalized)
    charge = atom.GetFormalCharge()
    features.append(np.clip(charge, -3, 3) / 3.0)

    return np.array(features, dtype=np.float32)


def get_bond_features(bond, is_periodic: bool = False) -> np.ndarray:
    """Extract bond features as one-hot encoded vector."""
    features = []

    # Bond type (one-hot)
    bond_type = bond.GetBondType() if bond is not None else None
    bond_type_feat = [0] * len(BOND_TYPES)

    if is_periodic:
        # Periodic edge - use zeros for bond type, mark as periodic
        features.extend(bond_type_feat)
        features.append(1.0)  # Periodic marker
    else:
        if bond_type in BOND_TYPES:
            bond_type_feat[BOND_TYPES.index(bond_type)] = 1
        features.extend(bond_type_feat)
        features.append(0.0)  # Not periodic

    return np.array(features, dtype=np.float32)


def parse_polymer_smiles(smiles: str) -> Tuple[str, int, int]:
    """
    Parse polymer SMILES to identify head and tail atoms.
    Returns: (standard_smiles, head_idx, tail_idx)

    Polymer SMILES use [*:1] and [*:2] notation for head and tail.
    We need to find these and remember their positions.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, -1, -1

    head_idx = -1
    tail_idx = -1

    # Find head and tail atoms (marked as [*:1] and [*:2])
    for atom in mol.GetAtoms():
        atom_num = atom.GetAtomicNum()
        if atom_num == 0:  # Dummy atom (*)
            # Check the atom map number
            map_num = atom.GetAtomMapNum()
            if map_num == 1:
                head_idx = atom.GetIdx()
            elif map_num == 2:
                tail_idx = atom.GetIdx()

    # If no explicit head/tail markers, try to find * atoms
    if head_idx == -1 or tail_idx == -1:
        dummy_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummy_atoms) >= 2:
            head_idx = dummy_atoms[0]
            tail_idx = dummy_atoms[1]
        elif len(dummy_atoms) == 1:
            # Single connection point - use same atom for both
            head_idx = tail_idx = dummy_atoms[0]

    return smiles, head_idx, tail_idx


def smiles_to_periodic_graph(smiles: str, target: float,
                              use_periodic: bool = True) -> Optional[Data]:
    """
    Convert polymer SMILES to PyTorch Geometric Data object with periodic edges.

    Args:
        smiles: Polymer SMILES string
        target: Target property value (Tg)
        use_periodic: Whether to add periodic edges

    Returns:
        PyTorch Geometric Data object or None if parsing fails
    """
    # Parse polymer SMILES
    smiles, head_idx, tail_idx = parse_polymer_smiles(smiles)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    if len(atom_features) == 0:
        return None

    x = torch.tensor(np.array(atom_features), dtype=torch.float)

    # Get bonds and edge features
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

        bond_feat = get_bond_features(bond, is_periodic=False)
        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)

    # Add periodic edge if head and tail are different atoms
    if use_periodic and head_idx >= 0 and tail_idx >= 0 and head_idx != tail_idx:
        # Add bidirectional periodic edge
        edge_index.append([head_idx, tail_idx])
        edge_index.append([tail_idx, head_idx])

        # Create periodic edge features (no actual bond type)
        periodic_feat = get_bond_features(None, is_periodic=True)
        edge_attr.append(periodic_feat)
        edge_attr.append(periodic_feat)

    if len(edge_index) == 0:
        # Single atom molecule - add self-loop
        edge_index = [[0, 0]]
        edge_attr = [get_bond_features(None, is_periodic=False)]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)

    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([target], dtype=torch.float),
        smiles=smiles,
        head_idx=head_idx,
        tail_idx=tail_idx,
        num_nodes=x.size(0)
    )

    return data


def compute_ecfp4_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """
    Compute ECFP4 fingerprint for a molecule.

    Args:
        smiles: SMILES string
        radius: Fingerprint radius (2 for ECFP4)
        n_bits: Number of bits in fingerprint

    Returns:
        Fingerprint as numpy array or None if computation fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


class PolymerDataset(Dataset):
    """PyTorch Geometric Dataset for polymer property prediction."""

    def __init__(self, csv_path: str, use_periodic: bool = True,
                 use_fingerprint: bool = True, transform=None):
        """
        Args:
            csv_path: Path to CSV file with SMILES and Tg columns
            use_periodic: Whether to include periodic edges
            use_fingerprint: Whether to compute ECFP4 fingerprints
            transform: Optional transform to apply to data
        """
        super().__init__(transform=transform)

        self.use_periodic = use_periodic
        self.use_fingerprint = use_fingerprint

        # Load data
        df = pd.read_csv(csv_path)
        self.smiles_list = df['Smiles'].tolist()
        self.targets = df['Tg'].values.astype(np.float32)

        # Pre-compute graphs and fingerprints
        self.data_list = []
        self.fingerprints = []
        self.valid_indices = []

        for i, (smiles, target) in enumerate(zip(self.smiles_list, self.targets)):
            # Compute graph
            data = smiles_to_periodic_graph(smiles, target, use_periodic)
            if data is None:
                continue

            # Compute fingerprint
            if use_fingerprint:
                fp = compute_ecfp4_fingerprint(smiles)
                if fp is None:
                    continue
                self.fingerprints.append(fp)
            else:
                self.fingerprints.append(np.zeros(2048, dtype=np.float32))

            self.data_list.append(data)
            self.valid_indices.append(i)

        self.fingerprints = np.array(self.fingerprints)

        print(f"Loaded {len(self.data_list)} valid samples from {len(self.smiles_list)} total")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data_list[idx].clone()
        if self.use_fingerprint:
            # Store fingerprint as a 2D tensor (1, fp_dim) for proper batching
            data.fingerprint = torch.tensor(self.fingerprints[idx], dtype=torch.float).unsqueeze(0)
        return data

    def get_targets(self):
        """Return all target values."""
        return self.targets[self.valid_indices]


def create_data_loaders(dataset: PolymerDataset,
                        train_idx: np.ndarray,
                        val_idx: np.ndarray,
                        batch_size: int = 64):
    """Create train and validation data loaders."""
    from torch_geometric.loader import DataLoader

    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    csv_path = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
    dataset = PolymerDataset(csv_path, use_periodic=True, use_fingerprint=True)

    print(f"\nDataset size: {len(dataset)}")
    print(f"Sample data:")
    data = dataset[0]
    print(f"  Node features shape: {data.x.shape}")
    print(f"  Edge index shape: {data.edge_index.shape}")
    print(f"  Edge attr shape: {data.edge_attr.shape}")
    print(f"  Fingerprint shape: {data.fingerprint.shape}")
    print(f"  Target: {data.y.item():.2f}")
    print(f"  Head idx: {data.head_idx}, Tail idx: {data.tail_idx}")

    # Check periodic edges
    periodic_count = 0
    for i in range(min(10, len(dataset))):
        d = dataset[i]
        if d.edge_attr.shape[0] > 0:
            periodic_edges = (d.edge_attr[:, -1] == 1).sum().item()
            periodic_count += periodic_edges > 0
    print(f"\nSamples with periodic edges: {periodic_count}/10")
