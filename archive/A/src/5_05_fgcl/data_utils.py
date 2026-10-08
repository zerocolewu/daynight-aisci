"""
Data preprocessing utilities for FGCL
"""

import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import KFold


# Atom feature dimensions
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),  # 118 possible atoms
    'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'is_aromatic': [False, True],
    'num_hydrogens': [0, 1, 2, 3, 4, 5, 6, 7, 8]
}


def one_hot_encoding(value, choices):
    """One-hot encoding with unknown handling"""
    encoding = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1  # Unknown
    return encoding


def atom_features(atom):
    """
    Generate feature vector for an atom

    Returns:
        List of features (length: 44)
    """
    features = []

    # Atomic number (118 + 1 unknown = 119)
    features += one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])

    # Degree (10 + 1 unknown = 11) - but we'll limit to first 5 features
    degree_enc = one_hot_encoding(atom.GetDegree(), ATOM_FEATURES['degree'])
    features += degree_enc[:6]  # Use only 6 features

    # Hybridization (6 + 1 unknown = 7)
    features += one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES['hybridization'])

    # Is aromatic (2)
    features += [int(atom.GetIsAromatic())]

    # Number of hydrogens (8 + 1 unknown = 9) - limit to 5 features
    h_enc = one_hot_encoding(atom.GetTotalNumHs(), ATOM_FEATURES['num_hydrogens'])
    features += h_enc[:6]  # Use only 6 features

    # Total: 119 + 6 + 7 + 1 + 6 = 139 (too many)
    # Simplified version:
    # - Atomic number one-hot (simplified to common atoms): 10
    # - Degree: 6
    # - Hybridization: 7
    # - Aromatic: 1
    # - Hydrogens: 6
    # - Formal charge: 1
    # - In ring: 1
    # Total: 32

    return features


def atom_features_simple(atom):
    """
    Simplified atom features (32 dimensions)
    """
    features = []

    # Atomic number (simplified to common atoms in polymers)
    atomic_num = atom.GetAtomicNum()
    common_atoms = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C, N, O, F, P, S, Cl, Br, I
    atom_enc = [0] * (len(common_atoms) + 1)
    try:
        idx = common_atoms.index(atomic_num)
        atom_enc[idx] = 1
    except ValueError:
        atom_enc[-1] = 1
    features += atom_enc

    # Degree (0-5)
    degree = min(atom.GetDegree(), 5)
    degree_enc = [0] * 6
    degree_enc[degree] = 1
    features += degree_enc

    # Hybridization
    hyb_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    hyb_enc = [0] * (len(hyb_types) + 1)
    try:
        idx = hyb_types.index(atom.GetHybridization())
        hyb_enc[idx] = 1
    except ValueError:
        hyb_enc[-1] = 1
    features += hyb_enc

    # Aromatic
    features.append(int(atom.GetIsAromatic()))

    # Formal charge (normalized)
    features.append(float(atom.GetFormalCharge()) / 10.0)

    # In ring
    features.append(int(atom.IsInRing()))

    # Number of hydrogens (0-5)
    num_h = min(atom.GetTotalNumHs(), 5)
    h_enc = [0] * 6
    h_enc[num_h] = 1
    features += h_enc

    # Total: 10 + 6 + 6 + 1 + 1 + 1 + 6 = 31
    # Pad to 32 for consistency
    features.append(0)

    return features


def smiles_to_graph(smiles, ecfp=None):
    """
    Convert SMILES to PyTorch Geometric Data object

    Args:
        smiles: SMILES string
        ecfp: Pre-computed ECFP fingerprint (optional)

    Returns:
        Data object or None if parsing fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Atom features
        atom_feats = []
        for atom in mol.GetAtoms():
            atom_feats.append(atom_features_simple(atom))

        x = torch.tensor(atom_feats, dtype=torch.float)

        # Edge indices
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])  # Undirected

        if len(edge_indices) == 0:
            # Single atom molecule
            edge_index = torch.tensor([[], []], dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        # Create data object
        data = Data(x=x, edge_index=edge_index)

        # Add ECFP if provided
        if ecfp is not None:
            data.ecfp = torch.tensor(ecfp, dtype=torch.float)

        return data

    except Exception as e:
        print(f"Error converting {smiles}: {e}")
        return None


def load_tg_dataset(csv_path, compute_ecfp=True):
    """
    Load Tg dataset

    Args:
        csv_path: Path to Tg.csv
        compute_ecfp: Whether to compute ECFP fingerprints

    Returns:
        List of (smiles, Tg, ecfp, graph) tuples
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    dataset = []
    failed = 0

    for idx, row in df.iterrows():
        smiles = row['Smiles']
        tg = row['Tg']

        # Compute ECFP
        if compute_ecfp:
            from fgcl_model import compute_ecfp_fingerprint
            ecfp = compute_ecfp_fingerprint(smiles, radius=2, n_bits=2048)
        else:
            ecfp = None

        # Convert to graph
        graph = smiles_to_graph(smiles, ecfp)

        if graph is not None:
            graph.y = torch.tensor([tg], dtype=torch.float)
            graph.smiles = smiles
            dataset.append(graph)
        else:
            failed += 1

    print(f"Successfully processed {len(dataset)} samples, {failed} failed")

    return dataset


def get_kfold_splits(dataset, n_splits=5, random_seed=42):
    """
    Get K-fold cross-validation splits

    Args:
        dataset: List of Data objects
        n_splits: Number of folds
        random_seed: Random seed

    Returns:
        List of (train_indices, test_indices) tuples
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    indices = np.arange(len(dataset))
    splits = []

    for train_idx, test_idx in kf.split(indices):
        splits.append((train_idx, test_idx))

    return splits


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    dataset = load_tg_dataset('/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv')

    print(f"\nDataset size: {len(dataset)}")
    print(f"Sample graph: {dataset[0]}")
    print(f"Node features shape: {dataset[0].x.shape}")
    print(f"Edge index shape: {dataset[0].edge_index.shape}")
    print(f"ECFP shape: {dataset[0].ecfp.shape}")
    print(f"Tg value: {dataset[0].y.item()}")

    # Test K-fold splits
    splits = get_kfold_splits(dataset, n_splits=5)
    print(f"\nNumber of folds: {len(splits)}")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
