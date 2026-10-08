"""
Dataset module for KA-GPS: Tg dataset processing with RWSE positional encoding.
Processes p-SMILES into PyTorch Geometric graph objects.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from rdkit import Chem
from scipy.sparse import csgraph
import scipy.sparse as sp


# ── Atom feature dimensions ──
ATOM_TYPES = ['C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

NUM_ATOM_TYPES = len(ATOM_TYPES)  # 10
NUM_HYBRIDIZATION = len(HYBRIDIZATION_TYPES)  # 5
# Atom features: atom_type(10) + degree(7) + formal_charge(1) + hybridization(5) + aromatic(1) = 24
ATOM_FEATURE_DIM = NUM_ATOM_TYPES + 7 + 1 + NUM_HYBRIDIZATION + 1  # 24

# Edge features: bond_type(4) + conjugated(1) + ring(1) = 6
EDGE_FEATURE_DIM = 4 + 1 + 1  # 6

# RWSE dimensions
RWSE_WALK_LENGTHS = [1, 2, 4, 8, 16, 32]
RWSE_DIM = len(RWSE_WALK_LENGTHS)  # 6


def atom_features(atom):
    """Compute atom feature vector."""
    features = []

    # Atom type one-hot
    atom_type = atom.GetSymbol()
    features += [1 if atom_type == t else 0 for t in ATOM_TYPES]

    # Degree (0-6, clipped)
    degree = min(atom.GetDegree(), 6)
    features += [1 if i == degree else 0 for i in range(7)]

    # Formal charge (normalized)
    features.append(float(atom.GetFormalCharge()))

    # Hybridization one-hot
    hyb = atom.GetHybridization()
    features += [1 if hyb == t else 0 for t in HYBRIDIZATION_TYPES]

    # Aromatic
    features.append(1.0 if atom.GetIsAromatic() else 0.0)

    return features


def bond_features(bond):
    """Compute bond feature vector."""
    features = []

    # Bond type one-hot
    bond_type = bond.GetBondType()
    bt_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    features += [1 if bond_type == t else 0 for t in bt_types]

    # Conjugated
    features.append(1.0 if bond.GetIsConjugated() else 0.0)

    # In ring
    features.append(1.0 if bond.IsInRing() else 0.0)

    return features


def compute_rwse(edge_index, num_nodes):
    """Compute Random Walk Structural Encoding (RWSE) features.

    For each node, compute the diagonal entries of A^k for k in RWSE_WALK_LENGTHS,
    where A is the normalized adjacency matrix.
    """
    if num_nodes == 0:
        return np.zeros((0, RWSE_DIM))

    # Build adjacency matrix
    row, col = edge_index[0], edge_index[1]
    data = np.ones(len(row), dtype=np.float32)
    A = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    # Add self-loops for RWSE computation (standard GraphGPS approach)
    A = A + sp.eye(num_nodes, dtype=np.float32)

    # Normalize: D^{-1/2} A D^{-1/2}
    deg = np.array(A.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    # Compute random walk landing probabilities (diagonal of A^k)
    rwse_features = np.zeros((num_nodes, RWSE_DIM), dtype=np.float32)
    Ak = A_norm.copy()
    current_power = 1

    for i, target_k in enumerate(RWSE_WALK_LENGTHS):
        # Compute A^target_k from current power
        while current_power < target_k:
            Ak = Ak @ A_norm
            current_power += 1

        # Extract diagonal
        rwse_features[:, i] = Ak.diagonal()

    return rwse_features


def psmiles_to_graph(psmiles, target):
    """Convert p-SMILES to PyG Data object with RWSE features.

    Processing strategy: Replace * with [H] (capping), then remove Hs to get
    clean molecular graph.
    """
    # Cap * atoms with H
    smiles = psmiles.replace('*', '[H]')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Remove explicit H atoms
    mol = Chem.RemoveHs(mol)

    # Build node features
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(atom_features(atom))

    if len(atom_feats) == 0:
        return None

    x = torch.tensor(atom_feats, dtype=torch.float32)

    # Build edge features and edge_index
    edge_indices = []
    edge_feats = []
    num_bonds = mol.GetNumBonds()

    if num_bonds > 0:
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bf = bond_features(bond)
            # Add both directions for undirected graph
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            edge_feats.append(bf)
            edge_feats.append(bf)

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
    else:
        # Single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32)

    num_nodes = x.size(0)

    # Compute RWSE
    if num_bonds > 0:
        rwse = compute_rwse(edge_index.numpy(), num_nodes)
    else:
        rwse = np.zeros((num_nodes, RWSE_DIM), dtype=np.float32)

    rwse_tensor = torch.tensor(rwse, dtype=torch.float32)

    # Concatenate RWSE to node features
    x = torch.cat([x, rwse_tensor], dim=1)

    y = torch.tensor([target], dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=num_nodes,
        smiles=psmiles,
    )

    return data


class TgDataset(Dataset):
    """Tg (Glass Transition Temperature) dataset from p-SMILES."""

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        print(f"Loaded Tg dataset: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        psmiles = row['Smiles']
        tg = row['Tg']

        graph = psmiles_to_graph(psmiles, tg)
        if graph is None:
            # Return a dummy graph (should not happen with proper data)
            graph = Data(
                x=torch.zeros(1, ATOM_FEATURE_DIM + RWSE_DIM),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, EDGE_FEATURE_DIM)),
                y=torch.tensor([0.0]),
                num_nodes=1,
                smiles='',
            )
        return graph


def collate_fn(batch):
    """Custom collate for batching PyG Data objects."""
    return Batch.from_data_list(batch)


def get_data_loaders(csv_path, seed=71, batch_size=32):
    """Create train/val/test data loaders with 72/8/20 split."""

    dataset = TgDataset(csv_path)
    n = len(dataset)

    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    n_train = int(n * 0.72)
    n_val = int(n * 0.08)
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx


if __name__ == '__main__':
    # Test dataset loading
    csv_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    train_loader, val_loader, test_loader, _, _, _ = get_data_loaders(csv_path, seed=71, batch_size=32)

    for batch in train_loader:
        print(f"Batch x shape: {batch.x.shape}")
        print(f"Batch edge_index shape: {batch.edge_index.shape}")
        print(f"Batch edge_attr shape: {batch.edge_attr.shape}")
        print(f"Batch y shape: {batch.y.shape}")
        print(f"Batch num_graphs: {batch.num_graphs}")
        break
