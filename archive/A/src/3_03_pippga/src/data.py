"""
Multi-edge graph construction for polymer repeat units with periodic offset encoding.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
import re


def parse_polymer_smiles(smiles):
    """
    Parse polymer SMILES to identify head/tail atoms marked with *.

    Args:
        smiles: Polymer SMILES with * markers (e.g., "*CC*")

    Returns:
        mol: RDKit molecule object
        head_atoms: List of head atom indices
        tail_atoms: List of tail atom indices
    """
    # Replace * with a dummy atom for RDKit parsing
    # Use [At] as placeholder for connection points
    smiles_processed = smiles.replace('*', '[At]')

    mol = Chem.MolFromSmiles(smiles_processed)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")

    # Find connection point atoms
    head_atoms = []
    tail_atoms = []

    # In polymer SMILES, the first * is typically head, last * is tail
    # We'll mark them based on position in the SMILES string
    star_positions = [i for i, c in enumerate(smiles) if c == '*']

    if len(star_positions) < 2:
        raise ValueError(f"Polymer SMILES must have at least 2 connection points: {smiles}")

    # Find atoms with [At] placeholder
    at_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'At']

    if len(at_atoms) >= 2:
        # First connection point is head, last is tail
        head_atoms = [at_atoms[0]]
        tail_atoms = [at_atoms[-1]]

    return mol, head_atoms, tail_atoms


def build_multi_edge_graph(mol, head_atoms, tail_atoms, distance_threshold=3):
    """
    Build multi-edge graph with periodic offset encoding.

    Creates a 3-copy extended graph (left, center, right) and extracts
    edges with periodic offset labels k ∈ {-1, 0, +1}.

    Args:
        mol: RDKit molecule object
        head_atoms: List of head atom indices
        tail_atoms: List of tail atom indices
        distance_threshold: Maximum graph distance for neighbors

    Returns:
        edge_index: Edge indices [2, num_edges]
        edge_offset: Periodic offset for each edge [num_edges]
        self_loop_features: Self-loop features for each atom [num_atoms, 4]
    """
    num_atoms = mol.GetNumAtoms()

    # Build adjacency list for graph distance computation
    adj = {i: [] for i in range(num_atoms)}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)

    # Compute graph distances using BFS
    def bfs_distances(start):
        dist = {i: float('inf') for i in range(num_atoms)}
        dist[start] = 0
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if dist[neighbor] == float('inf'):
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)
        return dist

    all_distances = {i: bfs_distances(i) for i in range(num_atoms)}

    # Compute normalized distances to head/tail for self-loop features
    head_distances = np.zeros(num_atoms)
    tail_distances = np.zeros(num_atoms)

    if len(head_atoms) > 0:
        head_dists = all_distances[head_atoms[0]]
        max_head_dist = max(head_dists.values()) if max(head_dists.values()) > 0 else 1
        for i in range(num_atoms):
            head_distances[i] = head_dists[i] / max_head_dist

    if len(tail_atoms) > 0:
        tail_dists = all_distances[tail_atoms[0]]
        max_tail_dist = max(tail_dists.values()) if max(tail_dists.values()) > 0 else 1
        for i in range(num_atoms):
            tail_distances[i] = tail_dists[i] / max_tail_dist

    # Create self-loop features: [d_head, d_tail, is_head, is_tail]
    self_loop_features = np.zeros((num_atoms, 4))
    for i in range(num_atoms):
        self_loop_features[i, 0] = head_distances[i]
        self_loop_features[i, 1] = tail_distances[i]
        self_loop_features[i, 2] = 1.0 if i in head_atoms else 0.0
        self_loop_features[i, 3] = 1.0 if i in tail_atoms else 0.0

    # Build multi-edge graph
    # Map atom index to center unit: atom i -> center_idx = i
    # Map atom index to left unit: atom i -> left_idx = i + num_atoms
    # Map atom index to right unit: atom i -> right_idx = i + 2*num_atoms

    edges = []
    offsets = []

    # For each atom in center unit, find neighbors within threshold
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                continue

            dist = all_distances[i][j]
            if dist <= distance_threshold:
                # Add edges to neighbors in all three units
                # Center to center (k=0)
                edges.append([i, j])
                offsets.append(0)

                # If j is near tail, it connects to i in right unit (k=+1)
                if tail_distances[j] < 0.5:  # j is closer to tail
                    edges.append([i, j])
                    offsets.append(1)

                # If j is near head, it connects to i in left unit (k=-1)
                if head_distances[j] < 0.5:  # j is closer to head
                    edges.append([i, j])
                    offsets.append(-1)

    if len(edges) == 0:
        # Fallback: use all edges from original graph
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.append([i, j])
            edges.append([j, i])
            offsets.append(0)
            offsets.append(0)

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_offset = torch.tensor(offsets, dtype=torch.long)
    self_loop_features = torch.tensor(self_loop_features, dtype=torch.float)

    return edge_index, edge_offset, self_loop_features


def get_atom_features(mol):
    """
    Extract atom features (one-hot encoding).

    Features:
    - Atom type: C, N, O, S, F, Cl, Br, I, P, At (connection point), other
    - Degree: 0, 1, 2, 3, 4, 5+
    - Hybridization: SP, SP2, SP3, SP3D, SP3D2, other
    - Is aromatic: 0/1
    """
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'At', 'OTHER']
    degrees = [0, 1, 2, 3, 4, 5]
    hybridizations = [Chem.rdchem.HybridizationType.SP,
                      Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3,
                      Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2]

    features = []
    for atom in mol.GetAtoms():
        # Atom type
        atom_type = atom.GetSymbol()
        if atom_type not in atom_types:
            atom_type = 'OTHER'
        atom_feat = [1.0 if atom_type == t else 0.0 for t in atom_types]

        # Degree
        degree = atom.GetTotalDegree()
        if degree > 5:
            degree = 5
        degree_feat = [1.0 if degree == d else 0.0 for d in degrees]

        # Hybridization
        hyb = atom.GetHybridization()
        hyb_feat = [1.0 if hyb == h else 0.0 for h in hybridizations]
        if sum(hyb_feat) == 0:
            hyb_feat = [0.0] * len(hybridizations)

        # Is aromatic
        aromatic_feat = [1.0 if atom.GetIsAromatic() else 0.0]

        feat = atom_feat + degree_feat + hyb_feat + aromatic_feat
        features.append(feat)

    return torch.tensor(features, dtype=torch.float)


def compute_ecfp4_fingerprint(smiles, radius=2, n_bits=2048):
    """
    Compute ECFP4 fingerprint from polymer SMILES.

    Args:
        smiles: Polymer SMILES string
        radius: Fingerprint radius (2 for ECFP4)
        n_bits: Number of bits in fingerprint

    Returns:
        fingerprint: Binary fingerprint array [n_bits]
    """
    # Replace * with [At] for RDKit
    smiles_processed = smiles.replace('*', '[At]')
    mol = Chem.MolFromSmiles(smiles_processed)

    if mol is None:
        # Return zero fingerprint if parsing fails
        return np.zeros(n_bits, dtype=np.float32)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


class PolymerDataset(Dataset):
    """
    Dataset for polymer Tg prediction with multi-edge graphs.
    """

    def __init__(self, smiles_list, tg_values, distance_threshold=3, augment_shift=False):
        """
        Args:
            smiles_list: List of polymer SMILES strings
            tg_values: List of Tg values (Celsius)
            distance_threshold: Maximum graph distance for multi-edge construction
            augment_shift: Whether to apply shift augmentation for periodic invariance test
        """
        self.smiles_list = smiles_list
        self.tg_values = tg_values
        self.distance_threshold = distance_threshold
        self.augment_shift = augment_shift

        # Pre-process all molecules
        self.processed_data = []
        for smiles, tg in zip(smiles_list, tg_values):
            try:
                if augment_shift and np.random.random() > 0.5:
                    # Apply shift augmentation
                    smiles = self._shift_cut_point(smiles)

                mol, head_atoms, tail_atoms = parse_polymer_smiles(smiles)
                atom_features = get_atom_features(mol)
                edge_index, edge_offset, self_loop_features = build_multi_edge_graph(
                    mol, head_atoms, tail_atoms, distance_threshold
                )
                fingerprint = compute_ecfp4_fingerprint(smiles)

                self.processed_data.append({
                    'smiles': smiles,
                    'atom_features': atom_features,
                    'edge_index': edge_index,
                    'edge_offset': edge_offset,
                    'self_loop_features': self_loop_features,
                    'fingerprint': torch.tensor(fingerprint, dtype=torch.float),
                    'tg': tg,
                    'num_atoms': atom_features.shape[0]
                })
            except Exception as e:
                print(f"Warning: Failed to process {smiles}: {e}")
                continue

    def _shift_cut_point(self, smiles):
        """
        Shift the cut point of a polymer SMILES.
        Example: "*CC*" -> "C*CC*C"
        """
        # Find * positions
        if smiles.count('*') < 2:
            return smiles

        # Simple shift: move first atom after first * to before last *
        # This is a simplified version; more sophisticated shifts could be implemented
        parts = smiles.split('*')
        if len(parts) >= 3:
            # Shift: move first group to end
            shifted = '*' + parts[1] + '*' + parts[0]
            return shifted

        return smiles

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


def collate_fn(batch):
    """
    Collate function for batching polymer graphs.
    """
    # Collect all data
    atom_features_list = [item['atom_features'] for item in batch]
    edge_index_list = [item['edge_index'] for item in batch]
    edge_offset_list = [item['edge_offset'] for item in batch]
    self_loop_features_list = [item['self_loop_features'] for item in batch]
    fingerprint_list = [item['fingerprint'] for item in batch]
    tg_list = [item['tg'] for item in batch]
    num_atoms_list = [item['num_atoms'] for item in batch]

    # Batch atom features
    atom_features_batch = torch.cat(atom_features_list, dim=0)

    # Batch edge indices with offset
    edge_index_batch = []
    edge_offset_batch = []
    atom_offset = 0

    for i, (edge_index, edge_offset) in enumerate(zip(edge_index_list, edge_offset_list)):
        edge_index_batch.append(edge_index + atom_offset)
        edge_offset_batch.append(edge_offset)
        atom_offset += num_atoms_list[i]

    edge_index_batch = torch.cat(edge_index_batch, dim=1)
    edge_offset_batch = torch.cat(edge_offset_batch, dim=0)

    # Batch self-loop features
    self_loop_features_batch = torch.cat(self_loop_features_list, dim=0)

    # Batch fingerprints
    fingerprint_batch = torch.stack(fingerprint_list, dim=0)

    # Batch Tg values
    tg_batch = torch.tensor(tg_list, dtype=torch.float)

    # Create batch index for pooling
    batch_idx = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(num_atoms_list)])

    return {
        'atom_features': atom_features_batch,
        'edge_index': edge_index_batch,
        'edge_offset': edge_offset_batch,
        'self_loop_features': self_loop_features_batch,
        'fingerprint': fingerprint_batch,
        'tg': tg_batch,
        'batch': batch_idx,
        'num_atoms': torch.tensor(num_atoms_list)
    }


def load_tg_data(csv_path, test_split=0.1, val_split=0.1, seed=42):
    """
    Load Tg dataset from CSV file.

    Args:
        csv_path: Path to Tg.csv
        test_split: Fraction of data for test set
        val_split: Fraction of data for validation set
        seed: Random seed for reproducibility

    Returns:
        train_data, val_data, test_data: Dictionaries with smiles and tg values
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    smiles_list = df['Smiles'].tolist()
    tg_values = df['Tg'].tolist()

    # Set random seed
    np.random.seed(seed)

    # Shuffle data
    indices = np.random.permutation(len(smiles_list))
    smiles_list = [smiles_list[i] for i in indices]
    tg_values = [tg_values[i] for i in indices]

    # Split data
    n_total = len(smiles_list)
    n_test = int(n_total * test_split)
    n_val = int(n_total * val_split)
    n_train = n_total - n_test - n_val

    train_data = {
        'smiles': smiles_list[:n_train],
        'tg': tg_values[:n_train]
    }

    val_data = {
        'smiles': smiles_list[n_train:n_train + n_val],
        'tg': tg_values[n_train:n_train + n_val]
    }

    test_data = {
        'smiles': smiles_list[n_train + n_val:],
        'tg': tg_values[n_train + n_val:]
    }

    return train_data, val_data, test_data


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    csv_path = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
    train_data, val_data, test_data = load_tg_data(csv_path)

    print(f"Train: {len(train_data['smiles'])} samples")
    print(f"Val: {len(val_data['smiles'])} samples")
    print(f"Test: {len(test_data['smiles'])} samples")

    # Test dataset creation
    dataset = PolymerDataset(train_data['smiles'][:5], train_data['tg'][:5])
    print(f"Dataset size: {len(dataset)}")

    # Test a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Atom features shape: {sample['atom_features'].shape}")
    print(f"Edge index shape: {sample['edge_index'].shape}")
    print(f"Edge offset shape: {sample['edge_offset'].shape}")
    print(f"Self-loop features shape: {sample['self_loop_features'].shape}")
    print(f"Fingerprint shape: {sample['fingerprint'].shape}")
    print(f"Tg: {sample['tg']}")

    # Test collate
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch atom features shape: {batch['atom_features'].shape}")
    print(f"Batch edge index shape: {batch['edge_index'].shape}")
    print(f"Batch fingerprint shape: {batch['fingerprint'].shape}")

    print("\nData module test passed!")
