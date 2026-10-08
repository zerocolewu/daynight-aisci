"""
Dataset module for PolyPerio-DenseGNN.
Handles polymer SMILES parsing, trimer construction, periodic edges, and angular encoding.

Approach:
- Monomer mode: Parse p-SMILES, keep wildcard atoms, add periodic edge between neighbors of wildcards
- Trimer mode: Build 3-repeat-unit chain by concatenating units and connecting backbone atoms
- Periodic edge: Virtual edge connecting terminal atoms of the repeat unit (encoding polymer periodicity)
- Angular encoding: RBF expansion of directional angles (theta_x, theta_y, theta_z) + distance
"""

import torch
import numpy as np
import pandas as pd
import copy
import warnings
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem, GetDistanceMatrix
from torch_geometric.data import Data

warnings.filterwarnings('ignore')

# ============================================================
# Feature Definitions
# ============================================================

# Compact atom features (similar to OGB mol-hiv style)
ATOM_FEATURES = {
    'atomic_num': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                   39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
                   57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
                   75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
                   93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
                   109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 0],  # 0 for wildcard
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
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

# Derived dimensions
ATOM_FEAT_DIM = (len(ATOM_FEATURES['atomic_num']) + len(ATOM_FEATURES['degree']) +
                 len(ATOM_FEATURES['formal_charge']) + len(ATOM_FEATURES['num_hs']) +
                 len(ATOM_FEATURES['hybridization']) + len(ATOM_FEATURES['is_aromatic']) +
                 len(ATOM_FEATURES['is_in_ring']))

BOND_FEAT_DIM = (len(BOND_FEATURES['bond_type']) + len(BOND_FEATURES['is_conjugated']) +
                 len(BOND_FEATURES['is_in_ring']))

# Additional features: periodic flag (1) + angular encoding (32 = 4*8 RBF)
PERIODIC_FLAG_DIM = 1
ANGULAR_DIM = 32  # 3 angular RBF (8 each) + 1 distance RBF (8)


# ============================================================
# Helper Functions
# ============================================================

def one_hot_encoding(value, choices):
    """Create one-hot encoding for a value given a list of possible choices."""
    encoding = [0] * len(choices)
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        pass
    return encoding


def get_atom_features(atom):
    """Get atom features as a flat vector."""
    features = (
        one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
        + one_hot_encoding(atom.GetTotalDegree(), ATOM_FEATURES['degree'])
        + one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
        + one_hot_encoding(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs'])
        + one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
        + one_hot_encoding(atom.GetIsAromatic(), ATOM_FEATURES['is_aromatic'])
        + one_hot_encoding(atom.IsInRing(), ATOM_FEATURES['is_in_ring'])
    )
    return features


def get_bond_features(bond):
    """Get bond features as a flat vector."""
    features = (
        one_hot_encoding(bond.GetBondType(), BOND_FEATURES['bond_type'])
        + one_hot_encoding(bond.GetIsConjugated(), BOND_FEATURES['is_conjugated'])
        + one_hot_encoding(bond.IsInRing(), BOND_FEATURES['is_in_ring'])
    )
    return features


def get_periodic_bond_features():
    """Get features for a periodic (virtual) bond - same dimension as regular bond features."""
    # 4 bond types (all 0 = no bond type) + 2 conjugated + 2 ring = 8
    return [0, 0, 0, 0, 0, 0, 0, 0]


# ============================================================
# Molecule Construction
# ============================================================

def build_trimer(smiles):
    """
    Build a trimer (3 repeat units) from polymer SMILES.
    Uses RDKit mol editing to concatenate 3 copies of the repeat unit.

    For *CC(*)C:
    - Unit 1: *CC(*)C
    - Unit 2: *CC(*)C
    - Unit 3: *CC(*)C
    Connect backbone atoms between units, remove wildcard atoms.
    Returns (mol, periodic_n1, periodic_n2) or (None, None, None) on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    # Find wildcard atoms and their neighbors
    wildcards = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            wildcards.append((atom.GetIdx(), neighbors))

    if len(wildcards) != 2:
        return None, None, None

    wc1_idx, wc1_neighbors = wildcards[0]
    wc2_idx, wc2_neighbors = wildcards[1]

    if not wc1_neighbors or not wc2_neighbors:
        return None, None, None

    conn1 = wc1_neighbors[0]  # Backbone atom connected to first wildcard
    conn2 = wc2_neighbors[0]  # Backbone atom connected to second wildcard

    # Create 3 copies of the unit
    mol1 = copy.deepcopy(mol)
    mol2 = copy.deepcopy(mol)
    mol3 = copy.deepcopy(mol)

    n_atoms = mol1.GetNumAtoms()

    # Build combined molecule using RWMol
    combo = Chem.RWMol()

    # Add atoms from mol1
    for atom in mol1.GetAtoms():
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetIsAromatic(atom.GetIsAromatic())
        combo.AddAtom(new_atom)

    # Add atoms from mol2
    offset2 = combo.GetNumAtoms()
    for atom in mol2.GetAtoms():
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetIsAromatic(atom.GetIsAromatic())
        combo.AddAtom(new_atom)

    # Add atoms from mol3
    offset3 = combo.GetNumAtoms()
    for atom in mol3.GetAtoms():
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetIsAromatic(atom.GetIsAromatic())
        combo.AddAtom(new_atom)

    # Add bonds from mol1
    for bond in mol1.GetBonds():
        combo.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())

    # Add bonds from mol2
    for bond in mol2.GetBonds():
        combo.AddBond(bond.GetBeginAtomIdx() + offset2,
                      bond.GetEndAtomIdx() + offset2, bond.GetBondType())

    # Add bonds from mol3
    for bond in mol3.GetBonds():
        combo.AddBond(bond.GetBeginAtomIdx() + offset3,
                      bond.GetEndAtomIdx() + offset3, bond.GetBondType())

    # Add inter-unit bonds:
    # conn2 of mol1 <-> conn1 of mol2
    combo.AddBond(conn2, conn1 + offset2, rdchem.BondType.SINGLE)
    # conn2 of mol2 <-> conn1 of mol3
    combo.AddBond(conn2 + offset2, conn1 + offset3, rdchem.BondType.SINGLE)

    # Remove all wildcard atoms (reverse order to preserve indices)
    wc_indices = sorted([wc1_idx, wc2_idx,
                         wc1_idx + offset2, wc2_idx + offset2,
                         wc1_idx + offset3, wc2_idx + offset3], reverse=True)

    for wc_idx in wc_indices:
        atom = combo.GetAtomWithIdx(wc_idx)
        if atom.GetAtomicNum() == 0 and atom.GetDegree() <= 1:
            combo.RemoveAtom(wc_idx)

    # Sanitize
    try:
        Chem.SanitizeMol(combo)
    except Exception:
        return None, None, None

    # Periodic edge: connect first and last backbone atoms
    # These represent the periodic boundary of the polymer chain
    n_final = combo.GetNumAtoms()
    if n_final < 2:
        return combo, None, None

    periodic_n1 = 0
    periodic_n2 = n_final - 1

    return combo, periodic_n1, periodic_n2


def build_monomer_with_periodic(smiles):
    """
    Build monomer graph from polymer SMILES.
    Keeps wildcard atoms in the graph for feature computation.
    Returns (mol, periodic_n1, periodic_n2, wc_indices) or (None, ...) on failure.

    Periodic edge connects the backbone neighbors of the two wildcard atoms.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None, None

    wildcards = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            wildcards.append((atom.GetIdx(), neighbors))

    if len(wildcards) != 2:
        return None, None, None, None

    wc1_idx, wc1_neighbors = wildcards[0]
    wc2_idx, wc2_neighbors = wildcards[1]

    if not wc1_neighbors or not wc2_neighbors:
        return None, None, None, None

    n1 = wc1_neighbors[0]
    n2 = wc2_neighbors[0]

    wc_indices = (wc1_idx, wc2_idx)

    return mol, n1, n2, wc_indices


def build_monomer_simple(smiles):
    """
    Build simple monomer graph by removing wildcard atoms.
    No periodic edge.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Find and remove wildcard atoms
    wildcard_indices = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            wildcard_indices.append(atom.GetIdx())

    if len(wildcard_indices) != 2:
        return None

    combo = Chem.RWMol(mol)
    for idx in sorted(wildcard_indices, reverse=True):
        combo.RemoveAtom(idx)

    try:
        Chem.SanitizeMol(combo)
        return combo
    except Exception:
        return None


# ============================================================
# Angular Encoding
# ============================================================

def rbf_expansion(values, centers, gamma=5.0):
    """Compute RBF expansion of values around centers."""
    diff = values - centers.unsqueeze(0)
    return torch.exp(-gamma * diff ** 2)


def compute_topological_coordinates(mol):
    """
    Compute pseudo-3D coordinates for atoms based on molecular topology.
    Uses graph distance from landmark atoms as coordinate axes.
    """
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return torch.zeros(0, 3)

    try:
        dist_matrix = GetDistanceMatrix(mol)
    except Exception:
        return torch.zeros(num_atoms, 3)

    positions = torch.zeros(num_atoms, 3)

    # Use distance from landmark atoms as pseudo-coordinates
    landmarks = [0, num_atoms - 1]
    if num_atoms > 2:
        landmarks.append(num_atoms // 2)

    for dim, landmark in enumerate(landmarks[:3]):
        if landmark < num_atoms:
            positions[:, dim] = torch.tensor(dist_matrix[landmark], dtype=torch.float)

    # Normalize each dimension
    for dim in range(min(3, len(landmarks))):
        col = positions[:, dim]
        std = col.std()
        if std > 1e-6:
            positions[:, dim] = (col - col.mean()) / std
        else:
            positions[:, dim] = 0

    return positions


def compute_angular_features(mol, edge_index, has_3d=False):
    """
    Compute angular features for each edge using RBF expansion.

    For each edge (i, j), compute:
    - cos(theta_x), cos(theta_y), cos(theta_z): angles with coordinate axes
    - Edge length

    All are encoded using RBF expansion (8 centers each).
    Total: 32 features per edge.
    """
    num_rbf = 8
    gamma = 5.0

    if has_3d:
        conf = mol.GetConformer()
        positions = torch.zeros(mol.GetNumAtoms(), 3)
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            positions[i, 0] = pos.x
            positions[i, 1] = pos.y
            positions[i, 2] = pos.z
    else:
        positions = compute_topological_coordinates(mol)

    src, dst = edge_index[0], edge_index[1]
    edge_vec = positions[dst] - positions[src]
    edge_len = edge_vec.norm(dim=1, keepdim=True).clamp(min=1e-6)
    edge_vec_norm = edge_vec / edge_len

    # Direction cosines with coordinate axes
    cos_x = edge_vec_norm[:, 0:1]
    cos_y = edge_vec_norm[:, 1:2]
    cos_z = edge_vec_norm[:, 2:3]

    # RBF expansion for angles
    angle_centers = torch.linspace(-1, 1, num_rbf)
    rbf_x = rbf_expansion(cos_x, angle_centers, gamma)
    rbf_y = rbf_expansion(cos_y, angle_centers, gamma)
    rbf_z = rbf_expansion(cos_z, angle_centers, gamma)

    # Distance RBF
    dist_centers = torch.linspace(0, 8, num_rbf)
    rbf_dist = rbf_expansion(edge_len, dist_centers, gamma)

    return torch.cat([rbf_x, rbf_y, rbf_z, rbf_dist], dim=1)


# ============================================================
# Graph Construction
# ============================================================

def mol_to_graph(smiles, use_trimer=True, use_periodic=True, use_angular=True, use_3d=False):
    """
    Convert polymer SMILES to PyG Data object.

    Args:
        smiles: polymer SMILES string (e.g., '*CC(*)C')
        use_trimer: if True, build trimer (3 repeat units); if False, use monomer
        use_periodic: if True, add periodic edge
        use_angular: if True, compute angular features for edges
        use_3d: if True, use 3D conformer for angular features

    Returns:
        PyG Data object or None on failure
    """
    smiles = smiles.strip()

    periodic_n1 = None
    periodic_n2 = None

    if use_trimer:
        mol, periodic_n1, periodic_n2 = build_trimer(smiles)
        if mol is None:
            return None
    else:
        if use_periodic:
            mol, periodic_n1, periodic_n2, wc_indices = build_monomer_with_periodic(smiles)
            if mol is None:
                # Fallback to simple monomer without periodic
                mol = build_monomer_simple(smiles)
                if mol is None:
                    return None
                use_periodic = False
        else:
            mol = build_monomer_simple(smiles)
            if mol is None:
                return None

    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None

    # Compute 3D coordinates if needed
    has_3d = False
    if use_angular and use_3d:
        try:
            mol_h = Chem.AddHs(mol)
            result = AllChem.EmbedMolecule(mol_h, randomSeed=42, maxAttempts=10)
            if result == -1:
                result = AllChem.EmbedMolecule(mol_h, randomSeed=42,
                                               maxAttempts=50, useRandomCoords=True)
            if result != -1:
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
                mol = Chem.RemoveHs(mol_h)
                has_3d = True
        except Exception:
            has_3d = False

    # ---- Node features ----
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(get_atom_features(atom))

    x = torch.tensor(atom_features_list, dtype=torch.float)

    # ---- Edge index and features ----
    edge_index_list = []
    edge_attr_list = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        features = get_bond_features(bond) + [0]  # periodic flag = 0
        edge_index_list.append([i, j])
        edge_index_list.append([j, i])
        edge_attr_list.append(features)
        edge_attr_list.append(features)

    # ---- Periodic edge ----
    if use_periodic and periodic_n1 is not None and periodic_n2 is not None:
        if periodic_n1 != periodic_n2 and periodic_n1 < num_atoms and periodic_n2 < num_atoms:
            periodic_features = get_periodic_bond_features() + [1]  # periodic flag = 1
            edge_index_list.append([periodic_n1, periodic_n2])
            edge_index_list.append([periodic_n2, periodic_n1])
            edge_attr_list.append(periodic_features)
            edge_attr_list.append(periodic_features)

    if len(edge_index_list) == 0:
        return None

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    # ---- Angular encoding ----
    if use_angular:
        angular_features = compute_angular_features(mol, edge_index, has_3d)
        edge_attr = torch.cat([edge_attr, angular_features], dim=1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_atoms)
    return data


# ============================================================
# Dataset Class
# ============================================================

class PolymerTgDataset(Dataset):
    """Dataset for polymer Tg prediction."""

    def __init__(self, csv_path, use_trimer=True, use_periodic=True,
                 use_angular=True, use_3d=False):
        df = pd.read_csv(csv_path)

        # Detect column names
        for col in df.columns:
            if col.lower() == 'smiles':
                self.smiles_col = col
                break
        else:
            raise ValueError(f"No SMILES column found. Columns: {df.columns.tolist()}")

        for col in df.columns:
            if col.lower() == 'tg':
                self.tg_col = col
                break
        else:
            raise ValueError(f"No Tg column found. Columns: {df.columns.tolist()}")

        self.smiles_list = df[self.smiles_col].tolist()
        self.tg_list = df[self.tg_col].astype(float).tolist()

        self.use_trimer = use_trimer
        self.use_periodic = use_periodic
        self.use_angular = use_angular
        self.use_3d = use_3d

        # Pre-compute graphs
        self.graphs = []
        self.valid_indices = []

        print(f"Processing {len(self.smiles_list)} molecules...")
        for i, (smi, tg) in enumerate(zip(self.smiles_list, self.tg_list)):
            graph = mol_to_graph(smi, use_trimer=use_trimer,
                                 use_periodic=use_periodic,
                                 use_angular=use_angular,
                                 use_3d=use_3d)
            if graph is not None:
                graph.y = torch.tensor([tg], dtype=torch.float)
                self.graphs.append(graph)
                self.valid_indices.append(i)

        print(f"Successfully processed {len(self.graphs)}/{len(self.smiles_list)} molecules")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def get_tg_stats(self):
        """Get mean and std of Tg values."""
        tgs = torch.tensor([g.y.item() for g in self.graphs])
        return tgs.mean().item(), tgs.std().item()
