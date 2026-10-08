"""
Data preprocessing utilities for Chemprop-Polymer-K.

Based on Day 5 Chemprop-Polymer with extensions:
1. K-node features (528 dims)
2. Trimer construction for polymer representation
3. Monomer and dimer construction for augmentation
4. Graph augmentation (bond dropping, feature masking)
5. PI1M and Tg dataset loading
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from typing import List, Dict, Optional, Tuple


# ============================================================
# Electronegativity values (Pauling scale)
# ============================================================
ELECTRONEGATIVITY = {
    1: 2.20,   # H
    5: 2.04,   # B
    6: 2.55,   # C
    7: 3.04,   # N
    8: 3.44,   # O
    9: 3.98,   # F
    14: 1.90,  # Si
    15: 2.19,  # P
    16: 2.58,  # S
    17: 3.16,  # Cl
    35: 2.96,  # Br
    53: 2.66,  # I
}
DEFAULT_EN = 2.0


# ============================================================
# Atom and Bond Feature Dimensions (same as Day 5)
# ============================================================
# Atomic number: 1-100 → 100 dims
ATOMIC_NUM_RANGE = list(range(1, 101))
# Degree: 0-5 → 6 dims
DEGREE_RANGE = list(range(6))
# Formal charge: -2,-1,0,1,2 → 5 dims
FORMAL_CHARGE_RANGE = [-2, -1, 0, 1, 2]
# Hybridization: SP,SP2,SP3,SP3D,SP3D2 → 5 dims
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
# NumHs: 0-4 → 5 dims
NUM_HS_RANGE = list(range(5))

ATOM_FDIM = 100 + 6 + 5 + 5 + 1 + 1 + 5  # = 123


# Bond features: 9 dims
# bond_type: SINGLE/DOUBLE/TRIPLE/AROMATIC (4)
# is_conjugated (1), is_in_ring (1), is_periodic (1)
# electronegativity_gradient (1), boundary_atom_degree (1)
BOND_FDIM = 9


# K-node features: 528 dims
K_FEAT_DIM = 528  # 1+1+1+10+1+1+1+512


def one_hot(value, choices):
    """One-hot encoding for a value given a list of choices."""
    vec = [0] * len(choices)
    try:
        idx = choices.index(value)
        vec[idx] = 1
    except ValueError:
        pass  # all zeros if not found
    return vec


def get_atom_features(atom):
    """Compute atom features for a single RDKit atom (123 dims)."""
    features = []
    # Atomic number one-hot (map 1-100 → index 0-99)
    atomic_num = atom.GetAtomicNum()
    an_vec = [0] * 100
    if 1 <= atomic_num <= 100:
        an_vec[atomic_num - 1] = 1
    features.extend(an_vec)
    # Degree
    features.extend(one_hot(atom.GetDegree(), DEGREE_RANGE))
    # Formal charge
    features.extend(one_hot(atom.GetFormalCharge(), FORMAL_CHARGE_RANGE))
    # Hybridization
    features.extend(one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES))
    # Is aromatic
    features.append(int(atom.GetIsAromatic()))
    # Is in ring
    features.append(int(atom.IsInRing()))
    # Number of Hs
    features.extend(one_hot(atom.GetTotalNumHs(), NUM_HS_RANGE))
    return features


def get_bond_features(bond, is_periodic=False, en_gradient=0.0, boundary_degree=0.0):
    """Compute bond features for a single RDKit bond or periodic edge (9 dims)."""
    features = []
    if is_periodic:
        # Bond type one-hot
        features.extend(one_hot(bond, ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']))
        features.append(0)  # is_conjugated
        features.append(1)  # is_in_ring (periodic edge creates ring)
        features.append(1)  # is_periodic
        features.append(float(en_gradient))  # electronegativity_gradient
        features.append(float(boundary_degree))  # boundary_atom_degree
    else:
        bt = bond.GetBondType()
        bond_type_map = {
            Chem.rdchem.BondType.SINGLE: 'SINGLE',
            Chem.rdchem.BondType.DOUBLE: 'DOUBLE',
            Chem.rdchem.BondType.TRIPLE: 'TRIPLE',
            Chem.rdchem.BondType.AROMATIC: 'AROMATIC',
        }
        bt_str = bond_type_map.get(bt, 'SINGLE')
        features.extend(one_hot(bt_str, ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']))
        features.append(int(bond.GetIsConjugated()))
        features.append(int(bond.IsInRing()))
        features.append(0)  # is_periodic
        features.append(0.0)  # electronegativity_gradient
        features.append(0.0)  # boundary_atom_degree
    return features


def get_electronegativity(atom):
    """Get Pauling electronegativity for an atom."""
    return ELECTRONEGATIVITY.get(atom.GetAtomicNum(), DEFAULT_EN)


def parse_psmiles(psmiles):
    """
    Parse a p-SMILES string and extract:
    - repeat unit mol (without star atoms)
    - star atom bonding info: [(neighbor_idx, bond_type, star_idx), ...]
    Returns (mol, star_info) or (None, None) on failure.
    """
    mol = Chem.MolFromSmiles(psmiles)
    if mol is None:
        return None, None

    # Find star atoms (atomic number 0)
    star_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(star_atoms) != 2:
        return None, None  # Skip if not exactly 2 star atoms

    star_info = []
    for star_atom in star_atoms:
        star_idx = star_atom.GetIdx()
        neighbors = star_atom.GetNeighbors()
        if len(neighbors) != 1:
            return None, None  # Star atom should have exactly 1 neighbor
        neighbor = neighbors[0]
        bond = mol.GetBondBetweenAtoms(star_idx, neighbor.GetIdx())
        bond_type = bond.GetBondType()
        star_info.append((neighbor.GetIdx(), bond_type, star_idx))

    # Sort star info: left star = first in SMILES order
    # The star atom with the smaller neighbor index is "left"
    star_info.sort(key=lambda x: x[0])

    # Create repeat unit by removing star atoms
    rwmol = Chem.RWMol(mol)
    # Remove star atoms (remove higher index first to preserve indices)
    star_indices = sorted([s[2] for s in star_info], reverse=True)
    for idx in star_indices:
        rwmol.RemoveAtom(idx)

    # Update neighbor indices after removal
    star_indices_asc = sorted([s[2] for s in star_info])

    # Recompute neighbor indices
    new_star_info = []
    for neighbor_idx, bond_type, star_idx in star_info:
        # After removing star atoms with higher index, adjust neighbor_idx
        new_idx = neighbor_idx
        for si in star_indices_asc:
            if si < neighbor_idx:
                new_idx -= 1
        new_star_info.append((new_idx, bond_type))

    # left neighbor and right neighbor
    left_neighbor_idx = new_star_info[0][0]
    left_bond_type = new_star_info[0][1]
    right_neighbor_idx = new_star_info[1][0]
    right_bond_type = new_star_info[1][1]

    repeat_unit = rwmol.GetMol()
    try:
        Chem.SanitizeMol(repeat_unit)
    except Exception:
        return None, None

    return repeat_unit, (left_neighbor_idx, left_bond_type, right_neighbor_idx, right_bond_type)


def compute_knode_features(mol):
    """
    Compute K-node features for a repeat unit (528 dims).

    Features:
    - molecular_weight (1)
    - num_heavy_atoms (1)
    - num_bonds (1)
    - composition_ratios (10): C, H, O, N, F, P, S, Cl, Br, I ratios
    - num_rings (1)
    - aromatic_ratio (1)
    - sp3_fraction (1)
    - morgan_fingerprint (512)

    Args:
        mol: RDKit mol object (repeat unit without star atoms)

    Returns:
        List of 528 float features
    """
    features = []

    # 1. Molecular weight
    mol_weight = Descriptors.MolWt(mol)
    features.append(mol_weight)

    # 2. Number of heavy atoms
    num_heavy = mol.GetNumAtoms()
    features.append(float(num_heavy))

    # 3. Number of bonds
    num_bonds = mol.GetNumBonds()
    features.append(float(num_bonds))

    # 4. Composition ratios (C, H, O, N, F, P, S, Cl, Br, I)
    element_counts = {}
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        symbol = atom.GetSymbol()
        element_counts[symbol] = element_counts.get(symbol, 0) + 1

    elements = ['C', 'H', 'O', 'N', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    for elem in elements:
        count = element_counts.get(elem, 0)
        ratio = count / num_heavy if num_heavy > 0 else 0.0
        features.append(ratio)

    # 5. Number of rings
    num_rings = Descriptors.RingCount(mol)
    features.append(float(num_rings))

    # 6. Aromatic ratio
    num_aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    aromatic_ratio = num_aromatic_atoms / num_heavy if num_heavy > 0 else 0.0
    features.append(aromatic_ratio)

    # 7. SP3 fraction
    sp3_atoms = 0
    for atom in mol.GetAtoms():
        if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3:
            sp3_atoms += 1
    sp3_fraction = sp3_atoms / num_heavy if num_heavy > 0 else 0.0
    features.append(sp3_fraction)

    # 8. Morgan fingerprint (512 bits, radius=2)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
    fp_array = np.zeros((512,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, fp_array)
    features.extend(fp_array.tolist())

    if len(features) != K_FEAT_DIM:
        raise ValueError(f"K-node features have wrong dimension: {len(features)} != {K_FEAT_DIM}")

    return features


def construct_monomer_graph(mol):
    """Construct graph data for monomer only (no periodic edge)."""
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    bonds = []
    bond_features_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = get_bond_features(bond)
        # Forward: i->j
        bonds.append((i, j))
        bond_features_list.append(bf)
        # Backward: j->i
        bonds.append((j, i))
        bond_features_list.append(bf)

    return {
        'atom_features': atom_features,
        'bonds': bonds,
        'bond_features': bond_features_list,
        'num_atoms': mol.GetNumAtoms(),
        'periodic_edge_bonds': [],  # no periodic edges
    }


def construct_trimer_graph(mol, left_neighbor, right_neighbor, add_periodic_edge=False):
    """
    Construct trimer graph from repeat unit.

    Args:
        mol: repeat unit RDKit mol
        left_neighbor: (idx, bond_type) for left star atom neighbor
        right_neighbor: (idx, bond_type) for right star atom neighbor
        add_periodic_edge: whether to add periodic edge

    Returns:
        Graph dict with atom_features, bonds, bond_features, num_atoms, periodic_edge_bonds
    """
    n_atoms = mol.GetNumAtoms()
    left_nidx, left_bt = left_neighbor
    right_nidx, right_bt = right_neighbor

    # Atom features for trimer (3 copies)
    atom_features = []
    for copy_idx in range(3):
        for atom in mol.GetAtoms():
            atom_features.append(get_atom_features(atom))

    # Internal bonds for each copy
    bonds = []
    bond_features_list = []

    for copy_idx in range(3):
        offset = copy_idx * n_atoms
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx() + offset
            j = bond.GetEndAtomIdx() + offset
            bf = get_bond_features(bond)
            bonds.append((i, j))
            bond_features_list.append(bf)
            bonds.append((j, i))
            bond_features_list.append(bf)

    # Inter-RU bonds
    # Connect right_partner of copy k to left_partner of copy k+1
    for k in range(2):
        right_partner = right_nidx + k * n_atoms
        left_partner = left_nidx + (k + 1) * n_atoms
        # Use right bond type for inter-RU connection
        bt_str_map = {
            Chem.rdchem.BondType.SINGLE: 'SINGLE',
            Chem.rdchem.BondType.DOUBLE: 'DOUBLE',
            Chem.rdchem.BondType.TRIPLE: 'TRIPLE',
            Chem.rdchem.BondType.AROMATIC: 'AROMATIC',
        }
        bt_str = bt_str_map.get(right_bt, 'SINGLE')
        bf = [0] * BOND_FDIM
        bf[:4] = one_hot(bt_str, ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'])
        bf[4] = 0  # not conjugated
        bf[5] = 0  # not in ring
        bf[6] = 0  # not periodic
        bf[7] = 0.0
        bf[8] = 0.0

        bonds.append((right_partner, left_partner))
        bond_features_list.append(bf)
        bonds.append((left_partner, right_partner))
        bond_features_list.append(bf)

    # Periodic edge
    periodic_edge_bonds = []
    if add_periodic_edge:
        # Left terminal = atom bonded to LEFT star atom in copy 1
        left_terminal = left_nidx  # copy 0
        # Right terminal = atom bonded to RIGHT star atom in copy 3 (last copy)
        right_terminal = right_nidx + 2 * n_atoms

        # Compute periodic edge features
        left_atom = mol.GetAtomWithIdx(left_terminal % n_atoms)
        right_atom = mol.GetAtomWithIdx(right_terminal % n_atoms)

        en_left = get_electronegativity(left_atom)
        en_right = get_electronegativity(right_atom)
        en_gradient = abs(en_right - en_left)

        # Boundary atom degree: degree in trimer before adding periodic edge
        def count_bonds(atom_idx, bonds_list):
            count = 0
            for (s, t) in bonds_list:
                if s == atom_idx or t == atom_idx:
                    count += 1
            # Each undirected bond appears twice (forward+backward), so divide by 2
            return count // 2

        deg_left = count_bonds(left_terminal, bonds)
        deg_right = count_bonds(right_terminal, bonds)
        boundary_degree = max(deg_left, deg_right)

        # Periodic edge bond type = same as RIGHT star atom bond type
        bt_str_map = {
            Chem.rdchem.BondType.SINGLE: 'SINGLE',
            Chem.rdchem.BondType.DOUBLE: 'DOUBLE',
            Chem.rdchem.BondType.TRIPLE: 'TRIPLE',
            Chem.rdchem.BondType.AROMATIC: 'AROMATIC',
        }
        bt_str = bt_str_map.get(right_bt, 'SINGLE')

        pf = [0] * BOND_FDIM
        pf[:4] = one_hot(bt_str, ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'])
        pf[4] = 0  # not conjugated
        pf[5] = 1  # in ring
        pf[6] = 1  # is periodic
        pf[7] = float(en_gradient)
        pf[8] = float(boundary_degree)

        # Forward: right_terminal -> left_terminal
        bonds.append((right_terminal, left_terminal))
        bond_features_list.append(pf)
        periodic_edge_bonds.append(len(bonds) - 1)

        # Backward: left_terminal -> right_terminal
        bonds.append((left_terminal, right_terminal))
        bond_features_list.append(pf)
        periodic_edge_bonds.append(len(bonds) - 1)

    return {
        'atom_features': atom_features,
        'bonds': bonds,
        'bond_features': bond_features_list,
        'num_atoms': 3 * n_atoms,
        'periodic_edge_bonds': periodic_edge_bonds,
    }


def construct_dimer_graph(mol, left_neighbor, right_neighbor):
    """
    Construct dimer graph (2 repeat units) for augmentation.

    Args:
        mol: repeat unit RDKit mol
        left_neighbor: (idx, bond_type) for left star atom neighbor
        right_neighbor: (idx, bond_type) for right star atom neighbor

    Returns:
        Graph dict
    """
    n_atoms = mol.GetNumAtoms()
    left_nidx, left_bt = left_neighbor
    right_nidx, right_bt = right_neighbor

    # Atom features for dimer (2 copies)
    atom_features = []
    for copy_idx in range(2):
        for atom in mol.GetAtoms():
            atom_features.append(get_atom_features(atom))

    # Internal bonds for each copy
    bonds = []
    bond_features_list = []

    for copy_idx in range(2):
        offset = copy_idx * n_atoms
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx() + offset
            j = bond.GetEndAtomIdx() + offset
            bf = get_bond_features(bond)
            bonds.append((i, j))
            bond_features_list.append(bf)
            bonds.append((j, i))
            bond_features_list.append(bf)

    # Inter-RU bond (connect copy 0 to copy 1)
    right_partner = right_nidx  # copy 0
    left_partner = left_nidx + n_atoms  # copy 1

    bt_str_map = {
        Chem.rdchem.BondType.SINGLE: 'SINGLE',
        Chem.rdchem.BondType.DOUBLE: 'DOUBLE',
        Chem.rdchem.BondType.TRIPLE: 'TRIPLE',
        Chem.rdchem.BondType.AROMATIC: 'AROMATIC',
    }
    bt_str = bt_str_map.get(right_bt, 'SINGLE')
    bf = [0] * BOND_FDIM
    bf[:4] = one_hot(bt_str, ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'])
    bf[4] = 0  # not conjugated
    bf[5] = 0  # not in ring
    bf[6] = 0  # not periodic
    bf[7] = 0.0
    bf[8] = 0.0

    bonds.append((right_partner, left_partner))
    bond_features_list.append(bf)
    bonds.append((left_partner, right_partner))
    bond_features_list.append(bf)

    return {
        'atom_features': atom_features,
        'bonds': bonds,
        'bond_features': bond_features_list,
        'num_atoms': 2 * n_atoms,
        'periodic_edge_bonds': [],
    }


def add_knode_to_graph(graph_data, k_node_features):
    """
    Add K-node to a graph.

    K-node is connected bidirectionally to ALL real atoms via virtual bonds.
    Virtual bond features are all zeros (9 dims).

    Args:
        graph_data: Graph dict with atom_features, bonds, bond_features, num_atoms
        k_node_features: List of K_FEAT_DIM features

    Returns:
        Updated graph dict with K-node
    """
    n_atoms = graph_data['num_atoms']
    bonds = graph_data['bonds'].copy()
    bond_features_list = graph_data['bond_features'].copy()

    # Virtual bond features (all zeros)
    virtual_bond_features = [0.0] * BOND_FDIM

    # Add bidirectional virtual edges from K-node to all real atoms
    for atom_idx in range(n_atoms):
        # Forward: K-node -> real atom
        bonds.append((n_atoms, atom_idx))
        bond_features_list.append(virtual_bond_features)
        # Backward: real atom -> K-node
        bonds.append((atom_idx, n_atoms))
        bond_features_list.append(virtual_bond_features)

    # Append K-node features to atom features
    atom_features = graph_data['atom_features'].copy()
    atom_features.append(k_node_features)

    return {
        'atom_features': atom_features,
        'bonds': bonds,
        'bond_features': bond_features_list,
        'num_atoms': n_atoms + 1,  # Include K-node
        'periodic_edge_bonds': graph_data['periodic_edge_bonds'],
    }


def augment_graph(graph_data, periodic_edge_bonds, drop_prob=0.1,
                  atom_mask_prob=0.1, bond_mask_prob=0.1, rng=None):
    """
    Augment graph with bond dropping and feature masking.

    Periodic edges are protected (never dropped).

    Args:
        graph_data: Graph dict
        periodic_edge_bonds: List of bond indices that are periodic edges
        drop_prob: Probability of dropping each non-periodic bond
        atom_mask_prob: Probability of masking each atom feature dimension
        bond_mask_prob: Probability of masking each bond feature dimension
        rng: Numpy random generator

    Returns:
        Augmented graph dict
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # Copy graph data
    atom_features = [feat.copy() for feat in graph_data['atom_features']]
    bond_features_list = [bf.copy() for bf in graph_data['bond_features']]
    bonds = graph_data['bonds'].copy()

    # Bond dropping (protect periodic edges)
    if drop_prob > 0:
        periodic_set = set(periodic_edge_bonds)
        bonds_to_keep = []
        bond_features_to_keep = []

        # We need to track bond pairs (forward+backward)
        # If we drop a bond, we drop both directions
        processed = set()
        for idx, (s, t) in enumerate(bonds):
            if idx in processed:
                continue

            # Find the reverse bond
            reverse_idx = -1
            for rev_i, (rs, rt) in enumerate(bonds):
                if rs == t and rt == s:
                    reverse_idx = rev_i
                    break

            # Decide whether to keep this bond pair
            if idx in periodic_set or reverse_idx in periodic_set:
                # Always keep periodic edges
                bonds_to_keep.append((s, t))
                bond_features_to_keep.append(bond_features_list[idx])
                if reverse_idx >= 0:
                    processed.add(reverse_idx)
            else:
                # Randomly drop non-periodic bonds
                if rng.rand() > drop_prob:
                    bonds_to_keep.append((s, t))
                    bond_features_to_keep.append(bond_features_list[idx])
                if reverse_idx >= 0:
                    processed.add(reverse_idx)

        bonds = bonds_to_keep
        bond_features_list = bond_features_to_keep

    # Atom feature masking
    if atom_mask_prob > 0:
        for feat in atom_features:
            mask = rng.rand(len(feat)) < atom_mask_prob
            for i in range(len(feat)):
                if mask[i]:
                    feat[i] = 0.0

    # Bond feature masking
    if bond_mask_prob > 0:
        for bf in bond_features_list:
            mask = rng.rand(len(bf)) < bond_mask_prob
            for i in range(len(bf)):
                if mask[i]:
                    bf[i] = 0.0

    return {
        'atom_features': atom_features,
        'bonds': bonds,
        'bond_features': bond_features_list,
        'num_atoms': graph_data['num_atoms'],
        'periodic_edge_bonds': periodic_edge_bonds,
    }


def build_reverse_edge_map(bonds):
    """For each directed bond index, find the index of its reverse bond."""
    bond_map = {}
    for idx, (s, t) in enumerate(bonds):
        bond_map[(s, t)] = idx

    reverse_map = []
    for idx, (s, t) in enumerate(bonds):
        rev_idx = bond_map.get((t, s), idx)  # fallback to self if not found
        reverse_map.append(rev_idx)

    return reverse_map


def graph_dict_to_tensors(graph_data, k_node_features=None, use_knode=True):
    """
    Convert graph data dict to tensor format.

    Args:
        graph_data: Graph dict with atom_features, bonds, bond_features, num_atoms, periodic_edge_bonds
        k_node_features: K-node features (if use_knode)
        use_knode: Whether to add K-node

    Returns:
        Tensor data dict
    """
    # Add K-node if requested - store K-node features separately
    # Virtual edges from K-node to all atoms are added in collate_batch (model.py)
    if use_knode and k_node_features is not None:
        # Mark that this graph has a K-node so collate_batch knows
        graph_data = dict(graph_data)  # shallow copy
        graph_data['has_knode'] = True

    atom_features = torch.tensor(graph_data['atom_features'], dtype=torch.float32)
    bond_features = torch.tensor(graph_data['bond_features'], dtype=torch.float32)

    bonds = graph_data['bonds']
    if len(bonds) > 0:
        edge_index = torch.tensor(bonds, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    reverse_map = build_reverse_edge_map(bonds)
    reverse_map = torch.tensor(reverse_map, dtype=torch.long)

    result = {
        'atom_features': atom_features,
        'bond_features': bond_features,
        'edge_index': edge_index,
        'reverse_map': reverse_map,
        'num_atoms': graph_data['num_atoms'],
        'periodic_edge_bonds': graph_data['periodic_edge_bonds'],
    }

    if use_knode and k_node_features is not None:
        result['k_node_features'] = torch.tensor(k_node_features, dtype=torch.float32).unsqueeze(0)  # [1, K_FEAT_DIM]
        result['has_knode'] = True

    return result


def load_tg_dataset(csv_path='/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'):
    """
    Load Tg dataset.

    Args:
        csv_path: Path to Tg.csv

    Returns:
        List of (psmiles, tg_value) tuples
    """
    df = pd.read_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        psmiles = row['Smiles']
        tg = row['Tg']
        data.append((psmiles, tg))
    print(f"Loaded {len(data)} polymers from {csv_path}")
    return data


def load_pi1m_dataset(csv_path='/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv',
                      max_samples=None):
    """
    Load PI1M dataset.

    Args:
        csv_path: Path to PI1M.csv
        max_samples: Maximum number of samples to load (for debugging)

    Returns:
        List of psmiles strings
    """
    df = pd.read_csv(csv_path)
    psmiles_list = df['SMILES'].tolist()
    if max_samples:
        psmiles_list = psmiles_list[:max_samples]
    print(f"Loaded {len(psmiles_list)} polymers from {csv_path}")
    return psmiles_list


def process_tg_dataset(csv_path='/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv',
                       use_knode=True, augment=False, seed=42):
    """
    Process Tg dataset and construct trimer graphs.

    Args:
        csv_path: Path to Tg.csv
        use_knode: Whether to add K-node
        augment: Whether to apply augmentation
        seed: Random seed

    Returns:
        List of tensor data dicts
    """
    data = load_tg_dataset(csv_path)
    rng = np.random.RandomState(seed)

    processed = []
    skipped = 0

    for psmiles, tg in data:
        mol, star_info = parse_psmiles(psmiles)
        if mol is None or star_info is None:
            skipped += 1
            continue

        left_nidx, left_bt, right_nidx, right_bt = star_info

        try:
            # Construct trimer with periodic edge
            graph = construct_trimer_graph(mol, (left_nidx, left_bt),
                                           (right_nidx, right_bt),
                                           add_periodic_edge=True)

            # Compute K-node features
            k_node_feat = compute_knode_features(mol) if use_knode else None

            # Apply augmentation if requested
            if augment:
                graph = augment_graph(graph, graph['periodic_edge_bonds'],
                                      drop_prob=0.1, atom_mask_prob=0.1,
                                      bond_mask_prob=0.1, rng=rng)

            # Convert to tensors
            tensor_data = graph_dict_to_tensors(graph, k_node_feat, use_knode=use_knode)
            tensor_data['target'] = torch.tensor([tg], dtype=torch.float32)
            tensor_data['psmiles'] = psmiles
            processed.append(tensor_data)

        except Exception as e:
            skipped += 1
            continue

    print(f"Processed: {len(processed)}, Skipped: {skipped}")
    return processed


def process_pi1m_dataset(csv_path='/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv',
                         use_knode=True, augment=False, max_samples=None, seed=42):
    """
    Process PI1M dataset and construct trimer graphs (no targets).

    Args:
        csv_path: Path to PI1M.csv
        use_knode: Whether to add K-node
        augment: Whether to apply augmentation
        max_samples: Maximum number of samples to process
        seed: Random seed

    Returns:
        List of tensor data dicts
    """
    psmiles_list = load_pi1m_dataset(csv_path, max_samples=max_samples)
    rng = np.random.RandomState(seed)

    processed = []
    skipped = 0

    for psmiles in psmiles_list:
        mol, star_info = parse_psmiles(psmiles)
        if mol is None or star_info is None:
            skipped += 1
            continue

        left_nidx, left_bt, right_nidx, right_bt = star_info

        try:
            # Construct trimer with periodic edge
            graph = construct_trimer_graph(mol, (left_nidx, left_bt),
                                           (right_nidx, right_bt),
                                           add_periodic_edge=True)

            # Compute K-node features
            k_node_feat = compute_knode_features(mol) if use_knode else None

            # Apply augmentation if requested
            if augment:
                graph = augment_graph(graph, graph['periodic_edge_bonds'],
                                      drop_prob=0.1, atom_mask_prob=0.1,
                                      bond_mask_prob=0.1, rng=rng)

            # Convert to tensors
            tensor_data = graph_dict_to_tensors(graph, k_node_feat, use_knode=use_knode)
            tensor_data['psmiles'] = psmiles
            processed.append(tensor_data)

        except Exception as e:
            skipped += 1
            continue

    print(f"Processed: {len(processed)}, Skipped: {skipped}")
    return processed


if __name__ == '__main__':
    # Test data loading and processing
    print("=== Testing Tg dataset processing ===")
    tg_data = process_tg_dataset(use_knode=True, augment=False)
    print(f"Sample data: {tg_data[0]}")

    print("\n=== Testing PI1M dataset processing (100 samples) ===")
    pi1m_data = process_pi1m_dataset(max_samples=100, use_knode=True, augment=False)
    print(f"Sample data: {pi1m_data[0]}")

    print("\n=== All tests passed ===")
