"""
Polymer-specific utilities for SMILES processing, functional group detection,
and graph construction
"""

import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger
from torch_geometric.data import Data
import torch
from collections import defaultdict

# Suppress RDKit logging
RDLogger.DisableLog('rdApp.*')


# Functional group SMARTS patterns for polymers
FUNCTIONAL_GROUP_SMARTS = {
    'aliphatic_chain': '[CX4]',  # sp3 carbons (aliphatic)
    'aromatic_ring': '[a]',  # aromatic atoms
    'carbonyl': '[CX3]=[OX1]',  # C=O
    'ether': '[OD2]([#6])[#6]',  # C-O-C
    'amine': '[NX3;H2,H1;!$(NC=O)]',  # primary/secondary amines
    'fluorine': '[F]',  # fluorine atoms
    'hydroxyl': '[OX2H]',  # -OH
    'carboxylic_acid': '[CX3](=[OX1])[OX2H]',  # -COOH
    'ester': '[CX3](=[OX1])[OX2][#6]',  # -COO-
    'amide': '[NX3][CX3](=[OX1])',  # -CONH-
    'silicon': '[Si]',  # silicon atoms (silicones)
    'sulfide': '[SX2]',  # sulfide
    'sulfone': '[SX4](=[OX1])(=[OX1])',  # sulfone
}


def get_functional_group_atoms(mol):
    """
    Identify atoms belonging to each functional group

    Returns:
        dict: {group_name: set of atom indices}
    """
    groups = {}
    for name, smarts in FUNCTIONAL_GROUP_SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            matches = mol.GetSubstructMatches(pattern)
            atoms = set()
            for match in matches:
                atoms.update(match)
            groups[name] = atoms
    return groups


def get_functional_group_mask_weights(mol, all_atom_indices):
    """
    Get masking weights for each atom based on functional group membership
    Atoms in functional groups get higher probability of being masked

    Args:
        mol: RDKit molecule
        all_atom_indices: list of all atom indices

    Returns:
        weights: numpy array of masking weights for each atom
    """
    groups = get_functional_group_atoms(mol)

    # Base weight
    weights = np.ones(len(all_atom_indices)) * 0.5

    # Increase weight for atoms in functional groups
    for group_name, atoms in groups.items():
        for atom_idx in atoms:
            if atom_idx < len(weights):
                weights[atom_idx] = 1.5  # Higher probability

    # Normalize
    weights = weights / weights.sum()

    return weights


def canonicalize_polymer_smiles(smiles):
    """
    Canonicalize polymer SMILES (remove asterisks for processing)
    """
    # Remove polymer markers
    clean = smiles.replace('*', '')
    # Handle empty or invalid
    if not clean or clean.isspace():
        return None
    return clean


def is_valid_polymer_smiles(smiles):
    """Check if SMILES represents a valid polymer"""
    if not smiles or len(smiles) < 2:
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def get_cyclic_rotations(smiles, k=3):
    """
    Generate cyclic rotations of polymer SMILES
    This is for polymer repeat units where rotation should not change meaning

    Args:
        smiles: SMILES string (should be canonicalized, no asterisks)
        k: number of rotations to generate

    Returns:
        list of rotated SMILES strings
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles] * k

    num_atoms = mol.GetNumAtoms()
    if num_atoms <= 2:
        return [smiles] * k

    rotations = []

    # For polymers, we can rotate the starting atom
    # But we need to be careful with ring closures
    # A simpler approach: use different atom ordering for canonicalization

    try:
        # Get different atom orderings by renumbering
        for i in range(k):
            # Create a new molecule with different atom ordering
            atom_order = list(range(num_atoms))
            # Rotate the order
            shift = (i * (num_atoms // k)) % num_atoms
            rotated_order = atom_order[shift:] + atom_order[:shift]

            # Renumber and canonicalize
            reordered_mol = Chem.RenumberAtoms(mol, rotated_order)
            canonical = Chem.MolToSmiles(reordered_mol, canonical=True)
            rotations.append(canonical)
    except:
        rotations = [Chem.MolToSmiles(mol, canonical=True)] * k

    return rotations


def compute_ecfp_fingerprint(smiles, radius=2, n_bits=2048):
    """
    Compute ECFP fingerprint for a SMILES string

    Args:
        smiles: SMILES string
        radius: fingerprint radius (2 for ECFP4)
        n_bits: fingerprint size

    Returns:
        numpy array of fingerprint bits
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def tanimoto_similarity(fp1, fp2):
    """Compute Tanimoto similarity between two fingerprints"""
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    if union == 0:
        return 0.0
    return intersection / union


# Atom feature definitions
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),  # Up to element 118
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    'is_aromatic': [False, True],
    'mass': None,  # Will be normalized
}

# Additional features for polymer atoms
POLYMER_ATOM_FEATURES = {
    'is_star': [False, True],  # Is this a polymer connection point (*)
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


def get_atom_features(atom):
    """
    Get feature vector for an atom

    Returns:
        list of feature values
    """
    features = []

    # Atomic number
    features += one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])

    # Degree
    features += one_hot_encoding(atom.GetDegree(), ATOM_FEATURES['degree'])

    # Formal charge
    features += one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])

    # Chiral tag
    features += one_hot_encoding(atom.GetChiralTag(), ATOM_FEATURES['chiral_tag'])

    # Number of Hs
    features += one_hot_encoding(atom.GetTotalNumHs(), ATOM_FEATURES['num_Hs'])

    # Hybridization
    features += one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES['hybridization'])

    # Is aromatic
    features += one_hot_encoding(atom.GetIsAromatic(), ATOM_FEATURES['is_aromatic'])

    # Is star (polymer marker)
    is_star = atom.GetAtomicNum() == 0  # Wildcard atoms
    features += one_hot_encoding(is_star, POLYMER_ATOM_FEATURES['is_star'])

    # Mass (normalized)
    features.append(atom.GetMass() / 100.0)

    return features


# Edge feature definitions
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def get_bond_features(bond):
    """
    Get feature vector for a bond

    Returns:
        list of feature values
    """
    features = []

    # Bond type
    features += one_hot_encoding(bond.GetBondType(), BOND_TYPES)

    # Conjugated
    features.append(int(bond.GetIsConjugated()))

    # In ring
    features.append(int(bond.IsInRing()))

    # Stereo
    features += one_hot_encoding(bond.GetStereo(), [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ])

    return features


def smiles_to_graph(smiles, add_self_loops=False):
    """
    Convert SMILES to PyTorch Geometric Data object

    Args:
        smiles: SMILES string
        add_self_loops: whether to add self-loops

    Returns:
        Data object or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge features and indices
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

    if len(edge_indices) == 0:
        # Single atom molecule - empty edge attributes (13 features: 5 bond_type + 1 conjugated + 1 in_ring + 6 stereo)
        edge_index = torch.tensor([[], []], dtype=torch.long)
        edge_attr = torch.zeros((0, 13), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # Ensure all attributes are present
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def batch_smiles_to_graphs(smiles_list, num_workers=1):
    """
    Convert a list of SMILES to a list of Data objects

    Args:
        smiles_list: list of SMILES strings
        num_workers: number of parallel workers (currently single-threaded)

    Returns:
        list of Data objects
    """
    graphs = []
    for smiles in smiles_list:
        graph = smiles_to_graph(smiles)
        if graph is not None:
            graphs.append(graph)
    return graphs
