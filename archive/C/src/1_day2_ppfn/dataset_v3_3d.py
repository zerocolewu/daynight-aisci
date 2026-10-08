"""
PPFN-v3 Dataset: 3D Conformer Features for polymer Tg prediction.

Key innovation: Extract 3D geometric features from conformers to capture
chain stiffness and mobility, which are fundamental to Tg.

Features:
- Reuses trimer construction from v2 (periodic edges)
- Adds 3D conformer generation (ETKDG + MMFF optimization)
- Extracts geometric features:
  - Bond lengths from conformer (1 feature per bond edge)
  - 3D distances between all pairs of atoms (within cutoff)
  - Dihedral angles for chains of 4 consecutive atoms
  - Bond angles for chains of 3 consecutive atoms
- Global graph features from conformer statistics
- Graceful fallback to 2D-only if conformer generation fails
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from typing import Optional, Tuple, List
import warnings

# Suppress RDKit warnings
RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings('ignore')

# Add parent data processing code to path
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/alldataset/some_dataprocesscode')
from multimer import generate_multimer_smiles


# ── Feature dimensions (same as v2) ────────────────────────────────────────
ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I',
    'B', 'Se', 'Te', 'Ge', 'As', 'Sb', 'Bi', '*'
]
ATOM_TYPES_DICT = {t: i for i, t in enumerate(ATOM_TYPES)}

HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
HYBRIDIZATION_DICT = {t: i for i, t in enumerate(HYBRIDIZATION_TYPES)}

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_TYPES_DICT = {t: i for i, t in enumerate(BOND_TYPES)}

# v3 dimensions
NUM_ATOM_FEATURES_V3 = 27  # Same as v2 (18 + 1 + 5 + 1 + 1 + 1)
NUM_EDGE_FEATURES_V3 = 8   # v2 has 6, we add 2 (bond_length, 3d_distance)
NUM_GLOBAL_FEATURES_V3 = 12  # Statistical features from conformer


def atom_features(atom: Chem.Atom) -> np.ndarray:
    """Compute enhanced atom features (same as v2)."""
    features = []
    # Atom type (one-hot)
    atom_type = atom.GetSymbol()
    type_feat = np.zeros(len(ATOM_TYPES))
    if atom_type in ATOM_TYPES_DICT:
        type_feat[ATOM_TYPES_DICT[atom_type]] = 1.0
    features.append(type_feat)
    # Degree (normalized)
    features.append(np.array([float(atom.GetDegree()) / 6.0]))
    # Hybridization (one-hot)
    hybrid = atom.GetHybridization()
    hybrid_feat = np.zeros(len(HYBRIDIZATION_TYPES))
    if hybrid in HYBRIDIZATION_DICT:
        hybrid_feat[HYBRIDIZATION_DICT[hybrid]] = 1.0
    features.append(hybrid_feat)
    # Aromatic
    features.append(np.array([float(atom.GetIsAromatic())]))
    # Formal charge (normalized)
    features.append(np.array([float(atom.GetFormalCharge()) / 4.0]))
    # In ring
    features.append(np.array([float(atom.IsInRing())]))
    return np.concatenate(features)


def bond_features(bond: Chem.Bond) -> np.ndarray:
    """Compute bond features (same as v2, will be extended with 3D info)."""
    features = []
    # Bond type (one-hot)
    bond_type = bond.GetBondType()
    type_feat = np.zeros(len(BOND_TYPES))
    if bond_type in BOND_TYPES_DICT:
        type_feat[BOND_TYPES_DICT[bond_type]] = 1.0
    else:
        type_feat[0] = 1.0  # default single
    features.append(type_feat)
    # Conjugated
    features.append(np.array([float(bond.GetIsConjugated())]))
    # In ring
    features.append(np.array([float(bond.IsInRing())]))
    return np.concatenate(features)


def build_trimer_graph(p_smiles: str) -> Optional[Chem.Mol]:
    """Build trimer (3 repeat units) from p-SMILES (same as v2)."""
    try:
        trimer_smiles = generate_multimer_smiles(
            num_repeat_units=3, smiles=p_smiles, replace_dummy_atoms=False
        )
        trimer_mol = Chem.MolFromSmiles(trimer_smiles)
        if trimer_mol is None:
            return None
        return trimer_mol
    except Exception:
        return None


def generate_3d_conformer(mol: Chem.Mol, num_attempts: int = 10) -> Optional[Chem.Mol]:
    """
    Generate 3D conformer using ETKDG and optimize with MMFF.

    Args:
        mol: RDKit molecule (2D)
        num_attempts: number of attempts to generate conformer

    Returns:
        Molecule with embedded conformer or None if fails
    """
    try:
        n_atoms = mol.GetNumAtoms()

        # For very large molecules, skip conformer generation (too slow)
        if n_atoms > 100:
            return None

        # Do NOT add Hs - keep atom indices aligned with the original molecule
        # This ensures edge indices map correctly to conformer positions

        # ETKDG embedding
        embed_params = AllChem.ETKDGv3()
        embed_params.randomSeed = 42
        embed_params.maxIterations = 10

        conf_id = AllChem.EmbedMolecule(mol, embed_params)

        if conf_id == -1:
            return None

        # Light MMFF optimization (without Hs, so faster)
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props)
                if ff is not None:
                    ff.Minimize(maxIts=50)
        except Exception:
            pass

        return mol

    except Exception:
        return None


def extract_3d_features(mol_3d: Chem.Mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 3D geometric features from conformer.

    Returns:
        bond_lengths: (n_bonds,) array of bond lengths
        bond_angles: (n_angles,) array of bond angles in degrees
        dihedral_angles: (n_dihedrals,) array of dihedral angles in degrees
    """
    conf = mol_3d.GetConformer()

    # 1. Bond lengths
    bond_lengths = []
    for bond in mol_3d.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        pos_i = np.array(conf.GetAtomPosition(i))
        pos_j = np.array(conf.GetAtomPosition(j))
        length = np.linalg.norm(pos_i - pos_j)
        bond_lengths.append(length)
    bond_lengths = np.array(bond_lengths)

    # 2. Bond angles (3 consecutive atoms)
    bond_angles = []
    for atom in mol_3d.GetAtoms():
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) >= 2:
            # All combinations of 2 neighbors
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    idx1, idx2, idx3 = neighbors[i], atom.GetIdx(), neighbors[j]
                    pos1 = np.array(conf.GetAtomPosition(idx1))
                    pos2 = np.array(conf.GetAtomPosition(idx2))
                    pos3 = np.array(conf.GetAtomPosition(idx3))

                    v1 = pos1 - pos2
                    v2 = pos3 - pos2

                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_angle))
                    bond_angles.append(angle)

    bond_angles = np.array(bond_angles)

    # 3. Dihedral angles (4 consecutive atoms along chains)
    dihedral_angles = []
    for bond in mol_3d.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Look for paths of length 4: a-b-c-d
        atom_i = mol_3d.GetAtomWithIdx(i)
        atom_j = mol_3d.GetAtomWithIdx(j)

        # Find neighbors of i (not j)
        neighbors_i = [n.GetIdx() for n in atom_i.GetNeighbors() if n.GetIdx() != j]
        # Find neighbors of j (not i)
        neighbors_j = [n.GetIdx() for n in atom_j.GetNeighbors() if n.GetIdx() != i]

        for a in neighbors_i:
            for d in neighbors_j:
                # Compute dihedral angle a-i-j-d
                pos_a = np.array(conf.GetAtomPosition(a))
                pos_i = np.array(conf.GetAtomPosition(i))
                pos_j = np.array(conf.GetAtomPosition(j))
                pos_d = np.array(conf.GetAtomPosition(d))

                # Vectors
                b1 = pos_i - pos_a
                b2 = pos_j - pos_i
                b3 = pos_d - pos_j

                # Normal vectors
                n1 = np.cross(b1, b2)
                n2 = np.cross(b2, b3)

                # Dihedral angle
                cos_dihedral = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-8)
                cos_dihedral = np.clip(cos_dihedral, -1.0, 1.0)

                # Sign using cross product
                m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-8))
                x = np.dot(n1, n2)
                y = np.dot(m1, n2)
                dihedral = np.degrees(np.arctan2(y, x))

                dihedral_angles.append(dihedral)

    dihedral_angles = np.array(dihedral_angles)

    return bond_lengths, bond_angles, dihedral_angles


def compute_global_features(bond_lengths: np.ndarray, bond_angles: np.ndarray,
                           dihedral_angles: np.ndarray, n_atoms: int) -> np.ndarray:
    """
    Compute global statistical features from 3D geometry.

    Returns:
        (12,) array of features:
        - mean, std, min, max of bond lengths (4)
        - mean, std of bond angles (2)
        - mean, std of dihedral angles (2)
        - volume estimate (1)
        - radius of gyration (1)
        - density (n_atoms / volume) (1)
        - shape descriptor (1)
    """
    features = []

    # Bond length statistics
    if len(bond_lengths) > 0:
        features.extend([
            np.mean(bond_lengths),
            np.std(bond_lengths),
            np.min(bond_lengths),
            np.max(bond_lengths),
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])

    # Bond angle statistics
    if len(bond_angles) > 0:
        features.extend([
            np.mean(bond_angles) / 180.0,  # Normalize to [0, 1]
            np.std(bond_angles) / 180.0,
        ])
    else:
        features.extend([0.0, 0.0])

    # Dihedral angle statistics
    if len(dihedral_angles) > 0:
        # Map to [0, 1] by dividing by 180
        features.extend([
            np.mean(np.abs(dihedral_angles)) / 180.0,
            np.std(dihedral_angles) / 180.0,
        ])
    else:
        features.extend([0.0, 0.0])

    # Volume estimate: sum of atomic volumes (approximation)
    # Simple approximation: volume ~ n_atoms * 20 Å^3 (average atomic volume)
    volume_estimate = n_atoms * 20.0
    features.append(np.log(volume_estimate + 1.0) / 10.0)  # Log normalize

    # Radius of gyration approximation (using n_atoms as proxy)
    # Rg ~ sqrt(n_atoms) * 1.5 Å
    rg_estimate = np.sqrt(n_atoms) * 1.5
    features.append(rg_estimate / 20.0)  # Normalize

    # Density descriptor
    density = n_atoms / (volume_estimate + 1e-8)
    features.append(np.log(density + 1e-8) / 5.0)  # Log normalize

    # Shape descriptor: bond angle std / dihedral std (flexibility measure)
    if len(bond_angles) > 0 and len(dihedral_angles) > 0:
        shape_desc = (np.std(bond_angles) + 1e-8) / (np.std(dihedral_angles) + 1e-8)
        features.append(np.log(shape_desc + 1e-8) / 5.0)
    else:
        features.append(0.0)

    return np.array(features)


def compute_3d_distance_features(mol_3d: Chem.Mol, cutoff: float = 5.0) -> np.ndarray:
    """
    Compute 3D distances between non-bonded atom pairs within cutoff.

    Returns:
        (n_atoms, n_atoms) array of distances (inf for distant pairs)
    """
    conf = mol_3d.GetConformer()
    n_atoms = mol_3d.GetNumAtoms()

    positions = []
    for i in range(n_atoms):
        pos = np.array(conf.GetAtomPosition(i))
        positions.append(pos)
    positions = np.array(positions)

    # Compute all pairwise distances
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)

    return distances


def process_polymer_v3(p_smiles: str, tg: float, use_periodic_edge: bool = True,
                      enable_3d: bool = True, distance_cutoff: float = 5.0) -> Optional[dict]:
    """
    Process a single polymer into graph data for PPFN-v3 with 3D features.

    Args:
        p_smiles: polymer SMILES
        tg: glass transition temperature
        use_periodic_edge: whether to add periodic edges between star atoms
        enable_3d: whether to generate 3D conformer features
        distance_cutoff: cutoff for 3D distance features

    Returns:
        Dictionary with graph data or None if processing fails
    """
    # Build trimer (same as v2)
    trimer_mol = build_trimer_graph(p_smiles)
    if trimer_mol is None:
        return None

    n_atoms = trimer_mol.GetNumAtoms()

    # Get atom features and edges (same as v2)
    atom_feats = []
    for atom in trimer_mol.GetAtoms():
        atom_feats.append(atom_features(atom))
    atom_feats = np.stack(atom_feats)

    edge_indices = []
    edge_feats_list = []  # 2D features

    for bond in trimer_mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        bf = bond_features(bond)
        edge_feats_list.append(bf)
        edge_feats_list.append(bf.copy())

    # 3D conformer generation
    conformer_success = False
    mol_3d = None
    distances_3d = None
    bond_lengths = None

    if enable_3d:
        mol_3d = generate_3d_conformer(trimer_mol)
        if mol_3d is not None:
            conformer_success = True
            # Extract 3D features
            bond_lengths, bond_angles, dihedral_angles = extract_3d_features(mol_3d)
            distances_3d = compute_3d_distance_features(mol_3d, cutoff=distance_cutoff)
        else:
            pass  # Silently fall back to 2D features

    # Add 3D features to edges
    if conformer_success and mol_3d is not None:
        conf = mol_3d.GetConformer()

        # Add bond length and 3D distance to edge features
        extended_edge_feats = []
        for idx, (i, j) in enumerate(edge_indices):
            base_feat = edge_feats_list[idx]
            # Add bond length
            if idx < len(edge_indices) // 2:  # Only for forward edges
                # Find corresponding bond
                length = 1.5  # Default
                for bond in trimer_mol.GetBonds():
                    bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    if (bi == i and bj == j) or (bi == j and bj == i):
                        pos_i = np.array(conf.GetAtomPosition(bi))
                        pos_j = np.array(conf.GetAtomPosition(bj))
                        length = np.linalg.norm(pos_i - pos_j)
                        break
                bond_length_feat = length / 2.0  # Normalize (~1.5 Å -> 0.75)
            else:
                bond_length_feat = 0.0  # Backward edge

            # 3D distance
            pos_i = np.array(conf.GetAtomPosition(i))
            pos_j = np.array(conf.GetAtomPosition(j))
            dist_3d = np.linalg.norm(pos_i - pos_j)
            dist_3d_feat = dist_3d / distance_cutoff  # Normalize by cutoff

            extended_feat = np.concatenate([
                base_feat,
                [bond_length_feat, dist_3d_feat]
            ])
            extended_edge_feats.append(extended_feat)

        edge_feats = np.stack(extended_edge_feats)
        current_num_edge_features = NUM_EDGE_FEATURES_V3
    else:
        # Use 2D only: pad edge features with zeros
        edge_feats = []
        for feat in edge_feats_list:
            padded_feat = np.concatenate([feat, [0.0, 0.0]])  # Pad with zeros
            edge_feats.append(padded_feat)
        edge_feats = np.stack(edge_feats)
        current_num_edge_features = NUM_EDGE_FEATURES_V3

    # Add periodic edges (same as v2)
    if use_periodic_edge:
        star_atoms = [a.GetIdx() for a in trimer_mol.GetAtoms() if a.GetSymbol() == '*']

        periodic_edges = []
        periodic_edge_feats = []
        if len(star_atoms) >= 2:
            for i in range(len(star_atoms)):
                for j in range(i + 1, len(star_atoms)):
                    s1, s2 = star_atoms[i], star_atoms[j]
                    periodic_edges.append([s1, s2])
                    periodic_edges.append([s2, s1])

                    # Periodic edge features (same as v2 + padding)
                    pe_feat = np.zeros(current_num_edge_features)
                    pe_feat[0] = 1.0  # single bond type
                    pe_feat[5] = 1.0  # in ring (periodic)
                    # 3D features are zero (no physical meaning)
                    periodic_edge_feats.append(pe_feat)
                    periodic_edge_feats.append(pe_feat)

        if periodic_edges:
            periodic_edge_index = np.array(periodic_edges, dtype=np.int64).T
            periodic_edge_features = np.stack(periodic_edge_feats)
            edge_index = np.concatenate([np.array(edge_indices, dtype=np.int64).T, periodic_edge_index], axis=1)
            edge_feats = np.concatenate([edge_feats, periodic_edge_features], axis=0)
        else:
            edge_index = np.array(edge_indices, dtype=np.int64).T
    else:
        edge_index = np.array(edge_indices, dtype=np.int64).T

    # Compute global features
    if conformer_success and mol_3d is not None:
        bond_lengths_extracted, bond_angles, dihedral_angles = extract_3d_features(mol_3d)
        global_feats = compute_global_features(bond_lengths_extracted, bond_angles, dihedral_angles, n_atoms)
    else:
        # Use default global features
        global_feats = np.zeros(NUM_GLOBAL_FEATURES_V3)

    return {
        'atom_features': atom_feats,
        'edge_index': edge_index,
        'edge_features': edge_feats,
        'global_features': global_feats,
        'tg': tg,
        'p_smiles': p_smiles,
        'n_atoms': n_atoms,
        'n_edges': edge_index.shape[1],
        'conformer_success': conformer_success,
    }


class PPFNv3Dataset(Dataset):
    """Dataset for PPFN-v3 with 3D conformer features."""

    def __init__(self, csv_path: str, split: str = 'train',
                 train_ratio: float = 0.72, val_ratio: float = 0.08,
                 seed: int = 42, use_periodic_edge: bool = True, enable_3d: bool = True):
        self.split = split
        self.use_periodic_edge = use_periodic_edge
        self.enable_3d = enable_3d

        df = pd.read_csv(csv_path)

        # Deterministic split
        np.random.seed(seed)
        n = len(df)
        indices = np.random.permutation(n)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if split == 'train':
            sel_indices = indices[:n_train]
        elif split == 'val':
            sel_indices = indices[n_train:n_train + n_val]
        else:  # test
            sel_indices = indices[n_train + n_val:]

        df_split = df.iloc[sel_indices].reset_index(drop=True)

        # Process all polymers
        self.samples = []
        self.failed = 0
        self.conformer_success_count = 0

        print(f"Processing {len(df_split)} {split} samples (periodic_edge={use_periodic_edge}, 3d={enable_3d})...")

        for idx, row in df_split.iterrows():
            result = process_polymer_v3(
                row['Smiles'], row['Tg'], use_periodic_edge, enable_3d
            )
            if result is not None:
                self.samples.append(result)
                if result['conformer_success']:
                    self.conformer_success_count += 1
            else:
                self.failed += 1

            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{len(df_split)}...")

        print(f"  Success: {len(self.samples)}, Failed: {self.failed}")
        print(f"  Conformer success rate: {self.conformer_success_count}/{len(self.samples)} ({100*self.conformer_success_count/len(self.samples):.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'atom_features': torch.FloatTensor(sample['atom_features']),
            'edge_index': torch.LongTensor(sample['edge_index']),
            'edge_features': torch.FloatTensor(sample['edge_features']),
            'global_features': torch.FloatTensor(sample['global_features']),
            'tg': torch.FloatTensor([sample['tg']]),
            'p_smiles': sample['p_smiles'],
            'n_atoms': sample['n_atoms'],
        }


def collate_fn_v3(batch):
    """Custom collate function for PPFN-v3 batching."""
    atom_features_list = []
    edge_index_list = []
    edge_features_list = []
    global_features_list = []
    tg_list = []
    batch_atom_indices = []
    p_smiles_list = []
    n_atoms_offset = 0

    for i, sample in enumerate(batch):
        n_atoms = sample['n_atoms']
        atom_features_list.append(sample['atom_features'])
        edge_index_list.append(sample['edge_index'] + n_atoms_offset)
        edge_features_list.append(sample['edge_features'])
        global_features_list.append(sample['global_features'])
        tg_list.append(sample['tg'])
        batch_atom_indices.append(torch.full((n_atoms,), i, dtype=torch.long))
        p_smiles_list.append(sample['p_smiles'])
        n_atoms_offset += n_atoms

    return {
        'atom_features': torch.cat(atom_features_list, dim=0),
        'edge_index': torch.cat(edge_index_list, dim=1),
        'edge_features': torch.cat(edge_features_list, dim=0),
        'global_features': torch.stack(global_features_list, dim=0),
        'tg': torch.cat(tg_list, dim=0),
        'batch_atom': torch.cat(batch_atom_indices, dim=0),
        'p_smiles_list': p_smiles_list,
    }


if __name__ == '__main__':
    # Quick test
    csv_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    dataset = PPFNv3Dataset(csv_path, split='train', enable_3d=True)

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Atoms: {sample['n_atoms']}")
        print(f"  Edges: {sample['edge_index'].shape[1]}")
        print(f"  Atom features: {sample['atom_features'].shape}")
        print(f"  Edge features: {sample['edge_features'].shape}")
        print(f"  Global features: {sample['global_features'].shape}")
        print(f"  Tg: {sample['tg'].item()}")

    # Test collation
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_v3)
    batch = next(iter(loader))
    print(f"\nBatch:")
    print(f"  Total atoms: {batch['atom_features'].shape[0]}")
    print(f"  Total edges: {batch['edge_index'].shape[1]}")
    print(f"  Global features: {batch['global_features'].shape}")
    print(f"  Batch size: {batch['tg'].shape[0]}")
