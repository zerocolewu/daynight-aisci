"""
Data loading utilities for HEPT polymer Tg prediction.
Handles p-SMILES parsing, 3D conformer generation, and graph construction.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

# Add dataset path
DATASET_PATH = Path("/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv")


def cyclize_p_smiles(p_smiles: str, keep_bond_type: bool = True) -> Chem.Mol:
    """
    Convert p-SMILES to cyclized molecule by connecting the two * sites.
    This creates a ring structure representing the polymer repeat unit.
    """
    mol = Chem.MolFromSmiles(p_smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {p_smiles}")

    star_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == "*"]
    if len(star_atoms) != 2:
        # If no star atoms, return as is (might already be cyclized or different format)
        return mol

    star_idx0, star_idx1 = star_atoms[0].GetIdx(), star_atoms[1].GetIdx()

    nei0 = star_atoms[0].GetNeighbors()
    nei1 = star_atoms[1].GetNeighbors()

    if len(nei0) != 1 or len(nei1) != 1:
        return mol  # Return original if structure is unexpected

    n0, n1 = nei0[0].GetIdx(), nei1[0].GetIdx()

    # Check if trying to create self-bond
    if n0 == n1:
        # Just remove the star atoms without creating bond
        ed_mol = Chem.EditableMol(mol)
        for idx in sorted([star_idx0, star_idx1], reverse=True):
            ed_mol.RemoveAtom(int(idx))
        final_mol = ed_mol.GetMol()
        try:
            Chem.SanitizeMol(final_mol)
        except:
            return mol
        return final_mol

    bond_type = Chem.BondType.SINGLE
    if keep_bond_type:
        b0 = mol.GetBondBetweenAtoms(star_idx0, n0)
        b1 = mol.GetBondBetweenAtoms(star_idx1, n1)
        if b0 is not None and b1 is not None and b0.GetBondType() == b1.GetBondType():
            bond_type = b0.GetBondType()

    ed_mol = Chem.EditableMol(mol)

    if mol.GetBondBetweenAtoms(n0, n1) is None:
        ed_mol.AddBond(int(n0), int(n1), order=bond_type)

    for idx in sorted([star_idx0, star_idx1], reverse=True):
        ed_mol.RemoveAtom(int(idx))

    final_mol = ed_mol.GetMol()
    try:
        Chem.SanitizeMol(final_mol)
    except:
        return mol  # Return original if sanitization fails

    return final_mol


def generate_optimized_conformer(mol: Chem.Mol, n_conformers: int = 5, max_attempts: int = 20) -> Tuple[Optional[np.ndarray], Optional[Chem.Mol]]:
    """
    Generate and optimize 3D conformer using ETKDG + MMFF94.
    Generates multiple conformers, optimizes each, and selects lowest energy.

    Returns:
        positions: Positions array of shape (num_atoms, 3) or None if failed
        mol_with_h: Molecule with hydrogens (for descriptor calculation)
    """
    try:
        mol = Chem.AddHs(mol)

        # Try ETKDGv3 with multiple seeds
        best_conf = None
        best_energy = float('inf')
        best_positions = None

        for seed in range(max_attempts):
            try:
                # Generate conformer
                res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), randomSeed=seed, maxAttempts=50)
                if res == -1:
                    continue

                # MMFF optimization
                try:
                    opt_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=200, confId=res)
                    if opt_result == -1:
                        continue
                except:
                    continue

                # Calculate energy
                try:
                    mp = AllChem.MMFFGetMoleculeProperties(mol)
                    if mp is None:
                        continue
                    ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=res)
                    if ff is None:
                        continue
                    energy = ff.CalcEnergy()

                    if energy < best_energy:
                        best_energy = energy
                        conf = mol.GetConformer(res)
                        best_positions = conf.GetPositions()
                        best_conf = res
                except:
                    continue

            except:
                continue

        if best_positions is not None:
            return best_positions, mol

        # Fallback: use 2D coordinates with z=0
        try:
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            positions = conf.GetPositions()
            return positions, mol
        except:
            return None, None

    except:
        return None, None


def compute_sp2_sp3_fractions(mol: Chem.Mol) -> Tuple[float, float]:
    """Compute fraction of sp2 and sp3 carbons."""
    sp2_count = 0
    sp3_count = 0
    total_c = 0

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:  # Carbon
            total_c += 1
            hyb = atom.GetHybridization()
            if hyb == Chem.rdchem.HybridizationType.SP2:
                sp2_count += 1
            elif hyb == Chem.rdchem.HybridizationType.SP3:
                sp3_count += 1

    if total_c == 0:
        return 0.0, 0.0

    return sp2_count / total_c, sp3_count / total_c


def compute_enhanced_molecular_features(mol: Chem.Mol) -> List[float]:
    """
    Compute enhanced molecular-level features for Tg prediction.
    These features capture properties known to correlate with Tg.
    """
    features = []

    try:
        # Basic descriptors
        features.append(Descriptors.MolWt(mol))  # Molecular weight
        features.append(Descriptors.MolLogP(mol))  # Lipophilicity
        features.append(Descriptors.TPSA(mol))  # Topological polar surface area

        # Flexibility indicators
        features.append(Descriptors.NumRotatableBonds(mol))  # Chain flexibility
        features.append(rdMolDescriptors.CalcNumAromaticRings(mol))  # Rigidity
        features.append(rdMolDescriptors.CalcNumAliphaticRings(mol))  # Flexibility

        # Hybridization
        sp2_frac, sp3_frac = compute_sp2_sp3_fractions(mol)
        features.append(sp2_frac)
        features.append(sp3_frac)

        # Hydrogen bonding
        features.append(Descriptors.NumHAcceptors(mol))
        features.append(Descriptors.NumHDonors(mol))

        # Ring and structural features
        features.append(rdMolDescriptors.CalcNumRings(mol))
        features.append(Descriptors.FractionCSP3(mol))

        # Additional descriptors relevant to Tg
        features.append(Descriptors.HeavyAtomCount(mol))
        features.append(Descriptors.NumHeteroatoms(mol))
        features.append(Descriptors.RingCount(mol))

        # Molecular complexity
        features.append(Descriptors.BertzCT(mol))  # Bertz complexity

        # Additional rigidity measures
        features.append(Descriptors.NumAromaticHeterocycles(mol))
        features.append(Descriptors.NumSaturatedRings(mol))

    except Exception as e:
        # Return zeros if calculation fails
        features = [0.0] * 18

    return features


# Number of enhanced molecular features
NUM_MOLECULAR_FEATURES = 18


def get_atom_features(atom: Chem.Atom) -> List[float]:
    """
    Extract atom features:
    - Atomic number (one-hot, up to 100)
    - Degree (0-5)
    - Formal charge (-2 to 2)
    - Hybridization (0-6)
    - Is aromatic (0/1)
    - Num hydrogens (0-4)
    """
    # Atomic number one-hot (considering common elements in polymers)
    atomic_num = atom.GetAtomicNum()
    atom_types = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H, C, N, O, F, P, S, Cl, Br, I
    atom_type_feat = [1 if atomic_num == t else 0 for t in atom_types]
    if not any(atom_type_feat):
        atom_type_feat = [0] * len(atom_types) + [1]  # Other
    else:
        atom_type_feat = atom_type_feat + [0]  # Other

    # Degree (0-5, capped)
    degree = min(atom.GetDegree(), 5)
    degree_feat = [1 if i == degree else 0 for i in range(6)]

    # Formal charge (-2 to 2)
    charge = atom.GetFormalCharge()
    charge = max(-2, min(2, charge))
    charge_feat = [1 if i == charge + 2 else 0 for i in range(5)]

    # Hybridization
    hyb_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    hyb = atom.GetHybridization()
    hyb_feat = [1 if hyb == t else 0 for t in hyb_types]
    if not any(hyb_feat):
        hyb_feat = hyb_feat + [1]  # Other
    else:
        hyb_feat = hyb_feat + [0]

    # Aromatic
    aromatic_feat = [1 if atom.GetIsAromatic() else 0]

    # Num hydrogens
    num_h = min(atom.GetTotalNumHs(), 4)
    num_h_feat = [1 if i == num_h else 0 for i in range(5)]

    # In ring
    in_ring_feat = [1 if atom.IsInRing() else 0]

    # Mass (normalized)
    mass_feat = [atom.GetMass() / 100.0]

    features = (
        atom_type_feat +
        degree_feat +
        charge_feat +
        hyb_feat +
        aromatic_feat +
        num_h_feat +
        in_ring_feat +
        mass_feat
    )

    return features


def get_bond_features(bond: Chem.Bond) -> List[float]:
    """
    Extract bond features:
    - Bond type (single, double, triple, aromatic)
    - Is conjugated
    - Is in ring
    """
    bond_type = bond.GetBondType()

    # Bond type one-hot
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    bond_type_feat = [1 if bond_type == t else 0 for t in bond_types]
    if not any(bond_type_feat):
        bond_type_feat = [0, 0, 0, 0, 1]  # Other
    else:
        bond_type_feat = bond_type_feat + [0]

    # Conjugated
    conj_feat = [1 if bond.GetIsConjugated() else 0]

    # In ring
    ring_feat = [1 if bond.IsInRing() else 0]

    # Stereo (not typically relevant for polymers, but included)
    stereo = bond.GetStereo()
    stereo_types = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]
    stereo_feat = [1 if stereo == t else 0 for t in stereo_types]

    features = bond_type_feat + conj_feat + ring_feat + stereo_feat
    return features


def mol_to_graph(mol: Chem.Mol, tg_value: float, smiles: str) -> Optional[Data]:
    """
    Convert RDKit molecule to PyTorch Geometric Data object.
    """
    if mol is None:
        return None

    # Generate 3D coordinates with MMFF optimization (function adds H internally)
    positions_3d, mol_with_conf = generate_optimized_conformer(mol)
    if positions_3d is None:
        return None

    # Remove hydrogens for efficiency (after conformer generation)
    mol_no_h = Chem.RemoveHs(mol_with_conf)
    num_atoms_no_h = mol_no_h.GetNumAtoms()

    if num_atoms_no_h == 0:
        return None

    # Atom features
    atom_features = []
    for atom in mol_no_h.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(atom_features, dtype=torch.float)

    # Bond features and edge indices
    edge_indices = []
    edge_features = []

    for bond in mol_no_h.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_indices.append([i, j])
        edge_indices.append([j, i])

        bond_feat = get_bond_features(bond)
        edge_features.append(bond_feat)
        edge_features.append(bond_feat)

    if len(edge_indices) == 0:
        # Single atom molecule
        edge_index = torch.tensor([[], []], dtype=torch.long)
        edge_attr = torch.tensor([], dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # 3D positions (for equivariant operations)
    # Get positions for non-H atoms
    try:
        conf = mol_no_h.GetConformer()
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
    except:
        pos = torch.zeros((num_atoms_no_h, 3), dtype=torch.float)

    # Compute enhanced molecular features (18 features total)
    mol_features = compute_enhanced_molecular_features(mol_no_h)
    mol_features_tensor = torch.tensor(mol_features, dtype=torch.float)

    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        mol_features=mol_features_tensor,  # Enhanced molecular features
        y=torch.tensor([tg_value], dtype=torch.float),
        smiles=smiles,
        num_nodes=num_atoms_no_h,
    )

    return data


def compute_molecular_features(mol: Chem.Mol) -> List[float]:
    """
    Compute additional molecular-level features for long-range module.
    """
    features = []

    # Basic descriptors
    features.append(Descriptors.MolWt(mol))
    features.append(Descriptors.MolLogP(mol))
    features.append(Descriptors.TPSA(mol))
    features.append(Descriptors.NumRotatableBonds(mol))
    features.append(Descriptors.NumHAcceptors(mol))
    features.append(Descriptors.NumHDonors(mol))
    features.append(rdMolDescriptors.CalcNumRings(mol))
    features.append(Descriptors.FractionCSP3(mol))

    return features


class PolymerTgDataset(Dataset):
    """
    Dataset for polymer Tg prediction from p-SMILES.
    """

    def __init__(
        self,
        csv_path: str = None,
        data_list: List[Data] = None,
        transform=None,
        pre_transform=None,
    ):
        self.transform = transform
        self.pre_transform = pre_transform

        if data_list is not None:
            self.data_list = data_list
        else:
            csv_path = csv_path or str(DATASET_PATH)
            self.data_list = self._load_data(csv_path)

    def _load_data(self, csv_path: str) -> List[Data]:
        """Load and process data from CSV file."""
        df = pd.read_csv(csv_path)
        print(f"Loading {len(df)} polymers from {csv_path}")

        data_list = []
        failed = 0

        for idx, row in df.iterrows():
            smiles = row['Smiles']
            tg = row['Tg']

            try:
                # Cyclize p-SMILES
                mol = cyclize_p_smiles(smiles)

                # Convert to graph
                data = mol_to_graph(mol, tg, smiles)

                if data is not None:
                    data_list.append(data)
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                continue

        print(f"Successfully processed: {len(data_list)}, Failed: {failed}")
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.transform:
            data = self.transform(data)
        return data


def stratified_split(
    data_list: List[Data],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_bins: int = 10,
    seed: int = 42
) -> Tuple[List[Data], List[Data], List[Data]]:
    """
    Perform stratified split based on Tg values.
    """
    np.random.seed(seed)

    # Get Tg values
    tg_values = np.array([data.y.item() for data in data_list])

    # Create bins for stratification
    bins = np.percentile(tg_values, np.linspace(0, 100, num_bins + 1))
    bin_indices = np.digitize(tg_values, bins) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    train_list, val_list, test_list = [], [], []

    for bin_idx in range(num_bins):
        bin_mask = bin_indices == bin_idx
        bin_data = [data_list[i] for i in range(len(data_list)) if bin_mask[i]]

        np.random.shuffle(bin_data)
        n = len(bin_data)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_list.extend(bin_data[:n_train])
        val_list.extend(bin_data[n_train:n_train + n_val])
        test_list.extend(bin_data[n_train + n_val:])

    print(f"Split sizes - Train: {len(train_list)}, Val: {len(val_list)}, Test: {len(test_list)}")
    return train_list, val_list, test_list


def collate_fn(data_list: List[Data]) -> Batch:
    """Custom collate function for PyG Data objects."""
    return Batch.from_data_list(data_list)


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    """
    # Load full dataset
    dataset = PolymerTgDataset()

    # Stratified split
    train_data, val_data, test_data = stratified_split(
        dataset.data_list,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=seed
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class RandomRotate:
    """Data augmentation: random rotation of 3D coordinates."""

    def __init__(self, rotate_prob: float = 0.5):
        self.rotate_prob = rotate_prob

    def __call__(self, data: Data) -> Data:
        if torch.rand(1).item() > self.rotate_prob:
            return data

        # Random rotation matrix
        angle = torch.rand(3) * 2 * np.pi
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)

        # Rotation around x-axis
        Rx = torch.tensor([
            [1, 0, 0],
            [0, cos_a[0], -sin_a[0]],
            [0, sin_a[0], cos_a[0]]
        ], dtype=torch.float)

        # Rotation around y-axis
        Ry = torch.tensor([
            [cos_a[1], 0, sin_a[1]],
            [0, 1, 0],
            [-sin_a[1], 0, cos_a[1]]
        ], dtype=torch.float)

        # Rotation around z-axis
        Rz = torch.tensor([
            [cos_a[2], -sin_a[2], 0],
            [sin_a[2], cos_a[2], 0],
            [0, 0, 1]
        ], dtype=torch.float)

        R = Rz @ Ry @ Rx
        data.pos = data.pos @ R.T

        return data


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test one batch
    for batch in train_loader:
        print(f"\nSample batch:")
        print(f"  x shape: {batch.x.shape}")
        print(f"  edge_index shape: {batch.edge_index.shape}")
        print(f"  pos shape: {batch.pos.shape}")
        print(f"  y shape: {batch.y.shape}")
        print(f"  batch shape: {batch.batch.shape}")
        break
