"""
Data preprocessing for Polymer JEPA Pre-training.
Converts PI1M p-SMILES and Tg p-SMILES to Trimer PyG datasets.
"""

import torch
import pandas as pd
import numpy as np
import os
import logging
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import rdmolops

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# --- Atom features (30 dims) ---
ATOM_TYPES = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'Si', 'B']
DEGREES = [0, 1, 2, 3, 4, 5]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
]
H_COUNTS = [0, 1, 2, 3, 4, 5]
CHIRALITY = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
]
ATOM_FEATURE_DIM = len(ATOM_TYPES) + len(DEGREES) + len(HYBRIDIZATIONS) + 1 + 1 + len(H_COUNTS) + len(CHIRALITY)

BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_FEATURE_DIM = len(BOND_TYPES) + 1 + 1

# ru_index feature adds 3 more dims (one-hot for RU0, RU1, RU2)
RU_FEATURE_DIM = 3
TOTAL_ATOM_FEATURE_DIM = ATOM_FEATURE_DIM + RU_FEATURE_DIM


def one_hot(value, choices):
    vec = [0] * len(choices)
    try:
        idx = choices.index(value)
        vec[idx] = 1
    except ValueError:
        pass
    return vec


def get_atom_features(atom):
    symbol = atom.GetSymbol()
    atom_type = one_hot(symbol, ATOM_TYPES)
    degree = one_hot(atom.GetTotalDegree(), DEGREES)
    hyb = atom.GetHybridization()
    hybridization = one_hot(hyb, HYBRIDIZATIONS)
    aromatic = [int(atom.GetIsAromatic())]
    ring = [int(atom.IsInRing())]
    h_count = one_hot(atom.GetTotalNumHs(), H_COUNTS)
    chirality = one_hot(atom.GetChiralTag(), CHIRALITY)
    return atom_type + degree + hybridization + aromatic + ring + h_count + chirality


def get_bond_features(bond):
    bond_type = one_hot(bond.GetBondType(), BOND_TYPES)
    conjugated = [int(bond.GetIsConjugated())]
    ring_feat = [int(bond.IsInRing())]
    return bond_type + conjugated + ring_feat


def safe_sanitize(mol):
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        pass
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        return True
    except Exception:
        pass
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)
        return True
    except Exception:
        return False


def find_star_atoms(mol):
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]


def get_star_neighbors(mol, star_indices):
    star_to_neighbor = {}
    for star_idx in star_indices:
        atom = mol.GetAtomWithIdx(star_idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() != 0:
                star_to_neighbor[star_idx] = nbr.GetIdx()
                break
    return star_to_neighbor


def psmiles_to_trimer(p_smiles):
    """Convert p-SMILES to trimer (3 repeat units connected).
    Returns (mol, unit_boundaries) or (None, None) on failure.
    unit_boundaries: list of (start_idx, end_idx) for each of the 3 RUs.
    """
    try:
        mol = Chem.MolFromSmiles(p_smiles)
        if mol is None:
            return None, None

        star_indices = find_star_atoms(mol)
        if len(star_indices) != 2:
            return None, None

        star_to_neighbor = get_star_neighbors(mol, star_indices)
        if len(star_to_neighbor) != 2:
            return None, None

        stars = list(star_to_neighbor.keys())
        head_star, tail_star = stars[0], stars[1]
        head_nbr_orig = star_to_neighbor[head_star]
        tail_nbr_orig = star_to_neighbor[tail_star]

        rw_mol = Chem.RWMol(mol)
        remove_order = sorted([head_star, tail_star], reverse=True)
        for idx in remove_order:
            rw_mol.RemoveAtom(idx)

        atoms_removed = sorted([head_star, tail_star])

        def update_idx(idx):
            new_idx = idx
            for removed_idx in atoms_removed:
                if idx > removed_idx:
                    new_idx -= 1
            return new_idx

        monomer_head = update_idx(head_nbr_orig)
        monomer_tail = update_idx(tail_nbr_orig)
        monomer = rw_mol.GetMol()
        n_monomer_atoms = monomer.GetNumAtoms()

        combined = Chem.RWMol(Chem.Mol())
        unit_boundaries = []

        for unit in range(3):
            offset = combined.GetNumAtoms()
            unit_boundaries.append((offset, offset + n_monomer_atoms))
            for atom in monomer.GetAtoms():
                new_atom = Chem.Atom(atom.GetSymbol())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                new_atom.SetIsAromatic(atom.GetIsAromatic())
                new_atom.SetChiralTag(atom.GetChiralTag())
                new_atom.SetHybridization(atom.GetHybridization())
                new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
                combined.AddAtom(new_atom)
            for bond in monomer.GetBonds():
                begin = bond.GetBeginAtomIdx() + offset
                end = bond.GetEndAtomIdx() + offset
                combined.AddBond(begin, end, bond.GetBondType())

        for i in range(2):
            tail_of_i = unit_boundaries[i][0] + monomer_tail
            head_of_next = unit_boundaries[i + 1][0] + monomer_head
            combined.AddBond(tail_of_i, head_of_next, Chem.BondType.SINGLE)

        result = combined.GetMol()
        if not safe_sanitize(result):
            return None, None

        return result, unit_boundaries
    except Exception as e:
        return None, None


def mol_to_graph(mol, unit_boundaries, y_value=None):
    """Convert RDKit trimer mol to PyG Data with ru_index features.

    Args:
        mol: RDKit molecule (trimer)
        unit_boundaries: list of (start, end) for each RU
        y_value: optional target value

    Returns:
        PyG Data object or None on failure
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return None

    # Atom features + ru_index
    atom_features = []
    for atom in mol.GetAtoms():
        base_feat = get_atom_features(atom)
        atom_idx = atom.GetIdx()
        # Determine which RU this atom belongs to
        ru_onehot = [0, 0, 0]
        for ru_idx, (start, end) in enumerate(unit_boundaries):
            if start <= atom_idx < end:
                ru_onehot[ru_idx] = 1
                break
        atom_features.append(base_feat + ru_onehot)

    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge features
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        feat = get_bond_features(bond)
        edge_indices.append([begin, end])
        edge_indices.append([end, begin])
        edge_attrs.append(feat)
        edge_attrs.append(feat)

    if len(edge_indices) == 0:
        edge_indices = [[0, 0]]
        edge_attrs = [[0.0] * BOND_FEATURE_DIM]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=n_atoms,
    )

    if y_value is not None:
        data.y = torch.tensor([y_value], dtype=torch.float)

    # Store unit boundaries for JEPA masking
    data.unit_boundaries = unit_boundaries

    return data


# --- PI1M Trimer Dataset ---

class PI1MTrimerDataset(InMemoryDataset):
    """PI1M pre-training dataset with trimer representation."""

    def __init__(self, root, csv_path, max_samples=None, transform=None,
                 pre_transform=None, pre_filter=None):
        self.csv_path = csv_path
        self.max_samples = max_samples
        super(PI1MTrimerDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['PI1M.csv']

    @property
    def processed_file_names(self):
        name = 'pi1m_trimer.pt'
        if self.max_samples is not None:
            name = f'pi1m_trimer_{self.max_samples}.pt'
        return [name]

    def download(self):
        pass

    def process(self):
        logger.info(f"Loading PI1M from {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        smiles_list = df['SMILES'].tolist()

        if self.max_samples is not None:
            smiles_list = smiles_list[:self.max_samples]
            logger.info(f"Using subset of {len(smiles_list)} samples")

        data_list = []
        failed = 0
        total = len(smiles_list)

        for i, smi in enumerate(smiles_list):
            if i % 50000 == 0:
                logger.info(f"Processing PI1M: {i}/{total} ({len(data_list)} success, {failed} failed)")

            mol, unit_boundaries = psmiles_to_trimer(smi)
            if mol is None:
                failed += 1
                continue

            graph = mol_to_graph(mol, unit_boundaries)
            if graph is None:
                failed += 1
                continue

            data_list.append(graph)

        logger.info(f"PI1M processing complete: {len(data_list)} success, {failed} failed out of {total}")

        if len(data_list) == 0:
            raise ValueError("No valid graphs found in PI1M!")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"Saved PI1M dataset to {self.processed_paths[0]}")


# --- Tg Trimer Dataset ---

class TgTrimerDataset(InMemoryDataset):
    """Tg fine-tuning dataset with trimer representation."""

    def __init__(self, root, csv_path, transform=None,
                 pre_transform=None, pre_filter=None):
        self.csv_path = csv_path
        super(TgTrimerDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['Tg.csv']

    @property
    def processed_file_names(self):
        return ['tg_trimer_jepa.pt']

    def download(self):
        pass

    def process(self):
        logger.info(f"Loading Tg from {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        smiles_list = df['Smiles'].tolist()
        tg_values = df['Tg'].tolist()

        data_list = []
        failed = 0

        for i, (smi, tg) in enumerate(zip(smiles_list, tg_values)):
            mol, unit_boundaries = psmiles_to_trimer(smi)
            if mol is None:
                failed += 1
                continue

            graph = mol_to_graph(mol, unit_boundaries, y_value=tg)
            if graph is None:
                failed += 1
                continue

            data_list.append(graph)

        logger.info(f"Tg processing: {len(data_list)} success, {failed} failed out of {len(smiles_list)}")

        if len(data_list) == 0:
            raise ValueError("No valid graphs found in Tg!")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"Saved Tg dataset to {self.processed_paths[0]}")


def get_data_splits(dataset, seed=71, train_ratio=0.72, val_ratio=0.08):
    """Split dataset deterministically."""
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, n))

    return train_indices, val_indices, test_indices


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pi1m_path', type=str,
                        default='/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv')
    parser.add_argument('--tg_path', type=str,
                        default='/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv')
    parser.add_argument('--output_dir', type=str, default='./data')
    parser.add_argument('--max_pi1m_samples', type=int, default=300000,
                        help='Max PI1M samples (None for all)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Process PI1M
    logger.info("=" * 60)
    logger.info("Processing PI1M dataset...")
    pi1m_dataset = PI1MTrimerDataset(
        root=args.output_dir,
        csv_path=args.pi1m_path,
        max_samples=args.max_pi1m_samples,
    )
    logger.info(f"PI1M dataset size: {len(pi1m_dataset)}")

    # Process Tg
    logger.info("=" * 60)
    logger.info("Processing Tg dataset...")
    tg_dataset = TgTrimerDataset(
        root=args.output_dir,
        csv_path=args.tg_path,
    )
    logger.info(f"Tg dataset size: {len(tg_dataset)}")

    # Verify splits
    train_idx, val_idx, test_idx = get_data_splits(tg_dataset)
    logger.info(f"Tg splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    logger.info("Data preprocessing complete!")
