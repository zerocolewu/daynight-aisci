"""
PCR-v2 Dataset: Dual-view monomer + trimer graphs per polymer.
Each polymer gets two graph views processed through the same GIN encoder.
"""

import os
import torch
import pandas as pd
import numpy as np
import logging
from torch_geometric.data import Data, InMemoryDataset, Batch
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Atom feature dimensions
ATOM_TYPES = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'Si', 'B']  # 10
DEGREES = [0, 1, 2, 3, 4, 5]  # 6
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
]  # 4
H_COUNTS = [0, 1, 2, 3, 4, 5]  # 6
CHIRALITY = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
]  # 2

ATOM_FEATURE_DIM = len(ATOM_TYPES) + len(DEGREES) + len(HYBRIDIZATIONS) + 1 + 1 + len(H_COUNTS) + len(CHIRALITY)  # 30

# Bond features
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]  # 4
BOND_FEATURE_DIM = len(BOND_TYPES) + 1 + 1  # 6 (bond_type + conjugated + ring)


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
    ring = [int(bond.IsInRing())]
    return bond_type + conjugated + ring


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
    star_atoms = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            star_atoms.append(atom.GetIdx())
    return star_atoms


def get_star_neighbors(mol, star_indices):
    star_to_neighbor = {}
    for star_idx in star_indices:
        atom = mol.GetAtomWithIdx(star_idx)
        neighbors = atom.GetNeighbors()
        for nbr in neighbors:
            if nbr.GetAtomicNum() != 0:
                star_to_neighbor[star_idx] = nbr.GetIdx()
                break
    return star_to_neighbor


def smiles_to_monomer(p_smiles):
    """
    Convert p-SMILES to monomer (single repeat unit with * replaced by H).
    Returns RDKit Mol or None.
    """
    try:
        mol = Chem.MolFromSmiles(p_smiles)
        if mol is None:
            return None

        star_indices = find_star_atoms(mol)
        if len(star_indices) != 2:
            return None

        # Replace * atoms with H atoms
        rw_mol = Chem.RWMol(mol)
        for star_idx in sorted(star_indices, reverse=True):
            atom = rw_mol.GetAtomWithIdx(star_idx)
            neighbors = [n for n in atom.GetNeighbors()]
            for nbr in neighbors:
                bond = rw_mol.GetBondBetweenAtoms(star_idx, nbr.GetIdx())
                if bond is not None:
                    rw_mol.RemoveBond(star_idx, nbr.GetIdx())
            rw_mol.RemoveAtom(star_idx)

        result = rw_mol.GetMol()
        if not safe_sanitize(result):
            return None
        return result
    except Exception as e:
        logger.warning(f"Failed monomer conversion for {p_smiles}: {e}")
        return None


def smiles_to_trimer(p_smiles):
    """
    Convert p-SMILES to trimer (3 repeat units connected).
    Terminal * positions are capped with H.
    Returns RDKit Mol or None.
    """
    try:
        mol = Chem.MolFromSmiles(p_smiles)
        if mol is None:
            return None

        star_indices = find_star_atoms(mol)
        if len(star_indices) != 2:
            return None

        star_to_neighbor = get_star_neighbors(mol, star_indices)
        if len(star_to_neighbor) != 2:
            return None

        stars = list(star_to_neighbor.keys())
        head_star, tail_star = stars[0], stars[1]
        head_nbr_orig = star_to_neighbor[head_star]
        tail_nbr_orig = star_to_neighbor[tail_star]

        # Create the monomer without star atoms
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

        # Build trimer by combining 3 copies
        combined = Chem.RWMol(Chem.Mol())

        for unit in range(3):
            offset = combined.GetNumAtoms()
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

        # Connect units: unit[i].tail -> unit[i+1].head
        for i in range(2):
            tail_of_i = i * n_monomer_atoms + monomer_tail
            head_of_next = (i + 1) * n_monomer_atoms + monomer_head
            combined.AddBond(tail_of_i, head_of_next, Chem.BondType.SINGLE)

        result = combined.GetMol()
        if not safe_sanitize(result):
            logger.warning(f"Failed sanitization for trimer: {p_smiles}")
            return None

        return result
    except Exception as e:
        logger.warning(f"Failed trimer conversion for {p_smiles}: {e}")
        return None


def mol_to_graph_data(mol):
    """Convert RDKit Mol to PyG Data object."""
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

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
        edge_attrs = [[0] * BOND_FEATURE_DIM]

    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_atoms)


class PCRDualViewDataset:
    """
    Dataset that stores monomer and trimer graphs for each polymer.
    Each entry is a pair (monomer_graph, trimer_graph) with shared Tg target.
    Uses simple list storage instead of InMemoryDataset for easier indexing.
    """

    def __init__(self, root, csv_path):
        self.root = root
        self.csv_path = csv_path
        self.monomer_graphs = []
        self.trimer_graphs = []
        self.targets = []
        self.valid_mask = []

        processed_path = os.path.join(root, 'dual_view_data.pt')
        if os.path.exists(processed_path):
            logger.info(f"Loading pre-processed data from {processed_path}")
            data = torch.load(processed_path, weights_only=False)
            self.monomer_graphs = data['monomer_graphs']
            self.trimer_graphs = data['trimer_graphs']
            self.targets = data['targets']
            self.valid_mask = data['valid_mask']
        else:
            self._process_and_save(processed_path)

        logger.info(f"Dataset: {len(self.targets)} total, {sum(self.valid_mask)} valid pairs")
        logger.info(f"Node features: {ATOM_FEATURE_DIM}, Edge features: {BOND_FEATURE_DIM}")

    def _process_and_save(self, save_path):
        df = pd.read_csv(self.csv_path)
        smiles_list = df['Smiles'].tolist()
        tg_values = df['Tg'].tolist()

        failed = 0
        for i, (smi, tg) in enumerate(zip(smiles_list, tg_values)):
            mono_mol = smiles_to_monomer(smi)
            tri_mol = smiles_to_trimer(smi)

            mono_graph = mol_to_graph_data(mono_mol)
            tri_graph = mol_to_graph_data(tri_mol)

            if mono_graph is None or tri_graph is None:
                failed += 1
                # Create minimal dummy graphs (will be masked out)
                self.monomer_graphs.append(Data(
                    x=torch.zeros(2, ATOM_FEATURE_DIM),
                    edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                    edge_attr=torch.zeros(2, BOND_FEATURE_DIM),
                    num_nodes=2,
                ))
                self.trimer_graphs.append(Data(
                    x=torch.zeros(2, ATOM_FEATURE_DIM),
                    edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                    edge_attr=torch.zeros(2, BOND_FEATURE_DIM),
                    num_nodes=2,
                ))
                self.targets.append(tg)
                self.valid_mask.append(False)
            else:
                self.monomer_graphs.append(mono_graph)
                self.trimer_graphs.append(tri_graph)
                self.targets.append(tg)
                self.valid_mask.append(True)

        logger.info(f"Processed {len(smiles_list)} polymers, {failed} failures")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'monomer_graphs': self.monomer_graphs,
            'trimer_graphs': self.trimer_graphs,
            'targets': self.targets,
            'valid_mask': self.valid_mask,
        }, save_path)

    def __len__(self):
        return len(self.targets)

    @property
    def num_node_features(self):
        return ATOM_FEATURE_DIM

    @property
    def num_edge_features(self):
        return BOND_FEATURE_DIM


def get_data_splits(dataset_len, seed=71, train_ratio=0.72, val_ratio=0.08):
    """Split dataset by index. 72/8/20 split (same as Day 1)."""
    n = dataset_len
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_indices = list(range(0, n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, n))
    return train_indices, val_indices, test_indices
