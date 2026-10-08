"""
p-SMILES to graph conversion utilities.
Handles both monomer (1 RU) and trimer (3 RU) generation.
"""

import torch
import numpy as np
import logging
from rdkit import Chem
from rdkit.Chem import rdmolops
from torch_geometric.data import Data
from feature_utils import get_atom_features, get_bond_features, ATOM_FEATURE_DIM, BOND_FEATURE_DIM

logger = logging.getLogger(__name__)


def safe_sanitize(mol):
    """Try multiple sanitization strategies."""
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
    """Find wildcard atom indices."""
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]


def get_star_neighbors(mol, star_indices):
    """Map star atoms to their non-star neighbor indices."""
    star_to_neighbor = {}
    for star_idx in star_indices:
        atom = mol.GetAtomWithIdx(star_idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() != 0:
                star_to_neighbor[star_idx] = nbr.GetIdx()
                break
    return star_to_neighbor


def psmiles_to_monomer(p_smiles):
    """Convert p-SMILES to monomer graph (1 repeat unit with [*] removed).
    Returns RDKit Mol or None on failure.
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

        # Remove star atoms
        rw_mol = Chem.RWMol(mol)
        remove_order = sorted(star_indices, reverse=True)
        for idx in remove_order:
            rw_mol.RemoveAtom(idx)

        result = rw_mol.GetMol()
        if not safe_sanitize(result):
            return None

        return result
    except Exception as e:
        logger.debug(f"Failed monomer conversion for {p_smiles}: {e}")
        return None


def psmiles_to_trimer(p_smiles):
    """Convert p-SMILES to trimer (3 repeat units connected).
    Returns RDKit Mol or None on failure.
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

        # Remove star atoms to get monomer
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

        # Connect adjacent units: tail of unit i -> head of unit i+1
        for i in range(2):
            tail_of_i = i * n_monomer_atoms + monomer_tail
            head_of_next = (i + 1) * n_monomer_atoms + monomer_head
            combined.AddBond(tail_of_i, head_of_next, Chem.BondType.SINGLE)

        result = combined.GetMol()
        if not safe_sanitize(result):
            return None

        return result
    except Exception as e:
        logger.debug(f"Failed trimer conversion for {p_smiles}: {e}")
        return None


def mol_to_data(mol, target=None):
    """Convert RDKit Mol to PyG Data object.
    Args:
        mol: RDKit Mol
        target: optional float target value
    Returns:
        PyG Data object or None
    """
    if mol is None:
        return None
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return None

    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
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

    if target is not None:
        data.y = torch.tensor([target], dtype=torch.float)

    return data
