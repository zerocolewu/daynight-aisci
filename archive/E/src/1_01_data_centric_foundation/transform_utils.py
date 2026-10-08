"""
p-SMILES to monomer/dimer/trimer/cyclized_trimer conversion utilities.
Handles conversion of polymer SMILES (with * wildcard atoms) into multimer graphs.
"""

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
import copy
import logging

logger = logging.getLogger(__name__)


def safe_sanitize(mol):
    """
    Try to sanitize a molecule, with fallback for kekulization failures.
    If normal sanitization fails, try with SANITIZE_NONE then manually set properties.
    Returns True if sanitization succeeded.
    """
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        pass
    try:
        # Try with relaxed sanitization
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        return True
    except Exception:
        pass
    try:
        # Last resort: skip all sanitization
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)
        return True
    except Exception:
        return False


def find_star_atoms(mol):
    """Find dummy atom indices (atomic num 0) in a molecule."""
    star_atoms = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            star_atoms.append(atom.GetIdx())
    return star_atoms


def get_star_neighbors(mol, star_indices):
    """
    For each star atom, find the neighbor atom (non-dummy) it's bonded to.
    Returns dict: star_idx -> neighbor_idx
    """
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
    Convert p-SMILES to monomer graph by removing star atoms.
    Returns (mol, head_neighbor, tail_neighbor) or (None, None, None) on failure.
    """
    try:
        mol = Chem.MolFromSmiles(p_smiles)
        if mol is None:
            return None, None, None

        star_indices = find_star_atoms(mol)
        if len(star_indices) != 2:
            logger.warning(f"Expected 2 star atoms, got {len(star_indices)}: {p_smiles}")
            return None, None, None

        star_to_neighbor = get_star_neighbors(mol, star_indices)
        if len(star_to_neighbor) != 2:
            logger.warning(f"Could not find neighbors for all star atoms: {p_smiles}")
            return None, None, None

        # Get the two star atoms and their neighbors
        stars = list(star_to_neighbor.keys())
        head_star, tail_star = stars[0], stars[1]
        head_nbr = star_to_neighbor[head_star]
        tail_nbr = star_to_neighbor[tail_star]

        # Remove star atoms
        rw_mol = Chem.RWMol(mol)
        # Remove higher index first to avoid reindexing issues
        remove_order = sorted([head_star, tail_star], reverse=True)
        for idx in remove_order:
            rw_mol.RemoveAtom(idx)

        # Update neighbor indices after removal
        # After removing atoms, indices shift down
        atoms_removed = sorted([head_star, tail_star])
        def update_idx(idx):
            new_idx = idx
            for removed_idx in atoms_removed:
                if idx > removed_idx:
                    new_idx -= 1
            return new_idx

        new_head_nbr = update_idx(head_nbr)
        new_tail_nbr = update_idx(tail_nbr)

        result_mol = rw_mol.GetMol()
        if not safe_sanitize(result_mol):
            logger.warning(f"Failed sanitization for monomer: {p_smiles}")
            return None, None, None

        return result_mol, new_head_nbr, new_tail_nbr
    except Exception as e:
        logger.warning(f"Failed monomer conversion for {p_smiles}: {e}")
        return None, None, None


def smiles_to_multimer(p_smiles, n_units=2):
    """
    Convert p-SMILES to multimer with n_units repeat units.
    Returns (mol, head_neighbor, tail_neighbor, unit_boundaries) or (None, None, None, None).
    unit_boundaries: list of (start_idx, end_idx) for each repeat unit (atom indices in final mol).
    """
    try:
        mol = Chem.MolFromSmiles(p_smiles)
        if mol is None:
            return None, None, None, None

        star_indices = find_star_atoms(mol)
        if len(star_indices) != 2:
            return None, None, None, None

        star_to_neighbor = get_star_neighbors(mol, star_indices)
        if len(star_to_neighbor) != 2:
            return None, None, None, None

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

        # Build multimer by combining n_units copies
        combined = Chem.RWMol(Chem.Mol())

        unit_boundaries = []

        for unit in range(n_units):
            offset = combined.GetNumAtoms()
            unit_boundaries.append((offset, offset + n_monomer_atoms))

            # Add atoms from this monomer copy
            for atom in monomer.GetAtoms():
                new_atom = Chem.Atom(atom.GetSymbol())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                new_atom.SetIsAromatic(atom.GetIsAromatic())
                new_atom.SetChiralTag(atom.GetChiralTag())
                new_atom.SetHybridization(atom.GetHybridization())
                new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
                combined.AddAtom(new_atom)

            # Add bonds within this monomer copy
            for bond in monomer.GetBonds():
                begin = bond.GetBeginAtomIdx() + offset
                end = bond.GetEndAtomIdx() + offset
                combined.AddBond(begin, end, bond.GetBondType())

        # Connect units: unit[i].tail -> unit[i+1].head
        for i in range(n_units - 1):
            tail_of_i = unit_boundaries[i][0] + monomer_tail
            head_of_next = unit_boundaries[i + 1][0] + monomer_head
            combined.AddBond(tail_of_i, head_of_next, Chem.BondType.SINGLE)

        result = combined.GetMol()
        if not safe_sanitize(result):
            logger.warning(f"Failed sanitization for multimer-{n_units}: {p_smiles}")
            return None, None, None, None

        # Global head/tail for the multimer
        global_head = unit_boundaries[0][0] + monomer_head
        global_tail = unit_boundaries[-1][0] + monomer_tail

        return result, global_head, global_tail, unit_boundaries
    except Exception as e:
        logger.warning(f"Failed multimer-{n_units} conversion for {p_smiles}: {e}")
        return None, None, None, None


def smiles_to_cyclized_trimer(p_smiles):
    """
    Convert p-SMILES to cyclized trimer (3 repeat units with periodic edge).
    Returns (mol, periodic_bond_atoms, unit_boundaries) or (None, None, None).
    periodic_bond_atoms: tuple of (atom_i, atom_j) for the periodic edge.
    """
    result = smiles_to_multimer(p_smiles, n_units=3)
    mol, global_head, global_tail, unit_boundaries = result

    if mol is None:
        return None, None, None

    try:
        rw_mol = Chem.RWMol(mol)
        # Add periodic edge: connect tail of last unit to head of first unit
        rw_mol.AddBond(global_tail, global_head, Chem.BondType.SINGLE)
        result_mol = rw_mol.GetMol()
        if not safe_sanitize(result_mol):
            logger.warning(f"Failed sanitization for cyclized trimer: {p_smiles}")
            return None, None, None

        periodic_bond_atoms = (global_tail, global_head)
        return result_mol, periodic_bond_atoms, unit_boundaries
    except Exception as e:
        logger.warning(f"Failed cyclized trimer conversion for {p_smiles}: {e}")
        return None, None, None


def get_graph_stats(mol):
    """Return basic graph statistics for a molecule."""
    return {
        'n_atoms': mol.GetNumAtoms(),
        'n_bonds': mol.GetNumBonds(),
    }


if __name__ == "__main__":
    # Quick test with valid p-SMILES
    test_smiles = "*CC(*)CC"
    print(f"Test p-SMILES: {test_smiles}")

    mono, h, t = smiles_to_monomer(test_smiles)
    if mono:
        print(f"Monomer: {Chem.MolToSmiles(mono)}, head={h}, tail={t}, atoms={mono.GetNumAtoms()}")

    dimer, h, t, ub = smiles_to_multimer(test_smiles, 2)
    if dimer:
        print(f"Dimer: {Chem.MolToSmiles(dimer)}, head={h}, tail={t}, atoms={dimer.GetNumAtoms()}, units={ub}")

    trimer, h, t, ub = smiles_to_multimer(test_smiles, 3)
    if trimer:
        print(f"Trimer: {Chem.MolToSmiles(trimer)}, head={h}, tail={t}, atoms={trimer.GetNumAtoms()}, units={ub}")

    ctrim, pb, ub = smiles_to_cyclized_trimer(test_smiles)
    if ctrim:
        print(f"Cyclized trimer: {Chem.MolToSmiles(ctrim)}, periodic={pb}, atoms={ctrim.GetNumAtoms()}, units={ub}")
