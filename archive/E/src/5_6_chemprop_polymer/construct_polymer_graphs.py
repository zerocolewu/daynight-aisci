"""
Construct polymer graphs from p-SMILES for Chemprop-Polymer.
Supports monomer, trimer (no periodic edge), and full polymer (trimer + periodic edge) graphs.
"""
import os
import pickle
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

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
# Atom features (one-hot encoded)
# ============================================================
# Atomic number: values 1-100 → 100 dims (we'll use indices 0-99 for atoms 1-100)
ATOMIC_NUM_RANGE = list(range(1, 101))  # 100 values
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
    """Compute atom features for a single RDKit atom."""
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
    """Compute bond features for a single RDKit bond or periodic edge."""
    features = []
    if is_periodic:
        # Bond type one-hot (use the provided bond type)
        # For periodic edge, bond type is passed separately
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


BOND_FDIM = 4 + 1 + 1 + 1 + 1 + 1  # = 9


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
    # We'll use SMILES order to determine left/right
    # The star atom with the smaller neighbor index is "left"
    star_info.sort(key=lambda x: x[0])

    # Create repeat unit by removing star atoms
    # Use Chem.RWMol for editing
    rwmol = Chem.RWMol(mol)
    # Remove star atoms (remove higher index first to preserve indices)
    star_indices = sorted([s[2] for s in star_info], reverse=True)
    for idx in star_indices:
        rwmol.RemoveAtom(idx)

    # Update neighbor indices after removal
    # After removing star atoms, we need to figure out new indices for neighbors
    # Sort star atoms by index (ascending) to compute offset
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


def get_electronegativity(atom):
    """Get Pauling electronegativity for an atom."""
    return ELECTRONEGATIVITY.get(atom.GetAtomicNum(), DEFAULT_EN)


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


def construct_trimer_graph(mol, left_neighbor, right_neighbor, add_periodic_edge=False,
                           minimal_periodic=False, random_periodic=False, rng=None):
    """
    Construct trimer graph from repeat unit.
    mol: repeat unit RDKit mol
    left_neighbor: (idx, bond_type) for left star atom neighbor
    right_neighbor: (idx, bond_type) for right star atom neighbor
    add_periodic_edge: whether to add periodic edge
    minimal_periodic: if True, only use is_periodic flag (no en_gradient, no boundary_degree)
    random_periodic: if True, connect random boundary atoms instead of correct terminals
    rng: numpy random generator for random periodic edge
    """
    n_atoms = mol.GetNumAtoms()
    left_nidx, left_bt = left_neighbor
    right_nidx, right_bt = right_neighbor

    # Atom features for trimer (3 copies)
    atom_features = []
    atom_objects = []  # store for computing properties
    for copy_idx in range(3):
        for atom in mol.GetAtoms():
            atom_features.append(get_atom_features(atom))
            atom_objects.append(atom)

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

        if random_periodic and rng is not None:
            # Select random atoms from boundary regions
            # Boundary atoms: first and last few atoms of each copy
            all_atoms = list(range(3 * n_atoms))
            correct_terminals = {left_terminal, right_terminal}
            # Candidates: atoms near the boundaries of each copy
            candidates = []
            for copy_idx in range(3):
                offset = copy_idx * n_atoms
                for i in range(n_atoms):
                    if (offset + i) not in correct_terminals:
                        candidates.append(offset + i)

            if len(candidates) >= 2:
                chosen = rng.choice(len(candidates), size=2, replace=False)
                left_terminal = candidates[chosen[0]]
                right_terminal = candidates[chosen[1]]
            # else fall back to correct terminals

        # Compute periodic edge features
        # Get atom objects for terminals
        left_atom = mol.GetAtomWithIdx(left_terminal % n_atoms)
        right_atom = mol.GetAtomWithIdx(right_terminal % n_atoms)

        en_left = get_electronegativity(left_atom)
        en_right = get_electronegativity(right_atom)
        en_gradient = abs(en_right - en_left)

        # Boundary atom degree: degree in trimer before adding periodic edge
        # Count existing bonds for each terminal atom
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

        if minimal_periodic:
            pf = [0] * BOND_FDIM
            pf[:4] = one_hot(bt_str, ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'])
            pf[4] = 0  # not conjugated
            pf[5] = 1  # in ring
            pf[6] = 1  # is periodic
            pf[7] = 0.0  # no en gradient
            pf[8] = 0.0  # no boundary degree
        else:
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


def build_reverse_edge_map(bonds):
    """For each directed bond index, find the index of its reverse bond."""
    # Build a dict from (source, target) -> index
    bond_map = {}
    for idx, (s, t) in enumerate(bonds):
        bond_map[(s, t)] = idx

    reverse_map = []
    for idx, (s, t) in enumerate(bonds):
        rev_idx = bond_map.get((t, s), idx)  # fallback to self if not found
        reverse_map.append(rev_idx)

    return reverse_map


def build_atom_incoming_bonds(bonds, num_atoms):
    """For each atom, list indices of incoming directed bonds."""
    incoming = [[] for _ in range(num_atoms)]
    for idx, (s, t) in enumerate(bonds):
        incoming[t].append(idx)
    return incoming


def graph_dict_to_tensors(graph_data):
    """Convert graph data dict to tensor format."""
    atom_features = torch.tensor(graph_data['atom_features'], dtype=torch.float32)
    bond_features = torch.tensor(graph_data['bond_features'], dtype=torch.float32)

    bonds = graph_data['bonds']
    if len(bonds) > 0:
        edge_index = torch.tensor(bonds, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    reverse_map = build_reverse_edge_map(bonds)
    reverse_map = torch.tensor(reverse_map, dtype=torch.long)

    incoming = build_atom_incoming_bonds(bonds, graph_data['num_atoms'])

    return {
        'atom_features': atom_features,
        'bond_features': bond_features,
        'edge_index': edge_index,
        'reverse_map': reverse_map,
        'atom_incoming_bonds': incoming,
        'num_atoms': graph_data['num_atoms'],
        'periodic_edge_bonds': graph_data['periodic_edge_bonds'],
    }


def process_dataset(csv_path, graph_type='polymer', minimal_periodic=False,
                    random_periodic=False, seed=42):
    """
    Process Tg dataset and construct graphs.

    graph_type: 'monomer', 'trimer', or 'polymer'
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} polymers")

    rng = np.random.RandomState(seed) if random_periodic else None
    processed = []
    skipped = 0

    for idx, row in df.iterrows():
        psmiles = row['Smiles']
        target = row['Tg']

        mol, star_info = parse_psmiles(psmiles)
        if mol is None or star_info is None:
            skipped += 1
            continue

        left_nidx, left_bt, right_nidx, right_bt = star_info

        try:
            if graph_type == 'monomer':
                graph = construct_monomer_graph(mol)
            elif graph_type == 'trimer':
                graph = construct_trimer_graph(mol, (left_nidx, left_bt),
                                               (right_nidx, right_bt),
                                               add_periodic_edge=False)
            elif graph_type == 'polymer':
                graph = construct_trimer_graph(mol, (left_nidx, left_bt),
                                               (right_nidx, right_bt),
                                               add_periodic_edge=True,
                                               minimal_periodic=minimal_periodic,
                                               random_periodic=random_periodic,
                                               rng=rng)
            else:
                raise ValueError(f"Unknown graph_type: {graph_type}")

            tensor_data = graph_dict_to_tensors(graph)
            tensor_data['target'] = torch.tensor([target], dtype=torch.float32)
            tensor_data['psmiles'] = psmiles
            processed.append(tensor_data)

        except Exception as e:
            skipped += 1
            if idx < 10:
                print(f"  Error processing {psmiles}: {e}")
            continue

    print(f"Processed: {len(processed)}, Skipped: {skipped}")
    return processed


if __name__ == '__main__':
    csv_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    out_dir = '/home/home_public/aisci/AIscientist/src/5_6_chemprop_polymer'

    # Process and save all graph types
    for gt in ['monomer', 'trimer', 'polymer']:
        print(f"\n=== Processing {gt} ===")
        data = process_dataset(csv_path, graph_type=gt)
        save_path = os.path.join(out_dir, f'graphs_{gt}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {len(data)} graphs to {save_path}")

    # Polymer with minimal periodic features
    print("\n=== Processing polymer_minimal ===")
    data = process_dataset(csv_path, graph_type='polymer', minimal_periodic=True)
    save_path = os.path.join(out_dir, 'graphs_polymer_minimal.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} graphs to {save_path}")

    # Polymer with random periodic edge
    print("\n=== Processing polymer_random ===")
    data = process_dataset(csv_path, graph_type='polymer', random_periodic=True, seed=42)
    save_path = os.path.join(out_dir, 'graphs_polymer_random.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} graphs to {save_path}")

    print("\n=== All graphs constructed ===")
