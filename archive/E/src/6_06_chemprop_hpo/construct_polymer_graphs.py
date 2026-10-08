"""
Construct polymer graphs from p-SMILES for Chemprop-Polymer.
Supports monomer, trimer (no periodic edge), and full polymer (trimer + periodic edge) graphs.
Copied from Day 5 (5_6_chemprop_polymer) with no modifications.
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
ATOMIC_NUM_RANGE = list(range(1, 101))  # 100 values
DEGREE_RANGE = list(range(6))
FORMAL_CHARGE_RANGE = [-2, -1, 0, 1, 2]
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
NUM_HS_RANGE = list(range(5))

ATOM_FDIM = 100 + 6 + 5 + 5 + 1 + 1 + 5  # = 123


def one_hot(value, choices):
    """One-hot encoding for a value given a list of choices."""
    vec = [0] * len(choices)
    try:
        idx = choices.index(value)
        vec[idx] = 1
    except ValueError:
        pass
    return vec


def get_atom_features(atom):
    """Compute atom features for a single RDKit atom."""
    features = []
    atomic_num = atom.GetAtomicNum()
    an_vec = [0] * 100
    if 1 <= atomic_num <= 100:
        an_vec[atomic_num - 1] = 1
    features.extend(an_vec)
    features.extend(one_hot(atom.GetDegree(), DEGREE_RANGE))
    features.extend(one_hot(atom.GetFormalCharge(), FORMAL_CHARGE_RANGE))
    features.extend(one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES))
    features.append(int(atom.GetIsAromatic()))
    features.append(int(atom.IsInRing()))
    features.extend(one_hot(atom.GetTotalNumHs(), NUM_HS_RANGE))
    return features


def get_bond_features(bond, is_periodic=False, en_gradient=0.0, boundary_degree=0.0):
    """Compute bond features for a single RDKit bond or periodic edge."""
    features = []
    if is_periodic:
        features.extend(one_hot(bond, ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']))
        features.append(0)  # is_conjugated
        features.append(1)  # is_in_ring
        features.append(1)  # is_periodic
        features.append(float(en_gradient))
        features.append(float(boundary_degree))
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
    """Parse p-SMILES and extract repeat unit mol and star atom info."""
    mol = Chem.MolFromSmiles(psmiles)
    if mol is None:
        return None, None

    star_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(star_atoms) != 2:
        return None, None

    star_info = []
    for star_atom in star_atoms:
        star_idx = star_atom.GetIdx()
        neighbors = star_atom.GetNeighbors()
        if len(neighbors) != 1:
            return None, None
        neighbor = neighbors[0]
        bond = mol.GetBondBetweenAtoms(star_idx, neighbor.GetIdx())
        bond_type = bond.GetBondType()
        star_info.append((neighbor.GetIdx(), bond_type, star_idx))

    star_info.sort(key=lambda x: x[0])

    rwmol = Chem.RWMol(mol)
    star_indices = sorted([s[2] for s in star_info], reverse=True)
    for idx in star_indices:
        rwmol.RemoveAtom(idx)

    star_indices_asc = sorted([s[2] for s in star_info])
    new_star_info = []
    for neighbor_idx, bond_type, star_idx in star_info:
        new_idx = neighbor_idx
        for si in star_indices_asc:
            if si < neighbor_idx:
                new_idx -= 1
        new_star_info.append((new_idx, bond_type))

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
    return ELECTRONEGATIVITY.get(atom.GetAtomicNum(), DEFAULT_EN)


def construct_trimer_graph(mol, left_neighbor, right_neighbor, add_periodic_edge=False,
                           minimal_periodic=False, random_periodic=False, rng=None):
    """Construct trimer graph from repeat unit."""
    n_atoms = mol.GetNumAtoms()
    left_nidx, left_bt = left_neighbor
    right_nidx, right_bt = right_neighbor

    atom_features = []
    atom_objects = []
    for copy_idx in range(3):
        for atom in mol.GetAtoms():
            atom_features.append(get_atom_features(atom))
            atom_objects.append(atom)

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
    for k in range(2):
        right_partner = right_nidx + k * n_atoms
        left_partner = left_nidx + (k + 1) * n_atoms
        bt_str_map = {
            Chem.rdchem.BondType.SINGLE: 'SINGLE',
            Chem.rdchem.BondType.DOUBLE: 'DOUBLE',
            Chem.rdchem.BondType.TRIPLE: 'TRIPLE',
            Chem.rdchem.BondType.AROMATIC: 'AROMATIC',
        }
        bt_str = bt_str_map.get(right_bt, 'SINGLE')
        bf = [0] * BOND_FDIM
        bf[:4] = one_hot(bt_str, ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'])
        bf[4] = 0
        bf[5] = 0
        bf[6] = 0
        bf[7] = 0.0
        bf[8] = 0.0
        bonds.append((right_partner, left_partner))
        bond_features_list.append(bf)
        bonds.append((left_partner, right_partner))
        bond_features_list.append(bf)

    periodic_edge_bonds = []
    if add_periodic_edge:
        left_terminal = left_nidx
        right_terminal = right_nidx + 2 * n_atoms

        if random_periodic and rng is not None:
            all_atoms = list(range(3 * n_atoms))
            correct_terminals = {left_terminal, right_terminal}
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

        left_atom = mol.GetAtomWithIdx(left_terminal % n_atoms)
        right_atom = mol.GetAtomWithIdx(right_terminal % n_atoms)
        en_left = get_electronegativity(left_atom)
        en_right = get_electronegativity(right_atom)
        en_gradient = abs(en_right - en_left)

        def count_bonds(atom_idx, bonds_list):
            count = 0
            for (s, t) in bonds_list:
                if s == atom_idx or t == atom_idx:
                    count += 1
            return count // 2

        deg_left = count_bonds(left_terminal, bonds)
        deg_right = count_bonds(right_terminal, bonds)
        boundary_degree = max(deg_left, deg_right)

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
            pf[4] = 0
            pf[5] = 1
            pf[6] = 1
            pf[7] = 0.0
            pf[8] = 0.0
        else:
            pf = [0] * BOND_FDIM
            pf[:4] = one_hot(bt_str, ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'])
            pf[4] = 0
            pf[5] = 1
            pf[6] = 1
            pf[7] = float(en_gradient)
            pf[8] = float(boundary_degree)

        bonds.append((right_terminal, left_terminal))
        bond_features_list.append(pf)
        periodic_edge_bonds.append(len(bonds) - 1)

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
    bond_map = {}
    for idx, (s, t) in enumerate(bonds):
        bond_map[(s, t)] = idx
    reverse_map = []
    for idx, (s, t) in enumerate(bonds):
        rev_idx = bond_map.get((t, s), idx)
        reverse_map.append(rev_idx)
    return reverse_map


def build_atom_incoming_bonds(bonds, num_atoms):
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
    """Process Tg dataset and construct graphs."""
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
            graph = construct_trimer_graph(mol, (left_nidx, left_bt),
                                           (right_nidx, right_bt),
                                           add_periodic_edge=True,
                                           minimal_periodic=minimal_periodic,
                                           random_periodic=random_periodic,
                                           rng=rng)
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
