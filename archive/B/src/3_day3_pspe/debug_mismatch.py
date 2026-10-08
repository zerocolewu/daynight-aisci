"""Debug why my graphs don't match the RWPE pickle."""
import sys, pickle, numpy as np, pandas as pd

csv_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'

# Day 1 data
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/1_day1_multimer_feature_gat')
from data_utils import build_datasets as bd1
tg1, vg1, tstg1, _ = bd1(csv_path, multimer='3-RU', feature_level='rich', seed=71)
all1 = tg1 + vg1 + tstg1

# My data
sys.path.insert(0, '.')
from pspe_data import build_datasets_pspe as bd2
tg2, vg2, tstg2, _, _ = bd2(csv_path, seed=71)
all2 = tg2 + vg2 + tstg2

# Load RWPE
with open('/home/home_public/aisci/AIscientist/src/2_day_2_lspe_gat/rwpe_k32.pkl', 'rb') as f:
    rwpe_data = pickle.load(f)
rwpe_dict = rwpe_data['rwpe']

# Create lookups: global_idx -> graph (num_nodes)
lookup1 = {}
for g in all1:
    # Day 1 graphs don't have global_idx, but they're built in order by split
    pass

# Actually, Day 1 code builds graphs with _build(indices_list)
# The indices are the CSV row numbers in sorted order within each split
# So let me just build the full graph list from scratch and track which CSV row each graph comes from

import random
from sklearn.model_selection import train_test_split
from copy import deepcopy
from rdkit import Chem
from torch_geometric.data import Data

# Rebuild exactly like Day 1 but track global_idx
def generate_multimer(psmiles, n_mer, cap_with_h=True):
    if n_mer == 1:
        smiles = psmiles.replace('*', '[H]')
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.RemoveHs(mol)
        return Chem.MolToSmiles(mol)

    monomer = Chem.MolFromSmiles(psmiles)
    if monomer is None:
        return None

    star_info = []
    for atom in monomer.GetAtoms():
        if atom.GetSymbol() == '*':
            neis = atom.GetNeighbors()
            if len(neis) != 1:
                return None
            bond = monomer.GetBondBetweenAtoms(atom.GetIdx(), neis[0].GetIdx())
            star_info.append({
                'star_idx': atom.GetIdx(),
                'neighbor_idx': neis[0].GetIdx(),
                'bond_type': bond.GetBondType()
            })

    if len(star_info) != 2:
        return None
    if star_info[0]['bond_type'] != star_info[1]['bond_type']:
        return None

    num_atoms = monomer.GetNumAtoms()
    oligomer = deepcopy(monomer)
    for _ in range(n_mer - 1):
        oligomer = Chem.CombineMols(oligomer, monomer)

    star_indices = np.zeros((n_mer, 2), dtype=int)
    neighbor_indices = np.zeros((n_mer, 2), dtype=int)
    for i in range(n_mer):
        offset = i * num_atoms
        star_indices[i] = [star_info[0]['star_idx'] + offset,
                           star_info[1]['star_idx'] + offset]
        neighbor_indices[i] = [star_info[0]['neighbor_idx'] + offset,
                               star_info[1]['neighbor_idx'] + offset]

    ed_mol = Chem.EditableMol(oligomer)
    removed_atoms = []

    for i in range(n_mer - 1):
        ed_mol.AddBond(
            int(neighbor_indices[i, 1]),
            int(neighbor_indices[i + 1, 0]),
            order=star_info[0]['bond_type']
        )
        removed_atoms.extend([
            int(star_indices[i, 1]),
            int(star_indices[i + 1, 0])
        ])

    if cap_with_h:
        ed_mol.ReplaceAtom(int(star_indices[0, 0]), Chem.Atom(1))
        ed_mol.ReplaceAtom(int(star_indices[n_mer - 1, 1]), Chem.Atom(1))

    for idx in sorted(removed_atoms, reverse=True):
        ed_mol.RemoveAtom(idx)

    final_mol = ed_mol.GetMol()
    final_mol = Chem.RemoveHs(final_mol)
    return Chem.MolToSmiles(final_mol)

# Check: build one specific CSV row and compare with pickle
df = pd.read_csv(csv_path)

# For each rwpe entry, check the corresponding graph
mismatches = 0
for global_idx in list(range(7170)):
    if global_idx not in rwpe_dict:
        continue
    rwpe_shape = rwpe_dict[global_idx].shape[0]

    psmiles = df['Smiles'].values[global_idx]
    smi = generate_multimer(psmiles, 3)
    if smi is None:
        mismatches += 1
        print(f'  Row {global_idx}: generate_multimer failed')
        continue

    mol = Chem.MolFromSmiles(smi)
    n_nodes = mol.GetNumAtoms()

    if n_nodes != rwpe_shape:
        mismatches += 1
        if mismatches <= 10:
            print(f'  Row {global_idx}: n_nodes={n_nodes}, rwpe_shape={rwpe_shape}, psmiles={psmiles}, smi={smi}')

print(f'\nTotal mismatches: {mismatches}')
