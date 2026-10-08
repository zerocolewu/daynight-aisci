import random
from typing import List
from copy import deepcopy
from rdkit import Chem
import numpy as np


def generate_multimer_smiles(num_repeat_units, smiles, replace_dummy_atoms=True):
    if num_repeat_units == 1:
        return smiles
    monomer = Chem.MolFromSmiles(smiles)
    repeat_points = []
    cnct_points = []
    bond_type = []
    num_dummy_atoms = 0
    for atom in monomer.GetAtoms():
        if atom.GetSymbol() == '*':
            repeat_points.append(atom.GetIdx())
            neis = atom.GetNeighbors()
            assert len(neis) == 1, f"*atom has more than one neighbor: {smiles}"
            cnct_points.append(atom.GetNeighbors()[0].GetIdx())
            bond_type.append(monomer.GetBondBetweenAtoms(atom.GetIdx(), atom.GetNeighbors()[0].GetIdx()).GetBondType())
            num_dummy_atoms += 1

    assert num_dummy_atoms == 2, "molecule has more than 2 *atoms"
    assert bond_type[0] == bond_type[1], "bond type of 2 *atoms are not same"

    num_atoms = monomer.GetNumAtoms()
    oligomer = deepcopy(monomer)
    for i in range(num_repeat_units-1):
        oligomer = Chem.CombineMols(oligomer, monomer)

    # create index list
    REPEAT_LIST, CNCT_LIST = np.zeros([num_repeat_units, 2]), np.zeros([num_repeat_units, 2])
    for i in range(num_repeat_units):
        REPEAT_LIST[i], CNCT_LIST[i] = np.array(repeat_points) + i * num_atoms, np.array(cnct_points) + i * num_atoms

    # add single bond between monomers
    ed_oligomer = Chem.EditableMol(oligomer)
    removed_atoms_idx = []
    for i in range(num_repeat_units - 1):
        ed_oligomer.AddBond(int(CNCT_LIST[i, 1]), int(CNCT_LIST[i + 1, 0]), order=bond_type[0])
        removed_atoms_idx.extend([int(REPEAT_LIST[i, 1]), int(REPEAT_LIST[i + 1, 0])])

    # Replace the atoms at both ends using H
    if replace_dummy_atoms:
        ed_oligomer.ReplaceAtom(int(REPEAT_LIST[0, 0]), Chem.Atom(1))
        ed_oligomer.ReplaceAtom(int(REPEAT_LIST[num_repeat_units - 1, 1]), Chem.Atom(1))

    # Remove * atoms
    for i in sorted(removed_atoms_idx, reverse=True):
        ed_oligomer.RemoveAtom(i)

    final_mol = ed_oligomer.GetMol()
    final_mol = Chem.RemoveHs(final_mol)

    return Chem.MolToSmiles(final_mol)


if __name__ == '__main__':
    print(generate_multimer_smiles(num_repeat_units=3, smiles="*Nc1c([2H])c([2H])c(*)c([2H])c1[2H]"))

# 增加重复单元