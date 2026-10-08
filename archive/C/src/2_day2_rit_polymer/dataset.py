"""
Dataset module for Repetition-Invariant Transformer (RIT) polymer Tg prediction.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


def parse_monomer(smiles: str) -> Optional[Chem.Mol]:
    """Replace * with [H] to get monomer mol."""
    try:
        s = smiles.replace('*', '[H]')
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except:
        return None


def gen_conformer(mol: Chem.Mol) -> Optional[Tuple[Chem.Mol, int]]:
    """Add Hs, generate 3D conformer with ETKDGv3+MMFF. Returns (mol_with_h, conf_id)."""
    try:
        mol = Chem.AddHs(mol)
        p = AllChem.ETKDGv3()
        p.randomSeed = 42
        cid = AllChem.EmbedMolecule(mol, p)
        if cid == -1:
            p2 = AllChem.ETKDG()
            p2.randomSeed = 42
            cid = AllChem.EmbedMolecule(mol, p2)
        if cid == -1:
            return None
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=cid)
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            if ff is not None:
                ff.Minimize(maxIts=200)
        except:
            pass
        return mol, cid
    except:
        return None


def extract_features(mol_h: Chem.Mol, cid: int):
    """Extract bond lengths dict (heavy-atom indexed) and 16-dim global stats."""
    conf = mol_h.GetConformer(cid)
    natoms_h = mol_h.GetNumAtoms()

    # Build heavy atom index map: position in heavy-only list -> index in mol_h
    heavy_idx_in_molh = []
    for a in mol_h.GetAtoms():
        if a.GetSymbol() != 'H':
            heavy_idx_in_molh.append(a.GetIdx())
    molh_to_heavy = {v: i for i, v in enumerate(heavy_idx_in_molh)}

    # Bond lengths indexed by heavy-atom pair indices
    bl_dict = {}
    all_bl = []
    for bond in mol_h.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ai = mol_h.GetAtomWithIdx(i)
        aj = mol_h.GetAtomWithIdx(j)
        if ai.GetSymbol() != 'H' and aj.GetSymbol() != 'H':
            pi = np.array(conf.GetAtomPosition(i))
            pj = np.array(conf.GetAtomPosition(j))
            d = np.linalg.norm(pi - pj)
            hi, hj = molh_to_heavy[i], molh_to_heavy[j]
            bl_dict[(hi, hj)] = d
            bl_dict[(hj, hi)] = d
            all_bl.append(d)

    # Angles
    angles = []
    mol_no_h = Chem.RemoveHs(mol_h)
    for a in mol_no_h.GetAtoms():
        nbs = [n.GetIdx() for n in a.GetNeighbors()]
        if len(nbs) >= 2:
            for ii in range(len(nbs)):
                for jj in range(ii + 1, len(nbs)):
                    v1 = np.array(conf.GetAtomPosition(heavy_idx_in_molh[nbs[ii]])) - np.array(conf.GetAtomPosition(heavy_idx_in_molh[a.GetIdx()]))
                    v2 = np.array(conf.GetAtomPosition(heavy_idx_in_molh[nbs[jj]])) - np.array(conf.GetAtomPosition(heavy_idx_in_molh[a.GetIdx()]))
                    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    angles.append(np.arccos(np.clip(c, -1, 1)))

    # Dihedrals
    dihedrals = []
    for bond in mol_no_h.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        n1 = [n.GetIdx() for n in mol_no_h.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() != j]
        n2 = [n.GetIdx() for n in mol_no_h.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != i]
        for a in n1:
            for b in n2:
                p0 = np.array(conf.GetAtomPosition(heavy_idx_in_molh[a]))
                p1 = np.array(conf.GetAtomPosition(heavy_idx_in_molh[i]))
                p2 = np.array(conf.GetAtomPosition(heavy_idx_in_molh[j]))
                p3 = np.array(conf.GetAtomPosition(heavy_idx_in_molh[b]))
                b1, b2, b3 = p1 - p0, p2 - p1, p2 - p3
                n1v, n2v = np.cross(b1, b2), np.cross(b2, b3)
                if np.linalg.norm(n1v) > 1e-8 and np.linalg.norm(n2v) > 1e-8:
                    y = np.dot(np.cross(n1v, n2v), b2 / (np.linalg.norm(b2) + 1e-8))
                    x = np.dot(n1v, n2v)
                    dihedrals.append(np.arctan2(y, x))

    # Global stats
    coords = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                        for k in heavy_idx_in_molh])
    center = coords.mean(axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))
    vol = np.prod(coords.max(axis=0) - coords.min(axis=0)) if len(coords) > 0 else 0

    try:
        mw = Descriptors.MolWt(mol_no_h)
        nrot = rdMolDescriptors.CalcNumRotatableBonds(mol_no_h)
        nrings = rdMolDescriptors.CalcNumRings(mol_no_h)
        tpsa = Descriptors.TPSA(mol_no_h)
        logp = Descriptors.MolLogP(mol_no_h)
        sp3 = sum(1 for a in mol_no_h.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP3)
        fsp3 = sp3 / max(mol_no_h.GetNumAtoms(), 1)
    except:
        mw = nrot = nrings = tpsa = logp = fsp3 = 0

    stats = np.array([
        len(all_bl), np.mean(all_bl) if all_bl else 0, np.std(all_bl) if len(all_bl) > 1 else 0,
        np.mean(angles) if angles else 0, np.std(angles) if len(angles) > 1 else 0,
        np.mean(np.abs(dihedrals)) if dihedrals else 0, np.std(dihedrals) if len(dihedrals) > 1 else 0,
        mol_no_h.GetNumAtoms(), mw, nrot, nrings, tpsa, logp, fsp3, rg, vol
    ], dtype=np.float32)

    return bl_dict, stats


def mol_to_graph(mol_no_h: Chem.Mol, bl_dict: dict, stats: np.ndarray, tg: float, num_ru: int = 3) -> Data:
    """Build PyG Data from heavy-atom mol."""
    natoms = mol_no_h.GetNumAtoms()
    atom_types = ['C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']

    x_list = []
    for atom in mol_no_h.GetAtoms():
        f = [1.0 if atom.GetSymbol() == t else 0.0 for t in atom_types]
        f.append(1.0 if atom.GetSymbol() not in atom_types else 0.0)
        for d in range(6):
            f.append(1.0 if atom.GetDegree() == d else 0.0)
        for h in [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
                   Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
                   Chem.HybridizationType.SP3D2]:
            f.append(1.0 if atom.GetHybridization() == h else 0.0)
        f.append(1.0 if atom.GetIsAromatic() else 0.0)
        f.append(1.0 if atom.IsInRing() else 0.0)
        f.append(float(atom.GetFormalCharge()))
        for h in range(5):
            f.append(1.0 if atom.GetTotalNumHs() == h else 0.0)
        x_list.append(f)

    x = torch.tensor(x_list, dtype=torch.float)

    ei, ef = [], []
    for bond in mol_no_h.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bl = bl_dict.get((i, j), bl_dict.get((j, i), 1.5)) / 3.0
        bt = bond.GetBondType()
        f = [1.0 if bt == t else 0.0 for t in [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                                                 Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]]
        f += [1.0 if bond.GetIsConjugated() else 0.0,
              1.0 if bond.IsInRing() else 0.0,
              bl]
        ei.append([i, j]); ei.append([j, i])
        ef.extend([f, f])

    return Data(
        x=x,
        edge_index=torch.tensor(ei, dtype=torch.long).t() if ei else torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.tensor(ef, dtype=torch.float) if ef else torch.zeros(0, 7, dtype=torch.float),
        conformer_stats=torch.tensor(stats, dtype=torch.float).unsqueeze(0),  # [1, 16]
        y=torch.tensor([tg], dtype=torch.float),
        num_nodes=natoms, num_ru=num_ru
    )


def get_tg_range(tg: float) -> int:
    return 0 if tg < 100 else (1 if tg < 200 else 2)


class PolymerDataset(Dataset):
    def __init__(self, smiles_list, tg_values, num_ru=3):
        self.data_list = []
        self.num_ru = num_ru
        failed = 0
        total = len(smiles_list)
        print(f"Processing {total} molecules...")
        for idx, (smi, tg) in enumerate(zip(smiles_list, tg_values)):
            if idx % 1000 == 0 and idx > 0:
                print(f"  {idx}/{total} done ({100*len(self.data_list)/idx:.1f}% success)")
            try:
                mol = parse_monomer(smi)
                if mol is None:
                    failed += 1; continue
                res = gen_conformer(mol)
                if res is None:
                    failed += 1; continue
                mol_h, cid = res
                bl, stats = extract_features(mol_h, cid)
                mol_no_h = Chem.RemoveHs(mol_h)
                data = mol_to_graph(mol_no_h, bl, stats, tg, num_ru)
                self.data_list.append(data)
            except:
                failed += 1
        print(f"Done: {len(self.data_list)}/{total} ({100*len(self.data_list)/total:.1f}%), failed={failed}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_fn(batch):
    return Batch.from_data_list(batch)
