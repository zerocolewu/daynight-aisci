"""
Standard baselines data utilities: Tg / Tm only.
- p-SMILES -> monomer (H-capped, n_mer=1) -> PyG graph with 35-dim one-hot
  categorical atomic node features (NO positional encoding, NO RWPE)
  -> standard vanilla baseline.
- Also computes ECFP4 (Morgan radius=2, 2048-bit) fingerprint for the ECFP+MLP baseline.
- Split: 72% train / 8% val / 20% test, configurable random seed (42 or 71).
- Targets are standardized (StandardScaler fit on train).
"""
import os
import pickle
import warnings
from copy import deepcopy
import numpy as np
import torch
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from torch_geometric.data import Data

RDLogger.DisableLog('rdApp.*')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')

BASE = "/home/home_public/aisci/AIscientist/archive/ablation6archive/src"
DATASET_DIR = os.path.join(BASE, "alldataset")
CACHE_DIR = os.path.join(BASE, "baseline", "standard", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

NODE_DIM = 35         # 35-dim one-hot atomic features
ECFP_RADIUS = 2
ECFP_BITS = 2048

# 12 common atoms in organic / polymer chemistry + 1 "other" bucket
COMMON_ATOMS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]   # H, B, C, N, O, F, Si, P, S, Cl, Br, I
# 5 hybridization types (rdkit enum) -- any unseen type falls outside this list
# (no "other" bucket, results in all-zero sub-vector)
HYBRID_TYPES = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
]


# ----------------------------------------------------------------------
# Head-to-tail multimer generation (per polymer-data-processing skill).
# ----------------------------------------------------------------------
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
            star_info.append({'star_idx': atom.GetIdx(),
                              'neighbor_idx': neis[0].GetIdx(),
                              'bond_type': bond.GetBondType()})
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
        off = i * num_atoms
        star_indices[i] = [star_info[0]['star_idx'] + off, star_info[1]['star_idx'] + off]
        neighbor_indices[i] = [star_info[0]['neighbor_idx'] + off, star_info[1]['neighbor_idx'] + off]
    ed = Chem.EditableMol(oligomer)
    removed = []
    for i in range(n_mer - 1):
        ed.AddBond(int(neighbor_indices[i, 1]), int(neighbor_indices[i + 1, 0]),
                   order=star_info[0]['bond_type'])
        removed.extend([int(star_indices[i, 1]), int(star_indices[i + 1, 0])])
    if cap_with_h:
        ed.ReplaceAtom(int(star_indices[0, 0]), Chem.Atom(1))
        ed.ReplaceAtom(int(star_indices[n_mer - 1, 1]), Chem.Atom(1))
    for idx in sorted(removed, reverse=True):
        ed.RemoveAtom(idx)
    fm = ed.GetMol()
    fm = Chem.RemoveHs(fm)
    return Chem.MolToSmiles(fm)


# ----------------------------------------------------------------------
# Node features: 35-dim one-hot categorical atomic features.
# Composition: 13 (atomic number) + 6 (degree) + 1 (aromatic) + 1 (ring)
#              + 1 (formal charge, scaled) + 6 (Hs) + 5 (hybridization)
#              + 1 (mass, scaled) + 1 (chirality possible) = 35
# ----------------------------------------------------------------------
def atom_features(atom: Chem.Atom) -> list:
    """Build atom feature vector, 35 dimensions total
    = 13(atomic number) + 6(degree) + 1(aromatic) + 1(ring) + 1(charge)
      + 6(H count) + 5(hybridization) + 1(mass) + 1(chirality)
    """
    features = []

    # 1. Atomic number one-hot (12 common + 1 other = 13 dims)
    atomic_num = atom.GetAtomicNum()
    atom_enc = [0] * (len(COMMON_ATOMS) + 1)
    if atomic_num in COMMON_ATOMS:
        atom_enc[COMMON_ATOMS.index(atomic_num)] = 1
    else:
        atom_enc[-1] = 1
    features += atom_enc

    # 2. Degree one-hot (0-5, 6 dims)
    degree = min(atom.GetDegree(), 5)
    features += [1 if i == degree else 0 for i in range(6)]

    # 3. Aromaticity (1 dim)
    features.append(int(atom.GetIsAromatic()))

    # 4. Whether in ring (1 dim)
    features.append(int(atom.IsInRing()))

    # 5. Formal charge (scaled, 1 dim)
    features.append(float(atom.GetFormalCharge()) / 10.0)

    # 6. Connected hydrogen count one-hot (0-5, 6 dims)
    num_h = min(atom.GetTotalNumHs(), 5)
    features += [1 if i == num_h else 0 for i in range(6)]

    # 7. Hybridization type one-hot (5 dims)
    hybrid = atom.GetHybridization()
    features += [1 if hybrid == ht else 0 for ht in HYBRID_TYPES]

    # 8. Atomic mass (scaled, 1 dim)
    features.append(atom.GetMass() / 100.0)

    # 9. Whether chiral center (1 dim)
    features.append(int(atom.HasProp('_ChiralityPossible')))

    return features  # 35 dims total


def extract_node_features(mol):
    feats = [atom_features(a) for a in mol.GetAtoms()]
    return torch.tensor(feats, dtype=torch.float32)


def mol_to_graph(mol, y):
    feats = extract_node_features(mol)
    if feats.shape[0] == 0:
        return None
    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=feats, edge_index=edge_index,
                y=torch.tensor([y], dtype=torch.float32))


_ECFP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_BITS)


def mol_to_ecfp(mol, radius=ECFP_RADIUS, n_bits=ECFP_BITS):
    bv = _ECFP_GEN.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    ConvertToNumpyArray(bv, arr)
    return arr


# ----------------------------------------------------------------------
# Dataset builder
# ----------------------------------------------------------------------
def build_dataset(name, csv_path, ycol, n_mer=1, seed=71):
    """name: 'Tg' or 'Tm'. n_mer=1 = monomer capped with H (vanilla baseline)."""
    cache_name = f"{name}_n{n_mer}_seed{seed}.pkl"
    cache_path = os.path.join(CACHE_DIR, cache_name)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    df = pd.read_csv(csv_path)
    psmiles_col = 'Smiles' if 'Smiles' in df.columns else 'smiles'
    psmiles_list = df[psmiles_col].values
    y_values = df[ycol].values.astype(np.float32)

    indices = np.arange(len(psmiles_list))
    train_idx, temp_idx = train_test_split(indices, test_size=0.28, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.20 / 0.28, random_state=seed)

    def build(idx_list):
        graphs, ecps, ys = [], [], []
        for idx in tqdm(idx_list, desc=name):
            ps = str(psmiles_list[idx]); y = float(y_values[idx])
            try:
                smi = generate_multimer(ps, n_mer)
                if smi is None:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None or mol.GetNumAtoms() == 0:
                    continue
                g = mol_to_graph(mol, y)
                if g is None:
                    continue
                ecfp = mol_to_ecfp(mol)
                graphs.append(g); ecps.append(ecfp); ys.append(y)
            except Exception:
                continue
        return graphs, np.array(ecps, dtype=np.float32), np.array(ys, dtype=np.float32)

    train_g, train_e, train_y = build(train_idx)
    val_g, val_e, val_y = build(val_idx)
    test_g, test_e, test_y = build(test_idx)

    # y scaler fit on train raw; standardize all targets (used by both GNN and MLP)
    scaler = StandardScaler()
    scaler.fit(train_y.reshape(-1, 1))
    for g in train_g + val_g + test_g:
        g.y = torch.tensor(scaler.transform(g.y.numpy().reshape(-1, 1))[0],
                           dtype=torch.float32)
    train_y_n = scaler.transform(train_y.reshape(-1, 1)).ravel().astype(np.float32)
    val_y_n = scaler.transform(val_y.reshape(-1, 1)).ravel().astype(np.float32)
    test_y_n = scaler.transform(test_y.reshape(-1, 1)).ravel().astype(np.float32)

    out = {
        'name': name,
        'train_graphs': train_g, 'val_graphs': val_g, 'test_graphs': test_g,
        'train_ecfp': train_e, 'val_ecfp': val_e, 'test_ecfp': test_e,
        'train_y_norm': train_y_n, 'val_y_norm': val_y_n, 'test_y_norm': test_y_n,
        'scaler': scaler, 'n_mer': n_mer, 'seed': seed,
        'node_dim': NODE_DIM, 'ecfp_bits': ECFP_BITS,
        'counts': {'train': len(train_g), 'val': len(val_g), 'test': len(test_g)},
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[{name}] seed={seed} -> train {len(train_g)} / val {len(val_g)} / test {len(test_g)}")
    return out


def load_dataset(name, seed):
    csvs = {
        'Tg': ('Tg.csv', 'Tg'),
        'Tm': ('Tm.csv', 'Tm'),
    }
    fname, ycol = csvs[name]
    return build_dataset(name, os.path.join(DATASET_DIR, fname), ycol, n_mer=1, seed=seed)
