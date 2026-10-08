"""
PPFN Dataset: Polymer Periodic Fragment Network dataset construction.

Builds trimer graphs with periodic edges, extracts BRICS fragments,
and creates hierarchical atom-fragment representations.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import BRICS, AllChem, Descriptors
from copy import deepcopy
from typing import List, Tuple, Dict, Optional

# Add parent data processing code to path
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/alldataset/some_dataprocesscode')
from multimer import generate_multimer_smiles
from cyclize_p_smiles import cyclize_p_smiles


# ── Atom feature dimensions ──────────────────────────────────────────────
ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I',
    'B', 'Se', 'Te', 'Ge', 'As', 'Sb', 'Bi', '*'
]
ATOM_TYPES_DICT = {t: i for i, t in enumerate(ATOM_TYPES)}

HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
HYBRIDIZATION_DICT = {t: i for i, t in enumerate(HYBRIDIZATION_TYPES)}

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_TYPES_DICT = {t: i for i, t in enumerate(BOND_TYPES)}

NUM_ATOM_FEATURES = len(ATOM_TYPES) + 1 + len(HYBRIDIZATION_TYPES) + 1 + 1 + 1  # type, degree, hybrid, aromatic, formal_charge, ring
NUM_EDGE_FEATURES = len(BOND_TYPES) + 1 + 1  # bond_type, conjugated, ring


def atom_features(atom: Chem.Atom) -> np.ndarray:
    """Compute atom features."""
    features = []
    # Atom type (one-hot)
    atom_type = atom.GetSymbol()
    type_feat = np.zeros(len(ATOM_TYPES))
    if atom_type in ATOM_TYPES_DICT:
        type_feat[ATOM_TYPES_DICT[atom_type]] = 1.0
    features.append(type_feat)
    # Degree
    features.append(np.array([float(atom.GetDegree())]))
    # Hybridization (one-hot)
    hybrid = atom.GetHybridization()
    hybrid_feat = np.zeros(len(HYBRIDIZATION_TYPES))
    if hybrid in HYBRIDIZATION_DICT:
        hybrid_feat[HYBRIDIZATION_DICT[hybrid]] = 1.0
    features.append(hybrid_feat)
    # Aromatic
    features.append(np.array([float(atom.GetIsAromatic())]))
    # Formal charge
    features.append(np.array([float(atom.GetFormalCharge())]))
    # In ring
    features.append(np.array([float(atom.IsInRing())]))
    return np.concatenate(features)


def bond_features(bond: Chem.Bond) -> np.ndarray:
    """Compute bond features."""
    features = []
    # Bond type (one-hot)
    bond_type = bond.GetBondType()
    type_feat = np.zeros(len(BOND_TYPES))
    if bond_type in BOND_TYPES_DICT:
        type_feat[BOND_TYPES_DICT[bond_type]] = 1.0
    else:
        type_feat[0] = 1.0  # default single
    features.append(type_feat)
    # Conjugated
    features.append(np.array([float(bond.GetIsConjugated())]))
    # In ring
    features.append(np.array([float(bond.IsInRing())]))
    return np.concatenate(features)


def build_trimer_graph(p_smiles: str) -> Optional[Chem.Mol]:
    """Build trimer (3 repeat units) from p-SMILES with periodic edge."""
    try:
        trimer_smiles = generate_multimer_smiles(num_repeat_units=3, smiles=p_smiles, replace_dummy_atoms=False)
        trimer_mol = Chem.MolFromSmiles(trimer_smiles)
        if trimer_mol is None:
            return None
        return trimer_mol
    except Exception:
        return None


def build_cyclized_mol(p_smiles: str) -> Optional[Chem.Mol]:
    """Build cyclized monomer (periodic unit cell) from p-SMILES."""
    try:
        cyclized_smiles = cyclize_p_smiles(p_smiles)
        mol = Chem.MolFromSmiles(cyclized_smiles)
        if mol is None:
            return None
        return mol
    except Exception:
        return None


def smart_fragmentation(mol: Chem.Mol) -> Tuple[List[List[int]], List[str]]:
    """
    Fragment a molecule using BRICS-breaking bonds with proper atom tracking.
    Falls back to ring-based or functional-group fragmentation if BRICS fails.
    """
    try:
        # Find BRICS breakable bonds
        bonds_to_break = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            s1, s2 = a1.GetSymbol(), a2.GetSymbol()
            bt = bond.GetBondType()
            # BRICS breaks single bonds between C and certain atoms, not in rings
            if bond.IsInRing():
                continue
            if bt != Chem.BondType.SINGLE:
                continue
            # Break at C-C single bonds in the backbone (not ring bonds)
            # Also at C-O, C-N, C-S, etc.
            if s1 in ['C', 'N', 'O', 'S'] and s2 in ['C', 'N', 'O', 'S']:
                # Don't break if either atom is a dummy
                if s1 == '*' or s2 == '*':
                    continue
                # Prefer to break at interesting junctions
                bonds_to_break.append(bond.GetIdx())

        if len(bonds_to_break) == 0:
            return _ring_fragmentation(mol)

        # Limit to breaking at most a few bonds to get reasonable fragments
        # Break at bonds that produce 2-8 fragments
        max_breaks = min(len(bonds_to_break), 6)
        bonds_to_break = bonds_to_break[:max_breaks]

        # Use FragmentOnBonds which preserves atom indices
        frag_mol = Chem.FragmentOnBonds(mol, bonds_to_break, dummyLabels=[(0, 0)] * len(bonds_to_break))
        frag_smiles_all = Chem.MolToSmiles(frag_mol)

        # Get the atom mapping using Chem.GetMolFrags
        frags = Chem.GetMolFrags(frag_mol, asMols=True)
        frag_atom_groups = Chem.GetMolFrags(frag_mol)

        if len(frags) <= 1:
            return _ring_fragmentation(mol)

        # Filter out tiny fragments (just dummy atoms)
        atom_groups = []
        frag_smiles_list = []
        for i, (group, frag) in enumerate(zip(frag_atom_groups, frags)):
            # Check if fragment has real atoms (not just dummies)
            real_atoms = [a for a in frag.GetAtoms() if a.GetSymbol() != '*']
            if len(real_atoms) == 0:
                continue
            atom_groups.append(list(group))
            frag_smiles_list.append(Chem.MolToSmiles(frag))

        if len(atom_groups) <= 1:
            return _ring_fragmentation(mol)

        return atom_groups, frag_smiles_list
    except Exception:
        return _ring_fragmentation(mol)


def _ring_fragmentation(mol: Chem.Mol) -> Tuple[List[List[int]], List[str]]:
    """
    Fragment by separating ring systems from chains.
    Fallback fragmentation strategy.
    """
    try:
        ring_info = mol.GetRingInfo()
        n_atoms = mol.GetNumAtoms()

        # Find atoms in rings
        ring_atoms = set()
        for ring in ring_info.AtomRings():
            ring_atoms.update(ring)

        if len(ring_atoms) == 0:
            # No rings: split into functional groups
            return _functional_group_fragmentation(mol)

        # Find connected components of ring atoms
        ring_groups = _connected_components(mol, ring_atoms)

        # Find chain groups (non-ring atoms connected together)
        chain_atoms = set(range(n_atoms)) - ring_atoms
        if chain_atoms:
            chain_groups = _connected_components(mol, chain_atoms)
        else:
            chain_groups = []

        atom_groups = ring_groups + chain_groups
        frag_smiles_list = []
        for group in atom_groups:
            # Extract substructure SMILES
            try:
                submol = Chem.RWMol(mol)
                atoms_to_remove = sorted(set(range(n_atoms)) - set(group), reverse=True)
                for idx in atoms_to_remove:
                    submol.RemoveAtom(idx)
                submol = submol.GetMol()
                frag_smiles_list.append(Chem.MolToSmiles(submol))
            except Exception:
                frag_smiles_list.append(f"group_{len(group)}")

        if not atom_groups:
            return [list(range(n_atoms))], [Chem.MolToSmiles(mol)]

        return atom_groups, frag_smiles_list
    except Exception:
        return [list(range(mol.GetNumAtoms()))], [Chem.MolToSmiles(mol)]


def _functional_group_fragmentation(mol: Chem.Mol) -> Tuple[List[List[int]], List[str]]:
    """Split by functional groups (O, N, S-containing substructures)."""
    n_atoms = mol.GetNumAtoms()
    if n_atoms <= 3:
        return [list(range(n_atoms))], [Chem.MolToSmiles(mol)]

    # Identify heteroatom positions and split around them
    hetero_atoms = set()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ['O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I']:
            hetero_atoms.add(atom.GetIdx())
            # Include neighbors
            for nei in atom.GetNeighbors():
                hetero_atoms.add(nei.GetIdx())

    if not hetero_atoms:
        # Split into halves
        mid = n_atoms // 2
        return [list(range(mid)), list(range(mid, n_atoms))], [f"chain_0_{mid}", f"chain_{mid}_{n_atoms}"]

    groups = _connected_components(mol, hetero_atoms)
    remaining = set(range(n_atoms)) - set().union(*[set(g) for g in groups]) if groups else set(range(n_atoms))
    if remaining:
        groups.extend(_connected_components(mol, remaining))

    smiles_list = []
    for g in groups:
        try:
            submol = Chem.RWMol(mol)
            to_remove = sorted(set(range(n_atoms)) - set(g), reverse=True)
            for idx in to_remove:
                submol.RemoveAtom(idx)
            submol = submol.GetMol()
            smiles_list.append(Chem.MolToSmiles(submol))
        except Exception:
            smiles_list.append(f"group_{len(g)}")

    if not groups:
        return [list(range(n_atoms))], [Chem.MolToSmiles(mol)]
    return groups, smiles_list


def _connected_components(mol: Chem.Mol, atom_set: set) -> List[List[int]]:
    """Find connected components within a subset of atoms."""
    visited = set()
    components = []
    for start in atom_set:
        if start in visited:
            continue
        component = []
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            atom = mol.GetAtomWithIdx(node)
            for nei in atom.GetNeighbors():
                nei_idx = nei.GetIdx()
                if nei_idx in atom_set and nei_idx not in visited:
                    stack.append(nei_idx)
        components.append(component)
    return components


def mol_to_graph_data(mol: Chem.Mol, atom_offset: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert RDKit mol to atom features, edge index, and edge features."""
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(atom_features(atom))
    atom_feats = np.stack(atom_feats)

    edge_indices = []
    edge_feats = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx() + atom_offset
        j = bond.GetEndAtomIdx() + atom_offset
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        bf = bond_features(bond)
        edge_feats.append(bf)
        edge_feats.append(bf)

    if len(edge_indices) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, NUM_EDGE_FEATURES))
    else:
        edge_index = np.array(edge_indices, dtype=np.int64).T
        edge_features = np.stack(edge_feats)

    return atom_feats, edge_index, edge_features


class PolymerGraph:
    """Stores all graph data for a single polymer sample."""
    def __init__(self):
        # Atom-level trimer graph
        self.atom_features = None       # (N_atoms, D_atom)
        self.edge_index = None          # (2, N_edges)
        self.edge_features = None       # (N_edges, D_edge)
        self.periodic_edge_index = None # (2, 2) - periodic edges
        self.periodic_edge_features = None

        # Fragment-level data
        self.frag_atom_mapping = None   # list of lists of atom indices in trimer
        self.frag_features = None       # (N_frags, D_frag)
        self.frag_edge_index = None     # (2, N_frag_edges)
        self.frag_edge_features = None

        # Bipartite atom-fragment edges
        self.atom_frag_edge_index = None  # (2, N_atom_frag_edges)

        # Fragment attention mask
        self.frag_mask = None
        self.frag_smiles_list = None

        # Target
        self.tg = None
        self.p_smiles = None


def process_polymer(p_smiles: str, tg: float) -> Optional[PolymerGraph]:
    """Process a single polymer into a PolymerGraph."""
    pg = PolymerGraph()
    pg.tg = tg
    pg.p_smiles = p_smiles

    # ── Step 1: Build trimer graph ───────────────────────────────────
    trimer_mol = build_trimer_graph(p_smiles)
    if trimer_mol is None:
        return None

    # Get atom features and edges
    atom_feats, edge_index, edge_feats = mol_to_graph_data(trimer_mol)
    n_atoms = atom_feats.shape[0]

    # Find the star atoms (dummy atoms) and add periodic edge
    star_atoms = [a.GetIdx() for a in trimer_mol.GetAtoms() if a.GetSymbol() == '*']

    periodic_edges = []
    periodic_edge_feats = []
    if len(star_atoms) >= 2:
        # Periodic edge: connect first and last star atoms
        # These represent the connection points of the trimer
        # We connect the terminal star atoms to create periodic boundary
        for i in range(len(star_atoms)):
            for j in range(i + 1, len(star_atoms)):
                s1, s2 = star_atoms[i], star_atoms[j]
                periodic_edges.append([s1, s2])
                periodic_edges.append([s2, s1])
                # Use single bond type for periodic edge
                pe_feat = np.zeros(NUM_EDGE_FEATURES)
                pe_feat[0] = 1.0  # single bond
                pe_feat[-1] = 1.0  # in ring (periodic)
                periodic_edge_feats.append(pe_feat)
                periodic_edge_feats.append(pe_feat)

    if periodic_edges:
        periodic_edge_index = np.array(periodic_edges, dtype=np.int64).T
        periodic_edge_features = np.stack(periodic_edge_feats)
        edge_index = np.concatenate([edge_index, periodic_edge_index], axis=1)
        edge_feats = np.concatenate([edge_feats, periodic_edge_features], axis=0)

    pg.atom_features = atom_feats
    pg.edge_index = edge_index
    pg.edge_features = edge_feats

    # ── Step 2: Build cyclized monomer for fragmentation ─────────────
    cyclized_mol = build_cyclized_mol(p_smiles)
    if cyclized_mol is None:
        # Fallback: use trimer directly for fragmentation
        cyclized_mol = trimer_mol

    # ── Step 3: Fragment extraction ──────────────────────────────────
    frag_atom_groups, frag_smiles_list = smart_fragmentation(cyclized_mol)

    # Map fragment atoms to trimer atoms
    # The cyclized mol is the repeat unit; in the trimer, atoms are repeated 3x
    n_monomer_atoms = cyclized_mol.GetNumAtoms()
    trimer_frag_mapping = []

    for group in frag_atom_groups:
        trimer_group = []
        for rep in range(3):
            offset = rep * n_monomer_atoms
            for atom_idx in group:
                trimer_idx = atom_idx + offset
                if trimer_idx < n_atoms:
                    trimer_group.append(trimer_idx)
        trimer_frag_mapping.append(list(set(trimer_group)))

    pg.frag_atom_mapping = trimer_frag_mapping
    pg.frag_smiles_list = frag_smiles_list

    # Compute fragment features: mean of member atom features + frag size + func group indicators
    n_frags = len(trimer_frag_mapping)
    frag_feats = []
    for fi, group in enumerate(trimer_frag_mapping):
        if len(group) > 0:
            group_arr = np.array(group)
            valid = group_arr[group_arr < n_atoms]
            if len(valid) > 0:
                mean_atom_feat = np.mean(atom_feats[valid], axis=0)
            else:
                mean_atom_feat = np.zeros(NUM_ATOM_FEATURES)
        else:
            mean_atom_feat = np.zeros(NUM_ATOM_FEATURES)
        frag_size = np.array([len(group) / max(n_atoms, 1)])
        # Functional group indicators from fragment SMILES
        frag_smiles = frag_smiles_list[fi] if fi < len(frag_smiles_list) else ""
        has_aromatic = np.array([1.0 if 'c' in str(frag_smiles) else 0.0])
        has_oh = np.array([1.0 if 'O' in str(frag_smiles) else 0.0])
        has_nh = np.array([1.0 if 'N' in str(frag_smiles) else 0.0])
        frag_feat = np.concatenate([mean_atom_feat, frag_size, has_aromatic, has_oh, has_nh])
        frag_feats.append(frag_feat)

    pg.frag_features = np.stack(frag_feats)

    # ── Step 4: Fragment graph edges ─────────────────────────────────
    # Two fragments are connected if any of their atoms are bonded in the trimer
    frag_edge_list = []
    frag_edge_feat_list = []
    for i in range(n_frags):
        for j in range(i + 1, n_frags):
            connected = False
            for ai in trimer_frag_mapping[i]:
                for aj in trimer_frag_mapping[j]:
                    # Check if these atoms share an edge
                    for e in range(edge_index.shape[1]):
                        if (edge_index[0, e] == ai and edge_index[1, e] == aj) or \
                           (edge_index[0, e] == aj and edge_index[1, e] == ai):
                            connected = True
                            break
                    if connected:
                        break
                if connected:
                    break
            if connected:
                frag_edge_list.append([i, j])
                frag_edge_list.append([j, i])
                fe_feat = np.zeros(NUM_EDGE_FEATURES)
                fe_feat[0] = 1.0
                frag_edge_feat_list.append(fe_feat)
                frag_edge_feat_list.append(fe_feat)

    if frag_edge_list:
        pg.frag_edge_index = np.array(frag_edge_list, dtype=np.int64).T
        pg.frag_edge_features = np.stack(frag_edge_feat_list)
    else:
        pg.frag_edge_index = np.zeros((2, 0), dtype=np.int64)
        pg.frag_edge_features = np.zeros((0, NUM_EDGE_FEATURES))

    # ── Step 5: Bipartite atom-fragment edges ────────────────────────
    af_edges = []
    for frag_idx, group in enumerate(trimer_frag_mapping):
        for atom_idx in group:
            af_edges.append([atom_idx, frag_idx])

    if af_edges:
        pg.atom_frag_edge_index = np.array(af_edges, dtype=np.int64).T
    else:
        pg.atom_frag_edge_index = np.zeros((2, 0), dtype=np.int64)

    pg.frag_mask = np.ones(n_frags)

    return pg


class PPFNDataset(Dataset):
    """Dataset for Polymer Periodic Fragment Network."""

    def __init__(self, csv_path: str, split: str = 'train',
                 train_ratio: float = 0.72, val_ratio: float = 0.08,
                 seed: int = 42, max_samples: int = None):
        """
        Args:
            csv_path: Path to Tg.csv
            split: 'train', 'val', or 'test'
            train_ratio: fraction for training
            val_ratio: fraction for validation
            seed: random seed for reproducibility
            max_samples: limit total samples (for debugging)
        """
        self.split = split
        df = pd.read_csv(csv_path)
        if max_samples:
            df = df.head(max_samples)

        # Deterministic split
        np.random.seed(seed)
        n = len(df)
        indices = np.random.permutation(n)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if split == 'train':
            sel_indices = indices[:n_train]
        elif split == 'val':
            sel_indices = indices[n_train:n_train + n_val]
        else:  # test
            sel_indices = indices[n_train + n_val:]

        df_split = df.iloc[sel_indices].reset_index(drop=True)

        # Preprocess all polymers
        self.graphs = []
        self.failed = 0
        print(f"Processing {len(df_split)} {split} samples...")

        for idx, row in df_split.iterrows():
            pg = process_polymer(row['Smiles'], row['Tg'])
            if pg is not None:
                self.graphs.append(pg)
            else:
                self.failed += 1

        print(f"  Successfully processed: {len(self.graphs)}, Failed: {self.failed}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        pg = self.graphs[idx]

        return {
            'atom_features': torch.FloatTensor(pg.atom_features),
            'edge_index': torch.LongTensor(pg.edge_index),
            'edge_features': torch.FloatTensor(pg.edge_features),
            'frag_features': torch.FloatTensor(pg.frag_features),
            'frag_edge_index': torch.LongTensor(pg.frag_edge_index),
            'frag_edge_features': torch.FloatTensor(pg.frag_edge_features),
            'atom_frag_edge_index': torch.LongTensor(pg.atom_frag_edge_index),
            'frag_mask': torch.FloatTensor(pg.frag_mask),
            'tg': torch.FloatTensor([pg.tg]),
            'n_atoms': pg.atom_features.shape[0],
            'n_frags': pg.frag_features.shape[0],
            'frag_atom_mapping': pg.frag_atom_mapping,
            'frag_smiles_list': pg.frag_smiles_list,
            'p_smiles': pg.p_smiles,
        }


def collate_fn(batch):
    """Custom collate function for batching graphs with different sizes."""
    atom_features_list = []
    edge_index_list = []
    edge_features_list = []
    frag_features_list = []
    frag_edge_index_list = []
    frag_edge_features_list = []
    atom_frag_edge_index_list = []
    frag_mask_list = []
    tg_list = []
    batch_atom_indices = []
    batch_frag_indices = []
    n_atoms_offset = 0
    n_frags_offset = 0

    # For interpretability
    frag_atom_mappings = []
    frag_smiles_lists = []
    p_smiles_list = []

    for i, sample in enumerate(batch):
        n_atoms = sample['n_atoms']
        n_frags = sample['n_frags']

        atom_features_list.append(sample['atom_features'])
        edge_index_list.append(sample['edge_index'] + n_atoms_offset)
        edge_features_list.append(sample['edge_features'])

        frag_features_list.append(sample['frag_features'])
        frag_edge_index_list.append(sample['frag_edge_index'] + n_frags_offset)
        frag_edge_features_list.append(sample['frag_edge_features'])

        # atom_frag_edge_index: rows are atom indices, cols are frag indices
        af_index = sample['atom_frag_edge_index'].clone()
        af_index[0] += n_atoms_offset
        af_index[1] += n_frags_offset
        atom_frag_edge_index_list.append(af_index)

        frag_mask_list.append(sample['frag_mask'])
        tg_list.append(sample['tg'])

        batch_atom_indices.append(torch.full((n_atoms,), i, dtype=torch.long))
        batch_frag_indices.append(torch.full((n_frags,), i, dtype=torch.long))

        frag_atom_mappings.append(sample['frag_atom_mapping'])
        frag_smiles_lists.append(sample['frag_smiles_list'])
        p_smiles_list.append(sample['p_smiles'])

        n_atoms_offset += n_atoms
        n_frags_offset += n_frags

    return {
        'atom_features': torch.cat(atom_features_list, dim=0),
        'edge_index': torch.cat(edge_index_list, dim=1),
        'edge_features': torch.cat(edge_features_list, dim=0),
        'frag_features': torch.cat(frag_features_list, dim=0),
        'frag_edge_index': torch.cat(frag_edge_index_list, dim=1),
        'frag_edge_features': torch.cat(frag_edge_features_list, dim=0),
        'atom_frag_edge_index': torch.cat(atom_frag_edge_index_list, dim=1),
        'frag_mask': torch.cat(frag_mask_list, dim=0),
        'tg': torch.cat(tg_list, dim=0),
        'batch_atom': torch.cat(batch_atom_indices, dim=0),
        'batch_frag': torch.cat(batch_frag_indices, dim=0),
        'n_atoms_offset': n_atoms_offset,
        'n_frags_offset': n_frags_offset,
        'frag_atom_mappings': frag_atom_mappings,
        'frag_smiles_lists': frag_smiles_lists,
        'p_smiles_list': p_smiles_list,
    }


if __name__ == '__main__':
    # Quick test
    csv_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    dataset = PPFNDataset(csv_path, split='train', max_samples=100)
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Atoms: {sample['n_atoms']}")
        print(f"  Edges: {sample['edge_index'].shape[1]}")
        print(f"  Fragments: {sample['n_frags']}")
        print(f"  Tg: {sample['tg'].item()}")
        print(f"  p-SMILES: {sample['p_smiles']}")

    # Test collation
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"\nBatch:")
    print(f"  Total atoms: {batch['atom_features'].shape[0]}")
    print(f"  Total frags: {batch['frag_features'].shape[0]}")
    print(f"  Total edges: {batch['edge_index'].shape[1]}")
    print(f"  Batch size: {batch['tg'].shape[0]}")
