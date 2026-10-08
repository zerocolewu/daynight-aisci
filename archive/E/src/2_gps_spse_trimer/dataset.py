"""
Dataset module for GPS-SPSE-Trimer Tg prediction.
Constructs trimer graphs. SPSE is computed on-the-fly in the model.
"""

import torch
import pandas as pd
import numpy as np
import os
import logging
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import rdmolops
from collections import deque

logger = logging.getLogger(__name__)

# --- Atom features (30 dims) ---
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
# Total: 10 + 6 + 4 + 1 + 1 + 6 + 2 = 30
ATOM_FEATURE_DIM = len(ATOM_TYPES) + len(DEGREES) + len(HYBRIDIZATIONS) + 1 + 1 + len(H_COUNTS) + len(CHIRALITY)

# Bond features (6 dims)
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]  # 4
BOND_FEATURE_DIM = len(BOND_TYPES) + 1 + 1


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


# --- p-SMILES to trimer conversion ---

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
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]


def get_star_neighbors(mol, star_indices):
    star_to_neighbor = {}
    for star_idx in star_indices:
        atom = mol.GetAtomWithIdx(star_idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() != 0:
                star_to_neighbor[star_idx] = nbr.GetIdx()
                break
    return star_to_neighbor


def psmiles_to_trimer(p_smiles):
    """Convert p-SMILES to trimer (3 repeat units connected).
    Returns (mol, unit_boundaries) or (None, None) on failure.
    """
    try:
        mol = Chem.MolFromSmiles(p_smiles)
        if mol is None:
            return None, None

        star_indices = find_star_atoms(mol)
        if len(star_indices) != 2:
            return None, None

        star_to_neighbor = get_star_neighbors(mol, star_indices)
        if len(star_to_neighbor) != 2:
            return None, None

        stars = list(star_to_neighbor.keys())
        head_star, tail_star = stars[0], stars[1]
        head_nbr_orig = star_to_neighbor[head_star]
        tail_nbr_orig = star_to_neighbor[tail_star]

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

        combined = Chem.RWMol(Chem.Mol())
        unit_boundaries = []

        for unit in range(3):
            offset = combined.GetNumAtoms()
            unit_boundaries.append((offset, offset + n_monomer_atoms))
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

        for i in range(2):
            tail_of_i = unit_boundaries[i][0] + monomer_tail
            head_of_next = unit_boundaries[i + 1][0] + monomer_head
            combined.AddBond(tail_of_i, head_of_next, Chem.BondType.SINGLE)

        result = combined.GetMol()
        if not safe_sanitize(result):
            return None, None

        return result, unit_boundaries
    except Exception as e:
        logger.warning(f"Failed trimer conversion for {p_smiles}: {e}")
        return None, None


# --- SPSE utilities ---

def compute_shortest_path_distances(n_nodes, edge_index_np):
    """Compute all-pairs shortest path distances using BFS.
    Args:
        n_nodes: number of nodes
        edge_index_np: (2, E) numpy array
    Returns:
        distances: (n_nodes, n_nodes) numpy array, 1000 for unreachable
    """
    adj = [[] for _ in range(n_nodes)]
    for e in range(edge_index_np.shape[1]):
        u, v = int(edge_index_np[0, e]), int(edge_index_np[1, e])
        if u != v:
            adj[u].append(v)

    distances = np.full((n_nodes, n_nodes), 1000, dtype=np.int32)

    for start in range(n_nodes):
        dist = [-1] * n_nodes
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for i in range(n_nodes):
            if dist[i] >= 0:
                distances[start, i] = dist[i]

    np.fill_diagonal(distances, 0)
    return distances


# --- Dataset ---

class TgTrimerDataset(InMemoryDataset):
    """Tg dataset with trimer representation. SPSE computed on-the-fly."""

    def __init__(self, root, csv_path, spse_dim=128, transform=None,
                 pre_transform=None, pre_filter=None):
        self.csv_path = csv_path
        self.spse_dim = spse_dim
        self._spse_distances = None  # lazy loaded
        super(TgTrimerDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def spse_distances(self):
        """Lazy-load SPSE distances."""
        if self._spse_distances is None:
            spse_path = os.path.join(self.processed_dir, 'spse_distances.pt')
            if os.path.exists(spse_path):
                self._spse_distances = torch.load(spse_path, weights_only=False)
            else:
                raise FileNotFoundError(f"SPSE distances not found at {spse_path}")
        return self._spse_distances

    @property
    def raw_file_names(self):
        return ['Tg.csv']

    @property
    def processed_file_names(self):
        return ['tg_trimer_spse.pt']

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.csv_path)
        smiles_list = df['Smiles'].tolist()
        tg_values = df['Tg'].tolist()

        data_list = []
        spse_list = []
        failed = 0

        for i, (smi, tg) in enumerate(zip(smiles_list, tg_values)):
            result, sp_dist = self._process_molecule(smi, tg)
            if result is not None:
                data_list.append(result)
                spse_list.append(sp_dist)
            else:
                failed += 1

        if failed > 0:
            logger.warning(f"Failed to process {failed}/{len(smiles_list)} molecules")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        # Save SPSE distances as separate file (list of tensors, variable sizes)
        torch.save(spse_list, os.path.join(self.processed_dir, 'spse_distances.pt'))
        logger.info(f"Processed {len(data_list)} molecules successfully")

    def _process_molecule(self, p_smiles, tg):
        mol, unit_boundaries = psmiles_to_trimer(p_smiles)
        if mol is None:
            return None, None

        n_atoms = mol.GetNumAtoms()
        if n_atoms == 0:
            return None, None

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

        # Compute shortest path distances for SPSE
        sp_distances = compute_shortest_path_distances(n_atoms, edge_index.numpy())

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([tg], dtype=torch.float),
            num_nodes=n_atoms,
        )
        return data, torch.tensor(sp_distances, dtype=torch.long)


def get_data_splits(dataset, seed=71, train_ratio=0.72, val_ratio=0.08):
    """Split dataset deterministically by index."""
    n = len(dataset)
    indices = list(range(n))
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    return train_indices, val_indices, test_indices
