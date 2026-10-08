"""
Data preprocessing for PolymerBipartite v2 experiment.
Constructs trimer graphs with RU (Repeat Unit) assignment metadata.

Trimer structure: RU0 - RU1 - RU2
Each RU is a monomer unit connected sequentially.
"""

import os
import logging
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from rdkit import Chem

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Feature Definitions (same as PCR v2 reference)
# ============================================================================

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
ATOM_FEATURE_DIM = 10 + 6 + 4 + 1 + 1 + 6 + 2

BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]  # 4
BOND_FEATURE_DIM = 4 + 1 + 1  # 6


# ============================================================================
# Helper Functions (from PCR v2 reference, exact match)
# ============================================================================

def one_hot(value, choices):
    vec = [0] * len(choices)
    try:
        idx = choices.index(value)
        vec[idx] = 1
    except ValueError:
        pass
    return vec


def get_atom_features(atom):
    """30-dim atom features: type(10) + degree(6) + hybridization(4) + aromatic(1) + ring(1) + Hs(6) + chirality(2)."""
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
    """6-dim bond features: type(4) + conjugated(1) + ring(1)."""
    bond_type = one_hot(bond.GetBondType(), BOND_TYPES)
    conjugated = [int(bond.GetIsConjugated())]
    ring_feat = [int(bond.IsInRing())]
    return bond_type + conjugated + ring_feat


def safe_sanitize(mol):
    """Multi-stage sanitization with fallbacks."""
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
    """Find star atoms (atomic number 0)."""
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]


def get_star_neighbors(mol, star_indices):
    """Map star atom index -> neighbor atom index."""
    star_to_neighbor = {}
    for star_idx in star_indices:
        atom = mol.GetAtomWithIdx(star_idx)
        neighbors = atom.GetNeighbors()
        for nbr in neighbors:
            if nbr.GetAtomicNum() != 0:
                star_to_neighbor[star_idx] = nbr.GetIdx()
                break
    return star_to_neighbor


# ============================================================================
# Trimer Construction with RU Metadata
# ============================================================================

def build_trimer_with_ru_metadata(p_smiles):
    """
    Build trimer graph with RU assignment metadata.

    Uses the proven PCR v2 trimer construction approach:
    1. Parse p-SMILES, identify star atoms
    2. Create monomer (remove star atoms)
    3. Build trimer from 3 copies connected at star positions
    4. Track RU assignment, boundary atoms, positions
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

        # Create monomer by removing star atoms
        rw_mol = Chem.RWMol(mol)
        remove_order = sorted([head_star, tail_star], reverse=True)
        for idx in remove_order:
            rw_mol.RemoveAtom(idx)

        # Update connection point indices after atom removal
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

        # Build trimer by combining 3 copies (exact PCR v2 approach)
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

        # Connect units: unit[i].tail -> unit[i+1].head
        inter_ru_bonds = []  # Track inter-RU bond pairs for boundary detection
        for i in range(2):
            tail_of_i = i * n_monomer_atoms + monomer_tail
            head_of_next = (i + 1) * n_monomer_atoms + monomer_head
            combined.AddBond(tail_of_i, head_of_next, Chem.BondType.SINGLE)
            inter_ru_bonds.append((tail_of_i, head_of_next))

        result = combined.GetMol()
        if not safe_sanitize(result):
            return None

        # Extract features
        num_nodes = result.GetNumAtoms()

        # Atom features
        x = []
        for atom in result.GetAtoms():
            x.append(get_atom_features(atom))
        x = torch.tensor(x, dtype=torch.float)

        # Edge features
        edge_index_list = []
        edge_attr_list = []
        for bond in result.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            feat = get_bond_features(bond)
            edge_index_list.append([begin, end])
            edge_index_list.append([end, begin])
            edge_attr_list.append(feat)
            edge_attr_list.append(feat)

        if len(edge_index_list) == 0:
            return None

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

        # Build RU metadata
        ru_assignment = torch.zeros(num_nodes, dtype=torch.long)
        ru_position = torch.zeros(num_nodes, dtype=torch.float)
        is_boundary = torch.zeros(num_nodes, dtype=torch.float)

        # Boundary atoms: those connected across RU boundaries
        boundary_set = set()
        for (a, b) in inter_ru_bonds:
            boundary_set.add(a)
            boundary_set.add(b)

        for unit in range(3):
            start = unit * n_monomer_atoms
            end = start + n_monomer_atoms
            ru_assignment[start:end] = unit

            for local_idx in range(n_monomer_atoms):
                global_idx = start + local_idx
                if n_monomer_atoms > 1:
                    ru_position[global_idx] = local_idx / (n_monomer_atoms - 1)
                else:
                    ru_position[global_idx] = 0.5
                if global_idx in boundary_set:
                    is_boundary[global_idx] = 1.0

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            ru_assignment=ru_assignment,
            is_boundary=is_boundary,
            ru_position=ru_position,
            num_nodes=num_nodes,
        )

        return data

    except Exception as e:
        logger.warning(f"Failed trimer construction for {p_smiles}: {e}")
        return None


# ============================================================================
# Dataset Class
# ============================================================================

class PolymerBipartiteDataset:
    """
    Simple list-based dataset for polymer trimer graphs with RU metadata.
    Caches preprocessed data to disk.
    """

    def __init__(self, csv_path, cache_dir=None):
        self.csv_path = csv_path
        self.cache_dir = cache_dir or os.path.dirname(csv_path)
        self.graphs = []
        self.targets = []
        self.valid_mask = []

        cache_path = os.path.join(self.cache_dir, 'bipartite_v2_cache.pt')
        if os.path.exists(cache_path):
            logger.info(f"Loading cached data from {cache_path}")
            cached = torch.load(cache_path, weights_only=False)
            self.graphs = cached['graphs']
            self.targets = cached['targets']
            self.valid_mask = cached['valid_mask']
        else:
            self._process_and_save(cache_path)

        self.targets = torch.tensor(self.targets, dtype=torch.float)
        self.valid_mask = torch.tensor(self.valid_mask, dtype=torch.bool)
        logger.info(f"Dataset: {len(self.targets)} total, {self.valid_mask.sum().item()} valid")

    def _process_and_save(self, save_path):
        df = pd.read_csv(self.csv_path)
        smiles_list = df['Smiles'].tolist()
        tg_values = df['Tg'].tolist()

        failed = 0
        for i, (smi, tg) in enumerate(zip(smiles_list, tg_values)):
            if (i + 1) % 1000 == 0:
                logger.info(f"  Processed {i + 1}/{len(smiles_list)}")

            data = build_trimer_with_ru_metadata(smi)

            if data is not None:
                self.graphs.append(data)
                self.targets.append(tg)
                self.valid_mask.append(True)
            else:
                # Dummy graph for failed conversions
                dummy = Data(
                    x=torch.zeros(1, ATOM_FEATURE_DIM),
                    edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                    edge_attr=torch.zeros(1, BOND_FEATURE_DIM),
                    ru_assignment=torch.zeros(1, dtype=torch.long),
                    is_boundary=torch.zeros(1, dtype=torch.float),
                    ru_position=torch.tensor([0.5]),
                    num_nodes=1,
                )
                self.graphs.append(dummy)
                self.targets.append(tg)
                self.valid_mask.append(False)
                failed += 1

        logger.info(f"Processed {len(smiles_list)} polymers, {failed} failures")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'graphs': self.graphs,
            'targets': self.targets,
            'valid_mask': self.valid_mask,
        }, save_path)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.graphs[idx]

    @property
    def num_node_features(self):
        return ATOM_FEATURE_DIM

    @property
    def num_edge_features(self):
        return BOND_FEATURE_DIM


# ============================================================================
# Data Splitting (exact match with reference: index-based, no shuffling)
# ============================================================================

def get_data_splits(dataset_len, seed=71, train_ratio=0.72, val_ratio=0.08):
    """Split dataset by index. 72/8/20 split (same as Day 1)."""
    n = dataset_len
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_indices = list(range(0, n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, n))
    return train_indices, val_indices, test_indices


# ============================================================================
# Diagnostic Helper: Random RU Assignment
# ============================================================================

def create_random_ru_dataset(graphs, targets, valid_mask, seed=71):
    """
    Create a dataset copy with randomly shuffled atom-to-RU assignments.
    For Diagnostic 2: verifies that structural RU correspondence matters.

    For each graph, randomly shuffles which atoms belong to which RU,
    keeping RU sizes the same but breaking structural correspondence.
    """
    rng = np.random.default_rng(seed)
    new_graphs = []

    for data in graphs:
        new_data = data.clone()
        n = data.num_nodes

        # Get original RU sizes
        ru_sizes = [(data.ru_assignment == k).sum().item() for k in range(3)]

        # Random permutation of atom indices
        perm = rng.permutation(n)

        new_assignment = torch.zeros(n, dtype=torch.long)
        pos = 0
        for k, size in enumerate(ru_sizes):
            indices = perm[pos:pos + size]
            new_assignment[indices] = k
            pos += size

        new_data.ru_assignment = new_assignment

        # Recompute is_boundary based on new assignment and actual edges
        new_is_boundary = torch.zeros(n, dtype=torch.float)
        src, dst = data.edge_index
        for i in range(n):
            neighbors = dst[src == i]
            for nbr in neighbors:
                if new_assignment[i] != new_assignment[nbr]:
                    new_is_boundary[i] = 1.0
                    break
        new_data.is_boundary = new_is_boundary

        # Recompute ru_position
        new_position = torch.zeros(n, dtype=torch.float)
        for k in range(3):
            mask = new_assignment == k
            indices = mask.nonzero(as_tuple=False).flatten().tolist()
            if len(indices) > 1:
                for rank, idx in enumerate(indices):
                    new_position[idx] = rank / (len(indices) - 1)
            elif len(indices) == 1:
                new_position[indices[0]] = 0.5
        new_data.ru_position = new_position

        new_graphs.append(new_data)

    return new_graphs, targets, valid_mask


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(os.path.dirname(current_dir), 'alldataset', 'Tg.csv')

    if not os.path.exists(csv_path):
        print(f"ERROR: Tg.csv not found at {csv_path}")
        exit(1)

    dataset = PolymerBipartiteDataset(csv_path, cache_dir=current_dir)

    print(f"\nDataset statistics:")
    print(f"  Total: {len(dataset)}")
    print(f"  Valid: {dataset.valid_mask.sum().item()}")
    print(f"  Node features: {dataset.num_node_features}")
    print(f"  Edge features: {dataset.num_edge_features}")

    train_idx, val_idx, test_idx = get_data_splits(len(dataset))
    print(f"\nSplits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    sample = dataset[0]
    print(f"\nSample 0: x={sample.x.shape}, edge_index={sample.edge_index.shape}")
    print(f"  ru_assignment={sample.ru_assignment}")
    print(f"  is_boundary={sample.is_boundary}")
    print(f"  ru_position={sample.ru_position}")
