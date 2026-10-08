"""
Augmentation strategies for contrastive pre-training.
1. MonomerTrimer (baseline): positive pair = (monomer, trimer)
2. SubgraphCrop: positive pair = (random_subgraph_1, random_subgraph_2) from same trimer
3. ShiftedRU: positive pair = (standard_trimer, shifted_trimer)
4. EdgeDropFeatMask: positive pair = (original_trimer, corrupted_trimer) with edge dropout + feature masking
5. MonomerDimer: positive pair = (monomer, dimer)
6. HybridCorrupt: positive pair = (corrupted_monomer, corrupted_trimer) with edge dropout
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import logging

logger = logging.getLogger(__name__)


def random_connected_subgraph(data, min_ratio=0.3, max_ratio=0.7):
    """
    Extract a random connected subgraph from a PyG Data object.

    Args:
        data: PyG Data with x, edge_index, edge_attr, num_nodes
        min_ratio: minimum fraction of nodes to keep
        max_ratio: maximum fraction of nodes to keep

    Returns:
        PyG Data representing the subgraph, or None on failure
    """
    n_nodes = data.num_nodes
    if n_nodes <= 3:
        return None

    # Target number of nodes
    target_n = max(3, int(n_nodes * np.random.uniform(min_ratio, max_ratio)))

    # Pick a random start node
    start_node = np.random.randint(0, n_nodes)

    # BFS to grow the subgraph
    visited = {start_node}
    frontier = [start_node]
    edge_index_np = data.edge_index.cpu().numpy()

    # Build adjacency list
    adj = [[] for _ in range(n_nodes)]
    for e in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, e], edge_index_np[1, e]
        adj[src].append((dst, e))

    selected_edges = set()

    while len(visited) < target_n and frontier:
        next_frontier = []
        np.random.shuffle(frontier)
        for node in frontier:
            neighbors = adj[node]
            np.random.shuffle(neighbors)
            for nbr, eidx in neighbors:
                if nbr not in visited and len(visited) < target_n:
                    visited.add(nbr)
                    selected_edges.add(eidx)
                    # Also add the reverse edge
                    next_frontier.append(nbr)
        frontier = next_frontier

    if len(visited) < 3:
        return None

    # Map old indices to new indices
    node_list = sorted(visited)
    old_to_new = {old: new for new, old in enumerate(node_list)}

    # Extract node features
    new_x = data.x[node_list]

    # Extract edges (both directions)
    new_edge_list = []
    new_edge_attrs = []
    edge_attr = data.edge_attr
    for e in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, e], edge_index_np[1, e]
        if src in visited and dst in visited:
            new_src = old_to_new[src]
            new_dst = old_to_new[dst]
            new_edge_list.append([new_src, new_dst])
            if edge_attr is not None:
                new_edge_attrs.append(edge_attr[e])

    if len(new_edge_list) == 0:
        return None

    new_edge_index = torch.tensor(new_edge_list, dtype=torch.long).t().contiguous()
    new_edge_attr = torch.stack(new_edge_attrs) if new_edge_attrs else None

    return Data(
        x=new_x,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        num_nodes=len(node_list),
    )


def shifted_trimer(p_smiles, shift_k=None):
    """
    Generate a shifted trimer view from p-SMILES.
    Instead of connecting unit[i].tail -> unit[i+1].head,
    start from a different atom in the repeat unit boundary.

    This creates a structurally different graph (different atom ordering,
    different boundary atoms) but semantically equivalent (same periodic polymer).

    Implementation: shift the head/tail connection points within the repeat unit.
    The repeat unit has monomer_head and monomer_tail atoms. For the shifted view,
    we swap which atom is considered head vs tail, effectively mirroring the
    polymer chain direction.

    Args:
        p_smiles: polymer SMILES string
        shift_k: ignored (we use head/tail swap instead)

    Returns:
        RDKit Mol or None
    """
    from rdkit import Chem
    from graph_utils import safe_sanitize, find_star_atoms, get_star_neighbors

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

        # Remove star atoms
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

        # Build trimer with REVERSED connection order
        # Standard: tail_of_i -> head_of_next
        # Shifted: head_of_i -> tail_of_next  (reversed chain direction)
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

        # Connect units with REVERSED direction: head_of_i -> tail_of_next
        for i in range(2):
            head_of_i = i * n_monomer_atoms + monomer_head
            tail_of_next = (i + 1) * n_monomer_atoms + monomer_tail
            combined.AddBond(head_of_i, tail_of_next, Chem.BondType.SINGLE)

        result = combined.GetMol()
        if not safe_sanitize(result):
            return None

        return result
    except Exception as e:
        logger.debug(f"Failed shifted trimer for {p_smiles}: {e}")
        return None


class SubgraphCropDataset(torch.utils.data.Dataset):
    """
    Dataset for subgraph cropping augmentation.
    Each item returns two random subgraphs cropped from the same trimer.
    Subgraphs are generated on-the-fly for maximum diversity.
    """

    def __init__(self, trimer_list):
        self.trimer_list = trimer_list

    def __len__(self):
        return len(self.trimer_list)

    def __getitem__(self, idx):
        trimer = self.trimer_list[idx]
        # Generate two different random subgraphs
        sub1 = random_connected_subgraph(trimer)
        sub2 = random_connected_subgraph(trimer)

        if sub1 is None:
            sub1 = trimer
        if sub2 is None:
            sub2 = trimer

        return sub1, sub2


class ShiftedRUDataset(torch.utils.data.Dataset):
    """
    Dataset for shifted RU augmentation.
    Each item returns (standard_trimer, shifted_trimer) as positive pair.
    The shifted trimer is pre-computed since it's deterministic.
    """

    def __init__(self, standard_trimer_list, shifted_trimer_list):
        self.standard_list = standard_trimer_list
        self.shifted_list = shifted_trimer_list

    def __len__(self):
        return len(self.standard_list)

    def __getitem__(self, idx):
        return self.standard_list[idx], self.shifted_list[idx]


# ============================================================
# Iteration 3: Moderate-difficulty augmentations
# ============================================================

def edge_dropout(data, p=0.25):
    """
    Randomly drop p fraction of edges from a PyG Data graph.
    Returns a new Data object with fewer edges.

    Args:
        data: PyG Data with x, edge_index, edge_attr, num_nodes
        p: probability of dropping each edge pair

    Returns:
        New PyG Data with some edges removed
    """
    n_edges = data.edge_index.size(1)
    if n_edges <= 2:
        return data

    # Process edge pairs (each bond has 2 directed edges)
    # We drop both directions together to keep the graph undirected
    n_undirected = n_edges // 2
    if n_undirected <= 1:
        return data

    # Create mask: keep each undirected edge pair with probability (1-p)
    keep_mask = np.random.rand(n_undirected) > p

    # Ensure at least 1 edge remains
    if not keep_mask.any():
        keep_mask[np.random.randint(n_undirected)] = True

    # Expand to directed mask
    directed_keep = np.repeat(keep_mask, 2)

    new_edge_index = data.edge_index[:, directed_keep]
    new_edge_attr = data.edge_attr[directed_keep] if data.edge_attr is not None else None

    return Data(
        x=data.x.clone(),
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        num_nodes=data.num_nodes,
    )


def feature_mask(data, p=0.25):
    """
    Randomly mask p fraction of node feature dimensions (set to 0).
    Applies the same mask to all nodes (dimension-level masking, like GraphCL).

    Args:
        data: PyG Data with x, edge_index, edge_attr, num_nodes
        p: fraction of feature dimensions to mask

    Returns:
        New PyG Data with some features zeroed out
    """
    n_feat = data.x.size(1)
    # Create a binary mask over feature dimensions
    mask = torch.rand(n_feat) > p
    # Ensure at least half the features remain
    if mask.sum() < n_feat // 2:
        mask[:n_feat // 2] = True

    new_x = data.x.clone()
    new_x = new_x * mask.unsqueeze(0).float()

    return Data(
        x=new_x,
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
        num_nodes=data.num_nodes,
    )


def edge_dropout_feat_mask(data, edge_p=0.25, feat_p=0.25):
    """
    Combined augmentation: edge dropout + feature masking.
    This creates a moderately corrupted view of the same graph.

    Args:
        data: PyG Data
        edge_p: fraction of edges to drop
        feat_p: fraction of feature dimensions to mask

    Returns:
        New PyG Data with both augmentations applied
    """
    corrupted = edge_dropout(data, p=edge_p)
    corrupted = feature_mask(corrupted, p=feat_p)
    return corrupted


def psmiles_to_dimer(p_smiles):
    """
    Convert p-SMILES to dimer (2 repeat units connected).
    Returns RDKit Mol or None on failure.
    """
    try:
        from graph_utils import safe_sanitize, find_star_atoms, get_star_neighbors
        from rdkit import Chem

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

        # Remove star atoms to get monomer
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

        # Build dimer (2 units)
        combined = Chem.RWMol(Chem.Mol())

        for unit in range(2):
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

        # Connect the 2 units: tail of unit 0 -> head of unit 1
        tail_of_0 = 0 * n_monomer_atoms + monomer_tail
        head_of_1 = 1 * n_monomer_atoms + monomer_head
        combined.AddBond(tail_of_0, head_of_1, Chem.BondType.SINGLE)

        result = combined.GetMol()
        if not safe_sanitize(result):
            return None

        return result
    except Exception as e:
        logger.debug(f"Failed dimer conversion for {p_smiles}: {e}")
        return None
