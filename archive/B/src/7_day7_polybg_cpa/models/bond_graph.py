"""
Bond graph (line graph) construction for polymer molecular graphs.

Given an atom-level PyG Data object, constructs a bond graph where:
- Each bond (edge in atom graph) becomes a node in the bond graph
- Two bond-nodes are connected if their corresponding bonds share a common atom
- Bond node features are computed from atom types, bond types, and polymer structure

The module also needs SMILES information to compute conjugation, rotatable bonds,
and backbone identification via RDKit.
"""

import torch
import numpy as np
from collections import deque
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


# ==============================================================================
# Atom-level PSPE computation (for backbone identification)
# ==============================================================================

def bfs_farthest(edge_index, num_nodes, start_node):
    adj = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        adj[u].append(v)
        adj[v].append(u)

    dist = [-1] * num_nodes
    dist[start_node] = 0
    q = deque([start_node])
    farthest_node = start_node

    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
                if dist[v] > dist[farthest_node]:
                    farthest_node = v

    return dist, farthest_node


def bfs_path(edge_index, num_nodes, start, end):
    adj = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        adj[u].append(v)
        adj[v].append(u)

    parent = [-1] * num_nodes
    parent[start] = start
    q = deque([start])
    found = False

    while q and not found:
        u = q.popleft()
        for v in adj[u]:
            if parent[v] == -1:
                parent[v] = u
                if v == end:
                    found = True
                    break
                q.append(v)

    if not found:
        return []

    path = []
    cur = end
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    path.reverse()
    return path


def find_backbone_path(edge_index, num_nodes):
    """Find longest shortest path (backbone) via double-BFS."""
    if num_nodes <= 1:
        path = [0] if num_nodes == 1 else []
        return set(path), path, len(path)

    _, node_a = bfs_farthest(edge_index, num_nodes, 0)
    _, node_b = bfs_farthest(edge_index, num_nodes, node_a)
    path = bfs_path(edge_index, num_nodes, node_a, node_b)
    return set(path), path, len(path)


def compute_bond_features_for_graph(data, smiles):
    """
    Compute 14-dim bond node features for each bond in the atom-level graph.

    Uses RDKit to parse the SMILES and extract bond-level properties,
    then maps them to the bonds in data.edge_index.

    Returns:
        bond_features: tensor [num_bonds, 14] where num_bonds = data.edge_index.shape[1]
        bond_to_atom: tensor [num_bonds, 2] mapping bond_idx -> (atom_u, atom_v)
    """
    edge_index = data.edge_index  # [2, E] - undirected, both directions
    num_edges = edge_index.shape[1]

    # Parse SMILES with RDKit
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(num_edges, 14), edge_index.t()

    mol = Chem.RemoveHs(mol)

    num_atoms = mol.GetNumAtoms()
    num_mol_bonds = mol.GetNumBonds()

    # Build mapping: (atom_u, atom_v) -> rdkit_bond_idx
    bond_map = {}
    for b_idx in range(num_mol_bonds):
        bond = mol.GetBondWithIdx(b_idx)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bond_map[(u, v)] = b_idx
        bond_map[(v, u)] = b_idx

    # Backbone identification
    backbone_set = set()
    backbone_path = []
    if num_atoms > 0:
        # Find [*] atoms (dummy atoms for polymer ends)
        star_atoms = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == '*':
                star_atoms.append(atom.GetIdx())

        # Find backbone: shortest path between the two [*] atoms
        if len(star_atoms) >= 2:
            # Build adjacency list from RDKit mol
            mol_adj = [[] for _ in range(num_atoms)]
            for bond in mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                mol_adj[u].append(v)
                mol_adj[v].append(u)

            # BFS between the two star atoms
            start, end = star_atoms[0], star_atoms[1]
            parent = [-1] * num_atoms
            parent[start] = start
            q = deque([start])
            found = False
            while q and not found:
                u = q.popleft()
                for v in mol_adj[u]:
                    if parent[v] == -1:
                        parent[v] = u
                        if v == end:
                            found = True
                            break
                        q.append(v)

            if found:
                cur = end
                backbone_path_list = []
                while cur != start:
                    backbone_path_list.append(cur)
                    cur = parent[cur]
                backbone_path_list.append(start)
                backbone_path_list.reverse()
                backbone_path = backbone_path_list
                backbone_set = set(backbone_path_list)

    # Build backbone position map for distance computation
    backbone_pos = {}
    for pos, node_idx in enumerate(backbone_path):
        backbone_pos[node_idx] = pos
    backbone_len = len(backbone_path)

    # Compute backbone bond distances: for each backbone bond, position along backbone
    backbone_bond_pos = {}
    for b_idx in range(num_mol_bonds):
        bond = mol.GetBondWithIdx(b_idx)
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if u in backbone_set and v in backbone_set:
            pos = (backbone_pos.get(u, -1) + backbone_pos.get(v, -1)) / 2.0
            backbone_bond_pos[b_idx] = pos

    # Compute atom electronegativity and monomer indices
    # Pauling electronegativity (simplified)
    en_map = {
        1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
        14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66
    }

    atom_en = {}
    monomer_idx = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom_en[idx] = en_map.get(atom.GetAtomicNum(), 2.5)
        if idx in backbone_pos:
            seg = min(int(backbone_pos[idx] / max(backbone_len / 3.0, 1)), 2)
            monomer_idx[idx] = seg / 2.0  # 0.0, 0.5, 1.0
        else:
            monomer_idx[idx] = -1.0

    # Average bond length estimates (typical, in Angstroms, normalized)
    bond_length_estimate = {
        Chem.BondType.SINGLE: 1.54,
        Chem.BondType.DOUBLE: 1.34,
        Chem.BondType.TRIPLE: 1.20,
        Chem.BondType.AROMATIC: 1.40,
    }

    # Now compute 14-dim features for each edge in data.edge_index
    atom_i = edge_index[0].numpy()  # [E]
    atom_j = edge_index[1].numpy()
    bond_feats = np.zeros((num_edges, 14), dtype=np.float32)

    for e_idx in range(num_edges):
        u, v = int(atom_i[e_idx]), int(atom_j[e_idx])

        # Find RDKit bond index
        rdkit_bond_idx = bond_map.get((u, v), -1)

        if rdkit_bond_idx >= 0:
            bond = mol.GetBondWithIdx(rdkit_bond_idx)

            # Feature 0-3: bond_type one-hot [single, double, triple, aromatic]
            bt = bond.GetBondType()
            if bt == Chem.BondType.SINGLE:
                bond_feats[e_idx, 0] = 1.0
            elif bt == Chem.BondType.DOUBLE:
                bond_feats[e_idx, 1] = 1.0
            elif bt == Chem.BondType.TRIPLE:
                bond_feats[e_idx, 2] = 1.0
            elif bt == Chem.BondType.AROMATIC:
                bond_feats[e_idx, 3] = 1.0

            # Feature 4: conjugation_flag
            bond_feats[e_idx, 4] = 1.0 if bond.GetIsConjugated() else 0.0

            # Feature 5: in_ring_flag
            bond_feats[e_idx, 5] = 1.0 if bond.IsInRing() else 0.0

            # Feature 6: rotatable_flag
            # RDKit's definition: single bond between non-ring heavy atoms
            # that is not in a ring, for which the smaller ring containing the bond is >6 atoms
            # But we need a simpler initial check
            a1 = mol.GetAtomWithIdx(u)
            a2 = mol.GetAtomWithIdx(v)
            # Simplified rotatable: single bond, not in ring, not terminal
            is_rotatable = (
                bt == Chem.BondType.SINGLE and
                not bond.IsInRing() and
                a1.GetDegree() > 1 and a2.GetDegree() > 1
            )
            bond_feats[e_idx, 6] = 1.0 if is_rotatable else 0.0

            # Feature 7: bond_stereo (simplified)
            stereo = bond.GetStereo()
            bond_feats[e_idx, 7] = 1.0 if stereo != Chem.BondStereo.STEREONONE else 0.0

            # Feature 8: bond_order_float (normalized to [0,1])
            order = bond.GetBondTypeAsDouble()
            bond_feats[e_idx, 8] = min(order / 3.0, 1.0)

            # Feature 9: is_backbone_bond
            is_bb = (u in backbone_set and v in backbone_set)
            bond_feats[e_idx, 9] = 1.0 if is_bb else 0.0

            # Feature 10: avg_endpoint_electronegativity (normalized)
            avg_en = (atom_en.get(u, 2.5) + atom_en.get(v, 2.5)) / 2.0
            bond_feats[e_idx, 10] = (avg_en - 1.5) / 3.0  # rough normalization

            # Feature 11: bond_distance_along_backbone (normalized [0,1])
            if is_bb and backbone_len > 1:
                pos = backbone_bond_pos.get(rdkit_bond_idx, -1)
                if pos >= 0:
                    bond_feats[e_idx, 11] = pos / max(backbone_len - 1, 1)
                else:
                    bond_feats[e_idx, 11] = -1.0
            else:
                bond_feats[e_idx, 11] = -1.0

            # Feature 12: monomer_index (0/0.5/1 for backbone, -1 for non-backbone)
            if is_bb:
                mi = (monomer_idx.get(u, -1) + monomer_idx.get(v, -1)) / 2.0
                bond_feats[e_idx, 12] = mi
            else:
                bond_feats[e_idx, 12] = -1.0

            # Feature 13: bond_length_estimate (normalized)
            bl = bond_length_estimate.get(bt, 1.5)
            bond_feats[e_idx, 13] = (bl - 1.0) / 1.0  # rough normalization

        else:
            # Fallback: infer basic features from atom types if bond not in RDKit mol
            # This only happens for PyG-generated self-loops or weird edges
            bond_feats[e_idx, 0] = 1.0  # default: single
            bond_feats[e_idx, 9] = 0.0  # not backbone
            bond_feats[e_idx, 11] = -1.0
            bond_feats[e_idx, 12] = -1.0
            avg_en = (atom_en.get(u, 2.5) + atom_en.get(v, 2.5)) / 2.0
            bond_feats[e_idx, 10] = (avg_en - 1.5) / 3.0
            bond_feats[e_idx, 13] = (1.54 - 1.0) / 1.0

    return torch.tensor(bond_feats, dtype=torch.float32), edge_index.t()


def construct_bond_graph(data, bond_features, bond_to_atom):
    """
    Build bond graph (line graph) from atom graph and bond features.

    In the bond graph:
    - Each bond (edge in atom graph) becomes a node
    - Two bond-nodes are adjacent if their corresponding bonds share a common atom
    - Bond node features are the provided bond_features

    Args:
        data: PyG Data object with atom graph (edge_index)
        bond_features: [num_bonds, 14] computed bond features
        bond_to_atom: [num_bonds, 2] mapping bond_idx -> (atom_u, atom_v)

    Returns:
        bond_data: PyG Data object with:
            x: [num_bonds, 14] bond node features
            edge_index: [2, num_bond_edges] bond graph adjacency
            num_nodes: num_bonds
            bond_to_atom: [num_bonds, 2] for later readout
    """
    edge_index = data.edge_index  # [2, E]
    num_bonds = edge_index.shape[1]

    # Build adjacency: bond i and bond j share an edge if they share a common atom
    # Efficient approach: for each atom, collect all incident bond indices
    num_atoms = data.num_nodes
    atom_to_bonds = [[] for _ in range(num_atoms)]
    for b in range(num_bonds):
        u, v = int(edge_index[0, b]), int(edge_index[1, b])
        atom_to_bonds[u].append(b)
        atom_to_bonds[v].append(b)

    bond_edges = []
    for a in range(num_atoms):
        incident = atom_to_bonds[a]
        for i_idx in range(len(incident)):
            for j_idx in range(i_idx + 1, len(incident)):
                bi, bj = incident[i_idx], incident[j_idx]
                bond_edges.append([bi, bj])
                bond_edges.append([bj, bi])  # undirected

    from torch_geometric.data import Data

    if len(bond_edges) > 0:
        bond_edge_index = torch.tensor(bond_edges, dtype=torch.long).t().contiguous()
    else:
        bond_edge_index = torch.zeros((2, 0), dtype=torch.long)

    bond_data = Data(
        x=bond_features,
        edge_index=bond_edge_index,
        num_nodes=num_bonds,
    )
    bond_data.bond_to_atom = bond_to_atom

    return bond_data
