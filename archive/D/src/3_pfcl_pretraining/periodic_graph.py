"""
Periodic Graph Construction Module
Builds PyTorch Geometric graphs from p-SMILES with periodicity edges
"""

import torch
from torch_geometric.data import Data
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Optional, Tuple
import os


class PeriodicGraphBuilder:
    """Build periodic graphs from polymer SMILES"""

    def __init__(
        self,
        node_feat_dim: int = 30,
        edge_feat_dim: int = 11,
        max_atoms: int = 200
    ):
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.max_atoms = max_atoms

    def get_atom_features(self, atom: Chem.Atom) -> np.ndarray:
        """Compute atom features (30-dim)"""
        features = []

        # Atom type (one-hot for common elements) - 10
        elements = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Other']
        atom_symbol = atom.GetSymbol()
        for elem in elements:
            if atom_symbol == elem:
                features.append(1.0)
            elif elem == 'Other' and atom_symbol not in elements[:-1]:
                features.append(1.0)
            else:
                features.append(0.0)

        # Degree - one-hot (0-6) - 7
        degree = atom.GetDegree()
        for i in range(7):
            features.append(1.0 if degree == i else 0.0)

        # Total number of Hs - one-hot (0-4) - 5
        total_h = atom.GetTotalNumHs()
        for i in range(5):
            features.append(1.0 if total_h == i else 0.0)

        # Implicit valence - one-hot (0-6) - 7
        implicit_valence = atom.GetImplicitValence()
        for i in range(7):
            features.append(1.0 if implicit_valence == i else 0.0)

        # Aromaticity - 1
        features.append(1.0 if atom.GetIsAromatic() else 0.0)

        # Hybridization - one-hot (SP, SP2, SP3, other) - 4
        hybridization = atom.GetHybridization()
        sp = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
              Chem.HybridizationType.SP3, Chem.HybridizationType.OTHER]
        for h in sp:
            features.append(1.0 if hybridization == h else 0.0)

        # Partial charge - 1
        try:
            features.append(atom.GetProp('_GasteigerCharge') or 0.0)
        except:
            features.append(0.0)

        # Formal charge - 1
        features.append(float(atom.GetFormalCharge()))

        # Is in ring - 1
        features.append(1.0 if atom.IsInRing() else 0.0)

        return np.array(features, dtype=np.float32)

    def get_bond_features(self, bond: Chem.Bond) -> np.ndarray:
        """Compute bond features (11-dim)"""
        features = []

        # Bond type - one-hot (single, double, triple, aromatic) - 4
        bond_type = bond.GetBondType()
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                      Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        for bt in bond_types:
            features.append(1.0 if bond_type == bt else 0.0)

        # Conjugation - 1
        features.append(1.0 if bond.GetIsConjugated() else 0.0)

        # In ring - 1
        features.append(1.0 if bond.IsInRing() else 0.0)

        # Stereo - one-hot (none, Z, E, any) - 4
        stereo = bond.GetStereo()
        stereos = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOZ,
                   Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOANY]
        for s in stereos:
            features.append(1.0 if stereo == s else 0.0)

        return np.array(features, dtype=np.float32)

    def find_terminal_stars(self, mol: Chem.Mol) -> List[int]:
        """Find atom indices connected to * (dummy atoms)"""
        terminal_atoms = []

        for atom in mol.GetAtoms():
            # Check if atom is connected to a dummy atom (atomic_num=0)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 0:  # Dummy atom
                    terminal_atoms.append(atom.GetIdx())
                    break

        return terminal_atoms

    def build_graph(
        self,
        smiles: str,
        add_periodic_edge: bool = True
    ) -> Optional[Data]:
        """Build a periodic graph from p-SMILES"""
        try:
            # Parse molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Add hydrogens
            mol = Chem.AddHs(mol)

            num_atoms = mol.GetNumAtoms()
            if num_atoms > self.max_atoms or num_atoms == 0:
                return None

            # Compute atom features
            node_features = []
            for atom in mol.GetAtoms():
                atom_feats = self.get_atom_features(atom)
                node_features.append(atom_feats)

            x = torch.tensor(np.array(node_features), dtype=torch.float32)

            # Build edge index and edge features
            edge_indices = []
            edge_features = []

            # Regular bonds
            for bond in mol.GetBonds():
                start_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()

                # Add both directions
                edge_indices.extend([[start_idx, end_idx], [end_idx, start_idx]])
                bond_feats = self.get_bond_features(bond)
                edge_features.extend([bond_feats, bond_feats])

            # Add periodicity edge between terminal atoms
            if add_periodic_edge:
                terminal_atoms = self.find_terminal_stars(mol)

                if len(terminal_atoms) >= 2:
                    # Connect first two terminal atoms
                    src, dst = terminal_atoms[0], terminal_atoms[1]

                    # Mark as periodic edge with special features
                    periodic_edge_feat = np.zeros(self.edge_feat_dim, dtype=np.float32)
                    periodic_edge_feat[0] = 1.0  # Special marker for periodic edge

                    edge_indices.extend([[src, dst], [dst, src]])
                    edge_features.extend([periodic_edge_feat, periodic_edge_feat])

            if len(edge_indices) == 0:
                return None

            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float32)

            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)

        except Exception as e:
            return None

    def build_graph_batch(
        self,
        smiles_list: List[str],
        add_periodic_edge: bool = True
    ) -> List[Data]:
        """Build graphs for a batch of SMILES"""
        graphs = []

        for smiles in smiles_list:
            graph = self.build_graph(smiles, add_periodic_edge)
            if graph is not None:
                graphs.append(graph)

        return graphs


class PeriodicAugmentor:
    """Generate augmented views for contrastive learning"""

    def __init__(self, graph_builder: PeriodicGraphBuilder):
        self.graph_builder = graph_builder

    def shift_boundary(self, smiles: str, shift: int = 1) -> Optional[str]:
        """Shift periodic boundary by rotating the repeating unit"""
        try:
            # Find * positions
            if smiles.count('*') < 2:
                return None

            # Extract content between first * and last *
            start = smiles.find('*') + 1
            end = smiles.rfind('*')
            content = smiles[start:end]

            # Remove trailing * if present
            if content.endswith('*'):
                content = content[:-1]

            # Rotate the content
            if shift > 0:
                shifted = content[shift:] + content[:shift]
            else:
                shifted = content[shift:] + content[:shift]

            new_smiles = '*' + shifted + '*'

            # Validate
            mol = Chem.MolFromSmiles(new_smiles)
            if mol is None:
                return None

            return new_smiles

        except Exception as e:
            return None

    def extend_unit(self, smiles: str, repeat: int = 2) -> Optional[str]:
        """Extend the repeating unit"""
        try:
            if smiles.count('*') < 2:
                return None

            # Extract content between * markers
            start = smiles.find('*') + 1
            end = smiles.rfind('*')
            content = smiles[start:end]

            # Remove trailing * if present
            if content.endswith('*'):
                content = content[:-1]

            # Repeat the unit
            extended = content * repeat

            new_smiles = '*' + extended + '*'

            # Validate
            mol = Chem.MolFromSmiles(new_smiles)
            if mol is None:
                return None

            return new_smiles

        except Exception as e:
            return None

    def augment_smiles(
        self,
        smiles: str,
        strategy: str = 'shift',
        **kwargs
    ) -> Optional[str]:
        """Apply augmentation strategy"""
        if strategy == 'shift':
            shift = kwargs.get('shift', 1)
            return self.shift_boundary(smiles, shift)
        elif strategy == 'extend':
            repeat = kwargs.get('repeat', 2)
            return self.extend_unit(smiles, repeat)
        else:
            return None

    def create_positive_pair(
        self,
        smiles: str,
        strategy: str = 'shift'
    ) -> Tuple[Optional[Data], Optional[Data]]:
        """Create a positive pair for contrastive learning"""
        # Original graph
        graph1 = self.graph_builder.build_graph(smiles, add_periodic_edge=True)

        # Augmented graph
        aug_smiles = None
        if strategy == 'shift':
            # Random shift 1-2
            import random
            shift = random.choice([1, 2])
            aug_smiles = self.shift_boundary(smiles, shift)
        elif strategy == 'extend':
            aug_smiles = self.extend_unit(smiles, repeat=2)

        if aug_smiles is None:
            return graph1, None

        graph2 = self.graph_builder.build_graph(aug_smiles, add_periodic_edge=True)

        return graph1, graph2


def main():
    """Test periodic graph construction"""
    builder = PeriodicGraphBuilder()
    augmentor = PeriodicAugmentor(builder)

    # Test polymers
    test_smiles = [
        "*CCC*",
        "*CC(*)C",
        "*C(=O)CNC(*)C(=O)OCCCCCNC"
    ]

    print("Testing periodic graph construction...")
    for smiles in test_smiles:
        print(f"\nSMILES: {smiles}")

        graph = builder.build_graph(smiles)
        if graph is not None:
            print(f"  Graph: {graph.num_nodes} nodes, {graph.num_edges/2:.0f} edges")

        # Test augmentation
        for strategy in ['shift', 'extend']:
            g1, g2 = augmentor.create_positive_pair(smiles, strategy=strategy)
            if g2 is not None:
                print(f"  {strategy} augmentation: {g2.num_nodes} nodes, {g2.num_edges/2:.0f} edges")
            else:
                print(f"  {strategy} augmentation: Failed")


if __name__ == "__main__":
    main()
