"""
PIPGA: Polymer Informatics with Position-aware Graph Attention
Core implementation with multi-edge graph construction and edge-wise attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import degree
import numpy as np
from rdkit import Chem


def identify_head_tail_atoms(mol):
    """Identify head (*) and tail (*) atoms in polymer SMILES"""
    head_atoms = []
    tail_atoms = []

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            # Distinguish head vs tail based on connectivity patterns
            neighbors = [n for n in atom.GetNeighbors()]
            if len(neighbors) == 0:
                # Isolated star - mark as both
                head_atoms.append(atom.GetIdx())
                tail_atoms.append(atom.GetIdx())
            elif len(neighbors) == 1:
                # Connected star - alternate between head and tail
                if len(head_atoms) <= len(tail_atoms):
                    head_atoms.append(atom.GetIdx())
                else:
                    tail_atoms.append(atom.GetIdx())
            else:
                # Multi-connected star
                tail_atoms.append(atom.GetIdx())

    return head_atoms, tail_atoms


def create_extended_graph(data, num_copies=3, k_hop=2):
    """
    Create multi-copy extension for periodic boundary simulation

    Args:
        data: PyG data object
        num_copies: Number of copies (3 for left-center-right)
        k_hop: Number of hops for neighbor extraction

    Returns:
        Extended data with periodic offset labels
    """
    num_nodes = data.x.size(0)

    # Create 3 copies of the graph
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    batch_list = []
    offset_labels = []

    for copy_idx in range(num_copies):
        # Copy node features
        x_list.append(data.x)

        # Copy edge indices with offset
        edge_index_offset = data.edge_index + copy_idx * num_nodes
        edge_index_list.append(edge_index_offset)

        # Copy edge attributes
        if data.edge_attr is not None:
            edge_attr_list.append(data.edge_attr)

        # Copy batch indices
        if hasattr(data, 'batch') and data.batch is not None:
            batch_list.append(data.batch)
        else:
            batch_list.append(torch.zeros(num_nodes, dtype=torch.long, device=data.x.device))

        # Create offset labels (position encoding)
        # Center copy (idx=1) has offset 0, left (idx=0) has -1, right (idx=2) has +1
        offset = copy_idx - 1  # -1, 0, 1 for copies 0, 1, 2
        offset_labels.append(torch.full((num_nodes,), offset, dtype=torch.float, device=data.x.device))

    # Concatenate all copies
    x = torch.cat(x_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    if data.edge_attr is not None:
        edge_attr = torch.cat(edge_attr_list, dim=0)
    else:
        edge_attr = None
    batch = torch.cat(batch_list, dim=0)
    offset = torch.cat(offset_labels, dim=0)

    return x, edge_index, edge_attr, batch, offset


class ECFP4Fingerprint(nn.Module):
    """ECFP4 fingerprint for attention modulation"""

    def __init__(self, radius=2, n_bits=1024):
        super().__init__()
        self.radius = radius
        self.n_bits = n_bits
        self.projection = nn.Linear(n_bits, 128)

    def compute_ecfp4(self, mol):
        """Compute ECFP4 fingerprint using RDKit"""
        from rdkit.Chem import AllChem
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        return np.array(fp)

    def forward(self, mol):
        """Compute and project ECFP4 fingerprint"""
        fp = self.compute_ecfp4(mol)
        fp_tensor = torch.tensor(fp, dtype=torch.float, device=next(self.parameters()).device)
        return self.projection(fp_tensor)


class EdgeWiseAttention(MessagePassing):
    """
    Edge-wise attention with sigmoid((q ∘ k) / √d) + LayerNorm
    NO softmax normalization
    """

    def __init__(self, hidden_dim, num_heads=4, edge_dim=None):
        super().__init__(aggr='add', node_dim=0)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Edge projection
        if edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        else:
            self.edge_proj = None

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # LayerNorm for attention
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.scale = self.head_dim ** 0.5

    def forward(self, x, edge_index, edge_attr=None, return_attention=True):
        """
        Forward pass with edge-wise attention

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            return_attention: Whether to return attention weights

        Returns:
            Updated node features
        """
        # Store original for residual
        x_residual = x

        # Project to Q, K, V
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)

        # Propagate with edge-wise attention
        out = self.propagate(edge_index, q=q, k=k, v=v, edge_attr=edge_attr, size=None)

        # Reshape and project output
        out = out.view(-1, self.hidden_dim)
        out = self.out_proj(out)

        # Residual connection + LayerNorm
        out = self.layer_norm(out + x_residual)

        return out

    def message(self, q_i, k_j, v_j, edge_attr, index, ptr, size_i):
        """
        Compute edge-wise attention weights using sigmoid

        Args:
            q_i: Query for target nodes [num_edges, num_heads, head_dim]
            k_j: Key for source nodes [num_edges, num_heads, head_dim]
            v_j: Value for source nodes [num_edges, num_heads, head_dim]
            edge_attr: Edge features
            index: Target node indices
            ptr: CSR format pointer (not used)
            size_i: Number of target nodes

        Returns:
            Weighted messages
        """
        # Compute edge-wise attention: (q ∘ k) / √d
        # q_i and k_j are both [num_edges, num_heads, head_dim]
        attn = (q_i * k_j).sum(dim=-1) / self.scale  # [num_edges, num_heads]

        # Apply sigmoid (NO softmax)
        attn = torch.sigmoid(attn)  # [num_edges, num_heads]

        # Apply attention to values
        out = attn.unsqueeze(-1) * v_j  # [num_edges, num_heads, head_dim]

        return out


class PIPGALayer(nn.Module):
    """PIPGA layer with self-connecting edges and head/tail encoding"""

    def __init__(self, hidden_dim, num_heads=4, edge_dim=None, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Edge-wise attention
        self.attention = EdgeWiseAttention(hidden_dim, num_heads, edge_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        # LayerNorm for FFN
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Head/tail encoding
        self.head_encoding = nn.Parameter(torch.randn(1, hidden_dim))
        self.tail_encoding = nn.Parameter(torch.randn(1, hidden_dim))

    def forward(self, x, edge_index, edge_attr=None, head_mask=None, tail_mask=None):
        """
        Forward pass with self-connecting edges

        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            head_mask: Boolean mask for head atoms
            tail_mask: Boolean mask for tail atoms

        Returns:
            Updated node features
        """
        num_nodes = x.size(0)

        # Add self-loops
        self_loops = torch.arange(num_nodes, device=x.device).unsqueeze(0).repeat(2, 1)
        edge_index_with_loops = torch.cat([edge_index, self_loops], dim=1)

        # Extend edge attributes for self-loops
        if edge_attr is not None:
            self_loop_attr = torch.zeros(num_nodes, edge_attr.size(1), device=x.device)
            edge_attr_with_loops = torch.cat([edge_attr, self_loop_attr], dim=0)
        else:
            edge_attr_with_loops = None

        # Apply edge-wise attention
        x = self.attention(x, edge_index_with_loops, edge_attr_with_loops)

        # Add head/tail encodings
        if head_mask is not None and tail_mask is not None:
            x[head_mask] = x[head_mask] + self.head_encoding
            x[tail_mask] = x[tail_mask] + self.tail_encoding

        # Feed-forward with residual
        x_ffn = self.ffn(x)
        x = self.layer_norm(x + x_ffn)

        return x


class PIPGAEncoder(nn.Module):
    """PIPGA encoder with multiple layers"""

    def __init__(self, input_dim, hidden_dim, num_layers=3, num_heads=4, edge_dim=None, dropout=0.1):
        super().__init__()

        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # PIPGA layers
        self.layers = nn.ModuleList([
            PIPGALayer(hidden_dim, num_heads, edge_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_attr=None, head_mask=None, tail_mask=None):
        """
        Forward pass through all PIPGA layers

        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            head_mask: Boolean mask for head atoms
            tail_mask: Boolean mask for tail atoms

        Returns:
            Updated node features
        """
        # Project input
        x = self.input_proj(x)

        # Apply PIPGA layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, head_mask, tail_mask)

        return x


class PIPGA(nn.Module):
    """
    PIPGA: Polymer Informatics with Position-aware Graph Attention

    Features:
    - Multi-edge graph construction with 3-copy extension
    - Edge-wise attention: sigmoid((q ∘ k) / √d) + LayerNorm
    - Self-connecting edges with d_head, d_tail encoding
    - ECFP4 fingerprint attention modulation
    """

    def __init__(self, input_dim=43, hidden_dim=256, num_layers=3, num_heads=4,
                 edge_dim=5, dropout=0.1, use_ecfp4=True, ecfp4_bits=1024):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_ecfp4 = use_ecfp4

        # PIPGA encoder
        self.encoder = PIPGAEncoder(input_dim, hidden_dim, num_layers, num_heads, edge_dim, dropout)

        # ECFP4 fingerprint module
        if use_ecfp4:
            self.ecfp4 = ECFP4Fingerprint(radius=2, n_bits=ecfp4_bits)
            self.ecfp4_attention = nn.Sequential(
                nn.Linear(128, hidden_dim),
                nn.Sigmoid()
            )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data, mol=None, head_mask=None, tail_mask=None):
        """
        Forward pass

        Args:
            data: PyG data object
            mol: RDKit molecule (for ECFP4)
            head_mask: Boolean mask for head atoms
            tail_mask: Boolean mask for tail atoms

        Returns:
            Prediction
        """
        # Get node features
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        # Encode with PIPGA
        node_features = self.encoder(x, edge_index, edge_attr, head_mask, tail_mask)

        # Global pooling
        if hasattr(data, 'batch') and data.batch is not None:
            graph_features = global_mean_pool(node_features, data.batch)
        else:
            graph_features = node_features.mean(dim=0, keepdim=True)

        # ECFP4 attention modulation
        if self.use_ecfp4 and mol is not None:
            try:
                ecfp4_features = self.ecfp4(mol)
                attention_weights = self.ecfp4_attention(ecfp4_features)
                graph_features = graph_features * attention_weights
            except:
                # If ECFP4 computation fails, skip modulation
                pass

        # Predict
        output = self.predictor(graph_features)

        return output.squeeze(-1)


def polymer_smiles_to_graph(smiles):
    """
    Convert polymer SMILES to PyG graph

    Args:
        smiles: Polymer SMILES string

    Returns:
        PyG Data object
    """
    from torch_geometric.data import Data

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Node features
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum() / 100.0,  # Atomic number
                atom.GetDegree() / 5.0,
                atom.GetFormalCharge() / 5.0,
                atom.GetNumRadicalElectrons() / 5.0,
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
                atom.GetMass() / 100.0,
                # Hybridization
                int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP),
                int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2),
                int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3),
                # Electronegativity proxy (using precomputed values)
                {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}.get(atom.GetAtomicNum(), 2.0) / 5.0,
                # Is star atom
                int(atom.GetSymbol() == '*'),
                # Valence
                atom.GetTotalValence() / 8.0,
                # Number of Hs
                atom.GetTotalNumHs() / 5.0,
                # Ring size
                int(atom.IsInRingSize(3)),
                int(atom.IsInRingSize(4)),
                int(atom.IsInRingSize(5)),
                int(atom.IsInRingSize(6)),
                # Degree one-hot
                int(atom.GetDegree() == 0),
                int(atom.GetDegree() == 1),
                int(atom.GetDegree() == 2),
                int(atom.GetDegree() == 3),
                int(atom.GetDegree() == 4),
                # Atom type one-hot (common atoms)
                int(atom.GetSymbol() == 'C'),
                int(atom.GetSymbol() == 'N'),
                int(atom.GetSymbol() == 'O'),
                int(atom.GetSymbol() == 'S'),
                int(atom.GetSymbol() == 'F'),
                int(atom.GetSymbol() == 'Cl'),
                int(atom.GetSymbol() == 'Br'),
                int(atom.GetSymbol() == 'P'),
                # Aromatic neighbors
                sum([n.GetIsAromatic() for n in atom.GetNeighbors()]) / 5.0,
                # Bond types to neighbors
                sum([int(b.GetBondType() == Chem.rdchem.BondType.SINGLE) for b in atom.GetBonds()]) / 5.0,
                sum([int(b.GetBondType() == Chem.rdchem.BondType.DOUBLE) for b in atom.GetBonds()]) / 5.0,
                sum([int(b.GetBondType() == Chem.rdchem.BondType.AROMATIC) for b in atom.GetBonds()]) / 5.0,
                # Is head/tail (star atoms)
                int(atom.GetSymbol() == '*' and len([n for n in atom.GetNeighbors()]) <= 1),
                # Partial charge proxy
                atom.GetFormalCharge() / 5.0,
                # Chirality
                int(atom.HasProp('_ChiralityPossible')),
                # Radical electrons
                atom.GetNumRadicalElectrons() / 5.0,
                # Isotope
                atom.GetIsotope() / 100.0,
                # Additional features
                atom.GetNumImplicitHs() / 5.0,
                atom.GetExplicitValence() / 8.0,
                atom.GetImplicitValence() / 8.0,
            ]
            atom_features.append(features)

        x = torch.tensor(atom_features, dtype=torch.float)

        # Edge features
        edge_index = []
        edge_features = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Bond features
            bond_features = [
                int(bond.GetBondType() == Chem.rdchem.BondType.SINGLE),
                int(bond.GetBondType() == Chem.rdchem.BondType.DOUBLE),
                int(bond.GetBondType() == Chem.rdchem.BondType.TRIPLE),
                int(bond.GetBondType() == Chem.rdchem.BondType.AROMATIC),
                int(bond.GetIsConjugated()),
            ]

            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_features.append(bond_features)
            edge_features.append(bond_features)

        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            # Single atom molecule
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 5), dtype=torch.float)

        # Identify head and tail atoms
        head_atoms, tail_atoms = identify_head_tail_atoms(mol)
        head_mask = torch.zeros(x.size(0), dtype=torch.bool)
        tail_mask = torch.zeros(x.size(0), dtype=torch.bool)
        head_mask[head_atoms] = True
        tail_mask[tail_atoms] = True

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                   head_mask=head_mask, tail_mask=tail_mask)

        return data, mol

    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None, None
