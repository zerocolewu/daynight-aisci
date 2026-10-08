"""
Full PolyBG-CPA models and all experiment variants.

Experiment 1 (BASELINE): CPA-GAT Atom-Only (CPAGATModel)
Experiment 2 (PRIMARY): PolyBG-CPA Parallel (PolyBGCPA)
Experiment 3 (ABLATION): PolyBG-noCPA (PolyBGnoCPA)
Experiment 4 (ABLATION): Bond-CPA-GAT Only (BondCPAModel)
Experiment 5 (ABLATION): Enriched-Edge CPA-GAT (EnrichedEdgeCPAGAT)

Iteration 2 - Hierarchical Bond->Atom Encoding:
Experiment 6: HierarchicalPolyBGCPA_Pre (Bond->Atom input augmentation)
Experiment 7: HierarchicalPolyBGCPA_Mid (Bond->Atom after atom layer 1)
Experiment 8: HierarchicalPolyBGCPA_PreGated (Pre with learned gate fusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cpa_gat import CPAGATv2Conv


# ==============================================================================
# Experiment 1: CPA-GAT Atom-Only (baseline, reproducing mem_146)
# ==============================================================================

class CPAGATBaseline(nn.Module):
    """
    CPA-GAT with 4 CPA layers, 256 hidden dim, 4 heads.
    Input: 42-dim (8 rich + 2 PSPE + 32 RWPE).
    This reproduces the CPA-GAT from 7_day7_cpa_pspe with PSPE included.
    """

    def __init__(self, in_dim=42, hidden_dim=256, num_layers=4, heads=4):
        super().__init__()
        self.head_dim = hidden_dim // heads

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                CPAGATv2Conv(hidden_dim, self.head_dim, heads=heads)
            )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, data, bond_data=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_proj(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)

        x = global_mean_pool(x, batch)
        x = self.predictor(x)
        return x.squeeze(-1)


# ==============================================================================
# Experiment 2: PolyBG-CPA Parallel
# ==============================================================================

class PolyBGCPA(nn.Module):
    """
    Parallel atom CPA-GAT (4-layer, 256-dim) + bond CPA-GAT (3-layer, 128-dim).
    Concatenated readout, no cross-attention.
    """

    def __init__(self, atom_in_dim=42, atom_hidden=256, atom_layers=4, atom_heads=4,
                 bond_in_dim=14, bond_hidden=128, bond_layers=3, bond_heads=4,
                 predictor_hidden=192, dropout=0.1):
        super().__init__()
        self.atom_hidden = atom_hidden
        self.bond_hidden = bond_hidden

        # Atom encoder
        self.atom_head_dim = atom_hidden // atom_heads
        self.atom_input_proj = nn.Linear(atom_in_dim, atom_hidden)
        self.atom_convs = nn.ModuleList([
            CPAGATv2Conv(atom_hidden, self.atom_head_dim, heads=atom_heads)
            for _ in range(atom_layers)
        ])

        # Bond encoder
        self.bond_head_dim = bond_hidden // bond_heads
        self.bond_input_proj = nn.Linear(bond_in_dim, bond_hidden)
        self.bond_convs = nn.ModuleList([
            CPAGATv2Conv(bond_hidden, self.bond_head_dim, heads=bond_heads)
            for _ in range(bond_layers)
        ])

        # Predictor
        joint_dim = atom_hidden + bond_hidden  # 256 + 128 = 384
        self.predictor = nn.Sequential(
            nn.Linear(joint_dim, predictor_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden, 1),
        )

    def forward(self, data, bond_data=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Atom encoder
        x = self.atom_input_proj(x)
        for conv in self.atom_convs:
            x = conv(x, edge_index)
            x = F.elu(x)
        atom_emb = global_mean_pool(x, batch)

        # Bond encoder (batched for efficiency)
        if bond_data is not None:
            bond_embs = self._forward_bond_encoder_batched(bond_data)
            joint_emb = torch.cat([atom_emb, bond_embs], dim=-1)
        else:
            bond_embs = torch.zeros(atom_emb.size(0), self.bond_hidden,
                                    device=atom_emb.device)
            joint_emb = torch.cat([atom_emb, bond_embs], dim=-1)

        out = self.predictor(joint_emb)
        return out.squeeze(-1)

    def _forward_bond_encoder_batched(self, bond_data):
        """Efficient bond encoder: batch all bond graphs into one PyG batch."""
        from torch_geometric.data import Data, Batch
        device = next(self.parameters()).device

        # Collect all bond nodes and edges with batch indices
        bond_graphs = []
        for i, bd in enumerate(bond_data):
            bg = Data(
                x=bd['bond_x'],
                edge_index=bd['bond_edge_index'],
                num_nodes=bd['bond_num_nodes'],
            )
            bond_graphs.append(bg)

        # Batch all bond graphs
        bond_batch = Batch.from_data_list(bond_graphs).to(device)

        bx = self.bond_input_proj(bond_batch.x)
        for conv in self.bond_convs:
            bx = conv(bx, bond_batch.edge_index)
            bx = F.elu(bx)

        # Global mean pool per molecule
        bond_embs = global_mean_pool(bx, bond_batch.batch)
        return bond_embs


# ==============================================================================
# Experiment 3: PolyBG-noCPA (ablation, no CPA in either encoder)
# ==============================================================================

class PolyBGnoCPA(nn.Module):
    """
    Same parallel architecture but NO CPA in either encoder.
    Plain GAT in both atom and bond encoders.
    """

    def __init__(self, atom_in_dim=42, atom_hidden=256, atom_layers=4, atom_heads=4,
                 bond_in_dim=14, bond_hidden=128, bond_layers=3, bond_heads=4,
                 predictor_hidden=192, dropout=0.1):
        super().__init__()
        self.atom_hidden = atom_hidden
        self.bond_hidden = bond_hidden

        # Atom encoder (plain GAT)
        self.atom_input_proj = nn.Linear(atom_in_dim, atom_hidden)
        self.atom_convs = nn.ModuleList([
            GATv2Conv(atom_hidden, atom_hidden // atom_heads, heads=atom_heads,
                      concat=True, add_self_loops=True, dropout=0.0)
            for _ in range(atom_layers)
        ])

        # Bond encoder (plain GAT)
        self.bond_input_proj = nn.Linear(bond_in_dim, bond_hidden)
        self.bond_convs = nn.ModuleList([
            GATv2Conv(bond_hidden, bond_hidden // bond_heads, heads=bond_heads,
                      concat=True, add_self_loops=True, dropout=0.0)
            for _ in range(bond_layers)
        ])

        # Predictor
        joint_dim = atom_hidden + bond_hidden
        self.predictor = nn.Sequential(
            nn.Linear(joint_dim, predictor_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden, 1),
        )

    def forward(self, data, bond_data=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Atom encoder
        x = self.atom_input_proj(x)
        for conv in self.atom_convs:
            x = conv(x, edge_index)
            x = F.elu(x)
        atom_emb = global_mean_pool(x, batch)

        # Bond encoder
        if bond_data is not None:
            bond_embs = self._forward_bond_encoder_batched(bond_data)
        else:
            bond_embs = torch.zeros(atom_emb.size(0), self.bond_hidden,
                                    device=atom_emb.device)

        joint_emb = torch.cat([atom_emb, bond_embs], dim=-1)
        out = self.predictor(joint_emb)
        return out.squeeze(-1)

    def _forward_bond_encoder_batched(self, bond_data):
        from torch_geometric.data import Data, Batch
        device = next(self.parameters()).device
        bond_graphs = [Data(x=bd['bond_x'], edge_index=bd['bond_edge_index'],
                           num_nodes=bd['bond_num_nodes']) for bd in bond_data]
        bond_batch = Batch.from_data_list(bond_graphs).to(device)
        bx = self.bond_input_proj(bond_batch.x)
        for conv in self.bond_convs:
            bx = conv(bx, bond_batch.edge_index)
            bx = F.elu(bx)
        return global_mean_pool(bx, bond_batch.batch)


# ==============================================================================
# Experiment 4: Bond-CPA-GAT Only (no atom encoder)
# ==============================================================================

class BondOnlyCPA(nn.Module):
    """
    Bond graph only with CPA (3-layer, 128-dim). No atom encoder.
    Tests standalone value of bond-level CPA.
    """

    def __init__(self, bond_in_dim=14, bond_hidden=128, bond_layers=3, bond_heads=4,
                 predictor_hidden=192, dropout=0.1):
        super().__init__()
        self.bond_hidden = bond_hidden

        # Bond encoder
        self.bond_head_dim = bond_hidden // bond_heads
        self.bond_input_proj = nn.Linear(bond_in_dim, bond_hidden)
        self.bond_convs = nn.ModuleList([
            CPAGATv2Conv(bond_hidden, self.bond_head_dim, heads=bond_heads)
            for _ in range(bond_layers)
        ])

        # Predictor (bond_hidden -> 1)
        self.predictor = nn.Sequential(
            nn.Linear(bond_hidden, predictor_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden, 1),
        )

    def forward(self, data, bond_data=None):
        if bond_data is None:
            device = next(self.parameters()).device
            return torch.zeros(data.num_graphs, device=device)

        from torch_geometric.data import Data, Batch
        device = next(self.parameters()).device
        bond_graphs = [Data(x=bd['bond_x'], edge_index=bd['bond_edge_index'],
                           num_nodes=bd['bond_num_nodes']) for bd in bond_data]
        bond_batch = Batch.from_data_list(bond_graphs).to(device)
        bx = self.bond_input_proj(bond_batch.x)
        for conv in self.bond_convs:
            bx = conv(bx, bond_batch.edge_index)
            bx = F.elu(bx)
        emb = global_mean_pool(bx, bond_batch.batch)
        out = self.predictor(emb)
        return out.squeeze(-1)


# ==============================================================================
# Experiment 5: Enriched-Edge CPA-GAT (atom-only with bond edge features)
# ==============================================================================

class EnrichedEdgeCPAGAT(nn.Module):
    """
    Atom-only CPA-GAT but with enriched edge features.
    Edge features: bond_type (4-dim one-hot) + conjugation + in_ring + rotatable + bond_order = 8-dim.
    Edge features are concatenated to attention computation.
    """

    def __init__(self, in_dim=42, hidden_dim=256, num_layers=4, heads=4, edge_dim=8):
        super().__init__()
        self.head_dim = hidden_dim // heads
        self.edge_dim = edge_dim

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Edge feature projection
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                EnrichedEdgeCPAGATv2Conv(hidden_dim, self.head_dim, heads=heads,
                                         edge_dim=edge_dim)
            )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, data, bond_data=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)

        x = self.input_proj(x)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.elu(x)

        x = global_mean_pool(x, batch)
        x = self.predictor(x)
        return x.squeeze(-1)


class EnrichedEdgeCPAGATv2Conv(nn.Module):
    """
    CPA-GATv2 layer with edge features aggregated to nodes via scatter mean.
    Each node's features are augmented with the mean of incident edge features.
    """
    def __init__(self, in_dim, out_dim, heads=4, edge_dim=8, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.head_dim = out_dim
        self.heads = heads
        self.edge_dim = edge_dim

        self.edge_proj = nn.Linear(edge_dim, in_dim)

        self.gat_heads = nn.ModuleList([
            GATv2Conv(in_dim * 2, out_dim, heads=1, concat=True,
                      add_self_loops=True, dropout=dropout)
            for _ in range(heads)
        ])

        self.W_g = nn.ParameterList([
            nn.Parameter(torch.Tensor(out_dim, in_dim))
            for _ in range(heads)
        ])
        self.b_g = nn.ParameterList([
            nn.Parameter(torch.Tensor(out_dim))
            for _ in range(heads)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for h in range(self.heads):
            nn.init.xavier_uniform_(self.W_g[h])
            nn.init.zeros_(self.b_g[h])

    def forward(self, x, edge_index, edge_attr=None):
        n = x.size(0)
        device = x.device

        self_idx = torch.arange(n, device=device)
        all_src = torch.cat([edge_index[0], self_idx])
        all_dst = torch.cat([edge_index[1], self_idx])

        from torch_scatter import scatter

        # Aggregate edge features to nodes (mean of incident edges)
        if edge_attr is not None:
            edge_feat = self.edge_proj(edge_attr)          # [E, in_dim]
            edge_sum = scatter(edge_feat, edge_index[0], dim=0,
                              dim_size=n, reduce='mean')     # [N, in_dim]
            edge_sum += scatter(edge_feat, edge_index[1], dim=0,
                               dim_size=n, reduce='mean')    # [N, in_dim]
            x_aug = torch.cat([x, edge_sum], dim=-1)         # [N, 2*in_dim]
        else:
            x_aug = x

        outputs = []
        for h in range(self.heads):
            if edge_attr is not None:
                o_std = self.gat_heads[h](x_aug, edge_index)
            else:
                o_std = self.gat_heads[h](x, edge_index)

            # CPA sum using original node features (before augmentation)
            v = self.gat_heads[h].lin_r(x_aug if edge_attr is not None else x)
            v_all = v[all_src]
            s = scatter(v_all, all_dst, dim=0, dim_size=n, reduce='sum')

            g = torch.sigmoid(F.linear(x, self.W_g[h], self.b_g[h]))
            o_cpa = g * s
            output = o_std + o_cpa
            outputs.append(output)

        x_out = torch.cat(outputs, dim=-1)
        return x_out


# ==============================================================================
# Pure GAT Baseline
# ==============================================================================

class PureGATBaseline(nn.Module):
    """Plain 4-layer GAT (no CPA), 42-dim input, for comparison."""

    def __init__(self, in_dim=42, hidden_dim=256, num_layers=4, heads=4):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                          concat=True, add_self_loops=True, dropout=0.0)
            )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, data, bond_data=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_proj(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)

        x = global_mean_pool(x, batch)
        x = self.predictor(x)
        return x.squeeze(-1)


# ==============================================================================
# Iteration 2: Hierarchical Bond->Atom Encoding
# ==============================================================================

class _BondToAtomProjector:
    """Mixin-style: project bond graph embeddings back to atom space via mean-pooling."""

    @staticmethod
    def project(bx, bond_data, atom_counts, bond_hidden):
        """
        Args:
            bx: [total_bonds, bond_hidden] bond node embeddings from bond encoder
            bond_data: list of dicts, each with 'bond_to_atom' [2, n_bonds_i]
            atom_counts: tensor [batch_size] number of atoms per molecule
            bond_hidden: int, dimension of bond embeddings

        Returns:
            atom_bond_feat: [total_atoms, bond_hidden] per-atom bond context
        """
        from torch_scatter import scatter
        device = bx.device

        all_src = []
        all_dst = []
        bond_offset = 0
        atom_offset = 0

        for i, bd in enumerate(bond_data):
            n_bonds = bd['bond_num_nodes']
            n_atoms_i = int(atom_counts[i])

            if n_bonds > 0 and n_atoms_i > 0:
                b2a = bd['bond_to_atom'].to(device)
                # Normalize shape: ensure [n_bonds, 2] format
                if b2a.shape[0] == 2 and b2a.shape[1] != 2:
                    b2a = b2a.t().contiguous()  # [2, n_bonds] -> [n_bonds, 2]
                # Each bond maps to 2 atoms: replicate bond indices
                bond_ids = (torch.arange(n_bonds, device=device)
                            .repeat_interleave(2) + bond_offset)
                atom_ids = torch.cat([b2a[:, 0], b2a[:, 1]]) + atom_offset
                all_src.append(bond_ids)
                all_dst.append(atom_ids)

            bond_offset += n_bonds
            atom_offset += n_atoms_i

        if len(all_src) == 0:
            return torch.zeros(atom_offset, bond_hidden, device=device)

        all_src = torch.cat(all_src)
        all_dst = torch.cat(all_dst)
        atom_bond_feat = scatter(bx[all_src], all_dst, dim=0,
                                  dim_size=atom_offset, reduce='mean')
        return atom_bond_feat


class _BondEncoderMixin:
    """Mixin providing bond graph encoding with batching."""

    def _forward_bond_encoder(self, bond_data):
        """Batch all bond graphs, run bond CPA-GAT, return node embeddings."""
        from torch_geometric.data import Data, Batch
        device = next(self.parameters()).device

        bond_graphs = []
        for bd in bond_data:
            bg = Data(
                x=bd['bond_x'],
                edge_index=bd['bond_edge_index'],
                num_nodes=bd['bond_num_nodes'],
            )
            bond_graphs.append(bg)

        bond_batch = Batch.from_data_list(bond_graphs).to(device)
        bx = self.bond_input_proj(bond_batch.x)
        for conv in self.bond_convs:
            bx = conv(bx, bond_batch.edge_index)
            bx = F.elu(bx)
        return bx


# --------------------------------------------------------------------------
# Experiment 6: HierarchicalPolyBGCPA_Pre (Bond->Atom input augmentation)
# --------------------------------------------------------------------------

class HierarchicalPolyBGCPA_Pre(nn.Module, _BondEncoderMixin):
    """
    Hierarchical encoding: bond graph -> atom augmentation BEFORE atom message passing.

    Architecture:
    1. Bond CPA-GAT (3-layer, 128-dim) on bond graphs
    2. Bond->atom projection (mean pool incident bond embeddings per atom)
    3. Concatenate: 42-dim original + 128-dim bond context = 170-dim atom input
    4. Atom CPA-GAT (4-layer, 256-dim) on augmented features
    5. Mean pool -> MLP(256->128->1)
    """

    def __init__(self, atom_in_dim=42, atom_hidden=256, atom_layers=4, atom_heads=4,
                 bond_in_dim=14, bond_hidden=128, bond_layers=3, bond_heads=4,
                 predictor_hidden=128, dropout=0.1):
        super().__init__()
        self.atom_hidden = atom_hidden
        self.bond_hidden = bond_hidden

        # Bond encoder (same as before)
        self.bond_head_dim = bond_hidden // bond_heads
        self.bond_input_proj = nn.Linear(bond_in_dim, bond_hidden)
        self.bond_convs = nn.ModuleList([
            CPAGATv2Conv(bond_hidden, self.bond_head_dim, heads=bond_heads)
            for _ in range(bond_layers)
        ])

        # Atom encoder with augmented input
        self.atom_head_dim = atom_hidden // atom_heads
        self.atom_input_dim = atom_in_dim + bond_hidden  # 42 + 128 = 170
        self.atom_input_proj = nn.Linear(self.atom_input_dim, atom_hidden)
        self.atom_convs = nn.ModuleList([
            CPAGATv2Conv(atom_hidden, self.atom_head_dim, heads=atom_heads)
            for _ in range(atom_layers)
        ])

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(atom_hidden, predictor_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden, 1),
        )

    def forward(self, data, bond_data=None):
        device = data.x.device
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Step 1: Bond encoder
        if bond_data is not None:
            bx = self._forward_bond_encoder(bond_data)

            # Step 2: Bond->Atom projection
            atom_counts = torch.bincount(batch)
            bond_context = _BondToAtomProjector.project(
                bx, bond_data, atom_counts, self.bond_hidden)

            # Step 3: Augment atom features
            x_aug = torch.cat([x, bond_context], dim=-1)
        else:
            x_aug = torch.cat([x, torch.zeros(x.size(0), self.bond_hidden, device=device)], dim=-1)

        # Step 4: Atom encoder
        x_aug = self.atom_input_proj(x_aug)
        for conv in self.atom_convs:
            x_aug = conv(x_aug, edge_index)
            x_aug = F.elu(x_aug)

        # Step 5: Readout
        x_aug = global_mean_pool(x_aug, batch)
        out = self.predictor(x_aug)
        return out.squeeze(-1)


# --------------------------------------------------------------------------
# Experiment 7: HierarchicalPolyBGCPA_Mid (Bond->Atom after layer 1)
# --------------------------------------------------------------------------

class HierarchicalPolyBGCPA_Mid(nn.Module, _BondEncoderMixin):
    """
    Hierarchical mid-fusion: bond context injected AFTER atom CPA-GAT layer 1.

    Architecture:
    1. Atom CPA-GAT layer 1 on original 42-dim -> 256-dim intermediate
    2. Bond CPA-GAT (3-layer, 128-dim) -> bond->atom projection -> 128-dim
    3. Concatenate: 256 + 128 = 384-dim per atom
    4. Projection: 384 -> 256
    5. Atom CPA-GAT layers 2-4 on 256-dim
    6. Mean pool -> MLP(256->128->1)
    """

    def __init__(self, atom_in_dim=42, atom_hidden=256, atom_layers=4, atom_heads=4,
                 bond_in_dim=14, bond_hidden=128, bond_layers=3, bond_heads=4,
                 predictor_hidden=128, dropout=0.1):
        super().__init__()
        self.atom_hidden = atom_hidden
        self.bond_hidden = bond_hidden

        # Bond encoder
        self.bond_head_dim = bond_hidden // bond_heads
        self.bond_input_proj = nn.Linear(bond_in_dim, bond_hidden)
        self.bond_convs = nn.ModuleList([
            CPAGATv2Conv(bond_hidden, self.bond_head_dim, heads=bond_heads)
            for _ in range(bond_layers)
        ])

        # Atom encoder: layer 1 processes original input
        self.atom_head_dim = atom_hidden // atom_heads
        self.atom_input_proj = nn.Linear(atom_in_dim, atom_hidden)
        self.atom_conv1 = CPAGATv2Conv(atom_hidden, self.atom_head_dim, heads=atom_heads)

        # Projection: 256 (atom) + 128 (bond context) -> 256
        self.mid_proj = nn.Linear(atom_hidden + bond_hidden, atom_hidden)

        # Atom encoder: layers 2-4
        self.atom_convs = nn.ModuleList([
            CPAGATv2Conv(atom_hidden, self.atom_head_dim, heads=atom_heads)
            for _ in range(atom_layers - 1)
        ])

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(atom_hidden, predictor_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden, 1),
        )

    def forward(self, data, bond_data=None):
        device = data.x.device
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Step 1: Atom encoder layer 1
        x = self.atom_input_proj(x)
        x = self.atom_conv1(x, edge_index)
        x = F.elu(x)

        # Step 2: Bond context
        if bond_data is not None:
            bx = self._forward_bond_encoder(bond_data)
            atom_counts = torch.bincount(batch)
            bond_context = _BondToAtomProjector.project(
                bx, bond_data, atom_counts, self.bond_hidden)
        else:
            bond_context = torch.zeros(x.size(0), self.bond_hidden, device=device)

        # Step 3: Concatenate & project
        x_mid = torch.cat([x, bond_context], dim=-1)  # 384-dim
        x = self.mid_proj(x_mid)  # -> 256-dim

        # Step 4: Atom encoder layers 2-4
        for conv in self.atom_convs:
            x = conv(x, edge_index)
            x = F.elu(x)

        # Step 5: Readout
        x = global_mean_pool(x, batch)
        out = self.predictor(x)
        return out.squeeze(-1)


# --------------------------------------------------------------------------
# Experiment 8: HierarchicalPolyBGCPA_PreGated (Pre with learned gate fusion)
# --------------------------------------------------------------------------

class HierarchicalPolyBGCPA_PreGated(nn.Module, _BondEncoderMixin):
    """
    Hierarchical pre-fusion with learned gate.

    Architecture:
    1. Bond CPA-GAT (3-layer, 128-dim) -> bond->atom projection -> 128-dim
    2. Learned gate: g = sigmoid(W_g[atom_feat || bond_context])
    3. Augmented input: concat(atom_feat, g * bond_context)
    4. Atom CPA-GAT (4-layer, 256-dim) on 170-dim input
    5. Mean pool -> MLP(256->128->1)
    """

    def __init__(self, atom_in_dim=42, atom_hidden=256, atom_layers=4, atom_heads=4,
                 bond_in_dim=14, bond_hidden=128, bond_layers=3, bond_heads=4,
                 predictor_hidden=128, dropout=0.1):
        super().__init__()
        self.atom_hidden = atom_hidden
        self.bond_hidden = bond_hidden

        # Bond encoder
        self.bond_head_dim = bond_hidden // bond_heads
        self.bond_input_proj = nn.Linear(bond_in_dim, bond_hidden)
        self.bond_convs = nn.ModuleList([
            CPAGATv2Conv(bond_hidden, self.bond_head_dim, heads=bond_heads)
            for _ in range(bond_layers)
        ])

        # Gate for bond context fusion
        self.gate_proj = nn.Linear(atom_in_dim + bond_hidden, bond_hidden)

        # Atom encoder with augmented input
        self.atom_head_dim = atom_hidden // atom_heads
        self.atom_input_dim = atom_in_dim + bond_hidden  # 42 + 128 = 170
        self.atom_input_proj = nn.Linear(self.atom_input_dim, atom_hidden)
        self.atom_convs = nn.ModuleList([
            CPAGATv2Conv(atom_hidden, self.atom_head_dim, heads=atom_heads)
            for _ in range(atom_layers)
        ])

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(atom_hidden, predictor_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden, 1),
        )

    def forward(self, data, bond_data=None):
        device = data.x.device
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Step 1: Bond encoder + projection
        if bond_data is not None:
            bx = self._forward_bond_encoder(bond_data)
            atom_counts = torch.bincount(batch)
            bond_context = _BondToAtomProjector.project(
                bx, bond_data, atom_counts, self.bond_hidden)
        else:
            bond_context = torch.zeros(x.size(0), self.bond_hidden, device=device)

        # Step 2: Gate
        gated_input = torch.cat([x, bond_context], dim=-1)  # 170-dim
        gate = torch.sigmoid(self.gate_proj(gated_input))  # [N, bond_hidden]
        gated_context = gate * bond_context

        # Step 3: Augment atom features
        x_aug = torch.cat([x, gated_context], dim=-1)  # 170-dim

        # Step 4: Atom encoder
        x_aug = self.atom_input_proj(x_aug)
        for conv in self.atom_convs:
            x_aug = conv(x_aug, edge_index)
            x_aug = F.elu(x_aug)

        # Step 5: Readout
        x_aug = global_mean_pool(x_aug, batch)
        out = self.predictor(x_aug)
        return out.squeeze(-1)
