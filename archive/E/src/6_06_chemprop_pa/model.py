"""
Chemprop-Polymer-K: Directed Bond-Level MPNN with K-node for Polymer Property Prediction.

Extensions over baseline Chemprop:
1. K-node support: virtual atom with polymer descriptors (528-dim) connected to all real atoms
2. Projector head for contrastive pre-training
3. Support for both K-node and non-K-node modes

Architecture:
- Real atoms: 123-dim features → W_a/W_o (shared with message passing)
- K-node: 528-dim features → W_k (separate embedding) → participates in message passing
- Virtual bidirectional edges connect K-node to all real atoms
- Graph representation: K-node hidden state (if use_knode) or mean atom pooling
"""
import torch
import torch.nn as nn
from torch_scatter import scatter


class ChempropPolymerKMPNN(nn.Module):
    """
    Directed bond-level Message Passing Neural Network with K-node support.
    """

    def __init__(self, atom_fdim, bond_fdim, k_feat_dim, hidden_dim=1200, depth=7,
                 dropout=0.0, use_knode=True, with_projector=False):
        super().__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.k_feat_dim = k_feat_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout
        self.use_knode = use_knode
        self.with_projector = with_projector

        # W_init: initial bond message
        # Input: atom_features_source + atom_features_target + bond_features
        # NOTE: When use_knode, K-node's "features" for W_init are its hidden_dim embedding
        # So we need two versions of W_init
        if use_knode and k_feat_dim > 0:
            # For bonds between real atoms: 123+123+9=255
            # For bonds involving K-node: hidden_dim+123+9 or 123+hidden_dim+9
            # Simplest approach: project K-node to hidden_dim first, then treat uniformly
            # But W_init expects atom_fdim-sized inputs for both source and target
            # Solution: use a unified hidden_dim input for all atoms after initial projection
            self.W_atom = nn.Linear(atom_fdim, hidden_dim)  # project real atom features
            self.W_knode = nn.Linear(k_feat_dim, hidden_dim)  # project K-node features
            self.W_init = nn.Linear(hidden_dim * 2 + bond_fdim, hidden_dim)
            self.W_a = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
            self.W_o = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        else:
            # Standard Chemprop without K-node
            self.W_init = nn.Linear(atom_fdim * 2 + bond_fdim, hidden_dim)
            self.W_a = nn.Linear(atom_fdim + hidden_dim, hidden_dim)
            self.W_o = nn.Linear(atom_fdim + hidden_dim, hidden_dim)

        # W_h: bond message update (shared across steps)
        self.W_h = nn.Linear(hidden_dim + hidden_dim + bond_fdim, hidden_dim)

        # Prediction MLP
        self.predict = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Projector for contrastive pre-training
        if with_projector:
            self.projector = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )

        self.relu = nn.ReLU()

    def forward(self, batch, return_embedding=False):
        """
        Forward pass.

        batch keys:
            - atom_features: [total_real_atoms, atom_fdim]
            - bond_features: [total_bonds, bond_fdim] (includes virtual bonds)
            - edge_index: [2, total_bonds]
            - reverse_map: [total_bonds]
            - atom_batch: [total_atoms_incl_knode]
            - k_node_indices: [batch_size] (if use_knode)
            - k_node_features: [batch_size, k_feat_dim] (if use_knode)
        """
        atom_features = batch['atom_features']
        bond_features = batch['bond_features']
        edge_index = batch['edge_index']
        reverse_map = batch['reverse_map']
        atom_batch = batch['atom_batch']

        n_real_atoms = atom_features.shape[0]
        n_bonds = bond_features.shape[0]
        device = atom_features.device

        if self.use_knode and self.k_feat_dim > 0:
            return self._forward_with_knode(
                atom_features, bond_features, edge_index, reverse_map,
                atom_batch, batch, n_real_atoms, n_bonds, device, return_embedding
            )
        else:
            return self._forward_standard(
                atom_features, bond_features, edge_index, reverse_map,
                atom_batch, n_real_atoms, n_bonds, device, return_embedding
            )

    def _forward_with_knode(self, atom_features, bond_features, edge_index,
                            reverse_map, atom_batch, batch,
                            n_real_atoms, n_bonds, device, return_embedding):
        """Forward pass with K-node: project all atoms to hidden_dim first."""
        k_node_features = batch['k_node_features']  # [batch_size, k_feat_dim]
        k_node_indices = batch['k_node_indices']  # [batch_size]

        # Step 0: Project all atom features to hidden_dim
        h_real = self.relu(self.W_atom(atom_features))  # [n_real_atoms, hidden_dim]
        h_k = self.relu(self.W_knode(k_node_features))  # [batch_size, hidden_dim]

        # Concatenate: real atoms followed by K-nodes (one per molecule)
        # But K-nodes are interleaved per molecule in collate_batch
        # Actually, collate_batch puts K-node right after each molecule's real atoms
        n_total = atom_batch.shape[0]

        # Build unified hidden state
        # h_all has real atoms and K-nodes interleaved per molecule
        h_all = torch.zeros(n_total, self.hidden_dim, device=device)

        # Fill in real atoms
        real_atom_mask = torch.ones(n_total, dtype=torch.bool, device=device)
        real_atom_mask[k_node_indices] = False
        h_all[real_atom_mask] = h_real

        # Fill in K-nodes
        h_all[k_node_indices] = h_k

        if n_bonds == 0:
            z_graph = h_all[k_node_indices]
            pred = self.predict(z_graph).squeeze(-1)
            if return_embedding:
                return pred, self._get_embedding(z_graph)
            return pred

        # Step 1: Initialize directed bond messages
        source = edge_index[0]
        target = edge_index[1]

        source_h = h_all[source]
        target_h = h_all[target]
        init_input = torch.cat([source_h, target_h, bond_features], dim=-1)
        messages = self.relu(self.W_init(init_input))

        # Step 2: Message passing
        for step in range(self.depth):
            agg_messages = scatter(messages, target, dim=0, dim_size=n_total, reduce='sum')

            atom_hidden = self.relu(self.W_a(
                torch.cat([h_all, agg_messages], dim=-1)
            ))

            h_source = atom_hidden[source]
            reverse_messages = messages[reverse_map]
            update_input = torch.cat([h_source, reverse_messages, bond_features], dim=-1)
            messages = self.relu(self.W_h(update_input))

        # Step 3: Final aggregation
        agg_final = scatter(messages, target, dim=0, dim_size=n_total, reduce='sum')
        z_atom = self.relu(self.W_o(torch.cat([h_all, agg_final], dim=-1)))

        # Graph representation = K-node hidden state
        z_graph = z_atom[k_node_indices]

        pred = self.predict(z_graph).squeeze(-1)

        if return_embedding:
            return pred, self._get_embedding(z_graph)
        return pred

    def _forward_standard(self, atom_features, bond_features, edge_index,
                          reverse_map, atom_batch,
                          n_atoms, n_bonds, device, return_embedding):
        """Standard Chemprop forward pass without K-node."""
        if n_bonds == 0:
            atom_hidden = self.relu(self.W_a(atom_features))
            z_atom = self.relu(self.W_o(torch.cat([atom_features, atom_hidden], dim=-1)))
            z_graph = scatter(z_atom, atom_batch, dim=0, reduce='mean')
            pred = self.predict(z_graph).squeeze(-1)
            if return_embedding:
                return pred, self._get_embedding(z_graph)
            return pred

        source = edge_index[0]
        target = edge_index[1]

        source_features = atom_features[source]
        target_features = atom_features[target]
        init_input = torch.cat([source_features, target_features, bond_features], dim=-1)
        messages = self.relu(self.W_init(init_input))

        for step in range(self.depth):
            agg_messages = scatter(messages, target, dim=0, dim_size=n_atoms, reduce='sum')
            atom_hidden = self.relu(self.W_a(
                torch.cat([atom_features, agg_messages], dim=-1)
            ))
            h_source = atom_hidden[source]
            reverse_messages = messages[reverse_map]
            update_input = torch.cat([h_source, reverse_messages, bond_features], dim=-1)
            messages = self.relu(self.W_h(update_input))

        agg_final = scatter(messages, target, dim=0, dim_size=n_atoms, reduce='sum')
        z_atom = self.relu(self.W_o(torch.cat([atom_features, agg_final], dim=-1)))
        z_graph = scatter(z_atom, atom_batch, dim=0, reduce='mean')

        pred = self.predict(z_graph).squeeze(-1)
        if return_embedding:
            return pred, self._get_embedding(z_graph)
        return pred

    def _get_embedding(self, z_graph):
        """Get embedding from graph representation."""
        if self.with_projector:
            embedding = self.projector(z_graph)
            embedding = embedding / (embedding.norm(dim=1, keepdim=True) + 1e-8)
            return embedding
        return z_graph

    def get_encoder_state_dict(self):
        """Get state dict of encoder (exclude predict head and projector)."""
        state_dict = self.state_dict()
        exclude = ['predict', 'projector']
        return {k: v for k, v in state_dict.items()
                if not any(k.startswith(e) for e in exclude)}

    def load_encoder_state_dict(self, state_dict, strict=False):
        """Load encoder weights, skipping predict head and projector."""
        model_state = self.state_dict()
        loaded = {}
        for key in model_state:
            if key in state_dict:
                loaded[key] = state_dict[key]
            else:
                loaded[key] = model_state[key]
        self.load_state_dict(loaded, strict=strict)


def collate_batch(data_list, use_knode=True):
    """
    Collate a list of graph data dicts into a batch.

    Each data dict should have:
        - atom_features: [n_atoms, atom_fdim] (real atoms only)
        - bond_features: [n_bonds, bond_fdim]
        - edge_index: [2, n_bonds]
        - reverse_map: [n_bonds]
        - num_atoms: int (number of real atoms)
        - target: [1]
        - psmiles: str
        - periodic_edge_bonds: list of bond indices
        - k_node_features: [1, k_feat_dim] (if use_knode)
    """
    atom_features_list = []
    bond_features_list = []
    edge_index_list = []
    reverse_map_list = []
    atom_batch_list = []
    bond_batch_list = []
    targets = []
    psmiles_list = []
    periodic_mask_list = []
    k_node_indices_list = []
    k_node_features_list = []

    atom_offset = 0
    bond_offset = 0
    bond_fdim = data_list[0]['bond_features'].shape[1] if len(data_list) > 0 else 9

    for i, data in enumerate(data_list):
        n_atoms = data['num_atoms']  # Real atoms only
        n_bonds = data['bond_features'].shape[0]

        if use_knode and data.get('has_knode', False):
            # Real atom features
            atom_features_list.append(data['atom_features'])

            # K-node features
            k_feat = data['k_node_features']  # [1, k_feat_dim] or [k_feat_dim]
            if k_feat.dim() == 1:
                k_feat = k_feat.unsqueeze(0)
            k_node_features_list.append(k_feat.squeeze(0))  # [k_feat_dim]

            # K-node index = atom_offset + n_atoms (right after real atoms)
            k_node_idx = atom_offset + n_atoms
            k_node_indices_list.append(k_node_idx)

            total_n_atoms = n_atoms + 1

            # Add virtual bonds: K-node <-> all real atoms (bidirectional)
            virtual_bond_feat = torch.zeros(1, bond_fdim)

            # Virtual edges for K-node
            virt_edges_src = []
            virt_edges_tgt = []
            virt_bond_feats = []

            for a_idx in range(n_atoms):
                global_a_idx = atom_offset + a_idx
                global_k_idx = k_node_idx
                # K-node -> atom
                virt_edges_src.append(global_k_idx)
                virt_edges_tgt.append(global_a_idx)
                virt_bond_feats.append(virtual_bond_feat)
                # atom -> K-node
                virt_edges_src.append(global_a_idx)
                virt_edges_tgt.append(global_k_idx)
                virt_bond_feats.append(virtual_bond_feat)

            # Real bonds
            if n_bonds > 0:
                real_edges = data['edge_index'] + atom_offset
                edge_index_list.append(real_edges)
                reverse_map_list.append(data['reverse_map'] + bond_offset)
                bond_features_list.append(data['bond_features'])

            # Virtual bonds
            if len(virt_edges_src) > 0:
                virt_edge_index = torch.tensor([virt_edges_src, virt_edges_tgt], dtype=torch.long)
                virt_bond_feats = torch.cat(virt_bond_feats, dim=0)
                edge_index_list.append(virt_edge_index)
                bond_features_list.append(virt_bond_feats)

                # Build reverse map for virtual bonds
                # Virtual bonds come after real bonds
                real_bond_offset = bond_offset
                virt_bond_offset = bond_offset + n_bonds
                n_virt_bonds = len(virt_edges_src)

                # Each virtual pair (K->A, A->K) are reverses of each other
                virt_reverse = []
                for j in range(0, n_virt_bonds, 2):
                    virt_reverse.append(virt_bond_offset + j + 1)  # K->A reverse = A->K
                    virt_reverse.append(virt_bond_offset + j)      # A->K reverse = K->A
                virt_reverse = torch.tensor(virt_reverse, dtype=torch.long)
                reverse_map_list.append(virt_reverse)

            total_n_bonds = n_bonds + 2 * n_atoms
        else:
            # No K-node
            atom_features_list.append(data['atom_features'])
            total_n_atoms = n_atoms

            if n_bonds > 0:
                edge_index_list.append(data['edge_index'] + atom_offset)
                reverse_map_list.append(data['reverse_map'] + bond_offset)
                bond_features_list.append(data['bond_features'])

            total_n_bonds = n_bonds

        # Atom batch assignment
        atom_batch_list.append(torch.full((total_n_atoms,), i, dtype=torch.long))
        bond_batch_list.append(torch.full((total_n_bonds,), i, dtype=torch.long))

        targets.append(data['target'])
        psmiles_list.append(data.get('psmiles', ''))

        # Periodic edge mask
        if n_bonds > 0:
            periodic_mask = torch.zeros(n_bonds, dtype=torch.bool)
            for pe in data.get('periodic_edge_bonds', []):
                if pe < n_bonds:
                    periodic_mask[pe] = True
            periodic_mask_list.append(periodic_mask)

        atom_offset += total_n_atoms
        bond_offset += total_n_bonds

    # Concatenate
    atom_features = torch.cat(atom_features_list, dim=0)
    bond_features = torch.cat(bond_features_list, dim=0) if bond_features_list else torch.zeros((0, bond_fdim))

    if edge_index_list:
        edge_index = torch.cat(edge_index_list, dim=1)
        reverse_map = torch.cat(reverse_map_list, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        reverse_map = torch.zeros(0, dtype=torch.long)

    atom_batch = torch.cat(atom_batch_list, dim=0)
    bond_batch = torch.cat(bond_batch_list, dim=0) if bond_batch_list else torch.zeros(0, dtype=torch.long)
    targets = torch.cat(targets, dim=0)

    batch_dict = {
        'atom_features': atom_features,
        'bond_features': bond_features,
        'edge_index': edge_index,
        'reverse_map': reverse_map,
        'atom_batch': atom_batch,
        'bond_batch': bond_batch,
        'targets': targets,
        'psmiles_list': psmiles_list,
        'periodic_mask': torch.cat(periodic_mask_list, dim=0) if periodic_mask_list else torch.zeros(0, dtype=torch.bool),
    }

    if use_knode and k_node_indices_list:
        batch_dict['k_node_indices'] = torch.tensor(k_node_indices_list, dtype=torch.long)
        batch_dict['k_node_features'] = torch.stack(k_node_features_list, dim=0)  # [batch_size, k_feat_dim]

    return batch_dict
