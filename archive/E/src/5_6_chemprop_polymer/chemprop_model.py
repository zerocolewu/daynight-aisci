"""
Chemprop-Polymer: Directed Bond-Level MPNN for Polymer Property Prediction.

Core architecture: Chemprop-style directed bond message passing where each undirected bond
has separate forward and backward message vectors. Periodic edges are treated as regular
directed bonds with special features.
"""
import torch
import torch.nn as nn
from torch_scatter import scatter


class ChempropPolymerMPNN(nn.Module):
    """
    Directed bond-level Message Passing Neural Network (Chemprop-style).

    Key differences from standard GNN:
    - Each undirected bond → 2 directed bonds with separate message vectors
    - Message update uses REVERSE bond message from previous step
    - Periodic edges participate as regular directed bonds
    """

    def __init__(self, atom_fdim, bond_fdim, hidden_dim=300, depth=3, dropout=0.0):
        super().__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout

        # W_init: initial bond message from atom features + bond features
        # Input: atom_features_source + atom_features_target + bond_features
        self.W_init = nn.Linear(atom_fdim * 2 + bond_fdim, hidden_dim)

        # W_h: bond message update (shared across steps)
        # Input: h_atom_source + reverse_bond_message + bond_features
        self.W_h = nn.Linear(hidden_dim + hidden_dim + bond_fdim, hidden_dim)

        # W_a: atom hidden state update (shared across steps)
        # Input: atom_features + aggregated_messages
        self.W_a = nn.Linear(atom_fdim + hidden_dim, hidden_dim)

        # W_o: final atom representation
        # Input: atom_features + aggregated_final_messages
        self.W_o = nn.Linear(atom_fdim + hidden_dim, hidden_dim)

        # Prediction MLP: 2 hidden layers + output
        self.predict = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.relu = nn.ReLU()

    def forward(self, batch, return_bond_messages=False):
        """
        Forward pass for a batch of molecule graphs.

        batch: dict with keys:
            - atom_features: [total_atoms, atom_fdim]
            - bond_features: [total_bonds, bond_fdim]
            - edge_index: [2, total_bonds] (source, target for each directed bond)
            - reverse_map: [total_bonds] index of reverse bond
            - bond_batch: [total_bonds] which molecule each bond belongs to
            - atom_batch: [total_atoms] which molecule each atom belongs to
            - num_atoms_list: list of num_atoms per molecule
            - (optional) periodic_edge_mask: [total_bonds] bool mask for periodic edges
        """
        atom_features = batch['atom_features']
        bond_features = batch['bond_features']
        edge_index = batch['edge_index']
        reverse_map = batch['reverse_map']
        atom_batch = batch['atom_batch']

        n_atoms = atom_features.shape[0]
        n_bonds = bond_features.shape[0]
        device = atom_features.device

        if n_bonds == 0:
            # No bonds - use atom features directly
            atom_hidden = self.relu(self.W_a(atom_features))
            z_atom = self.relu(self.W_o(torch.cat([atom_features, atom_hidden], dim=-1)))
            z_graph = scatter(z_atom, atom_batch, dim=0, reduce='mean')
            pred = self.predict(z_graph)
            if return_bond_messages:
                return pred, None
            return pred

        # Step 1: Initialize directed bond messages
        # m_ij^0 = ReLU(W_init · concat(atom_features_i, atom_features_j, bond_features_ij))
        source = edge_index[0]  # [n_bonds]
        target = edge_index[1]  # [n_bonds]

        source_features = atom_features[source]
        target_features = atom_features[target]
        init_input = torch.cat([source_features, target_features, bond_features], dim=-1)
        messages = self.relu(self.W_init(init_input))  # [n_bonds, hidden_dim]

        # Step 2: Message passing steps (shared W_h and W_a)
        for step in range(self.depth):
            # Aggregate incoming messages for each atom
            # a_i = sum of messages where target == i
            agg_messages = scatter(messages, target, dim=0, dim_size=n_atoms, reduce='sum')

            # Update atom hidden state: h_i = ReLU(W_a · concat(atom_features_i, a_i))
            atom_hidden = self.relu(self.W_a(
                torch.cat([atom_features, agg_messages], dim=-1)
            ))

            # Update directed bond messages
            # m_ij^t = ReLU(W_h · concat(h_i^t, m_ji^{t-1}, bond_features_ij))
            # Get atom hidden state for source atoms
            h_source = atom_hidden[source]  # [n_bonds, hidden_dim]

            # Get REVERSE bond messages from previous step
            reverse_messages = messages[reverse_map]  # [n_bonds, hidden_dim]

            update_input = torch.cat([h_source, reverse_messages, bond_features], dim=-1)
            messages = self.relu(self.W_h(update_input))

        # Step 3: Final aggregation and output
        agg_final = scatter(messages, target, dim=0, dim_size=n_atoms, reduce='sum')

        # Atom representation: z_i = ReLU(W_o · concat(atom_features_i, a_i^final))
        z_atom = self.relu(self.W_o(
            torch.cat([atom_features, agg_final], dim=-1)
        ))

        # Graph pooling: mean over atoms per molecule
        z_graph = scatter(z_atom, atom_batch, dim=0, reduce='mean')

        # Prediction
        pred = self.predict(z_graph).squeeze(-1)

        if return_bond_messages:
            return pred, messages
        return pred


def collate_batch(data_list):
    """
    Collate a list of graph data dicts into a batch.

    Each data dict should have:
        - atom_features: [n_atoms, atom_fdim]
        - bond_features: [n_bonds, bond_fdim]
        - edge_index: [2, n_bonds]
        - reverse_map: [n_bonds]
        - atom_incoming_bonds: list of lists
        - num_atoms: int
        - target: [1]
        - psmiles: str
        - periodic_edge_bonds: list of bond indices
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

    atom_offset = 0
    bond_offset = 0

    for i, data in enumerate(data_list):
        n_atoms = data['num_atoms']
        n_bonds = data['bond_features'].shape[0]

        atom_features_list.append(data['atom_features'])
        bond_features_list.append(data['bond_features'])

        if n_bonds > 0:
            edge_index_list.append(data['edge_index'] + atom_offset)
            reverse_map_list.append(data['reverse_map'] + bond_offset)

        atom_batch_list.append(torch.full((n_atoms,), i, dtype=torch.long))
        bond_batch_list.append(torch.full((n_bonds,), i, dtype=torch.long))

        targets.append(data['target'])
        psmiles_list.append(data.get('psmiles', ''))

        # Periodic edge mask
        if n_bonds > 0:
            periodic_mask = torch.zeros(n_bonds, dtype=torch.bool)
            for pe in data.get('periodic_edge_bonds', []):
                if pe < n_bonds:
                    periodic_mask[pe] = True
            periodic_mask_list.append(periodic_mask)

        atom_offset += n_atoms
        bond_offset += n_bonds

    atom_features = torch.cat(atom_features_list, dim=0)
    bond_features = torch.cat(bond_features_list, dim=0) if bond_features_list else torch.zeros((0, data_list[0]['bond_features'].shape[1]))

    if edge_index_list:
        edge_index = torch.cat(edge_index_list, dim=1)
        reverse_map = torch.cat(reverse_map_list, dim=0)
        periodic_mask = torch.cat(periodic_mask_list, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        reverse_map = torch.zeros(0, dtype=torch.long)
        periodic_mask = torch.zeros(0, dtype=torch.bool)

    atom_batch = torch.cat(atom_batch_list, dim=0)
    bond_batch = torch.cat(bond_batch_list, dim=0) if bond_batch_list else torch.zeros(0, dtype=torch.long)
    targets = torch.cat(targets, dim=0)

    return {
        'atom_features': atom_features,
        'bond_features': bond_features,
        'edge_index': edge_index,
        'reverse_map': reverse_map,
        'atom_batch': atom_batch,
        'bond_batch': bond_batch,
        'targets': targets,
        'psmiles_list': psmiles_list,
        'periodic_mask': periodic_mask,
    }
