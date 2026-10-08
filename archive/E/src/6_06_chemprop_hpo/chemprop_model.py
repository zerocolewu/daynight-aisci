"""
Chemprop-Polymer MPNN with optional Dense Skip Connections (DenseGNN-inspired).

Base architecture from Day 5 (5_6_chemprop_polymer) with added dense skip connection support.
When dense_connections=True, message vectors from ALL previous MP steps are concatenated
and projected before each MP update, enabling shorter gradient paths and feature reuse.
"""
import torch
import torch.nn as nn
from torch_scatter import scatter


class ChempropPolymerMPNN(nn.Module):
    """
    Directed bond-level MPNN with optional dense skip connections.

    Dense skip connections: At MP step i, concatenate messages from steps 0..i-1,
    project back to hidden_dim, then use as input to the MP update.
    """

    def __init__(self, atom_fdim, bond_fdim, hidden_dim=300, depth=3, dropout=0.0,
                 dense_connections=False):
        super().__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout
        self.dense_connections = dense_connections

        # W_init: initial bond message from atom features + bond features
        self.W_init = nn.Linear(atom_fdim * 2 + bond_fdim, hidden_dim)

        # W_h: bond message update (shared across steps)
        self.W_h = nn.Linear(hidden_dim + hidden_dim + bond_fdim, hidden_dim)

        # W_a: atom hidden state update (shared across steps)
        self.W_a = nn.Linear(atom_fdim + hidden_dim, hidden_dim)

        # W_o: final atom representation
        self.W_o = nn.Linear(atom_fdim + hidden_dim, hidden_dim)

        # Dense skip connection projections
        # At step i, we concatenate messages from steps 0..i-1 → dimension i*hidden_dim
        # Then project back to hidden_dim
        if dense_connections and depth > 1:
            self.dense_projections = nn.ModuleList()
            for i in range(1, depth):
                self.dense_projections.append(
                    nn.Linear(i * hidden_dim, hidden_dim)
                )

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

        self.relu = nn.ReLU()

    def forward(self, batch, return_bond_messages=False):
        atom_features = batch['atom_features']
        bond_features = batch['bond_features']
        edge_index = batch['edge_index']
        reverse_map = batch['reverse_map']
        atom_batch = batch['atom_batch']

        n_atoms = atom_features.shape[0]
        n_bonds = bond_features.shape[0]
        device = atom_features.device

        if n_bonds == 0:
            atom_hidden = self.relu(self.W_a(atom_features))
            z_atom = self.relu(self.W_o(torch.cat([atom_features, atom_hidden], dim=-1)))
            z_graph = scatter(z_atom, atom_batch, dim=0, reduce='mean')
            pred = self.predict(z_graph)
            if return_bond_messages:
                return pred, None
            return pred

        # Step 1: Initialize directed bond messages
        source = edge_index[0]
        target = edge_index[1]

        source_features = atom_features[source]
        target_features = atom_features[target]
        init_input = torch.cat([source_features, target_features, bond_features], dim=-1)
        messages = self.relu(self.W_init(init_input))

        # Step 2: Message passing with optional dense skip connections
        if self.dense_connections and self.depth > 1:
            # Store ALL message states (including initial) for dense connections
            # message_history stores the state BEFORE each MP update
            all_messages = [messages.clone()]  # initial messages (step 0 input)

            for step in range(self.depth):
                # Dense skip: at step i>0, concatenate messages from steps 0..i-1
                # message_history has (step+1) entries: [init, step0_output, step1_output, ...]
                # At step i, we want messages from steps 0..i-1, which is (step) entries
                if step > 0:
                    # Concatenate messages from all previous steps
                    concat_messages = torch.cat(all_messages[:step], dim=-1)
                    # Project back to hidden_dim
                    projected = self.dense_projections[step - 1](concat_messages)
                    # Use projected for aggregation
                    agg_messages = scatter(projected, target, dim=0, dim_size=n_atoms, reduce='sum')
                else:
                    agg_messages = scatter(messages, target, dim=0, dim_size=n_atoms, reduce='sum')

                # Update atom hidden state
                atom_hidden = self.relu(self.W_a(
                    torch.cat([atom_features, agg_messages], dim=-1)
                ))

                # Update directed bond messages
                h_source = atom_hidden[source]
                reverse_messages = messages[reverse_map]
                update_input = torch.cat([h_source, reverse_messages, bond_features], dim=-1)
                messages = self.relu(self.W_h(update_input))

                # Store this step's output messages
                all_messages.append(messages.clone())
        else:
            # Standard MP (no dense connections)
            for step in range(self.depth):
                agg_messages = scatter(messages, target, dim=0, dim_size=n_atoms, reduce='sum')
                atom_hidden = self.relu(self.W_a(
                    torch.cat([atom_features, agg_messages], dim=-1)
                ))
                h_source = atom_hidden[source]
                reverse_messages = messages[reverse_map]
                update_input = torch.cat([h_source, reverse_messages, bond_features], dim=-1)
                messages = self.relu(self.W_h(update_input))

        # Step 3: Final aggregation and output
        agg_final = scatter(messages, target, dim=0, dim_size=n_atoms, reduce='sum')
        z_atom = self.relu(self.W_o(
            torch.cat([atom_features, agg_final], dim=-1)
        ))
        z_graph = scatter(z_atom, atom_batch, dim=0, reduce='mean')
        pred = self.predict(z_graph).squeeze(-1)

        if return_bond_messages:
            return pred, messages
        return pred


def collate_batch(data_list):
    """Collate a list of graph data dicts into a batch."""
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
