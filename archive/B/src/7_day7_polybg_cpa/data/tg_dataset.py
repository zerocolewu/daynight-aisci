"""
Tg dataset loading with bond graph construction support.

Loads precomputed graphs from pickle, matches each to its SMILES
via Tg-value + node-count matching, then computes bond features
and constructs bond graphs using RDKit.
"""

import pickle
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.bond_graph import compute_bond_features_for_graph, construct_bond_graph
from config import DATA_PATH, TG_CSV

sys.path.insert(0, "/home/home_public/aisci/AIscientist/src/alldataset/some_dataprocesscode")
from multimer import generate_multimer_smiles


def load_data():
    with open(DATA_PATH, 'rb') as f:
        pdata = pickle.load(f)

    train_graphs = pdata['train_graphs']
    val_graphs = pdata['val_graphs']
    test_graphs = pdata['test_graphs']
    scaler = pdata['scaler']

    df = pd.read_csv(TG_CSV)
    print("Matching SMILES to precomputed graphs...")

    csv_data = []
    for idx, row in df.iterrows():
        psmiles = row['Smiles']
        tg = float(row['Tg'])
        try:
            multi = generate_multimer_smiles(3, psmiles)
            mol = Chem.MolFromSmiles(multi)
            if mol is not None:
                n_atoms = mol.GetNumAtoms()
                csv_data.append({
                    'psmiles': psmiles, 'multi_smiles': multi,
                    'n_atoms': n_atoms, 'tg': tg, 'idx': idx,
                })
        except Exception:
            pass

    lookup = {}
    for entry in csv_data:
        key = (round(entry['tg'], 2), entry['n_atoms'])
        lookup.setdefault(key, []).append(entry)

    used_indices = set()

    def match_graphs(graphs):
        smiles_list = []
        matched = 0
        for g in graphs:
            tg_denorm = g.y.item() * scaler.scale_[0] + scaler.mean_[0]
            n_atoms = g.num_nodes
            best_match = None
            for precision in [4, 3, 2, 1, 0]:
                candidates = lookup.get((round(tg_denorm, precision), n_atoms), [])
                for c in candidates:
                    if c['idx'] not in used_indices:
                        best_match = c
                        break
                if best_match:
                    break
            if not best_match:
                for c in csv_data:
                    if c['idx'] not in used_indices and abs(c['tg'] - tg_denorm) < 0.5:
                        best_match = c
                        break
            if best_match:
                used_indices.add(best_match['idx'])
                smiles_list.append(best_match['multi_smiles'])
                matched += 1
            else:
                smiles_list.append(None)
        print(f"  Matched {matched}/{len(graphs)} ({100*matched/max(len(graphs),1):.1f}%)")
        return smiles_list

    train_smiles = match_graphs(train_graphs)
    val_smiles = match_graphs(val_graphs)
    test_smiles = match_graphs(test_graphs)
    return train_graphs, val_graphs, test_graphs, scaler, train_smiles, val_smiles, test_smiles


def compute_bond_data(graphs, smiles_list, edge_only_dim=None):
    """
    Compute bond features and bond graphs for each atom graph.
    Returns tuple: (graph_list, bond_data_list).

    bond_data_list[i] = {'bond_x': ..., 'bond_edge_index': ..., 'bond_num_nodes': ...}
    """
    bond_data_list = []
    for i, (g, smi) in enumerate(zip(graphs, smiles_list)):
        if edge_only_dim is not None:
            if smi is not None:
                bond_features, _ = compute_bond_features_for_graph(g, smi)
                g.edge_attr = bond_features[:, :edge_only_dim]
            else:
                g.edge_attr = torch.zeros(g.edge_index.shape[1], edge_only_dim)
            bond_data_list.append(None)
        else:
            if smi is not None:
                bond_features, bond_to_atom = compute_bond_features_for_graph(g, smi)
                bond_g = construct_bond_graph(g, bond_features, bond_to_atom)
                bond_data_list.append({
                    'bond_x': bond_g.x,
                    'bond_edge_index': bond_g.edge_index,
                    'bond_num_nodes': bond_g.num_nodes,
                    'bond_to_atom': bond_g.bond_to_atom,  # for hierarchical encoding
                })
            else:
                num_bonds = g.edge_index.shape[1]
                bond_data_list.append({
                    'bond_x': torch.zeros(num_bonds, 14),
                    'bond_edge_index': torch.zeros((2, 0), dtype=torch.long),
                    'bond_num_nodes': num_bonds,
                    'bond_to_atom': torch.zeros((2, num_bonds), dtype=torch.long),
                })
    return graphs, bond_data_list


# ==============================================================================
# Custom dataset & collation for bond-graph models
# ==============================================================================

class BondGraphDataset(torch.utils.data.Dataset):
    """
    Dataset that stores atom graphs and their bond data separately.
    Uses torch DataLoader (not PyG) for full control over collation.
    """
    def __init__(self, atom_graphs, bond_data_list):
        self.atom_graphs = atom_graphs
        self.bond_data_list = bond_data_list

    def __len__(self):
        return len(self.atom_graphs)

    def __getitem__(self, idx):
        return self.atom_graphs[idx], self.bond_data_list[idx]


def collate_bond_batch(batch):
    """
    Custom collate: batch atom graphs via PyG's Batch, keep bond data as list.
    Returns (atom_batch, bond_batch_list)
    """
    atom_graphs = [item[0] for item in batch]
    bond_data = [item[1] for item in batch]
    atom_batch = Batch.from_data_list(atom_graphs)
    return atom_batch, bond_data
