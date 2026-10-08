"""
Run ablation studies for v5 (main experiment already completed).
"""

import os
import sys
import json
import random
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, JumpingKnowledge, global_mean_pool, global_add_pool
from torch_geometric.loader import DataLoader as PyGDataLoader

warnings.filterwarnings("ignore")

# Import training functions from train_v5
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/6_day6_polykanssm')
from train_v5 import (
    CONFIG, atom_features, bond_features, gaussian_distance_encoding,
    smiles_to_graph, TgDataset, collate_fn, load_and_process_data,
    split_data, train_one_epoch, evaluate, train_single_seed
)
from model_v5 import PolyKANSSM_v5, FourierKANLayer
import math

# ============================================================
# Ablation Models
# ============================================================

class PolyKANSSM_v5_NoFourier(nn.Module):
    """Ablation: No Fourier-KAN."""

    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_layers=4,
                 dropout=0.25, spectral_radius=0.99, num_frequencies=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.edge_embed = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.vn_emb = nn.Parameter(torch.zeros(1, hidden_dim))
        self.vn_updates = nn.ModuleList()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.log_lambdas = nn.ParameterList()

        for i in range(num_layers):
            nn_i = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.LayerNorm(2 * hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn_i, edge_dim=hidden_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
            self.vn_updates.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            ))
            self.log_lambdas.append(
                nn.Parameter(torch.full((hidden_dim,), math.log(spectral_radius)))
            )

        self.jk = JumpingKnowledge(mode='cat', channels=hidden_dim, num_layers=num_layers)
        jk_dim = hidden_dim * num_layers

        self.pool_attn = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch, batch_size):
        h = self.node_embed(x)
        edge_h = self.edge_embed(edge_attr)

        batch_size_val = batch.max().item() + 1
        vn_h = self.vn_emb.expand(batch_size_val, -1)

        layer_outputs = []

        for i, (conv, norm, do) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            vn_per_node = vn_h[batch]
            h_in = h + vn_per_node

            h_msg = conv(h_in, edge_index, edge_h)

            lam = torch.exp(self.log_lambdas[i]).clamp(max=0.999)
            h_ssm = h * lam + h_msg

            h = do(norm(h + h_ssm))

            layer_outputs.append(h)

            vn_agg = global_mean_pool(h, batch, batch_size_val)
            vn_h = vn_h + self.vn_updates[i](vn_agg)

        # NO Fourier-KAN

        h_jk = self.jk(layer_outputs)

        attn = torch.sigmoid(self.pool_attn(h_jk))
        h_w = h_jk * attn
        graph_attn = global_add_pool(h_w, batch, batch_size_val)
        graph_mean = global_mean_pool(h_jk, batch, batch_size_val)
        graph_repr = graph_attn + graph_mean
        out = self.head(graph_repr)
        return out.squeeze(-1)


class PolyKANSSM_v5_NoJK(nn.Module):
    """Ablation: No JumpingKnowledge."""

    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_layers=4,
                 dropout=0.25, spectral_radius=0.99, num_frequencies=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.edge_embed = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.vn_emb = nn.Parameter(torch.zeros(1, hidden_dim))
        self.vn_updates = nn.ModuleList()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.log_lambdas = nn.ParameterList()

        self.kan_gate = nn.Parameter(torch.tensor(0.1))
        self.kan = FourierKANLayer(hidden_dim, hidden_dim, num_frequencies)

        for i in range(num_layers):
            nn_i = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.LayerNorm(2 * hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn_i, edge_dim=hidden_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
            self.vn_updates.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            ))
            self.log_lambdas.append(
                nn.Parameter(torch.full((hidden_dim,), math.log(spectral_radius)))
            )

        # NO JK
        jk_dim = hidden_dim

        self.pool_attn = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch, batch_size):
        h = self.node_embed(x)
        edge_h = self.edge_embed(edge_attr)

        batch_size_val = batch.max().item() + 1
        vn_h = self.vn_emb.expand(batch_size_val, -1)

        for i, (conv, norm, do) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            vn_per_node = vn_h[batch]
            h_in = h + vn_per_node

            h_msg = conv(h_in, edge_index, edge_h)

            lam = torch.exp(self.log_lambdas[i]).clamp(max=0.999)
            h_ssm = h * lam + h_msg

            h = do(norm(h + h_ssm))

            vn_agg = global_mean_pool(h, batch, batch_size_val)
            vn_h = vn_h + self.vn_updates[i](vn_agg)

        kan_out = self.kan(h)
        h_final = h + torch.sigmoid(self.kan_gate) * kan_out

        h_jk = h_final

        attn = torch.sigmoid(self.pool_attn(h_jk))
        h_w = h_jk * attn
        graph_attn = global_add_pool(h_w, batch, batch_size_val)
        graph_mean = global_mean_pool(h_jk, batch, batch_size_val)
        graph_repr = graph_attn + graph_mean
        out = self.head(graph_repr)
        return out.squeeze(-1)


class PolyKANSSM_v5_NoVirtualNode(nn.Module):
    """Ablation: No virtual node."""

    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_layers=4,
                 dropout=0.25, spectral_radius=0.99, num_frequencies=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.edge_embed = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.log_lambdas = nn.ParameterList()

        self.kan_gate = nn.Parameter(torch.tensor(0.1))
        self.kan = FourierKANLayer(hidden_dim, hidden_dim, num_frequencies)

        for i in range(num_layers):
            nn_i = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.LayerNorm(2 * hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn_i, edge_dim=hidden_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
            self.log_lambdas.append(
                nn.Parameter(torch.full((hidden_dim,), math.log(spectral_radius)))
            )

        self.jk = JumpingKnowledge(mode='cat', channels=hidden_dim, num_layers=num_layers)
        jk_dim = hidden_dim * num_layers

        self.pool_attn = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch, batch_size):
        h = self.node_embed(x)
        edge_h = self.edge_embed(edge_attr)

        layer_outputs = []

        for i, (conv, norm, do) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            h_msg = conv(h, edge_index, edge_h)

            lam = torch.exp(self.log_lambdas[i]).clamp(max=0.999)
            h_ssm = h * lam + h_msg

            h = do(norm(h + h_ssm))

            layer_outputs.append(h)

        kan_out = self.kan(h)
        h_final = h + torch.sigmoid(self.kan_gate) * kan_out

        layer_outputs[-1] = h_final

        h_jk = self.jk(layer_outputs)

        attn = torch.sigmoid(self.pool_attn(h_jk))
        h_w = h_jk * attn
        graph_attn = global_add_pool(h_w, batch, batch_size)
        graph_mean = global_mean_pool(h_jk, batch, batch_size)
        graph_repr = graph_attn + graph_mean
        out = self.head(graph_repr)
        return out.squeeze(-1)


# ============================================================
# Main
# ============================================================

def main():
    config = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = os.path.join(config["work_dir"], "checkpoints")

    ablation_results = {}

    print(f"\n{'='*60}")
    print(f"ABLATION 1: No Fourier-KAN")
    print(f"{'='*60}")

    graphs, targets = load_and_process_data(config)
    result = train_single_seed(42, graphs, targets, config, device, checkpoint_dir, PolyKANSSM_v5_NoFourier)
    ablation_results["no_fourier"] = result
    print(f"Test RMSE (no Fourier-KAN): {result['test_rmse']:.4f}")

    print(f"\n{'='*60}")
    print(f"ABLATION 2: No JumpingKnowledge")
    print(f"{'='*60}")

    graphs, targets = load_and_process_data(config)
    result = train_single_seed(42, graphs, targets, config, device, checkpoint_dir, PolyKANSSM_v5_NoJK)
    ablation_results["no_jk"] = result
    print(f"Test RMSE (no JK): {result['test_rmse']:.4f}")

    print(f"\n{'='*60}")
    print(f"ABLATION 3: No Virtual Node")
    print(f"{'='*60}")

    graphs, targets = load_and_process_data(config)
    result = train_single_seed(42, graphs, targets, config, device, checkpoint_dir, PolyKANSSM_v5_NoVirtualNode)
    ablation_results["no_virtual_node"] = result
    print(f"Test RMSE (no virtual node): {result['test_rmse']:.4f}")

    ablation_summary = {
        "ablation_1_no_fourier": {
            "test_rmse": float(ablation_results["no_fourier"]["test_rmse"]),
            "test_mae": float(ablation_results["no_fourier"]["test_mae"]),
            "test_r2": float(ablation_results["no_fourier"]["test_r2"]),
        },
        "ablation_2_no_jk": {
            "test_rmse": float(ablation_results["no_jk"]["test_rmse"]),
            "test_mae": float(ablation_results["no_jk"]["test_mae"]),
            "test_r2": float(ablation_results["no_jk"]["test_r2"]),
        },
        "ablation_3_no_virtual_node": {
            "test_rmse": float(ablation_results["no_virtual_node"]["test_rmse"]),
            "test_mae": float(ablation_results["no_virtual_node"]["test_mae"]),
            "test_r2": float(ablation_results["no_virtual_node"]["test_r2"]),
        },
        "timestamp": datetime.now().isoformat(),
    }

    ablation_path = os.path.join(config["work_dir"], "ablation_results_v5.json")
    with open(ablation_path, "w") as f:
        json.dump(ablation_summary, f, indent=2)
    print(f"\nAblation results saved to {ablation_path}")

    print(f"\n{'='*60}")
    print(f"ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"No Fourier-KAN:  {ablation_summary['ablation_1_no_fourier']['test_rmse']:.4f}")
    print(f"No JK:           {ablation_summary['ablation_2_no_jk']['test_rmse']:.4f}")
    print(f"No Virtual Node: {ablation_summary['ablation_3_no_virtual_node']['test_rmse']:.4f}")

    return ablation_summary


if __name__ == "__main__":
    results = main()
