#!/usr/bin/env python3
"""
GraphMAE-Style Masked Node Feature Pretraining on PI1M for Tg Prediction.
ITERATION 2: Fixed graph construction using 3-RU multimer (not cyclization).

Pipeline:
  1. Graph construction from p-SMILES (3-RU multimer + rich features + RWPE k=32)
  2. GraphMAE pretraining on 50K PI1M subset
  3. Finetuning on Tg (with pretrained encoder)
  4. Baseline training on Tg (random init)
  5. Diagnostic analysis & reporting
"""

import os
import sys
import json
import pickle
import warnings
import random
from copy import deepcopy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, BatchNorm, global_mean_pool
from torch_geometric.transforms import AddRandomWalkPE, Compose
from torch.utils.data import Dataset as TorchDataset

from rdkit import Chem
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    work_dir = "/home/home_public/aisci/AIscientist/src/3_day3_graphmae_pretrain"
    data_dir = "/home/home_public/aisci/AIscientist/src/alldataset"
    pi1m_csv = os.path.join(data_dir, "PI1M.csv")
    tg_csv = os.path.join(data_dir, "Tg.csv")

    cache_dir = os.path.join(work_dir, "cache")
    pi1m_cache = os.path.join(cache_dir, "pi1m_50k_graphs_3ru.pkl")
    tg_cache = os.path.join(cache_dir, "tg_graphs_3ru.pkl")

    pretrain_ckpt = os.path.join(work_dir, "best_pretrained_encoder.pt")
    pretrain_loss_csv = os.path.join(work_dir, "pretrain_loss.csv")
    pretrain_loss_png = os.path.join(work_dir, "pretrain_loss.png")

    finetune_json = os.path.join(work_dir, "finetune_results.json")
    finetune_curves_png = os.path.join(work_dir, "finetune_curves.png")
    diagnostic_txt = os.path.join(work_dir, "diagnostic_analysis.txt")

    # Graph
    rwpe_k = 32
    # Rich features: atomic_number(1) + hybrid(6) + aromatic(1) + degree(1) +
    #   formal_charge(1) + ring(1) + chirality(1) + hydrogens(1) = 13
    rich_feat_dim = 13
    total_input_dim = rich_feat_dim + rwpe_k  # 45

    # GAT Encoder
    gat_hidden = 256
    gat_heads = 4
    gat_layers = 4
    gat_dropout = 0.1

    # Decoder
    decoder_hidden = [256, 128]
    decoder_output = 13  # reconstruct rich features only

    # Predictor
    predictor_hidden = [256, 128]
    predictor_output = 1

    # Masking
    mask_ratio = 0.25

    # Pretraining
    pretrain_subset_size = 50000
    pretrain_batch_size = 64  # reduced for larger ~100-node 3-RU graphs
    pretrain_epochs = 50
    pretrain_lr = 1e-3
    pretrain_weight_decay = 1e-5

    # Finetuning / Baseline
    tg_split_ratios = [0.72, 0.08, 0.20]
    tg_split_seed = 71
    ft_batch_size = 32  # reduced for larger ~100-node 3-RU graphs
    ft_max_epochs = 300
    ft_patience = 30
    ft_lr = 3e-4
    ft_weight_decay = 1e-5

    # Random seeds
    pretrain_seed = 42
    baseline_seed = 42

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()
os.makedirs(cfg.cache_dir, exist_ok=True)


# ============================================================================
# 3-RU Multimer Graph Construction (from Day 2 proven pipeline)
# ============================================================================

def generate_multimer(psmiles: str, n_mer: int) -> str:
    """Generate N-mer oligomer SMILES from p-SMILES.

    For n_mer=1: cap * with H, return standard SMILES.
    For n_mer>1: chain n repeat units, remove internal *, cap ends with H.
    """
    monomer = Chem.MolFromSmiles(psmiles)
    if monomer is None:
        return None

    star_info = []
    for atom in monomer.GetAtoms():
        if atom.GetSymbol() == '*':
            neis = atom.GetNeighbors()
            if len(neis) != 1:
                return None
            bond = monomer.GetBondBetweenAtoms(atom.GetIdx(), neis[0].GetIdx())
            star_info.append({
                'star_idx': atom.GetIdx(),
                'neighbor_idx': neis[0].GetIdx(),
                'bond_type': bond.GetBondType()
            })

    if len(star_info) != 2:
        return None
    if star_info[0]['bond_type'] != star_info[1]['bond_type']:
        return None

    num_atoms = monomer.GetNumAtoms()
    oligomer = deepcopy(monomer)
    for _ in range(n_mer - 1):
        oligomer = Chem.CombineMols(oligomer, monomer)

    star_indices = np.zeros((n_mer, 2), dtype=int)
    neighbor_indices = np.zeros((n_mer, 2), dtype=int)
    for i in range(n_mer):
        offset = i * num_atoms
        star_indices[i] = [star_info[0]['star_idx'] + offset,
                           star_info[1]['star_idx'] + offset]
        neighbor_indices[i] = [star_info[0]['neighbor_idx'] + offset,
                               star_info[1]['neighbor_idx'] + offset]

    ed_mol = Chem.EditableMol(oligomer)
    removed_atoms = []

    for i in range(n_mer - 1):
        ed_mol.AddBond(
            int(neighbor_indices[i, 1]),
            int(neighbor_indices[i + 1, 0]),
            order=star_info[0]['bond_type']
        )
        removed_atoms.extend([
            int(star_indices[i, 1]),
            int(star_indices[i + 1, 0])
        ])

    # Cap ends with H
    ed_mol.ReplaceAtom(int(star_indices[0, 0]), Chem.Atom(1))
    ed_mol.ReplaceAtom(int(star_indices[n_mer - 1, 1]), Chem.Atom(1))

    for idx in sorted(removed_atoms, reverse=True):
        ed_mol.RemoveAtom(idx)

    final_mol = ed_mol.GetMol()
    final_mol = Chem.RemoveHs(final_mol)
    return Chem.MolToSmiles(final_mol)


HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
}


def _hyb_onehot(hyb) -> list:
    """One-hot 6-dim: SP, SP2, SP3, SP3D, SP3D2, OTHER."""
    enc = [0.0] * 6
    idx = HYBRIDIZATION_MAP.get(hyb, 5)
    enc[idx] = 1.0
    return enc


def extract_rich_features(mol):
    """Extract 13-dim rich atom features:
    atomic_number/100, hybrid_onehot(6), aromatic, degree/10,
    formal_charge/5, ring, chirality/10, hydrogens/10
    """
    features = []
    for atom in mol.GetAtoms():
        feat = []
        # 1. atomic_number (normalized by 100)
        feat.append(float(atom.GetAtomicNum()) / 100.0)

        # 2. hybridization (one-hot 6-dim)
        feat.extend(_hyb_onehot(atom.GetHybridization()))

        # 3. aromaticity
        feat.append(1.0 if atom.GetIsAromatic() else 0.0)

        # 4. degree (normalized by 10)
        feat.append(min(float(atom.GetDegree()) / 10.0, 1.0))

        # 5. formal_charge (normalized by 5)
        feat.append(max(-1.0, min(float(atom.GetFormalCharge()) / 5.0, 1.0)))

        # 6. ring_membership
        feat.append(1.0 if atom.IsInRing() else 0.0)

        # 7. chirality (int, normalized by 10)
        chiral_val = float(int(atom.GetChiralTag()))
        feat.append(chiral_val / 10.0)

        # 8. hydrogen_count (normalized by 10)
        feat.append(min(float(atom.GetTotalNumHs(includeNeighbors=False)) / 10.0, 1.0))

        features.append(feat)
    return torch.tensor(features, dtype=torch.float32)


def psmiles_to_pyg_data(psmiles, rwpe_transform):
    """Convert a single p-SMILES to PyG Data object via 3-RU multimer.

    Steps:
      1. Generate 3-RU multimer SMILES
      2. Convert to RDKit mol
      3. Extract 13-dim rich features
      4. Build edge_index
      5. Add RWPE (k=32)
      6. Concatenate: total input = 45 dim
    """
    # Step 1: Generate 3-RU multimer
    try:
        smiles = generate_multimer(psmiles, n_mer=3)
    except Exception:
        return None
    if smiles is None:
        return None

    # Step 2: RDKit mol
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # Step 3: Rich features (13-dim)
    x = extract_rich_features(mol)

    # Step 4: Edges (undirected)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index, num_nodes=x.size(0))

    # Step 5: Add RWPE
    try:
        data = rwpe_transform(data)
    except Exception:
        data.random_walk_pe = torch.zeros(data.num_nodes, cfg.rwpe_k)

    # Step 6: Concatenate rich features with RWPE
    data.x = torch.cat([data.x, data.random_walk_pe], dim=-1)
    del data.random_walk_pe

    return data


def build_graphs_from_csv(csv_path, smiles_col, max_count=None, desc="Building graphs"):
    """Build PyG Data objects from a CSV file of p-SMILES."""
    import csv
    rwpe_transform = AddRandomWalkPE(walk_length=cfg.rwpe_k)

    all_smiles = []
    all_targets = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_smiles.append(row[smiles_col])
            for key in row:
                if key.lower() in ['tg', 'target', 'value']:
                    all_targets.append(float(row[key]))
                    break
            else:
                all_targets.append(None)

    if max_count is not None and len(all_smiles) > max_count:
        indices = np.random.RandomState(cfg.pretrain_seed).choice(
            len(all_smiles), max_count, replace=False)
        all_smiles = [all_smiles[i] for i in indices]

    graphs = []
    valid_targets = []
    failures = 0

    for i in tqdm(range(len(all_smiles)), desc=desc):
        data = psmiles_to_pyg_data(all_smiles[i], rwpe_transform)
        if data is None:
            failures += 1
            continue
        graphs.append(data)
        if all_targets[i] is not None:
            valid_targets.append(all_targets[i])

    print(f"  Built {len(graphs)} graphs ({failures} failures)")
    return graphs, valid_targets


# ============================================================================
# Dataset Classes
# ============================================================================

class GraphListDataset(TorchDataset):
    """Simple dataset wrapping a list of PyG Data objects."""
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


class LabeledGraphDataset(TorchDataset):
    """Dataset of graphs with scalar labels."""
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx].clone()
        data.y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return data


# ============================================================================
# Models
# ============================================================================

class GATEncoderWithResidual(nn.Module):
    """4-layer GAT encoder with residual connections and dimension projection."""

    def __init__(self, input_dim=40, hidden_dim=256, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.residual_proj = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim * num_heads
            concat = (i < num_layers - 1)
            out_dim_actual = hidden_dim * num_heads if concat else hidden_dim

            self.convs.append(GATConv(in_dim, hidden_dim, heads=num_heads, concat=concat, dropout=dropout))
            self.bns.append(BatchNorm(out_dim_actual))

            # Residual projection
            if i > 0 and in_dim != out_dim_actual:
                self.residual_proj.append(nn.Linear(in_dim, out_dim_actual, bias=False))
            else:
                self.residual_proj.append(nn.Identity())

        self.output_dim = hidden_dim  # last layer uses mean, output = 256

    def forward(self, x, edge_index, batch=None):
        for i in range(self.num_layers):
            identity = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if i > 0:
                identity = self.residual_proj[i](identity)
                x = x + identity

        return x


class DecoderMLP(nn.Module):
    """MLP decoder: 256 -> 128 -> rich_feat_dim."""
    def __init__(self, input_dim=256, output_dim=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class GraphMAE(nn.Module):
    """GraphMAE: encoder + masking + decoder."""

    def __init__(self, input_dim=45, hidden_dim=256, num_heads=4, num_layers=4,
                 mask_ratio=0.25, dropout=0.1):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.rich_dim = cfg.rich_feat_dim  # 13

        self.encoder = GATEncoderWithResidual(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_heads=num_heads, num_layers=num_layers, dropout=dropout
        )
        self.decoder = DecoderMLP(
            input_dim=self.encoder.output_dim,
            output_dim=self.rich_dim
        )

    def forward(self, x, edge_index, batch=None):
        num_nodes = x.size(0)

        # Generate random mask: mask 25% of nodes
        num_mask = max(1, int(num_nodes * self.mask_ratio))
        perm = torch.randperm(num_nodes, device=x.device)
        mask_indices = perm[:num_mask]
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        mask[mask_indices] = True

        # Store original rich features as targets
        targets = x[mask, :self.rich_dim].clone()

        # Create masked input: zero out rich features for masked nodes
        x_masked = x.clone()
        x_masked[mask, :self.rich_dim] = 0.0

        # Encode
        h = self.encoder(x_masked, edge_index)

        # Decode only masked nodes
        h_masked = h[mask]
        pred = self.decoder(h_masked)

        return pred, targets, mask

    def compute_loss(self, pred, targets):
        return F.mse_loss(pred, targets)


class GATPredictor(nn.Module):
    """GAT encoder + mean pooling + predictor head."""

    def __init__(self, encoder, encoder_output_dim=256,
                 predictor_hidden=[256, 128], predictor_output=1):
        super().__init__()
        self.encoder = encoder
        self.encoder_output_dim = encoder_output_dim

        layers = []
        in_dim = encoder_output_dim
        for h in predictor_hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, predictor_output))
        self.predictor = nn.Sequential(*layers)

    def forward(self, x, edge_index, batch):
        h = self.encoder(x, edge_index)
        h_pooled = global_mean_pool(h, batch)
        out = self.predictor(h_pooled)
        return out.squeeze(-1)


# ============================================================================
# Pretraining
# ============================================================================

def pretrain_graphmae(train_graphs):
    """Run GraphMAE pretraining."""
    print("\n" + "=" * 60)
    print("GRAPHMAE PRETRAINING")
    print("=" * 60)

    torch.manual_seed(cfg.pretrain_seed)
    np.random.seed(cfg.pretrain_seed)
    random.seed(cfg.pretrain_seed)

    dataset = GraphListDataset(train_graphs)
    loader = DataLoader(dataset, batch_size=cfg.pretrain_batch_size, shuffle=True)

    model = GraphMAE(
        input_dim=cfg.total_input_dim,
        hidden_dim=cfg.gat_hidden,
        num_heads=cfg.gat_heads,
        num_layers=cfg.gat_layers,
        mask_ratio=cfg.mask_ratio,
        dropout=cfg.gat_dropout,
    ).to(cfg.device)

    optimizer = AdamW(model.parameters(), lr=cfg.pretrain_lr, weight_decay=cfg.pretrain_weight_decay)

    print(f"  Graphs: {len(train_graphs)}, Batch size: {cfg.pretrain_batch_size}")
    print(f"  Input dim: {cfg.total_input_dim} (rich={cfg.rich_feat_dim} + RWPE={cfg.rwpe_k})")
    print(f"  Epochs: {cfg.pretrain_epochs}, LR: {cfg.pretrain_lr}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_history = []
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, cfg.pretrain_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{cfg.pretrain_epochs}", leave=False):
            batch = batch.to(cfg.device)
            optimizer.zero_grad()

            pred, targets, mask = model(batch.x, batch.edge_index)
            loss = model.compute_loss(pred, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save(model.encoder.state_dict(), cfg.pretrain_ckpt)
            print(f"    -> Best checkpoint saved (loss={best_loss:.6f})")

    print(f"\nPretraining complete. Best loss: {best_loss:.6f} at epoch {best_epoch}")

    # Save loss CSV
    with open(cfg.pretrain_loss_csv, 'w') as f:
        f.write("epoch,loss\n")
        for i, loss in enumerate(loss_history, 1):
            f.write(f"{i},{loss:.6f}\n")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-', linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Pretraining MSE Loss")
    plt.title("GraphMAE Pretraining Loss (PI1M 50K, 3-RU Multimer)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.pretrain_loss_png, dpi=120)
    plt.close()
    print(f"  Loss curve saved to {cfg.pretrain_loss_png}")

    return model.encoder, loss_history


# ============================================================================
# Finetuning
# ============================================================================

def create_tg_split(graphs, labels):
    """Create train/val/test split with seed=71, ratios 72/8/20."""
    indices = np.arange(len(graphs))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.28, random_state=cfg.tg_split_seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=20.0 / 28.0, random_state=cfg.tg_split_seed
    )

    train_graphs = [graphs[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    return (train_graphs, train_labels), (val_graphs, val_labels), (test_graphs, test_labels)


def train_finetune(model, train_loader, val_loader, optimizer, max_epochs, patience, label=""):
    """Train predictor with early stopping."""
    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0

    train_loss_hist = []
    val_rmse_hist = []
    train_rmse_hist = []

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        total_samples = 0
        train_preds = []
        train_targets = []

        for batch in train_loader:
            batch = batch.to(cfg.device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.batch)
            loss = F.mse_loss(pred, batch.y.squeeze())
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
            train_preds.append(pred.detach().cpu())
            train_targets.append(batch.y.squeeze().detach().cpu())

        avg_train_loss = total_loss / total_samples
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_rmse = torch.sqrt(F.mse_loss(train_preds, train_targets)).item()

        train_loss_hist.append(avg_train_loss)
        train_rmse_hist.append(train_rmse)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(cfg.device)
                pred = model(batch.x, batch.edge_index, batch.batch)
                val_preds.append(pred.cpu())
                val_targets.append(batch.y.squeeze().cpu())

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_rmse = torch.sqrt(F.mse_loss(val_preds, val_targets)).item()
        val_rmse_hist.append(val_rmse)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  {label} Epoch {epoch:3d} | Train RMSE: {train_rmse:.2f}K | Val RMSE: {val_rmse:.2f}K")

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  {label} Early stopping at epoch {epoch} (best val={best_val_rmse:.2f}K at epoch {best_epoch})")
                break

    if epoch == max_epochs:
        print(f"  {label} Max epochs reached (best val={best_val_rmse:.2f}K at epoch {best_epoch})")

    model.load_state_dict(best_state)

    return (train_loss_hist, train_rmse_hist, val_rmse_hist,
            best_val_rmse, best_epoch)


def evaluate(model, test_loader):
    """Evaluate model on test set."""
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(cfg.device)
            pred = model(batch.x, batch.edge_index, batch.batch)
            preds.append(pred.cpu())
            targets.append(batch.y.squeeze().cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    rmse = torch.sqrt(F.mse_loss(preds, targets)).item()
    mae = F.l1_loss(preds, targets).item()
    return rmse, mae, preds.numpy(), targets.numpy()


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    print("=" * 60)
    print("GRAPHMAE EXPERIMENT: Pretraining on PI1M for Tg Prediction")
    print("ITERATION 2: 3-RU Multimer Graph Construction")
    print("=" * 60)
    print(f"Device: {cfg.device}")

    # ------------------------------------------------------------------
    # Stage 1: Build / Load 3-RU Graphs
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("STAGE 1: 3-RU MULTIMER GRAPH CONSTRUCTION")
    print("-" * 40)

    # PI1M graphs (50K subset)
    if os.path.exists(cfg.pi1m_cache):
        print(f"Loading cached PI1M 3-RU graphs from {cfg.pi1m_cache}...")
        with open(cfg.pi1m_cache, 'rb') as f:
            pi1m_graphs = pickle.load(f)
        print(f"  Loaded {len(pi1m_graphs)} graphs")
    else:
        print(f"Building PI1M 3-RU graphs (50K subset)...")
        print("  NOTE: This may take 30-60 min for 50K graphs with 3-RU + RWPE")
        pi1m_graphs, _ = build_graphs_from_csv(
            cfg.pi1m_csv, smiles_col="SMILES",
            max_count=cfg.pretrain_subset_size, desc="Building PI1M 3-RU graphs"
        )
        with open(cfg.pi1m_cache, 'wb') as f:
            pickle.dump(pi1m_graphs, f)
        print(f"  Saved to {cfg.pi1m_cache}")

    # Tg graphs
    if os.path.exists(cfg.tg_cache):
        print(f"Loading cached Tg 3-RU graphs from {cfg.tg_cache}...")
        with open(cfg.tg_cache, 'rb') as f:
            tg_graphs, tg_labels = pickle.load(f)
        print(f"  Loaded {len(tg_graphs)} graphs")
    else:
        print(f"Building Tg 3-RU graphs...")
        tg_graphs, tg_labels = build_graphs_from_csv(
            cfg.tg_csv, smiles_col="Smiles",
            max_count=None, desc="Building Tg 3-RU graphs"
        )
        with open(cfg.tg_cache, 'wb') as f:
            pickle.dump((tg_graphs, tg_labels), f)
        print(f"  Saved to {cfg.tg_cache}")

    # Print graph statistics
    print("\n  Graph Statistics:")
    if pi1m_graphs:
        n_nodes = [g.num_nodes for g in pi1m_graphs]
        print(f"    PI1M: {len(pi1m_graphs)} graphs, avg nodes: {np.mean(n_nodes):.1f}, "
              f"min/max: {min(n_nodes)}/{max(n_nodes)}")
    if tg_graphs:
        n_nodes = [g.num_nodes for g in tg_graphs]
        print(f"    Tg: {len(tg_graphs)} graphs, avg nodes: {np.mean(n_nodes):.1f}, "
              f"min/max: {min(n_nodes)}/{max(n_nodes)}")

    # Create Tg splits
    (train_g, train_l), (val_g, val_l), (test_g, test_l) = create_tg_split(tg_graphs, tg_labels)

    tg_train_dataset = LabeledGraphDataset(train_g, train_l)
    tg_val_dataset = LabeledGraphDataset(val_g, val_l)
    tg_test_dataset = LabeledGraphDataset(test_g, test_l)

    # ------------------------------------------------------------------
    # Stage 2: GraphMAE Pretraining
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("STAGE 2: GRAPHMAE PRETRAINING")
    print("-" * 40)

    def create_encoder():
        return GATEncoderWithResidual(
            input_dim=cfg.total_input_dim,
            hidden_dim=cfg.gat_hidden,
            num_heads=cfg.gat_heads,
            num_layers=cfg.gat_layers,
            dropout=cfg.gat_dropout,
        )

    pretrained_encoder, pretrain_loss_hist = pretrain_graphmae(pi1m_graphs)

    # ------------------------------------------------------------------
    # Stage 3: Finetuning (Pretrained)
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("STAGE 3: FINETUNING (PRETRAINED ENCODER)")
    print("-" * 40)

    torch.manual_seed(cfg.tg_split_seed)
    np.random.seed(cfg.tg_split_seed)

    ft_encoder = create_encoder()
    ft_encoder.load_state_dict(torch.load(cfg.pretrain_ckpt, map_location='cpu'))
    ft_encoder.to(cfg.device)

    ft_model = GATPredictor(
        ft_encoder,
        encoder_output_dim=cfg.gat_hidden,
        predictor_hidden=cfg.predictor_hidden,
        predictor_output=cfg.predictor_output,
    ).to(cfg.device)

    ft_optimizer = AdamW(ft_model.parameters(), lr=cfg.ft_lr, weight_decay=cfg.ft_weight_decay)

    ft_train_loader = DataLoader(tg_train_dataset, batch_size=cfg.ft_batch_size, shuffle=True)
    ft_val_loader = DataLoader(tg_val_dataset, batch_size=cfg.ft_batch_size, shuffle=False)
    ft_test_loader = DataLoader(tg_test_dataset, batch_size=cfg.ft_batch_size, shuffle=False)

    (ft_train_loss, ft_train_rmse, ft_val_rmse,
     ft_best_val_rmse, ft_best_epoch) = train_finetune(
        ft_model, ft_train_loader, ft_val_loader,
        ft_optimizer, cfg.ft_max_epochs, cfg.ft_patience,
        label="[FT]"
    )

    ft_test_rmse, ft_test_mae, ft_test_preds, ft_test_targets = evaluate(ft_model, ft_test_loader)
    print(f"\nPretrained Test RMSE: {ft_test_rmse:.2f}K, MAE: {ft_test_mae:.2f}K")

    # ------------------------------------------------------------------
    # Stage 4: Baseline (Random Init)
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("STAGE 4: BASELINE (RANDOM INIT)")
    print("-" * 40)

    torch.manual_seed(cfg.baseline_seed)
    np.random.seed(cfg.baseline_seed)

    bl_encoder = create_encoder().to(cfg.device)
    bl_model = GATPredictor(
        bl_encoder,
        encoder_output_dim=cfg.gat_hidden,
        predictor_hidden=cfg.predictor_hidden,
        predictor_output=cfg.predictor_output,
    ).to(cfg.device)

    bl_optimizer = AdamW(bl_model.parameters(), lr=cfg.ft_lr, weight_decay=cfg.ft_weight_decay)

    (bl_train_loss, bl_train_rmse, bl_val_rmse,
     bl_best_val_rmse, bl_best_epoch) = train_finetune(
        bl_model, ft_train_loader, ft_val_loader,
        bl_optimizer, cfg.ft_max_epochs, cfg.ft_patience,
        label="[BL]"
    )

    bl_test_rmse, bl_test_mae, bl_test_preds, bl_test_targets = evaluate(bl_model, ft_test_loader)
    print(f"\nBaseline Test RMSE: {bl_test_rmse:.2f}K, MAE: {bl_test_mae:.2f}K")

    # ------------------------------------------------------------------
    # Stage 5: Results & Analysis
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("STAGE 5: RESULTS & ANALYSIS")
    print("-" * 40)

    improvement_k = bl_test_rmse - ft_test_rmse
    pct_improvement = (improvement_k / bl_test_rmse) * 100 if bl_test_rmse > 0 else 0

    results = {
        "graph_construction": "3-RU multimer",
        "pretrained_test_rmse": round(ft_test_rmse, 4),
        "baseline_test_rmse": round(bl_test_rmse, 4),
        "pretrained_best_val_rmse": round(ft_best_val_rmse, 4),
        "baseline_best_val_rmse": round(bl_best_val_rmse, 4),
        "improvement_k": round(improvement_k, 4),
        "improvement_pct": round(pct_improvement, 2),
        "pretrained_test_mae": round(ft_test_mae, 4),
        "baseline_test_mae": round(bl_test_mae, 4),
        "pretrained_epochs_to_best": ft_best_epoch,
        "baseline_epochs_to_best": bl_best_epoch,
        "expected_baseline_k": 35.94,
        "baseline_deviation_k": round(bl_test_rmse - 35.94, 4),
    }

    with open(cfg.finetune_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {cfg.finetune_json}")

    # Plot finetuning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pretrained
    ax = axes[0]
    epochs = range(1, len(ft_train_rmse) + 1)
    ax.plot(epochs, ft_train_rmse, 'b-', alpha=0.5, label='Train RMSE')
    ax.plot(epochs, ft_val_rmse, 'r-', label='Val RMSE')
    ax.axvline(x=ft_best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best epoch {ft_best_epoch}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE (K)')
    ax.set_title('Pretrained GAT Finetuning')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Baseline
    ax = axes[1]
    epochs = range(1, len(bl_train_rmse) + 1)
    ax.plot(epochs, bl_train_rmse, 'b-', alpha=0.5, label='Train RMSE')
    ax.plot(epochs, bl_val_rmse, 'r-', label='Val RMSE')
    ax.axvline(x=bl_best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best epoch {bl_best_epoch}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE (K)')
    ax.set_title('Baseline GAT (Random Init)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'3-RU Multimer | Pretrained vs Baseline | '
                 f'Baseline={bl_test_rmse:.1f}K | \u0394RMSE = {improvement_k:+.2f}K ({pct_improvement:+.1f}%)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(cfg.finetune_curves_png, dpi=120)
    plt.close()
    print(f"Finetuning curves saved to {cfg.finetune_curves_png}")

    # ------------------------------------------------------------------
    # Diagnostic Analysis
    # ------------------------------------------------------------------
    diagnostic_lines = []
    diagnostic_lines.append("=" * 60)
    diagnostic_lines.append("GRAPHMAE PRETRAINING DIAGNOSTIC ANALYSIS (3-RU Multimer)")
    diagnostic_lines.append("=" * 60)
    diagnostic_lines.append("")

    diagnostic_lines.append("--- Critical Fix ---")
    diagnostic_lines.append("Previous iteration used cyclization-based graph construction")
    diagnostic_lines.append("(connect * neighbors, remove * atoms) which caused a 4K RMSE")
    diagnostic_lines.append("regression (39.94K vs expected 35.94K).")
    diagnostic_lines.append("This iteration uses 3-RU multimer construction (identical to Day 2),")
    diagnostic_lines.append("which replicates the monomer 3x, connects via * attachment points,")
    diagnostic_lines.append("removes internal *, and caps ends with H.")
    diagnostic_lines.append("")

    diagnostic_lines.append("--- Baseline Validation ---")
    deviation = bl_test_rmse - 35.94
    if abs(deviation) <= 1.0:
        diagnostic_lines.append(f"PASS: Baseline {bl_test_rmse:.2f}K is within ±1K of expected 35.94K")
        diagnostic_lines.append(f"  (deviation = {deviation:+.2f}K). Graph construction matches Day 2 pipeline.")
    else:
        diagnostic_lines.append(f"WARNING: Baseline {bl_test_rmse:.2f}K deviates by {deviation:+.2f}K from expected 35.94K")
        diagnostic_lines.append(f"  Possible causes: feature definition differences, architecture differences,")
        diagnostic_lines.append(f"  or dataset processing variations.")

    diagnostic_lines.append("")

    diagnostic_lines.append("--- Results Summary ---")
    diagnostic_lines.append(f"Pretrained Test RMSE:  {ft_test_rmse:.2f} K")
    diagnostic_lines.append(f"Baseline Test RMSE:    {bl_test_rmse:.2f} K")
    diagnostic_lines.append(f"Improvement:           {improvement_k:.2f} K ({pct_improvement:+.1f}%)")
    diagnostic_lines.append(f"Pretrained Best Val:   {ft_best_val_rmse:.2f} K (epoch {ft_best_epoch})")
    diagnostic_lines.append(f"Baseline Best Val:     {bl_best_val_rmse:.2f} K (epoch {bl_best_epoch})")
    diagnostic_lines.append("")

    diagnostic_lines.append("--- Convergence Analysis ---")
    if ft_best_epoch < bl_best_epoch:
        diagnostic_lines.append(f"Pretrained model converged {bl_best_epoch - ft_best_epoch} epochs faster "
                                f"({ft_best_epoch} vs {bl_best_epoch}), suggesting better initialization.")
    else:
        diagnostic_lines.append(f"Pretrained model converged after {ft_best_epoch} epochs "
                                f"(baseline: {bl_best_epoch}).")
    diagnostic_lines.append("")

    diagnostic_lines.append("--- Hypothesis Check ---")
    if improvement_k >= 1.5:
        diagnostic_lines.append(f"HYPOTHESIS SUPPORTED: Pretraining improved RMSE by {improvement_k:.2f}K "
                                f"(>= 1.5K threshold).")
    elif improvement_k > 0:
        diagnostic_lines.append(f"HYPOTHESIS PARTIALLY SUPPORTED: Small improvement of {improvement_k:.2f}K "
                                f"(< 1.5K threshold), but direction is correct.")
    else:
        diagnostic_lines.append(f"HYPOTHESIS NOT SUPPORTED: Pretraining did not improve RMSE "
                                f"({improvement_k:+.2f}K).")
    diagnostic_lines.append("")

    diagnostic_lines.append("--- Pretraining Loss Curve ---")
    if pretrain_loss_hist:
        diagnostic_lines.append(f"Initial loss: {pretrain_loss_hist[0]:.6f}")
        diagnostic_lines.append(f"Final loss:   {pretrain_loss_hist[-1]:.6f}")
        reduction = (pretrain_loss_hist[0] - pretrain_loss_hist[-1]) / pretrain_loss_hist[0] * 100 if pretrain_loss_hist[0] > 0 else 0
        diagnostic_lines.append(f"Reduction:    {reduction:.1f}%")
    diagnostic_lines.append("")

    diagnostic_lines.append("--- Weight Transfer Analysis ---")
    ft_encoder_state = {k: v.cpu().clone() for k, v in ft_encoder.state_dict().items()}
    original_pretrained = torch.load(cfg.pretrain_ckpt, map_location='cpu')

    weight_change = 0.0
    param_count = 0
    for key in original_pretrained:
        if key in ft_encoder_state:
            diff = (ft_encoder_state[key] - original_pretrained[key]).norm().item()
            orig_norm = original_pretrained[key].norm().item()
            rel_change = diff / max(orig_norm, 1e-8) * 100
            diagnostic_lines.append(f"  {key}: orig_norm={orig_norm:.2f}, Δ={diff:.2f} ({rel_change:.1f}%)")
            weight_change += rel_change
            param_count += 1
    if param_count > 0:
        diagnostic_lines.append(f"  Average relative change: {weight_change/param_count:.1f}%")
    diagnostic_lines.append("")

    diagnostic_lines.append("--- Data Statistics ---")
    diagnostic_lines.append(f"Graph construction: 3-RU multimer")
    diagnostic_lines.append(f"Rich feature dim: {cfg.rich_feat_dim} (atomic_num/100, hyb_oh6, aromatic, degree/10, "
                            f"charge/5, ring, chiral/10, H/10)")
    diagnostic_lines.append(f"Total input dim: {cfg.total_input_dim} ({cfg.rich_feat_dim} rich + {cfg.rwpe_k} RWPE)")
    diagnostic_lines.append(f"PI1M pretraining graphs: {len(pi1m_graphs)}")
    diagnostic_lines.append(f"Tg total graphs: {len(tg_graphs)}")
    diagnostic_lines.append(f"Tg train: {len(train_g)}, val: {len(val_g)}, test: {len(test_g)}")

    if pi1m_graphs:
        nodes_per_graph = [g.num_nodes for g in pi1m_graphs]
        edges_per_graph = [g.edge_index.size(1) // 2 for g in pi1m_graphs]
        diagnostic_lines.append(f"PI1M avg nodes/graph: {np.mean(nodes_per_graph):.1f}, "
                                f"avg edges/graph: {np.mean(edges_per_graph):.1f}")
    if tg_graphs:
        nodes_per_graph = [g.num_nodes for g in tg_graphs]
        edges_per_graph = [g.edge_index.size(1) // 2 for g in tg_graphs]
        diagnostic_lines.append(f"Tg avg nodes/graph: {np.mean(nodes_per_graph):.1f}, "
                                f"avg edges/graph: {np.mean(edges_per_graph):.1f}")

    diagnostic_text = "\n".join(diagnostic_lines)
    with open(cfg.diagnostic_txt, 'w') as f:
        f.write(diagnostic_text)
    print(diagnostic_text)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"All outputs in: {cfg.work_dir}")


if __name__ == "__main__":
    main()
