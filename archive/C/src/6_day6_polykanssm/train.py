"""
Training pipeline for PolyKAN-SSM on Tg prediction dataset.

Handles:
- Data loading and graph construction with non-covalent edges
- Training with early stopping
- Multi-seed experiments
- Evaluation and results reporting
"""

import os
import sys
import json
import random
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.rdchem import GetPeriodicTable

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader

from model_v3 import PolyKANSSM_v3

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    "data_path": "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv",
    "work_dir": "/home/home_public/aisci/AIscientist/src/6_day6_polykanssm",
    "hidden_dim": 256,
    "num_layers": 4,
    "dropout": 0.2,
    "spectral_radius": 0.99,
    "num_frequencies": 4,
    "lr": 5e-4,
    "weight_decay": 1e-3,
    "batch_size": 32,
    "max_epochs": 500,
    "patience": 40,
    "scheduler_patience": 15,
    "scheduler_factor": 0.5,
    "num_frequencies": 8,
    "noncovalent_cutoff": 5.0,  # Angstroms
    "num_seeds": 5,
    "train_ratio": 0.70,
    "val_ratio": 0.10,
    # test_ratio = 0.20
    "seeds": [42, 123, 456, 789, 2024],
}

# ============================================================
# Atom and Bond Feature Extraction
# ============================================================

# Atom types for one-hot encoding
ATOM_TYPES = ["C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I", "other"]
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    "other",
]

NUM_ATOM_FEATURES = len(ATOM_TYPES) + 6  # type(OH) + degree + hybrid(OH) + aromatic + formal_charge + in_ring
NUM_BOND_FEATURES = 7 + 16  # type(OH) + conjugated + in_ring + covalent_flag + distance_encoding(16)


def atom_features(atom):
    """Extract atom features as a list of floats."""
    features = []

    # Atom type one-hot
    atom_type = atom.GetSymbol()
    type_vec = [0.0] * len(ATOM_TYPES)
    if atom_type in ATOM_TYPES:
        type_vec[ATOM_TYPES.index(atom_type)] = 1.0
    else:
        type_vec[-1] = 1.0
    features.extend(type_vec)

    # Degree (integer, normalized)
    features.append(float(atom.GetDegree()) / 10.0)

    # Hybridization one-hot
    hyb = atom.GetHybridization()
    hyb_vec = [0.0] * len(HYBRIDIZATION_TYPES)
    if hyb in HYBRIDIZATION_TYPES:
        hyb_vec[HYBRIDIZATION_TYPES.index(hyb)] = 1.0
    else:
        hyb_vec[-1] = 1.0
    features.extend(hyb_vec)

    # Aromaticity
    features.append(float(atom.GetIsAromatic()))

    # Formal charge (normalized)
    features.append(float(atom.GetFormalCharge()) / 8.0)

    # Is in ring
    features.append(float(atom.IsInRing()))

    return features


def bond_features(bond, is_covalent=True):
    """Extract bond features (without distance) as a list of floats."""
    features = []

    if is_covalent:
        # Bond type one-hot
        bond_type = bond.GetBondType()
        type_vec = [
            float(bond_type == Chem.rdchem.BondType.SINGLE),
            float(bond_type == Chem.rdchem.BondType.DOUBLE),
            float(bond_type == Chem.rdchem.BondType.TRIPLE),
            float(bond_type == Chem.rdchem.BondType.AROMATIC),
        ]
        features.extend(type_vec)

        # Conjugated
        features.append(float(bond.GetIsConjugated()))

        # In ring
        features.append(float(bond.IsInRing()))
    else:
        # Non-covalent: no bond type, not conjugated, not in ring
        features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Covalent flag (1.0 for covalent, 0.0 for non-covalent)
    features.append(float(is_covalent))

    return features


def gaussian_distance_encoding(distance, num_basis=16, min_dist=0.0, max_dist=8.0):
    """Encode distance using Gaussian basis functions."""
    centres = np.linspace(min_dist, max_dist, num_basis)
    width = (max_dist - min_dist) / num_basis
    features = np.exp(-((distance - centres) ** 2) / (2 * width ** 2))
    return features.tolist()


# ============================================================
# Graph Construction
# ============================================================

def smiles_to_graph(smiles, noncovalent_cutoff=5.0):
    """
    Convert a polymer SMILES to a PyG Data object with:
    - Covalent edges from chemical bonds
    - Non-covalent edges from distance-based cutoff
    - 3D conformer-based distances (or 2D fallback)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Add hydrogens for 3D conformer generation
    mol_h = Chem.AddHs(mol)

    # Try 3D conformer generation
    use_3d = False
    conf = None
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.maxAttempts = 1
        params.numThreads = 1
        status = AllChem.EmbedMolecule(mol_h, params)
        if status == 0:  # success
            try:
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=50)
            except:
                pass
            conf = mol_h.GetConformer()
            use_3d = True
    except:
        pass

    # Use mol without H for feature extraction
    mol_no_h = Chem.RemoveHs(mol_h) if use_3d else mol

    num_atoms = mol_no_h.GetNumAtoms()
    if num_atoms == 0:
        return None

    # Node features
    node_feats = []
    for atom in mol_no_h.GetAtoms():
        node_feats.append(atom_features(atom))
    x = torch.tensor(node_feats, dtype=torch.float32)

    # Build covalent edge index and features
    src_list, dst_list = [], []
    cov_edge_feats = []

    # Track covalent bonds for non-covalent exclusion
    covalent_pairs = set()
    for bond in mol_no_h.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        src_list.extend([i, j])
        dst_list.extend([j, i])
        covalent_pairs.add((min(i, j), max(i, j)))

        # Bond features with distance
        bf = bond_features(bond, is_covalent=True)
        if use_3d:
            pos_i = conf.GetAtomPosition(i)
            pos_j = conf.GetAtomPosition(j)
            dist = pos_i.Distance(pos_j)
            bf.extend(gaussian_distance_encoding(dist))
        else:
            bf.extend(gaussian_distance_encoding(1.5))  # default bond length
        cov_edge_feats.extend([bf, bf])

    # Non-covalent edges based on distance cutoff
    nc_src_list, nc_dst_list = [], []
    nc_edge_feats = []

    if use_3d and conf is not None:
        # Use heavy atom positions from mol_h, need to map indices
        # For mol_no_h, we use its positions directly from conf
        # But conf was generated for mol_h, so we need to map
        # Actually, let's re-embed mol_no_h or use the conf differently
        # Simpler: get positions for heavy atoms from mol_h conformer
        heavy_atom_indices = []
        for atom in mol_h.GetAtoms():
            if atom.GetAtomicNum() > 1:
                heavy_atom_indices.append(atom.GetIdx())

        if len(heavy_atom_indices) == num_atoms:
            for idx_a in range(num_atoms):
                for idx_b in range(idx_a + 1, num_atoms):
                    pair = (min(idx_a, idx_b), max(idx_a, idx_b))
                    if pair in covalent_pairs:
                        continue

                    pos_a = conf.GetAtomPosition(heavy_atom_indices[idx_a])
                    pos_b = conf.GetAtomPosition(heavy_atom_indices[idx_b])
                    dist = pos_a.Distance(pos_b)

                    if dist < noncovalent_cutoff:
                        nc_src_list.extend([idx_a, idx_b])
                        nc_dst_list.extend([idx_b, idx_a])

                        # Edge features: bond features + distance encoding
                        ef = bond_features(None, is_covalent=False)
                        ef.extend(gaussian_distance_encoding(dist))
                        nc_edge_feats.extend([ef, ef])
    else:
        # 2D fallback: use shortest path distances
        # For small molecules, add edges between atoms within topological distance <= 3
        try:
            from rdkit.Chem import GetDistanceMatrix
            dist_matrix = Chem.GetDistanceMatrix(mol_no_h)
            for i in range(num_atoms):
                for j in range(i + 1, num_atoms):
                    pair = (min(i, j), max(i, j))
                    if pair in covalent_pairs:
                        continue

                    topo_dist = dist_matrix[i, j]
                    if topo_dist <= 3.0:  # Short topological distance
                        est_dist = topo_dist * 1.5  # Rough distance estimate
                        if est_dist < noncovalent_cutoff:
                            nc_src_list.extend([i, j])
                            nc_dst_list.extend([j, i])
                            ef = bond_features(None, is_covalent=False)
                            ef.extend(gaussian_distance_encoding(est_dist))
                            nc_edge_feats.extend([ef, ef])
        except:
            pass

    # Combine covalent and non-covalent edges
    all_src = src_list + nc_src_list
    all_dst = dst_list + nc_dst_list
    all_edge_feats = cov_edge_feats + nc_edge_feats

    if len(all_src) == 0:
        # No edges at all - create self-loops
        all_src = list(range(num_atoms))
        all_dst = list(range(num_atoms))
        all_edge_feats = [[0.0] * NUM_BOND_FEATURES for _ in range(num_atoms)]

    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    edge_attr = torch.tensor(all_edge_feats, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ============================================================
# Dataset
# ============================================================

class TgDataset(Dataset):
    """Dataset for polymer Tg prediction."""

    def __init__(self, graphs, targets):
        self.graphs = graphs
        self.targets = targets

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.targets[idx]


def collate_fn(batch):
    """Custom collate function for PyG graphs."""
    graphs, targets = zip(*batch)
    targets = torch.tensor(targets, dtype=torch.float32)
    batch_pyg = Batch.from_data_list(graphs)
    return batch_pyg, targets


def load_and_process_data(config):
    """Load dataset, build graphs, and split into train/val/test."""
    print("Loading dataset...")
    df = pd.read_csv(config["data_path"])
    print(f"Dataset loaded: {len(df)} samples, columns: {df.columns.tolist()}")

    # Standardize column names
    df.columns = ["SMILES", "Tg"]

    # Filter out NaN values
    df = df.dropna(subset=["SMILES", "Tg"])
    print(f"After filtering NaN: {len(df)} samples")

    # Build graphs
    print("Building graphs with non-covalent edges...")
    graphs = []
    targets = []
    failed = 0
    for i, row in df.iterrows():
        smiles = row["SMILES"]
        tg = row["Tg"]
        graph = smiles_to_graph(smiles, config["noncovalent_cutoff"])
        if graph is not None:
            graphs.append(graph)
            targets.append(float(tg))
        else:
            failed += 1

    print(f"Graph building complete: {len(graphs)} success, {failed} failed")

    return graphs, targets


def split_data(graphs, targets, seed, config):
    """Split data into train/val/test."""
    n = len(graphs)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    n_train = int(n * config["train_ratio"])
    n_val = int(n * config["val_ratio"])

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    train_targets = [targets[i] for i in train_idx]
    val_targets = [targets[i] for i in val_idx]
    test_targets = [targets[i] for i in test_idx]

    print(
        f"Split (seed={seed}): train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}"
    )

    return (
        train_graphs,
        val_graphs,
        test_graphs,
        train_targets,
        val_targets,
        test_targets,
    )


# ============================================================
# Training
# ============================================================

def train_one_epoch(model, loader, optimizer, device, target_mean=0, target_std=1):
    model.train()
    total_loss = 0
    num_samples = 0

    for batch_pyg, targets in loader:
        batch_pyg = batch_pyg.to(device)
        targets = targets.to(device)

        # Normalize targets
        targets_norm = (targets - target_mean) / (target_std + 1e-8)

        optimizer.zero_grad()
        preds = model(
            batch_pyg.x,
            batch_pyg.edge_index,
            batch_pyg.edge_attr,
            batch_pyg.batch,
            targets.shape[0],
        )
        # Huber loss for robustness
        loss = F.huber_loss(preds, targets_norm, delta=1.0)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Denormalize for RMSE tracking
        preds_denorm = preds * (target_std + 1e-8) + target_mean
        total_loss += F.mse_loss(preds_denorm, targets, reduction='sum').item()
        num_samples += targets.shape[0]

    return (total_loss / num_samples) ** 0.5  # RMSE


@torch.no_grad()
def evaluate(model, loader, device, target_mean=0, target_std=1):
    model.eval()
    total_loss = 0
    num_samples = 0
    all_preds = []
    all_targets = []

    for batch_pyg, targets in loader:
        batch_pyg = batch_pyg.to(device)
        targets = targets.to(device)

        preds = model(
            batch_pyg.x,
            batch_pyg.edge_index,
            batch_pyg.edge_attr,
            batch_pyg.batch,
            targets.shape[0],
        )
        # Denormalize predictions
        preds_denorm = preds * (target_std + 1e-8) + target_mean

        loss = F.mse_loss(preds_denorm, targets)

        total_loss += loss.item() * targets.shape[0]
        num_samples += targets.shape[0]

        all_preds.extend(preds_denorm.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

    rmse = (total_loss / num_samples) ** 0.5

    # MAE
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    mae = np.mean(np.abs(all_preds - all_targets))

    # R2
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return rmse, mae, r2


def train_single_seed(
    seed, graphs, targets, config, device, save_dir
):
    """Train and evaluate for a single seed."""
    print(f"\n{'='*60}")
    print(f"Training seed {seed}")
    print(f"{'='*60}")

    # Split data
    (
        train_graphs,
        val_graphs,
        test_graphs,
        train_targets,
        val_targets,
        test_targets,
    ) = split_data(graphs, targets, seed, config)

    # Compute target normalization from training set
    target_mean = np.mean(train_targets)
    target_std = np.std(train_targets)
    print(f"Target mean: {target_mean:.2f}, std: {target_std:.2f}")

    # Create dataloaders
    train_dataset = TgDataset(train_graphs, train_targets)
    val_dataset = TgDataset(val_graphs, val_targets)
    test_dataset = TgDataset(test_graphs, test_targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Determine dimensions from data
    node_dim = train_graphs[0].x.shape[1]
    edge_dim = train_graphs[0].edge_attr.shape[1]
    print(f"Node dim: {node_dim}, Edge dim: {edge_dim}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create model
    model = PolyKANSSM_v3(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        spectral_radius=config["spectral_radius"],
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config["scheduler_patience"],
        factor=config["scheduler_factor"],
        verbose=True,
        min_lr=1e-6,
    )

    # Training loop
    best_val_rmse = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_rmse": [], "val_rmse": [], "lr": []}

    for epoch in range(1, config["max_epochs"] + 1):
        train_rmse = train_one_epoch(model, train_loader, optimizer, device, target_mean, target_std)
        val_rmse, val_mae, val_r2 = evaluate(model, val_loader, device, target_mean, target_std)

        scheduler.step(val_rmse)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)
        history["lr"].append(current_lr)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"best_model_seed{seed}.pt"),
            )
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | Train RMSE: {train_rmse:.4f} | "
                f"Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f} | "
                f"Val R2: {val_r2:.4f} | LR: {current_lr:.6f} | "
                f"Best: {best_val_rmse:.4f} @ {best_epoch}"
            )

        if patience_counter >= config["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest val RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")

    # Load best model and evaluate on test set
    model.load_state_dict(
        torch.load(os.path.join(save_dir, f"best_model_seed{seed}.pt"), map_location=device)
    )
    test_rmse, test_mae, test_r2 = evaluate(model, test_loader, device, target_mean, target_std)
    train_rmse_final, _, _ = evaluate(model, train_loader, device, target_mean, target_std)
    val_rmse_final, _, _ = evaluate(model, val_loader, device, target_mean, target_std)

    print(f"\n--- Seed {seed} Final Results ---")
    print(f"Train RMSE: {train_rmse_final:.4f}")
    print(f"Val RMSE:   {val_rmse_final:.4f}")
    print(f"Test RMSE:  {test_rmse:.4f}")
    print(f"Test MAE:   {test_mae:.4f}")
    print(f"Test R2:    {test_r2:.4f}")

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "train_rmse": train_rmse_final,
        "val_rmse": val_rmse_final,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "num_params": num_params,
        "history": history,
    }


# ============================================================
# Main
# ============================================================

def main():
    config = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    checkpoint_dir = os.path.join(config["work_dir"], "checkpoints")
    diagnostics_dir = os.path.join(config["work_dir"], "diagnostics")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(diagnostics_dir, exist_ok=True)

    # Load and process data (once, shared across seeds)
    graphs, targets = load_and_process_data(config)

    # Check dimensions
    print(f"\nSample graph: nodes={graphs[0].x.shape[0]}, edges={graphs[0].edge_index.shape[1]}")
    print(f"Node features: {graphs[0].x.shape[1]}")
    print(f"Edge features: {graphs[0].edge_attr.shape[1]}")

    # Analyze non-covalent edges
    nc_counts = []
    for g in graphs[:100]:
        num_edges = g.edge_index.shape[1]
        # Count non-covalent edges (those with non-covalent flag = 1.0 in last feature)
        num_cov = 0
        num_nc = 0
        for bond_feat in g.edge_attr:
            # First 4 features are bond type one-hot
            if bond_feat[:4].sum() > 0:
                num_cov += 1
            else:
                num_nc += 1
        nc_counts.append((num_cov, num_nc))
    avg_cov = np.mean([c[0] for c in nc_counts])
    avg_nc = np.mean([c[1] for c in nc_counts])
    print(f"\nEdge statistics (first 100 graphs):")
    print(f"  Avg covalent edges: {avg_cov:.1f}")
    print(f"  Avg non-covalent edges: {avg_nc:.1f}")

    # Train with multiple seeds
    all_results = []
    for i, seed in enumerate(config["seeds"][: config["num_seeds"]]):
        print(f"\n{'#'*60}")
        print(f"# Seed {i+1}/{config['num_seeds']}: {seed}")
        print(f"{'#'*60}")

        result = train_single_seed(seed, graphs, targets, config, device, checkpoint_dir)
        all_results.append(result)

    # Aggregate results
    test_rmses = [r["test_rmse"] for r in all_results]
    test_maes = [r["test_mae"] for r in all_results]
    test_r2s = [r["test_r2"] for r in all_results]
    train_rmses = [r["train_rmse"] for r in all_results]
    val_rmses = [r["val_rmse"] for r in all_results]

    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS ({config['num_seeds']} seeds)")
    print(f"{'='*60}")
    print(f"Test RMSE: {np.mean(test_rmses):.4f} +/- {np.std(test_rmses):.4f}")
    print(f"Test MAE:  {np.mean(test_maes):.4f} +/- {np.std(test_maes):.4f}")
    print(f"Test R2:   {np.mean(test_r2s):.4f} +/- {np.std(test_r2s):.4f}")
    print(f"Train RMSE: {np.mean(train_rmses):.4f} +/- {np.std(train_rmses):.4f}")
    print(f"Val RMSE: {np.mean(val_rmses):.4f} +/- {np.std(val_rmses):.4f}")
    print(f"Train-Val gap: {np.mean(val_rmses) - np.mean(train_rmses):.4f}")
    print(f"Best single seed test RMSE: {min(test_rmses):.4f}")
    print(f"SOTA target: 30.8")

    # Save results
    results_summary = {
        "model": "PolyKAN-SSM-v3",
        "config": {k: v for k, v in config.items() if not callable(v)},
        "node_dim": graphs[0].x.shape[1],
        "edge_dim": graphs[0].edge_attr.shape[1],
        "num_samples": len(graphs),
        "mean_test_rmse": float(np.mean(test_rmses)),
        "std_test_rmse": float(np.std(test_rmses)),
        "mean_test_mae": float(np.mean(test_maes)),
        "std_test_mae": float(np.std(test_maes)),
        "mean_test_r2": float(np.mean(test_r2s)),
        "mean_train_rmse": float(np.mean(train_rmses)),
        "mean_val_rmse": float(np.mean(val_rmses)),
        "train_val_gap": float(np.mean(val_rmses) - np.mean(train_rmses)),
        "best_test_rmse": float(min(test_rmses)),
        "sota_target": 30.8,
        "beat_sota": bool(np.mean(test_rmses) < 30.8),
        "per_seed_results": [
            {
                "seed": r["seed"],
                "best_epoch": r["best_epoch"],
                "train_rmse": float(r["train_rmse"]),
                "val_rmse": float(r["val_rmse"]),
                "test_rmse": float(r["test_rmse"]),
                "test_mae": float(r["test_mae"]),
                "test_r2": float(r["test_r2"]),
                "num_params": r["num_params"],
            }
            for r in all_results
        ],
        "timestamp": datetime.now().isoformat(),
    }

    results_path = os.path.join(config["work_dir"], "results.json")
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save training curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for r in all_results:
        h = r["history"]
        axes[0].plot(h["train_rmse"], alpha=0.5, label=f"Train (seed={r['seed']})")
        axes[0].plot(h["val_rmse"], alpha=0.5, linestyle="--", label=f"Val (seed={r['seed']})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Training Curves - PolyKAN-SSM")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # Bar chart of per-seed test RMSE
    seeds = [r["seed"] for r in all_results]
    rmses = [r["test_rmse"] for r in all_results]
    axes[1].bar(range(len(seeds)), rmses, tick_label=[str(s) for s in seeds])
    axes[1].axhline(y=30.8, color="r", linestyle="--", label="SOTA (30.8)")
    axes[1].axhline(y=np.mean(rmses), color="b", linestyle="--", label=f"Mean ({np.mean(rmses):.2f})")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Test RMSE")
    axes[1].set_title("Per-Seed Test RMSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(diagnostics_dir, "training_curves.png"), dpi=150)
    plt.close()

    # Train vs Val gap analysis
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in all_results:
        h = r["history"]
        gap = [v - t for t, v in zip(h["train_rmse"], h["val_rmse"])]
        ax.plot(gap, alpha=0.5, label=f"Seed {r['seed']}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val RMSE - Train RMSE")
    ax.set_title("Overfitting Gap Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(diagnostics_dir, "overfitting_gap.png"), dpi=150)
    plt.close()

    print(f"Diagnostics saved to {diagnostics_dir}/")

    return results_summary


if __name__ == "__main__":
    from torch.nn import functional as F
    results = main()
