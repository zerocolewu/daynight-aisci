#!/usr/bin/env python3
"""
SelfGNN Pretraining on PI1M for Polymer Tg Prediction.

Trains a Siamese GAT with Split+Standardize augmentations on 200K PI1M polymers.
Saves the best student encoder weights for downstream finetuning.
"""

import os
import sys
import pickle
import json
import csv
import argparse
import random
import warnings
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.selfgnn import SelfGNN
from utils.diagnostics import (
    compute_embedding_stats,
    compute_nearest_neighbor_consistency,
    plot_pretrain_loss,
    plot_embedding_diagnostics,
)

warnings.filterwarnings('ignore')


# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
WORK_DIR = "/home/home_public/aisci/AIscientist/src/4_day4_selfgnn_polymer"
PI1M_PKL = os.path.join(WORK_DIR, "data", "pi1m_pretrain_cache", "pi1m_200k_1ru_10dim.pkl")
RESULTS_DIR = os.path.join(WORK_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_standardize', action='store_true', default=True,
                        help='Use Standardize augmentation')
    parser.add_argument('--use_split', action='store_true', default=True,
                        help='Use Split augmentation')
    parser.add_argument('--no_standardize', action='store_true', default=False,
                        help='Disable Standardize augmentation')
    parser.add_argument('--no_split', action='store_true', default=False,
                        help='Disable Split augmentation')
    parser.add_argument('--use_batchnorm', action='store_true', default=True,
                        help='Use BatchNorm in encoder')
    parser.add_argument('--no_batchnorm', action='store_true', default=False,
                        help='Remove BatchNorm from encoder (Ablation H)')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--output_name', type=str, default='selfgnn_pretrain',
                        help='Output filename prefix')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


class PI1MPretrainDataset(TorchDataset):
    """Dataset wrapper for PI1M pretraining graphs."""
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def collate_fn(batch):
    """Custom collate for PyG Data objects."""
    return Batch.from_data_list(batch)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_diagnostics(model, loader, device, num_samples=100):
    """Run embedding quality diagnostics."""
    model.eval()
    all_z_student = []
    all_z_teacher = []

    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if batch.batch.max() + 1 < 2 and count > 0:
                continue  # skip single-graph batches

            z_student, z_teacher = model(batch)
            all_z_student.append(F.normalize(z_student, dim=-1))
            all_z_teacher.append(F.normalize(z_teacher, dim=-1))
            count += batch.batch.max().item() + 1
            if count >= num_samples:
                break

    z_s = torch.cat(all_z_student, dim=0)
    z_t = torch.cat(all_z_teacher, dim=0)

    stats = compute_embedding_stats(z_s)
    nn_cons = compute_nearest_neighbor_consistency(z_s[:num_samples], z_t[:num_samples])

    return stats, nn_cons


def main():
    args = get_args()

    # Handle negation flags
    if args.no_standardize:
        args.use_standardize = False
    if args.no_split:
        args.use_split = False
    if args.no_batchnorm:
        args.use_batchnorm = False

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Print config
    print("\n=== SelfGNN Pretraining Configuration ===")
    print(f"  Augmentations: Split={args.use_split}, Standardize={args.use_standardize}")
    print(f"  BatchNorm: {args.use_batchnorm}")
    print(f"  LR: {args.lr}, Epochs: {args.epochs}, Batch: {args.batch_size}")
    print(f"  Patience: {args.patience}, Seed: {args.seed}")

    # Load data
    print(f"\nLoading PI1M pretraining data from: {PI1M_PKL}")
    with open(PI1M_PKL, 'rb') as f:
        graphs = pickle.load(f)
    print(f"Loaded {len(graphs)} graphs, feature dim: {graphs[0].x.size(1)}")

    input_dim = graphs[0].x.size(1)

    # Split: 95% train, 5% held-out for diagnostics
    random.shuffle(graphs)
    split_idx = int(0.95 * len(graphs))
    train_graphs = graphs[:split_idx]
    diag_graphs = graphs[split_idx:split_idx + 500]
    print(f"Train: {len(train_graphs)}, Diag: {len(diag_graphs)}")

    # DataLoader
    train_ds = PI1MPretrainDataset(train_graphs)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    diag_ds = PI1MPretrainDataset(diag_graphs)
    diag_loader = DataLoader(
        diag_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # Model
    model = SelfGNN(
        input_dim=input_dim,
        hidden_dim=256,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        tau=0.999,
        use_batchnorm=args.use_batchnorm,
        use_standardize=args.use_standardize,
        use_split=args.use_split,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable params: {n_params:,}")

    # Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    losses = []
    diag_history = []

    checkpoint_path = os.path.join(RESULTS_DIR, f"{args.output_name}_encoder.pt")
    loss_csv_path = os.path.join(RESULTS_DIR, f"{args.output_name}_loss.csv")

    print(f"\n=== Starting Pretraining ({args.epochs} epochs) ===")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            batch = batch.to(device)

            optimizer.zero_grad()

            z_student, z_teacher = model(batch)
            loss = model.compute_loss(z_student, z_teacher)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # EMA update teacher
            model.update_teacher()

            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        epoch_loss = np.mean(epoch_losses)
        losses.append(epoch_loss)

        # Diagnostics every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            cos_stats, nn_cons = run_diagnostics(model, diag_loader, device)
            diag_history.append({
                'epoch': epoch,
                'loss': epoch_loss,
                'cos_sim_mean': cos_stats['mean'],
                'cos_sim_std': cos_stats['std'],
                'cos_sim_min': cos_stats['min'],
                'cos_sim_max': cos_stats['max'],
                'nn_consistency': nn_cons,
            })

            print(f"  Epoch {epoch:3d} | Loss: {epoch_loss:.6f} | "
                  f"CosSim: {cos_stats['mean']:.4f}±{cos_stats['std']:.4f} "
                  f"[{cos_stats['min']:.4f}, {cos_stats['max']:.4f}] | "
                  f"NN Cons: {nn_cons:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

        else:
            if epoch % 5 == 0:
                print(f"  Epoch {epoch:3d} | Loss: {epoch_loss:.6f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Early stopping
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.get_student_encoder().state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    print(f"\n=== Pretraining Complete ===")
    print(f"  Best epoch: {best_epoch}, Best loss: {best_loss:.6f}")
    print(f"  Model saved: {checkpoint_path}")

    # Save loss CSV
    with open(loss_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss'])
        for i, l in enumerate(losses):
            writer.writerow([i + 1, l])

    # Save diagnostics
    diag_path = os.path.join(RESULTS_DIR, f"{args.output_name}_diagnostics.json")
    with open(diag_path, 'w') as f:
        json.dump(diag_history, f, indent=2)

    # Plot
    plot_pretrain_loss(losses, os.path.join(RESULTS_DIR, f"{args.output_name}_loss.png"))
    if diag_history:
        plot_embedding_diagnostics(diag_history, os.path.join(RESULTS_DIR, f"{args.output_name}_diagnostics.png"))

    print(f"\n  Loss plot: {RESULTS_DIR}/{args.output_name}_loss.png")
    if diag_history:
        print(f"  Diag plot: {RESULTS_DIR}/{args.output_name}_diagnostics.png")


if __name__ == "__main__":
    main()
