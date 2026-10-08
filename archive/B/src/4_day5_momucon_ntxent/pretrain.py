#!/usr/bin/env python3
"""
MoMuCon NT-Xent Pretraining on PI1M.

Trains a Siamese GAT encoder with NT-Xent contrastive loss on
100K PI1M polymers, using (1-RU, 3-RU) graph views as positive pairs.
"""

import os
import sys
import pickle
import json
import csv
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from data_utils import (
    prepare_pi1m_pretrain_data,
    TOTAL_DIM,
    EDGE_DIM,
    RWPE_K,
    PI1M_SUBSET,
)
from model import MoMuConSiamese, nt_xent_loss

warnings.filterwarnings('ignore')

WORK_DIR = "/home/home_public/aisci/AIscientist/src/4_day5_momucon_ntxent"
RESULTS_DIR = os.path.join(WORK_DIR, "results")
MODELS_DIR = os.path.join(WORK_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
class Config:
    # Data
    batch_size = 128          # B polymers → 256 graphs per batch
    train_frac = 0.95         # 95% training, 5% diagnostic held-out

    # Model
    hidden_dim = 256
    num_heads = 4
    num_layers = 4
    dropout = 0.1
    proj_hidden_dim = 128
    proj_output_dim = 128
    tau = 0.1

    # Training
    lr = 3e-4
    weight_decay = 1e-4
    epochs = 100
    warmup_epochs = 10
    grad_clip = 1.0
    seed = 42

    # Diagnostics frequency
    diag_every = 10


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
    """Linear warmup then cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return max(eta_min / optimizer.defaults['lr'],
                   0.5 * (1 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def collate_pairs(batch):
    """Collate function for (graph_1ru, graph_3ru) pairs.

    Returns:
        batch_1ru: Batched 1-RU graphs
        batch_3ru: Batched 3-RU graphs
    """
    graphs_1ru = [item[0] for item in batch]
    graphs_3ru = [item[1] for item in batch]
    return Batch.from_data_list(graphs_1ru), Batch.from_data_list(graphs_3ru)


# --------------------------------------------------------------------
# Diagnostics: collapse monitoring
# --------------------------------------------------------------------
@torch.no_grad()
def compute_embedding_stats(model, loader, device, max_batches=10):
    """Compute pairwise cosine similarity statistics for collapse monitoring.

    Returns:
        dict with:
          - mean_pairwise: mean cosine similarity among all embeddings
          - pos_sim: mean similarity of positive pairs
          - neg_sim: mean similarity of negative pairs
          - std: standard deviation of pairwise similarities
    """
    model.eval()
    all_z = []

    n_batches = 0
    for batch_1ru, batch_3ru in loader:
        batch_1ru = batch_1ru.to(device)
        batch_3ru = batch_3ru.to(device)

        # Handle edge_attr
        edge_attr_1 = batch_1ru.edge_attr if hasattr(batch_1ru, 'edge_attr') else None
        edge_attr_3 = batch_3ru.edge_attr if hasattr(batch_3ru, 'edge_attr') else None

        with torch.no_grad():
            h1 = model.encoder(batch_1ru.x, batch_1ru.edge_index, edge_attr_1, batch_1ru.batch)
            h2 = model.encoder(batch_3ru.x, batch_3ru.edge_index, edge_attr_3, batch_3ru.batch)
            z1 = model.projection(h1)
            z2 = model.projection(h2)

        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        all_z.append(z.cpu())

        n_batches += 1
        if n_batches >= max_batches:
            break

    all_z = torch.cat(all_z, dim=0)  # [N, D]
    all_z = F.normalize(all_z, dim=-1)

    B_all = all_z.shape[0] // 2

    # Pairwise similarity matrix (only first B_all*2 samples for speed if too large)
    max_n = min(all_z.shape[0], 2000)
    all_z = all_z[:max_n]
    B_eff = all_z.shape[0] // 2

    sim_matrix = all_z @ all_z.T  # [2B, 2B]

    # Positive pair similarities: (i, i+B) for i in [0, B)
    pos_sims = []
    for i in range(B_eff):
        pos_sims.append(sim_matrix[i, i + B_eff])
    pos_sims = torch.tensor(pos_sims)

    # Negative pair similarities: everything except self and positive
    mask = torch.ones_like(sim_matrix)
    for i in range(B_eff):
        mask[i, i] = 0           # self
        mask[i, i + B_eff] = 0   # positive
        mask[i + B_eff, i] = 0   # positive (reverse)
        mask[i + B_eff, i + B_eff] = 0  # self

    neg_sims = sim_matrix[mask.bool()]

    # Overall pairwise (excluding self)
    self_mask = torch.eye(sim_matrix.shape[0], device=sim_matrix.device)
    pair_sims = sim_matrix[self_mask == 0]

    return {
        'mean_pairwise': pair_sims.mean().item(),
        'std_pairwise': pair_sims.std().item(),
        'pos_sim': pos_sims.mean().item(),
        'neg_sim': neg_sims.mean().item(),
        'pos_neg_gap': (pos_sims.mean() - neg_sims.mean()).item(),
    }


# --------------------------------------------------------------------
# Main training loop
# --------------------------------------------------------------------
def main():
    config = Config()

    # Adjust batch size based on GPU memory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_mem_gb < 12:
            config.batch_size = 64
            print(f"GPU memory: {gpu_mem_gb:.1f}GB, reducing batch_size to {config.batch_size}")

    set_seed(config.seed)

    print("=" * 70)
    print("MoMuCon NT-Xent Pretraining on PI1M")
    print("=" * 70)
    print(f"Batch size: {config.batch_size} polymers ({config.batch_size * 2} graphs)")
    print(f"Epochs: {config.epochs}, LR: {config.lr}, Warmup: {config.warmup_epochs}")
    print(f"Temperature: {config.tau}")
    print(f"Hidden dim: {config.hidden_dim}, Heads: {config.num_heads}, Layers: {config.num_layers}")
    print(f"Device: {device}")

    # Load PI1M data
    pairs = prepare_pi1m_pretrain_data(force_rebuild=False)

    input_dim = pairs[0][0].x.size(1)
    edge_dim = pairs[0][0].edge_attr.size(1) if hasattr(pairs[0][0], 'edge_attr') and pairs[0][0].edge_attr is not None else None
    print(f"\nInput dim: {input_dim}, Edge dim: {edge_dim}")

    # Split: train / diagnostic
    random.seed(config.seed)
    indices = list(range(len(pairs)))
    random.shuffle(indices)
    split = int(config.train_frac * len(pairs))
    train_pairs = [pairs[i] for i in indices[:split]]
    diag_pairs = [pairs[i] for i in indices[split:split + 500]]

    print(f"Train pairs: {len(train_pairs)}, Diag pairs: {len(diag_pairs)}")

    # DataLoaders
    class PairDataset(TorchDataset):
        def __init__(self, pairs):
            self.pairs = pairs
        def __len__(self):
            return len(self.pairs)
        def __getitem__(self, idx):
            p = self.pairs[idx]
            return p[0], p[1]  # 1-RU, 3-RU

    train_ds = PairDataset(train_pairs)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
        collate_fn=collate_pairs
    )
    diag_ds = PairDataset(diag_pairs)
    diag_loader = DataLoader(
        diag_ds, batch_size=min(config.batch_size, 64), shuffle=False,
        num_workers=2, pin_memory=True,
        collate_fn=collate_pairs
    )

    # Model
    model = MoMuConSiamese(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        proj_hidden_dim=config.proj_hidden_dim,
        proj_output_dim=config.proj_output_dim,
        tau=config.tau,
        edge_dim=edge_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable params: {n_params:,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = get_warmup_cosine_scheduler(optimizer, config.warmup_epochs, config.epochs)

    # Training loop
    best_loss = float('inf')
    best_epoch = 0
    losses = []
    diag_history = []

    encoder_path = os.path.join(MODELS_DIR, "pretrained_encoder.pt")
    full_path = os.path.join(MODELS_DIR, "pretrained_full.pt")
    loss_csv_path = os.path.join(RESULTS_DIR, "pretrain_loss.csv")

    print(f"\n=== Starting Pretraining ({config.epochs} epochs) ===")

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []

        for batch_1ru, batch_3ru in tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False):
            batch_1ru = batch_1ru.to(device)
            batch_3ru = batch_3ru.to(device)

            # Handle edge_attr
            edge_attr_1 = batch_1ru.edge_attr if hasattr(batch_1ru, 'edge_attr') else None
            edge_attr_3 = batch_3ru.edge_attr if hasattr(batch_3ru, 'edge_attr') else None

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                h1 = model.encoder(batch_1ru.x, batch_1ru.edge_index, edge_attr_1, batch_1ru.batch)
                h2 = model.encoder(batch_3ru.x, batch_3ru.edge_index, edge_attr_3, batch_3ru.batch)
                z1 = model.projection(h1)
                z2 = model.projection(h2)
                loss = model.compute_loss(z1, z2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()

            epoch_losses.append(loss.item())

        scheduler.step()
        epoch_loss = np.mean(epoch_losses)
        losses.append(epoch_loss)

        # Diagnostics
        if epoch % config.diag_every == 0 or epoch == 1 or epoch == config.epochs:
            cos_stats = compute_embedding_stats(model, diag_loader, device)
            diag_history.append({
                'epoch': epoch,
                'loss': epoch_loss,
                **cos_stats,
            })

            print(f"  Epoch {epoch:3d} | Loss: {epoch_loss:.6f} | "
                  f"PosSim: {cos_stats['pos_sim']:.4f} | "
                  f"NegSim: {cos_stats['neg_sim']:.4f} | "
                  f"Gap: {cos_stats['pos_neg_gap']:.4f} | "
                  f"MeanCos: {cos_stats['mean_pairwise']:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")
        elif epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {epoch_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_epoch = epoch
            torch.save(model.encoder.state_dict(), encoder_path)
            torch.save({
                'encoder': model.encoder.state_dict(),
                'projection': model.projection.state_dict(),
                'epoch': epoch,
                'loss': epoch_loss,
            }, full_path)

    print(f"\n=== Pretraining Complete ===")
    print(f"  Best epoch: {best_epoch}, Best loss: {best_loss:.6f}")
    print(f"  Encoder saved to: {encoder_path}")
    print(f"  Full model saved to: {full_path}")

    # Save loss history
    with open(loss_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss'])
        for i, l in enumerate(losses):
            writer.writerow([i + 1, l])

    # Save diagnostics
    diag_path = os.path.join(RESULTS_DIR, "pretrain_diagnostics.json")
    with open(diag_path, 'w') as f:
        json.dump(diag_history, f, indent=2)

    # Save config
    config_path = os.path.join(RESULTS_DIR, "pretrain_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            'lr': config.lr,
            'warmup_epochs': config.warmup_epochs,
            'tau': config.tau,
            'hidden_dim': config.hidden_dim,
            'num_heads': config.num_heads,
            'num_layers': config.num_layers,
            'input_dim': input_dim,
            'edge_dim': edge_dim,
            'seed': config.seed,
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'final_loss': losses[-1] if losses else None,
            'n_trainable_params': n_params,
            'n_pairs': len(pairs),
            'n_train_pairs': len(train_pairs),
        }, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
