#!/usr/bin/env python3
"""
MoMuCon-NT-Xent Finetuning on Tg Dataset.

Loads pretrained GAT encoder from src/5_day5_momucon_ntxent/checkpoints/best_pretrain.pt,
adds fresh predictor head, finetunes on Tg dataset with 3-RU graphs.

Architecture:
  - Input: rich features (8) + PSPE (2) + RWPE (32) = 42 dims
  - GAT encoder: 4 layers, 256 hidden, 4 heads, mean pool
  - Predictor: Linear(256->128) + BN + ReLU + Dropout(0.1) + Linear(128->1)
  - AdamW lr=1e-4, weight_decay=1e-4
  - ReduceLROnPlateau patience=10 factor=0.5
  - Early stop patience=30, max 300 epochs, batch 64
  - 72/8/20 split, seed=71
"""

import os
import sys
import json
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models import GATEncoder, Predictor
from dataset import (
    prepare_tg_data,
    TOTAL_NODE_DIM,
    EDGE_DIM,
    TgDataset,
)
from utils import plot_finetune_curve

warnings.filterwarnings('ignore')

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORK_DIR, 'results')
CHECKPOINTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '5_day5_momucon_ntxent', 'checkpoints'
)
PRETRAINED_PATH = os.path.join(CHECKPOINTS_DIR, 'best_pretrain.pt')

os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------
# Training helpers
# --------------------------------------------------------------------
def train_epoch(model, loader, optimizer, amp_scaler, device):
    """One training epoch with AMP."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for data in tqdm(loader, desc="Train", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        with amp.autocast():
            pred = model(data)
            loss = nn.MSELoss()(pred, data.y.squeeze())
        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        amp_scaler.step(optimizer)
        amp_scaler.update()

        total_loss += loss.item()
        all_preds.append(pred.detach().cpu())
        all_targets.append(data.y.squeeze().detach().cpu())

    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    rmse = float(torch.sqrt(nn.MSELoss()(all_preds, all_targets)).item())
    return avg_loss, rmse


@torch.no_grad()
def eval_epoch(model, loader, device):
    """One evaluation epoch."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for data in tqdm(loader, desc="Eval", leave=False):
        data = data.to(device)
        with amp.autocast():
            pred = model(data)
        loss = nn.MSELoss()(pred, data.y.squeeze())
        total_loss += loss.item()
        all_preds.append(pred.cpu())
        all_targets.append(data.y.squeeze().cpu())

    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    rmse = float(torch.sqrt(nn.MSELoss()(all_preds, all_targets)).item())
    return avg_loss, rmse


# --------------------------------------------------------------------
# Model wrapper
# --------------------------------------------------------------------
class FinetuneModel(nn.Module):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, data):
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        h = self.encoder(data.x, data.edge_index, edge_attr, data.batch)
        return self.predictor(h)


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    seed = 42
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Config
    lr = 1e-4
    weight_decay = 1e-4
    batch_size = 64
    max_epochs = 300
    patience = 30
    plateau_patience = 10
    plateau_factor = 0.5
    hidden_dim = 256
    num_layers = 4
    heads = 4
    dropout = 0.1
    hidden_dim_predictor = 128

    print("=" * 70)
    print("MoMuCon-NT-Xent Finetuning on Tg")
    print("=" * 70)
    print(f"Pretrained: {PRETRAINED_PATH}")
    print(f"LR: {lr}, WD: {weight_decay}, Batch: {batch_size}")
    print(f"Max epochs: {max_epochs}, Patience: {patience}")
    print(f"GAT: {num_layers} layers, {hidden_dim} hidden, {heads} heads")
    print(f"Predictor: {hidden_dim}->{hidden_dim_predictor}->1, dropout={dropout}")

    # Load data
    print("\n[1/4] Loading Tg data...")
    tg_data = prepare_tg_data(cache_dir=WORK_DIR, force_rebuild=False)
    train_graphs = tg_data['train_graphs']
    val_graphs = tg_data['val_graphs']
    test_graphs = tg_data['test_graphs']
    scaler = tg_data['scaler']
    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    # Data dimensions
    input_dim = train_graphs[0].x.size(1)
    edge_dim = train_graphs[0].edge_attr.size(1)
    print(f"  Node dim: {input_dim}, Edge dim: {edge_dim}")

    # DataLoaders
    train_loader = DataLoader(
        TgDataset(train_graphs), batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        TgDataset(val_graphs), batch_size=batch_size, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        TgDataset(test_graphs), batch_size=batch_size, shuffle=False, pin_memory=True
    )

    # Load pretrained encoder
    print("\n[2/4] Loading pretrained encoder...")
    encoder = GATEncoder(
        in_dim=input_dim, hidden=hidden_dim, num_layers=num_layers, heads=heads,
        dropout=dropout, edge_dim=edge_dim,
    ).to(device)

    if os.path.exists(PRETRAINED_PATH):
        ckpt = torch.load(PRETRAINED_PATH, map_location=device)
        encoder.load_state_dict(ckpt['encoder'])
        print(f"  Loaded from: {PRETRAINED_PATH}")
        print(f"  Pretrained epoch: {ckpt.get('epoch', '?')}")
        print(f"  Pretrained loss: {ckpt.get('loss', '?'):.4f}")
    else:
        raise FileNotFoundError(f"Pretrained checkpoint not found: {PRETRAINED_PATH}")

    # Fresh predictor
    predictor = Predictor(in_dim=hidden_dim, hidden_dim=hidden_dim_predictor, dropout=dropout).to(device)

    model = FinetuneModel(encoder, predictor).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # Optimizer & scheduler
    print("\n[3/4] Setting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=plateau_factor,
        patience=plateau_patience, min_lr=1e-6
    )
    amp_scaler = amp.GradScaler()

    # Training
    print("\n[4/4] Training...")
    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    val_rmse_history = []
    train_rmse_history = []

    for epoch in range(max_epochs):
        train_loss, train_rmse = train_epoch(
            model, train_loader, optimizer, amp_scaler, device
        )
        train_rmse_history.append(train_rmse)

        val_loss, val_rmse = eval_epoch(model, val_loader, device)
        val_rmse_history.append(val_rmse)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_rmse)

        if (epoch + 1) % 5 == 0 or val_rmse < best_val_rmse:
            print(f"  Epoch {epoch+1:3d}: train_rmse={train_rmse:.4f}, "
                  f"val_rmse={val_rmse:.4f}, lr={current_lr:.2e}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'encoder': encoder.state_dict(),
                'predictor': predictor.state_dict(),
                'best_val_rmse': best_val_rmse,
            }, os.path.join(RESULTS_DIR, 'best_finetune.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch + 1} "
                      f"(patience={patience})")
                break

    # Evaluate on test set
    print(f"\n[Test] Loading best model (epoch {best_epoch}, "
          f"val_rmse={best_val_rmse:.4f})...")
    best_ckpt = torch.load(
        os.path.join(RESULTS_DIR, 'best_finetune.pt'), map_location=device
    )
    encoder.load_state_dict(best_ckpt['encoder'])
    predictor.load_state_dict(best_ckpt['predictor'])
    model = FinetuneModel(encoder, predictor).to(device)

    model.eval()
    all_preds = []
    all_targets = []
    all_targets_orig = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            data = data.to(device)
            with amp.autocast():
                pred = model(data)
            all_preds.append(pred.cpu())
            all_targets.append(data.y.cpu())
            if hasattr(data, 'y_original'):
                all_targets_orig.append(data.y_original.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    if all_targets_orig:
        all_targets_orig = torch.cat(all_targets_orig)

    # RMSE in normalized space and K units
    norm_rmse = float(torch.sqrt(
        nn.MSELoss()(all_preds, all_targets.squeeze())).item()
    )
    if scaler is not None and len(all_targets_orig) > 0:
        preds_k = scaler.inverse_transform(all_preds.numpy().reshape(-1, 1))
        targets_k = all_targets_orig.numpy().reshape(-1, 1)
        test_rmse_k = float(np.sqrt(np.mean((preds_k - targets_k) ** 2)))
    else:
        test_rmse_k = norm_rmse

    baseline = 35.44  # K
    delta = baseline - test_rmse_k

    print("\n" + "=" * 70)
    print("MoMuCon-NT-Xent Finetuning Results")
    print("=" * 70)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation RMSE (norm): {best_val_rmse:.4f}")
    print(f"Test RMSE (normalized): {norm_rmse:.4f}")
    print(f"Test RMSE: {test_rmse_k:.2f} K")
    print(f"Baseline: {baseline:.2f} K")
    print(f"Improvement vs baseline: {delta:+.2f} K")
    print()

    # Required comparison format
    print(f"MoMuCon-NT-Xent: {test_rmse_k:.2f} K -> "
          f"\u0394 = {delta:+.2f} K")
    print()

    # Save results
    results = {
        'best_epoch': best_epoch,
        'best_val_rmse_norm': best_val_rmse,
        'test_rmse_norm': norm_rmse,
        'test_rmse_k': test_rmse_k,
        'baseline_k': baseline,
        'delta_k': delta,
        'val_rmse_history': val_rmse_history,
        'train_rmse_history': train_rmse_history,
        'pretrained_epoch': ckpt.get('epoch', 30),
        'pretrained_loss': ckpt.get('loss', 0.4641),
    }
    with open(os.path.join(RESULTS_DIR, 'finetune_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_DIR}/finetune_results.json")

    # Plot
    plot_finetune_curve(val_rmse_history, RESULTS_DIR)

    return test_rmse_k, delta


if __name__ == '__main__':
    main()
