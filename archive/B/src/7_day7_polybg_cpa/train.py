"""
Unified training script for all PolyBG-CPA experiments.
Handles data loading, training, and evaluation.
"""

import os
import sys
import json
import time
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    RESULTS_DIR, BATCH_SIZE, MAX_EPOCHS,
    EARLY_STOP_PATIENCE, LEARNING_RATE, WEIGHT_DECAY, ETA_MIN, GRAD_CLIP,
)
from data.tg_dataset import BondGraphDataset, collate_bond_batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, device, use_bond_graph=False):
    model.train()
    total_loss = 0.0
    num_samples = 0
    for batch_data in loader:
        if use_bond_graph:
            atom_batch, bond_data = batch_data
            atom_batch = atom_batch.to(device)
            target = atom_batch.y.view(-1)
        else:
            atom_batch = batch_data.to(device)
            bond_data = None
            target = atom_batch.y.view(-1)
        optimizer.zero_grad()
        pred = model(atom_batch, bond_data=bond_data)
        loss = criterion(pred, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        n = atom_batch.num_graphs if hasattr(atom_batch, 'num_graphs') else len(bond_data or [1])
        total_loss += loss.item() * n
        num_samples += n
    return total_loss / max(num_samples, 1)


def evaluate(model, loader, criterion, device, scaler=None, use_bond_graph=False):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_data in loader:
            if use_bond_graph:
                atom_batch, bond_data = batch_data
                atom_batch = atom_batch.to(device)
                target = atom_batch.y.view(-1)
            else:
                atom_batch = batch_data.to(device)
                bond_data = None
                target = atom_batch.y.view(-1)
            pred = model(atom_batch, bond_data=bond_data)
            loss = criterion(pred, target)
            n = atom_batch.num_graphs if hasattr(atom_batch, 'num_graphs') else len(bond_data or [1])
            total_loss += loss.item() * n
            num_samples += n
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
    avg_loss = total_loss / max(num_samples, 1)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    if scaler is not None:
        all_preds = all_preds * scaler.scale_[0] + scaler.mean_[0]
        all_targets = all_targets * scaler.scale_[0] + scaler.mean_[0]
    mse = ((all_preds - all_targets) ** 2).mean().item()
    rmse = np.sqrt(mse)
    mae = (all_preds - all_targets).abs().mean().item()
    return avg_loss, rmse, mae, all_preds, all_targets


def train_model(model, model_name, seed,
                train_graphs, val_graphs, test_graphs, scaler,
                train_bond_data=None, val_bond_data=None, test_bond_data=None,
                num_epochs=MAX_EPOCHS):
    """Train a model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    print(f"\n{'='*60}")
    print(f"Training {model_name} | Seed {seed} | Device: {device}")
    print(f"{'='*60}")

    use_bond = train_bond_data is not None

    if use_bond:
        train_ds = BondGraphDataset(train_graphs, train_bond_data)
        val_ds = BondGraphDataset(val_graphs, val_bond_data)
        test_ds = BondGraphDataset(test_graphs, test_bond_data)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_bond_batch)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_bond_batch)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_bond_batch)
    else:
        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}, Params: {num_params:,}")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=ETA_MIN)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = []
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                 use_bond_graph=use_bond)
        val_loss, val_rmse, val_mae, _, _ = evaluate(
            model, val_loader, criterion, device, scaler, use_bond_graph=use_bond)
        current_lr = optimizer.param_groups[0]['lr']
        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'val_rmse': val_rmse, 'val_mae': val_mae, 'lr': current_lr,
        })
        scheduler.step()
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch % 20 == 1 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val RMSE: {val_rmse:.2f} K | Best: {best_val_rmse:.2f} K (ep {best_epoch}) | LR: {current_lr:.6f}")
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    train_time = time.time() - start_time
    model.load_state_dict(best_state)
    model.to(device)
    test_loss, test_rmse, test_mae, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device, scaler, use_bond_graph=use_bond)

    print(f"\nFinal Results for {model_name} Seed {seed}:")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Val RMSE: {best_val_rmse:.4f} K")
    print(f"  Test RMSE: {test_rmse:.4f} K")
    print(f"  Test MAE: {test_mae:.4f} K")
    print(f"  Training Time: {train_time:.1f}s")

    result = {
        'model_name': model_name, 'seed': seed, 'best_epoch': best_epoch,
        'best_val_rmse': float(best_val_rmse), 'test_rmse': float(test_rmse),
        'test_mae': float(test_mae), 'train_time': train_time, 'num_params': num_params,
        'test_preds': test_preds.numpy().tolist(),
        'test_targets': test_targets.numpy().tolist(),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{model_name}_seed{seed}.json"), 'w') as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(RESULTS_DIR, f"{model_name}_seed{seed}_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    torch.save(best_state, os.path.join(RESULTS_DIR, f"{model_name}_seed{seed}.pt"))
    return result
