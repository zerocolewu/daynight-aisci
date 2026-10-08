"""
Training pipeline for PSPE experiments — matches Day 2's training configuration EXACTLY.

Key differences from previous PSPE round:
- CosineAnnealingLR (NOT ReduceLROnPlateau)
- AdamW with wd=1e-4 (NOT 1e-5)
- patience=50 (NOT 30)
- max_epochs=300 (NOT 200)
- GATConv with ELU (NOT GATv2Conv with ReLU)
"""

import os
import json
import time
import random
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import r2_score, mean_absolute_error

from model import GATPredictor
from pspe_data import get_input_dim, EDGE_DIM_RICH, assemble_features

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration — matches Day 2 EXACTLY
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'csv_path': os.path.join(BASE_DIR, '..', 'alldataset', 'Tg.csv'),
    'rwpe_path': os.path.join(BASE_DIR, '..', '2_day_2_lspe_gat', 'rwpe_k32.pkl'),
    'batch_size': 64,
    'max_epochs': 300,
    'patience': 50,
    'lr': 1e-3,
    'weight_decay': 1e-4,       # NOT 1e-5
    'hidden_dim': 256,
    'heads': 4,
    'dropout': 0.1,
    'max_grad_norm': 1.0,
    'use_amp': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

SEED = 42  # Model seed


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scaler, criterion, device, condition):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        x = assemble_features(batch, condition)

        optimizer.zero_grad()

        with autocast(enabled=CONFIG['use_amp']):
            out = model(x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
            loss = criterion(out, batch.y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, scaler_obj, condition):
    """Evaluate model, return scaled loss + unscaled metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        x = assemble_features(batch, condition)
        out = model(x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)

        loss = criterion(out, batch.y.view(-1))
        total_loss += loss.item()
        n_batches += 1

        all_preds.append(out.cpu())
        all_targets.append(batch.y.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy().flatten()

    # Unscale
    preds_unscaled = scaler_obj.inverse_transform(preds.reshape(-1, 1)).flatten()
    targets_unscaled = scaler_obj.inverse_transform(targets.reshape(-1, 1)).flatten()

    rmse = np.sqrt(np.mean((preds_unscaled - targets_unscaled) ** 2))
    mae = mean_absolute_error(targets_unscaled, preds_unscaled)
    r2 = r2_score(targets_unscaled, preds_unscaled)

    return total_loss / max(n_batches, 1), rmse, mae, r2, preds_unscaled, targets_unscaled


# ---------------------------------------------------------------------------
# Main training run
# ---------------------------------------------------------------------------

def run_training(
    condition: str,
    train_loader,
    val_loader,
    test_loader,
    scaler,
    save_dir: str,
):
    """Run full training pipeline for one condition."""
    set_seed(SEED)

    in_dim = get_input_dim(condition)
    device = torch.device(CONFIG['device'])

    model = GATPredictor(
        in_dim=in_dim,
        edge_dim=EDGE_DIM_RICH,
        hidden_dim=CONFIG['hidden_dim'],
        heads=CONFIG['heads'],
        dropout=CONFIG['dropout'],
    ).to(device)

    n_params = model.count_parameters()
    print(f"[{condition}] Model: {n_params:,} parameters, input_dim={in_dim}")

    # Optimizer & scheduler — matches Day 2
    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['max_epochs'])
    scaler_amp = GradScaler(enabled=CONFIG['use_amp'])
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_epoch = 0
    best_state = None
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'val_r2': [],
    }

    start_time = time.time()

    for epoch in range(1, CONFIG['max_epochs'] + 1):
        epoch_start = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, scaler_amp, criterion, device, condition
        )
        val_loss, val_rmse, val_mae, val_r2, _, _ = evaluate(
            model, val_loader, criterion, device, scaler, condition
        )

        scheduler.step()

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_rmse'].append(float(val_rmse))
        history['val_mae'].append(float(val_mae))
        history['val_r2'].append(float(val_r2))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 50 == 0 or patience_counter == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val RMSE: {val_rmse:.2f}K | Val MAE: {val_mae:.2f}K | "
                  f"Val R²: {val_r2:.4f} | LR: {lr:.2e} | "
                  f"Time: {time.time()-epoch_start:.1f}s")

        if patience_counter >= CONFIG['patience']:
            print(f"  Early stopping at epoch {epoch}")
            break

    train_time = time.time() - start_time

    # Load best model for test evaluation
    model.load_state_dict(best_state)
    test_loss, test_rmse, test_mae, test_r2, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device, scaler, condition
    )

    print(f"\n  Best epoch: {best_epoch}")
    print(f"  Test  RMSE: {test_rmse:.2f}K")
    print(f"  Test  MAE:  {test_mae:.2f}K")
    print(f"  Test  R²:   {test_r2:.4f}")
    print(f"  Time: {train_time:.1f}s")

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"model_{condition}.pt")
    torch.save({
        'model_state_dict': {k: v.cpu().clone() for k, v in best_state.items()},
        'condition': condition,
        'in_dim': in_dim,
        'best_val_rmse': best_val_rmse,
        'best_epoch': best_epoch,
    }, model_path)

    # Collect full test predictions
    model.eval()
    all_test_preds = []
    all_test_targets = []
    all_test_dce_mean = []
    all_test_mi_mean = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            x = assemble_features(batch, condition)
            out = model(x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
            all_test_preds.append(out.cpu().numpy())
            all_test_targets.append(batch.y.cpu().numpy())
            for i in range(batch.num_graphs):
                node_mask = (batch.batch == i)
                all_test_dce_mean.append(batch.dce[node_mask].mean().item())
                all_test_mi_mean.append(batch.mi[node_mask].mean().item())

    test_preds_full = np.concatenate(all_test_preds)
    test_targets_full = np.concatenate(all_test_targets)
    test_preds_unscaled = scaler.inverse_transform(test_preds_full.reshape(-1, 1)).ravel()
    test_targets_unscaled = scaler.inverse_transform(test_targets_full.reshape(-1, 1)).ravel()

    results = {
        'condition': condition,
        'seed': SEED,
        'in_dim': in_dim,
        'n_params': n_params,
        'best_epoch': best_epoch,
        'best_val_rmse': float(best_val_rmse),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'train_time': train_time,
        'history': history,
        'test_targets': test_targets_unscaled.tolist(),
        'test_predictions': test_preds_unscaled.tolist(),
        'test_dce_mean': all_test_dce_mean,
        'test_mi_mean': all_test_mi_mean,
    }

    results_path = os.path.join(save_dir, f"results_{condition}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results
