"""
Training pipeline for PSPE experiments.
"""

import os
import json
import time
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import r2_score, mean_absolute_error

from model import GATv2Predictor
from pspe_data import get_input_dim, EDGE_DIM_RICH, assemble_features

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    'csv_path': '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv',
    'rwpe_path': '/home/home_public/aisci/AIscientist/src/2_day_2_lspe_gat/rwpe_k32.pkl',
    'batch_size': 64,
    'max_epochs': 200,
    'patience': 30,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'hidden_dim': 256,
    'heads': 4,
    'dropout': 0.1,
    'max_grad_norm': 1.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

SEED = 42  # Model seed (data split uses seed=71 separately)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


@torch.no_grad()
def evaluate(model, loader, criterion, device, scaler, condition):
    """Evaluate model on a data loader, returning unscaled metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        x = assemble_features(batch, condition)
        out = model(x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
        all_preds.append(out.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # Compute MSE loss on scaled data
    loss = np.mean((preds - targets) ** 2)

    # Unscale for RMSE/MAE/R2
    preds_unscaled = scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
    targets_unscaled = scaler.inverse_transform(targets.reshape(-1, 1)).ravel()

    rmse = np.sqrt(np.mean((preds_unscaled - targets_unscaled) ** 2))
    mae = mean_absolute_error(targets_unscaled, preds_unscaled)
    r2 = r2_score(targets_unscaled, preds_unscaled)

    return loss, rmse, mae, r2, targets_unscaled, preds_unscaled


def train_epoch(model, loader, optimizer, criterion, device, condition, scaler, use_amp=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    scaler_amp = GradScaler(enabled=use_amp)

    for batch in loader:
        batch = batch.to(device)
        x = assemble_features(batch, condition)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            out = model(x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
            loss = criterion(out, batch.y.squeeze(-1))

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
        scaler_amp.step(optimizer)
        scaler_amp.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def run_training(
    condition: str,
    train_loader,
    val_loader,
    test_loader,
    scaler,
    save_dir: str,
):
    """Run full training pipeline for one condition.

    Returns: dict with results
    """
    set_seed(SEED)

    in_dim = get_input_dim(condition)
    device = torch.device(CONFIG['device'])

    model = GATv2Predictor(
        in_dim=in_dim,
        edge_dim=EDGE_DIM_RICH,
        hidden_dim=CONFIG['hidden_dim'],
        heads=CONFIG['heads'],
        dropout=CONFIG['dropout'],
    ).to(device)

    n_params = model.count_parameters()
    print(f"[{condition}] Model: {n_params:,} parameters, input_dim={in_dim}")

    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay'],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )
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
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, condition, scaler
        )
        val_loss, val_rmse, val_mae, val_r2, _, _ = evaluate(
            model, val_loader, criterion, device, scaler, condition
        )

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_rmse'].append(float(val_rmse))
        history['val_mae'].append(float(val_mae))
        history['val_r2'].append(float(val_r2))

        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_rmse={val_rmse:.2f}K, "
                  f"val_r2={val_r2:.4f}, lr={lr:.2e}")

        if patience_counter >= CONFIG['patience']:
            print(f"  Early stopping at epoch {epoch}")
            break

    train_time = time.time() - start_time

    # Load best model for test evaluation
    model.load_state_dict(best_state)
    test_loss, test_rmse, test_mae, test_r2, test_targets, test_preds = evaluate(
        model, test_loader, criterion, device, scaler, condition
    )

    print(f"  Best epoch: {best_epoch}")
    print(f"  Test  RMSE: {test_rmse:.2f}K")
    print(f"  Test  MAE:  {test_mae:.2f}K")
    print(f"  Test  R²:   {test_r2:.4f}")
    print(f"  Time: {train_time:.1f}s")

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"model_{condition}.pt")
    torch.save({
        'model_state_dict': best_state,
        'condition': condition,
        'in_dim': in_dim,
        'best_val_rmse': best_val_rmse,
        'best_epoch': best_epoch,
    }, model_path)

    # Get test predictions for all samples (for error analysis)
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
            # Get per-sample PSPE means
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

    # Save results
    results_path = os.path.join(save_dir, f"results_{condition}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results
