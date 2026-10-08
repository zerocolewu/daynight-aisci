"""
PPFN Training Script: Full training pipeline with early stopping.
"""

import sys
import os
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/1_day2_ppfn')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import json
from datetime import datetime

from dataset import PPFNDataset, collate_fn
from model import PPFN

torch.manual_seed(42)
np.random.seed(42)

CSV_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
SAVE_DIR = '/home/home_public/aisci/AIscientist/src/1_day2_ppfn'
BATCH_SIZE = 64  # Increased from 32 for more stable gradients
MAX_EPOCHS = 500
LR = 1e-3
WEIGHT_DECAY = 1e-4  # Increased from 1e-5 for better regularization
ES_PATIENCE = 50  # Reduced from 30 to allow more exploration


def move_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def compute_metrics(preds, targets):
    p = preds.cpu().numpy().flatten()
    t = targets.cpu().numpy().flatten()
    rmse = np.sqrt(np.mean((p - t) ** 2))
    mae = np.mean(np.abs(p - t))
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return rmse, mae, r2


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad()
        pred = model(batch)
        target = batch['tg']
        loss = criterion(pred.squeeze(-1), target.squeeze(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * target.shape[0]
        all_preds.append(pred.detach())
        all_targets.append(target.detach())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    rmse, mae, r2 = compute_metrics(all_preds, all_targets)
    return total_loss / len(all_preds), rmse, mae, r2


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch)
        target = batch['tg']
        loss = criterion(pred.squeeze(-1), target.squeeze(-1))
        total_loss += loss.item() * target.shape[0]
        all_preds.append(pred)
        all_targets.append(target)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    rmse, mae, r2 = compute_metrics(all_preds, all_targets)
    return total_loss / len(all_preds), rmse, mae, r2


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading datasets...")
    train_ds = PPFNDataset(CSV_PATH, split='train')
    val_ds = PPFNDataset(CSV_PATH, split='val')
    test_ds = PPFNDataset(CSV_PATH, split='test')
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    model = PPFN(hidden_dim=256).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Use Huber Loss (SmoothL1Loss) for robustness to outliers
    criterion = nn.SmoothL1Loss(beta=10.0)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Use CosineAnnealingWarmRestarts for better learning rate scheduling
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

    best_val_rmse = float('inf')
    best_epoch = 0
    no_improve = 0
    log_path = os.path.join(SAVE_DIR, 'training_log.txt')

    with open(log_path, 'w') as log:
        log.write(f"PPFN Training Log - {datetime.now()}\n")
        log.write(f"Params: {n_params}, LR: {LR}, BS: {BATCH_SIZE}\n\n")

        for epoch in range(1, MAX_EPOCHS + 1):
            t_loss, t_rmse, t_mae, t_r2 = train_epoch(
                model, train_loader, criterion, optimizer, device)
            v_loss, v_rmse, v_mae, v_r2 = eval_epoch(
                model, val_loader, criterion, device)

            # Step scheduler every epoch (not based on validation loss)
            scheduler.step()

            entry = (f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
                     f"Train RMSE: {t_rmse:.4f} MAE: {t_mae:.4f} R2: {t_r2:.4f} | "
                     f"Val RMSE: {v_rmse:.4f} MAE: {v_mae:.4f} R2: {v_r2:.4f} | "
                     f"Best Val RMSE: {best_val_rmse:.4f}\n")
            log.write(entry)
            log.flush()

            if epoch % 10 == 0 or epoch == 1:
                print(entry.strip())

            if v_rmse < best_val_rmse:
                best_val_rmse = v_rmse
                best_epoch = epoch
                no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_rmse': v_rmse,
                    'val_mae': v_mae,
                    'val_r2': v_r2,
                }, os.path.join(SAVE_DIR, 'best_model.pt'))
            else:
                no_improve += 1

            if no_improve >= ES_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nBest epoch: {best_epoch}, Best val RMSE: {best_val_rmse:.4f}")

    ckpt = torch.load(os.path.join(SAVE_DIR, 'best_model.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    _, train_rmse, train_mae, train_r2 = eval_epoch(model, train_loader, criterion, device)
    _, val_rmse, val_mae, val_r2 = eval_epoch(model, val_loader, criterion, device)
    _, test_rmse, test_mae, test_r2 = eval_epoch(model, test_loader, criterion, device)

    print(f"\nFinal Results:")
    print(f"  Train: RMSE={train_rmse:.4f} MAE={train_mae:.4f} R2={train_r2:.4f}")
    print(f"  Val:   RMSE={val_rmse:.4f} MAE={val_mae:.4f} R2={val_r2:.4f}")
    print(f"  Test:  RMSE={test_rmse:.4f} MAE={test_mae:.4f} R2={test_r2:.4f}")

    results = {
        'best_epoch': best_epoch,
        'train': {'rmse': float(train_rmse), 'mae': float(train_mae), 'r2': float(train_r2)},
        'val': {'rmse': float(val_rmse), 'mae': float(val_mae), 'r2': float(val_r2)},
        'test': {'rmse': float(test_rmse), 'mae': float(test_mae), 'r2': float(test_r2)},
        'n_params': n_params,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(SAVE_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {SAVE_DIR}/results.json")


if __name__ == '__main__':
    main()
