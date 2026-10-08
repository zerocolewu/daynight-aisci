#!/usr/bin/env python
"""
Treatment B v2 (Iteration): Residual MaxGAT + Rich features + 3-RU + SPSE.
Adds residual connections to MaxGATConv to prevent variance collapse.
Learnable gate per layer controls residual vs max-aggregation blend.
"""

from __future__ import annotations

import os
import sys
import time
import random
import json

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(WORK_DIR)
sys.path.insert(0, PARENT_DIR)

from data_utils import (
    build_datasets, create_dataloaders,
    NODE_DIM_RICH, EDGE_DIM_RICH, EDGE_DIM_SPSE,
)
from model import GATResidualMax
from spse_encoding import SPSEEncoder, precompute_spse_for_graphs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DGX_CSV = os.path.join(os.path.dirname(PARENT_DIR), 'alldataset', 'Tg.csv')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_EPOCHS = 300
PATIENCE = 30
LR = 5e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
AMP_ENABLED = True
SEED = 71

RESULTS_DIR = os.path.join(WORK_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

LOG_FILE = os.path.join(RESULTS_DIR, 'train_log.txt')
CURVES_PNG = os.path.join(RESULTS_DIR, 'learning_curves.png')
CHECKPOINT = os.path.join(RESULTS_DIR, 'best_model.pt')
RESULTS_JSON = os.path.join(RESULTS_DIR, 'results.json')
SPSE_PARAMS_TXT = os.path.join(RESULTS_DIR, 'spse_params.txt')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, spse_encoder, loader, optimizer, scaler, criterion, device):
    model.train()
    spse_encoder.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        edge_attr = getattr(batch, 'edge_attr', None)
        spse_raw = getattr(batch, 'spse_raw', None)
        if edge_attr is not None and spse_raw is not None:
            spse_encoded = spse_encoder(spse_raw.to(device))
            edge_attr = torch.cat([edge_attr, spse_encoded], dim=1)

        if AMP_ENABLED:
            with autocast('cuda'):
                out = model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)
                loss = criterion(out, batch.y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)
            loss = criterion(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, spse_encoder, loader, criterion, device):
    model.eval()
    spse_encoder.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in loader:
        batch = batch.to(device)
        edge_attr = getattr(batch, 'edge_attr', None)
        spse_raw = getattr(batch, 'spse_raw', None)
        if edge_attr is not None and spse_raw is not None:
            spse_encoded = spse_encoder(spse_raw.to(device))
            edge_attr = torch.cat([edge_attr, spse_encoded], dim=1)
        out = model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)
        loss = criterion(out, batch.y)
        total_loss += loss.item() * batch.num_graphs
        all_preds.append(out.cpu())
        all_targets.append(batch.y.cpu())

    mse = total_loss / len(loader.dataset)
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    return mse, np.sqrt(max(mse, 0)), preds, targets


def main():
    set_seed(SEED)

    with open(LOG_FILE, 'w') as log_fp:
        log_fp.write(f"Treatment B v2: Residual MaxGAT + Rich + 3-RU + SPSE\n")
        log_fp.write(f"Device: {DEVICE} | Seed: {SEED}\n")
        log_fp.write(f"LR: {LR}, WD: {WEIGHT_DECAY}, Batch: {BATCH_SIZE}\n")
        log_fp.write("=" * 80 + "\n")
        log_fp.flush()

        print("Building datasets...")
        t0 = time.time()
        train_graphs, val_graphs, test_graphs, scaler = build_datasets(DGX_CSV, seed=SEED)
        dt = time.time() - t0
        print(f"  Built: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)} ({dt:.0f}s)")

        print("Precomputing SPSE...")
        t0_spse = time.time()
        precompute_spse_for_graphs(train_graphs, encoder=None, max_k=4)
        precompute_spse_for_graphs(val_graphs, encoder=None, max_k=4)
        precompute_spse_for_graphs(test_graphs, encoder=None, max_k=4)
        print(f"  Done: {time.time()-t0_spse:.0f}s")

        train_loader, val_loader, test_loader = create_dataloaders(
            train_graphs, val_graphs, test_graphs, scaler, BATCH_SIZE
        )

        spse_encoder = SPSEEncoder(embed_dim=4).to(DEVICE)
        model = GATResidualMax(
            node_dim=NODE_DIM_RICH,
            edge_dim=EDGE_DIM_RICH + EDGE_DIM_SPSE,
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model params: {n_params:,} + SPSE: 8")
        log_fp.write(f"  Model params: {n_params:,} + SPSE: 8\n")
        log_fp.flush()

        all_params = list(model.parameters()) + list(spse_encoder.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        criterion = nn.MSELoss()
        scaler_amp = GradScaler('cuda') if AMP_ENABLED else None

        best_val_rmse = float('inf')
        best_epoch = 0
        best_state = None
        best_spse_state = None
        train_rmse_hist = []
        val_rmse_hist = []

        t0_train = time.time()

        for epoch in range(1, MAX_EPOCHS + 1):
            train_mse = train_epoch(model, spse_encoder, train_loader, optimizer, scaler_amp, criterion, DEVICE)
            val_mse, val_rmse_raw, _, _ = evaluate(model, spse_encoder, val_loader, criterion, DEVICE)

            train_rmse = np.sqrt(max(train_mse, 0))
            train_rmse_hist.append(float(train_rmse))
            val_rmse_hist.append(float(val_rmse_raw))

            scheduler.step(val_rmse_raw)

            if val_rmse_raw < best_val_rmse:
                best_val_rmse = val_rmse_raw
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_spse_state = {k: v.cpu().clone() for k, v in spse_encoder.state_dict().items()}
                waited = 0
            else:
                waited = epoch - best_epoch

            if epoch % 10 == 0 or waited >= PATIENCE:
                elapsed = time.time() - t0_train
                info = (f"  epoch {epoch:3d}/{MAX_EPOCHS} | "
                        f"train={train_rmse:.4f} | val={val_rmse_raw:.4f} | "
                        f"best={best_val_rmse:.4f}@{best_epoch} | "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                        f"waited={waited} | time={elapsed:.0f}s")
                print(info)
                log_fp.write(info + '\n')
                # Print gate values
                gates = []
                for name, p in model.named_parameters():
                    if 'residual_gate' in name:
                        gates.append(f"{torch.sigmoid(p).item():.3f}")
                log_fp.write(f"    gates={gates}\n")
                log_fp.flush()

            if waited >= PATIENCE:
                break

        # Restore best model
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        spse_encoder.load_state_dict({k: v.to(DEVICE) for k, v in best_spse_state.items()})
        torch.save({'model': best_state, 'spse_encoder': best_spse_state}, CHECKPOINT)

        # Evaluate
        _, train_rmse, _, _ = evaluate(model, spse_encoder, train_loader, criterion, DEVICE)
        _, val_rmse, _, _ = evaluate(model, spse_encoder, val_loader, criterion, DEVICE)
        _, test_rmse, test_preds, test_targets = evaluate(model, spse_encoder, test_loader, criterion, DEVICE)

        scale = scaler.scale_[0]
        mean_val = scaler.mean_[0]
        train_rmse_orig = float(train_rmse * scale)
        val_rmse_orig = float(val_rmse * scale)
        test_rmse_orig = float(test_rmse * scale)
        test_preds_orig = test_preds * scale + mean_val
        test_targets_orig = test_targets * scale + mean_val
        test_r2 = float(r2_score(test_targets_orig, test_preds_orig))
        test_mae = float(mean_absolute_error(test_targets_orig, test_preds_orig))
        total_time = time.time() - t0_train

        # Final gate values
        gate_values = {}
        for name, p in model.named_parameters():
            if 'residual_gate' in name:
                gate_values[name] = torch.sigmoid(p).item()

        alpha, beta = spse_encoder.get_params()

        msg = (
            f"\n=== Final Results (Treatment B v2: Residual MaxGAT) ===\n"
            f"  Best epoch: {best_epoch} / {epoch}\n"
            f"  Train RMSE: {train_rmse_orig:.3f} K\n"
            f"  Val RMSE:   {val_rmse_orig:.3f} K\n"
            f"  Test RMSE:  {test_rmse_orig:.3f} K\n"
            f"  Test MAE:   {test_mae:.3f} K\n"
            f"  Test R²:    {test_r2:.4f}\n"
            f"  Total time: {total_time:.0f}s\n"
            f"  Gates: {gate_values}\n"
        )
        print(msg)
        log_fp.write(msg)

        results = {
            'model': 'Treatment B v2 (Residual MaxGAT + Rich + 3-RU + SPSE)',
            'train_rmse': train_rmse_orig,
            'val_rmse': val_rmse_orig,
            'test_rmse': test_rmse_orig,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'best_epoch': best_epoch,
            'total_epochs': epoch,
            'time_s': total_time,
            'train_rmse_hist': train_rmse_hist,
            'val_rmse_hist': val_rmse_hist,
            'test_preds': test_preds_orig.tolist(),
            'test_targets': test_targets_orig.tolist(),
            'spse_alpha': alpha.tolist(),
            'spse_beta': beta.tolist(),
            'gates': gate_values,
        }
        with open(RESULTS_JSON, 'w') as f:
            json.dump(results, f, indent=2)

        with open(SPSE_PARAMS_TXT, 'w') as fp:
            fp.write("Residual MaxGAT: SPSE Parameters + Gates\n")
            fp.write("=" * 50 + "\n")
            fp.write(f"Residual gates: {gate_values}\n\n")
            for k in range(4):
                fp.write(f"k={k+1}: alpha={alpha[k].item():.6f}, beta={beta[k].item():.6f}\n")

        # Plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(range(1, len(train_rmse_hist)+1), train_rmse_hist, 'b-', alpha=0.7, label='Train')
        ax1.plot(range(1, len(val_rmse_hist)+1), val_rmse_hist, 'r-', alpha=0.7, label='Val')
        ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best@{best_epoch}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('RMSE (normalized)')
        ax1.set_title('Learning Curves - Residual MaxGAT')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.scatter(test_targets_orig, test_preds_orig, alpha=0.5, s=20)
        lim_min = min(test_targets_orig.min(), test_preds_orig.min())
        lim_max = max(test_targets_orig.max(), test_preds_orig.max())
        ax2.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.5)
        ax2.set_xlabel('Actual Tg (K)')
        ax2.set_ylabel('Predicted Tg (K)')
        ax2.set_title(f'Residual MaxGAT: RMSE={test_rmse_orig:.1f}K, R²={test_r2:.3f}')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(CURVES_PNG, dpi=150)
        plt.close()
        print(f"  Plots saved to {CURVES_PNG}")

    print("Treatment B v2 training complete.")


if __name__ == '__main__':
    main()
