"""
CPA-GAT single model training script (3 seeds: 76, 14, 71).
"""
import os
import sys
import json
import time
import pickle
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cpa_gat import CPAGAT

DATA_PATH = "/home/home_public/aisci/AIscientist/src/4_day4_selfgnn_polymer/data/tg_cache/tg_graphs_3ru_42dim.pkl"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [76, 14, 71]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['train_graphs'], data['val_graphs'], data['test_graphs'], data['scaler']


def train_epoch(model, loader, optimizer, criterion, device, clip_norm=1.0):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, scaler=None):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = criterion(pred, batch.y.view(-1))
            total_loss += loss.item() * batch.num_graphs
            all_preds.append(pred.cpu())
            all_targets.append(batch.y.view(-1).cpu())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    if scaler is not None:
        all_preds = all_preds * scaler.scale_[0] + scaler.mean_[0]
        all_targets = all_targets * scaler.scale_[0] + scaler.mean_[0]

    mse = ((all_preds - all_targets) ** 2).mean().item()
    rmse = np.sqrt(mse)
    mae = (all_preds - all_targets).abs().mean().item()
    return avg_loss, rmse, mae, all_preds, all_targets


def train_cpa_gat(seed):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    print(f"\n{'='*60}")
    print(f"Training CPAGAT | Seed {seed} | Device: {device}")
    print(f"{'='*60}")

    train_graphs, val_graphs, test_graphs, scaler = load_data()

    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)

    model = CPAGAT(in_dim=42).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: CPAGAT, Params: {num_params:,}")

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-6)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_epoch = 0
    best_state = None
    patience = 40
    patience_counter = 0
    history = []
    start_time = time.time()

    for epoch in range(1, 301):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_rmse, val_mae, _, _ = evaluate(model, val_loader, criterion, device, scaler)
        current_lr = optimizer.param_groups[0]['lr']

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'lr': current_lr,
        })

        scheduler.step()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val RMSE: {val_rmse:.2f} K | Best: {best_val_rmse:.2f} K (ep {best_epoch}) | LR: {current_lr:.6f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    train_time = time.time() - start_time

    model.load_state_dict(best_state)
    model.to(device)
    test_loss, test_rmse, test_mae, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device, scaler
    )

    print(f"\nFinal Results for CPAGAT Seed {seed}:")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Val RMSE: {best_val_rmse:.4f} K")
    print(f"  Test RMSE: {test_rmse:.4f} K")
    print(f"  Test MAE: {test_mae:.4f} K")
    print(f"  Training Time: {train_time:.1f}s")

    result = {
        'model_name': 'cpa_gat',
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_rmse': float(best_val_rmse),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'train_time': train_time,
        'num_params': num_params,
        'test_preds': test_preds.numpy().tolist(),
        'test_targets': test_targets.numpy().tolist(),
    }

    result_file = os.path.join(RESULTS_DIR, f"cpa_gat_seed{seed}_v2.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    history_file = os.path.join(RESULTS_DIR, f"cpa_gat_seed{seed}_v2_history.json")
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    model_file = os.path.join(RESULTS_DIR, f"cpa_gat_seed{seed}_v2.pt")
    torch.save(best_state, model_file)

    return result


if __name__ == "__main__":
    all_results = []
    for seed in SEEDS:
        result = train_cpa_gat(seed)
        all_results.append(result)

    rmses = [r['test_rmse'] for r in all_results]
    mean_rmse = np.mean(rmses)
    std_rmse = np.std(rmses)

    print(f"\n{'='*60}")
    print(f"CPAGAT Summary (seeds {SEEDS})")
    print(f"  Test RMSE: {mean_rmse:.2f} ± {std_rmse:.2f} K")
    for r in all_results:
        print(f"    Seed {r['seed']}: {r['test_rmse']:.4f} K")
    print(f"{'='*60}")

    summary_file = os.path.join(RESULTS_DIR, "cpa_gat_v2_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'seeds': SEEDS,
            'test_rmse_mean': float(mean_rmse),
            'test_rmse_std': float(std_rmse),
            'per_seed': all_results,
        }, f, indent=2)
    print(f"Summary saved to {summary_file}")
