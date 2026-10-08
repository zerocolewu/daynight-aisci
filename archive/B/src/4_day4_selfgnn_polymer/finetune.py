#!/usr/bin/env python3
"""
Finetune SelfGNN pretrained GAT encoder on Tg prediction.

Supports:
  - Pretrained init + full finetuning
  - Random init (baseline)
  - Frozen encoder (Ablation G)
  - Multiple independent seeds
"""

import os
import sys
import pickle
import json
import csv
import argparse
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Set2Set, global_mean_pool
from torch.utils.data import Dataset as TorchDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gat_encoder import GATEncoder, GATEncoderWithProjection
from utils.diagnostics import plot_finetune_curves, compute_regression_metrics

warnings.filterwarnings('ignore')

WORK_DIR = "/home/home_public/aisci/AIscientist/src/4_day4_selfgnn_polymer"
TG_PKL = os.path.join(WORK_DIR, "data", "tg_cache", "tg_graphs_3ru_42dim.pkl")
RESULTS_DIR = os.path.join(WORK_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------------------------------------------------------
# Finetuning Model
# --------------------------------------------------------------------
class FinetuneModel(nn.Module):
    """GAT encoder + Set2Set readout + MLP predictor for Tg regression."""

    def __init__(self, encoder, encoder_output_dim=256, set2set_steps=3,
                 hidden1=256, hidden2=128, dropout=0.1, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.encoder_output_dim = encoder_output_dim
        self.freeze_encoder = freeze_encoder

        # Set2Set: input dim = encoder output, output = 2 * input_dim
        self.readout = Set2Set(encoder_output_dim, processing_steps=set2set_steps)

        # MLP: 512 -> 256 -> 128 -> 1
        self.mlp = nn.Sequential(
            nn.Linear(2 * encoder_output_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x, edge_index, batch):
        h = self.encoder.forward_node(x, edge_index)
        h_pooled = self.readout(h, batch)
        return self.mlp(h_pooled).squeeze(-1)


# --------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------
class TgDataset(TorchDataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_linear_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
    """Linear warmup then cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            return max(eta_min / optimizer.defaults['lr'], 0.5 * (1 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = F.mse_loss(pred, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.y.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_pred = []
    all_true = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = F.mse_loss(pred, batch.y.squeeze())
        total_loss += loss.item() * batch.y.size(0)
        all_pred.append(pred.cpu())
        all_true.append(batch.y.squeeze().cpu())

    pred = torch.cat(all_pred).numpy()
    true = torch.cat(all_true).numpy()
    rmse = np.sqrt(mean_squared_error(true, pred))
    return total_loss / len(loader.dataset), rmse, pred, true


def finetune_one_seed(seed, train_graphs, val_graphs, test_graphs, scaler,
                      pretrained_path=None, freeze_encoder=False, encoder_type='gatv2',
                      lr=1e-4, weight_decay=1e-4, batch_size=64,
                      warmup=5, max_epochs=200, patience=50):
    """Run one finetuning run with a specific seed."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataloaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=0)

    input_dim = train_graphs[0].x.size(1)

    # Build encoder
    if encoder_type == 'gatv2':
        encoder = GATEncoder(
            input_dim=input_dim, hidden_dim=256, num_heads=4,
            num_layers=4, dropout=0.1, use_batchnorm=True
        )
    else:
        encoder = GATEncoderWithProjection(
            input_dim=input_dim, hidden_dim=256, num_heads=4,
            num_layers=4, dropout=0.1
        )

    # Load pretrained weights if provided
    if pretrained_path is not None:
        print(f"  Loading pretrained weights from: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        # Filter to match encoder keys
        encoder_state = encoder.state_dict()
        matched = {k: v for k, v in state_dict.items() if k in encoder_state}
        print(f"  Matched {len(matched)}/{len(encoder_state)} encoder params")
        encoder_state.update(matched)
        encoder.load_state_dict(encoder_state)

    model = FinetuneModel(
        encoder=encoder,
        encoder_output_dim=256,
        set2set_steps=3,
        hidden1=256,
        hidden2=128,
        dropout=0.1,
        freeze_encoder=freeze_encoder,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable: {n_trainable:,}, Frozen: {n_frozen:,}")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay
    )
    scheduler = get_linear_warmup_cosine_scheduler(optimizer, warmup, max_epochs, eta_min=1e-6)

    best_val_rmse = float('inf')
    best_state = None
    best_epoch = 0
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_rmses_hist = []
    val_rmses_hist = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_rmse, _, _ = evaluate(model, val_loader, device)
        _, train_rmse, _, _ = evaluate(model, train_loader, device)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_rmses_hist.append(train_rmse)
        val_rmses_hist.append(val_rmse)

        if val_rmse < best_val_rmse - 1e-6:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    # Final test evaluation
    model.load_state_dict(best_state)
    model = model.to(device)
    test_loss, test_rmse_scaled, test_pred_scaled, test_true_scaled = evaluate(model, test_loader, device)

    # Inverse transform to original scale
    test_pred = scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
    test_true = scaler.inverse_transform(test_true_scaled.reshape(-1, 1)).flatten()
    test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
    test_mae = mean_absolute_error(test_true, test_pred)

    result = {
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_rmse_scaled': float(best_val_rmse),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'train_rmses': [float(x) for x in train_rmses_hist],
        'val_rmses': [float(x) for x in val_rmses_hist],
    }
    return result


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained encoder weights')
    parser.add_argument('--output_name', type=str, default='finetune',
                        help='Output filename prefix')
    parser.add_argument('--seeds', type=int, nargs='+', default=[71, 42, 123])
    parser.add_argument('--freeze_encoder', action='store_true', default=False)
    parser.add_argument('--encoder_type', type=str, default='gatv2',
                        choices=['gatv2', 'gatconv'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=50)
    args = parser.parse_args()

    print(f"=== SelfGNN Finetuning ===")
    print(f"  Seeds: {args.seeds}")
    print(f"  Pretrained: {args.pretrained_path is not None}")
    print(f"  Freeze encoder: {args.freeze_encoder}")
    print(f"  Encoder type: {args.encoder_type}")
    print(f"  LR: {args.lr}, Patience: {args.patience}")

    # Load Tg data
    print(f"\nLoading Tg data from: {TG_PKL}")
    with open(TG_PKL, 'rb') as f:
        data = pickle.load(f)

    train_graphs = data['train_graphs']
    val_graphs = data['val_graphs']
    test_graphs = data['test_graphs']
    scaler = data['scaler']
    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    print(f"  Feature dim: {train_graphs[0].x.size(1)}")

    # Run per seed
    all_results = []
    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        result = finetune_one_seed(
            seed=seed,
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            test_graphs=test_graphs,
            scaler=scaler,
            pretrained_path=args.pretrained_path,
            freeze_encoder=args.freeze_encoder,
            encoder_type=args.encoder_type,
            lr=args.lr,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            patience=args.patience,
        )
        all_results.append(result)
        print(f"  Test RMSE: {result['test_rmse']:.4f}K, Test MAE: {result['test_mae']:.4f}K")
        print(f"  Best epoch: {result['best_epoch']}")

    # Aggregate
    test_rmses = [r['test_rmse'] for r in all_results]
    test_maes = [r['test_mae'] for r in all_results]

    summary = {
        'config': {
            'pretrained': args.pretrained_path is not None,
            'freeze_encoder': args.freeze_encoder,
            'encoder_type': args.encoder_type,
            'seeds': args.seeds,
        },
        'per_seed': all_results,
        'mean_test_rmse': float(np.mean(test_rmses)),
        'std_test_rmse': float(np.std(test_rmses)),
        'mean_test_mae': float(np.mean(test_maes)),
        'std_test_mae': float(np.std(test_maes)),
    }

    print(f"\n=== Final Results ===")
    print(f"  Test RMSE: {summary['mean_test_rmse']:.4f}K ± {summary['std_test_rmse']:.4f}K")
    print(f"  Test MAE:  {summary['mean_test_mae']:.4f}K ± {summary['std_test_mae']:.4f}K")

    # Save
    result_path = os.path.join(RESULTS_DIR, f"{args.output_name}_results.json")
    with open(result_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot curves for first seed
    if all_results:
        plot_finetune_curves(
            all_results[0]['train_losses'],
            all_results[0]['val_losses'],
            all_results[0]['train_rmses'],
            all_results[0]['val_rmses'],
            os.path.join(RESULTS_DIR, f"{args.output_name}_curves.png")
        )

    print(f"\n  Results saved: {result_path}")


if __name__ == "__main__":
    main()
