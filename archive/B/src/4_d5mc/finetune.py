#!/usr/bin/env python3
"""
MoMuCon Finetuning on Tg Prediction.

Runs three comparison modes:
  1. Non-pretrained baseline (random init GAT encoder + predictor)
  2. Linear probe (frozen pretrained encoder + linear regressor)
  3. Full finetuning (pretrained encoder + predictor, all weights trainable)
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
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

from data_utils import prepare_tg_data, TOTAL_DIM, EDGE_DIM
from model import GATEncoder, FinetuneModel, LinearProbe

warnings.filterwarnings('ignore')

WORK_DIR = "/home/home_public/aisci/AIscientist/src/4_d5mc"
RESULTS_DIR = os.path.join(WORK_DIR, "results")
MODELS_DIR = os.path.join(WORK_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


class Config:
    lr = 1e-4
    weight_decay = 1e-4
    batch_size = 64
    max_epochs = 200
    early_stop_patience = 50
    warmup_epochs = 5
    seed = 71
    hidden_dim = 256
    num_heads = 4
    num_layers = 4
    dropout = 0.1
    predictor_hidden = 128


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return max(eta_min / optimizer.defaults['lr'],
                   0.5 * (1 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = F.mse_loss(pred, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.y.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        loss = F.mse_loss(pred, batch.y.squeeze())
        total_loss += loss.item() * batch.y.size(0)
        all_pred.append(pred.cpu())
        all_true.append(batch.y.squeeze().cpu())

    pred = torch.cat(all_pred).numpy()
    true = torch.cat(all_true).numpy()
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    return total_loss / len(loader.dataset), rmse, mae, r2, pred, true


def run_finetuning(train_graphs, val_graphs, test_graphs, scaler,
                   pretrained_encoder_path=None, mode='finetune', config=None):
    if config is None:
        config = Config()

    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = train_graphs[0].x.size(1)
    edge_dim = EDGE_DIM

    encoder = GATEncoder(
        input_dim=input_dim, hidden_dim=config.hidden_dim,
        num_heads=config.num_heads, num_layers=config.num_layers,
        dropout=config.dropout, edge_dim=edge_dim
    )

    if pretrained_encoder_path is not None and mode in ('finetune', 'linear_probe'):
        print(f"  Loading pretrained encoder from: {pretrained_encoder_path}")
        state_dict = torch.load(pretrained_encoder_path, map_location='cpu')
        encoder_state = encoder.state_dict()
        matched = {k: v for k, v in state_dict.items() if k in encoder_state}
        missing = [k for k in encoder_state if k not in state_dict]
        print(f"  Loaded {len(matched)}/{len(encoder_state)} params")
        if missing:
            print(f"  Missing keys: {missing[:5]}...")
        encoder_state.update(matched)
        encoder.load_state_dict(encoder_state)

    if mode == 'linear_probe':
        model = LinearProbe(encoder, encoder_output_dim=config.hidden_dim * config.num_heads).to(device)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Linear probe: {n_trainable} trainable params")
    else:
        model = FinetuneModel(
            encoder, encoder_output_dim=config.hidden_dim * config.num_heads,
            predictor_hidden=config.predictor_hidden, dropout=config.dropout
        ).to(device)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Finetune model: {n_trainable} trainable params")

    train_loader = DataLoader(train_graphs, batch_size=config.batch_size, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_graphs, batch_size=config.batch_size, shuffle=False,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=config.batch_size, shuffle=False,
                             pin_memory=True, num_workers=0)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = get_warmup_cosine_scheduler(optimizer, config.warmup_epochs, config.max_epochs)

    best_val_rmse = float('inf')
    best_state = None
    best_epoch = 0
    patience_counter = 0

    train_losses, val_losses = [], []
    train_rmses, val_rmses = [], []

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_rmse, val_mae, val_r2, _, _ = evaluate(model, val_loader, device)
        _, train_rmse, _, _, _, _ = evaluate(model, train_loader, device)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)

        if val_rmse < best_val_rmse - 1e-8:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stop_patience:
            if epoch <= config.warmup_epochs + 5:
                continue
            print(f"    Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model = model.to(device)
    test_loss, test_rmse_scaled, test_mae_scaled, test_r2_scaled, pred_scaled, true_scaled = evaluate(
        model, test_loader, device
    )

    test_pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    test_true = scaler.inverse_transform(true_scaled.reshape(-1, 1)).flatten()
    test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
    test_mae = mean_absolute_error(test_true, test_pred)
    test_r2 = r2_score(test_true, test_pred)

    result = {
        'mode': mode,
        'best_epoch': best_epoch,
        'best_val_rmse_scaled': float(best_val_rmse),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'train_rmses': [float(x) for x in train_rmses],
        'val_rmses': [float(x) for x in val_rmses],
        'n_train': len(train_graphs),
    }
    return result


def main():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained_path = os.path.join(MODELS_DIR, "pretrained_encoder.pt")

    print("=" * 70)
    print("MoMuCon Finetuning on Tg Prediction")
    print("=" * 70)

    tg_data = prepare_tg_data(force_rebuild=False)
    train_graphs = tg_data['train_graphs']
    val_graphs = tg_data['val_graphs']
    test_graphs = tg_data['test_graphs']
    scaler = tg_data['scaler']

    print(f"\nTg data: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")
    print(f"Feature dim: {train_graphs[0].x.size(1)}")

    all_results = {}

    # 1. Non-pretrained baseline
    print("\n" + "=" * 50)
    print("[1/3] Non-pretrained baseline (random init)")
    print("=" * 50)
    result_random = run_finetuning(
        train_graphs, val_graphs, test_graphs, scaler,
        pretrained_encoder_path=None, mode='random', config=config
    )
    all_results['random_init'] = result_random
    print(f"  Test RMSE: {result_random['test_rmse']:.4f}K")
    print(f"  Test MAE:  {result_random['test_mae']:.4f}K")
    print(f"  Test R^2:   {result_random['test_r2']:.4f}")
    print(f"  Best epoch: {result_random['best_epoch']}")

    # 2. Linear probe
    if os.path.exists(pretrained_path):
        print("\n" + "=" * 50)
        print("[2/3] Linear probe (frozen pretrained encoder)")
        print("=" * 50)
        result_probe = run_finetuning(
            train_graphs, val_graphs, test_graphs, scaler,
            pretrained_encoder_path=pretrained_path, mode='linear_probe', config=config
        )
        all_results['linear_probe'] = result_probe
        print(f"  Test RMSE: {result_probe['test_rmse']:.4f}K")
        print(f"  Test MAE:  {result_probe['test_mae']:.4f}K")
        print(f"  Test R^2:   {result_probe['test_r2']:.4f}")
        print(f"  Best epoch: {result_probe['best_epoch']}")
    else:
        print(f"\n[2/3] SKIP: No pretrained encoder found at {pretrained_path}")

    # 3. Full finetuning
    if os.path.exists(pretrained_path):
        print("\n" + "=" * 50)
        print("[3/3] Full finetuning (pretrained encoder)")
        print("=" * 50)
        result_finetune = run_finetuning(
            train_graphs, val_graphs, test_graphs, scaler,
            pretrained_encoder_path=pretrained_path, mode='finetune', config=config
        )
        all_results['full_finetune'] = result_finetune
        print(f"  Test RMSE: {result_finetune['test_rmse']:.4f}K")
        print(f"  Test MAE:  {result_finetune['test_mae']:.4f}K")
        print(f"  Test R^2:   {result_finetune['test_r2']:.4f}")
        print(f"  Best epoch: {result_finetune['best_epoch']}")
    else:
        print(f"\n[3/3] SKIP: No pretrained encoder found at {pretrained_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("FINETUNING RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<25} {'Test RMSE':>10} {'Test MAE':>10} {'Test R^2':>10} {'Epoch':>8}")
    print("-" * 65)
    for mode, r in all_results.items():
        print(f"{mode:<25} {r['test_rmse']:>10.4f} {r['test_mae']:>10.4f} {r['test_r2']:>10.4f} {r['best_epoch']:>8}")

    result_path = os.path.join(RESULTS_DIR, "finetune_results.json")
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
