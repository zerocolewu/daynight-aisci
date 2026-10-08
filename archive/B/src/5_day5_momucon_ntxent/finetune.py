#!/usr/bin/env python3
"""
MoMuCon-NT-Xent Finetuning on Tg Dataset.

Loads pretrained GAT encoder, adds fresh predictor head,
finetunes on Tg dataset with 3-RU graphs.

Command:
  python finetune.py --pretrained_path checkpoints/best_pretrain.pt --data_path src/alldataset/Tg
"""

import os
import sys
import argparse
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

warnings.filterwarnings('ignore')

WORK_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_rmse(pred, target, scaler=None):
    """Compute RMSE. If scaler provided, unnormalize first."""
    if scaler is not None:
        pred = scaler.inverse_transform(pred.cpu().numpy().reshape(-1, 1))
        target = scaler.inverse_transform(target.cpu().numpy().reshape(-1, 1))
        rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    else:
        rmse = float(torch.sqrt(nn.MSELoss()(pred, target)).item())
    return rmse


def finetune_epoch(model, loader, optimizer, scaler, device, is_train=True):
    """Run one epoch of finetuning."""
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for data in tqdm(loader, desc="Train" if is_train else "Eval", leave=False):
        data = data.to(device)
        if is_train:
            optimizer.zero_grad()
            with amp.autocast():
                pred = model(data)
                loss = nn.MSELoss()(pred, data.y.squeeze())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                with amp.autocast():
                    pred = model(data)

        total_loss += loss.item() if is_train else nn.MSELoss()(pred, data.y.squeeze()).item()
        all_preds.append(pred.cpu().detach())
        all_targets.append(data.y.squeeze().cpu().detach())

    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    rmse = float(torch.sqrt(nn.MSELoss()(all_preds, all_targets)).item())

    return avg_loss, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str,
                        default='checkpoints/best_pretrain.pt')
    parser.add_argument('--data_path', type=str,
                        default='/home/home_public/aisci/AIscientist/src/alldataset/Tg')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--freeze_encoder', type=int, default=0,
                        help='Epochs to freeze encoder (0 = no freeze)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem_gb:.1f} GB)")

    results_dir = os.path.join(WORK_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("MoMuCon-NT-Xent Finetuning on Tg")
    print("=" * 70)
    print(f"Pretrained: {args.pretrained_path}")
    print(f"LR: {args.lr}, Batch size: {args.batch_size}, Max epochs: {args.max_epochs}")
    print(f"Patience: {args.patience}, Freeze epochs: {args.freeze_encoder}")

    # Load Tg data
    print("\nLoading Tg data...")
    tg_data = prepare_tg_data(cache_dir=WORK_DIR, force_rebuild=False)
    train_graphs = tg_data['train_graphs']
    val_graphs = tg_data['val_graphs']
    test_graphs = tg_data['test_graphs']
    scaler = tg_data['scaler']
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    # Determine feature dimensions
    input_dim = train_graphs[0].x.size(1)
    edge_dim = train_graphs[0].edge_attr.size(1)
    print(f"Input node dim: {input_dim}, Edge dim: {edge_dim}")

    # DataLoaders
    train_ds = TgDataset(train_graphs)
    val_ds = TgDataset(val_graphs)
    test_ds = TgDataset(test_graphs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True)

    # Load pretrained encoder
    encoder = GATEncoder(
        in_dim=input_dim, hidden=256, num_layers=4, heads=4,
        dropout=0.1, edge_dim=edge_dim,
    ).to(device)

    pretrained_path_full = os.path.join(WORK_DIR, args.pretrained_path) \
        if not os.path.isabs(args.pretrained_path) else args.pretrained_path

    if os.path.exists(pretrained_path_full):
        ckpt = torch.load(pretrained_path_full, map_location=device)
        encoder.load_state_dict(ckpt['encoder'])
        print(f"Loaded pretrained encoder from {pretrained_path_full}")
        print(f"  Pretrained epoch: {ckpt.get('epoch', '?')}")
        print(f"  Pretrained loss: {ckpt.get('loss', '?'):.4f}" if 'loss' in ckpt else "")
    else:
        print(f"WARNING: Pretrained checkpoint not found at {pretrained_path_full}")
        print("Finetuning from scratch (random init)")

    # Predictor (fresh, randomly initialized)
    predictor = Predictor(in_dim=256, hidden_dim=128, dropout=0.1).to(device)

    # FinetuneModel wrapper
    class FinetuneModel(nn.Module):
        def __init__(self, encoder, predictor):
            super().__init__()
            self.encoder = encoder
            self.predictor = predictor

        def forward(self, data):
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
            h = self.encoder(data.x, data.edge_index, edge_attr, data.batch)
            return self.predictor(h)

        def freeze_encoder(self):
            for param in self.encoder.parameters():
                param.requires_grad = False

        def unfreeze_encoder(self):
            for param in self.encoder.parameters():
                param.requires_grad = True

    model = FinetuneModel(encoder, predictor).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,} total, {n_trainable:,} trainable")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                   patience=10, min_lr=1e-6)
    amp_scaler = amp.GradScaler()

    # Training loop
    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    val_rmse_history = []
    train_rmse_history = []

    print(f"\nStarting finetuning...\n")

    for epoch in range(args.max_epochs):
        # Freeze/unfreeze logic
        if args.freeze_encoder > 0 and epoch < args.freeze_encoder:
            model.freeze_encoder()
            if epoch == 0:
                print(f"Freezing encoder for first {args.freeze_encoder} epochs")
        elif args.freeze_encoder > 0 and epoch == args.freeze_encoder:
            model.unfreeze_encoder()
            print(f"Unfreezing encoder at epoch {epoch + 1}")

        # Train
        train_loss, train_rmse = finetune_epoch(
            model, train_loader, optimizer, amp_scaler, device, is_train=True
        )
        train_rmse_history.append(train_rmse)

        # Validate
        val_loss, val_rmse = finetune_epoch(
            model, val_loader, None, None, device, is_train=False
        )
        val_rmse_history.append(val_rmse)

        # Learning rate before scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_rmse)

        print(f"  Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_rmse={train_rmse:.4f}, "
              f"val_rmse={val_rmse:.4f}, lr={current_lr:.2e}")

        # Checkpoint
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'encoder': encoder.state_dict(),
                'predictor': predictor.state_dict(),
                'best_val_rmse': best_val_rmse,
            }, os.path.join(results_dir, 'best_finetune.pt'))
            print(f"    New best! val_rmse={best_val_rmse:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Load best model and evaluate on test set
    print(f"\nLoading best model (epoch {best_epoch}, val_rmse={best_val_rmse:.4f})...")
    best_ckpt = torch.load(os.path.join(results_dir, 'best_finetune.pt'), map_location=device)
    encoder.load_state_dict(best_ckpt['encoder'])
    predictor.load_state_dict(best_ckpt['predictor'])
    model = FinetuneModel(encoder, predictor).to(device)

    # Test evaluation
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

    # Compute RMSE (normalized and K units)
    norm_rmse = float(torch.sqrt(nn.MSELoss()(all_preds, all_targets.squeeze())).item())
    if scaler is not None and all_targets_orig:
        preds_k = scaler.inverse_transform(all_preds.numpy().reshape(-1, 1))
        targets_k = scaler.inverse_transform(all_targets.numpy().reshape(-1, 1))
        test_rmse_k = float(np.sqrt(np.mean((preds_k - targets_k) ** 2)))
    else:
        test_rmse_k = norm_rmse

    print("\n" + "=" * 70)
    print("MoMuCon-NT-Xent Finetuning Results")
    print("=" * 70)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation RMSE (norm): {best_val_rmse:.4f}")
    print(f"Test RMSE (normalized): {norm_rmse:.4f}")
    print(f"Test RMSE (K): {test_rmse_k:.2f} K")
    print(f"Baseline: 35.44 K (GAT+RWPE+PSPE, no pretraining)")
    delta = 35.44 - test_rmse_k
    print(f"Delta vs baseline: {delta:+.2f} K")
    print(f"Val RMSE history length: {len(val_rmse_history)}")

    # Save results
    import json
    results = {
        'best_epoch': best_epoch,
        'best_val_rmse_norm': best_val_rmse,
        'test_rmse_norm': norm_rmse,
        'test_rmse_k': test_rmse_k,
        'baseline_k': 35.44,
        'delta_k': delta,
        'val_rmse_history': val_rmse_history,
        'train_rmse_history': train_rmse_history,
    }
    with open(os.path.join(results_dir, 'finetune_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_dir}/finetune_results.json")

    # Plot
    from utils import plot_finetune_curve
    plot_finetune_curve(val_rmse_history, results_dir)

    return test_rmse_k, delta


if __name__ == '__main__':
    main()
