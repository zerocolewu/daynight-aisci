"""Finetuning script: GAT encoder + Set2Set + MLP for Tg prediction (REWRITE).

Architecture:
  GATEncoder (42-dim in, 256-dim out)
  → Set2Set(256, processing_steps=3) → 512-dim
  → Linear(512→256) + BN + ReLU + Dropout(0.1)
  → Linear(256→128) + BN + ReLU + Dropout(0.1)
  → Linear(128→1)

Hyperparams (matching task spec):
  - AdamW lr=3e-4, weight_decay=1e-4
  - CosineAnnealingLR (T_max=epochs, eta_min=1e-6)
  - batch=64, max 200 epochs, patience=50
  - 3 seeds: 71, 42, 123
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.serialization
import torch.nn as nn
import torch.nn.functional as F

# Fix for PyTorch 2.6+ weights_only default
import numpy.core.multiarray
torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
torch.serialization.add_safe_globals([numpy.dtype])
try:
    torch.serialization.add_safe_globals([numpy.ndarray])
except Exception:
    pass
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_utils import TgDataset, split_dataset
from models import GATEncoder, FinetuneModel, NoPretrainModel


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, scaler=None):
    model.eval()
    preds, targets = [], []
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
        preds.append(pred.detach().cpu().numpy())
        targets.append(data.y.detach().cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    if scaler is not None:
        preds = scaler.inverse_transform(preds.reshape(-1, 1)).ravel()

    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    return rmse, mae, r2


def finetune(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results = {}

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Finetuning seed={seed}")
        print(f"{'='*60}")
        set_seed(seed)

        # ── Load Tg dataset (42-dim: 8 rich + 2 PSPE + 32 RWPE) ──────────────
        dataset = TgDataset(
            csv_path=args.tg_path,
            n_ru=args.n_ru,
            add_rwpe=True,
            rwpe_dim=32,
            root=f'{args.data_root}_seed{seed}',
        )
        print(f"Tg dataset size: {len(dataset)}")

        # Split: 72/8/20
        train_idx, val_idx, test_idx = split_dataset(
            dataset, train_ratio=0.72, val_ratio=0.08, seed=71,
        )
        print(f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)

        # Target normalization
        train_y = np.array([dataset[i].y.item() for i in train_idx]).reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(train_y)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        # ── Build model ──────────────────────────────────────────────────────
        if args.no_pretrain:
            model = NoPretrainModel(in_dim=42, hidden_dim=args.hidden_dim, heads=args.heads)
            print("Using NO-pretrain baseline (random init)")
        else:
            # Load pretrained encoder (42-dim input)
            encoder = GATEncoder(
                in_dim=42, hidden_dim=args.hidden_dim, heads=args.heads,
                use_bn=not args.no_batchnorm,
            )

            checkpoint = torch.load(args.pretrain_path, map_location='cpu', weights_only=False)
            student_state = checkpoint.get('student_state_dict', checkpoint)
            encoder_state = {}
            for k, v in student_state.items():
                if k.startswith('encoder.'):
                    encoder_state[k[8:]] = v  # remove 'encoder.' prefix
            encoder.load_state_dict(encoder_state, strict=False)
            print(f"Loaded pretrained encoder from {args.pretrain_path}")

            freeze_enc = args.freeze_encoder
            model = FinetuneModel(
                encoder, hidden_dim=args.hidden_dim, heads=args.heads,
                freeze_encoder=freeze_enc,
            )
            if freeze_enc:
                print("Encoder FROZEN")

        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {n_params:,}")

        # ── Training (hyperparams from task spec) ─────────────────────────────
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=1e-4,
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6,
        )

        best_val_rmse = float('inf')
        best_state = None
        patience_counter = 0
        best_epoch = 0

        for epoch in range(1, args.epochs + 1):
            # Train
            train_loss = 0.0
            model.train()
            for data in train_loader:
                data = data.to(device)
                y_scaled = torch.tensor(
                    scaler.transform(data.y.detach().cpu().numpy().reshape(-1, 1)),
                    dtype=torch.float, device=device,
                ).squeeze(-1)

                optimizer.zero_grad()
                pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = criterion(pred, y_scaled)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            scheduler.step()

            # Validation
            val_rmse, val_mae, val_r2 = evaluate(
                model, val_loader, device, scaler=scaler,
            )

            if epoch % 10 == 0 or epoch <= 5:
                print(f"Epoch {epoch:3d} | TrainLoss: {train_loss:.4f} | "
                      f"Val RMSE: {val_rmse:.2f} | Val MAE: {val_mae:.2f} | Val R²: {val_r2:.4f}")

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # ── Test evaluation ──────────────────────────────────────────────────
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        test_rmse, test_mae, test_r2 = evaluate(
            model, test_loader, device, scaler=scaler,
        )

        results[seed] = {
            'best_epoch': best_epoch,
            'best_val_rmse': float(best_val_rmse),
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae),
            'test_r2': float(test_r2),
        }
        print(f"\nSeed {seed} Results: Test RMSE={test_rmse:.2f} "
              f"MAE={test_mae:.2f} R²={test_r2:.4f} (best epoch={best_epoch})")

        # Save model
        save_name = 'finetune'
        if args.no_pretrain:
            save_name = 'finetune_nopretrain'
        elif args.freeze_encoder:
            save_name = 'finetune_frozen'
        torch.save({
            'model_state_dict': best_state,
            'seed': seed,
            'test_rmse': float(test_rmse),
        }, os.path.join(args.save_dir, f'{save_name}_seed{seed}.pt'))

    # ── Report ───────────────────────────────────────────────────────────────
    test_rmses = [results[s]['test_rmse'] for s in args.seeds]
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    for seed in args.seeds:
        print(f"  Seed {seed}: Test RMSE = {results[seed]['test_rmse']:.2f} K")
    print(f"  Mean ± Std: {np.mean(test_rmses):.2f} ± {np.std(test_rmses):.2f} K")
    print(f"{'='*60}")

    with open(os.path.join(args.save_dir, 'finetune_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tg_path', type=str,
                        default='/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv')
    parser.add_argument('--pretrain_path', type=str, default='./pretrain_output/pretrain_best.pt')
    parser.add_argument('--data_root', type=str, default='./data_tg')
    parser.add_argument('--save_dir', type=str, default='./finetune_output')
    parser.add_argument('--n_ru', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--no_batchnorm', action='store_true', default=False)
    parser.add_argument('--no_pretrain', action='store_true', default=False)
    parser.add_argument('--freeze_encoder', action='store_true', default=False)
    parser.add_argument('--seeds', type=int, nargs='+', default=[71, 42, 123])

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    finetune(args)
