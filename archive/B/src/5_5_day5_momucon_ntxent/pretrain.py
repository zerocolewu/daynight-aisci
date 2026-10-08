#!/usr/bin/env python3
"""
MoMuCon-NT-Xent Pretraining on PI1M.

Contrastive pretraining with explicit NT-Xent loss:
  - Maps 1-RU (monomer) and 3-RU (multimer) graph embeddings close
  - Pushes apart embeddings of different polymers

Command:
  python pretrain.py --data_path src/alldataset/PI1M --num_polymers 200000 \\
      --batch_size 128 --epochs 50 --lr 1e-4 --temperature 0.07
"""

import os
import sys
import argparse
import json
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models import GATEncoder, ProjectionHead
from dataset import (
    prepare_pi1m_pretrain_data,
    TOTAL_NODE_DIM,
    EDGE_DIM,
    PairDataset,
    collate_pairs,
)
from utils import (
    nt_xent_loss_with_two_heads,
    MetricTracker,
    plot_pretrain_diagnostics,
    compute_grad_norm,
)

warnings.filterwarnings('ignore')

WORK_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs,
                                 eta_min=1e-6, base_lr=1e-4):
    """Linear warmup then cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return max(eta_min / base_lr, 0.5 * (1 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='/home/home_public/aisci/AIscientist/src/alldataset/PI1M')
    parser.add_argument('--num_polymers', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--accum_steps', type=int, default=4,
                        help='Gradient accumulation steps (4 = micro-batch 32)')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint path (e.g., checkpoints/pretrain_epoch30.pt)')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem_gb:.1f} GB)")

    # Directories
    checkpoint_dir = os.path.join(WORK_DIR, 'checkpoints')
    results_dir = os.path.join(WORK_DIR, 'results')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("MoMuCon-NT-Xent Pretraining on PI1M")
    print("=" * 70)
    print(f"Polymers: {args.num_polymers}, Batch size: {args.batch_size}")
    print(f"Accum steps: {args.accum_steps} (effective micro-batch: {args.batch_size // args.accum_steps})")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Temperature: {args.temperature}")
    print(f"Hidden: {args.hidden_dim}, Heads: {args.heads}, Layers: {args.num_layers}")
    print(f"Device: {device}")

    # Load data
    pairs = prepare_pi1m_pretrain_data(
        n_polymers=args.num_polymers,
        seed=args.seed,
        cache_dir=WORK_DIR,
        force_rebuild=False,
    )
    input_dim = pairs[0][0].x.size(1)
    edge_dim = pairs[0][0].edge_attr.size(1)
    print(f"\nInput node dim: {input_dim}, Edge dim: {edge_dim}")
    print(f"Total pairs: {len(pairs)}")

    # Shuffle and split
    random.shuffle(pairs)
    train_frac = 0.95
    split = int(train_frac * len(pairs))
    train_pairs = pairs[:split]
    diag_pairs = pairs[split:min(len(pairs), split + 500)]
    print(f"Train pairs: {len(train_pairs)}, Diag pairs: {len(diag_pairs)}")

    # DataLoaders
    train_ds = PairDataset(train_pairs)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=2, drop_last=True,
        collate_fn=collate_pairs
    )
    diag_ds = PairDataset(diag_pairs)
    diag_loader = DataLoader(
        diag_ds, batch_size=min(args.batch_size, 64), shuffle=False,
        pin_memory=True, num_workers=0, drop_last=False,
        collate_fn=collate_pairs
    )

    # Model: shared encoder + two separate projection heads
    encoder = GATEncoder(
        in_dim=input_dim, hidden=args.hidden_dim,
        num_layers=args.num_layers, heads=args.heads,
        dropout=args.dropout, edge_dim=edge_dim,
    ).to(device)

    monomer_head = ProjectionHead(
        in_dim=args.hidden_dim, hidden_dim=128, out_dim=128,
    ).to(device)

    multimer_head = ProjectionHead(
        in_dim=args.hidden_dim, hidden_dim=128, out_dim=128,
    ).to(device)

    params = (list(encoder.parameters()) +
              list(monomer_head.parameters()) +
              list(multimer_head.parameters()))
    n_params = sum(p.numel() for p in params)
    print(f"\nTotal parameters: {n_params:,}")
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Monomer head: {sum(p.numel() for p in monomer_head.parameters()):,}")
    print(f"  Multimer head: {sum(p.numel() for p in multimer_head.parameters()):,}")

    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_warmup_cosine_scheduler(
        optimizer, args.warmup_epochs, args.epochs,
        eta_min=1e-6, base_lr=args.lr
    )
    scaler = amp.GradScaler()

    tracker = MetricTracker()
    best_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint if specified
    if args.resume:
        resume_path = args.resume
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(WORK_DIR, resume_path)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        print(f"\nResuming from: {resume_path}")
        ckpt_resume = torch.load(resume_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt_resume['encoder'])
        monomer_head.load_state_dict(ckpt_resume['monomer_head'])
        multimer_head.load_state_dict(ckpt_resume['multimer_head'])
        optimizer.load_state_dict(ckpt_resume['optimizer'])
        if 'scheduler' in ckpt_resume:
            scheduler.load_state_dict(ckpt_resume['scheduler'])
        start_epoch = ckpt_resume['epoch']
        # Restore best_loss from the checkpoint's tracked loss
        if 'loss' in ckpt_resume:
            best_loss = ckpt_resume['loss']
        print(f"  Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")
        del ckpt_resume

    steps_per_epoch = len(train_loader) // args.accum_steps

    print(f"\nSteps per epoch: {steps_per_epoch}")
    print("Starting pretraining...\n")

    for epoch in range(start_epoch, args.epochs):
        encoder.train()
        monomer_head.train()
        multimer_head.train()

        total_loss = 0.0
        total_pos_sim = 0.0
        n_steps = 0

        optimizer.zero_grad()
        acc_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_idx, (batch_1ru, batch_3ru) in enumerate(pbar):
            batch_1ru = batch_1ru.to(device)
            batch_3ru = batch_3ru.to(device)

            effective_bs = batch_1ru.num_graphs
            if effective_bs < args.batch_size:
                continue  # skip incomplete last batch

            with amp.autocast():
                # Encode monomer
                edge_attr_1 = batch_1ru.edge_attr if hasattr(batch_1ru, 'edge_attr') else None
                h_mono = encoder(batch_1ru.x, batch_1ru.edge_index, edge_attr_1, batch_1ru.batch)
                z_mono = monomer_head(h_mono)

                # Encode multimer
                edge_attr_3 = batch_3ru.edge_attr if hasattr(batch_3ru, 'edge_attr') else None
                h_multi = encoder(batch_3ru.x, batch_3ru.edge_index, edge_attr_3, batch_3ru.batch)
                z_multi = multimer_head(h_multi)

                loss, pos_sim, neg_sim_max, stats = nt_xent_loss_with_two_heads(
                    z_mono, z_multi, tau=args.temperature
                )

            # Gradient accumulation
            loss = loss / args.accum_steps
            scaler.scale(loss).backward()
            acc_loss += loss.item() * args.accum_steps
            total_pos_sim += pos_sim

            if (batch_idx + 1) % args.accum_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                total_loss += acc_loss
                n_steps += 1
                acc_loss = 0.0

                pbar.set_postfix({
                    'loss': f'{total_loss / n_steps:.4f}',
                    'pos_sim': f'{pos_sim:.3f}',
                    'neg_max': f'{neg_sim_max:.3f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                })

        scheduler.step()

        avg_loss = total_loss / max(n_steps, 1)
        avg_pos_sim = total_pos_sim / max(n_steps, 1)

        # Diagnostic evaluation
        encoder.eval()
        monomer_head.eval()
        multimer_head.eval()
        diag_loss = 0.0
        diag_pos_sim = 0.0
        diag_neg_max = 0.0
        diag_n = 0
        with torch.no_grad():
            for batch_1ru, batch_3ru in diag_loader:
                batch_1ru = batch_1ru.to(device)
                batch_3ru = batch_3ru.to(device)
                with amp.autocast():
                    e1 = batch_1ru.edge_attr if hasattr(batch_1ru, 'edge_attr') else None
                    e3 = batch_3ru.edge_attr if hasattr(batch_3ru, 'edge_attr') else None
                    h_m = encoder(batch_1ru.x, batch_1ru.edge_index, e1, batch_1ru.batch)
                    h_3 = encoder(batch_3ru.x, batch_3ru.edge_index, e3, batch_3ru.batch)
                    z_m = monomer_head(h_m)
                    z_3 = multimer_head(h_3)
                    dl, ps, nm, _ = nt_xent_loss_with_two_heads(z_m, z_3, tau=args.temperature)
                diag_loss += dl.item()
                diag_pos_sim += ps
                diag_neg_max += nm
                diag_n += 1

        diag_loss /= max(diag_n, 1)
        diag_pos_sim /= max(diag_n, 1)
        diag_neg_max /= max(diag_n, 1)

        tracker.add(
            epoch + 1,
            loss=avg_loss,
            pos_sim_mean=diag_pos_sim,
            neg_sim_max=diag_neg_max,
            diag_loss=diag_loss,
            grad_norm=grad_norm if 'grad_norm' in dir() else 0.0,
        )

        print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}, diag_loss={diag_loss:.4f}, "
              f"pos_sim={diag_pos_sim:.3f}, neg_max={diag_neg_max:.3f}")

        # Collapse check
        if diag_pos_sim > 0.95:
            print(f"  WARNING: Positive similarity {diag_pos_sim:.3f} near collapse threshold")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt = {
                'epoch': epoch + 1,
                'encoder': encoder.state_dict(),
                'monomer_head': monomer_head.state_dict(),
                'multimer_head': multimer_head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': avg_loss,
            }
            ckpt_path = os.path.join(checkpoint_dir, f'pretrain_epoch{epoch + 1}.pt')
            torch.save(ckpt, ckpt_path)
            print(f"  Saved: {ckpt_path}")

        # Track best
        if diag_loss < best_loss:
            best_loss = diag_loss
            best_ckpt = {
                'epoch': epoch + 1,
                'encoder': encoder.state_dict(),
                'monomer_head': monomer_head.state_dict(),
                'multimer_head': multimer_head.state_dict(),
                'loss': avg_loss,
            }
            best_path = os.path.join(checkpoint_dir, 'best_pretrain.pt')
            torch.save(best_ckpt, best_path)

    # Save final
    final_ckpt = {
        'epoch': args.epochs,
        'encoder': encoder.state_dict(),
        'monomer_head': monomer_head.state_dict(),
        'multimer_head': multimer_head.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss': avg_loss,
    }
    torch.save(final_ckpt, os.path.join(checkpoint_dir, 'final_pretrain.pt'))

    # Save metrics and plot
    tracker.save(os.path.join(results_dir, 'pretrain_metrics.json'))
    plot_pretrain_diagnostics(tracker, results_dir)

    print("\n=== Pretraining Complete ===")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final pos_sim: {diag_pos_sim:.3f}")
    print(f"Final neg_max: {diag_neg_max:.3f}")

    # Final diagnostics
    if diag_pos_sim > 0.95:
        print("WARNING: Possible embedding collapse detected (pos_sim > 0.95)")
    elif diag_pos_sim < 0.05:
        print("WARNING: No learning detected (pos_sim < 0.05)")
    else:
        print("OK: Positive similarity in healthy range (0.05-0.95)")


if __name__ == '__main__':
    main()
