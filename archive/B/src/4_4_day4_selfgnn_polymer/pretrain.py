"""Pretraining script: Siamese GAT with Split+Standardize augmentations (REWRITE).

Self-supervised pretraining on PI1M (200K 1-RU polymers):
  - 42-dim input (8 rich + 2 PSPE + 32 RWPE) — same as finetuning
  - Siamese GAT: 4 layers, 256 hidden, 4 heads, BatchNorm
  - Student = GAT + prediction MLP g (256→128→256)
  - Teacher = GAT (EMA τ=0.999w, no grad)
  - Augmentations: Split (feature dim) + Standardize (z-score norm)
  - Loss: MSE cosine similarity = 2 - 2*cos(z_student, z_teacher)
  - Diagnostics every 10 epochs
"""

import os
import sys
import time
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
from torch_geometric.nn import global_mean_pool
from torch.utils.data import Subset

from data_utils import (
    PI1MDataset, augment_split, augment_standardize, augment_combined,
)
from models import StudentModel, TeacherModel


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_similarity_loss(z1, z2):
    """MSE cosine similarity loss: mean(2 - 2*cos(z1_i, z2_i))."""
    z1_norm = F.normalize(z1, p=2, dim=-1)
    z2_norm = F.normalize(z2, p=2, dim=-1)
    cos_sim = (z1_norm * z2_norm).sum(dim=-1)
    return (2.0 - 2.0 * cos_sim).mean()


def global_pool(x, batch):
    """Mean pool node embeddings to graph level."""
    return global_mean_pool(x, batch)


@torch.no_grad()
def compute_embedding_stats(student, teacher, loader, device):
    """Compute embedding statistics: mean cosine similarity across all graphs."""
    student.eval()
    teacher.eval()

    all_cos_sims = []
    all_norms_student = []
    all_norms_teacher = []

    for data in loader:
        data = data.to(device)

        z_s = student.encode(data.x, data.edge_index, data.edge_attr, data.batch)
        z_t = teacher(data.x, data.edge_index, data.edge_attr, data.batch)

        batch = data.batch
        z_s_graph = global_pool(z_s, batch)
        z_t_graph = global_pool(z_t, batch)

        z_s_norm = F.normalize(z_s_graph, p=2, dim=-1)
        z_t_norm = F.normalize(z_t_graph, p=2, dim=-1)
        cos = (z_s_norm * z_t_norm).sum(dim=-1)

        all_cos_sims.append(cos.detach().cpu().numpy())
        all_norms_student.append(z_s_graph.norm(dim=-1).detach().cpu().numpy())
        all_norms_teacher.append(z_t_graph.norm(dim=-1).detach().cpu().numpy())

    all_cos = np.concatenate(all_cos_sims)
    all_ns = np.concatenate(all_norms_student)
    all_nt = np.concatenate(all_norms_teacher)

    return {
        'cos_mean': float(all_cos.mean()),
        'cos_std': float(all_cos.std()),
        'cos_min': float(all_cos.min()),
        'cos_max': float(all_cos.max()),
        'cos_median': float(np.median(all_cos)),
        'norm_student_mean': float(all_ns.mean()),
        'norm_student_std': float(all_ns.std()),
        'norm_teacher_mean': float(all_nt.mean()),
        'norm_teacher_std': float(all_nt.std()),
        'collapse_warning': bool(all_cos.std() < 0.01 or all_cos.mean() > 0.99),
    }


def pretrain(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    dataset = PI1MDataset(
        csv_path=args.pi1m_path,
        n_samples=args.n_samples,
        seed=args.seed,
        root=args.data_root,
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Diagnostic loader (small subset)
    diag_n = min(500, len(dataset))
    diag_subset = Subset(dataset, range(diag_n))
    diag_loader = DataLoader(
        diag_subset, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )

    print(f"Dataset size: {len(dataset)}, Batches/epoch: {len(loader)}")

    # ── Models (42-dim input, same as finetuning) ────────────────────────────
    use_bn = not args.no_batchnorm
    student = StudentModel(
        in_dim=42, hidden_dim=args.hidden_dim, heads=args.heads,
        use_bn=use_bn,
    ).to(device)
    teacher = TeacherModel(
        in_dim=42, hidden_dim=args.hidden_dim, heads=args.heads,
        use_bn=use_bn,
    ).to(device)

    # Copy student weights to teacher
    teacher.encoder.load_state_dict(student.encoder.state_dict())

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # ── Augmentation selection ───────────────────────────────────────────────
    aug_map = {
        'split': augment_split,
        'standardize': augment_standardize,
        'combined': augment_combined,
    }
    augment_fn = aug_map[args.augmentation]

    # ── Training loop ────────────────────────────────────────────────────────
    best_loss = float('inf')
    patience_counter = 0
    history = {'epoch': [], 'loss': [], 'diag': []}

    print(f"\n{'='*60}")
    print(f"Pretraining: {args.augmentation} augmentation")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Patience: {args.patience}, BN: {use_bn}")
    print(f"Input dim: 42 (8 rich + 2 PSPE + 32 RWPE)")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_losses = []
        t0 = time.time()

        for batch_idx, data in enumerate(loader):
            data = data.to(device)

            # Generate two augmented views
            view1, view2 = augment_fn(data)

            # Student encodes both views and predicts
            z1_s = student(view1.x, view1.edge_index, view1.edge_attr, view1.batch)
            z2_s = student(view2.x, view2.edge_index, view2.edge_attr, view2.batch)

            # Teacher encodes both views (no grad, EMA)
            with torch.no_grad():
                z1_t = teacher(view1.x, view1.edge_index, view1.edge_attr, view1.batch)
                z2_t = teacher(view2.x, view2.edge_index, view2.edge_attr, view2.batch)

            # Loss: predict cross-views
            loss1 = cosine_similarity_loss(z1_s, z2_t)
            loss2 = cosine_similarity_loss(z2_s, z1_t)
            loss = (loss1 + loss2) / 2.0

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            # EMA update teacher
            teacher.update_ema(student.encoder, tau=args.ema_tau)

            epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        history['epoch'].append(epoch)
        history['loss'].append(float(avg_loss))

        t1 = time.time()
        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f} | Time: {t1-t0:.1f}s")

        # ── Diagnostics every 10 epochs ──────────────────────────────────────
        if epoch % 10 == 0:
            diag = compute_embedding_stats(student, teacher, diag_loader, device)
            history['diag'].append({'epoch': epoch, **diag})
            print(f"  Diag | cos_mean={diag['cos_mean']:.4f} cos_std={diag['cos_std']:.4f} "
                  f"cos_median={diag['cos_median']:.4f} n_s={diag['norm_student_mean']:.2f} "
                  f"n_t={diag['norm_teacher_mean']:.2f}")
            if diag['collapse_warning']:
                print(f"  *** COLLAPSE WARNING: low cosine diversity ***")

        # ── Early stopping ───────────────────────────────────────────────────
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
            }, os.path.join(args.save_dir, 'pretrain_best.pt'))
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # ── Save final model and history ─────────────────────────────────────────
    torch.save({
        'student_state_dict': student.state_dict(),
        'teacher_state_dict': teacher.state_dict(),
        'epoch': epoch,
    }, os.path.join(args.save_dir, 'pretrain_final.pt'))

    with open(os.path.join(args.save_dir, 'pretrain_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nPretraining complete. Best loss: {best_loss:.4f}")
    return best_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pi1m_path', type=str,
                        default='/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv')
    parser.add_argument('--data_root', type=str, default='./data_pretrain')
    parser.add_argument('--save_dir', type=str, default='./pretrain_output')
    parser.add_argument('--n_samples', type=int, default=200000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--ema_tau', type=float, default=0.999)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--augmentation', type=str, default='combined',
                        choices=['split', 'standardize', 'combined'])
    parser.add_argument('--no_batchnorm', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    pretrain(args)
