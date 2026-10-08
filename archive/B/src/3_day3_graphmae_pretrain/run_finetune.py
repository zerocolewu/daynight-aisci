#!/usr/bin/env python3
"""
Stages 3-5: Finetuning + Baseline on 3-RU Multimer Graphs.

Uses cached 3-RU graphs and pretrained encoder from Stage 2.
Does NOT rebuild graphs or rerun pretraining.
"""

import os
import sys
import json
import pickle
import warnings
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, BatchNorm, global_mean_pool
from torch.utils.data import Dataset as TorchDataset

from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration (matching train.py exactly)
# ============================================================================
class Config:
    work_dir = "/home/home_public/aisci/AIscientist/src/3_day3_graphmae_pretrain"
    cache_dir = os.path.join(work_dir, "cache")
    pi1m_cache = os.path.join(cache_dir, "pi1m_50k_graphs_3ru.pkl")
    tg_cache = os.path.join(cache_dir, "tg_graphs_3ru.pkl")
    pretrain_ckpt = os.path.join(work_dir, "best_pretrained_encoder.pt")

    finetune_json = os.path.join(work_dir, "finetune_results.json")
    finetune_curves_png = os.path.join(work_dir, "finetune_curves.png")
    diagnostic_txt = os.path.join(work_dir, "diagnostic_analysis.txt")
    pretrain_loss_csv = os.path.join(work_dir, "pretrain_loss.csv")

    rwpe_k = 32
    rich_feat_dim = 13
    total_input_dim = 45

    gat_hidden = 256
    gat_heads = 4
    gat_layers = 4
    gat_dropout = 0.1

    predictor_hidden = [256, 128]
    predictor_output = 1

    tg_split_ratios = [0.72, 0.08, 0.20]
    tg_split_seed = 71
    ft_batch_size = 32
    ft_max_epochs = 300
    ft_patience = 30
    ft_lr = 3e-4
    ft_weight_decay = 1e-5

    baseline_seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()


# ============================================================================
# Models (matching train.py exactly)
# ============================================================================

class GATEncoderWithResidual(nn.Module):
    def __init__(self, input_dim=45, hidden_dim=256, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.residual_proj = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim * num_heads
            concat = (i < num_layers - 1)
            out_dim_actual = hidden_dim * num_heads if concat else hidden_dim

            self.convs.append(GATConv(in_dim, hidden_dim, heads=num_heads, concat=concat, dropout=dropout))
            self.bns.append(BatchNorm(out_dim_actual))

            if i > 0 and in_dim != out_dim_actual:
                self.residual_proj.append(nn.Linear(in_dim, out_dim_actual, bias=False))
            else:
                self.residual_proj.append(nn.Identity())

        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch=None):
        for i in range(self.num_layers):
            identity = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i > 0:
                identity = self.residual_proj[i](identity)
                x = x + identity
        return x


class GATPredictor(nn.Module):
    def __init__(self, encoder, encoder_output_dim=256,
                 predictor_hidden=[256, 128], predictor_output=1):
        super().__init__()
        self.encoder = encoder
        layers = []
        in_dim = encoder_output_dim
        for h in predictor_hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, predictor_output))
        self.predictor = nn.Sequential(*layers)

    def forward(self, x, edge_index, batch):
        h = self.encoder(x, edge_index)
        h_pooled = global_mean_pool(h, batch)
        out = self.predictor(h_pooled)
        return out.squeeze(-1)


# ============================================================================
# Dataset
# ============================================================================

class LabeledGraphDataset(TorchDataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx].clone()
        data.y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return data


# ============================================================================
# Training Utilities
# ============================================================================

def create_tg_split(graphs, labels):
    indices = np.arange(len(graphs))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.28, random_state=cfg.tg_split_seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=20.0 / 28.0, random_state=cfg.tg_split_seed
    )

    train_graphs = [graphs[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    return (train_graphs, train_labels), (val_graphs, val_labels), (test_graphs, test_labels)


def train_finetune(model, train_loader, val_loader, optimizer, max_epochs, patience, label=""):
    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0

    train_loss_hist = []
    val_rmse_hist = []
    train_rmse_hist = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        train_preds = []
        train_targets = []

        for batch in train_loader:
            batch = batch.to(cfg.device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.batch)
            loss = F.mse_loss(pred, batch.y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
            train_preds.append(pred.detach().cpu())
            train_targets.append(batch.y.squeeze().detach().cpu())

        avg_train_loss = total_loss / total_samples
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_rmse = torch.sqrt(F.mse_loss(train_preds, train_targets)).item()

        train_loss_hist.append(avg_train_loss)
        train_rmse_hist.append(train_rmse)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(cfg.device)
                pred = model(batch.x, batch.edge_index, batch.batch)
                val_preds.append(pred.cpu())
                val_targets.append(batch.y.squeeze().cpu())

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_rmse = torch.sqrt(F.mse_loss(val_preds, val_targets)).item()
        val_rmse_hist.append(val_rmse)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  {label} Epoch {epoch:3d} | Train RMSE: {train_rmse:.2f}K | Val RMSE: {val_rmse:.2f}K")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  {label} Early stopping at epoch {epoch} (best val={best_val_rmse:.2f}K at epoch {best_epoch})")
                break

    if epoch == max_epochs:
        print(f"  {label} Max epochs reached (best val={best_val_rmse:.2f}K at epoch {best_epoch})")

    model.load_state_dict(best_state)
    return (train_loss_hist, train_rmse_hist, val_rmse_hist,
            best_val_rmse, best_epoch)


def evaluate(model, test_loader):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(cfg.device)
            pred = model(batch.x, batch.edge_index, batch.batch)
            preds.append(pred.cpu())
            targets.append(batch.y.squeeze().cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    rmse = torch.sqrt(F.mse_loss(preds, targets)).item()
    mae = F.l1_loss(preds, targets).item()
    r2 = 1 - F.mse_loss(preds, targets) / torch.var(targets)
    return rmse, mae, r2.item(), preds.numpy(), targets.numpy()


def create_encoder():
    return GATEncoderWithResidual(
        input_dim=cfg.total_input_dim,
        hidden_dim=cfg.gat_hidden,
        num_heads=cfg.gat_heads,
        num_layers=cfg.gat_layers,
        dropout=cfg.gat_dropout,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("GRAPHMAE FINETUNING + BASELINE (3-RU Multimer Graphs)")
    print("=" * 60)
    print(f"Device: {cfg.device}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Loading 3-RU graphs and pretrained encoder...")
    print("-" * 40)

    with open(cfg.tg_cache, 'rb') as f:
        tg_graphs, tg_labels = pickle.load(f)
    print(f"  Loaded {len(tg_graphs)} Tg graphs ({len(tg_labels)} labels)")

    # Graph statistics
    n_nodes = [g.num_nodes for g in tg_graphs]
    print(f"  Tg avg nodes: {np.mean(n_nodes):.1f}, min={min(n_nodes)}, max={max(n_nodes)}")
    print(f"  Tg label range: [{min(tg_labels):.1f}, {max(tg_labels):.1f}]")

    # Validate pretrained encoder
    print(f"\n  Loading pretrained encoder from {cfg.pretrain_ckpt}")
    state = torch.load(cfg.pretrain_ckpt, map_location='cpu')
    print(f"  Encoder keys: {len(state)}, e.g. first layer weight: {list(state.values())[0].shape}")

    # Load pretrain loss for diagnostics
    pretrain_loss_hist = []
    if os.path.exists(cfg.pretrain_loss_csv):
        import csv
        with open(cfg.pretrain_loss_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pretrain_loss_hist.append(float(row['loss']))
    if pretrain_loss_hist:
        print(f"  Pretraining: {len(pretrain_loss_hist)} epochs, "
              f"initial loss={pretrain_loss_hist[0]:.6f}, final={pretrain_loss_hist[-1]:.6f}")

    # Create splits
    print("\nCreating Tg train/val/test split (seed=71, 72/8/20)...")
    (train_g, train_l), (val_g, val_l), (test_g, test_l) = create_tg_split(tg_graphs, tg_labels)

    tg_train_dataset = LabeledGraphDataset(train_g, train_l)
    tg_val_dataset = LabeledGraphDataset(val_g, val_l)
    tg_test_dataset = LabeledGraphDataset(test_g, test_l)

    ft_train_loader = DataLoader(tg_train_dataset, batch_size=cfg.ft_batch_size, shuffle=True)
    ft_val_loader = DataLoader(tg_val_dataset, batch_size=cfg.ft_batch_size, shuffle=False)
    ft_test_loader = DataLoader(tg_test_dataset, batch_size=cfg.ft_batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # Stage 3: Finetuning (Pretrained Encoder)
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("STAGE 3: FINETUNING (PRETRAINED ENCODER)")
    print("-" * 40)

    torch.manual_seed(cfg.tg_split_seed)
    np.random.seed(cfg.tg_split_seed)

    ft_encoder = create_encoder()
    ft_encoder.load_state_dict(state)
    ft_encoder.to(cfg.device)

    ft_model = GATPredictor(
        ft_encoder,
        encoder_output_dim=cfg.gat_hidden,
        predictor_hidden=cfg.predictor_hidden,
        predictor_output=cfg.predictor_output,
    ).to(cfg.device)

    ft_optimizer = AdamW(ft_model.parameters(), lr=cfg.ft_lr, weight_decay=cfg.ft_weight_decay)

    print(f"  Parameters: {sum(p.numel() for p in ft_model.parameters()):,}")
    print(f"  LR: {cfg.ft_lr}, WD: {cfg.ft_weight_decay}, Batch: {cfg.ft_batch_size}")

    (ft_train_loss, ft_train_rmse, ft_val_rmse,
     ft_best_val_rmse, ft_best_epoch) = train_finetune(
        ft_model, ft_train_loader, ft_val_loader,
        ft_optimizer, cfg.ft_max_epochs, cfg.ft_patience,
        label="[FT]"
    )

    ft_test_rmse, ft_test_mae, ft_test_r2, ft_test_preds, ft_test_targets = evaluate(ft_model, ft_test_loader)
    print(f"\nPretrained Test RMSE: {ft_test_rmse:.2f}K, MAE: {ft_test_mae:.2f}K, R2: {ft_test_r2:.4f}")

    # ------------------------------------------------------------------
    # Stage 4: Baseline (Random Init)
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("STAGE 4: BASELINE (RANDOM INIT)")
    print("-" * 40)

    torch.manual_seed(cfg.baseline_seed)
    np.random.seed(cfg.baseline_seed)

    bl_encoder = create_encoder().to(cfg.device)
    bl_model = GATPredictor(
        bl_encoder,
        encoder_output_dim=cfg.gat_hidden,
        predictor_hidden=cfg.predictor_hidden,
        predictor_output=cfg.predictor_output,
    ).to(cfg.device)

    bl_optimizer = AdamW(bl_model.parameters(), lr=cfg.ft_lr, weight_decay=cfg.ft_weight_decay)

    print(f"  Parameters: {sum(p.numel() for p in bl_model.parameters()):,}")
    print(f"  LR: {cfg.ft_lr}, WD: {cfg.ft_weight_decay}, Batch: {cfg.ft_batch_size}")

    (bl_train_loss, bl_train_rmse, bl_val_rmse,
     bl_best_val_rmse, bl_best_epoch) = train_finetune(
        bl_model, ft_train_loader, ft_val_loader,
        bl_optimizer, cfg.ft_max_epochs, cfg.ft_patience,
        label="[BL]"
    )

    bl_test_rmse, bl_test_mae, bl_test_r2, bl_test_preds, bl_test_targets = evaluate(bl_model, ft_test_loader)
    print(f"\nBaseline Test RMSE: {bl_test_rmse:.2f}K, MAE: {bl_test_mae:.2f}K, R2: {bl_test_r2:.4f}")

    # ------------------------------------------------------------------
    # Stage 5: Results & Analysis
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("STAGE 5: RESULTS & ANALYSIS")
    print("-" * 40)

    improvement_k = bl_test_rmse - ft_test_rmse
    pct_improvement = (improvement_k / bl_test_rmse) * 100 if bl_test_rmse > 0 else 0

    results = {
        "graph_construction": "3-RU multimer",
        "pretrained_test_rmse": round(ft_test_rmse, 4),
        "baseline_test_rmse": round(bl_test_rmse, 4),
        "pretrained_best_val_rmse": round(ft_best_val_rmse, 4),
        "baseline_best_val_rmse": round(bl_best_val_rmse, 4),
        "improvement_k": round(improvement_k, 4),
        "improvement_pct": round(pct_improvement, 2),
        "pretrained_test_mae": round(ft_test_mae, 4),
        "baseline_test_mae": round(bl_test_mae, 4),
        "pretrained_test_r2": round(ft_test_r2, 4),
        "baseline_test_r2": round(bl_test_r2, 4),
        "pretrained_epochs_to_best": ft_best_epoch,
        "baseline_epochs_to_best": bl_best_epoch,
        "expected_baseline_k": 35.94,
        "baseline_deviation_k": round(bl_test_rmse - 35.94, 4),
    }

    with open(cfg.finetune_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {cfg.finetune_json}")

    # Plot finetuning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pretrained
    ax = axes[0]
    epochs = range(1, len(ft_train_rmse) + 1)
    ax.plot(epochs, ft_train_rmse, 'b-', alpha=0.5, label='Train RMSE')
    ax.plot(epochs, ft_val_rmse, 'r-', label='Val RMSE')
    ax.axvline(x=ft_best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best epoch {ft_best_epoch}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE (K)')
    ax.set_title('Pretrained GAT Finetuning')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Baseline
    ax = axes[1]
    epochs = range(1, len(bl_train_rmse) + 1)
    ax.plot(epochs, bl_train_rmse, 'b-', alpha=0.5, label='Train RMSE')
    ax.plot(epochs, bl_val_rmse, 'r-', label='Val RMSE')
    ax.axvline(x=bl_best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best epoch {bl_best_epoch}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE (K)')
    ax.set_title('Baseline GAT (Random Init)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'3-RU Multimer | Pretrained vs Baseline | '
                 f'Baseline={bl_test_rmse:.1f}K | \u0394RMSE = {improvement_k:+.2f}K ({pct_improvement:+.1f}%)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(cfg.finetune_curves_png, dpi=120)
    plt.close()
    print(f"Finetuning curves saved to {cfg.finetune_curves_png}")

    # Diagnostic Analysis
    print("\nWriting diagnostic analysis...")
    diagnostic_lines = []
    diagnostic_lines.append("=" * 60)
    diagnostic_lines.append("GRAPHMAE PRETRAINING DIAGNOSTIC ANALYSIS (3-RU Multimer)")
    diagnostic_lines.append("=" * 60)
    diagnostic_lines.append("")

    diagnostic_lines.append("--- Critical Fix ---")
    diagnostic_lines.append("Previous iteration used cyclization-based graph construction")
    diagnostic_lines.append("(connect * neighbors, remove * atoms) which caused a 4K RMSE")
    diagnostic_lines.append("deviation (39.94K vs expected 35.94K).")
    diagnostic_lines.append("This iteration uses 3-RU multimer construction (identical to Day 2),")
    diagnostic_lines.append("which replicates the monomer 3x, connects via * attachment points,")
    diagnostic_lines.append("removes internal *, and caps ends with H.")
    diagnostic_lines.append("")

    diagnostic_lines.append("--- Baseline Validation ---")
    deviation = bl_test_rmse - 35.94
    if abs(deviation) <= 1.0:
        diagnostic_lines.append(f"PASS: Baseline {bl_test_rmse:.2f}K is within +-1K of expected 35.94K")
        diagnostic_lines.append(f"  (deviation = {deviation:+.2f}K). Graph construction matches Day 2 pipeline.")
    else:
        diagnostic_lines.append(f"WARNING: Baseline {bl_test_rmse:.2f}K deviates by {deviation:+.2f}K from expected 35.94K")
        diagnostic_lines.append(f"  Possible causes: feature definition differences, architecture differences,")
        diagnostic_lines.append(f"  or dataset processing variations.")
    diagnostic_lines.append("")

    diagnostic_lines.append("--- Results Summary ---")
    diagnostic_lines.append(f"Pretrained Test RMSE:  {ft_test_rmse:.2f} K")
    diagnostic_lines.append(f"Baseline Test RMSE:    {bl_test_rmse:.2f} K")
    diagnostic_lines.append(f"Improvement:           {improvement_k:.2f} K ({pct_improvement:+.1f}%)")
    diagnostic_lines.append(f"Pretrained Best Val:   {ft_best_val_rmse:.2f} K (epoch {ft_best_epoch})")
    diagnostic_lines.append(f"Baseline Best Val:     {bl_best_val_rmse:.2f} K (epoch {bl_best_epoch})")
    diagnostic_lines.append(f"Pretrained Test R2:    {ft_test_r2:.4f}")
    diagnostic_lines.append(f"Baseline Test R2:      {bl_test_r2:.4f}")
    diagnostic_lines.append("")

    diagnostic_lines.append("--- Convergence Analysis ---")
    if ft_best_epoch < bl_best_epoch:
        diagnostic_lines.append(f"Pretrained model converged {bl_best_epoch - ft_best_epoch} epochs faster "
                                f"({ft_best_epoch} vs {bl_best_epoch}), suggesting better initialization.")
    else:
        diagnostic_lines.append(f"Pretrained model converged after {ft_best_epoch} epochs "
                                f"(baseline: {bl_best_epoch}).")
    diagnostic_lines.append("")

    diagnostic_lines.append("--- Hypothesis Check ---")
    if improvement_k >= 1.5:
        diagnostic_lines.append(f"HYPOTHESIS SUPPORTED: Pretraining improved RMSE by {improvement_k:.2f}K "
                                f"(>= 1.5K threshold).")
    elif improvement_k > 0:
        diagnostic_lines.append(f"HYPOTHESIS PARTIALLY SUPPORTED: Small improvement of {improvement_k:.2f}K "
                                f"(< 1.5K threshold), but direction is correct.")
    else:
        diagnostic_lines.append(f"HYPOTHESIS NOT SUPPORTED: Pretraining did not improve RMSE "
                                f"({improvement_k:+.2f}K).")
    diagnostic_lines.append("")

    diagnostic_lines.append("--- Pretraining Loss Curve ---")
    if pretrain_loss_hist:
        diagnostic_lines.append(f"Initial loss: {pretrain_loss_hist[0]:.6f}")
        diagnostic_lines.append(f"Final loss:   {pretrain_loss_hist[-1]:.6f}")
        reduction = ((pretrain_loss_hist[0] - pretrain_loss_hist[-1]) / pretrain_loss_hist[0] * 100
                     if pretrain_loss_hist[0] > 0 else 0)
        diagnostic_lines.append(f"Reduction:    {reduction:.1f}%")
    diagnostic_lines.append("")

    diagnostic_lines.append("--- Weight Transfer Analysis ---")
    original_pretrained = torch.load(cfg.pretrain_ckpt, map_location='cpu')
    weight_change_total = 0.0
    param_count = 0
    for key in original_pretrained:
        if hasattr(ft_encoder, 'state_dict'):
            pass
        if key in state:
            diff = (ft_encoder.state_dict()[key].cpu() - original_pretrained[key]).norm().item()
            orig_norm = original_pretrained[key].norm().item()
            rel_change = diff / max(orig_norm, 1e-8) * 100
            if param_count < 12:  # show first 12 layers
                diagnostic_lines.append(f"  {key}: orig_norm={orig_norm:.2f}, delta={diff:.2f} ({rel_change:.1f}%)")
            weight_change_total += rel_change
            param_count += 1
    if param_count > 0:
        diagnostic_lines.append(f"  Average relative change: {weight_change_total/param_count:.1f}% "
                                f"(over {param_count} parameters)")
    diagnostic_lines.append("")

    diagnostic_lines.append("--- Data Statistics ---")
    diagnostic_lines.append(f"Graph construction: 3-RU multimer")
    diagnostic_lines.append(f"Rich feature dim: {cfg.rich_feat_dim}")
    diagnostic_lines.append(f"Total input dim: {cfg.total_input_dim} ({cfg.rich_feat_dim} rich + {cfg.rwpe_k} RWPE)")
    diagnostic_lines.append(f"Tg total graphs: {len(tg_graphs)}")
    diagnostic_lines.append(f"Tg train: {len(train_g)}, val: {len(val_g)}, test: {len(test_g)}")

    tg_nodes = [g.num_nodes for g in tg_graphs]
    diagnostic_lines.append(f"Tg avg nodes/graph: {np.mean(tg_nodes):.1f}, "
                            f"min={min(tg_nodes)}, max={max(tg_nodes)}")
    diagnostic_lines.append(f"Tg target range: [{min(tg_labels):.1f}, {max(tg_labels):.1f}]")

    # Load PI1M info if available
    if os.path.exists(cfg.pi1m_cache):
        with open(cfg.pi1m_cache, 'rb') as f:
            pi1m_graphs = pickle.load(f)
        pi1m_nodes = [g.num_nodes for g in pi1m_graphs]
        diagnostic_lines.append(f"PI1M pretraining graphs: {len(pi1m_graphs)}")
        diagnostic_lines.append(f"PI1M avg nodes/graph: {np.mean(pi1m_nodes):.1f}, "
                                f"min={min(pi1m_nodes)}, max={max(pi1m_nodes)}")

    diagnostic_text = "\n".join(diagnostic_lines)
    with open(cfg.diagnostic_txt, 'w') as f:
        f.write(diagnostic_text)
    print(diagnostic_text)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Results: {cfg.finetune_json}")
    print(f"Curves:  {cfg.finetune_curves_png}")
    print(f"Diagnostics: {cfg.diagnostic_txt}")


if __name__ == "__main__":
    main()
