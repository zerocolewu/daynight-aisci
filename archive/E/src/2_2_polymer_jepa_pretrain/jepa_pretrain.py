"""
JEPA (Joint Embedding Predictive Architecture) pre-training on PI1M Trimer graphs.

Iteration 3: Fix within-graph node collapse + architecture improvements.

Key changes from v2:
1. Reduced dropout from 0.5 to 0.1 (0.5 was killing node-level information)
2. Use mean pooling instead of sum pooling (robust to graph size)
3. Add node-level VICReg loss (penalize within-graph node collapse)
4. Add projection head with L2 normalization
5. Keep all v2 fixes (VICReg, EMA ramping, L2 norm targets, predictor weight decay)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np
import os
import logging
import time
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# GIN Encoder
# ============================================================

class GINEncoder(nn.Module):
    """GIN encoder with 5 layers, reduced dropout."""

    def __init__(self, in_dim, hidden_dim=256, out_dim=256, num_layers=5, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                )
            else:
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch, edge_attr=None):
        """Forward pass, returns node-level embeddings."""
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x  # node-level embeddings


# ============================================================
# Projection Head (for VICReg loss on projected space)
# ============================================================

class ProjectionHead(nn.Module):
    """MLP projection head for VICReg loss computation."""

    def __init__(self, in_dim=256, hidden_dim=256, out_dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.head(x)


# ============================================================
# JEPA Predictor
# ============================================================

class JEPAPredictor(nn.Module):
    """Predictor: context_embedding + positional_info -> predicted target embedding."""

    def __init__(self, context_dim=256, pos_dim=3, hidden_dim=256):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(context_dim + pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, context_emb, positional_info):
        x = torch.cat([context_emb, positional_info], dim=-1)
        return self.predictor(x)


# ============================================================
# VICReg Loss Functions
# ============================================================

def variance_loss(z, eps=1e-4, gamma=1.0):
    """VICReg variance term."""
    std_z = torch.sqrt(z.var(dim=0) + eps)
    loss = torch.mean(torch.relu(gamma - std_z))
    return loss


def covariance_loss(z):
    """VICReg covariance term."""
    n, d = z.shape
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / n
    diag_mask = torch.eye(d, device=z.device, dtype=torch.bool)
    off_diag = cov[~diag_mask]
    loss = off_diag.pow(2).sum() / d
    return loss


def node_diversity_loss(node_emb, batch, eps=1e-4):
    """Penalize within-graph node collapse.

    For each graph, compute the mean pairwise cosine distance between
    node embeddings. Penalize if nodes within a graph are too similar.

    Args:
        node_emb: (total_nodes, embed_dim) node embeddings
        batch: (total_nodes,) graph assignment vector

    Returns:
        scalar loss encouraging diverse node embeddings within each graph
    """
    unique_graphs = batch.unique()
    total_loss = 0.0
    count = 0

    for g_id in unique_graphs:
        mask = batch == g_id
        g_nodes = node_emb[mask]  # (n_nodes_g, dim)

        if g_nodes.shape[0] < 2:
            continue

        # Compute per-dimension variance across nodes within this graph
        var_per_dim = g_nodes.var(dim=0)  # (dim,)
        # Penalize low variance (node collapse)
        std_per_dim = torch.sqrt(var_per_dim + eps)
        loss = torch.mean(torch.relu(0.1 - std_per_dim))  # threshold 0.1 per dim
        total_loss += loss
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=node_emb.device)

    return total_loss / count


# ============================================================
# JEPA Masking Utilities
# ============================================================

def create_jepa_masks_for_graph(edge_index_cpu, n_nodes, x_cpu, context_ratio=0.6, target_ratio=0.1):
    """Create context and target masks for a single graph."""
    edge_index_np = edge_index_cpu.numpy()

    adj = [[] for _ in range(n_nodes)]
    for e in range(edge_index_np.shape[1]):
        u, v = int(edge_index_np[0, e]), int(edge_index_np[1, e])
        if u != v:
            adj[u].append(v)

    n_context = max(1, int(n_nodes * context_ratio))
    context_mask = np.zeros(n_nodes, dtype=bool)

    start = np.random.randint(0, n_nodes)
    queue = deque([start])
    context_mask[start] = True
    count = 1

    while queue and count < n_context:
        node = queue.popleft()
        neighbors = adj[node]
        if len(neighbors) > 1:
            perm = np.random.permutation(len(neighbors))
            neighbors = [neighbors[p] for p in perm]
        for nbr in neighbors:
            if not context_mask[nbr] and count < n_context:
                context_mask[nbr] = True
                queue.append(nbr)
                count += 1

    if count < n_context:
        remaining = np.where(~context_mask)[0]
        if len(remaining) > 0:
            extra = np.random.choice(remaining, min(n_context - count, len(remaining)), replace=False)
            context_mask[extra] = True

    non_context = ~context_mask
    non_context_indices = np.where(non_context)[0]
    n_target = max(1, int(n_nodes * target_ratio))

    if len(non_context_indices) <= n_target:
        target_mask = non_context.copy()
    else:
        target_indices = np.random.choice(non_context_indices, n_target, replace=False)
        target_mask = np.zeros(n_nodes, dtype=bool)
        target_mask[target_indices] = True

    ru_features = x_cpu[:, -3:].numpy()
    ru_assignment = ru_features.argmax(axis=1)

    target_indices_np = np.where(target_mask)[0]
    if len(target_indices_np) > 0:
        target_rus = ru_assignment[target_indices_np]
        pos_info = np.array([
            np.sum(target_rus == ru) / max(len(target_indices_np), 1)
            for ru in range(3)
        ], dtype=np.float32)
    else:
        pos_info = np.array([1.0/3, 1.0/3, 1.0/3], dtype=np.float32)

    return context_mask, target_mask, pos_info


# ============================================================
# EMA Update
# ============================================================

def update_ema(student_model, teacher_model, momentum=0.996):
    """Update teacher parameters via EMA of student parameters."""
    for param_s, param_t in zip(student_model.parameters(), teacher_model.parameters()):
        param_t.data.mul_(momentum).add_(param_s.data, alpha=1 - momentum)


# ============================================================
# Collapse Monitoring
# ============================================================

def check_collapse(embeddings, node_emb=None, batch=None, threshold=0.01):
    """Check for representation collapse."""
    with torch.no_grad():
        # Graph-level diversity
        var_per_dim = embeddings.var(dim=0)
        mean_var = var_per_dim.mean().item()

        if embeddings.shape[0] >= 2:
            normed = F.normalize(embeddings, dim=-1)
            n_pairs = min(100, embeddings.shape[0])
            idx1 = torch.randint(0, embeddings.shape[0], (n_pairs,))
            idx2 = torch.randint(0, embeddings.shape[0], (n_pairs,))
            cos_sims = (normed[idx1] * normed[idx2]).sum(dim=-1)
            mean_cos_sim = cos_sims.mean().item()
        else:
            mean_cos_sim = 0.0

        # Node-level diversity within graphs
        node_diversity = -1.0
        if node_emb is not None and batch is not None:
            unique_graphs = batch.unique()
            within_cos_sims = []
            for g_id in unique_graphs[:10]:  # sample 10 graphs
                mask = batch == g_id
                g_nodes = node_emb[mask]
                if g_nodes.shape[0] > 2:
                    g_normed = F.normalize(g_nodes, dim=-1)
                    n_sample = min(20, g_nodes.shape[0])
                    i1 = torch.randint(0, g_nodes.shape[0], (n_sample,))
                    i2 = torch.randint(0, g_nodes.shape[0], (n_sample,))
                    sims = (g_normed[i1] * g_normed[i2]).sum(dim=-1)
                    within_cos_sims.append(sims.mean().item())
            if within_cos_sims:
                node_diversity = np.mean(within_cos_sims)

        collapsed = mean_var < threshold or mean_cos_sim > 0.99

    return {
        'mean_var': mean_var,
        'mean_cos_sim': mean_cos_sim,
        'node_within_cos_sim': node_diversity,
        'collapsed': collapsed,
    }


# ============================================================
# Training Loop
# ============================================================

def train_jepa(
    student_encoder,
    teacher_encoder,
    predictor,
    student_proj,
    teacher_proj,
    train_loader,
    device,
    num_epochs=30,
    lr=1e-3,
    weight_decay=1e-5,
    predictor_weight_decay=1e-4,
    ema_momentum_start=0.996,
    ema_momentum_end=0.999,
    ema_ramp_epochs=5,
    grad_clip=5.0,
    save_dir='./',
    context_ratio=0.6,
    target_ratio=0.1,
    # VICReg loss weights
    lambda_inv=1.0,
    lambda_var=25.0,
    lambda_cov=1.0,
    lambda_node_div=5.0,
    variance_threshold=1.0,
):
    """Train JEPA on PI1M trimer graphs with VICReg + node diversity."""

    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.Adam(
        [
            {'params': student_encoder.parameters(), 'weight_decay': weight_decay},
            {'params': predictor.parameters(), 'weight_decay': predictor_weight_decay},
            {'params': student_proj.parameters(), 'weight_decay': weight_decay},
        ],
        lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5
    )

    loss_history = []
    loss_components = {'jepa': [], 'variance': [], 'covariance': [], 'node_div': [], 'total': []}
    collapse_history = []
    best_loss = float('inf')

    logger.info(f"Starting JEPA pre-training (v6: very strong VICReg var+cov) for {num_epochs} epochs")
    logger.info(f"Student params: {sum(p.numel() for p in student_encoder.parameters()):,}")
    logger.info(f"Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")
    logger.info(f"Projection head params: {sum(p.numel() for p in student_proj.parameters()):,}")
    logger.info(f"VICReg weights: lambda_inv={lambda_inv}, lambda_var={lambda_var}, lambda_cov={lambda_cov}, lambda_node_div={lambda_node_div}")
    logger.info(f"EMA momentum: {ema_momentum_start} -> {ema_momentum_end} over {ema_ramp_epochs} epochs")
    logger.info(f"Encoder dropout: {student_encoder.dropout}")

    for epoch in range(num_epochs):
        if epoch < ema_ramp_epochs:
            ema_momentum = ema_momentum_start + (ema_momentum_end - ema_momentum_start) * (epoch / max(ema_ramp_epochs - 1, 1))
        else:
            ema_momentum = ema_momentum_end

        student_encoder.train()
        teacher_encoder.eval()
        predictor.train()
        student_proj.train()

        epoch_loss = 0.0
        epoch_jepa = 0.0
        epoch_var = 0.0
        epoch_cov = 0.0
        epoch_node_div = 0.0
        epoch_steps = 0
        epoch_start = time.time()
        epoch_collapse_checks = []

        for batch_idx, batch in enumerate(train_loader):
            B = batch.num_graphs
            n_nodes = batch.num_nodes

            # ---- Create masks on CPU ----
            all_context_masks = np.zeros(n_nodes, dtype=bool)
            all_target_masks = np.zeros(n_nodes, dtype=bool)
            positional_infos = []
            ptr = batch.ptr.numpy()

            for g in range(B):
                g_start = int(ptr[g])
                g_end = int(ptr[g + 1])
                g_n_nodes = g_end - g_start
                g_edge_index = batch.edge_index[:, (batch.edge_index[0] >= g_start) & (batch.edge_index[0] < g_end)]
                g_edge_index = g_edge_index - g_start
                g_x = batch.x[g_start:g_end]

                ctx_mask, tgt_mask, pos_info = create_jepa_masks_for_graph(
                    g_edge_index, g_n_nodes, g_x, context_ratio, target_ratio
                )
                all_context_masks[g_start:g_end] = ctx_mask
                all_target_masks[g_start:g_end] = tgt_mask
                positional_infos.append(torch.tensor(pos_info, dtype=torch.float))

            batch = batch.to(device)
            all_context_masks_t = torch.tensor(all_context_masks, dtype=torch.bool, device=device)
            all_target_masks_t = torch.tensor(all_target_masks, dtype=torch.bool, device=device)
            positional_info_batch = torch.stack(positional_infos).to(device)

            # ---- Student: process context subgraph ----
            context_indices = torch.where(all_context_masks_t)[0]
            if len(context_indices) == 0:
                continue

            node_mapping = -torch.ones(n_nodes, dtype=torch.long, device=device)
            node_mapping[context_indices] = torch.arange(len(context_indices), device=device)

            src_in_ctx = all_context_masks_t[batch.edge_index[0]]
            tgt_in_ctx = all_context_masks_t[batch.edge_index[1]]
            edge_mask = src_in_ctx & tgt_in_ctx

            if edge_mask.sum() == 0:
                continue

            ctx_edge_index = node_mapping[batch.edge_index[:, edge_mask]]
            ctx_x = batch.x[context_indices]
            ctx_batch = batch.batch[context_indices]
            ctx_edge_attr = batch.edge_attr[edge_mask] if batch.edge_attr is not None else None

            # Student forward on context subgraph
            ctx_node_emb = student_encoder(ctx_x, ctx_edge_index, ctx_batch, ctx_edge_attr)  # (ctx_nodes, 256)
            # MEAN pooling instead of sum pooling (robust to graph size)
            context_graph_emb = global_mean_pool(ctx_node_emb, ctx_batch)  # (B, 256)

            # Project for VICReg loss
            context_graph_proj = student_proj(context_graph_emb)  # (B, 256)

            # ---- Teacher: process FULL graph (no grad) ----
            with torch.no_grad():
                teacher_node_emb = teacher_encoder(
                    batch.x, batch.edge_index, batch.batch, batch.edge_attr
                )

                # Pool only target nodes - MEAN pooling
                target_node_emb = teacher_node_emb * all_target_masks_t.unsqueeze(-1).float()
                target_graph_emb = global_mean_pool(target_node_emb, batch.batch)

                # Count target nodes per graph to normalize (already handled by mean_pool but still useful)
                # Actually mean_pool counts only non-zero nodes, but with masking, zero nodes contribute 0
                # Let's do it manually for correctness
                target_counts = torch.zeros(B, device=device)
                target_counts.scatter_add_(0, batch.batch, all_target_masks_t.float())
                target_counts = target_counts.clamp(min=1.0)
                # Use add pool on masked, then divide by count
                target_graph_emb = global_add_pool(target_node_emb, batch.batch) / target_counts.unsqueeze(-1)

                # Project teacher target
                target_graph_proj = teacher_proj(target_graph_emb)

                # L2 normalize
                target_graph_proj = F.normalize(target_graph_proj, dim=-1)

            # ---- Predictor ----
            predicted_target = predictor(context_graph_proj, positional_info_batch)
            predicted_target = F.normalize(predicted_target, dim=-1)

            # ---- Loss computation ----
            # JEPA invariance loss
            jepa_loss = F.mse_loss(predicted_target, target_graph_proj.detach())

            # VICReg variance on RAW encoder output (not projected!)
            # This is critical: prevents dimensional collapse in the encoder itself
            var_loss = variance_loss(context_graph_emb, gamma=variance_threshold)

            # VICReg covariance on RAW encoder output
            cov_loss = covariance_loss(context_graph_emb)

            # Node diversity loss on student context node embeddings
            nd_loss = node_diversity_loss(ctx_node_emb, ctx_batch)

            # Total loss
            total_loss = (lambda_inv * jepa_loss +
                         lambda_var * var_loss +
                         lambda_cov * cov_loss +
                         lambda_node_div * nd_loss)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student_encoder.parameters()) +
                list(predictor.parameters()) +
                list(student_proj.parameters()),
                grad_clip
            )
            optimizer.step()

            # EMA update teacher + projection head
            update_ema(student_encoder, teacher_encoder, ema_momentum)
            update_ema(student_proj, teacher_proj, ema_momentum)

            epoch_loss += total_loss.item()
            epoch_jepa += jepa_loss.item()
            epoch_var += var_loss.item()
            epoch_cov += cov_loss.item()
            epoch_node_div += nd_loss.item()
            epoch_steps += 1

            # Collapse monitoring every 100 steps
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    collapse_info = check_collapse(
                        context_graph_emb,
                        node_emb=ctx_node_emb,
                        batch=ctx_batch
                    )
                    epoch_collapse_checks.append(collapse_info)

            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | Step {batch_idx}/{len(train_loader)} | "
                    f"Total: {total_loss.item():.6f} (JEPA: {jepa_loss.item():.6f}, "
                    f"Var: {var_loss.item():.6f}, Cov: {cov_loss.item():.6f}, "
                    f"NodeDiv: {nd_loss.item():.6f}) | EMA: {ema_momentum:.4f}"
                )

        scheduler.step()

        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_jepa = epoch_jepa / max(epoch_steps, 1)
        avg_var = epoch_var / max(epoch_steps, 1)
        avg_cov = epoch_cov / max(epoch_steps, 1)
        avg_nd = epoch_node_div / max(epoch_steps, 1)
        elapsed = time.time() - epoch_start

        loss_history.append(avg_loss)
        loss_components['jepa'].append(avg_jepa)
        loss_components['variance'].append(avg_var)
        loss_components['covariance'].append(avg_cov)
        loss_components['node_div'].append(avg_nd)
        loss_components['total'].append(avg_loss)

        # Aggregate collapse info
        if epoch_collapse_checks:
            avg_var_emb = np.mean([c['mean_var'] for c in epoch_collapse_checks])
            avg_cos_sim = np.mean([c['mean_cos_sim'] for c in epoch_collapse_checks])
            avg_node_cos = np.mean([c['node_within_cos_sim'] for c in epoch_collapse_checks if c['node_within_cos_sim'] >= 0])
            any_collapsed = any(c['collapsed'] for c in epoch_collapse_checks)
            collapse_history.append({
                'epoch': epoch + 1,
                'mean_embedding_var': avg_var_emb,
                'mean_cosine_sim': avg_cos_sim,
                'node_within_cos_sim': avg_node_cos,
                'collapsed': any_collapsed,
            })
        else:
            collapse_history.append({
                'epoch': epoch + 1,
                'mean_embedding_var': -1,
                'mean_cosine_sim': -1,
                'node_within_cos_sim': -1,
                'collapsed': False,
            })

        ci = collapse_history[-1]
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | Total: {avg_loss:.6f} "
            f"(JEPA: {avg_jepa:.6f}, Var: {avg_var:.6f}, Cov: {avg_cov:.6f}, NodeDiv: {avg_nd:.6f}) | "
            f"Emb Var: {ci['mean_embedding_var']:.4f} | "
            f"Cos Sim: {ci['mean_cosine_sim']:.4f} | "
            f"Node Cos: {ci['node_within_cos_sim']:.4f} | "
            f"Collapse: {ci['collapsed']} | "
            f"Time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if ci['collapsed']:
            logger.warning(f"  *** COLLAPSE DETECTED at epoch {epoch+1}! ***")

        # Save best teacher encoder
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(save_dir, 'pretrained_encoder_v6.pt')
            torch.save({
                'teacher_encoder': teacher_encoder.state_dict(),
                'student_encoder': student_encoder.state_dict(),
                'predictor': predictor.state_dict(),
                'teacher_proj': teacher_proj.state_dict(),
                'student_proj': student_proj.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'loss_components': {k: v[-1] if v else 0 for k, v in loss_components.items()},
                'ema_momentum': ema_momentum,
                'collapse_history': collapse_history,
            }, save_path)
            logger.info(f"Saved best encoder (loss={avg_loss:.6f}) to {save_path}")

    # Final save
    final_path = os.path.join(save_dir, 'pretrained_encoder_v6_final.pt')
    torch.save({
        'teacher_encoder': teacher_encoder.state_dict(),
        'student_encoder': student_encoder.state_dict(),
        'predictor': predictor.state_dict(),
        'teacher_proj': teacher_proj.state_dict(),
        'student_proj': student_proj.state_dict(),
        'epoch': num_epochs,
        'loss': loss_history[-1] if loss_history else 0,
        'loss_components': loss_components,
        'collapse_history': collapse_history,
    }, final_path)

    np.savez(
        os.path.join(save_dir, 'jepa_loss_history_v6.npz'),
        total=np.array(loss_history),
        jepa=np.array(loss_components['jepa']),
        variance=np.array(loss_components['variance']),
        covariance=np.array(loss_components['covariance']),
        node_div=np.array(loss_components['node_div']),
    )

    logger.info(f"JEPA pre-training complete. Best loss: {best_loss:.6f}")
    return loss_history, loss_components, collapse_history
