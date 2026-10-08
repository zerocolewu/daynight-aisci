"""
JEPA (Joint Embedding Predictive Architecture) pre-training on PI1M Trimer graphs.

Architecture:
- Student Encoder: GIN with 5 layers, hidden_dim=256
- Teacher Encoder: Same architecture, updated via EMA
- Predictor: 2-layer MLP predicting target embeddings from context embeddings
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
    """GIN encoder with 5 layers."""

    def __init__(self, in_dim, hidden_dim=256, out_dim=256, num_layers=5, dropout=0.5):
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

        self.pool = global_add_pool

    def forward(self, x, edge_index, batch, edge_attr=None):
        """Forward pass, returns node-level embeddings."""
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x  # node-level embeddings

    def get_graph_embedding(self, x, edge_index, batch, node_mask=None, edge_attr=None):
        """Get graph-level embedding (sum readout).
        If node_mask is provided, only pool masked nodes.
        """
        node_emb = self.forward(x, edge_index, batch, edge_attr)
        if node_mask is not None:
            node_emb = node_emb * node_mask.unsqueeze(-1)
            # Use sum pooling on masked nodes
            graph_emb = global_add_pool(node_emb, batch)
        else:
            graph_emb = global_add_pool(node_emb, batch)
        return graph_emb, node_emb


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
        """
        Args:
            context_emb: (B, 256) context graph embedding
            positional_info: (B, pos_dim) fraction of target nodes per RU
        Returns:
            (B, 256) predicted target embedding
        """
        x = torch.cat([context_emb, positional_info], dim=-1)
        return self.predictor(x)


# ============================================================
# JEPA Masking Utilities
# ============================================================

def create_jepa_masks_for_graph(edge_index_cpu, n_nodes, x_cpu, context_ratio=0.6, target_ratio=0.1):
    """Create context and target masks for a single graph (all CPU tensors).

    Args:
        edge_index_cpu: (2, E) CPU LongTensor
        n_nodes: int
        x_cpu: (N, F) CPU FloatTensor (atom features)
        context_ratio: fraction of nodes for context
        target_ratio: fraction of nodes for target

    Returns:
        context_mask: numpy boolean array (n_nodes,)
        target_mask: numpy boolean array (n_nodes,)
        positional_info: numpy float32 array (3,)
    """
    edge_index_np = edge_index_cpu.numpy()

    # Build adjacency list
    adj = [[] for _ in range(n_nodes)]
    for e in range(edge_index_np.shape[1]):
        u, v = int(edge_index_np[0, e]), int(edge_index_np[1, e])
        if u != v:
            adj[u].append(v)

    # BFS from random start to select context
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

    # Fill remaining context randomly if BFS didn't reach enough
    if count < n_context:
        remaining = np.where(~context_mask)[0]
        if len(remaining) > 0:
            extra = np.random.choice(remaining, min(n_context - count, len(remaining)), replace=False)
            context_mask[extra] = True

    # Select target from non-context nodes
    non_context = ~context_mask
    non_context_indices = np.where(non_context)[0]
    n_target = max(1, int(n_nodes * target_ratio))

    if len(non_context_indices) <= n_target:
        target_mask = non_context.copy()
    else:
        target_indices = np.random.choice(non_context_indices, n_target, replace=False)
        target_mask = np.zeros(n_nodes, dtype=bool)
        target_mask[target_indices] = True

    # Positional info: fraction of target nodes in each RU
    ru_features = x_cpu[:, -3:].numpy()  # Last 3 dims are ru_index one-hot
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
# Training Loop
# ============================================================

def train_jepa(
    student_encoder,
    teacher_encoder,
    predictor,
    train_loader,
    device,
    num_epochs=15,
    lr=1e-3,
    weight_decay=1e-5,
    ema_momentum_start=0.996,
    ema_momentum_end=0.999,
    grad_clip=5.0,
    save_dir='./',
    context_ratio=0.6,
    target_ratio=0.1,
):
    """Train JEPA on PI1M trimer graphs."""

    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.Adam(
        list(student_encoder.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5
    )

    loss_history = []
    global_step = 0
    best_loss = float('inf')

    logger.info(f"Starting JEPA pre-training for {num_epochs} epochs")
    logger.info(f"Student params: {sum(p.numel() for p in student_encoder.parameters()):,}")
    logger.info(f"Teacher params: {sum(p.numel() for p in teacher_encoder.parameters()):,}")
    logger.info(f"Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")

    for epoch in range(num_epochs):
        # EMA momentum schedule
        ema_momentum = ema_momentum_start + (ema_momentum_end - ema_momentum_start) * (epoch / max(num_epochs - 1, 1))

        student_encoder.train()
        teacher_encoder.eval()
        predictor.train()

        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Keep batch on CPU for masking, then move to GPU
            B = batch.num_graphs
            n_nodes = batch.num_nodes

            # ---- Create masks on CPU (numpy operations) ----
            all_context_masks = np.zeros(n_nodes, dtype=bool)
            all_target_masks = np.zeros(n_nodes, dtype=bool)
            positional_infos = []

            # Get per-graph node ranges using ptr
            ptr = batch.ptr.numpy()  # (B+1,) cumulative node counts

            for g in range(B):
                g_start = int(ptr[g])
                g_end = int(ptr[g + 1])
                g_n_nodes = g_end - g_start

                # Get edge index and x for this graph
                g_edge_index = batch.edge_index[:, (batch.edge_index[0] >= g_start) & (batch.edge_index[0] < g_end)]
                g_edge_index = g_edge_index - g_start  # make local indices
                g_x = batch.x[g_start:g_end]

                ctx_mask, tgt_mask, pos_info = create_jepa_masks_for_graph(
                    g_edge_index, g_n_nodes, g_x, context_ratio, target_ratio
                )

                all_context_masks[g_start:g_end] = ctx_mask
                all_target_masks[g_start:g_end] = tgt_mask
                positional_infos.append(torch.tensor(pos_info, dtype=torch.float))

            # Move everything to device
            batch = batch.to(device)
            all_context_masks_t = torch.tensor(all_context_masks, dtype=torch.bool, device=device)
            all_target_masks_t = torch.tensor(all_target_masks, dtype=torch.bool, device=device)
            positional_info_batch = torch.stack(positional_infos).to(device)  # (B, 3)

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

            # Student forward
            ctx_node_emb = student_encoder(ctx_x, ctx_edge_index, ctx_batch, ctx_edge_attr)
            context_graph_emb = global_add_pool(ctx_node_emb, ctx_batch)

            # ---- Teacher: process FULL graph (no grad) ----
            with torch.no_grad():
                teacher_node_emb = teacher_encoder(
                    batch.x, batch.edge_index, batch.batch, batch.edge_attr
                )

                # Pool only target nodes for each graph
                target_node_emb = teacher_node_emb * all_target_masks_t.unsqueeze(-1).float()
                target_graph_emb = global_add_pool(target_node_emb, batch.batch)

                # Count target nodes per graph to normalize
                target_counts = torch.zeros(B, device=device)
                target_counts.scatter_add_(0, batch.batch, all_target_masks_t.float())
                target_counts = target_counts.clamp(min=1.0)
                target_graph_emb = target_graph_emb / target_counts.unsqueeze(-1)

            # ---- Predictor ----
            predicted_target = predictor(context_graph_emb, positional_info_batch)

            # ---- Loss ----
            loss = F.mse_loss(predicted_target, target_graph_emb.detach())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student_encoder.parameters()) + list(predictor.parameters()),
                grad_clip
            )
            optimizer.step()

            # EMA update teacher
            update_ema(student_encoder, teacher_encoder, ema_momentum)

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | Step {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.6f} | EMA mom: {ema_momentum:.4f}"
                )

        scheduler.step()

        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - epoch_start
        loss_history.append(avg_loss)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.6f} | "
            f"Time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save best teacher encoder
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(save_dir, 'pretrained_encoder.pt')
            torch.save({
                'teacher_encoder': teacher_encoder.state_dict(),
                'student_encoder': student_encoder.state_dict(),
                'predictor': predictor.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'ema_momentum': ema_momentum,
            }, save_path)
            logger.info(f"Saved best teacher encoder (loss={avg_loss:.6f}) to {save_path}")

    # Final save
    final_path = os.path.join(save_dir, 'pretrained_encoder_final.pt')
    torch.save({
        'teacher_encoder': teacher_encoder.state_dict(),
        'student_encoder': student_encoder.state_dict(),
        'predictor': predictor.state_dict(),
        'epoch': num_epochs,
        'loss': loss_history[-1] if loss_history else 0,
    }, final_path)

    # Save loss history
    np.save(os.path.join(save_dir, 'jepa_loss_history.npy'), np.array(loss_history))

    logger.info(f"JEPA pre-training complete. Best loss: {best_loss:.6f}")
    logger.info(f"Loss history: {loss_history}")

    return loss_history


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    from data_preprocessing import PI1MTrimerDataset, TOTAL_ATOM_FEATURE_DIM, BOND_FEATURE_DIM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load PI1M dataset
    pi1m_dataset = PI1MTrimerDataset(
        root='./data',
        csv_path='/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv',
        max_samples=300000,
    )
    logger.info(f"PI1M dataset size: {len(pi1m_dataset)}")

    train_loader = DataLoader(pi1m_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    # Create models
    in_dim = TOTAL_ATOM_FEATURE_DIM  # 33
    hidden_dim = 256

    student_encoder = GINEncoder(in_dim, hidden_dim, dropout=0.5).to(device)
    teacher_encoder = GINEncoder(in_dim, hidden_dim, dropout=0.5).to(device)

    # Initialize teacher with student weights
    teacher_encoder.load_state_dict(student_encoder.state_dict())

    predictor = JEPAPredictor(context_dim=hidden_dim, pos_dim=3, hidden_dim=hidden_dim).to(device)

    # Train
    loss_history = train_jepa(
        student_encoder=student_encoder,
        teacher_encoder=teacher_encoder,
        predictor=predictor,
        train_loader=train_loader,
        device=device,
        num_epochs=15,
        lr=1e-3,
        weight_decay=1e-5,
        save_dir='./',
    )
