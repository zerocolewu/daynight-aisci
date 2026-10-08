"""
Contrastive Pre-training for Polymer Trimer Graphs (SimCLR-style).

Instead of JEPA masking, uses graph augmentations to create positive pairs
and learns discriminative representations via NT-Xent loss.

Key design choices:
1. Graph augmentations: node dropping, edge masking, subgraph sampling
2. Projection head for contrastive loss
3. VICReg on encoder output to prevent dimensional collapse
4. Same GIN encoder architecture for fair comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import logging
from torch_geometric.nn import global_mean_pool, global_add_pool

logger = logging.getLogger(__name__)

# Import GINEncoder from jepa_pretrain for architectural compatibility
from jepa_pretrain import GINEncoder


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, in_dim=256, hidden_dim=256, out_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.head(x)


# ---- Graph Augmentations ----

def augment_graph(batch, device, drop_rate=0.2, edge_drop_rate=0.2):
    """Apply random augmentations to a batch of graphs.

    Augmentations:
    1. Node feature masking (randomly zero out some node features)
    2. Edge dropping (randomly remove edges)

    Returns augmented batch.
    """
    x = batch.x.clone()
    edge_index = batch.edge_index.clone()

    # Node feature masking: randomly mask some node features
    feat_mask = torch.bernoulli(torch.full((x.shape[0], 1), 1 - drop_rate, device=device))
    x = x * feat_mask

    # Edge dropping: randomly remove some edges
    num_edges = edge_index.shape[1]
    edge_mask = torch.bernoulli(torch.full((num_edges,), 1 - edge_drop_rate, device=device)).bool()
    # Keep at least some edges per node to maintain connectivity
    edge_index = edge_index[:, edge_mask]

    # Create new batch object
    aug_batch = batch.clone()
    aug_batch.x = x
    aug_batch.edge_index = edge_index

    return aug_batch


def augment_subgraph(batch, device, sample_ratio=0.8):
    """Subgraph sampling augmentation.

    For each graph, randomly sample a subset of nodes and extract the induced subgraph.
    """
    # For simplicity, use node dropping (mask out dropped nodes in features)
    x = batch.x.clone()
    node_mask = torch.bernoulli(torch.full((x.shape[0],), sample_ratio, device=device))
    x = x * node_mask.unsqueeze(-1)

    aug_batch = batch.clone()
    aug_batch.x = x
    return aug_batch


# ---- Loss Functions ----

def nt_xent_loss(z1, z2, temperature=0.1):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    Args:
        z1: (B, D) projected embeddings from view 1
        z2: (B, D) projected embeddings from view 2
        temperature: softmax temperature

    Returns:
        scalar loss
    """
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    z = F.normalize(z, dim=1)

    # Compute similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Labels: for each i in [0, B), positive is i+B; for each i in [B, 2B), positive is i-B
    labels = torch.cat([torch.arange(batch_size, 2 * batch_size, device=z.device),
                        torch.arange(0, batch_size, device=z.device)], dim=0)

    loss = F.cross_entropy(sim, labels)
    return loss


def variance_loss(z, gamma=1.0):
    """VICReg variance term: penalize low per-dimension std."""
    std_per_dim = torch.sqrt(z.var(dim=0) + 1e-4)
    loss = torch.mean(torch.relu(gamma - std_per_dim))
    return loss


def covariance_loss(z):
    """VICReg covariance term: penalize off-diagonal correlation."""
    n, d = z.shape
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / n
    diag_mask = torch.eye(d, device=z.device, dtype=torch.bool)
    off_diag = cov[~diag_mask]
    loss = off_diag.pow(2).sum() / d
    return loss


def check_collapse(graph_emb):
    """Check if embeddings have collapsed."""
    with torch.no_grad():
        var_per_dim = graph_emb.var(dim=0)
        mean_var = var_per_dim.mean().item()

        # Cosine similarity
        normed = F.normalize(graph_emb, dim=1)
        cos_sim = torch.mm(normed, normed.t())
        n = cos_sim.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=cos_sim.device)
        mean_cos_sim = cos_sim[mask].mean().item()

        collapsed = mean_var < 0.01 or mean_cos_sim > 0.999

    return {
        'mean_var': mean_var,
        'mean_cos_sim': mean_cos_sim,
        'collapsed': collapsed,
    }


def train_contrastive(
    student_encoder,
    projection_head,
    train_loader,
    device,
    num_epochs=30,
    lr=1e-3,
    weight_decay=1e-5,
    grad_clip=5.0,
    save_dir='.',
    temperature=0.1,
    lambda_var=25.0,
    lambda_cov=1.0,
    variance_threshold=1.0,
    drop_rate=0.2,
    edge_drop_rate=0.2,
):
    """Train encoder with contrastive learning."""

    optimizer = torch.optim.Adam(
        list(student_encoder.parameters()) + list(projection_head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )

    loss_history = []
    loss_components = {'contrastive': [], 'variance': [], 'covariance': [], 'total': []}
    collapse_history = []
    best_loss = float('inf')

    logger.info(f"Starting contrastive pre-training for {num_epochs} epochs")
    logger.info(f"Encoder params: {sum(p.numel() for p in student_encoder.parameters()):,}")
    logger.info(f"Projection head params: {sum(p.numel() for p in projection_head.parameters()):,}")
    logger.info(f"Temperature: {temperature}, lambda_var: {lambda_var}, lambda_cov: {lambda_cov}")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        student_encoder.train()
        projection_head.train()

        epoch_loss = 0.0
        epoch_cont = 0.0
        epoch_var = 0.0
        epoch_cov = 0.0
        epoch_steps = 0
        epoch_collapse_checks = []

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)

            # Create two augmented views
            view1 = augment_graph(batch, device, drop_rate=drop_rate, edge_drop_rate=edge_drop_rate)
            view2 = augment_subgraph(batch, device, sample_ratio=0.85)

            # Encode both views
            node_emb1 = student_encoder(view1.x, view1.edge_index, view1.batch, view1.edge_attr)
            node_emb2 = student_encoder(view2.x, view2.edge_index, view2.batch, view2.edge_attr)

            graph_emb1 = global_mean_pool(node_emb1, view1.batch)
            graph_emb2 = global_mean_pool(node_emb2, view2.batch)

            # Project for contrastive loss
            proj1 = projection_head(graph_emb1)
            proj2 = projection_head(graph_emb2)

            # Contrastive loss
            cont_loss = nt_xent_loss(proj1, proj2, temperature=temperature)

            # VICReg on raw encoder output (view 1)
            var_loss = variance_loss(graph_emb1, gamma=variance_threshold)
            cov_loss = covariance_loss(graph_emb1)

            # Total loss
            total_loss = cont_loss + lambda_var * var_loss + lambda_cov * cov_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student_encoder.parameters()) + list(projection_head.parameters()),
                grad_clip
            )
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_cont += cont_loss.item()
            epoch_var += var_loss.item()
            epoch_cov += cov_loss.item()
            epoch_steps += 1

            # Collapse monitoring every 100 steps
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    collapse_info = check_collapse(graph_emb1)
                    epoch_collapse_checks.append(collapse_info)

            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | Step {batch_idx}/{len(train_loader)} | "
                    f"Total: {total_loss.item():.4f} (Cont: {cont_loss.item():.4f}, "
                    f"Var: {var_loss.item():.6f}, Cov: {cov_loss.item():.6f})"
                )

        scheduler.step()

        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_cont = epoch_cont / max(epoch_steps, 1)
        avg_var = epoch_var / max(epoch_steps, 1)
        avg_cov = epoch_cov / max(epoch_steps, 1)
        elapsed = time.time() - epoch_start

        loss_history.append(avg_loss)
        loss_components['contrastive'].append(avg_cont)
        loss_components['variance'].append(avg_var)
        loss_components['covariance'].append(avg_cov)
        loss_components['total'].append(avg_loss)

        # Aggregate collapse info
        if epoch_collapse_checks:
            avg_var_emb = np.mean([c['mean_var'] for c in epoch_collapse_checks])
            avg_cos_sim = np.mean([c['mean_cos_sim'] for c in epoch_collapse_checks])
            any_collapsed = any(c['collapsed'] for c in epoch_collapse_checks)
            collapse_history.append({
                'epoch': epoch + 1,
                'mean_embedding_var': avg_var_emb,
                'mean_cosine_sim': avg_cos_sim,
                'collapsed': any_collapsed,
            })
        else:
            collapse_history.append({
                'epoch': epoch + 1,
                'mean_embedding_var': -1,
                'mean_cosine_sim': -1,
                'collapsed': False,
            })

        ci = collapse_history[-1]
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | Total: {avg_loss:.4f} "
            f"(Cont: {avg_cont:.4f}, Var: {avg_var:.6f}, Cov: {avg_cov:.6f}) | "
            f"Emb Var: {ci['mean_embedding_var']:.4f} | "
            f"Cos Sim: {ci['mean_cosine_sim']:.4f} | "
            f"Collapse: {ci['collapsed']} | "
            f"Time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if ci['collapsed']:
            logger.warning(f"COLLAPSE DETECTED at epoch {epoch+1}!")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(save_dir, 'pretrained_encoder_contrastive.pt')
            torch.save({
                'epoch': epoch + 1,
                'student_encoder': student_encoder.state_dict(),
                'projection_head': projection_head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            logger.info(f"Saved best encoder (loss={avg_loss:.6f}) to {save_path}")

    # Save final model
    final_path = os.path.join(save_dir, 'pretrained_encoder_contrastive_final.pt')
    torch.save({
        'epoch': num_epochs,
        'student_encoder': student_encoder.state_dict(),
        'projection_head': projection_head.state_dict(),
        'loss': avg_loss,
    }, final_path)

    # Save loss history
    np.savez(
        os.path.join(save_dir, 'contrastive_loss_history.npz'),
        loss_history=loss_history,
        contrastive=loss_components['contrastive'],
        variance=loss_components['variance'],
        covariance=loss_components['covariance'],
    )

    return loss_history, loss_components, collapse_history
