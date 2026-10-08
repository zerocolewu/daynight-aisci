"""
Training script for PolymerBipartite v2 Iteration 2.

Fixes from iteration 1:
- FIX1: Gate penalty L_gate = 0.01 * mean(g_i)
- FIX2: Discriminability from epoch 0, lambda=0.1, margin=0.85, 128 pairs
- FIX3: Gradient clipping max_norm=1.0, separate LR for backbone vs RU params
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_preprocessing import (
    PolymerBipartiteDataset, get_data_splits, create_random_ru_dataset,
    ATOM_FEATURE_DIM, BOND_FEATURE_DIM
)
from model import (
    PolymerBipartiteV2, GINTrimerBaseline, count_parameters,
    create_wide_baseline
)


def prepare_batch(graphs, targets, indices, device):
    """Prepare a batch of graphs for the PolymerBipartite v2 model."""
    batch_graphs = [graphs[i] for i in indices]
    batch_targets = targets[indices].to(device)

    batch_data = Batch.from_data_list(batch_graphs)
    batch_data = batch_data.to(device)

    n_atoms = batch_data.x.shape[0]
    batch_vec = batch_data.batch
    ru_assignment = batch_data.ru_assignment

    B = len(batch_graphs)
    atom_to_ru_idx = batch_vec * 3 + ru_assignment
    ru_batch = torch.arange(B, device=device).repeat_interleave(3)
    n_rus = 3 * B

    data_dict = {
        'x': batch_data.x,
        'edge_index': batch_data.edge_index,
        'edge_attr': batch_data.edge_attr,
        'batch': batch_vec,
        'ru_assignment': ru_assignment,
        'is_boundary': batch_data.is_boundary,
        'ru_position': batch_data.ru_position,
        'atom_to_ru_idx': atom_to_ru_idx,
        'ru_batch': ru_batch,
        'n_atoms': n_atoms,
        'n_rus': n_rus,
    }

    return data_dict, batch_targets


def prepare_baseline_batch(graphs, targets, indices, device):
    """Prepare batch for standard GIN baseline."""
    batch_graphs = [graphs[i] for i in indices]
    batch_targets = targets[indices].to(device)

    batch_data = Batch.from_data_list(batch_graphs)
    batch_data = batch_data.to(device)

    return batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch, batch_targets


# ============================================================================
# FIX2: Stronger discriminability loss from epoch 0
# ============================================================================

def compute_discriminability_loss(h_ru, ru_batch, n_pairs=128, margin=0.85):
    """
    Discriminability regularizer (FIX2: stronger from epoch 0).

    L_disc = mean(max(0, cos_sim(h_RU_k^i, h_RU_k^j) - margin))
    - 128 pairs (doubled from 64)
    - margin = 0.85 (tighter than 0.90)
    """
    B = ru_batch.max().item() + 1
    if B < 2:
        return torch.tensor(0.0, device=h_ru.device)

    # Get RU_1 (middle RU) for each graph as representative
    ru1_indices = torch.arange(B, device=h_ru.device) * 3 + 1
    h_ru1 = h_ru[ru1_indices]

    n_actual = min(n_pairs, B * (B - 1) // 2)
    if n_actual <= 0:
        return torch.tensor(0.0, device=h_ru.device)

    idx_i = torch.randint(0, B, (n_actual * 2,), device=h_ru.device)
    idx_j = torch.randint(0, B, (n_actual * 2,), device=h_ru.device)
    mask = idx_i != idx_j
    idx_i = idx_i[mask][:n_actual]
    idx_j = idx_j[mask][:n_actual]
    if len(idx_i) == 0:
        return torch.tensor(0.0, device=h_ru.device)

    cos_sim = F.cosine_similarity(h_ru1[idx_i], h_ru1[idx_j], dim=-1)
    loss = F.relu(cos_sim - margin).mean()
    return loss


def compute_gate_penalty(model, is_boundary, device):
    """
    FIX1: Gate regularization penalty.
    L_gate = lambda_gate * mean(g_i)
    Discourages fully open gates.
    """
    boundary_mask = (is_boundary > 0.5)
    n_total = is_boundary.shape[0]
    if n_total == 0:
        return torch.tensor(0.0, device=device)

    g_boundary = torch.sigmoid(model.boundary_gate_scalar)
    g_interior = torch.sigmoid(model.interior_gate_scalar)
    n_boundary = boundary_mask.sum().item()
    n_interior = n_total - n_boundary

    mean_gate = (n_boundary * g_boundary + n_interior * g_interior) / n_total
    return mean_gate


# ============================================================================
# Training Functions
# ============================================================================

def train_bipartite_v2(dataset, train_idx, val_idx, test_idx, seed=71,
                       epochs=300, backbone_lr=1e-3, ru_lr=5e-4,
                       weight_decay=1e-4, batch_size=32, hidden_dim=256,
                       patience=30, disc_lambda=0.1, gate_lambda=0.01,
                       device='cuda', save_dir='.'):
    """
    Train PolymerBipartite v2 iter2 with all fixes.

    FIX2: disc_lambda=0.1 from epoch 0, margin=0.85
    FIX3: Separate LR for backbone (1e-3) and RU params (5e-4)
    FIX3: Gradient clipping max_norm=1.0
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)

    valid_mask = dataset.valid_mask
    train_valid = [i for i in train_idx if valid_mask[i]]
    val_valid = [i for i in val_idx if valid_mask[i]]
    test_valid = [i for i in test_idx if valid_mask[i]]

    print(f"Training PolymerBipartite v2 (iter2): train={len(train_valid)}, val={len(val_valid)}, test={len(test_valid)}")

    model = PolymerBipartiteV2(
        node_dim=ATOM_FEATURE_DIM, edge_dim=BOND_FEATURE_DIM,
        hidden_dim=hidden_dim, num_layers=5,
        dropout_atom=0.5, dropout_ru_msg=0.3, dropout_head=0.5,
    ).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"  Initial boundary gate: {torch.sigmoid(model.boundary_gate_scalar).item():.4f}")
    print(f"  Initial interior gate: {torch.sigmoid(model.interior_gate_scalar).item():.4f}")

    # FIX3: Separate learning rate groups
    optimizer = torch.optim.Adam([
        {'params': model.get_backbone_params(), 'lr': backbone_lr},
        {'params': model.get_ru_params(), 'lr': ru_lr},
    ], weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_rmse = float('inf')
    best_epoch = 0
    best_state = None
    history = []
    disc_history = []

    for epoch in range(1, epochs + 1):
        model.train()

        perm = torch.randperm(len(train_valid)).tolist()
        train_perm = [train_valid[p] for p in perm]

        epoch_loss = 0.0
        epoch_disc = 0.0
        epoch_gate = 0.0
        n_batches = 0

        for i in range(0, len(train_perm), batch_size):
            batch_indices = train_perm[i:i + batch_size]
            data_dict, targets = prepare_batch(dataset.graphs, dataset.targets, batch_indices, device)

            optimizer.zero_grad()
            pred, h_ru = model(data_dict)

            # Supervised loss
            sup_loss = F.mse_loss(pred, targets)

            # FIX2: Discriminability from epoch 0 (no warmup)
            disc_loss = compute_discriminability_loss(h_ru, data_dict['ru_batch'], n_pairs=128, margin=0.85)

            # FIX1: Gate penalty
            gate_loss = compute_gate_penalty(model, data_dict['is_boundary'], device)

            # Total loss
            total_loss = sup_loss + disc_lambda * disc_loss + gate_lambda * gate_loss

            total_loss.backward()

            # FIX3: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += sup_loss.item()
            epoch_disc += disc_loss.item()
            epoch_gate += gate_loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        if epoch % 5 == 0 or epoch <= 30:
            model.eval()
            with torch.no_grad():
                train_preds = []
                train_true = []
                for i in range(0, len(train_valid), batch_size):
                    bi = train_valid[i:i + batch_size]
                    dd, tt = prepare_batch(dataset.graphs, dataset.targets, bi, device)
                    p, _ = model(dd)
                    train_preds.append(p.cpu())
                    train_true.append(tt.cpu())
                train_preds = torch.cat(train_preds)
                train_true = torch.cat(train_true)
                train_rmse = torch.sqrt(F.mse_loss(train_preds, train_true)).item()

                val_preds = []
                val_true = []
                for i in range(0, len(val_valid), batch_size):
                    bi = val_valid[i:i + batch_size]
                    dd, tt = prepare_batch(dataset.graphs, dataset.targets, bi, device)
                    p, _ = model(dd)
                    val_preds.append(p.cpu())
                    val_true.append(tt.cpu())
                val_preds = torch.cat(val_preds)
                val_true = torch.cat(val_true)
                val_rmse = torch.sqrt(F.mse_loss(val_preds, val_true)).item()

            b_gate = torch.sigmoid(model.boundary_gate_scalar).item()
            i_gate = torch.sigmoid(model.interior_gate_scalar).item()

            history.append({
                'epoch': epoch,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_loss': epoch_loss / max(n_batches, 1),
                'disc_loss': epoch_disc / max(n_batches, 1),
                'gate_loss': epoch_gate / max(n_batches, 1),
                'boundary_gate': b_gate,
                'interior_gate': i_gate,
            })

            # Track discriminability at key epochs
            if epoch in [10, 30, 50, 100] or epoch == epochs:
                cos_sim = compute_val_discriminability(model, dataset, val_valid, device)
                disc_history.append({'epoch': epoch, 'mean_cos_sim': cos_sim})
                print(f"  RU cos_sim at epoch {epoch}: {cos_sim:.4f}")

            improved = ''
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                improved = ' *'

            print(f"Epoch {epoch:3d} | Train: {train_rmse:.2f} | Val: {val_rmse:.2f} | "
                  f"Disc: {epoch_disc/max(n_batches,1):.4f} | Gate(b/i): {b_gate:.3f}/{i_gate:.3f}{improved}")

            if epoch - best_epoch >= patience:
                print(f"Early stopping at epoch {epoch}, best was epoch {best_epoch}")
                break

    # Evaluate on all splits with best model
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    test_results = evaluate_model(model, dataset, test_valid, device, model_type='bipartite')
    train_results = evaluate_model(model, dataset, train_valid, device, model_type='bipartite')
    val_results = evaluate_model(model, dataset, val_valid, device, model_type='bipartite')

    b_gate = torch.sigmoid(model.boundary_gate_scalar).item()
    i_gate = torch.sigmoid(model.interior_gate_scalar).item()
    print(f"\n=== PolymerBipartite v2 iter2 Results (seed={seed}) ===")
    print(f"  Train RMSE: {train_results['rmse']:.2f}")
    print(f"  Val RMSE:   {val_results['rmse']:.2f}")
    print(f"  Test RMSE:  {test_results['rmse']:.2f}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Final gates: boundary={b_gate:.4f}, interior={i_gate:.4f}")

    results = {
        'model': 'PolymerBipartiteV2_iter2',
        'seed': seed,
        'best_epoch': best_epoch,
        'train_rmse': train_results['rmse'],
        'val_rmse': val_results['rmse'],
        'test_rmse': test_results['rmse'],
        'overfitting_gap': train_results['rmse'] - test_results['rmse'],
        'history': history,
        'disc_history': disc_history,
        'n_params': count_parameters(model),
        'final_boundary_gate': b_gate,
        'final_interior_gate': i_gate,
    }

    save_path = os.path.join(save_dir, f'bpv2_iter2_seed{seed}.pt')
    torch.save({'model_state': best_state, 'results': results}, save_path)
    print(f"Saved to {save_path}")

    return model, results


def train_baseline(dataset, train_idx, val_idx, test_idx, seed=71,
                   epochs=300, lr=1e-3, weight_decay=1e-4,
                   batch_size=32, hidden_dim=256, patience=30,
                   device='cuda', save_dir='.', save_name=None):
    """Train standard GIN+Trimer baseline."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)

    valid_mask = dataset.valid_mask
    train_valid = [i for i in train_idx if valid_mask[i]]
    val_valid = [i for i in val_idx if valid_mask[i]]
    test_valid = [i for i in test_idx if valid_mask[i]]

    model_name = f'GIN+Trimer(h={hidden_dim})'
    print(f"\nTraining {model_name}: train={len(train_valid)}, val={len(val_valid)}, test={len(test_valid)}")

    model = GINTrimerBaseline(
        node_dim=ATOM_FEATURE_DIM, edge_dim=BOND_FEATURE_DIM,
        hidden_dim=hidden_dim,
    ).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_rmse = float('inf')
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()

        perm = torch.randperm(len(train_valid)).tolist()
        train_perm = [train_valid[p] for p in perm]

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_perm), batch_size):
            batch_indices = train_perm[i:i + batch_size]
            x, ei, ea, batch_vec, targets = prepare_baseline_batch(
                dataset.graphs, dataset.targets, batch_indices, device)
            optimizer.zero_grad()
            pred = model(x, ei, ea, batch_vec)
            loss = F.mse_loss(pred, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch <= 30:
            model.eval()
            with torch.no_grad():
                train_rmse = eval_baseline_rmse(model, dataset.graphs, dataset.targets, train_valid, device)
                val_rmse = eval_baseline_rmse(model, dataset.graphs, dataset.targets, val_valid, device)

            history.append({
                'epoch': epoch,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_loss': epoch_loss / max(n_batches, 1),
            })

            improved = ''
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                improved = ' *'

            print(f"Epoch {epoch:3d} | Train: {train_rmse:.2f} | Val: {val_rmse:.2f} | "
                  f"Loss: {epoch_loss/max(n_batches,1):.2f}{improved}")

            if epoch - best_epoch >= patience:
                print(f"Early stopping at epoch {epoch}, best was epoch {best_epoch}")
                break

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    test_rmse = eval_baseline_rmse(model, dataset.graphs, dataset.targets, test_valid, device)
    train_rmse = eval_baseline_rmse(model, dataset.graphs, dataset.targets, train_valid, device)
    val_rmse = eval_baseline_rmse(model, dataset.graphs, dataset.targets, val_valid, device)

    print(f"\n=== {model_name} Results (seed={seed}) ===")
    print(f"  Train RMSE: {train_rmse:.2f}")
    print(f"  Val RMSE:   {val_rmse:.2f}")
    print(f"  Test RMSE:  {test_rmse:.2f}")
    print(f"  Best epoch: {best_epoch}")

    results = {
        'model': model_name,
        'seed': seed,
        'best_epoch': best_epoch,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse,
        'overfitting_gap': train_rmse - test_rmse,
        'history': history,
        'n_params': count_parameters(model),
    }

    name = save_name or f'baseline_seed{seed}.pt'
    torch.save({'model_state': best_state, 'results': results},
               os.path.join(save_dir, name))

    return model, results


def train_random_ru(dataset, train_idx, val_idx, test_idx, seed=71,
                    epochs=300, backbone_lr=1e-3, ru_lr=5e-4,
                    weight_decay=1e-4, batch_size=32, hidden_dim=256,
                    patience=30, device='cuda', save_dir='.'):
    """Train PolymerBipartite v2 iter2 with randomly shuffled RU assignments."""
    print(f"\n=== Training with Random RU Assignment (Diagnostic 2, iter2) ===")
    torch.manual_seed(seed)
    np.random.seed(seed)

    random_graphs, _, _ = create_random_ru_dataset(
        dataset.graphs, dataset.targets, dataset.valid_mask, seed=seed + 1000)

    valid_mask = dataset.valid_mask
    train_valid = [i for i in train_idx if valid_mask[i]]
    val_valid = [i for i in val_idx if valid_mask[i]]
    test_valid = [i for i in test_idx if valid_mask[i]]

    model = PolymerBipartiteV2(
        node_dim=ATOM_FEATURE_DIM, edge_dim=BOND_FEATURE_DIM,
        hidden_dim=hidden_dim, num_layers=5,
    ).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.get_backbone_params(), 'lr': backbone_lr},
        {'params': model.get_ru_params(), 'lr': ru_lr},
    ], weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_rmse = float('inf')
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(train_valid)).tolist()
        train_perm = [train_valid[p] for p in perm]

        for i in range(0, len(train_perm), batch_size):
            batch_indices = train_perm[i:i + batch_size]
            data_dict, targets = prepare_batch(random_graphs, dataset.targets, batch_indices, device)
            optimizer.zero_grad()
            pred, h_ru = model(data_dict)
            loss = F.mse_loss(pred, targets)
            # Also add disc and gate regularizers for fair comparison
            disc_loss = compute_discriminability_loss(h_ru, data_dict['ru_batch'], n_pairs=128, margin=0.85)
            gate_loss = compute_gate_penalty(model, data_dict['is_boundary'], device)
            total = loss + 0.1 * disc_loss + 0.01 * gate_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        if epoch % 5 == 0 or epoch <= 30:
            model.eval()
            with torch.no_grad():
                val_preds = []
                val_true = []
                for i in range(0, len(val_valid), batch_size):
                    bi = val_valid[i:i + batch_size]
                    dd, tt = prepare_batch(random_graphs, dataset.targets, bi, device)
                    p, _ = model(dd)
                    val_preds.append(p.cpu())
                    val_true.append(tt.cpu())
                val_preds = torch.cat(val_preds)
                val_true = torch.cat(val_true)
                val_rmse = torch.sqrt(F.mse_loss(val_preds, val_true)).item()

            improved = ''
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                improved = ' *'

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Val RMSE: {val_rmse:.2f}{improved}")

            if epoch - best_epoch >= patience:
                print(f"Early stopping at epoch {epoch}, best was epoch {best_epoch}")
                break

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    test_rmse = eval_bipartite_rmse(model, random_graphs, dataset.targets, test_valid, device)
    train_rmse = eval_bipartite_rmse(model, random_graphs, dataset.targets, train_valid, device)
    val_rmse = eval_bipartite_rmse(model, random_graphs, dataset.targets, val_valid, device)

    b_gate = torch.sigmoid(model.boundary_gate_scalar).item()
    i_gate = torch.sigmoid(model.interior_gate_scalar).item()

    print(f"\n=== Random RU iter2 Results (seed={seed}) ===")
    print(f"  Train RMSE: {train_rmse:.2f}")
    print(f"  Val RMSE:   {val_rmse:.2f}")
    print(f"  Test RMSE:  {test_rmse:.2f}")
    print(f"  Final gates: boundary={b_gate:.4f}, interior={i_gate:.4f}")

    results = {
        'model': 'RandomRU_iter2',
        'seed': seed,
        'best_epoch': best_epoch,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse,
        'overfitting_gap': train_rmse - test_rmse,
        'n_params': count_parameters(model),
        'final_boundary_gate': b_gate,
        'final_interior_gate': i_gate,
    }

    torch.save({'model_state': best_state, 'results': results},
               os.path.join(save_dir, f'random_ru_iter2_seed{seed}.pt'))

    return model, results


# ============================================================================
# Evaluation Helpers
# ============================================================================

def eval_baseline_rmse(model, graphs, targets, indices, device, batch_size=32):
    model.eval()
    preds = []
    true = []
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            bi = indices[i:i + batch_size]
            x, ei, ea, batch_vec, tt = prepare_baseline_batch(graphs, targets, bi, device)
            p = model(x, ei, ea, batch_vec)
            preds.append(p.cpu())
            true.append(tt.cpu())
    preds = torch.cat(preds)
    true = torch.cat(true)
    return torch.sqrt(F.mse_loss(preds, true)).item()


def eval_bipartite_rmse(model, graphs, targets, indices, device, batch_size=32):
    model.eval()
    preds = []
    true = []
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            bi = indices[i:i + batch_size]
            dd, tt = prepare_batch(graphs, targets, bi, device)
            p, _ = model(dd)
            preds.append(p.cpu())
            true.append(tt.cpu())
    preds = torch.cat(preds)
    true = torch.cat(true)
    return torch.sqrt(F.mse_loss(preds, true)).item()


def evaluate_model(model, dataset, indices, device, model_type='bipartite', batch_size=32):
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            bi = indices[i:i + batch_size]
            if model_type == 'bipartite':
                dd, tt = prepare_batch(dataset.graphs, dataset.targets, bi, device)
                p, _ = model(dd)
            else:
                x, ei, ea, bv, tt = prepare_baseline_batch(
                    dataset.graphs, dataset.targets, bi, device)
                p = model(x, ei, ea, bv)
            all_preds.append(p.cpu())
            all_true.append(tt.cpu())

    preds = torch.cat(all_preds)
    true = torch.cat(all_true)
    rmse = torch.sqrt(F.mse_loss(preds, true)).item()
    mae = torch.mean(torch.abs(preds - true)).item()

    return {'rmse': rmse, 'mae': mae, 'predictions': preds, 'targets': true}


def compute_val_discriminability(model, dataset, val_indices, device, batch_size=64):
    """Compute pairwise cosine similarity of RU representations."""
    model.eval()
    all_ru_h = []

    with torch.no_grad():
        for i in range(0, len(val_indices), batch_size):
            bi = val_indices[i:i + batch_size]
            dd, _ = prepare_batch(dataset.graphs, dataset.targets, bi, device)
            _, h_ru, _ = model(dd, return_gate_values=True)
            B = len(bi)
            ru1_idx = torch.arange(B, device=device) * 3 + 1
            all_ru_h.append(h_ru[ru1_idx].cpu())

    all_ru_h = torch.cat(all_ru_h, dim=0)

    n = all_ru_h.shape[0]
    if n > 500:
        idx = torch.randperm(n)[:500]
        all_ru_h = all_ru_h[idx]
        n = 500

    normed = F.normalize(all_ru_h, dim=-1)
    cos_sim_matrix = normed @ normed.t()

    triu_idx = torch.triu_indices(n, n, offset=1)
    pairwise_cos = cos_sim_matrix[triu_idx[0], triu_idx[1]]

    return pairwise_cos.mean().item()
