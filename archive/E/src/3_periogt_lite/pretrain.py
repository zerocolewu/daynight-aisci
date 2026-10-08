"""
Contrastive pre-training on PI1M using InfoNCE loss.
Supports six augmentation strategies:
1. 'monomer_trimer' (baseline): positive pair = (monomer, trimer) from same polymer
2. 'subgraph_crop': positive pair = (random_subgraph_1, random_subgraph_2) from same trimer
3. 'shifted_ru': positive pair = (standard_trimer, shifted_trimer)
4. 'edge_drop_feat_mask': positive pair = (original_trimer, corrupted_trimer) with edge dropout + feature masking
5. 'monomer_dimer': positive pair = (monomer, dimer) from same polymer
6. 'hybrid_corrupt': positive pair = (corrupted_monomer, corrupted_trimer) both with edge dropout
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PerioGTPretrainModel, GINEncoder, count_parameters
from augment import SubgraphCropDataset, shifted_trimer, edge_dropout_feat_mask, edge_dropout, psmiles_to_dimer
from graph_utils import psmiles_to_trimer, mol_to_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


class PairedDataset(torch.utils.data.Dataset):
    """Dataset that returns (view1, view2) pairs."""

    def __init__(self, view1_list, view2_list):
        self.view1_list = view1_list
        self.view2_list = view2_list

    def __len__(self):
        return len(self.view1_list)

    def __getitem__(self, idx):
        return self.view1_list[idx], self.view2_list[idx]


def info_nce_loss(z1, z2, temperature=0.07):
    """
    Symmetric InfoNCE loss.
    z1, z2: [batch_size, proj_dim] L2-normalized embeddings.
    """
    batch_size = z1.size(0)
    sim_matrix = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(batch_size, device=z1.device)
    loss_12 = F.cross_entropy(sim_matrix, labels)
    loss_21 = F.cross_entropy(sim_matrix.t(), labels)
    return (loss_12 + loss_21) / 2.0


def prepare_shifted_views(trimer_graphs, psmiles_list):
    """Generate shifted trimer views for the shifted RU augmentation."""
    logger.info("Generating shifted RU views...")
    shifted_list = []
    n_failed = 0
    for i, (tri_graph, smi) in enumerate(zip(trimer_graphs, psmiles_list)):
        shifted_mol = shifted_trimer(smi)
        if shifted_mol is not None:
            shifted_data = mol_to_data(shifted_mol)
            if shifted_data is not None:
                shifted_list.append(shifted_data)
            else:
                shifted_list.append(tri_graph)
                n_failed += 1
        else:
            shifted_list.append(tri_graph)
            n_failed += 1

    if n_failed > 0:
        logger.warning(f"Shifted RU: {n_failed}/{len(trimer_graphs)} failed, using standard trimer as fallback")
    return shifted_list


def pretrain(
    augment='subgraph_crop',
    subset_size=300000,
    epochs=50,
    batch_size=32,
    accum_steps=16,
    lr=1e-3,
    weight_decay=1e-5,
    temperature=0.07,
    hidden_dim=256,
    num_layers=3,
    proj_dim=64,
    seed=42,
    corrupt_rate=0.25,
):
    """
    Contrastive pre-training with configurable augmentation.
    """
    print("=" * 60)
    print(f"Phase: Contrastive Pre-training ({augment}, corrupt_rate={corrupt_rate})")
    print("=" * 60)

    # Build a tag that includes corruption rate for saving
    if augment in ('edge_drop_feat_mask', 'hybrid_corrupt'):
        tag = f"{augment}_{int(corrupt_rate * 100)}"
    else:
        tag = augment

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load preprocessed data
    mono_path = os.path.join(OUTPUT_DIR, 'pi1m_monomer_graphs.pt')
    tri_path = os.path.join(OUTPUT_DIR, 'pi1m_trimer_graphs.pt')

    if not os.path.exists(tri_path):
        print("ERROR: pi1m_trimer_graphs.pt not found. Run preprocess_pi1m.py first.")
        return

    print("Loading preprocessed graphs...")
    tri_list = torch.load(tri_path, weights_only=False)
    print(f"Loaded {len(tri_list)} trimer graphs")

    # Prepare dataset based on augmentation strategy
    if augment == 'monomer_trimer':
        if not os.path.exists(mono_path):
            print("ERROR: pi1m_monomer_graphs.pt not found.")
            return
        mono_list = torch.load(mono_path, weights_only=False)
        dataset_v1 = mono_list
        dataset_v2 = tri_list
        print(f"Augmentation: monomer-trimer pairs ({len(mono_list)} pairs)")

    elif augment == 'subgraph_crop':
        # Subgraph cropping generates views on-the-fly
        dataset_v1 = tri_list  # Will be cropped dynamically
        dataset_v2 = tri_list
        print(f"Augmentation: subgraph cropping ({len(tri_list)} trimers)")

    elif augment == 'shifted_ru':
        # Need p-SMILES to generate shifted views
        import pandas as pd
        csv_path = os.path.join(
            os.path.dirname(OUTPUT_DIR), 'alldataset', 'PI1M.csv'
        )
        if not os.path.exists(csv_path):
            print("ERROR: PI1M.csv not found for shifted RU generation")
            return

        df = pd.read_csv(csv_path)
        smiles_col = df.columns[0]
        all_smiles = df[smiles_col].tolist()

        # Reproduce the same subset sampling as preprocessing
        np.random.seed(42)
        indices = np.random.choice(len(all_smiles), min(subset_size, len(all_smiles)), replace=False)
        psmiles_subset = [all_smiles[i] for i in indices]
        # Trim to match trimer count (some may have been filtered)
        psmiles_subset = psmiles_subset[:len(tri_list)]

        shifted_list = prepare_shifted_views(tri_list, psmiles_subset)
        dataset_v1 = tri_list
        dataset_v2 = shifted_list
        print(f"Augmentation: shifted RU pairs ({len(tri_list)} pairs)")

    elif augment == 'edge_drop_feat_mask':
        # View 1: original trimer, View 2: corrupted trimer (edge dropout + feature masking)
        dataset_v1 = tri_list
        dataset_v2 = tri_list  # corruption applied on-the-fly
        print(f"Augmentation: edge dropout + feature masking ({len(tri_list)} trimers)")

    elif augment == 'monomer_dimer':
        # Need monomer + dimer graphs
        if not os.path.exists(mono_path):
            print("ERROR: pi1m_monomer_graphs.pt not found.")
            return
        mono_list = torch.load(mono_path, weights_only=False)

        # Generate dimer graphs
        import pandas as pd
        csv_path = os.path.join(
            os.path.dirname(OUTPUT_DIR), 'alldataset', 'PI1M.csv'
        )
        if not os.path.exists(csv_path):
            print("ERROR: PI1M.csv not found for dimer generation")
            return

        df = pd.read_csv(csv_path)
        smiles_col = df.columns[0]
        all_smiles = df[smiles_col].tolist()

        np.random.seed(42)
        indices = np.random.choice(len(all_smiles), min(subset_size, len(all_smiles)), replace=False)
        psmiles_subset = [all_smiles[i] for i in indices]
        psmiles_subset = psmiles_subset[:len(tri_list)]

        print("Generating dimer graphs...")
        dimer_list = []
        for i, smi in enumerate(tqdm(psmiles_subset, desc="Dimers")):
            dimer_mol = psmiles_to_dimer(smi)
            if dimer_mol is not None:
                dimer_data = mol_to_data(dimer_mol)
                if dimer_data is not None:
                    dimer_list.append(dimer_data)
                else:
                    dimer_list.append(tri_list[i])
            else:
                dimer_list.append(tri_list[i])

        dataset_v1 = mono_list
        dataset_v2 = dimer_list
        print(f"Augmentation: monomer-dimer pairs ({len(mono_list)} pairs)")

    elif augment == 'hybrid_corrupt':
        # Monomer + Trimer with edge dropout applied to BOTH
        if not os.path.exists(mono_path):
            print("ERROR: pi1m_monomer_graphs.pt not found.")
            return
        mono_list = torch.load(mono_path, weights_only=False)
        dataset_v1 = mono_list  # corruption applied on-the-fly
        dataset_v2 = tri_list   # corruption applied on-the-fly
        print(f"Augmentation: hybrid corruption on monomer+trimer ({len(mono_list)} pairs)")

    else:
        raise ValueError(f"Unknown augmentation: {augment}")

    # Model
    model = PerioGTPretrainModel(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        proj_dim=proj_dim,
    ).to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Training log
    log = {
        'config': {
            'augment': augment,
            'tag': tag,
            'corrupt_rate': corrupt_rate,
            'subset_size': len(tri_list),
            'epochs': epochs,
            'batch_size': batch_size,
            'accum_steps': accum_steps,
            'effective_batch_size': batch_size * accum_steps,
            'lr': lr,
            'weight_decay': weight_decay,
            'temperature': temperature,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'proj_dim': proj_dim,
            'n_params': n_params,
        },
        'epoch_metrics': [],
    }

    best_loss = float('inf')
    best_epoch = 0
    n_data = len(tri_list)

    print(f"\nEffective batch size: {batch_size * accum_steps} pairs")
    print(f"Starting training for {epochs} epochs...\n")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_cos_sim = 0.0
        n_batches = 0

        # Shuffle indices each epoch
        perm = torch.randperm(n_data).tolist()
        optimizer.zero_grad()

        pbar = tqdm(range(0, n_data, batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, start_idx in enumerate(pbar):
            end_idx = min(start_idx + batch_size, n_data)
            batch_indices = perm[start_idx:end_idx]

            # Generate views based on augmentation
            if augment == 'subgraph_crop':
                from augment import random_connected_subgraph
                batch_v1 = []
                batch_v2 = []
                for idx in batch_indices:
                    sub1 = random_connected_subgraph(tri_list[idx])
                    sub2 = random_connected_subgraph(tri_list[idx])
                    batch_v1.append(sub1 if sub1 is not None else tri_list[idx])
                    batch_v2.append(sub2 if sub2 is not None else tri_list[idx])
            elif augment == 'edge_drop_feat_mask':
                batch_v1 = [dataset_v1[idx] for idx in batch_indices]  # original trimer
                batch_v2 = [edge_dropout_feat_mask(dataset_v2[idx], edge_p=corrupt_rate, feat_p=corrupt_rate)
                            for idx in batch_indices]  # corrupted trimer
            elif augment == 'hybrid_corrupt':
                batch_v1 = [edge_dropout(dataset_v1[idx], p=corrupt_rate) for idx in batch_indices]  # corrupted monomer
                batch_v2 = [edge_dropout(dataset_v2[idx], p=corrupt_rate) for idx in batch_indices]  # corrupted trimer
            else:
                batch_v1 = [dataset_v1[idx] for idx in batch_indices]
                batch_v2 = [dataset_v2[idx] for idx in batch_indices]

            # Create PyG batches
            loader_v1 = DataLoader(batch_v1, batch_size=len(batch_v1), shuffle=False)
            loader_v2 = DataLoader(batch_v2, batch_size=len(batch_v2), shuffle=False)

            v1_batch = next(iter(loader_v1)).to(device)
            v2_batch = next(iter(loader_v2)).to(device)

            with torch.cuda.amp.autocast(enabled=True):
                z1 = model(v1_batch.x, v1_batch.edge_index, v1_batch.edge_attr, v1_batch.batch)
                z2 = model(v2_batch.x, v2_batch.edge_index, v2_batch.edge_attr, v2_batch.batch)

                loss = info_nce_loss(z1, z2, temperature)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            with torch.no_grad():
                cos_sim = F.cosine_similarity(z1, z2, dim=1).mean().item()

            epoch_loss += loss.item() * accum_steps
            epoch_cos_sim += cos_sim
            n_batches += 1

            # Gradient accumulation
            if (batch_idx + 1) % accum_steps == 0 or (end_idx == n_data):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix({
                'loss': f'{loss.item() * accum_steps:.4f}',
                'cos': f'{cos_sim:.3f}'
            })

        avg_loss = epoch_loss / n_batches
        avg_cos = epoch_cos_sim / n_batches

        log['epoch_metrics'].append({
            'epoch': epoch + 1,
            'avg_loss': float(avg_loss),
            'avg_cosine_sim': float(avg_cos),
        })

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, cosine_sim={avg_cos:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            encoder_path = os.path.join(OUTPUT_DIR, f'pretrained_encoder_{tag}.pt')
            torch.save(model.encoder.state_dict(), encoder_path)
            print(f"  -> New best model saved (loss={best_loss:.4f})")

    log['best_epoch'] = best_epoch
    log['best_loss'] = float(best_loss)

    # Save training log
    log_path = os.path.join(OUTPUT_DIR, f'pretrain_log_{tag}.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining log saved to {log_path}")
    print(f"Best epoch: {best_epoch}, Best loss: {best_loss:.4f}")

    return log


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--augment', type=str, default='edge_drop_feat_mask',
                       choices=['monomer_trimer', 'subgraph_crop', 'shifted_ru',
                                'edge_drop_feat_mask', 'monomer_dimer', 'hybrid_corrupt'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accum_steps', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--corrupt_rate', type=float, default=0.25,
                       help='Corruption rate for edge dropout and feature masking')
    args = parser.parse_args()

    pretrain(
        augment=args.augment,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        temperature=args.temperature,
        corrupt_rate=args.corrupt_rate,
    )
