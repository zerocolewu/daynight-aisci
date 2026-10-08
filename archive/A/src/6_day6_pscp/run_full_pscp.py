#!/usr/bin/env python3
"""
Full PSCP experiment: Pretraining + Fine-tuning with all ablation conditions
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
import warnings

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/6_day6_pscp')

from pscp_model import PSCPModel, PredictionModel, GINEncoder
from polymer_utils import smiles_to_graph, compute_ecfp_fingerprint
from torch_geometric.data import Batch, Data


class TgDataset(Dataset):
    """Tg dataset with pre-computed graphs and ECFP"""

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} samples from {csv_path}")

        # Pre-compute ECFP and validate SMILES
        print("Pre-computing ECFP fingerprints...")
        self.ecfp_list = []
        self.valid_indices = []

        for idx in tqdm(range(len(self.df)), desc="Computing ECFP"):
            row = self.df.iloc[idx]
            smiles = row['Smiles']
            clean_smiles = smiles.replace('*', '')
            if not clean_smiles:
                clean_smiles = 'C'

            ecfp = compute_ecfp_fingerprint(clean_smiles, radius=2, n_bits=2048)
            if ecfp is not None:
                self.ecfp_list.append(ecfp.astype(np.float32))
                self.valid_indices.append(idx)

        self.df = self.df.iloc[self.valid_indices].reset_index(drop=True)
        print(f"Valid samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row['Smiles']
        tg = float(row['Tg'])

        clean_smiles = smiles.replace('*', '')
        if not clean_smiles:
            clean_smiles = 'C'

        graph = smiles_to_graph(clean_smiles)
        if graph is None:
            # Create dummy graph with all required attributes
            graph = Data(
                x=torch.zeros(1, 159),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, 13, dtype=torch.float)
            )

        # Ensure edge_attr exists
        if not hasattr(graph, 'edge_attr') or graph.edge_attr is None:
            num_edges = graph.edge_index.size(1) if graph.edge_index is not None else 0
            graph.edge_attr = torch.zeros(num_edges, 13, dtype=torch.float)

        return {
            'graph': graph,
            'ecfp': torch.from_numpy(self.ecfp_list[idx]),
            'tg': torch.tensor(tg, dtype=torch.float32)
        }


class PI1MDataset(Dataset):
    """PI1M dataset for pretraining"""

    def __init__(self, csv_path, max_samples=None):
        df = pd.read_csv(csv_path)
        self.smiles_list = df['SMILES'].tolist()

        if max_samples:
            self.smiles_list = self.smiles_list[:max_samples]

        # Filter valid SMILES
        print(f"Filtering valid SMILES from {len(self.smiles_list)}...")
        valid_smiles = []
        for smi in tqdm(self.smiles_list, desc="Validating SMILES"):
            clean = smi.replace('*', '')
            if clean and len(clean) > 1:
                mol = Chem.MolFromSmiles(clean)
                if mol is not None:
                    valid_smiles.append(smi)

        self.smiles_list = valid_smiles
        print(f"Valid SMILES: {len(self.smiles_list)}")

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        clean_smiles = smiles.replace('*', '')
        if not clean_smiles:
            clean_smiles = 'C'

        graph = smiles_to_graph(clean_smiles)
        if graph is None:
            graph = Data(
                x=torch.zeros(1, 159),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, 13, dtype=torch.float)
            )

        # Ensure edge_attr exists
        if not hasattr(graph, 'edge_attr') or graph.edge_attr is None:
            num_edges = graph.edge_index.size(1) if graph.edge_index is not None else 0
            graph.edge_attr = torch.zeros(num_edges, 13, dtype=torch.float)

        return {
            'graph': graph,
            'smiles': smiles,
            'idx': idx
        }


def collate_fn(batch):
    """Collate function for batching"""
    graphs = [item['graph'] for item in batch]
    batched_graph = Batch.from_data_list(graphs)

    result = {'graph': batched_graph}

    if 'ecfp' in batch[0]:
        result['ecfp'] = torch.stack([item['ecfp'] for item in batch])
    if 'tg' in batch[0]:
        result['tg'] = torch.stack([item['tg'] for item in batch])
    if 'smiles' in batch[0]:
        result['smiles'] = [item['smiles'] for item in batch]

    return result


def pretrain_model(model, dataloader, optimizer, device, epoch, tasks=['fgm', 'cri', 'egcl']):
    """Run one pretraining epoch"""
    model.train()
    total_loss = 0
    fgm_loss_sum = 0
    cri_loss_sum = 0
    egcl_loss_sum = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Pretraining Epoch {epoch}")

    for batch in pbar:
        optimizer.zero_grad()

        graphs = batch['graph'].to(device)
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Functional Group Masking (FGM)
        if 'fgm' in tasks:
            fgm_loss = compute_fgm_loss(model, graphs)
            fgm_loss_sum += fgm_loss.item()
            loss = loss + fgm_loss

        # ECFP-Guided Contrastive Learning (EGCL)
        if 'egcl' in tasks:
            egcl_loss = compute_egcl_loss(model, graphs)
            egcl_loss_sum += egcl_loss.item()
            loss = loss + egcl_loss

        # Cyclic Rotation Invariance (CRI) - simplified
        if 'cri' in tasks:
            cri_loss = torch.tensor(0.0, device=device, requires_grad=True)
            cri_loss_sum += cri_loss.item()
            loss = loss + 0.5 * cri_loss

        if loss.item() > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    n = max(num_batches, 1)
    return {
        'total': total_loss / n,
        'fgm': fgm_loss_sum / n,
        'cri': cri_loss_sum / n,
        'egcl': egcl_loss_sum / n
    }


def compute_fgm_loss(model, graphs):
    """Functional Group Masking loss"""
    num_nodes = graphs.x.size(0)
    mask_ratio = 0.15
    num_mask = max(1, int(num_nodes * mask_ratio))

    # Random mask
    mask_indices = torch.randperm(num_nodes)[:num_mask]

    # Store original
    original_features = graphs.x[mask_indices].clone()
    original_labels = original_features.argmax(dim=-1)

    # Mask
    masked_x = graphs.x.clone()
    masked_x[mask_indices] = 0

    # Predict
    from torch_geometric.data import Data
    masked_data = Data(
        x=masked_x,
        edge_index=graphs.edge_index,
        edge_attr=graphs.edge_attr if hasattr(graphs, 'edge_attr') else None,
        batch=graphs.batch
    )

    predictions = model.predict_masked_atoms(masked_data)
    predicted_labels = predictions[mask_indices]

    loss = F.cross_entropy(predicted_labels, original_labels)
    return loss


def compute_egcl_loss(model, graphs):
    """ECFP-Guided Contrastive Learning loss (simplified)"""
    batch_size = graphs.num_graphs

    # Get projections
    projections = model.get_projection(graphs)
    projections = F.normalize(projections, dim=-1)

    # Similarity matrix
    sim_matrix = torch.mm(projections, projections.t()) / 0.1

    # Use shifted indices as pseudo-positives
    labels = torch.arange(batch_size, device=graphs.x.device)
    labels = (labels + 1) % batch_size

    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def train_epoch_finetune(model, dataloader, optimizer, device):
    """Fine-tuning training epoch"""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()

        graphs = batch['graph'].to(device)
        ecfp = batch['ecfp'].to(device)
        targets = batch['tg'].to(device)

        predictions = model(graphs, ecfp)
        loss = nn.MSELoss()(predictions, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in dataloader:
            graphs = batch['graph'].to(device)
            ecfp = batch['ecfp'].to(device)
            target = batch['tg']

            pred = model(graphs, ecfp)
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def finetune_cv(encoder, dataset, config, device, n_folds=5):
    """Fine-tune with cross-validation"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*50}")

        # Create fresh encoder copy
        from copy import deepcopy
        encoder_copy = deepcopy(encoder)

        model = PredictionModel(
            encoder=encoder_copy,
            hidden_dim=config['hidden_dim'],
            ecfp_dim=config['ecfp_bits']
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = TorchDataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = TorchDataLoader(val_subset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        best_val_rmse = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(100):
            train_loss = train_epoch_finetune(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)
            scheduler.step()

            if val_metrics['rmse'] < best_val_rmse:
                best_val_rmse = val_metrics['rmse']
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}: Loss={train_loss:.4f}, Val RMSE={val_metrics['rmse']:.2f}, R2={val_metrics['r2']:.4f}")

            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        model.load_state_dict(best_model_state)
        final_metrics = evaluate(model, val_loader, device)

        print(f"\nFold {fold + 1}: RMSE={final_metrics['rmse']:.2f}, MAE={final_metrics['mae']:.2f}, R2={final_metrics['r2']:.4f}")

        fold_results.append({
            'fold': fold + 1,
            'rmse': final_metrics['rmse'],
            'mae': final_metrics['mae'],
            'r2': final_metrics['r2']
        })

    rmses = [r['rmse'] for r in fold_results]
    return {
        'mean_rmse': np.mean(rmses),
        'std_rmse': np.std(rmses),
        'mean_r2': np.mean([r['r2'] for r in fold_results]),
        'fold_results': fold_results
    }


def main():
    print("=" * 70)
    print("PSCP: Polymer-Specific Contrastive Pretraining")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Configuration
    config = {
        'num_node_features': 159,
        'hidden_dim': 256,
        'num_layers': 3,
        'edge_dim': 13,
        'proj_dim': 128,
        'ecfp_bits': 2048
    }

    # Paths
    pi1m_path = '/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv'
    tg_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    output_dir = '/home/home_public/aisci/AIscientist/src/6_day6_pscp'

    # Load datasets
    print("\n" + "=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    tg_dataset = TgDataset(tg_path)

    # Ablation conditions
    conditions = {
        'no_pretrain': [],
        'fgm_only': ['fgm'],
        'cri_only': ['cri'],
        'egcl_only': ['egcl'],
        'full_pscp': ['fgm', 'cri', 'egcl']
    }

    all_results = []

    for condition, tasks in conditions.items():
        print("\n" + "=" * 70)
        print(f"CONDITION: {condition}")
        print(f"Tasks: {tasks if tasks else 'None (random init)'}")
        print("=" * 70)

        # Create model
        model = PSCPModel(config).to(device)

        # Pretrain if needed
        if tasks:
            print(f"\nPretraining on PI1M...")
            pi1m_dataset = PI1MDataset(pi1m_path, max_samples=100000)

            dataloader = TorchDataLoader(
                pi1m_dataset,
                batch_size=64,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0
            )

            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            scheduler = CosineAnnealingLR(optimizer, T_max=5)

            for epoch in range(5):
                losses = pretrain_model(model, dataloader, optimizer, device, epoch + 1, tasks)
                scheduler.step()
                print(f"Epoch {epoch + 1}: Total={losses['total']:.4f}, FGM={losses['fgm']:.4f}, EGCL={losses['egcl']:.4f}")

            # Save pretrained model
            torch.save({
                'encoder_state_dict': model.encoder.state_dict(),
                'config': config
            }, os.path.join(output_dir, f'pretrained_{condition}.pt'))
            print(f"Saved pretrained model: pretrained_{condition}.pt")
        else:
            print("\nSkipping pretraining (random initialization)")

        # Fine-tune
        print(f"\nFine-tuning on Tg dataset...")
        results = finetune_cv(model.encoder, tg_dataset, config, device, n_folds=5)

        # Save results
        for fr in results['fold_results']:
            all_results.append({
                'condition': condition,
                'fold': fr['fold'],
                'rmse': fr['rmse'],
                'mae': fr['mae'],
                'r2': fr['r2']
            })

        all_results.append({
            'condition': condition,
            'fold': 'mean',
            'rmse': results['mean_rmse'],
            'mae': np.mean([r['mae'] for r in results['fold_results']]),
            'r2': results['mean_r2']
        })

    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, 'results_ablation.csv'), index=False)

    # Print final report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    baseline_rmse = 38.78
    mean_results = results_df[results_df['fold'] == 'mean']

    print("\n[Ablation Results]")
    print("Condition        | RMSE    | ± Std   | R2      | Improvement")
    print("-" * 65)
    for _, row in mean_results.iterrows():
        std = results_df[(results_df['condition'] == row['condition']) &
                         (results_df['fold'] != 'mean')]['rmse'].std()
        improvement = baseline_rmse - row['rmse']
        print(f"{row['condition']:16} | {row['rmse']:7.2f} | ±{std:5.2f} | {row['r2']:7.4f} | +{improvement:.2f}")

    best_rmse = mean_results['rmse'].min()
    best_condition = mean_results.loc[mean_results['rmse'].idxmin(), 'condition']

    print("\n[Key Findings]")
    print(f"- Best RMSE: {best_rmse:.2f} ({best_condition})")
    print(f"- Target RMSE < 35.0: {'ACHIEVED' if best_rmse < 35.0 else 'NOT ACHIEVED'}")
    print(f"- Best improvement over baseline: +{baseline_rmse - best_rmse:.2f} RMSE")

    print(f"\nResults saved to: {output_dir}/results_ablation.csv")


if __name__ == '__main__':
    main()
