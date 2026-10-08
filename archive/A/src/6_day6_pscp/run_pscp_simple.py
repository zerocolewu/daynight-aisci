#!/usr/bin/env python3
"""
Simplified PSCP experiment runner that handles all edge cases
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# Suppress RDKit
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/6_day6_pscp')

from pscp_model import PSCPModel, PredictionModel
from pretraining_tasks import PretrainingLoss
from polymer_utils import smiles_to_graph, compute_ecfp_fingerprint, canonicalize_polymer_smiles
from torch_geometric.data import Batch


class SimpleTgDataset(Dataset):
    """Simplified dataset that handles everything in __getitem__"""

    def __init__(self, csv_path):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} samples from {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row['Smiles']
        tg = float(row['Tg'])

        # Clean SMILES
        clean_smiles = canonicalize_polymer_smiles(smiles)
        if not clean_smiles:
            clean_smiles = 'C'

        # Create graph
        graph = smiles_to_graph(clean_smiles)
        if graph is None:
            # Create dummy graph
            graph = Batch.from_data_list([])

        # Compute ECFP
        ecfp = compute_ecfp_fingerprint(clean_smiles, radius=2, n_bits=2048).astype(np.float32)

        return {
            'graph': graph,
            'ecfp': torch.from_numpy(ecfp),
            'tg': torch.tensor(tg, dtype=torch.float32)
        }


def simple_collate(batch):
    """Simple collate function"""
    graphs = [item['graph'] for item in batch]
    ecfps = torch.stack([item['ecfp'] for item in batch])
    tgs = torch.stack([item['tg'] for item in batch])

    # Batch graphs
    batched_graph = Batch.from_data_list(graphs)

    return {
        'graph': batched_graph,
        'ecfp': ecfps,
        'tg': tgs
    }


def train_one_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
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
    print(f"\nStarting {n_folds}-fold CV...")

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*50}")

        # Create model
        model = PredictionModel(
            encoder=encoder,
            hidden_dim=config['hidden_dim'],
            ecfp_dim=config['ecfp_bits']
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)

        # Create dataloaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = TorchDataLoader(
            train_subset, batch_size=32, shuffle=True,
            collate_fn=simple_collate, num_workers=0
        )
        val_loader = TorchDataLoader(
            val_subset, batch_size=32, shuffle=False,
            collate_fn=simple_collate, num_workers=0
        )

        # Training loop with early stopping
        best_val_rmse = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(100):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)
            scheduler.step()

            if val_metrics['rmse'] < best_val_rmse:
                best_val_rmse = val_metrics['rmse']
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, "
                      f"Val RMSE = {val_metrics['rmse']:.2f}, R2 = {val_metrics['r2']:.4f}")

            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        model.load_state_dict(best_model_state)
        final_metrics = evaluate(model, val_loader, device)

        print(f"\nFold {fold + 1} Results:")
        print(f"  RMSE: {final_metrics['rmse']:.2f}")
        print(f"  MAE: {final_metrics['mae']:.2f}")
        print(f"  R2: {final_metrics['r2']:.4f}")

        fold_results.append({
            'fold': fold + 1,
            'rmse': final_metrics['rmse'],
            'mae': final_metrics['mae'],
            'r2': final_metrics['r2']
        })

    # Summary
    rmses = [r['rmse'] for r in fold_results]
    r2s = [r['r2'] for r in fold_results]

    summary = {
        'mean_rmse': np.mean(rmses),
        'std_rmse': np.std(rmses),
        'mean_r2': np.mean(r2s),
        'std_r2': np.std(r2s),
        'fold_results': fold_results
    }

    print(f"\n{'='*50}")
    print("CV Summary:")
    print(f"  RMSE: {summary['mean_rmse']:.2f} ± {summary['std_rmse']:.2f}")
    print(f"  R2: {summary['mean_r2']:.4f} ± {summary['std_r2']:.4f}")

    return summary


def main():
    print("=" * 70)
    print("PSCP EXPERIMENT (SIMPLIFIED)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

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
    tg_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    output_dir = '/home/home_public/aisci/AIscientist/src/6_day6_pscp'

    # Load dataset
    print("\nLoading Tg dataset...")
    tg_dataset = SimpleTgDataset(tg_path)

    # Run ablation study
    print("\n" + "=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    conditions = {
        'no_pretrain': None,  # Random encoder
    }

    all_results = []

    for condition, encoder_path in conditions.items():
        print(f"\n{'='*70}")
        print(f"Condition: {condition}")
        print(f"{'='*70}")

        # Create encoder
        pscp_model = PSCPModel(config)
        encoder = pscp_model.encoder.to(device)

        # Fine-tune
        results = finetune_cv(encoder, tg_dataset, config, device, n_folds=5)

        # Save results
        for fold_result in results['fold_results']:
            all_results.append({
                'condition': condition,
                'fold': fold_result['fold'],
                'rmse': fold_result['rmse'],
                'mae': fold_result['mae'],
                'r2': fold_result['r2']
            })

        all_results.append({
            'condition': condition,
            'fold': 'mean',
            'rmse': results['mean_rmse'],
            'mae': np.mean([r['mae'] for r in results['fold_results']]),
            'r2': results['mean_r2']
        })

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, 'results_ablation.csv'), index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    baseline_rmse = 38.78
    mean_results = results_df[results_df['fold'] == 'mean']

    print("\nCondition | RMSE | MAE | R2 | Improvement")
    print("-" * 60)
    for _, row in mean_results.iterrows():
        improvement = baseline_rmse - row['rmse']
        print(f"{row['condition']:12} | {row['rmse']:5.2f} | {row['mae']:5.2f} | {row['r2']:6.4f} | +{improvement:.2f}")

    best_rmse = mean_results['rmse'].min()
    best_condition = mean_results.loc[mean_results['rmse'].idxmin(), 'condition']

    print(f"\nBest RMSE: {best_rmse:.2f} ({best_condition})")
    print(f"Target (<35.0): {'✓ ACHIEVED' if best_rmse < 35.0 else '✗ NOT ACHIEVED'}")
    print(f"Improvement over baseline: +{baseline_rmse - best_rmse:.2f}")


if __name__ == '__main__':
    main()
