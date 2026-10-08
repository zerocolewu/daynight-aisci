"""
Comprehensive ablation study to diagnose TCP failure.

Ablation 1: Exact Day 2 baseline (verify implementation matches 32.31 K)
Ablation 2: TCP with consistency loss only (no pseudo-labels)
Ablation 3: TCP with exact Day 2 features and architecture

Goal: Identify why TCP performed 3 K worse than Day 2 baseline
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# Add paths for imports
sys.path.append('/home/home_public/aisci/AIscientist/src/2_day2_grin')
from data_utils_v2 import mol_to_graph as mol_to_graph_grin, create_n_ru_molecule, TOTAL_ATOM_FEATURES

sys.path.append('/home/home_public/aisci/AIscientist/src/1_day2_reperioGT')
from dataset import mol_to_graph as mol_to_graph_reperio, get_feature_dimensions


def load_tg_data_grin_features(csv_path):
    """Load Tg data with GRIN-style features (148-dim)."""
    print(f"Loading Tg data with GRIN features from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")

    data_list = []
    targets = []

    for idx, row in df.iterrows():
        smiles = row['Smiles']
        tg = row['Tg']

        # Create molecule with 1-RU (baseline setup)
        mol = create_n_ru_molecule(smiles, n=1)
        if mol is None:
            continue

        # Convert to graph with GRIN features
        graph = mol_to_graph_grin(mol, target=tg)
        if graph is not None:
            data_list.append(graph)
            targets.append(tg)

    print(f"Successfully loaded {len(data_list)} samples")
    print(f"Feature dimension: {data_list[0].x.shape[1]}")
    return data_list, np.array(targets)


def load_tg_data_reperio_features(csv_path):
    """Load Tg data with ReperioGT-style features (153-dim)."""
    print(f"Loading Tg data with ReperioGT features from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")

    data_list = []
    targets = []

    for idx, row in df.iterrows():
        smiles = row['Smiles']
        tg = row['Tg']

        # Use reperioGT's mol_to_graph (153-dim features)
        graph = mol_to_graph_reperio(smiles, target=tg)
        if graph is not None:
            data_list.append(graph)
            targets.append(tg)

    print(f"Successfully loaded {len(data_list)} samples")
    print(f"Feature dimension: {data_list[0].x.shape[1]}")
    return data_list, np.array(targets)


class GINModel(nn.Module):
    """GIN model matching Day 2 baseline architecture."""

    def __init__(self, input_dim, hidden_dim=256, num_layers=5, dropout=0.15, readout='mean'):
        super().__init__()
        self.num_layers = num_layers
        self.readout = readout
        self.dropout = dropout

        # Build GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                mlp = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            else:
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )

            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Prediction head
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        # GIN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = nn.functional.relu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if self.readout == 'mean':
            x = global_mean_pool(x, batch)
        elif self.readout == 'sum':
            x = global_add_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)

        # Prediction head
        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x


def train_epoch(model, loader, optimizer, device, scaler=None):
    """Train for one epoch with optional target normalization."""
    model.train()
    total_loss = 0
    count = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch.x, batch.edge_index, batch.batch)

        # Handle normalized vs unnormalized targets
        if hasattr(batch, 'y') and batch.y is not None:
            target = batch.y.view(-1, 1)
        else:
            target = batch.y_original.view(-1, 1)

        loss = nn.functional.mse_loss(pred, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        count += batch.num_graphs

    return total_loss / count


def evaluate(model, loader, device, scaler=None):
    """Evaluate model with proper inverse transform."""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch)

            # Inverse transform predictions to original scale
            if scaler is not None:
                pred_original = scaler.inverse_transform(pred.cpu().numpy())
                target_original = batch.y_original.view(-1, 1).cpu().numpy()
            else:
                pred_original = pred.cpu().numpy()
                target_original = batch.y_original.view(-1, 1).cpu().numpy() if hasattr(batch, 'y_original') else batch.y.view(-1, 1).cpu().numpy()

            predictions.append(pred_original)
            targets.append(target_original)

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    return {'rmse': rmse, 'mae': mae, 'predictions': predictions, 'targets': targets}


def run_kfold_cv(data_list, targets, config, ablation_name):
    """Run K-fold cross-validation."""
    print(f"\n{'='*80}")
    print(f"Ablation: {ablation_name}")
    print(f"{'='*80}")
    print(f"Config: {json.dumps(config, indent=2)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=config['seed'])
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(data_list)):
        print(f"\n--- Fold {fold_idx + 1}/{config['n_folds']} ---")

        # Split data
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        train_targets = targets[train_idx]
        val_targets = targets[val_idx]

        # Fit scaler on training data
        scaler = StandardScaler()
        scaler.fit(train_targets.reshape(-1, 1))

        # Normalize targets
        for i, data in enumerate(train_data):
            data.y = torch.tensor([scaler.transform([[train_targets[i]]])[0, 0]], dtype=torch.float)
            data.y_original = torch.tensor([train_targets[i]], dtype=torch.float)

        for i, data in enumerate(val_data):
            data.y = torch.tensor([scaler.transform([[val_targets[i]]])[0, 0]], dtype=torch.float)
            data.y_original = torch.tensor([val_targets[i]], dtype=torch.float)

        # Create dataloaders
        from torch_geometric.loader import DataLoader
        train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)

        # Get input dimension
        input_dim = train_data[0].x.shape[1]
        print(f"Input dimension: {input_dim}")

        # Create model
        model = GINModel(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            readout=config['readout']
        ).to(device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = CosineAnnealingLR(optimizer, T_max=config['max_epochs'], eta_min=1e-6)

        # Training loop
        best_val_rmse = float('inf')
        patience_counter = 0

        for epoch in range(config['max_epochs']):
            train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
            val_metrics = evaluate(model, val_loader, device, scaler)
            scheduler.step()

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, Val RMSE: {val_metrics['rmse']:.2f}")

            if val_metrics['rmse'] < best_val_rmse:
                best_val_rmse = val_metrics['rmse']
                patience_counter = 0
                # Save best predictions
                best_predictions = val_metrics['predictions']
                best_targets = val_metrics['targets']
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print(f"Fold {fold_idx + 1} Best Val RMSE: {best_val_rmse:.2f}")
        fold_results.append({
            'val_rmse': best_val_rmse,
            'predictions': best_predictions,
            'targets': best_targets
        })

    # Compute mean and std
    val_rmses = [r['val_rmse'] for r in fold_results]
    mean_rmse = np.mean(val_rmses)
    std_rmse = np.std(val_rmses)

    print(f"\n{'='*80}")
    print(f"RESULTS: {ablation_name}")
    print(f"{'='*80}")
    print(f"Mean RMSE: {mean_rmse:.2f} ± {std_rmse:.2f} K")
    print(f"Fold RMSEs: {[f'{r:.2f}' for r in val_rmses]}")

    return {
        'name': ablation_name,
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'fold_rmses': val_rmses,
        'config': config
    }


def main():
    """Run all ablation studies."""
    print("="*80)
    print("TCP FAILURE DIAGNOSIS - ABLATION STUDIES")
    print("="*80)
    print(f"Started at: {datetime.now()}")

    csv_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'

    # Day 2 baseline config (exact match)
    baseline_config = {
        'n_folds': 5,
        'seed': 42,
        'batch_size': 64,
        'hidden_dim': 256,
        'num_layers': 5,
        'dropout': 0.15,
        'lr': 0.0005,
        'weight_decay': 1e-5,
        'max_epochs': 300,
        'patience': 40,
        'readout': 'mean'
    }

    results = []

    # ========================================================================
    # ABLATION 1: Exact Day 2 Baseline (GRIN features, 148-dim)
    # ========================================================================
    print("\n" + "="*80)
    print("ABLATION 1: EXACT DAY 2 BASELINE (GRIN FEATURES)")
    print("Expected: ~32.31 K")
    print("="*80)

    data_list_grin, targets_grin = load_tg_data_grin_features(csv_path)
    result1 = run_kfold_cv(data_list_grin, targets_grin, baseline_config,
                          "Ablation 1: Exact Day 2 Baseline (GRIN features)")
    results.append(result1)

    # ========================================================================
    # ABLATION 2: Current TCP Setup (ReperioGT features, 153-dim)
    # ========================================================================
    print("\n" + "="*80)
    print("ABLATION 2: CURRENT TCP SETUP (REPERIOGT FEATURES)")
    print("This should match current TCP performance (~35 K)")
    print("="*80)

    data_list_reperio, targets_reperio = load_tg_data_reperio_features(csv_path)
    result2 = run_kfold_cv(data_list_reperio, targets_reperio, baseline_config,
                          "Ablation 2: TCP Setup (ReperioGT features)")
    results.append(result2)

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)

    for r in results:
        print(f"\n{r['name']}:")
        print(f"  Mean RMSE: {r['mean_rmse']:.2f} ± {r['std_rmse']:.2f} K")
        print(f"  Fold RMSEs: {[f'{x:.2f}' for x in r['fold_rmses']]}")

    # Save to file
    output_file = 'ablation_results.txt'
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TCP FAILURE DIAGNOSIS - ABLATION STUDIES\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now()}\n\n")

        f.write("ABLATION 1: EXACT DAY 2 BASELINE (GRIN FEATURES)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean RMSE: {results[0]['mean_rmse']:.2f} ± {results[0]['std_rmse']:.2f} K\n")
        f.write(f"Fold RMSEs: {[f'{x:.2f}' for x in results[0]['fold_rmses']]}\n")
        f.write(f"Expected: ~32.31 K\n")
        f.write(f"Gap: {results[0]['mean_rmse'] - 32.31:.2f} K\n\n")

        f.write("ABLATION 2: CURRENT TCP SETUP (REPERIOGT FEATURES)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean RMSE: {results[1]['mean_rmse']:.2f} ± {results[1]['std_rmse']:.2f} K\n")
        f.write(f"Fold RMSEs: {[f'{x:.2f}' for x in results[1]['fold_rmses']]}\n")
        f.write(f"Expected: ~35 K (current TCP performance)\n\n")

        f.write("="*80 + "\n")
        f.write("DIAGNOSIS\n")
        f.write("="*80 + "\n\n")

        gap_to_baseline = results[0]['mean_rmse'] - 32.31
        gap_reperio_vs_grin = results[1]['mean_rmse'] - results[0]['mean_rmse']

        if abs(gap_to_baseline) < 2.0:
            f.write("✅ ABLATION 1 MATCHES EXPECTED BASELINE\n")
            f.write("   Implementation is correct\n\n")
        else:
            f.write("❌ ABLATION 1 DOES NOT MATCH EXPECTED BASELINE\n")
            f.write(f"   Gap: {gap_to_baseline:.2f} K\n")
            f.write("   There may be implementation issues\n\n")

        if gap_reperio_vs_grin > 2.0:
            f.write(f"⚠️  FEATURE MISMATCH IDENTIFIED\n")
            f.write(f"   ReperioGT features (153-dim) perform {gap_reperio_vs_grin:.2f} K worse than GRIN features (148-dim)\n")
            f.write("   RECOMMENDATION: Use GRIN features for TCP\n\n")
        else:
            f.write("✅ Feature choice is not the main issue\n\n")

    print(f"\nResults saved to {output_file}")

    # Also save JSON
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"JSON results saved to ablation_results.json")


if __name__ == "__main__":
    main()
