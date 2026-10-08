"""
Training script for PIPGA on Tg dataset
5-fold cross-validation with early stopping
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model import PIPGA, polymer_smiles_to_graph
import warnings
warnings.filterwarnings('ignore')


def load_data(csv_path):
    """Load and preprocess Tg dataset"""
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")

    # Filter out invalid entries
    df = df.dropna(subset=['Smiles', 'Tg'])
    print(f"After removing NaN: {len(df)} samples")

    # Convert to graphs
    graphs = []
    mols = []
    targets = []

    failed = 0
    for idx, row in df.iterrows():
        smiles = row['Smiles']
        target = row['Tg']

        result = polymer_smiles_to_graph(smiles)
        if result[0] is not None:
            graph, mol = result
            graphs.append(graph)
            mols.append(mol)
            targets.append(target)
        else:
            failed += 1

    print(f"Successfully converted {len(graphs)} graphs, failed {failed}")

    return graphs, mols, targets


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        mol = batch.mol if hasattr(batch, 'mol') else None
        head_mask = batch.head_mask if hasattr(batch, 'head_mask') else None
        tail_mask = batch.tail_mask if hasattr(batch, 'tail_mask') else None

        optimizer.zero_grad()

        # Forward pass
        pred = model(batch, mol, head_mask, tail_mask)
        target = batch.y.to(device)

        loss = criterion(pred, target)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    for batch in loader:
        batch = batch.to(device)
        mol = batch.mol if hasattr(batch, 'mol') else None
        head_mask = batch.head_mask if hasattr(batch, 'head_mask') else None
        tail_mask = batch.tail_mask if hasattr(batch, 'tail_mask') else None

        pred = model(batch, mol, head_mask, tail_mask)
        target = batch.y.to(device)

        loss = criterion(pred, target)
        total_loss += loss.item() * batch.num_graphs

        predictions.extend(pred.cpu().numpy())
        targets.extend(target.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)

    return total_loss / len(loader.dataset), rmse, mae


class MolData(Data):
    """Custom data class to store mol object"""
    def __init__(self, mol=None, **kwargs):
        super().__init__(**kwargs)
        self.mol = mol


def collate_fn(batch):
    """Custom collate function to handle mol objects"""
    from torch_geometric.data import Batch

    # Separate graphs and mols
    graphs = [item[0] for item in batch]
    mols = [item[1] for item in batch]
    targets = [item[2] for item in batch]

    # Create batch
    batched_graph = Batch.from_data_list(graphs)
    batched_graph.y = torch.tensor(targets, dtype=torch.float)
    batched_graph.mols = mols  # Store mols list

    return batched_graph


def run_fold(train_graphs, train_mols, train_targets,
             val_graphs, val_mols, val_targets,
             fold_idx, device, args):
    """Run training for one fold"""

    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}")
    print(f"{'='*60}")

    # Create data loaders
    train_data = list(zip(train_graphs, train_mols, train_targets))
    val_data = list(zip(val_graphs, val_mols, val_targets))

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args['batch_size'], shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_fn
    )

    # Initialize model
    model = PIPGA(
        input_dim=43,
        hidden_dim=args['hidden_dim'],
        num_layers=args['num_layers'],
        num_heads=args['num_heads'],
        edge_dim=5,
        dropout=args['dropout'],
        use_ecfp4=True,
        ecfp4_bits=1024
    ).to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    criterion = nn.MSELoss()

    # Training loop
    best_val_rmse = float('inf')
    best_val_mae = float('inf')
    patience_counter = 0

    for epoch in range(args['epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_rmse, val_mae = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_rmse)

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_val_mae = val_mae
            patience_counter = 0

            # Save best model
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/best_fold_{fold_idx}.pt')
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args['epochs']} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val RMSE: {val_rmse:.4f} - Val MAE: {val_mae:.4f} - "
                  f"Best Val RMSE: {best_val_rmse:.4f}")

        if patience_counter >= args['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(torch.load(f'checkpoints/best_fold_{fold_idx}.pt'))

    return model, best_val_rmse, best_val_mae


def main():
    # Arguments
    args = {
        'hidden_dim': 256,
        'num_layers': 3,
        'num_heads': 4,
        'dropout': 0.1,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'batch_size': 32,
        'epochs': 500,
        'patience': 50,
        'n_folds': 5
    }

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    csv_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    graphs, mols, targets = load_data(csv_path)

    # Normalize targets
    targets = np.array(targets)
    target_mean = targets.mean()
    target_std = targets.std()
    targets_normalized = (targets - target_mean) / target_std

    print(f"\nDataset statistics:")
    print(f"  Mean Tg: {target_mean:.2f}")
    print(f"  Std Tg: {target_std:.2f}")
    print(f"  Min Tg: {targets.min():.2f}")
    print(f"  Max Tg: {targets.max():.2f}")

    # 5-fold cross-validation
    kf = KFold(n_splits=args['n_folds'], shuffle=True, random_state=42)

    fold_results = []
    all_val_rmses = []
    all_val_maes = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(graphs)):
        # Split data
        train_graphs = [graphs[i] for i in train_idx]
        train_mols = [mols[i] for i in train_idx]
        train_targets = targets_normalized[train_idx]

        val_graphs = [graphs[i] for i in val_idx]
        val_mols = [mols[i] for i in val_idx]
        val_targets = targets_normalized[val_idx]

        # Train fold
        model, val_rmse_norm, val_mae_norm = run_fold(
            train_graphs, train_mols, train_targets,
            val_graphs, val_mols, val_targets,
            fold_idx, device, args
        )

        # Denormalize metrics
        val_rmse = val_rmse_norm * target_std
        val_mae = val_mae_norm * target_std

        all_val_rmses.append(val_rmse)
        all_val_maes.append(val_mae)

        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  Val RMSE: {val_rmse:.4f}")
        print(f"  Val MAE: {val_mae:.4f}")

        fold_results.append({
            'fold': fold_idx + 1,
            'val_rmse': val_rmse,
            'val_mae': val_mae
        })

    # Calculate final statistics
    mean_rmse = np.mean(all_val_rmses)
    std_rmse = np.std(all_val_rmses)
    mean_mae = np.mean(all_val_maes)
    std_mae = np.std(all_val_maes)

    print(f"\n{'='*60}")
    print("FINAL RESULTS (5-Fold Cross-Validation)")
    print(f"{'='*60}")
    print(f"RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"MAE:  {mean_mae:.4f} ± {std_mae:.4f}")
    print(f"{'='*60}")

    # Save results
    with open('results.txt', 'w') as f:
        f.write("PIPGA Results on Tg Dataset\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model Configuration:\n")
        for key, value in args.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nDataset Statistics:\n")
        f.write(f"  Total samples: {len(graphs)}\n")
        f.write(f"  Target mean: {target_mean:.2f}\n")
        f.write(f"  Target std: {target_std:.2f}\n")
        f.write(f"  Target range: [{targets.min():.2f}, {targets.max():.2f}]\n")
        f.write(f"\n5-Fold Cross-Validation Results:\n")
        for result in fold_results:
            f.write(f"  Fold {result['fold']}: RMSE = {result['val_rmse']:.4f}, MAE = {result['val_mae']:.4f}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"FINAL RESULTS:\n")
        f.write(f"  RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}\n")
        f.write(f"  MAE:  {mean_mae:.4f} ± {std_mae:.4f}\n")
        f.write(f"{'='*60}\n")

    print("\nResults saved to results.txt")

    return mean_rmse, std_rmse


if __name__ == '__main__':
    main()
