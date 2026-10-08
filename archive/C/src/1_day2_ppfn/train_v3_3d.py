"""
Training script for PPFN-v3 with 3D Conformer Features.

Training strategy:
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau (patience=15, factor=0.5)
- Loss: Huber loss (delta=1.0)
- Batch size: 32
- Max epochs: 300 with early stopping (patience=30)
- Label normalization using StandardScaler
- Multiple configurations and ensemble
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress RDKit warnings for cleaner output
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings('ignore')

# Local imports
from dataset_v3_3d import PPFNv3Dataset, collate_fn_v3
from model_v3_3d import PPFNv3_3D, PPFNv3_3D_Wide, count_parameters


# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CSV_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
WORK_DIR = '/home/home_public/aisci/AIscientist/src/1_day2_ppfn'

# Training hyperparameters
BATCH_SIZE = 32
MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
SCHEDULER_PATIENCE = 15
SCHEDULER_FACTOR = 0.5

# Loss function
HUBER_DELTA = 1.0


class HuberLoss(nn.Module):
    """Huber loss for robust regression."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = torch.abs(pred - target)
        quadratic = torch.min(residual, torch.tensor(self.delta).to(pred.device))
        linear = residual - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return loss.mean()


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, scaler: StandardScaler, device: torch.device) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        # Move to device
        atom_features = batch['atom_features'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_features = batch['edge_features'].to(device)
        global_features = batch['global_features'].to(device)
        batch_atom = batch['batch_atom'].to(device)
        targets = batch['tg'].to(device)

        # Normalize targets
        targets_np = targets.cpu().numpy().reshape(-1, 1)
        targets_norm = torch.FloatTensor(scaler.transform(targets_np)).to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model({
            'atom_features': atom_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'global_features': global_features,
            'batch_atom': batch_atom,
        })

        # Compute loss
        loss = criterion(predictions, targets_norm)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * predictions.shape[0]

        # Denormalize for metrics
        preds_np = predictions.detach().cpu().numpy().reshape(-1, 1)
        preds_denorm = torch.FloatTensor(scaler.inverse_transform(preds_np))
        all_preds.extend(preds_denorm.numpy().flatten())
        all_targets.extend(targets.cpu().numpy().flatten())

    avg_loss = total_loss / len(all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

    return avg_loss, rmse


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
             scaler: StandardScaler, device: torch.device) -> Tuple[float, float, float, float, List[float], List[float]]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            atom_features = batch['atom_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_features = batch['edge_features'].to(device)
            global_features = batch['global_features'].to(device)
            batch_atom = batch['batch_atom'].to(device)
            targets = batch['tg'].to(device)

            # Normalize targets
            targets_norm = torch.FloatTensor(scaler.transform(targets.cpu().numpy().reshape(-1, 1))).to(device)

            # Forward pass
            predictions = model({
                'atom_features': atom_features,
                'edge_index': edge_index,
                'edge_features': edge_features,
                'global_features': global_features,
                'batch_atom': batch_atom,
            })

            # Compute loss
            loss = criterion(predictions, targets_norm)
            total_loss += loss.item() * predictions.shape[0]

            # Denormalize
            preds_denorm = torch.FloatTensor(scaler.inverse_transform(predictions.cpu().numpy().reshape(-1, 1)))
            all_preds.extend(preds_denorm.numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    avg_loss = total_loss / len(all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    return avg_loss, rmse, mae, r2, all_targets, all_preds


def test_model(model: nn.Module, dataloader: DataLoader, scaler: StandardScaler,
               device: torch.device) -> Dict[str, float]:
    """Test model and return metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            atom_features = batch['atom_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_features = batch['edge_features'].to(device)
            global_features = batch['global_features'].to(device)
            batch_atom = batch['batch_atom'].to(device)
            targets = batch['tg'].to(device)

            predictions = model({
                'atom_features': atom_features,
                'edge_index': edge_index,
                'edge_features': edge_features,
                'global_features': global_features,
                'batch_atom': batch_atom,
            })

            # Denormalize
            preds_denorm = torch.FloatTensor(scaler.inverse_transform(predictions.cpu().numpy().reshape(-1, 1)))
            all_preds.extend(preds_denorm.numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': all_preds,
        'targets': all_targets,
    }


def train_single_model(config_name: str, model_config: Dict, train_loader: DataLoader,
                       val_loader: DataLoader, test_loader: DataLoader, scaler: StandardScaler,
                       device: torch.device) -> Dict:
    """Train a single model configuration."""
    print(f"\n{'='*80}")
    print(f"Training {config_name}")
    print(f"{'='*80}")

    # Create model
    model = model_config['model_class'](**model_config['model_params']).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, verbose=True
    )

    # Loss function
    criterion = HuberLoss(delta=HUBER_DELTA)

    # Training loop
    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    training_log = []

    for epoch in range(MAX_EPOCHS):
        start_time = time.time()

        # Train
        train_loss, train_rmse = train_epoch(model, train_loader, optimizer, criterion, scaler, device)

        # Validate
        val_loss, val_rmse, val_mae, val_r2, _, _ = validate(model, val_loader, criterion, scaler, device)

        # Scheduler step
        scheduler.step(val_rmse)

        epoch_time = time.time() - start_time

        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_rmse': train_rmse,
            'val_loss': val_loss,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time,
        }
        training_log.append(log_entry)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{MAX_EPOCHS} | "
                  f"Train RMSE: {train_rmse:.2f} | Val RMSE: {val_rmse:.2f} | "
                  f"Val R2: {val_r2:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {epoch_time:.1f}s")

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # Test
    print(f"\nBest validation RMSE: {best_val_rmse:.2f} at epoch {best_epoch}")
    test_results = test_model(model, test_loader, scaler, device)
    print(f"Test RMSE: {test_results['rmse']:.2f}")
    print(f"Test MAE: {test_results['mae']:.2f}")
    print(f"Test R2: {test_results['r2']:.4f}")

    # Save model
    model_path = os.path.join(WORK_DIR, f'best_model_v3_3d_{config_name}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    return {
        'config_name': config_name,
        'model_params': model_config['model_params'],
        'best_val_rmse': best_val_rmse,
        'best_epoch': best_epoch,
        'test_rmse': test_results['rmse'],
        'test_mae': test_results['mae'],
        'test_r2': test_results['r2'],
        'training_log': training_log,
        'model_path': model_path,
    }


def train_ensemble(models_config: List[Tuple[str, Dict]], train_loader: DataLoader,
                   val_loader: DataLoader, test_loader: DataLoader, scaler: StandardScaler,
                   device: torch.device, ensemble_size: int = 5) -> Dict:
    """Train an ensemble of models."""
    print(f"\n{'='*80}")
    print(f"Training Ensemble ({ensemble_size} models)")
    print(f"{'='*80}")

    config_name, model_config = models_config[0]

    ensemble_results = []
    test_predictions_list = []

    for i in range(ensemble_size):
        print(f"\n{'='*80}")
        print(f"Training ensemble model {i+1}/{ensemble_size}")
        print(f"{'='*80}")

        # Create model
        model = model_config['model_class'](**model_config['model_params']).to(device)

        # Optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, verbose=True
        )

        criterion = HuberLoss(delta=HUBER_DELTA)

        # Training loop
        best_val_rmse = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_model_state = None

        for epoch in range(MAX_EPOCHS):
            # Train
            train_loss, train_rmse = train_epoch(model, train_loader, optimizer, criterion, scaler, device)

            # Validate
            val_loss, val_rmse, val_mae, val_r2, _, _ = validate(model, val_loader, criterion, scaler, device)

            scheduler.step(val_rmse)

            # Early stopping
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch + 1
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 50 == 0:
                print(f"Model {i+1} - Epoch {epoch+1:3d}/{MAX_EPOCHS} | "
                      f"Val RMSE: {val_rmse:.2f} | Best: {best_val_rmse:.2f}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break

        # Load best model
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

        # Test
        test_results = test_model(model, test_loader, scaler, device)
        ensemble_results.append({
            'model_idx': i,
            'best_val_rmse': best_val_rmse,
            'best_epoch': best_epoch,
            'test_rmse': test_results['rmse'],
            'test_mae': test_results['mae'],
            'test_r2': test_results['r2'],
        })
        test_predictions_list.append(test_results['predictions'])

        # Save model
        model_path = os.path.join(WORK_DIR, f'best_model_v3_3d_ensemble_{i}.pt')
        torch.save(model.state_dict(), model_path)

        print(f"Model {i+1} - Test RMSE: {test_results['rmse']:.2f}")

    # Ensemble prediction (average)
    ensemble_preds = np.mean(test_predictions_list, axis=0)
    ensemble_targets = test_results['targets']

    ensemble_rmse = np.sqrt(mean_squared_error(ensemble_targets, ensemble_preds))
    ensemble_mae = mean_absolute_error(ensemble_targets, ensemble_preds)
    ensemble_r2 = r2_score(ensemble_targets, ensemble_preds)

    print(f"\n{'='*80}")
    print(f"Ensemble Results ({ensemble_size} models)")
    print(f"{'='*80}")
    print(f"Ensemble Test RMSE: {ensemble_rmse:.2f}")
    print(f"Ensemble Test MAE: {ensemble_mae:.2f}")
    print(f"Ensemble Test R2: {ensemble_r2:.4f}")

    return {
        'config_name': f'{config_name}_ensemble',
        'ensemble_size': ensemble_size,
        'individual_results': ensemble_results,
        'ensemble_rmse': ensemble_rmse,
        'ensemble_mae': ensemble_mae,
        'ensemble_r2': ensemble_r2,
    }


def main():
    """Main training function."""
    print("="*80)
    print("PPFN-v3-3D: Polymer Property Prediction with 3D Conformer Features")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max epochs: {MAX_EPOCHS}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Loss: Huber (delta={HUBER_DELTA})")
    print()

    # Create datasets
    print("Creating datasets...")
    train_dataset = PPFNv3Dataset(CSV_PATH, split='train', use_periodic_edge=True, enable_3d=True)
    val_dataset = PPFNv3Dataset(CSV_PATH, split='val', use_periodic_edge=True, enable_3d=True)
    test_dataset = PPFNv3Dataset(CSV_PATH, split='test', use_periodic_edge=True, enable_3d=True)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn_v3, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn_v3, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn_v3, num_workers=0)

    # Fit scaler on training labels
    train_labels = np.array([sample['tg'].item() for sample in train_dataset])
    scaler = StandardScaler()
    scaler.fit(train_labels.reshape(-1, 1))
    print(f"\nLabel scaler: mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}")

    # Model configurations to train
    models_config = [
        ('base_256', {
            'model_class': PPFNv3_3D,
            'model_params': {'hidden_dim': 256, 'num_gnn_layers': 4, 'dropout': 0.1},
        }),
        ('wide_512', {
            'model_class': PPFNv3_3D_Wide,
            'model_params': {'hidden_dim': 512, 'num_gnn_layers': 6, 'dropout': 0.1},
        }),
    ]

    # Train all configurations
    all_results = {}

    for config_name, model_config in models_config:
        result = train_single_model(
            config_name, model_config, train_loader, val_loader, test_loader, scaler, DEVICE
        )
        all_results[config_name] = result

    # Train ensemble (using best config)
    print("\n" + "="*80)
    print("Training ensemble...")
    ensemble_result = train_ensemble(
        [models_config[0]], train_loader, val_loader, test_loader, scaler, DEVICE, ensemble_size=5
    )
    all_results['ensemble'] = ensemble_result

    # Save results
    results_path = os.path.join(WORK_DIR, 'results_v3_3d.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("Final Results Summary")
    print(f"{'='*80}")
    print(f"\nResults saved to {results_path}\n")

    for config_name, result in all_results.items():
        if 'ensemble' in config_name:
            print(f"{config_name:30s} | Test RMSE: {result['ensemble_rmse']:.2f} | "
                  f"MAE: {result['ensemble_mae']:.2f} | R2: {result['ensemble_r2']:.4f}")
        else:
            print(f"{config_name:30s} | Test RMSE: {result['test_rmse']:.2f} | "
                  f"MAE: {result['test_mae']:.2f} | R2: {result['test_r2']:.4f}")

    # Comparison with baselines
    print(f"\n{'='*80}")
    print("Comparison with Previous Baselines")
    print(f"{'='*80}")
    print(f"PPFNv2 single best (wide_512):  Test RMSE = 39.21")
    print(f"PPFNv2 ensemble (5 models):      Test RMSE = 36.74")
    print(f"Target SOTA:                     Test RMSE < 30.8")
    print()

    # Find best model
    best_config = min(all_results.items(), key=lambda x: x[1].get('ensemble_rmse', x[1]['test_rmse']))
    best_rmse = best_config[1].get('ensemble_rmse', best_config[1]['test_rmse'])

    if 'ensemble' in best_config[0]:
        print(f"PPFNv3-3D {best_config[0]:25s} | Test RMSE = {best_rmse:.2f}")
    else:
        print(f"PPFNv3-3D {best_config[0]:25s} | Test RMSE = {best_rmse:.2f}")

    if best_rmse < 30.8:
        print(f"\n SUCCESS: Achieved target SOTA (RMSE < 30.8)!")
    elif best_rmse < 36.74:
        print(f"\n IMPROVEMENT: Beat PPFNv2 ensemble!")
    else:
        print(f"\n Note: Did not beat previous best. Consider further tuning.")

    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()
