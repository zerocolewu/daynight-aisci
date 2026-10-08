"""
Training script for HEPT polymer Tg prediction.
Includes training loop, validation, early stopping, and checkpointing.
"""

import os
import sys
import json
import time
import copy
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch

# Suppress warnings
warnings.filterwarnings('ignore')

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Local imports
from data_loader import get_dataloaders, PolymerTgDataset, stratified_split, RandomRotate
from model import HEPT, HEPT_NoEquivariant, HEPT_NoPeriodic, HEPT_NoLongRange, SimpleGNN


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Data
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 42

    # Model
    hidden_dim: int = 256
    num_equivariant_layers: int = 3
    num_periodic_layers: int = 2
    num_heads: int = 8
    num_rbf: int = 64
    cutoff: float = 5.0
    dropout: float = 0.1
    num_periodic_images: int = 2  # Added parameter

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 200
    patience: int = 20
    gradient_clip: float = 1.0

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Ablation
    use_equivariant: bool = True
    use_periodic: bool = True
    use_long_range: bool = True


# ============================================================================
# Training Utilities
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_input_dim(data_loader: DataLoader) -> int:
    """Get input dimension from data."""
    for batch in data_loader:
        return batch.x.size(-1)
    raise ValueError("Data loader is empty")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    config: Config,
    augmenter: Optional[RandomRotate] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch.x, batch.pos, batch.edge_index, batch.batch)

        # Compute loss
        loss = criterion(predictions.squeeze(), batch.y.squeeze())

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs

    return {"train_loss": total_loss / total_samples}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict[str, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_targets = []

    for batch in data_loader:
        batch = batch.to(device)

        predictions = model(batch.x, batch.pos, batch.edge_index, batch.batch)
        loss = criterion(predictions.squeeze(), batch.y.squeeze())

        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs

        all_predictions.append(predictions.squeeze().cpu())
        all_targets.append(batch.y.squeeze().cpu())

    # Compute metrics
    predictions = torch.cat(all_predictions).numpy()
    targets = torch.cat(all_targets).numpy()

    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    # R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "val_loss": total_loss / total_samples,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": predictions,
        "targets": targets
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_rmse: float,
    config: Config,
    filepath: str
):
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_rmse": best_rmse,
        "config": {k: v for k, v in vars(config).items() if not k.startswith('_')}
    }, filepath)


def load_checkpoint(model: nn.Module, filepath: str, device: str) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


# ============================================================================
# Main Training Function
# ============================================================================

def train(config: Config, model_name: str = "hept") -> Tuple[nn.Module, Dict]:
    """Main training function."""

    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} model")
    print(f"{'='*60}")
    print(f"Device: {config.device}")
    print(f"Hidden dim: {config.hidden_dim}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max epochs: {config.max_epochs}")
    print(f"Patience: {config.patience}")
    print(f"{'='*60}\n")

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Create directories
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed
    )

    input_dim = get_input_dim(train_loader)
    print(f"Input dimension: {input_dim}")

    # Create model
    if model_name == "hept":
        model = HEPT(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            num_equivariant_layers=config.num_equivariant_layers,
            num_periodic_layers=config.num_periodic_layers,
            num_heads=config.num_heads,
            num_rbf=config.num_rbf,
            cutoff=config.cutoff,
            dropout=config.dropout,
            use_equivariant=config.use_equivariant,
            use_periodic=config.use_periodic,
            use_long_range=config.use_long_range
        )
    elif model_name == "hept_no_equiv":
        model = HEPT_NoEquivariant(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            num_equivariant_layers=config.num_equivariant_layers,
            num_periodic_layers=config.num_periodic_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
    elif model_name == "hept_no_periodic":
        model = HEPT_NoPeriodic(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            num_equivariant_layers=config.num_equivariant_layers,
            num_periodic_layers=config.num_periodic_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
    elif model_name == "hept_no_longrange":
        model = HEPT_NoLongRange(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            num_equivariant_layers=config.num_equivariant_layers,
            num_periodic_layers=config.num_periodic_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
    elif model_name == "simple_gnn":
        model = SimpleGNN(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            dropout=config.dropout
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = model.to(config.device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )

    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    best_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "val_mae": [],
        "val_r2": [],
        "lr": []
    }

    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(config.max_epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            config.device, config, None
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, config.device)

        # Update scheduler
        scheduler.step(val_metrics["rmse"])

        # Record history
        history["train_loss"].append(train_metrics["train_loss"])
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_r2"].append(val_metrics["r2"])
        history["lr"].append(optimizer.param_groups[0]['lr'])

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch+1:3d}/{config.max_epochs} | "
              f"Train Loss: {train_metrics['train_loss']:.4f} | "
              f"Val RMSE: {val_metrics['rmse']:.2f} | "
              f"Val MAE: {val_metrics['mae']:.2f} | "
              f"Val R²: {val_metrics['r2']:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())

            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"{model_name}_best.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_rmse, config, str(checkpoint_path)
            )
            print(f"  -> New best model saved! RMSE: {best_rmse:.2f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Best validation RMSE: {best_rmse:.2f} at epoch {best_epoch}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, config.device)

    print(f"\nTest Results:")
    print(f"  RMSE: {test_metrics['rmse']:.2f}")
    print(f"  MAE: {test_metrics['mae']:.2f}")
    print(f"  R²: {test_metrics['r2']:.4f}")

    # Save final model
    final_checkpoint_path = checkpoint_dir / f"{model_name}_final.pt"
    save_checkpoint(
        model, optimizer, scheduler, epoch, best_rmse, config, str(final_checkpoint_path)
    )

    # Return model and results
    results = {
        "model_name": model_name,
        "best_val_rmse": best_rmse,
        "best_epoch": best_epoch,
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_predictions": test_metrics["predictions"],
        "test_targets": test_metrics["targets"],
        "history": history,
        "total_time": total_time,
        "num_parameters": count_parameters(model)
    }

    return model, results


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    config = Config()

    # Train main model
    model, results = train(config, model_name="hept")

    # Save results
    results_save = {k: v for k, v in results.items() if k not in ["test_predictions", "test_targets", "history"]}
    results_save["history_length"] = len(results["history"]["train_loss"])

    with open("training_results.json", "w") as f:
        json.dump(results_save, f, indent=2)

    # Save predictions
    predictions_df = pd.DataFrame({
        "target": results["test_targets"],
        "prediction": results["test_predictions"]
    })
    predictions_df.to_csv("predictions.csv", index=False)

    # Save history
    history_df = pd.DataFrame(results["history"])
    history_df.to_csv("training_history.csv", index=False)

    print("\nResults saved to training_results.json, predictions.csv, and training_history.csv")
