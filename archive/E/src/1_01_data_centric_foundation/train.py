"""
Training loop with early stopping for GIN model.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch_geometric.loader import DataLoader as PyGDataLoader

from gin_model import GINModel

logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    n_samples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(pred.squeeze(), batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_samples += batch.num_graphs
    return total_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns average loss."""
    model.eval()
    total_loss = 0
    n_samples = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(pred.squeeze(), batch.y.squeeze())
        total_loss += loss.item() * batch.num_graphs
        n_samples += batch.num_graphs
    return total_loss / n_samples


def train_model(dataset, train_idx, val_idx, test_idx, node_dim, edge_dim,
                hidden_dim=256, num_layers=3, lr=0.001, batch_size=32,
                max_epochs=500, patience=30, seed=71, device='cuda',
                save_dir=None, exp_name='experiment'):
    """
    Train GIN model with early stopping.

    Args:
        dataset: PyG dataset
        train_idx, val_idx, test_idx: index lists for splits
        node_dim: input node feature dimension
        edge_dim: input edge feature dimension
        hidden_dim: GIN hidden dimension
        num_layers: number of GIN layers
        lr: learning rate
        batch_size: batch size
        max_epochs: max training epochs
        patience: early stopping patience
        seed: random seed
        device: torch device
        save_dir: directory to save best model
        exp_name: experiment name for logging

    Returns:
        dict with train/val/test RMSE, best epoch, learning curves
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Create data loaders
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = GINModel(node_dim=node_dim, edge_dim=edge_dim,
                     hidden_dim=hidden_dim, num_layers=num_layers)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop with early stopping
    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0

    train_rmses = []
    val_rmses = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_rmse = np.sqrt(train_loss)
        val_rmse = np.sqrt(val_loss)

        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, f'{exp_name}_best.pt'))
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            logger.info(f"[{exp_name}] Epoch {epoch}: train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}")

        if patience_counter >= patience:
            logger.info(f"[{exp_name}] Early stopping at epoch {epoch}, best epoch={best_epoch}, best val_rmse={best_val_rmse:.4f}")
            break

    # Load best model and evaluate on test set
    if save_dir:
        model.load_state_dict(torch.load(os.path.join(save_dir, f'{exp_name}_best.pt'), weights_only=True))

    test_loss = evaluate(model, test_loader, criterion, device)
    test_rmse = np.sqrt(test_loss)

    # Final train/val at best epoch (already tracked)
    best_train_rmse = train_rmses[best_epoch - 1]

    result = {
        'train_rmse': best_train_rmse,
        'val_rmse': best_val_rmse,
        'test_rmse': test_rmse,
        'best_epoch': best_epoch,
        'total_epochs': len(train_rmses),
        'train_rmses': train_rmses,
        'val_rmses': val_rmses,
    }

    logger.info(f"[{exp_name}] Final: train={best_train_rmse:.4f}, val={best_val_rmse:.4f}, test={test_rmse:.4f}, best_epoch={best_epoch}")

    return result
