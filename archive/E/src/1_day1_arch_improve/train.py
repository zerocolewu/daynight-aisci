"""
Training loop for all 4 experiments in Day 1 Experiment 2.
Experiments:
  1. GIN-GAT Hybrid on Trimer
  2. GIN + Virtual Node on Trimer
  3. GIN-GAT Hybrid on Cyclized Trimer
  4. GIN + Virtual Node on Cyclized Trimer
"""

import os
import sys
import logging
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader

from data_preprocessing import TgDataset, get_data_splits, MODES, prepare_all_datasets
from models import GINModel, GINGATModel, count_parameters

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Constants
CSV_PATH = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
PROCESSED_DIR = "./data"
MODELS_DIR = "./models"
SEED = 71
HIDDEN_DIM = 256
NUM_LAYERS = 3
LR = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 500
PATIENCE = 30


def train_one_epoch(model, loader, optimizer, criterion, device, has_virtual_node=False):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    n_samples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        if has_virtual_node:
            # Compute num real nodes per graph (exclude virtual node)
            num_real_nodes = []
            _, counts = torch.unique(batch.batch, return_counts=True)
            for count in counts:
                num_real_nodes.append(count.item() - 1)  # -1 for virtual node
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                         num_nodes_per_graph=num_real_nodes)
        else:
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(pred.squeeze(), batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_samples += batch.num_graphs
    return total_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, has_virtual_node=False):
    """Evaluate model. Returns average loss."""
    model.eval()
    total_loss = 0
    n_samples = 0
    for batch in loader:
        batch = batch.to(device)
        if has_virtual_node:
            num_real_nodes = []
            _, counts = torch.unique(batch.batch, return_counts=True)
            for count in counts:
                num_real_nodes.append(count.item() - 1)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                         num_nodes_per_graph=num_real_nodes)
        else:
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(pred.squeeze(), batch.y.squeeze())
        total_loss += loss.item() * batch.num_graphs
        n_samples += batch.num_graphs
    return total_loss / n_samples


def train_experiment(exp_name, dataset, model_type, has_virtual_node=False,
                     hidden_dim=256, num_layers=3, lr=0.001, batch_size=32,
                     max_epochs=500, patience=30, seed=71, device='cuda'):
    """
    Train a single experiment.

    Args:
        exp_name: experiment name for logging/saving
        dataset: PyG dataset
        model_type: 'gingat' or 'gin'
        has_virtual_node: whether data has virtual node
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Data splits
    train_idx, val_idx, test_idx = get_data_splits(len(dataset), seed=seed)

    train_loader = PyGDataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = PyGDataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = PyGDataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

    node_dim = dataset.num_node_features
    edge_dim = dataset.num_edge_features

    # Initialize model
    if model_type == 'gingat':
        model = GINGATModel(node_dim=node_dim, edge_dim=edge_dim,
                            hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_type == 'gin':
        model = GINModel(node_dim=node_dim, edge_dim=edge_dim,
                         hidden_dim=hidden_dim, num_layers=num_layers,
                         has_virtual_node=has_virtual_node)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    n_params = count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_rmses = []
    val_rmses = []

    save_dir = MODELS_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{exp_name}_best.pt')

    logger.info(f"[{exp_name}] Starting training: model={model_type}, vn={has_virtual_node}, "
                f"params={n_params:,}, node_dim={node_dim}, edge_dim={edge_dim}")

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                     has_virtual_node=has_virtual_node)
        val_loss = evaluate(model, val_loader, criterion, device,
                            has_virtual_node=has_virtual_node)

        train_rmse = np.sqrt(train_loss)
        val_rmse = np.sqrt(val_loss)

        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            logger.info(f"[{exp_name}] Epoch {epoch}: train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}")

        if patience_counter >= patience:
            logger.info(f"[{exp_name}] Early stopping at epoch {epoch}, "
                        f"best epoch={best_epoch}, best val_rmse={best_val_rmse:.4f}")
            break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_loss = evaluate(model, test_loader, criterion, device, has_virtual_node=has_virtual_node)
    test_rmse = np.sqrt(test_loss)

    best_train_rmse = train_rmses[best_epoch - 1]

    result = {
        'exp_name': exp_name,
        'model_type': model_type,
        'has_virtual_node': has_virtual_node,
        'train_rmse': best_train_rmse,
        'val_rmse': best_val_rmse,
        'test_rmse': test_rmse,
        'best_epoch': best_epoch,
        'total_epochs': len(train_rmses),
        'n_params': n_params,
        'overfitting_gap': test_rmse - best_train_rmse,
        'train_rmses': train_rmses,
        'val_rmses': val_rmses,
    }

    logger.info(f"[{exp_name}] FINAL: train={best_train_rmse:.4f}, val={best_val_rmse:.4f}, "
                f"test={test_rmse:.4f}, best_epoch={best_epoch}, params={n_params:,}")

    return result


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Prepare all datasets
    datasets = prepare_all_datasets(CSV_PATH, PROCESSED_DIR)

    # Define experiments
    experiments = [
        # (exp_name, data_mode, model_type, has_virtual_node)
        ("gin_gat_trimer", "trimer", "gingat", False),
        ("gin_gat_cyclized", "cyclized", "gingat", False),
        ("gin_vn_trimer", "trimer_vn", "gin", True),
        ("gin_vn_cyclized", "cyclized_vn", "gin", True),
    ]

    all_results = []

    for exp_name, data_mode, model_type, has_vn in experiments:
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {exp_name} | Data: {data_mode} | Model: {model_type} | VN: {has_vn}")
        logger.info(f"{'='*60}")

        result = train_experiment(
            exp_name=exp_name,
            dataset=datasets[data_mode],
            model_type=model_type,
            has_virtual_node=has_vn,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            lr=LR,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            seed=SEED,
            device=device,
        )
        all_results.append(result)

    # Save results to CSV
    csv_path = "./results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', 'Model', 'Data', 'Train RMSE', 'Val RMSE', 'Test RMSE',
                         'Best Epoch', 'Total Epochs', 'Params', 'Overfitting Gap'])
        for r in all_results:
            data_label = r['exp_name'].replace('gin_gat_', 'GIN-GAT ').replace('gin_vn_', 'GIN+VN ')
            writer.writerow([
                r['exp_name'],
                r['model_type'],
                data_label,
                f"{r['train_rmse']:.4f}",
                f"{r['val_rmse']:.4f}",
                f"{r['test_rmse']:.4f}",
                r['best_epoch'],
                r['total_epochs'],
                f"{r['n_params']:,}",
                f"{r['overfitting_gap']:.4f}",
            ])
    logger.info(f"\nResults saved to {csv_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Model':<20} {'Data':<20} {'Train RMSE':<12} {'Val RMSE':<12} {'Test RMSE':<12} "
          f"{'Best Epoch':<12} {'Params':<12} {'Gap':<10}")
    print("-" * 100)

    # Reference baselines from Experiment 1
    baselines = [
        ("GIN (ref)", "Monomer", 36.38, 41.37, 52.71, 30, "445,700", 16.33),
        ("GIN (ref)", "Trimer", 41.29, 39.65, 50.19, 15, "~445K", 8.90),
        ("GIN (ref)", "Cyclized", 26.53, 39.62, 53.28, 64, "~445K", 26.75),
    ]
    for b in baselines:
        print(f"{b[0]:<20} {b[1]:<20} {b[2]:<12.2f} {b[3]:<12.2f} {b[4]:<12.2f} "
              f"{b[5]:<12} {b[6]:<12} {b[7]:<10.2f}")

    for r in all_results:
        data_label = r['exp_name'].split('_', 2)[-1].replace('_', ' + ')
        model_label = "GIN-GAT" if r['model_type'] == 'gingat' else "GIN+VN"
        print(f"{model_label:<20} {data_label:<20} {r['train_rmse']:<12.4f} {r['val_rmse']:<12.4f} "
              f"{r['test_rmse']:<12.4f} {r['best_epoch']:<12} {r['n_params']:<12,} {r['overfitting_gap']:<10.4f}")

    return all_results


if __name__ == "__main__":
    results = main()
