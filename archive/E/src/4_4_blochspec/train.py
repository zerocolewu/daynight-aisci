"""
Training script for BlochSpec experiment.

Supports:
- (b) FULL MODEL: GIN+Trimer + Bloch spectral summaries
- (e) Pure Spectral MLP
- (f) GIN+Trimer + Random spectral features
- (g) GIN+Trimer + Unweighted Laplacian eigenvalues
- Baseline: GIN+Trimer (no spectral features)

Multi-seed support: seeds={42, 71, 123}
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import BlochSpecGIN, PureSpectralMLP, BaselineGIN, count_parameters
from dataset import (BlochSpecDataset, SpectralOnlyDataset, get_data_splits, prepare_dataset)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Constants
CSV_PATH = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
SPECTRAL_DIR = "/home/home_public/aisci/AIscientist/src/4_4_blochspec/spectral_data"
PROCESSED_DIR = "/home/home_public/aisci/AIscientist/src/4_4_blochspec/data"
WORK_DIR = "/home/home_public/aisci/AIscientist/src/4_4_blochspec"
MODELS_DIR = os.path.join(WORK_DIR, "models")
RESULTS_DIR = os.path.join(WORK_DIR, "results")

HIDDEN_DIM = 256
NUM_LAYERS = 3
LR = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 500
PATIENCE = 30
N_SPECTRAL = 12

SEEDS = [42, 71, 123]
PRIMARY_SEED = 71


def set_seed(seed):
    """Set all random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_batch_spectral_features(batch, n_features=12):
    """Extract spectral features from batch and reshape to [batch_size, n_features]."""
    sf = batch.spectral_features
    if sf.dim() == 1:
        sf = sf.view(batch.num_graphs, n_features)
    return sf


def train_blochspec_epoch(model, loader, optimizer, criterion, device):
    """Train BlochSpecGIN for one epoch."""
    model.train()
    total_loss = 0
    n_samples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        spectral_feats = get_batch_spectral_features(batch)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                     spectral_feats)
        loss = criterion(pred.squeeze(), batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_samples += batch.num_graphs
    return total_loss / n_samples


@torch.no_grad()
def evaluate_blochspec(model, loader, criterion, device):
    """Evaluate BlochSpecGIN."""
    model.eval()
    total_loss = 0
    n_samples = 0
    for batch in loader:
        batch = batch.to(device)
        spectral_feats = get_batch_spectral_features(batch)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                     spectral_feats)
        loss = criterion(pred.squeeze(), batch.y.squeeze())
        total_loss += loss.item() * batch.num_graphs
        n_samples += batch.num_graphs
    return total_loss / n_samples


def train_baseline_epoch(model, loader, optimizer, criterion, device):
    """Train BaselineGIN for one epoch."""
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
def evaluate_baseline(model, loader, criterion, device):
    """Evaluate BaselineGIN."""
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


def train_spectral_mlp_epoch(model, loader, optimizer, criterion, device):
    """Train PureSpectralMLP for one epoch."""
    model.train()
    total_loss = 0
    n_samples = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        pred = model(features)
        loss = criterion(pred.squeeze(), targets.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(features)
        n_samples += len(features)
    return total_loss / n_samples


@torch.no_grad()
def evaluate_spectral_mlp(model, loader, criterion, device):
    """Evaluate PureSpectralMLP."""
    model.eval()
    total_loss = 0
    n_samples = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        pred = model(features)
        loss = criterion(pred.squeeze(), targets.squeeze())
        total_loss += loss.item() * len(features)
        n_samples += len(features)
    return total_loss / n_samples


def train_blochspec_experiment(dataset, seed, device, exp_name="blochspec"):
    """Train the full BlochSpec model (GIN + spectral features)."""
    set_seed(seed)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_idx, val_idx, test_idx = get_data_splits(len(dataset), seed=seed)

    train_loader = PyGDataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = PyGDataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = PyGDataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    node_dim = dataset.num_node_features
    edge_dim = dataset.num_edge_features

    model = BlochSpecGIN(node_dim=node_dim, edge_dim=edge_dim,
                         n_spectral_features=N_SPECTRAL,
                         hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    model = model.to(device)
    n_params = count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_rmses = []
    val_rmses = []

    save_path = os.path.join(MODELS_DIR, f'{exp_name}_seed{seed}_best.pt')

    logger.info(f"[{exp_name}/seed{seed}] Starting: params={n_params:,}, "
                f"node_dim={node_dim}, edge_dim={edge_dim}")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_blochspec_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_blochspec(model, val_loader, criterion, device)

        train_rmse = np.sqrt(train_loss)
        val_rmse = np.sqrt(val_loss)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)

        scheduler.step()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if epoch % 50 == 0 or epoch == 1:
            logger.info(f"[{exp_name}/seed{seed}] Epoch {epoch}: "
                        f"train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}")

        if patience_counter >= PATIENCE:
            logger.info(f"[{exp_name}/seed{seed}] Early stop at epoch {epoch}, "
                        f"best epoch={best_epoch}")
            break

    # Evaluate on test set
    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_loss = evaluate_blochspec(model, test_loader, criterion, device)
    test_rmse = np.sqrt(test_loss)

    # Compute train RMSE at best epoch
    best_train_rmse = train_rmses[best_epoch - 1]

    logger.info(f"[{exp_name}/seed{seed}] FINAL: train={best_train_rmse:.4f}, "
                f"val={best_val_rmse:.4f}, test={test_rmse:.4f}")

    return {
        'exp_name': exp_name,
        'seed': seed,
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


def train_baseline_experiment(dataset, seed, device, exp_name="baseline_gin"):
    """Train baseline GIN on trimer (no spectral features)."""
    set_seed(seed)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_idx, val_idx, test_idx = get_data_splits(len(dataset), seed=seed)

    train_loader = PyGDataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = PyGDataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = PyGDataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    node_dim = dataset.num_node_features
    edge_dim = dataset.num_edge_features

    model = BaselineGIN(node_dim=node_dim, edge_dim=edge_dim,
                        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    model = model.to(device)
    n_params = count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_rmses = []
    val_rmses = []

    save_path = os.path.join(MODELS_DIR, f'{exp_name}_seed{seed}_best.pt')

    logger.info(f"[{exp_name}/seed{seed}] Starting: params={n_params:,}")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_baseline_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_baseline(model, val_loader, criterion, device)

        train_rmse = np.sqrt(train_loss)
        val_rmse = np.sqrt(val_loss)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)

        scheduler.step()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if epoch % 50 == 0 or epoch == 1:
            logger.info(f"[{exp_name}/seed{seed}] Epoch {epoch}: "
                        f"train={train_rmse:.4f}, val={val_rmse:.4f}")

        if patience_counter >= PATIENCE:
            logger.info(f"[{exp_name}/seed{seed}] Early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_loss = evaluate_baseline(model, test_loader, criterion, device)
    test_rmse = np.sqrt(test_loss)
    best_train_rmse = train_rmses[best_epoch - 1]

    logger.info(f"[{exp_name}/seed{seed}] FINAL: train={best_train_rmse:.4f}, "
                f"val={best_val_rmse:.4f}, test={test_rmse:.4f}")

    return {
        'exp_name': exp_name,
        'seed': seed,
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


def train_spectral_mlp_experiment(spectral_features, tgs, seed, device, exp_name="spectral_mlp"):
    """Train Pure Spectral MLP (no GNN)."""
    set_seed(seed)
    os.makedirs(MODELS_DIR, exist_ok=True)

    n_samples = len(tgs)
    train_idx, val_idx, test_idx = get_data_splits(n_samples, seed=seed)

    train_dataset = SpectralOnlyDataset(spectral_features[train_idx], tgs[train_idx])
    val_dataset = SpectralOnlyDataset(spectral_features[val_idx], tgs[val_idx])
    test_dataset = SpectralOnlyDataset(spectral_features[test_idx], tgs[test_idx])

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = PureSpectralMLP(n_features=N_SPECTRAL)
    model = model.to(device)
    n_params = count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_rmses = []
    val_rmses = []

    save_path = os.path.join(MODELS_DIR, f'{exp_name}_seed{seed}_best.pt')

    logger.info(f"[{exp_name}/seed{seed}] Starting: params={n_params:,}")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_spectral_mlp_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_spectral_mlp(model, val_loader, criterion, device)

        train_rmse = np.sqrt(train_loss)
        val_rmse = np.sqrt(val_loss)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)

        scheduler.step()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if epoch % 50 == 0 or epoch == 1:
            logger.info(f"[{exp_name}/seed{seed}] Epoch {epoch}: "
                        f"train={train_rmse:.4f}, val={val_rmse:.4f}")

        if patience_counter >= PATIENCE:
            logger.info(f"[{exp_name}/seed{seed}] Early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_loss = evaluate_spectral_mlp(model, test_loader, criterion, device)
    test_rmse = np.sqrt(test_loss)
    best_train_rmse = train_rmses[best_epoch - 1]

    logger.info(f"[{exp_name}/seed{seed}] FINAL: train={best_train_rmse:.4f}, "
                f"val={best_val_rmse:.4f}, test={test_rmse:.4f}")

    return {
        'exp_name': exp_name,
        'seed': seed,
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


def train_with_alternative_features(dataset, alt_features, seed, device, exp_name):
    """
    Train BlochSpecGIN with alternative spectral features (random or unweighted).
    This works by temporarily replacing the spectral_features in the dataset.
    """
    set_seed(seed)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_idx, val_idx, test_idx = get_data_splits(len(dataset), seed=seed)

    # We need to create a modified dataset that uses the alternative features
    # The simplest approach: modify the data object's spectral_features before creating loaders
    # We'll use a wrapper approach

    # Store original features
    orig_features = dataset.data.spectral_features.clone()

    # Set alternative features - flatten to match InMemoryDataset format
    alt_tensor = torch.tensor(alt_features.flatten(), dtype=torch.float)
    dataset.data.spectral_features = alt_tensor

    train_loader = PyGDataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = PyGDataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = PyGDataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    node_dim = dataset.num_node_features
    edge_dim = dataset.num_edge_features

    model = BlochSpecGIN(node_dim=node_dim, edge_dim=edge_dim,
                         n_spectral_features=N_SPECTRAL,
                         hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    model = model.to(device)
    n_params = count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_rmses = []
    val_rmses = []

    save_path = os.path.join(MODELS_DIR, f'{exp_name}_seed{seed}_best.pt')

    logger.info(f"[{exp_name}/seed{seed}] Starting: params={n_params:,}")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_blochspec_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_blochspec(model, val_loader, criterion, device)

        train_rmse = np.sqrt(train_loss)
        val_rmse = np.sqrt(val_loss)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)

        scheduler.step()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if epoch % 50 == 0 or epoch == 1:
            logger.info(f"[{exp_name}/seed{seed}] Epoch {epoch}: "
                        f"train={train_rmse:.4f}, val={val_rmse:.4f}")

        if patience_counter >= PATIENCE:
            logger.info(f"[{exp_name}/seed{seed}] Early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_loss = evaluate_blochspec(model, test_loader, criterion, device)
    test_rmse = np.sqrt(test_loss)
    best_train_rmse = train_rmses[best_epoch - 1]

    # Restore original features
    dataset.data.spectral_features = orig_features

    logger.info(f"[{exp_name}/seed{seed}] FINAL: train={best_train_rmse:.4f}, "
                f"val={best_val_rmse:.4f}, test={test_rmse:.4f}")

    return {
        'exp_name': exp_name,
        'seed': seed,
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


def aggregate_results(results_list, exp_name):
    """Aggregate multi-seed results: compute mean ± std."""
    test_rmses = [r['test_rmse'] for r in results_list]
    train_rmses = [r['train_rmse'] for r in results_list]
    val_rmses = [r['val_rmse'] for r in results_list]
    gaps = [r['overfitting_gap'] for r in results_list]

    return {
        'exp_name': exp_name,
        'test_rmse_mean': np.mean(test_rmses),
        'test_rmse_std': np.std(test_rmses),
        'train_rmse_mean': np.mean(train_rmses),
        'train_rmse_std': np.std(train_rmses),
        'val_rmse_mean': np.mean(val_rmses),
        'val_rmse_std': np.std(val_rmses),
        'gap_mean': np.mean(gaps),
        'gap_std': np.std(gaps),
        'individual_results': results_list,
    }


def save_results_summary(all_agg_results, output_path):
    """Save results summary to file."""
    with open(output_path, 'w') as f:
        f.write("BlochSpec Experiment Results Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Experiment':<35} {'Test RMSE':>20} {'Train RMSE':>20} {'Gap':>15}\n")
        f.write("-" * 80 + "\n")

        for r in all_agg_results:
            test_str = f"{r['test_rmse_mean']:.2f} +/- {r['test_rmse_std']:.2f}"
            train_str = f"{r['train_rmse_mean']:.2f} +/- {r['train_rmse_std']:.2f}"
            gap_str = f"{r['gap_mean']:.2f} +/- {r['gap_std']:.2f}"
            f.write(f"{r['exp_name']:<35} {test_str:>20} {train_str:>20} {gap_str:>15}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Individual Seed Results:\n")
        f.write("-" * 80 + "\n")
        for r in all_agg_results:
            f.write(f"\n{r['exp_name']}:\n")
            for ind in r['individual_results']:
                f.write(f"  seed={ind['seed']}: test={ind['test_rmse']:.4f}, "
                        f"train={ind['train_rmse']:.4f}, val={ind['val_rmse']:.4f}, "
                        f"best_epoch={ind['best_epoch']}, gap={ind['overfitting_gap']:.4f}\n")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load spectral data
    spectral_features = np.load(os.path.join(SPECTRAL_DIR, 'spectral_features.npy'))
    spectral_indices = np.load(os.path.join(SPECTRAL_DIR, 'feature_indices.npy'))
    feature_tgs = np.load(os.path.join(SPECTRAL_DIR, 'feature_tgs.npy'))

    # Load alternative features
    spectral_random = np.load(os.path.join(SPECTRAL_DIR, 'spectral_features_random.npy'))
    spectral_unweighted = np.load(os.path.join(SPECTRAL_DIR, 'spectral_features_unweighted.npy'))

    # Prepare dataset (with weighted spectral features)
    dataset = prepare_dataset(CSV_PATH, SPECTRAL_DIR, PROCESSED_DIR)
    logger.info(f"Dataset: {len(dataset)} graphs, node_dim={dataset.num_node_features}, "
                f"edge_dim={dataset.num_edge_features}")

    # Verify dataset and spectral features are aligned
    assert len(dataset) == len(spectral_features), \
        f"Dataset size {len(dataset)} != spectral features {len(spectral_features)}"

    all_agg_results = []

    # ==========================================
    # Experiment (a): Baseline GIN+Trimer
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("Experiment (a): Baseline GIN+Trimer (no spectral)")
    logger.info("=" * 60)

    baseline_results = []
    for seed in SEEDS:
        r = train_baseline_experiment(dataset, seed, device, exp_name="baseline_gin")
        baseline_results.append(r)
    all_agg_results.append(aggregate_results(baseline_results, "Baseline GIN+Trimer"))

    # ==========================================
    # Experiment (b): FULL MODEL: GIN+Trimer + Bloch spectral
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("Experiment (b): FULL BlochSpec (GIN + Bloch spectral)")
    logger.info("=" * 60)

    blochspec_results = []
    for seed in SEEDS:
        r = train_blochspec_experiment(dataset, seed, device, exp_name="blochspec_full")
        blochspec_results.append(r)
    all_agg_results.append(aggregate_results(blochspec_results, "BlochSpec (GIN+Bloch)"))

    # ==========================================
    # Experiment (e): Pure Spectral MLP
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("Experiment (e): Pure Spectral MLP (no GNN)")
    logger.info("=" * 60)

    mlp_results = []
    for seed in SEEDS:
        r = train_spectral_mlp_experiment(spectral_features, feature_tgs, seed, device,
                                           exp_name="spectral_mlp")
        mlp_results.append(r)
    all_agg_results.append(aggregate_results(mlp_results, "Pure Spectral MLP"))

    # ==========================================
    # Experiment (f): GIN+Trimer + Random features
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("Experiment (f): GIN+Trimer + Random features")
    logger.info("=" * 60)

    random_results = []
    for seed in SEEDS:
        # Use seed-specific random features
        rng = np.random.RandomState(seed)
        n_samples, n_feats = spectral_features.shape
        rand_feats = np.zeros_like(spectral_features)
        for j in range(n_feats):
            col_mean = np.mean(spectral_features[:, j])
            col_std = np.std(spectral_features[:, j])
            rand_feats[:, j] = rng.normal(col_mean, col_std, n_samples)
        rand_feats = rand_feats.astype(np.float32)

        r = train_with_alternative_features(dataset, rand_feats, seed, device,
                                            exp_name="blochspec_random")
        random_results.append(r)
    all_agg_results.append(aggregate_results(random_results, "GIN + Random Features"))

    # ==========================================
    # Experiment (g): GIN+Trimer + Unweighted Laplacian
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("Experiment (g): GIN+Trimer + Unweighted Laplacian features")
    logger.info("=" * 60)

    unweighted_results = []
    for seed in SEEDS:
        r = train_with_alternative_features(dataset, spectral_unweighted, seed, device,
                                            exp_name="blochspec_unweighted")
        unweighted_results.append(r)
    all_agg_results.append(aggregate_results(unweighted_results, "GIN + Unweighted Lap"))

    # Save results
    results_path = os.path.join(RESULTS_DIR, "results_summary.txt")
    save_results_summary(all_agg_results, results_path)
    logger.info(f"\nResults saved to {results_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("BLOCHSPEC EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<35} {'Test RMSE':>20} {'Train RMSE':>20} {'Gap':>15}")
    print("-" * 80)
    for r in all_agg_results:
        test_str = f"{r['test_rmse_mean']:.2f} +/- {r['test_rmse_std']:.2f}"
        train_str = f"{r['train_rmse_mean']:.2f} +/- {r['train_rmse_std']:.2f}"
        gap_str = f"{r['gap_mean']:.2f} +/- {r['gap_std']:.2f}"
        print(f"{r['exp_name']:<35} {test_str:>20} {train_str:>20} {gap_str:>15}")

    return all_agg_results


if __name__ == "__main__":
    results = main()
