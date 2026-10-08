"""
Training script for BlochSpec Iteration 2.

CRITICAL FIX: global_mean_pool in ALL models.
Experiments:
  (a) Baseline GIN+Trimer (global_mean_pool) - should match Day 1 ~50.19
  (b) BlochSpec: GIN+Trimer + 12 Bloch spectral features
  (c) GIN+Trimer + Wider MLP (parameter count control)
  (d) GIN (reduced dim=244) + 12 Bloch features (total dim=256, same as baseline)
  (e) Pure Spectral MLP (12 features only)
  (f) GIN+Trimer + Random features
  (g) GIN+Trimer + Unweighted Laplacian features

Multi-seed: seeds={42, 71, 123}
Statistical significance: paired t-test vs baseline
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import ttest_rel

from model import (BaselineGIN, BlochSpecGIN, WiderMLPGIN, ReducedGNNBlochSpec,
                   PureSpectralMLP, count_parameters)
from dataset import (BlochSpecDataset, SpectralOnlyDataset, get_data_splits, prepare_dataset)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

CSV_PATH = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
SPECTRAL_DIR = os.path.join(os.path.dirname(__file__), "spectral_data")
WORK_DIR = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(WORK_DIR, "data")
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


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_batch_spectral_features(batch, n_features=12):
    sf = batch.spectral_features
    if sf.dim() == 1:
        sf = sf.view(batch.num_graphs, n_features)
    return sf


# ============================================================
# Training loop - generic for all GNN-based models
# ============================================================

def train_gnn_epoch(model, loader, optimizer, criterion, device, use_spectral=False):
    model.train()
    total_loss = 0
    n_samples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        if use_spectral:
            spectral_feats = get_batch_spectral_features(batch)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, spectral_feats)
        else:
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(pred.squeeze(), batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_samples += batch.num_graphs
    return total_loss / n_samples


@torch.no_grad()
def evaluate_gnn(model, loader, criterion, device, use_spectral=False):
    model.eval()
    total_loss = 0
    n_samples = 0
    for batch in loader:
        batch = batch.to(device)
        if use_spectral:
            spectral_feats = get_batch_spectral_features(batch)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, spectral_feats)
        else:
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(pred.squeeze(), batch.y.squeeze())
        total_loss += loss.item() * batch.num_graphs
        n_samples += batch.num_graphs
    return total_loss / n_samples


def train_spectral_mlp_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_samples = 0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
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
    model.eval()
    total_loss = 0
    n_samples = 0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        pred = model(features)
        loss = criterion(pred.squeeze(), targets.squeeze())
        total_loss += loss.item() * len(features)
        n_samples += len(features)
    return total_loss / n_samples


# ============================================================
# Experiment runners
# ============================================================

def run_gnn_experiment(model, dataset, seed, device, exp_name, use_spectral=False):
    """Generic training loop for GNN-based models."""
    set_seed(seed)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_idx, val_idx, test_idx = get_data_splits(len(dataset), seed=seed)
    train_loader = PyGDataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = PyGDataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = PyGDataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    model = model.to(device)
    n_params = count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_rmses, val_rmses = [], []

    save_path = os.path.join(MODELS_DIR, f'{exp_name}_seed{seed}_best.pt')

    logger.info(f"[{exp_name}/seed{seed}] params={n_params:,}")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_gnn_epoch(model, train_loader, optimizer, criterion, device, use_spectral)
        val_loss = evaluate_gnn(model, val_loader, criterion, device, use_spectral)

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
            logger.info(f"[{exp_name}/seed{seed}] Early stop at epoch {epoch}, best={best_epoch}")
            break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_loss = evaluate_gnn(model, test_loader, criterion, device, use_spectral)
    test_rmse = np.sqrt(test_loss)
    best_train_rmse = train_rmses[best_epoch - 1]

    logger.info(f"[{exp_name}/seed{seed}] FINAL: train={best_train_rmse:.4f}, "
                f"val={best_val_rmse:.4f}, test={test_rmse:.4f}")

    return {
        'exp_name': exp_name, 'seed': seed,
        'train_rmse': best_train_rmse, 'val_rmse': best_val_rmse, 'test_rmse': test_rmse,
        'best_epoch': best_epoch, 'total_epochs': len(train_rmses),
        'n_params': n_params, 'overfitting_gap': test_rmse - best_train_rmse,
    }


def run_gnn_with_alt_features(model_fn, dataset, alt_features, seed, device, exp_name):
    """Train a GNN model with alternative spectral features (temporarily swapped in)."""
    set_seed(seed)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Swap spectral features in dataset
    orig_features = dataset.data.spectral_features.clone()
    alt_tensor = torch.tensor(alt_features.flatten(), dtype=torch.float)
    dataset.data.spectral_features = alt_tensor

    train_idx, val_idx, test_idx = get_data_splits(len(dataset), seed=seed)
    train_loader = PyGDataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = PyGDataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = PyGDataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    node_dim = dataset.num_node_features
    edge_dim = dataset.num_edge_features
    model = model_fn(node_dim, edge_dim)
    model = model.to(device)
    n_params = count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_rmses, val_rmses = [], []

    save_path = os.path.join(MODELS_DIR, f'{exp_name}_seed{seed}_best.pt')

    logger.info(f"[{exp_name}/seed{seed}] params={n_params:,}")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_gnn_epoch(model, train_loader, optimizer, criterion, device, use_spectral=True)
        val_loss = evaluate_gnn(model, val_loader, criterion, device, use_spectral=True)

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
            logger.info(f"[{exp_name}/seed{seed}] Early stop at epoch {epoch}, best={best_epoch}")
            break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_loss = evaluate_gnn(model, test_loader, criterion, device, use_spectral=True)
    test_rmse = np.sqrt(test_loss)
    best_train_rmse = train_rmses[best_epoch - 1]

    # Restore original features
    dataset.data.spectral_features = orig_features

    logger.info(f"[{exp_name}/seed{seed}] FINAL: train={best_train_rmse:.4f}, "
                f"val={best_val_rmse:.4f}, test={test_rmse:.4f}")

    return {
        'exp_name': exp_name, 'seed': seed,
        'train_rmse': best_train_rmse, 'val_rmse': best_val_rmse, 'test_rmse': test_rmse,
        'best_epoch': best_epoch, 'total_epochs': len(train_rmses),
        'n_params': n_params, 'overfitting_gap': test_rmse - best_train_rmse,
    }


def run_spectral_mlp_experiment(spectral_features, tgs, seed, device, exp_name="spectral_mlp"):
    """Train Pure Spectral MLP (no GNN)."""
    set_seed(seed)
    os.makedirs(MODELS_DIR, exist_ok=True)

    n_samples = len(tgs)
    train_idx, val_idx, test_idx = get_data_splits(n_samples, seed=seed)

    train_dataset = SpectralOnlyDataset(spectral_features[train_idx], tgs[train_idx])
    val_dataset = SpectralOnlyDataset(spectral_features[val_idx], tgs[val_idx])
    test_dataset = SpectralOnlyDataset(spectral_features[test_idx], tgs[test_idx])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = PureSpectralMLP(n_features=N_SPECTRAL).to(device)
    n_params = count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_rmses, val_rmses = [], []

    save_path = os.path.join(MODELS_DIR, f'{exp_name}_seed{seed}_best.pt')

    logger.info(f"[{exp_name}/seed{seed}] params={n_params:,}")

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
        'exp_name': exp_name, 'seed': seed,
        'train_rmse': best_train_rmse, 'val_rmse': best_val_rmse, 'test_rmse': test_rmse,
        'best_epoch': best_epoch, 'total_epochs': len(train_rmses),
        'n_params': n_params, 'overfitting_gap': test_rmse - best_train_rmse,
    }


# ============================================================
# Aggregation & Statistics
# ============================================================

def aggregate_results(results_list, exp_name):
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


def compute_paired_ttest(baseline_test_rmses, experiment_test_rmses):
    """Compute paired t-test between baseline and experiment across seeds."""
    if len(baseline_test_rmses) != len(experiment_test_rmses):
        return None, None
    if len(baseline_test_rmses) < 2:
        return None, None
    t_stat, p_value = ttest_rel(baseline_test_rmses, experiment_test_rmses)
    return t_stat, p_value


def save_full_results(all_agg_results, baseline_agg, output_path):
    """Save comprehensive results summary."""
    baseline_tests = [r['test_rmse'] for r in baseline_agg['individual_results']]

    with open(output_path, 'w') as f:
        f.write("BlochSpec Iteration 2 Results Summary\n")
        f.write("CRITICAL FIX: global_mean_pool (matching Day 1 reference ~50.19)\n")
        f.write("=" * 100 + "\n\n")

        # Summary table
        f.write(f"{'Experiment':<40} {'Test RMSE':>20} {'Train RMSE':>20} {'Gap':>15} {'Params':>12}\n")
        f.write("-" * 100 + "\n")

        for r in all_agg_results:
            test_str = f"{r['test_rmse_mean']:.2f} +/- {r['test_rmse_std']:.2f}"
            train_str = f"{r['train_rmse_mean']:.2f} +/- {r['train_rmse_std']:.2f}"
            gap_str = f"{r['gap_mean']:.2f} +/- {r['gap_std']:.2f}"
            n_params = r['individual_results'][0]['n_params']
            param_str = f"{n_params:,}"
            f.write(f"{r['exp_name']:<40} {test_str:>20} {train_str:>20} {gap_str:>15} {param_str:>12}\n")

        # Per-seed results
        f.write("\n" + "=" * 100 + "\n")
        f.write("Per-Seed Results:\n")
        f.write("-" * 100 + "\n")
        for r in all_agg_results:
            f.write(f"\n{r['exp_name']}:\n")
            for ind in r['individual_results']:
                f.write(f"  seed={ind['seed']}: test={ind['test_rmse']:.4f}, "
                        f"train={ind['train_rmse']:.4f}, val={ind['val_rmse']:.4f}, "
                        f"best_epoch={ind['best_epoch']}, gap={ind['overfitting_gap']:.4f}\n")

        # Statistical significance testing
        f.write("\n" + "=" * 100 + "\n")
        f.write("Statistical Significance (Paired t-test vs Baseline):\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Experiment':<40} {'t-stat':>12} {'p-value':>12} {'Significant?':>15}\n")
        f.write("-" * 100 + "\n")

        for r in all_agg_results:
            if r['exp_name'] == baseline_agg['exp_name']:
                f.write(f"{r['exp_name']:<40} {'---':>12} {'---':>12} {'baseline':>15}\n")
                continue

            exp_tests = [ind['test_rmse'] for ind in r['individual_results']]
            t_stat, p_value = compute_paired_ttest(baseline_tests, exp_tests)
            if t_stat is not None:
                sig = "YES (p<0.05)" if p_value < 0.05 else "NO (p>=0.05)"
                # Direction: positive t means baseline > experiment (experiment better)
                if t_stat > 0:
                    direction = "(exp better)"
                else:
                    direction = "(exp worse)"
                f.write(f"{r['exp_name']:<40} {t_stat:>12.4f} {p_value:>12.4f} {sig + ' ' + direction:>15}\n")
            else:
                f.write(f"{r['exp_name']:<40} {'N/A':>12} {'N/A':>12} {'N/A':>15}\n")

        # Delta from baseline
        f.write("\n" + "=" * 100 + "\n")
        f.write("Delta from Baseline (negative = improvement):\n")
        f.write("-" * 100 + "\n")
        baseline_mean = baseline_agg['test_rmse_mean']
        for r in all_agg_results:
            delta = r['test_rmse_mean'] - baseline_mean
            pct = (delta / baseline_mean) * 100
            f.write(f"{r['exp_name']:<40} delta={delta:+.2f} ({pct:+.1f}%)\n")

        # Key analysis
        f.write("\n" + "=" * 100 + "\n")
        f.write("Analysis:\n")
        f.write("-" * 100 + "\n")

        # Find blochspec and random results
        blochspec = next((r for r in all_agg_results if 'Bloch' in r['exp_name'] and 'Random' not in r['exp_name']
                         and 'Unweighted' not in r['exp_name'] and 'Reduced' not in r['exp_name']), None)
        random_res = next((r for r in all_agg_results if 'Random' in r['exp_name']), None)
        wider_res = next((r for r in all_agg_results if 'Wider' in r['exp_name']), None)
        reduced_res = next((r for r in all_agg_results if 'Reduced' in r['exp_name']), None)

        f.write(f"\n1. Baseline (global_mean_pool): {baseline_agg['test_rmse_mean']:.2f} +/- {baseline_agg['test_rmse_std']:.2f}\n")
        f.write(f"   Day 1 reference: ~50.19\n")

        if blochspec:
            delta_b = blochspec['test_rmse_mean'] - baseline_mean
            f.write(f"\n2. BlochSpec improvement: {delta_b:+.2f}\n")

        if random_res:
            delta_r = random_res['test_rmse_mean'] - baseline_mean
            f.write(f"   Random features delta: {delta_r:+.2f}\n")
            if blochspec and random_res:
                bloch_vs_random = blochspec['test_rmse_mean'] - random_res['test_rmse_mean']
                f.write(f"   Bloch vs Random: {bloch_vs_random:+.2f}\n")
                if abs(bloch_vs_random) < 2.0:
                    f.write("   WARNING: Bloch and Random features produce similar results.\n")
                    f.write("   This suggests the improvement is from dimensionality, not physics.\n")
                else:
                    f.write("   Bloch features differ meaningfully from random.\n")

        if wider_res:
            delta_w = wider_res['test_rmse_mean'] - baseline_mean
            f.write(f"\n3. Wider MLP (param control) delta: {delta_w:+.2f}\n")
            if blochspec:
                f.write(f"   If Wider MLP matches BlochSpec, improvement is from params not features.\n")

        if reduced_res:
            delta_red = reduced_res['test_rmse_mean'] - baseline_mean
            f.write(f"\n4. Reduced GNN (dim control, total=256) delta: {delta_red:+.2f}\n")
            f.write(f"   If this matches BlochSpec, the input dimension matters more than feature content.\n")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load spectral data
    spectral_features = np.load(os.path.join(SPECTRAL_DIR, 'spectral_features.npy'))
    spectral_indices = np.load(os.path.join(SPECTRAL_DIR, 'feature_indices.npy'))
    feature_tgs = np.load(os.path.join(SPECTRAL_DIR, 'feature_tgs.npy'))
    spectral_random_pre = np.load(os.path.join(SPECTRAL_DIR, 'spectral_features_random.npy'))
    spectral_unweighted = np.load(os.path.join(SPECTRAL_DIR, 'spectral_features_unweighted.npy'))

    # Prepare dataset
    dataset = prepare_dataset(CSV_PATH, SPECTRAL_DIR, PROCESSED_DIR)
    logger.info(f"Dataset: {len(dataset)} graphs, node_dim={dataset.num_node_features}, "
                f"edge_dim={dataset.num_edge_features}")

    assert len(dataset) == len(spectral_features), \
        f"Dataset size {len(dataset)} != spectral features {len(spectral_features)}"

    node_dim = dataset.num_node_features
    edge_dim = dataset.num_edge_features

    all_agg_results = []

    # ==========================================
    # (a) Baseline GIN+Trimer with global_mean_pool
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("(a) Baseline GIN+Trimer (global_mean_pool)")
    logger.info("=" * 60)

    baseline_results = []
    for seed in SEEDS:
        model = BaselineGIN(node_dim=node_dim, edge_dim=edge_dim,
                            hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
        r = run_gnn_experiment(model, dataset, seed, device, "baseline_gin_meanpool")
        baseline_results.append(r)
    baseline_agg = aggregate_results(baseline_results, "(a) Baseline GIN+Trimer")
    all_agg_results.append(baseline_agg)

    # ==========================================
    # (b) BlochSpec: GIN+Trimer + 12 Bloch features
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("(b) BlochSpec (GIN+Bloch spectral)")
    logger.info("=" * 60)

    blochspec_results = []
    for seed in SEEDS:
        model = BlochSpecGIN(node_dim=node_dim, edge_dim=edge_dim,
                             n_spectral_features=N_SPECTRAL,
                             hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
        r = run_gnn_experiment(model, dataset, seed, device, "blochspec_full",
                               use_spectral=True)
        blochspec_results.append(r)
    all_agg_results.append(aggregate_results(blochspec_results, "(b) BlochSpec (GIN+Bloch)"))

    # ==========================================
    # (c) Wider MLP GIN (parameter count control)
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("(c) Wider MLP GIN (param count control, no extra features)")
    logger.info("=" * 60)

    wider_results = []
    for seed in SEEDS:
        model = WiderMLPGIN(node_dim=node_dim, edge_dim=edge_dim,
                            hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
        r = run_gnn_experiment(model, dataset, seed, device, "wider_mlp_gin")
        wider_results.append(r)
    all_agg_results.append(aggregate_results(wider_results, "(c) GIN+WiderMLP (param ctrl)"))

    # ==========================================
    # (d) Reduced GNN dim + Bloch features (total input dim = 256)
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("(d) Reduced GNN (dim=244) + 12 Bloch features (total=256)")
    logger.info("=" * 60)

    reduced_results = []
    for seed in SEEDS:
        model = ReducedGNNBlochSpec(node_dim=node_dim, edge_dim=edge_dim,
                                    n_spectral_features=N_SPECTRAL,
                                    hidden_dim=244, num_layers=NUM_LAYERS)
        r = run_gnn_experiment(model, dataset, seed, device, "reduced_gnn_bloch",
                               use_spectral=True)
        reduced_results.append(r)
    all_agg_results.append(aggregate_results(reduced_results, "(d) ReducedGNN+Bloch (dim ctrl)"))

    # ==========================================
    # (e) Pure Spectral MLP
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("(e) Pure Spectral MLP (12 features only)")
    logger.info("=" * 60)

    mlp_results = []
    for seed in SEEDS:
        r = run_spectral_mlp_experiment(spectral_features, feature_tgs, seed, device,
                                         exp_name="spectral_mlp")
        mlp_results.append(r)
    all_agg_results.append(aggregate_results(mlp_results, "(e) Pure Spectral MLP"))

    # ==========================================
    # (f) GIN+Trimer + Random features
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("(f) GIN+Trimer + Random features")
    logger.info("=" * 60)

    def make_blochspec_model(nd, ed):
        return BlochSpecGIN(node_dim=nd, edge_dim=ed,
                           n_spectral_features=N_SPECTRAL,
                           hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)

    random_results = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        n_samples, n_feats = spectral_features.shape
        rand_feats = np.zeros_like(spectral_features)
        for j in range(n_feats):
            col_mean = np.mean(spectral_features[:, j])
            col_std = np.std(spectral_features[:, j])
            rand_feats[:, j] = rng.normal(col_mean, col_std, n_samples)
        rand_feats = rand_feats.astype(np.float32)

        r = run_gnn_with_alt_features(make_blochspec_model, dataset, rand_feats,
                                      seed, device, "blochspec_random")
        random_results.append(r)
    all_agg_results.append(aggregate_results(random_results, "(f) GIN + Random Features"))

    # ==========================================
    # (g) GIN+Trimer + Unweighted Laplacian
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("(g) GIN+Trimer + Unweighted Laplacian features")
    logger.info("=" * 60)

    unweighted_results = []
    for seed in SEEDS:
        r = run_gnn_with_alt_features(make_blochspec_model, dataset, spectral_unweighted,
                                      seed, device, "blochspec_unweighted")
        unweighted_results.append(r)
    all_agg_results.append(aggregate_results(unweighted_results, "(g) GIN + Unweighted Lap"))

    # Save results
    results_path = os.path.join(RESULTS_DIR, "results_summary.txt")
    save_full_results(all_agg_results, baseline_agg, results_path)
    logger.info(f"\nResults saved to {results_path}")

    # Print summary
    print("\n" + "=" * 100)
    print("BLOCHSPEC ITERATION 2 RESULTS (global_mean_pool)")
    print("=" * 100)
    print(f"{'Experiment':<40} {'Test RMSE':>20} {'Train RMSE':>20} {'Gap':>15}")
    print("-" * 100)
    for r in all_agg_results:
        test_str = f"{r['test_rmse_mean']:.2f} +/- {r['test_rmse_std']:.2f}"
        train_str = f"{r['train_rmse_mean']:.2f} +/- {r['train_rmse_std']:.2f}"
        gap_str = f"{r['gap_mean']:.2f} +/- {r['gap_std']:.2f}"
        print(f"{r['exp_name']:<40} {test_str:>20} {train_str:>20} {gap_str:>15}")

    # Statistical tests
    print("\n" + "=" * 100)
    print("SIGNIFICANCE TESTS vs Baseline")
    print("=" * 100)
    baseline_tests = [r['test_rmse'] for r in baseline_agg['individual_results']]
    for r in all_agg_results:
        if r['exp_name'] == baseline_agg['exp_name']:
            continue
        exp_tests = [ind['test_rmse'] for ind in r['individual_results']]
        t_stat, p_value = compute_paired_ttest(baseline_tests, exp_tests)
        if t_stat is not None:
            sig = "SIGNIFICANT" if p_value < 0.05 else "not significant"
            delta = r['test_rmse_mean'] - baseline_agg['test_rmse_mean']
            print(f"  {r['exp_name']:<40} t={t_stat:+.3f}, p={p_value:.4f}, "
                  f"delta={delta:+.2f} [{sig}]")

    return all_agg_results


if __name__ == "__main__":
    results = main()
