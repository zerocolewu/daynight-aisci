"""
Diagnostic analysis for BlochSpec experiment.

1. Feature importance: train GIN+Trimer with 1 spectral feature at a time
2. Overfitting analysis
3. Correlation analysis (detailed)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import BlochSpecGIN, count_parameters
from dataset import get_data_splits, prepare_dataset
from compute_spectral import FEATURE_NAMES

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

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


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_batch_spectral_features(batch, n_features=12):
    """Extract spectral features from batch and reshape to [batch_size, n_features]."""
    sf = batch.spectral_features
    if sf.dim() == 1:
        sf = sf.view(batch.num_graphs, n_features)
    return sf


def train_single_feature_experiment(dataset, feature_idx, seed, device):
    """Train BlochSpecGIN with only 1 spectral feature (rest set to zero)."""
    set_seed(seed)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_idx, val_idx, test_idx = get_data_splits(len(dataset), seed=seed)

    # Modify spectral features: keep only feature_idx, zero out others
    # Data is stored as flat 1D tensor: [f1_1, f2_1, ..., f12_1, f1_2, f2_2, ..., f12_2, ...]
    orig_features = dataset.data.spectral_features.clone()
    n_samples = len(dataset)
    n_feat = 12
    orig_2d = orig_features.view(n_samples, n_feat)
    single_feat_2d = torch.zeros_like(orig_2d)
    single_feat_2d[:, feature_idx] = orig_2d[:, feature_idx]
    dataset.data.spectral_features = single_feat_2d.flatten()

    train_loader = PyGDataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = PyGDataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = PyGDataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    node_dim = dataset.num_node_features
    edge_dim = dataset.num_edge_features

    model = BlochSpecGIN(node_dim=node_dim, edge_dim=edge_dim,
                         n_spectral_features=12,
                         hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_train_rmse = float('inf')

    exp_name = f"single_feat_{feature_idx}"
    save_path = os.path.join(MODELS_DIR, f'{exp_name}_seed{seed}_best.pt')

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0
        n_samples = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                         get_batch_spectral_features(batch))
            loss = criterion(pred.squeeze(), batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            n_samples += batch.num_graphs
        train_rmse = np.sqrt(total_loss / n_samples)
        scheduler.step()

        model.eval()
        total_loss = 0
        n_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                             get_batch_spectral_features(batch))
                loss = criterion(pred.squeeze(), batch.y.squeeze())
                total_loss += loss.item() * batch.num_graphs
                n_samples += batch.num_graphs
        val_rmse = np.sqrt(total_loss / n_samples)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_train_rmse = train_rmse
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    # Test evaluation
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    total_loss = 0
    n_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                         get_batch_spectral_features(batch))
            loss = criterion(pred.squeeze(), batch.y.squeeze())
            total_loss += loss.item() * batch.num_graphs
            n_samples += batch.num_graphs
    test_rmse = np.sqrt(total_loss / n_samples)

    # Restore
    dataset.data.spectral_features = orig_features

    return {
        'feature_idx': feature_idx,
        'feature_name': FEATURE_NAMES[feature_idx],
        'test_rmse': test_rmse,
        'val_rmse': best_val_rmse,
        'train_rmse': best_train_rmse,
        'best_epoch': best_epoch,
    }


def run_feature_importance(dataset, seed, device):
    """Run feature importance analysis: one feature at a time."""
    logger.info("\n=== Feature Importance Analysis ===")

    results = []
    for feat_idx in range(12):
        logger.info(f"Training with feature {feat_idx}: {FEATURE_NAMES[feat_idx]}")
        r = train_single_feature_experiment(dataset, feat_idx, seed, device)
        results.append(r)
        logger.info(f"  test_rmse={r['test_rmse']:.4f}")

    # Sort by test RMSE (lower is better, more important feature)
    results.sort(key=lambda x: x['test_rmse'])

    # Save
    output_path = os.path.join(RESULTS_DIR, "feature_importance.txt")
    with open(output_path, 'w') as f:
        f.write("Feature Importance Analysis (1 feature at a time, seed={})\n".format(seed))
        f.write("Lower test RMSE = more important feature\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Rank':<6} {'Feature':<25} {'Test RMSE':<12} {'Val RMSE':<12} {'Best Epoch':<12}\n")
        f.write("-" * 70 + "\n")
        for rank, r in enumerate(results, 1):
            f.write(f"{rank:<6} {r['feature_name']:<25} {r['test_rmse']:<12.4f} "
                    f"{r['val_rmse']:<12.4f} {r['best_epoch']:<12}\n")

    logger.info(f"\nFeature importance saved to {output_path}")
    return results


def run_diagnostics():
    """Run all diagnostic analyses."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load dataset
    dataset = prepare_dataset(CSV_PATH, SPECTRAL_DIR, PROCESSED_DIR)

    # Load spectral features for correlation analysis
    spectral_features = np.load(os.path.join(SPECTRAL_DIR, 'spectral_features.npy'))
    feature_tgs = np.load(os.path.join(SPECTRAL_DIR, 'feature_tgs.npy'))

    # 1. Correlation analysis (already done in compute_spectral.py, but redo here)
    logger.info("\n=== Correlation Analysis ===")
    from scipy.stats import pearsonr
    correlations = []
    for i, name in enumerate(FEATURE_NAMES):
        r, p = pearsonr(spectral_features[:, i], feature_tgs)
        correlations.append((name, r, p))
        logger.info(f"  {name:25s}: r={r:+.4f}, p={p:.2e}")

    # 2. Feature importance
    feature_importance = run_feature_importance(dataset, seed=71, device=device)

    # 3. Overfitting analysis
    logger.info("\n=== Overfitting Analysis ===")
    results_files = os.path.join(RESULTS_DIR, "results_summary.txt")
    if os.path.exists(results_files):
        logger.info("Results summary file exists - overfitting analysis included in results_summary.txt")

    # 4. Save comprehensive diagnostics
    diag_path = os.path.join(RESULTS_DIR, "diagnostics.txt")
    with open(diag_path, 'w') as f:
        f.write("BlochSpec Diagnostic Analysis\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. Feature Correlations with Tg\n")
        f.write("-" * 50 + "\n")
        for name, r, p in correlations:
            flag = "***" if abs(r) > 0.2 else ""
            f.write(f"  {name:25s}: r={r:+.4f}, p={p:.2e} {flag}\n")

        f.write(f"\n2. Feature Importance (single feature, ranked)\n")
        f.write("-" * 50 + "\n")
        for rank, r in enumerate(feature_importance, 1):
            f.write(f"  {rank:2d}. {r['feature_name']:25s}: test_rmse={r['test_rmse']:.4f}\n")

    logger.info(f"Diagnostics saved to {diag_path}")
    return feature_importance


if __name__ == "__main__":
    run_diagnostics()
