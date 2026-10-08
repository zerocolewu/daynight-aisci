"""
PCR-v2 Iteration 3: Training script.

5 Experiments:
1. Baseline: EXACT Day 1 architecture (3-layer GINEConv, mean_pool, no contrastive)
2. PCR alpha=0.01 with L2 norm on contrastive projector input
3. PCR alpha=0.05
4. PCR alpha=0.1
5. PCR alpha=0.5

Training settings (matching Day 1):
- seed=71, 72/8/20 split, patience=30
- LR=1e-3, Adam optimizer (NO cosine annealing to match Day 1)
- Batch_size=64 polymers (128 graphs per batch for PCR)
- Alpha warmup: 0->target over 20 epochs

InfoNCE: tau=0.07, positive = (monomer_i, trimer_i), negatives = all others in batch
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch

from dataset import PCRDualViewDataset, get_data_splits
from model_v3 import PCRModel, info_nce_loss, cosine_similarity_pairs, count_parameters

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Paths
CSV_PATH = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
DATA_DIR = "./data"
MODELS_DIR = "./saved_models_v3"
RESULTS_PATH = "./results_v3.json"

# Training hyperparameters (matching Day 1 exactly)
SEED = 71
HIDDEN_DIM = 256
NUM_LAYERS = 3
LR = 1e-3
BATCH_SIZE = 32  # Match Day 1 exactly (32 polymers, 64 graphs for PCR)
MAX_EPOCHS = 500
PATIENCE = 30
WARMUP_EPOCHS = 20


class DualViewBatch:
    """Collates monomer+trimer pairs into a single batch."""

    @staticmethod
    def collate(mono_graphs, tri_graphs, targets):
        mono_batch = Batch.from_data_list(mono_graphs)
        tri_batch = Batch.from_data_list(tri_graphs)
        targets_t = torch.tensor(targets, dtype=torch.float)
        return mono_batch, tri_batch, targets_t


def make_data_loaders(dataset, train_idx, val_idx, test_idx, batch_size):
    """Create data loaders for dual-view training."""

    def get_graphs(dataset, indices):
        mono = [dataset.monomer_graphs[i] for i in indices]
        tri = [dataset.trimer_graphs[i] for i in indices]
        tg = [dataset.targets[i] for i in indices]
        return mono, tri, tg

    class DualViewLoader:
        def __init__(self, mono, tri, tg, batch_size, shuffle=False):
            self.mono = mono
            self.tri = tri
            self.tg = tg
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.n = len(tg)
            self.indices = list(range(self.n))

        def __len__(self):
            return (self.n + self.batch_size - 1) // self.batch_size

        def __iter__(self):
            if self.shuffle:
                order = np.random.permutation(self.n).tolist()
            else:
                order = list(range(self.n))

            for start in range(0, self.n, self.batch_size):
                batch_idx = order[start:start + self.batch_size]
                mono_batch = [self.mono[i] for i in batch_idx]
                tri_batch = [self.tri[i] for i in batch_idx]
                tg_batch = [self.tg[i] for i in batch_idx]
                yield DualViewBatch.collate(mono_batch, tri_batch, tg_batch)

    mono_train, tri_train, tg_train = get_graphs(dataset, train_idx)
    mono_val, tri_val, tg_val = get_graphs(dataset, val_idx)
    mono_test, tri_test, tg_test = get_graphs(dataset, test_idx)

    train_loader = DualViewLoader(mono_train, tri_train, tg_train, batch_size, shuffle=True)
    val_loader = DualViewLoader(mono_val, tri_val, tg_val, batch_size, shuffle=False)
    test_loader = DualViewLoader(mono_test, tri_test, tg_test, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_baseline(exp_name, dataset, train_idx, val_idx, test_idx, device):
    """Train baseline model (no contrastive)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment: {exp_name} (BASELINE)")
    logger.info(f"{'='*60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    train_loader, val_loader, test_loader = make_data_loaders(
        dataset, train_idx, val_idx, test_idx, BATCH_SIZE)

    node_dim = dataset.num_node_features
    edge_dim = dataset.num_edge_features

    model = PCRModel(
        node_dim=node_dim, edge_dim=edge_dim,
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        mode='baseline',
    ).to(device)

    n_params = count_parameters(model)
    logger.info(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, f'{exp_name}_best.pt')

    train_rmses = []
    val_rmses = []

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0
        n_samples = 0
        for mono_batch, tri_batch, tg_batch in train_loader:
            tri_batch = tri_batch.to(device)
            tg_batch = tg_batch.to(device)

            optimizer.zero_grad()
            pred, _ = model.forward_baseline(
                tri_batch.x, tri_batch.edge_index, tri_batch.edge_attr, tri_batch.batch)
            loss = criterion(pred, tg_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(tg_batch)
            n_samples += len(tg_batch)

        train_rmse = np.sqrt(total_loss / n_samples)

        # Validate
        model.eval()
        total_loss = 0
        n_samples = 0
        with torch.no_grad():
            for mono_batch, tri_batch, tg_batch in val_loader:
                tri_batch = tri_batch.to(device)
                tg_batch = tg_batch.to(device)
                pred, _ = model.forward_baseline(
                    tri_batch.x, tri_batch.edge_index, tri_batch.edge_attr, tri_batch.batch)
                loss = criterion(pred, tg_batch)
                total_loss += loss.item() * len(tg_batch)
                n_samples += len(tg_batch)

        val_rmse = np.sqrt(total_loss / n_samples)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"[{exp_name}] Epoch {epoch}: train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}")

        if patience_counter >= PATIENCE:
            logger.info(f"[{exp_name}] Early stopping at epoch {epoch}, best={best_epoch}")
            break

    # Test
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    total_loss = 0
    n_samples = 0
    with torch.no_grad():
        for mono_batch, tri_batch, tg_batch in test_loader:
            tri_batch = tri_batch.to(device)
            tg_batch = tg_batch.to(device)
            pred, _ = model.forward_baseline(
                tri_batch.x, tri_batch.edge_index, tri_batch.edge_attr, tri_batch.batch)
            loss = criterion(pred, tg_batch)
            total_loss += loss.item() * len(tg_batch)
            n_samples += len(tg_batch)

    test_rmse = np.sqrt(total_loss / n_samples)
    best_train_rmse = train_rmses[best_epoch - 1]

    logger.info(f"[{exp_name}] FINAL: train={best_train_rmse:.4f}, val={best_val_rmse:.4f}, "
                f"test={test_rmse:.4f}, best_epoch={best_epoch}")

    return {
        'exp_name': exp_name,
        'mode': 'baseline',
        'train_rmse': float(best_train_rmse),
        'val_rmse': float(best_val_rmse),
        'test_rmse': float(test_rmse),
        'best_epoch': best_epoch,
        'total_epochs': len(train_rmses),
        'n_params': n_params,
    }


def train_pcr(exp_name, dataset, train_idx, val_idx, test_idx, alpha, device):
    """Train PCR model with contrastive loss."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment: {exp_name} (PCR, alpha={alpha})")
    logger.info(f"{'='*60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    train_loader, val_loader, test_loader = make_data_loaders(
        dataset, train_idx, val_idx, test_idx, BATCH_SIZE)

    node_dim = dataset.num_node_features
    edge_dim = dataset.num_edge_features

    model = PCRModel(
        node_dim=node_dim, edge_dim=edge_dim,
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        mode='pcr',
    ).to(device)

    n_params = count_parameters(model)
    logger.info(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, f'{exp_name}_best.pt')

    train_rmses = []
    val_rmses = []
    contrastive_losses = []

    for epoch in range(1, MAX_EPOCHS + 1):
        # Alpha warmup
        current_alpha = alpha * min(1.0, epoch / WARMUP_EPOCHS)

        # Train
        model.train()
        total_sup_loss = 0
        total_con_loss = 0
        n_samples = 0

        for mono_batch, tri_batch, tg_batch in train_loader:
            mono_batch = mono_batch.to(device)
            tri_batch = tri_batch.to(device)
            tg_batch = tg_batch.to(device)

            optimizer.zero_grad()
            pred, z_mono, z_tri, _ = model.forward_pcr(
                mono_batch.x, mono_batch.edge_index, mono_batch.edge_attr, mono_batch.batch,
                tri_batch.x, tri_batch.edge_index, tri_batch.edge_attr, tri_batch.batch,
            )

            sup_loss = criterion(pred, tg_batch)
            con_loss = info_nce_loss(z_mono, z_tri)
            loss = sup_loss + current_alpha * con_loss

            loss.backward()
            optimizer.step()

            total_sup_loss += sup_loss.item() * len(tg_batch)
            total_con_loss += con_loss.item() * len(tg_batch)
            n_samples += len(tg_batch)

        train_rmse = np.sqrt(total_sup_loss / n_samples)
        avg_con_loss = total_con_loss / n_samples

        # Validate using baseline forward (fair comparison)
        model.eval()
        total_loss = 0
        n_samples = 0
        with torch.no_grad():
            for mono_batch, tri_batch, tg_batch in val_loader:
                tri_batch = tri_batch.to(device)
                tg_batch = tg_batch.to(device)
                pred, _ = model.forward_baseline(
                    tri_batch.x, tri_batch.edge_index, tri_batch.edge_attr, tri_batch.batch)
                loss = criterion(pred, tg_batch)
                total_loss += loss.item() * len(tg_batch)
                n_samples += len(tg_batch)

        val_rmse = np.sqrt(total_loss / n_samples)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)
        contrastive_losses.append(avg_con_loss)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            cos_sim = cosine_similarity_pairs(z_mono, z_tri)
            logger.info(f"[{exp_name}] Epoch {epoch}: train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}, "
                        f"con_loss={avg_con_loss:.4f}, alpha={current_alpha:.4f}, cos_sim={cos_sim:.4f}")

        if patience_counter >= PATIENCE:
            logger.info(f"[{exp_name}] Early stopping at epoch {epoch}, best={best_epoch}")
            break

    # Test using baseline forward
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    total_loss = 0
    n_samples = 0
    with torch.no_grad():
        for mono_batch, tri_batch, tg_batch in test_loader:
            tri_batch = tri_batch.to(device)
            tg_batch = tg_batch.to(device)
            pred, _ = model.forward_baseline(
                tri_batch.x, tri_batch.edge_index, tri_batch.edge_attr, tri_batch.batch)
            loss = criterion(pred, tg_batch)
            total_loss += loss.item() * len(tg_batch)
            n_samples += len(tg_batch)

    test_rmse = np.sqrt(total_loss / n_samples)
    best_train_rmse = train_rmses[best_epoch - 1]

    logger.info(f"[{exp_name}] FINAL: train={best_train_rmse:.4f}, val={best_val_rmse:.4f}, "
                f"test={test_rmse:.4f}, best_epoch={best_epoch}")

    return {
        'exp_name': exp_name,
        'mode': 'pcr',
        'alpha': alpha,
        'train_rmse': float(best_train_rmse),
        'val_rmse': float(best_val_rmse),
        'test_rmse': float(test_rmse),
        'best_epoch': best_epoch,
        'total_epochs': len(train_rmses),
        'n_params': n_params,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    logger.info(f"Architecture: Day 1 exact - EdgeAwareGINConv, 3 layers, mean_pool, MLP(256->128->1)")

    # Load dataset
    dataset = PCRDualViewDataset(root=DATA_DIR, csv_path=CSV_PATH)
    train_idx, val_idx, test_idx = get_data_splits(len(dataset), seed=SEED)
    logger.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    logger.info(f"Node features: {dataset.num_node_features}, Edge features: {dataset.num_edge_features}")

    all_results = []

    # Experiment 1: Baseline
    result = train_baseline('v3_baseline', dataset, train_idx, val_idx, test_idx, device)
    all_results.append(result)

    # Verify baseline is reasonable (~50.19)
    baseline_test = result['test_rmse']
    logger.info(f"\n*** BASELINE TEST RMSE: {baseline_test:.4f} ***")
    if baseline_test > 60:
        logger.warning(f"BASELINE RMSE ({baseline_test:.2f}) is much worse than expected 50.19!")
        logger.warning("Architecture mismatch detected. Check implementation.")

    # Experiments 2-5: PCR with different alphas
    alphas = [0.01, 0.05, 0.1, 0.5]
    for alpha in alphas:
        exp_name = f'v3_pcr_a{alpha}'
        result = train_pcr(exp_name, dataset, train_idx, val_idx, test_idx, alpha, device)
        all_results.append(result)

    # Save results
    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {RESULTS_PATH}")

    # Print summary
    print("\n" + "=" * 80)
    print("PCR-v2 Iteration 3 RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<25} {'Train RMSE':<12} {'Val RMSE':<12} {'Test RMSE':<12} "
          f"{'Best Epoch':<12} {'Params':<12}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['exp_name']:<25} {r['train_rmse']:<12.4f} {r['val_rmse']:<12.4f} "
              f"{r['test_rmse']:<12.4f} {r['best_epoch']:<12} {r['n_params']:<12,}")

    return all_results


if __name__ == "__main__":
    results = main()
