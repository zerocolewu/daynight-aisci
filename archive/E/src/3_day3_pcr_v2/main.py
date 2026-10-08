"""
PCR-v2 Iteration 2: Hybrid pooling experiments.

Fix from iteration 1: global_mean_pool degraded baseline from 50.19 to 52.68.
Solution: sum_pool for supervised head + mean_pool for contrastive projector.

4 Experiments:
1. Baseline: GIN+sum_pool+trimer (validates hybrid pooling matches ~50.19)
2. PCR-v2 alpha=0.01 (very weak contrastive)
3. PCR-v2 alpha=0.05 (auxiliary contrastive)
4. PCR-v2 alpha=0.1  (for comparison with iteration 1)

All use seed=71, 72/8/20 split, patience=50.
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from dataset import PCRDualViewDataset, get_data_splits, ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from model import (PCRModel, info_nce_loss, cosine_similarity_pairs,
                   alignment_metric, uniformity_metric, count_parameters)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Constants
CSV_PATH = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
DATA_DIR = "./data"
MODELS_DIR = "./saved_models_v2"
PLOTS_DIR = "./diagnostic_plots_v2"
RESULTS_PATH = "./results_v2.json"

SEED = 71
HIDDEN_DIM = 256
NUM_LAYERS = 5
DROPOUT = 0.5
LR = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 500
PATIENCE = 50
TAU = 0.07
WARMUP_EPOCHS = 30


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def make_data_loader(dataset, indices, batch_size, shuffle=True):
    """Custom batch collation for dual-view dataset."""
    class DualViewLoader:
        def __init__(self, dataset, indices, batch_size, shuffle):
            self.dataset = dataset
            self.indices = list(indices)
            self.batch_size = batch_size
            self.shuffle = shuffle

        def __iter__(self):
            idxs = list(self.indices)
            if self.shuffle:
                np.random.shuffle(idxs)
            for start in range(0, len(idxs), self.batch_size):
                batch_idxs = idxs[start:start + self.batch_size]
                mono_graphs = [self.dataset.monomer_graphs[i] for i in batch_idxs]
                tri_graphs = [self.dataset.trimer_graphs[i] for i in batch_idxs]
                targets = torch.tensor([self.dataset.targets[i] for i in batch_idxs],
                                       dtype=torch.float)
                mono_batch = Batch.from_data_list(mono_graphs)
                tri_batch = Batch.from_data_list(tri_graphs)
                yield mono_batch, tri_batch, targets

        def __len__(self):
            return (len(self.indices) + self.batch_size - 1) // self.batch_size

    return DualViewLoader(dataset, indices, batch_size, shuffle)


# ========== EXPERIMENT 1: BASELINE (sum_pool) ==========
def train_baseline(dataset, train_idx, val_idx, test_idx, device):
    """Baseline: GIN+sum_pool+trimer, no contrastive."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Baseline (GIN+sum_pool+trimer, no contrastive)")
    logger.info("=" * 60)

    set_seed(SEED)
    model = PCRModel(
        node_dim=ATOM_FEATURE_DIM,
        edge_dim=BOND_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        mode='baseline',
    ).to(device)

    n_params = count_parameters(model)
    logger.info(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.MSELoss()

    loader = make_data_loader(dataset, train_idx, BATCH_SIZE, shuffle=True)
    val_loader = make_data_loader(dataset, val_idx, BATCH_SIZE, shuffle=False)
    test_loader = make_data_loader(dataset, test_idx, BATCH_SIZE, shuffle=False)

    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_rmses, val_rmses = [], []

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, 'baseline_best.pt')

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        epoch_loss = 0
        n_samples = 0
        for mono_batch, tri_batch, targets in loader:
            tri_batch = tri_batch.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            pred, _ = model.forward_baseline(
                tri_batch.x, tri_batch.edge_index, tri_batch.edge_attr, tri_batch.batch)
            loss = criterion(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(targets)
            n_samples += len(targets)

        scheduler.step()
        train_rmse = np.sqrt(epoch_loss / n_samples)
        train_rmses.append(train_rmse)

        # Validate
        model.eval()
        val_loss = 0
        val_n = 0
        with torch.no_grad():
            for mono_batch, tri_batch, targets in val_loader:
                tri_batch = tri_batch.to(device)
                targets = targets.to(device)
                pred, _ = model.forward_baseline(
                    tri_batch.x, tri_batch.edge_index, tri_batch.edge_attr, tri_batch.batch)
                loss = criterion(pred, targets)
                val_loss += loss.item() * len(targets)
                val_n += len(targets)

        val_rmse = np.sqrt(val_loss / val_n)
        val_rmses.append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch}: train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}")

        if patience_counter >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Test
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    test_loss = 0
    test_n = 0
    with torch.no_grad():
        for mono_batch, tri_batch, targets in test_loader:
            tri_batch = tri_batch.to(device)
            targets = targets.to(device)
            pred, _ = model.forward_baseline(
                tri_batch.x, tri_batch.edge_index, tri_batch.edge_attr, tri_batch.batch)
            loss = criterion(pred, targets)
            test_loss += loss.item() * len(targets)
            test_n += len(targets)

    test_rmse = np.sqrt(test_loss / test_n)
    best_train_rmse = train_rmses[best_epoch - 1]

    logger.info(f"BASELINE: train={best_train_rmse:.4f}, val={best_val_rmse:.4f}, "
                f"test={test_rmse:.4f}, best_epoch={best_epoch}")

    return {
        'exp_name': 'baseline_sum_pool',
        'test_rmse': test_rmse,
        'val_rmse': best_val_rmse,
        'train_rmse': best_train_rmse,
        'best_epoch': best_epoch,
        'total_epochs': len(train_rmses),
        'n_params': n_params,
        'overfitting_gap': test_rmse - best_train_rmse,
        'train_rmses': train_rmses,
        'val_rmses': val_rmses,
    }


# ========== EXPERIMENTS 2-4: PCR with hybrid pooling ==========
def train_pcr(dataset, train_idx, val_idx, test_idx, device, alpha, exp_name):
    """PCR-v2 with hybrid pooling: sum_pool→supervised, mean_pool→contrastive."""
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT: {exp_name} (alpha={alpha})")
    logger.info("=" * 60)

    set_seed(SEED)
    model = PCRModel(
        node_dim=ATOM_FEATURE_DIM,
        edge_dim=BOND_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        mode='pcr',
    ).to(device)

    n_params = count_parameters(model)
    logger.info(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.MSELoss()

    loader = make_data_loader(dataset, train_idx, BATCH_SIZE, shuffle=True)
    val_loader = make_data_loader(dataset, val_idx, BATCH_SIZE, shuffle=False)
    test_loader = make_data_loader(dataset, test_idx, BATCH_SIZE, shuffle=False)

    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_rmses, val_rmses = [], []
    sup_losses, con_losses = [], []
    cos_sims, alignments, uniformities = [], [], []

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, f'{exp_name}_best.pt')

    for epoch in range(1, MAX_EPOCHS + 1):
        current_alpha = alpha * min(1.0, epoch / WARMUP_EPOCHS)

        # Train
        model.train()
        epoch_sup_loss = 0
        epoch_con_loss = 0
        n_samples = 0
        epoch_cos_sims = []

        for mono_batch, tri_batch, targets in loader:
            mono_batch = mono_batch.to(device)
            tri_batch = tri_batch.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            pred, z_mono, z_tri, h_tri_sum, h_tri_mean = model.forward_pcr(
                mono_batch.x, mono_batch.edge_index, mono_batch.edge_attr, mono_batch.batch,
                tri_batch.x, tri_batch.edge_index, tri_batch.edge_attr, tri_batch.batch,
            )

            sup_loss = criterion(pred, targets)
            con_loss = info_nce_loss(z_mono, z_tri, temperature=TAU)
            loss = sup_loss + current_alpha * con_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_sup_loss += sup_loss.item() * len(targets)
            epoch_con_loss += con_loss.item() * len(targets)
            n_samples += len(targets)
            epoch_cos_sims.append(cosine_similarity_pairs(z_mono.detach(), z_tri.detach()))

        scheduler.step()

        train_rmse = np.sqrt(epoch_sup_loss / n_samples)
        train_rmses.append(train_rmse)
        sup_losses.append(epoch_sup_loss / n_samples)
        con_losses.append(epoch_con_loss / n_samples)
        cos_sims.append(np.mean(epoch_cos_sims))

        # Diagnostic: alignment and uniformity (on train set, every 10 epochs)
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                diag_cos = []
                diag_align = []
                diag_unif = []
                for mono_batch, tri_batch, targets in loader:
                    mono_batch = mono_batch.to(device)
                    tri_batch = tri_batch.to(device)
                    _, z_mono, z_tri, h_tri_sum, h_tri_mean = model.forward_pcr(
                        mono_batch.x, mono_batch.edge_index, mono_batch.edge_attr, mono_batch.batch,
                        tri_batch.x, tri_batch.edge_index, tri_batch.edge_attr, tri_batch.batch,
                    )
                    diag_cos.append(cosine_similarity_pairs(z_mono, z_tri))
                    diag_align.append(alignment_metric(z_mono, z_tri))
                    diag_unif.append(uniformity_metric(z_tri))
                alignments.append(np.mean(diag_align))
                uniformities.append(np.mean(diag_unif))
                logger.info(f"  Diag: cos_sim={np.mean(diag_cos):.4f}, "
                            f"align={np.mean(diag_align):.4f}, unif={np.mean(diag_unif):.4f}")

        # Validate (use forward_baseline for fair comparison — same sum_pool path)
        model.eval()
        val_loss = 0
        val_n = 0
        with torch.no_grad():
            for mono_batch, tri_batch, targets in val_loader:
                tri_batch = tri_batch.to(device)
                targets = targets.to(device)
                pred, _ = model.forward_baseline(
                    tri_batch.x, tri_batch.edge_index, tri_batch.edge_attr, tri_batch.batch)
                loss = criterion(pred, targets)
                val_loss += loss.item() * len(targets)
                val_n += len(targets)

        val_rmse = np.sqrt(val_loss / val_n)
        val_rmses.append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch}: train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}, "
                        f"cos_sim={cos_sims[-1]:.4f}, alpha_eff={current_alpha:.4f}")

        if patience_counter >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Test
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    test_loss = 0
    test_n = 0
    with torch.no_grad():
        for mono_batch, tri_batch, targets in test_loader:
            tri_batch = tri_batch.to(device)
            targets = targets.to(device)
            pred, _ = model.forward_baseline(
                tri_batch.x, tri_batch.edge_index, tri_batch.edge_attr, tri_batch.batch)
            loss = criterion(pred, targets)
            test_loss += loss.item() * len(targets)
            test_n += len(targets)

    test_rmse = np.sqrt(test_loss / test_n)
    best_train_rmse = train_rmses[best_epoch - 1]

    logger.info(f"{exp_name}: train={best_train_rmse:.4f}, val={best_val_rmse:.4f}, "
                f"test={test_rmse:.4f}, best_epoch={best_epoch}")

    return {
        'exp_name': exp_name,
        'alpha': alpha,
        'test_rmse': test_rmse,
        'val_rmse': best_val_rmse,
        'train_rmse': best_train_rmse,
        'best_epoch': best_epoch,
        'total_epochs': len(train_rmses),
        'n_params': n_params,
        'overfitting_gap': test_rmse - best_train_rmse,
        'train_rmses': train_rmses,
        'val_rmses': val_rmses,
        'sup_losses': sup_losses,
        'con_losses': con_losses,
        'cos_sims': cos_sims,
        'alignments': alignments,
        'uniformities': uniformities,
        'final_cos_sim': cos_sims[best_epoch - 1] if cos_sims else 0,
    }


def generate_plots(all_results):
    """Generate diagnostic plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for r in all_results:
        if 'train_rmses' in r and r['train_rmses']:
            label = r['exp_name']
            axes[0].plot(r['train_rmses'], label=f"{label} (train)", alpha=0.7)
            axes[0].plot(r['val_rmses'], label=f"{label} (val)", alpha=0.7, linestyle='--')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('Training/Validation RMSE Curves')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 2. Test RMSE comparison bar chart
    names = [r['exp_name'] for r in all_results]
    test_rmses = [r['test_rmse'] for r in all_results]
    gaps = [r['overfitting_gap'] for r in all_results]

    x = np.arange(len(names))
    width = 0.35

    axes[1].bar(x - width/2, test_rmses, width, label='Test RMSE', color='steelblue')
    axes[1].bar(x + width/2, gaps, width, label='Overfitting Gap', color='coral')
    axes[1].axhline(y=50.19, color='red', linestyle='--', alpha=0.7, label='Day 1 ref (50.19)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('Value')
    axes[1].set_title('Test RMSE and Overfitting Gap')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Cosine similarity evolution
    pcr_results = [r for r in all_results if 'cos_sims' in r and r['cos_sims']]
    if pcr_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        for r in pcr_results:
            ax.plot(r['cos_sims'], label=r['exp_name'], alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cosine Similarity (mean_pooled projected)')
        ax.set_title('Monomer-Trimer Cosine Similarity (mean_pool projection)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(PLOTS_DIR, 'cosine_similarity.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Loss decomposition
    pcr_with_losses = [r for r in all_results if 'sup_losses' in r and r['sup_losses']]
    if pcr_with_losses:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for r in pcr_with_losses:
            axes[0].plot(r['sup_losses'], label=r['exp_name'], alpha=0.8)
            axes[1].plot(r['con_losses'], label=r['exp_name'], alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].set_title('Supervised Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('InfoNCE Loss')
        axes[1].set_title('Contrastive Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'loss_decomposition.png'), dpi=150, bbox_inches='tight')
        plt.close()

    logger.info(f"Diagnostic plots saved to {PLOTS_DIR}/")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = PCRDualViewDataset(root=DATA_DIR, csv_path=CSV_PATH)

    train_idx, val_idx, test_idx = get_data_splits(len(dataset), seed=SEED)
    logger.info(f"Data split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    all_results = []

    # ---- Experiment 1: Baseline (sum_pool) ----
    result = train_baseline(dataset, train_idx, val_idx, test_idx, device)
    all_results.append(result)

    # ---- Experiment 2: PCR alpha=0.01 ----
    result = train_pcr(dataset, train_idx, val_idx, test_idx, device,
                       alpha=0.01, exp_name='pcr_hybrid_a0.01')
    all_results.append(result)

    # ---- Experiment 3: PCR alpha=0.05 ----
    result = train_pcr(dataset, train_idx, val_idx, test_idx, device,
                       alpha=0.05, exp_name='pcr_hybrid_a0.05')
    all_results.append(result)

    # ---- Experiment 4: PCR alpha=0.1 ----
    result = train_pcr(dataset, train_idx, val_idx, test_idx, device,
                       alpha=0.1, exp_name='pcr_hybrid_a0.1')
    all_results.append(result)

    # Generate plots
    generate_plots(all_results)

    # Save results
    results_to_save = []
    for r in all_results:
        r_copy = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                r_copy[k] = v.tolist()
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (np.floating, float)):
                r_copy[k] = [float(x) for x in v]
            elif isinstance(v, (np.integer,)):
                r_copy[k] = int(v)
            else:
                r_copy[k] = v
        results_to_save.append(r_copy)

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    logger.info(f"Results saved to {RESULTS_PATH}")

    # Print summary
    print("\n" + "=" * 120)
    print("PCR-v2 ITERATION 2 RESULTS (Hybrid Pooling)")
    print("=" * 120)
    print(f"{'Experiment':<25} {'Test RMSE':<12} {'Val RMSE':<12} {'Train RMSE':<12} "
          f"{'Overfit Gap':<12} {'Best Epoch':<12} {'Params':<12} {'Final CosSim':<12}")
    print("-" * 120)

    print(f"{'Day1 GIN Trimer (ref)':<25} {'50.19':<12} {'39.65':<12} {'41.29':<12} "
          f"{'8.90':<12} {'15':<12} {'~445K':<12} {'N/A':<12}")
    print(f"{'Iter1 mean_pool bl (ref)':<25} {'52.68':<12} {'39.68':<12} {'45.27':<12} "
          f"{'7.41':<12} {'26':<12} {'681K':<12} {'N/A':<12}")

    for r in all_results:
        cos_sim_str = f"{r.get('final_cos_sim', 0):.4f}" if 'final_cos_sim' in r else 'N/A'
        print(f"{r['exp_name']:<25} {r['test_rmse']:<12.4f} {r['val_rmse']:<12.4f} "
              f"{r['train_rmse']:<12.4f} {r['overfitting_gap']:<12.4f} "
              f"{r['best_epoch']:<12} {r['n_params']:<12,} {cos_sim_str:<12}")

    # Analysis
    print("\n" + "=" * 120)
    print("ANALYSIS")
    print("=" * 120)

    baseline = all_results[0]
    print(f"\nBaseline validation (sum_pool should match ~50.19):")
    print(f"  Baseline test RMSE: {baseline['test_rmse']:.4f} (Day 1 ref: 50.19)")
    print(f"  Delta from Day 1: {baseline['test_rmse'] - 50.19:+.4f}")

    pcr_results = all_results[1:]
    best_pcr = min(pcr_results, key=lambda x: x['test_rmse'])

    print(f"\nPCR contrastive results:")
    for r in pcr_results:
        delta = r['test_rmse'] - baseline['test_rmse']
        print(f"  {r['exp_name']}: test={r['test_rmse']:.4f} "
              f"(delta from baseline: {delta:+.4f}), "
              f"gap={r['overfitting_gap']:.4f}, cos_sim={r.get('final_cos_sim', 0):.4f}")

    if best_pcr['test_rmse'] < baseline['test_rmse']:
        print(f"\n  BEST PCR ({best_pcr['exp_name']}) improves over baseline by "
              f"{baseline['test_rmse'] - best_pcr['test_rmse']:.4f}")
    else:
        print(f"\n  No PCR variant improves over baseline")

    if best_pcr['test_rmse'] < 50.19:
        print(f"  PCR BEATS Day 1 reference by {50.19 - best_pcr['test_rmse']:.4f}")
    else:
        print(f"  PCR does NOT beat Day 1 reference (gap: {best_pcr['test_rmse'] - 50.19:.4f})")

    return all_results


if __name__ == "__main__":
    results = main()
