"""
Multi-Seed Verification of 50% Corruption Pre-training Result.

This script verifies whether the 49.83 RMSE achieved with 50% corruption
pre-training is robust across multiple random seeds, or just a statistical fluke.

Plan:
- Seeds: [42, 71, 123, 456, 789]
- For each seed:
  1. Load all Tg graphs (preprocessed), re-split with seed-based random shuffle (72/8/20)
  2. Run baseline (random init) fine-tuning
  3. Run pre-trained fine-tuning (using the existing 50% corruption encoder)
- Compare per-seed and aggregate results
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PerioGTFineTuneModel, count_parameters

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 71, 123, 456, 789]
PRETRAINED_ENCODER = 'pretrained_encoder_edge_drop_feat_mask_50.pt'


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_and_split_tg_data(seed):
    """Load preprocessed Tg graphs and split with given seed.

    Instead of the index-based split in preprocess_tg.py, we use
    a random permutation with the given seed for proper multi-seed evaluation.
    """
    # Load all pre-split data (train + val + test)
    train_data = torch.load(os.path.join(OUTPUT_DIR, 'tg_train.pt'), weights_only=False)
    val_data = torch.load(os.path.join(OUTPUT_DIR, 'tg_val.pt'), weights_only=False)
    test_data = torch.load(os.path.join(OUTPUT_DIR, 'tg_test.pt'), weights_only=False)

    # Combine all data
    all_data = train_data + val_data + test_data
    n = len(all_data)
    print(f"  Total Tg samples: {n}")

    # Random permutation with seed
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    # Split 72/8/20
    n_train = int(n * 0.72)
    n_val = int(n * 0.08)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    seed_train = [all_data[i] for i in train_idx]
    seed_val = [all_data[i] for i in val_idx]
    seed_test = [all_data[i] for i in test_idx]

    print(f"  Split (seed={seed}): train={len(seed_train)}, val={len(seed_val)}, test={len(seed_test)}")

    return seed_train, seed_val, seed_test


def compute_rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))


def evaluate(model, loader, device):
    """Evaluate model and return RMSE and predictions."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    rmse = compute_rmse(all_preds, all_targets)
    return rmse, all_preds, all_targets


def run_finetune(
    mode='baseline',  # 'baseline' or 'pretrained'
    seed=71,
    train_data=None,
    val_data=None,
    test_data=None,
    lr=1e-3,
    weight_decay=0.0,
    batch_size=32,
    max_epochs=300,
    patience=30,
    hidden_dim=256,
    num_layers=3,
):
    """Run a single fine-tuning experiment with given seed and data split."""

    # Set seed for model initialization and training
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Create model
    model = PerioGTFineTuneModel(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    if mode == 'pretrained':
        encoder_path = os.path.join(OUTPUT_DIR, PRETRAINED_ENCODER)
        if os.path.exists(encoder_path):
            model.load_pretrained_encoder(encoder_path)
        else:
            raise FileNotFoundError(f"Pre-trained encoder not found at {encoder_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Training loop
    best_val_rmse = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        train_rmse = np.sqrt(train_loss / n_train_batches)
        val_rmse, _, _ = evaluate(model, val_loader, device)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}: train={train_rmse:.4f}, val={val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    # Load best model and evaluate
    model.load_state_dict(best_state)
    model = model.to(device)

    train_rmse_final, _, _ = evaluate(model, train_loader, device)
    val_rmse_final, _, _ = evaluate(model, val_loader, device)
    test_rmse, test_preds, test_targets = evaluate(model, test_loader, device)

    results = {
        'seed': seed,
        'mode': mode,
        'train_rmse': float(train_rmse_final),
        'val_rmse': float(val_rmse_final),
        'test_rmse': float(test_rmse),
        'best_epoch': best_epoch,
        'total_epochs': epoch,
    }

    return results


def main():
    print("=" * 70)
    print("Multi-Seed Verification of 50% Corruption Pre-training")
    print("=" * 70)
    print(f"Seeds: {SEEDS}")
    print(f"Pre-trained encoder: {PRETRAINED_ENCODER}")
    print(f"Split ratio: 72/8/20 (random, seed-dependent)")
    print(f"Architecture: 3-layer GINEConv, hidden=256, mean_pool")
    print(f"Fine-tuning: Adam lr=1e-3, patience=30, batch_size=32")
    print("=" * 70)

    all_results = {}

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        # Load and split data
        print("Loading and splitting data...")
        train_data, val_data, test_data = load_and_split_tg_data(seed)

        # --- Baseline ---
        print(f"\n--- Baseline (seed={seed}) ---")
        t0 = time.time()
        baseline_result = run_finetune(
            mode='baseline',
            seed=seed,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )
        baseline_time = time.time() - t0
        baseline_result['time_sec'] = baseline_time
        print(f"  => Test RMSE: {baseline_result['test_rmse']:.4f} (best epoch {baseline_result['best_epoch']}, {baseline_time:.0f}s)")

        # --- Pre-trained ---
        print(f"\n--- Pre-trained (seed={seed}) ---")
        t0 = time.time()
        pretrained_result = run_finetune(
            mode='pretrained',
            seed=seed,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )
        pretrained_time = time.time() - t0
        pretrained_result['time_sec'] = pretrained_time
        print(f"  => Test RMSE: {pretrained_result['test_rmse']:.4f} (best epoch {pretrained_result['best_epoch']}, {pretrained_time:.0f}s)")

        # Comparison
        diff = baseline_result['test_rmse'] - pretrained_result['test_rmse']
        print(f"\n  Difference (baseline - pretrained): {diff:+.4f}")
        if diff > 0:
            print(f"  Pre-trained WINS by {diff:.4f} RMSE")
        else:
            print(f"  Baseline WINS by {-diff:.4f} RMSE")

        all_results[seed] = {
            'baseline': baseline_result,
            'pretrained': pretrained_result,
            'diff': diff,
        }

    # ========== Summary ==========
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    baseline_rmses = []
    pretrained_rmses = []
    diffs = []

    print(f"\n{'Seed':>6} | {'Baseline':>10} | {'Pretrained':>10} | {'Diff':>8} | {'Winner':>10}")
    print("-" * 60)

    for seed in SEEDS:
        r = all_results[seed]
        b = r['baseline']['test_rmse']
        p = r['pretrained']['test_rmse']
        d = r['diff']
        winner = 'Pretrained' if d > 0 else 'Baseline'
        print(f"{seed:>6} | {b:>10.4f} | {p:>10.4f} | {d:>+8.4f} | {winner:>10}")
        baseline_rmses.append(b)
        pretrained_rmses.append(p)
        diffs.append(d)

    print("-" * 60)
    b_mean = np.mean(baseline_rmses)
    b_std = np.std(baseline_rmses)
    p_mean = np.mean(pretrained_rmses)
    p_std = np.std(pretrained_rmses)
    d_mean = np.mean(diffs)
    d_std = np.std(diffs)

    print(f"{'Mean':>6} | {b_mean:>10.4f} | {p_mean:>10.4f} | {d_mean:>+8.4f} |")
    print(f"{'Std':>6} | {b_std:>10.4f} | {p_std:>10.4f} | {d_std:>8.4f} |")

    # Wins count
    pretrained_wins = sum(1 for d in diffs if d > 0)
    baseline_wins = sum(1 for d in diffs if d <= 0)
    print(f"\nPretrained wins: {pretrained_wins}/{len(SEEDS)}")
    print(f"Baseline wins:   {baseline_wins}/{len(SEEDS)}")

    # Statistical significance (paired t-test)
    from scipy import stats as scipy_stats
    if len(diffs) >= 3:
        t_stat, p_value = scipy_stats.ttest_1samp(diffs, 0)
        print(f"\nPaired t-test: t={t_stat:.4f}, p={p_value:.4f}")
        if p_value < 0.05:
            print("  => Statistically significant (p < 0.05)")
        else:
            print("  => NOT statistically significant (p >= 0.05)")

    # Conclusion
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")

    improvement_pct = (b_mean - p_mean) / b_mean * 100
    print(f"Baseline:    {b_mean:.4f} ± {b_std:.4f}")
    print(f"Pretrained:  {p_mean:.4f} ± {p_std:.4f}")
    print(f"Improvement: {improvement_pct:+.2f}%")

    if pretrained_wins >= 4 and improvement_pct > 1.0:
        print("\nVERDICT: The 50% corruption pre-training benefit is ROBUST across seeds.")
    elif pretrained_wins >= 3 and improvement_pct > 0:
        print("\nVERDICT: The pre-training shows a CONSISTENT but modest benefit.")
        print("  The improvement may not be statistically significant with 5 seeds.")
    else:
        print("\nVERDICT: The 50% corruption result is likely a STATISTICAL FLUCTUATION.")
        print("  Pre-training does not consistently improve over baseline.")

    # Save all results
    results_path = os.path.join(OUTPUT_DIR, 'multiseed_verification_results.json')
    save_data = {
        'seeds': SEEDS,
        'encoder': PRETRAINED_ENCODER,
        'split_ratio': '72/8/20',
        'per_seed_results': {},
        'summary': {
            'baseline_mean': float(b_mean),
            'baseline_std': float(b_std),
            'pretrained_mean': float(p_mean),
            'pretrained_std': float(p_std),
            'improvement_pct': float(improvement_pct),
            'pretrained_wins': pretrained_wins,
            'total_seeds': len(SEEDS),
        },
    }
    for seed in SEEDS:
        r = all_results[seed]
        save_data['per_seed_results'][str(seed)] = r

    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
