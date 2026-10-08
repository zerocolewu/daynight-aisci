"""
Multi-Seed Verification using the ORIGINAL index-based split.

This version uses the same fixed train/val/test split (from tg_train.pt,
tg_val.pt, tg_test.pt) for all seeds. Only the model initialization and
training randomness (shuffling, dropout, etc.) varies with seed.

This is a more conservative test: same data split, different training randomness.
If pre-training is beneficial, it should help regardless of training randomness.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PerioGTFineTuneModel, count_parameters

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 71, 123, 456, 789]
PRETRAINED_ENCODER = 'pretrained_encoder_edge_drop_feat_mask_50.pt'


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))


def evaluate(model, loader, device):
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
    mode='baseline',
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
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = PerioGTFineTuneModel(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    if mode == 'pretrained':
        encoder_path = os.path.join(OUTPUT_DIR, PRETRAINED_ENCODER)
        model.load_pretrained_encoder(encoder_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_rmse = np.sqrt(train_loss / n_batches)
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

    model.load_state_dict(best_state)
    model = model.to(device)
    train_rmse_final, _, _ = evaluate(model, train_loader, device)
    val_rmse_final, _, _ = evaluate(model, val_loader, device)
    test_rmse, _, _ = evaluate(model, test_loader, device)

    return {
        'seed': seed,
        'mode': mode,
        'train_rmse': float(train_rmse_final),
        'val_rmse': float(val_rmse_final),
        'test_rmse': float(test_rmse),
        'best_epoch': best_epoch,
        'total_epochs': epoch,
    }


def main():
    print("=" * 70)
    print("Multi-Seed Verification (FIXED INDEX-BASED SPLIT)")
    print("=" * 70)
    print(f"Seeds: {SEEDS}")
    print(f"Pre-trained encoder: {PRETRAINED_ENCODER}")
    print(f"Split: ORIGINAL index-based (same for all seeds)")
    print(f"Note: Only training randomness varies with seed")
    print("=" * 70)

    # Load the ORIGINAL fixed split
    train_data = torch.load(os.path.join(OUTPUT_DIR, 'tg_train.pt'), weights_only=False)
    val_data = torch.load(os.path.join(OUTPUT_DIR, 'tg_val.pt'), weights_only=False)
    test_data = torch.load(os.path.join(OUTPUT_DIR, 'tg_test.pt'), weights_only=False)
    print(f"Data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    all_results = {}

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        # Baseline
        print(f"\n--- Baseline (seed={seed}) ---")
        t0 = time.time()
        baseline = run_finetune(mode='baseline', seed=seed,
                                train_data=train_data, val_data=val_data, test_data=test_data)
        baseline['time_sec'] = time.time() - t0
        print(f"  => Test RMSE: {baseline['test_rmse']:.4f} (best epoch {baseline['best_epoch']})")

        # Pre-trained
        print(f"\n--- Pre-trained (seed={seed}) ---")
        t0 = time.time()
        pretrained = run_finetune(mode='pretrained', seed=seed,
                                  train_data=train_data, val_data=val_data, test_data=test_data)
        pretrained['time_sec'] = time.time() - t0
        print(f"  => Test RMSE: {pretrained['test_rmse']:.4f} (best epoch {pretrained['best_epoch']})")

        diff = baseline['test_rmse'] - pretrained['test_rmse']
        winner = 'Pretrained' if diff > 0 else 'Baseline'
        print(f"\n  Diff: {diff:+.4f} => {winner} wins")

        all_results[seed] = {
            'baseline': baseline,
            'pretrained': pretrained,
            'diff': diff,
        }

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY (Fixed Split)")
    print(f"{'='*70}")

    baseline_rmses = [all_results[s]['baseline']['test_rmse'] for s in SEEDS]
    pretrained_rmses = [all_results[s]['pretrained']['test_rmse'] for s in SEEDS]
    diffs = [all_results[s]['diff'] for s in SEEDS]

    print(f"\n{'Seed':>6} | {'Baseline':>10} | {'Pretrained':>10} | {'Diff':>8} | {'Winner':>10}")
    print("-" * 60)
    for seed in SEEDS:
        r = all_results[seed]
        print(f"{seed:>6} | {r['baseline']['test_rmse']:>10.4f} | {r['pretrained']['test_rmse']:>10.4f} | {r['diff']:>+8.4f} | {'Pretrained' if r['diff'] > 0 else 'Baseline':>10}")

    print("-" * 60)
    b_mean, b_std = np.mean(baseline_rmses), np.std(baseline_rmses)
    p_mean, p_std = np.mean(pretrained_rmses), np.std(pretrained_rmses)
    d_mean, d_std = np.mean(diffs), np.std(diffs)
    print(f"{'Mean':>6} | {b_mean:>10.4f} | {p_mean:>10.4f} | {d_mean:>+8.4f} |")
    print(f"{'Std':>6} | {b_std:>10.4f} | {p_std:>10.4f} | {d_std:>8.4f} |")

    pretrained_wins = sum(1 for d in diffs if d > 0)
    print(f"\nPretrained wins: {pretrained_wins}/{len(SEEDS)}")

    from scipy import stats as scipy_stats
    if len(diffs) >= 3:
        t_stat, p_value = scipy_stats.ttest_1samp(diffs, 0)
        print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
        if p_value < 0.05:
            print("  => Statistically significant (p < 0.05)")
        else:
            print("  => NOT statistically significant (p >= 0.05)")

    improvement_pct = (b_mean - p_mean) / b_mean * 100
    print(f"\nBaseline:    {b_mean:.4f} +/- {b_std:.4f}")
    print(f"Pretrained:  {p_mean:.4f} +/- {p_std:.4f}")
    print(f"Improvement: {improvement_pct:+.2f}%")

    # Save
    results_path = os.path.join(OUTPUT_DIR, 'multiseed_fixedsplit_results.json')
    save_data = {
        'seeds': SEEDS,
        'encoder': PRETRAINED_ENCODER,
        'split_type': 'fixed_index_based',
        'per_seed_results': {str(s): all_results[s] for s in SEEDS},
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
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
