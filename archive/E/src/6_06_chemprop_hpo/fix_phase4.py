"""
Re-run Phase 4 multi-seed with correct batch_size=16 (matching Phase 3 best config).
Also test combined_best (no dense) with multi-seed for fair comparison.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chemprop_hpo import (
    load_graphs, train_model, WORK_DIR
)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    data = load_graphs()

    # Best dense config: dense_d5_h1200 (depth=5, hidden=1200, lr=0.001, dropout=0.1)
    # Phase 3 best used bs=16, Phase 4 mistakenly used bs=50
    seeds = [42, 71, 123]
    results = []

    print("\n=== Re-running Phase 4: dense_d5_h1200 with bs=16 ===")
    for seed in seeds:
        name = f"v2_dense_d5_h1200_seed{seed}"
        r = train_model(data, name, depth=5, hidden_dim=1200, lr=0.001, dropout=0.1,
                        dense_connections=True, seed=seed, epochs=200, patience=30,
                        batch_size=16, device=device)
        results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(WORK_DIR, 'results_phase4_v2.csv'), index=False)

    test_rmses = [r['test_rmse'] for r in results]
    train_rmses = [r['train_rmse'] for r in results]
    gaps = [r['overfitting_gap'] for r in results]

    print(f"\n=== Dense d5 h1200 Multi-Seed (bs=16) ===")
    for r in results:
        print(f"  Seed {r['seed']}: test={r['test_rmse']:.4f}, train={r['train_rmse']:.4f}, gap={r['overfitting_gap']:.4f}")
    print(f"  Test RMSE: {np.mean(test_rmses):.2f} ± {np.std(test_rmses):.2f}")
    print(f"  Train RMSE: {np.mean(train_rmses):.2f} ± {np.std(train_rmses):.2f}")
    print(f"  Overfitting gap: {np.mean(gaps):.2f} ± {np.std(gaps):.2f}")

    # Also test combined_best (no dense) with multi-seed for fair comparison
    print("\n=== Multi-seed: combined_best (no dense) d7 h1200 lr=0.001 dropout=0.1 ===")
    results_nodense = []
    for seed in seeds:
        name = f"v2_nodense_d7_h1200_seed{seed}"
        r = train_model(data, name, depth=7, hidden_dim=1200, lr=0.001, dropout=0.1,
                        dense_connections=False, seed=seed, epochs=200, patience=30,
                        batch_size=25, device=device)
        results_nodense.append(r)

    df2 = pd.DataFrame(results_nodense)
    df2.to_csv(os.path.join(WORK_DIR, 'results_phase4_nodense.csv'), index=False)

    test_rmses2 = [r['test_rmse'] for r in results_nodense]
    print(f"\n=== No-dense d7 h1200 Multi-Seed ===")
    for r in results_nodense:
        print(f"  Seed {r['seed']}: test={r['test_rmse']:.4f}, train={r['train_rmse']:.4f}, gap={r['overfitting_gap']:.4f}")
    print(f"  Test RMSE: {np.mean(test_rmses2):.2f} ± {np.std(test_rmses2):.2f}")

    print("\nDONE.")


if __name__ == '__main__':
    main()
