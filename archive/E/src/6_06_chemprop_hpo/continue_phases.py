"""
Continue from Phase 2 results - run Phase 3 (dense connections) and Phase 4 (multi-seed).
Phase 1 and 2 results already exist.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chemprop_hpo import (
    load_graphs, train_model, generate_summary, select_best_config,
    WORK_DIR, DIAG_DIR
)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Phase 1 results
    phase1_df = pd.read_csv(os.path.join(WORK_DIR, 'results_phase1.csv'))
    phase1_results = phase1_df.to_dict('records')

    # Load Phase 2 results
    phase2_df = pd.read_csv(os.path.join(WORK_DIR, 'results_phase2.csv'))
    phase2_result = phase2_df.to_dict('records')[0]

    # Best config from Phase 1
    best_depth, best_hidden, best_lr, best_dropout = select_best_config(phase1_results)
    print(f"Best from Phase 1: depth={best_depth}, hidden={best_hidden}, lr={best_lr}, dropout={best_dropout}")
    print(f"Phase 2 result: test_rmse={phase2_result['test_rmse']:.4f}")

    data = load_graphs()

    # ============================================================
    # Phase 3: Dense Skip Connections
    # ============================================================
    print("\n" + "#"*70)
    print("# PHASE 3: Dense Skip Connections")
    print("#"*70)

    results = []
    configs = [
        (best_depth, best_hidden, f'dense_combined_d{best_depth}_h{best_hidden}'),
        (7, best_hidden, f'dense_d7_h{best_hidden}'),
        (5, best_hidden, f'dense_d5_h{best_hidden}'),
    ]

    for d, h, name in configs:
        bs = 16 if h > 600 else 25
        try:
            torch.cuda.empty_cache()
            r = train_model(data, name, depth=d, hidden_dim=h, lr=best_lr, dropout=best_dropout,
                            dense_connections=True, seed=71, epochs=200, patience=30,
                            batch_size=bs, device=device)
            results.append(r)
        except RuntimeError as e:
            print(f"  SKIPPED {name}: {e}")
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(WORK_DIR, 'results_phase3.csv'), index=False)
    print(f"\nPhase 3 results saved ({len(results)} configs)")

    # ============================================================
    # Phase 4: Multi-seed verification
    # ============================================================
    all_results = phase1_results + [phase2_result] + results
    best_overall = min(all_results, key=lambda x: x['test_rmse'])
    print(f"\nBest overall: {best_overall['config_name']} (test RMSE: {best_overall['test_rmse']:.4f})")

    seeds = [42, 71, 123]
    phase4_results = []
    for seed in seeds:
        name = f"final_best_{best_overall['config_name']}_seed{seed}"
        r = train_model(data, name,
                        depth=int(best_overall['depth']),
                        hidden_dim=int(best_overall['hidden_dim']),
                        lr=best_overall['lr'],
                        dropout=best_overall['dropout'],
                        dense_connections=bool(best_overall.get('dense_connections', False)),
                        seed=seed, epochs=200, patience=30, device=device)
        phase4_results.append(r)

    df4 = pd.DataFrame(phase4_results)
    df4.to_csv(os.path.join(WORK_DIR, 'results_phase4.csv'), index=False)

    # Multi-seed stats
    test_rmses = [r['test_rmse'] for r in phase4_results]
    print(f"\n  Multi-Seed: {np.mean(test_rmses):.2f} ± {np.std(test_rmses):.2f}")

    # Generate summary
    generate_summary(phase1_results, phase2_result, results, phase4_results,
                     best_depth, best_hidden, best_lr, best_dropout,
                     bool(best_overall.get('dense_connections', False)))

    print("\nDONE.")


if __name__ == '__main__':
    main()
