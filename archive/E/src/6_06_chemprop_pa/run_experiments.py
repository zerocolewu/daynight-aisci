"""
Run all Chemprop-Polymer-K experiments sequentially.

Experiments:
  Variant A: No pre-training baseline (3 LRs x 3 seeds)
  Variant B: PI1M contrastive pre-training + K-node
  Variant C: Random graph pre-training control
  Variant D: PI1M pre-training WITHOUT K-node

Then fine-tuning for B, C, D (3 LRs each).
"""
import sys
import os
import pickle
import time
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from finetune import finetune
from pretrain import pretrain

# Configuration
HIDDEN_DIM = 1200
DEPTH = 7
DROPOUT = 0.0
SEEDS = (71, 42, 123)
SPLIT_SEED = 71
EPOCHS_FINETUNE = 300
PATIENCE = 30
BATCH_SIZE = 64
PRETRAIN_EPOCHS = 50
PRETRAIN_BATCH = 32
PRETRAIN_LR = 1e-3
TEMPERATURE = 0.07
FINETUNE_LRS = [1e-4, 5e-4, 1e-3]
DEVICE = 'cuda'

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_results(results, name):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {path}")


def load_results(name):
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load Tg data
    print("Loading Tg data...")
    tg_data_k = load_data(os.path.join(RESULTS_DIR, 'tg_graphs.pkl'))
    tg_data_no_k = load_data(os.path.join(RESULTS_DIR, 'tg_graphs_no_k.pkl'))
    print(f"  With K-node: {len(tg_data_k)} graphs")
    print(f"  Without K-node: {len(tg_data_no_k)} graphs")

    all_results = {}

    # ============================================================
    # Variant A: No pre-training baseline (3 LRs x 3 seeds)
    # ============================================================
    print("\n" + "="*60)
    print("VARIANT A: No pre-training baseline")
    print("="*60)

    for lr in FINETUNE_LRS:
        result_name = f'variant_A_lr{lr:.0e}.pkl'
        existing = load_results(result_name)
        if existing is not None and 'summary' in existing:
            print(f"  Skipping {result_name} (already done: {existing['summary']['mean_test_rmse']:.2f})")
            all_results[f'A_lr{lr:.0e}'] = existing
            continue

        print(f"\n--- Variant A, LR={lr} ---")
        results = finetune(
            pretrain_checkpoint_path=None,
            tg_data=tg_data_k,
            use_knode=True,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH,
            dropout=DROPOUT,
            lr=lr,
            seeds=SEEDS,
            split_seed=SPLIT_SEED,
            epochs=EPOCHS_FINETUNE,
            patience=PATIENCE,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            variant_name=f'A_baseline_lr{lr:.0e}',
        )
        save_results(results, result_name)
        all_results[f'A_lr{lr:.0e}'] = results
        torch.cuda.empty_cache()

    # ============================================================
    # Variant B: PI1M contrastive pre-training + K-node
    # ============================================================
    print("\n" + "="*60)
    print("VARIANT B: PI1M pre-training + K-node")
    print("="*60)

    pretrain_ckpt_B = os.path.join(RESULTS_DIR, 'pretrain_pi1m_knode.pt')
    if not os.path.exists(pretrain_ckpt_B):
        print("Pre-training on PI1M with K-node...")
        history_B = pretrain(
            data_path=os.path.join(RESULTS_DIR, 'pi1m_graphs.pkl'),
            checkpoint_path=pretrain_ckpt_B,
            use_knode=True,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH,
            dropout=DROPOUT,
            batch_size=PRETRAIN_BATCH,
            epochs=PRETRAIN_EPOCHS,
            lr=PRETRAIN_LR,
            temperature=TEMPERATURE,
            device=DEVICE,
            variant_name='B_pi1m_knode',
        )
        save_results(history_B, 'pretrain_history_B.pkl')
        torch.cuda.empty_cache()
    else:
        print(f"  Pre-training checkpoint exists: {pretrain_ckpt_B}")

    # Fine-tune Variant B (3 LRs)
    for lr in FINETUNE_LRS:
        result_name = f'variant_B_lr{lr:.0e}.pkl'
        existing = load_results(result_name)
        if existing is not None and 'summary' in existing:
            print(f"  Skipping {result_name} (already done)")
            all_results[f'B_lr{lr:.0e}'] = existing
            continue

        print(f"\n--- Variant B fine-tune, LR={lr} ---")
        results = finetune(
            pretrain_checkpoint_path=pretrain_ckpt_B,
            tg_data=tg_data_k,
            use_knode=True,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH,
            dropout=DROPOUT,
            lr=lr,
            seeds=SEEDS,
            split_seed=SPLIT_SEED,
            epochs=EPOCHS_FINETUNE,
            patience=PATIENCE,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            variant_name=f'B_pi1m_knode_lr{lr:.0e}',
        )
        save_results(results, result_name)
        all_results[f'B_lr{lr:.0e}'] = results
        torch.cuda.empty_cache()

    # ============================================================
    # Variant C: Random graph pre-training control
    # ============================================================
    print("\n" + "="*60)
    print("VARIANT C: Random graph pre-training")
    print("="*60)

    random_graph_path = os.path.join(RESULTS_DIR, 'random_graphs.pkl')
    pretrain_ckpt_C = os.path.join(RESULTS_DIR, 'pretrain_random_knode.pt')

    if os.path.exists(random_graph_path) and not os.path.exists(pretrain_ckpt_C):
        print("Pre-training on random graphs with K-node...")
        history_C = pretrain(
            data_path=random_graph_path,
            checkpoint_path=pretrain_ckpt_C,
            use_knode=True,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH,
            dropout=DROPOUT,
            batch_size=PRETRAIN_BATCH,
            epochs=PRETRAIN_EPOCHS,
            lr=PRETRAIN_LR,
            temperature=TEMPERATURE,
            device=DEVICE,
            variant_name='C_random_knode',
        )
        save_results(history_C, 'pretrain_history_C.pkl')
        torch.cuda.empty_cache()
    elif os.path.exists(pretrain_ckpt_C):
        print(f"  Pre-training checkpoint exists: {pretrain_ckpt_C}")
    else:
        print(f"  Random graphs not found at {random_graph_path}, skipping Variant C")

    # Fine-tune Variant C
    if os.path.exists(pretrain_ckpt_C):
        for lr in FINETUNE_LRS:
            result_name = f'variant_C_lr{lr:.0e}.pkl'
            existing = load_results(result_name)
            if existing is not None and 'summary' in existing:
                all_results[f'C_lr{lr:.0e}'] = existing
                continue

            print(f"\n--- Variant C fine-tune, LR={lr} ---")
            results = finetune(
                pretrain_checkpoint_path=pretrain_ckpt_C,
                tg_data=tg_data_k,
                use_knode=True,
                hidden_dim=HIDDEN_DIM,
                depth=DEPTH,
                dropout=DROPOUT,
                lr=lr,
                seeds=SEEDS,
                split_seed=SPLIT_SEED,
                epochs=EPOCHS_FINETUNE,
                patience=PATIENCE,
                batch_size=BATCH_SIZE,
                device=DEVICE,
                variant_name=f'C_random_knode_lr{lr:.0e}',
            )
            save_results(results, result_name)
            all_results[f'C_lr{lr:.0e}'] = results
            torch.cuda.empty_cache()

    # ============================================================
    # Variant D: PI1M pre-training WITHOUT K-node
    # ============================================================
    print("\n" + "="*60)
    print("VARIANT D: PI1M pre-training WITHOUT K-node")
    print("="*60)

    pretrain_ckpt_D = os.path.join(RESULTS_DIR, 'pretrain_pi1m_nok.pt')
    if not os.path.exists(pretrain_ckpt_D):
        print("Pre-training on PI1M without K-node...")
        history_D = pretrain(
            data_path=os.path.join(RESULTS_DIR, 'pi1m_graphs_no_k.pkl'),
            checkpoint_path=pretrain_ckpt_D,
            use_knode=False,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH,
            dropout=DROPOUT,
            batch_size=PRETRAIN_BATCH,
            epochs=PRETRAIN_EPOCHS,
            lr=PRETRAIN_LR,
            temperature=TEMPERATURE,
            device=DEVICE,
            variant_name='D_pi1m_nok',
        )
        save_results(history_D, 'pretrain_history_D.pkl')
        torch.cuda.empty_cache()
    else:
        print(f"  Pre-training checkpoint exists: {pretrain_ckpt_D}")

    # Fine-tune Variant D
    for lr in FINETUNE_LRS:
        result_name = f'variant_D_lr{lr:.0e}.pkl'
        existing = load_results(result_name)
        if existing is not None and 'summary' in existing:
            all_results[f'D_lr{lr:.0e}'] = existing
            continue

        print(f"\n--- Variant D fine-tune, LR={lr} ---")
        results = finetune(
            pretrain_checkpoint_path=pretrain_ckpt_D,
            tg_data=tg_data_no_k,
            use_knode=False,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH,
            dropout=DROPOUT,
            lr=lr,
            seeds=SEEDS,
            split_seed=SPLIT_SEED,
            epochs=EPOCHS_FINETUNE,
            patience=PATIENCE,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            variant_name=f'D_pi1m_nok_lr{lr:.0e}',
        )
        save_results(results, result_name)
        all_results[f'D_lr{lr:.0e}'] = results
        torch.cuda.empty_cache()

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    for variant in ['A', 'B', 'C', 'D']:
        print(f"\n--- Variant {variant} ---")
        best_rmse = float('inf')
        best_lr = None
        for lr in FINETUNE_LRS:
            key = f'{variant}_lr{lr:.0e}'
            if key in all_results and 'summary' in all_results[key]:
                summary = all_results[key]['summary']
                mean = summary['mean_test_rmse']
                std = summary['std_test_rmse']
                gap = summary['mean_overfitting_gap']
                print(f"  LR={lr:.0e}: Test RMSE = {mean:.2f} ± {std:.2f} K, Overfitting gap = {gap:.2f} K")
                if mean < best_rmse:
                    best_rmse = mean
                    best_lr = lr
        if best_lr:
            print(f"  Best: LR={best_lr:.0e}, RMSE = {best_rmse:.2f} K")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, 'all_results.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nAll results saved to: {summary_path}")


if __name__ == '__main__':
    main()
