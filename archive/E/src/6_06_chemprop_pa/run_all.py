"""
Chemprop-Polymer-K + PI1M Contrastive Pre-training Experiment
Main orchestrator: runs all experimental variants (A, B, C, D) + diagnostics.

Usage:
    python run_all.py                    # Run everything
    python run_all.py --skip-preprocess  # Skip data preprocessing
    python run_all.py --variant A        # Run only variant A
    python run_all.py --variant B        # Run only variant B
    python run_all.py --variant C        # Run only variant C
    python run_all.py --variant D        # Run only variant D
    python run_all.py --variant diag     # Run only diagnostics
"""
import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import ATOM_FDIM, BOND_FDIM, K_FEAT_DIM, process_tg_dataset, process_pi1m_dataset
from model import ChempropPolymerKMPNN, collate_batch
from generate_random import generate_random_graphs

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORK_DIR, 'results')
CHECKPOINTS_DIR = os.path.join(WORK_DIR, 'checkpoints')
CSV_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
PI1M_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# ============================================================
# Configuration
# ============================================================
SEEDS = [71, 42, 123]
SPLIT_SEED = 71

# Model
HIDDEN_DIM = 1200
DEPTH = 7
DROPOUT = 0.0

# Pre-training
PRETRAIN_LR = 1e-4
PRETRAIN_EPOCHS = 50
PRETRAIN_BATCH = 32
TEMPERATURE = 0.07
PI1M_SUBSET = 300000

# Fine-tuning
FINETUNE_LRS = [1e-4, 5e-4, 1e-3]
FINETUNE_EPOCHS = 200
FINETUNE_BATCH = 50
FINETUNE_PATIENCE = 30

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def preprocess_data():
    """Step 1: Preprocess Tg and PI1M datasets."""
    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 70)

    # Tg dataset with K-node
    tg_path = os.path.join(RESULTS_DIR, 'tg_graphs.pkl')
    if os.path.exists(tg_path):
        print(f"  Tg graphs already exist at {tg_path}")
    else:
        print("  Processing Tg dataset with K-node...")
        tg_data = process_tg_dataset(CSV_PATH, graph_type='polymer', use_knode=True)
        with open(tg_path, 'wb') as f:
            pickle.dump(tg_data, f)
        print(f"  Saved {len(tg_data)} Tg graphs")

    # Tg dataset without K-node (for Variant D)
    tg_nok_path = os.path.join(RESULTS_DIR, 'tg_graphs_no_k.pkl')
    if os.path.exists(tg_nok_path):
        print(f"  Tg no-K graphs already exist")
    else:
        print("  Processing Tg dataset without K-node...")
        tg_nok = process_tg_dataset(CSV_PATH, graph_type='polymer', use_knode=False)
        with open(tg_nok_path, 'wb') as f:
            pickle.dump(tg_nok, f)
        print(f"  Saved {len(tg_nok)} Tg no-K graphs")

    # PI1M dataset (300K subset) with K-node
    pi1m_path = os.path.join(RESULTS_DIR, 'pi1m_graphs.pkl')
    if os.path.exists(pi1m_path):
        print(f"  PI1M graphs already exist at {pi1m_path}")
    else:
        print("  Processing PI1M dataset (300K subset) with K-node...")
        pi1m_data = process_pi1m_dataset(PI1M_PATH, max_samples=PI1M_SUBSET,
                                          graph_type='polymer', use_knode=True, seed=42)
        with open(pi1m_path, 'wb') as f:
            pickle.dump(pi1m_data, f)
        print(f"  Saved {len(pi1m_data)} PI1M graphs")

    # PI1M without K-node (for Variant D)
    pi1m_nok_path = os.path.join(RESULTS_DIR, 'pi1m_graphs_no_k.pkl')
    if os.path.exists(pi1m_nok_path):
        print(f"  PI1M no-K graphs already exist")
    else:
        print("  Processing PI1M dataset without K-node...")
        pi1m_nok = process_pi1m_dataset(PI1M_PATH, max_samples=PI1M_SUBSET,
                                         graph_type='polymer', use_knode=False, seed=42)
        with open(pi1m_nok_path, 'wb') as f:
            pickle.dump(pi1m_nok, f)
        print(f"  Saved {len(pi1m_nok)} PI1M no-K graphs")

    # Random graphs (for Variant C)
    random_path = os.path.join(RESULTS_DIR, 'random_graphs.pkl')
    if os.path.exists(random_path):
        print(f"  Random graphs already exist")
    else:
        print("  Generating random graphs (300K)...")
        random_data = generate_random_graphs(
            n_graphs=PI1M_SUBSET,
            atom_fdim=ATOM_FDIM,
            bond_fdim=BOND_FDIM,
            k_feat_dim=K_FEAT_DIM,
            seed=42,
        )
        with open(random_path, 'wb') as f:
            pickle.dump(random_data, f)
        print(f"  Saved {len(random_data)} random graphs")

    print("  Data preprocessing complete.")


def run_diagnostics():
    """Step 2: Run augmentation diagnostics with random encoder."""
    print("\n" + "=" * 70)
    print("STEP 2: AUGMENTATION DIAGNOSTICS")
    print("=" * 70)

    from diagnostics import augmentation_spectrum

    # Load Tg data
    tg_path = os.path.join(RESULTS_DIR, 'tg_graphs.pkl')
    with open(tg_path, 'rb') as f:
        tg_data = pickle.load(f)

    # Use a subset for diagnostics speed
    sample_graphs = tg_data[:200]

    # Create a fresh model for diagnostics (with projector for embeddings)
    encoder = ChempropPolymerKMPNN(
        atom_fdim=ATOM_FDIM, bond_fdim=BOND_FDIM,
        k_feat_dim=K_FEAT_DIM,
        hidden_dim=HIDDEN_DIM, depth=DEPTH, dropout=DROPOUT,
        use_knode=True, with_projector=True,
    ).to(DEVICE)

    results = augmentation_spectrum(
        encoder=encoder,
        sample_graphs=sample_graphs,
        device=DEVICE,
    )

    # Save diagnostic results
    # Convert numpy types to float for JSON serialization
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    diag_path = os.path.join(RESULTS_DIR, 'diagnostics_augmentation.json')
    with open(diag_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"  Augmentation diagnostics saved to {diag_path}")

    # Print summary
    print("\n  Augmentation Spectrum Summary:")
    for key, label in [('bond_drop_rates', 'Bond dropping'),
                       ('feature_mask_rates', 'Feature masking')]:
        if key in results:
            print(f"    {label}:")
            for r in results[key]:
                print(f"      rate={r['rate']}: sim={r['mean_cosine_sim']:.4f}")
    if 'trimer_shift' in results:
        ts = results['trimer_shift']
        print(f"    Trimer-shift: sim={ts.get('mean_cosine_sim', 'N/A')}")


def run_variant_a():
    """Variant A: No pre-training baseline with K-node."""
    print("\n" + "=" * 70)
    print("VARIANT A: NO PRE-TRAINING BASELINE (K-NODE)")
    print("=" * 70)

    from finetune import finetune

    tg_path = os.path.join(RESULTS_DIR, 'tg_graphs.pkl')
    with open(tg_path, 'rb') as f:
        tg_data = pickle.load(f)
    print(f"  Loaded {len(tg_data)} Tg graphs")

    all_results = {}
    for lr in FINETUNE_LRS:
        print(f"\n  --- Variant A, lr={lr} ---")
        results = finetune(
            pretrain_checkpoint_path=None,  # No pre-training
            tg_data=tg_data,
            use_knode=True,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH,
            dropout=DROPOUT,
            lr=lr,
            seeds=tuple(SEEDS),
            split_seed=SPLIT_SEED,
            epochs=FINETUNE_EPOCHS,
            patience=FINETUNE_PATIENCE,
            batch_size=FINETUNE_BATCH,
            device=DEVICE,
            variant_name=f'A_knode_lr{lr}',
        )
        all_results[f'lr_{lr}'] = results

    save_path = os.path.join(RESULTS_DIR, 'variant_a_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\n  Variant A results saved to {save_path}")

    print_variant_summary('A', all_results)
    return all_results


def run_pretraining(variant_name, graph_path, use_knode=True):
    """Run contrastive pre-training for a given variant."""
    print(f"\n  Pre-training for {variant_name}...")

    from pretrain import pretrain as do_pretrain

    ckpt_path = os.path.join(CHECKPOINTS_DIR, f'{variant_name}_pretrained.pt')
    if os.path.exists(ckpt_path):
        print(f"  Pre-trained checkpoint already exists at {ckpt_path}")
        return ckpt_path

    do_pretrain(
        data_path=graph_path,
        checkpoint_path=ckpt_path,
        use_knode=use_knode,
        hidden_dim=HIDDEN_DIM,
        depth=DEPTH,
        dropout=DROPOUT,
        batch_size=PRETRAIN_BATCH,
        epochs=PRETRAIN_EPOCHS,
        lr=PRETRAIN_LR,
        temperature=TEMPERATURE,
        device=DEVICE,
        variant_name=variant_name,
    )
    print(f"  Pre-trained model saved to {ckpt_path}")
    return ckpt_path


def run_variant_b():
    """Variant B: PI1M contrastive pre-training + K-node (PRIMARY)."""
    print("\n" + "=" * 70)
    print("VARIANT B: PI1M PRE-TRAINING + K-NODE (PRIMARY)")
    print("=" * 70)

    from finetune import finetune

    # Pre-train
    pi1m_path = os.path.join(RESULTS_DIR, 'pi1m_graphs.pkl')
    ckpt_path = run_pretraining('B_pi1m_knode', pi1m_path, use_knode=True)

    # Fine-tune
    tg_path = os.path.join(RESULTS_DIR, 'tg_graphs.pkl')
    with open(tg_path, 'rb') as f:
        tg_data = pickle.load(f)
    print(f"  Loaded {len(tg_data)} Tg graphs")

    all_results = {}
    for lr in FINETUNE_LRS:
        print(f"\n  --- Variant B, lr={lr} ---")
        results = finetune(
            pretrain_checkpoint_path=ckpt_path,
            tg_data=tg_data,
            use_knode=True,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH,
            dropout=DROPOUT,
            lr=lr,
            seeds=tuple(SEEDS),
            split_seed=SPLIT_SEED,
            epochs=FINETUNE_EPOCHS,
            patience=FINETUNE_PATIENCE,
            batch_size=FINETUNE_BATCH,
            device=DEVICE,
            variant_name=f'B_pi1m_knode_lr{lr}',
        )
        all_results[f'lr_{lr}'] = results

    save_path = os.path.join(RESULTS_DIR, 'variant_b_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\n  Variant B results saved to {save_path}")
    print_variant_summary('B', all_results)
    return all_results


def run_variant_c():
    """Variant C: Random graph pre-training control."""
    print("\n" + "=" * 70)
    print("VARIANT C: RANDOM GRAPH PRE-TRAINING CONTROL")
    print("=" * 70)

    from finetune import finetune

    # Pre-train on random graphs
    random_path = os.path.join(RESULTS_DIR, 'random_graphs.pkl')
    ckpt_path = run_pretraining('C_random_knode', random_path, use_knode=True)

    # Fine-tune
    tg_path = os.path.join(RESULTS_DIR, 'tg_graphs.pkl')
    with open(tg_path, 'rb') as f:
        tg_data = pickle.load(f)
    print(f"  Loaded {len(tg_data)} Tg graphs")

    all_results = {}
    for lr in FINETUNE_LRS:
        print(f"\n  --- Variant C, lr={lr} ---")
        results = finetune(
            pretrain_checkpoint_path=ckpt_path,
            tg_data=tg_data,
            use_knode=True,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH,
            dropout=DROPOUT,
            lr=lr,
            seeds=tuple(SEEDS),
            split_seed=SPLIT_SEED,
            epochs=FINETUNE_EPOCHS,
            patience=FINETUNE_PATIENCE,
            batch_size=FINETUNE_BATCH,
            device=DEVICE,
            variant_name=f'C_random_knode_lr{lr}',
        )
        all_results[f'lr_{lr}'] = results

    save_path = os.path.join(RESULTS_DIR, 'variant_c_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\n  Variant C results saved to {save_path}")
    print_variant_summary('C', all_results)
    return all_results


def run_variant_d():
    """Variant D: PI1M pre-training WITHOUT K-node."""
    print("\n" + "=" * 70)
    print("VARIANT D: PI1M PRE-TRAINING WITHOUT K-NODE")
    print("=" * 70)

    from finetune import finetune

    # Pre-train (no K-node)
    pi1m_nok_path = os.path.join(RESULTS_DIR, 'pi1m_graphs_no_k.pkl')
    ckpt_path = run_pretraining('D_pi1m_noknode', pi1m_nok_path, use_knode=False)

    # Fine-tune (also no K-node)
    tg_nok_path = os.path.join(RESULTS_DIR, 'tg_graphs_no_k.pkl')
    with open(tg_nok_path, 'rb') as f:
        tg_nok_data = pickle.load(f)
    print(f"  Loaded {len(tg_nok_data)} Tg no-K graphs")

    all_results = {}
    for lr in FINETUNE_LRS:
        print(f"\n  --- Variant D, lr={lr} ---")
        results = finetune(
            pretrain_checkpoint_path=ckpt_path,
            tg_data=tg_nok_data,
            use_knode=False,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH,
            dropout=DROPOUT,
            lr=lr,
            seeds=tuple(SEEDS),
            split_seed=SPLIT_SEED,
            epochs=FINETUNE_EPOCHS,
            patience=FINETUNE_PATIENCE,
            batch_size=FINETUNE_BATCH,
            device=DEVICE,
            variant_name=f'D_pi1m_noknode_lr{lr}',
        )
        all_results[f'lr_{lr}'] = results

    save_path = os.path.join(RESULTS_DIR, 'variant_d_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\n  Variant D results saved to {save_path}")
    print_variant_summary('D', all_results)
    return all_results


def print_variant_summary(variant, all_results):
    """Print summary for a variant across learning rates."""
    print(f"\n  === Variant {variant} Summary ===")
    for lr_key, lr_results in all_results.items():
        if 'summary' in lr_results:
            s = lr_results['summary']
            mean_rmse = s.get('mean_test_rmse', 0)
            std_rmse = s.get('std_test_rmse', 0)
            mean_gap = s.get('mean_overfitting_gap', 0)
            print(f"    {lr_key}: Test RMSE = {mean_rmse:.2f} +/- {std_rmse:.2f}, "
                  f"Gap = {mean_gap:.2f}")


def run_embedding_diagnostics():
    """Post-training embedding diagnostics."""
    print("\n" + "=" * 70)
    print("STEP: EMBEDDING DIVERSITY DIAGNOSTICS")
    print("=" * 70)

    from diagnostics import embedding_diversity

    tg_path = os.path.join(RESULTS_DIR, 'tg_graphs.pkl')
    with open(tg_path, 'rb') as f:
        tg_data = pickle.load(f)

    sample_graphs = tg_data[:500]

    results = {}
    for variant_name, use_knode, ckpt_name in [
        ('A_random_knode', True, None),
        ('B_pi1m_knode', True, 'B_pi1m_knode_pretrained.pt'),
        ('C_random_knode', True, 'C_random_knode_pretrained.pt'),
        ('D_pi1m_noknode', False, 'D_pi1m_noknode_pretrained.pt'),
    ]:
        ckpt_path = os.path.join(CHECKPOINTS_DIR, ckpt_name) if ckpt_name else None
        if ckpt_path and not os.path.exists(ckpt_path):
            print(f"  Skipping {variant_name}: checkpoint not found")
            continue

        encoder = ChempropPolymerKMPNN(
            atom_fdim=ATOM_FDIM, bond_fdim=BOND_FDIM,
            k_feat_dim=K_FEAT_DIM if use_knode else 0,
            hidden_dim=HIDDEN_DIM, depth=DEPTH, dropout=DROPOUT,
            use_knode=use_knode, with_projector=False,
        ).to(DEVICE)

        if ckpt_path:
            state = torch.load(ckpt_path, map_location=DEVICE)
            if 'model_state_dict' in state:
                encoder.load_encoder_state_dict(state['model_state_dict'], strict=False)
            else:
                encoder.load_encoder_state_dict(state, strict=False)

        div = embedding_diversity(encoder, sample_graphs, device=DEVICE)
        results[variant_name] = div
        print(f"  {variant_name}: mean_cos_sim={div.get('mean_cosine_sim', 'N/A'):.4f}")

    # Convert for JSON
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    div_path = os.path.join(RESULTS_DIR, 'diagnostics_embedding.json')
    with open(div_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"  Embedding diversity saved to {div_path}")
    return results


def write_final_report(all_variant_results):
    """Write the comprehensive final report."""
    print("\n" + "=" * 70)
    print("WRITING FINAL REPORT")
    print("=" * 70)

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("CHEMPROP-POLYMER-K + PI1M CONTRASTIVE PRE-TRAINING")
    report_lines.append("HYPOTHESIS H11: PARADIGM RE-EVALUATION PRINCIPLE")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Configuration
    report_lines.append("CONFIGURATION")
    report_lines.append("-" * 40)
    report_lines.append(f"  Model: Chemprop-Polymer-K (hidden={HIDDEN_DIM}, depth={DEPTH})")
    report_lines.append(f"  K-node features: {K_FEAT_DIM} (Morgan FP 512 + descriptors 16)")
    report_lines.append(f"  Pre-training: PI1M 300K subset, {PRETRAIN_EPOCHS} epochs, lr={PRETRAIN_LR}")
    report_lines.append(f"  Fine-tuning: Tg dataset, {FINETUNE_EPOCHS} epochs max, patience={FINETUNE_PATIENCE}")
    report_lines.append(f"  Split: 72/8/20 (seed={SPLIT_SEED})")
    report_lines.append(f"  FFN seeds: {SEEDS}")
    report_lines.append(f"  Fine-tune LRs: {FINETUNE_LRS}")
    report_lines.append(f"  Device: {DEVICE}")
    report_lines.append("")

    # Reference baselines
    report_lines.append("REFERENCE BASELINES")
    report_lines.append("-" * 40)
    report_lines.append("  Day 5 Chemprop baseline: 43.46 +/- 0.89 RMSE")
    report_lines.append("  HPO best (single seed):  35.48 RMSE")
    report_lines.append("  HPO best (multi-seed):   40.00 +/- 2.09 RMSE")
    report_lines.append("")

    # Results per variant
    report_lines.append("RESULTS")
    report_lines.append("-" * 70)
    report_lines.append(f"  {'Variant':<35} {'Test RMSE':<20} {'Gap':<15}")
    report_lines.append("  " + "-" * 65)

    best_overall = None
    best_rmse = float('inf')

    for variant_name, variant_data in all_variant_results.items():
        for lr_key, lr_results in variant_data.items():
            if not isinstance(lr_results, dict) or 'summary' not in lr_results:
                continue
            s = lr_results['summary']
            name = f"{variant_name}_{lr_key}"
            mean_rmse = s.get('mean_test_rmse', 0)
            std_rmse = s.get('std_test_rmse', 0)
            mean_gap = s.get('mean_overfitting_gap', 0)
            test_str = f"{mean_rmse:.2f} +/- {std_rmse:.2f}"
            gap_str = f"{mean_gap:.2f}"
            report_lines.append(f"  {name:<35} {test_str:<20} {gap_str:<15}")

            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_overall = name

    report_lines.append("")

    # Per-seed detail
    report_lines.append("PER-SEED DETAIL")
    report_lines.append("-" * 70)
    for variant_name, variant_data in all_variant_results.items():
        for lr_key, lr_results in variant_data.items():
            if not isinstance(lr_results, dict):
                continue
            for seed in SEEDS:
                if seed in lr_results and isinstance(lr_results[seed], dict):
                    r = lr_results[seed]
                    report_lines.append(
                        f"  {variant_name}_{lr_key}_s{seed}: "
                        f"test={r.get('test_rmse', 'N/A'):.2f}, "
                        f"train={r.get('train_rmse_at_best', 'N/A'):.2f}, "
                        f"gap={r.get('overfitting_gap', 'N/A'):.2f}, "
                        f"best_ep={r.get('best_epoch', 'N/A')}"
                    )
    report_lines.append("")

    # Hypothesis analysis
    report_lines.append("HYPOTHESIS H11 ANALYSIS")
    report_lines.append("-" * 40)
    report_lines.append("")
    report_lines.append("H11a: Contrastive pre-training requires domain structure")
    report_lines.append("  Compare: B (PI1M+K-node) vs C (Random+K-node)")
    report_lines.append("  If B >> C, domain structure is essential for pre-training benefit")
    report_lines.append("")
    report_lines.append("H11b: K-node captures complementary molecular information")
    report_lines.append("  Compare: B (PI1M+K-node) vs D (PI1M no K-node)")
    report_lines.append("  If B > D, K-node's molecular descriptors add value")
    report_lines.append("")

    # Best result
    if best_overall:
        report_lines.append(f"BEST OVERALL: {best_overall}, Test RMSE = {best_rmse:.2f}")
        # Compare to baselines
        baseline = 43.46
        hpo = 40.00
        if best_rmse < baseline:
            report_lines.append(f"  vs Day 5 baseline: {best_rmse:.2f} < {baseline} (IMPROVED by {baseline - best_rmse:.2f})")
        if best_rmse < hpo:
            report_lines.append(f"  vs HPO multi-seed: {best_rmse:.2f} < {hpo} (IMPROVED by {hpo - best_rmse:.2f})")
    else:
        report_lines.append("  No results yet")

    report_lines.append("")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = os.path.join(RESULTS_DIR, 'report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\n  Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Chemprop-Polymer-K Experiments')
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='Skip data preprocessing step')
    parser.add_argument('--variant', type=str, default='all',
                        choices=['all', 'A', 'B', 'C', 'D', 'diag'],
                        help='Which variant to run')
    args = parser.parse_args()

    print("=" * 70)
    print("CHEMPROP-POLYMER-K + PI1M CONTRASTIVE PRE-TRAINING EXPERIMENT")
    print(f"Device: {DEVICE}")
    print(f"Working dir: {WORK_DIR}")
    print("=" * 70)

    # Step 1: Data preprocessing
    if not args.skip_preprocess and args.variant in ['all', 'A', 'B', 'C', 'D', 'diag']:
        preprocess_data()

    all_variant_results = {}

    if args.variant in ['all', 'diag']:
        run_diagnostics()

    if args.variant in ['all', 'A']:
        all_variant_results['variant_A'] = run_variant_a()

    if args.variant in ['all', 'B']:
        all_variant_results['variant_B'] = run_variant_b()

    if args.variant in ['all', 'C']:
        all_variant_results['variant_C'] = run_variant_c()

    if args.variant in ['all', 'D']:
        all_variant_results['variant_D'] = run_variant_d()

    # Post-training diagnostics
    if args.variant == 'all':
        run_embedding_diagnostics()

    # Final report
    if all_variant_results:
        write_final_report(all_variant_results)
    elif args.variant == 'all':
        # Try loading existing results for report
        for vname, fname in [('variant_A', 'variant_a_results.pkl'),
                              ('variant_B', 'variant_b_results.pkl'),
                              ('variant_C', 'variant_c_results.pkl'),
                              ('variant_D', 'variant_d_results.pkl')]:
            fpath = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(fpath):
                with open(fpath, 'rb') as f:
                    all_variant_results[vname] = pickle.load(f)
        if all_variant_results:
            write_final_report(all_variant_results)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
