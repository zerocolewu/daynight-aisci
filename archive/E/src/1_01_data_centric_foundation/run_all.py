"""
Main experiment runner for Data Centric Foundation experiments.
Runs all 5 experiments and collects results.
"""

import os
import sys
import time
import logging
import json
import numpy as np
import torch

from dataset import TgDataset, get_data_splits, compute_graph_stats
from train import train_model
from feature_utils import BASE_ATOM_FEATURE_DIM, BASE_BOND_FEATURE_DIM, AUGMENTED_ATOM_FEATURE_DIM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('experiment.log', mode='w'),
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CSV_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
WORK_DIR = '/home/home_public/aisci/AIscientist/src/1_01_data_centric_foundation'
SEED = 71
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EXPERIMENTS = [
    ('monomer', 'a_monomer'),
    ('dimer', 'b_dimer'),
    ('trimer', 'c_trimer'),
    ('cyclized', 'd_cyclized'),
    ('augmented', 'e_augmented'),
]


def save_learning_curves(results, exp_name, work_dir):
    """Save learning curve data as JSON and plot."""
    curves_dir = os.path.join(work_dir, 'learning_curves')
    os.makedirs(curves_dir, exist_ok=True)

    curve_data = {
        'train_rmse': results['train_rmses'],
        'val_rmse': results['val_rmses'],
        'best_epoch': results['best_epoch'],
    }

    json_path = os.path.join(curves_dir, f'{exp_name}_curves.json')
    with open(json_path, 'w') as f:
        json.dump(curve_data, f)

    # Try to plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        epochs = list(range(1, len(results['train_rmses']) + 1))
        ax.plot(epochs, results['train_rmses'], label='Train RMSE')
        ax.plot(epochs, results['val_rmses'], label='Val RMSE')
        ax.axvline(x=results['best_epoch'], color='r', linestyle='--', label=f"Best epoch ({results['best_epoch']})")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title(f'{exp_name} Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(curves_dir, f'{exp_name}_curves.png'), dpi=150)
        plt.close()
        logger.info(f"Saved learning curve plot for {exp_name}")
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")


def run_experiment(mode, exp_name, work_dir):
    """Run a single experiment."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting experiment: {exp_name} (mode={mode})")
    logger.info(f"{'='*60}")

    # Create dataset
    dataset_dir = os.path.join(work_dir, 'data', exp_name)
    os.makedirs(dataset_dir, exist_ok=True)

    dataset = TgDataset(
        root=dataset_dir,
        csv_path=CSV_PATH,
        mode=mode,
    )

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Failed conversions: {len(dataset.failed_indices)}")

    # Compute graph stats
    stats = compute_graph_stats(dataset, min(100, len(dataset)))
    logger.info(f"Graph stats: {stats}")

    # Get data splits
    train_idx, val_idx, test_idx = get_data_splits(dataset, seed=SEED)
    logger.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Determine feature dimensions
    sample = dataset[0]
    node_dim = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1)
    logger.info(f"Feature dims: node={node_dim}, edge={edge_dim}")

    # Save dir for best models
    save_dir = os.path.join(work_dir, 'models')
    os.makedirs(save_dir, exist_ok=True)

    # Train
    start_time = time.time()
    results = train_model(
        dataset=dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=256,
        num_layers=3,
        lr=0.001,
        batch_size=32,
        max_epochs=500,
        patience=30,
        seed=SEED,
        device=DEVICE,
        save_dir=save_dir,
        exp_name=exp_name,
    )
    elapsed = time.time() - start_time
    results['time_seconds'] = elapsed
    results['graph_stats'] = stats
    results['node_dim'] = node_dim
    results['edge_dim'] = edge_dim
    results['dataset_size'] = len(dataset)
    results['failed'] = len(dataset.failed_indices)

    logger.info(f"Experiment {exp_name} completed in {elapsed:.1f}s")
    logger.info(f"Results: train={results['train_rmse']:.4f}, val={results['val_rmse']:.4f}, test={results['test_rmse']:.4f}")

    # Save learning curves
    save_learning_curves(results, exp_name, work_dir)

    return results


def write_results_summary(all_results, work_dir):
    """Write final results summary."""
    summary_path = os.path.join(work_dir, 'results_summary.txt')

    with open(summary_path, 'w') as f:
        f.write("Data Centric Foundation Experiment Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset: Tg (7170 p-SMILES), Seed={SEED}, Split=72/8/20\n")
        f.write("Base Model: GIN (3 layers, hidden_dim=256, mean pooling + MLP)\n")
        f.write(f"Device: {DEVICE}\n\n")

        for mode, exp_name in EXPERIMENTS:
            r = all_results[exp_name]
            f.write(f"Experiment ({exp_name[0]}) - {exp_name[2:].replace('_', ' ').title()}:\n")
            f.write(f"  Dataset size: {r['dataset_size']} (failed: {r['failed']})\n")
            f.write(f"  Graph stats: avg_nodes={r['graph_stats']['avg_nodes']:.1f}, "
                    f"avg_edges={r['graph_stats']['avg_edges']:.1f}\n")
            f.write(f"  Feature dims: node={r['node_dim']}, edge={r['edge_dim']}\n")
            f.write(f"  Train RMSE: {r['train_rmse']:.4f}\n")
            f.write(f"  Val RMSE:   {r['val_rmse']:.4f}\n")
            f.write(f"  Test RMSE:  {r['test_rmse']:.4f}\n")
            f.write(f"  Best epoch: {r['best_epoch']}\n")
            f.write(f"  Total epochs: {r['total_epochs']}\n")
            f.write(f"  Time: {r['time_seconds']:.1f}s\n\n")

        f.write("\nAnalysis:\n")
        f.write("-" * 60 + "\n")

        baseline = all_results['a_monomer']['test_rmse']
        f.write(f"\nBaseline (Monomer) Test RMSE: {baseline:.4f}\n\n")

        for mode, exp_name in EXPERIMENTS[1:]:
            r = all_results[exp_name]
            diff = r['test_rmse'] - baseline
            pct = (diff / baseline) * 100
            if diff < 0:
                f.write(f"  {exp_name[2:].replace('_', ' ').title()}: IMPROVED by {abs(diff):.4f} ({abs(pct):.2f}%)\n")
            else:
                f.write(f"  {exp_name[2:].replace('_', ' ').title()}: WORSENED by {diff:.4f} ({pct:.2f}%)\n")

        f.write("\n  Key Observations:\n")

        # Multimer effect
        mono = all_results['a_monomer']['test_rmse']
        dimer = all_results['b_dimer']['test_rmse']
        trimer = all_results['c_trimer']['test_rmse']
        f.write(f"  - Dimer vs Monomer: {dimer - mono:+.4f} RMSE change\n")
        f.write(f"  - Trimer vs Monomer: {trimer - mono:+.4f} RMSE change\n")

        # Cyclization effect
        cycl = all_results['d_cyclized']['test_rmse']
        f.write(f"  - Cyclized trimer vs Trimer: {cycl - trimer:+.4f} RMSE change\n")
        f.write(f"  - Cyclized trimer vs Monomer: {cycl - mono:+.4f} RMSE change\n")

        # Feature augmentation effect
        aug = all_results['e_augmented']['test_rmse']
        f.write(f"  - Augmented trimer vs Trimer: {aug - trimer:+.4f} RMSE change\n")
        f.write(f"  - Augmented trimer vs Monomer: {aug - mono:+.4f} RMSE change\n")

        # Convergence speed
        f.write("\n  Convergence Analysis:\n")
        for mode, exp_name in EXPERIMENTS:
            r = all_results[exp_name]
            f.write(f"    {exp_name[2:].replace('_', ' ').title()}: best_epoch={r['best_epoch']}, "
                    f"total_epochs={r['total_epochs']}\n")

        # Computational cost
        f.write("\n  Computational Cost:\n")
        for mode, exp_name in EXPERIMENTS:
            r = all_results[exp_name]
            f.write(f"    {exp_name[2:].replace('_', ' ').title()}: "
                    f"avg_nodes={r['graph_stats']['avg_nodes']:.1f}, "
                    f"avg_edges={r['graph_stats']['avg_edges']:.1f}, "
                    f"time={r['time_seconds']:.1f}s\n")

    logger.info(f"Results summary written to {summary_path}")
    return summary_path


def main():
    logger.info("Starting Data Centric Foundation Experiments")
    logger.info(f"Working directory: {WORK_DIR}")
    logger.info(f"Device: {DEVICE}")

    all_results = {}

    for mode, exp_name in EXPERIMENTS:
        try:
            results = run_experiment(mode, exp_name, WORK_DIR)
            all_results[exp_name] = results
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}", exc_info=True)
            all_results[exp_name] = {
                'error': str(e),
                'train_rmse': float('nan'),
                'val_rmse': float('nan'),
                'test_rmse': float('nan'),
                'best_epoch': 0,
                'total_epochs': 0,
                'time_seconds': 0,
                'graph_stats': {'avg_nodes': 0, 'avg_edges': 0, 'max_nodes': 0, 'min_nodes': 0},
                'node_dim': 0,
                'edge_dim': 0,
                'dataset_size': 0,
                'failed': 0,
                'train_rmses': [],
                'val_rmses': [],
            }

    # Write summary
    summary_path = write_results_summary(all_results, WORK_DIR)
    logger.info(f"All experiments completed. Summary: {summary_path}")

    # Print quick summary to stdout
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for mode, exp_name in EXPERIMENTS:
        r = all_results[exp_name]
        if 'error' in r:
            print(f"  {exp_name}: FAILED - {r['error']}")
        else:
            print(f"  {exp_name}: test_rmse={r['test_rmse']:.4f}, best_epoch={r['best_epoch']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
