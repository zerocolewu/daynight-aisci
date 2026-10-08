"""
Main experiment runner for CPA-PSE experiment.
Runs all conditions across 5 seeds, aggregates results, computes diagnostics.
"""
import os
import sys
import json
import numpy as np
import torch
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train_model, load_data, load_pse_features, prepare_graphs, evaluate, DATA_PATH
from model import CPAGAT, NonCPAGAT

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [71, 42, 123, 888, 2024]

# Experiment conditions
# Priority order: A, B first; then C, D, E, F; then H; then I
CONDITIONS = {
    # Primary
    'A': {
        'model_class': CPAGAT,
        'model_name': 'cpa_baseline',
        'description': 'CPA-GAT baseline (42-dim, no PSE)',
        'priority': 1,
    },
    'B': {
        'model_class': CPAGAT,
        'model_name': 'cpa_pse',
        'description': 'CPA-PSE full (45-dim, all 3 PSE features)',
        'priority': 1,
    },
    # Key ablations
    'C': {
        'model_class': CPAGAT,
        'model_name': 'cpa_modpos',
        'description': 'CPA + modular_position only (43-dim)',
        'priority': 2,
    },
    'D': {
        'model_class': CPAGAT,
        'model_name': 'cpa_perdist',
        'description': 'CPA + periodic_distance only (43-dim)',
        'priority': 2,
    },
    'E': {
        'model_class': CPAGAT,
        'model_name': 'cpa_backbone',
        'description': 'CPA + is_backbone only (43-dim)',
        'priority': 2,
    },
    'F': {
        'model_class': CPAGAT,
        'model_name': 'cpa_pse_nopspe',
        'description': 'CPA + PSE without PSPE (43-dim, remove dce+mi)',
        'priority': 3,
    },
    # Non-CPA ablation
    'H': {
        'model_class': NonCPAGAT,
        'model_name': 'noncpa_pse',
        'description': 'Non-CPA GAT + full PSE (45-dim)',
        'priority': 4,
    },
}


def run_experiments(conditions_to_run=None):
    """Run all specified conditions across 5 seeds."""
    if conditions_to_run is None:
        conditions_to_run = ['A', 'B']

    summary = {}

    for cond in conditions_to_run:
        if cond not in CONDITIONS:
            print(f"Skipping unknown condition: {cond}")
            continue

        info = CONDITIONS[cond]
        model_class = info['model_class']
        model_name = info['model_name']
        description = info['description']

        print(f"\n{'#'*60}")
        print(f"# CONDITION {cond}: {description}")
        print(f"{'#'*60}")

        cond_results = []
        for seed in SEEDS:
            result = train_model(model_class, model_name, seed, cond)
            cond_results.append(result)

        rmses = [r['test_rmse'] for r in cond_results]
        maes = [r['test_mae'] for r in cond_results]
        r2s = [r['test_r2'] for r in cond_results]
        val_rmses = [r['best_val_rmse'] for r in cond_results]

        mean_rmse = np.mean(rmses)
        mean_mae = np.mean(maes)
        mean_r2 = np.mean(r2s)
        mean_val_rmse = np.mean(val_rmses)

        std_rmse = np.std(rmses, ddof=1)
        std_mae = np.std(maes, ddof=1)
        std_r2 = np.std(r2s, ddof=1)
        std_val_rmse = np.std(val_rmses, ddof=1)

        print(f"\n{'='*60}")
        print(f"Condition {cond}: {description}")
        print(f"  Test RMSE:  {mean_rmse:.4f} ± {std_rmse:.4f} K")
        print(f"  Test MAE:   {mean_mae:.4f} ± {std_mae:.4f} K")
        print(f"  Test R²:    {mean_r2:.4f} ± {std_r2:.4f}")
        print(f"  Val RMSE:   {mean_val_rmse:.4f} ± {std_val_rmse:.4f}")
        print(f"  Per-seed RMSE: {rmses}")
        print(f"{'='*60}")

        summary[cond] = {
            'description': description,
            'model_name': model_name,
            'test_rmse_mean': float(mean_rmse),
            'test_rmse_std': float(std_rmse),
            'test_mae_mean': float(mean_mae),
            'test_mae_std': float(std_mae),
            'test_r2_mean': float(mean_r2),
            'test_r2_std': float(std_r2),
            'val_rmse_mean': float(mean_val_rmse),
            'val_rmse_std': float(std_val_rmse),
            'seeds': [{'seed': r['seed'], 'test_rmse': r['test_rmse'],
                       'test_mae': r['test_mae'], 'test_r2': r['test_r2'],
                       'best_val_rmse': r['best_val_rmse']} for r in cond_results],
            'num_params': cond_results[0]['num_params'],
        }

    return summary


def compute_diagnostic_analysis(summary):
    """Compute diagnostic analyses from the best results."""
    diagnostics = {}

    # H1: B - A improvement
    if 'A' in summary and 'B' in summary:
        improvement = summary['A']['test_rmse_mean'] - summary['B']['test_rmse_mean']
        diagnostics['H1_B_vs_A'] = {
            'A_rmse': summary['A']['test_rmse_mean'],
            'B_rmse': summary['B']['test_rmse_mean'],
            'improvement_K': improvement,
            'hypothesis': 'H1: PSE improves over CPA baseline by >= 0.3K',
            'supported': improvement >= 0.3,
        }

    # H2: modular_position alone vs full PSE
    if 'C' in summary and 'B' in summary:
        diagnostics['H2_modpos_vs_full'] = {
            'C_rmse': summary['C']['test_rmse_mean'],
            'B_rmse': summary['B']['test_rmse_mean'],
            'diff_K': summary['C']['test_rmse_mean'] - summary['B']['test_rmse_mean'],
            'hypothesis': 'H2: modular_position alone provides most of the gain',
        }

    # H3: is_backbone redundancy
    if 'E' in summary and 'A' in summary:
        improvement_E = summary['A']['test_rmse_mean'] - summary['E']['test_rmse_mean']
        diagnostics['H3_isbackbone'] = {
            'E_rmse': summary['E']['test_rmse_mean'],
            'A_rmse': summary['A']['test_rmse_mean'],
            'improvement_K': improvement_E,
            'hypothesis': 'H3: is_backbone is redundant with PSPE',
            'supported': abs(improvement_E) < 0.2,
        }

    # H4: PSE benefit is CPA-dependent
    if 'H' in summary and 'B' in summary and 'A' in summary:
        cpa_improvement = summary['A']['test_rmse_mean'] - summary['B']['test_rmse_mean']
        noncpa_improvement = summary.get('_noncpa_baseline', {}).get('test_rmse_mean', 0) - summary['H']['test_rmse_mean']
        diagnostics['H4_cpa_dependency'] = {
            'cpa_improvement': cpa_improvement,
            'hypothesis': 'H4: PSE benefit is CPA-dependent',
        }

    return diagnostics


def run_strata_analysis(best_condition='B', seed=71):
    """
    Run chain-length strata analysis on the best model.
    Strata: short (<60 nodes), medium (60-99), long (>138 nodes).
    """
    import torch
    from torch_geometric.loader import DataLoader
    from train import load_data, load_pse_features, prepare_graphs, set_seed
    from model import CPAGAT

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    train_graphs, val_graphs, test_graphs, scaler = load_data()
    pse_data = load_pse_features()

    # Use condition features
    test_graphs_prep = prepare_graphs(test_graphs, pse_data['test'], best_condition)

    # Find model file
    model_file = os.path.join(RESULTS_DIR, f"cpa_pse_{best_condition}_{seed}.pt")
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        return None

    in_dim = test_graphs_prep[0].x.shape[1]
    model = CPAGAT(in_dim=in_dim).to(device)
    model.load_state_dict(torch.load(model_file, weights_only=True))

    # Split test graphs by node count strata
    short_graphs = []
    med_graphs = []
    long_graphs = []
    short_targets = []
    med_targets = []
    long_targets = []

    for g in test_graphs_prep:
        n = g.x.shape[0]
        if n < 60:
            short_graphs.append(g)
            short_targets.append(g.y.item())
        elif n <= 99:
            med_graphs.append(g)
            med_targets.append(g.y.item())
        else:
            long_graphs.append(g)
            long_targets.append(g.y.item())

    strata_results = {}

    for strata_name, graphs, targets in [
        ('short_<60', short_graphs, short_targets),
        ('medium_60-99', med_graphs, med_targets),
        ('long_>138', long_graphs, long_targets),
    ]:
        if len(graphs) == 0:
            strata_results[strata_name] = {'count': 0}
            continue

        loader = DataLoader(graphs, batch_size=64, shuffle=False)
        criterion = torch.nn.MSELoss()
        _, rmse, mae, r2, preds, tgts = evaluate(
            model, loader, criterion, device, scaler
        )

        strata_results[strata_name] = {
            'count': len(graphs),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
        }

        print(f"  {strata_name}: n={len(graphs)}, RMSE={rmse:.2f}K, MAE={mae:.2f}K, R²={r2:.4f}")

    return strata_results


def compute_per_sample_improvement(cond_better='B', cond_baseline='A', seed=71):
    """
    Compute per-sample improvement: what fraction of test samples improve with PSE?
    """
    # Load results files
    def find_result_file(model_name, condition, seed):
        cond_tag = f"{condition}_{seed}"
        fpath = os.path.join(RESULTS_DIR, f"{model_name}_{cond_tag}.json")
        return fpath if os.path.exists(fpath) else None

    better_file = find_result_file('cpa_pse', cond_better, seed)
    baseline_file = find_result_file('cpa_baseline', cond_baseline, seed)

    if better_file is None or baseline_file is None:
        print("Could not find result files for per-sample improvement analysis")
        return None

    with open(better_file) as f:
        better = json.load(f)
    with open(baseline_file) as f:
        baseline = json.load(f)

    better_preds = np.array(better['test_preds'])
    baseline_preds = np.array(baseline['test_preds'])
    targets = np.array(better['test_targets'])

    # For each sample, which model is closer?
    better_err = np.abs(better_preds - targets)
    baseline_err = np.abs(baseline_preds - targets)

    improved = better_err < baseline_err
    n_improved = improved.sum()
    n_total = len(improved)

    mean_improvement = (baseline_err[improved] - better_err[improved]).mean()

    print(f"\n=== Per-sample Improvement Analysis ===")
    print(f"  Total test samples: {n_total}")
    print(f"  Improved with PSE: {n_improved} ({100*n_improved/n_total:.1f}%)")
    print(f"  Mean improvement per improved sample: {mean_improvement:.2f} K")
    print(f"  PSE worse in: {n_total - n_improved} samples ({100*(n_total-n_improved)/n_total:.1f}%)")

    return {
        'n_total': int(n_total),
        'n_improved': int(n_improved),
        'pct_improved': float(100 * n_improved / n_total),
        'mean_improvement_K': float(mean_improvement),
    }


def main():
    """Main entry point for the experiment."""
    import argparse
    parser = argparse.ArgumentParser(description='Run CPA-PSE experiments')
    parser.add_argument('--conditions', nargs='+', default=['A', 'B', 'C', 'D', 'E', 'F', 'H'],
                        help='Conditions to run (default: all)')
    parser.add_argument('--priority', type=int, default=None,
                        help='Only run conditions with this priority or higher')
    parser.add_argument('--analysis-only', action='store_true',
                        help='Only run analysis on existing results')
    args = parser.parse_args()

    if not args.analysis_only:
        # Filter by priority if specified
        if args.priority is not None:
            conditions_to_run = [c for c in args.conditions
                                 if CONDITIONS[c]['priority'] <= args.priority]
        else:
            conditions_to_run = args.conditions

        # Run experiments
        summary = run_experiments(conditions_to_run)
    else:
        # Load existing summary
        summary_file = os.path.join(RESULTS_DIR, "results_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file) as f:
                summary = json.load(f)
        else:
            print("No existing summary found. Run experiments first.")
            return

    # Compute diagnostics
    diagnostics = compute_diagnostic_analysis(summary)
    summary['diagnostics'] = diagnostics

    # Run strata analysis
    print(f"\n{'='*60}")
    print("Chain-length strata analysis")
    print(f"{'='*60}")
    strata = run_strata_analysis(best_condition='B', seed=71)
    if strata:
        summary['strata_analysis'] = strata

    # Per-sample improvement
    per_sample = compute_per_sample_improvement(cond_better='B', cond_baseline='A', seed=71)
    if per_sample:
        summary['per_sample_improvement'] = per_sample

    # Save final summary
    summary_file = os.path.join(RESULTS_DIR, "results_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Final Summary saved to: {summary_file}")
    print(f"{'='*60}")

    # Print full summary table
    print(f"\n{'Condition':<20} {'Test RMSE':<16} {'Test MAE':<16} {'Test R²':<12} {'#Params':<10}")
    print("-" * 74)
    for cond in ['A', 'B', 'C', 'D', 'E', 'F', 'H']:
        if cond in summary and cond != 'diagnostics' and cond != 'strata_analysis' and cond != 'per_sample_improvement':
            s = summary[cond]
            print(f"{cond + ' (' + s.get('description', '')[:15] + ')':<20} "
                  f"{s['test_rmse_mean']:.2f}±{s['test_rmse_std']:.2f} K  "
                  f"{s['test_mae_mean']:.2f}±{s['test_mae_std']:.2f} K  "
                  f"{s['test_r2_mean']:.3f}±{s['test_r2_std']:.3f}  "
                  f"{s.get('num_params', 0):,}")

    return summary


if __name__ == "__main__":
    main()
