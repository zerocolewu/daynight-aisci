"""
Main experiment runner for PolyBG-CPA project.
Runs 5 experiments on seed=71 (primary), plus seeds 42, 123 for variance.
Iteration 2: adds Experiments 6-8 (Hierarchical Bond->Atom Encoding).
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_DIR, SEEDS
from data.tg_dataset import load_data, compute_bond_data
from models.polybg_cpa import (
    CPAGATBaseline, PolyBGCPA, PolyBGnoCPA, BondOnlyCPA, EnrichedEdgeCPAGAT,
    HierarchicalPolyBGCPA_Pre, HierarchicalPolyBGCPA_Mid, HierarchicalPolyBGCPA_PreGated,
)
from train import train_model


def run_experiments():
    print("=" * 70)
    print("PolyBG-CPA: Parallel Bond-Graph CPA-GAT for Polymer Tg Prediction")
    print("=" * 70)

    print("\nLoading data...")
    train_graphs, val_graphs, test_graphs, scaler, train_smiles, val_smiles, test_smiles = load_data()

    # Bond data for bond-graph experiments (Exp 2,3,4)
    print("\nComputing bond data...")
    train_atoms, train_bond = compute_bond_data(
        [g.clone() for g in train_graphs], train_smiles, edge_only_dim=None)
    val_atoms, val_bond = compute_bond_data(
        [g.clone() for g in val_graphs], val_smiles, edge_only_dim=None)
    test_atoms, test_bond = compute_bond_data(
        [g.clone() for g in test_graphs], test_smiles, edge_only_dim=None)

    # Edge features for Exp 5
    train_exp5, _ = compute_bond_data(
        [g.clone() for g in train_graphs], train_smiles, edge_only_dim=8)
    val_exp5, _ = compute_bond_data(
        [g.clone() for g in val_graphs], val_smiles, edge_only_dim=8)
    test_exp5, _ = compute_bond_data(
        [g.clone() for g in test_graphs], test_smiles, edge_only_dim=8)

    all_results = {}

    # Exp 1
    print("\n" + "#" * 70)
    print("# Experiment 1: CPA-GAT Atom-Only Baseline")
    print("#" * 70)
    exp1_results = []
    for seed in SEEDS:
        model = CPAGATBaseline(in_dim=42, hidden_dim=256, num_layers=4, heads=4)
        r = train_model(model, "exp1_cpa_gat_baseline", seed,
                        train_graphs, val_graphs, test_graphs, scaler)
        exp1_results.append(r)
    all_results['exp1_cpa_gat_baseline'] = {
        'results': exp1_results, 'description': 'CPA-GAT Atom-Only (42-dim, 4L)'}

    # Exp 2
    print("\n" + "#" * 70)
    print("# Experiment 2: PolyBG-CPA Parallel")
    print("#" * 70)
    exp2_results = []
    for seed in SEEDS:
        model = PolyBGCPA(
            atom_in_dim=42, atom_hidden=256, atom_layers=4, atom_heads=4,
            bond_in_dim=14, bond_hidden=128, bond_layers=3, bond_heads=4,
            predictor_hidden=192, dropout=0.1)
        r = train_model(model, "exp2_polybg_cpa", seed,
                        train_atoms, val_atoms, test_atoms, scaler,
                        train_bond_data=train_bond, val_bond_data=val_bond, test_bond_data=test_bond)
        exp2_results.append(r)
    all_results['exp2_polybg_cpa'] = {
        'results': exp2_results, 'description': 'PolyBG-CPA Parallel (Atom 256d 4L + Bond 128d 3L)'}

    # Exp 3
    print("\n" + "#" * 70)
    print("# Experiment 3: PolyBG-noCPA")
    print("#" * 70)
    exp3_results = []
    for seed in [SEEDS[0]]:
        model = PolyBGnoCPA(
            atom_in_dim=42, atom_hidden=256, atom_layers=4, atom_heads=4,
            bond_in_dim=14, bond_hidden=128, bond_layers=3, bond_heads=4,
            predictor_hidden=192, dropout=0.1)
        r = train_model(model, "exp3_polybg_nocpa", seed,
                        train_atoms, val_atoms, test_atoms, scaler,
                        train_bond_data=train_bond, val_bond_data=val_bond, test_bond_data=test_bond)
        exp3_results.append(r)
    all_results['exp3_polybg_nocpa'] = {
        'results': exp3_results, 'description': 'PolyBG-noCPA: Plain GAT both'}

    # Exp 4
    print("\n" + "#" * 70)
    print("# Experiment 4: Bond-CPA-GAT Only")
    print("#" * 70)
    exp4_results = []
    for seed in [SEEDS[0]]:
        model = BondOnlyCPA(
            bond_in_dim=14, bond_hidden=128, bond_layers=3, bond_heads=4,
            predictor_hidden=192, dropout=0.1)
        r = train_model(model, "exp4_bond_cpa_only", seed,
                        train_atoms, val_atoms, test_atoms, scaler,
                        train_bond_data=train_bond, val_bond_data=val_bond, test_bond_data=test_bond)
        exp4_results.append(r)
    all_results['exp4_bond_cpa_only'] = {
        'results': exp4_results, 'description': 'Bond-CPA-GAT Only (3L, 128d)'}

    # Exp 5
    print("\n" + "#" * 70)
    print("# Experiment 5: Enriched-Edge CPA-GAT")
    print("#" * 70)
    exp5_results = []
    for seed in [SEEDS[0]]:
        model = EnrichedEdgeCPAGAT(
            in_dim=42, hidden_dim=256, num_layers=4, heads=4, edge_dim=8)
        r = train_model(model, "exp5_enriched_edge_cpa", seed,
                        train_exp5, val_exp5, test_exp5, scaler)
        exp5_results.append(r)
    all_results['exp5_enriched_edge_cpa'] = {
        'results': exp5_results, 'description': 'Enriched-Edge CPA-GAT (8d edge)'}

    # Summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    summary = {}
    for exp_name, exp_data in all_results.items():
        results = exp_data['results']
        rmses = [r['test_rmse'] for r in results]
        maes = [r['test_mae'] for r in results]
        summary[exp_name] = {
            'description': exp_data['description'],
            'test_rmse_mean': float(np.mean(rmses)),
            'test_rmse_std': float(np.std(rmses)),
            'test_mae_mean': float(np.mean(maes)),
            'num_seeds': len(results),
            'seeds': [{'seed': r['seed'], 'test_rmse': r['test_rmse'],
                       'test_mae': r['test_mae'], 'best_epoch': r['best_epoch'],
                       'train_time': r['train_time']} for r in results],
        }

    exp_names = ['exp1_cpa_gat_baseline', 'exp2_polybg_cpa', 'exp3_polybg_nocpa',
                 'exp4_bond_cpa_only', 'exp5_enriched_edge_cpa']
    baseline_rmse = summary['exp1_cpa_gat_baseline']['test_rmse_mean']

    print(f"\n{'Experiment':<40s} {'RMSE (K)':<15s} {'MAE (K)':<12s} {'vs Baseline'}")
    print("-" * 82)
    for en in exp_names:
        if en in summary:
            d = summary[en]
            mean_r = d['test_rmse_mean']
            std_r = d['test_rmse_std']
            delta = baseline_rmse - mean_r
            ds = f"-{delta:.2f} K" if delta > 0 else f"+{-delta:.2f} K"
            print(f"{d['description']:<40s} {mean_r:.2f} +/- {std_r:.2f}  {d['test_mae_mean']:.2f}       {ds}")
    print("-" * 82)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "results.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {RESULTS_DIR}/results.json")
    return all_results, summary


def run_iteration2():
    """Run Iteration 2 experiments: Exp6-8 (Hierarchical Bond->Atom Encoding)."""
    print("=" * 70)
    print("PolyBG-CPA Iteration 2: Hierarchical Bond->Atom Encoding")
    print("=" * 70)

    print("\nLoading data...")
    train_graphs, val_graphs, test_graphs, scaler, train_smiles, val_smiles, test_smiles = load_data()

    print("\nComputing bond data (with bond_to_atom mapping)...")
    train_atoms, train_bond = compute_bond_data(
        [g.clone() for g in train_graphs], train_smiles, edge_only_dim=None)
    val_atoms, val_bond = compute_bond_data(
        [g.clone() for g in val_graphs], val_smiles, edge_only_dim=None)
    test_atoms, test_bond = compute_bond_data(
        [g.clone() for g in test_graphs], test_smiles, edge_only_dim=None)

    all_results = {}

    # Load existing results if available
    existing_results_path = os.path.join(RESULTS_DIR, "results.json")
    if os.path.exists(existing_results_path):
        with open(existing_results_path, 'r') as f:
            existing_results = json.load(f)
    else:
        existing_results = {}

    # =====================================================================
    # Exp 6: Hierarchical-Pre (Bond->Atom input augmentation)
    # =====================================================================
    print("\n" + "#" * 70)
    print("# Experiment 6: Hierarchical-Pre (Bond->Atom Input Augmentation)")
    print("#" * 70)
    exp6_results = []
    for seed in SEEDS:
        model = HierarchicalPolyBGCPA_Pre(
            atom_in_dim=42, atom_hidden=256, atom_layers=4, atom_heads=4,
            bond_in_dim=14, bond_hidden=128, bond_layers=3, bond_heads=4,
            predictor_hidden=128, dropout=0.1)
        r = train_model(model, "exp6_hierarchical_pre", seed,
                        train_atoms, val_atoms, test_atoms, scaler,
                        train_bond_data=train_bond, val_bond_data=val_bond, test_bond_data=test_bond)
        exp6_results.append(r)
    all_results['exp6_hierarchical_pre'] = {
        'results': exp6_results,
        'description': 'Exp6: Hierarchical-Pre (Bond->Atom Input Aug)'
    }

    # =====================================================================
    # Exp 7: Hierarchical-Mid (Bond->Atom after layer 1)
    # =====================================================================
    print("\n" + "#" * 70)
    print("# Experiment 7: Hierarchical-Mid (Bond->Atom After Layer 1)")
    print("#" * 70)
    exp7_results = []
    for seed in [SEEDS[0]]:
        model = HierarchicalPolyBGCPA_Mid(
            atom_in_dim=42, atom_hidden=256, atom_layers=4, atom_heads=4,
            bond_in_dim=14, bond_hidden=128, bond_layers=3, bond_heads=4,
            predictor_hidden=128, dropout=0.1)
        r = train_model(model, "exp7_hierarchical_mid", seed,
                        train_atoms, val_atoms, test_atoms, scaler,
                        train_bond_data=train_bond, val_bond_data=val_bond, test_bond_data=test_bond)
        exp7_results.append(r)
    all_results['exp7_hierarchical_mid'] = {
        'results': exp7_results,
        'description': 'Exp7: Hierarchical-Mid (Bond->Atom After L1)'
    }

    # =====================================================================
    # Exp 8: Hierarchical-Pre Gated
    # =====================================================================
    print("\n" + "#" * 70)
    print("# Experiment 8: Hierarchical-Pre Gated Fusion")
    print("#" * 70)
    exp8_results = []
    for seed in [SEEDS[0]]:
        model = HierarchicalPolyBGCPA_PreGated(
            atom_in_dim=42, atom_hidden=256, atom_layers=4, atom_heads=4,
            bond_in_dim=14, bond_hidden=128, bond_layers=3, bond_heads=4,
            predictor_hidden=128, dropout=0.1)
        r = train_model(model, "exp8_hierarchical_pre_gated", seed,
                        train_atoms, val_atoms, test_atoms, scaler,
                        train_bond_data=train_bond, val_bond_data=val_bond, test_bond_data=test_bond)
        exp8_results.append(r)
    all_results['exp8_hierarchical_pre_gated'] = {
        'results': exp8_results,
        'description': 'Exp8: Hierarchical-Pre Gated (Learned Gate)'
    }

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 80)
    print("ITERATION 2 FINAL RESULTS")
    print("=" * 80)

    summary = dict(existing_results)
    for exp_name, exp_data in all_results.items():
        results = exp_data['results']
        rmses = [r['test_rmse'] for r in results]
        maes = [r['test_mae'] for r in results]
        summary[exp_name] = {
            'description': exp_data['description'],
            'test_rmse_mean': float(np.mean(rmses)),
            'test_rmse_std': float(np.std(rmses)),
            'test_mae_mean': float(np.mean(maes)),
            'num_seeds': len(results),
            'seeds': [{'seed': r['seed'], 'test_rmse': r['test_rmse'],
                       'test_mae': r['test_mae'], 'best_epoch': r['best_epoch'],
                       'train_time': r['train_time']} for r in results],
        }

    baseline_rmse = summary.get('exp1_cpa_gat_baseline', {}).get('test_rmse_mean', None)

    exp_keys = list(summary.keys())
    iter2_keys = ['exp6_hierarchical_pre', 'exp7_hierarchical_mid', 'exp8_hierarchical_pre_gated']

    print(f"\n{'Experiment':<45s} {'RMSE (K)':<15s} {'MAE (K)':<12s} {'vs Baseline'}")
    print("-" * 92)
    for ek in exp_keys:
        if ek in summary:
            d = summary[ek]
            if 'test_rmse_std' in d and d.get('num_seeds', 1) > 1:
                mean_r = d['test_rmse_mean']
                std_r = d['test_rmse_std']
                print(f"{d['description']:<45s} {mean_r:.2f} +/- {std_r:.2f}  {d['test_mae_mean']:.2f}       "
                      f"(vs Exp2: {mean_r - summary.get('exp2_polybg_cpa', {}).get('test_rmse_mean', 0):+.2f}K)")
            else:
                mean_r = d['test_rmse_mean']
                print(f"{d['description']:<45s} {mean_r:.2f}          {d['test_mae_mean']:.2f}       "
                      f"(vs Exp2: {mean_r - summary.get('exp2_polybg_cpa', {}).get('test_rmse_mean', 0):+.2f}K)")
    print("-" * 92)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "results.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {RESULTS_DIR}/results.json")
    return all_results, summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "iter2":
        run_iteration2()
    else:
        run_experiments()
