"""
Diagnostics for PolymerBipartite v2 Iteration 2.

Includes all 5 mandatory diagnostics plus gate value analysis:
1. RU Virtual Node Discriminability (cos_sim)
2. Random RU Assignment Control
3. Parameter-Matched Baseline (wide hidden_dim)
4. Overfitting Gap Analysis
5. Multi-Seed Verification
+ Gate Value Distribution (scalar gate params)
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch
from train import (
    prepare_batch, prepare_baseline_batch, evaluate_model,
    count_parameters
)
from data_preprocessing import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from model import PolymerBipartiteV2, GINTrimerBaseline, create_wide_baseline


# ============================================================================
# Diagnostic 1: RU Virtual Node Discriminability
# ============================================================================

def diagnostic_ru_discriminability(model, dataset, test_indices, device, batch_size=32):
    """
    Compute pairwise cosine similarity between final RU representations
    across ALL test set polymers.

    Target: mean cos_sim < 0.90 (discriminative)
    Degenerate: mean cos_sim > 0.95 (all RU representations collapsed)
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1: RU Virtual Node Discriminability")
    print("=" * 70)

    model.eval()
    valid_mask = dataset.valid_mask
    test_valid = [i for i in test_indices if valid_mask[i]]

    all_ru_h = []
    with torch.no_grad():
        for i in range(0, len(test_valid), batch_size):
            bi = test_valid[i:i + batch_size]
            dd, _ = prepare_batch(dataset.graphs, dataset.targets, bi, device)
            _, h_ru, _ = model(dd, return_gate_values=True)

            B = len(bi)
            for k in range(3):
                ru_idx = torch.arange(B, device=device) * 3 + k
                all_ru_h.append(h_ru[ru_idx].cpu())

    all_ru_h = torch.cat(all_ru_h, dim=0)
    n = all_ru_h.shape[0]
    print(f"Total RU representations: {n}")

    # Sample if too large
    if n > 1000:
        idx = torch.randperm(n)[:1000]
        sample_h = all_ru_h[idx]
    else:
        sample_h = all_ru_h

    ns = sample_h.shape[0]
    normed = F.normalize(sample_h, dim=-1)
    cos_sim_matrix = normed @ normed.t()

    triu_idx = torch.triu_indices(ns, ns, offset=1)
    pairwise_cos = cos_sim_matrix[triu_idx[0], triu_idx[1]]

    mean_cos = pairwise_cos.mean().item()
    median_cos = pairwise_cos.median().item()
    std_cos = pairwise_cos.std().item()
    pct_above_95 = (pairwise_cos > 0.95).float().mean().item() * 100

    print(f"  Pairwise cosine similarity statistics:")
    print(f"    Mean:   {mean_cos:.4f}")
    print(f"    Median: {median_cos:.4f}")
    print(f"    Std:    {std_cos:.4f}")
    print(f"    % > 0.95: {pct_above_95:.1f}%")

    if mean_cos < 0.90:
        print(f"  PASS: RU representations are discriminative (mean cos_sim = {mean_cos:.4f} < 0.90)")
    else:
        print(f"  FAIL: RU representations may be degenerate (mean cos_sim = {mean_cos:.4f} >= 0.90)")

    return {
        'mean_cos_sim': mean_cos,
        'median_cos_sim': median_cos,
        'std_cos_sim': std_cos,
        'pct_above_95': pct_above_95,
        'n_representations': n,
    }


# ============================================================================
# Diagnostic 2: Random RU Assignment Control
# ============================================================================

def diagnostic_random_ru_summary(real_results, random_results):
    """Compare real vs random RU assignment results."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2: Random RU Assignment Control")
    print("=" * 70)

    real_test = real_results['test_rmse']
    random_test = random_results['test_rmse']
    diff = random_test - real_test

    print(f"  Real RU assignment test RMSE:    {real_test:.2f}")
    print(f"  Random RU assignment test RMSE:  {random_test:.2f}")
    print(f"  Difference (random - real):      {diff:+.2f}")

    if real_test < random_test:
        print(f"  PASS: Real RU assignment outperforms random (delta = {diff:.2f})")
    else:
        print(f"  FAIL: Random RU assignment is as good or better (delta = {diff:.2f})")

    return {'real_test_rmse': real_test, 'random_test_rmse': random_test, 'delta': diff}


# ============================================================================
# Diagnostic 3: Parameter-Matched Baseline (wide hidden_dim)
# ============================================================================

def diagnostic_parameter_matched(bipartite_params, baseline_results, matched_results):
    """Compare PolymerBipartite v2 vs parameter-matched baseline."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 3: Parameter-Matched Baseline (wide hidden_dim)")
    print("=" * 70)

    print(f"  PolymerBipartite v2 params:       {bipartite_params:,}")
    print(f"  Standard GIN+Trimer params:       {baseline_results['n_params']:,}")
    print(f"  Wide parameter-matched params:    {matched_results['n_params']:,}")

    bp_test = matched_results.get('test_rmse', 0)
    matched_test = matched_results['test_rmse']

    if matched_test < baseline_results['test_rmse']:
        print(f"  Note: Wide baseline ({matched_test:.2f}) outperforms standard baseline ({baseline_results['test_rmse']:.2f})")

    return {
        'bipartite_params': bipartite_params,
        'baseline_params': baseline_results['n_params'],
        'matched_params': matched_results['n_params'],
        'baseline_test_rmse': baseline_results['test_rmse'],
        'matched_test_rmse': matched_results['test_rmse'],
    }


# ============================================================================
# Diagnostic 4: Overfitting Gap Analysis
# ============================================================================

def diagnostic_overfitting_gap(all_results):
    """Report overfitting gaps for all models."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 4: Overfitting Gap Analysis")
    print("=" * 70)

    for name, results in all_results.items():
        train_rmse = results['train_rmse']
        test_rmse = results['test_rmse']
        gap = train_rmse - test_rmse
        print(f"  {name:30s} | Train: {train_rmse:6.2f} | Val: {results['val_rmse']:6.2f} | "
              f"Test: {test_rmse:6.2f} | Gap: {gap:+6.2f}")

    return {name: {
        'train_rmse': r['train_rmse'],
        'val_rmse': r['val_rmse'],
        'test_rmse': r['test_rmse'],
        'gap': r['train_rmse'] - r['test_rmse'],
    } for name, r in all_results.items()}


# ============================================================================
# Diagnostic 5: Multi-Seed Verification
# ============================================================================

def diagnostic_multi_seed(seed_results):
    """Report multi-seed results."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 5: Multi-Seed Verification")
    print("=" * 70)

    for model_name, seeds in seed_results.items():
        test_rmses = [s['test_rmse'] for s in seeds.values()]
        mean_rmse = np.mean(test_rmses)
        std_rmse = np.std(test_rmses)
        print(f"  {model_name:30s} | Seeds: {list(seeds.keys())}")
        for seed, r in seeds.items():
            print(f"    seed={seed}: Test RMSE = {r['test_rmse']:.2f}")
        print(f"    Mean +/- Std: {mean_rmse:.2f} +/- {std_rmse:.2f}")

    model_names = list(seed_results.keys())
    if len(model_names) >= 2:
        m1 = model_names[0]
        m2 = model_names[1]
        mean1 = np.mean([s['test_rmse'] for s in seed_results[m1].values()])
        mean2 = np.mean([s['test_rmse'] for s in seed_results[m2].values()])
        delta = mean2 - mean1
        print(f"\n  Difference ({m2} - {m1}): {delta:+.2f}")
        if delta > 0:
            print(f"  {m1} is better by {delta:.2f} RMSE on average")

    return seed_results


# ============================================================================
# Gate Value Distribution Analysis (scalar gates for iter2)
# ============================================================================

def analyze_gate_values(model, dataset, test_indices, device, batch_size=32):
    """
    Analyze gate values for iter2 (scalar gate params).
    Report learned boundary and interior gate values + per-atom distributions.
    """
    print("\n" + "=" * 70)
    print("GATE VALUE DISTRIBUTION ANALYSIS (iter2 scalar gates)")
    print("=" * 70)

    model.eval()
    valid_mask = dataset.valid_mask
    test_valid = [i for i in test_indices if valid_mask[i]]

    # Report scalar gate values directly
    b_scalar = model.boundary_gate_scalar.item()
    i_scalar = model.interior_gate_scalar.item()
    b_gate = torch.sigmoid(model.boundary_gate_scalar).item()
    i_gate = torch.sigmoid(model.interior_gate_scalar).item()

    print(f"  Learned boundary gate scalar: {b_scalar:.4f} -> sigmoid = {b_gate:.4f}")
    print(f"  Learned interior gate scalar: {i_scalar:.4f} -> sigmoid = {i_gate:.4f}")
    print(f"  Ratio (boundary/interior): {b_gate / max(i_gate, 1e-8):.2f}x")

    # Verify per-atom gate values match the scalars
    all_boundary_gates = []
    all_interior_gates = []

    with torch.no_grad():
        for i in range(0, len(test_valid), batch_size):
            bi = test_valid[i:i + batch_size]
            dd, _ = prepare_batch(dataset.graphs, dataset.targets, bi, device)
            _, _, gate_values_all = model(dd, return_gate_values=True)

            last_gates = gate_values_all[-1]
            g = last_gates['gate'].squeeze(-1)
            is_b = last_gates['is_boundary']

            boundary_mask = is_b > 0.5
            all_boundary_gates.append(g[boundary_mask].cpu())
            all_interior_gates.append(g[~boundary_mask].cpu())

    boundary_gates = torch.cat(all_boundary_gates)
    interior_gates = torch.cat(all_interior_gates)

    print(f"\n  Per-atom verification (last layer):")
    print(f"    Boundary atoms ({len(boundary_gates)}): mean={boundary_gates.mean():.4f}, std={boundary_gates.std():.4f}")
    print(f"    Interior atoms ({len(interior_gates)}): mean={interior_gates.mean():.4f}, std={interior_gates.std():.4f}")

    b_mean = boundary_gates.mean().item()
    i_mean = interior_gates.mean().item()

    if b_mean > 0.7 and i_mean < 0.5:
        print(f"  PASS: Boundary gates ({b_mean:.3f}) >> Interior gates ({i_mean:.3f})")
    elif b_mean > 0.9 and i_mean > 0.9:
        print(f"  WARN: All gates ~1.0 — gating collapsed (flat broadcast)")
    elif b_mean < 0.1 and i_mean < 0.1:
        print(f"  WARN: All gates ~0.0 — model ignores RU channel")
    else:
        print(f"  INFO: Boundary={b_mean:.3f}, Interior={i_mean:.3f}")

    # Per-layer gate scalars (all same for scalar params, but verify)
    print(f"\n  Per-layer gate scalars (should be identical):")
    with torch.no_grad():
        for layer_idx in range(5):
            all_b_g = []
            all_i_g = []
            for i in range(0, min(len(test_valid), 200), batch_size):
                bi = test_valid[i:i + batch_size]
                dd, _ = prepare_batch(dataset.graphs, dataset.targets, bi, device)
                _, _, gate_values_all = model(dd, return_gate_values=True)
                g = gate_values_all[layer_idx]['gate'].squeeze(-1)
                is_b = gate_values_all[layer_idx]['is_boundary']
                bm = is_b > 0.5
                all_b_g.append(g[bm].cpu())
                all_i_g.append(g[~bm].cpu())
            bg = torch.cat(all_b_g) if all_b_g else torch.tensor([])
            ig = torch.cat(all_i_g) if all_i_g else torch.tensor([])
            if len(bg) > 0 and len(ig) > 0:
                print(f"    Layer {layer_idx}: boundary={bg.mean():.4f}, interior={ig.mean():.4f}")

    return {
        'boundary_mean': b_mean,
        'boundary_std': boundary_gates.std().item(),
        'interior_mean': i_mean,
        'interior_std': interior_gates.std().item(),
        'n_boundary': len(boundary_gates),
        'n_interior': len(interior_gates),
        'boundary_scalar': b_scalar,
        'interior_scalar': i_scalar,
    }


# ============================================================================
# Summary Report
# ============================================================================

def write_results_summary(filepath, all_results):
    """Write comprehensive results summary to file."""
    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PolymerBipartite v2 Iteration 2 — Results Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. MAIN RESULT (seed=71)\n")
        f.write("-" * 40 + "\n")
        for name in ['PolymerBipartiteV2', 'GIN+Trimer']:
            if name in all_results:
                r = all_results[name]
                f.write(f"  {name}:\n")
                f.write(f"    Train RMSE: {r['train_rmse']:.2f}\n")
                f.write(f"    Val RMSE:   {r['val_rmse']:.2f}\n")
                f.write(f"    Test RMSE:  {r['test_rmse']:.2f}\n")
                f.write(f"    Overfitting gap: {r['overfitting_gap']:+.2f}\n")
                f.write(f"    Best epoch: {r['best_epoch']}\n")
                f.write(f"    Parameters: {r['n_params']:,}\n\n")

        f.write("2. MULTI-SEED RESULTS\n")
        f.write("-" * 40 + "\n")
        if 'multi_seed' in all_results:
            for model_name, seeds in all_results['multi_seed'].items():
                rmses = [s['test_rmse'] for s in seeds.values()]
                f.write(f"  {model_name}: {np.mean(rmses):.2f} +/- {np.std(rmses):.2f}\n")
                for s, r in seeds.items():
                    f.write(f"    seed={s}: {r['test_rmse']:.2f}\n")
                f.write("\n")

        f.write("3. DIAGNOSTICS\n")
        f.write("-" * 40 + "\n")

        if 'ru_discriminability' in all_results:
            d = all_results['ru_discriminability']
            f.write(f"  D1: RU Discriminability\n")
            f.write(f"    Mean cos_sim: {d['mean_cos_sim']:.4f}\n")
            f.write(f"    Median: {d['median_cos_sim']:.4f}\n")
            f.write(f"    Std: {d['std_cos_sim']:.4f}\n")
            f.write(f"    % > 0.95: {d['pct_above_95']:.1f}%\n\n")

        if 'random_ru' in all_results:
            d = all_results['random_ru']
            f.write(f"  D2: Random RU Control\n")
            f.write(f"    Real RMSE: {d['real_test_rmse']:.2f}\n")
            f.write(f"    Random RMSE: {d['random_test_rmse']:.2f}\n")
            f.write(f"    Delta: {d['delta']:+.2f}\n\n")

        if 'param_matched' in all_results:
            d = all_results['param_matched']
            f.write(f"  D3: Parameter-Matched Baseline\n")
            f.write(f"    Bipartite params: {d['bipartite_params']:,}\n")
            f.write(f"    Baseline params: {d['baseline_params']:,}\n")
            f.write(f"    Matched params: {d['matched_params']:,}\n")
            f.write(f"    Baseline test RMSE: {d['baseline_test_rmse']:.2f}\n")
            f.write(f"    Matched test RMSE: {d['matched_test_rmse']:.2f}\n\n")

        if 'gate_values' in all_results:
            g = all_results['gate_values']
            f.write(f"  Gate Value Distribution\n")
            f.write(f"    Boundary atoms: mean={g['boundary_mean']:.4f}, std={g['boundary_std']:.4f}\n")
            f.write(f"    Interior atoms: mean={g['interior_mean']:.4f}, std={g['interior_std']:.4f}\n")
            f.write(f"    Learned scalars: boundary={g['boundary_scalar']:.4f}, interior={g['interior_scalar']:.4f}\n\n")

        if 'overfitting' in all_results:
            f.write(f"  D4: Overfitting Gap Analysis\n")
            for name, r in all_results['overfitting'].items():
                f.write(f"    {name}: gap={r['gap']:+.2f}\n")
            f.write("\n")

    print(f"\nResults summary written to {filepath}")
