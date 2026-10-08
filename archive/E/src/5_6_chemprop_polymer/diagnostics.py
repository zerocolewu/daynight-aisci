"""
Diagnostics for Chemprop-Polymer model analysis.

1. Periodic Edge Bond Message Discriminability (cos_sim diagnostic)
2. Overfitting Gap Analysis
3. Architecture Comparison
4. Progressive Improvement Analysis
5. Random Periodic Edge Control
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chemprop_model import ChempropPolymerMPNN, collate_batch
from construct_polymer_graphs import ATOM_FDIM, BOND_FDIM

WORK_DIR = '/home/home_public/aisci/AIscientist/src/5_6_chemprop_polymer'
MODEL_DIR = os.path.join(WORK_DIR, 'trained_models')


def load_best_model(ablation_name, seed=71, device='cuda'):
    """Load the best trained model for a given ablation and seed."""
    model_path = os.path.join(MODEL_DIR, f'{ablation_name}_seed{seed}.pt')
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None, None

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    hidden_dim = checkpoint.get('hidden_dim', 300)

    model = ChempropPolymerMPNN(
        atom_fdim=ATOM_FDIM,
        bond_fdim=BOND_FDIM,
        hidden_dim=hidden_dim,
        depth=3,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model, checkpoint


def extract_periodic_edge_messages(model, data, device='cuda'):
    """
    Extract periodic edge bond messages for all test polymers.

    Returns:
        periodic_messages: [n_polymers, hidden_dim] tensor of periodic edge forward messages
        periodic_messages_rev: [n_polymers, hidden_dim] tensor of periodic edge backward messages
    """
    model.eval()
    periodic_msgs = []
    periodic_msgs_rev = []
    valid_indices = []

    with torch.no_grad():
        for i, graph in enumerate(data):
            if len(graph.get('periodic_edge_bonds', [])) == 0:
                continue

            batch = collate_batch([graph])
            for k in ['atom_features', 'bond_features', 'edge_index', 'reverse_map',
                       'atom_batch', 'bond_batch', 'targets']:
                batch[k] = batch[k].to(device)

            pred, messages = model(batch, return_bond_messages=True)

            # Get periodic edge messages
            periodic_indices = graph['periodic_edge_bonds']
            if len(periodic_indices) >= 2:
                # Forward periodic message
                fwd_idx = periodic_indices[0]
                bwd_idx = periodic_indices[1]
                periodic_msgs.append(messages[fwd_idx].cpu().numpy())
                periodic_msgs_rev.append(messages[bwd_idx].cpu().numpy())
                valid_indices.append(i)

    if len(periodic_msgs) == 0:
        return None, None, []

    return (np.array(periodic_msgs),
            np.array(periodic_msgs_rev),
            valid_indices)


def compute_cosine_similarity_analysis(messages, labels=None):
    """
    Compute pairwise cosine similarity between message vectors.

    Returns dict with mean, std, min, max of pairwise cosine similarities.
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    if messages is None or len(messages) < 2:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'n': 0}

    # Subsample for efficiency if too many
    if len(messages) > 500:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(messages), 500, replace=False)
        messages = messages[idx]
        if labels is not None:
            labels = [labels[i] for i in idx]

    sim_matrix = cos_sim(messages)
    n = sim_matrix.shape[0]

    # Get upper triangle (exclude diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    sims = sim_matrix[mask]

    result = {
        'mean': float(np.mean(sims)),
        'std': float(np.std(sims)),
        'min': float(np.min(sims)),
        'max': float(np.max(sims)),
        'n': n,
    }

    # If labels provided, compute within-group and cross-group similarities
    if labels is not None:
        unique_labels = sorted(set(labels))
        if len(unique_labels) >= 2:
            for label in unique_labels:
                mask_label = [i for i, l in enumerate(labels) if l == label]
                if len(mask_label) >= 2:
                    sub_messages = messages[mask_label]
                    sub_sim = cos_sim(sub_messages)
                    sub_n = sub_sim.shape[0]
                    sub_mask = np.triu(np.ones((sub_n, sub_n), dtype=bool), k=1)
                    sub_sims = sub_sim[sub_mask]
                    result[f'within_{label}_mean'] = float(np.mean(sub_sims))
                    result[f'within_{label}_std'] = float(np.std(sub_sims))

    return result


def get_backbone_bond_type(graph):
    """Determine the backbone bond type for a polymer graph."""
    # Check periodic edge bond features (first 4 dims are bond type one-hot)
    for pe in graph.get('periodic_edge_bonds', []):
        bf = graph['bond_features'][pe]
        # bond_type: [SINGLE, DOUBLE, TRIPLE, AROMATIC]
        if bf[0] == 1:
            return 'SINGLE'
        elif bf[1] == 1:
            return 'DOUBLE'
        elif bf[2] == 1:
            return 'TRIPLE'
        elif bf[3] == 1:
            return 'AROMATIC'
    return 'UNKNOWN'


def run_diagnostics():
    """Run all diagnostic analyses."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("CHEMPROP-POLYMER DIAGNOSTIC REPORT")
    report_lines.append("=" * 70)

    # Load ablation results
    results_path = os.path.join(WORK_DIR, 'ablation_results.csv')
    summary_path = os.path.join(WORK_DIR, 'ablation_summary.csv')
    summaries_path = os.path.join(WORK_DIR, 'all_summaries.pkl')

    if not os.path.exists(results_path):
        print("No results found. Run train_ablations.py first.")
        return

    results_df = pd.read_csv(results_path)
    summary_df = pd.read_csv(summary_path)

    # ============================================================
    # 1. Periodic Edge Bond Message Discriminability
    # ============================================================
    report_lines.append("\n" + "=" * 70)
    report_lines.append("1. PERIODIC EDGE BOND MESSAGE DISCRIMINABILITY")
    report_lines.append("=" * 70)

    # Load polymer graph data for seed=71 (best validation seed)
    polymer_graphs_path = os.path.join(WORK_DIR, 'graphs_polymer.pkl')
    if os.path.exists(polymer_graphs_path):
        with open(polymer_graphs_path, 'rb') as f:
            polymer_data = pickle.load(f)

        # Load best model for full polymer (C)
        for seed in [42, 71, 123]:
            model, checkpoint = load_best_model('C_polymer_full', seed=seed, device=device)
            if model is not None:
                print(f"\nExtracting periodic edge messages (model C, seed={seed})...")
                messages_fwd, messages_bwd, valid_idx = extract_periodic_edge_messages(
                    model, polymer_data, device
                )

                if messages_fwd is not None:
                    # Overall cosine similarity
                    cos_result = compute_cosine_similarity_analysis(messages_fwd)
                    report_lines.append(f"\nSeed {seed} - Forward periodic edge messages:")
                    report_lines.append(f"  N polymers: {cos_result['n']}")
                    report_lines.append(f"  Mean pairwise cos_sim: {cos_result['mean']:.4f}")
                    report_lines.append(f"  Std: {cos_result['std']:.4f}")
                    report_lines.append(f"  Min: {cos_result['min']:.4f}")
                    report_lines.append(f"  Max: {cos_result['max']:.4f}")

                    # Stratify by backbone bond type
                    backbone_types = [get_backbone_bond_type(polymer_data[i]) for i in valid_idx]
                    cos_stratified = compute_cosine_similarity_analysis(
                        messages_fwd, labels=backbone_types
                    )
                    report_lines.append(f"  Stratified by backbone bond type:")
                    for bt in ['SINGLE', 'DOUBLE', 'AROMATIC', 'TRIPLE']:
                        key = f'within_{bt}_mean'
                        if key in cos_stratified:
                            report_lines.append(f"    {bt}: mean={cos_stratified[key]:.4f} ± {cos_stratified.get(f'within_{bt}_std', 0):.4f}")

                    # Backward messages
                    cos_result_bwd = compute_cosine_similarity_analysis(messages_bwd)
                    report_lines.append(f"\n  Backward periodic edge messages:")
                    report_lines.append(f"  Mean pairwise cos_sim: {cos_result_bwd['mean']:.4f}")
                    report_lines.append(f"  Std: {cos_result_bwd['std']:.4f}")

                    # Save cos_sim data
                    cos_sim_data = {
                        'seed': seed,
                        'fwd_mean': cos_result['mean'],
                        'fwd_std': cos_result['std'],
                        'fwd_min': cos_result['min'],
                        'fwd_max': cos_result['max'],
                        'bwd_mean': cos_result_bwd['mean'],
                        'bwd_std': cos_result_bwd['std'],
                    }
                    cos_df = pd.DataFrame([cos_sim_data])
                    cos_path = os.path.join(WORK_DIR, f'bond_message_cos_sim_seed{seed}.csv')
                    cos_df.to_csv(cos_path, index=False)

                    # Interpretation
                    if cos_result['mean'] > 0.95:
                        report_lines.append(f"\n  WARNING: cos_sim > 0.95 → periodic edge messages are NEAR-IDENTICAL")
                        report_lines.append(f"  Chemprop NOT utilizing periodic edge chemistry effectively")
                    elif cos_result['mean'] > 0.90:
                        report_lines.append(f"\n  CAUTION: cos_sim > 0.90 → moderate discriminability")
                    else:
                        report_lines.append(f"\n  GOOD: cos_sim < 0.90 → periodic edge messages are DISCRIMINATIVE")
                        report_lines.append(f"  Chemprop IS utilizing periodic edge chemistry")

                    break  # Use first available seed

    # ============================================================
    # 2. Overfitting Gap Analysis
    # ============================================================
    report_lines.append("\n" + "=" * 70)
    report_lines.append("2. OVERFITTING GAP ANALYSIS")
    report_lines.append("=" * 70)
    report_lines.append(f"\n{'Ablation':<25} {'Train RMSE':<15} {'Val RMSE':<15} {'Test RMSE':<15} {'Gap':<10}")
    report_lines.append("-" * 80)

    for _, row in summary_df.iterrows():
        name = row['ablation']
        train_rmse = row['train_rmse_mean']
        val_rmse = row['val_rmse_mean']
        test_rmse = row['test_rmse_mean']
        gap = row['overfitting_gap_mean']
        gap_std = row['overfitting_gap_std']
        report_lines.append(
            f"{name:<25} {train_rmse:<15.2f} {val_rmse:<15.2f} {test_rmse:<15.2f} {gap:.2f}±{gap_std:.2f}"
        )

    # ============================================================
    # 3. Architecture Comparison
    # ============================================================
    report_lines.append("\n" + "=" * 70)
    report_lines.append("3. ARCHITECTURE COMPARISON")
    report_lines.append("=" * 70)

    gin_trimer_ref = 50.19
    gin_polymer_ref = 53.28

    c_row = summary_df[summary_df['ablation'] == 'C_polymer_full']
    if len(c_row) > 0:
        c_rmse = c_row['test_rmse_mean'].values[0]
        c_std = c_row['test_rmse_std'].values[0]

        report_lines.append(f"\nChemprop+polymer (C):     {c_rmse:.2f} ± {c_std:.2f}")
        report_lines.append(f"GIN+trimer (reference):   {gin_trimer_ref:.2f}")
        report_lines.append(f"GIN+polymer (reference):  {gin_polymer_ref:.2f}")

        if c_rmse < gin_trimer_ref:
            gap = gin_trimer_ref - c_rmse
            report_lines.append(f"\n  Chemprop+polymer BEATS GIN+trimer by {gap:.2f} RMSE")
            report_lines.append(f"  → Architecture-periodicity compatibility CONFIRMED")
        else:
            gap = c_rmse - gin_trimer_ref
            report_lines.append(f"\n  Chemprop+polymer LOSES to GIN+trimer by {gap:.2f} RMSE")
            report_lines.append(f"  → Architecture-periodicity compatibility NOT confirmed")

        if c_rmse < gin_polymer_ref:
            gap = gin_polymer_ref - c_rmse
            report_lines.append(f"\n  Chemprop+polymer BEATS GIN+polymer by {gap:.2f} RMSE")
            report_lines.append(f"  → Directed bond messages handle periodicity better than atom-level aggregation")
        else:
            gap = c_rmse - gin_polymer_ref
            report_lines.append(f"\n  Chemprop+polymer LOSES to GIN+polymer by {gap:.2f} RMSE")

    # ============================================================
    # 4. Progressive Improvement Analysis
    # ============================================================
    report_lines.append("\n" + "=" * 70)
    report_lines.append("4. PROGRESSIVE IMPROVEMENT ANALYSIS")
    report_lines.append("=" * 70)

    a_row = summary_df[summary_df['ablation'] == 'A_monomer']
    b_row = summary_df[summary_df['ablation'] == 'B_trimer']

    if len(a_row) > 0 and len(b_row) > 0 and len(c_row) > 0:
        a_rmse = a_row['test_rmse_mean'].values[0]
        b_rmse = b_row['test_rmse_mean'].values[0]
        c_rmse = c_row['test_rmse_mean'].values[0]

        ab_gap = a_rmse - b_rmse
        bc_gap = b_rmse - c_rmse

        report_lines.append(f"\n  Monomer (A):    {a_rmse:.2f}")
        report_lines.append(f"  Trimer (B):     {b_rmse:.2f}  (improvement: {ab_gap:+.2f})")
        report_lines.append(f"  Polymer (C):    {c_rmse:.2f}  (improvement: {bc_gap:+.2f})")
        report_lines.append(f"\n  Monomer→Trimer: {'POSITIVE' if ab_gap > 0 else 'NEGATIVE'} ({ab_gap:+.2f})")
        report_lines.append(f"  Trimer→Polymer: {'POSITIVE' if bc_gap > 0 else 'NEGATIVE'} ({bc_gap:+.2f})")

    # ============================================================
    # 5. Random Periodic Edge Control
    # ============================================================
    report_lines.append("\n" + "=" * 70)
    report_lines.append("5. RANDOM PERIODIC EDGE CONTROL")
    report_lines.append("=" * 70)

    e_row = summary_df[summary_df['ablation'] == 'E_polymer_random']
    if len(e_row) > 0 and len(c_row) > 0:
        c_rmse = c_row['test_rmse_mean'].values[0]
        e_rmse = e_row['test_rmse_mean'].values[0]
        diff = e_rmse - c_rmse

        report_lines.append(f"\n  Correct periodic edge (C):    {c_rmse:.2f}")
        report_lines.append(f"  Random periodic edge (E):     {e_rmse:.2f}")
        report_lines.append(f"  Difference: {diff:+.2f}")

        if abs(diff) < 1.0:
            report_lines.append(f"\n  RESULT: Similar performance → specific chemistry does NOT matter")
            report_lines.append(f"  Extra connection alone provides the benefit")
        elif diff > 1.0:
            report_lines.append(f"\n  RESULT: Correct periodic edge significantly better")
            report_lines.append(f"  Specific inter-RU chemistry matters → H8f SUPPORTED")
        else:
            report_lines.append(f"\n  RESULT: Random periodic edge better (unexpected)")

    # ============================================================
    # 6. Minimal vs Full Periodic Features
    # ============================================================
    report_lines.append("\n" + "=" * 70)
    report_lines.append("6. MINIMAL vs FULL PERIODIC EDGE FEATURES")
    report_lines.append("=" * 70)

    d_row = summary_df[summary_df['ablation'] == 'D_polymer_minimal']
    if len(d_row) > 0 and len(c_row) > 0:
        c_rmse = c_row['test_rmse_mean'].values[0]
        d_rmse = d_row['test_rmse_mean'].values[0]
        diff = d_rmse - c_rmse

        report_lines.append(f"\n  Full features (C):     {c_rmse:.2f}")
        report_lines.append(f"  Minimal features (D):  {d_rmse:.2f}")
        report_lines.append(f"  Difference: {diff:+.2f}")

        if abs(diff) < 1.0:
            report_lines.append(f"\n  RESULT: Rich features NOT essential → topology marker sufficient")
        else:
            report_lines.append(f"\n  RESULT: Rich features help → H8f SUPPORTED")

    # ============================================================
    # 7. Hidden Dimension Comparison
    # ============================================================
    report_lines.append("\n" + "=" * 70)
    report_lines.append("7. HIDDEN DIMENSION COMPARISON (300 vs 256)")
    report_lines.append("=" * 70)

    f_row = summary_df[summary_df['ablation'] == 'F_polymer_256']
    if len(f_row) > 0 and len(c_row) > 0:
        c_rmse = c_row['test_rmse_mean'].values[0]
        f_rmse = f_row['test_rmse_mean'].values[0]
        diff = c_rmse - f_rmse

        report_lines.append(f"\n  hidden_dim=300 (C): {c_rmse:.2f}")
        report_lines.append(f"  hidden_dim=256 (F): {f_rmse:.2f}")
        report_lines.append(f"  Difference: {diff:+.2f}")

        if abs(diff) < 1.0:
            report_lines.append(f"\n  RESULT: Parameter count not the main factor")
        elif diff < 0:
            report_lines.append(f"\n  RESULT: 300-dim better → more capacity helps")
        else:
            report_lines.append(f"\n  RESULT: 256-dim better → overfitting concern with 300")

    # ============================================================
    # Write report
    # ============================================================
    report_text = "\n".join(report_lines)
    report_path = os.path.join(WORK_DIR, 'diagnostic_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(report_text)
    print(f"\nReport saved to {report_path}")


if __name__ == '__main__':
    run_diagnostics()
