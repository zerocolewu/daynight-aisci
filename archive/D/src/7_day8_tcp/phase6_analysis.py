"""
Phase 6: Analysis and comparison.

Computes:
- Thermodynamic consistency: % predictions satisfying Tg < Tm
- Comparison with baseline
- Comprehensive results summary
"""
import numpy as np
import torch
from torch_geometric.data import DataLoader
import sys
sys.path.append('/home/home_public/aisci/AIscientist/src/1_day2_reperioGT')
from dataset import mol_to_graph
from tcp_dataset import load_tg_data
from phase3_pretraining import TCPModel, evaluate_tcp
from phase4_finetuning import TCPFinetuneModel, BaselineModel

def torch_load_safe(path):
    """Safe torch.load wrapper for loading data with numpy arrays."""
    import torch
    return torch.load(path, weights_only=False)


def torch_load_state(path):
    """Safe torch.load wrapper for loading model state_dict."""
    import torch
    return torch.load(path, weights_only=False)



def compute_thermodynamic_consistency(model, data_loader, device):
    """
    Compute thermodynamic consistency metrics.

    Returns:
        - % predictions satisfying Tg < Tm
        - Tg/Tm ratio statistics
        - Egc correlation with Tg
    """
    model.eval()

    tg_preds, tm_preds, egc_preds = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            if hasattr(model, 'tg_head') and hasattr(model, 'tm_head'):
                # TCP model
                tg_pred, tm_pred, egc_pred = model(batch)
                tg_preds.extend(tg_pred.cpu().numpy())
                tm_preds.extend(tm_pred.cpu().numpy())
                egc_preds.extend(egc_pred.cpu().numpy())

    tg_preds = np.array(tg_preds)
    tm_preds = np.array(tm_preds)
    egc_preds = np.array(egc_preds)

    # Consistency metrics
    consistent = (tm_preds > tg_preds).sum()
    consistency_pct = 100 * consistent / len(tg_preds)

    ratio = tg_preds / tm_preds
    ratio_mean = ratio.mean()
    ratio_std = ratio.std()

    # Egc-Tg correlation
    egc_tg_corr = np.corrcoef(egc_preds, tg_preds)[0, 1]

    return {
        'consistency_pct': consistency_pct,
        'ratio_mean': ratio_mean,
        'ratio_std': ratio_std,
        'egc_tg_corr': egc_tg_corr,
        'tg_mean': tg_preds.mean(),
        'tg_std': tg_preds.std(),
        'tm_mean': tm_preds.mean(),
        'tm_std': tm_preds.std(),
        'egc_mean': egc_preds.mean(),
        'egc_std': egc_preds.std(),
    }


def phase6_analysis():
    """
    Complete Phase 6: Analysis and comparison.
    """
    print("=" * 80)
    print("PHASE 6: ANALYSIS AND COMPARISON")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load models
    print("\nLoading models...")

    # TCP pretrained model
    from dataset import get_feature_dimensions
    atom_dim, _ = get_feature_dimensions()

    tcp_model = TCPModel(atom_dim).to(device)
    checkpoint = torch_load_safe('/home/home_public/aisci/AIscientist/src/7_day8_tcp/TCP_pretrained_model.pt')
    tcp_model.load_state_dict(checkpoint['model_state_dict'])
    tcp_model.eval()
    print("  ✓ Loaded TCP pretrained model")

    # Load data for consistency analysis
    print("\nLoading PI1M validation data...")
    pseudo_labels = torch_load_safe('/home/home_public/aisci/AIscientist/src/7_day8_tcp/PI1M_pseudo_labels.pt')
    selected_indices = torch_load_safe('/home/home_public/aisci/AIscientist/src/7_day8_tcp/selected_indices.pt')

    import pandas as pd
    pi1m_df = pd.read_csv('/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv')

    valid_indices = np.where(pseudo_labels['valid'])[0]

    # Sample subset for analysis
    sample_indices = selected_indices[:10000]  # Use 10K for efficiency

    from tcp_dataset import PolymerMultiDataset
    selected_data = []
    selected_targets = []

    for idx in sample_indices:
        real_idx = valid_indices[idx]
        # Handle both 'SMILES' and 'Smiles' column names
        smiles = pi1m_df.iloc[real_idx].get('SMILES', pi1m_df.iloc[real_idx].get('Smiles', None))
        if smiles is None:
            continue
        data = mol_to_graph(smiles, target=None)
        if data is not None:
            selected_data.append(data)
            selected_targets.append(pseudo_labels['targets'][real_idx])

    selected_targets = np.array(selected_targets, dtype=np.float32)
    dataset = PolymerMultiDataset(selected_data, selected_targets)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=False)

    print(f"  Loaded {len(dataset)} samples for analysis")

    # Compute thermodynamic consistency
    print("\n" + "-" * 80)
    print("THERMODYNAMIC CONSISTENCY ANALYSIS")
    print("-" * 80)

    consistency_metrics = compute_thermodynamic_consistency(tcp_model, data_loader, device)

    print(f"\nTCP Model Thermodynamic Consistency:")
    print(f"  Tg < Tm: {consistency_metrics['consistency_pct']:.1f}%")
    print(f"  Tg/Tm ratio: {consistency_metrics['ratio_mean']:.3f} ± {consistency_metrics['ratio_std']:.3f}")
    print(f"  Egc-Tg correlation: {consistency_metrics['egc_tg_corr']:.3f}")
    print(f"\nPrediction Statistics:")
    print(f"  Tg: {consistency_metrics['tg_mean']:.1f} ± {consistency_metrics['tg_std']:.1f} K")
    print(f"  Tm: {consistency_metrics['tm_mean']:.1f} ± {consistency_metrics['tm_std']:.1f} K")
    print(f"  Egc: {consistency_metrics['egc_mean']:.1f} ± {consistency_metrics['egc_std']:.1f} kJ/mol")

    # Save thermodynamic consistency results
    with open('/home/home_public/aisci/AIscientist/src/7_day8_tcp/thermodynamic_consistency.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("THERMODYNAMIC CONSISTENCY ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write("TCP Model Results:\n")
        f.write(f"  Tg < Tm: {consistency_metrics['consistency_pct']:.1f}%\n")
        f.write(f"  Tg/Tm ratio: {consistency_metrics['ratio_mean']:.3f} ± {consistency_metrics['ratio_std']:.3f}\n")
        f.write(f"  Egc-Tg correlation: {consistency_metrics['egc_tg_corr']:.3f}\n\n")

        f.write("Prediction Statistics:\n")
        f.write(f"  Tg: {consistency_metrics['tg_mean']:.1f} ± {consistency_metrics['tg_std']:.1f} K\n")
        f.write(f"  Tm: {consistency_metrics['tm_mean']:.1f} ± {consistency_metrics['tm_std']:.1f} K\n")
        f.write(f"  Egc: {consistency_metrics['egc_mean']:.1f} ± {consistency_metrics['egc_std']:.1f} kJ/mol\n")

        f.write("\n" + "=" * 80 + "\n")

    # Load and compare results from previous phases
    print("\n" + "-" * 80)
    print("COMPARISON WITH BASELINE")
    print("-" * 80)

    # Read Phase 4 results
    with open('/home/home_public/aisci/AIscientist/src/7_day8_tcp/Tg_finetuned_results.txt', 'r') as f:
        phase4_content = f.read()

    # Read ablation results
    try:
        with open('/home/home_public/aisci/AIscientist/src/7_day8_tcp/ablation_results.txt', 'r') as f:
            ablation_content = f.read()
    except:
        ablation_content = "Not yet available\n"

    # Parse Phase 4 results
    import re
    tcp_match = re.search(r'TCP Fine-tuned Model:\s*\n\s*Mean RMSE: ([\d.]+) ± ([\d.]+) K', phase4_content)
    baseline_match = re.search(r'Baseline Model \(Train from Scratch\):\s*\n\s*Mean RMSE: ([\d.]+) ± ([\d.]+) K', phase4_content)

    if tcp_match and baseline_match:
        tcp_rmse = float(tcp_match.group(1))
        baseline_rmse = float(baseline_match.group(1))

        print(f"\nTg Prediction Performance:")
        print(f"  TCP Fine-tuned:  {tcp_rmse:.2f} K")
        print(f"  Baseline:        {baseline_rmse:.2f} K")
        print(f"  Improvement:     {baseline_rmse - tcp_rmse:+.2f} K ({100*(baseline_rmse - tcp_rmse)/baseline_rmse:+.1f}%)")

        if tcp_rmse < baseline_rmse:
            print(f"\n  ✓ TCP pre-training IMPROVES performance")
        else:
            print(f"\n  ✗ TCP pre-training shows NEGATIVE TRANSFER")

        print(f"\n  Target baseline RMSE: 32.31 K")
        print(f"  TCP model vs target: {32.31 - tcp_rmse:+.2f} K")

    # Save comparison results
    with open('/home/home_public/aisci/AIscientist/src/7_day8_tcp/comparison_with_baseline.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPARISON WITH BASELINE\n")
        f.write("=" * 80 + "\n\n")

        f.write(phase4_content)
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("ABLATION STUDIES\n")
        f.write("=" * 80 + "\n\n")
        f.write(ablation_content)
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("THERMODYNAMIC CONSISTENCY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Consistency (Tg < Tm): {consistency_metrics['consistency_pct']:.1f}%\n")
        f.write(f"Target baseline RMSE: 32.31 K\n")

        if tcp_match:
            f.write(f"\nTCP model RMSE: {tcp_rmse:.2f} K\n")
            f.write(f"Improvement over target: {32.31 - tcp_rmse:+.2f} K\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"\n{'='*80}")
    print("PHASE 6 COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved:")
    print(f"  - Thermodynamic consistency: thermodynamic_consistency.txt")
    print(f"  - Comparison with baseline: comparison_with_baseline.txt")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    phase6_analysis()
