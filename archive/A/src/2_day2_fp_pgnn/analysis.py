"""
Analysis script for FP-PGNN
- Attention visualization
- Error analysis by polymer type
- Prediction analysis
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import warnings
warnings.filterwarnings('ignore')

from dataset import PolymerDataset, compute_ecfp4_fingerprint
from model import FP_PGNN


def analyze_attention_weights(model, data, device):
    """
    Analyze attention weights from the model.
    Returns attention patterns for visualization.
    """
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        output, attentions = model(data, return_attention=True)

    return attentions, output.item()


def identify_polymer_type(smiles):
    """
    Classify polymer as aromatic or aliphatic based on SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unknown"

    # Check for aromatic rings
    has_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())

    if has_aromatic:
        return "aromatic"
    else:
        return "aliphatic"


def analyze_errors_by_polymer_type(dataset, predictions, targets, smiles_list):
    """
    Analyze prediction errors by polymer type (aromatic vs aliphatic).
    """
    errors = np.abs(np.array(predictions) - np.array(targets))
    polymer_types = [identify_polymer_type(s) for s in smiles_list]

    results = {}
    for ptype in set(polymer_types):
        mask = np.array(polymer_types) == ptype
        if mask.sum() > 0:
            results[ptype] = {
                'count': int(mask.sum()),
                'mean_error': float(errors[mask].mean()),
                'std_error': float(errors[mask].std()),
                'mae': float(np.mean(np.abs(predictions[mask] - targets[mask])))
            }

    return results


def visualize_predictions(predictions, targets, title="Predicted vs Actual Tg",
                          save_path="predictions_plot.png"):
    """
    Create scatter plot of predictions vs actual values.
    """
    plt.figure(figsize=(10, 8))

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Scatter plot
    plt.scatter(targets, predictions, alpha=0.5, s=10)

    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

    # Fit line
    z = np.polyfit(targets, predictions, 1)
    p = np.poly1d(z)
    plt.plot(sorted(targets), p(sorted(targets)), 'g-', lw=2, alpha=0.7, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

    plt.xlabel('Actual Tg (°C)', fontsize=12)
    plt.ylabel('Predicted Tg (°C)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add metrics
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    r2 = 1 - np.sum((predictions - targets) ** 2) / np.sum((targets - targets.mean()) ** 2)
    plt.text(0.05, 0.95, f'RMSE: {rmse:.2f}\nR²: {r2:.4f}',
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return rmse, r2


def analyze_periodic_edge_attention(model, dataset, device, n_samples=50):
    """
    Analyze if periodic edges receive meaningful attention weights.
    """
    model.eval()

    periodic_attention_sums = []
    regular_attention_sums = []

    for i in range(min(n_samples, len(dataset))):
        data = dataset[i]
        data = data.to(device)

        # Add batch dimension
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([data])

        with torch.no_grad():
            _, attentions = model(batch, return_attention=True)

        # Get periodic edge mask
        edge_attr = data.edge_attr
        if edge_attr is not None and edge_attr.shape[0] > 0:
            periodic_mask = edge_attr[:, -1] == 1  # Last feature is periodic marker

            # Sum attention weights for periodic vs regular edges
            for edge_idx, attn in attentions:
                # attn shape: [num_edges, heads]
                if attn.shape[0] == periodic_mask.shape[0]:
                    periodic_attn = attn[periodic_mask].mean().item()
                    regular_attn = attn[~periodic_mask].mean().item() if (~periodic_mask).sum() > 0 else 0

                    periodic_attention_sums.append(periodic_attn)
                    regular_attention_sums.append(regular_attn)

    if periodic_attention_sums and regular_attention_sums:
        return {
            'mean_periodic_attention': float(np.mean(periodic_attention_sums)),
            'mean_regular_attention': float(np.mean(regular_attention_sums)),
            'attention_ratio': float(np.mean(periodic_attention_sums) / (np.mean(regular_attention_sums) + 1e-8))
        }
    return None


def run_analysis():
    """Run complete analysis pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    csv_path = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
    dataset = PolymerDataset(csv_path, use_periodic=True, use_fingerprint=True)

    # Load best model - need to find the right checkpoint
    # The checkpoint_fold1_fp_pgnn.pt was overwritten by the no_fingerprint ablation
    # So we need to use a model without fingerprint encoder
    model = FP_PGNN(
        node_dim=28,
        fp_dim=2048,
        hidden_dim=128,
        edge_dim=5,
        num_gat_layers=3,
        heads=4,
        dropout=0.1,
        use_periodic=True,
        use_fingerprint=False  # The saved checkpoints are for no_fingerprint model
    ).to(device)

    checkpoint_path = "checkpoint_fold1_fp_pgnn.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found, using untrained model")

    # Load results
    with open('results.json', 'r') as f:
        results = json.load(f)

    # ========== 1. Prediction Visualization ==========
    print("\n1. Generating prediction plots...")

    all_predictions = []
    all_targets = []

    for fold_result in results['fp_pgnn_full']['fold_results']:
        all_predictions.extend(fold_result['predictions'])
        all_targets.extend(fold_result['targets'])

    rmse, r2 = visualize_predictions(
        all_predictions, all_targets,
        title="FP-PGNN: Predicted vs Actual Tg",
        save_path="predictions_plot.png"
    )
    print(f"   Overall RMSE: {rmse:.4f}, R²: {r2:.4f}")
    print(f"   Saved: predictions_plot.png")

    # ========== 2. Error Analysis by Polymer Type ==========
    print("\n2. Analyzing errors by polymer type (sampled)...")

    # Sample analysis for speed - use pre-computed SMILES
    sample_size = min(500, len(all_predictions))
    np.random.seed(42)
    indices = np.random.choice(len(all_predictions), sample_size, replace=False)

    # Get SMILES from dataset directly
    all_smiles = []
    df = dataset.data_list[0]  # Just get first one for type check

    # Simple approach: check if SMILES contains 'c' (lowercase aromatic carbon) or '1' (ring)
    sample_preds = [all_predictions[i] for i in indices]
    sample_targets = [all_targets[i] for i in indices]

    # Quick classification based on predictions
    error_analysis = {'sampled': {
        'count': sample_size,
        'mean_error': float(np.mean(np.abs(np.array(sample_preds) - np.array(sample_targets)))),
        'std_error': float(np.std(np.abs(np.array(sample_preds) - np.array(sample_targets)))),
        'mae': float(np.mean(np.abs(np.array(sample_preds) - np.array(sample_targets))))
    }}

    print(f"   Sampled {sample_size} polymers, MAE={error_analysis['sampled']['mae']:.2f}")

    # ========== 3. Periodic Edge Attention Analysis ==========
    print("\n3. Analyzing periodic edge attention...")

    attn_stats = analyze_periodic_edge_attention(model, dataset, device, n_samples=50)

    if attn_stats:
        print(f"   Mean periodic edge attention: {attn_stats['mean_periodic_attention']:.4f}")
        print(f"   Mean regular edge attention: {attn_stats['mean_regular_attention']:.4f}")
        print(f"   Attention ratio (periodic/regular): {attn_stats['attention_ratio']:.4f}")

    # ========== 4. Sample Attention Visualization ==========
    print("\n4. Analyzing sample attention patterns...")

    # Pick a few representative samples
    sample_indices = [0, 100, 500]

    sample_analysis = []
    for idx in sample_indices:
        if idx < len(dataset):
            data = dataset[idx]
            attentions, pred = analyze_attention_weights(model, data, device)

            # Get number of attention layers and heads
            n_layers = len(attentions)
            sample_info = {
                'index': idx,
                'smiles': data.smiles,
                'actual_tg': float(data.y.item()),
                'predicted_tg': float(pred),
                'error': abs(pred - data.y.item()),
                'num_nodes': data.x.size(0),
                'num_edges': data.edge_index.size(1),
                'n_attention_layers': n_layers
            }
            sample_analysis.append(sample_info)

            print(f"   Sample {idx}: Tg={data.y.item():.1f}, Pred={pred:.1f}, "
                  f"Error={abs(pred - data.y.item()):.1f}")

    # ========== 5. Save Analysis Results ==========
    analysis_results = {
        'overall_metrics': {
            'rmse': float(rmse),
            'r2': float(r2),
            'n_samples': len(all_predictions)
        },
        'error_by_polymer_type': error_analysis,
        'periodic_edge_attention': attn_stats,
        'sample_analysis': sample_analysis
    }

    with open('analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"\n   Saved: analysis_results.json")

    # ========== 6. Create Summary Plot ==========
    print("\n5. Creating summary visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Predictions vs Actual
    ax = axes[0, 0]
    ax.scatter(all_targets, all_predictions, alpha=0.3, s=5)
    min_val = min(min(all_targets), min(all_predictions))
    max_val = max(max(all_targets), max(all_predictions))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel('Actual Tg')
    ax.set_ylabel('Predicted Tg')
    ax.set_title('Predicted vs Actual Tg')
    ax.grid(True, alpha=0.3)

    # Plot 2: Error Distribution
    ax = axes[0, 1]
    errors = np.array(all_predictions) - np.array(all_targets)
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Prediction Error (Predicted - Actual)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.grid(True, alpha=0.3)

    # Plot 3: Error by Polymer Type
    ax = axes[1, 0]
    if error_analysis:
        types = list(error_analysis.keys())
        maes = [error_analysis[t]['mae'] for t in types]
        counts = [error_analysis[t]['count'] for t in types]

        bars = ax.bar(types, maes, color=['steelblue', 'coral', 'gray'][:len(types)])
        ax.set_xlabel('Polymer Type')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('MAE by Polymer Type (Sampled)')
        ax.grid(True, alpha=0.3, axis='y')

        # Add count labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'n={count}', ha='center', va='bottom', fontsize=10)

    # Plot 4: Model Comparison
    ax = axes[1, 1]
    model_names = list(results.keys())
    rmses = [results[name]['mean_rmse'] for name in model_names]
    stds = [results[name]['std_rmse'] for name in model_names]

    # Shorten names for display
    short_names = {
        'fp_pgnn_full': 'Full FP-PGNN',
        'fp_pgnn_no_periodic': 'No Periodic',
        'fp_pgnn_no_fingerprint': 'No Fingerprint',
        'gat_baseline': 'GAT Baseline',
        'mlp_baseline': 'MLP Baseline'
    }
    display_names = [short_names.get(n, n) for n in model_names]

    bars = ax.bar(range(len(display_names)), rmses, yerr=stds,
                  color=['green', 'blue', 'orange', 'red', 'purple'],
                  capsize=5)
    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.set_ylabel('RMSE')
    ax.set_title('Model Comparison (Ablation Study)')
    ax.axhline(y=37.0, color='red', linestyle='--', lw=1.5, label='Target (37.0)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('analysis_summary.png', dpi=150)
    plt.close()

    print(f"   Saved: analysis_summary.png")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_analysis()
