"""
Periodic invariance test: Test if PIPGA is invariant to shifted cut points.
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import PolymerDataset, collate_fn, parse_polymer_smiles
from model import PIPGA
from torch.utils.data import DataLoader


def shift_polymer_cut_point(smiles):
    """
    Shift the cut point of a polymer SMILES.
    Example: "*CC*" -> "C*CC*C" (simplified shift)

    Args:
        smiles: Original polymer SMILES

    Returns:
        shifted_smiles: Polymer SMILES with shifted cut point
    """
    # Count connection points
    if smiles.count('*') < 2:
        return smiles

    # Find the positions of connection points
    parts = []
    current_part = ""
    for char in smiles:
        if char == '*':
            if current_part:
                parts.append(current_part)
            parts.append('*')
            current_part = ""
        else:
            current_part += char
    if current_part:
        parts.append(current_part)

    # Filter out empty parts
    parts = [p for p in parts if p]

    if len(parts) < 3:
        return smiles

    # Shift: move first non-* part to end
    # Find first non-* part
    first_non_star_idx = None
    for i, p in enumerate(parts):
        if p != '*':
            first_non_star_idx = i
            break

    if first_non_star_idx is None:
        return smiles

    # Create shifted version
    shifted_parts = parts.copy()
    part_to_move = shifted_parts.pop(first_non_star_idx)
    shifted_parts.append(part_to_move)

    shifted_smiles = ''.join(shifted_parts)
    return shifted_smiles


def test_periodic_invariance(model_path, test_data, device, distance_threshold=3):
    """
    Test if model predictions are invariant to shifted cut points.

    Args:
        model_path: Path to trained model checkpoint
        test_data: Dictionary with 'smiles' and 'tg' lists
        device: Torch device
        distance_threshold: Graph distance threshold

    Returns:
        Dictionary with invariance test results
    """
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    model = PIPGA(
        node_input_dim=23,
        node_dim=config['node_dim'],
        edge_dim=config['edge_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        use_softmax=config['use_softmax'],
        use_fingerprint=config['use_fingerprint'],
        use_self_loop=config['use_self_loop'],
        use_multi_edge=config['use_multi_edge']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Test on original and shifted SMILES
    original_predictions = []
    shifted_predictions = []
    valid_indices = []

    print(f"Testing periodic invariance on {len(test_data['smiles'])} samples...")

    for i, (smiles, tg) in enumerate(zip(test_data['smiles'][:100], test_data['tg'][:100])):
        try:
            # Original prediction
            dataset_orig = PolymerDataset([smiles], [tg], distance_threshold)
            loader_orig = DataLoader(dataset_orig, batch_size=1, collate_fn=collate_fn)
            batch_orig = next(iter(loader_orig))

            with torch.no_grad():
                pred_orig = model(
                    batch_orig['atom_features'].to(device),
                    batch_orig['edge_index'].to(device),
                    batch_orig['edge_offset'].to(device),
                    batch_orig['self_loop_features'].to(device),
                    batch_orig['fingerprint'].to(device),
                    batch_orig['batch'].to(device),
                    batch_orig['num_atoms']
                )

            # Shifted prediction
            shifted_smiles = shift_polymer_cut_point(smiles)
            dataset_shift = PolymerDataset([shifted_smiles], [tg], distance_threshold)
            loader_shift = DataLoader(dataset_shift, batch_size=1, collate_fn=collate_fn)
            batch_shift = next(iter(loader_shift))

            with torch.no_grad():
                pred_shift = model(
                    batch_shift['atom_features'].to(device),
                    batch_shift['edge_index'].to(device),
                    batch_shift['edge_offset'].to(device),
                    batch_shift['self_loop_features'].to(device),
                    batch_shift['fingerprint'].to(device),
                    batch_shift['batch'].to(device),
                    batch_shift['num_atoms']
                )

            original_predictions.append(pred_orig.item())
            shifted_predictions.append(pred_shift.item())
            valid_indices.append(i)

        except Exception as e:
            print(f"Warning: Failed to process sample {i}: {e}")
            continue

    original_predictions = np.array(original_predictions)
    shifted_predictions = np.array(shifted_predictions)

    # Compute metrics
    prediction_diff = np.abs(original_predictions - shifted_predictions)
    mean_diff = np.mean(prediction_diff)
    std_diff = np.std(prediction_diff)
    max_diff = np.max(prediction_diff)

    results = {
        'num_tested': len(valid_indices),
        'mean_prediction_diff': float(mean_diff),
        'std_prediction_diff': float(std_diff),
        'max_prediction_diff': float(max_diff),
        'original_predictions': original_predictions.tolist(),
        'shifted_predictions': shifted_predictions.tolist()
    }

    return results


def run_invariance_test_all_models(base_dir, test_data, device):
    """
    Run periodic invariance test on all models.
    """
    print("\n" + "="*60)
    print("Periodic Invariance Test")
    print("="*60)

    results = {}

    # Test full model
    model_configs = [
        ('Full PIPGA', 'full_model'),
        ('No Fingerprint', 'no_fingerprint'),
        ('Softmax Attention', 'softmax_attention'),
        ('No Self-loop', 'no_self_loop'),
        ('Single-edge', 'single_edge')
    ]

    for model_name, model_dir in model_configs:
        model_path = os.path.join(base_dir, 'results', model_dir, 'model', 'best_model_fold0.pt')

        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue

        print(f"\nTesting {model_name}...")
        result = test_periodic_invariance(model_path, test_data, device)

        results[model_name] = result

        print(f"  Samples tested: {result['num_tested']}")
        print(f"  Mean prediction difference: {result['mean_prediction_diff']:.2f} ± {result['std_prediction_diff']:.2f}")
        print(f"  Max prediction difference: {result['max_prediction_diff']:.2f}")

    # Save results
    results_path = os.path.join(base_dir, 'results', 'periodic_invariance_test.txt')
    with open(results_path, 'w') as f:
        f.write("Periodic Invariance Test Results\n")
        f.write("="*60 + "\n\n")
        f.write("Testing if model predictions are invariant to shifted cut points.\n")
        f.write("Lower difference = better periodic invariance.\n\n")

        for model_name, result in results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Samples tested: {result['num_tested']}\n")
            f.write(f"  Mean prediction difference: {result['mean_prediction_diff']:.2f} ± {result['std_prediction_diff']:.2f}\n")
            f.write(f"  Max prediction difference: {result['max_prediction_diff']:.2f}\n\n")

    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    # Load test data
    csv_path = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
    df = pd.read_csv(csv_path)

    # Use a subset for testing
    np.random.seed(42)
    test_indices = np.random.choice(len(df), size=200, replace=False)
    test_data = {
        'smiles': [df['Smiles'].iloc[i] for i in test_indices],
        'tg': [df['Tg'].iloc[i] for i in test_indices]
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = "/home/home_public/aisci/AIscientist/src/3_03_pippga"

    results = run_invariance_test_all_models(base_dir, test_data, device)

    print("\nPeriodic invariance test completed!")
