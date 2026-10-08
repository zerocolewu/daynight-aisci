"""
Attention visualization for PIPGA model.
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import PolymerDataset, collate_fn
from model import PIPGA
from torch.utils.data import DataLoader


def visualize_attention_patterns(model_path, smiles_list, device, save_path,
                                 distance_threshold=3, num_samples=5):
    """
    Visualize attention patterns for a set of polymer SMILES.

    Args:
        model_path: Path to trained model checkpoint
        smiles_list: List of polymer SMILES to visualize
        device: Torch device
        save_path: Path to save visualization
        distance_threshold: Graph distance threshold
        num_samples: Number of samples to visualize
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

    fig, axes = plt.subplots(num_samples, config['num_layers'], figsize=(4*config['num_layers'], 3*num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for sample_idx, smiles in enumerate(smiles_list[:num_samples]):
        # Create dataset
        dataset = PolymerDataset([smiles], [0.0], distance_threshold)
        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
        batch = next(iter(loader))

        # Get attention weights
        with torch.no_grad():
            _, attention_weights = model(
                batch['atom_features'].to(device),
                batch['edge_index'].to(device),
                batch['edge_offset'].to(device),
                batch['self_loop_features'].to(device),
                batch['fingerprint'].to(device),
                batch['batch'].to(device),
                batch['num_atoms'],
                return_attention=True
            )

        # Plot attention for each layer
        num_atoms = batch['num_atoms'][0].item()
        for layer_idx, attn in enumerate(attention_weights):
            # Average over heads
            attn_mean = attn.mean(dim=1).cpu().numpy()

            # Create attention matrix
            edge_index = batch['edge_index'].cpu().numpy()
            attn_matrix = np.zeros((num_atoms, num_atoms))

            for e, (src, dst) in enumerate(edge_index.T):
                if src < num_atoms and dst < num_atoms:
                    attn_matrix[src, dst] = attn_mean[e]

            # Plot
            ax = axes[sample_idx, layer_idx]
            sns.heatmap(attn_matrix, ax=ax, cmap='viridis', cbar=True,
                       square=True, xticklabels=False, yticklabels=False)
            ax.set_title(f'Layer {layer_idx+1}', fontsize=10)
            ax.set_xlabel('Target', fontsize=8)
            ax.set_ylabel('Source', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Attention visualization saved to {save_path}")


def analyze_periodic_attention(model_path, smiles_list, device, save_path,
                               distance_threshold=3):
    """
    Analyze attention patterns on edges with different periodic offsets.

    Args:
        model_path: Path to trained model checkpoint
        smiles_list: List of polymer SMILES
        device: Torch device
        save_path: Path to save analysis
        distance_threshold: Graph distance threshold
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

    # Collect attention weights by periodic offset
    offset_attention = {-1: [], 0: [], 1: []}

    for smiles in smiles_list[:50]:
        try:
            dataset = PolymerDataset([smiles], [0.0], distance_threshold)
            loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
            batch = next(iter(loader))

            with torch.no_grad():
                _, attention_weights = model(
                    batch['atom_features'].to(device),
                    batch['edge_index'].to(device),
                    batch['edge_offset'].to(device),
                    batch['self_loop_features'].to(device),
                    batch['fingerprint'].to(device),
                    batch['batch'].to(device),
                    batch['num_atoms'],
                    return_attention=True
                )

            # Get edge offsets
            edge_offset = batch['edge_offset'].cpu().numpy()

            # Average attention over layers and heads
            for layer_attn in attention_weights:
                attn_mean = layer_attn.mean(dim=1).cpu().numpy()

                for offset in [-1, 0, 1]:
                    mask = edge_offset == offset
                    if mask.sum() > 0:
                        offset_attention[offset].extend(attn_mean[mask].tolist())

        except Exception as e:
            continue

    # Plot distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = [1, 2, 3]
    labels = ['k=-1 (left)', 'k=0 (center)', 'k=+1 (right)']
    colors = ['#ff7f7f', '#7fbf7f', '#7f7fff']

    for i, offset in enumerate([-1, 0, 1]):
        if len(offset_attention[offset]) > 0:
            bp = ax.boxplot([offset_attention[offset]], positions=[positions[i]],
                           widths=0.6, patch_artist=True)
            bp['boxes'][0].set_facecolor(colors[i])
            bp['boxes'][0].set_alpha(0.7)

    ax.set_xticklabels(labels)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_xlabel('Periodic Offset', fontsize=12)
    ax.set_title('Attention Distribution by Periodic Offset', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Periodic attention analysis saved to {save_path}")

    # Print statistics
    print("\nAttention statistics by periodic offset:")
    for offset in [-1, 0, 1]:
        if len(offset_attention[offset]) > 0:
            print(f"  k={offset}: mean={np.mean(offset_attention[offset]):.4f}, "
                  f"std={np.std(offset_attention[offset]):.4f}, "
                  f"n={len(offset_attention[offset])}")


def run_attention_analysis(base_dir, smiles_list, device):
    """
    Run attention analysis on the full PIPGA model.
    """
    print("\n" + "="*60)
    print("Attention Visualization and Analysis")
    print("="*60)

    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Full model path
    model_path = os.path.join(results_dir, 'full_model', 'model', 'best_model_fold0.pt')

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    # Visualize attention patterns
    print("\nVisualizing attention patterns...")
    visualize_attention_patterns(
        model_path, smiles_list[:5], device,
        os.path.join(results_dir, 'attention_analysis.png')
    )

    # Analyze periodic attention
    print("\nAnalyzing periodic attention distribution...")
    analyze_periodic_attention(
        model_path, smiles_list, device,
        os.path.join(results_dir, 'periodic_attention_analysis.png')
    )

    print("\nAttention analysis completed!")


if __name__ == "__main__":
    # Load data
    csv_path = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
    df = pd.read_csv(csv_path)
    smiles_list = df['Smiles'].tolist()[:100]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = "/home/home_public/aisci/AIscientist/src/3_03_pippga"

    run_attention_analysis(base_dir, smiles_list, device)
