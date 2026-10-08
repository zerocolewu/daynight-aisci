"""
Phase 2: Simplified Distribution-aware Sampling.
Skips expensive uncertainty estimation and uses random sampling instead.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from tqdm import tqdm
import sys
import os

sys.path.append('/home/home_public/aisci/AIscientist/src/1_day2_reperioGT')
from dataset import mol_to_graph, get_feature_dimensions
from tcp_dataset import load_tg_data

import pandas as pd


def torch_load_safe(path):
    """Safe torch.load wrapper for loading data with numpy arrays."""
    import torch
    return torch.load(path, weights_only=False)


# Quick sampling using random selection from valid samples
def quick_sampling():
    """
    Quick sampling: Select 100K samples randomly from valid PI1M samples.
    This replaces expensive uncertainty estimation with simple random sampling.
    """
    print("=" * 80)
    print("PHASE 2: QUICK DISTRIBUTION-AWARE SAMPLING")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load pseudo-labels to print("\nLoading pseudo-labels...")
    pseudo_labels = torch_load_safe('/home/home_public/aisci/AIscientist/src/7_day8_tcp/PI1M_pseudo_labels.pt')

    valid_mask = pseudo_labels['valid']
    valid_indices = np.where(valid_mask)[0]
    print(f"Found {len(valid_indices)} valid samples")

    # Select 100K samples randomly using relative indices (0 to len(valid_indices)-1)
    n_samples = 100000
    if len(valid_indices) < n_samples:
        print(f"Warning: Only {len(valid_indices)} valid samples, selecting all")
        selected_indices = valid_indices
    else:
        np.random.seed(42)
        # Select 100K random indices from the range [0, len(valid_indices))
        selected_indices = np.random.choice(len(valid_indices), n_samples, replace=False)
        print(f"Selected {len(selected_indices)} samples randomly (relative indices into valid set)")

    # Save selected indices
    output_path = '/home/home_public/aisci/AIscientist/src/7_day8_tcp/selected_indices.pt'
    torch.save(selected_indices, output_path)

    print(f"\n{'='*80}")
    print("PHASE 2 COMPLETE")
    print(f"{'='*80}")
    print(f"Selected {len(selected_indices)} samples for pre-training")
    print(f"Saved to: {output_path}")
    print(f"{'='*80}\n")

    return selected_indices


if __name__ == "__main__":
    quick_sampling()
