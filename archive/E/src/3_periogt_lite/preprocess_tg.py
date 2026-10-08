"""
Preprocess Tg dataset: generate trimer PyG graphs with train/val/test split.
Split: 72/8/20 with seed=71 (deterministic by index).
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_utils import psmiles_to_trimer, mol_to_data

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'alldataset')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def preprocess_tg():
    """Generate trimer graphs from Tg dataset and save with splits."""
    print("Loading Tg dataset...")
    csv_path = os.path.join(DATA_DIR, 'Tg.csv')
    df = pd.read_csv(csv_path)
    print(f"Total Tg entries: {len(df)}")

    smiles_list = df['Smiles'].tolist()
    tg_values = df['Tg'].tolist()

    # Deterministic split by index (72/8/20)
    n = len(smiles_list)
    n_train = int(n * 0.72)
    n_val = int(n * 0.08)

    # Process all molecules
    data_list = []
    valid_indices = []
    failed = 0

    print("Generating trimer graphs...")
    for i, (smi, tg) in enumerate(tqdm(zip(smiles_list, tg_values), total=n)):
        tri_mol = psmiles_to_trimer(smi)
        if tri_mol is None:
            failed += 1
            continue

        data = mol_to_data(tri_mol, target=tg)
        if data is None:
            failed += 1
            continue

        data_list.append(data)
        valid_indices.append(i)

    print(f"\nProcessed {len(data_list)} molecules successfully, {failed} failed")

    # Split based on original indices
    train_data, val_data, test_data = [], [], []
    for data, orig_idx in zip(data_list, valid_indices):
        if orig_idx < n_train:
            train_data.append(data)
        elif orig_idx < n_train + n_val:
            val_data.append(data)
        else:
            test_data.append(data)

    print(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Save
    torch.save(train_data, os.path.join(OUTPUT_DIR, 'tg_train.pt'))
    torch.save(val_data, os.path.join(OUTPUT_DIR, 'tg_val.pt'))
    torch.save(test_data, os.path.join(OUTPUT_DIR, 'tg_test.pt'))

    # Stats
    tg_all = [d.y.item() for d in data_list]
    print(f"\nTg stats: mean={np.mean(tg_all):.1f}, std={np.std(tg_all):.1f}, "
          f"min={np.min(tg_all):.1f}, max={np.max(tg_all):.1f}")

    sizes = [d.num_nodes for d in data_list]
    print(f"Trimer sizes: avg={np.mean(sizes):.1f}, min={np.min(sizes)}, max={np.max(sizes)}")


if __name__ == '__main__':
    preprocess_tg()
