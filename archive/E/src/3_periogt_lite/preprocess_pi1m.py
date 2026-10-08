"""
Preprocess PI1M dataset: generate monomer and trimer PyG graphs.
Saves preprocessed data to disk for efficient training.
"""

import os
import sys
import time
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_utils import psmiles_to_monomer, psmiles_to_trimer, mol_to_data

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'alldataset')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def preprocess_pi1m(subset_size=300000, seed=42):
    """Generate monomer and trimer graphs from PI1M p-SMILES."""
    print(f"Loading PI1M dataset...")
    csv_path = os.path.join(DATA_DIR, 'PI1M.csv')
    df = pd.read_csv(csv_path)
    print(f"Total PI1M entries: {len(df)}")

    # Sample subset
    if len(df) > subset_size:
        df = df.sample(n=subset_size, random_state=seed).reset_index(drop=True)
        print(f"Sampled {subset_size} entries for pre-training")

    smiles_list = df['SMILES'].tolist()

    # Process in batches and save
    mono_data_list = []
    tri_data_list = []
    failed = 0
    start = time.time()

    print("Generating monomer and trimer graphs...")
    for i, smi in enumerate(tqdm(smiles_list)):
        mono_mol = psmiles_to_monomer(smi)
        tri_mol = psmiles_to_trimer(smi)

        if mono_mol is None or tri_mol is None:
            failed += 1
            continue

        mono_data = mol_to_data(mono_mol)
        tri_data = mol_to_data(tri_mol)

        if mono_data is None or tri_data is None:
            failed += 1
            continue

        mono_data_list.append(mono_data)
        tri_data_list.append(tri_data)

    elapsed = time.time() - start
    print(f"\nProcessed {len(mono_data_list)} polymers successfully, {failed} failed")
    print(f"Time: {elapsed:.1f}s ({len(mono_data_list)/elapsed:.0f} mol/s)")

    # Save to disk
    mono_path = os.path.join(OUTPUT_DIR, 'pi1m_monomer_graphs.pt')
    tri_path = os.path.join(OUTPUT_DIR, 'pi1m_trimer_graphs.pt')
    torch.save(mono_data_list, mono_path)
    torch.save(tri_data_list, tri_path)
    print(f"Saved {len(mono_data_list)} monomer graphs to {mono_path}")
    print(f"Saved {len(tri_data_list)} trimer graphs to {tri_path}")

    # Stats
    mono_sizes = [d.num_nodes for d in mono_data_list]
    tri_sizes = [d.num_nodes for d in tri_data_list]
    print(f"\nMonomer: avg {np.mean(mono_sizes):.1f} atoms, "
          f"min {np.min(mono_sizes)}, max {np.max(mono_sizes)}")
    print(f"Trimer: avg {np.mean(tri_sizes):.1f} atoms, "
          f"min {np.min(tri_sizes)}, max {np.max(tri_sizes)}")


if __name__ == '__main__':
    preprocess_pi1m(subset_size=300000)
