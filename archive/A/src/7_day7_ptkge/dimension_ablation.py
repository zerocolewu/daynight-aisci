"""
Dimensionality Ablation: Test random embeddings with different dimensions.
Tests dimensions: 4, 16, 32, 64 (8 is the baseline, already tested).
"""

import os
import sys
import json
import random
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import PTKGEModel, compute_ecfp4_fingerprint

warnings.filterwarnings('ignore')


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def polymer_smiles_to_mol(smiles: str):
    try:
        processed = smiles.replace('*', '[H]')
        mol = Chem.MolFromSmiles(processed)
        return mol
    except:
        return None


def mol_to_graph(mol: Chem.Mol):
    if mol is None:
        return None
    try:
        atoms = []
        for atom in mol.GetAtoms():
            atoms.append(atom.GetAtomicNum())
        if len(atoms) == 0:
            return None
        x = torch.tensor(atoms, dtype=torch.long)
        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.append([i, j])
            edges.append([j, i])
        if len(edges) == 0:
            edges = [[0, 0]]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)
    except:
        return None


class PolymerDataset(Dataset):
    def __init__(self, smiles_list: List[str], targets: Optional[np.ndarray] = None,
                 fp_dim: int = 2048):
        self.smiles_list = smiles_list
        self.targets = targets
        self.fp_dim = fp_dim
        self.graphs = []
        self.fingerprints = []
        self.valid_indices = []

        for i, smi in enumerate(smiles_list):
            mol = polymer_smiles_to_mol(smi)
            graph = mol_to_graph(mol) if mol else None
            fp = compute_ecfp4_fingerprint(smi, n_bits=fp_dim) if mol else None
            if graph is not None and fp is not None and fp.sum() > 0:
                self.graphs.append(graph)
                self.fingerprints.append(fp)
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        fp = self.fingerprints[idx]
        if self.targets is not None:
            target = self.targets[self.valid_indices[idx]]
            return graph, fp, target
        return graph, fp


def collate_fn(batch):
    if len(batch[0]) == 3:
        graphs, fps, targets = zip(*batch)
        targets = torch.tensor(np.array(targets), dtype=torch.float32)
    else:
        graphs, fps = zip(*batch)
        targets = None
    batched_graphs = Batch.from_data_list(graphs)
    fps = torch.tensor(np.array(fps), dtype=torch.float32)
    if targets is not None:
        return batched_graphs, fps, targets
    return batched_graphs, fps


def finetune_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for graphs, fps, tgt in loader:
        graphs = graphs.to(device)
        fps = fps.to(device)
        tgt = tgt.to(device)
        optimizer.zero_grad()
        pred = model(graphs.x, graphs.edge_index, graphs.batch, fps)
        loss = criterion(pred.squeeze(), tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for graphs, fps, tgt in loader:
            graphs = graphs.to(device)
            fps = fps.to(device)
            tgt = tgt.to(device)
            pred = model(graphs.x, graphs.edge_index, graphs.batch, fps)
            predictions.extend(pred.squeeze().cpu().numpy())
            targets.extend(tgt.cpu().numpy())
    predictions = np.array(predictions)
    targets = np.array(targets)
    metrics = {
        'rmse': np.sqrt(mean_squared_error(targets, predictions)),
        'mae': mean_absolute_error(targets, predictions),
        'r2': r2_score(targets, predictions)
    }
    return metrics


def train_single_model(
    train_smiles: List[str],
    train_targets: np.ndarray,
    val_smiles: List[str],
    val_targets: np.ndarray,
    embedding_strategy: str,
    embedding_dim: int,
    seed: int,
    device: torch.device
):
    """Train a single model and return validation RMSE."""
    set_seed(seed)

    train_dataset = PolymerDataset(train_smiles, train_targets)
    val_dataset = PolymerDataset(val_smiles, val_targets)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )

    model = PTKGEModel(
        num_layers=5,
        hidden_dim=256,
        embedding_strategy=embedding_strategy,
        embedding_dim=embedding_dim,
        dropout=0.1
    ).to(device)

    optimizer = AdamW([
        {'params': model.get_encoder_params(), 'lr': 1e-4},
        {'params': model.get_predictor_params(), 'lr': 1e-3}
    ], weight_decay=1e-5)

    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    patience_counter = 0
    patience = 30

    for epoch in range(300):
        train_loss = finetune_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_rmse = val_metrics['rmse']

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_rmse


def run_dimensionality_ablation(
    train_smiles: List[str],
    train_targets: np.ndarray,
    val_smiles: List[str],
    val_targets: np.ndarray,
    dimensions: List[int],
    n_seeds: int,
    device: torch.device
):
    """Test different embedding dimensions."""
    print(f"\n{'='*60}")
    print(f"Dimensionality Ablation with Random Embeddings")
    print(f"Dimensions: {dimensions}")
    print(f"Seeds per dimension: {n_seeds}")
    print(f"{'='*60}")

    results = {}
    seeds = [42, 123, 456][:n_seeds]

    for dim in dimensions:
        print(f"\n--- Dimension: {dim} ---")
        dim_rmses = []

        for seed in seeds:
            print(f"  Training with seed {seed}...")
            val_rmse = train_single_model(
                train_smiles, train_targets, val_smiles, val_targets,
                'random', dim, seed, device
            )
            dim_rmses.append(val_rmse)
            print(f"  Seed {seed}: RMSE = {val_rmse:.4f}")

        results[dim] = {
            'rmses': dim_rmses,
            'mean': float(np.mean(dim_rmses)),
            'std': float(np.std(dim_rmses))
        }
        print(f"  Mean RMSE: {np.mean(dim_rmses):.4f} +/- {np.std(dim_rmses):.4f}")

    return results


def plot_dimensionality_results(results, save_path):
    """Plot dimensionality ablation results."""
    dims = sorted(results.keys())
    means = [results[d]['mean'] for d in dims]
    stds = [results[d]['std'] for d in dims]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(dims, means, yerr=stds, fmt='o-', capsize=5, capthick=2, markersize=10)
    ax.set_xlabel('Embedding Dimension', fontsize=12)
    ax.set_ylabel('Validation RMSE', fontsize=12)
    ax.set_title('Effect of Embedding Dimension on Performance (Random Initialization)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    for d, m, s in zip(dims, means, stds):
        ax.annotate(f'{m:.2f}', (d, m), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved dimensionality plot to {save_path}")


def main():
    set_seed(42)

    base_dir = '/home/home_public/aisci/AIscientist/src/7_day7_ptkge'
    data_dir = '/home/home_public/aisci/AIscientist/src/alldataset'
    figures_dir = os.path.join(base_dir, 'figures_iter2')
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Tg dataset
    print("\nLoading Tg dataset...")
    tg_df = pd.read_csv(os.path.join(data_dir, 'Tg.csv'))
    tg_smiles = tg_df['Smiles'].tolist()
    tg_targets = tg_df['Tg'].values.astype(np.float32)
    print(f"Tg dataset: {len(tg_smiles)} samples")

    # Split into train/val (80/20)
    train_smiles, val_smiles, train_targets, val_targets = train_test_split(
        tg_smiles, tg_targets, test_size=0.2, random_state=42
    )
    print(f"Train: {len(train_smiles)}, Val: {len(val_smiles)}")

    # Run dimensionality ablation
    results = run_dimensionality_ablation(
        train_smiles, train_targets, val_smiles, val_targets,
        dimensions=[4, 16, 32, 64],
        n_seeds=3,
        device=device
    )

    # Save results
    output = {
        'dimensionality_ablation': {
            str(k): {'mean': v['mean'], 'std': v['std'], 'rmses': v['rmses']}
            for k, v in results.items()
        }
    }

    with open(os.path.join(base_dir, 'dimension_ablation_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {base_dir}/dimension_ablation_results.json")

    # Plot
    plot_dimensionality_results(results, os.path.join(figures_dir, 'dimensionality_ablation.png'))

    # Print summary
    print("\n" + "="*60)
    print("Dimensionality Ablation Summary:")
    print("="*60)
    for dim in sorted(results.keys()):
        r = results[dim]
        print(f"  Dim {dim}: RMSE = {r['mean']:.4f} +/- {r['std']:.4f}")

    return output


if __name__ == "__main__":
    main()
