"""
Per-Element Analysis: Compute RMSE for polymers containing each element.
Compares physics vs random embeddings.
"""

import os
import sys
import json
import random
import warnings
from collections import defaultdict
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
    return metrics, predictions, targets


def train_model(
    train_smiles: List[str],
    train_targets: np.ndarray,
    val_smiles: List[str],
    val_targets: np.ndarray,
    embedding_strategy: str,
    seed: int,
    device: torch.device
):
    """Train a single model."""
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
        embedding_dim=8,
        dropout=0.1
    ).to(device)

    optimizer = AdamW([
        {'params': model.get_encoder_params(), 'lr': 1e-4},
        {'params': model.get_predictor_params(), 'lr': 1e-3}
    ], weight_decay=1e-5)

    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 30

    for epoch in range(300):
        train_loss = finetune_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, val_preds, val_tgts = evaluate(model, val_loader, criterion, device)
        val_rmse = val_metrics['rmse']

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_preds = val_preds
            best_val_targets = val_tgts
            best_valid_indices = val_dataset.valid_indices
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_rmse, best_val_preds, best_val_targets, best_valid_indices


def compute_per_element_rmse(
    predictions: np.ndarray,
    targets: np.ndarray,
    smiles_list: List[str],
    valid_indices: List[int]
):
    """Compute per-element RMSE breakdown."""
    element_to_errors = defaultdict(list)
    element_to_count = defaultdict(int)

    for i, orig_idx in enumerate(valid_indices):
        smi = smiles_list[orig_idx]
        mol = polymer_smiles_to_mol(smi)
        if mol is None:
            continue

        elements = set()
        for atom in mol.GetAtoms():
            elements.add(atom.GetSymbol())

        error_sq = (predictions[i] - targets[i]) ** 2
        for elem in elements:
            element_to_errors[elem].append(error_sq)
            element_to_count[elem] += 1

    element_rmse = {}
    for elem, errors in element_to_errors.items():
        if len(errors) >= 5:
            element_rmse[elem] = {
                'rmse': float(np.sqrt(np.mean(errors))),
                'count': len(errors)
            }

    return element_rmse


def run_per_element_analysis(
    train_smiles: List[str],
    train_targets: np.ndarray,
    val_smiles: List[str],
    val_targets: np.ndarray,
    device: torch.device
):
    """Compare per-element RMSE between physics and random embeddings."""
    print(f"\n{'='*60}")
    print("Per-Element RMSE Analysis: Physics vs Random")
    print(f"{'='*60}")

    # Train with physics embeddings
    print("\nTraining with physics embeddings...")
    physics_rmse, physics_preds, physics_tgts, physics_valid = train_model(
        train_smiles, train_targets, val_smiles, val_targets,
        'physics', 42, device
    )
    print(f"Physics overall RMSE: {physics_rmse:.4f}")
    physics_elem_rmse = compute_per_element_rmse(physics_preds, physics_tgts, val_smiles, physics_valid)

    # Train with random embeddings
    print("\nTraining with random embeddings...")
    random_rmse, random_preds, random_tgts, random_valid = train_model(
        train_smiles, train_targets, val_smiles, val_targets,
        'random', 42, device
    )
    print(f"Random overall RMSE: {random_rmse:.4f}")
    random_elem_rmse = compute_per_element_rmse(random_preds, random_tgts, val_smiles, random_valid)

    # Compare
    print("\n" + "="*60)
    print("Per-Element RMSE Comparison:")
    print("="*60)
    print(f"{'Element':<8} {'Count':<8} {'Physics':<12} {'Random':<12} {'Diff':<12}")
    print("-" * 52)

    common_elements = set(physics_elem_rmse.keys()) & set(random_elem_rmse.keys())

    comparison = []
    for elem in sorted(common_elements, key=lambda x: physics_elem_rmse[x]['count'], reverse=True):
        p_rmse = physics_elem_rmse[elem]['rmse']
        r_rmse = random_elem_rmse[elem]['rmse']
        diff = p_rmse - r_rmse
        count = physics_elem_rmse[elem]['count']
        comparison.append({
            'element': elem,
            'count': count,
            'physics_rmse': p_rmse,
            'random_rmse': r_rmse,
            'diff': diff
        })
        print(f"{elem:<8} {count:<8} {p_rmse:<12.2f} {r_rmse:<12.2f} {diff:<+12.2f}")

    # Summary
    better_count = sum(1 for c in comparison if c['diff'] < 0)
    worse_count = sum(1 for c in comparison if c['diff'] > 0)
    print(f"\nElements where physics is better: {better_count}")
    print(f"Elements where random is better: {worse_count}")

    return {
        'physics_elem_rmse': physics_elem_rmse,
        'random_elem_rmse': random_elem_rmse,
        'comparison': comparison,
        'physics_overall_rmse': float(physics_rmse),
        'random_overall_rmse': float(random_rmse),
        'better_count': better_count,
        'worse_count': worse_count
    }


def plot_per_element_comparison(results, save_path):
    """Plot per-element RMSE comparison."""
    comparison = results['comparison'][:15]

    elements = [c['element'] for c in comparison]
    physics_rmses = [c['physics_rmse'] for c in comparison]
    random_rmses = [c['random_rmse'] for c in comparison]
    counts = [c['count'] for c in comparison]

    x = np.arange(len(elements))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, physics_rmses, width, label='Physics', color='steelblue')
    bars2 = ax.bar(x + width/2, random_rmses, width, label='Random', color='coral')

    ax.set_xlabel('Element')
    ax.set_ylabel('RMSE')
    ax.set_title('Per-Element RMSE: Physics vs Random Embeddings')
    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.legend()

    for i, (p, r, c) in enumerate(zip(physics_rmses, random_rmses, counts)):
        ax.annotate(f'n={c}', (i, max(p, r) + 2), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-element comparison plot to {save_path}")


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

    # Run analysis
    results = run_per_element_analysis(
        train_smiles, train_targets, val_smiles, val_targets, device
    )

    # Save results
    output = {
        'per_element': {
            'physics_overall_rmse': results['physics_overall_rmse'],
            'random_overall_rmse': results['random_overall_rmse'],
            'comparison': results['comparison'],
            'better_count': results['better_count'],
            'worse_count': results['worse_count']
        }
    }

    with open(os.path.join(base_dir, 'per_element_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {base_dir}/per_element_results.json")

    # Plot
    plot_per_element_comparison(results, os.path.join(figures_dir, 'per_element_comparison.png'))

    return output


if __name__ == "__main__":
    main()
