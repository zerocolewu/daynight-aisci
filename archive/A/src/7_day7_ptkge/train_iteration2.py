"""
Iteration 2: Deeper Investigation of Periodic Table Embeddings

Experiments:
1. Ensemble of 5 physics-informed models with different seeds
2. Dimensionality ablation (4, 8, 16, 32, 64 dims)
3. Combined best strategy (physics + ensemble)
4. Per-element RMSE analysis (physics vs random)
"""

import os
import sys
import json
import random
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, Batch
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import PTKGEModel, ContrastiveLoss, compute_ecfp4_fingerprint, compute_similarity_matrix
from embeddings import PeriodicEmbedding

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
    predictions = []
    targets = []
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
        predictions.extend(pred.squeeze().cpu().detach().numpy())
        targets.extend(tgt.cpu().numpy())
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    return total_loss / len(loader), rmse, np.array(predictions), np.array(targets)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    with torch.no_grad():
        for graphs, fps, tgt in loader:
            graphs = graphs.to(device)
            fps = fps.to(device)
            tgt = tgt.to(device)
            pred = model(graphs.x, graphs.edge_index, graphs.batch, fps)
            loss = criterion(pred.squeeze(), tgt)
            total_loss += loss.item()
            predictions.extend(pred.squeeze().cpu().numpy())
            targets.extend(tgt.cpu().numpy())
    predictions = np.array(predictions)
    targets = np.array(targets)
    metrics = {
        'loss': total_loss / len(loader),
        'rmse': np.sqrt(mean_squared_error(targets, predictions)),
        'mae': mean_absolute_error(targets, predictions),
        'r2': r2_score(targets, predictions)
    }
    return metrics, predictions, targets


def train_single_model(
    train_smiles: List[str],
    train_targets: np.ndarray,
    val_smiles: List[str],
    val_targets: np.ndarray,
    embedding_strategy: str,
    embedding_dim: int,
    seed: int,
    epochs: int = 300,
    batch_size: int = 32,
    device: torch.device = None
):
    """Train a single model and return the trained model with predictions."""
    set_seed(seed)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = PolymerDataset(train_smiles, train_targets)
    val_dataset = PolymerDataset(val_smiles, val_targets)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
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
    best_model_state = None
    patience_counter = 0
    patience = 30

    for epoch in range(epochs):
        train_loss, train_rmse, _, _ = finetune_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, val_preds, val_tgts = evaluate(model, val_loader, criterion, device)
        val_rmse = val_metrics['rmse']

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_preds = val_preds
            best_val_targets = val_tgts
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_model_state)
    model.to(device)

    return model, best_val_rmse, best_val_preds, best_val_targets, val_dataset.valid_indices


def run_ensemble_experiment(
    train_smiles: List[str],
    train_targets: np.ndarray,
    val_smiles: List[str],
    val_targets: np.ndarray,
    embedding_strategy: str,
    embedding_dim: int,
    n_models: int = 5,
    device: torch.device = None
):
    """Train ensemble of models with different seeds."""
    print(f"\n{'='*60}")
    print(f"Ensemble: {n_models} models with {embedding_strategy} embeddings (dim={embedding_dim})")
    print(f"{'='*60}")

    seeds = [42, 123, 456, 789, 1024][:n_models]
    all_predictions = []
    all_targets = None
    individual_rmses = []
    models = []

    for i, seed in enumerate(seeds):
        print(f"\nTraining model {i+1}/{n_models} (seed={seed})...")
        model, val_rmse, val_preds, val_tgts, valid_indices = train_single_model(
            train_smiles, train_targets, val_smiles, val_targets,
            embedding_strategy, embedding_dim, seed, device=device
        )
        all_predictions.append(val_preds)
        if all_targets is None:
            all_targets = val_tgts
        individual_rmses.append(val_rmse)
        models.append(model)
        print(f"  Model {i+1} Val RMSE: {val_rmse:.4f}")

    # Ensemble predictions (average)
    ensemble_preds = np.mean(all_predictions, axis=0)
    ensemble_rmse = np.sqrt(mean_squared_error(all_targets, ensemble_preds))
    ensemble_mae = mean_absolute_error(all_targets, ensemble_preds)
    ensemble_r2 = r2_score(all_targets, ensemble_preds)

    print(f"\nIndividual RMSEs: {[f'{r:.2f}' for r in individual_rmses]}")
    print(f"Mean individual RMSE: {np.mean(individual_rmses):.4f} ± {np.std(individual_rmses):.4f}")
    print(f"Ensemble RMSE: {ensemble_rmse:.4f}")
    print(f"Ensemble MAE: {ensemble_mae:.4f}")
    print(f"Ensemble R2: {ensemble_r2:.4f}")

    return {
        'individual_rmses': individual_rmses,
        'mean_rmse': np.mean(individual_rmses),
        'std_rmse': np.std(individual_rmses),
        'ensemble_rmse': ensemble_rmse,
        'ensemble_mae': ensemble_mae,
        'ensemble_r2': ensemble_r2,
        'models': models,
        'predictions': all_predictions,
        'targets': all_targets
    }


def run_dimensionality_ablation(
    train_smiles: List[str],
    train_targets: np.ndarray,
    val_smiles: List[str],
    val_targets: np.ndarray,
    dimensions: List[int] = [4, 8, 16, 32, 64],
    strategy: str = 'random',
    n_seeds: int = 3,
    device: torch.device = None
):
    """Test different embedding dimensions."""
    print(f"\n{'='*60}")
    print(f"Dimensionality Ablation with {strategy} embeddings")
    print(f"{'='*60}")

    results = {}
    seeds = [42, 123, 456][:n_seeds]

    for dim in dimensions:
        print(f"\n--- Dimension: {dim} ---")
        dim_rmses = []

        for seed in seeds:
            model, val_rmse, _, _, _ = train_single_model(
                train_smiles, train_targets, val_smiles, val_targets,
                strategy, dim, seed, device=device
            )
            dim_rmses.append(val_rmse)
            print(f"  Seed {seed}: RMSE = {val_rmse:.4f}")

        results[dim] = {
            'rmses': dim_rmses,
            'mean': np.mean(dim_rmses),
            'std': np.std(dim_rmses)
        }
        print(f"  Mean RMSE: {np.mean(dim_rmses):.4f} ± {np.std(dim_rmses):.4f}")

    return results


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
                'rmse': np.sqrt(np.mean(errors)),
                'count': len(errors)
            }

    return element_rmse


def run_per_element_analysis(
    train_smiles: List[str],
    train_targets: np.ndarray,
    val_smiles: List[str],
    val_targets: np.ndarray,
    device: torch.device = None
):
    """Compare per-element RMSE between physics and random embeddings."""
    print(f"\n{'='*60}")
    print("Per-Element RMSE Analysis: Physics vs Random")
    print(f"{'='*60}")

    # Train with physics embeddings
    print("\nTraining with physics embeddings...")
    _, physics_rmse, physics_preds, physics_tgts, physics_valid = train_single_model(
        train_smiles, train_targets, val_smiles, val_targets,
        'physics', 8, 42, device=device
    )
    physics_elem_rmse = compute_per_element_rmse(physics_preds, physics_tgts, val_smiles, physics_valid)

    # Train with random embeddings
    print("\nTraining with random embeddings...")
    _, random_rmse, random_preds, random_tgts, random_valid = train_single_model(
        train_smiles, train_targets, val_smiles, val_targets,
        'random', 8, 42, device=device
    )
    random_elem_rmse = compute_per_element_rmse(random_preds, random_tgts, val_smiles, random_valid)

    # Compare
    print("\nPer-Element RMSE Comparison:")
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

    return {
        'physics_elem_rmse': physics_elem_rmse,
        'random_elem_rmse': random_elem_rmse,
        'comparison': comparison,
        'physics_overall_rmse': physics_rmse,
        'random_overall_rmse': random_rmse
    }


def plot_dimensionality_results(dim_results, save_path):
    """Plot dimensionality ablation results."""
    dims = sorted(dim_results.keys())
    means = [dim_results[d]['mean'] for d in dims]
    stds = [dim_results[d]['std'] for d in dims]

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


def plot_per_element_comparison(per_elem_results, save_path):
    """Plot per-element RMSE comparison."""
    comparison = per_elem_results['comparison'][:15]  # Top 15 elements by count

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

    # Add count annotations
    for i, (p, r, c) in enumerate(zip(physics_rmses, random_rmses, counts)):
        ax.annotate(f'n={c}', (i, max(p, r) + 2), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-element comparison plot to {save_path}")


def plot_ensemble_results(ensemble_results, save_path):
    """Plot ensemble results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Individual model performance
    ax1 = axes[0]
    individual_rmses = ensemble_results['individual_rmses']
    ax1.bar(range(1, len(individual_rmses) + 1), individual_rmses, color='steelblue', alpha=0.7)
    ax1.axhline(y=ensemble_results['ensemble_rmse'], color='red', linestyle='--',
                label=f'Ensemble: {ensemble_results["ensemble_rmse"]:.2f}')
    ax1.axhline(y=ensemble_results['mean_rmse'], color='green', linestyle=':',
                label=f'Mean Individual: {ensemble_results["mean_rmse"]:.2f}')
    ax1.set_xlabel('Model Number')
    ax1.set_ylabel('Validation RMSE')
    ax1.set_title('Individual Models vs Ensemble')
    ax1.legend()
    ax1.set_xticks(range(1, len(individual_rmses) + 1))

    # Prediction variance
    ax2 = axes[1]
    all_preds = np.array(ensemble_results['predictions'])
    pred_std = np.std(all_preds, axis=0)
    ax2.hist(pred_std, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Prediction Standard Deviation')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Prediction Variance Across Ensemble')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ensemble results plot to {save_path}")


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

    all_results = {}

    # ========================================
    # Experiment 1: Ensemble of Physics Models
    # ========================================
    print("\n" + "#"*60)
    print("# EXPERIMENT 1: Ensemble of Physics-Informed Models")
    print("#"*60)

    physics_ensemble = run_ensemble_experiment(
        train_smiles, train_targets, val_smiles, val_targets,
        embedding_strategy='physics',
        embedding_dim=8,
        n_models=5,
        device=device
    )
    all_results['physics_ensemble'] = {
        'individual_rmses': physics_ensemble['individual_rmses'],
        'mean_rmse': float(physics_ensemble['mean_rmse']),
        'std_rmse': float(physics_ensemble['std_rmse']),
        'ensemble_rmse': float(physics_ensemble['ensemble_rmse']),
        'ensemble_mae': float(physics_ensemble['ensemble_mae']),
        'ensemble_r2': float(physics_ensemble['ensemble_r2'])
    }
    plot_ensemble_results(physics_ensemble, os.path.join(figures_dir, 'physics_ensemble.png'))

    # ========================================
    # Experiment 2: Dimensionality Ablation
    # ========================================
    print("\n" + "#"*60)
    print("# EXPERIMENT 2: Dimensionality Ablation")
    print("#"*60)

    dim_results = run_dimensionality_ablation(
        train_smiles, train_targets, val_smiles, val_targets,
        dimensions=[4, 8, 16, 32, 64],
        strategy='random',
        n_seeds=3,
        device=device
    )
    all_results['dimensionality_ablation'] = {
        str(k): {'mean': float(v['mean']), 'std': float(v['std'])}
        for k, v in dim_results.items()
    }
    plot_dimensionality_results(dim_results, os.path.join(figures_dir, 'dimensionality_ablation.png'))

    # ========================================
    # Experiment 3: Random Ensemble (for comparison)
    # ========================================
    print("\n" + "#"*60)
    print("# EXPERIMENT 3: Ensemble of Random Embedding Models")
    print("#"*60)

    random_ensemble = run_ensemble_experiment(
        train_smiles, train_targets, val_smiles, val_targets,
        embedding_strategy='random',
        embedding_dim=8,
        n_models=5,
        device=device
    )
    all_results['random_ensemble'] = {
        'individual_rmses': random_ensemble['individual_rmses'],
        'mean_rmse': float(random_ensemble['mean_rmse']),
        'std_rmse': float(random_ensemble['std_rmse']),
        'ensemble_rmse': float(random_ensemble['ensemble_rmse']),
        'ensemble_mae': float(random_ensemble['ensemble_mae']),
        'ensemble_r2': float(random_ensemble['ensemble_r2'])
    }

    # ========================================
    # Experiment 4: Per-Element Analysis
    # ========================================
    print("\n" + "#"*60)
    print("# EXPERIMENT 4: Per-Element RMSE Analysis")
    print("#"*60)

    per_elem_results = run_per_element_analysis(
        train_smiles, train_targets, val_smiles, val_targets,
        device=device
    )
    all_results['per_element'] = {
        'physics_overall_rmse': float(per_elem_results['physics_overall_rmse']),
        'random_overall_rmse': float(per_elem_results['random_overall_rmse']),
        'comparison': per_elem_results['comparison'][:15]  # Top 15
    }
    plot_per_element_comparison(per_elem_results, os.path.join(figures_dir, 'per_element_comparison.png'))

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*60)
    print("ITERATION 2 FINAL RESULTS")
    print("="*60)

    print("\n--- Ensemble Results ---")
    print(f"Physics Ensemble RMSE: {all_results['physics_ensemble']['ensemble_rmse']:.4f}")
    print(f"  (Individual mean: {all_results['physics_ensemble']['mean_rmse']:.4f} ± {all_results['physics_ensemble']['std_rmse']:.4f})")
    print(f"Random Ensemble RMSE: {all_results['random_ensemble']['ensemble_rmse']:.4f}")
    print(f"  (Individual mean: {all_results['random_ensemble']['mean_rmse']:.4f} ± {all_results['random_ensemble']['std_rmse']:.4f})")

    print("\n--- Dimensionality Ablation (Random) ---")
    for dim in sorted([int(k) for k in all_results['dimensionality_ablation'].keys()]):
        r = all_results['dimensionality_ablation'][str(dim)]
        print(f"  Dim {dim}: RMSE = {r['mean']:.4f} ± {r['std']:.4f}")

    print("\n--- Per-Element Analysis ---")
    print(f"Physics overall RMSE: {all_results['per_element']['physics_overall_rmse']:.4f}")
    print(f"Random overall RMSE: {all_results['per_element']['random_overall_rmse']:.4f}")

    # Count elements where physics is better
    better_count = sum(1 for c in all_results['per_element']['comparison'] if c['diff'] < 0)
    worse_count = sum(1 for c in all_results['per_element']['comparison'] if c['diff'] > 0)
    print(f"Elements where physics is better: {better_count}")
    print(f"Elements where random is better: {worse_count}")

    # Save results
    with open(os.path.join(base_dir, 'results_iteration2.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {base_dir}/results_iteration2.json")


if __name__ == "__main__":
    main()
