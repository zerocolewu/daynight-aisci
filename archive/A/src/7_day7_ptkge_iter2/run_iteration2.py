"""
Day 7 PTKGE Iteration 2: Ensemble Analysis, Per-Element Analysis, and Dimensionality Ablation

This script runs three analyses:
1. Ensemble Analysis - Train 5 physics models with different seeds and ensemble them
2. Per-Element Analysis - Compare physics vs random embeddings for different elements
3. Dimensionality Ablation - Test random embeddings with dimensions 4, 8, 16, 32, 64
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

# Import from copied files
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PTKGEModel, compute_ecfp4_fingerprint
from embeddings import PeriodicEmbedding

warnings.filterwarnings('ignore')


# ========== Utility Functions ==========

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def polymer_smiles_to_mol(smiles: str):
    """Convert polymer SMILES to RDKit mol."""
    try:
        processed = smiles.replace('*', '[H]')
        mol = Chem.MolFromSmiles(processed)
        return mol
    except:
        return None


def mol_to_graph(mol: Chem.Mol):
    """Convert RDKit mol to PyTorch Geometric graph."""
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


# ========== Dataset ==========

class PolymerDataset(Dataset):
    """Dataset for polymer SMILES."""

    def __init__(self, smiles_list: List[str], targets: Optional[np.ndarray] = None,
                 fp_dim: int = 2048):
        self.smiles_list = smiles_list
        self.targets = targets
        self.fp_dim = fp_dim

        self.graphs = []
        self.fingerprints = []
        self.valid_indices = []

        print(f"Processing {len(smiles_list)} samples...")
        for i, smi in enumerate(smiles_list):
            mol = polymer_smiles_to_mol(smi)
            graph = mol_to_graph(mol) if mol else None
            fp = compute_ecfp4_fingerprint(smi, n_bits=fp_dim) if mol else None

            if graph is not None and fp is not None and fp.sum() > 0:
                self.graphs.append(graph)
                self.fingerprints.append(fp)
                self.valid_indices.append(i)

        print(f"Valid samples: {len(self.valid_indices)} / {len(smiles_list)}")

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
    """Custom collate function for batching graphs."""
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


# ========== Training Functions ==========

def train_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch."""
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
    return total_loss / len(loader), rmse


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
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


def train_model(
    train_smiles: List[str],
    train_targets: np.ndarray,
    val_smiles: List[str],
    val_targets: np.ndarray,
    embedding_strategy: str,
    embedding_dim: int = 8,
    epochs: int = 150,
    batch_size: int = 32,
    lr: float = 1e-4,
    patience: int = 20,
    hidden_dim: int = 256,
    num_layers: int = 5,
    device: torch.device = None,
    seed: int = 42
):
    """Train a single model with early stopping."""

    set_seed(seed)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets
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

    # Create model
    model = PTKGEModel(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        embedding_strategy=embedding_strategy,
        embedding_dim=embedding_dim,
        dropout=0.1
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Training loop with early stopping
    best_val_rmse = float('inf')
    best_model_state = None
    best_predictions = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_rmse = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, val_preds, val_tgts = evaluate(model, val_loader, criterion, device)

        val_rmse = val_metrics['rmse']

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train RMSE: {train_rmse:.4f} - Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_predictions = val_preds.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_model_state)
    model.to(device)

    # Final evaluation
    final_metrics, final_preds, final_tgts = evaluate(model, val_loader, criterion, device)

    return model, final_metrics, final_preds, final_tgts


# ========== Task 1: Ensemble Analysis ==========

def run_ensemble_analysis(
    smiles_list: List[str],
    targets: np.ndarray,
    n_models: int = 5,
    save_dir: str = './',
    device: torch.device = None
):
    """
    Train multiple physics-informed models with different seeds and ensemble them.
    """
    print("\n" + "="*60)
    print("TASK 1: ENSEMBLE ANALYSIS")
    print("="*60)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split data
    train_idx, test_idx = train_test_split(
        range(len(smiles_list)), test_size=0.2, random_state=42
    )

    train_smiles = [smiles_list[i] for i in train_idx]
    test_smiles = [smiles_list[i] for i in test_idx]
    train_targets = targets[train_idx]
    test_targets = targets[test_idx]

    # Further split train into train/val
    train_idx2, val_idx = train_test_split(
        range(len(train_smiles)), test_size=0.125, random_state=42  # 0.125 * 0.8 = 0.1
    )

    final_train_smiles = [train_smiles[i] for i in train_idx2]
    val_smiles = [train_smiles[i] for i in val_idx]
    final_train_targets = train_targets[train_idx2]
    val_targets = train_targets[val_idx]

    print(f"Train size: {len(final_train_smiles)}")
    print(f"Val size: {len(val_smiles)}")
    print(f"Test size: {len(test_smiles)}")

    # Train multiple models with different seeds
    all_predictions = []
    individual_results = []

    for i in range(n_models):
        print(f"\n--- Training Model {i+1}/{n_models} (seed={42+i*100}) ---")

        model, metrics, val_preds, val_tgts = train_model(
            final_train_smiles, final_train_targets,
            val_smiles, val_targets,
            embedding_strategy='physics',
            embedding_dim=8,
            seed=42 + i * 100,
            device=device
        )

        print(f"Model {i+1} Val RMSE: {metrics['rmse']:.4f}")

        # Evaluate on test set
        test_dataset = PolymerDataset(test_smiles, test_targets)
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False,
            collate_fn=collate_fn, num_workers=4, pin_memory=True
        )

        test_metrics, test_preds, test_tgts = evaluate(model, test_loader, nn.MSELoss(), device)

        all_predictions.append(test_preds)
        individual_results.append({
            'model_id': i+1,
            'seed': 42 + i * 100,
            'val_rmse': metrics['rmse'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'test_r2': test_metrics['r2']
        })

        print(f"Model {i+1} Test RMSE: {test_metrics['rmse']:.4f}")

    # Ensemble predictions (average)
    ensemble_predictions = np.mean(all_predictions, axis=0)

    # Compute ensemble metrics
    ensemble_rmse = np.sqrt(mean_squared_error(test_targets, ensemble_predictions))
    ensemble_mae = mean_absolute_error(test_targets, ensemble_predictions)
    ensemble_r2 = r2_score(test_targets, ensemble_predictions)

    print("\n" + "="*60)
    print("ENSEMBLE RESULTS")
    print("="*60)

    print("\nIndividual Model Performance:")
    for res in individual_results:
        print(f"  Model {res['model_id']}: Test RMSE = {res['test_rmse']:.4f}")

    print(f"\nEnsemble Performance:")
    print(f"  RMSE: {ensemble_rmse:.4f}")
    print(f"  MAE: {ensemble_mae:.4f}")
    print(f"  R2: {ensemble_r2:.4f}")

    avg_individual_rmse = np.mean([r['test_rmse'] for r in individual_results])
    print(f"\nImprovement over average individual: {avg_individual_rmse - ensemble_rmse:.4f}")

    # Save results
    results = {
        'individual_models': individual_results,
        'ensemble': {
            'rmse': ensemble_rmse,
            'mae': ensemble_mae,
            'r2': ensemble_r2
        },
        'average_individual_rmse': avg_individual_rmse,
        'improvement': avg_individual_rmse - ensemble_rmse
    }

    with open(os.path.join(save_dir, 'ensemble_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    model_ids = [f"Model {i+1}" for i in range(n_models)] + ['Ensemble']
    rmses = [r['test_rmse'] for r in individual_results] + [ensemble_rmse]

    colors = ['steelblue'] * n_models + ['red']
    bars = ax.bar(model_ids, rmses, color=colors)

    ax.axhline(y=avg_individual_rmse, color='gray', linestyle='--', label=f'Avg Individual: {avg_individual_rmse:.2f}')
    ax.set_ylabel('Test RMSE')
    ax.set_title('Ensemble vs Individual Models (Physics Embeddings)')
    ax.legend()

    # Add value labels
    for bar, rmse in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rmse:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ensemble_comparison.png'), dpi=150)
    plt.close()

    print(f"\nResults saved to {save_dir}/ensemble_results.json")
    print(f"Plot saved to {save_dir}/ensemble_comparison.png")

    return results


# ========== Task 2: Per-Element Analysis ==========

def get_elements_in_polymer(smiles: str) -> set:
    """Get set of elements in a polymer SMILES."""
    mol = polymer_smiles_to_mol(smiles)
    if mol is None:
        return set()

    elements = set()
    for atom in mol.GetAtoms():
        elements.add(atom.GetSymbol())

    return elements


def run_per_element_analysis(
    smiles_list: List[str],
    targets: np.ndarray,
    save_dir: str = './',
    device: torch.device = None
):
    """
    Compare physics vs random embeddings for different elements.
    """
    print("\n" + "="*60)
    print("TASK 2: PER-ELEMENT ANALYSIS")
    print("="*60)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split data
    train_smiles, test_smiles, train_targets, test_targets = train_test_split(
        smiles_list, targets, test_size=0.2, random_state=42
    )

    # Train physics model
    print("\n--- Training Physics Model ---")
    physics_model, physics_metrics, physics_preds, physics_tgts = train_model(
        train_smiles, train_targets,
        test_smiles, test_targets,
        embedding_strategy='physics',
        embedding_dim=8,
        seed=42,
        device=device
    )

    print(f"Physics Model Test RMSE: {physics_metrics['rmse']:.4f}")

    # Train random model
    print("\n--- Training Random Model ---")
    random_model, random_metrics, random_preds, random_tgts = train_model(
        train_smiles, train_targets,
        test_smiles, test_targets,
        embedding_strategy='random',
        embedding_dim=8,
        seed=42,
        device=device
    )

    print(f"Random Model Test RMSE: {random_metrics['rmse']:.4f}")

    # Per-element analysis
    element_data = defaultdict(lambda: {
        'physics_errors': [],
        'random_errors': [],
        'targets': [],
        'count': 0
    })

    # Collect errors per element
    for i, smi in enumerate(test_smiles):
        elements = get_elements_in_polymer(smi)
        if not elements:
            continue

        physics_error = abs(physics_preds[i] - test_targets[i])
        random_error = abs(random_preds[i] - test_targets[i])

        for elem in elements:
            element_data[elem]['physics_errors'].append(physics_error)
            element_data[elem]['random_errors'].append(random_error)
            element_data[elem]['targets'].append(test_targets[i])
            element_data[elem]['count'] += 1

    # Compute per-element RMSE
    element_results = []
    elements_of_interest = {'Si', 'P', 'S', 'Cl', 'Br', 'I', 'F', 'C', 'H', 'N', 'O'}

    for elem in sorted(element_data.keys()):
        data = element_data[elem]
        if data['count'] >= 10:  # Only analyze elements with enough samples
            physics_rmse = np.sqrt(np.mean(np.array(data['physics_errors'])**2))
            random_rmse = np.sqrt(np.mean(np.array(data['random_errors'])**2))

            result = {
                'element': elem,
                'count': data['count'],
                'physics_rmse': physics_rmse,
                'random_rmse': random_rmse,
                'difference': random_rmse - physics_rmse,  # Positive = physics better
                'underrepresented': elem in elements_of_interest - {'C', 'H', 'N', 'O'}
            }
            element_results.append(result)

    # Sort by count
    element_results.sort(key=lambda x: x['count'], reverse=True)

    print("\n" + "="*60)
    print("PER-ELEMENT RMSE COMPARISON")
    print("="*60)
    print(f"\n{'Element':<6} {'Count':>6} {'Physics':>10} {'Random':>10} {'Diff':>8} {'Type':>15}")
    print("-" * 60)

    for res in element_results:
        elem_type = "Underrepresented" if res['underrepresented'] else "Common"
        print(f"{res['element']:<6} {res['count']:>6} {res['physics_rmse']:>10.2f} "
              f"{res['random_rmse']:>10.2f} {res['difference']:>8.2f} {elem_type:>15}")

    # Save results
    with open(os.path.join(save_dir, 'per_element_results.json'), 'w') as f:
        json.dump({
            'physics_overall_rmse': physics_metrics['rmse'],
            'random_overall_rmse': random_metrics['rmse'],
            'element_results': element_results
        }, f, indent=2)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: RMSE comparison for all elements
    elements = [r['element'] for r in element_results[:15]]
    physics_rmses = [r['physics_rmse'] for r in element_results[:15]]
    random_rmses = [r['random_rmse'] for r in element_results[:15]]
    counts = [r['count'] for r in element_results[:15]]

    x = np.arange(len(elements))
    width = 0.35

    ax1 = axes[0]
    ax1.bar(x - width/2, physics_rmses, width, label='Physics', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, random_rmses, width, label='Random', color='coral', alpha=0.8)

    ax1.set_xlabel('Element')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Per-Element RMSE: Physics vs Random Embeddings')
    ax1.set_xticks(x)
    ax1.set_xticklabels(elements)
    ax1.legend()

    # Add count annotations
    for i, (elem, count) in enumerate(zip(elements, counts)):
        ax1.text(i, max(physics_rmses[i], random_rmses[i]) + 1,
                f'n={count}', ha='center', fontsize=7)

    # Plot 2: Difference (Physics - Random) for underrepresented elements
    underrep_results = [r for r in element_results if r['underrepresented']]

    if underrep_results:
        elements_u = [r['element'] for r in underrep_results]
        diffs = [r['difference'] for r in underrep_results]
        colors = ['green' if d > 0 else 'red' for d in diffs]

        ax2 = axes[1]
        bars = ax2.bar(elements_u, diffs, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Element')
        ax2.set_ylabel('Random RMSE - Physics RMSE')
        ax2.set_title('Improvement from Physics Embeddings\n(Positive = Physics Better)')

        for bar, diff in zip(bars, diffs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{diff:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_element_comparison.png'), dpi=150)
    plt.close()

    print(f"\nResults saved to {save_dir}/per_element_results.json")
    print(f"Plot saved to {save_dir}/per_element_comparison.png")

    return element_results


# ========== Task 3: Dimensionality Ablation ==========

def run_dimensionality_ablation(
    smiles_list: List[str],
    targets: np.ndarray,
    dimensions: List[int] = [4, 8, 16, 32, 64],
    save_dir: str = './',
    device: torch.device = None
):
    """
    Test random embeddings with different dimensions.
    """
    print("\n" + "="*60)
    print("TASK 3: DIMENSIONALITY ABLATION")
    print("="*60)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split data
    train_smiles, test_smiles, train_targets, test_targets = train_test_split(
        smiles_list, targets, test_size=0.2, random_state=42
    )

    results = []

    for dim in dimensions:
        print(f"\n--- Testing Dimension: {dim} ---")

        model, metrics, preds, tgts = train_model(
            train_smiles, train_targets,
            test_smiles, test_targets,
            embedding_strategy='random',
            embedding_dim=dim,
            seed=42,
            device=device
        )

        result = {
            'dimension': dim,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2']
        }

        results.append(result)
        print(f"Dimension {dim}: RMSE = {metrics['rmse']:.4f}, MAE = {metrics['mae']:.4f}")

    # Find optimal dimension
    best_result = min(results, key=lambda x: x['rmse'])

    print("\n" + "="*60)
    print("DIMENSIONALITY ABLATION RESULTS")
    print("="*60)
    print(f"\n{'Dimension':<12} {'RMSE':>10} {'MAE':>10} {'R2':>10}")
    print("-" * 45)

    for res in results:
        marker = " <-- Best" if res == best_result else ""
        print(f"{res['dimension']:<12} {res['rmse']:>10.4f} {res['mae']:>10.4f} {res['r2']:>10.4f}{marker}")

    print(f"\nOptimal dimension: {best_result['dimension']} (RMSE: {best_result['rmse']:.4f})")

    # Save results
    with open(os.path.join(save_dir, 'dimensionality_results.json'), 'w') as f:
        json.dump({
            'results': results,
            'optimal_dimension': best_result['dimension'],
            'optimal_rmse': best_result['rmse']
        }, f, indent=2)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    dimensions_plot = [r['dimension'] for r in results]
    rmses = [r['rmse'] for r in results]
    maes = [r['mae'] for r in results]

    # Plot RMSE
    axes[0].plot(dimensions_plot, rmses, 'o-', linewidth=2, markersize=8, color='steelblue')
    axes[0].axvline(x=best_result['dimension'], color='red', linestyle='--',
                    label=f'Best: dim={best_result["dimension"]}')
    axes[0].set_xlabel('Embedding Dimension')
    axes[0].set_ylabel('Test RMSE')
    axes[0].set_title('RMSE vs Embedding Dimension (Random Embeddings)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Add value labels
    for dim, rmse in zip(dimensions_plot, rmses):
        axes[0].text(dim, rmse + 0.3, f'{rmse:.2f}', ha='center', fontsize=9)

    # Plot MAE
    axes[1].plot(dimensions_plot, maes, 'o-', linewidth=2, markersize=8, color='coral')
    axes[1].axvline(x=best_result['dimension'], color='red', linestyle='--',
                    label=f'Best: dim={best_result["dimension"]}')
    axes[1].set_xlabel('Embedding Dimension')
    axes[1].set_ylabel('Test MAE')
    axes[1].set_title('MAE vs Embedding Dimension (Random Embeddings)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Add value labels
    for dim, mae in zip(dimensions_plot, maes):
        axes[1].text(dim, mae + 0.2, f'{mae:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dimensionality_ablation.png'), dpi=150)
    plt.close()

    print(f"\nResults saved to {save_dir}/dimensionality_results.json")
    print(f"Plot saved to {save_dir}/dimensionality_ablation.png")

    return results


# ========== Main ==========

def main():
    print("="*60)
    print("DAY 7 PTKGE ITERATION 2")
    print("="*60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Paths
    base_dir = '/home/home_public/aisci/AIscientist/src/7_day7_ptkge_iter2'
    data_dir = '/home/home_public/aisci/AIscientist/src/alldataset'
    save_dir = os.path.join(base_dir, 'results')
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    print("\nLoading Tg dataset...")
    tg_path = os.path.join(data_dir, 'Tg.csv')
    tg_df = pd.read_csv(tg_path)
    smiles_list = tg_df['Smiles'].tolist()
    targets = tg_df['Tg'].values.astype(np.float32)
    print(f"Dataset size: {len(smiles_list)} samples")

    # ========== Task 1: Ensemble Analysis ==========
    ensemble_results = run_ensemble_analysis(
        smiles_list, targets,
        n_models=5,
        save_dir=save_dir,
        device=device
    )

    # ========== Task 2: Per-Element Analysis ==========
    per_element_results = run_per_element_analysis(
        smiles_list, targets,
        save_dir=save_dir,
        device=device
    )

    # ========== Task 3: Dimensionality Ablation ==========
    dimensionality_results = run_dimensionality_ablation(
        smiles_list, targets,
        dimensions=[4, 8, 16, 32, 64],
        save_dir=save_dir,
        device=device
    )

    # ========== Final Summary ==========
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    print("\n1. ENSEMBLE ANALYSIS:")
    print(f"   - Average Individual RMSE: {ensemble_results['average_individual_rmse']:.4f}")
    print(f"   - Ensemble RMSE: {ensemble_results['ensemble']['rmse']:.4f}")
    print(f"   - Improvement: {ensemble_results['improvement']:.4f}")

    print("\n2. PER-ELEMENT ANALYSIS:")
    underrep_results = [r for r in per_element_results if r['underrepresented']]
    if underrep_results:
        avg_diff = np.mean([r['difference'] for r in underrep_results])
        print(f"   - Average improvement for underrepresented elements: {avg_diff:.4f}")

    print("\n3. DIMENSIONALITY ABLATION:")
    best_dim = min(dimensionality_results, key=lambda x: x['rmse'])
    print(f"   - Optimal dimension: {best_dim['dimension']}")
    print(f"   - Best RMSE: {best_dim['rmse']:.4f}")

    # Save final summary
    summary = {
        'ensemble': ensemble_results,
        'per_element': {
            'underrepresented_avg_improvement': np.mean([r['difference'] for r in underrep_results]) if underrep_results else 0
        },
        'dimensionality': {
            'optimal_dimension': best_dim['dimension'],
            'optimal_rmse': best_dim['rmse']
        }
    }

    with open(os.path.join(save_dir, 'final_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {save_dir}/")
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
