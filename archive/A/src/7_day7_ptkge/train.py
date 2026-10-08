"""
Training script for PTKGE model with pretraining and fine-tuning.
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
from torch_geometric.loader import DataLoader as GeomDataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import PTKGEModel, ContrastiveLoss, compute_ecfp4_fingerprint, compute_similarity_matrix
from embeddings import PeriodicEmbedding

warnings.filterwarnings('ignore')

# Set random seeds
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ========== Data Processing ==========

def polymer_smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """Convert polymer SMILES (with * for connections) to RDKit mol."""
    try:
        # Replace * with [H] for processing
        processed = smiles.replace('*', '[H]')
        mol = Chem.MolFromSmiles(processed)
        return mol
    except:
        return None


def mol_to_graph(mol: Chem.Mol) -> Optional[Data]:
    """Convert RDKit mol to PyTorch Geometric graph."""
    if mol is None:
        return None

    try:
        # Get atom features
        atoms = []
        for atom in mol.GetAtoms():
            atoms.append(atom.GetAtomicNum())

        if len(atoms) == 0:
            return None

        x = torch.tensor(atoms, dtype=torch.long)

        # Get edges
        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.append([i, j])
            edges.append([j, i])

        if len(edges) == 0:
            # Single atom - add self loop
            edges = [[0, 0]]

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)
    except:
        return None


class PolymerDataset(Dataset):
    """Dataset for polymer SMILES."""

    def __init__(self, smiles_list: List[str], targets: Optional[np.ndarray] = None,
                 fp_dim: int = 2048):
        self.smiles_list = smiles_list
        self.targets = targets
        self.fp_dim = fp_dim

        # Pre-compute graphs and fingerprints
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


class PretrainDataset(Dataset):
    """Dataset for contrastive pretraining."""

    def __init__(self, smiles_list: List[str], fp_dim: int = 2048):
        self.smiles_list = smiles_list
        self.fp_dim = fp_dim

        # Pre-compute graphs and fingerprints
        self.graphs = []
        self.fingerprints = []

        print(f"Processing {len(smiles_list)} samples for pretraining...")
        for smi in smiles_list:
            mol = polymer_smiles_to_mol(smi)
            graph = mol_to_graph(mol) if mol else None
            fp = compute_ecfp4_fingerprint(smi, n_bits=fp_dim) if mol else None

            if graph is not None and fp is not None and fp.sum() > 0:
                self.graphs.append(graph)
                self.fingerprints.append(fp)

        print(f"Valid pretrain samples: {len(self.graphs)}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.fingerprints[idx]


def collate_fn(batch):
    """Custom collate function for batching graphs."""
    if len(batch[0]) == 3:
        # With targets
        graphs, fps, targets = zip(*batch)
        targets = torch.tensor(np.array(targets), dtype=torch.float32)
    else:
        # Without targets
        graphs, fps = zip(*batch)
        targets = None

    # Batch graphs
    batched_graphs = Batch.from_data_list(graphs)
    fps = torch.tensor(np.array(fps), dtype=torch.float32)

    if targets is not None:
        return batched_graphs, fps, targets
    return batched_graphs, fps


# ========== Training Functions ==========

def pretrain_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: ContrastiveLoss,
    device: torch.device,
    accumulation_steps: int = 1
) -> float:
    """Run one pretraining epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, (graphs, fps) in enumerate(loader):
        graphs = graphs.to(device)
        fps = fps.to(device)

        # Compute similarity matrix for contrastive loss
        with torch.no_grad():
            sim_matrix = compute_similarity_matrix(fps.cpu().numpy())
            sim_matrix = torch.from_numpy(sim_matrix).to(device)

        # Forward pass
        embeddings = model.get_graph_embedding(
            graphs.x, graphs.edge_index, graphs.batch, fps
        )

        # Compute loss
        loss = criterion(embeddings, sim_matrix)
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        num_batches += 1

    return total_loss / max(num_batches, 1)


def finetune_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Run one fine-tuning epoch."""
    model.train()
    total_loss = 0.0
    predictions = []
    targets = []

    for graphs, fps, tgt in loader:
        graphs = graphs.to(device)
        fps = fps.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        # Forward pass
        pred = model(graphs.x, graphs.edge_index, graphs.batch, fps)
        loss = criterion(pred.squeeze(), tgt)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        predictions.extend(pred.squeeze().cpu().detach().numpy())
        targets.extend(tgt.cpu().numpy())

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    return total_loss / len(loader), rmse


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
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


# ========== Main Training Pipeline ==========

def run_pretraining(
    smiles_list: List[str],
    embedding_strategy: str,
    save_dir: str,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-4,
    temperature: float = 0.05,
    hidden_dim: int = 256,
    num_layers: int = 5,
    device: torch.device = None
) -> nn.Module:
    """Run contrastive pretraining."""

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"Pretraining with {embedding_strategy} embeddings")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Create dataset
    dataset = PretrainDataset(smiles_list)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )

    # Create model
    model = PTKGEModel(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        embedding_strategy=embedding_strategy,
        embedding_dim=8,
        dropout=0.1
    ).to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = ContrastiveLoss(temperature=temperature)

    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        loss = pretrain_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'embedding_strategy': embedding_strategy,
                'epoch': epoch
            }, os.path.join(save_dir, f'pretrained_{embedding_strategy}.pt'))

    print(f"Pretraining complete. Best loss: {best_loss:.4f}")
    return model


def run_finetuning(
    train_smiles: List[str],
    train_targets: np.ndarray,
    val_smiles: List[str],
    val_targets: np.ndarray,
    embedding_strategy: str,
    pretrained_path: Optional[str],
    epochs: int = 100,
    batch_size: int = 32,
    encoder_lr: float = 1e-4,
    head_lr: float = 1e-3,
    patience: int = 15,
    hidden_dim: int = 256,
    num_layers: int = 5,
    device: torch.device = None
) -> Tuple[nn.Module, Dict[str, float]]:
    """Run fine-tuning with early stopping."""

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
        embedding_dim=8,
        dropout=0.1
    ).to(device)

    # Load pretrained weights
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Optimizer with different learning rates
    optimizer = AdamW([
        {'params': model.get_encoder_params(), 'lr': encoder_lr},
        {'params': model.get_predictor_params(), 'lr': head_lr}
    ], weight_decay=1e-5)

    criterion = nn.MSELoss()

    # Training loop with early stopping
    best_val_rmse = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_rmse = finetune_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

        val_rmse = val_metrics['rmse']

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train RMSE: {train_rmse:.4f} - Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_model_state)
    model.to(device)

    # Final evaluation
    final_metrics, _, _ = evaluate(model, val_loader, criterion, device)

    return model, final_metrics


def run_cross_validation(
    smiles_list: List[str],
    targets: np.ndarray,
    embedding_strategy: str,
    pretrained_path: Optional[str],
    n_folds: int = 5,
    save_dir: str = './',
    **kwargs
) -> Dict[str, float]:
    """Run K-fold cross-validation."""

    print(f"\n{'='*60}")
    print(f"Running {n_folds}-fold CV with {embedding_strategy} embeddings")
    print(f"{'='*60}")

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    all_predictions = []
    all_targets = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(smiles_list)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")

        train_smiles = [smiles_list[i] for i in train_idx]
        val_smiles = [smiles_list[i] for i in val_idx]
        train_targets = targets[train_idx]
        val_targets = targets[val_idx]

        model, metrics = run_finetuning(
            train_smiles, train_targets,
            val_smiles, val_targets,
            embedding_strategy=embedding_strategy,
            pretrained_path=pretrained_path,
            **kwargs
        )

        fold_results.append(metrics)
        print(f"Fold {fold+1} - Val RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")

    # Aggregate results
    avg_results = {
        'rmse_mean': np.mean([r['rmse'] for r in fold_results]),
        'rmse_std': np.std([r['rmse'] for r in fold_results]),
        'mae_mean': np.mean([r['mae'] for r in fold_results]),
        'mae_std': np.std([r['mae'] for r in fold_results]),
        'r2_mean': np.mean([r['r2'] for r in fold_results]),
        'r2_std': np.std([r['r2'] for r in fold_results]),
    }

    print(f"\n{embedding_strategy} Results:")
    print(f"  RMSE: {avg_results['rmse_mean']:.4f} ± {avg_results['rmse_std']:.4f}")
    print(f"  MAE: {avg_results['mae_mean']:.4f} ± {avg_results['mae_std']:.4f}")
    print(f"  R2: {avg_results['r2_mean']:.4f} ± {avg_results['r2_std']:.4f}")

    return avg_results


# ========== Visualization ==========

def plot_embedding_tsne(model: nn.Module, save_path: str, device: torch.device):
    """Plot t-SNE visualization of element embeddings."""
    from embeddings import ELEMENT_PROPERTIES

    model.eval()

    # Get element embeddings
    with torch.no_grad():
        atomic_nums = torch.arange(1, 119).to(device)
        embeddings = model.gnn_encoder.atom_embedding.embedding(atomic_nums).cpu().numpy()

    # Element names
    element_names = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe'
    ]

    # Common elements in polymers (to highlight)
    polymer_elements = {'H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I'}

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Color by group
    groups = [ELEMENT_PROPERTIES[i][3] for i in range(1, 119)]  # Group number

    scatter = ax.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=groups, cmap='tab20', s=100, alpha=0.7
    )

    # Add labels for common elements
    for i, name in enumerate(element_names[:len(embeddings_2d)]):
        if name in polymer_elements:
            ax.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                       fontsize=10, fontweight='bold', ha='center', va='bottom')

    ax.set_title('t-SNE Visualization of Element Embeddings', fontsize=14)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')

    plt.colorbar(scatter, label='Group Number')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")


def plot_per_element_rmse(
    predictions: np.ndarray,
    targets: np.ndarray,
    smiles_list: List[str],
    save_path: str
):
    """Plot per-element RMSE analysis."""
    from collections import Counter

    # Get elements for each polymer
    element_to_rmse = defaultdict(list)

    for smi, pred, tgt in zip(smiles_list, predictions, targets):
        mol = polymer_smiles_to_mol(smi)
        if mol is None:
            continue

        elements = set()
        for atom in mol.GetAtoms():
            elements.add(atom.GetSymbol())

        error = abs(pred - tgt)
        for elem in elements:
            element_to_rmse[elem].append(error)

    # Compute RMSE per element
    element_rmse = {}
    for elem, errors in element_to_rmse.items():
        if len(errors) >= 5:  # Only elements with enough samples
            element_rmse[elem] = np.sqrt(np.mean(np.array(errors)**2))

    # Sort by RMSE
    sorted_elements = sorted(element_rmse.items(), key=lambda x: x[1], reverse=True)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    elements = [e[0] for e in sorted_elements[:20]]
    rmses = [e[1] for e in sorted_elements[:20]]

    # Highlight underrepresented elements
    colors = ['red' if e in {'Si', 'P', 'S', 'Cl', 'Br', 'I', 'F'} else 'steelblue' for e in elements]

    ax.bar(elements, rmses, color=colors)
    ax.set_xlabel('Element')
    ax.set_ylabel('RMSE')
    ax.set_title('Per-Element RMSE Analysis (Red = Underrepresented Elements)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-element RMSE plot to {save_path}")


def plot_embedding_similarity(model: nn.Module, save_path: str, device: torch.device):
    """Plot cosine similarity matrix of element embeddings."""
    from embeddings import analyze_element_similarity

    model.eval()

    elements = ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']

    with torch.no_grad():
        similarity, element_names = analyze_element_similarity(
            model.gnn_encoder.atom_embedding.embedding, elements
        )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        similarity,
        xticklabels=element_names,
        yticklabels=element_names,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        vmin=-1, vmax=1,
        ax=ax
    )

    ax.set_title('Element Embedding Cosine Similarity')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity matrix to {save_path}")


# ========== Main ==========

def main():
    set_seed(42)

    # Paths
    base_dir = '/home/home_public/aisci/AIscientist/src/7_day7_ptkge'
    data_dir = '/home/home_public/aisci/AIscientist/src/alldataset'

    tg_path = os.path.join(data_dir, 'Tg.csv')
    pi1m_path = os.path.join(data_dir, 'PI1M.csv')

    save_dir = os.path.join(base_dir, 'checkpoints')
    figures_dir = os.path.join(base_dir, 'figures')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Tg dataset
    print("\nLoading Tg dataset...")
    tg_df = pd.read_csv(tg_path)
    tg_smiles = tg_df['Smiles'].tolist()
    tg_targets = tg_df['Tg'].values.astype(np.float32)
    print(f"Tg dataset: {len(tg_smiles)} samples")

    # Load PI1M for pretraining (sample 100k)
    print("\nLoading PI1M dataset...")
    pi1m_df = pd.read_csv(pi1m_path)
    sample_size = min(100000, len(pi1m_df))
    pi1m_sample = pi1m_df.sample(n=sample_size, random_state=42)
    pi1m_smiles = pi1m_sample['SMILES'].tolist()
    print(f"PI1M sample: {len(pi1m_smiles)} samples")

    # Results storage
    results = {}

    # ========== Experiment Conditions ==========

    strategies = ['onehot', 'random', 'physics', 'learnable']

    # Load existing results if available
    results = {}
    results_path = os.path.join(base_dir, 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded existing results for: {list(results.keys())}")

    # Common training parameters
    pretrain_params = {
        'epochs': 20,
        'batch_size': 64,
        'lr': 1e-4,
        'temperature': 0.05,
        'hidden_dim': 256,
        'num_layers': 5,
        'device': device
    }

    finetune_params = {
        'epochs': 100,
        'batch_size': 32,
        'encoder_lr': 1e-4,
        'head_lr': 1e-3,
        'patience': 15,
        'hidden_dim': 256,
        'num_layers': 5,
        'device': device
    }

    # ========== Run Experiments ==========

    for strategy in strategies:
        # Skip if already completed
        if strategy in results:
            print(f"\nSkipping {strategy} - already completed")
            continue
        print(f"\n{'#'*60}")
        print(f"# Strategy: {strategy}")
        print(f"{'#'*60}")

        # Pretraining
        pretrained_path = os.path.join(save_dir, f'pretrained_{strategy}.pt')

        if strategy != 'onehot':  # One-hot doesn't benefit from physics-based pretraining
            run_pretraining(
                pi1m_smiles,
                embedding_strategy=strategy,
                save_dir=save_dir,
                **pretrain_params
            )
        else:
            pretrained_path = None

        # Fine-tuning with cross-validation
        cv_results = run_cross_validation(
            smiles_list=tg_smiles,
            targets=tg_targets,
            embedding_strategy=strategy,
            pretrained_path=pretrained_path,
            save_dir=save_dir,
            **finetune_params
        )

        results[strategy] = cv_results

        # Save intermediate results
        with open(os.path.join(base_dir, 'results_intermediate.json'), 'w') as f:
            json.dump(results, f, indent=2)

    # ========== Visualization ==========

    print("\nGenerating visualizations...")

    # Load best model for visualization
    best_strategy = min(results.keys(), key=lambda x: results[x]['rmse_mean'])
    print(f"Best strategy: {best_strategy}")

    pretrained_path = os.path.join(save_dir, f'pretrained_{best_strategy}.pt')
    model = PTKGEModel(
        num_layers=5,
        hidden_dim=256,
        embedding_strategy=best_strategy,
        embedding_dim=8
    ).to(device)

    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # t-SNE visualization
    plot_embedding_tsne(model, os.path.join(figures_dir, 'embedding_tsne.png'), device)

    # Similarity matrix
    plot_embedding_similarity(model, os.path.join(figures_dir, 'embedding_similarity.png'), device)

    # ========== Final Results ==========

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    for strategy, res in results.items():
        print(f"\n{strategy}:")
        print(f"  RMSE: {res['rmse_mean']:.4f} ± {res['rmse_std']:.4f}")
        print(f"  MAE: {res['mae_mean']:.4f} ± {res['mae_std']:.4f}")
        print(f"  R2: {res['r2_mean']:.4f} ± {res['r2_std']:.4f}")

    # Save final results
    with open(os.path.join(base_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {base_dir}/results.json")
    print(f"Figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
