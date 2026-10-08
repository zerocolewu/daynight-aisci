"""
Ensemble script: Load 5 physics models from checkpoints and average predictions.
Trains 5 models with different seeds and computes ensemble RMSE.
"""

import os
import sys
import json
import random
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    patience: int = 30,
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

    for epoch in range(epochs):
        train_loss, train_rmse = finetune_epoch(model, train_loader, optimizer, criterion, device)
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
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: Train RMSE={train_rmse:.4f}, Val RMSE={val_rmse:.4f}")

    model.load_state_dict(best_model_state)

    return model, best_val_rmse, best_val_preds, best_val_targets


def run_ensemble(
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
        model, val_rmse, val_preds, val_tgts = train_single_model(
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

    print(f"\n{'='*60}")
    print(f"Ensemble Results:")
    print(f"{'='*60}")
    print(f"Individual RMSEs: {[f'{r:.2f}' for r in individual_rmses]}")
    print(f"Mean individual RMSE: {np.mean(individual_rmses):.4f} +/- {np.std(individual_rmses):.4f}")
    print(f"Ensemble RMSE: {ensemble_rmse:.4f}")
    print(f"Ensemble MAE: {ensemble_mae:.4f}")
    print(f"Ensemble R2: {ensemble_r2:.4f}")

    return {
        'individual_rmses': individual_rmses,
        'mean_rmse': float(np.mean(individual_rmses)),
        'std_rmse': float(np.std(individual_rmses)),
        'ensemble_rmse': float(ensemble_rmse),
        'ensemble_mae': float(ensemble_mae),
        'ensemble_r2': float(ensemble_r2),
        'models': models,
        'predictions': all_predictions,
        'targets': all_targets
    }


def main():
    set_seed(42)

    base_dir = '/home/home_public/aisci/AIscientist/src/7_day7_ptkge'
    data_dir = '/home/home_public/aisci/AIscientist/src/alldataset'

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

    # Run physics ensemble
    results = run_ensemble(
        train_smiles, train_targets, val_smiles, val_targets,
        embedding_strategy='physics',
        embedding_dim=8,
        n_models=5,
        device=device
    )

    # Save results
    output = {
        'physics_ensemble': {
            'individual_rmses': results['individual_rmses'],
            'mean_rmse': results['mean_rmse'],
            'std_rmse': results['std_rmse'],
            'ensemble_rmse': results['ensemble_rmse'],
            'ensemble_mae': results['ensemble_mae'],
            'ensemble_r2': results['ensemble_r2']
        }
    }

    with open(os.path.join(base_dir, 'ensemble_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {base_dir}/ensemble_results.json")

    return output


if __name__ == "__main__":
    main()
