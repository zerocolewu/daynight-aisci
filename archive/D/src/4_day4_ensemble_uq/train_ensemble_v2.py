#!/usr/bin/env python3
"""
Day 4: Multi-seed Ensemble with Uncertainty Quantification for Tm and Egc
Version 2: Simplified batching approach
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.loader import DataLoader

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, rdMolDescriptors

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Random seeds for ensemble
SEEDS = [42, 123, 456, 789, 1024]
N_FOLDS = 5


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_148_features(mol) -> np.ndarray:
    """Compute 148-dimensional feature vector from RDKit mol object."""
    features = []

    try:
        # Basic molecular properties (15)
        features.append(Descriptors.MolWt(mol))
        features.append(Descriptors.HeavyAtomCount(mol))
        features.append(mol.GetNumAtoms())
        features.append(mol.GetNumBonds())
        features.append(mol.GetNumHeavyAtoms())
        features.append(Descriptors.NumValenceElectrons(mol))
        features.append(Descriptors.NumRadicalElectrons(mol))

        # Ring information (10)
        features.append(Descriptors.RingCount(mol))
        features.append(Descriptors.FractionCSP3(mol))
        features.append(rdMolDescriptors.CalcNumRings(mol))
        features.append(rdMolDescriptors.CalcNumAromaticRings(mol))
        features.append(rdMolDescriptors.CalcNumAliphaticRings(mol))
        features.append(rdMolDescriptors.CalcNumSaturatedRings(mol))
        features.append(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))
        features.append(rdMolDescriptors.CalcNumAromaticCarbocycles(mol))
        features.append(rdMolDescriptors.CalcNumHeterocycles(mol))
        features.append(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))

        # Electronic properties (8)
        features.append(Descriptors.MaxPartialCharge(mol))
        features.append(Descriptors.MinPartialCharge(mol))
        features.append(Descriptors.MaxAbsPartialCharge(mol))
        features.append(Descriptors.MinAbsPartialCharge(mol))
        features.append(Descriptors.MolLogP(mol))
        features.append(Descriptors.MolMR(mol))
        features.append(Descriptors.TPSA(mol))
        features.append(Descriptors.LabuteASA(mol))

        # Topological indices (15)
        features.append(Descriptors.BertzCT(mol))
        features.append(Descriptors.BalabanJ(mol))
        features.append(Descriptors.HallKierAlpha(mol))
        features.append(Descriptors.Kappa1(mol))
        features.append(Descriptors.Kappa2(mol))
        features.append(Descriptors.Kappa3(mol))
        features.append(Descriptors.Chi0(mol))
        features.append(Descriptors.Chi1(mol))
        features.append(Descriptors.Chi0n(mol))
        features.append(Descriptors.Chi1n(mol))
        features.append(Descriptors.Chi2n(mol))
        features.append(Descriptors.Chi3n(mol))
        features.append(Descriptors.Chi4n(mol))
        features.append(Descriptors.Chi0v(mol))
        features.append(Descriptors.Chi1v(mol))

        # Hydrogen bond donors/acceptors (6)
        features.append(Descriptors.NumHDonors(mol))
        features.append(Descriptors.NumHAcceptors(mol))
        features.append(Descriptors.NumHeteroatoms(mol))
        features.append(Descriptors.NumHeterocycles(mol))
        features.append(Lipinski.NHOHCount(mol))
        features.append(Lipinski.NOCount(mol))

        # Rotatable bonds (5)
        total_bonds = mol.GetNumBonds()
        features.append(Descriptors.NumRotatableBonds(mol))
        features.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        features.append(Lipinski.FractionCSP3(mol))

        # Complexity (5)
        features.append(Descriptors.BertzCT(mol))
        features.append(Descriptors.HallKierAlpha(mol))
        features.append(rdMolDescriptors.CalcNumAmideBonds(mol))

        # Atom type counts (9)
        atom_types = {'C': 0, 'N': 0, 'O': 0, 'S': 0, 'F': 0, 'Cl': 0, 'Br': 0, 'I': 0, 'P': 0}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in atom_types:
                atom_types[symbol] += 1
        features.extend(list(atom_types.values()))

        # Hybridization counts (4)
        hybridization = {Chem.rdchem.HybridizationType.SP: 0,
                        Chem.rdchem.HybridizationType.SP2: 0,
                        Chem.rdchem.HybridizationType.SP3: 0,
                        Chem.rdchem.HybridizationType.SP3D: 0}
        for atom in mol.GetAtoms():
            hyb = atom.GetHybridization()
            if hyb in hybridization:
                hybridization[hyb] += 1
        features.extend(list(hybridization.values()))

        # Bond type counts (4)
        bond_types = {Chem.rdchem.BondType.SINGLE: 0,
                     Chem.rdchem.BondType.DOUBLE: 0,
                     Chem.rdchem.BondType.TRIPLE: 0,
                     Chem.rdchem.BondType.AROMATIC: 0}
        for bond in mol.GetBonds():
            btype = bond.GetBondType()
            if btype in bond_types:
                bond_types[btype] += 1
        features.extend(list(bond_types.values()))

        # Degree distribution (6)
        degrees = [0] * 6
        for atom in mol.GetAtoms():
            deg = min(atom.GetDegree(), 5)
            degrees[deg] += 1
        features.extend(degrees)

        # Ring sizes (6)
        ring_info = mol.GetRingInfo()
        ring_sizes = [0] * 6
        for ring in ring_info.AtomRings():
            size = len(ring)
            if 3 <= size <= 8:
                ring_sizes[size - 3] += 1
        features.extend(ring_sizes)

        # Functional group counts (20 common groups)
        functional_groups = {
            'alcohol': '[OH]',
            'amine': '[NH2,NH]',
            'carbonyl': '[C]=[O]',
            'carboxylic': 'C(=O)[OH]',
            'ester': 'C(=O)[O]',
            'ether': '[O]([C])[C]',
            'amide': 'C(=O)[N]',
            'nitrile': 'C#N',
            'halogen': '[F,Cl,Br,I]',
            'nitro': '[N+](=[O])[O-]',
            'sulfide': '[S]',
            'phosphate': 'P(=O)([O])[O]',
        }

        for name, smarts in functional_groups.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                matches = mol.GetSubstructMatches(pattern)
                features.append(len(matches))
            else:
                features.append(0)

        # Additional features to reach 148
        features.append(Descriptors.NumRotatableBonds(mol) / total_bonds if total_bonds > 0 else 0)
        features.append(rdMolDescriptors.CalcNumAromaticRings(mol))
        features.append(rdMolDescriptors.CalcNumAliphaticRings(mol))
        features.append(rdMolDescriptors.CalcNumSaturatedRings(mol))
        features.append(mol.GetNumConformers() if mol.GetNumConformers() else 0)
        features.append(Descriptors.HeavyAtomMolWt(mol) / Descriptors.MolWt(mol) if Descriptors.MolWt(mol) > 0 else 0)

        # Complexity metrics
        features.append(Descriptors.BertzCT(mol))
        features.append(Descriptors.BalabanJ(mol))

        # Ensure exactly 148 features
        features = features[:148]
        while len(features) < 148:
            features.append(0.0)

        # Replace NaN/Inf with 0
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    except Exception as e:
        print(f"Error computing features: {e}")
        return np.zeros(148, dtype=np.float32)


def process_psmiles(psmiles: str) -> Tuple[Chem.Mol, str]:
    """Process p-SMILES to get RDKit mol object."""
    smiles = psmiles.replace('*', '[H]')

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None, None

        Chem.SanitizeMol(mol)
        return mol, smiles

    except Exception as e:
        try:
            smiles_simple = psmiles.replace('*', '')
            mol = Chem.MolFromSmiles(smiles_simple)
            if mol is not None:
                return mol, smiles_simple
        except:
            pass
        return None, None


def mol_to_pyg_data(mol, target: float, features: np.ndarray) -> Optional[Data]:
    """Convert RDKit mol to PyG Data object with features."""
    try:
        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_feat = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetNumRadicalElectrons(),
                atom.GetHybridization(),
                atom.GetIsAromatic(),
                atom.GetMass(),
            ]
            atom_features.append(atom_feat)

        x = torch.tensor(atom_features, dtype=torch.float)

        # Get edge indices
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])

            edge_attr = [
                bond.GetBondTypeAsDouble(),
                bond.GetIsConjugated(),
                bond.IsInRing(),
            ]
            edge_attrs.append(edge_attr)
            edge_attrs.append(edge_attr)

        if len(edge_indices) == 0:
            return None

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        # Create data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([target], dtype=torch.float),
            features=torch.tensor(features, dtype=torch.float)
        )

        return data

    except Exception as e:
        print(f"Error converting mol to PyG data: {e}")
        return None


class GINWithFeatures(nn.Module):
    """GIN model with concatenated features."""

    def __init__(self, num_node_features, num_features=148, hidden_dim=256, num_layers=5):
        super(GINWithFeatures, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Node embedding layers
        self.node_embedding = nn.Linear(num_node_features, hidden_dim)

        # GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            self.convs.append(GINConv(nn_layer))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Feature processing
        self.feature_bn = nn.BatchNorm1d(num_features)
        self.feature_proj = nn.Linear(num_features, hidden_dim)

        # Output head - 2 layers, 128 hidden dim
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, batch):
        x, edge_index, batch_idx, features = batch.x, batch.edge_index, batch.batch, batch.features

        # Node embedding
        x = self.node_embedding(x)

        # GIN layers with mean pooling
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        # Global mean pooling
        x = global_mean_pool(x, batch_idx)

        # Process features - need to handle batched features
        # Features are stacked as [n_graphs * 148] but we need [n_graphs, 148]
        n_graphs = x.size(0)
        features = features.view(n_graphs, -1)
        features = self.feature_bn(features)
        features = self.feature_proj(features)
        features = F.relu(features)

        # Concatenate graph and feature embeddings
        combined = torch.cat([x, features], dim=1)

        # Output
        out = self.output(combined)
        return out.squeeze(-1)


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        output = model(batch)
        loss = criterion(output, batch.y.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch.y.view(-1))

            total_loss += loss.item()
            predictions.extend(output.cpu().numpy())
            targets.extend(batch.y.view(-1).cpu().numpy())

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0

    predictions = np.array(predictions)
    targets = np.array(targets)

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    return avg_loss, rmse, mae, r2, predictions, targets


def train_single_model(
    train_data,
    val_data,
    test_data,
    seed: int,
    fold: int,
    property_name: str,
    checkpoint_dir: str,
    max_epochs: int = 500,
    patience: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    batch_size: int = 32
) -> Dict:
    """Train a single model with given seed and fold."""

    set_seed(seed)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize model
    num_node_features = train_data[0].x.shape[1]
    model = GINWithFeatures(num_node_features=num_node_features, num_features=148, hidden_dim=256, num_layers=5)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_rmse, val_mae, val_r2, _, _ = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{max_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val RMSE={val_rmse:.4f}")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    model = model.to(device)
    test_loss, test_rmse, test_mae, test_r2, predictions, targets = evaluate(model, test_loader, criterion, device)

    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"{property_name}_fold{fold}_seed{seed}.pt")
    torch.save({
        'model_state_dict': best_model_state,
        'seed': seed,
        'fold': fold,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
    }, checkpoint_path)

    print(f"    Seed {seed}: Test RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, R2={test_r2:.4f}")

    return {
        'seed': seed,
        'fold': fold,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'predictions': predictions,
        'targets': targets,
    }


def run_ensemble_training(
    csv_path: str,
    property_name: str,
    checkpoint_dir: str,
    n_folds: int = 5,
    seeds: List[int] = SEEDS
) -> Dict:
    """Run ensemble training for a property."""

    print(f"\n{'='*80}")
    print(f"Training ensemble for {property_name}")
    print(f"{'='*80}")

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")

    # Determine column names
    if 'psmiles' in df.columns:
        psmiles_col = 'psmiles'
    elif 'p_smiles' in df.columns:
        psmiles_col = 'p_smiles'
    elif 'Smiles' in df.columns:
        psmiles_col = 'Smiles'
    elif 'smiles' in df.columns:
        psmiles_col = 'smiles'
    else:
        psmiles_col = df.columns[0]

    # Determine target column
    target_col_candidates = [c for c in df.columns if c.lower() in ['tm', 'egc', 'tg', 'value']]
    if target_col_candidates:
        target_col = target_col_candidates[0]
    elif 'value' in df.columns:
        target_col = 'value'
    else:
        # Use second column as fallback
        target_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    print(f"Using columns: psmiles={psmiles_col}, target={target_col}")

    # Process molecules
    print("Processing molecules...")
    data_list = []
    failed = 0

    for idx, row in df.iterrows():
        psmiles = row[psmiles_col]
        target = row[target_col]

        mol, smiles = process_psmiles(psmiles)
        if mol is None:
            failed += 1
            continue

        features = compute_148_features(mol)
        data = mol_to_pyg_data(mol, target, features)

        if data is not None:
            data_list.append(data)
        else:
            failed += 1

    print(f"Successfully processed {len(data_list)} molecules, {failed} failed")

    # K-Fold cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_results = {
        'property': property_name,
        'n_samples': len(data_list),
        'n_failed': failed,
        'n_folds': n_folds,
        'n_seeds': len(seeds),
        'seeds': seeds,
        'fold_results': [],
    }

    # Store all predictions for uncertainty analysis
    all_fold_predictions = {}

    for fold_idx, (train_val_idx, test_idx) in enumerate(kfold.split(data_list), 1):
        print(f"\n{'='*80}")
        print(f"Fold {fold_idx}/{n_folds}")
        print(f"{'='*80}")

        # Split train into train/val
        train_val_data = [data_list[i] for i in train_val_idx]
        test_data = [data_list[i] for i in test_idx]

        # Further split train_val into train and validation (80/20)
        split_idx = int(len(train_val_data) * 0.8)
        train_data = train_val_data[:split_idx]
        val_data = train_val_data[split_idx:]

        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        fold_seed_results = []
        fold_predictions = []

        # Train multiple seeds
        for seed in seeds:
            print(f"\n  Training seed {seed}...")

            result = train_single_model(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                seed=seed,
                fold=fold_idx,
                property_name=property_name,
                checkpoint_dir=checkpoint_dir,
            )

            fold_seed_results.append({
                'seed': seed,
                'test_rmse': result['test_rmse'],
                'test_mae': result['test_mae'],
                'test_r2': result['test_r2'],
                'best_epoch': result['best_epoch'],
                'best_val_loss': result['best_val_loss'],
            })

            fold_predictions.append(result['predictions'])

        # Compute ensemble predictions (mean of seeds)
        fold_predictions = np.array(fold_predictions)  # Shape: (n_seeds, n_test_samples)
        ensemble_pred = np.mean(fold_predictions, axis=0)
        ensemble_std = np.std(fold_predictions, axis=0)  # Uncertainty
        targets = result['targets']  # Same for all seeds

        # Ensemble metrics
        ensemble_rmse = np.sqrt(mean_squared_error(targets, ensemble_pred))
        ensemble_mae = mean_absolute_error(targets, ensemble_pred)
        ensemble_r2 = r2_score(targets, ensemble_pred)

        # Uncertainty-error correlation
        abs_errors = np.abs(targets - ensemble_pred)
        correlation, p_value = stats.pearsonr(ensemble_std, abs_errors)

        # Identify high-uncertainty samples (top 20%)
        threshold = np.percentile(ensemble_std, 80)
        high_uncertainty_mask = ensemble_std >= threshold

        high_unc_rmse = np.sqrt(mean_squared_error(
            targets[high_uncertainty_mask],
            ensemble_pred[high_uncertainty_mask]
        )) if np.sum(high_uncertainty_mask) > 0 else 0

        low_unc_rmse = np.sqrt(mean_squared_error(
            targets[~high_uncertainty_mask],
            ensemble_pred[~high_uncertainty_mask]
        )) if np.sum(~high_uncertainty_mask) > 0 else 0

        fold_result = {
            'fold': fold_idx,
            'seed_results': fold_seed_results,
            'individual_seed_rmse': [r['test_rmse'] for r in fold_seed_results],
            'individual_seed_rmse_mean': float(np.mean([r['test_rmse'] for r in fold_seed_results])),
            'individual_seed_rmse_std': float(np.std([r['test_rmse'] for r in fold_seed_results])),
            'ensemble_rmse': float(ensemble_rmse),
            'ensemble_mae': float(ensemble_mae),
            'ensemble_r2': float(ensemble_r2),
            'uncertainty_error_correlation': float(correlation),
            'uncertainty_error_pvalue': float(p_value),
            'high_uncertainty_rmse': float(high_unc_rmse),
            'low_uncertainty_rmse': float(low_unc_rmse),
            'n_high_uncertainty': int(np.sum(high_uncertainty_mask)),
        }

        all_results['fold_results'].append(fold_result)

        # Store predictions for this fold
        all_fold_predictions[fold_idx] = {
            'predictions': fold_predictions,
            'ensemble_pred': ensemble_pred,
            'ensemble_std': ensemble_std,
            'targets': targets,
        }

        print(f"\n  Fold {fold_idx} Results:")
        print(f"    Individual seeds RMSE: {fold_result['individual_seed_rmse_mean']:.4f} ± {fold_result['individual_seed_rmse_std']:.4f}")
        print(f"    Ensemble RMSE: {ensemble_rmse:.4f}")
        print(f"    Uncertainty-Error correlation: r={correlation:.4f} (p={p_value:.4e})")
        print(f"    High uncertainty RMSE: {high_unc_rmse:.4f}, Low uncertainty RMSE: {low_unc_rmse:.4f}")

    # Compute cross-fold statistics
    ensemble_rmses = [f['ensemble_rmse'] for f in all_results['fold_results']]
    individual_means = [f['individual_seed_rmse_mean'] for f in all_results['fold_results']]

    all_results['cross_fold_ensemble'] = {
        'mean_rmse': float(np.mean(ensemble_rmses)),
        'std_rmse': float(np.std(ensemble_rmses)),
        'best_rmse': float(np.min(ensemble_rmses)),
        'best_fold': int(np.argmin(ensemble_rmses) + 1),
        'worst_rmse': float(np.max(ensemble_rmses)),
        'worst_fold': int(np.argmax(ensemble_rmses) + 1),
        'variance_reduction': float((np.std(individual_means) - np.std(ensemble_rmses)) / np.std(individual_means) * 100) if np.std(individual_means) > 0 else 0.0,
    }

    print(f"\n{'='*80}")
    print(f"{property_name} Cross-Fold Results:")
    print(f"  Ensemble RMSE: {all_results['cross_fold_ensemble']['mean_rmse']:.4f} ± {all_results['cross_fold_ensemble']['std_rmse']:.4f}")
    print(f"  Best fold: {all_results['cross_fold_ensemble']['best_fold']} (RMSE: {all_results['cross_fold_ensemble']['best_rmse']:.4f})")
    print(f"  Worst fold: {all_results['cross_fold_ensemble']['worst_fold']} (RMSE: {all_results['cross_fold_ensemble']['worst_rmse']:.4f})")
    print(f"  Variance reduction: {all_results['cross_fold_ensemble']['variance_reduction']:.2f}%")
    print(f"{'='*80}\n")

    # Save predictions for analysis
    predictions_path = os.path.join(checkpoint_dir, f"{property_name}_predictions.pkl")
    with open(predictions_path, 'wb') as f:
        pickle.dump(all_fold_predictions, f)

    return all_results


def check_training_complete(property_name: str, checkpoint_dir: str, n_folds: int = 5, seeds: List[int] = SEEDS) -> bool:
    """Check if all checkpoints for a property exist."""
    for fold in range(1, n_folds + 1):
        for seed in seeds:
            checkpoint_path = os.path.join(checkpoint_dir, f"{property_name}_fold{fold}_seed{seed}.pt")
            if not os.path.exists(checkpoint_path):
                return False
    return True


def main():
    """Main function to run ensemble training."""

    # Paths
    base_dir = Path(__file__).parent
    checkpoint_dir = base_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    dataset_dir = Path(__file__).parent.parent / "alldataset"

    # Run ensemble training for Tm
    if check_training_complete("Tm", str(checkpoint_dir)):
        print("\nTm training already complete. Skipping...")
        # Try to load existing results
        tm_results_path = base_dir / "tm_results.json"
        results_path = base_dir / "results.json"
        if tm_results_path.exists():
            with open(tm_results_path, 'r') as f:
                tm_results = json.load(f)
            print(f"Loaded Tm results from {tm_results_path}")
        elif results_path.exists():
            with open(results_path, 'r') as f:
                saved_results = json.load(f)
                tm_results = saved_results.get('tm', {})
            print(f"Loaded Tm results from {results_path}")
        else:
            tm_results = {}
    else:
        tm_results = run_ensemble_training(
            csv_path=str(dataset_dir / "Tm.csv"),
            property_name="Tm",
            checkpoint_dir=str(checkpoint_dir),
        )

    # Run ensemble training for Egc
    if check_training_complete("Egc", str(checkpoint_dir)):
        print("\nEgc training already complete. Skipping...")
        # Try to load existing results
        results_path = base_dir / "results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                saved_results = json.load(f)
                egc_results = saved_results.get('egc', {})
        else:
            egc_results = {}
    else:
        egc_results = run_ensemble_training(
            csv_path=str(dataset_dir / "Egc.csv"),
            property_name="Egc",
            checkpoint_dir=str(checkpoint_dir),
        )

    # Save final results
    results = {
        'tm': tm_results,
        'egc': egc_results,
        'timestamp': datetime.now().isoformat(),
    }

    results_path = base_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\nTm:")
    print(f"  Mean RMSE: {tm_results['cross_fold_ensemble']['mean_rmse']:.4f} ± {tm_results['cross_fold_ensemble']['std_rmse']:.4f}")
    print(f"  Variance reduction: {tm_results['cross_fold_ensemble']['variance_reduction']:.2f}%")

    print(f"\nEgc:")
    print(f"  Mean RMSE: {egc_results['cross_fold_ensemble']['mean_rmse']:.4f} ± {egc_results['cross_fold_ensemble']['std_rmse']:.4f}")
    print(f"  Variance reduction: {egc_results['cross_fold_ensemble']['variance_reduction']:.2f}%")

    print(f"\nResults saved to: {results_path}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
