#!/usr/bin/env python3
"""
Day 4: Feature Baseline for Tm and Egc Prediction (Version 2 - Fixed p-SMILES handling)
Extends Day 2's validated feature engineering approach to Tm and Egc datasets.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.loader import DataLoader

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, rdMolDescriptors

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


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
        features.append(Descriptors.NumRotatableBonds(mol))
        features.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        features.append(Descriptors.FractionCSP3(mol))
        features.append(rdMolDescriptors.CalcNumAmideBonds(mol))
        features.append(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))

        # EState indices (4)
        features.append(Descriptors.MaxEStateIndex(mol))
        features.append(Descriptors.MinEStateIndex(mol))
        features.append(Descriptors.MaxAbsEStateIndex(mol))
        features.append(Descriptors.MinAbsEStateIndex(mol))

        # Additional descriptors (25)
        features.append(Descriptors.HeavyAtomMolWt(mol))
        features.append(Descriptors.ExactMolWt(mol))
        features.append(Descriptors.NumValenceElectrons(mol))
        features.append(Descriptors.NumRadicalElectrons(mol))
        features.append(Descriptors.FpDensityMorgan1(mol))
        features.append(Descriptors.FpDensityMorgan2(mol))
        features.append(Descriptors.FpDensityMorgan3(mol))
        features.append(Descriptors.HeavyAtomCount(mol))
        features.append(Descriptors.NHOHCount(mol))
        features.append(Descriptors.NOCount(mol))
        features.append(Descriptors.NumAliphaticCarbocycles(mol))
        features.append(Descriptors.NumAliphaticHeterocycles(mol))
        features.append(Descriptors.NumAliphaticRings(mol))
        features.append(Descriptors.NumAromaticCarbocycles(mol))
        features.append(Descriptors.NumAromaticHeterocycles(mol))
        features.append(Descriptors.NumAromaticRings(mol))
        features.append(Descriptors.NumSaturatedCarbocycles(mol))
        features.append(Descriptors.NumSaturatedHeterocycles(mol))
        features.append(Descriptors.NumSaturatedRings(mol))
        features.append(Descriptors.qed(mol))
        features.append(Descriptors.SPS(mol))
        features.append(Descriptors.MolLogP(mol))
        features.append(Descriptors.MolMR(mol))
        features.append(Descriptors.TPSA(mol))
        features.append(Descriptors.LabuteASA(mol))

        # Polymer-specific features (60)
        # Atom type counts
        atom_types = {'C': 0, 'N': 0, 'O': 0, 'S': 0, 'F': 0, 'Cl': 0, 'Br': 0, 'P': 0, 'Si': 0}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in atom_types:
                atom_types[symbol] += 1
        features.extend(list(atom_types.values()))

        # Bond type counts
        single_bonds = 0
        double_bonds = 0
        triple_bonds = 0
        aromatic_bonds = 0
        for bond in mol.GetBonds():
            bt = bond.GetBondType()
            if bt == Chem.BondType.SINGLE:
                single_bonds += 1
            elif bt == Chem.BondType.DOUBLE:
                double_bonds += 1
            elif bt == Chem.BondType.TRIPLE:
                triple_bonds += 1
            elif bt == Chem.BondType.AROMATIC:
                aromatic_bonds += 1

        total_bonds = mol.GetNumBonds() if mol.GetNumBonds() > 0 else 1
        features.extend([
            single_bonds,
            double_bonds,
            triple_bonds,
            aromatic_bonds,
            single_bonds / total_bonds,
            double_bonds / total_bonds,
            triple_bonds / total_bonds,
            aromatic_bonds / total_bonds
        ])

        # Hybridization counts
        sp = 0
        sp2 = 0
        sp3 = 0
        for atom in mol.GetAtoms():
            hyb = atom.GetHybridization()
            if hyb == Chem.HybridizationType.SP:
                sp += 1
            elif hyb == Chem.HybridizationType.SP2:
                sp2 += 1
            elif hyb == Chem.HybridizationType.SP3:
                sp3 += 1

        total_atoms = mol.GetNumAtoms() if mol.GetNumAtoms() > 0 else 1
        features.extend([
            sp, sp2, sp3,
            sp / total_atoms, sp2 / total_atoms, sp3 / total_atoms
        ])

        # Ring size distribution (3-8 membered rings)
        ring_info = mol.GetRingInfo()
        ring_sizes = [0] * 6  # for 3-8 membered rings
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

        # Additional 8 features to reach 148
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
    """
    Process p-SMILES to get RDKit mol object.
    Polymer SMILES format: *-connected repeat units, asterisks mark attachment points.

    Strategy: Replace asterisks with hydrogens to create a valid monomer SMILES.
    This preserves the chemical structure while making it processable by RDKit.
    """
    # Convert p-SMILES to regular SMILES by replacing asterisks with hydrogens
    # The asterisk represents the polymer connection point
    smiles = psmiles.replace('*', '[H]')

    # Try to create mol object
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None, None

        # Sanitize molecule
        Chem.SanitizeMol(mol)

        # Return the modified SMILES for PyG processing
        return mol, smiles

    except Exception as e:
        # If sanitization fails, try without explicit H replacement
        try:
            smiles_simple = psmiles.replace('*', '')
            mol = Chem.MolFromSmiles(smiles_simple)
            if mol is not None:
                return mol, smiles_simple
        except:
            pass
        return None, None


class PolymerDataset(torch.utils.data.Dataset):
    """Custom dataset for polymer data with features."""

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_fn(batch):
    """Custom collate function to handle variable-size graphs."""
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


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

        # Output head
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        features = data.features

        # Ensure x is float
        x = x.float()

        # Node embeddings
        x = self.node_embedding(x)

        # GIN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)

        # Mean aggregation
        x_pool = global_mean_pool(x, batch)

        # Process features - squeeze to (batch_size, num_features)
        if features.dim() == 3:
            features = features.squeeze(1)
        features = self.feature_bn(features)
        features_proj = self.feature_proj(features)

        # Concatenate
        combined = torch.cat([x_pool, features_proj], dim=1)

        # Output
        out = self.output(combined)
        return out.squeeze(-1)


def create_pyg_graph(mol, smiles):
    """Create PyG Data object from RDKit mol."""
    from torch_geometric.utils import from_smiles

    try:
        data = from_smiles(smiles)
        if data is None or data.x is None:
            return None
        return data
    except:
        return None


def train_model(model, train_loader, val_loader, optimizer, criterion, device, patience=50, max_epochs=500):
    """Train model with early stopping."""
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch.y)
                val_loss += loss.item()
                val_preds.extend(output.cpu().numpy())
                val_targets.extend(batch.y.cpu().numpy())

        val_loss /= len(val_loader)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))

        # Early stopping
        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val RMSE={val_rmse:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_loss


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            predictions.extend(output.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    return rmse, mae, r2, predictions, targets


def run_cross_validation(dataset_name, data_path, output_dir, n_folds=5):
    """Run 5-fold cross-validation for a dataset."""
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} dataset")
    print(f"{'='*60}\n")

    # Load dataset
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")

    # Determine column names
    if dataset_name == 'Tm':
        smiles_col = 'Smiles'
        value_col = 'Tm'
    else:  # Egc
        smiles_col = 'smiles'
        value_col = 'value'

    # Process molecules
    print("Processing molecules and computing features...")
    all_data = []
    all_features = []
    all_targets = []
    failed = 0
    failed_examples = []

    for idx, row in df.iterrows():
        psmiles = row[smiles_col]
        target = row[value_col]

        mol, smiles = process_psmiles(psmiles)

        if mol is None:
            failed += 1
            if len(failed_examples) < 5:
                failed_examples.append(psmiles)
            continue

        # Create PyG graph
        pyg_data = create_pyg_graph(mol, smiles)
        if pyg_data is None:
            failed += 1
            continue

        # Compute features
        features = compute_148_features(mol)

        # Store data
        pyg_data.y = torch.tensor([target], dtype=torch.float)
        pyg_data.features = torch.tensor(features, dtype=torch.float).unsqueeze(0)
        all_data.append(pyg_data)
        all_features.append(features)
        all_targets.append(target)

    print(f"Successfully processed {len(all_data)} molecules ({failed} failed)")
    if failed_examples:
        print(f"Example failed SMILES: {failed_examples[:3]}")

    if len(all_data) == 0:
        print("ERROR: No valid molecules processed!")
        return None, None, None, None, None

    # Convert to arrays
    all_features = np.array(all_features)
    all_targets = np.array(all_targets)

    # Normalize targets
    target_scaler = StandardScaler()
    all_targets_scaled = target_scaler.fit_transform(all_targets.reshape(-1, 1)).flatten()

    # Update target values
    for i, data in enumerate(all_data):
        data.y = torch.tensor([all_targets_scaled[i]], dtype=torch.float)

    # Cross-validation
    print(f"\nRunning {n_folds}-fold cross-validation...")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    all_predictions = []
    all_targets_list = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_data)):
        print(f"\nFold {fold + 1}/{n_folds}")

        # Split data
        train_data = [all_data[i] for i in train_idx]
        test_data = [all_data[i] for i in test_idx]

        # Further split train into train/val (80/10/10 overall)
        val_size = int(len(train_data) * 0.111)  # ~10% of total
        val_data = train_data[-val_size:]
        train_data = train_data[:-val_size]

        print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        # Normalize features
        train_features = np.array([d.features.numpy().squeeze(0) for d in train_data])
        feature_scaler = StandardScaler()
        feature_scaler.fit(train_features)

        # Apply normalization to all splits
        for data_list in [train_data, val_data, test_data]:
            for d in data_list:
                normalized = feature_scaler.transform(d.features.numpy().squeeze(0).reshape(1, -1))
                d.features = torch.tensor(normalized, dtype=torch.float)

        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        # Initialize model
        num_node_features = train_data[0].x.shape[1]
        model = GINWithFeatures(num_node_features=num_node_features, num_features=148).to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = nn.MSELoss()

        # Train
        model, best_val_loss = train_model(
            model, train_loader, val_loader, optimizer, criterion, device,
            patience=50, max_epochs=500
        )

        # Evaluate
        rmse_scaled, mae_scaled, r2, preds, targets = evaluate_model(model, test_loader, device)

        # Convert back to original scale
        preds_original = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        targets_original = target_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

        rmse = np.sqrt(mean_squared_error(targets_original, preds_original))
        mae = mean_absolute_error(targets_original, preds_original)

        print(f"  Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        fold_results.append({
            'fold': fold + 1,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'best_val_loss': best_val_loss
        })

        all_predictions.extend(preds_original)
        all_targets_list.extend(targets_original)

        # Save model
        model_path = output_dir / 'trained_models' / f'{dataset_name.lower()}_fold{fold+1}.pt'
        torch.save(model.state_dict(), model_path)

    # Aggregate results
    rmses = [r['rmse'] for r in fold_results]
    maes = [r['mae'] for r in fold_results]
    r2s = [r['r2'] for r in fold_results]

    results = {
        'dataset': dataset_name,
        'n_samples': len(all_data),
        'n_failed': failed,
        'n_folds': n_folds,
        'fold_results': fold_results,
        'mean_rmse': float(np.mean(rmses)),
        'std_rmse': float(np.std(rmses)),
        'mean_mae': float(np.mean(maes)),
        'std_mae': float(np.std(maes)),
        'mean_r2': float(np.mean(r2s)),
        'std_r2': float(np.std(r2s)),
        'best_rmse': float(np.min(rmses)),
        'best_fold': int(np.argmin(rmses) + 1)
    }

    print(f"\n{dataset_name} Results:")
    print(f"  Mean RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
    print(f"  Mean MAE: {results['mean_mae']:.4f} ± {results['std_mae']:.4f}")
    print(f"  Mean R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
    print(f"  Best RMSE: {results['best_rmse']:.4f} (Fold {results['best_fold']})")

    return results, all_features, all_targets, all_predictions, all_targets_list


def analyze_feature_importance(features, targets, dataset_name, output_dir, top_k=20):
    """Analyze feature importance using Random Forest."""
    print(f"\nAnalyzing feature importance for {dataset_name}...")

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(features, targets)

    # Feature names (simplified)
    feature_names = [
        'MolWt', 'HeavyAtomCount', 'NumAtoms', 'NumBonds', 'NumHeavyAtoms',
        'NumValenceElectrons', 'NumRadicalElectrons', 'RingCount', 'FractionCSP3',
        'NumRings', 'NumAromaticRings', 'NumAliphaticRings', 'NumSaturatedRings',
        'NumAromaticHeterocycles', 'NumAromaticCarbocycles', 'NumHeterocycles',
        'NumBridgeheadAtoms', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge',
        'MinAbsPartialCharge', 'MolLogP', 'MolMR', 'TPSA', 'LabuteASA',
        'BertzCT', 'BalabanJ', 'HallKierAlpha', 'Kappa1', 'Kappa2',
        'Kappa3', 'Chi0', 'Chi1', 'Chi0n', 'Chi1n',
        'Chi2n', 'Chi3n', 'Chi4n', 'Chi0v', 'Chi1v',
        'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms', 'NumHeterocycles', 'NHOHCount',
        'NOCount', 'NumRotatableBonds', 'NumRotatableBonds2', 'FractionCSP3_2', 'NumAmideBonds',
        'NumBridgeheadAtoms2', 'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex',
        'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons2', 'NumRadicalElectrons2', 'FpDensityMorgan1',
        'FpDensityMorgan2', 'FpDensityMorgan3', 'HeavyAtomCount2', 'NHOHCount2', 'NOCount2',
        'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings2', 'NumAromaticCarbocycles2',
        'NumAromaticHeterocycles2', 'NumAromaticRings2', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles',
        'NumSaturatedRings2', 'qed', 'SPS', 'MolLogP2', 'MolMR2',
        'TPSA2', 'LabuteASA2', 'C_count', 'N_count', 'O_count',
        'S_count', 'F_count', 'Cl_count', 'Br_count', 'P_count',
        'Si_count', 'single_bonds', 'double_bonds', 'triple_bonds', 'aromatic_bonds',
        'single_bond_frac', 'double_bond_frac', 'triple_bond_frac', 'aromatic_bond_frac', 'sp_count',
        'sp2_count', 'sp3_count', 'sp_frac', 'sp2_frac', 'sp3_frac',
        'ring3', 'ring4', 'ring5', 'ring6', 'ring7',
        'ring8', 'alcohol', 'amine', 'carbonyl', 'carboxylic',
        'ester', 'ether', 'amide', 'nitrile', 'halogen',
        'nitro', 'sulfide', 'phosphate', 'rotatable_bond_frac', 'NumAromaticRings3',
        'NumAliphaticRings3', 'NumSaturatedRings3', 'NumConformers', 'HeavyAtomMolWt_ratio', 'BertzCT2',
        'BalabanJ2'
    ]

    # Get importance
    importance = rf.feature_importances_
    indices = np.argsort(importance)[::-1]

    # Save top features
    top_features = []
    for i in range(min(top_k, len(feature_names))):
        idx = indices[i]
        top_features.append({
            'rank': i + 1,
            'feature': feature_names[idx] if idx < len(feature_names) else f'feature_{idx}',
            'importance': float(importance[idx])
        })

    importance_results = {
        'dataset': dataset_name,
        'top_features': top_features
    }

    # Save
    output_path = output_dir / f'{dataset_name.lower()}_feature_importance.json'
    with open(output_path, 'w') as f:
        json.dump(importance_results, f, indent=2)

    print(f"  Top 5 features for {dataset_name}:")
    for i in range(min(5, len(top_features))):
        tf = top_features[i]
        print(f"    {tf['rank']}. {tf['feature']}: {tf['importance']:.4f}")

    return importance_results


def main():
    """Main function."""
    print(f"\n{'='*60}")
    print(f"Day 4: Feature Baseline for Tm and Egc (Version 2)")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Setup
    base_dir = Path('/home/home_public/aisci/AIscientist/src/3_day4_feature_baseline_tm_egc')
    data_dir = Path('/home/home_public/aisci/AIscientist/src/alldataset')
    output_dir = base_dir
    model_dir = output_dir / 'trained_models'
    model_dir.mkdir(exist_ok=True)

    # Process Tm dataset
    tm_results, tm_features, tm_targets, tm_preds, tm_true = run_cross_validation(
        'Tm', data_dir / 'Tm.csv', output_dir, n_folds=5
    )

    if tm_results is not None:
        # Save Tm results
        with open(output_dir / 'tm_baseline_results_v2.json', 'w') as f:
            json.dump(tm_results, f, indent=2)

        # Analyze Tm feature importance
        tm_importance = analyze_feature_importance(tm_features, tm_targets, 'Tm', output_dir)

    # Process Egc dataset
    egc_results, egc_features, egc_targets, egc_preds, egc_true = run_cross_validation(
        'Egc', data_dir / 'Egc.csv', output_dir, n_folds=5
    )

    if egc_results is not None:
        # Save Egc results
        with open(output_dir / 'egc_baseline_results_v2.json', 'w') as f:
            json.dump(egc_results, f, indent=2)

        # Analyze Egc feature importance
        egc_importance = analyze_feature_importance(egc_features, egc_targets, 'Egc', output_dir)

    # Cross-property comparison
    if tm_results is not None and egc_results is not None:
        print("\n" + "="*60)
        print("Cross-property Feature Comparison")
        print("="*60)

        tm_top_features = set([f['feature'] for f in tm_importance['top_features'][:10]])
        egc_top_features = set([f['feature'] for f in egc_importance['top_features'][:10]])

        common_features = tm_top_features.intersection(egc_top_features)

        comparison = {
            'tm_top_10': list(tm_top_features),
            'egc_top_10': list(egc_top_features),
            'common_features': list(common_features),
            'n_common': len(common_features)
        }

        with open(output_dir / 'cross_property_comparison_v2.json', 'w') as f:
            json.dump(comparison, f, indent=2)

        print(f"\nCommon top-10 features between Tm and Egc: {len(common_features)}")
        for feat in common_features:
            print(f"  - {feat}")

    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    if tm_results is not None:
        print(f"\nTm Prediction:")
        print(f"  Samples used: {tm_results['n_samples']}/{tm_results['n_samples']+tm_results['n_failed']}")
        print(f"  Mean RMSE: {tm_results['mean_rmse']:.4f} ± {tm_results['std_rmse']:.4f}")
        print(f"  Target: < 60.0 (GAT baseline: 79.3)")
        print(f"  Status: {'ACHIEVED' if tm_results['mean_rmse'] < 60.0 else 'NOT ACHIEVED'}")

    if egc_results is not None:
        print(f"\nEgc Prediction:")
        print(f"  Samples used: {egc_results['n_samples']}/{egc_results['n_samples']+egc_results['n_failed']}")
        print(f"  Mean RMSE: {egc_results['mean_rmse']:.4f} ± {egc_results['std_rmse']:.4f}")
        print(f"  Target: < 0.50 (GAT baseline: 0.63)")
        print(f"  Status: {'ACHIEVED' if egc_results['mean_rmse'] < 0.50 else 'NOT ACHIEVED'}")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == '__main__':
    main()
