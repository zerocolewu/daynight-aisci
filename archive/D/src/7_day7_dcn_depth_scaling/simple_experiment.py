#!/usr/bin/env python3
"""
Simple DCN-GIN Experiment - Focused comparison at depth 10
Tests: DCN-GIN-10 vs Standard-GIN-10 vs CFD-MTL baseline
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# Simplified feature extraction (80-dim)
# =============================================================================
def compute_features_simple(mol):
    """Compute 80 features."""
    features = []

    try:
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        # Basic properties
        features.extend([
            Descriptors.MolWt(mol),
            Descriptors.HeavyAtomCount(mol),
            num_atoms,
            num_bonds,
            Descriptors.RingCount(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol) / max(num_bonds, 1),
        ])

        # Atom type counts
        atom_types = {'C': 0, 'N': 0, 'O': 0, 'S': 0, 'F': 0, 'Cl': 0, 'Br': 0, 'P': 0}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in atom_types:
                atom_types[symbol] += 1
        features.extend(list(atom_types.values()))

        # Bond type fractions
        single_bonds = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE)
        double_bonds = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
        aromatic_bonds = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.AROMATIC)

        total = max(num_bonds, 1)
        features.extend([
            single_bonds / total,
            double_bonds / total,
            aromatic_bonds / total,
        ])

        # Hybridization fractions
        sp2_count = sum(1 for a in mol.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP2)
        sp3_count = sum(1 for a in mol.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP3)
        features.extend([
            sp2_count / max(num_atoms, 1),
            sp3_count / max(num_atoms, 1),
        ])

        # Pad to 80
        while len(features) < 80:
            features.append(0.0)

        features = np.array(features[:80], dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    except Exception as e:
        return np.zeros(80, dtype=np.float32)


def mol_to_graph(mol, target, features):
    """Convert to PyG graph."""
    try:
        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append([
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
            ])

        x = torch.tensor(atom_features, dtype=torch.float)

        # Edges
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])
            edge_attr.extend([
                [int(bond.GetBondType() == Chem.BondType.SINGLE),
                 int(bond.GetBondType() == Chem.BondType.DOUBLE),
                 int(bond.GetBondType() == Chem.BondType.AROMATIC)],
            ] * 2)

        if not edge_index:
            edge_index = [[0, 0]]
            edge_attr = [[0, 0, 0]]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([target]))
        data.mol_features = torch.tensor(features, dtype=torch.float).unsqueeze(0)

        return data
    except:
        return None


def load_dataset(csv_path):
    """Load Tg dataset."""
    print(f"Loading from {csv_path}")
    df = pd.read_csv(csv_path)

    data_list = []
    features_list = []

    for _, row in df.iterrows():
        psmiles = row['Smiles']
        tg = row['Tg']

        # Process SMILES
        smiles = psmiles.replace('*', '[H]')
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                continue
            Chem.SanitizeMol(mol)
        except:
            continue

        # Compute features
        features = compute_features_simple(mol)

        # Create graph
        data = mol_to_graph(mol, tg, features)
        if data is not None:
            data_list.append(data)
            features_list.append(features)

    print(f"Loaded {len(data_list)} samples")

    features = np.array(features_list)
    scaler = StandardScaler()
    scaler.fit(features)

    return data_list, features, scaler


# =============================================================================
# Models
# =============================================================================
class DenseGINLayer(nn.Module):
    """Single dense GIN layer."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index):
        # Simple aggregation
        row, col = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, col, x[row])
        return self.mlp((1 + self.eps) * x + out)


class DCNGIN(nn.Module):
    """Dense connectivity GIN."""
    def __init__(self, num_layers=10, hidden_dim=256, feature_dim=80, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers

        self.atom_encoder = nn.Linear(6, hidden_dim)
        self.feature_proj = nn.Linear(feature_dim, 128)

        # Dense layers
        self.layers = nn.ModuleList()
        growth_rate = 32
        current_dim = hidden_dim

        for i in range(num_layers):
            self.layers.append(DenseGINLayer(current_dim, growth_rate))
            current_dim += growth_rate

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(current_dim + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x, edge_index, features, batch):
        x = self.atom_encoder(x)

        # Dense connectivity
        features_list = [x]
        for layer in self.layers:
            out = layer(torch.cat(features_list, dim=-1), edge_index)
            features_list.append(out)

        # Pool
        graph_emb = global_mean_pool(torch.cat(features_list, dim=-1), batch)

        # Feature fusion
        feat_emb = self.feature_proj(features)
        combined = torch.cat([graph_emb, feat_emb], dim=-1)

        return self.head(combined)


class StandardGIN(nn.Module):
    """Standard GIN without dense connectivity."""
    def __init__(self, num_layers=10, hidden_dim=256, feature_dim=80, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers

        self.atom_encoder = nn.Linear(6, hidden_dim)
        self.feature_proj = nn.Linear(feature_dim, 128)

        # Standard layers
        self.layers = nn.ModuleList([
            DenseGINLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x, edge_index, features, batch):
        x = self.atom_encoder(x)

        # Standard sequential layers
        for layer, bn in zip(self.layers, self.bns):
            x = layer(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        # Pool
        graph_emb = global_mean_pool(x, batch)

        # Feature fusion
        feat_emb = self.feature_proj(features)
        combined = torch.cat([graph_emb, feat_emb], dim=-1)

        return self.head(combined)


# =============================================================================
# Training
# =============================================================================
def train_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        features = batch.mol_features

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            pred = model(batch.x, batch.edge_index, features, batch.batch)
            loss = nn.MSELoss()(pred.squeeze(), batch.y.squeeze())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, targets = [], []

    for batch in loader:
        batch = batch.to(device)
        features = batch.mol_features
        pred = model(batch.x, batch.edge_index, features, batch.batch)
        preds.append(pred.cpu())
        targets.append(batch.y.cpu())

    preds = torch.cat(preds).squeeze().numpy()
    targets = torch.cat(targets).squeeze().numpy()

    return np.sqrt(mean_squared_error(targets, preds))


def train_model(model_class, model_params, data_list, features, scaler, epochs=200, patience=30):
    """Train with 5-fold CV."""
    fold_rmses = []

    for fold in range(5):
        print(f"  Fold {fold}...", end=' ')

        # Split data
        n = len(data_list)
        indices = np.arange(n)
        np.random.seed(42 + fold)
        np.random.shuffle(indices)

        val_start = int(n * 0.8)
        test_start = int(n * 0.9)

        train_idx = indices[:val_start]
        val_idx = indices[val_start:test_start]
        test_idx = indices[test_start:]

        # Prepare data
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        test_data = [data_list[i] for i in test_idx]

        # Normalize features
        for i, idx in enumerate(train_idx):
            train_data[i].mol_features = torch.tensor(
                scaler.transform(features[idx:idx+1]), dtype=torch.float
            )
        for i, idx in enumerate(val_idx):
            val_data[i].mol_features = torch.tensor(
                scaler.transform(features[idx:idx+1]), dtype=torch.float
            )
        for i, idx in enumerate(test_idx):
            test_data[i].mol_features = torch.tensor(
                scaler.transform(features[idx:idx+1]), dtype=torch.float
            )

        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=64)
        test_loader = DataLoader(test_data, batch_size=64)

        # Train
        model = model_class(**model_params).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        scaler_amp = torch.amp.GradScaler('cuda')

        best_val = float('inf')
        patience_count = 0

        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, scaler_amp)
            val_rmse = evaluate(model, val_loader)
            scheduler.step(val_rmse)

            if val_rmse < best_val:
                best_val = val_rmse
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    break

        test_rmse = evaluate(model, test_loader)
        fold_rmses.append(test_rmse)
        print(f"Test RMSE: {test_rmse:.2f}")

    return np.mean(fold_rmses), np.std(fold_rmses), fold_rmses


# =============================================================================
# Main
# =============================================================================
def main():
    print("="*80)
    print("DCN-GIN Depth Scaling Experiment")
    print("="*80)

    # Load data
    csv_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    data_list, features, scaler = load_dataset(csv_path)

    results = []

    # Test DCN-GIN-10
    print("\n" + "="*80)
    print("Training DCN-GIN-10")
    print("="*80)

    mean_rmse, std_rmse, fold_rmses = train_model(
        DCNGIN,
        {'num_layers': 10, 'hidden_dim': 256, 'feature_dim': 80, 'dropout': 0.2},
        data_list, features, scaler
    )

    results.append({
        'model': 'DCN',
        'depth': 10,
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'fold_rmses': fold_rmses
    })

    # Test Standard-GIN-10
    print("\n" + "="*80)
    print("Training Standard-GIN-10")
    print("="*80)

    mean_rmse, std_rmse, fold_rmses = train_model(
        StandardGIN,
        {'num_layers': 10, 'hidden_dim': 256, 'feature_dim': 80, 'dropout': 0.2},
        data_list, features, scaler
    )

    results.append({
        'model': 'Standard',
        'depth': 10,
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'fold_rmses': fold_rmses
    })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)

    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    baselines = {
        'Day 2 5-layer GIN': 32.31,
        'CFD-MTL 3-layer GIN': 44.12
    }

    for r in results:
        print(f"\n{r['model']}-GIN-{r['depth']}: {r['mean_rmse']:.2f} ± {r['std_rmse']:.2f}")
        for name, baseline in baselines.items():
            improvement = (baseline - r['mean_rmse']) / baseline * 100
            print(f"  vs {name}: {improvement:+.1f}%")

    # Comparison
    dcn = [r for r in results if r['model'] == 'DCN'][0]
    std = [r for r in results if r['model'] == 'Standard'][0]

    improvement = (std['mean_rmse'] - dcn['mean_rmse']) / std['mean_rmse'] * 100
    print(f"\nDCN vs Standard at depth 10: {improvement:+.1f}% improvement")

    print("\n" + "="*80)
    print("Experiment completed!")
    print("="*80)


if __name__ == "__main__":
    main()
