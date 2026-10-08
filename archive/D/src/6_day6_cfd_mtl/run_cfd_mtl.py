#!/usr/bin/env python3
"""
Corrected Feature-Decomposed Multi-Task Learning (CFD-MTL) - Optimized Version

This version is optimized for faster execution while maintaining the key features:
1. Validated features with strong correlations
2. Feature decomposition based on property-specific correlations
3. Multi-task learning with shared encoder
"""

import os
import sys
import json
import warnings
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_smiles

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, rdMolDescriptors

warnings.filterwarnings('ignore')
torch.set_num_threads(4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# VALIDATED FEATURES
# =============================================================================

def compute_validated_features(mol) -> Tuple[np.ndarray, List[str]]:
    """Compute validated features with strong property correlations."""
    features = []
    feature_names = []

    try:
        mol_weight = Descriptors.MolWt(mol)
        heavy_atoms = Descriptors.HeavyAtomCount(mol)
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        total_bonds = num_bonds if num_bonds > 0 else 1
        total_atoms = num_atoms if num_atoms > 0 else 1

        # Basic properties
        features.extend([mol_weight, heavy_atoms, num_atoms, num_bonds])
        feature_names.extend(['mol_weight', 'heavy_atoms', 'num_atoms', 'num_bonds'])

        # Ring info
        ring_count = Descriptors.RingCount(mol)
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        frac_csp3 = Descriptors.FractionCSP3(mol)

        features.extend([ring_count, num_rings, num_aromatic_rings, frac_csp3])
        feature_names.extend(['RingCount', 'NumRings', 'NumAromaticRings', 'FractionCSP3'])

        # CRITICAL: Rotatable bonds (r=-0.73 with Tg)
        num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
        rotatable_bond_frac = num_rotatable / total_bonds
        features.extend([num_rotatable, rotatable_bond_frac])
        feature_names.extend(['NumRotatableBonds', 'rotatable_bond_frac'])

        # Atom types
        atom_types = {'C': 0, 'N': 0, 'O': 0, 'S': 0, 'F': 0, 'Cl': 0, 'Br': 0}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in atom_types:
                atom_types[symbol] += 1
        features.extend(list(atom_types.values()))
        feature_names.extend([f'{k}_count' for k in atom_types.keys()])

        # CRITICAL: Bond type fractions
        single_bonds = double_bonds = triple_bonds = aromatic_bonds = 0
        for bond in mol.GetBonds():
            bt = bond.GetBondType()
            if bt == Chem.BondType.SINGLE: single_bonds += 1
            elif bt == Chem.BondType.DOUBLE: double_bonds += 1
            elif bt == Chem.BondType.TRIPLE: triple_bonds += 1
            elif bt == Chem.BondType.AROMATIC: aromatic_bonds += 1

        single_bond_frac = single_bonds / total_bonds
        double_bond_frac = double_bonds / total_bonds
        aromatic_bond_frac = aromatic_bonds / total_bonds

        features.extend([single_bonds, double_bonds, aromatic_bonds,
                        single_bond_frac, double_bond_frac, aromatic_bond_frac])
        feature_names.extend(['single_bonds', 'double_bonds', 'aromatic_bonds',
                             'single_bond_frac', 'double_bond_frac', 'aromatic_bond_frac'])

        # CRITICAL: Hybridization (sp2_frac r=0.70 with Tg, r=-0.72 with Egc)
        sp2_count = sp3_count = 0
        for atom in mol.GetAtoms():
            hyb = atom.GetHybridization()
            if hyb == Chem.HybridizationType.SP2: sp2_count += 1
            elif hyb == Chem.HybridizationType.SP3: sp3_count += 1

        sp2_frac = sp2_count / total_atoms
        sp3_frac = sp3_count / total_atoms
        features.extend([sp2_count, sp3_count, sp2_frac, sp3_frac])
        feature_names.extend(['sp2_count', 'sp3_count', 'sp2_frac', 'sp3_frac'])

        # CRITICAL: Conjugation features (r=-0.747 with Egc)
        conjugated_bonds = sp2_sp2_bonds = aromatic_atoms = 0
        for bond in mol.GetBonds():
            if bond.GetIsConjugated() or bond.GetBondType() == Chem.BondType.AROMATIC:
                conjugated_bonds += 1
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            if (begin_atom.GetHybridization() == Chem.HybridizationType.SP2 and
                end_atom.GetHybridization() == Chem.HybridizationType.SP2):
                sp2_sp2_bonds += 1

        for atom in mol.GetAtoms():
            if atom.GetIsAromatic():
                aromatic_atoms += 1

        conjugated_bond_fraction = conjugated_bonds / total_bonds
        sp2_sp2_bond_fraction = sp2_sp2_bonds / total_bonds
        aromatic_atom_fraction = aromatic_atoms / total_atoms

        features.extend([conjugated_bonds, conjugated_bond_fraction, sp2_sp2_bonds,
                        sp2_sp2_bond_fraction, aromatic_atoms, aromatic_atom_fraction])
        feature_names.extend(['conjugated_bonds', 'conjugated_bond_fraction', 'sp2_sp2_bonds',
                             'sp2_sp2_bond_fraction', 'aromatic_atoms', 'aromatic_atom_fraction'])

        # Additional descriptors
        features.extend([
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHeteroatoms(mol),
        ])
        feature_names.extend(['MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumHeteroatoms'])

        # Ring sizes
        ring_info = mol.GetRingInfo()
        ring_sizes = [0] * 6
        for ring in ring_info.AtomRings():
            size = len(ring)
            if 3 <= size <= 8:
                ring_sizes[size - 3] += 1
        features.extend(ring_sizes)
        feature_names.extend(['ring3', 'ring4', 'ring5', 'ring6', 'ring7', 'ring8'])

        # Pad to 80 features (enough for validation)
        while len(features) < 80:
            features.append(0.0)
            feature_names.append(f'padding_{len(features)}')

        features = features[:80]
        feature_names = feature_names[:80]

        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features, feature_names

    except Exception as e:
        return np.zeros(80, dtype=np.float32), ['error'] * 80


def process_psmiles(psmiles: str):
    smiles = psmiles.replace('*', '[H]')
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None, None
        Chem.SanitizeMol(mol)
        return mol, smiles
    except:
        return None, None


# =============================================================================
# MODELS
# =============================================================================

class GINEncoder(nn.Module):
    def __init__(self, num_node_features, hidden_dim=128, num_layers=3):
        super().__init__()
        self.node_embedding = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(nn_layer))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.node_embedding(x)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        return global_mean_pool(x, batch)


class SingleTaskModel(nn.Module):
    def __init__(self, num_node_features, num_features=80, hidden_dim=128):
        super().__init__()
        self.encoder = GINEncoder(num_node_features, hidden_dim)
        self.feature_bn = nn.BatchNorm1d(num_features)
        self.feature_proj = nn.Linear(num_features, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        z = self.encoder(data.x, data.edge_index, data.batch)
        f = data.features
        if f.dim() == 3: f = f.squeeze(1)
        f = self.feature_proj(self.feature_bn(f))
        return self.head(torch.cat([z, f], dim=-1)).squeeze(-1)


class CFD_MTL(nn.Module):
    def __init__(self, num_node_features, num_features=80, hidden_dim=128,
                 shared_idx=None, tg_idx=None, tm_idx=None, egc_idx=None):
        super().__init__()
        self.encoder = GINEncoder(num_node_features, hidden_dim)

        self.shared_idx = shared_idx if shared_idx else list(range(num_features))
        self.tg_idx = tg_idx if tg_idx else list(range(num_features))
        self.tm_idx = tm_idx if tm_idx else list(range(num_features))
        self.egc_idx = egc_idx if egc_idx else list(range(num_features))

        self.shared_bn = nn.BatchNorm1d(len(self.shared_idx))
        self.tg_bn = nn.BatchNorm1d(len(self.tg_idx))
        self.tm_bn = nn.BatchNorm1d(len(self.tm_idx))
        self.egc_bn = nn.BatchNorm1d(len(self.egc_idx))

        self.shared_proj = nn.Linear(len(self.shared_idx), hidden_dim)
        self.tg_proj = nn.Linear(len(self.tg_idx), hidden_dim)
        self.tm_proj = nn.Linear(len(self.tm_idx), hidden_dim)
        self.egc_proj = nn.Linear(len(self.egc_idx), hidden_dim)

        def make_head(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1)
            )

        self.tg_head = make_head(hidden_dim * 2)
        self.tm_head = make_head(hidden_dim * 2)
        self.egc_head = make_head(hidden_dim * 2)

    def forward(self, data, task='all'):
        z = self.encoder(data.x, data.edge_index, data.batch)
        f = data.features
        if f.dim() == 3: f = f.squeeze(1)

        shared_f = self.shared_proj(self.shared_bn(f[:, self.shared_idx]))

        outputs = {}
        if task in ['tg', 'all']:
            tg_f = self.tg_proj(self.tg_bn(f[:, self.tg_idx]))
            outputs['tg'] = self.tg_head(torch.cat([z, tg_f], dim=-1)).squeeze(-1)
        if task in ['tm', 'all']:
            tm_f = self.tm_proj(self.tm_bn(f[:, self.tm_idx]))
            outputs['tm'] = self.tm_head(torch.cat([z, tm_f], dim=-1)).squeeze(-1)
        if task in ['egc', 'all']:
            egc_f = self.egc_proj(self.egc_bn(f[:, self.egc_idx]))
            outputs['egc'] = self.egc_head(torch.cat([z, egc_f], dim=-1)).squeeze(-1)

        return outputs


# =============================================================================
# DATA PROCESSING
# =============================================================================

def create_pyg_graph(smiles):
    try:
        data = from_smiles(smiles)
        return data if data and data.x is not None else None
    except:
        return None


def load_dataset(property_name, data_path, cache_path=None):
    print(f"\nLoading {property_name}...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} samples")

    if property_name == 'Tg':
        smiles_col, value_col = 'Smiles', 'Tg'
    elif property_name == 'Tm':
        smiles_col, value_col = 'Smiles', 'Tm'
    else:
        smiles_col, value_col = 'smiles', 'value'

    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)

    all_data, all_features, all_targets = [], [], []

    for idx, row in df.iterrows():
        mol, smiles = process_psmiles(row[smiles_col])
        if mol is None: continue

        pyg_data = create_pyg_graph(smiles)
        if pyg_data is None: continue

        if row[smiles_col] in cache:
            features, feature_names = cache[row[smiles_col]]
        else:
            features, feature_names = compute_validated_features(mol)
            cache[row[smiles_col]] = (features, feature_names)

        pyg_data.y = torch.tensor([row[value_col]], dtype=torch.float)
        pyg_data.features = torch.tensor(features, dtype=torch.float).unsqueeze(0)
        all_data.append(pyg_data)
        all_features.append(features)
        all_targets.append(row[value_col])

    print(f"  Processed {len(all_data)} molecules")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)

    return all_data, np.array(all_features), np.array(all_targets), feature_names


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def analyze_correlations(tg_f, tg_y, tm_f, tm_y, egc_f, egc_y, names, output_file=None):
    print("\n" + "="*60)
    print("FEATURE-PROPERTY CORRELATIONS")
    print("="*60)

    correlations = []
    for i in range(min(len(names), tg_f.shape[1])):
        try:
            tg_r, _ = pearsonr(tg_f[:, i], tg_y)
            tm_r, _ = pearsonr(tm_f[:, i], tm_y)
            egc_r, _ = pearsonr(egc_f[:, i], egc_y)
        except:
            tg_r = tm_r = egc_r = 0.0

        correlations.append({
            'idx': i, 'name': names[i],
            'tg': tg_r, 'tm': tm_r, 'egc': egc_r,
            'abs_tg': abs(tg_r), 'abs_tm': abs(tm_r), 'abs_egc': abs(egc_r)
        })

    threshold = 0.3
    shared, tg_spec, tm_spec, egc_spec = [], [], [], []

    for c in correlations:
        imp_tg = c['abs_tg'] > threshold
        imp_tm = c['abs_tm'] > threshold
        imp_egc = c['abs_egc'] > threshold
        n_imp = sum([imp_tg, imp_tm, imp_egc])

        if n_imp >= 2: shared.append(c['idx'])
        elif imp_tg: tg_spec.append(c['idx'])
        elif imp_tm: tm_spec.append(c['idx'])
        elif imp_egc: egc_spec.append(c['idx'])

    print(f"  Shared: {len(shared)}, Tg-spec: {len(tg_spec)}, Tm-spec: {len(tm_spec)}, Egc-spec: {len(egc_spec)}")

    # Print top correlations
    print("\n  Top correlations:")
    for prop, key in [('Tg', 'tg'), ('Tm', 'tm'), ('Egc', 'egc')]:
        sorted_corrs = sorted(correlations, key=lambda x: x[f'abs_{key}'], reverse=True)[:5]
        print(f"\n  {prop}:")
        for c in sorted_corrs:
            print(f"    {c['name']}: r={c[key]:.4f}")

    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"Shared: {shared}\n")
            f.write(f"Tg-specific: {tg_spec}\n")
            f.write(f"Tm-specific: {tm_spec}\n")
            f.write(f"Egc-specific: {egc_spec}\n")

    return {'shared': shared, 'tg_specific': tg_spec, 'tm_specific': tm_spec, 'egc_specific': egc_spec}


# =============================================================================
# TRAINING
# =============================================================================

def train_single(model, train_loader, val_loader, epochs=300, patience=30):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()
    best_val, patience_cnt, best_state = float('inf'), 0, None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_preds.extend(model(batch).cpu().numpy())
                val_targets.extend(batch.y.cpu().numpy())

        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))

        if val_rmse < best_val:
            best_val = val_rmse
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def train_mtl(model, train_loaders, val_data, epochs=300, patience=30):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()
    best_val, patience_cnt, best_state = float('inf'), 0, None

    for epoch in range(epochs):
        model.train()
        for task, loader in train_loaders.items():
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch, task)[task], batch.y)
                loss.backward()
                optimizer.step()

        model.eval()
        val_rmses = []
        with torch.no_grad():
            for task, vdata in val_data.items():
                preds, targets = [], []
                for batch in DataLoader(vdata, batch_size=64, shuffle=False):
                    batch = batch.to(device)
                    preds.extend(model(batch, task)[task].cpu().numpy())
                    targets.extend(batch.y.cpu().numpy())
                val_rmses.append(np.sqrt(mean_squared_error(targets, preds)))

        avg_val = np.mean(val_rmses)
        if avg_val < best_val:
            best_val = avg_val
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate(model, test_loader, scaler=None, task='single'):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            if task == 'single':
                output = model(batch)
            else:
                output = model(batch, task=task)[task]
            preds.extend(output.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())

    preds, targets = np.array(preds), np.array(targets)

    if scaler:
        preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        targets = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(targets, preds))
    return rmse, preds, targets


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    print("\n" + "="*80)
    print("CFD-MTL EXPERIMENT")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")

    base_dir = Path('/home/home_public/aisci/AIscientist/src/6_day6_cfd_mtl')
    data_dir = Path('/home/home_public/aisci/AIscientist/src/alldataset')
    cache_dir = base_dir / 'cache'
    cache_dir.mkdir(exist_ok=True)

    # Load datasets
    tg_data, tg_f, tg_y, names = load_dataset('Tg', data_dir/'Tg.csv', cache_dir/'tg.pkl')
    tm_data, tm_f, tm_y, _ = load_dataset('Tm', data_dir/'Tm.csv', cache_dir/'tm.pkl')
    egc_data, egc_f, egc_y, _ = load_dataset('Egc', data_dir/'Egc.csv', cache_dir/'egc.pkl')

    # Analyze correlations
    feature_info = analyze_correlations(tg_f, tg_y, tm_f, tm_y, egc_f, egc_y, names,
                                        base_dir/'feature_indices.txt')

    # Scale targets
    tg_scaler = StandardScaler().fit(tg_y.reshape(-1,1))
    tm_scaler = StandardScaler().fit(tm_y.reshape(-1,1))
    egc_scaler = StandardScaler().fit(egc_y.reshape(-1,1))

    for i, d in enumerate(tg_data):
        d.y = torch.tensor([tg_scaler.transform([[tg_y[i]]])[0,0]], dtype=torch.float)
    for i, d in enumerate(tm_data):
        d.y = torch.tensor([tm_scaler.transform([[tm_y[i]]])[0,0]], dtype=torch.float)
    for i, d in enumerate(egc_data):
        d.y = torch.tensor([egc_scaler.transform([[egc_y[i]]])[0,0]], dtype=torch.float)

    num_node_features = tg_data[0].x.shape[1]
    num_features = len(names)

    results = {}

    # ==========================================================================
    # Ablation 1: Single-task
    # ==========================================================================
    print("\n" + "-"*60)
    print("ABLATION 1: SINGLE-TASK")
    print("-"*60)

    results['single'] = {}
    for prop, data, scaler in [('Tg', tg_data, tg_scaler), ('Tm', tm_data, tm_scaler), ('Egc', egc_data, egc_scaler)]:
        print(f"\n  {prop}...")
        fold_rmses = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
            train_data = [data[i] for i in train_idx]
            test_data = [data[i] for i in test_idx]

            # Normalize features
            train_f = np.array([d.features.numpy().squeeze(0) for d in train_data])
            feat_scaler = StandardScaler().fit(train_f)
            for d in train_data + test_data:
                d.features = torch.tensor(feat_scaler.transform(d.features.numpy().squeeze(0).reshape(1,-1)), dtype=torch.float)

            # Split train/val
            val_data = train_data[-int(len(train_data)*0.111):]
            train_data = train_data[:-int(len(train_data)*0.111)]

            train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

            model = SingleTaskModel(num_node_features, num_features).to(device)
            model = train_single(model, train_loader, val_loader)

            rmse, _, _ = evaluate(model, test_loader, scaler)
            fold_rmses.append(rmse)
            print(f"    Fold {fold+1}: {rmse:.4f}")

        results['single'][prop] = {
            'fold_rmses': fold_rmses,
            'mean_rmse': float(np.mean(fold_rmses)),
            'std_rmse': float(np.std(fold_rmses))
        }

    # ==========================================================================
    # Ablation 2: Naive MTL
    # ==========================================================================
    print("\n" + "-"*60)
    print("ABLATION 2: NAIVE MTL (all features)")
    print("-"*60)

    all_idx = list(range(num_features))
    results['mtl_naive'] = run_mtl(
        tg_data, tm_data, egc_data,
        all_idx, all_idx, all_idx,
        tg_scaler, tm_scaler, egc_scaler,
        num_node_features, num_features, 5
    )

    # ==========================================================================
    # Ablation 3: CFD-MTL
    # ==========================================================================
    print("\n" + "-"*60)
    print("ABLATION 3: CFD-MTL (feature decomposition)")
    print("-"*60)

    shared = feature_info['shared']
    tg_idx = shared + feature_info['tg_specific']
    tm_idx = shared + feature_info['tm_specific']
    egc_idx = shared + feature_info['egc_specific']

    if not tg_idx: tg_idx = all_idx
    if not tm_idx: tm_idx = all_idx
    if not egc_idx: egc_idx = all_idx

    results['cfd_mtl'] = run_mtl(
        tg_data, tm_data, egc_data,
        tg_idx, tm_idx, egc_idx,
        tg_scaler, tm_scaler, egc_scaler,
        num_node_features, num_features, 5
    )

    # ==========================================================================
    # Report
    # ==========================================================================
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print(f"\n{'Model':<25} {'Tg':<15} {'Tm':<15} {'Egc':<15}")
    print("-"*60)

    for name, key in [('Single-Task', 'single'), ('Naive MTL', 'mtl_naive'), ('CFD-MTL', 'cfd_mtl')]:
        r = results[key]
        print(f"{name:<25} "
              f"{r['Tg']['mean_rmse']:.2f}±{r['Tg']['std_rmse']:.2f}   "
              f"{r['Tm']['mean_rmse']:.2f}±{r['Tm']['std_rmse']:.2f}   "
              f"{r['Egc']['mean_rmse']:.4f}±{r['Egc']['std_rmse']:.4f}")

    # Save results
    with open(base_dir/'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Generate report
    with open(base_dir/'results.txt', 'w') as f:
        f.write("CFD-MTL RESULTS\n")
        f.write("="*60 + "\n\n")

        f.write("Baselines (literature):\n")
        f.write("  Tg: 32.31\n")
        f.write("  Tm: 59.18±1.12\n")
        f.write("  Egc: 0.5262±0.0471\n\n")

        for name, key in [('Single-Task', 'single'), ('Naive MTL', 'mtl_naive'), ('CFD-MTL', 'cfd_mtl')]:
            r = results[key]
            f.write(f"{name}:\n")
            f.write(f"  Tg: {r['Tg']['mean_rmse']:.4f}±{r['Tg']['std_rmse']:.4f}\n")
            f.write(f"  Tm: {r['Tm']['mean_rmse']:.4f}±{r['Tm']['std_rmse']:.4f}\n")
            f.write(f"  Egc: {r['Egc']['mean_rmse']:.4f}±{r['Egc']['std_rmse']:.4f}\n\n")

    print(f"\nEnd: {datetime.now().strftime('%H:%M:%S')}")
    return results


def run_mtl(tg_data, tm_data, egc_data, tg_idx, tm_idx, egc_idx,
            tg_scaler, tm_scaler, egc_scaler, num_node_features, num_features, n_folds):
    fold_results = {'Tg': [], 'Tm': [], 'Egc': []}

    for fold in range(n_folds):
        print(f"  Fold {fold+1}/{n_folds}")

        np.random.seed(42 + fold)
        tg_idx_shuf = np.random.permutation(len(tg_data))
        tm_idx_shuf = np.random.permutation(len(tm_data))
        egc_idx_shuf = np.random.permutation(len(egc_data))

        # Split
        tg_train = [tg_data[i] for i in tg_idx_shuf[:int(0.8*len(tg_idx_shuf))]]
        tg_test = [tg_data[i] for i in tg_idx_shuf[int(0.8*len(tg_idx_shuf)):]]
        tm_train = [tm_data[i] for i in tm_idx_shuf[:int(0.8*len(tm_idx_shuf))]]
        tm_test = [tm_data[i] for i in tm_idx_shuf[int(0.8*len(tm_idx_shuf)):]]
        egc_train = [egc_data[i] for i in egc_idx_shuf[:int(0.8*len(egc_idx_shuf))]]
        egc_test = [egc_data[i] for i in egc_idx_shuf[int(0.8*len(egc_idx_shuf)):]]

        # Val split
        tg_val = tg_train[-int(len(tg_train)*0.111):]
        tg_train = tg_train[:-int(len(tg_train)*0.111)]
        tm_val = tm_train[-int(len(tm_train)*0.111):]
        tm_train = tm_train[:-int(len(tm_train)*0.111)]
        egc_val = egc_train[-int(len(egc_train)*0.111):]
        egc_train = egc_train[:-int(len(egc_train)*0.111)]

        # Normalize features
        all_train = tg_train + tm_train + egc_train
        train_f = np.array([d.features.numpy().squeeze(0) for d in all_train])
        feat_scaler = StandardScaler().fit(train_f)

        for d in tg_train + tg_val + tg_test + tm_train + tm_val + tm_test + egc_train + egc_val + egc_test:
            d.features = torch.tensor(feat_scaler.transform(d.features.numpy().squeeze(0).reshape(1,-1)), dtype=torch.float)

        # Create loaders
        train_loaders = {
            'tg': DataLoader(tg_train, batch_size=64, shuffle=True),
            'tm': DataLoader(tm_train, batch_size=64, shuffle=True),
            'egc': DataLoader(egc_train, batch_size=64, shuffle=True)
        }
        val_data = {'tg': tg_val, 'tm': tm_val, 'egc': egc_val}

        # Train MTL model
        model = CFD_MTL(num_node_features, num_features, 128,
                       shared_idx=tg_idx, tg_idx=tg_idx, tm_idx=tm_idx, egc_idx=egc_idx).to(device)
        model = train_mtl(model, train_loaders, val_data)

        # Evaluate
        for prop, test_data, scaler in [('Tg', tg_test, tg_scaler), ('Tm', tm_test, tm_scaler), ('Egc', egc_test, egc_scaler)]:
            test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
            rmse, _, _ = evaluate(model, test_loader, scaler, prop.lower())
            fold_results[prop].append(rmse)

        print(f"    Tg: {fold_results['Tg'][-1]:.4f}, Tm: {fold_results['Tm'][-1]:.4f}, Egc: {fold_results['Egc'][-1]:.4f}")

    return {
        prop: {
            'fold_rmses': fold_results[prop],
            'mean_rmse': float(np.mean(fold_results[prop])),
            'std_rmse': float(np.std(fold_results[prop]))
        }
        for prop in ['Tg', 'Tm', 'Egc']
    }


if __name__ == '__main__':
    results = run_experiment()
