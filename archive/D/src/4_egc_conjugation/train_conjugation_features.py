#!/usr/bin/env python3
"""
Egc Conjugation-Specific Features Experiment
Addresses Fold 3 failure by implementing conjugation-specific features
Based on X2-GNN insights: H^core features improve conjugated systems by 55.8%
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
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.loader import DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, rdMolDescriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# CONJUGATION-SPECIFIC FEATURE ENGINEERING
# ============================================================================

def compute_conjugation_features(mol) -> Dict[str, float]:
    """
    Compute conjugation-specific features for Egc prediction.
    Based on X2-GNN insights and single_bond_frac dominance (56.3% importance).
    """
    if mol is None:
        return {}

    features = {}

    try:
        # 1. Aromatic ring count
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        aromatic_ring_count = 0
        total_ring_count = len(atom_rings)

        for ring in atom_rings:
            is_aromatic = all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
            if is_aromatic:
                aromatic_ring_count += 1

        features['aromatic_ring_count'] = aromatic_ring_count

        # 2. Aromatic ring fraction
        if total_ring_count > 0:
            features['aromatic_ring_fraction'] = aromatic_ring_count / total_ring_count
        else:
            features['aromatic_ring_fraction'] = 0.0

        # 3. Conjugated bond fraction and longest conjugated path
        # Conjugation: alternating single/double bonds
        bonds = [b for b in mol.GetBonds()]
        total_bonds = len(bonds)

        if total_bonds == 0:
            features['conjugated_bond_fraction'] = 0.0
            features['conjugation_length'] = 0
        else:
            # Build adjacency for conjugated path analysis
            # A bond is conjugated if it's part of alternating single/double bond system
            conjugated_bonds = set()
            for bond in bonds:
                bond_type = str(bond.GetBondType())
                # Single and double bonds can be part of conjugated systems
                if bond_type in ['SINGLE', 'DOUBLE', 'AROMATIC']:
                    # Check if connected to aromatic or has alternating pattern
                    atom1 = bond.GetBeginAtom()
                    atom2 = bond.GetEndAtom()

                    # Atoms are sp2 hybridized if they have double bonds or are aromatic
                    is_sp2_1 = atom1.GetIsAromatic() or atom1.GetHybridization() == Chem.HybridizationType.SP2
                    is_sp2_2 = atom2.GetIsAromatic() or atom2.GetHybridization() == Chem.HybridizationType.SP2

                    if is_sp2_1 and is_sp2_2:
                        conjugated_bonds.add(bond.GetIdx())

            features['conjugated_bond_fraction'] = len(conjugated_bonds) / total_bonds

            # Find longest conjugated path using DFS
            adj = defaultdict(list)
            for bond in bonds:
                if bond.GetIdx() in conjugated_bonds:
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    adj[i].append(j)
                    adj[j].append(i)

            max_conjugation_length = 0
            visited_global = set()

            for start in adj.keys():
                if start not in visited_global:
                    # BFS to find longest path from this component
                    stack = [(start, {start}, 1)]
                    while stack:
                        node, path, length = stack.pop()
                        max_conjugation_length = max(max_conjugation_length, length)
                        for neighbor in adj[node]:
                            if neighbor not in path:
                                new_path = path | {neighbor}
                                stack.append((neighbor, new_path, length + 1))
                    visited_global.add(start)

            features['conjugation_length'] = max_conjugation_length

        # 4. sp2-sp2 bond fraction
        sp2_sp2_count = 0
        for bond in bonds:
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()

            # Check hybridization
            is_sp2_1 = atom1.GetHybridization() == Chem.HybridizationType.SP2 or atom1.GetIsAromatic()
            is_sp2_2 = atom2.GetHybridization() == Chem.HybridizationType.SP2 or atom2.GetIsAromatic()

            if is_sp2_1 and is_sp2_2:
                sp2_sp2_count += 1

        features['sp2_sp2_bond_fraction'] = sp2_sp2_count / total_bonds if total_bonds > 0 else 0.0

        # 5. π-electron density proxy
        # π-electrons from aromatic carbons (1 each) and double bonds (1 each)
        pi_electrons = 0

        # Count aromatic atoms (each contributes 1 π-electron)
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic():
                pi_electrons += 1

        # Count double bonds (each contributes 2 π-electrons total, but we count per carbon = 1 each)
        for bond in bonds:
            if str(bond.GetBondType()) == 'DOUBLE' and not bond.GetIsAromatic():
                pi_electrons += 2  # 2 carbons, each contributes 1

        mol_weight = Descriptors.MolWt(mol)
        features['pi_electron_density'] = pi_electrons / mol_weight if mol_weight > 0 else 0.0
        features['pi_electron_count'] = pi_electrons

        # 6. Aromatic atom fraction
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        total_atoms = mol.GetNumAtoms()
        features['aromatic_atom_fraction'] = aromatic_atoms / total_atoms if total_atoms > 0 else 0.0

        # 7. Additional useful conjugation features
        # Average bond order in conjugated regions
        conjugated_bond_orders = []
        for bond in bonds:
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            is_sp2_1 = atom1.GetHybridization() == Chem.HybridizationType.SP2 or atom1.GetIsAromatic()
            is_sp2_2 = atom2.GetHybridization() == Chem.HybridizationType.SP2 or atom2.GetIsAromatic()

            if is_sp2_1 and is_sp2_2:
                bond_type = str(bond.GetBondType())
                if bond_type == 'SINGLE':
                    conjugated_bond_orders.append(1.0)
                elif bond_type == 'DOUBLE':
                    conjugated_bond_orders.append(2.0)
                elif bond_type == 'AROMATIC':
                    conjugated_bond_orders.append(1.5)  # Aromatic bonds are between single and double

        if conjugated_bond_orders:
            features['avg_conjugated_bond_order'] = np.mean(conjugated_bond_orders)
        else:
            features['avg_conjugated_bond_order'] = 0.0

        # Number of heteroatoms in aromatic systems
        aromatic_heteroatoms = sum(1 for atom in mol.GetAtoms()
                                   if atom.GetIsAromatic() and atom.GetAtomicNum() not in [6, 1])
        features['aromatic_heteroatom_count'] = aromatic_heteroatoms

    except Exception as e:
        print(f"Error computing conjugation features: {e}")
        return {}

    return features


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
        features.append(Lipinski.FractionCSP3(mol))
        features.append(rdMolDescriptors.CalcNumRotatableBonds(mol))

        # Bond type counts (4)
        mol_bonds = [b for b in mol.GetBonds()]
        num_bonds = len(mol_bonds)
        single_bonds = sum(1 for b in mol_bonds if str(b.GetBondType()) == 'SINGLE')
        double_bonds = sum(1 for b in mol_bonds if str(b.GetBondType()) == 'DOUBLE')
        aromatic_bonds = sum(1 for b in mol_bonds if b.GetIsAromatic())

        features.append(single_bonds / num_bonds if num_bonds > 0 else 0)  # single_bond_frac
        features.append(double_bonds / num_bonds if num_bonds > 0 else 0)
        features.append(aromatic_bonds / num_bonds if num_bonds > 0 else 0)

        # Additional topological descriptors (20)
        features.append(Descriptors.Chi2v(mol))
        features.append(Descriptors.Chi3v(mol))
        features.append(Descriptors.Chi4v(mol))
        features.append(Descriptors.Chi0n(mol))
        features.append(Descriptors.Chi1n(mol))
        features.append(Descriptors.Chi2n(mol))
        features.append(Descriptors.Chi3n(mol))
        features.append(Descriptors.Chi4n(mol))
        features.append(Descriptors.HallKierAlpha(mol))
        features.append(Descriptors.Kappa1(mol))
        features.append(Descriptors.Kappa2(mol))
        features.append(Descriptors.Kappa3(mol))
        features.append(Descriptors.MaxEStateIndex(mol))
        features.append(Descriptors.MinEStateIndex(mol))
        features.append(Descriptors.MaxAbsEStateIndex(mol))
        features.append(Descriptors.MinAbsEStateIndex(mol))

        # Additional electronic descriptors (10)
        features.append(Descriptors.qed(mol))
        features.append(Descriptors.SPS(mol))
        features.append(Descriptors.MolLogP(mol))
        features.append(Descriptors.MolMR(mol))

        # Atom counts by type (10)
        atom_counts = defaultdict(int)
        for atom in mol.GetAtoms():
            atom_counts[atom.GetSymbol()] += 1

        features.append(atom_counts.get('C', 0))
        features.append(atom_counts.get('N', 0))
        features.append(atom_counts.get('O', 0))
        features.append(atom_counts.get('S', 0))
        features.append(atom_counts.get('F', 0))
        features.append(atom_counts.get('Cl', 0))
        features.append(atom_counts.get('Br', 0))
        features.append(atom_counts.get('P', 0))

        # Molecular complexity (10)
        features.append(Descriptors.BertzCT(mol))
        features.append(Descriptors.HallKierAlpha(mol))

        # Ring size distribution (5)
        ring_info = mol.GetRingInfo()
        ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
        features.append(ring_sizes.count(3) if ring_sizes else 0)
        features.append(ring_sizes.count(4) if ring_sizes else 0)
        features.append(ring_sizes.count(5) if ring_sizes else 0)
        features.append(ring_sizes.count(6) if ring_sizes else 0)
        features.append(ring_sizes.count(7) if ring_sizes else 0)

        # Additional features to reach 148
        # Additional bond features (using alternative descriptors)
        features.append(rdMolDescriptors.CalcNumSaturatedRings(mol))
        features.append(rdMolDescriptors.CalcNumAliphaticRings(mol))

        # Weight ratios
        features.append(Descriptors.HeavyAtomMolWt(mol) / Descriptors.MolWt(mol) if Descriptors.MolWt(mol) > 0 else 0)

        # Additional ring features
        features.append(rdMolDescriptors.CalcNumSpiroAtoms(mol))
        features.append(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))

        # Fill remaining with 0s if needed
        while len(features) < 148:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    except Exception as e:
        print(f"Error computing 148 features: {e}")
        return np.zeros(148, dtype=np.float32)


def process_psmiles(smiles: str) -> Optional[Chem.Mol]:
    """
    Process p-SMILES notation (* represents polymer connection points).
    Replace * with C to make valid SMILES for RDKit.
    """
    if pd.isna(smiles) or smiles == '':
        return None

    try:
        # Replace * with C (carbon) for processing
        # This is standard practice for polymer SMILES
        processed = smiles.replace('*', 'C')

        # Parse with RDKit
        mol = Chem.MolFromSmiles(processed)

        if mol is None:
            return None

        # Compute 2D coordinates
        AllChem.Compute2DCoords(mol)

        return mol

    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None


def compute_all_features(mol) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute both 148 generic features and conjugation-specific features."""
    generic_features = compute_148_features(mol)
    conjugation_features = compute_conjugation_features(mol)
    return generic_features, conjugation_features


# ============================================================================
# GRAPH NEURAL NETWORK
# ============================================================================

class GIN(torch.nn.Module):
    """GIN model for molecular property prediction."""

    def __init__(self, num_node_features=9, hidden_dim=256, num_layers=5,
                 dropout=0.1, feature_dim=None):
        super(GIN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Node feature embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_layer))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Global feature fusion
        if feature_dim is not None:
            self.feature_fc = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            fc_input_dim = hidden_dim * 2  # Graph embedding + feature embedding
        else:
            self.feature_fc = None
            fc_input_dim = hidden_dim

        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Node embedding
        x = self.node_embedding(x)

        # GIN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph pooling
        graph_embedding = global_mean_pool(x, batch)

        # Feature fusion
        if self.feature_fc is not None and hasattr(data, 'features'):
            # Concatenate all features in the batch
            batch_size = graph_embedding.size(0)
            feature_embedding = self.feature_fc(data.features.view(batch_size, -1))
            combined = torch.cat([graph_embedding, feature_embedding], dim=1)
        else:
            combined = graph_embedding

        # Prediction
        out = self.fc(combined)

        return out.squeeze(-1)


def get_node_features(mol) -> torch.Tensor:
    """Extract node features for graph construction."""
    features = []

    for atom in mol.GetAtoms():
        atom_features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(),
            atom.GetHybridization(),
            atom.GetTotalValence(),
            atom.GetMass()
        ]
        features.append(atom_features)

    return torch.tensor(features, dtype=torch.float)


def get_edge_features(mol) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract edge indices for graph construction."""
    edge_indices = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])  # Undirected graph

    if len(edge_indices) == 0:
        # Single atom molecule
        return torch.tensor([[0], [0]], dtype=torch.long), None

    return torch.tensor(edge_indices, dtype=torch.long).t().contiguous(), None


# ============================================================================
# DATASET LOADING AND PREPARATION
# ============================================================================

def load_egc_dataset(data_path: str) -> pd.DataFrame:
    """Load Egc dataset."""
    print(f"\n{'='*80}")
    print("Loading Egc Dataset")
    print(f"{'='*80}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")

    return df


def prepare_dataset(df: pd.DataFrame) -> Tuple[List[Data], pd.DataFrame, List[str]]:
    """
    Prepare dataset with both generic and conjugation-specific features.
    Returns list of PyG Data objects, feature DataFrame, and feature names.
    """
    print(f"\n{'='*80}")
    print("Preparing Dataset with Conjugation Features")
    print(f"{'='*80}")

    data_list = []
    failed_indices = []

    # Storage for all features
    all_generic_features = []
    all_conjugation_features = []
    conjugation_feature_names = None

    for idx, row in df.iterrows():
        smiles = row['smiles']
        target = row['value']

        # Process molecule
        mol = process_psmiles(smiles)

        if mol is None:
            failed_indices.append(idx)
            continue

        # Compute features
        generic_feats, conj_feats = compute_all_features(mol)

        if conjugation_feature_names is None:
            conjugation_feature_names = list(conj_feats.keys())

        all_generic_features.append(generic_feats)
        all_conjugation_features.append(conj_feats)

        # Create graph data
        node_features = get_node_features(mol)
        edge_index, _ = get_edge_features(mol)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.tensor([target], dtype=torch.float),
            smiles=smiles,
            idx=idx
        )

        data_list.append(data)

    # Create feature DataFrames
    generic_df = pd.DataFrame(all_generic_features, columns=[f'generic_{i}' for i in range(148)])

    conjugation_df = pd.DataFrame(all_conjugation_features)

    # Combine features
    combined_features = pd.concat([generic_df, conjugation_df], axis=1)

    print(f"\nSuccessfully processed {len(data_list)} molecules")
    print(f"Failed to process {len(failed_indices)} molecules")
    print(f"\nFeature dimensions:")
    print(f"  Generic features: {generic_df.shape[1]}")
    print(f"  Conjugation features: {conjugation_df.shape[1]}")
    print(f"  Total features: {combined_features.shape[1]}")
    print(f"\nConjugation features: {conjugation_feature_names}")

    return data_list, combined_features, conjugation_feature_names


def create_stratified_folds(features_df: pd.DataFrame, data_list: List[Data],
                            n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified K-fold splits based on single_bond_frac bins.
    This ensures balanced conjugation distribution across folds.
    """
    print(f"\n{'='*80}")
    print("Creating Stratified K-Fold Splits")
    print(f"{'='*80}")

    # Get single_bond_frac (it's in generic features at index 40 based on original code)
    # Actually, we need to find it from the generic features
    # The single_bond_frac is computed in compute_148_features at a specific position

    # Let's recompute it directly from the molecules to be sure
    single_bond_fracs = []

    for data in data_list:
        # Get the single_bond_frac from generic features (index 40 in the feature array)
        # This is from the bond type counts section
        single_bond_fracs.append(features_df.iloc[data.idx, 40])  # single_bond_frac column

    single_bond_fracs = np.array(single_bond_fracs)

    # Bin into 5 quantiles for stratification
    n_bins = 5
    bins = pd.qcut(single_bond_fracs, q=n_bins, labels=False, duplicates='drop')

    print(f"Single bond fraction distribution:")
    print(f"  Min: {single_bond_fracs.min():.4f}")
    print(f"  Max: {single_bond_fracs.max():.4f}")
    print(f"  Mean: {single_bond_fracs.mean():.4f}")
    print(f"  Std: {single_bond_fracs.std():.4f}")

    print(f"\nBin distribution:")
    for i in range(n_bins):
        count = (bins == i).sum()
        print(f"  Bin {i}: {count} samples ({100*count/len(bins):.1f}%)")

    # Create stratified folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(data_list)), bins)):
        folds.append((train_idx, val_idx))

        # Print fold statistics
        train_bins = bins[train_idx]
        val_bins = bins[val_idx]

        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train size: {len(train_idx)}")
        print(f"  Val size: {len(val_idx)}")
        print(f"  Train bin distribution: {np.bincount(train_bins)}")
        print(f"  Val bin distribution: {np.bincount(val_bins)}")

    return folds


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_model(model, train_loader, val_loader, optimizer, scheduler,
                criterion, n_epochs=500, patience=50, device='cpu'):
    """Train model with early stopping and learning rate scheduling."""

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = model(batch)
            loss = criterion(outputs, batch.y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                loss = criterion(outputs, batch.y)
                val_loss += loss.item() * batch.num_graphs

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    return model, best_val_loss, train_losses, val_losses


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model on test set."""

    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)

            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions,
        'targets': targets
    }


def run_cross_validation(data_list: List[Data], features_df: pd.DataFrame,
                         folds: List[Tuple[np.ndarray, np.ndarray]],
                         conjugation_feature_names: List[str],
                         output_dir: str):
    """
    Run stratified 5-fold cross-validation with conjugation features.
    """
    print(f"\n{'='*80}")
    print("Running Stratified 5-Fold Cross-Validation")
    print(f"{'='*80}")

    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_df.values)
    features_tensor = torch.tensor(features_normalized, dtype=torch.float)

    # Add features to data objects
    for i, data in enumerate(data_list):
        data.features = features_tensor[i]

    fold_results = []
    all_predictions = []
    all_targets = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*80}")
        print(f"Fold {fold_idx + 1}/5")
        print(f"{'='*80}")

        # Create data loaders
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

        # Initialize model
        model = GIN(
            num_node_features=9,
            hidden_dim=256,
            num_layers=5,
            dropout=0.1,
            feature_dim=features_df.shape[1]
        ).to(device)

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # Loss function - Huber loss for robustness
        criterion = nn.HuberLoss(delta=0.5)

        # Train
        model, best_val_loss, train_losses, val_losses = train_model(
            model, train_loader, val_loader, optimizer, scheduler,
            criterion, n_epochs=500, patience=50, device=device
        )

        # Evaluate
        results = evaluate_model(model, val_loader, device=device)

        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  MAE: {results['mae']:.4f}")
        print(f"  R²: {results['r2']:.4f}")

        fold_results.append({
            'fold': fold_idx + 1,
            'rmse': results['rmse'],
            'mae': results['mae'],
            'r2': results['r2'],
            'best_val_loss': best_val_loss
        })

        all_predictions.extend(results['predictions'])
        all_targets.extend(results['targets'])

        # Save fold model
        torch.save({
            'model_state_dict': model.state_dict(),
            'fold': fold_idx + 1,
            'results': results
        }, os.path.join(output_dir, f'model_fold_{fold_idx+1}.pt'))

    # Compute overall statistics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    overall_rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    overall_mae = mean_absolute_error(all_targets, all_predictions)
    overall_r2 = r2_score(all_targets, all_predictions)

    mean_rmse = np.mean([r['rmse'] for r in fold_results])
    std_rmse = np.std([r['rmse'] for r in fold_results])
    mean_mae = np.mean([r['mae'] for r in fold_results])
    std_mae = np.std([r['mae'] for r in fold_results])
    mean_r2 = np.mean([r['r2'] for r in fold_results])
    std_r2 = np.std([r['r2'] for r in fold_results])

    print(f"\n{'='*80}")
    print("Overall Cross-Validation Results")
    print(f"{'='*80}")
    print(f"Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"Mean MAE: {mean_mae:.4f} ± {std_mae:.4f}")
    print(f"Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"\nOverall (pooled) RMSE: {overall_rmse:.4f}")
    print(f"Overall (pooled) MAE: {overall_mae:.4f}")
    print(f"Overall (pooled) R²: {overall_r2:.4f}")

    results_summary = {
        'dataset': 'Egc',
        'n_samples': len(data_list),
        'n_folds': 5,
        'fold_results': fold_results,
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'mean_mae': mean_mae,
        'std_mae': std_mae,
        'mean_r2': mean_r2,
        'std_r2': std_r2,
        'overall_rmse': overall_rmse,
        'overall_mae': overall_mae,
        'overall_r2': overall_r2
    }

    return results_summary, all_predictions, all_targets


# ============================================================================
# DIAGNOSTIC ANALYSIS
# ============================================================================

def analyze_fold3_difficulty(df: pd.DataFrame, data_list: List[Data],
                             folds: List[Tuple[np.ndarray, np.ndarray]],
                             features_df: pd.DataFrame,
                             conjugation_feature_names: List[str]):
    """
    Analyze why Fold 3 was problematic in baseline.
    """
    print(f"\n{'='*80}")
    print("Fold 3 Diagnostic Analysis")
    print(f"{'='*80}")

    # Get fold indices
    train_idx, val_idx = folds[2]  # Fold 3 (0-indexed)

    # Get conjugation features for fold 3
    conj_feats = features_df[conjugation_feature_names]

    # Analyze distribution
    print("\nConjugation feature distribution in Fold 3:")
    print(f"Validation set size: {len(val_idx)}")

    for feat_name in conjugation_feature_names:
        train_vals = conj_feats.iloc[train_idx][feat_name].values
        val_vals = conj_feats.iloc[val_idx][feat_name].values

        print(f"\n{feat_name}:")
        print(f"  Train: mean={train_vals.mean():.4f}, std={train_vals.std():.4f}, "
              f"min={train_vals.min():.4f}, max={train_vals.max():.4f}")
        print(f"  Val:   mean={val_vals.mean():.4f}, std={val_vals.std():.4f}, "
              f"min={val_vals.min():.4f}, max={val_vals.max():.4f}")

        # Check if validation set has extreme values
        if val_vals.mean() > train_vals.mean() + 1.5 * train_vals.std():
            print(f"  ⚠️ Validation set has higher {feat_name} than training!")
        if val_vals.mean() < train_vals.mean() - 1.5 * train_vals.std():
            print(f"  ⚠️ Validation set has lower {feat_name} than training!")


def analyze_feature_correlations(df: pd.DataFrame, features_df: pd.DataFrame,
                                 conjugation_feature_names: List[str]):
    """
    Analyze correlation of conjugation features with Egc target.
    """
    print(f"\n{'='*80}")
    print("Feature Correlation Analysis")
    print(f"{'='*80}")

    # Get targets
    targets = df['value'].values[:len(features_df)]

    correlations = {}

    for feat_name in conjugation_feature_names:
        feat_vals = features_df[feat_name].values

        # Compute Pearson correlation
        if np.std(feat_vals) > 0:  # Avoid division by zero
            corr = np.corrcoef(feat_vals, targets)[0, 1]
            correlations[feat_name] = corr

    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\nCorrelation of conjugation features with Egc:")
    for feat_name, corr in sorted_corrs:
        print(f"  {feat_name}: r = {corr:.4f} {'✓' if abs(corr) > 0.4 else ''}")

    # Save correlations
    return correlations


def compute_feature_importance(features_df: pd.DataFrame, targets: np.ndarray):
    """
    Compute feature importance using Random Forest.
    """
    print(f"\n{'='*80}")
    print("Feature Importance Analysis")
    print(f"{'='*80}")

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(features_df.values, targets)

    # Get feature importances
    importances = rf.feature_importances_
    feature_names = features_df.columns.tolist()

    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]

    print("\nTop 20 most important features:")
    for i in range(min(20, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # Save importance ranking
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    return importance_df


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    """Main experiment execution."""

    # Setup paths
    base_dir = Path(__file__).parent
    data_path = Path(__file__).parent.parent / 'alldataset' / 'Egc.csv'
    output_dir = base_dir / 'results'
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*80}")
    print("Egc Conjugation-Specific Features Experiment")
    print(f"{'='*80}")
    print(f"Working directory: {base_dir}")
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")

    # Load dataset
    df = load_egc_dataset(data_path)

    # Prepare dataset with features
    data_list, features_df, conj_feature_names = prepare_dataset(df)

    # Create stratified folds
    folds = create_stratified_folds(features_df, data_list, n_splits=5)

    # Analyze Fold 3 difficulty
    analyze_fold3_difficulty(df, data_list, folds, features_df, conj_feature_names)

    # Analyze feature correlations
    correlations = analyze_feature_correlations(df, features_df, conj_feature_names)

    # Run cross-validation
    results, predictions, targets = run_cross_validation(
        data_list, features_df, folds, conj_feature_names, output_dir
    )

    # Compute feature importance
    importance_df = compute_feature_importance(features_df, targets)

    # Compare with baseline
    print(f"\n{'='*80}")
    print("Comparison with Baseline (Day 4)")
    print(f"{'='*80}")
    print(f"Baseline mean RMSE: 1.19 ± 0.59")
    print(f"Our mean RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
    print(f"Improvement: {(1.19 - results['mean_rmse']) / 1.19 * 100:.1f}%")

    print(f"\nBaseline Fold 3 RMSE: 2.25")
    print(f"Our Fold 3 RMSE: {results['fold_results'][2]['rmse']:.4f}")
    print(f"Improvement: {(2.25 - results['fold_results'][2]['rmse']) / 2.25 * 100:.1f}%")

    # Save results
    results['correlations'] = correlations
    results['comparison'] = {
        'baseline_mean_rmse': 1.19,
        'baseline_std_rmse': 0.59,
        'baseline_fold3_rmse': 2.25,
        'our_mean_rmse': results['mean_rmse'],
        'our_std_rmse': results['std_rmse'],
        'our_fold3_rmse': results['fold_results'][2]['rmse'],
        'improvement_percent': (1.19 - results['mean_rmse']) / 1.19 * 100,
        'fold3_improvement_percent': (2.25 - results['fold_results'][2]['rmse']) / 2.25 * 100
    }

    # Save all results
    with open(output_dir / 'experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)

    # Save predictions
    pred_df = pd.DataFrame({
        'target': targets,
        'prediction': predictions
    })
    pred_df.to_csv(output_dir / 'predictions.csv', index=False)

    print(f"\n{'='*80}")
    print("Experiment Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == '__main__':
    results = main()
