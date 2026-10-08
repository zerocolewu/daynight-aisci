#!/usr/bin/env python3
"""
Feature-Decomposed Multi-Task Learning (FD-MTL) for Tg/Tm/Egc Prediction

CRITICAL: Uses GIN architecture with 148-dim features (validated from Day 2-5)
- Single-task models should match baselines: Tg ~32, Tm ~59, Egc ~0.53
- Multi-task model with feature decomposition for transfer learning

Architecture:
1. GNN Encoder: GIN with mean aggregation (from Day 2-5)
2. Feature Decomposition: 148-dim features categorized by property correlation
3. Property-specific heads for Tg, Tm, Egc
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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# 148-DIMENSIONAL FEATURE COMPUTATION (From Day 4)
# =============================================================================

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

        # Functional group counts (12 common groups)
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
    Replace asterisks with hydrogens to create a valid monomer SMILES.
    """
    # Convert p-SMILES to regular SMILES by replacing asterisks with hydrogens
    smiles = psmiles.replace('*', '[H]')

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None, None

        # Sanitize molecule
        Chem.SanitizeMol(mol)
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


# =============================================================================
# GIN MODEL WITH FEATURES (From Day 4 - Validated Architecture)
# =============================================================================

class GINWithFeatures(nn.Module):
    """GIN model with concatenated features - Validated from Day 4."""

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


# =============================================================================
# MULTI-TASK LEARNING MODEL
# =============================================================================

class FD_MTL(nn.Module):
    """
    Feature-Decomposed Multi-Task Learning Model

    Architecture:
    1. Shared GNN encoder (GIN)
    2. Feature decomposition based on property correlations
    3. Property-specific heads
    """

    def __init__(self, num_node_features, num_features=148, hidden_dim=256, num_layers=5,
                 tg_feature_indices=None, tm_feature_indices=None, egc_feature_indices=None):
        super(FD_MTL, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_features = num_features

        # Store feature indices for each property
        self.tg_feature_indices = tg_feature_indices if tg_feature_indices is not None else list(range(num_features))
        self.tm_feature_indices = tm_feature_indices if tm_feature_indices is not None else list(range(num_features))
        self.egc_feature_indices = egc_feature_indices if egc_feature_indices is not None else list(range(num_features))

        # Node embedding layers
        self.node_embedding = nn.Linear(num_node_features, hidden_dim)

        # GIN layers (shared encoder)
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

        # Feature processing for each property
        self.feature_bn_tg = nn.BatchNorm1d(len(self.tg_feature_indices))
        self.feature_bn_tm = nn.BatchNorm1d(len(self.tm_feature_indices))
        self.feature_bn_egc = nn.BatchNorm1d(len(self.egc_feature_indices))

        self.feature_proj_tg = nn.Linear(len(self.tg_feature_indices), hidden_dim)
        self.feature_proj_tm = nn.Linear(len(self.tm_feature_indices), hidden_dim)
        self.feature_proj_egc = nn.Linear(len(self.egc_feature_indices), hidden_dim)

        # Property-specific heads
        self.tg_head = nn.Sequential(
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

        self.tm_head = nn.Sequential(
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

        self.egc_head = nn.Sequential(
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

    def forward(self, data, task='all'):
        """
        Forward pass

        Args:
            data: PyG Data object with features
            task: 'tg', 'tm', 'egc', or 'all'
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        features = data.features

        # Ensure x is float
        x = x.float()

        # Node embeddings
        x = self.node_embedding(x)

        # GIN layers (shared encoder)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)

        # Mean aggregation
        x_pool = global_mean_pool(x, batch)

        # Process features
        if features.dim() == 3:
            features = features.squeeze(1)

        outputs = {}

        if task in ['tg', 'all']:
            tg_features = features[:, self.tg_feature_indices]
            tg_features_proj = self.feature_proj_tg(self.feature_bn_tg(tg_features))
            tg_combined = torch.cat([x_pool, tg_features_proj], dim=1)
            outputs['tg'] = self.tg_head(tg_combined).squeeze(-1)

        if task in ['tm', 'all']:
            tm_features = features[:, self.tm_feature_indices]
            tm_features_proj = self.feature_proj_tm(self.feature_bn_tm(tm_features))
            tm_combined = torch.cat([x_pool, tm_features_proj], dim=1)
            outputs['tm'] = self.tm_head(tm_combined).squeeze(-1)

        if task in ['egc', 'all']:
            egc_features = features[:, self.egc_feature_indices]
            egc_features_proj = self.feature_proj_egc(self.feature_bn_egc(egc_features))
            egc_combined = torch.cat([x_pool, egc_features_proj], dim=1)
            outputs['egc'] = self.egc_head(egc_combined).squeeze(-1)

        return outputs


# =============================================================================
# DATA PROCESSING
# =============================================================================

def create_pyg_graph(mol, smiles):
    """Create PyG Data object from RDKit mol."""
    try:
        data = from_smiles(smiles)
        if data is None or data.x is None:
            return None
        return data
    except:
        return None


def load_dataset(property_name, data_path):
    """Load and process dataset for a specific property."""
    print(f"\nLoading {property_name} dataset...")

    # Load CSV
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} samples")

    # Determine column names
    if property_name == 'Tg':
        smiles_col = 'Smiles'
        value_col = 'Tg'
    elif property_name == 'Tm':
        smiles_col = 'Smiles'
        value_col = 'Tm'
    else:  # Egc
        smiles_col = 'smiles'
        value_col = 'value'

    # Process molecules
    all_data = []
    all_features = []
    all_targets = []
    failed = 0

    for idx, row in df.iterrows():
        psmiles = row[smiles_col]
        target = row[value_col]

        mol, smiles = process_psmiles(psmiles)

        if mol is None:
            failed += 1
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

    print(f"  Successfully processed {len(all_data)} molecules ({failed} failed)")

    return all_data, np.array(all_features), np.array(all_targets)


# =============================================================================
# FEATURE-PROPERTY CORRELATION ANALYSIS
# =============================================================================

def analyze_feature_correlations(tg_features, tg_targets, tm_features, tm_targets,
                                 egc_features, egc_targets, output_file='feature_correlations.txt'):
    """Analyze feature-property correlations and categorize features."""
    print("\n" + "="*80)
    print("FEATURE-PROPERTY CORRELATION ANALYSIS")
    print("="*80)

    n_features = 148
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

    correlations = []

    for i in range(n_features):
        try:
            tg_corr, tg_p = pearsonr(tg_features[:, i], tg_targets)
            if np.isnan(tg_corr):
                tg_corr = 0.0
        except:
            tg_corr = 0.0

        try:
            tm_corr, tm_p = pearsonr(tm_features[:, i], tm_targets)
            if np.isnan(tm_corr):
                tm_corr = 0.0
        except:
            tm_corr = 0.0

        try:
            egc_corr, egc_p = pearsonr(egc_features[:, i], egc_targets)
            if np.isnan(egc_corr):
                egc_corr = 0.0
        except:
            egc_corr = 0.0

        correlations.append({
            'feature_idx': i,
            'feature_name': feature_names[i] if i < len(feature_names) else f'F{i}',
            'tg_corr': tg_corr,
            'tm_corr': tm_corr,
            'egc_corr': egc_corr,
            'abs_tg': abs(tg_corr),
            'abs_tm': abs(tm_corr),
            'abs_egc': abs(egc_corr)
        })

    # Categorize features based on correlation threshold
    threshold = 0.3

    shared_features = []
    tg_specific = []
    tm_specific = []
    egc_specific = []

    for corr in correlations:
        important_tg = corr['abs_tg'] > threshold
        important_tm = corr['abs_tm'] > threshold
        important_egc = corr['abs_egc'] > threshold

        n_important = sum([important_tg, important_tm, important_egc])

        if n_important >= 2:
            shared_features.append(corr['feature_idx'])
        elif important_tg:
            tg_specific.append(corr['feature_idx'])
        elif important_tm:
            tm_specific.append(corr['feature_idx'])
        elif important_egc:
            egc_specific.append(corr['feature_idx'])

    # If no features meet threshold, use all features
    if len(shared_features) == 0 and len(tg_specific) == 0 and len(tm_specific) == 0 and len(egc_specific) == 0:
        print(f"  No features met correlation threshold {threshold}")
        print(f"  Using all features for all properties")
        shared_features = list(range(n_features))
        tg_specific = []
        tm_specific = []
        egc_specific = []

    # Write analysis
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FEATURE-PROPERTY CORRELATION ANALYSIS\n")
        f.write("="*80 + "\n\n")

        f.write(f"Correlation threshold: {threshold}\n\n")

        f.write(f"Shared features (≥2 properties): {len(shared_features)}\n")
        f.write(f"Tg-specific features: {len(tg_specific)}\n")
        f.write(f"Tm-specific features: {len(tm_specific)}\n")
        f.write(f"Egc-specific features: {len(egc_specific)}\n\n")

        f.write("\nTop correlations for each property:\n")
        f.write("-"*40 + "\n")

        # Top Tg features
        tg_sorted = sorted(correlations, key=lambda x: x['abs_tg'], reverse=True)[:10]
        f.write("\nTg top 10:\n")
        for i, c in enumerate(tg_sorted):
            f.write(f"  {i+1}. {c['feature_name']}: r={c['tg_corr']:.4f}\n")

        # Top Tm features
        tm_sorted = sorted(correlations, key=lambda x: x['abs_tm'], reverse=True)[:10]
        f.write("\nTm top 10:\n")
        for i, c in enumerate(tm_sorted):
            f.write(f"  {i+1}. {c['feature_name']}: r={c['tm_corr']:.4f}\n")

        # Top Egc features
        egc_sorted = sorted(correlations, key=lambda x: x['abs_egc'], reverse=True)[:10]
        f.write("\nEgc top 10:\n")
        for i, c in enumerate(egc_sorted):
            f.write(f"  {i+1}. {c['feature_name']}: r={c['egc_corr']:.4f}\n")

        # Feature indices for each property
        f.write("\n\nFeature indices for each property:\n")
        f.write("-"*40 + "\n")
        f.write(f"Shared: {shared_features[:20]}...\n")
        f.write(f"Tg-specific: {tg_specific}\n")
        f.write(f"Tm-specific: {tm_specific}\n")
        f.write(f"Egc-specific: {egc_specific}\n")

    print(f"\n  Shared features: {len(shared_features)}")
    print(f"  Tg-specific: {len(tg_specific)}")
    print(f"  Tm-specific: {len(tm_specific)}")
    print(f"  Egc-specific: {len(egc_specific)}")

    # Check for expected correlations from Day 4-5
    rigidity_idx = 46  # NumRotatableBonds
    conjugation_idx = 96  # aromatic_bond_frac (approximate)

    print("\n  Validated correlations (Day 4-5):")
    if rigidity_idx < len(correlations):
        print(f"    Rigidity (rotatable bonds) vs Tg: r={correlations[rigidity_idx]['tg_corr']:.4f}")
    if conjugation_idx < len(correlations):
        print(f"    Conjugation (aromatic bonds) vs Egc: r={correlations[conjugation_idx]['egc_corr']:.4f}")

    return {
        'shared': shared_features,
        'tg_specific': tg_specific,
        'tm_specific': tm_specific,
        'egc_specific': egc_specific,
        'correlations': correlations
    }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_single_task(model, train_loader, val_loader, optimizer, criterion, device,
                      patience=50, max_epochs=500, verbose=True):
    """Train single-task model with early stopping."""
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
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1}")
                break

        if verbose and (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val RMSE={val_rmse:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_loss


def train_mtl(model, train_loaders, val_loaders, optimizer, criterion, device,
              patience=50, max_epochs=500, verbose=True):
    """Train multi-task model with early stopping."""
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        n_batches = 0

        # Iterate through all task loaders
        for task, loader in train_loaders.items():
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                outputs = model(batch, task=task)
                loss = criterion(outputs[task], batch.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

        train_loss /= max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = {'tg': [], 'tm': [], 'egc': []}
        val_targets = {'tg': [], 'tm': [], 'egc': []}

        with torch.no_grad():
            for task, loader in val_loaders.items():
                for batch in loader:
                    batch = batch.to(device)
                    outputs = model(batch, task=task)
                    loss = criterion(outputs[task], batch.y)
                    val_loss += loss.item()
                    val_preds[task].extend(outputs[task].cpu().numpy())
                    val_targets[task].extend(batch.y.cpu().numpy())

        val_loss /= max(sum(len(list(l)) for l in val_loaders.values()), 1)

        # Compute average RMSE across all tasks
        task_rmses = []
        for task in val_preds:
            if len(val_preds[task]) > 0:
                rmse = np.sqrt(mean_squared_error(val_targets[task], val_preds[task]))
                task_rmses.append(rmse)

        avg_val_rmse = np.mean(task_rmses) if task_rmses else float('inf')

        # Early stopping
        if avg_val_rmse < best_val_loss:
            best_val_loss = avg_val_rmse
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1}")
                break

        if verbose and (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}: Train Loss={train_loss:.4f}, Avg Val RMSE={avg_val_rmse:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_loss


def evaluate_model(model, test_loader, device, task='single'):
    """Evaluate model on test set."""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            if task == 'single':
                output = model(batch)
            else:
                output = model(batch, task=task)[task]
            predictions.extend(output.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    return rmse, mae, r2, predictions, targets


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_ablation_studies(tg_data, tm_data, egc_data, feature_info, output_dir):
    """Run all ablation studies."""
    print("\n" + "="*80)
    print("ABLATION STUDIES")
    print("="*80)

    results = {}

    # Get number of node features
    num_node_features = tg_data[0].x.shape[1]

    # =========================================================================
    # Ablation 1: Single-task models (baseline)
    # =========================================================================
    print("\n" + "-"*80)
    print("Ablation 1: Single-task Models (Baseline)")
    print("-"*80)

    # Normalize targets for each property
    tg_targets = np.array([d.y.item() for d in tg_data])
    tm_targets = np.array([d.y.item() for d in tm_data])
    egc_targets = np.array([d.y.item() for d in egc_data])

    tg_scaler = StandardScaler()
    tm_scaler = StandardScaler()
    egc_scaler = StandardScaler()

    tg_targets_scaled = tg_scaler.fit_transform(tg_targets.reshape(-1, 1)).flatten()
    tm_targets_scaled = tm_scaler.fit_transform(tm_targets.reshape(-1, 1)).flatten()
    egc_targets_scaled = egc_scaler.fit_transform(egc_targets.reshape(-1, 1)).flatten()

    # Update data objects
    for i, d in enumerate(tg_data):
        d.y = torch.tensor([tg_targets_scaled[i]], dtype=torch.float)
    for i, d in enumerate(tm_data):
        d.y = torch.tensor([tm_targets_scaled[i]], dtype=torch.float)
    for i, d in enumerate(egc_data):
        d.y = torch.tensor([egc_targets_scaled[i]], dtype=torch.float)

    # Tg single-task
    print("\n  Training Tg single-task model...")
    tg_single_results = run_single_task_cv(
        tg_data, num_node_features, 'Tg', tg_scaler, n_folds=5
    )
    results['single_tg'] = tg_single_results

    # Tm single-task
    print("\n  Training Tm single-task model...")
    tm_single_results = run_single_task_cv(
        tm_data, num_node_features, 'Tm', tm_scaler, n_folds=5
    )
    results['single_tm'] = tm_single_results

    # Egc single-task
    print("\n  Training Egc single-task model...")
    egc_single_results = run_single_task_cv(
        egc_data, num_node_features, 'Egc', egc_scaler, n_folds=5
    )
    results['single_egc'] = egc_single_results

    # =========================================================================
    # Ablation 2: Multi-task with ALL features (naive)
    # =========================================================================
    print("\n" + "-"*80)
    print("Ablation 2: Multi-task with ALL Features (Naive)")
    print("-"*80)

    all_features = list(range(148))
    mtl_all_results = run_mtl_cv(
        tg_data, tm_data, egc_data,
        num_node_features, all_features, all_features, all_features,
        tg_scaler, tm_scaler, egc_scaler,
        n_folds=5, name="MTL-All"
    )
    results['mtl_all'] = mtl_all_results

    # =========================================================================
    # Ablation 3: Multi-task with feature decomposition (full model)
    # =========================================================================
    print("\n" + "-"*80)
    print("Ablation 3: Multi-task with Feature Decomposition (Full Model)")
    print("-"*80)

    # Use feature decomposition from analysis
    shared_features = feature_info['shared']
    tg_features = feature_info['shared'] + feature_info['tg_specific']
    tm_features = feature_info['shared'] + feature_info['tm_specific']
    egc_features = feature_info['shared'] + feature_info['egc_specific']

    mtl_decomp_results = run_mtl_cv(
        tg_data, tm_data, egc_data,
        num_node_features, tg_features, tm_features, egc_features,
        tg_scaler, tm_scaler, egc_scaler,
        n_folds=5, name="MTL-Decomp"
    )
    results['mtl_decomp'] = mtl_decomp_results

    return results


def run_single_task_cv(data, num_node_features, property_name, target_scaler, n_folds=5):
    """Run 5-fold cross-validation for single-task model."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    all_preds = []
    all_targets = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        print(f"\n    Fold {fold + 1}/{n_folds}")

        # Split data
        train_data = [data[i] for i in train_idx]
        test_data = [data[i] for i in test_idx]

        # Further split train into train/val
        val_size = int(len(train_data) * 0.111)
        val_data = train_data[-val_size:]
        train_data = train_data[:-val_size]

        # Normalize features
        train_features = np.array([d.features.numpy().squeeze(0) for d in train_data])
        feature_scaler = StandardScaler()
        feature_scaler.fit(train_features)

        for data_list in [train_data, val_data, test_data]:
            for d in data_list:
                normalized = feature_scaler.transform(d.features.numpy().squeeze(0).reshape(1, -1))
                d.features = torch.tensor(normalized, dtype=torch.float)

        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        # Initialize model
        model = GINWithFeatures(num_node_features=num_node_features, num_features=148).to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = nn.MSELoss()

        # Train
        model, best_val_loss = train_single_task(
            model, train_loader, val_loader, optimizer, criterion, device,
            patience=50, max_epochs=500, verbose=False
        )

        # Evaluate
        rmse_scaled, mae_scaled, r2, preds, targets = evaluate_model(model, test_loader, device)

        # Convert back to original scale
        preds_original = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        targets_original = target_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

        rmse = np.sqrt(mean_squared_error(targets_original, preds_original))
        mae = mean_absolute_error(targets_original, preds_original)

        print(f"      Test RMSE: {rmse:.4f}")

        fold_results.append({
            'fold': fold + 1,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })

        all_preds.extend(preds_original)
        all_targets.extend(targets_original)

    # Aggregate results
    rmses = [r['rmse'] for r in fold_results]
    maes = [r['mae'] for r in fold_results]

    return {
        'fold_results': fold_results,
        'mean_rmse': float(np.mean(rmses)),
        'std_rmse': float(np.std(rmses)),
        'mean_mae': float(np.mean(maes)),
        'std_mae': float(np.std(maes)),
        'all_preds': all_preds,
        'all_targets': all_targets
    }


def run_mtl_cv(tg_data, tm_data, egc_data, num_node_features,
               tg_feature_indices, tm_feature_indices, egc_feature_indices,
               tg_scaler, tm_scaler, egc_scaler, n_folds=5, name="MTL"):
    """Run 5-fold cross-validation for multi-task model."""
    # Use same splits across all properties
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = {'tg': [], 'tm': [], 'egc': []}

    for fold in range(n_folds):
        print(f"\n    Fold {fold + 1}/{n_folds}")

        # Get indices for this fold
        tg_indices = list(range(len(tg_data)))
        tm_indices = list(range(len(tm_data)))
        egc_indices = list(range(len(egc_data)))

        np.random.seed(42)
        np.random.shuffle(tg_indices)
        np.random.shuffle(tm_indices)
        np.random.shuffle(egc_indices)

        # Split indices
        tg_train_size = int(len(tg_indices) * 0.8)
        tm_train_size = int(len(tm_indices) * 0.8)
        egc_train_size = int(len(egc_indices) * 0.8)

        tg_train_idx = tg_indices[:tg_train_size]
        tg_test_idx = tg_indices[tg_train_size:]
        tm_train_idx = tm_indices[:tm_train_size]
        tm_test_idx = tm_indices[tm_train_size:]
        egc_train_idx = egc_indices[:egc_train_size]
        egc_test_idx = egc_indices[egc_train_size:]

        # Split data
        tg_train = [tg_data[i] for i in tg_train_idx]
        tg_test = [tg_data[i] for i in tg_test_idx]
        tm_train = [tm_data[i] for i in tm_train_idx]
        tm_test = [tm_data[i] for i in tm_test_idx]
        egc_train = [egc_data[i] for i in egc_train_idx]
        egc_test = [egc_data[i] for i in egc_test_idx]

        # Further split for validation
        val_frac = 0.111
        tg_val = tg_train[-int(len(tg_train) * val_frac):]
        tg_train = tg_train[:-int(len(tg_train) * val_frac)]
        tm_val = tm_train[-int(len(tm_train) * val_frac):]
        tm_train = tm_train[:-int(len(tm_train) * val_frac)]
        egc_val = egc_train[-int(len(egc_train) * val_frac):]
        egc_train = egc_train[:-int(len(egc_train) * val_frac)]

        # Normalize features (use all training data)
        all_train_data = tg_train + tm_train + egc_train
        train_features = np.array([d.features.numpy().squeeze(0) for d in all_train_data])
        feature_scaler = StandardScaler()
        feature_scaler.fit(train_features)

        for data_list in [tg_train, tg_val, tg_test, tm_train, tm_val, tm_test, egc_train, egc_val, egc_test]:
            for d in data_list:
                normalized = feature_scaler.transform(d.features.numpy().squeeze(0).reshape(1, -1))
                d.features = torch.tensor(normalized, dtype=torch.float)

        # Create data loaders
        tg_train_loader = DataLoader(tg_train, batch_size=64, shuffle=True)
        tg_val_loader = DataLoader(tg_val, batch_size=64, shuffle=False)
        tg_test_loader = DataLoader(tg_test, batch_size=64, shuffle=False)

        tm_train_loader = DataLoader(tm_train, batch_size=64, shuffle=True)
        tm_val_loader = DataLoader(tm_val, batch_size=64, shuffle=False)
        tm_test_loader = DataLoader(tm_test, batch_size=64, shuffle=False)

        egc_train_loader = DataLoader(egc_train, batch_size=64, shuffle=True)
        egc_val_loader = DataLoader(egc_val, batch_size=64, shuffle=False)
        egc_test_loader = DataLoader(egc_test, batch_size=64, shuffle=False)

        train_loaders = {'tg': tg_train_loader, 'tm': tm_train_loader, 'egc': egc_train_loader}
        val_loaders = {'tg': tg_val_loader, 'tm': tm_val_loader, 'egc': egc_val_loader}

        # Initialize model
        model = FD_MTL(
            num_node_features=num_node_features,
            num_features=148,
            hidden_dim=256,
            num_layers=5,
            tg_feature_indices=tg_feature_indices,
            tm_feature_indices=tm_feature_indices,
            egc_feature_indices=egc_feature_indices
        ).to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = nn.MSELoss()

        # Train
        model, best_val_loss = train_mtl(
            model, train_loaders, val_loaders, optimizer, criterion, device,
            patience=50, max_epochs=500, verbose=False
        )

        # Evaluate each task
        for task, test_loader, scaler in [
            ('tg', tg_test_loader, tg_scaler),
            ('tm', tm_test_loader, tm_scaler),
            ('egc', egc_test_loader, egc_scaler)
        ]:
            rmse_scaled, mae_scaled, r2, preds, targets = evaluate_model(model, test_loader, device, task)

            # Convert back to original scale
            preds_original = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            targets_original = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

            rmse = np.sqrt(mean_squared_error(targets_original, preds_original))
            mae = mean_absolute_error(targets_original, preds_original)

            fold_results[task].append({
                'fold': fold + 1,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            })

        print(f"      Tg RMSE: {fold_results['tg'][-1]['rmse']:.4f}, "
              f"Tm RMSE: {fold_results['tm'][-1]['rmse']:.4f}, "
              f"Egc RMSE: {fold_results['egc'][-1]['rmse']:.4f}")

    # Aggregate results
    results = {}
    for task in ['tg', 'tm', 'egc']:
        rmses = [r['rmse'] for r in fold_results[task]]
        results[task] = {
            'fold_results': fold_results[task],
            'mean_rmse': float(np.mean(rmses)),
            'std_rmse': float(np.std(rmses))
        }

    return results


def generate_report(results, output_dir):
    """Generate final report."""
    report_file = output_dir / 'results.txt'

    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FEATURE-DECOMPOSED MULTI-TASK LEARNING RESULTS\n")
        f.write("="*80 + "\n\n")

        # Ablation 1: Single-task
        f.write("="*80 + "\n")
        f.write("Ablation 1: Single-Task Models (Baseline)\n")
        f.write("="*80 + "\n\n")

        f.write("Tg:\n")
        f.write(f"  RMSE: {results['single_tg']['mean_rmse']:.4f} +/- {results['single_tg']['std_rmse']:.4f}\n")
        tg_folds = ', '.join([f"{r['rmse']:.4f}" for r in results['single_tg']['fold_results']])
        f.write(f"  Fold results: {tg_folds}\n\n")

        f.write("Tm:\n")
        f.write(f"  RMSE: {results['single_tm']['mean_rmse']:.4f} +/- {results['single_tm']['std_rmse']:.4f}\n")
        tm_folds = ', '.join([f"{r['rmse']:.4f}" for r in results['single_tm']['fold_results']])
        f.write(f"  Fold results: {tm_folds}\n\n")

        f.write("Egc:\n")
        f.write(f"  RMSE: {results['single_egc']['mean_rmse']:.4f} +/- {results['single_egc']['std_rmse']:.4f}\n")
        egc_folds = ', '.join([f"{r['rmse']:.4f}" for r in results['single_egc']['fold_results']])
        f.write(f"  Fold results: {egc_folds}\n\n")

        # Ablation 2: MTL with all features
        f.write("="*80 + "\n")
        f.write("Ablation 2: MTL with ALL Features (Naive)\n")
        f.write("="*80 + "\n\n")

        f.write("Tg:\n")
        f.write(f"  RMSE: {results['mtl_all']['tg']['mean_rmse']:.4f} +/- {results['mtl_all']['tg']['std_rmse']:.4f}\n\n")

        f.write("Tm:\n")
        f.write(f"  RMSE: {results['mtl_all']['tm']['mean_rmse']:.4f} +/- {results['mtl_all']['tm']['std_rmse']:.4f}\n\n")

        f.write("Egc:\n")
        f.write(f"  RMSE: {results['mtl_all']['egc']['mean_rmse']:.4f} +/- {results['mtl_all']['egc']['std_rmse']:.4f}\n\n")

        # Ablation 3: MTL with decomposition
        f.write("="*80 + "\n")
        f.write("Ablation 3: MTL with Feature Decomposition (Full Model)\n")
        f.write("="*80 + "\n\n")

        f.write("Tg:\n")
        f.write(f"  RMSE: {results['mtl_decomp']['tg']['mean_rmse']:.4f} +/- {results['mtl_decomp']['tg']['std_rmse']:.4f}\n\n")

        f.write("Tm:\n")
        f.write(f"  RMSE: {results['mtl_decomp']['tm']['mean_rmse']:.4f} +/- {results['mtl_decomp']['tm']['std_rmse']:.4f}\n\n")

        f.write("Egc:\n")
        f.write(f"  RMSE: {results['mtl_decomp']['egc']['mean_rmse']:.4f} +/- {results['mtl_decomp']['egc']['std_rmse']:.4f}\n\n")

        # Summary table
        f.write("="*80 + "\n")
        f.write("SUMMARY TABLE\n")
        f.write("="*80 + "\n\n")

        f.write(f"{'Ablation':<45} {'Tg RMSE':<15} {'Tm RMSE':<15} {'Egc RMSE':<15}\n")
        f.write("-"*80 + "\n")

        f.write(f"{'Single-Task (Baseline)':<45} "
                f"{results['single_tg']['mean_rmse']:.2f}+/-{results['single_tg']['std_rmse']:.2f}  "
                f"{results['single_tm']['mean_rmse']:.2f}+/-{results['single_tm']['std_rmse']:.2f}  "
                f"{results['single_egc']['mean_rmse']:.4f}+/-{results['single_egc']['std_rmse']:.4f}\n")

        f.write(f"{'MTL with ALL Features (Naive)':<45} "
                f"{results['mtl_all']['tg']['mean_rmse']:.2f}+/-{results['mtl_all']['tg']['std_rmse']:.2f}  "
                f"{results['mtl_all']['tm']['mean_rmse']:.2f}+/-{results['mtl_all']['tm']['std_rmse']:.2f}  "
                f"{results['mtl_all']['egc']['mean_rmse']:.4f}+/-{results['mtl_all']['egc']['std_rmse']:.4f}\n")

        f.write(f"{'MTL with Feature Decomposition':<45} "
                f"{results['mtl_decomp']['tg']['mean_rmse']:.2f}+/-{results['mtl_decomp']['tg']['std_rmse']:.2f}  "
                f"{results['mtl_decomp']['tm']['mean_rmse']:.2f}+/-{results['mtl_decomp']['tm']['std_rmse']:.2f}  "
                f"{results['mtl_decomp']['egc']['mean_rmse']:.4f}+/-{results['mtl_decomp']['egc']['std_rmse']:.4f}\n")

        # Baseline comparison
        f.write("\n" + "="*80 + "\n")
        f.write("BASELINE COMPARISON\n")
        f.write("="*80 + "\n\n")

        f.write("Baseline from literature:\n")
        f.write("  Tg:  32.31\n")
        f.write("  Tm:  59.18 +/- 1.12\n")
        f.write("  Egc: 0.5262 +/- 0.0471\n\n")

        best_tg = min(results['single_tg']['mean_rmse'], results['mtl_all']['tg']['mean_rmse'], results['mtl_decomp']['tg']['mean_rmse'])
        best_tm = min(results['single_tm']['mean_rmse'], results['mtl_all']['tm']['mean_rmse'], results['mtl_decomp']['tm']['mean_rmse'])
        best_egc = min(results['single_egc']['mean_rmse'], results['mtl_all']['egc']['mean_rmse'], results['mtl_decomp']['egc']['mean_rmse'])

        f.write(f"Our best model:\n")
        f.write(f"  Tg:  {best_tg:.4f}\n")
        f.write(f"  Tm:  {best_tm:.4f}\n")
        f.write(f"  Egc: {best_egc:.4f}\n\n")

        f.write(f"Performance change:\n")
        f.write(f"  Tg:  {((best_tg - 32.31) / 32.31 * 100):+.2f}%\n")
        f.write(f"  Tm:  {((best_tm - 59.18) / 59.18 * 100):+.2f}%\n")
        f.write(f"  Egc: {((best_egc - 0.5262) / 0.5262 * 100):+.2f}%\n")

    print(f"\nReport saved to {report_file}")


def main():
    """Main function."""
    print("\n" + "="*80)
    print("FEATURE-DECOMPOSED MULTI-TASK LEARNING (FD-MTL)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Setup
    base_dir = Path('/home/home_public/aisci/AIscientist/src/5_day5_fd_mtl')
    data_dir = Path('/home/home_public/aisci/AIscientist/src/alldataset')
    output_dir = base_dir

    # Load datasets
    tg_data, tg_features, tg_targets = load_dataset('Tg', data_dir / 'Tg.csv')
    tm_data, tm_features, tm_targets = load_dataset('Tm', data_dir / 'Tm.csv')
    egc_data, egc_features, egc_targets = load_dataset('Egc', data_dir / 'Egc.csv')

    # Analyze feature correlations
    feature_info = analyze_feature_correlations(
        tg_features, tg_targets,
        tm_features, tm_targets,
        egc_features, egc_targets,
        output_file=str(output_dir / 'feature_correlations.txt')
    )

    # Run ablation studies
    results = run_ablation_studies(
        tg_data, tm_data, egc_data,
        feature_info, output_dir
    )

    # Generate report
    generate_report(results, output_dir)

    # Save results as JSON
    results_json = {
        'single_tg': {
            'mean_rmse': results['single_tg']['mean_rmse'],
            'std_rmse': results['single_tg']['std_rmse'],
            'fold_rmses': [r['rmse'] for r in results['single_tg']['fold_results']]
        },
        'single_tm': {
            'mean_rmse': results['single_tm']['mean_rmse'],
            'std_rmse': results['single_tm']['std_rmse'],
            'fold_rmses': [r['rmse'] for r in results['single_tm']['fold_results']]
        },
        'single_egc': {
            'mean_rmse': results['single_egc']['mean_rmse'],
            'std_rmse': results['single_egc']['std_rmse'],
            'fold_rmses': [r['rmse'] for r in results['single_egc']['fold_results']]
        },
        'mtl_all': {
            'tg': results['mtl_all']['tg'],
            'tm': results['mtl_all']['tm'],
            'egc': results['mtl_all']['egc']
        },
        'mtl_decomp': {
            'tg': results['mtl_decomp']['tg'],
            'tm': results['mtl_decomp']['tm'],
            'egc': results['mtl_decomp']['egc']
        }
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    print(f"\nSingle-Task Baseline:")
    print(f"  Tg:  {results['single_tg']['mean_rmse']:.4f} +/- {results['single_tg']['std_rmse']:.4f}")
    print(f"  Tm:  {results['single_tm']['mean_rmse']:.4f} +/- {results['single_tm']['std_rmse']:.4f}")
    print(f"  Egc: {results['single_egc']['mean_rmse']:.4f} +/- {results['single_egc']['std_rmse']:.4f}")

    print(f"\nMTL with ALL Features:")
    print(f"  Tg:  {results['mtl_all']['tg']['mean_rmse']:.4f} +/- {results['mtl_all']['tg']['std_rmse']:.4f}")
    print(f"  Tm:  {results['mtl_all']['tm']['mean_rmse']:.4f} +/- {results['mtl_all']['tm']['std_rmse']:.4f}")
    print(f"  Egc: {results['mtl_all']['egc']['mean_rmse']:.4f} +/- {results['mtl_all']['egc']['std_rmse']:.4f}")

    print(f"\nMTL with Feature Decomposition:")
    print(f"  Tg:  {results['mtl_decomp']['tg']['mean_rmse']:.4f} +/- {results['mtl_decomp']['tg']['std_rmse']:.4f}")
    print(f"  Tm:  {results['mtl_decomp']['tm']['mean_rmse']:.4f} +/- {results['mtl_decomp']['tm']['std_rmse']:.4f}")
    print(f"  Egc: {results['mtl_decomp']['egc']['mean_rmse']:.4f} +/- {results['mtl_decomp']['egc']['std_rmse']:.4f}")

    print(f"\nTarget Assessment:")
    print(f"  Tg:  Target <31.5, Achieved {results['single_tg']['mean_rmse']:.4f} - {'PASS' if results['single_tg']['mean_rmse'] < 31.5 else 'FAIL'}")
    print(f"  Tm:  Target <55.0, Achieved {results['single_tm']['mean_rmse']:.4f} - {'PASS' if results['single_tm']['mean_rmse'] < 55.0 else 'FAIL'}")
    print(f"  Egc: Target <0.50, Achieved {results['single_egc']['mean_rmse']:.4f} - {'PASS' if results['single_egc']['mean_rmse'] < 0.50 else 'FAIL'}")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == '__main__':
    main()
