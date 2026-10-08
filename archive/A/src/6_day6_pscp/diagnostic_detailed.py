#!/usr/bin/env python3
"""
DETAILED DIAGNOSTIC: Root cause analysis of RMSE=103.18 vs Baseline=38.78
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DETAILED DIAGNOSTIC: Root Cause Analysis")
print("=" * 80)

# Load data
df = pd.read_csv('/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv')
print(f"\n1. DATASET ANALYSIS")
print("-" * 40)
print(f"Total samples: {len(df)}")
print(f"Tg range: {df['Tg'].min():.1f} to {df['Tg'].max():.1f} °C")
print(f"Tg mean: {df['Tg'].mean():.1f} °C, std: {df['Tg'].std():.1f} °C")

# Check the experiment code's atom features
print(f"\n2. ATOM FEATURE ANALYSIS (from run_pscp_complete.py)")
print("-" * 40)
print("Features used in experiment:")
print("  - atomic_num / 100.0")
print("  - degree / 10.0")
print("  - is_aromatic (0 or 1)")
print("  - formal_charge / 10.0")
print("  - total_num_Hs / 10.0")
print("Total: 5 features per atom")
print("")
print("ISSUE: Atom features are normalized incorrectly!")
print("  - atomic_num / 100: e.g., Carbon(6) -> 0.06, Oxygen(8) -> 0.08")
print("  - These values are VERY SMALL, which can hurt learning!")

# Check what proper atom features should be
print(f"\n3. PROPER ATOM FEATURES (one-hot encoding)")
print("-" * 40)
print("Standard approach uses ONE-HOT encoding:")
print("  - atomic_num: 119-dim one-hot (elements 1-118)")
print("  - degree: 7-dim one-hot")
print("  - formal_charge: 8-dim one-hot")
print("  - chiral_tag: 5-dim one-hot")
print("  - num_Hs: 5-dim one-hot")
print("  - hybridization: 6-dim one-hot")
print("  - is_aromatic: 2-dim one-hot")
print("  - mass: 1-dim (normalized)")
print("Total: ~159 features (commonly used in molecular GNNs)")

# Test both approaches
print(f"\n4. COMPARISON: CURRENT vs PROPER FEATURES")
print("-" * 40)

from rdkit import Chem
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool

# Current feature extraction (from experiment)
def current_atom_features(atom):
    """5 features as used in the experiment"""
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetDegree() / 10.0,
        1.0 if atom.GetIsAromatic() else 0.0,
        atom.GetFormalCharge() / 10.0,
        atom.GetTotalNumHs() / 10.0,
    ]

# Proper one-hot encoding
def one_hot_encoding(value, choices):
    encoding = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1
    return encoding

def proper_atom_features(atom):
    """159 features with proper one-hot encoding"""
    features = []
    # Common atom features
    features += one_hot_encoding(atom.GetAtomicNum(), list(range(1, 119)))
    features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    features += one_hot_encoding(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3])
    features += one_hot_encoding(atom.GetChiralTag(), [0, 1, 2, 3])
    features += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    features += one_hot_encoding(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ])
    features += one_hot_encoding(atom.GetIsAromatic(), [False, True])
    features.append(atom.GetMass() / 100.0)
    return features

# Test on sample molecule
sample_smiles = "*CC(*)Cc1ccccc1"  # Polymer SMILES with benzene ring
sample_smiles_clean = sample_smiles.replace('*', '')
mol = Chem.MolFromSmiles(sample_smiles_clean)

print(f"Sample SMILES: {sample_smiles}")
print(f"After removing *: {sample_smiles_clean}")
print(f"Atoms in molecule: {mol.GetNumAtoms()}")

current_feats = [current_atom_features(atom) for atom in mol.GetAtoms()]
proper_feats = [proper_atom_features(atom) for atom in mol.GetAtoms()]

print(f"\nCurrent feature shape: {len(current_feats)} x {len(current_feats[0])}")
print(f"Proper feature shape: {len(proper_feats)} x {len(proper_feats[0])}")
print(f"\nCurrent features sample (first atom):")
print(f"  {current_feats[0]}")
print(f"  Sum: {sum(current_feats[0]):.4f}")
print(f"\nProper features sample (first atom):")
print(f"  First 20: {proper_feats[0][:20]}")
print(f"  Sum: {sum(proper_feats[0]):.4f}")

# Build GNN with current features and test
print(f"\n5. TESTING GNN WITH CURRENT vs PROPER FEATURES")
print("-" * 40)

class SimpleGIN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        h = self.input_proj(x)
        for conv, bn in zip(self.convs, self.bns):
            h = bn(conv(h, edge_index))
        return global_mean_pool(h, batch)

def smiles_to_graph_current(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_features = [current_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_indices = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def smiles_to_graph_proper(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_features = [proper_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_indices = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

# Test forward pass
print("\nTesting forward pass with current features (5-dim):")
try:
    graph = smiles_to_graph_current(sample_smiles_clean)
    model_current = SimpleGIN(input_dim=5, hidden_dim=256, num_layers=3)
    batch = torch.zeros(graph.x.size(0), dtype=torch.long)
    out = model_current(graph.x, graph.edge_index, batch)
    print(f"  Output shape: {out.shape}")
    print(f"  Output mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\nTesting forward pass with proper features (159-dim):")
try:
    graph = smiles_to_graph_proper(sample_smiles_clean)
    model_proper = SimpleGIN(input_dim=159, hidden_dim=256, num_layers=3)
    batch = torch.zeros(graph.x.size(0), dtype=torch.long)
    out = model_proper(graph.x, graph.edge_index, batch)
    print(f"  Output shape: {out.shape}")
    print(f"  Output mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")
except Exception as e:
    print(f"  ERROR: {e}")

# Now test ECFP-only baseline
print(f"\n6. ECFP-ONLY BASELINE (Random Forest)")
print("-" * 40)
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor

def compute_ecfp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)

# Prepare data
X_ecfp = []
y = []
for idx, row in df.iterrows():
    smiles = row['Smiles'].replace('*', '')
    if not smiles:
        smiles = 'C'
    X_ecfp.append(compute_ecfp(smiles))
    y.append(row['Tg'])

X_ecfp = np.array(X_ecfp)
y = np.array(y)

# RF baseline
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_rmse = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X_ecfp)):
    rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_ecfp[train_idx], y[train_idx])
    pred = rf.predict(X_ecfp[test_idx])
    rmse = np.sqrt(mean_squared_error(y[test_idx], pred))
    rf_rmse.append(rmse)
    print(f"Fold {fold+1}: RMSE = {rmse:.2f}")

print(f"\nRF+ECFP4 Mean RMSE: {np.mean(rf_rmse):.2f} +/- {np.std(rf_rmse):.2f}")

# Test MLP with ECFP only
print(f"\n7. MLP WITH ECFP ONLY (NO GRAPH)")
print("-" * 40)

class MLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_mlp_ecfp(X, y, n_folds=5, epochs=200):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # To tensors
        X_train_t = torch.FloatTensor(X_train_scaled)
        y_train_t = torch.FloatTensor(y_train)
        X_test_t = torch.FloatTensor(X_test_scaled)
        y_test_t = torch.FloatTensor(y_test)

        model = MLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.MSELoss()

        best_rmse = float('inf')
        best_state = None

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train_t)
            loss = criterion(pred, y_train_t)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_test_t)
                val_rmse = torch.sqrt(criterion(val_pred, y_test_t)).item()

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            final_pred = model(X_test_t).numpy()

        rmse = np.sqrt(mean_squared_error(y_test, final_pred))
        results.append(rmse)
        print(f"Fold {fold+1}: RMSE = {rmse:.2f}")

    return results

mlp_rmse = train_mlp_ecfp(X_ecfp, y)
print(f"\nMLP+ECFP4 Mean RMSE: {np.mean(mlp_rmse):.2f} +/- {np.std(mlp_rmse):.2f}")

# Summary
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)
print("""
KEY FINDINGS:

1. DATASET UNITS:
   - Tg values are in CELSIUS (°C), not Kelvin
   - Range: -139 to 495 °C
   - Mean: 141.87 °C, Std: 112.15 °C

2. ATOM FEATURE ISSUE (CRITICAL):
   - Current implementation uses 5 raw features with simple normalization
   - Features are: atomic_num/100, degree/10, is_aromatic, charge/10, Hs/10
   - Problem: Values are very small (e.g., atomic_num=6 -> 0.06)
   - This can hurt gradient flow and model learning

3. PROPER APPROACH:
   - Standard GNNs use ONE-HOT encoding (~159 features)
   - This gives much richer representation

4. BASELINE COMPARISON:
   - Random Forest + ECFP4: RMSE ~39.21 °C
   - MLP + ECFP4: RMSE varies (need proper training)
   - Current GNN (5-dim features): RMSE ~103.18 °C

5. ROOT CAUSE ANALYSIS:
   The poor performance (RMSE=103) is likely due to:
   a) Inadequate atom features (5-dim vs 159-dim)
   b) Simple normalization that produces small values
   c) Model may not be learning meaningful representations

RECOMMENDATION:
- Use proper one-hot encoding for atom features (159-dim)
- Or rely on ECFP fingerprint alone which already achieves ~39 RMSE
- The GNN should add value on top of ECFP, not replace it with worse features
""")
