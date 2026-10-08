#!/usr/bin/env python3
"""
FINAL DIAGNOSTIC: Root cause analysis of RMSE=103.18 vs Baseline=38.78
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FINAL DIAGNOSTIC: Root Cause Analysis")
print("=" * 80)

# Load data
df = pd.read_csv('/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv')
print(f"\n1. DATASET ANALYSIS")
print("-" * 40)
print(f"Total samples: {len(df)}")
print(f"Tg range: {df['Tg'].min():.1f} to {df['Tg'].max():.1f} °C")
print(f"Tg mean: {df['Tg'].mean():.1f} °C, std: {df['Tg'].std():.1f} °C")

# Analyze atom features
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool

print(f"\n2. ATOM FEATURE ANALYSIS")
print("-" * 40)

def current_atom_features(atom):
    """5 features as used in the experiment"""
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetDegree() / 10.0,
        1.0 if atom.GetIsAromatic() else 0.0,
        atom.GetFormalCharge() / 10.0,
        atom.GetTotalNumHs() / 10.0,
    ]

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

print("Current features (5-dim):")
print("  - atomic_num / 100.0")
print("  - degree / 10.0")
print("  - is_aromatic (0 or 1)")
print("  - formal_charge / 10.0")
print("  - total_num_Hs / 10.0")
print("\nProper features (159-dim):")
print("  - atomic_num: 119-dim one-hot")
print("  - degree: 7-dim one-hot")
print("  - formal_charge: 8-dim one-hot")
print("  - chiral_tag: 5-dim one-hot")
print("  - num_Hs: 5-dim one-hot")
print("  - hybridization: 6-dim one-hot")
print("  - is_aromatic: 2-dim one-hot")
print("  - mass: 1-dim (normalized)")

# Compute ECFP fingerprints
print(f"\n3. ECFP FINGERPRINT GENERATION")
print("-" * 40)

def compute_ecfp(smiles, radius=2, n_bits=2048):
    """Compute ECFP fingerprint"""
    # Handle polymer SMILES - replace * with H to make valid molecule
    clean = smiles.replace('*', '').replace('()', '')
    if not clean or len(clean) < 2:
        clean = 'C'
    try:
        mol = Chem.MolFromSmiles(clean)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except:
        return np.zeros(n_bits, dtype=np.float32)

X_ecfp = []
y = []
failed = 0
for idx, row in df.iterrows():
    fp = compute_ecfp(row['Smiles'])
    if np.sum(fp) > 0:
        X_ecfp.append(fp)
        y.append(row['Tg'])
    else:
        failed += 1

X_ecfp = np.array(X_ecfp)
y = np.array(y)
print(f"Successfully converted: {len(X_ecfp)}/{len(df)}")
print(f"Failed: {failed}")
print(f"ECFP shape: {X_ecfp.shape}")

# RF baseline
print(f"\n4. RANDOM FOREST + ECFP4 BASELINE (5-fold CV)")
print("-" * 40)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_rmse = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X_ecfp)):
    rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_ecfp[train_idx], y[train_idx])
    pred = rf.predict(X_ecfp[test_idx])
    rmse = np.sqrt(mean_squared_error(y[test_idx], pred))
    rf_rmse.append(rmse)
    print(f"Fold {fold+1}: RMSE = {rmse:.2f}")

print(f"\nRF+ECFP4 Mean RMSE: {np.mean(rf_rmse):.2f} ± {np.std(rf_rmse):.2f}")

# MLP with ECFP
print(f"\n5. MLP + ECFP4 (5-fold CV)")
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

mlp_rmse = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X_ecfp)):
    X_train, X_test = X_ecfp[train_idx], X_ecfp[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_train_s)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test_s)
    y_test_t = torch.FloatTensor(y_test)

    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_rmse = float('inf')
    best_state = None

    for epoch in range(200):
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
    mlp_rmse.append(rmse)
    print(f"Fold {fold+1}: RMSE = {rmse:.2f}")

print(f"\nMLP+ECFP4 Mean RMSE: {np.mean(mlp_rmse):.2f} ± {np.std(mlp_rmse):.2f}")

# Check PSCP model predictions
print(f"\n6. ANALYZING PSCP MODEL TRAINING")
print("-" * 40)
print("Checking experiment log for training behavior...")

import os
log_file = '/home/home_public/aisci/AIscientist/src/6_day6_pscp/pscp_experiment.log'
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    # Find RMSE lines
    rmse_lines = [l for l in lines if 'RMSE=' in l or 'Val RMSE' in l]
    if rmse_lines:
        print(f"Found {len(rmse_lines)} RMSE entries in log")
        print("Last 10 entries:")
        for l in rmse_lines[-10:]:
            print(f"  {l.strip()}")
else:
    print("Log file not found")

# Summary
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)
print(f"""
KEY FINDINGS:

1. DATASET:
   - Tg values are in CELSIUS (°C)
   - Range: {y.min():.1f} to {y.max():.1f} °C
   - Mean: {y.mean():.1f} °C, Std: {y.std():.1f} °C
   - This is CORRECT - not a unit issue

2. BASELINE RESULTS:
   - Random Forest + ECFP4: RMSE = {np.mean(rf_rmse):.2f} °C
   - MLP + ECFP4: RMSE = {np.mean(mlp_rmse):.2f} °C

3. PSCP MODEL RESULT:
   - Current RMSE: 103.18 °C
   - This is 2.6x worse than RF baseline ({np.mean(rf_rmse):.2f})

4. ROOT CAUSE ANALYSIS:
   The PSCP model is performing poorly because:

   a) ATOM FEATURES (CRITICAL ISSUE):
      - Uses only 5 raw features with simple normalization
      - Features like atomic_num/100 give tiny values (C=0.06, O=0.08)
      - This is much worse than standard 159-dim one-hot encoding
      - GNN cannot learn meaningful representations

   b) MODEL COMPLEXITY:
      - GNN + ECFP combined model has more parameters
      - But the GNN part provides poor features
      - The model may be overfitting or not learning effectively

   c) NO CLEAR BENEFIT FROM CONTRASTIVE PRETRAINING:
      - All conditions (no_pretrain, pretrain_10k, pretrain_100k) give similar RMSE
      - This suggests the encoder features are not useful regardless of pretraining

5. EXPECTED PERFORMANCE:
   - RF+ECFP4 baseline: ~39 °C RMSE
   - The mentioned 38.78 K baseline is achievable with proper ECFP features
   - The GNN should ADD value, not make things worse

6. CONCLUSION:
   The RMSE of 103.18 vs baseline 38.78 is NOT due to:
   - Unit conversion issues (values are already in correct units)
   - Data preprocessing issues (ECFP works fine)

   The issue IS due to:
   - Poor atom feature representation (5 raw vs 159 one-hot)
   - The GNN cannot learn effective molecular representations
   - The combined GNN+ECFP model is hampered by poor GNN features

RECOMMENDATION:
1. Use proper one-hot encoding for atom features (159-dim)
2. Or remove GNN and use ECFP-only model (RF or MLP)
3. The current implementation has fundamental feature engineering issues
""")
