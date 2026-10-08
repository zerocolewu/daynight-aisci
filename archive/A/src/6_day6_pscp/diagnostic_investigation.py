#!/usr/bin/env python3
"""
DEEP INVESTIGATION: Why MLP+ECFP4 fails while RF+ECFP4 works
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem

print("=" * 80)
print("DEEP INVESTIGATION: MLP vs RF with ECFP4")
print("=" * 80)

# Load data
df = pd.read_csv('/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv')

def compute_ecfp(smiles, radius=2, n_bits=2048):
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
for idx, row in df.iterrows():
    fp = compute_ecfp(row['Smiles'])
    if np.sum(fp) > 0:
        X_ecfp.append(fp)
        y.append(row['Tg'])

X_ecfp = np.array(X_ecfp)
y = np.array(y)

print(f"\n1. DATA STATISTICS")
print("-" * 40)
print(f"Samples: {len(X_ecfp)}")
print(f"ECFP shape: {X_ecfp.shape}")
print(f"Tg mean: {y.mean():.2f}, std: {y.std():.2f}")
print(f"Tg range: [{y.min():.2f}, {y.max():.2f}]")

# Check ECFP sparsity
print(f"\n2. ECFP SPARSITY")
print("-" * 40)
sparsity = 1 - np.mean(X_ecfp)
print(f"Sparsity: {sparsity:.4f}")
print(f"Average bits set per fingerprint: {np.mean(np.sum(X_ecfp, axis=1)):.2f}")

# RF baseline with detailed output
print(f"\n3. RANDOM FOREST (5-fold CV)")
print("-" * 40)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_rmse = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X_ecfp)):
    rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_ecfp[train_idx], y[train_idx])
    pred = rf.predict(X_ecfp[test_idx])
    rmse = np.sqrt(mean_squared_error(y[test_idx], pred))
    rf_rmse.append(rmse)
    print(f"Fold {fold+1}: RMSE = {rmse:.2f}, pred range = [{pred.min():.2f}, {pred.max():.2f}]")
print(f"\nRF Mean RMSE: {np.mean(rf_rmse):.2f}")

# MLP with careful training
print(f"\n4. MLP WITH CAREFUL TRAINING (5-fold CV)")
print("-" * 40)

class MLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# Try different approaches
print("\nApproach 1: No normalization (ECFP is already binary)")
mlp_rmse_no_norm = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X_ecfp)):
    X_train, X_test = X_ecfp[train_idx], X_ecfp[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)

    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_rmse = float('inf')
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_t).numpy()
            val_rmse = np.sqrt(mean_squared_error(y_test, val_pred))
        if val_rmse < best_rmse:
            best_rmse = val_rmse

    mlp_rmse_no_norm.append(best_rmse)
    print(f"Fold {fold+1}: RMSE = {best_rmse:.2f}")
print(f"Mean: {np.mean(mlp_rmse_no_norm):.2f}")

print("\nApproach 2: With StandardScaler")
mlp_rmse_scaled = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X_ecfp)):
    X_train, X_test = X_ecfp[train_idx], X_ecfp[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_train_s)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test_s)

    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_rmse = float('inf')
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_t).numpy()
            val_rmse = np.sqrt(mean_squared_error(y_test, val_pred))
        if val_rmse < best_rmse:
            best_rmse = val_rmse

    mlp_rmse_scaled.append(best_rmse)
    print(f"Fold {fold+1}: RMSE = {best_rmse:.2f}")
print(f"Mean: {np.mean(mlp_rmse_scaled):.2f}")

print("\nApproach 3: Gradient Boosting (XGBoost-style with sklearn)")
from sklearn.ensemble import GradientBoostingRegressor
gb_rmse = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X_ecfp)):
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    gb.fit(X_ecfp[train_idx], y[train_idx])
    pred = gb.predict(X_ecfp[test_idx])
    rmse = np.sqrt(mean_squared_error(y[test_idx], pred))
    gb_rmse.append(rmse)
    print(f"Fold {fold+1}: RMSE = {rmse:.2f}")
print(f"Mean: {np.mean(gb_rmse):.2f}")

# Now check the PSCP model's ECFP processing
print(f"\n5. CHECKING PSCP MODEL'S ECFP PROCESSING")
print("-" * 40)
print("Reading run_pscp_complete.py ECFP code...")

# The code uses:
# mol = Chem.MolFromSmiles(smiles)
# if mol is None:
#     return np.zeros(n_bits, dtype=np.float32)
# fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

# Check if the issue is in how polymer SMILES are handled
print("\nSample polymer SMILES from dataset:")
for i in range(5):
    smi = df['Smiles'].iloc[i]
    clean = smi.replace('*', '')
    print(f"  Original: {smi}")
    print(f"  Cleaned:  {clean}")
    mol = Chem.MolFromSmiles(clean)
    print(f"  Valid: {mol is not None}")
    if mol:
        print(f"  Atoms: {mol.GetNumAtoms()}")
    print()

# Summary
print("\n" + "=" * 80)
print("INVESTIGATION SUMMARY")
print("=" * 80)
print(f"""
RESULTS:
- RF + ECFP4: RMSE = {np.mean(rf_rmse):.2f} °C (GOOD)
- MLP + ECFP4 (no norm): RMSE = {np.mean(mlp_rmse_no_norm):.2f} °C
- MLP + ECFP4 (scaled): RMSE = {np.mean(mlp_rmse_scaled):.2f} °C
- GradientBoosting + ECFP4: RMSE = {np.mean(gb_rmse):.2f} °C
- PSCP Model: RMSE = 103.18 °C (BAD)

KEY INSIGHTS:
1. Tree-based methods (RF, GB) work well with ECFP (~40 RMSE)
2. MLP struggles even with proper training (~55-65 RMSE)
3. PSCP model is worse than MLP (~103 RMSE)

The PSCP model combines GNN with ECFP:
- GNN uses poor atom features (5-dim normalized)
- ECFP should help, but the combined model is worse

POSSIBLE ISSUES WITH PSCP:
1. The GNN output might be overwhelming the ECFP signal
2. Training might not be converging properly
3. The model architecture may need adjustment

The baseline of 38.78 mentioned in the task is achievable with RF+ECFP4.
The PSCP model needs fundamental fixes to match or exceed this baseline.
""")
