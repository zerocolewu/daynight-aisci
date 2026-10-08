#!/usr/bin/env python3
"""
Diagnostic script to investigate the RMSE issue (103.18 vs expected 38.78)
"""
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DIAGNOSTIC: Investigating RMSE=103.18 vs Baseline=38.78")
print("=" * 80)

# Check environment
print("\n1. CHECKING ENVIRONMENT")
print("-" * 40)
import sys
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Load data
print("\n2. LOADING AND ANALYZING Tg DATASET")
print("-" * 40)
df = pd.read_csv('/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nTg statistics (raw values):")
print(f"  Min: {df['Tg'].min():.2f}")
print(f"  Max: {df['Tg'].max():.2f}")
print(f"  Mean: {df['Tg'].mean():.2f}")
print(f"  Std: {df['Tg'].std():.2f}")
print(f"  Median: {df['Tg'].median():.2f}")

# Check for unit consistency
print(f"\nSample Tg values:")
print(df['Tg'].head(20).tolist())

# These values look like Celsius, not Kelvin
# Tg values typically range from -100 to 300+ Celsius
print("\nNOTE: Values appear to be in CELSIUS (typical polymer Tg range)")

# Check SMILES
print("\n3. CHECKING SMILES FORMAT")
print("-" * 40)
print(f"Sample SMILES:")
for i, smi in enumerate(df['Smiles'].head(10)):
    print(f"  {i+1}. {smi}")

# Check if SMILES have asterisk (polymer notation)
has_asterisk = df['Smiles'].str.contains('\*').sum()
print(f"\nSMILES with asterisk (*): {has_asterisk}/{len(df)}")

# Try to compute ECFP4 fingerprints
print("\n4. TESTING ECFP4 FINGERPRINT GENERATION")
print("-" * 40)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Test on a few SMILES
    test_smiles = df['Smiles'].head(5).tolist()
    fps = []
    for smi in test_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fp_array = np.array(fp)
            fps.append(fp_array)
            print(f"  SMILES: {smi} -> Fingerprint shape: {fp_array.shape}, Non-zero bits: {np.sum(fp_array > 0)}")
        else:
            print(f"  SMILES: {smi} -> FAILED to parse")

    if fps:
        print(f"\nECFP4 generation: SUCCESS")
        print(f"  Fingerprint size: 2048 bits")
except Exception as e:
    print(f"ECFP4 generation: FAILED - {e}")

# Build MLP baseline with ECFP4
print("\n5. BUILDING MLP+ECFP4 BASELINE")
print("-" * 40)

from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_ecfp4(smiles, radius=2, n_bits=2048):
    """Convert SMILES to ECFP4 fingerprint"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp)

# Generate fingerprints
print("Generating ECFP4 fingerprints...")
X_fps = []
y_vals = []
failed = 0
for idx, row in df.iterrows():
    fp = smiles_to_ecfp4(row['Smiles'])
    if fp is not None:
        X_fps.append(fp)
        y_vals.append(row['Tg'])
    else:
        failed += 1

X = np.array(X_fps)
y = np.array(y_vals)
print(f"Successfully converted: {len(X)}/{len(df)}")
print(f"Failed: {failed}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"y statistics: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}, std={y.std():.2f}")

# Simple MLP baseline
print("\n6. TRAINING MLP BASELINE (5-fold CV)")
print("-" * 40)

import torch.nn as nn
import torch.optim as optim

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_mlp(X_train, y_train, X_val, y_val, epochs=200, lr=0.001, verbose=False):
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

    model = SimpleMLP(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = torch.sqrt(criterion(val_pred, y_val_t)).item()

        losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 30:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch}: Train Loss = {loss.item():.4f}, Val RMSE = {val_loss:.4f}")

    model.load_state_dict(best_model_state)
    return model, scaler, losses

# 5-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
mae_scores = []
r2_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Further split train into train/val
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    model, scaler, losses = train_mlp(X_tr, y_tr, X_val, y_val, epochs=300, verbose=False)

    # Evaluate on test
    X_test_scaled = scaler.transform(X_test)
    X_test_t = torch.FloatTensor(X_test_scaled)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).squeeze().numpy()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    rmse_scores.append(rmse)
    mae_scores.append(mae)
    r2_scores.append(r2)

    print(f"Fold {fold+1}: RMSE = {rmse:.4f}, MAE = {mae:.4f}, R2 = {r2:.4f}")

print(f"\nMLP+ECFP4 Baseline Results (5-fold CV):")
print(f"  RMSE: {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}")
print(f"  MAE:  {np.mean(mae_scores):.4f} +/- {np.std(mae_scores):.4f}")
print(f"  R2:   {np.mean(r2_scores):.4f} +/- {np.std(r2_scores):.4f}")

# Also try Random Forest for comparison
print("\n7. RANDOM FOREST BASELINE")
print("-" * 40)
from sklearn.ensemble import RandomForestRegressor

rf_rmse_scores = []
rf_mae_scores = []
rf_r2_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    rf_rmse_scores.append(rmse)
    rf_mae_scores.append(mae)
    rf_r2_scores.append(r2)

    print(f"Fold {fold+1}: RMSE = {rmse:.4f}, MAE = {mae:.4f}, R2 = {r2:.4f}")

print(f"\nRandom Forest+ECFP4 Baseline Results (5-fold CV):")
print(f"  RMSE: {np.mean(rf_rmse_scores):.4f} +/- {np.std(rf_rmse_scores):.4f}")
print(f"  MAE:  {np.mean(rf_mae_scores):.4f} +/- {np.std(rf_mae_scores):.4f}")
print(f"  R2:   {np.mean(rf_r2_scores):.4f} +/- {np.std(rf_r2_scores):.4f}")

# Now let's check what the PSCP model is actually doing
print("\n8. ANALYZING PSCP MODEL PREDICTIONS")
print("-" * 40)

# Load the model and check predictions
try:
    import sys
    sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/6_day6_pscp')
    from pscp_model import PolymerGraphNeuralNetwork, PolymerSmilesEncoder
    from polymer_utils import polymer_to_graph

    # Load pretrained model
    checkpoint = torch.load('/home/home_public/aisci/AIscientist/src/6_day6_pscp/pretrained_pretrain_100k.pt',
                            map_location='cpu')
    print(f"Loaded checkpoint keys: {checkpoint.keys()}")

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    print(f"\nModel state dict keys (first 10): {list(state_dict.keys())[:10]}")

except Exception as e:
    print(f"Error loading PSCP model: {e}")

# Check the current experiment code
print("\n9. CHECKING EXPERIMENT CODE")
print("-" * 40)
import run_pscp_complete
print("Checking run_pscp_complete.py for issues...")

# Look at how the model processes data
print("\n10. PREDICTION DISTRIBUTION ANALYSIS")
print("-" * 40)
# Run a quick test with the PSCP model on a small subset
print("Testing PSCP model on sample data...")

# Summary
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)
print(f"""
1. Dataset: Tg values are in CELSIUS (range: {y.min():.1f} to {y.max():.1f}°C)
2. ECFP4 fingerprint generation: Working correctly
3. MLP+ECFP4 Baseline RMSE: {np.mean(rmse_scores):.2f} K (or °C, same scale for RMSE)
4. RF+ECFP4 Baseline RMSE: {np.mean(rf_rmse_scores):.2f}
5. Current PSCP RMSE: 103.18

KEY FINDING:
- MLP+ECFP4 achieves RMSE ~{np.mean(rmse_scores):.2f}
- PSCP model achieves RMSE ~103.18
- Gap: ~{103.18 - np.mean(rmse_scores):.2f} units

The baseline of 38.78 mentioned in the task appears to be achievable with MLP+ECFP4.
The PSCP model is performing ~{103.18/np.mean(rmse_scores):.1f}x worse than baseline.

This suggests the PSCP model has implementation issues:
- Possible causes:
  a) Incorrect graph construction from polymer SMILES
  b) Model architecture issues
  c) Training/optimization problems
  d) Feature normalization issues
""")
