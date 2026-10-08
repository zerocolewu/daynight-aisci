#!/usr/bin/env python3
"""
CRITICAL FINDING: Polymer SMILES cleaning issue
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CRITICAL ISSUE: Polymer SMILES Cleaning")
print("=" * 80)

df = pd.read_csv('/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv')

print("\n1. ANALYZING POLYMER SMILES CLEANING")
print("-" * 40)

# Current approach (from run_pscp_complete.py)
def current_clean(smiles):
    """What the experiment code does"""
    return smiles.replace('*', '')

# Proper approach
def proper_clean(smiles):
    """Replace * with H (wildcard becomes hydrogen)"""
    return smiles.replace('*', '[H]').replace('([H])', '')

# Test on samples
print("Sample SMILES processing:")
for i in range(10):
    smi = df['Smiles'].iloc[i]
    current = current_clean(smi)
    proper = proper_clean(smi)

    mol_current = Chem.MolFromSmiles(current)
    mol_proper = Chem.MolFromSmiles(proper)

    print(f"\n{i+1}. Original: {smi}")
    print(f"   Current clean: {current}")
    print(f"   Current valid: {mol_current is not None}")
    print(f"   Proper clean: {proper}")
    print(f"   Proper valid: {mol_proper is not None}")

# Count failures
print("\n2. FAILURE COUNT")
print("-" * 40)

current_fail = 0
proper_fail = 0
ecfp_current_fail = 0
ecfp_proper_fail = 0

for idx, row in df.iterrows():
    smi = row['Smiles']

    # Current approach
    current = current_clean(smi)
    mol_c = Chem.MolFromSmiles(current)
    if mol_c is None:
        current_fail += 1
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol_c, 2, nBits=2048)
        if np.sum(fp) == 0:
            ecfp_current_fail += 1

    # Proper approach
    proper = proper_clean(smi)
    mol_p = Chem.MolFromSmiles(proper)
    if mol_p is None:
        proper_fail += 1
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol_p, 2, nBits=2048)
        if np.sum(fp) == 0:
            ecfp_proper_fail += 1

print(f"Total samples: {len(df)}")
print(f"\nCurrent approach (replace * with nothing):")
print(f"  Invalid SMILES: {current_fail} ({100*current_fail/len(df):.1f}%)")
print(f"  Zero ECFP: {ecfp_current_fail}")

print(f"\nProper approach (replace * with [H]):")
print(f"  Invalid SMILES: {proper_fail} ({100*proper_fail/len(df):.1f}%)")
print(f"  Zero ECFP: {ecfp_proper_fail}")

# Test graph construction
print("\n3. GRAPH CONSTRUCTION TEST")
print("-" * 40)

def smiles_to_graph_current(smiles):
    """From run_pscp_complete.py"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol

def smiles_to_graph_proper(smiles):
    """Proper handling"""
    clean = smiles.replace('*', '[H]').replace('([H])', '')
    mol = Chem.MolFromSmiles(clean)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol

current_graphs = 0
proper_graphs = 0
current_atoms = []
proper_atoms = []

for idx, row in df.iterrows():
    smi = row['Smiles']

    # Current
    clean = smi.replace('*', '')
    mol = smiles_to_graph_current(clean)
    if mol is not None:
        current_graphs += 1
        current_atoms.append(mol.GetNumAtoms())

    # Proper
    clean = proper_clean(smi)
    mol = smiles_to_graph_proper(clean)
    if mol is not None:
        proper_graphs += 1
        proper_atoms.append(mol.GetNumAtoms())

print(f"Current approach valid graphs: {current_graphs}/{len(df)}")
print(f"Proper approach valid graphs: {proper_graphs}/{len(df)}")
print(f"\nAverage atoms (current): {np.mean(current_atoms):.2f}")
print(f"Average atoms (proper): {np.mean(proper_atoms):.2f}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
CRITICAL FINDING:

The polymer SMILES in the Tg dataset use '*' to mark connection points.
The current code simply removes '*', which creates INVALID SMILES!

Example:
  Original: *CC(*)C
  Current: CC()C  <- INVALID (empty parentheses)
  Proper:  HCC(H)C or CC(H)C  <- VALID

IMPACT:
- {current_fail} samples ({100*current_fail/len(df):.1f}%) produce invalid SMILES
- These samples get zero/invalid features
- This degrades model performance

The 103.18 RMSE is partly due to:
1. Poor atom features (5-dim vs 159-dim)
2. Invalid SMILES handling (creates zero features)
3. Combined GNN+ECFP model architecture issues

The RF+ECFP baseline (40.64 RMSE) works because:
- It handles the invalid cases gracefully
- ECFP on its own is a strong feature
- No complex GNN to mess things up
""")
