"""
Phase 1: Pseudo-label Generation for PI1M using Bicerano-style group contribution.
Generates Tg_pseudo, Tm_pseudo, Egc_pseudo from molecular features.
"""
import pandas as pd
import numpy as np
import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, GraphDescriptors
from tqdm import tqdm
import sys
sys.path.append('/home/home_public/aisci/AIscientist/src/1_day2_reperioGT')
from dataset import mol_to_graph, compute_rotatable_bond_fraction


def compute_molecular_features(mol) -> dict:
    """
    Compute molecular features for Bicerano-style group contribution.

    Features:
    - MW: Molecular weight
    - NumRotatableBonds: Flexibility
    - NumHBD: Hydrogen bond donors
    - NumHBA: Hydrogen bond acceptors
    - NumAromaticRings: Aromaticity
    - NumSaturatedRings: Cyclic structure
    - TPSA: Topological polar surface area
    - FractionCSP3: sp3 hybridization fraction
    - NumHeteroatoms: Heteroatom count
    - MaxRingSize: Largest ring
    - Chi0v, Chi1n: Connectivity indices
    """
    try:
        features = {
            'MW': rdMolDescriptors.CalcExactMolWt(mol),
            'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'NumHBD': rdMolDescriptors.CalcNumHBD(mol),
            'NumHBA': rdMolDescriptors.CalcNumHBA(mol),
            'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(mol),
            'TPSA': Descriptors.TPSA(mol),
            'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
            'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
            'MaxRingSize': max([len(r) for r in mol.GetRingInfo().AtomRings()]) if mol.GetRingInfo().AtomRings() else 0,
            'Chi0v': GraphDescriptors.Chi0v(mol),
            'Chi1n': GraphDescriptors.Chi1n(mol),
        }
        return features
    except Exception as e:
        return None


def bicerano_tg_prediction(features: dict) -> float:
    """
    Predict Tg using Bicerano-style group contribution model.

    Simplified model: Tg = base + sum(contributions)
    Base Tg contribution from chain stiffness and intermolecular forces.
    """
    if features is None:
        return 300.0  # Default value

    # Base temperature (K)
    tg = 200.0

    # Molecular weight contribution (size effect)
    mw_factor = np.log1p(features['MW']) * 20
    tg += mw_factor

    # Rotatable bonds (flexibility reduces Tg)
    rot_penalty = features['NumRotatableBonds'] * 8
    tg -= rot_penalty

    # Aromatic rings increase Tg (rigidity)
    aromatic_boost = features['NumAromaticRings'] * 40
    tg += aromatic_boost

    # Saturated rings moderate increase
    ring_boost = features['NumSaturatedRings'] * 15
    tg += ring_boost

    # Polar interactions (H-bonding increases Tg)
    hb_boost = (features['NumHBD'] + features['NumHBA']) * 12
    tg += hb_boost

    # TPSA contribution (polarity)
    psa_contrib = features['TPSA'] * 0.3
    tg += psa_contrib

    # sp3 fraction (more flexible)
    sp3_penalty = features['FractionCSP3'] * 30
    tg -= sp3_penalty

    # Connectivity indices (topology)
    chi_contrib = features['Chi0v'] * 15
    tg += chi_contrib

    # Add some noise-like variability
    tg += np.random.normal(0, 10)

    # Physical constraints
    tg = max(150, min(600, tg))  # Tg between 150K and 600K

    return float(tg)


def predict_tm_from_tg(tg: float, features: dict) -> float:
    """
    Predict Tm from Tg using empirical relationship.

    For polymers: Tm ≈ 1.5 * Tg (rough rule of thumb)
    Modified by structural features.
    """
    if features is None:
        return tg * 1.5

    # Base ratio
    ratio = 1.5

    # Adjust based on crystallinity tendency
    # More regular/linear structures have higher Tm/Tg ratio
    if features['NumRotatableBonds'] < 5:
        ratio += 0.3  # Stiffer chains

    if features['NumAromaticRings'] > 0:
        ratio += 0.2  # Aromatics increase Tm more

    if features['FractionCSP3'] > 0.7:
        ratio -= 0.2  # More aliphatic

    tm = tg * ratio
    tm = max(tg + 50, min(800, tm))  # Tm > Tg, max 800K

    return float(tm)


def predict_egc_from_tg_tm(tg: float, tm: float, features: dict) -> float:
    """
    Predict Egc (energy gap at Tg) from Tg and Tm.

    Egc represents energy barrier for segmental motion.
    Empirical: Egc ≈ R * Tg * ln(Tm/Tg) simplified.
    """
    if features is None:
        return 50.0

    # Simplified relationship (kJ/mol)
    # Egc correlates with chain stiffness
    egc = (tg / 10.0) + (tm / 20.0)

    # Adjust for molecular features
    if features['NumAromaticRings'] > 0:
        egc += features['NumAromaticRings'] * 5

    if features['NumRotatableBonds'] > 10:
        egc -= 15

    egc = max(20, min(150, egc))

    return float(egc)


def generate_pseudo_labels(csv_path: str, output_path: str):
    """
    Generate pseudo-labels for all PI1M polymers.

    Saves dictionary with:
    - smiles: List of SMILES strings
    - tg_pseudo: Tg predictions
    - tm_pseudo: Tm predictions
    - egc_pseudo: Egc predictions
    - features: Molecular features used
    """
    print("=" * 80)
    print("PHASE 1: PSEUDO-LABEL GENERATION")
    print("=" * 80)

    # Load PI1M data
    print(f"\nLoading PI1M from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")

    # Process all polymers
    results = {
        'smiles': [],
        'tg_pseudo': [],
        'tm_pseudo': [],
        'egc_pseudo': [],
        'valid': []
    }

    print("\nGenerating pseudo-labels...")
    failed_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Handle both 'SMILES' and 'Smiles' column names
        smiles = row.get('SMILES', row.get('Smiles', None))
        if smiles is None:
            results['smiles'].append('')
            results['tg_pseudo'].append(0.0)
            results['tm_pseudo'].append(0.0)
            results['egc_pseudo'].append(0.0)
            results['valid'].append(False)
            failed_count += 1
            continue

        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                results['smiles'].append(smiles)
                results['tg_pseudo'].append(0.0)
                results['tm_pseudo'].append(0.0)
                results['egc_pseudo'].append(0.0)
                results['valid'].append(False)
                failed_count += 1
                continue

            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                            Chem.SanitizeFlags.SANITIZE_ADJUSTHS)

            # Compute features
            features = compute_molecular_features(mol)

            if features is None:
                results['smiles'].append(smiles)
                results['tg_pseudo'].append(0.0)
                results['tm_pseudo'].append(0.0)
                results['egc_pseudo'].append(0.0)
                results['valid'].append(False)
                failed_count += 1
                continue

            # Generate pseudo-labels
            tg = bicerano_tg_prediction(features)
            tm = predict_tm_from_tg(tg, features)
            egc = predict_egc_from_tg_tm(tg, tm, features)

            results['smiles'].append(smiles)
            results['tg_pseudo'].append(tg)
            results['tm_pseudo'].append(tm)
            results['egc_pseudo'].append(egc)
            results['valid'].append(True)

        except Exception as e:
            results['smiles'].append(smiles)
            results['tg_pseudo'].append(0.0)
            results['tm_pseudo'].append(0.0)
            results['egc_pseudo'].append(0.0)
            results['valid'].append(False)
            failed_count += 1

    # Convert to arrays
    tg_array = np.array(results['tg_pseudo'], dtype=np.float32)
    tm_array = np.array(results['tm_pseudo'], dtype=np.float32)
    egc_array = np.array(results['egc_pseudo'], dtype=np.float32)
    valid_array = np.array(results['valid'], dtype=bool)

    # Create stacked targets [N, 3]
    targets = np.stack([tg_array, tm_array, egc_array], axis=1)

    # Save to file
    save_dict = {
        'smiles': results['smiles'],
        'targets': targets,  # [N, 3] array
        'valid': valid_array,
        'tg_pseudo': tg_array,
        'tm_pseudo': tm_array,
        'egc_pseudo': egc_array,
    }

    torch.save(save_dict, output_path)

    # Print statistics
    print(f"\n{'='*80}")
    print("PSEUDO-LABEL GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total processed: {len(df)}")
    print(f"Valid: {valid_array.sum()}")
    print(f"Failed: {failed_count}")
    print(f"\nTg pseudo-statistics (K):")
    print(f"  Mean: {tg_array[valid_array].mean():.2f}")
    print(f"  Std: {tg_array[valid_array].std():.2f}")
    print(f"  Min: {tg_array[valid_array].min():.2f}")
    print(f"  Max: {tg_array[valid_array].max():.2f}")
    print(f"\nTm pseudo-statistics (K):")
    print(f"  Mean: {tm_array[valid_array].mean():.2f}")
    print(f"  Std: {tm_array[valid_array].std():.2f}")
    print(f"  Min: {tm_array[valid_array].min():.2f}")
    print(f"  Max: {tm_array[valid_array].max():.2f}")
    print(f"\nEgc pseudo-statistics (kJ/mol):")
    print(f"  Mean: {egc_array[valid_array].mean():.2f}")
    print(f"  Std: {egc_array[valid_array].std():.2f}")
    print(f"  Min: {egc_array[valid_array].min():.2f}")
    print(f"  Max: {egc_array[valid_array].max():.2f}")
    print(f"\nThermodynamic consistency check:")
    consistent = (tm_array[valid_array] > tg_array[valid_array]).sum()
    print(f"  Tm > Tg: {consistent}/{valid_array.sum()} ({100*consistent/valid_array.sum():.1f}%)")
    print(f"\nSaved to: {output_path}")
    print(f"{'='*80}\n")

    return True  # Add explicit return for success


if __name__ == "__main__":
    # Generate pseudo-labels for PI1M
    generate_pseudo_labels(
        csv_path='/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv',
        output_path='/home/home_public/aisci/AIscientist/src/7_day8_tcp/PI1M_pseudo_labels.pt'
    )
