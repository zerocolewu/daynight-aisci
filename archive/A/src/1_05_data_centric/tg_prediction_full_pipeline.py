#!/usr/bin/env python
"""
Comprehensive Data-Centric Foundation Experiment for Polymer Tg Prediction
Implements all phases: Data Analysis, Multimer Generation, Baseline Models, Diagnostics
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict
import json
from collections import defaultdict
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# PyTorch Geometric
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, global_mean_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch Geometric not available. GCN models will be skipped.")
    TORCH_GEOMETRIC_AVAILABLE = False

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# =============================================================================
# PHASE 1: Data Analysis Pipeline
# =============================================================================

class DataAnalyzer:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.figures_dir = Path("figures")
        self.figures_dir.mkdir(exist_ok=True)

    def load_data(self):
        """Load Tg dataset"""
        print("\n" + "="*80)
        print("PHASE 1: Data Analysis Pipeline")
        print("="*80)
        print("\nLoading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} samples")
        print(f"Columns: {list(self.df.columns)}")
        print(f"\nFirst few samples:\n{self.df.head()}")
        return self.df

    def plot_histogram(self):
        """Plot histogram of Tg values"""
        print("\nPlotting Tg distribution...")
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(self.df['Tg'], bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Tg (Celsius)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Glass Transition Temperature')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.hist(self.df['Tg'], bins=50, edgecolor='black', alpha=0.7, log=True)
        plt.xlabel('Tg (Celsius)')
        plt.ylabel('Frequency (log scale)')
        plt.title('Tg Distribution (Log Scale)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'tg_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.figures_dir / 'tg_distribution.png'}")
        plt.close()

    def check_outliers(self):
        """Check for outliers in Tg values"""
        print("\nChecking for outliers...")
        tg_values = self.df['Tg'].values

        # Original threshold check (in case data is in Kelvin)
        outliers_kelvin = self.df[(self.df['Tg'] < 100) | (self.df['Tg'] > 800)]
        print(f"Samples with Tg < 100K or > 800K: {len(outliers_kelvin)}")

        # More reasonable thresholds for Celsius
        min_celsius = -150  # Very low but possible for some polymers
        max_celsius = 600   # Very high but possible

        outliers_celsius = self.df[(self.df['Tg'] < min_celsius) | (self.df['Tg'] > max_celsius)]
        print(f"Samples with Tg < {min_celsius}°C or > {max_celsius}°C: {len(outliers_celsius)}")

        if len(outliers_celsius) > 0:
            print(f"Outlier samples:\n{outliers_celsius}")

        # Statistical outliers using IQR method
        Q1 = np.percentile(tg_values, 25)
        Q3 = np.percentile(tg_values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        statistical_outliers = self.df[(self.df['Tg'] < lower_bound) | (self.df['Tg'] > upper_bound)]
        print(f"Statistical outliers (IQR method): {len(statistical_outliers)}")
        print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

        return {
            'count_kelvin_threshold': len(outliers_kelvin),
            'count_celsius_threshold': len(outliers_celsius),
            'count_statistical_outliers': len(statistical_outliers),
            'q1': Q1,
            'q3': Q3,
            'iqr': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

    def analyze_distribution(self):
        """Analyze skewness and kurtosis"""
        print("\nAnalyzing distribution characteristics...")
        tg_values = self.df['Tg'].values

        mean_tg = np.mean(tg_values)
        std_tg = np.std(tg_values)
        median_tg = np.median(tg_values)
        skewness = stats.skew(tg_values)
        kurtosis = stats.kurtosis(tg_values)

        print(f"Mean Tg: {mean_tg:.2f}°C")
        print(f"Median Tg: {median_tg:.2f}°C")
        print(f"Std Tg: {std_tg:.2f}°C")
        print(f"Skewness: {skewness:.4f}")
        print(f"Kurtosis: {kurtosis:.4f}")

        # Test for normality
        shapiro_stat, shapiro_p = stats.shapiro(tg_values[:5000])  # Limit sample size
        print(f"Shapiro-Wilk test (n=5000): statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4e}")

        return {
            'mean': mean_tg,
            'std': std_tg,
            'median': median_tg,
            'min': np.min(tg_values),
            'max': np.max(tg_values),
            'skewness': skewness,
            'kurtosis': kurtosis,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p
        }

    def identify_duplicates(self):
        """Identify duplicate SMILES with different Tg values"""
        print("\nChecking for duplicate SMILES...")
        smiles_groups = self.df.groupby('Smiles')['Tg'].agg(['count', 'mean', 'std', 'min', 'max'])

        duplicates = smiles_groups[smiles_groups['count'] > 1]
        print(f"Found {len(duplicates)} unique SMILES with multiple entries")

        if len(duplicates) > 0:
            print(f"Total duplicate entries: {duplicates['count'].sum()}")
            print(f"\nTop 10 SMILES with most entries:")
            print(duplicates.nlargest(10, 'count'))

            # Analyze variability
            high_variability = duplicates[duplicates['std'] > 10]
            print(f"\nSMILES with high variability (std > 10°C): {len(high_variability)}")

            if len(high_variability) > 0:
                print(f"\nHigh variability examples:")
                print(high_variability.nlargest(5, 'std'))

        return {
            'num_unique_duplicates': len(duplicates),
            'total_duplicate_entries': duplicates['count'].sum() if len(duplicates) > 0 else 0,
            'high_variability_count': len(duplicates[duplicates['std'] > 10]) if len(duplicates) > 0 else 0
        }

    def check_smiles_validity(self):
        """Check SMILES validity with RDKit"""
        print("\nChecking SMILES validity...")
        valid_count = 0
        invalid_count = 0
        invalid_examples = []

        for idx, row in self.df.iterrows():
            smiles = row['Smiles']
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_count += 1
                else:
                    invalid_count += 1
                    if len(invalid_examples) < 10:
                        invalid_examples.append((idx, smiles, row['Tg']))
            except Exception as e:
                invalid_count += 1
                if len(invalid_examples) < 10:
                    invalid_examples.append((idx, smiles, row['Tg']))

        print(f"Valid SMILES: {valid_count}/{len(self.df)} ({100*valid_count/len(self.df):.2f}%)")
        print(f"Invalid SMILES: {invalid_count}/{len(self.df)} ({100*invalid_count/len(self.df):.2f}%)")

        if invalid_count > 0:
            print(f"\nInvalid SMILES examples:")
            for idx, smiles, tg in invalid_examples[:5]:
                print(f"  [{idx}] {smiles} -> Tg={tg}")

        # Check for connection points (*)
        has_wildcard = self.df['Smiles'].str.contains('\*').sum()
        print(f"\nSMILES with wildcard (*) connection points: {has_wildcard}/{len(self.df)}")

        return {
            'valid_count': valid_count,
            'invalid_count': invalid_count,
            'validity_percentage': 100*valid_count/len(self.df),
            'has_wildcard': has_wildcard
        }

    def generate_report(self, outlier_stats, dist_stats, duplicate_stats, validity_stats):
        """Generate comprehensive data analysis report"""
        report_path = "data_analysis_report.txt"
        print(f"\nGenerating data analysis report: {report_path}")

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DATA ANALYSIS REPORT - Polymer Tg Prediction\n")
            f.write("="*80 + "\n\n")

            f.write("1. DATASET OVERVIEW\n")
            f.write("-"*80 + "\n")
            f.write(f"Total samples: {len(self.df)}\n")
            f.write(f"Columns: {list(self.df.columns)}\n")
            f.write(f"Data path: {self.data_path}\n\n")

            f.write("2. Tg DISTRIBUTION STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Mean Tg: {dist_stats['mean']:.2f}°C\n")
            f.write(f"Median Tg: {dist_stats['median']:.2f}°C\n")
            f.write(f"Std Tg: {dist_stats['std']:.2f}°C\n")
            f.write(f"Min Tg: {dist_stats['min']:.2f}°C\n")
            f.write(f"Max Tg: {dist_stats['max']:.2f}°C\n")
            f.write(f"Skewness: {dist_stats['skewness']:.4f}\n")
            f.write(f"Kurtosis: {dist_stats['kurtosis']:.4f}\n")
            f.write(f"Shapiro-Wilk test: statistic={dist_stats['shapiro_stat']:.4f}, p-value={dist_stats['shapiro_p']:.4e}\n\n")

            f.write("3. OUTLIER ANALYSIS\n")
            f.write("-"*80 + "\n")
            f.write(f"Samples outside [100K, 800K]: {outlier_stats['count_kelvin_threshold']}\n")
            f.write(f"Samples outside reasonable Celsius range: {outlier_stats['count_celsius_threshold']}\n")
            f.write(f"Statistical outliers (IQR method): {outlier_stats['count_statistical_outliers']}\n")
            f.write(f"Q1: {outlier_stats['q1']:.2f}, Q3: {outlier_stats['q3']:.2f}, IQR: {outlier_stats['iqr']:.2f}\n")
            f.write(f"IQR bounds: [{outlier_stats['lower_bound']:.2f}, {outlier_stats['upper_bound']:.2f}]\n\n")

            f.write("4. DUPLICATE ANALYSIS\n")
            f.write("-"*80 + "\n")
            f.write(f"Unique SMILES with multiple entries: {duplicate_stats['num_unique_duplicates']}\n")
            f.write(f"Total duplicate entries: {duplicate_stats['total_duplicate_entries']}\n")
            f.write(f"High variability duplicates (std > 10°C): {duplicate_stats['high_variability_count']}\n\n")

            f.write("5. SMILES VALIDITY\n")
            f.write("-"*80 + "\n")
            f.write(f"Valid SMILES: {validity_stats['valid_count']}/{len(self.df)} ({validity_stats['validity_percentage']:.2f}%)\n")
            f.write(f"Invalid SMILES: {validity_stats['invalid_count']}/{len(self.df)}\n")
            f.write(f"SMILES with connection points (*): {validity_stats['has_wildcard']}/{len(self.df)}\n\n")

            f.write("6. VISUALIZATIONS\n")
            f.write("-"*80 + "\n")
            f.write(f"Tg distribution plot: {self.figures_dir / 'tg_distribution.png'}\n\n")

            f.write("7. RECOMMENDATIONS\n")
            f.write("-"*80 + "\n")
            if validity_stats['invalid_count'] > 0:
                f.write("- WARNING: Invalid SMILES found. Consider cleaning or removing these entries.\n")
            if duplicate_stats['high_variability_count'] > 0:
                f.write("- WARNING: High variability in duplicate SMILES. Consider investigating measurement consistency.\n")
            if dist_stats['shapiro_p'] < 0.05:
                f.write("- Tg distribution is not normal. Consider transformations or non-parametric methods.\n")
            f.write("- All SMILES contain connection points (*), indicating they are polymer repeat units.\n")
            f.write("- Consider data augmentation through multimer generation for improved model performance.\n")

        print(f"Report saved to: {report_path}")

    def run_full_analysis(self):
        """Run complete data analysis pipeline"""
        self.load_data()
        self.plot_histogram()
        outlier_stats = self.check_outliers()
        dist_stats = self.analyze_distribution()
        duplicate_stats = self.identify_duplicates()
        validity_stats = self.check_smiles_validity()
        self.generate_report(outlier_stats, dist_stats, duplicate_stats, validity_stats)
        return self.df


# =============================================================================
# PHASE 2: Multimer 3D Structure Generation
# =============================================================================

class MultimerGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.log_path = "multimer_generation_log.txt"

    def create_multimer(self, smiles: str, num_units: int) -> str:
        """Create multimer by connecting repeat units"""
        try:
            # Replace wildcard (*) with connections
            # For simplicity, create linear chain
            if '*' not in smiles:
                return None

            # Remove wildcards and connect units
            # Simple approach: replace * with single bonds
            unit = smiles.replace('*', '')

            # Connect units in a linear fashion
            multimer = unit
            for _ in range(num_units - 1):
                multimer += unit

            return multimer
        except Exception as e:
            return None

    def generate_3d_conformer(self, smiles: str, max_attempts: int = 5) -> Tuple[object, Dict]:
        """Generate 3D conformer using RDKit MMFF94"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, {}

        try:
            # Add hydrogens
            mol = Chem.AddHs(mol)

            # Generate 3D coordinates with multiple conformers
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=max_attempts,
                params=AllChem.ETKDGv3()
            )

            if len(conformer_ids) == 0:
                return None, {}

            # Minimize each conformer and select the best
            best_energy = float('inf')
            best_conf_id = None

            for conf_id in conformer_ids:
                # MMFF94 minimization
                try:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf_id)
                    if ff is not None:
                        ff.Minimize(maxIts=200)
                        energy = ff.CalcEnergy()
                        if energy < best_energy:
                            best_energy = energy
                            best_conf_id = conf_id
                except:
                    pass

            if best_conf_id is None:
                # Fallback to UFF if MMFF fails
                for conf_id in conformer_ids:
                    try:
                        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                        if ff is not None:
                            ff.Minimize(maxIts=200)
                            energy = ff.CalcEnergy()
                            if energy < best_energy:
                                best_energy = energy
                                best_conf_id = conf_id
                    except:
                        pass

            if best_conf_id is None:
                return None, {}

            # Extract statistics
            stats = self.extract_mol_stats(mol, best_conf_id)

            return mol, stats

        except Exception as e:
            return None, {}

    def extract_mol_stats(self, mol: object, conf_id: int) -> Dict:
        """Extract molecular statistics"""
        try:
            conformer = mol.GetConformer(conf_id)

            # Basic properties
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            num_heavy_atoms = rdMolDescriptors.CalcNumHeavyAtoms(mol)
            mol_weight = Descriptors.MolWt(mol)

            # 3D properties
            coords = np.array([conformer.GetAtomPosition(i) for i in range(num_atoms)])

            # Center of mass (approximate)
            center = np.mean(coords, axis=0)

            # Radius of gyration
            rg = np.sqrt(np.mean(np.sum((coords - center)**2, axis=1)))

            # End-to-end distance (distance between first and last heavy atom)
            if num_heavy_atoms >= 2:
                heavy_indices = [i.GetIdx() for i in mol.GetAtoms() if i.GetAtomicNum() > 1]
                if len(heavy_indices) >= 2:
                    first_pos = conformer.GetAtomPosition(heavy_indices[0])
                    last_pos = conformer.GetAtomPosition(heavy_indices[-1])
                    e2e_dist = np.sqrt(
                        (first_pos.x - last_pos.x)**2 +
                        (first_pos.y - last_pos.y)**2 +
                        (first_pos.z - last_pos.z)**2
                    )
                else:
                    e2e_dist = 0.0
            else:
                e2e_dist = 0.0

            return {
                'num_atoms': num_atoms,
                'num_bonds': num_bonds,
                'num_heavy_atoms': num_heavy_atoms,
                'molecular_weight': mol_weight,
                'radius_of_gyration': rg,
                'end_to_end_distance': e2e_dist
            }
        except Exception as e:
            return {}

    def generate_multimers_sample(self, sample_size: int = 100):
        """Generate multimers for a sample of polymers"""
        print("\n" + "="*80)
        print("PHASE 2: Multimer 3D Structure Generation")
        print("="*80)

        print(f"\nGenerating multimers for {sample_size} random samples...")

        results = []
        success_count = {2: 0, 3: 0, 4: 0, 5: 0}

        # Sample random polymers
        sample_df = self.df.sample(n=min(sample_size, len(self.df)), random_state=RANDOM_SEED)

        for idx, row in sample_df.iterrows():
            smiles = row['Smiles']
            tg = row['Tg']

            result = {
                'smiles': smiles,
                'tg': tg,
                'multimers': {}
            }

            for num_units in [2, 3, 4, 5]:
                multimer_smiles = self.create_multimer(smiles, num_units)

                if multimer_smiles:
                    mol, stats = self.generate_3d_conformer(multimer_smiles)

                    if mol is not None and stats:
                        result['multimers'][num_units] = {
                            'success': True,
                            'stats': stats
                        }
                        success_count[num_units] += 1
                    else:
                        result['multimers'][num_units] = {
                            'success': False,
                            'stats': {}
                        }
                else:
                    result['multimers'][num_units] = {
                        'success': False,
                        'stats': {}
                    }

            results.append(result)

            if (len(results) % 20) == 0:
                print(f"Processed {len(results)}/{sample_size} samples...")

        # Compile statistics
        print(f"\nSuccess rates:")
        for num_units in [2, 3, 4, 5]:
            rate = 100 * success_count[num_units] / sample_size
            print(f"  {num_units}-mer: {success_count[num_units]}/{sample_size} ({rate:.1f}%)")

        # Save log
        self.save_generation_log(results, success_count, sample_size)

        return results

    def save_generation_log(self, results: List[Dict], success_count: Dict, sample_size: int):
        """Save multimer generation log"""
        print(f"\nSaving multimer generation log: {self.log_path}")

        with open(self.log_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MULTIMER GENERATION LOG - Polymer 3D Structures\n")
            f.write("="*80 + "\n\n")

            f.write("SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Sample size: {sample_size}\n")
            f.write(f"Total polymers processed: {len(results)}\n\n")

            f.write("Success Rates:\n")
            for num_units in [2, 3, 4, 5]:
                rate = 100 * success_count[num_units] / sample_size
                f.write(f"  {num_units}-mer: {success_count[num_units]}/{sample_size} ({rate:.1f}%)\n")
            f.write("\n")

            # Aggregate statistics
            f.write("AGGREGATE STATISTICS\n")
            f.write("-"*80 + "\n\n")

            for num_units in [2, 3, 4, 5]:
                unit_results = [r['multimers'][num_units] for r in results if r['multimers'][num_units]['success']]

                if len(unit_results) > 0:
                    f.write(f"{num_units}-mer Statistics (n={len(unit_results)}):\n")

                    stats_list = [r['stats'] for r in unit_results]

                    for key in ['num_atoms', 'num_bonds', 'num_heavy_atoms', 'molecular_weight',
                               'radius_of_gyration', 'end_to_end_distance']:
                        values = [s.get(key, 0) for s in stats_list if key in s]
                        if values:
                            f.write(f"  {key}: mean={np.mean(values):.2f}, std={np.std(values):.2f}, "
                                   f"min={np.min(values):.2f}, max={np.max(values):.2f}\n")

                    f.write("\n")

            # Sample detailed entries
            f.write("SAMPLE DETAILED ENTRIES (first 10)\n")
            f.write("-"*80 + "\n\n")

            for i, result in enumerate(results[:10]):
                f.write(f"Entry {i+1}:\n")
                f.write(f"  SMILES: {result['smiles']}\n")
                f.write(f"  Tg: {result['tg']:.2f}°C\n")

                for num_units in [2, 3, 4, 5]:
                    m = result['multimers'][num_units]
                    f.write(f"  {num_units}-mer: {'Success' if m['success'] else 'Failed'}\n")
                    if m['success'] and m['stats']:
                        for key, val in m['stats'].items():
                            f.write(f"    {key}: {val:.4f}\n")

                f.write("\n")

        print(f"Log saved to: {self.log_path}")


# =============================================================================
# PHASE 3: Vanilla Baseline Models
# =============================================================================

class TgDataset(Dataset):
    """PyTorch Dataset for Tg prediction"""
    def __init__(self, smiles_list, tg_values, transform=None):
        self.smiles_list = smiles_list
        self.tg_values = torch.FloatTensor(tg_values).unsqueeze(1)
        self.transform = transform

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        tg = self.tg_values[idx]

        if self.transform:
            smiles = self.transform(smiles)

        return smiles, tg


class FingerprintExtractor:
    """Extract ECFP fingerprints from SMILES"""
    def __init__(self, radius=2, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits

    def extract(self, smiles: str) -> np.ndarray:
        """Extract ECFP fingerprint"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits)

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.radius,
            nBits=self.n_bits
        )
        return np.array(fp)


class MLPModel(nn.Module):
    """3-layer MLP for Tg prediction"""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super(MLPModel, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GraphDataset(Dataset):
    """PyTorch Geometric Dataset for graph data"""
    def __init__(self, smiles_list, tg_values):
        self.smiles_list = smiles_list
        self.tg_values = torch.FloatTensor(tg_values)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        tg = self.tg_values[idx]

        # Convert SMILES to graph
        data = self.smiles_to_graph(smiles)
        data.y = tg
        return data

    def smiles_to_graph(self, smiles):
        """Convert SMILES to PyTorch Geometric Data object"""
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            # Return empty graph
            return Data(
                x=torch.zeros(1, 1),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros(0, 1)
            )

        # Node features
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetHybridization(),
                atom.GetIsAromatic(),
            ]
            atom_features.append(features)

        x = torch.tensor(atom_features, dtype=torch.float)

        # Edge indices
        edge_indices = []
        edge_attrs = []

        for bond in mol.GetBonds():
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()

            edge_indices.append([idx1, idx2])
            edge_indices.append([idx2, idx1])

            # Bond type
            bond_type = bond.GetBondType()
            bond_feat = [
                bond_type == Chem.rdchem.BondType.SINGLE,
                bond_type == Chem.rdchem.BondType.DOUBLE,
                bond_type == Chem.rdchem.BondType.TRIPLE,
                bond_type == Chem.rdchem.BondType.AROMATIC,
            ]

            edge_attrs.append(bond_feat)
            edge_attrs.append(bond_feat)

        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 4), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class SimpleGCN(nn.Module):
    """3-layer GCN for graph-based Tg prediction"""
    def __init__(self, node_dim, hidden_dim=128, dropout=0.3):
        super(SimpleGCN, self).__init__()

        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)

        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)

        x = F.relu(self.conv3(x, edge_index))

        # Global pooling
        x = global_mean_pool(x, batch)

        # FC layers
        x = self.fc(x)

        return x


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class ModelTrainer:
    """Train and evaluate baseline models"""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}

    def split_data(self):
        """Split data into train/val/test (80/10/10)"""
        print("\nSplitting data into train/val/test (80/10/10)...")

        smiles = self.df['Smiles'].values
        tg = self.df['Tg'].values

        # First split: 80% train, 20% temp
        smiles_train, smiles_temp, tg_train, tg_temp = train_test_split(
            smiles, tg, test_size=0.2, random_state=RANDOM_SEED
        )

        # Second split: 10% val, 10% test
        smiles_val, smiles_test, tg_val, tg_test = train_test_split(
            smiles_temp, tg_temp, test_size=0.5, random_state=RANDOM_SEED
        )

        print(f"Train: {len(smiles_train)}, Val: {len(smiles_val)}, Test: {len(smiles_test)}")

        return {
            'train': (smiles_train, tg_train),
            'val': (smiles_val, tg_val),
            'test': (smiles_test, tg_test)
        }

    def train_mlp(self, splits):
        """Train MLP baseline with ECFP fingerprints"""
        print("\n" + "="*80)
        print("PHASE 3: Vanilla Baseline Models - MLP")
        print("="*80)

        # Extract fingerprints
        print("\nExtracting ECFP fingerprints...")
        fp_extractor = FingerprintExtractor(radius=2, n_bits=2048)

        X_train = np.array([fp_extractor.extract(s) for s in splits['train'][0]], dtype=np.float32)
        X_val = np.array([fp_extractor.extract(s) for s in splits['val'][0]], dtype=np.float32)
        X_test = np.array([fp_extractor.extract(s) for s in splits['test'][0]], dtype=np.float32)

        y_train = splits['train'][1]
        y_val = splits['val'][1]
        y_test = splits['test'][1]

        print(f"Feature dimensions: {X_train.shape}")

        # Create datasets
        train_dataset = TgDataset(X_train, y_train)
        val_dataset = TgDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Create model
        model = MLPModel(input_dim=X_train.shape[1]).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.MSELoss()

        # Training loop
        print("\nTraining MLP...")
        max_epochs = 200
        early_stopping = EarlyStopping(patience=20, min_delta=0.001)

        train_losses = []
        val_losses = []

        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(max_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()

            # Early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(best_model_state)

        # Evaluate
        print("\nEvaluating MLP...")
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
            y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

            predictions = model(X_test_tensor)
            predictions = predictions.cpu().numpy().flatten()
            y_true = y_test

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)

        print(f"\nMLP Results:")
        print(f"  RMSE: {rmse:.4f}°C")
        print(f"  MAE: {mae:.4f}°C")
        print(f"  R²: {r2:.4f}")

        # Save model
        model_path = "trained_models/mlp_baseline.pth"
        torch.save({
            'model_state_dict': best_model_state,
            'input_dim': X_train.shape[1],
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }, model_path)
        print(f"Model saved: {model_path}")

        return {
            'model': model,
            'predictions': predictions,
            'y_true': y_true,
            'metrics': {'rmse': rmse, 'mae': mae, 'r2': r2},
            'train_losses': train_losses,
            'val_losses': val_losses
        }

    def train_gcn(self, splits):
        """Train Simple GCN baseline"""
        if not TORCH_GEOMETRIC_AVAILABLE:
            print("\nSkipping GCN - PyTorch Geometric not available")
            return None

        print("\n" + "="*80)
        print("PHASE 3: Vanilla Baseline Models - Simple GCN")
        print("="*80)

        # Create graph datasets
        print("\nConverting SMILES to graphs...")
        train_dataset = GraphDataset(splits['train'][0], splits['train'][1])
        val_dataset = GraphDataset(splits['val'][0], splits['val'][1])
        test_dataset = GraphDataset(splits['test'][0], splits['test'][1])

        # Get node feature dimension
        sample_data = train_dataset[0]
        node_dim = sample_data.x.shape[1]

        print(f"Node feature dimension: {node_dim}")

        # Create dataloaders
        def collate_fn(batch):
            return Batch.from_data_list(batch)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        # Create model
        model = SimpleGCN(node_dim=node_dim).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.MSELoss()

        # Training loop
        print("\nTraining GCN...")
        max_epochs = 200
        early_stopping = EarlyStopping(patience=20, min_delta=0.001)

        train_losses = []
        val_losses = []

        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(max_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(DEVICE)

                optimizer.zero_grad()
                predictions = model(batch)
                loss = criterion(predictions, batch.y.unsqueeze(1))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(DEVICE)
                    predictions = model(batch)
                    loss = criterion(predictions, batch.y.unsqueeze(1))
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()

            # Early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(best_model_state)

        # Evaluate
        print("\nEvaluating GCN...")
        model.eval()
        all_predictions = []
        all_true = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                predictions = model(batch)
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_true.extend(batch.y.cpu().numpy().flatten())

        predictions = np.array(all_predictions)
        y_true = np.array(all_true)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)

        print(f"\nGCN Results:")
        print(f"  RMSE: {rmse:.4f}°C")
        print(f"  MAE: {mae:.4f}°C")
        print(f"  R²: {r2:.4f}")

        # Save model
        model_path = "trained_models/gcn_baseline.pth"
        torch.save({
            'model_state_dict': best_model_state,
            'node_dim': node_dim,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }, model_path)
        print(f"Model saved: {model_path}")

        return {
            'model': model,
            'predictions': predictions,
            'y_true': y_true,
            'metrics': {'rmse': rmse, 'mae': mae, 'r2': r2},
            'train_losses': train_losses,
            'val_losses': val_losses
        }

    def save_baseline_results(self, mlp_results, gcn_results):
        """Save baseline results to file"""
        results_path = "baseline_results.txt"
        print(f"\nSaving baseline results: {results_path}")

        with open(results_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BASELINE MODEL RESULTS - Polymer Tg Prediction\n")
            f.write("="*80 + "\n\n")

            f.write("EXPERIMENT CONFIGURATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Total samples: {len(self.df)}\n")
            f.write(f"Train/Val/Test split: 80/10/10\n")
            f.write(f"Random seed: {RANDOM_SEED}\n")
            f.write(f"Batch size: 32\n")
            f.write(f"Max epochs: 200\n")
            f.write(f"Early stopping patience: 20\n")
            f.write(f"Device: {DEVICE}\n\n")

            f.write("MLP BASELINE (ECFP Fingerprints)\n")
            f.write("-"*80 + "\n")
            f.write(f"Architecture: 3-layer MLP\n")
            f.write(f"Hidden dimensions: [512, 256, 128]\n")
            f.write(f"Fingerprint: ECFP4 (radius=2, 2048 bits)\n")
            f.write(f"Dropout: 0.3\n")
            f.write(f"Optimizer: Adam (lr=0.001, weight_decay=1e-5)\n\n")
            f.write(f"RESULTS:\n")
            f.write(f"  RMSE: {mlp_results['metrics']['rmse']:.4f}°C\n")
            f.write(f"  MAE: {mlp_results['metrics']['mae']:.4f}°C\n")
            f.write(f"  R²: {mlp_results['metrics']['r2']:.4f}\n")
            f.write(f"  Best val loss: {min(mlp_results['val_losses']):.4f}\n")
            f.write(f"  Epochs trained: {len(mlp_results['train_losses'])}\n\n")
            f.write(f"Model saved: trained_models/mlp_baseline.pth\n\n")

            if gcn_results is not None:
                f.write("SIMPLE GCN BASELINE (Graph Structure)\n")
                f.write("-"*80 + "\n")
                f.write(f"Architecture: 3-layer GCN\n")
                f.write(f"Hidden dimension: 128\n")
                f.write(f"Node features: 5 (atomic num, degree, charge, hybridization, aromatic)\n")
                f.write(f"Edge features: 4 (bond types)\n")
                f.write(f"Pooling: Global mean pooling\n")
                f.write(f"Dropout: 0.3\n")
                f.write(f"Optimizer: Adam (lr=0.001, weight_decay=1e-5)\n\n")
                f.write(f"RESULTS:\n")
                f.write(f"  RMSE: {gcn_results['metrics']['rmse']:.4f}°C\n")
                f.write(f"  MAE: {gcn_results['metrics']['mae']:.4f}°C\n")
                f.write(f"  R²: {gcn_results['metrics']['r2']:.4f}\n")
                f.write(f"  Best val loss: {min(gcn_results['val_losses']):.4f}\n")
                f.write(f"  Epochs trained: {len(gcn_results['train_losses'])}\n\n")
                f.write(f"Model saved: trained_models/gcn_baseline.pth\n\n")

                # Compare models
                f.write("MODEL COMPARISON\n")
                f.write("-"*80 + "\n")
                if mlp_results['metrics']['rmse'] < gcn_results['metrics']['rmse']:
                    f.write("Best model (by RMSE): MLP\n")
                else:
                    f.write("Best model (by RMSE): GCN\n")

                if mlp_results['metrics']['r2'] > gcn_results['metrics']['r2']:
                    f.write("Best model (by R²): MLP\n")
                else:
                    f.write("Best model (by R²): GCN\n")

        print(f"Results saved: {results_path}")


# =============================================================================
# PHASE 4: Diagnostic Analysis
# =============================================================================

class DiagnosticAnalyzer:
    """Analyze model performance and failure modes"""
    def __init__(self, results: Dict, figures_dir: str = "figures"):
        self.results = results
        self.figures_dir = Path(figures_dir)

    def plot_predictions(self):
        """Plot predicted vs actual for best model"""
        print("\n" + "="*80)
        print("PHASE 4: Diagnostic Analysis")
        print("="*80)

        # Determine best model
        mlp_r2 = self.results['mlp']['metrics']['r2']
        gcn_r2 = self.results['gcn']['metrics']['r2'] if self.results['gcn'] else 0

        if mlp_r2 > gcn_r2:
            best_model = 'mlp'
            best_name = 'MLP'
        else:
            best_model = 'gcn'
            best_name = 'GCN'

        print(f"\nBest model: {best_name}")

        predictions = self.results[best_model]['predictions']
        y_true = self.results[best_model]['y_true']

        # Plot predicted vs actual
        plt.figure(figsize=(10, 8))

        plt.scatter(y_true, predictions, alpha=0.5, s=20)

        # Perfect prediction line
        min_val = min(y_true.min(), predictions.min())
        max_val = max(y_true.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        plt.xlabel('Actual Tg (°C)', fontsize=12)
        plt.ylabel('Predicted Tg (°C)', fontsize=12)
        plt.title(f'{best_name}: Predicted vs Actual Tg', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add metrics text
        rmse = self.results[best_model]['metrics']['rmse']
        mae = self.results[best_model]['metrics']['mae']
        r2 = self.results[best_model]['metrics']['r2']

        textstr = f'RMSE: {rmse:.2f}°C\nMAE: {mae:.2f}°C\nR²: {r2:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'predicted_vs_actual.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.figures_dir / 'predicted_vs_actual.png'}")
        plt.close()

        return best_model

    def analyze_error_distribution(self, best_model: str):
        """Analyze error distribution across Tg ranges"""
        print("\nAnalyzing error distribution...")

        predictions = self.results[best_model]['predictions']
        y_true = self.results[best_model]['y_true']
        errors = predictions - y_true
        abs_errors = np.abs(errors)

        # Define Tg ranges
        min_tg = y_true.min()
        max_tg = y_true.max()
        bins = np.linspace(min_tg, max_tg, 11)  # 10 bins

        # Calculate error statistics per bin
        bin_centers = []
        bin_rmse = []
        bin_mae = []
        bin_count = []

        for i in range(len(bins) - 1):
            mask = (y_true >= bins[i]) & (y_true < bins[i+1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i+1]) / 2)
                bin_rmse.append(np.sqrt(np.mean(errors[mask]**2)))
                bin_mae.append(np.mean(abs_errors[mask]))
                bin_count.append(mask.sum())

        # Plot error vs Tg range
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(bin_centers, bin_rmse, 'o-', linewidth=2, markersize=8, label='RMSE')
        plt.plot(bin_centers, bin_mae, 's-', linewidth=2, markersize=8, label='MAE')
        plt.xlabel('Tg Range (°C)', fontsize=12)
        plt.ylabel('Error (°C)', fontsize=12)
        plt.title('Error vs Tg Range', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.bar(bin_centers, bin_count, width=np.diff(bins), alpha=0.7, edgecolor='black')
        plt.xlabel('Tg (°C)', fontsize=12)
        plt.ylabel('Sample Count', fontsize=12)
        plt.title('Test Set Distribution', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.figures_dir / 'error_distribution.png'}")
        plt.close()

        # Error histogram
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        plt.xlabel('Prediction Error (Predicted - Actual, °C)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Error Distribution', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        plt.axvline(mean_error, color='g', linestyle='-', linewidth=2, label=f'Mean: {mean_error:.2f}')

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'error_histogram.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.figures_dir / 'error_histogram.png'}")
        plt.close()

        return {
            'bin_centers': bin_centers,
            'bin_rmse': bin_rmse,
            'bin_mae': bin_mae,
            'bin_count': bin_count
        }

    def identify_failure_modes(self, best_model: str):
        """Identify failure modes and problematic samples"""
        print("\nIdentifying failure modes...")

        predictions = self.results[best_model]['predictions']
        y_true = self.results[best_model]['y_true']
        errors = predictions - y_true
        abs_errors = np.abs(errors)

        # Find worst predictions
        worst_idx = np.argsort(abs_errors)[-20:]  # Top 20 worst

        # Analyze failure patterns
        high_error_threshold = np.percentile(abs_errors, 90)  # Top 10%
        high_error_mask = abs_errors >= high_error_threshold

        print(f"\nHigh error threshold (90th percentile): {high_error_threshold:.2f}°C")
        print(f"Samples with high errors: {high_error_mask.sum()}")

        # Analyze characteristics of high-error samples
        high_errors_tg = y_true[high_error_mask]
        low_errors_tg = y_true[~high_error_mask]

        print(f"\nTg statistics:")
        print(f"  High-error samples: mean={np.mean(high_errors_tg):.2f}, std={np.std(high_errors_tg):.2f}")
        print(f"  Low-error samples: mean={np.mean(low_errors_tg):.2f}, std={np.std(low_errors_tg):.2f}")

        # Check if errors correlate with Tg magnitude
        correlation = np.corrcoef(abs_errors, y_true)[0, 1]
        print(f"\nCorrelation between |error| and Tg: {correlation:.4f}")

        return {
            'worst_idx': worst_idx,
            'high_error_threshold': high_error_threshold,
            'correlation': correlation
        }

    def save_diagnostic_report(self, error_stats, failure_stats):
        """Save diagnostic analysis report"""
        report_path = "diagnostic_analysis.txt"
        print(f"\nSaving diagnostic report: {report_path}")

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DIAGNOSTIC ANALYSIS - Polymer Tg Prediction\n")
            f.write("="*80 + "\n\n")

            f.write("1. BEST MODEL PERFORMANCE\n")
            f.write("-"*80 + "\n")

            # Determine best model
            mlp_r2 = self.results['mlp']['metrics']['r2']
            gcn_r2 = self.results['gcn']['metrics']['r2'] if self.results['gcn'] else 0

            if mlp_r2 > gcn_r2:
                best_model = 'mlp'
                best_name = 'MLP'
            else:
                best_model = 'gcn'
                best_name = 'GCN'

            metrics = self.results[best_model]['metrics']
            f.write(f"Best model: {best_name}\n")
            f.write(f"RMSE: {metrics['rmse']:.4f}°C\n")
            f.write(f"MAE: {metrics['mae']:.4f}°C\n")
            f.write(f"R²: {metrics['r2']:.4f}\n\n")

            f.write("2. ERROR DISTRIBUTION ACROSS TG RANGES\n")
            f.write("-"*80 + "\n")
            f.write("Tg Range (°C) | RMSE (°C) | MAE (°C) | Count\n")
            f.write("-"*80 + "\n")

            for i, center in enumerate(error_stats['bin_centers']):
                f.write(f"{center:8.1f}     | {error_stats['bin_rmse'][i]:6.2f}   | "
                       f"{error_stats['bin_mae'][i]:6.2f}  | {error_stats['bin_count'][i]:4d}\n")

            f.write("\n")

            f.write("3. FAILURE MODE ANALYSIS\n")
            f.write("-"*80 + "\n")
            f.write(f"High error threshold (90th percentile): {failure_stats['high_error_threshold']:.2f}°C\n")
            f.write(f"Correlation between |error| and Tg magnitude: {failure_stats['correlation']:.4f}\n\n")

            if abs(failure_stats['correlation']) > 0.3:
                if failure_stats['correlation'] > 0:
                    f.write("OBSERVATION: Errors increase with Tg magnitude.\n")
                    f.write("  The model struggles more with high-Tg polymers.\n")
                else:
                    f.write("OBSERVATION: Errors decrease with Tg magnitude.\n")
                    f.write("  The model performs better on high-Tg polymers.\n")
            else:
                f.write("OBSERVATION: Weak correlation between errors and Tg magnitude.\n")
                f.write("  Errors are relatively uniform across Tg ranges.\n")

            f.write("\n")

            f.write("4. RECOMMENDATIONS\n")
            f.write("-"*80 + "\n")
            f.write("Based on diagnostic analysis:\n\n")

            if failure_stats['correlation'] > 0.3:
                f.write("- Consider separate models or specialized features for high-Tg polymers\n")
                f.write("- Investigate if high-Tg polymers have distinct structural characteristics\n")

            if error_stats['bin_rmse']:
                max_rmse_idx = np.argmax(error_stats['bin_rmse'])
                min_rmse_idx = np.argmin(error_stats['bin_rmse'])
                f.write(f"- Model performs best in Tg range: {error_stats['bin_centers'][min_rmse_idx]:.1f}°C\n")
                f.write(f"- Model struggles most in Tg range: {error_stats['bin_centers'][max_rmse_idx]:.1f}°C\n")

            f.write("\n5. VISUALIZATIONS\n")
            f.write("-"*80 + "\n")
            f.write(f"- Predicted vs Actual: {self.figures_dir / 'predicted_vs_actual.png'}\n")
            f.write(f"- Error Distribution: {self.figures_dir / 'error_distribution.png'}\n")
            f.write(f"- Error Histogram: {self.figures_dir / 'error_histogram.png'}\n")

            f.write("\n6. NEXT STEPS FOR IMPROVEMENT\n")
            f.write("-"*80 + "\n")
            f.write("Data-centric approaches:\n")
            f.write("- Incorporate multimer representations (2-5 repeat units)\n")
            f.write("- Augment dataset with 3D conformer features\n")
            f.write("- Consider ensemble methods combining multiple representations\n")
            f.write("- Investigate data cleaning for high-error samples\n")
            f.write("- Implement uncertainty quantification\n")

        print(f"Report saved: {report_path}")

    def run_diagnostics(self):
        """Run complete diagnostic analysis"""
        best_model = self.plot_predictions()
        error_stats = self.analyze_error_distribution(best_model)
        failure_stats = self.identify_failure_modes(best_model)
        self.save_diagnostic_report(error_stats, failure_stats)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution pipeline"""
    print("\n" + "="*80)
    print("COMPREHENSIVE DATA-CENTRIC FOUNDATION EXPERIMENT")
    print("Polymer Glass Transition Temperature (Tg) Prediction")
    print("="*80)

    # Paths
    data_path = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"

    # Phase 1: Data Analysis
    analyzer = DataAnalyzer(data_path)
    df = analyzer.run_full_analysis()

    # Phase 2: Multimer Generation (sample of 100 due to computational cost)
    multimer_gen = MultimerGenerator(df)
    multimer_results = multimer_gen.generate_multimers_sample(sample_size=100)

    # Phase 3: Baseline Models
    trainer = ModelTrainer(df)
    splits = trainer.split_data()

    mlp_results = trainer.train_mlp(splits)
    gcn_results = trainer.train_gcn(splits)
    trainer.save_baseline_results(mlp_results, gcn_results)

    # Phase 4: Diagnostic Analysis
    diagnostic = DiagnosticAnalyzer({
        'mlp': mlp_results,
        'gcn': gcn_results
    })
    diagnostic.run_diagnostics()

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print("\nGenerated outputs:")
    print("  - data_analysis_report.txt")
    print("  - multimer_generation_log.txt")
    print("  - baseline_results.txt")
    print("  - diagnostic_analysis.txt")
    print("  - figures/*.png")
    print("  - trained_models/*.pth")
    print("\n")

if __name__ == "__main__":
    main()
