"""
Analyze correlation between rotatable bond fraction and Tg
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

def compute_rotatable_bond_fraction(smiles):
    """
    Compute rotatable bond fraction from p-SMILES.
    Handle polymer notation with * (wildcard atom).
    """
    try:
        # Use RDKit's sanitize option that allows wildcard atoms
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None

        # Sanitize but allow wildcard atoms (*)
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                        Chem.SanitizeFlags.SANITIZE_ADJUSTHS)

        # Get number of rotatable bonds
        num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)

        # Get total number of bonds
        total_bonds = mol.GetNumBonds()

        if total_bonds == 0:
            return 0.0

        return num_rotatable / total_bonds
    except Exception as e:
        # Silent fail for invalid SMILES
        return None

def main():
    # Load data
    df = pd.read_csv('/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv')
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Tg range: [{df['Tg'].min()}, {df['Tg'].max()}]")
    print(f"Tg mean: {df['Tg'].mean():.2f}, std: {df['Tg'].std():.2f}")

    # Compute rigidity
    print("\nComputing rotatable bond fractions...")
    df['rigidity'] = df['Smiles'].apply(compute_rotatable_bond_fraction)

    # Remove invalid entries
    valid_mask = df['rigidity'].notna()
    df_valid = df[valid_mask]
    print(f"Valid samples: {len(df_valid)}")

    # Compute correlations
    rigidity = df_valid['rigidity'].values
    tg = df_valid['Tg'].values

    pearson_r, pearson_p = pearsonr(rigidity, tg)
    spearman_r, spearman_p = spearmanr(rigidity, tg)

    print(f"\n=== Correlation Analysis ===")
    print(f"Pearson correlation: r = {pearson_r:.4f}, p = {pearson_p:.4e}")
    print(f"Spearman correlation: rho = {spearman_r:.4f}, p = {spearman_p:.4e}")

    # Rigidity statistics
    print(f"\n=== Rigidity Statistics ===")
    print(f"Rigidity range: [{rigidity.min():.4f}, {rigidity.max():.4f}]")
    print(f"Rigidity mean: {rigidity.mean():.4f}, std: {rigidity.std():.4f}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    ax = axes[0]
    ax.scatter(rigidity, tg, alpha=0.3, s=5)
    ax.set_xlabel('Rotatable Bond Fraction')
    ax.set_ylabel('Tg (°C)')
    ax.set_title(f'Rotatable Bond Fraction vs Tg\nPearson r={pearson_r:.3f}')

    # Add regression line
    z = np.polyfit(rigidity, tg, 1)
    p = np.poly1d(z)
    x_line = np.linspace(rigidity.min(), rigidity.max(), 100)
    ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f'y={z[0]:.1f}x+{z[1]:.1f}')
    ax.legend()

    # Histogram of rigidity
    ax = axes[1]
    ax.hist(rigidity, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Rotatable Bond Fraction')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Rotatable Bond Fraction')

    plt.tight_layout()
    plt.savefig('/home/home_public/aisci/AIscientist/src/1_day2_reperioGT/analysis/rigidity_correlation.png', dpi=150)
    print(f"\nSaved figure to analysis/rigidity_correlation.png")

    # Save processed data
    df_valid.to_csv('/home/home_public/aisci/AIscientist/src/1_day2_reperioGT/processed_data.csv', index=False)
    print(f"Saved processed data to processed_data.csv")

    return pearson_r, spearman_r

if __name__ == "__main__":
    import os
    os.makedirs('/home/home_public/aisci/AIscientist/src/1_day2_reperioGT/analysis', exist_ok=True)
    main()
