"""
BlochSpec Step 1: Pre-compute Bloch spectral features for all polymers.

For each polymer p-SMILES:
1. Construct chemistry-weighted Laplacian
2. Compute Bloch band structure
3. Extract compact spectral summaries (12 features)

Also computes unweighted Laplacian features for ablation (g).
"""

import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
from scipy.stats import skew, kurtosis
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Pauling electronegativity values
ELECTRONEGATIVITY = {
    'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96,
    'I': 2.66, 'B': 2.04, 'Se': 2.55, 'Te': 2.10, 'Sn': 1.96,
    'Ge': 2.01, 'As': 2.18, 'Sb': 2.05, 'Bi': 2.02,
}

# Bond type weights
BOND_TYPE_WEIGHT = {
    Chem.BondType.SINGLE: 1.0,
    Chem.BondType.DOUBLE: 2.0,
    Chem.BondType.AROMATIC: 1.5,
    Chem.BondType.TRIPLE: 3.0,
}

# Default electronegativity for unknown elements
DEFAULT_EN = 2.0

# k-points: 24 points from 0 to 2*pi
K_POINTS = np.linspace(0, 2 * np.pi, 24, endpoint=False)

# Feature names for reference
FEATURE_NAMES = [
    'spectral_gap',        # 1: chain connectivity
    'spectral_radius',     # 2: overall coupling strength
    'spectral_centroid',   # 3: average coupling
    'spectral_variance',   # 4: coupling diversity
    'low_freq_density',    # 5: chain flexibility
    'band_width_max',      # 6: strongest inter-RU coupling
    'band_width_mean',     # 7: average inter-RU coupling
    'band_flatness',       # 8: mode localization
    'spectral_gap_kpi',    # 9: band gap analog
    'density_at_Fermi',    # 10: states near median
    'spectral_skew',       # 11: asymmetry
    'spectral_kurtosis',   # 12: peakedness
]


def get_electronegativity(symbol):
    """Get electronegativity for an element symbol."""
    return ELECTRONEGATIVITY.get(symbol, DEFAULT_EN)


def get_bond_weight(bond):
    """Get the weight for a bond based on bond type and electronegativity difference."""
    bond_type = bond.GetBondType()
    bt_weight = BOND_TYPE_WEIGHT.get(bond_type, 1.0)

    # Electronegativity difference between bonded atoms
    a1 = bond.GetBeginAtom()
    a2 = bond.GetEndAtom()
    en1 = get_electronegativity(a1.GetSymbol())
    en2 = get_electronegativity(a2.GetSymbol())
    delta_en = abs(en1 - en2)

    w = bt_weight * (1 + delta_en)
    return w


def get_bond_weight_unweighted(bond):
    """Unweighted version: all bonds have weight 1.0."""
    return 1.0


def parse_monomer(p_smiles):
    """
    Parse a p-SMILES string to extract monomer unit information.

    Returns:
        success: bool
        n_atoms: number of atoms in monomer (after removing star atoms)
        atoms: list of atom symbols
        bonds: list of (i, j, bond_type) internal bonds
        boundary_atoms: list of atom indices connected to star atoms
        inter_ru_bonds: list of (boundary_i, boundary_j, bond_type) for inter-RU connections
        If fails, returns (False, None, None, None, None, None)
    """
    try:
        mol = Chem.MolFromSmiles(p_smiles)
        if mol is None:
            return False, None, None, None, None, None

        # Find star atoms (atomic num 0)
        star_atoms = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                star_atoms.append(atom.GetIdx())

        if len(star_atoms) == 0:
            # Non-periodic polymer: no coupling matrix
            # Just use standard Laplacian
            n_atoms = mol.GetNumAtoms()
            atoms = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n_atoms)]
            bonds = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bonds.append((i, j, bond.GetBondType()))
            return True, n_atoms, atoms, bonds, [], []

        if len(star_atoms) != 2:
            return False, None, None, None, None, None

        # Get boundary atoms (neighbors of star atoms)
        star_to_neighbor = {}
        for star_idx in star_atoms:
            star_atom = mol.GetAtomWithIdx(star_idx)
            for nbr in star_atom.GetNeighbors():
                if nbr.GetAtomicNum() != 0:
                    star_to_neighbor[star_idx] = nbr.GetIdx()
                    break

        if len(star_to_neighbor) != 2:
            return False, None, None, None, None, None

        # Get bond types from star atoms to boundary atoms
        star_bond_types = {}
        for star_idx in star_atoms:
            if star_idx in star_to_neighbor:
                nbr_idx = star_to_neighbor[star_idx]
                bond = mol.GetBondBetweenAtoms(star_idx, nbr_idx)
                star_bond_types[star_idx] = bond.GetBondType()

        # Remove star atoms - need to track index remapping
        atoms_removed = sorted(star_atoms)

        def update_idx(idx):
            new_idx = idx
            for removed_idx in atoms_removed:
                if idx > removed_idx:
                    new_idx -= 1
            return new_idx

        boundary_atoms = []
        inter_ru_bonds = []
        for star_idx in star_atoms:
            if star_idx in star_to_neighbor:
                orig_nbr = star_to_neighbor[star_idx]
                new_nbr = update_idx(orig_nbr)
                boundary_atoms.append(new_nbr)
                inter_ru_bonds.append((new_nbr, star_bond_types[star_idx]))

        # Collect atoms and bonds after removing star atoms
        remaining_atoms = []
        for i in range(mol.GetNumAtoms()):
            if i not in star_atoms:
                new_i = update_idx(i)
                remaining_atoms.append((new_i, mol.GetAtomWithIdx(i).GetSymbol()))

        remaining_atoms.sort(key=lambda x: x[0])
        atoms = [sym for _, sym in remaining_atoms]
        n_atoms = len(atoms)

        # Collect internal bonds (not involving star atoms)
        bonds = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if i in star_atoms or j in star_atoms:
                continue
            new_i = update_idx(i)
            new_j = update_idx(j)
            bonds.append((new_i, new_j, bond.GetBondType()))

        return True, n_atoms, atoms, bonds, boundary_atoms, inter_ru_bonds

    except Exception as e:
        logger.warning(f"Failed to parse p-SMILES '{p_smiles}': {e}")
        return False, None, None, None, None, None


def construct_laplacian(n_atoms, bonds, atoms, weight_func=get_bond_weight):
    """
    Construct the chemistry-weighted Laplacian for a monomer unit cell.

    Args:
        n_atoms: number of atoms
        bonds: list of (i, j, bond_type)
        atoms: list of atom symbols
        weight_func: function to compute bond weight

    Returns:
        L_UC: unit cell Laplacian (n_atoms x n_atoms)
    """
    W = np.zeros((n_atoms, n_atoms))

    for i, j, bond_type in bonds:
        # Create a minimal bond representation for weight computation
        # We need electronegativity-based weights
        en_i = get_electronegativity(atoms[i])
        en_j = get_electronegativity(atoms[j])
        delta_en = abs(en_i - en_j)

        bt_weight = {Chem.BondType.SINGLE: 1.0, Chem.BondType.DOUBLE: 2.0,
                     Chem.BondType.AROMATIC: 1.5, Chem.BondType.TRIPLE: 3.0}

        if weight_func == get_bond_weight_unweighted:
            w = 1.0
        else:
            w = bt_weight.get(bond_type, 1.0) * (1 + delta_en)

        W[i, j] = w
        W[j, i] = w

    D_w = np.diag(W.sum(axis=1))
    L_UC = D_w - W

    return L_UC


def construct_coupling_matrix(n_atoms, boundary_atoms, inter_ru_bonds, weight_func=get_bond_weight):
    """
    Construct the coupling matrix between adjacent repeat units.

    The coupling matrix encodes how boundary atoms connect to the next RU.
    L_coupling[i,j] = -coupling_weight for each boundary atom pair.

    For a polymer *A...B*, the inter-RU bond connects B of unit n to A of unit n+1.
    So if boundary_atoms = [b1, b2] and b1 connects through the first * and b2 through the second *,
    then in the coupling matrix:
    - L_coupling[b2, b1] = -w  (b2 of current RU connects to b1 of next RU)

    Args:
        n_atoms: number of atoms in monomer
        boundary_atoms: list of boundary atom indices [b1, b2]
        inter_ru_bonds: list of (boundary_idx, bond_type)
        weight_func: function to compute bond weight

    Returns:
        L_coupling: coupling matrix (n_atoms x n_atoms)
    """
    L_coupling = np.zeros((n_atoms, n_atoms))

    if len(boundary_atoms) < 2:
        return L_coupling

    # boundary_atoms[0] is connected to first star atom (head)
    # boundary_atoms[1] is connected to second star atom (tail)
    # In a linear polymer: ...-tail_unit_n-head-unit_{n+1}-tail-...
    # So boundary_atoms[1] (tail of current RU) connects to boundary_atoms[0] (head of next RU)
    # L_coupling[boundary_atoms[1], boundary_atoms[0]] = -w

    bt_weight = {Chem.BondType.SINGLE: 1.0, Chem.BondType.DOUBLE: 2.0,
                 Chem.BondType.AROMATIC: 1.5, Chem.BondType.TRIPLE: 3.0}

    if len(inter_ru_bonds) == 2:
        # inter_ru_bonds[0]: (boundary_atoms[0], bond_type) - head connection
        # inter_ru_bonds[1]: (boundary_atoms[1], bond_type) - tail connection
        # The coupling is tail -> head of next unit
        tail_idx = boundary_atoms[1]
        head_idx = boundary_atoms[0]

        # Use the bond type of the tail connection (typically single bond for backbone)
        tail_bond_type = inter_ru_bonds[1][1]
        if weight_func == get_bond_weight_unweighted:
            w = 1.0
        else:
            w = bt_weight.get(tail_bond_type, 1.0)

        L_coupling[tail_idx, head_idx] = -w

    return L_coupling


def compute_bloch_bands(L_UC, L_coupling, k_points=K_POINTS):
    """
    Compute Bloch band structure.

    For each k: H(k) = L_UC + exp(ik) * L_coupling + exp(-ik) * L_coupling^T

    Args:
        L_UC: unit cell Laplacian (N x N)
        L_coupling: coupling matrix (N x N)
        k_points: array of k-points

    Returns:
        eigenvalues: array of shape (N, len(k_points))
    """
    N = L_UC.shape[0]
    n_k = len(k_points)
    eigenvalues = np.zeros((N, n_k))

    for ki, k in enumerate(k_points):
        # Bloch Hamiltonian
        phase = np.exp(1j * k)
        H_k = L_UC + phase * L_coupling + np.conj(phase) * L_coupling.T

        # H(k) should be Hermitian for real eigenvalues
        # Force Hermitian to avoid numerical issues
        H_k = (H_k + H_k.conj().T) / 2

        eigs = np.linalg.eigvalsh(H_k)
        eigenvalues[:, ki] = eigs

    return eigenvalues


def extract_spectral_features(eigenvalues):
    """
    Extract 12 compact spectral summary features from the N x 24 eigenvalue matrix.

    Args:
        eigenvalues: array of shape (N, 24) - Bloch band eigenvalues

    Returns:
        features: array of 12 features, or None if computation fails
    """
    try:
        N, n_k = eigenvalues.shape
        all_eigs = eigenvalues.flatten()

        # Sort eigenvalues at each k-point (should already be sorted from eigvalsh)
        eigs_sorted = np.sort(eigenvalues, axis=0)

        # 1. spectral_gap = λ_2(0) - λ_1(0)
        if N >= 2:
            spectral_gap = eigs_sorted[1, 0] - eigs_sorted[0, 0]
        else:
            spectral_gap = 0.0

        # 2. spectral_radius = max over all eigenvalues
        spectral_radius = np.max(all_eigs)

        # 3. spectral_centroid = mean of all eigenvalues
        spectral_centroid = np.mean(all_eigs)

        # 4. spectral_variance = std of all eigenvalues
        spectral_variance = np.std(all_eigs)

        # 5. low_freq_density = fraction of eigenvalues < 0.5
        low_freq_density = np.sum(all_eigs < 0.5) / len(all_eigs)

        # 6. band_width_max = max over n of (max_k λ_n(k) - min_k λ_n(k))
        band_widths = np.max(eigenvalues, axis=1) - np.min(eigenvalues, axis=1)
        band_width_max = np.max(band_widths)

        # 7. band_width_mean = mean of band widths
        band_width_mean = np.mean(band_widths)

        # 8. band_flatness = mean over n of (1 - band_width_n / centroid_n)
        # Protect against division by zero
        band_centroids = np.mean(eigenvalues, axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flatness_terms = 1 - band_widths / (band_centroids + 1e-10)
        band_flatness = np.mean(flatness_terms)

        # 9. spectral_gap_kpi = minimum band gap across all k-points
        # Sort eigenvalues at each k-point and find minimum gap between consecutive bands
        min_gap = float('inf')
        for ki in range(n_k):
            sorted_eigs = np.sort(eigenvalues[:, ki])
            for n in range(len(sorted_eigs) - 1):
                gap = sorted_eigs[n + 1] - sorted_eigs[n]
                if gap < min_gap:
                    min_gap = gap
        spectral_gap_kpi = min_gap

        # 10. density_at_Fermi = fraction near median
        eig_median = np.median(all_eigs)
        density_at_Fermi = np.sum(np.abs(all_eigs - eig_median) < 0.3) / len(all_eigs)

        # 11. spectral_skew
        spectral_skew = skew(all_eigs)

        # 12. spectral_kurtosis
        spectral_kurt = kurtosis(all_eigs)

        features = np.array([
            spectral_gap,
            spectral_radius,
            spectral_centroid,
            spectral_variance,
            low_freq_density,
            band_width_max,
            band_width_mean,
            band_flatness,
            spectral_gap_kpi,
            density_at_Fermi,
            spectral_skew,
            spectral_kurt,
        ], dtype=np.float32)

        # Sanity check for NaN/Inf
        if not np.all(np.isfinite(features)):
            return None

        return features

    except Exception as e:
        logger.warning(f"Failed to extract spectral features: {e}")
        return None


def compute_features_for_polymer(p_smiles, weighted=True):
    """
    Compute Bloch spectral features for a single polymer.

    Args:
        p_smiles: polymer SMILES string
        weighted: if True, use chemistry-weighted Laplacian; if False, unweighted

    Returns:
        features: array of 12 features, or None if computation fails
    """
    try:
        result = parse_monomer(p_smiles)
        success, n_atoms, atoms, bonds, boundary_atoms, inter_ru_bonds = result

        if not success:
            return None

        if n_atoms == 0:
            return None

        # Choose weight function
        wf = get_bond_weight if weighted else get_bond_weight_unweighted

        # Construct Laplacian
        L_UC = construct_laplacian(n_atoms, bonds, atoms, weight_func=wf)

        # Construct coupling matrix
        L_coupling = construct_coupling_matrix(n_atoms, boundary_atoms, inter_ru_bonds, weight_func=wf)

        # Compute Bloch bands
        eigenvalues = compute_bloch_bands(L_UC, L_coupling)

        # Extract features
        features = extract_spectral_features(eigenvalues)

        return features

    except Exception as e:
        logger.warning(f"Failed to compute features for '{p_smiles}': {e}")
        return None


def precompute_all_features(csv_path, output_dir):
    """
    Pre-compute Bloch spectral features for all polymers in the dataset.

    Saves:
        spectral_features.npy - weighted Bloch features (n_polymers x 12)
        spectral_features_unweighted.npy - unweighted Bloch features (n_polymers x 12)
        feature_indices.npy - indices of successfully processed polymers
        feature_names.txt - feature names
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    smiles_list = df['Smiles'].tolist()
    tg_values = df['Tg'].tolist()
    n_total = len(smiles_list)

    logger.info(f"Computing spectral features for {n_total} polymers...")

    weighted_features = []
    unweighted_features = []
    valid_indices = []
    valid_tgs = []
    failed_count = 0

    for i, (smi, tg) in enumerate(zip(smiles_list, tg_values)):
        if (i + 1) % 500 == 0:
            logger.info(f"  Processed {i+1}/{n_total} polymers...")

        # Weighted features
        wf = compute_features_for_polymer(smi, weighted=True)
        # Unweighted features
        uf = compute_features_for_polymer(smi, weighted=False)

        if wf is not None and uf is not None:
            weighted_features.append(wf)
            unweighted_features.append(uf)
            valid_indices.append(i)
            valid_tgs.append(tg)
        else:
            failed_count += 1

    weighted_features = np.array(weighted_features)
    unweighted_features = np.array(unweighted_features)
    valid_indices = np.array(valid_indices)
    valid_tgs = np.array(valid_tgs)

    logger.info(f"Successfully computed features for {len(valid_indices)}/{n_total} polymers")
    logger.info(f"Failed: {failed_count}")

    # Save
    np.save(os.path.join(output_dir, 'spectral_features.npy'), weighted_features)
    np.save(os.path.join(output_dir, 'spectral_features_unweighted.npy'), unweighted_features)
    np.save(os.path.join(output_dir, 'feature_indices.npy'), valid_indices)
    np.save(os.path.join(output_dir, 'feature_tgs.npy'), valid_tgs)

    # Save feature names
    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        for name in FEATURE_NAMES:
            f.write(name + '\n')

    logger.info(f"Features saved to {output_dir}")
    logger.info(f"  Weighted features shape: {weighted_features.shape}")
    logger.info(f"  Unweighted features shape: {unweighted_features.shape}")

    return weighted_features, unweighted_features, valid_indices, valid_tgs


def compute_correlations(features, tgs, feature_names, output_path):
    """
    Compute Pearson correlations between each spectral feature and Tg.
    """
    from scipy.stats import pearsonr

    logger.info("\n=== Correlation Analysis ===")
    results = []

    for i, name in enumerate(feature_names):
        feat_vals = features[:, i]
        r, p = pearsonr(feat_vals, tgs)
        results.append((name, r, p))
        logger.info(f"  {name:25s}: r = {r:+.4f}, p = {p:.2e}")

    with open(output_path, 'w') as f:
        f.write("Feature Correlations with Tg\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Feature':25s} {'Pearson r':>12s} {'p-value':>15s} {'|r|>0.2':>10s}\n")
        f.write("-" * 60 + "\n")
        for name, r, p in results:
            flag = "***" if abs(r) > 0.2 else ""
            f.write(f"{name:25s} {r:+12.4f} {p:15.2e} {flag:>10s}\n")

    # Count features with meaningful correlation
    n_meaningful = sum(1 for _, r, _ in results if abs(r) > 0.2)
    logger.info(f"\nFeatures with |r| > 0.2: {n_meaningful}/{len(feature_names)}")

    return results


def generate_random_features(features, seed=42):
    """
    Generate random features with same mean/std as real features for ablation (f).
    """
    rng = np.random.RandomState(seed)
    n_samples, n_features = features.shape
    random_feats = np.zeros_like(features)

    for j in range(n_features):
        col_mean = np.mean(features[:, j])
        col_std = np.std(features[:, j])
        random_feats[:, j] = rng.normal(col_mean, col_std, n_samples)

    return random_feats.astype(np.float32)


if __name__ == "__main__":
    CSV_PATH = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
    OUTPUT_DIR = "/home/home_public/aisci/AIscientist/src/4_4_blochspec/spectral_data"

    # Pre-compute features
    weighted, unweighted, indices, tgs = precompute_all_features(CSV_PATH, OUTPUT_DIR)

    # Correlation analysis
    corr_path = os.path.join(OUTPUT_DIR, "correlation_analysis.txt")
    compute_correlations(weighted, tgs, FEATURE_NAMES, corr_path)

    # Generate and save random features for ablation
    random_feats = generate_random_features(weighted, seed=42)
    np.save(os.path.join(OUTPUT_DIR, 'spectral_features_random.npy'), random_feats)
    logger.info(f"Random features saved: {random_feats.shape}")

    # Print summary statistics
    logger.info("\n=== Feature Summary Statistics ===")
    for i, name in enumerate(FEATURE_NAMES):
        vals = weighted[:, i]
        logger.info(f"  {name:25s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
                     f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")

    logger.info("\nDone!")
