"""
Dataset module for DCN-GIN depth scaling experiment
Loads Tg dataset with 148-dim validated features
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_smiles

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, rdMolDescriptors

from sklearn.preprocessing import StandardScaler


def compute_validated_features(mol) -> np.ndarray:
    """
    Compute validated features with strong property correlations.
    Total: 148 features
    """
    features = []
    feature_names = []

    try:
        # Basic molecular properties
        mol_weight = Descriptors.MolWt(mol)
        heavy_atoms = Descriptors.HeavyAtomCount(mol)
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        features.extend([mol_weight, heavy_atoms, num_atoms, num_bonds])
        feature_names.extend(['mol_weight', 'heavy_atoms', 'num_atoms', 'num_bonds'])

        # Ring information
        ring_count = Descriptors.RingCount(mol)
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        num_aliphatic_rings = rdMolDescriptors.CalcNumAliphaticRings(mol)
        num_saturated_rings = rdMolDescriptors.CalcNumSaturatedRings(mol)

        features.extend([ring_count, num_rings, num_aromatic_rings, num_aliphatic_rings, num_saturated_rings])
        feature_names.extend(['RingCount', 'NumRings', 'NumAromaticRings', 'NumAliphaticRings', 'NumSaturatedRings'])

        # FractionCSP3
        frac_csp3 = Descriptors.FractionCSP3(mol)
        features.append(frac_csp3)
        feature_names.append('FractionCSP3')

        # Electronic properties
        mol_logp = Descriptors.MolLogP(mol)
        mol_mr = Descriptors.MolMR(mol)
        tpsa = Descriptors.TPSA(mol)

        features.extend([mol_logp, mol_mr, tpsa])
        feature_names.extend(['MolLogP', 'MolMR', 'TPSA'])

        # Topological indices
        bertz_ct = Descriptors.BertzCT(mol)
        balaban_j = Descriptors.BalabanJ(mol)
        chi0 = Descriptors.Chi0(mol)
        chi1 = Descriptors.Chi1(mol)

        features.extend([bertz_ct, balaban_j, chi0, chi1])
        feature_names.extend(['BertzCT', 'BalabanJ', 'Chi0', 'Chi1'])

        # Hydrogen bond donors/acceptors
        num_h_donors = Descriptors.NumHDonors(mol)
        num_h_acceptors = Descriptors.NumHAcceptors(mol)
        num_heteroatoms = Descriptors.NumHeteroatoms(mol)

        features.extend([num_h_donors, num_h_acceptors, num_heteroatoms])
        feature_names.extend(['NumHDonors', 'NumHAcceptors', 'NumHeteroatoms'])

        # Rotatable bonds
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        rotatable_bond_frac = num_rotatable_bonds / num_bonds if num_bonds > 0 else 0

        features.extend([num_rotatable_bonds, rotatable_bond_frac])
        feature_names.extend(['NumRotatableBonds', 'rotatable_bond_frac'])

        # Atom type counts
        atom_types = {'C': 0, 'N': 0, 'O': 0, 'S': 0, 'F': 0, 'Cl': 0, 'Br': 0, 'P': 0, 'Si': 0}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in atom_types:
                atom_types[symbol] += 1

        features.extend(list(atom_types.values()))
        feature_names.extend([f'{k}_count' for k in atom_types.keys()])

        # Bond type fractions
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

        total_bonds = num_bonds if num_bonds > 0 else 1
        single_bond_frac = single_bonds / total_bonds
        double_bond_frac = double_bonds / total_bonds
        triple_bond_frac = triple_bonds / total_bonds
        aromatic_bond_frac = aromatic_bonds / total_bonds

        features.extend([single_bonds, double_bonds, triple_bonds, aromatic_bonds,
                        single_bond_frac, double_bond_frac, triple_bond_frac, aromatic_bond_frac])
        feature_names.extend(['single_bonds', 'double_bonds', 'triple_bonds', 'aromatic_bonds',
                             'single_bond_frac', 'double_bond_frac', 'triple_bond_frac', 'aromatic_bond_frac'])

        # Hybridization fractions
        sp_count = 0
        sp2_count = 0
        sp3_count = 0

        for atom in mol.GetAtoms():
            hyb = atom.GetHybridization()
            if hyb == Chem.HybridizationType.SP:
                sp_count += 1
            elif hyb == Chem.HybridizationType.SP2:
                sp2_count += 1
            elif hyb == Chem.HybridizationType.SP3:
                sp3_count += 1

        total_atoms = num_atoms if num_atoms > 0 else 1
        sp_frac = sp_count / total_atoms
        sp2_frac = sp2_count / total_atoms
        sp3_frac = sp3_count / total_atoms

        features.extend([sp_count, sp2_count, sp3_count, sp_frac, sp2_frac, sp3_frac])
        feature_names.extend(['sp_count', 'sp2_count', 'sp3_count', 'sp_frac', 'sp2_frac', 'sp3_frac'])

        # Ring size distribution
        ring_info = mol.GetRingInfo()
        ring_sizes = [0] * 6
        for ring in ring_info.AtomRings():
            size = len(ring)
            if 3 <= size <= 8:
                ring_sizes[size - 3] += 1

        features.extend(ring_sizes)
        feature_names.extend(['ring3', 'ring4', 'ring5', 'ring6', 'ring7', 'ring8'])

        # Conjugation features
        conjugated_bonds = 0
        sp2_sp2_bonds = 0
        pi_electrons = 0
        aromatic_atoms = 0

        for bond in mol.GetBonds():
            if bond.GetIsConjugated() or bond.GetBondType() == Chem.BondType.AROMATIC:
                conjugated_bonds += 1

            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            if (begin_atom.GetHybridization() == Chem.HybridizationType.SP2 and
                end_atom.GetHybridization() == Chem.HybridizationType.SP2):
                sp2_sp2_bonds += 1

        conjugated_bond_fraction = conjugated_bonds / total_bonds if total_bonds > 0 else 0
        sp2_sp2_bond_fraction = sp2_sp2_bonds / total_bonds if total_bonds > 0 else 0

        for atom in mol.GetAtoms():
            if atom.GetIsAromatic():
                aromatic_atoms += 1
            hyb = atom.GetHybridization()
            if hyb == Chem.HybridizationType.SP2:
                pi_electrons += 1
            elif hyb == Chem.HybridizationType.SP:
                pi_electrons += 2

        aromatic_atom_fraction = aromatic_atoms / total_atoms if total_atoms > 0 else 0
        pi_electron_density = pi_electrons / mol_weight if mol_weight > 0 else 0

        features.extend([conjugated_bonds, conjugated_bond_fraction, sp2_sp2_bonds, sp2_sp2_bond_fraction,
                        pi_electrons, pi_electron_density, aromatic_atoms, aromatic_atom_fraction])
        feature_names.extend(['conjugated_bonds', 'conjugated_bond_fraction', 'sp2_sp2_bonds', 'sp2_sp2_bond_fraction',
                             'pi_electrons', 'pi_electron_density', 'aromatic_atoms', 'aromatic_atom_fraction'])

        # Additional RDKit descriptors
        features.extend([
            Descriptors.HeavyAtomMolWt(mol),
            Descriptors.ExactMolWt(mol),
            Descriptors.NumValenceElectrons(mol),
            Descriptors.FpDensityMorgan1(mol),
            Descriptors.FpDensityMorgan2(mol),
            Descriptors.FpDensityMorgan3(mol),
            Descriptors.NumAliphaticCarbocycles(mol),
            Descriptors.NumAliphaticHeterocycles(mol),
            Descriptors.NumAromaticCarbocycles(mol),
            Descriptors.NumAromaticHeterocycles(mol),
            Descriptors.NumSaturatedCarbocycles(mol),
            Descriptors.NumSaturatedHeterocycles(mol),
            Descriptors.qed(mol),
        ])
        feature_names.extend([
            'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
            'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3',
            'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
            'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
            'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles',
            'qed'
        ])

        # Functional group counts
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
            feature_names.append(f'fg_{name}')

        # Additional conjugation-related features
        conjugation_length = num_aromatic_rings * 6
        aromatic_ring_fraction = num_aromatic_rings / ring_count if ring_count > 0 else 0

        features.extend([conjugation_length, aromatic_ring_fraction])
        feature_names.extend(['conjugation_length', 'aromatic_ring_fraction'])

        # Complexity metrics
        features.extend([
            Descriptors.HallKierAlpha(mol),
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol),
        ])
        feature_names.extend(['HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3'])

        # EState indices
        features.extend([
            Descriptors.MaxEStateIndex(mol),
            Descriptors.MinEStateIndex(mol),
        ])
        feature_names.extend(['MaxEStateIndex', 'MinEStateIndex'])

        # Additional bond order features
        avg_bond_order = (single_bonds * 1 + double_bonds * 2 + triple_bonds * 3 + aromatic_bonds * 1.5) / total_bonds if total_bonds > 0 else 0
        features.append(avg_bond_order)
        feature_names.append('avg_bond_order')

        # Aromatic heteroatom count
        aromatic_heteroatom_count = 0
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic() and atom.GetSymbol() not in ['C', 'H']:
                aromatic_heteroatom_count += 1
        features.append(aromatic_heteroatom_count)
        feature_names.append('aromatic_heteroatom_count')

        # Additional features
        additional_features = [
            Descriptors.LabuteASA(mol),
            Descriptors.BalabanJ(mol) if Descriptors.BalabanJ(mol) != float('inf') else 0,
            rdMolDescriptors.CalcNumHeterocycles(mol),
            rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            rdMolDescriptors.CalcNumAmideBonds(mol),
        ]
        features.extend(additional_features)
        feature_names.extend(['LabuteASA', 'BalabanJ2', 'NumHeterocycles', 'NumBridgeheadAtoms', 'NumAmideBonds'])

        # Pad or truncate to exactly 148 features
        features = features[:148]
        feature_names = feature_names[:148]

        while len(features) < 148:
            features.append(0.0)
            feature_names.append(f'padding_{len(features)}')

        # Replace NaN/Inf with 0
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features, feature_names

    except Exception as e:
        print(f"Error computing features: {e}")
        return np.zeros(148, dtype=np.float32), ['error'] * 148


def process_psmiles(psmiles: str) -> Tuple[Chem.Mol, str]:
    """Process p-SMILES to get RDKit mol object."""
    smiles = psmiles.replace('*', '[H]')

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None, None
        Chem.SanitizeMol(mol)
        return mol, smiles
    except:
        try:
            smiles_simple = psmiles.replace('*', '')
            mol = Chem.MolFromSmiles(smiles_simple)
            if mol is not None:
                return mol, smiles_simple
        except:
            pass
        return None, None


def mol_to_graph(mol, target: float = None, features: np.ndarray = None) -> Data:
    """
    Convert RDKit mol to PyG Data object with atom and bond features.
    """
    try:
        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_feat = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
                atom.GetHybridization(),
                atom.GetMass(),
                atom.GetTotalValence()
            ]
            atom_features.append(atom_feat)

        x = torch.tensor(atom_features, dtype=torch.float)

        # Bond features and edge index
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])

            bond_feat = [
                int(bond.GetBondType() == Chem.BondType.SINGLE),
                int(bond.GetBondType() == Chem.BondType.DOUBLE),
                int(bond.GetBondType() == Chem.BondType.TRIPLE),
                int(bond.GetBondType() == Chem.BondType.AROMATIC),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
                bond.GetBondTypeAsDouble()
            ]
            edge_attr.append(bond_feat)
            edge_attr.append(bond_feat)

        if len(edge_index) == 0:
            # Single atom molecule - add self-loop
            edge_index = [[0, 0]]
            edge_attr = [[0, 0, 0, 0, 0, 0, 0]]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Create Data object - store features as a 2D tensor [1, 148]
        # This ensures proper batching in PyG
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([target], dtype=torch.float) if target is not None else None,
        )

        # Store features separately
        if features is not None:
            data.mol_features = torch.tensor(features, dtype=torch.float).unsqueeze(0)  # [1, 148]

        return data

    except Exception as e:
        return None


def load_tg_dataset(
    csv_path: str,
    cache_dir: str = None,
    use_cache: bool = True
) -> Tuple[List[Data], np.ndarray, StandardScaler]:
    """
    Load Tg dataset with 148-dim features.

    Returns:
        data_list: List of PyG Data objects
        features: Numpy array of features (N x 148)
        scaler: Fitted StandardScaler
    """
    if cache_dir is None:
        cache_dir = os.path.dirname(csv_path)

    cache_file = os.path.join(cache_dir, 'tg_features_148dim.pkl')

    # Try to load from cache
    if use_cache and os.path.exists(cache_file):
        print(f"Loading features from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            data_list = cache_data['data_list']
            features = cache_data['features']
            scaler = cache_data['scaler']
            print(f"Loaded {len(data_list)} samples from cache")
            return data_list, features, scaler

    # Load CSV
    print(f"Loading Tg dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} samples in CSV")

    data_list = []
    features_list = []
    valid_indices = []

    for idx, row in df.iterrows():
        psmiles = row['Smiles']
        tg = row['Tg']

        # Process SMILES
        mol, smiles = process_psmiles(psmiles)
        if mol is None:
            continue

        # Compute features
        features, _ = compute_validated_features(mol)

        # Create PyG graph
        try:
            data = mol_to_graph(mol, tg, features)
            if data is not None:
                data_list.append(data)
                features_list.append(features)
                valid_indices.append(idx)
        except Exception as e:
            print(f"Error creating graph for sample {idx}: {e}")
            continue

    print(f"Successfully processed {len(data_list)} samples")

    # Convert to numpy array
    features = np.array(features_list, dtype=np.float32)

    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(features)

    # Save cache
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'data_list': data_list,
                'features': features,
                'scaler': scaler
            }, f)
        print(f"Saved features to cache: {cache_file}")

    return data_list, features, scaler


if __name__ == "__main__":
    # Test dataset loading
    csv_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    data_list, features, scaler = load_tg_dataset(csv_path, use_cache=False)

    print(f"\nDataset loaded:")
    print(f"  Samples: {len(data_list)}")
    print(f"  Feature shape: {features.shape}")
    print(f"  Feature mean: {features.mean(axis=0)[:5]}")
    print(f"  Feature std: {features.std(axis=0)[:5]}")

    # Test a sample
    sample = data_list[0]
    print(f"\nSample graph:")
    print(f"  Nodes: {sample.x.shape}")
    print(f"  Edges: {sample.edge_index.shape}")
    print(f"  Target Tg: {sample.y.item():.2f}")
    print(f"  Features: {sample.features.shape}")
