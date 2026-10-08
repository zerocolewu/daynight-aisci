"""
Dataset preparation for LEMP model
Handles SMILES to 3D structure conversion and feature extraction
"""

import torch
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def get_atom_features(atom):
    """
    Extract atom features for GNN input
    Features: atom type (one-hot), degree, hybridization, aromatic, etc.
    """
    # Atom types (common in polymers)
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si']

    # One-hot encoding of atom type
    atom_type = atom.GetSymbol()
    type_feat = [1 if atom_type == t else 0 for t in atom_types]

    # Degree (0-5)
    degree = min(atom.GetDegree(), 5)
    degree_feat = [1 if i == degree else 0 for i in range(6)]

    # Hybridization (SP, SP2, SP3, SP3D, SP3D2)
    hyb_types = [Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3,
                 Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2]
    hyb = atom.GetHybridization()
    hyb_feat = [1 if hyb == h else 0 for h in hyb_types]

    # Other features
    aromatic = int(atom.GetIsAromatic())
    formal_charge = atom.GetFormalCharge()
    num_hs = atom.GetTotalNumHs()
    in_ring = int(atom.IsInRing())

    # Normalize continuous features
    formal_charge = np.clip(formal_charge, -2, 2) / 2.0
    num_hs = min(num_hs, 4) / 4.0

    features = type_feat + degree_feat + hyb_feat + [aromatic, formal_charge, num_hs, in_ring]

    return features


def get_bond_features(bond):
    """
    Extract bond features
    """
    bond_type = bond.GetBondType()

    # Bond types
    type_feat = [
        int(bond_type == Chem.rdchem.BondType.SINGLE),
        int(bond_type == Chem.rdchem.BondType.DOUBLE),
        int(bond_type == Chem.rdchem.BondType.TRIPLE),
        int(bond_type == Chem.rdchem.BondType.AROMATIC)
    ]

    # Conjugated and in ring
    conjugated = int(bond.GetIsConjugated())
    in_ring = int(bond.IsInRing())

    return type_feat + [conjugated, in_ring]


def generate_3d_conformation(mol, max_attempts=10):
    """
    Generate 3D conformation using ETKDG
    Returns positions as numpy array
    For polymers with wildcard atoms, use distance-based approximation
    """
    if mol is None:
        return None

    # Check for wildcard atoms (*)
    has_wildcard = any(atom.GetSymbol() == '*' for atom in mol.GetAtoms())

    # If has wildcard atoms, use simple 2D coordinate-based approximation for 3D
    if has_wildcard:
        try:
            # Compute 2D coordinates
            AllChem.Compute2DCoords(mol)

            # Get 2D positions and add a small z-component based on atom index
            conf = mol.GetConformer()
            positions = []
            for i in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(i)
                if atom.GetSymbol() != 'H':
                    pos = conf.GetAtomPosition(i)
                    # Create pseudo-3D from 2D + depth based on connectivity
                    z = (i % 3 - 1) * 0.5  # Simple alternating z
                    positions.append([pos.x, pos.y, z])

            return np.array(positions) if positions else None
        except:
            return None

    # Add hydrogens for proper 3D
    mol_h = Chem.AddHs(mol)

    # Try multiple attempts
    for attempt in range(max_attempts):
        try:
            res = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
            if res != -1:
                # Optimize with UFF (more robust than MMFF)
                AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
                break
        except:
            continue

    if mol_h.GetNumConformers() == 0:
        return None

    # Get positions (excluding hydrogens for simplicity)
    conf = mol_h.GetConformer()
    positions = []
    for i in range(mol_h.GetNumAtoms()):
        atom = mol_h.GetAtomWithIdx(i)
        if atom.GetSymbol() != 'H':  # Skip hydrogens
            pos = conf.GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])

    return np.array(positions) if positions else None


def get_atom_types(mol):
    """Get atom types (as integers) for voxel encoding"""
    if mol is None:
        return None

    atom_types = []
    atom_type_map = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4,
                     'Cl': 5, 'Br': 6, 'I': 7, 'P': 8, 'Si': 9}

    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'H':
            atom_types.append(atom_type_map.get(atom.GetSymbol(), 0))

    return atom_types


def compute_ecfp4_fingerprint(mol, radius=2, n_bits=2048):
    """
    Compute ECFP4 fingerprint using RDKit
    """
    if mol is None:
        return np.zeros(n_bits)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def smiles_to_graph(smiles, fp_size=2048):
    """
    Convert SMILES to PyG Data object with all features
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Get node features
        atom_features = []
        atom_positions_3d = []
        atom_types_for_voxel = []

        # First pass: get all atom features
        for atom in mol.GetAtoms():
            feat = get_atom_features(atom)
            atom_features.append(feat)

        x = torch.tensor(atom_features, dtype=torch.float)

        # Get edge indices and features
        edge_index = []
        edge_attr = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_index.append([i, j])
            edge_index.append([j, i])

            bond_feat = get_bond_features(bond)
            edge_attr.append(bond_feat)
            edge_attr.append(bond_feat)

        if len(edge_index) == 0:
            # Single atom molecule
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 6), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Compute ECFP4 fingerprint
        fp = compute_ecfp4_fingerprint(mol, n_bits=fp_size)
        fp = torch.tensor(fp, dtype=torch.float)

        # Generate 3D conformation
        positions_3d = generate_3d_conformation(mol)

        # Convert to torch tensor if valid
        if positions_3d is not None:
            positions_3d = torch.tensor(positions_3d, dtype=torch.float)

        # Get atom types for voxel encoding
        atom_types = get_atom_types(mol)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            fp=fp,
            positions_3d=positions_3d,
            atom_types=atom_types,
            smiles=smiles
        )

        return data

    except Exception as e:
        return None


class PolymerTgDataset(InMemoryDataset):
    """
    Dataset for polymer Tg prediction
    """
    def __init__(self, root, csv_path, fp_size=2048, transform=None):
        self.csv_path = csv_path
        self.fp_size = fp_size
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Tg.csv']

    @property
    def processed_file_names(self):
        return ['tg_processed.pt']

    def download(self):
        pass

    def process(self):
        """Process the dataset"""
        df = pd.read_csv(self.csv_path)
        print(f"Processing {len(df)} polymer repeat units...")

        data_list = []
        failed = 0

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            smiles = row['Smiles']
            tg = row['Tg']

            data = smiles_to_graph(smiles, self.fp_size)

            if data is not None:
                data.y = torch.tensor([tg], dtype=torch.float)
                data_list.append(data)
            else:
                failed += 1

        print(f"Successfully processed: {len(data_list)}, Failed: {failed}")

        if len(data_list) > 0:
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
        else:
            raise ValueError("No valid data processed!")


def load_tg_dataset(csv_path, fp_size=2048, force_reprocess=False):
    """
    Load Tg dataset, processing if necessary
    Returns a list of Data objects directly
    """
    # Process from scratch (always return list of Data objects)
    df = pd.read_csv(csv_path)
    print(f"Processing {len(df)} polymer repeat units...")

    data_list = []
    failed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row['Smiles']
        tg = row['Tg']

        data = smiles_to_graph(smiles, fp_size)

        if data is not None:
            data.y = torch.tensor([tg], dtype=torch.float)
            data_list.append(data)
        else:
            failed += 1

    print(f"Successfully processed: {len(data_list)}, Failed: {failed}")

    return data_list


def collate_fn(data_list):
    """
    Custom collate function for batching with 3D positions
    """
    from torch_geometric.data import Batch

    batch = Batch.from_data_list(data_list)

    # Extract positions and atom types for each graph in batch
    positions_list = []
    atom_types_list = []

    for data in data_list:
        positions_list.append(data.positions_3d)
        atom_types_list.append(data.atom_types)

    return batch, positions_list, atom_types_list


if __name__ == "__main__":
    # Test dataset loading
    csv_path = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
    data_list = load_tg_dataset(csv_path, force_reprocess=True)
    print(f"Loaded {len(data_list)} samples")

    if len(data_list) > 0:
        sample = data_list[0]
        print(f"Sample features:")
        print(f"  - Node features: {sample.x.shape}")
        print(f"  - Edge index: {sample.edge_index.shape}")
        print(f"  - Fingerprint: {sample.fp.shape}")
        print(f"  - 3D positions: {sample.positions_3d.shape if sample.positions_3d is not None else None}")
        print(f"  - Target: {sample.y}")
