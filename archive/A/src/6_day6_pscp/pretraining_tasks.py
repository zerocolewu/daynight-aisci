"""
Pretraining tasks for PSCP:
1. Functional Group Masking (FGM)
2. Cyclic Rotation Invariance (CRI)
3. ECFP-Guided Contrastive Learning (EGCL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit import RDLogger
import random
from polymer_utils import (
    get_functional_group_mask_weights,
    get_cyclic_rotations,
    compute_ecfp_fingerprint,
    tanimoto_similarity,
    smiles_to_graph
)

# Suppress RDKit logging
RDLogger.DisableLog('rdApp.*')


class FunctionalGroupMasking:
    """
    Functional Group Masking (FGM) pretraining task
    Mask atoms preferentially in functional groups and predict their types
    """

    def __init__(self, mask_ratio=0.15, functional_group_weight=2.0):
        self.mask_ratio = mask_ratio
        self.functional_group_weight = functional_group_weight

    def mask_graph(self, data, mol):
        """
        Mask atoms in the graph, preferentially selecting functional group atoms

        Args:
            data: PyTorch Geometric Data object
            mol: RDKit molecule

        Returns:
            masked_data: Data with masked atoms
            mask_indices: indices of masked atoms
            original_features: original features of masked atoms
        """
        num_atoms = data.x.size(0)
        num_mask = max(1, int(num_atoms * self.mask_ratio))

        # Get masking weights based on functional groups
        if mol is not None:
            weights = get_functional_group_mask_weights(mol, list(range(num_atoms)))
        else:
            weights = np.ones(num_atoms) / num_atoms

        # Sample atoms to mask
        mask_indices = np.random.choice(
            num_atoms,
            size=min(num_mask, num_atoms),
            replace=False,
            p=weights
        )

        # Create masked data
        masked_data = data.clone()
        original_features = data.x[mask_indices].clone()

        # Replace masked atoms with mask token (zeros or learnable)
        masked_data.x[mask_indices] = 0

        return masked_data, mask_indices, original_features

    def __call__(self, data, mol):
        return self.mask_graph(data, mol)


class CyclicRotationInvariance:
    """
    Cyclic Rotation Invariance (CRI) pretraining task
    Enforce that rotated versions of polymer SMILES have similar representations
    """

    def __init__(self, num_rotations=3, temperature=0.1):
        self.num_rotations = num_rotations
        self.temperature = temperature

    def get_rotated_graphs(self, smiles):
        """
        Generate rotated versions of a polymer SMILES

        Args:
            smiles: original SMILES string

        Returns:
            list of Data objects for different rotations
        """
        rotations = get_cyclic_rotations(smiles, k=self.num_rotations)
        graphs = []

        for rot_smiles in rotations:
            graph = smiles_to_graph(rot_smiles)
            if graph is not None:
                graphs.append(graph)

        return graphs


class ECFPGuidedContrastive:
    """
    ECFP-Guided Contrastive Learning (EGCL)
    Use ECFP similarity to define positive/negative pairs
    """

    def __init__(self, pos_threshold=0.7, neg_threshold=0.3,
                 num_positives=1, num_negatives=5, radius=2, n_bits=2048):
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.radius = radius
        self.n_bits = n_bits

    def compute_fingerprint(self, smiles):
        """Compute ECFP fingerprint"""
        return compute_ecfp_fingerprint(smiles, self.radius, self.n_bits)

    def find_positives_negatives(self, anchor_fp, all_fps, all_indices,
                                  anchor_idx=None):
        """
        Find positive and negative samples based on ECFP similarity

        Args:
            anchor_fp: fingerprint of anchor
            all_fps: list of all fingerprints
            all_indices: corresponding indices
            anchor_idx: index of anchor (to exclude)

        Returns:
            positive_indices, negative_indices
        """
        similarities = []
        for i, fp in enumerate(all_fps):
            if anchor_idx is not None and all_indices[i] == anchor_idx:
                continue
            sim = tanimoto_similarity(anchor_fp, fp)
            similarities.append((sim, all_indices[i]))

        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Get positives (high similarity)
        positives = [idx for sim, idx in similarities[:self.num_positives]
                     if sim >= self.pos_threshold]

        # Get negatives (low similarity)
        negatives = [idx for sim, idx in similarities[-self.num_negatives:]
                     if sim <= self.neg_threshold]

        # If not enough, just take closest/furthest
        if len(positives) == 0:
            positives = [similarities[0][1]] if similarities else []
        if len(negatives) == 0:
            negatives = [similarities[-1][1]] if similarities else []

        return positives[:self.num_positives], negatives[:self.num_negatives]


class PretrainingLoss(nn.Module):
    """
    Combined pretraining loss
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def mask_prediction_loss(self, predictions, targets):
        """Cross-entropy loss for masked atom prediction"""
        return F.cross_entropy(predictions, targets)

    def contrastive_loss(self, anchor, positive, negatives):
        """InfoNCE loss for contrastive learning"""
        # Normalize
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)

        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature

        # Negative similarities
        if negatives.dim() == 2:
            negatives = negatives.unsqueeze(1)
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature

        # Combine
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)

        return F.cross_entropy(logits, labels)

    def rotation_invariance_loss(self, embeddings_list):
        """
        Loss to enforce similar embeddings for rotated versions
        """
        if len(embeddings_list) < 2:
            return torch.tensor(0.0, device=embeddings_list[0].device)

        total_loss = 0.0
        count = 0

        for i in range(len(embeddings_list)):
            for j in range(i + 1, len(embeddings_list)):
                emb1 = F.normalize(embeddings_list[i], dim=-1)
                emb2 = F.normalize(embeddings_list[j], dim=-1)

                # Cosine similarity loss
                sim = torch.sum(emb1 * emb2, dim=-1)
                loss = 1 - sim.mean()  # Want similarity close to 1

                total_loss += loss
                count += 1

        return total_loss / count if count > 0 else torch.tensor(0.0)


class PI1MDataset(torch.utils.data.Dataset):
    """
    Dataset for PI1M pretraining
    Streams data from disk to avoid memory issues
    """

    def __init__(self, csv_path, max_samples=None, use_cache=True):
        self.csv_path = csv_path
        self.max_samples = max_samples
        self.use_cache = use_cache

        # Load SMILES
        print(f"Loading PI1M data from {csv_path}...")
        import pandas as pd
        df = pd.read_csv(csv_path)
        raw_smiles = df['SMILES'].tolist()

        # Filter valid SMILES upfront
        print("Filtering valid SMILES...")
        self.smiles_list = []
        for smi in raw_smiles:
            if max_samples is not None and len(self.smiles_list) >= max_samples:
                break
            clean_smiles = smi.replace('*', '')
            if clean_smiles and len(clean_smiles) > 1:
                mol = Chem.MolFromSmiles(clean_smiles)
                if mol is not None:
                    self.smiles_list.append(smi)

        print(f"Loaded {len(self.smiles_list)} valid SMILES (from {len(raw_smiles)} total)")

        # Cache for graphs
        self.cache = {}

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        if self.use_cache and idx in self.cache:
            return self.cache[idx]

        smiles = self.smiles_list[idx]

        # Clean SMILES (remove asterisks for graph construction)
        clean_smiles = smiles.replace('*', '')
        if not clean_smiles:
            clean_smiles = 'C'  # Fallback

        # Create graph
        graph = smiles_to_graph(clean_smiles)
        mol = Chem.MolFromSmiles(clean_smiles)

        if graph is None:
            # Return a dummy graph (159 features to match actual atom features, 13 edge features)
            graph = Data(x=torch.zeros(1, 159), edge_index=torch.zeros(2, 0, dtype=torch.long), edge_attr=torch.zeros(0, 13, dtype=torch.float))
            mol = None

        result = {
            'graph': graph,
            'smiles': smiles,
            'clean_smiles': clean_smiles,
            'mol': mol,
            'idx': int(idx)
        }

        if self.use_cache:
            self.cache[idx] = result

        return result


class TgDataset(torch.utils.data.Dataset):
    """
    Dataset for Tg fine-tuning
    """

    def __init__(self, csv_path, compute_ecfp=True, ecfp_radius=2, ecfp_bits=2048):
        import pandas as pd
        self.df = pd.read_csv(csv_path)

        print(f"Loading Tg dataset from {csv_path}...")
        print(f"Columns: {self.df.columns.tolist()}")
        print(f"Total samples: {len(self.df)}")

        self.compute_ecfp = compute_ecfp
        self.ecfp_radius = ecfp_radius
        self.ecfp_bits = ecfp_bits

        # Pre-compute ECFP fingerprints
        if compute_ecfp:
            print("Computing ECFP fingerprints...")
            self.ecfp_cache = {}
            for idx, row in self.df.iterrows():
                smiles = row['Smiles']
                clean_smiles = smiles.replace('*', '')
                if not clean_smiles:
                    clean_smiles = 'C'
                self.ecfp_cache[idx] = compute_ecfp_fingerprint(
                    clean_smiles, ecfp_radius, ecfp_bits
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row['Smiles']
        tg = row['Tg']

        # Clean SMILES
        clean_smiles = smiles.replace('*', '')
        if not clean_smiles:
            clean_smiles = 'C'

        # Create graph
        graph = smiles_to_graph(clean_smiles)

        if graph is None:
            graph = Data(x=torch.zeros(1, 159), edge_index=torch.zeros(2, 0, dtype=torch.long), edge_attr=torch.zeros(0, 13, dtype=torch.float))

        result = {
            'graph': graph,
            'smiles': smiles,
            'clean_smiles': clean_smiles,
            'tg': float(tg),
            'idx': int(idx)
        }

        if self.compute_ecfp:
            result['ecfp'] = self.ecfp_cache[idx].astype(np.float32)

        return result


def collate_fn(batch):
    """Custom collate function for batching"""
    graphs = [item['graph'] for item in batch]
    batched_graph = Batch.from_data_list(graphs)

    result = {
        'graph': batched_graph,
        'idx': [item['idx'] for item in batch],
        'smiles': [item['smiles'] for item in batch],
    }

    if 'tg' in batch[0]:
        result['tg'] = torch.tensor([item['tg'] for item in batch], dtype=torch.float32)

    if 'ecfp' in batch[0]:
        result['ecfp'] = torch.tensor(np.stack([item['ecfp'] for item in batch]), dtype=torch.float32)

    if 'mol' in batch[0]:
        result['mol'] = [item['mol'] for item in batch]

    return result
