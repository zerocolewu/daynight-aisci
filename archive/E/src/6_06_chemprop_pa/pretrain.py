"""
Contrastive pre-training of Chemprop-Polymer-K on PI1M dataset.
Uses trimer-shift augmentation + bond-level perturbation + InfoNCE loss.
"""
import os
import sys
import pickle
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from model import ChempropPolymerKMPNN, collate_batch
from data_utils import ATOM_FDIM, BOND_FDIM, K_FEAT_DIM


class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive pre-training.
    Loads pre-constructed polymer graphs and applies on-the-fly augmentation.

    Key features:
    - Loads graphs with monomer/dimer/trimer fragments
    - Randomly selects 2 fragment sizes per sample for trimer-shift augmentation
    - Applies bond dropping + feature masking
    - K-node features are NEVER augmented
    """

    def __init__(self, graphs, aug_bond_drop_rate=0.15, aug_feat_mask_rate=0.25):
        """
        Args:
            graphs: List of graph dicts with keys:
                - atom_features, k_node_features, bond_features
                - edge_index, reverse_map, num_atoms
                - monomer_features, dimer_features, trimer_features (for trimer-shift)
            aug_bond_drop_rate: Probability of dropping each bond
            aug_feat_mask_rate: Probability of masking each feature
        """
        self.graphs = graphs
        self.aug_bond_drop_rate = aug_bond_drop_rate
        self.aug_feat_mask_rate = aug_feat_mask_rate

    def __len__(self):
        return len(self.graphs)

    def _augment_graph(self, graph):
        """
        Apply bond-level augmentation to a graph.
        K-node features are NEVER modified.
        """
        aug_graph = copy.deepcopy(graph)

        # Get number of real atoms (excluding K-node)
        n_real_atoms = aug_graph['num_atoms']

        # Bond dropping: only for bonds between real atoms
        if 'bond_features' in aug_graph and aug_graph['bond_features'].shape[0] > 0:
            n_bonds = aug_graph['bond_features'].shape[0]

            # Create mask for bonds to keep (don't drop virtual bonds to/from K-node)
            # Virtual bonds are those involving atoms >= n_real_atoms
            bond_mask = torch.ones(n_bonds, dtype=torch.bool)

            # Identify bonds between real atoms only
            edge_index = aug_graph['edge_index']
            real_bond_mask = (edge_index[0] < n_real_atoms) & (edge_index[1] < n_real_atoms)

            # Only drop bonds between real atoms
            if real_bond_mask.any():
                real_bond_indices = torch.where(real_bond_mask)[0]
                drop_probs = torch.rand(len(real_bond_indices))
                drop_indices = real_bond_indices[drop_probs < self.aug_bond_drop_rate]
                bond_mask[drop_indices] = False

            # Filter bonds, edge_index, reverse_map
            if not bond_mask.all():
                # Keep only bonds that are not dropped
                aug_graph['bond_features'] = aug_graph['bond_features'][bond_mask]
                aug_graph['edge_index'] = aug_graph['edge_index'][:, bond_mask]

                # Update reverse map (need to remap indices)
                # This is complex, so we'll rebuild it
                kept_indices = torch.where(bond_mask)[0]
                old_reverse = aug_graph['reverse_map']

                # Create new reverse map
                new_reverse = []
                for idx in kept_indices:
                    # Find where old reverse points to
                    old_rev = old_reverse[idx].item()
                    # Check if reverse is also kept
                    new_idx = (kept_indices == old_rev).nonzero(as_tuple=True)[0]
                    if len(new_idx) > 0:
                        new_reverse.append(new_idx[0].item())
                    else:
                        # If reverse was dropped, point to self
                        new_reverse.append((kept_indices == idx).nonzero(as_tuple=True)[0].item())

                aug_graph['reverse_map'] = torch.tensor(new_reverse, dtype=torch.long)

        # Feature masking: only for real atom features, not K-node
        if 'atom_features' in aug_graph:
            atom_feat = aug_graph['atom_features']
            n_atoms = atom_feat.shape[0]

            # Mask only real atoms (first n_real_atoms)
            if n_real_atoms > 0:
                # Create mask tensor
                mask = torch.rand(n_real_atoms, atom_feat.shape[1]) < self.aug_feat_mask_rate
                atom_feat[:n_real_atoms][mask] = 0.0

            aug_graph['atom_features'] = atom_feat

        return aug_graph

    def _trimer_shift(self, graph):
        """
        Trimer-shift augmentation: randomly select monomer/dimer/trimer representation.
        """
        # Check if we have trimer-shift options
        has_monomer = 'monomer_atom_features' in graph
        has_dimer = 'dimer_atom_features' in graph
        has_trimer = 'trimer_atom_features' in graph

        if not (has_monomer or has_dimer or has_trimer):
            # No trimer-shift options, return original
            return graph

        # Randomly select fragment size
        options = []
        if has_monomer:
            options.append('monomer')
        if has_dimer:
            options.append('dimer')
        if has_trimer:
            options.append('trimer')

        if len(options) == 0:
            return graph

        selected = np.random.choice(options)

        # Create augmented graph with selected fragment
        aug_graph = copy.deepcopy(graph)

        if selected == 'monomer':
            aug_graph['atom_features'] = graph['monomer_atom_features']
            aug_graph['bond_features'] = graph['monomer_bond_features']
            aug_graph['edge_index'] = graph['monomer_edge_index']
            aug_graph['reverse_map'] = graph['monomer_reverse_map']
            aug_graph['num_atoms'] = graph['monomer_num_atoms']
        elif selected == 'dimer':
            aug_graph['atom_features'] = graph['dimer_atom_features']
            aug_graph['bond_features'] = graph['dimer_bond_features']
            aug_graph['edge_index'] = graph['dimer_edge_index']
            aug_graph['reverse_map'] = graph['dimer_reverse_map']
            aug_graph['num_atoms'] = graph['dimer_num_atoms']
        elif selected == 'trimer':
            aug_graph['atom_features'] = graph['trimer_atom_features']
            aug_graph['bond_features'] = graph['trimer_bond_features']
            aug_graph['edge_index'] = graph['trimer_edge_index']
            aug_graph['reverse_map'] = graph['trimer_reverse_map']
            aug_graph['num_atoms'] = graph['trimer_num_atoms']

        return aug_graph

    def __getitem__(self, idx):
        """
        Return two augmented views of the same polymer.
        """
        graph = self.graphs[idx]

        # Apply trimer-shift augmentation (randomly select fragment)
        view1 = self._trimer_shift(graph)
        view2 = self._trimer_shift(graph)

        # Apply bond-level augmentation
        view1_aug = self._augment_graph(view1)
        view2_aug = self._augment_graph(view2)

        # Ensure both views have necessary keys
        # Add target placeholder
        view1_aug['target'] = graph.get('target', torch.tensor([0.0]))
        view2_aug['target'] = graph.get('target', torch.tensor([0.0]))

        # Ensure psmiles is present
        view1_aug['psmiles'] = graph.get('psmiles', '')
        view2_aug['psmiles'] = graph.get('psmiles', '')

        return view1_aug, view2_aug


def info_nce_loss(embeddings, temperature=0.07):
    """
    InfoNCE loss for contrastive learning.

    Args:
        embeddings: [2N, dim] tensor of L2-normalized embeddings
                    First N are view1, next N are view2
        temperature: Temperature parameter

    Returns:
        loss: Scalar loss value
    """
    n = embeddings.shape[0] // 2  # Batch size

    # Split into two views
    z1 = embeddings[:n]  # [N, dim]
    z2 = embeddings[n:]  # [N, dim]

    # Compute similarity matrix
    # Similarity between all pairs: [2N, 2N]
    sim_matrix = torch.mm(embeddings, embeddings.t()) / temperature  # [2N, 2N]

    # Create positive mask: diagonal pairs (z1_i with z2_i)
    # For each sample i, positive pair is (z1_i, z2_{i}) and (z2_i, z1_{i})
    # In the concatenated embeddings, positive pairs are:
    # - z1_i with z2_{i+n} for i in [0, n-1]
    # - z2_i with z1_{i-n} for i in [n, 2n-1]

    # Create positive mask
    positive_mask = torch.zeros(2*n, 2*n, dtype=torch.bool, device=embeddings.device)

    # For view1 samples (indices 0 to n-1), positive is corresponding view2 (indices n to 2n-1)
    for i in range(n):
        positive_mask[i, n + i] = True

    # For view2 samples (indices n to 2n-1), positive is corresponding view1 (indices 0 to n-1)
    for i in range(n, 2*n):
        positive_mask[i, i - n] = True

    # Remove diagonals (self-similarity)
    diag_mask = ~torch.eye(2*n, dtype=torch.bool, device=embeddings.device)
    positive_mask = positive_mask & diag_mask

    # Compute logits
    # For each sample, we compute exp(sim) / sum(exp(sim))
    # where positive is the paired view

    # Extract positive similarities
    positive_sim = sim_matrix[positive_mask].view(2*n, 1)  # [2N, 1]

    # Compute logsumexp over all negatives for each sample
    # Excluding self (diagonal)
    mask = torch.eye(2*n, dtype=torch.bool, device=embeddings.device)
    sim_matrix_masked = sim_matrix.masked_fill(mask, -1e9)

    # Log-sum-exp trick
    log_sum_exp = torch.logsumexp(sim_matrix_masked, dim=1, keepdim=True)  # [2N, 1]

    # Compute loss: -log(exp(pos) / sum(exp(all)))
    loss = -positive_sim + log_sum_exp

    # Average over all samples
    loss = loss.mean()

    return loss


def pretrain(
    data_path,
    checkpoint_path,
    use_knode=True,
    hidden_dim=1200,
    depth=7,
    dropout=0.0,
    batch_size=32,
    epochs=30,
    lr=1e-3,
    temperature=0.07,
    warmup_epochs=5,
    device='cuda',
    variant_name='default',
    aug_bond_drop_rate=0.15,
    aug_feat_mask_rate=0.25,
    num_samples=None,
    seed=42,
):
    """
    Contrastive pre-training on PI1M dataset.

    Args:
        data_path: Path to pre-constructed graphs (pickle file)
        checkpoint_path: Where to save checkpoints
        use_knode: Whether to use K-node
        hidden_dim: Hidden dimension
        depth: Number of message passing steps
        dropout: Dropout rate
        batch_size: Batch size (may be reduced if OOM)
        epochs: Number of training epochs
        lr: Learning rate
        temperature: InfoNCE temperature
        warmup_epochs: Number of warmup epochs
        device: Device to use
        variant_name: Name for saving checkpoints
        aug_bond_drop_rate: Bond dropping augmentation rate
        aug_feat_mask_rate: Feature masking augmentation rate
        num_samples: Limit number of samples (for testing)
        seed: Random seed
    """
    print(f"\n{'='*60}")
    print(f"Contrastive Pre-Training: {variant_name}")
    print(f"{'='*60}")
    print(f"Data: {data_path}")
    print(f"K-node: {use_knode}")
    print(f"Hidden dim: {hidden_dim}, Depth: {depth}")
    print(f"Batch size: {batch_size}, Epochs: {epochs}")
    print(f"LR: {lr}, Temperature: {temperature}")
    print(f"Augmentation: bond_drop={aug_bond_drop_rate}, feat_mask={aug_feat_mask_rate}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        graphs = pickle.load(f)

    print(f"Loaded {len(graphs)} graphs")

    # Optionally limit samples
    if num_samples is not None and num_samples < len(graphs):
        graphs = graphs[:num_samples]
        print(f"Using subset of {num_samples} samples")

    # Create dataset
    print("Creating contrastive dataset...")
    dataset = ContrastiveDataset(
        graphs,
        aug_bond_drop_rate=aug_bond_drop_rate,
        aug_feat_mask_rate=aug_feat_mask_rate
    )

    # Create dataloader
    # Each sample returns 2 graphs (view1, view2)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        collate_fn=lambda batch: collate_dual_views(batch, use_knode=use_knode)
    )

    print(f"Dataset: {len(dataset)} samples")
    print(f"Batch size: {batch_size} ({batch_size * 2} graphs per batch)")

    # Create model
    print("\nInitializing model...")
    model = ChempropPolymerKMPNN(
        atom_fdim=ATOM_FDIM,
        bond_fdim=BOND_FDIM,
        k_feat_dim=K_FEAT_DIM,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
        use_knode=use_knode,
        with_projector=True  # Enable projector for pre-training
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Scheduler: cosine annealing with warmup
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs
    )
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=lr * 0.01
    )

    # Training loop
    print("\nStarting training...")
    best_loss = float('inf')
    history = {'loss': [], 'epoch_time': []}

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            try:
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)

                # Forward pass for both views
                view1_batch = {k: v for k, v in batch.items() if not k.endswith('_view2')}
                view2_batch = {k.replace('_view2', ''): v for k, v in batch.items() if k.endswith('_view2')}

                # Get embeddings
                _, emb1 = model(view1_batch, return_embedding=True)
                _, emb2 = model(view2_batch, return_embedding=True)

                # Concatenate embeddings [2N, dim]
                embeddings = torch.cat([emb1, emb2], dim=0)

                # Compute loss
                loss = info_nce_loss(embeddings, temperature=temperature)

                # Backward
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # Print progress
                if (batch_idx + 1) % 50 == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.4f}")

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"\nOOM at batch {batch_idx}, reducing batch size...")
                    torch.cuda.empty_cache()
                    # Return early to let user reduce batch size
                    raise RuntimeError("Out of memory - reduce batch size") from e
                else:
                    raise

        # Update scheduler
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        # Compute epoch metrics
        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start

        history['loss'].append(avg_loss)
        history['epoch_time'].append(epoch_time)

        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'variant': variant_name,
                'use_knode': use_knode,
                'hidden_dim': hidden_dim,
                'depth': depth,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved best checkpoint: {checkpoint_path}")

    # Save final checkpoint
    final_checkpoint_path = checkpoint_path.replace('.pt', '_final.pt')
    checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'variant': variant_name,
        'use_knode': use_knode,
        'hidden_dim': hidden_dim,
        'depth': depth,
    }
    torch.save(checkpoint, final_checkpoint_path)
    print(f"\nSaved final checkpoint: {final_checkpoint_path}")

    print("\nTraining completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final loss: {avg_loss:.4f}")

    return history


def collate_dual_views(batch, use_knode=True):
    """
    Collate a batch of dual-view samples.

    Args:
        batch: List of (view1, view2) tuples
        use_knode: Whether to use K-node

    Returns:
        Batch dict with all view1 and view2 data
    """
    view1_list = [item[0] for item in batch]
    view2_list = [item[1] for item in batch]

    # Collate each view separately
    batch1 = collate_batch(view1_list, use_knode=use_knode)
    batch2 = collate_batch(view2_list, use_knode=use_knode)

    # Merge with _view2 suffix
    merged = {}
    for key in batch1:
        merged[key] = batch1[key]

    for key in batch2:
        merged[f"{key}_view2"] = batch2[key]

    return merged


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Contrastive pre-training on PI1M')
    parser.add_argument('--data', type=str, required=True, help='Path to PI1M graphs')
    parser.add_argument('--checkpoint', type=str, required=True, help='Output checkpoint path')
    parser.add_argument('--no-knode', action='store_true', help='Disable K-node')
    parser.add_argument('--hidden-dim', type=int, default=1200, help='Hidden dimension')
    parser.add_argument('--depth', type=int, default=7, help='MPNN depth')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--variant', type=str, default='default', help='Variant name')
    parser.add_argument('--num-samples', type=int, default=None, help='Limit samples')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    pretrain(
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        use_knode=not args.no_knode,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        variant_name=args.variant,
        num_samples=args.num_samples,
        device=args.device,
    )
