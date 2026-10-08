"""
Diagnostics: augmentation spectrum, embedding diversity, overfitting gap analysis.
"""
import os
import sys
import pickle
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from model import ChempropPolymerKMPNN, collate_batch
from data_utils import ATOM_FDIM, BOND_FDIM, K_FEAT_DIM


def cosine_similarity_torch(x, y):
    """
    Compute cosine similarity between two tensors.

    Args:
        x: [N, D] tensor
        y: [M, D] tensor

    Returns:
        [N, M] tensor of cosine similarities
    """
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)
    return torch.mm(x_norm, y_norm.t())


def create_augmented_view(graph, bond_drop_rate=0.0, feat_mask_rate=0.0,
                         fragment_type='trimer', seed=42):
    """
    Create an augmented view of a graph.

    Args:
        graph: Original graph dict
        bond_drop_rate: Bond dropping probability
        feat_mask_rate: Feature masking probability
        fragment_type: 'monomer', 'dimer', or 'trimer'
        seed: Random seed

    Returns:
        Augmented graph dict
    """
    rng = np.random.RandomState(seed)
    aug_graph = copy.deepcopy(graph)

    # Apply fragment selection (trimer-shift)
    if fragment_type != 'trimer':
        # Check if alternative fragments are available
        if fragment_type == 'monomer' and 'monomer_atom_features' in graph:
            aug_graph['atom_features'] = graph['monomer_atom_features']
            aug_graph['bond_features'] = graph['monomer_bond_features']
            aug_graph['edge_index'] = graph['monomer_edge_index']
            aug_graph['reverse_map'] = graph['monomer_reverse_map']
            aug_graph['num_atoms'] = graph['monomer_num_atoms']
        elif fragment_type == 'dimer' and 'dimer_atom_features' in graph:
            aug_graph['atom_features'] = graph['dimer_atom_features']
            aug_graph['bond_features'] = graph['dimer_bond_features']
            aug_graph['edge_index'] = graph['dimer_edge_index']
            aug_graph['reverse_map'] = graph['dimer_reverse_map']
            aug_graph['num_atoms'] = graph['dimer_num_atoms']

    # Apply bond dropping
    if bond_drop_rate > 0:
        n_real_atoms = aug_graph['num_atoms']
        n_bonds = aug_graph['bond_features'].shape[0]

        if n_bonds > 0:
            # Identify bonds between real atoms only
            edge_index = aug_graph['edge_index']
            real_bond_mask = (edge_index[0] < n_real_atoms) & (edge_index[1] < n_real_atoms)

            # Create mask for bonds to keep
            bond_mask = torch.ones(n_bonds, dtype=torch.bool)

            # Only drop bonds between real atoms
            if real_bond_mask.any():
                real_bond_indices = torch.where(real_bond_mask)[0]
                drop_probs = torch.rand(len(real_bond_indices))
                drop_indices = real_bond_indices[drop_probs < bond_drop_rate]
                bond_mask[drop_indices] = False

            # Filter bonds
            if not bond_mask.all():
                aug_graph['bond_features'] = aug_graph['bond_features'][bond_mask]
                aug_graph['edge_index'] = aug_graph['edge_index'][:, bond_mask]

                # Update reverse map
                kept_indices = torch.where(bond_mask)[0]
                old_reverse = aug_graph['reverse_map']

                new_reverse = []
                for idx in kept_indices:
                    old_rev = old_reverse[idx].item()
                    new_idx = (kept_indices == old_rev).nonzero(as_tuple=True)[0]
                    if len(new_idx) > 0:
                        new_reverse.append(new_idx[0].item())
                    else:
                        new_reverse.append((kept_indices == idx).nonzero(as_tuple=True)[0].item())

                aug_graph['reverse_map'] = torch.tensor(new_reverse, dtype=torch.long)

    # Apply feature masking
    if feat_mask_rate > 0:
        atom_feat = aug_graph['atom_features']
        n_atoms = atom_feat.shape[0]
        n_real_atoms = aug_graph['num_atoms']

        # Mask only real atoms
        if n_real_atoms > 0:
            mask = torch.rand(n_real_atoms, atom_feat.shape[1]) < feat_mask_rate
            atom_feat[:n_real_atoms][mask] = 0.0

        aug_graph['atom_features'] = atom_feat

    return aug_graph


def augmentation_spectrum(encoder, sample_graphs, device='cuda', batch_size=32):
    """
    Analyze augmentation spectrum: how cosine similarity between augmented
    views changes with augmentation intensity.

    Args:
        encoder: ChempropPolymerKMPNN model
        sample_graphs: List of graph dicts (should have monomer/dimer/trimer variants)
        device: Device to use
        batch_size: Batch size

    Returns:
        Dict with augmentation spectrum results
    """
    print(f"\n{'='*60}")
    print("Augmentation Spectrum Analysis")
    print(f"{'='*60}")
    print(f"Samples: {len(sample_graphs)}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    encoder.eval()
    results = {}

    # Sample subset (max 1000)
    num_samples = min(len(sample_graphs), 1000)
    sample_graphs = sample_graphs[:num_samples]
    print(f"Using {num_samples} samples for analysis\n")

    # Test 1: Bond dropping
    print("Testing bond dropping...")
    bond_drop_rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results['bond_dropping'] = {}

    for rate in bond_drop_rates:
        cos_sims = []

        for i in range(0, num_samples, batch_size):
            batch_graphs = sample_graphs[i:i+batch_size]

            # Create original and augmented views
            original_views = []
            augmented_views = []

            for graph in batch_graphs:
                original_views.append(graph)
                aug_view = create_augmented_view(
                    graph,
                    bond_drop_rate=rate,
                    feat_mask_rate=0.0,
                    fragment_type='trimer'
                )
                augmented_views.append(aug_view)

            # Collate batches
            batch_orig = collate_batch(original_views, use_knode=encoder.use_knode)
            batch_aug = collate_batch(augmented_views, use_knode=encoder.use_knode)

            # Move to device
            for key in batch_orig:
                if isinstance(batch_orig[key], torch.Tensor):
                    batch_orig[key] = batch_orig[key].to(device)
            for key in batch_aug:
                if isinstance(batch_aug[key], torch.Tensor):
                    batch_aug[key] = batch_aug[key].to(device)

            # Get embeddings
            with torch.no_grad():
                _, emb_orig = encoder(batch_orig, return_embedding=True)
                _, emb_aug = encoder(batch_aug, return_embedding=True)

            # Compute cosine similarity
            sim = cosine_similarity_torch(emb_orig, emb_aug)
            cos_sims.extend(sim.diag().cpu().numpy())

        mean_sim = np.mean(cos_sims)
        std_sim = np.std(cos_sims)
        results['bond_dropping'][rate] = {'mean': mean_sim, 'std': std_sim}
        print(f"  Bond drop {rate*100:.0f}%: cos_sim = {mean_sim:.4f} ± {std_sim:.4f}")

    # Test 2: Feature masking
    print("\nTesting feature masking...")
    feat_mask_rates = [0.10, 0.15, 0.25, 0.35, 0.50]
    results['feature_masking'] = {}

    for rate in feat_mask_rates:
        cos_sims = []

        for i in range(0, num_samples, batch_size):
            batch_graphs = sample_graphs[i:i+batch_size]

            original_views = []
            augmented_views = []

            for graph in batch_graphs:
                original_views.append(graph)
                aug_view = create_augmented_view(
                    graph,
                    bond_drop_rate=0.0,
                    feat_mask_rate=rate,
                    fragment_type='trimer'
                )
                augmented_views.append(aug_view)

            batch_orig = collate_batch(original_views, use_knode=encoder.use_knode)
            batch_aug = collate_batch(augmented_views, use_knode=encoder.use_knode)

            for key in batch_orig:
                if isinstance(batch_orig[key], torch.Tensor):
                    batch_orig[key] = batch_orig[key].to(device)
            for key in batch_aug:
                if isinstance(batch_aug[key], torch.Tensor):
                    batch_aug[key] = batch_aug[key].to(device)

            with torch.no_grad():
                _, emb_orig = encoder(batch_orig, return_embedding=True)
                _, emb_aug = encoder(batch_aug, return_embedding=True)

            sim = cosine_similarity_torch(emb_orig, emb_aug)
            cos_sims.extend(sim.diag().cpu().numpy())

        mean_sim = np.mean(cos_sims)
        std_sim = np.std(cos_sims)
        results['feature_masking'][rate] = {'mean': mean_sim, 'std': std_sim}
        print(f"  Feature mask {rate*100:.0f}%: cos_sim = {mean_sim:.4f} ± {std_sim:.4f}")

    # Test 3: Trimer-shift
    print("\nTesting trimer-shift...")
    fragment_pairs = [
        ('monomer', 'dimer'),
        ('dimer', 'trimer'),
        ('monomer', 'trimer')
    ]
    results['trimer_shift'] = {}

    for frag1, frag2 in fragment_pairs:
        cos_sims = []

        for i in range(0, num_samples, batch_size):
            batch_graphs = sample_graphs[i:i+batch_size]

            view1_list = []
            view2_list = []

            for graph in batch_graphs:
                view1 = create_augmented_view(graph, fragment_type=frag1)
                view2 = create_augmented_view(graph, fragment_type=frag2)
                view1_list.append(view1)
                view2_list.append(view2)

            batch1 = collate_batch(view1_list, use_knode=encoder.use_knode)
            batch2 = collate_batch(view2_list, use_knode=encoder.use_knode)

            for key in batch1:
                if isinstance(batch1[key], torch.Tensor):
                    batch1[key] = batch1[key].to(device)
            for key in batch2:
                if isinstance(batch2[key], torch.Tensor):
                    batch2[key] = batch2[key].to(device)

            with torch.no_grad():
                _, emb1 = encoder(batch1, return_embedding=True)
                _, emb2 = encoder(batch2, return_embedding=True)

            sim = cosine_similarity_torch(emb1, emb2)
            cos_sims.extend(sim.diag().cpu().numpy())

        mean_sim = np.mean(cos_sims)
        std_sim = np.std(cos_sims)
        pair_name = f"{frag1}-{frag2}"
        results['trimer_shift'][pair_name] = {'mean': mean_sim, 'std': std_sim}
        print(f"  {frag1} vs {frag2}: cos_sim = {mean_sim:.4f} ± {std_sim:.4f}")

    print("\nAugmentation spectrum analysis complete!")

    return results


def embedding_diversity(encoder, sample_graphs, device='cuda', batch_size=32):
    """
    Analyze embedding diversity: check for collapse and measure pairwise similarities.

    Args:
        encoder: ChempropPolymerKMPNN model
        sample_graphs: List of graph dicts
        device: Device to use
        batch_size: Batch size

    Returns:
        Dict with diversity metrics
    """
    print(f"\n{'='*60}")
    print("Embedding Diversity Analysis")
    print(f"{'='*60}")
    print(f"Samples: {len(sample_graphs)}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    encoder.eval()

    # Sample subset (max 1000)
    num_samples = min(len(sample_graphs), 1000)
    sample_graphs = sample_graphs[:num_samples]
    print(f"Using {num_samples} samples for analysis\n")

    # Get all embeddings
    all_embeddings = []

    print("Computing embeddings...")
    for i in range(0, num_samples, batch_size):
        batch_graphs = sample_graphs[i:i+batch_size]
        batch = collate_batch(batch_graphs, use_knode=encoder.use_knode)

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        with torch.no_grad():
            _, emb = encoder(batch, return_embedding=True)

        all_embeddings.append(emb.cpu())

        if (i + batch_size) % 100 == 0:
            print(f"  Processed {min(i + batch_size, num_samples)}/{num_samples} samples...")

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, dim]

    print(f"\nEmbedding shape: {all_embeddings.shape}")

    # Compute pairwise cosine similarities
    print("Computing pairwise similarities...")
    # For efficiency, sample pairs
    n_samples = all_embeddings.shape[0]
    n_pairs_to_sample = min(10000, n_samples * (n_samples - 1) // 2)

    # Sample random pairs
    indices_i = []
    indices_j = []
    rng = np.random.RandomState(42)

    for _ in range(n_pairs_to_sample):
        i, j = rng.choice(n_samples, size=2, replace=False)
        indices_i.append(i)
        indices_j.append(j)

    indices_i = torch.tensor(indices_i)
    indices_j = torch.tensor(indices_j)

    emb_i = all_embeddings[indices_i]
    emb_j = all_embeddings[indices_j]

    similarities = cosine_similarity_torch(emb_i, emb_j).diag().numpy()

    # Compute metrics
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)
    median_sim = np.median(similarities)

    # Check for collapse (high similarity)
    collapse_threshold = 0.95
    collapse_ratio = np.sum(similarities > collapse_threshold) / len(similarities)

    # Compute embedding norm statistics
    norms = torch.norm(all_embeddings, dim=1).numpy()
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)

    results = {
        'num_samples': num_samples,
        'embedding_dim': all_embeddings.shape[1],
        'mean_cosine_similarity': float(mean_sim),
        'std_cosine_similarity': float(std_sim),
        'min_cosine_similarity': float(min_sim),
        'max_cosine_similarity': float(max_sim),
        'median_cosine_similarity': float(median_sim),
        'collapse_ratio': float(collapse_ratio),
        'mean_embedding_norm': float(mean_norm),
        'std_embedding_norm': float(std_norm),
    }

    print(f"\nResults:")
    print(f"  Mean cosine similarity: {mean_sim:.4f} ± {std_sim:.4f}")
    print(f"  Range: [{min_sim:.4f}, {max_sim:.4f}]")
    print(f"  Median: {median_sim:.4f}")
    print(f"  Collapse ratio (> {collapse_threshold}): {collapse_ratio:.2%}")
    print(f"  Mean norm: {mean_norm:.4f} ± {std_norm:.4f}")

    if collapse_ratio > 0.1:
        print(f"\n  WARNING: High collapse ratio detected ({collapse_ratio:.2%})!")
        print(f"  Consider increasing augmentation or temperature.")

    return results


def overfitting_gap_analysis(results_dict):
    """
    Analyze overfitting gaps across different variants.

    Args:
        results_dict: Dict with results from fine-tuning different variants
                      Keys are variant names, values are dicts with:
                      - 'train_rmse_at_best': train RMSE at best epoch
                      - 'test_rmse': test RMSE at best epoch
                      - (or similar keys from finetune.py output)

    Returns:
        Dict with overfitting gap analysis
    """
    print(f"\n{'='*60}")
    print("Overfitting Gap Analysis")
    print(f"{'='*60}\n")

    gaps = {}

    for variant_name, variant_results in results_dict.items():
        if isinstance(variant_results, dict):
            # Handle single-seed results
            if 'train_rmse_at_best' in variant_results and 'test_rmse' in variant_results:
                train_rmse = variant_results['train_rmse_at_best']
                test_rmse = variant_results['test_rmse']
                gap = train_rmse - test_rmse
                gaps[variant_name] = {
                    'gap': gap,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                }
            # Handle multi-seed results (from finetune.py)
            elif 'summary' in variant_results:
                summary = variant_results['summary']
                gaps[variant_name] = {
                    'mean_gap': summary.get('mean_overfitting_gap', 0.0),
                    'std_gap': summary.get('std_overfitting_gap', 0.0),
                    'mean_test_rmse': summary.get('mean_test_rmse', 0.0),
                    'std_test_rmse': summary.get('std_test_rmse', 0.0),
                }
            # Handle aggregated results across seeds
            elif 'mean_overfitting_gap' in variant_results:
                gaps[variant_name] = {
                    'mean_gap': variant_results['mean_overfitting_gap'],
                    'std_gap': variant_results.get('std_overfitting_gap', 0.0),
                    'mean_test_rmse': variant_results.get('mean_test_rmse', 0.0),
                    'std_test_rmse': variant_results.get('std_test_rmse', 0.0),
                }

    # Print results
    print("Overfitting gaps:")
    for variant_name, metrics in gaps.items():
        if 'mean_gap' in metrics:
            print(f"  {variant_name}:")
            print(f"    Test RMSE: {metrics['mean_test_rmse']:.2f} ± {metrics['std_test_rmse']:.2f} K")
            print(f"    Gap (train-test): {metrics['mean_gap']:.2f} ± {metrics['std_gap']:.2f} K")
        else:
            print(f"  {variant_name}:")
            print(f"    Train RMSE: {metrics['train_rmse']:.2f} K")
            print(f"    Test RMSE: {metrics['test_rmse']:.2f} K")
            print(f"    Gap: {metrics['gap']:.2f} K")

    # Interpretation
    print(f"\nInterpretation:")
    for variant_name, metrics in gaps.items():
        gap = metrics.get('mean_gap', metrics.get('gap', 0.0))

        if gap < 5:
            interpretation = "Low overfitting (good generalization)"
        elif gap < 15:
            interpretation = "Moderate overfitting"
        elif gap < 30:
            interpretation = "High overfitting"
        else:
            interpretation = "Severe overfitting (poor generalization)"

        print(f"  {variant_name}: {interpretation}")

    print(f"\n{'='*60}\n")

    return gaps


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Diagnostic analysis')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['augmentation_spectrum', 'embedding_diversity', 'overfitting_gap'],
                       help='Analysis mode')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to encoder checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data graphs (pickle)')
    parser.add_argument('--results', type=str, default=None,
                       help='Path to results pickle (for overfitting_gap mode)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output', type=str, default=None, help='Output results path')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Determine if K-node is used
    use_knode = checkpoint.get('use_knode', True)
    hidden_dim = checkpoint.get('hidden_dim', 1200)
    depth = checkpoint.get('depth', 7)

    model = ChempropPolymerKMPNN(
        atom_fdim=ATOM_FDIM,
        bond_fdim=BOND_FDIM,
        k_feat_dim=K_FEAT_DIM,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=0.0,
        use_knode=use_knode,
        with_projector=True  # Enable projector for embedding extraction
    ).to(args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")

    # Load data
    print(f"\nLoading data from {args.data}...")
    with open(args.data, 'rb') as f:
        graphs = pickle.load(f)
    print(f"Loaded {len(graphs)} graphs")

    # Run analysis
    if args.mode == 'augmentation_spectrum':
        results = augmentation_spectrum(model, graphs, device=args.device, batch_size=args.batch_size)

    elif args.mode == 'embedding_diversity':
        results = embedding_diversity(model, graphs, device=args.device, batch_size=args.batch_size)

    elif args.mode == 'overfitting_gap':
        # Load results dict
        if args.results is None:
            raise ValueError("--results required for overfitting_gap mode")

        with open(args.results, 'rb') as f:
            results_dict = pickle.load(f)

        results = overfitting_gap_analysis(results_dict)

    # Save results
    if args.output:
        print(f"\nSaving results to {args.output}...")
        with open(args.output, 'wb') as f:
            pickle.dump(results, f)
        print("Results saved successfully")
