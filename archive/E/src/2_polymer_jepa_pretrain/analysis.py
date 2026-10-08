"""
Diagnostic analysis for Polymer JEPA Pre-training experiment.
- Pre-training loss curve
- Fine-tuning comparison table
- t-SNE visualization of pre-trained embeddings
- Cosine similarity between monomer and trimer embeddings
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def plot_pretraining_loss(loss_history_path, save_dir='./figures'):
    """Plot JEPA pre-training loss curve."""
    os.makedirs(save_dir, exist_ok=True)

    losses = np.load(loss_history_path)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('JEPA Loss (MSE)', fontsize=12)
    ax.set_title('JEPA Pre-training Loss Curve', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'jepa_pretrain_loss.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved pre-training loss curve to {save_path}")

    return losses


def plot_finetune_comparison(results, save_dir='./figures'):
    """Plot comparison bar chart of fine-tuning strategies."""
    os.makedirs(save_dir, exist_ok=True)

    strategies = []
    test_rmses = []
    train_rmses = []
    val_rmses = []
    gaps = []

    for name, res in results.items():
        strategies.append(name)
        test_rmses.append(res['test_rmse'])
        train_rmses.append(res['train_rmse'])
        val_rmses.append(res['val_rmse'])
        gaps.append(res['overfitting_gap'])

    x = np.arange(len(strategies))
    width = 0.2

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    bars1 = ax.bar(x - 1.5*width, train_rmses, width, label='Train RMSE', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, val_rmses, width, label='Val RMSE', color='#FF9800', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, test_rmses, width, label='Test RMSE', color='#F44336', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, gaps, width, label='Overfitting Gap', color='#9C27B0', alpha=0.8)

    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Fine-tuning Strategy Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'finetune_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comparison plot to {save_path}")


def plot_training_curves(results, save_dir='./figures'):
    """Plot training/validation curves for each strategy."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (name, res) in enumerate(results.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        train_hist = res.get('train_history', [])
        val_hist = res.get('val_history', [])
        epochs = range(1, len(train_hist) + 1)

        ax.plot(epochs, train_hist, 'b-', label='Train RMSE', linewidth=1.5)
        ax.plot(epochs, val_hist, 'r-', label='Val RMSE', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title(f'{name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training Curves for Each Strategy', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training curves to {save_path}")


def compute_tsne(encoder, dataset, device, save_dir='./figures', max_samples=2000):
    """Compute t-SNE visualization of pre-trained embeddings."""
    os.makedirs(save_dir, exist_ok=True)

    encoder.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    all_embeddings = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            node_emb = encoder(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            graph_emb = global_add_pool(node_emb, batch.batch)
            all_embeddings.append(graph_emb.cpu().numpy())
            all_targets.append(batch.y.squeeze(-1).cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        targets = targets[indices]

    logger.info(f"Computing t-SNE for {len(embeddings)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                        c=targets, cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Tg (K)')
    ax.set_title('t-SNE of Pre-trained Embeddings (colored by Tg)', fontsize=14)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'tsne_embeddings.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved t-SNE plot to {save_path}")

    return embeddings_2d, targets


def compute_monomer_trimer_similarity(encoder, tg_dataset, device, n_samples=500):
    """Compute cosine similarity between monomer and trimer embeddings.

    For each polymer, we compute the embedding of the full trimer and
    compare with the embedding of just the first repeat unit (RU0).
    High similarity indicates periodicity-invariant representations.
    """
    from data_preprocessing import psmiles_to_trimer, mol_to_graph
    import pandas as pd

    encoder.eval()

    # Load Tg data to get SMILES
    df = pd.read_csv('/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv')
    smiles_list = df['Smiles'].tolist()

    # Sample subset
    np.random.seed(42)
    if len(smiles_list) > n_samples:
        indices = np.random.choice(len(smiles_list), n_samples, replace=False)
        smiles_list = [smiles_list[i] for i in indices]

    similarities = []

    with torch.no_grad():
        for smi in smiles_list:
            try:
                mol, unit_boundaries = psmiles_to_trimer(smi)
                if mol is None:
                    continue

                # Full trimer graph
                trimer_graph = mol_to_graph(mol, unit_boundaries)
                if trimer_graph is None:
                    continue

                # Create monomer subgraph (RU0 only)
                start, end = unit_boundaries[0]
                n_monomer = end - start

                monomer_x = trimer_graph.x[start:end]
                # Update ru_index: all zeros for monomer
                monomer_x[:, -3:] = torch.tensor([1, 0, 0], dtype=torch.float)

                # Get edges within RU0
                edge_index = trimer_graph.edge_index
                edge_mask = (edge_index[0] >= start) & (edge_index[0] < end) & \
                           (edge_index[1] >= start) & (edge_index[1] < end)
                monomer_edges = edge_index[:, edge_mask] - start

                if monomer_edges.shape[1] == 0:
                    continue

                # Get trimer embedding
                trimer_batch = torch.zeros(trimer_graph.num_nodes, dtype=torch.long, device=device)
                trimer_graph = trimer_graph.to(device)
                trimer_node_emb = encoder(trimer_graph.x, trimer_graph.edge_index, trimer_batch, trimer_graph.edge_attr)
                trimer_emb = global_add_pool(trimer_node_emb, trimer_batch)  # (1, 256)

                # Get monomer embedding
                monomer_graph = trimer_graph.clone()
                monomer_batch = torch.zeros(n_monomer, dtype=torch.long, device=device)
                monomer_node_emb = encoder(monomer_x.to(device), monomer_edges.to(device), monomer_batch,
                                           monomer_graph.edge_attr[edge_mask].to(device) if monomer_graph.edge_attr is not None else None)
                monomer_emb = global_add_pool(monomer_node_emb, monomer_batch)  # (1, 256)

                # Cosine similarity
                sim = F.cosine_similarity(trimer_emb, monomer_emb, dim=-1).item()
                similarities.append(sim)

            except Exception:
                continue

    if len(similarities) == 0:
        logger.warning("No valid monomer-trimer pairs for similarity computation")
        return 0.0, []

    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)

    logger.info(f"Monomer-Trimer Cosine Similarity: {mean_sim:.4f} +/- {std_sim:.4f} (n={len(similarities)})")

    # Plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(similarities, bins=50, alpha=0.7, color='#2196F3', edgecolor='black')
    ax.axvline(mean_sim, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_sim:.4f}')
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Monomer-Trimer Embedding Similarity', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('./figures', exist_ok=True)
    plt.savefig('./figures/monomer_trimer_similarity.png', dpi=150, bbox_inches='tight')
    plt.close()

    return mean_sim, similarities


def generate_results_summary(all_results, jepa_losses, mean_similarity, n_pi1m, save_path='./results_summary.txt'):
    """Generate a text summary of all results."""
    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("POLYMER JEPA PRE-TRAINING EXPERIMENT RESULTS\n")
        f.write("=" * 70 + "\n\n")

        # Pre-training
        f.write("PHASE 1: JEPA Pre-training on PI1M\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of PI1M samples used: {n_pi1m}\n")
        f.write(f"Number of epochs: {len(jepa_losses)}\n")
        if len(jepa_losses) > 0:
            f.write(f"Initial loss: {jepa_losses[0]:.6f}\n")
            f.write(f"Final loss: {jepa_losses[-1]:.6f}\n")
            f.write(f"Best loss: {min(jepa_losses):.6f}\n")
        f.write("\n")

        # Fine-tuning
        f.write("PHASE 2: Fine-tuning on Tg\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Strategy':<30} {'Test RMSE':>10} {'Val RMSE':>10} {'Train RMSE':>10} {'Gap':>8} {'Epoch':>6}\n")
        f.write("-" * 74 + "\n")

        for name, res in all_results.items():
            f.write(f"{name:<30} {res['test_rmse']:>10.4f} {res['val_rmse']:>10.4f} "
                   f"{res['train_rmse']:>10.4f} {res['overfitting_gap']:>8.4f} {res['best_epoch']:>6}\n")

        f.write("\n")

        # Key findings
        f.write("KEY FINDINGS\n")
        f.write("-" * 40 + "\n")
        baseline_rmse = all_results.get('baseline', {}).get('test_rmse', None)
        if baseline_rmse is not None:
            f.write(f"Baseline (no pre-train) Test RMSE: {baseline_rmse:.4f}\n")
            for name in ['strategy_A', 'strategy_B', 'strategy_C']:
                if name in all_results:
                    rmse = all_results[name]['test_rmse']
                    diff = rmse - baseline_rmse
                    f.write(f"{name} Test RMSE: {rmse:.4f} (delta: {diff:+.4f})\n")

            any_improvement = any(
                all_results[k]['test_rmse'] < baseline_rmse
                for k in ['strategy_A', 'strategy_B', 'strategy_C']
                if k in all_results
            )
            f.write(f"\nAny pre-training strategy beats baseline: {any_improvement}\n")

        f.write(f"\nMonomer-Trimer Cosine Similarity: {mean_similarity:.4f}\n")

        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("SUCCESS CRITERION: ANY strategy with Test RMSE < 50.19 validates hypothesis\n")
        f.write("=" * 70 + "\n")

    logger.info(f"Results summary saved to {save_path}")


if __name__ == '__main__':
    from data_preprocessing import TgTrimerDataset, get_data_splits, TOTAL_ATOM_FEATURE_DIM
    from jepa_pretrain import GINEncoder

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load results
    if os.path.exists('finetune_results.json'):
        with open('finetune_results.json', 'r') as f:
            results = json.load(f)

        plot_finetune_comparison(results)

    # Plot pre-training loss
    loss_path = './jepa_loss_history.npy'
    if os.path.exists(loss_path):
        jepa_losses = plot_pretraining_loss(loss_path)
    else:
        jepa_losses = []

    # t-SNE
    checkpoint_path = './pretrained_encoder.pt'
    if os.path.exists(checkpoint_path):
        encoder = GINEncoder(TOTAL_ATOM_FEATURE_DIM, hidden_dim=256, dropout=0.5).to(device)
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        encoder.load_state_dict(checkpoint['teacher_encoder'])

        tg_dataset = TgTrimerDataset(
            root='./data',
            csv_path='/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv',
        )

        _, _, test_idx = get_data_splits(tg_dataset)
        from torch.utils.data import Subset
        test_dataset = Subset(tg_dataset, test_idx)

        compute_tsne(encoder, test_dataset, device)
        mean_sim, _ = compute_monomer_trimer_similarity(encoder, tg_dataset, device)
    else:
        mean_sim = 0.0

    logger.info("Analysis complete!")
