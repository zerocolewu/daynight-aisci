"""
Visualization script for PTKGE results.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from model import PTKGEModel
from embeddings import ELEMENT_PROPERTIES, analyze_element_similarity

# Paths
base_dir = '/home/home_public/aisci/AIscientist/src/7_day7_ptkge'
figures_dir = os.path.join(base_dir, 'figures')
checkpoints_dir = os.path.join(base_dir, 'checkpoints')
os.makedirs(figures_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load results
with open(os.path.join(base_dir, 'results.json'), 'r') as f:
    results = json.load(f)

print("\nResults Summary:")
print("-" * 60)
for strategy, res in results.items():
    print(f"{strategy}: RMSE={res['rmse_mean']:.4f}±{res['rmse_std']:.4f}")

# ========== Plot 1: Results Comparison Bar Chart ==========
print("\nGenerating results comparison chart...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

strategies = list(results.keys())
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

# RMSE
ax = axes[0]
means = [results[s]['rmse_mean'] for s in strategies]
stds = [results[s]['rmse_std'] for s in strategies]
bars = ax.bar(strategies, means, yerr=stds, color=colors, capsize=5)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title('RMSE Comparison (Lower is Better)', fontsize=14)
ax.axhline(y=35.42, color='red', linestyle='--', label='PSCP Baseline (35.42)')
ax.legend()
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{mean:.2f}',
            ha='center', va='bottom', fontsize=10)

# MAE
ax = axes[1]
means = [results[s]['mae_mean'] for s in strategies]
stds = [results[s]['mae_std'] for s in strategies]
bars = ax.bar(strategies, means, yerr=stds, color=colors, capsize=5)
ax.set_ylabel('MAE', fontsize=12)
ax.set_title('MAE Comparison (Lower is Better)', fontsize=14)
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{mean:.2f}',
            ha='center', va='bottom', fontsize=10)

# R2
ax = axes[2]
means = [results[s]['r2_mean'] for s in strategies]
stds = [results[s]['r2_std'] for s in strategies]
bars = ax.bar(strategies, means, yerr=stds, color=colors, capsize=5)
ax.set_ylabel('R2', fontsize=12)
ax.set_title('R2 Comparison (Higher is Better)', fontsize=14)
ax.set_ylim(0.85, 0.92)
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{mean:.4f}',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'results_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved results comparison to {figures_dir}/results_comparison.png")

# ========== Plot 2: t-SNE Visualization ==========
print("\nGenerating t-SNE visualization...")
# Load physics model for visualization
pretrained_path = os.path.join(checkpoints_dir, 'pretrained_physics.pt')
if os.path.exists(pretrained_path):
    model = PTKGEModel(
        num_layers=5,
        hidden_dim=256,
        embedding_strategy='physics',
        embedding_dim=8
    ).to(device)

    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get element embeddings
    with torch.no_grad():
        atomic_nums = torch.arange(1, 119).to(device)
        embeddings = model.gnn_encoder.atom_embedding.embedding(atomic_nums).cpu().numpy()

    # Element names
    element_names = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe'
    ]

    # Common elements in polymers (to highlight)
    polymer_elements = {'H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I'}

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Color by group
    groups = [ELEMENT_PROPERTIES[i][3] for i in range(1, 119)]  # Group number

    scatter = ax.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=groups, cmap='tab20', s=100, alpha=0.7
    )

    # Add labels for common elements
    for i, name in enumerate(element_names[:len(embeddings_2d)]):
        if name in polymer_elements:
            ax.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                       fontsize=10, fontweight='bold', ha='center', va='bottom')

    ax.set_title('t-SNE Visualization of Physics-Informed Element Embeddings', fontsize=14)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')

    plt.colorbar(scatter, label='Group Number')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'embedding_tsne.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to {figures_dir}/embedding_tsne.png")

# ========== Plot 3: Similarity Matrix ==========
print("\nGenerating similarity matrix...")
if os.path.exists(pretrained_path):
    elements = ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']

    similarity, element_names = analyze_element_similarity(
        model.gnn_encoder.atom_embedding.embedding, elements
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        similarity,
        xticklabels=element_names,
        yticklabels=element_names,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        vmin=-1, vmax=1,
        ax=ax
    )

    ax.set_title('Element Embedding Cosine Similarity (Physics-Informed)', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'embedding_similarity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity matrix to {figures_dir}/embedding_similarity.png")

# ========== Plot 4: Improvement over Baseline ==========
print("\nGenerating improvement chart...")
baseline_rmse = 35.42  # PSCP baseline

fig, ax = plt.subplots(figsize=(10, 6))

strategies = list(results.keys())
improvements = [(baseline_rmse - results[s]['rmse_mean']) / baseline_rmse * 100 for s in strategies]
colors_imp = ['#e74c3c' if imp < 0 else '#2ecc71' for imp in improvements]

bars = ax.bar(strategies, improvements, color=colors_imp)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('Improvement over PSCP Baseline (%)', fontsize=12)
ax.set_title('RMSE Improvement over PSCP Baseline (35.42)', fontsize=14)

for bar, imp in zip(bars, improvements):
    y_pos = bar.get_height() + 0.2 if imp >= 0 else bar.get_height() - 0.5
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{imp:.2f}%',
            ha='center', va='bottom' if imp >= 0 else 'top', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'improvement_over_baseline.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved improvement chart to {figures_dir}/improvement_over_baseline.png")

print("\n" + "="*60)
print("All visualizations generated successfully!")
print("="*60)
