"""
Iteration 3: Fix previous run (RMSE 35.60) targeting RMSE < 30.8.
Key changes from previous iteration:
1. Use 300K PI1M samples for pretraining
2. 65% mask ratio with edge masking
3. Contrastive loss (MoCL-style)
4. Dropout 0.3, label normalization
5. Fix overfitting: progressive unfreezing, better regularization
6. Multi-scale encoder features (JK-net style)
7. Larger fine-tuning batch size for more stable gradients
"""
import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import warnings
warnings.filterwarnings('ignore')

from dataset import load_pretrain_dataset, load_finetune_dataset, MoCLAUG, ContrastiveAug
from model import (GraphMAE, FineTuner, GINEncoder, SCELoss,
                   EdgeReconLoss, ContrastiveLoss, MultiScaleEncoder)

BASE_DIR = "/home/home_public/aisci/AIscientist/src/5_day5_pretraining"

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def pretrain(epochs=60, save_path=None, max_samples=300000):
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "pretrain_model.pt")

    print("\n" + "="*80)
    print("PRETRAINING PHASE (Iteration 3) - GraphMAE + Contrastive on PI1M")
    print("="*80)
    print(f"Config: {max_samples} samples, {epochs} epochs, mask_ratio=0.65")

    t0 = time.time()
    train_loader, train_dataset = load_pretrain_dataset(
        "/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv",
        max_samples=max_samples,
        batch_size=256
    )
    print(f"Data loading took {time.time()-t0:.1f}s")

    # GraphMAE with 65% mask ratio
    model = GraphMAE(in_dim=27, hidden_dim=256, out_dim=256, num_layers=5,
                     mask_ratio=0.65, drop_rate=0.15).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # LR warmup + cosine decay (no restart)
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss functions
    node_criterion = SCELoss(power=5.0)
    edge_criterion = EdgeReconLoss()
    cl_criterion = ContrastiveLoss(temperature=0.1)

    # Augmentation
    aug = MoCLAUG(atom_mask_ratio=0.2, edge_drop_ratio=0.1)
    cl_aug = ContrastiveAug(atom_mask_ratio=0.25, edge_drop_ratio=0.15, feature_noise=0.1)

    # Loss weights
    lambda_node = 1.0
    lambda_edge = 0.5
    lambda_cl = 0.1

    train_losses = []
    node_losses_hist = []
    edge_losses_hist = []
    cl_losses_hist = []
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_node_loss = 0
        epoch_edge_loss = 0
        epoch_cl_loss = 0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)

            # Apply MoCL augmentation for MAE
            aug_batch = aug(batch)

            optimizer.zero_grad()

            # Forward through GraphMAE
            x_recon, x_orig, mask_idx, z, edge_recon, edge_orig, proj = model(
                aug_batch.x, aug_batch.edge_index, aug_batch.edge_attr, aug_batch.batch
            )

            # Node reconstruction loss
            node_loss = node_criterion(x_recon, x_orig, mask_idx)

            # Edge reconstruction loss
            edge_loss = edge_criterion(edge_recon, edge_orig)

            # Contrastive loss: create two augmented views
            view1, view2 = cl_aug(batch)
            with torch.no_grad():
                z1 = model.encoder(view1.x.to(device), view1.edge_index.to(device),
                                   view1.edge_attr.to(device), view1.batch.to(device))
                z2 = model.encoder(view2.x.to(device), view2.edge_index.to(device),
                                   view2.edge_attr.to(device), view2.batch.to(device))
            proj1 = model.proj_head(global_mean_pool(z1, view1.batch.to(device)))
            proj2 = model.proj_head(global_mean_pool(z2, view2.batch.to(device)))
            cl_loss = cl_criterion(proj1, proj2)

            # Total loss
            loss = lambda_node * node_loss + lambda_edge * edge_loss + lambda_cl * cl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_node_loss += node_loss.item()
            epoch_edge_loss += edge_loss.item()
            epoch_cl_loss += cl_loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_node = epoch_node_loss / num_batches
        avg_edge = epoch_edge_loss / num_batches
        avg_cl = epoch_cl_loss / num_batches

        train_losses.append(avg_loss)
        node_losses_hist.append(avg_node)
        edge_losses_hist.append(avg_edge)
        cl_losses_hist.append(avg_cl)

        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f} "
              f"(Node: {avg_node:.4f}, Edge: {avg_edge:.4f}, CL: {avg_cl:.4f}), "
              f"Best: {best_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save pretrained encoder (save full model state for multi-scale)
    torch.save({
        'model_state_dict': model.encoder.state_dict(),
        'full_model_state_dict': model.state_dict(),
        'epoch': epochs,
        'loss': train_losses[-1],
        'node_losses': node_losses_hist,
        'edge_losses': edge_losses_hist,
        'cl_losses': cl_losses_hist,
    }, save_path)
    print(f"\nSaved pretrained encoder: {save_path}")
    print(f"Pretraining took {time.time()-t0:.1f}s")

    return model, train_losses, node_losses_hist, edge_losses_hist, cl_losses_hist


def finetune(pretrain_path=None, epochs=500, save_path=None):
    if pretrain_path is None:
        pretrain_path = os.path.join(BASE_DIR, "pretrain_model.pt")
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "finetune_model.pt")

    print("\n" + "="*80)
    print("FINE-TUNING PHASE (Iteration 3) - Tg Prediction")
    print("="*80)

    t0 = time.time()
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = \
        load_finetune_dataset("/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv")
    print(f"Data loading took {time.time()-t0:.1f}s")

    # Compute label normalization stats from training set
    all_labels = []
    for i in range(len(train_ds)):
        graph = train_ds.get(i)
        if hasattr(graph, 'y') and graph.y is not None:
            all_labels.append(graph.y.item())
    all_labels = np.array(all_labels)
    label_mean = np.mean(all_labels)
    label_std = np.std(all_labels)
    print(f"Label stats: mean={label_mean:.2f}, std={label_std:.2f}")
    print(f"Label range: [{all_labels.min():.1f}, {all_labels.max():.1f}]")

    # Load pretrained encoder into multi-scale encoder
    print(f"\nLoading pretrained encoder from {pretrain_path}...")
    checkpoint = torch.load(pretrain_path, map_location=device)

    # Use MultiScaleEncoder which wraps the pretrained GINEncoder
    encoder = MultiScaleEncoder(in_dim=27, hidden_dim=256, out_dim=256,
                                num_layers=5, drop_rate=0.1)
    encoder.gine.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded pretrained encoder into MultiScaleEncoder!")

    model = FineTuner(encoder, hidden_dim=768, num_heads=4, drop_rate=0.3).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Phase 1: Freeze encoder, train only prediction head (first 30 epochs)
    # Phase 2: Unfreeze with low LR for fine-tuning
    freeze_epochs = 30

    # Optimizer with differential learning rates
    head_params = list(model.attention_pooling.parameters()) + \
                  list(model.aux_proj.parameters()) + \
                  list(model.combine_proj.parameters()) + \
                  list(model.predictor.parameters())
    encoder_params = list(model.encoder.parameters())

    optimizer = optim.AdamW([
        {'params': head_params, 'lr': 5e-4, 'weight_decay': 1e-4},
        {'params': encoder_params, 'lr': 0.0},  # Frozen initially
    ])

    # Smooth cosine annealing without warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 100
    patience_counter = 0

    for epoch in range(epochs):
        # Progressive unfreezing
        if epoch == freeze_epochs:
            print(f"\n--- Unfreezing encoder at epoch {epoch+1} ---")
            optimizer.param_groups[1]['lr'] = 1e-5  # Low LR for encoder

        # Training
        model.train()
        epoch_loss = 0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)

            # Normalize labels
            y_normalized = (batch.y - label_mean) / label_std

            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            # Loss on normalized labels
            loss = torch.sqrt(criterion(pred, y_normalized))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Convert back to original scale for tracking
            epoch_loss += (loss.item() * label_std)
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss_sum = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                y_normalized = (batch.y - label_mean) / label_std
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

                # Denormalize prediction
                pred_orig = pred * label_std + label_mean
                val_loss = torch.sqrt(criterion(pred_orig, batch.y))
                val_loss_sum += val_loss.item()

                val_preds.extend(pred_orig.cpu().numpy())
                val_targets.extend(batch.y.cpu().numpy())
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches
        val_losses.append(avg_val_loss)

        scheduler.step()

        if (epoch + 1) % 10 == 0 or avg_val_loss < best_val_loss:
            print(f"Epoch {epoch+1}/{epochs}, Train RMSE: {avg_train_loss:.4f}, "
                  f"Val RMSE: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'encoder_type': 'multiscale',
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'label_mean': label_mean,
                'label_std': label_std,
            }, save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\nFine-tuning done! Best Val RMSE: {best_val_loss:.4f}")
    print(f"Fine-tuning took {time.time()-t0:.1f}s")

    return model, train_losses, val_losses, label_mean, label_std


def evaluate(model_path=None, label_mean=None, label_std=None):
    if model_path is None:
        model_path = os.path.join(BASE_DIR, "finetune_model.pt")

    print("\n" + "="*80)
    print("EVALUATION PHASE")
    print("="*80)

    _, _, test_loader, _, _, _ = load_finetune_dataset(
        "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
    )

    checkpoint = torch.load(model_path, map_location=device)

    # Load normalization stats from checkpoint if not provided
    if label_mean is None:
        label_mean = checkpoint.get('label_mean', 0.0)
    if label_std is None:
        label_std = checkpoint.get('label_std', 1.0)
    print(f"Label normalization: mean={label_mean:.2f}, std={label_std:.2f}")

    encoder_type = checkpoint.get('encoder_type', 'single')
    print(f"Encoder type: {encoder_type}")

    encoder = MultiScaleEncoder(in_dim=27, hidden_dim=256, out_dim=256,
                                num_layers=5, drop_rate=0.1)
    model = FineTuner(encoder, hidden_dim=768, num_heads=4, drop_rate=0.3).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            # Denormalize predictions
            pred_orig = pred * label_std + label_mean
            test_preds.extend(pred_orig.cpu().numpy())
            test_targets.extend(batch.y.cpu().numpy())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)

    rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    mae = mean_absolute_error(test_targets, test_preds)
    r2 = r2_score(test_targets, test_preds)

    print(f"\nTest Set Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2:   {r2:.4f}")

    results = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'train_rmse': float(checkpoint.get('train_loss', 0)),
        'val_rmse': float(checkpoint.get('val_loss', 0)),
        'label_mean': float(label_mean),
        'label_std': float(label_std)
    }

    return model, test_preds, test_targets, results


def plot_training_curves(pretrain_losses, node_losses, edge_losses, cl_losses,
                         finetune_train, finetune_val, save_path=None):
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "training_curves.png")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Pretraining losses
    axes[0, 0].plot(pretrain_losses, linewidth=2, color='steelblue')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Pretraining Total Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(node_losses, linewidth=2, color='coral')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('SCE Loss')
    axes[0, 1].set_title('Node Reconstruction Loss')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(edge_losses, linewidth=2, color='green', label='Edge')
    axes[0, 2].plot(cl_losses, linewidth=2, color='purple', label='Contrastive')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('Edge Recon + Contrastive Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Fine-tuning losses
    axes[1, 0].plot(finetune_train, linewidth=2, color='steelblue', label='Train RMSE')
    axes[1, 0].plot(finetune_val, linewidth=2, color='coral', label='Val RMSE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('Tg Prediction Fine-tuning')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Summary text
    axes[1, 1].text(0.1, 0.5, 'Iteration 3\n'
                    'GraphMAE (65% mask)\n'
                    '+ Edge Recon\n'
                    '+ Contrastive Loss\n'
                    '+ Label Norm\n'
                    '+ Dropout 0.3\n'
                    '+ 300K PI1M samples\n'
                    '+ MultiScaleEncoder\n'
                    '+ Progressive Unfreeze',
                    ha='left', va='center', transform=axes[1, 1].transAxes,
                    fontsize=12, family='monospace')
    axes[1, 1].set_title('Config')
    axes[1, 1].axis('off')

    axes[1, 2].axis('off')

    plt.suptitle('Iteration 3: GraphMAE + Contrastive Pretraining', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_parity(test_preds, test_targets, save_path=None):
    """Plot parity (predicted vs actual) scatter plot."""
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "parity_plot.png")

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(test_targets, test_preds, alpha=0.3, s=10, c='steelblue')
    min_val = min(test_targets.min(), test_preds.min())
    max_val = max(test_targets.max(), test_preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

    rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    mae = mean_absolute_error(test_targets, test_preds)
    r2 = r2_score(test_targets, test_preds)

    ax.set_xlabel('Actual Tg', fontsize=12)
    ax.set_ylabel('Predicted Tg', fontsize=12)
    ax.set_title(f'Parity Plot (Test Set)\nRMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Parity plot saved to {save_path}")
    plt.close()


def plot_tsne(model, save_path=None):
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "tsne_visualization.png")

    print("\nGenerating t-SNE visualization...")

    _, _, test_loader, _, _, _ = load_finetune_dataset(
        "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
    )

    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            emb = model.get_embeddings(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            embeddings.append(emb.cpu().numpy())
            labels.extend(batch.y.cpu().numpy())

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    perplexity = min(30, len(labels) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=labels, cmap='RdYlBu_r', s=20, alpha=0.7)
    plt.colorbar(scatter, label='Tg Value')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE of Test Set Graph Embeddings (colored by Tg)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"t-SNE visualization saved to {save_path}")
    plt.close()


def main():
    print("\n" + "="*80)
    print("POLYMER Tg PREDICTION PIPELINE - ITERATION 3")
    print("GraphMAE (65% mask + edge + CL) pretraining on 300K PI1M")
    print("Fine-tuning with label normalization + dropout 0.3 + progressive unfreeze")
    print("="*80)

    # Phase 1: Pretraining (300K samples, 60 epochs)
    _, pretrain_losses, node_losses, edge_losses, cl_losses = pretrain(
        epochs=60, max_samples=300000
    )

    # Phase 2: Fine-tuning with anti-overfitting measures
    _, finetune_train, finetune_val, label_mean, label_std = finetune(epochs=500)

    # Phase 3: Evaluation
    model, test_preds, test_targets, results = evaluate(
        label_mean=label_mean, label_std=label_std
    )

    # Save results
    results_path = os.path.join(BASE_DIR, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Phase 4: Visualizations
    plot_training_curves(pretrain_losses, node_losses, edge_losses, cl_losses,
                         finetune_train, finetune_val)
    plot_parity(test_preds, test_targets)
    plot_tsne(model)

    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE - ITERATION 3!")
    print("="*80)
    print(f"\nTest Results:")
    print(f"  RMSE: {results['rmse']:.4f} (target: < 30.8)")
    print(f"  MAE:  {results['mae']:.4f}")
    print(f"  R2:   {results['r2']:.4f}")

    if results['rmse'] < 30.8:
        print("\n*** TARGET ACHIEVED! RMSE < 30.8 ***")
    else:
        print(f"\n*** Target not met. Gap: {results['rmse'] - 30.8:.2f} ***")

    print(f"\nOutput files in {BASE_DIR}/:")
    for f in ['pretrain_model.pt', 'finetune_model.pt', 'results.json',
              'training_curves.png', 'parity_plot.png', 'tsne_visualization.png']:
        print(f"  - {f}")
    print("="*80)


if __name__ == "__main__":
    main()
