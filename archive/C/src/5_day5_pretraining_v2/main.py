"""
Iteration v2: Pre-training on PI1M + Fine-tuning on Tg.
Fixes from previous run (RMSE 35.60):
1. Use 300K PI1M samples (vs 100K)
2. 65% mask ratio with edge masking
3. Contrastive loss (InfoNCE)
4. Dropout 0.3, label normalization
5. ReduceLROnPlateau (not CosineAnnealingWarmRestarts)
6. Progressive unfreezing with low encoder LR
7. Early stopping patience=80
8. Smaller fine-tuning batch_size=32
"""
import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import warnings
warnings.filterwarnings('ignore')

from dataset import load_pretrain_dataset, load_finetune_dataset
from model import (GraphMAE, FineTuner, GINEncoder, SCELoss,
                   EdgeReconLoss, ContrastiveLoss)

BASE_DIR = "/home/home_public/aisci/AIscientist/src/5_day5_pretraining_v2"

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def pretrain(epochs=60, save_path=None, max_samples=300000):
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "pretrain_model.pt")

    print("\n" + "="*80)
    print("PRETRAINING PHASE - GraphMAE + Contrastive on PI1M")
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
                     mask_ratio=0.65, drop_rate=0.1).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # LR warmup + cosine decay
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

            optimizer.zero_grad()

            # Forward through GraphMAE
            x_recon, x_orig, mask_idx, z, edge_recon, edge_orig, proj = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )

            # Node reconstruction loss (SCE)
            node_loss = node_criterion(x_recon, x_orig, mask_idx)

            # Edge reconstruction loss
            edge_loss = edge_criterion(edge_recon, edge_orig)

            # Contrastive loss: create two augmented views
            # Apply noise to get second view
            with torch.no_grad():
                z2 = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            proj2 = model.proj_head(global_mean_pool(z2, batch.batch))
            cl_loss = cl_criterion(proj, proj2) if proj is not None else torch.tensor(0.0, device=device)

            # Total loss
            loss = lambda_node * node_loss + lambda_edge * edge_loss + lambda_cl * cl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_node_loss += node_loss.item()
            epoch_edge_loss += edge_loss.item()
            epoch_cl_loss += cl_loss.item() if isinstance(cl_loss, torch.Tensor) else cl_loss
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

    # Save pretrained encoder
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
    print("FINE-TUNING PHASE - Tg Prediction")
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

    # Load pretrained encoder
    print(f"\nLoading pretrained encoder from {pretrain_path}...")
    checkpoint = torch.load(pretrain_path, map_location=device, weights_only=False)

    # Create encoder and load pretrained weights
    encoder = GINEncoder(in_dim=27, hidden_dim=256, out_dim=256,
                         num_layers=5, drop_rate=0.1)
    encoder.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded pretrained encoder!")

    # Create fine-tuning model
    model = FineTuner(encoder, hidden_dim=256, num_heads=4, drop_rate=0.3).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Progressive unfreezing: freeze encoder initially
    freeze_epochs = 50
    for param in model.encoder.parameters():
        param.requires_grad = False

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

    # ReduceLROnPlateau scheduler (key fix from previous iteration)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
    )

    criterion = nn.SmoothL1Loss(beta=1.0)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 80
    patience_counter = 0

    for epoch in range(epochs):
        # Progressive unfreezing
        if epoch == freeze_epochs:
            print(f"\n--- Unfreezing encoder at epoch {epoch+1} ---")
            for param in model.encoder.parameters():
                param.requires_grad = True
            # Re-create optimizer with encoder at low LR
            optimizer = optim.AdamW([
                {'params': head_params, 'lr': 5e-4, 'weight_decay': 1e-4},
                {'params': encoder_params, 'lr': 1e-5, 'weight_decay': 1e-5},
            ])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
            )

        # Training
        model.train()
        epoch_loss = 0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)

            # Normalize labels
            y_normalized = (batch.y.squeeze() - label_mean) / label_std

            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            # Loss on normalized labels
            loss = criterion(pred, y_normalized)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
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
                y_normalized = (batch.y.squeeze() - label_mean) / label_std
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

                # Denormalize prediction for RMSE calculation
                pred_orig = pred * label_std + label_mean
                val_loss = torch.sqrt(nn.MSELoss()(pred_orig, batch.y.squeeze()))
                val_loss_sum += val_loss.item()

                val_preds.extend(pred_orig.cpu().numpy())
                val_targets.extend(batch.y.squeeze().cpu().numpy())
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0 or avg_val_loss < best_val_loss:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
                  f"Val RMSE: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
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

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if label_mean is None:
        label_mean = checkpoint.get('label_mean', 0.0)
    if label_std is None:
        label_std = checkpoint.get('label_std', 1.0)
    print(f"Label normalization: mean={label_mean:.2f}, std={label_std:.2f}")

    # Rebuild model
    encoder = GINEncoder(in_dim=27, hidden_dim=256, out_dim=256,
                         num_layers=5, drop_rate=0.1)
    model = FineTuner(encoder, hidden_dim=256, num_heads=4, drop_rate=0.3).to(device)
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
            test_targets.extend(batch.y.squeeze().cpu().numpy())

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
    axes[1, 0].plot(finetune_train, linewidth=2, color='steelblue', label='Train Loss')
    axes[1, 0].plot(finetune_val, linewidth=2, color='coral', label='Val RMSE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss/RMSE')
    axes[1, 0].set_title('Tg Prediction Fine-tuning')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Summary
    axes[1, 1].text(0.1, 0.5, 'Iteration v2\n'
                    'GraphMAE (65% mask)\n'
                    '+ Edge Recon\n'
                    '+ Contrastive Loss\n'
                    '+ Label Norm\n'
                    '+ Dropout 0.3\n'
                    '+ 300K PI1M samples\n'
                    '+ ReduceLROnPlateau\n'
                    '+ Progressive Unfreeze\n'
                    '+ SmoothL1 Loss',
                    ha='left', va='center', transform=axes[1, 1].transAxes,
                    fontsize=11, family='monospace')
    axes[1, 1].set_title('Config')
    axes[1, 1].axis('off')

    axes[1, 2].axis('off')

    plt.suptitle('Iteration v2: GraphMAE Pretraining + Fine-tuning', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def main():
    print("\n" + "="*80)
    print("POLYMER Tg PREDICTION PIPELINE - ITERATION v2")
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

    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE - ITERATION v2!")
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
              'training_curves.png']:
        print(f"  - {f}")
    print("="*80)


if __name__ == "__main__":
    main()
