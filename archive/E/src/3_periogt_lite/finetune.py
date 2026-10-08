"""
Fine-tuning on Tg dataset.
Supports both baseline (random init) and pre-trained modes.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PerioGTFineTuneModel, count_parameters

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))


def evaluate(model, loader, device):
    """Evaluate model and return RMSE and predictions."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    rmse = compute_rmse(all_preds, all_targets)
    return rmse, all_preds, all_targets


def finetune(
    mode='pretrained',  # 'baseline' or 'pretrained'
    augment='subgraph_crop',  # which pretrained encoder to use
    lr=1e-3,
    weight_decay=0.0,
    batch_size=32,
    max_epochs=300,
    patience=30,
    hidden_dim=256,
    num_layers=3,
    seed=71,
):
    """
    Fine-tune on Tg dataset.

    Args:
        mode: 'baseline' (random init) or 'pretrained' (load pre-trained encoder)
    """
    print("=" * 60)
    print(f"Phase: Fine-tuning ({mode})")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load preprocessed data
    train_data = torch.load(os.path.join(OUTPUT_DIR, 'tg_train.pt'), weights_only=False)
    val_data = torch.load(os.path.join(OUTPUT_DIR, 'tg_val.pt'), weights_only=False)
    test_data = torch.load(os.path.join(OUTPUT_DIR, 'tg_test.pt'), weights_only=False)
    print(f"Data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Create model
    model = PerioGTFineTuneModel(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    if mode == 'pretrained':
        encoder_path = os.path.join(OUTPUT_DIR, f'pretrained_encoder_{augment}.pt')
        if os.path.exists(encoder_path):
            model.load_pretrained_encoder(encoder_path)
        else:
            print(f"WARNING: Pre-trained encoder not found at {encoder_path}")
            print("Falling back to random initialization")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Training loop
    best_val_rmse = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    best_state = None

    results_log = {
        'mode': mode,
        'config': {
            'lr': lr,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'patience': patience,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'seed': seed,
            'n_params': n_params,
        },
        'epoch_metrics': [],
    }

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        train_rmse = np.sqrt(train_loss / n_train_batches)

        # Validate
        val_rmse, _, _ = evaluate(model, val_loader, device)

        results_log['epoch_metrics'].append({
            'epoch': epoch,
            'train_rmse': float(train_rmse),
            'val_rmse': float(val_rmse),
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}")

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    model = model.to(device)

    train_rmse_final, train_preds, train_targets = evaluate(model, train_loader, device)
    val_rmse_final, val_preds, val_targets = evaluate(model, val_loader, device)
    test_rmse, test_preds, test_targets = evaluate(model, test_loader, device)

    overfitting_gap = train_rmse_final - test_rmse

    results = {
        'exp_name': f'{mode}_finetune',
        'mode': mode,
        'train_rmse': float(train_rmse_final),
        'val_rmse': float(val_rmse_final),
        'test_rmse': float(test_rmse),
        'best_epoch': best_epoch,
        'total_epochs': epoch,
        'overfitting_gap': float(overfitting_gap),
        'n_params': n_params,
    }

    results_log['results'] = results
    results_log['train_preds'] = train_preds.tolist()
    results_log['train_targets'] = train_targets.tolist()
    results_log['val_preds'] = val_preds.tolist()
    results_log['val_targets'] = val_targets.tolist()
    results_log['test_preds'] = test_preds.tolist()
    results_log['test_targets'] = test_targets.tolist()

    # Save results
    if mode == 'baseline':
        results_path = os.path.join(OUTPUT_DIR, 'results_baseline.json')
        model_path = os.path.join(OUTPUT_DIR, 'model_baseline.pt')
    else:
        results_path = os.path.join(OUTPUT_DIR, f'results_pretrained_{augment}.json')
        model_path = os.path.join(OUTPUT_DIR, f'model_pretrained_{augment}.pt')

    with open(results_path, 'w') as f:
        json.dump(results_log, f, indent=2)
    torch.save(model.state_dict(), model_path)

    print(f"\n{'='*60}")
    print(f"Results ({mode}):")
    print(f"  Train RMSE: {train_rmse_final:.4f}")
    print(f"  Val RMSE:   {val_rmse_final:.4f}")
    print(f"  Test RMSE:  {test_rmse:.4f}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Overfit gap: {overfitting_gap:.4f}")
    print(f"  Saved to {results_path}")
    print(f"{'='*60}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='baseline',
                       choices=['baseline', 'pretrained'])
    parser.add_argument('--augment', type=str, default='edge_drop_feat_mask',
                       help='Tag name for pretrained encoder to load (for pretrained mode). '
                            'Examples: edge_drop_feat_mask, edge_drop_feat_mask_40, edge_drop_feat_mask_50')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=30)
    args = parser.parse_args()

    finetune(mode=args.mode, lr=args.lr, max_epochs=args.epochs,
             patience=args.patience, augment=args.augment)
