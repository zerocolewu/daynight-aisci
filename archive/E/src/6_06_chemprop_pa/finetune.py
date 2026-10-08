"""
Fine-tuning of pre-trained Chemprop-Polymer-K on Tg dataset.
"""
import os
import sys
import pickle
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from model import ChempropPolymerKMPNN, collate_batch
from data_utils import ATOM_FDIM, BOND_FDIM, K_FEAT_DIM


class TgDataset(Dataset):
    """Dataset for Tg prediction."""

    def __init__(self, graphs):
        """
        Args:
            graphs: List of graph dicts with targets
        """
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def create_prediction_head(hidden_dim=1200, dropout=0.0):
    """
    Create FFN prediction head for fine-tuning.

    Args:
        hidden_dim: Input dimension
        dropout: Dropout rate

    Returns:
        nn.Sequential: Prediction head
    """
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


def finetune(
    pretrain_checkpoint_path,
    tg_data,
    use_knode=True,
    hidden_dim=1200,
    depth=7,
    dropout=0.0,
    lr=1e-3,
    seeds=(71, 42, 123),
    split_seed=71,
    epochs=200,
    patience=30,
    batch_size=64,
    device='cuda',
    variant_name='default',
):
    """
    Fine-tune pre-trained encoder on Tg dataset.

    Args:
        pretrain_checkpoint_path: Path to pre-trained checkpoint (None for random init)
        tg_data: List of pre-constructed Tg graphs
        use_knode: Whether to use K-node
        hidden_dim: Hidden dimension
        depth: MPNN depth
        dropout: Dropout rate
        lr: Learning rate
        seeds: Tuple of seeds for FFN initialization
        split_seed: Seed for train/val/test split
        epochs: Maximum epochs
        patience: Early stopping patience
        batch_size: Batch size
        device: Device to use
        variant_name: Name for logging

    Returns:
        results: Dict with results for all seeds
    """
    print(f"\n{'='*60}")
    print(f"Fine-Tuning: {variant_name}")
    print(f"{'='*60}")
    print(f"Pre-trained: {pretrain_checkpoint_path is not None}")
    print(f"K-node: {use_knode}")
    print(f"Hidden dim: {hidden_dim}, Depth: {depth}")
    print(f"LR: {lr}, Batch size: {batch_size}")
    print(f"Seeds: {seeds}")
    print(f"Split seed: {split_seed}")
    print(f"Patience: {patience}, Max epochs: {epochs}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Create train/val/test split (72/8/20)
    train_data, temp_data = train_test_split(
        tg_data,
        train_size=0.72,
        random_state=split_seed
    )
    val_data, test_data = train_test_split(
        temp_data,
        train_size=0.08 / (0.08 + 0.20),  # 8/28 of remaining
        random_state=split_seed
    )

    print(f"Dataset split:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Test: {len(test_data)}")

    # Store results for all seeds
    all_results = {}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed}/{seeds} ({seed_idx+1}/{len(seeds)})")
        print(f"{'='*60}\n")

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create model
        print("Initializing model...")
        model = ChempropPolymerKMPNN(
            atom_fdim=ATOM_FDIM,
            bond_fdim=BOND_FDIM,
            k_feat_dim=K_FEAT_DIM,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            use_knode=use_knode,
            with_projector=False  # No projector for fine-tuning
        ).to(device)

        # Load pre-trained encoder if provided
        if pretrain_checkpoint_path is not None:
            print(f"Loading pre-trained encoder from {pretrain_checkpoint_path}")
            checkpoint = torch.load(pretrain_checkpoint_path, map_location=device)
            model.load_encoder_state_dict(checkpoint['model_state_dict'], strict=False)
            print("  Loaded encoder weights")
        else:
            print("  Random initialization (no pre-training)")

        # Create new prediction head (randomly initialized)
        print(f"Creating new prediction head (seed={seed})...")
        prediction_head = create_prediction_head(hidden_dim=hidden_dim, dropout=dropout).to(device)
        model.predict = prediction_head

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create dataloaders
        train_dataset = TgDataset(train_data)
        val_dataset = TgDataset(val_data)
        test_dataset = TgDataset(test_data)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_batch(batch, use_knode=use_knode)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, use_knode=use_knode)
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, use_knode=use_knode)
        )

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training loop with early stopping
        best_val_rmse = float('inf')
        best_epoch = 0
        best_model_state = None
        patience_counter = 0

        history = {
            'train_rmse': [],
            'val_rmse': [],
            'test_rmse': [],
            'epoch_time': []
        }

        print("\nStarting training...")

        for epoch in range(epochs):
            epoch_start = time.time()

            # Training
            model.train()
            train_preds = []
            train_targets = []
            train_loss = 0.0

            for batch in train_loader:
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)

                # Forward
                targets = batch['targets']
                preds = model(batch)

                # Compute loss
                loss = nn.MSELoss()(preds, targets)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_preds.extend(preds.detach().cpu().numpy())
                train_targets.extend(targets.detach().cpu().numpy())

            train_rmse = np.sqrt(np.mean((np.array(train_preds) - np.array(train_targets))**2))

            # Validation
            model.eval()
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    # Move to device
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(device)

                    targets = batch['targets']
                    preds = model(batch)

                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())

            val_rmse = np.sqrt(np.mean((np.array(val_preds) - np.array(val_targets))**2))

            # Test
            test_preds = []
            test_targets = []

            with torch.no_grad():
                for batch in test_loader:
                    # Move to device
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(device)

                    targets = batch['targets']
                    preds = model(batch)

                    test_preds.extend(preds.cpu().numpy())
                    test_targets.extend(targets.cpu().numpy())

            test_rmse = np.sqrt(np.mean((np.array(test_preds) - np.array(test_targets))**2))

            epoch_time = time.time() - epoch_start

            history['train_rmse'].append(train_rmse)
            history['val_rmse'].append(val_rmse)
            history['test_rmse'].append(test_rmse)
            history['epoch_time'].append(epoch_time)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"  Train RMSE: {train_rmse:.2f} K")
                print(f"  Val RMSE: {val_rmse:.2f} K")
                print(f"  Test RMSE: {test_rmse:.2f} K")
                print(f"  Time: {epoch_time:.2f}s")

            # Early stopping
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Load best model
        print(f"\nLoading best model from epoch {best_epoch+1}")
        model.load_state_dict(best_model_state)

        # Evaluate on test set
        model.eval()
        test_preds = []
        test_targets = []

        with torch.no_grad():
            for batch in test_loader:
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)

                targets = batch['targets']
                preds = model(batch)

                test_preds.extend(preds.cpu().numpy())
                test_targets.extend(targets.cpu().numpy())

        test_rmse = np.sqrt(np.mean((np.array(test_preds) - np.array(test_targets))**2))

        # Compute train RMSE at best epoch
        train_rmse_at_best = history['train_rmse'][best_epoch]
        test_rmse_at_best = history['test_rmse'][best_epoch]
        overfitting_gap = train_rmse_at_best - test_rmse_at_best

        print(f"\nResults for seed {seed}:")
        print(f"  Best epoch: {best_epoch+1}")
        print(f"  Best val RMSE: {best_val_rmse:.2f} K")
        print(f"  Test RMSE: {test_rmse:.2f} K")
        print(f"  Train RMSE (at best): {train_rmse_at_best:.2f} K")
        print(f"  Overfitting gap: {overfitting_gap:.2f} K")

        # Store results
        all_results[seed] = {
            'best_epoch': best_epoch + 1,
            'val_rmse': best_val_rmse,
            'test_rmse': test_rmse,
            'train_rmse_at_best': train_rmse_at_best,
            'overfitting_gap': overfitting_gap,
            'history': history,
            'test_preds': np.array(test_preds),
            'test_targets': np.array(test_targets),
        }

    # Compute statistics across seeds
    test_rmses = [all_results[s]['test_rmse'] for s in seeds]
    gaps = [all_results[s]['overfitting_gap'] for s in seeds]

    mean_test_rmse = np.mean(test_rmses)
    std_test_rmse = np.std(test_rmses, ddof=1)
    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps, ddof=1)

    print(f"\n{'='*60}")
    print(f"Summary across {len(seeds)} seeds:")
    print(f"{'='*60}")
    print(f"Test RMSE: {mean_test_rmse:.2f} ± {std_test_rmse:.2f} K")
    print(f"Overfitting gap: {mean_gap:.2f} ± {std_gap:.2f} K")

    all_results['summary'] = {
        'mean_test_rmse': mean_test_rmse,
        'std_test_rmse': std_test_rmse,
        'mean_overfitting_gap': mean_gap,
        'std_overfitting_gap': std_gap,
    }

    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fine-tuning on Tg')
    parser.add_argument('--checkpoint', type=str, default=None, help='Pre-trained checkpoint path')
    parser.add_argument('--tg-data', type=str, required=True, help='Path to Tg graphs')
    parser.add_argument('--no-knode', action='store_true', help='Disable K-node')
    parser.add_argument('--hidden-dim', type=int, default=1200, help='Hidden dimension')
    parser.add_argument('--depth', type=int, default=7, help='MPNN depth')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seeds', type=int, nargs='+', default=[71, 42, 123], help='Random seeds')
    parser.add_argument('--split-seed', type=int, default=71, help='Split seed')
    parser.add_argument('--epochs', type=int, default=200, help='Max epochs')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--variant', type=str, default='default', help='Variant name')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output', type=str, default=None, help='Output results path')

    args = parser.parse_args()

    # Load Tg data
    print(f"Loading Tg data from {args.tg_data}...")
    with open(args.tg_data, 'rb') as f:
        tg_data = pickle.load(f)
    print(f"Loaded {len(tg_data)} graphs")

    # Run fine-tuning
    results = finetune(
        pretrain_checkpoint_path=args.checkpoint,
        tg_data=tg_data,
        use_knode=not args.no_knode,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        lr=args.lr,
        seeds=args.seeds,
        split_seed=args.split_seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
        variant_name=args.variant,
    )

    # Save results
    if args.output:
        with open(args.output, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nSaved results to {args.output}")
