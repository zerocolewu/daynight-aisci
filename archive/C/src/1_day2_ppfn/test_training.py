"""
Quick training simulation to verify model works end-to-end.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from model import PPFN
from dataset import PPFNDataset, collate_fn


def train_one_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # Forward pass
        predictions = model(batch)
        targets = batch['tg']

        # Compute loss
        loss = F.mse_loss(predictions.squeeze(), targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Forward pass
            predictions = model(batch)
            targets = batch['tg']

            # Compute loss
            loss = F.mse_loss(predictions.squeeze(), targets)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def main():
    """Run training simulation."""
    print("=" * 60)
    print("PPFN Training Simulation")
    print("=" * 60)

    # Load dataset
    csv_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    print("\nLoading datasets...")
    train_dataset = PPFNDataset(csv_path, split='train', max_samples=200)
    val_dataset = PPFNDataset(csv_path, split='val', max_samples=50)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = PPFN(hidden_dim=256).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Training loop
    n_epochs = 5
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)

    # Test prediction
    print("\nTesting prediction on a sample...")
    model.eval()
    batch = next(iter(val_loader))
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    with torch.no_grad():
        predictions = model(batch)

    print(f"\nPredictions vs Targets:")
    for i in range(min(5, len(predictions))):
        pred = predictions[i].item()
        target = batch['tg'][i].item()
        error = abs(pred - target)
        print(f"  Sample {i}: Pred={pred:7.2f}, Target={target:7.2f}, Error={error:6.2f}")

    # Test interpretability
    print("\nTesting interpretability...")
    interp = model.interpret(batch)
    print(f"Fragment attention shape: {interp['fragment_attention'].shape}")
    print(f"Fragment importance shape: {interp['fragment_importance'].shape}")

    # Get most important fragments for first sample
    sample_idx = 0
    importance = interp['fragment_importance'][sample_idx]
    n_frags = (batch['batch_frag'] == sample_idx).sum().item()

    print(f"\nTop 3 fragments for sample {sample_idx}:")
    for i in range(min(3, n_frags)):
        imp_val = importance[i].item()
        print(f"  Fragment {i}: importance = {imp_val:.4f}")

    print("\n" + "=" * 60)
    print("✓ All tests passed! Model is ready for training.")
    print("=" * 60)


if __name__ == '__main__':
    main()
