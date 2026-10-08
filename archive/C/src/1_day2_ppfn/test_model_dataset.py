"""
Test PPFN model with real dataset.
"""

import torch
from torch.utils.data import DataLoader
from model import PPFN
from dataset import PPFNDataset, collate_fn


def test_with_dataset():
    """Test model with real polymer data."""
    print("Testing PPFN with real dataset...")

    # Load dataset
    csv_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    dataset = PPFNDataset(csv_path, split='train', max_samples=50)

    if len(dataset) == 0:
        print("No samples in dataset!")
        return

    print(f"Dataset loaded: {len(dataset)} samples")

    # Create dataloader
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))

    print(f"\nBatch details:")
    print(f"  Atom features: {batch['atom_features'].shape}")
    print(f"  Edge index: {batch['edge_index'].shape}")
    print(f"  Edge features: {batch['edge_features'].shape}")
    print(f"  Frag features: {batch['frag_features'].shape}")
    print(f"  Frag edge index: {batch['frag_edge_index'].shape}")
    print(f"  Atom-frag edge index: {batch['atom_frag_edge_index'].shape}")
    print(f"  Batch size: {batch['tg'].shape[0]}")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = PPFN(hidden_dim=256).to(device)

    # Move batch to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(batch)

    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Predictions: {predictions.squeeze().cpu().numpy()}")
    print(f"Targets: {batch['tg'].squeeze().cpu().numpy()}")

    # Test interpretability
    interp = model.interpret(batch)
    print(f"\nInterpretability:")
    print(f"  Fragment attention shape: {interp['fragment_attention'].shape}")
    print(f"  Fragment importance shape: {interp['fragment_importance'].shape}")

    # Test with different batch sizes
    print("\nTesting with batch_size=1...")
    loader_single = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    batch_single = next(iter(loader_single))

    for key in batch_single:
        if isinstance(batch_single[key], torch.Tensor):
            batch_single[key] = batch_single[key].to(device)

    with torch.no_grad():
        pred_single = model(batch_single)

    print(f"Single prediction: {pred_single.item():.2f}, Target: {batch_single['tg'].item():.2f}")

    print("\n✓ All tests passed!")


if __name__ == '__main__':
    test_with_dataset()
