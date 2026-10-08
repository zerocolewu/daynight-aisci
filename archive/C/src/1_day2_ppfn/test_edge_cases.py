"""
Test edge cases for PPFN model.
"""

import torch
from model import PPFN


def test_edge_cases():
    """Test various edge cases."""
    print("Testing PPFN edge cases...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PPFN(hidden_dim=256).to(device)
    model.eval()

    # Test 1: Empty edges
    print("\n1. Testing empty edges...")
    batch = {
        'atom_features': torch.randn(10, 27).to(device),
        'edge_index': torch.zeros((2, 0), dtype=torch.long).to(device),
        'edge_features': torch.zeros((0, 6)).to(device),
        'frag_features': torch.randn(3, 31).to(device),
        'frag_edge_index': torch.zeros((2, 0), dtype=torch.long).to(device),
        'frag_edge_features': torch.zeros((0, 6)).to(device),
        'atom_frag_edge_index': torch.stack([
            torch.arange(10, dtype=torch.long),
            torch.randint(0, 3, (10,))
        ]).to(device),
        'batch_atom': torch.zeros(10, dtype=torch.long).to(device),
        'batch_frag': torch.zeros(3, dtype=torch.long).to(device),
    }
    with torch.no_grad():
        pred = model(batch)
    print(f"  ✓ Empty edges: predictions shape {pred.shape}")

    # Test 2: Single atom
    print("\n2. Testing single atom...")
    batch = {
        'atom_features': torch.randn(1, 27).to(device),
        'edge_index': torch.zeros((2, 0), dtype=torch.long).to(device),
        'edge_features': torch.zeros((0, 6)).to(device),
        'frag_features': torch.randn(1, 31).to(device),
        'frag_edge_index': torch.zeros((2, 0), dtype=torch.long).to(device),
        'frag_edge_features': torch.zeros((0, 6)).to(device),
        'atom_frag_edge_index': torch.tensor([[0], [0]], dtype=torch.long).to(device),
        'batch_atom': torch.zeros(1, dtype=torch.long).to(device),
        'batch_frag': torch.zeros(1, dtype=torch.long).to(device),
    }
    with torch.no_grad():
        pred = model(batch)
    print(f"  ✓ Single atom: predictions shape {pred.shape}")

    # Test 3: Multiple batches with varying sizes
    print("\n3. Testing multiple batches with varying sizes...")
    batch = {
        'atom_features': torch.randn(30, 27).to(device),
        'edge_index': torch.randint(0, 30, (2, 60)).to(device),
        'edge_features': torch.randn(60, 6).to(device),
        'frag_features': torch.randn(9, 31).to(device),
        'frag_edge_index': torch.randint(0, 9, (2, 12)).to(device),
        'frag_edge_features': torch.randn(12, 6).to(device),
        'atom_frag_edge_index': torch.stack([
            torch.randint(0, 30, (50,)),
            torch.randint(0, 9, (50,))
        ]).to(device),
        'batch_atom': torch.cat([
            torch.zeros(10, dtype=torch.long),
            torch.ones(10, dtype=torch.long),
            torch.full((10,), 2, dtype=torch.long)
        ]).to(device),
        'batch_frag': torch.cat([
            torch.zeros(3, dtype=torch.long),
            torch.ones(3, dtype=torch.long),
            torch.full((3,), 2, dtype=torch.long)
        ]).to(device),
    }
    with torch.no_grad():
        pred = model(batch)
    print(f"  ✓ Multiple batches: predictions shape {pred.shape}")

    # Test 4: Large batch
    print("\n4. Testing larger batch...")
    batch = {
        'atom_features': torch.randn(500, 27).to(device),
        'edge_index': torch.randint(0, 500, (2, 1000)).to(device),
        'edge_features': torch.randn(1000, 6).to(device),
        'frag_features': torch.randn(100, 31).to(device),
        'frag_edge_index': torch.randint(0, 100, (2, 150)).to(device),
        'frag_edge_features': torch.randn(150, 6).to(device),
        'atom_frag_edge_index': torch.stack([
            torch.randint(0, 500, (400,)),
            torch.randint(0, 100, (400,))
        ]).to(device),
        'batch_atom': torch.cat([
            torch.full((100,), i, dtype=torch.long)
            for i in range(5)
        ]).to(device),
        'batch_frag': torch.cat([
            torch.full((20,), i, dtype=torch.long)
            for i in range(5)
        ]).to(device),
    }
    with torch.no_grad():
        pred = model(batch)
    print(f"  ✓ Large batch: predictions shape {pred.shape}")

    # Test 5: Training mode
    print("\n5. Testing training mode...")
    model.train()
    pred = model(batch)
    loss = pred.sum()
    loss.backward()
    print(f"  ✓ Training mode: backward pass successful")

    print("\n✓ All edge case tests passed!")


if __name__ == '__main__':
    test_edge_cases()
