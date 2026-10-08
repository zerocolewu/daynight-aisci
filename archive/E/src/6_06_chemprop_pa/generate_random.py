"""
Generate random graphs as control for contrastive pre-training.
Creates 300K synthetic random graphs with same dimensions as real polymer data.
"""
import os
import sys
import pickle
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from data_utils import ATOM_FDIM, BOND_FDIM, K_FEAT_DIM


def generate_random_graph(num_atoms=None, num_bonds=None, use_knode=True):
    """
    Generate a single random graph with specified dimensions.

    Graph structure:
    - Erdos-Renyi random graph with edge_probability = 0.15
    - Random atom features (dim=123)
    - Random bond features (dim=9)
    - NO periodic edges
    - NO polymer structure
    - K-node with random features (dim=528) if use_knode=True

    Args:
        num_atoms: Number of atoms (random in [20, 60] if None)
        num_bonds: Number of undirected bonds (estimated if None)
        use_knode: Whether to add K-node

    Returns:
        Graph dict with all necessary fields
    """
    # Random number of atoms
    if num_atoms is None:
        num_atoms = np.random.randint(20, 61)

    # Generate atom features (random in [0,1])
    atom_features = np.random.rand(num_atoms, ATOM_FDIM).astype(np.float32)

    # Generate bonds using Erdos-Renyi model
    edge_probability = 0.15
    max_possible_bonds = num_atoms * (num_atoms - 1) // 2

    # Randomly decide which undirected bonds exist
    bond_exists = np.random.rand(max_possible_bonds) < edge_probability
    num_undirected_bonds = bond_exists.sum()

    # Create bond list (directed: both ways)
    bonds = []
    bond_features = []
    bond_idx = 0

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if bond_exists[bond_idx]:
                # Forward: i -> j
                bonds.append((i, j))
                # Random bond features
                bf = np.random.rand(BOND_FDIM).astype(np.float32)
                # Set periodic-related features to 0
                bf[6] = 0.0  # is_periodic
                bond_features.append(bf)

                # Backward: j -> i
                bonds.append((j, i))
                bond_features.append(bf)

            bond_idx += 1

    # Convert to lists
    bonds = bonds
    bond_features = np.array(bond_features, dtype=np.float32)

    # Build reverse map
    bond_map = {}
    for idx, (s, t) in enumerate(bonds):
        bond_map[(s, t)] = idx

    reverse_map = []
    for idx, (s, t) in enumerate(bonds):
        rev_idx = bond_map.get((t, s), idx)
        reverse_map.append(rev_idx)

    # Create graph dict
    graph_data = {
        'atom_features': torch.from_numpy(atom_features),
        'bond_features': torch.from_numpy(bond_features),
        'edge_index': torch.tensor(bonds, dtype=torch.long).t().contiguous() if bonds else torch.zeros((2, 0), dtype=torch.long),
        'reverse_map': torch.tensor(reverse_map, dtype=torch.long),
        'num_atoms': num_atoms,
        'periodic_edge_bonds': [],  # No periodic edges
    }

    # Add K-node if requested (features stored separately, virtual edges added in collate_batch)
    if use_knode:
        k_node_features = np.random.rand(K_FEAT_DIM).astype(np.float32)
        graph_data['k_node_features'] = torch.tensor(k_node_features, dtype=torch.float32).unsqueeze(0)  # [1, K_FEAT_DIM]
        graph_data['has_knode'] = True

    # Add placeholder target (zero)
    graph_data['target'] = torch.tensor([0.0], dtype=torch.float32)

    # Add placeholder psmiles
    graph_data['psmiles'] = f'random_{np.random.randint(1000000)}'

    return graph_data


def generate_random_graphs(num_graphs=300000, use_knode=True, output_path=None, seed=42):
    """
    Generate multiple random graphs and save to pickle file.

    Args:
        num_graphs: Number of graphs to generate
        use_knode: Whether to add K-node to graphs
        output_path: Path to save pickle file
        seed: Random seed

    Returns:
        List of graph dicts
    """
    print(f"\n{'='*60}")
    print(f"Generating {num_graphs} random graphs")
    print(f"{'='*60}")
    print(f"K-node: {use_knode}")
    print(f"Output: {output_path}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")

    np.random.seed(seed)

    graphs = []

    print("Generating graphs...")
    for i in range(num_graphs):
        if (i + 1) % 50000 == 0 or i == 0:
            print(f"  Generated {i+1}/{num_graphs} graphs...")

        graph = generate_random_graph(use_knode=use_knode)
        graphs.append(graph)

    print(f"Generated {len(graphs)} graphs")

    # Save to file
    if output_path is not None:
        print(f"\nSaving to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(graphs, f)
        print(f"Saved successfully")

    return graphs


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate random graphs')
    parser.add_argument('--num-graphs', type=int, default=300000, help='Number of graphs to generate')
    parser.add_argument('--no-knode', action='store_true', help='Disable K-node')
    parser.add_argument('--output', type=str, required=True, help='Output pickle file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    generate_random_graphs(
        num_graphs=args.num_graphs,
        use_knode=not args.no_knode,
        output_path=args.output,
        seed=args.seed,
    )
