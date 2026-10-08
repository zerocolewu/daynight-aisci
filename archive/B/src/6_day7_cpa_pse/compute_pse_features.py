"""
Compute PSE (Periodicity-Structural Encoding) features for all Tg polymer graphs.

Features:
  1. modular_position  [0,1]: atom's normalized position within its RU template
  2. periodic_distance [0,1]: shortest-path distance to nearest structurally equivalent
     atom in adjacent RU, normalized
  3. is_backbone       {0,1}: whether atom is on the backbone chain

Algorithm:
  Step 1: Find backbone atoms via double-BFS (same as PSPE backbone detection)
  Step 2: Compute modular_position for backbone + side-chain inheritance
  Step 3: Compute periodic_distance for backbone atoms across adjacent RUs
  Step 4: Compute is_backbone binary flag

Output: src/alldataset/tg_pse_features.pt
  { 'train': [tensor(N_i,3), ...], 'val': [...], 'test': [...] }
"""
import os
import pickle
import sys
import warnings
from collections import deque

import numpy as np
import torch

warnings.filterwarnings('ignore')

DATA_PATH = "/home/home_public/aisci/AIscientist/src/4_day4_selfgnn_polymer/data/tg_cache/tg_graphs_3ru_42dim.pkl"
OUTPUT_PATH = "/home/home_public/aisci/AIscientist/src/alldataset/tg_pse_features.pt"


def build_adjacency(edge_index, num_nodes):
    """Build adjacency list from edge_index."""
    adj = {i: [] for i in range(num_nodes)}
    ei = edge_index.numpy()
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        adj[u].append(v)
        adj[v].append(u)
    return adj


def bfs_farthest(adj, start):
    """BFS from start, return (farthest_node, distances dict, num_visited)."""
    dist = {start: 0}
    q = deque([start])
    visited = {start}
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                dist[v] = dist[u] + 1
                q.append(v)
    if not dist:
        return start, dist, 0
    farthest = max(dist, key=dist.get)
    return farthest, dist, len(dist)


def bfs_path(adj, start, target):
    """BFS from start to target, return shortest path list."""
    if start == target:
        return [start]
    parent = {start: None}
    q = deque([start])
    visited = {start}
    while q:
        u = q.popleft()
        if u == target:
            break
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                q.append(v)
    # Reconstruct path
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path


def shortest_distance(adj, start, target):
    """Compute shortest path distance between two nodes using BFS."""
    if start == target:
        return 0
    dist = {start: 0}
    q = deque([start])
    visited = {start}
    while q:
        u = q.popleft()
        if u == target:
            return dist[u]
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                dist[v] = dist[u] + 1
                q.append(v)
    return float('inf')


def compute_pse_for_graph(adj, edge_index, num_nodes, n_ru=3):
    """
    Compute PSE features for a single polymer graph.

    Args:
        adj: adjacency list
        edge_index: PyG edge_index tensor (for reference)
        num_nodes: number of nodes
        n_ru: number of repeat units (3 for 3-RU multimers)

    Returns:
        pse: numpy array [num_nodes, 3] with (modular_position, periodic_distance, is_backbone)
    """
    pse = np.zeros((num_nodes, 3), dtype=np.float32)

    # Step 1: Find backbone via double-BFS
    # Start BFS from node 0
    node_a, _, visited_count = bfs_farthest(adj, 0)
    if visited_count == 0:
        return pse

    # Second BFS from node_a to find node_b
    node_b, _, _ = bfs_farthest(adj, node_a)

    # Find the shortest path from node_a to node_b (this is the backbone)
    backbone_path = bfs_path(adj, node_a, node_b)
    backbone_set = set(backbone_path)
    backbone_length = len(backbone_path)

    if backbone_length < 3:
        # Too short for meaningful PSE
        return pse

    # Step 2: Compute backbone position mapping
    pos_map = {}  # node_idx -> position along backbone
    for p, node_idx in enumerate(backbone_path):
        pos_map[node_idx] = p

    # Step 4: is_backbone (column 2)
    for node_idx in backbone_set:
        pse[node_idx, 2] = 1.0

    # Split backbone into n_ru equal segments using integer division
    base_seg_len = backbone_length // n_ru
    remainder = backbone_length % n_ru
    # Segment lengths: first 'remainder' segments get base_seg_len+1, rest get base_seg_len
    seg_lengths = []
    for i in range(n_ru):
        seg_lengths.append(base_seg_len + (1 if i < remainder else 0))
    # Build segment boundaries
    seg_starts = [0]
    for i in range(1, n_ru):
        seg_starts.append(seg_starts[i-1] + seg_lengths[i-1])

    # modular_position for backbone atoms (column 0)
    for node_idx in backbone_path:
        p = pos_map[node_idx]
        # Find which segment this position belongs to
        segment_idx = 0
        for si in range(1, n_ru):
            if p >= seg_starts[si]:
                segment_idx = si
        seg_start = seg_starts[segment_idx]
        seg_length = seg_lengths[segment_idx]
        pos_in_seg = p - seg_start
        if seg_length > 1:
            pse[node_idx, 0] = pos_in_seg / (seg_length - 1)
        else:
            pse[node_idx, 0] = 0.0
        pse[node_idx, 0] = np.clip(pse[node_idx, 0], 0.0, 1.0)

    # Step 3: periodic_distance for backbone atoms (column 1)
    # For each backbone atom in RU k, find closest modular_position match in RU k+1
    # (or RU k-1 for last RU)
    max_periodic_dist = 0.0
    periodic_dists = {}

    # Group backbone atoms by segment (using same integer division)
    segment_atoms = [[] for _ in range(n_ru)]
    for node_idx in backbone_path:
        p = pos_map[node_idx]
        segment_idx = 0
        for si in range(1, n_ru):
            if p >= seg_starts[si]:
                segment_idx = si
        segment_atoms[segment_idx].append((node_idx, pse[node_idx, 0]))

    for ru in range(n_ru):
        if ru < n_ru - 1:
            adj_ru = ru + 1
        else:
            adj_ru = ru - 1

        for node_idx, mp in segment_atoms[ru]:
            # Find closest modular_position match in adjacent RU
            best_dist = float('inf')
            best_match = None
            for other_idx, other_mp in segment_atoms[adj_ru]:
                if abs(mp - other_mp) < 0.05:
                    d = shortest_distance(adj, node_idx, other_idx)
                    if d < best_dist:
                        best_dist = d
                        best_match = other_idx

            if best_dist < float('inf'):
                periodic_dists[node_idx] = best_dist
                max_periodic_dist = max(max_periodic_dist, best_dist)

    # Normalize periodic distances
    if max_periodic_dist > 0:
        for node_idx, dist in periodic_dists.items():
            pse[node_idx, 1] = dist / max_periodic_dist

    # Step 2 continued: side-chain atoms inherit modular_position from nearest backbone atom
    # Find nearest backbone atom via BFS for each non-backbone atom
    for node_idx in range(num_nodes):
        if node_idx in backbone_set:
            continue
        # Find nearest backbone atom via BFS
        dist_map = {node_idx: 0}
        q = deque([node_idx])
        visited = {node_idx}
        found_backbone = None
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    dist_map[v] = dist_map[u] + 1
                    if v in backbone_set:
                        found_backbone = v
                        break
                    q.append(v)
            if found_backbone is not None:
                break
        if found_backbone is not None:
            pse[node_idx, 0] = pse[found_backbone, 0]

    return pse


def compute_all_pse():
    """Compute PSE features for all graphs and save to disk."""
    print(f"Loading graphs from {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    train_graphs = data['train_graphs']
    val_graphs = data['val_graphs']
    test_graphs = data['test_graphs']

    total = len(train_graphs) + len(val_graphs) + len(test_graphs)
    print(f"Total graphs: {total} (train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)})")

    results = {}
    all_counts = 0

    for split_name, graphs in [('train', train_graphs), ('val', val_graphs), ('test', test_graphs)]:
        pse_list = []
        for i, g in enumerate(graphs):
            num_nodes = g.x.shape[0]
            edge_index = g.edge_index
            adj = build_adjacency(edge_index, num_nodes)
            pse = compute_pse_for_graph(adj, edge_index, num_nodes, n_ru=3)
            pse_list.append(torch.tensor(pse, dtype=torch.float32))
            all_counts += 1
            if all_counts % 500 == 0:
                print(f"  Processed {all_counts}/{total} graphs...")
        results[split_name] = pse_list
        print(f"  {split_name}: {len(pse_list)} graphs processed")

    # Compute summary statistics
    all_pse = torch.cat([torch.cat(results['train'], dim=0),
                          torch.cat(results['val'], dim=0),
                          torch.cat(results['test'], dim=0)], dim=0)

    print(f"\nSaving PSE features to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(results, OUTPUT_PATH)

    print(f"\n=== PSE Feature Statistics ===")
    print(f"Total atoms: {all_pse.shape[0]}")
    print(f"modular_position:  mean={all_pse[:,0].mean():.4f}, std={all_pse[:,0].std():.4f}")
    print(f"                    min={all_pse[:,0].min():.4f}, max={all_pse[:,0].max():.4f}")
    print(f"periodic_distance: mean={all_pse[:,1].mean():.4f}, std={all_pse[:,1].std():.4f}")
    print(f"                    min={all_pse[:,1].min():.4f}, max={all_pse[:,1].max():.4f}")
    print(f"is_backbone:       mean={all_pse[:,2].mean():.4f} (fraction of backbone atoms)")
    print(f"\nNon-zero periodic_distance atoms: {(all_pse[:,1] > 0).sum().item()}")
    print(f"Backbone atoms: {(all_pse[:,2] > 0).sum().item()}")
    print(f"\nDone!")

    return results


if __name__ == "__main__":
    compute_all_pse()
