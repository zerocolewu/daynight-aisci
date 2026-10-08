"""
PSPE (Polymer-Specific Positional Encoding) computation.
Computes Distance from Chain End (DCE) and Monomer Index (MI) features
from polymer molecular graphs.

These features capture chain-geometry signals at the node level:
- DCE: How close a node is to the nearest polymer chain end (0=at end, 1=in middle)
- MI: Which of the 3 repeat unit segments a node belongs to (0.0, 0.5, or 1.0)
"""

import torch
from collections import deque


def bfs_farthest(edge_index, num_nodes, start_node):
    """BFS from start_node, return (distances, farthest_node)."""
    adj = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        adj[u].append(v)
        adj[v].append(u)

    dist = [-1] * num_nodes
    dist[start_node] = 0
    q = deque([start_node])
    farthest_node = start_node

    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
                if dist[v] > dist[farthest_node]:
                    farthest_node = v

    return dist, farthest_node


def bfs_path(edge_index, num_nodes, start, end):
    """BFS from start to end, return shortest path as list of nodes."""
    adj = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        adj[u].append(v)
        adj[v].append(u)

    parent = [-1] * num_nodes
    parent[start] = start
    q = deque([start])
    found = False

    while q and not found:
        u = q.popleft()
        for v in adj[u]:
            if parent[v] == -1:
                parent[v] = u
                if v == end:
                    found = True
                    break
                q.append(v)

    if not found:
        return []

    path = []
    cur = end
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    path.reverse()
    return path


def find_backbone_path(edge_index, num_nodes):
    """
    Find the longest simple path (backbone) in the polymer graph using double-BFS.
    Returns (backbone_nodes_set, backbone_path_list, backbone_length).
    """
    if num_nodes <= 1:
        path = [0] if num_nodes == 1 else []
        return set(path), path, len(path)

    # First BFS: from node 0, find farthest node A
    _, node_a = bfs_farthest(edge_index, num_nodes, 0)

    # Second BFS: from node A, find farthest node B
    _, node_b = bfs_farthest(edge_index, num_nodes, node_a)

    # BFS path from A to B
    path = bfs_path(edge_index, num_nodes, node_a, node_b)

    return set(path), path, len(path)


def compute_pspe(edge_index, num_nodes):
    """
    Compute PSPE features for a polymer graph.

    Args:
        edge_index: torch.Tensor [2, E], undirected edge indices
        num_nodes: int, number of nodes in the graph

    Returns:
        dce: torch.Tensor [num_nodes], distance to chain end, normalized [0,1]
        mi: torch.Tensor [num_nodes], monomer index along backbone, {0.0, 0.5, 1.0}
        info: dict with backbone_length, valid flag
    """
    backbone_set, backbone_path, backbone_len = find_backbone_path(edge_index, num_nodes)

    dce = torch.zeros(num_nodes, dtype=torch.float)
    mi = torch.zeros(num_nodes, dtype=torch.float)
    info = {'backbone_length': backbone_len, 'valid': True}

    if backbone_len < 3:
        info['valid'] = False
        return dce, mi, info

    # --- DCE: Distance to Chain End ---
    # For each backbone node at position p:
    #   raw = min(p, backbone_len - 1 - p)
    #   dce = raw / max_possible
    max_possible = (backbone_len - 1) / 2.0

    for pos, node_id in enumerate(backbone_path):
        raw_dce = min(pos, backbone_len - 1 - pos)
        dce[node_id] = raw_dce / max_possible

    # Side-chain (non-backbone) atoms get dce=0 (left at default)

    # --- MI: Monomer Index ---
    segment_size = backbone_len / 3.0

    for pos, node_id in enumerate(backbone_path):
        segment_idx = min(int(pos / segment_size), 2)
        mi[node_id] = segment_idx / 2.0  # 0.0, 0.5, or 1.0

    # Side-chain atoms get mi=0 (left at default)

    return dce, mi, info
