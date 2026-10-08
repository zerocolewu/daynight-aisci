"""
SPSE (Simple Path Structural Encoding) for polymer graphs.

Precomputes simple path counts (k=1..4) for each edge and encodes them
via learnable log-transform parameters: f_k(x) = alpha_k * ln(1 + x) + beta_k.

Reference: The SPSE hypothesis posits that counting simple paths of different
lengths between edge endpoints captures polymer backbone connectivity patterns
(rings, linear chains, branches) that raw edge features miss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Path counting
# ---------------------------------------------------------------------------

def _count_paths_k1(num_edges: int) -> torch.Tensor:
    """k=1 paths: always 1 per edge (the edge itself)."""
    return torch.ones(num_edges, dtype=torch.float32)


def _count_paths_k2(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """k=2 paths: |N(u) ∩ N(v)| — common neighbors of u and v."""
    # Build adjacency list
    adj = [set() for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        adj[u].add(v)

    counts = []
    for u, v in edge_index.t().tolist():
        n_u = adj[u] - {v}
        n_v = adj[v] - {u}
        counts.append(len(n_u & n_v))
    return torch.tensor(counts, dtype=torch.float32)


def _count_paths_k3(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """k=3 paths: paths u→a→b→v with a,b distinct, not endpoints."""
    adj = [set() for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        adj[u].add(v)

    counts = []
    for u, v in edge_index.t().tolist():
        c = 0
        # For each a in N(u)\{v}
        for a in adj[u] - {v}:
            # For each b in N(a)\{u} (simple path: b≠u)
            for b in adj[a] - {u}:
                if b == a:  # should not happen but guard
                    continue
                if b != v and b in adj[v]:
                    c += 1
        counts.append(c)
    return torch.tensor(counts, dtype=torch.float32)


def _count_paths_k4(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """k=4 paths: paths u→a→b→c→v with all nodes distinct."""
    adj = [set() for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        adj[u].add(v)

    counts = []
    for u, v in edge_index.t().tolist():
        c = 0
        # For each a in N(u)\{v}
        for a in adj[u] - {v}:
            # For each b in N(a)\{u}
            for b in adj[a] - {u}:
                if b == a:
                    continue
                # For each c in N(b)\{a,u} (simple path constraint)
                for cc in adj[b] - {a, u}:
                    if cc == b:
                        continue
                    if cc != v and cc in adj[v]:
                        c += 1
        counts.append(c)
    return torch.tensor(counts, dtype=torch.float32)


def compute_spse_path_counts(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_k: int = 4,
) -> torch.Tensor:
    """Compute simple path counts for all edges.

    Args:
        edge_index: [2, num_edges] tensor
        num_nodes: total number of nodes
        max_k: maximum path length (default 4)

    Returns:
        [num_edges, max_k] tensor of path counts
    """
    k1 = _count_paths_k1(edge_index.shape[1])
    k2 = _count_paths_k2(edge_index, num_nodes)
    k3 = _count_paths_k3(edge_index, num_nodes)
    k4 = _count_paths_k4(edge_index, num_nodes) if max_k >= 4 else torch.zeros_like(k1)
    return torch.stack([k1, k2, k3, k4], dim=1)


# ---------------------------------------------------------------------------
# Learnable SPSE encoder
# ---------------------------------------------------------------------------

class SPSEEncoder(nn.Module):
    """Learnable log-transform encoder for SPSE path counts.

    f_k(x) = alpha_k * ln(1 + x) + beta_k
    where alpha_k, beta_k are learnable scalar parameters.

    This produces 4-dimensional edge encoding vectors from 4 path count values.
    """

    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim

        # Learnable alpha and beta per path length k
        self.alpha = nn.Parameter(torch.ones(embed_dim) * 0.1)
        self.beta = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, path_counts: torch.Tensor) -> torch.Tensor:
        """Encode path counts.

        Args:
            path_counts: [num_edges, embed_dim] raw path counts

        Returns:
            [num_edges, embed_dim] encoded features
        """
        return self.alpha.unsqueeze(0) * torch.log1p(path_counts) + self.beta.unsqueeze(0)

    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (alpha, beta) as numpy-friendly values."""
        return self.alpha.detach().cpu(), self.beta.detach().cpu()


# ---------------------------------------------------------------------------
# Precompute SPSE for all graphs
# ---------------------------------------------------------------------------

def precompute_spse_for_graphs(
    graphs: list,
    encoder: Optional[SPSEEncoder] = None,
    max_k: int = 4,
) -> list:
    """Precompute SPSE path counts for all graphs and optionally encode them.

    For each graph, computes raw path counts and stores as graph.spse_raw.
    If encoder is provided, also computes encoded features and extends edge_attr.

    Args:
        graphs: list of PyG Data objects
        encoder: SPSEEncoder module (if None, only raw counts stored)
        max_k: max path length

    Returns:
        Modified graphs list (in-place with spse_raw attribute added)
    """
    for g in graphs:
        num_edges = g.edge_index.shape[1]
        num_nodes = g.x.shape[0]
        raw_counts = compute_spse_path_counts(g.edge_index, num_nodes, max_k)
        g.spse_raw = raw_counts  # [num_edges, max_k]

        if encoder is not None:
            encoded = encoder(raw_counts)
            # Concatenate to existing edge_attr
            if g.edge_attr is not None:
                g.edge_attr = torch.cat([g.edge_attr, encoded], dim=1)
            else:
                g.edge_attr = encoded

    return graphs
