"""
Model variants for LSPE-GAT polymer Tg prediction.

Variants:
  1. GATBaseline — Standard 4-layer GAT with rich features (reference)
  2. GATStaticRWPE — GAT with precomputed RWPE concatenated to input
  3. LSPEGAT — Decoupled structural/positional updates (LSPE-style)
  4. LSPEGATEDGE — LSPE-GAT with edge gating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool, MessagePassing
from torch_geometric.utils import softmax


# ==============================================================================
# Baseline GAT
# ==============================================================================

class GATBaseline(nn.Module):
    """4-layer GAT with global mean pooling + 2-layer MLP predictor.

    Mirrors the baseline from 2_day2_spse_maxgat/baseline/model.py
    but with customizable node_dim and edge_dim.
    """

    def __init__(
        self,
        node_dim: int = 47,
        edge_dim: int = 10,
        hidden_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % heads == 0
        out_per_head = hidden_dim // heads

        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.heads = heads

        use_edge = edge_dim > 0

        self.conv1 = GATConv(
            node_dim, out_per_head, heads=heads,
            concat=True, dropout=dropout,
            edge_dim=edge_dim if use_edge else None,
        )
        self.conv2 = GATConv(
            hidden_dim, out_per_head, heads=heads,
            concat=True, dropout=dropout,
            edge_dim=edge_dim if use_edge else None,
        )
        self.conv3 = GATConv(
            hidden_dim, out_per_head, heads=heads,
            concat=True, dropout=dropout,
            edge_dim=edge_dim if use_edge else None,
        )
        self.conv4 = GATConv(
            hidden_dim, out_per_head, heads=heads,
            concat=True, dropout=dropout,
            edge_dim=edge_dim if use_edge else None,
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = F.dropout(F.elu(self.conv1(x, edge_index, edge_attr=edge_attr)), p=self.dropout, training=self.training)
        x = F.dropout(F.elu(self.conv2(x, edge_index, edge_attr=edge_attr)), p=self.dropout, training=self.training)
        x = F.dropout(F.elu(self.conv3(x, edge_index, edge_attr=edge_attr)), p=self.dropout, training=self.training)
        x = F.dropout(F.elu(self.conv4(x, edge_index, edge_attr=edge_attr)), p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.predictor(x).squeeze(-1)


# ==============================================================================
# Treatment A: Static RWPE
# ==============================================================================

class GATStaticRWPE(nn.Module):
    """GAT with precomputed RWPE concatenated to node features.

    x_i' = [x_i; RWPE_i] → standard 4-layer GAT.

    The RWPE tensor is provided as a node attribute in each graph.
    """

    def __init__(
        self,
        node_dim: int = 47,
        rwpe_dim: int = 16,
        edge_dim: int = 10,
        hidden_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % heads == 0
        out_per_head = hidden_dim // heads

        in_dim = node_dim + rwpe_dim  # 47 + 16 = 63
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        use_edge = edge_dim > 0

        self.conv1 = GATConv(
            in_dim, out_per_head, heads=heads,
            concat=True, dropout=dropout,
            edge_dim=edge_dim if use_edge else None,
        )
        self.conv2 = GATConv(
            hidden_dim, out_per_head, heads=heads,
            concat=True, dropout=dropout,
            edge_dim=edge_dim if use_edge else None,
        )
        self.conv3 = GATConv(
            hidden_dim, out_per_head, heads=heads,
            concat=True, dropout=dropout,
            edge_dim=edge_dim if use_edge else None,
        )
        self.conv4 = GATConv(
            hidden_dim, out_per_head, heads=heads,
            concat=True, dropout=dropout,
            edge_dim=edge_dim if use_edge else None,
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, edge_index, batch, edge_attr=None, rwpe=None):
        if rwpe is not None:
            x = torch.cat([x, rwpe], dim=-1)
        x = F.dropout(F.elu(self.conv1(x, edge_index, edge_attr=edge_attr)), p=self.dropout, training=self.training)
        x = F.dropout(F.elu(self.conv2(x, edge_index, edge_attr=edge_attr)), p=self.dropout, training=self.training)
        x = F.dropout(F.elu(self.conv3(x, edge_index, edge_attr=edge_attr)), p=self.dropout, training=self.training)
        x = F.dropout(F.elu(self.conv4(x, edge_index, edge_attr=edge_attr)), p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.predictor(x).squeeze(-1)


# ==============================================================================
# LSPE Helper: Neighbor aggregation for positional features
# ==============================================================================

class MeanNeighborAgg(MessagePassing):
    """Compute mean of neighbor positional features."""
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, p, edge_index):
        return self.propagate(edge_index, x=p)


# ==============================================================================
# LSPE Positional Update MLP
# ==============================================================================

class PositionalUpdate(nn.Module):
    """Positional feature update: p_new = tanh(Linear2(tanh(Linear1([p; mean(p_neighbors)])))).

    Input: [p_i; mean_j p_j] → 128-dim
    Hidden: 64-dim
    Output: 64-dim
    No batch norm per LSPE specification.
    """

    def __init__(self, pos_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * pos_dim, pos_dim),
            nn.Tanh(),
            nn.Linear(pos_dim, pos_dim),
            nn.Tanh(),
        )

    def forward(self, p, p_neighbor_mean):
        inp = torch.cat([p, p_neighbor_mean], dim=-1)
        return self.mlp(inp)


# ==============================================================================
# Treatment B: LSPE-GAT
# ==============================================================================

class LSPEGAT(nn.Module):
    """Decoupled structural/positional updates with GATv2 backbone.

    - h_i^0: Linear(node_feat) → hidden_dim
    - p_i^0: Linear(RWPE) → pos_dim
    - Each layer:
        * Structural: GATv2Conv([h; p]) → LayerNorm → ReLU → Dropout
        * Positional: p = tanh(MLP([p; mean_j p_j]))
    - Readout: concat[mean_pool(h^L); mean_pool(p^L)] → MLP predictor
    """

    def __init__(
        self,
        node_dim: int = 47,
        rwpe_dim: int = 16,
        edge_dim: int = 10,
        hidden_dim: int = 256,
        pos_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
        num_layers: int = 4,
    ):
        super().__init__()
        assert hidden_dim % heads == 0
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        self.heads = heads
        self.dropout = dropout
        self.num_layers = num_layers
        out_per_head = hidden_dim // heads

        # Initial projections
        self.h_proj = nn.Linear(node_dim, hidden_dim)
        self.p_proj = nn.Linear(rwpe_dim, pos_dim)

        # Neighbor aggregation for p
        self.p_agg = MeanNeighborAgg()

        # GATv2 layers: input is [h; p] → hidden_dim + pos_dim
        gat_in_dim = hidden_dim + pos_dim
        use_edge = edge_dim > 0

        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.p_updates = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    gat_in_dim, out_per_head, heads=heads,
                    concat=True, dropout=dropout,
                    edge_dim=edge_dim if use_edge else None,
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.p_updates.append(PositionalUpdate(pos_dim))

        # Predictor: [mean_pool(h); mean_pool(p)] → 320 → 128 → 1
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + pos_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, edge_index, batch, edge_attr=None, rwpe=None, return_diagnostics=False):
        """
        Args:
            x: Node features [N_total, node_dim]
            edge_index: Edge indices [2, E]
            batch: Batch assignment [N_total]
            edge_attr: Edge features [E, edge_dim]
            rwpe: Random walk PE [N_total, rwpe_dim]
            return_diagnostics: If True, return (pred, diagnostics_dict)
        """
        diag = {} if return_diagnostics else None

        # Initial projections
        h = self.h_proj(x)  # [N, hidden_dim]
        if rwpe is not None:
            p = self.p_proj(rwpe)  # [N, pos_dim]
        else:
            p = torch.zeros(x.shape[0], self.pos_dim, device=x.device)

        # Track p norms across layers
        if return_diagnostics:
            p_norms = [p.norm(dim=-1).mean().item()]

        # Layers
        for layer_idx in range(self.num_layers):
            # --- Structural update ---
            h_p_cat = torch.cat([h, p], dim=-1)  # [N, hidden_dim + pos_dim]
            h_new = self.convs[layer_idx](h_p_cat, edge_index, edge_attr=edge_attr)
            h_new = self.layer_norms[layer_idx](h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # Residual

            # --- Positional update ---
            p_neighbor_mean = self.p_agg(p, edge_index)
            p_new = self.p_updates[layer_idx](p, p_neighbor_mean)
            p = p_new  # [N, pos_dim]

            if return_diagnostics:
                p_norms.append(p.norm(dim=-1).mean().item())

        # Readout
        h_graph = global_mean_pool(h, batch)
        p_graph = global_mean_pool(p, batch)
        out = self.predictor(torch.cat([h_graph, p_graph], dim=-1)).squeeze(-1)

        if return_diagnostics:
            diag['h'] = h
            diag['p'] = p
            diag['p_norms'] = p_norms
            diag['edge_index'] = edge_index
            diag['batch'] = batch
            return out, diag

        return out


# ==============================================================================
# Custom GATv2 with Edge Gate for Treatment C
# ==============================================================================

class GATv2EdgeGateConv(MessagePassing):
    """GATv2 with learnable edge gates.

    Edge gate: g_ij = σ(Linear([h_i; h_j; p_i; p_j; e_ij]))
    Output dim = num_heads (one scalar per head)
    Modulates attention: α'_ij = α_ij ⊙ g_ij
    """

    def __init__(
        self,
        h_dim: int,
        p_dim: int,
        edge_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.h_dim = h_dim
        self.p_dim = p_dim
        self.heads = heads
        self.dropout = dropout
        self.concat = concat
        self.out_channels = h_dim // heads  # Per-head output dim

        # Full input dim for [h; p]
        in_dim = h_dim + p_dim

        # GATv2 linear projections
        self.lin_l = nn.Linear(in_dim, heads * self.out_channels, bias=False)
        self.lin_r = nn.Linear(in_dim, heads * self.out_channels, bias=False)
        self.att = Parameter(torch.empty(1, heads, self.out_channels))

        # Edge encoder
        self.lin_edge = nn.Linear(edge_dim, heads * self.out_channels, bias=False) if edge_dim > 0 else None

        # Bias
        self.bias = Parameter(torch.empty(heads * self.out_channels))

        # Edge gate network: [h_i; h_j; p_i; p_j; e_ij]
        # Total input: 2 * h_dim + 2 * p_dim + edge_dim
        gate_input_dim = 2 * h_dim + 2 * p_dim + edge_dim
        self.edge_gate = nn.Sequential(
            nn.Linear(gate_input_dim, heads),
            nn.Sigmoid(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)

    def forward(self, h, p, edge_index, edge_attr=None, return_attention=False):
        # h: [N, h_dim], p: [N, p_dim]
        out = self.propagate(
            edge_index, h=h, p=p, edge_attr=edge_attr,
        )
        out = out + self.bias

        if return_attention:
            return out, getattr(self, '_last_alpha', None)

        return out

    def message(self, h_i, h_j, p_i, p_j, edge_attr, index, ptr, size_i):
        # h_i, h_j: [E, h_dim]
        # p_i, p_j: [E, p_dim]
        E = h_i.shape[0]

        # Concatenate [h; p] for attention
        x_i = torch.cat([h_i, p_i], dim=-1)  # [E, h_dim + p_dim]
        x_j = torch.cat([h_j, p_j], dim=-1)  # [E, h_dim + p_dim]

        # GATv2 attention
        x_i_lin = self.lin_l(x_i).view(-1, self.heads, self.out_channels)  # [E, H, D]
        x_j_lin = self.lin_r(x_j).view(-1, self.heads, self.out_channels)  # [E, H, D]

        # alpha = LeakyReLU((x_i_lin + x_j_lin) · att)
        alpha = (x_i_lin + x_j_lin) * self.att  # [E, H, D]
        alpha = F.leaky_relu(alpha, 0.2).sum(dim=-1)  # [E, H]

        if edge_attr is not None and self.lin_edge is not None:
            edge_emb = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            alpha = alpha + (edge_emb * self.att).sum(dim=-1)

        # Softmax over neighbors
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # --- Edge gate ---
        gate_input = torch.cat([h_i, h_j, p_i, p_j, edge_attr], dim=-1)  # [E, gate_input_dim]
        gate = self.edge_gate(gate_input)  # [E, H]

        # Apply gate: α'_ij = α_ij * g_ij
        alpha = alpha * gate

        # Store for attention visualization
        self._last_alpha = alpha.detach()

        # Weighted message
        out = x_j_lin * alpha.unsqueeze(-1)  # [E, H, D]
        if self.concat:
            out = out.view(E, -1)  # [E, H*D]
        else:
            out = out.mean(dim=1)  # [E, D]

        return out


# ==============================================================================
# Treatment C: LSPE-GAT + Edge Gate
# ==============================================================================

class LSPEGATEDGE(nn.Module):
    """LSPE-GAT with learnable edge gating.

    Same as LSPEGAT, but each GAT layer includes edge gates:
    g_ij = σ(Linear([h_i; h_j; p_i; p_j; e_ij]))
    α'_ij = α_ij ⊙ g_ij
    """

    def __init__(
        self,
        node_dim: int = 47,
        rwpe_dim: int = 16,
        edge_dim: int = 10,
        hidden_dim: int = 256,
        pos_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
        num_layers: int = 4,
    ):
        super().__init__()
        assert hidden_dim % heads == 0
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        self.heads = heads
        self.dropout = dropout
        self.num_layers = num_layers

        # Initial projections
        self.h_proj = nn.Linear(node_dim, hidden_dim)
        self.p_proj = nn.Linear(rwpe_dim, pos_dim)

        # Neighbor aggregation for p
        self.p_agg = MeanNeighborAgg()

        # GATv2 layers with edge gates
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.p_updates = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                GATv2EdgeGateConv(
                    h_dim=hidden_dim,
                    p_dim=pos_dim,
                    edge_dim=edge_dim if edge_dim > 0 else 0,
                    heads=heads,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.p_updates.append(PositionalUpdate(pos_dim))

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + pos_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, edge_index, batch, edge_attr=None, rwpe=None, return_diagnostics=False):
        diag = {} if return_diagnostics else None

        # Initial projections
        h = self.h_proj(x)  # [N, hidden_dim]
        if rwpe is not None:
            p = self.p_proj(rwpe)  # [N, pos_dim]
        else:
            p = torch.zeros(x.shape[0], self.pos_dim, device=x.device)

        if return_diagnostics:
            p_norms = [p.norm(dim=-1).mean().item()]
            all_attentions = []

        for layer_idx in range(self.num_layers):
            # Structural update with edge gate
            h_new, attn = self.convs[layer_idx](h, p, edge_index, edge_attr=edge_attr, return_attention=True)
            h_new = self.layer_norms[layer_idx](h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # Residual

            # Positional update
            p_neighbor_mean = self.p_agg(p, edge_index)
            p_new = self.p_updates[layer_idx](p, p_neighbor_mean)
            p = p_new

            if return_diagnostics:
                p_norms.append(p.norm(dim=-1).mean().item())
                if attn is not None:
                    all_attentions.append(attn)

        # Readout
        h_graph = global_mean_pool(h, batch)
        p_graph = global_mean_pool(p, batch)
        out = self.predictor(torch.cat([h_graph, p_graph], dim=-1)).squeeze(-1)

        if return_diagnostics:
            diag['h'] = h
            diag['p'] = p
            diag['p_norms'] = p_norms
            diag['edge_index'] = edge_index
            diag['batch'] = batch
            if all_attentions:
                diag['attentions'] = all_attentions
            return out, diag

        return out


# ==============================================================================
# Treatment E: SkipRWPE-GAT — Layer-wise RWPE Injection
# ==============================================================================

class SkipRWPE_GAT(nn.Module):
    """GAT with static RWPE re-injected at every layer as a skip connection.

    Instead of learning positional updates (LSPE), this model simply
    re-concatenates the ORIGINAL static RWPE at each GAT layer input.
    This prevents RWPE signal from being "washed out" through deep
    message passing.

    At each layer l:
      h_in = [h^l; RWPE_i]  (re-concatenate original RWPE)
      h_new = GATConv(h_in) -> LayerNorm -> ReLU -> Dropout
      h^{l+1} = h^l + h_new  (residual)

    Architecture:
      Input: [x; RWPE] -> Linear -> hidden_dim (= h^0)
      For each layer:
        [h^l; RWPE] -> GATConv -> LayerNorm -> ReLU -> Dropout -> +residual -> h^{l+1}
      Readout: global_mean_pool(h^L) -> MLP predictor
    """

    def __init__(
        self,
        node_dim: int = 47,
        rwpe_dim: int = 16,
        edge_dim: int = 10,
        hidden_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
        num_layers: int = 4,
    ):
        super().__init__()
        assert hidden_dim % heads == 0
        self.hidden_dim = hidden_dim
        self.rwpe_dim = rwpe_dim
        self.heads = heads
        self.dropout = dropout
        self.num_layers = num_layers
        out_per_head = hidden_dim // heads

        # Input projection: [node_feat; rwpe] -> hidden_dim
        self.input_proj = nn.Linear(node_dim + rwpe_dim, hidden_dim)

        # GAT layers: each takes [h; rwpe] as input
        use_edge = edge_dim > 0
        conv_in_dim = hidden_dim + rwpe_dim

        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                GATConv(
                    conv_in_dim, out_per_head, heads=heads,
                    concat=True, dropout=dropout,
                    edge_dim=edge_dim if use_edge else None,
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Predictor: hidden_dim -> 128 -> 1
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, edge_index, batch, edge_attr=None, rwpe=None,
                return_diagnostics=False):
        """
        Args:
            x: Node features [N_total, node_dim]
            edge_index: Edge indices [2, E]
            batch: Batch assignment [N_total]
            edge_attr: Edge features [E, edge_dim]
            rwpe: Random walk PE [N_total, rwpe_dim]
            return_diagnostics: If True, returns (pred, diagnostic_dict)
        """
        diag = {} if return_diagnostics else None

        if rwpe is None:
            rwpe = torch.zeros(x.shape[0], self.rwpe_dim, device=x.device)

        # Initial projection
        h = F.relu(self.input_proj(torch.cat([x, rwpe], dim=-1)))
        h = F.dropout(h, p=self.dropout, training=self.training)

        if return_diagnostics:
            rwpe_similarities = []

        for layer_idx in range(self.num_layers):
            # Re-concatenate the ORIGINAL static RWPE at each layer
            h_in = torch.cat([h, rwpe], dim=-1)

            h_new = self.convs[layer_idx](h_in, edge_index, edge_attr=edge_attr)
            h_new = self.layer_norms[layer_idx](h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # Residual connection

            if return_diagnostics:
                # Track cosine similarity between RWPE projection and hidden state
                sim = F.cosine_similarity(
                    h.mean(dim=-1, keepdim=True),
                    rwpe.mean(dim=-1, keepdim=True),
                    dim=-1
                ).mean().item()
                rwpe_similarities.append(sim)

        h_graph = global_mean_pool(h, batch)
        out = self.predictor(h_graph).squeeze(-1)

        if return_diagnostics:
            diag['h'] = h
            diag['rwpe_similarities'] = rwpe_similarities
            diag['edge_index'] = edge_index
            diag['batch'] = batch
            return out, diag

        return out


# ==============================================================================
# Factory
# ==============================================================================

def create_model(treatment: str, node_dim: int = 47, rwpe_dim: int = 16,
                 edge_dim: int = 10, hidden_dim: int = 256, pos_dim: int = 64,
                 heads: int = 4, dropout: float = 0.1):
    """Create model for a given treatment."""
    if treatment == 'baseline':
        return GATBaseline(
            node_dim=node_dim, edge_dim=edge_dim,
            hidden_dim=hidden_dim, heads=heads, dropout=dropout,
        )
    elif treatment in ('static_rwpe_k16',):
        # Alias for static_rwpe with k=16 (backward compat)
        return GATStaticRWPE(
            node_dim=node_dim, rwpe_dim=rwpe_dim, edge_dim=edge_dim,
            hidden_dim=hidden_dim, heads=heads, dropout=dropout,
        )
    elif treatment in ('static_rwpe_k24', 'static_rwpe_k32'):
        return GATStaticRWPE(
            node_dim=node_dim, rwpe_dim=rwpe_dim, edge_dim=edge_dim,
            hidden_dim=hidden_dim, heads=heads, dropout=dropout,
        )
    elif treatment in ('skip_rwpe_k16', 'skip_rwpe_k24', 'skip_rwpe_k32'):
        return SkipRWPE_GAT(
            node_dim=node_dim, rwpe_dim=rwpe_dim, edge_dim=edge_dim,
            hidden_dim=hidden_dim, heads=heads, dropout=dropout,
        )
    elif treatment == 'lspe_gat':
        return LSPEGAT(
            node_dim=node_dim, rwpe_dim=rwpe_dim, edge_dim=edge_dim,
            hidden_dim=hidden_dim, pos_dim=pos_dim, heads=heads, dropout=dropout,
        )
    elif treatment == 'lspe_gat_edge':
        return LSPEGATEDGE(
            node_dim=node_dim, rwpe_dim=rwpe_dim, edge_dim=edge_dim,
            hidden_dim=hidden_dim, pos_dim=pos_dim, heads=heads, dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown treatment: {treatment}")
