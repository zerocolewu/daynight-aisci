"""
Copolymer variant of PTKGE for the EA/IP datasets.

It reuses the pretrained-able PTKGE encoder verbatim (atomic-number GIN +
periodic-table physics embedding + ECFP4 branch + fusion), but is fed a
CONNECTED A-B copolymer graph so messages cross the monomer interface, and
concatenates composition extras [rA, rB, type-one-hot] to the fused embedding.

Trained from scratch (no pretraining): PI1M is single-monomer, so the
contrastive prior does not transfer to the two-monomer copolymer graphs, and the
EA/IP corpora (~30k train each) are large enough to train the encoder directly.

Output is in normalized target space; the training loop handles denormalization.
"""
import torch
import torch.nn as nn

from model import PTKGEModel


class PTKGECopolymer(nn.Module):
    def __init__(self, extras_dim, num_layers=5, hidden_dim=256,
                 embedding_strategy="physics", embedding_dim=8, fp_dim=2048, dropout=0.1):
        super().__init__()
        # Reuse the exact PTKGE sub-modules (same architecture / parameter shapes).
        base = PTKGEModel(num_layers=num_layers, hidden_dim=hidden_dim,
                          embedding_strategy=embedding_strategy, embedding_dim=embedding_dim,
                          fp_dim=fp_dim, dropout=dropout, output_dim=1)
        self.gnn_encoder = base.gnn_encoder
        self.ecfp_encoder = base.ecfp_encoder
        self.fusion = base.fusion
        self.hidden_dim = hidden_dim

        # Head over [fused embedding, composition extras].
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + extras_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch):
        g = self.gnn_encoder(batch.z, batch.edge_index, batch.batch)
        f = self.ecfp_encoder(batch.fp)
        h = self.fusion(torch.cat([g, f], dim=1))
        h = torch.cat([h, batch.extras], dim=1)
        return self.head(h).squeeze(-1)
