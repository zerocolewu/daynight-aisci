"""
Reproduce the PTKGE contrastive pretraining on PI1M (physics embeddings).

STATUS: the pretrained checkpoint `checkpoints/pretrained_physics.pt` is already
provided and is what the fine-tuning runs load. This script exists for full
reproducibility -- rerun it only if you want to regenerate the checkpoint from
scratch (~1.5-2 h on one GPU over the full PI1M corpus).

Recipe (ptkge_scaled): InfoNCE contrastive learning where ECFP4 Tanimoto
similarity defines positive pairs; physics element-identity embedding; GIN
encoder; AdamW lr 5e-5 wd 1e-5; cosine schedule; 30 epochs; temperature 0.05.
"""
import os
import sys
import math
import random
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from model import PTKGEModel, ContrastiveLoss, compute_ecfp4_fingerprint, compute_similarity_matrix
import data_utils as DU

PRETRAIN_EPOCHS = 30
PRETRAIN_BATCH = 64
PRETRAIN_LR = 5e-5
PRETRAIN_TEMP = 0.05
PI1M_SAMPLE = 990000


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_pretrain_dataset(smiles_list, fp_dim=2048):
    graphs, fps = [], []
    for smi in smiles_list:
        mol = DU.polymer_smiles_to_mol(smi)
        if mol is None:
            continue
        gr = DU.mol_to_atom_graph(mol)
        if gr is None:
            continue
        z, edge_index = gr
        fp = compute_ecfp4_fingerprint(str(smi), n_bits=fp_dim)
        if fp.sum() == 0:
            continue
        d = gr.__class__()  # not used; build a Data directly
        from torch_geometric.data import Data
        d = Data(z=z, edge_index=edge_index, num_nodes=int(z.numel()))
        d.fp = torch.tensor(fp, dtype=torch.float)
        graphs.append(d); fps.append(fp)
    return graphs, np.asarray(fps, dtype=np.float32)


def collate(batch):
    graphs, fps = zip(*batch)
    return Batch.from_data_list(graphs), torch.tensor(np.asarray(fps), dtype=torch.float32)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    md = dict(C.MODEL_DEFAULTS)

    df = pd.read_csv(os.path.join(C.DATASET_DIR, "PI1M.csv"))
    smis = df["SMILES"].tolist()
    if len(smis) > PI1M_SAMPLE:
        smis = df.sample(n=PI1M_SAMPLE, random_state=42)["SMILES"].tolist()
    print(f"PI1M: featurizing {len(smis)} polymers for pretraining ...", flush=True)
    graphs, fps = build_pretrain_dataset(smis, fp_dim=md["fp_dim"])
    print(f"PI1M valid pretrain samples: {len(graphs)}", flush=True)

    loader = DataLoader(list(zip(graphs, fps)), batch_size=PRETRAIN_BATCH,
                        shuffle=True, collate_fn=collate, num_workers=4, pin_memory=True)
    model = PTKGEModel(**md).to(device)
    optimizer = AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS)
    criterion = ContrastiveLoss(temperature=PRETRAIN_TEMP)

    best = float("inf")
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train(); total = 0.0; nb = 0
        for graphs_b, fps_b in loader:
            graphs_b = graphs_b.to(device); fps_b = fps_b.to(device)
            with torch.no_grad():
                sim = torch.from_numpy(compute_similarity_matrix(fps_b.cpu().numpy())).to(device)
            emb = model.get_graph_embedding(graphs_b.z, graphs_b.edge_index,
                                            graphs_b.batch, fps_b)
            loss = criterion(emb, sim)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item(); nb += 1
        scheduler.step()
        avg = total / max(nb, 1)
        print(f"pretrain ep{epoch:2d}/{PRETRAIN_EPOCHS} loss={avg:.4f}", flush=True)
        if avg < best:
            best = avg
            torch.save(dict(model_state_dict=model.state_dict(),
                            embedding_strategy=md["embedding_strategy"],
                            hidden_dim=md["hidden_dim"], num_layers=md["num_layers"],
                            embedding_dim=md["embedding_dim"], fp_dim=md["fp_dim"],
                            epoch=epoch), C.PRETRAINED_PATH)
    print(f"pretraining done, best loss {best:.4f}; saved {C.PRETRAINED_PATH}", flush=True)


if __name__ == "__main__":
    main()
