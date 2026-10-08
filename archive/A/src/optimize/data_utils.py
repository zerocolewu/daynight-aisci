"""
Self-contained data pipeline for the PTKGE optimization campaign.

Single-polymer graphs (Tg, Tm):
    p-SMILES -> cap connection stars '*' with [H] -> RDKit mol -> atomic-number
    graph (node = atomic number, edges = bonds) + ECFP4 fingerprint. This is the
    exact representation used to pretrain the encoder on PI1M, so the pretrained
    weights transfer without a distribution shift.

Copolymer graphs (EA, IP) -- THE KEY FIX:
    The two monomers are joined into ONE connected A-B molecule (an explicit bond
    between monomer A's exit and monomer B's entry; outer stars capped with H),
    so the GIN can pass messages across the A-B interface. The previous version
    used a disjoint union of the two monomer graphs, which left A and B
    disconnected and could not model the interface -- the root cause of the poor
    EA/IP numbers. Composition scalars [rA, rB, type-one-hot] are concatenated at
    the head. ECFP4 is computed on the connected A-B molecule.

Splits are decoupled from training randomness:
    * tg, tm: 72/8/20 partition generated from split_seed in {42, 71}.
    * ea, ip: the official split loaded from {ea,ip}_split.pkl.
    The whole dataset is featurized once (cached), then re-indexed per split, so
    the expensive RDKit work happens only once even with multiple splits.
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from sklearn.model_selection import train_test_split

from config import (DATASET_DIR, CACHE_DIR, DATASETS, SPLIT_PKL, GIVEN_SPLIT_DATASETS, CACHE_VERSION)
from model import compute_ecfp4_fingerprint

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

ECFP_BITS = 2048


# --------------------------------------------------------------------------- #
# Molecule construction
# --------------------------------------------------------------------------- #
def polymer_smiles_to_mol(smiles):
    """Replace polymer connection star '*' with [H] and parse."""
    try:
        return Chem.MolFromSmiles(str(smiles).replace("*", "[H]"))
    except Exception:
        return None


def _star_info(mol):
    """Locate the two connection stars in a monomer; None if not exactly two."""
    info = []
    for at in mol.GetAtoms():
        if at.GetSymbol() != "*":
            continue
        neis = at.GetNeighbors()
        if len(neis) != 1:
            return None
        b = mol.GetBondBetweenAtoms(at.GetIdx(), neis[0].GetIdx())
        info.append({"star": at.GetIdx(), "nei": neis[0].GetIdx(), "bt": b.GetBondType()})
    if len(info) != 2:
        return None
    return info


def build_connected_copolymer_mol(smi_a, smi_b):
    """Join monomers A and B into a single connected molecule (A-exit -> B-entry)."""
    ma = Chem.MolFromSmiles(str(smi_a))
    mb = Chem.MolFromSmiles(str(smi_b))
    if ma is None or mb is None:
        return None
    sa = _star_info(ma)
    sb = _star_info(mb)
    if sa is None or sb is None:
        return None
    if sa[0]["bt"] != sa[1]["bt"] or sb[0]["bt"] != sb[1]["bt"]:
        return None
    bt = sa[0]["bt"]
    na = ma.GetNumAtoms()
    combined = Chem.CombineMols(ma, mb)
    ed = Chem.EditableMol(combined)
    # Link A's exit neighbour to B's entry neighbour.
    ed.AddBond(int(sa[1]["nei"]), int(sb[0]["nei"] + na), order=bt)
    # Cap the two outer stars with H.
    ed.ReplaceAtom(int(sa[0]["star"]), Chem.Atom(1))
    ed.ReplaceAtom(int(sb[1]["star"] + na), Chem.Atom(1))
    # Remove the two inner stars (now redundant).
    for idx in sorted([int(sa[1]["star"]), int(sb[0]["star"] + na)], reverse=True):
        ed.RemoveAtom(idx)
    mol = ed.GetMol()
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    if mol.GetNumAtoms() == 0:
        return None
    return mol


def mol_to_atom_graph(mol):
    """Atomic-number node graph + bidirectional bond edges. No bond/atom features
    (matches the pretrained PTKGE encoder, which only consumes atomic numbers)."""
    zs = [a.GetAtomicNum() for a in mol.GetAtoms()]
    if not zs:
        return None
    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges.append([i, j]); edges.append([j, i])
    if not edges:
        edges = [[0, 0]]
    z = torch.tensor(zs, dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return z, edge_index


def ecfp4_of_mol(mol):
    """ECFP4 (Morgan radius 2) of an RDKit mol object."""
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=ECFP_BITS)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return np.zeros(ECFP_BITS, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Graph builders
# --------------------------------------------------------------------------- #
def build_single_graph(smiles, y):
    """Single-polymer graph: atomic-number graph + ECFP4 (on the *-SMILES)."""
    mol = polymer_smiles_to_mol(smiles)
    if mol is None:
        return None
    gr = mol_to_atom_graph(mol)
    if gr is None:
        return None
    z, edge_index = gr
    fp = compute_ecfp4_fingerprint(str(smiles), n_bits=ECFP_BITS)
    if fp.sum() == 0:
        return None
    d = Data()
    d.z = z
    d.edge_index = edge_index
    d.fp = torch.tensor(fp, dtype=torch.float).unsqueeze(0)   # [1, fp_dim]
    d.num_nodes = int(z.numel())
    d.y = torch.tensor([float(y)], dtype=torch.float)
    return d


def build_copolymer_graph(smi_a, smi_b, rA, rB, ctype, type_to_i, y):
    """Connected A-B copolymer graph + ECFP4 (on the connected mol) + extras."""
    mol = build_connected_copolymer_mol(smi_a, smi_b)
    if mol is None:
        return None
    gr = mol_to_atom_graph(mol)
    if gr is None:
        return None
    z, edge_index = gr
    fp = ecfp4_of_mol(mol)
    if fp.sum() == 0:
        return None
    n_types = len(type_to_i)
    type_oh = np.zeros(n_types, dtype=np.float32)
    if str(ctype) in type_to_i:
        type_oh[type_to_i[str(ctype)]] = 1.0
    extras = np.concatenate([np.asarray([rA, rB], dtype=np.float32), type_oh])

    d = Data()
    d.z = z
    d.edge_index = edge_index
    d.fp = torch.tensor(fp, dtype=torch.float).unsqueeze(0)
    d.extras = torch.tensor(extras, dtype=torch.float).unsqueeze(0)  # [1, extras_dim]
    d.num_nodes = int(z.numel())
    d.y = torch.tensor([float(y)], dtype=torch.float)
    return d


# --------------------------------------------------------------------------- #
# Full-dataset featurization (cached)
# --------------------------------------------------------------------------- #
def _cache_path(dataset):
    return os.path.join(CACHE_DIR, f"{dataset}_{CACHE_VERSION}_full.pkl")


def featurize_full(dataset, max_rows=None, force=False):
    """Featurize the entire dataset once and cache it. Returns a dict with the
    graph list, the list of valid original CSV row indices, and (for copolymers)
    the type vocabulary + extras_dim."""
    cpath = _cache_path(dataset)
    if not force and max_rows is None and os.path.exists(cpath):
        with open(cpath, "rb") as f:
            feat = pickle.load(f)
        print(f"[featurize {dataset}] loaded from cache {os.path.basename(cpath)} "
              f"(valid {feat['n_valid']}/{feat['n_total']})", flush=True)
        return feat

    meta = DATASETS[dataset]
    df = pd.read_csv(os.path.join(DATASET_DIR, meta["csv"]))
    if max_rows is not None:
        df = df.iloc[:max_rows].copy()
    ycol = meta["y"]
    kind = meta["kind"]

    graphs, valid_csv_idx, type_to_i, extras_dim = [], [], None, 0
    if kind == "copolymer":
        types = sorted(df["type"].astype(str).unique().tolist())
        type_to_i = {t: i for i, t in enumerate(types)}
        extras_dim = 2 + len(types)

    n_total = len(df)
    for i in range(n_total):
        row = df.iloc[i]
        y = float(row[ycol])
        if kind == "single":
            g = build_single_graph(row[meta["smi"]], y)
        else:
            g = build_copolymer_graph(row["smiles_A"], row["smiles_B"],
                                      float(row["rA"]), float(row["rB"]),
                                      str(row["type"]), type_to_i, y)
        if g is not None:
            graphs.append(g)
            valid_csv_idx.append(i)

    feat = dict(graphs=graphs, valid_csv_idx=valid_csv_idx, n_total=n_total,
                n_valid=len(graphs), kind=kind,
                type_to_i=type_to_i, extras_dim=extras_dim)
    if max_rows is None:
        with open(cpath, "wb") as f:
            pickle.dump(feat, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[featurize {dataset}] valid {len(graphs)}/{n_total} "
          f"({n_total - len(graphs)} dropped)", flush=True)
    return feat


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def load_given_split(dataset):
    """Official split -> (train_csv_idx, val_csv_idx, test_csv_idx) numpy arrays."""
    p = os.path.join(DATASET_DIR, SPLIT_PKL[dataset])
    with open(p, "rb") as f:
        obj = pickle.load(f)
    assert isinstance(obj, (list, tuple)) and len(obj) == 3, \
        f"unexpected split format in {p}: {type(obj)}"
    return tuple(np.asarray(a) for a in obj)


def get_split(dataset, split_seed):
    """Return dict(train, val, test graph lists, y_mean, y_std, split_tag, counts).

    split_seed=None -> use the official split pkl (ea/ip).
    split_seed=int  -> generate a 72/8/20 split deterministically (tg/tm).
    """
    feat = featurize_full(dataset)
    graphs = feat["graphs"]
    valid = feat["valid_csv_idx"]                      # original CSV row indices
    csv2pos = {c: i for i, c in enumerate(valid)}      # csv row -> position in graphs

    if split_seed is None:
        tr_csv, va_csv, te_csv = load_given_split(dataset)
        if int(max(tr_csv.max(), va_csv.max(), te_csv.max())) >= feat["n_total"]:
            print(f"[WARN] {dataset}: split index exceeds CSV rows ({feat['n_total']}); "
                  f"check pkl provenance", flush=True)
        tr = [csv2pos[c] for c in tr_csv if int(c) in csv2pos]
        va = [csv2pos[c] for c in va_csv if int(c) in csv2pos]
        te = [csv2pos[c] for c in te_csv if int(c) in csv2pos]
        split_tag = "given"
    else:
        pos = np.arange(len(valid))
        pos_tv, pos_te = train_test_split(pos, test_size=0.20, random_state=split_seed)
        pos_tr, pos_va = train_test_split(pos_tv, test_size=0.10, random_state=split_seed)
        tr, va, te = pos_tr.tolist(), pos_va.tolist(), pos_te.tolist()
        split_tag = str(split_seed)

    train_g = [graphs[i] for i in tr]
    val_g = [graphs[i] for i in va]
    test_g = [graphs[i] for i in te]

    y_tr = np.array([float(g.y.item()) for g in train_g], dtype=np.float64)
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std() + 1e-8)

    print(f"[{dataset} split={split_tag}] train/val/test = "
          f"{len(tr)}/{len(va)}/{len(te)}  "
          f"(valid molecules used: {len(valid)}/{feat['n_total']}, "
          f"{feat['n_total'] - len(valid)} dropped before split)", flush=True)

    return dict(train=train_g, val=val_g, test=test_g,
                y_mean=y_mean, y_std=y_std, split_tag=split_tag,
                counts=(len(tr), len(va), len(te)),
                extras_dim=feat["extras_dim"], kind=feat["kind"])


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else "tg"
    sp = get_split(ds, split_seed=None if ds in GIVEN_SPLIT_DATASETS else 71)
    g = sp["train"][0]
    print(f"{ds}: train/val/test = {sp['counts']}  y_mean={sp['y_mean']:.3f} "
          f"y_std={sp['y_std']:.3f}  extras_dim={sp['extras_dim']}")
    print(f"sample graph: z{tuple(g.z.shape)} edge_index{tuple(g.edge_index.shape)} "
          f"fp{tuple(g.fp.shape)} extras={getattr(g, 'extras', None) is not None}")
