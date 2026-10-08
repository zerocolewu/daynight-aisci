"""
Self-contained data pipeline for the CPA-GAT optimization campaign
(35-dim rich atom features variant).

Graphs (single-polymer Tg/Tm): a head-to-tail N-mer oligomer (N=3 repeat
units) of the p-SMILES, with rich atom features (35:
   13 atomic-number one-hot + 6 degree + 1 aromatic + 1 in-ring
   + 1 formal charge + 6 H-count + 5 hybridization + 1 mass + 1 chirality)
+ a backbone positional encoding (dce, mi) (2) + a random-walk positional
encoding (32) + the is_backbone flag (1) = 70-dim nodes, and 4-dim edge features.

Copolymer graphs (EA/IP): the two monomers joined into ONE connected A-B molecule
(an explicit bond between A's exit and B's entry), with the same node/edge
features plus a composition vector u = [rA, rB, type-one-hot] consumed by the head.

The whole dataset is featurized once (cached), then re-indexed per split so the
expensive RDKit + RWPE work happens only once even with multiple splits.

Splits are decoupled from training randomness:
    * tg, tm: 72/8/20 partition generated from split_seed in {42, 71}.
    * ea, ip: the official split loaded from {ea,ip}_split.pkl.
"""
import os
import pickle
import warnings
from collections import deque
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.transforms import AddRandomWalkPE
from sklearn.model_selection import train_test_split

from config import (DATASET_DIR, CACHE_DIR, DATASETS, SPLIT_PKL, GIVEN_SPLIT_DATASETS,
                    N_MER, RWPE_K, CACHE_VERSION)

warnings.filterwarnings("ignore")

EDGE_DIM = 4  # [single, multi(aromatic/double/...), in_ring, conjugated]

# 12 common polymer atoms + 1 "other" bucket = 13-dim atomic-number one-hot.
# Covers every atom that appears in Tg/Tm/EA/IP plus H (always 0 after RemoveHs,
# harmless) and B (rare but present in Tg). Anything else falls into the "other" slot.
COMMON_ATOMS = [1, 6, 7, 8, 9, 5, 14, 15, 16, 17, 35, 53]
#                         H  C  N  O  F  B  Si  P   S  Cl  Br  I

# 5 hybridization types (matches the legacy HYBRIDIZATION_MAP key set).
HYBRID_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

COPOLYMER_TYPES = ["alternating", "block", "random", "graft"]


# --------------------------------------------------------------------------- #
# is_backbone: double-BFS longest path (the project's single best feature add).
# --------------------------------------------------------------------------- #
def _bfs_farthest(adj, start):
    dist = {start: 0}; q = deque([start]); vis = {start}
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in vis:
                vis.add(v); dist[v] = dist[u] + 1; q.append(v)
    if not dist:
        return start, dist
    return max(dist, key=dist.get), dist


def _bfs_path(adj, start, target):
    if start == target:
        return [start]
    parent = {start: None}; q = deque([start]); vis = {start}
    while q:
        u = q.popleft()
        if u == target:
            break
        for v in adj.get(u, []):
            if v not in vis:
                vis.add(v); parent[v] = u; q.append(v)
    path = []; cur = target
    while cur is not None:
        path.append(cur); cur = parent.get(cur)
    path.reverse()
    return path


def compute_is_backbone(edge_index, num_nodes):
    adj = {i: [] for i in range(num_nodes)}
    ei = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        if v not in adj[u]:
            adj[u].append(v)
    a, _ = _bfs_farthest(adj, 0)
    b, _ = _bfs_farthest(adj, a)
    backbone = set(_bfs_path(adj, a, b))
    return np.array([1.0 if i in backbone else 0.0 for i in range(num_nodes)], dtype=np.float32)


# --------------------------------------------------------------------------- #
# PSPE: backbone positional encoding (degree-to-closest-end, motif index).
# --------------------------------------------------------------------------- #
def find_backbone_path(edge_index, num_nodes):
    adj = {i: [] for i in range(num_nodes)}
    ei = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        adj[u].append(v); adj[v].append(u)

    def bfs_farthest(start):
        dist = {start: 0}; q = deque([start]); vis = {start}
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v not in vis:
                    vis.add(v); dist[v] = dist[u] + 1; q.append(v)
        if not dist:
            return start
        return max(dist, key=dist.get)

    def bfs_path(s, t):
        if s == t:
            return [s]
        par = {s: None}; q = deque([s]); vis = {s}
        while q:
            u = q.popleft()
            if u == t:
                break
            for v in adj.get(u, []):
                if v not in vis:
                    vis.add(v); par[v] = u; q.append(v)
        path = []; cur = t
        while cur is not None:
            path.append(cur); cur = par.get(cur)
        path.reverse()
        return path

    a = bfs_farthest(0)
    b = bfs_farthest(a)
    return bfs_path(a, b)


def compute_pspe(edge_index, num_nodes):
    backbone = find_backbone_path(edge_index, num_nodes)
    bl = len(backbone)
    dce = np.zeros(num_nodes, dtype=np.float32)
    mi = np.zeros(num_nodes, dtype=np.float32)
    if bl < 3:
        return dce, mi
    pos_map = {n: p for p, n in enumerate(backbone)}
    max_p = (bl - 1) / 2.0
    for n in backbone:
        p = pos_map[n]
        raw = min(p, bl - 1 - p)
        dce[n] = raw / max_p if max_p > 0 else 0.0
    seg = bl / 3.0
    for n in backbone:
        p = pos_map[n]
        si = min(int(p / seg), 2)
        mi[n] = si / 2.0
    return dce, mi


# --------------------------------------------------------------------------- #
# Multimer generation (head-to-tail), per the polymer-data-processing convention.
# --------------------------------------------------------------------------- #
def generate_multimer(psmiles, n_mer, cap_with_h=True):
    if n_mer == 1:
        smiles = psmiles.replace("*", "[H]")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    monomer = Chem.MolFromSmiles(psmiles)
    if monomer is None:
        return None
    star_info = []
    for atom in monomer.GetAtoms():
        if atom.GetSymbol() == "*":
            neis = atom.GetNeighbors()
            if len(neis) != 1:
                return None
            bond = monomer.GetBondBetweenAtoms(atom.GetIdx(), neis[0].GetIdx())
            star_info.append({"star_idx": atom.GetIdx(), "neighbor_idx": neis[0].GetIdx(),
                              "bond_type": bond.GetBondType()})
    if len(star_info) != 2:
        return None
    if star_info[0]["bond_type"] != star_info[1]["bond_type"]:
        return None
    num_atoms = monomer.GetNumAtoms()
    oligomer = deepcopy(monomer)
    for _ in range(n_mer - 1):
        oligomer = Chem.CombineMols(oligomer, monomer)
    star_indices = np.zeros((n_mer, 2), dtype=int)
    neighbor_indices = np.zeros((n_mer, 2), dtype=int)
    for i in range(n_mer):
        off = i * num_atoms
        star_indices[i] = [star_info[0]["star_idx"] + off, star_info[1]["star_idx"] + off]
        neighbor_indices[i] = [star_info[0]["neighbor_idx"] + off, star_info[1]["neighbor_idx"] + off]
    ed = Chem.EditableMol(oligomer)
    removed = []
    for i in range(n_mer - 1):
        ed.AddBond(int(neighbor_indices[i, 1]), int(neighbor_indices[i + 1, 0]),
                   order=star_info[0]["bond_type"])
        removed.extend([int(star_indices[i, 1]), int(star_indices[i + 1, 0])])
    if cap_with_h:
        ed.ReplaceAtom(int(star_indices[0, 0]), Chem.Atom(1))
        ed.ReplaceAtom(int(star_indices[n_mer - 1, 1]), Chem.Atom(1))
    for idx in sorted(removed, reverse=True):
        ed.RemoveAtom(idx)
    fm = ed.GetMol()
    fm = Chem.RemoveHs(fm)
    return Chem.MolToSmiles(fm)


# --------------------------------------------------------------------------- #
# Node / edge features
# --------------------------------------------------------------------------- #
def atom_features(atom):
    """Build atom feature vector, 35 dimensions total
    = 13(atomic number) + 6(degree) + 1(aromatic) + 1(ring) + 1(charge)
      + 6(H count) + 5(hybridization) + 1(mass) + 1(chirality)
    """
    features = []

    # 1. Atomic number one-hot (12 common + 1 other = 13 dims)
    atomic_num = atom.GetAtomicNum()
    atom_enc = [0] * (len(COMMON_ATOMS) + 1)
    if atomic_num in COMMON_ATOMS:
        atom_enc[COMMON_ATOMS.index(atomic_num)] = 1
    else:
        atom_enc[-1] = 1
    features += atom_enc

    # 2. Degree one-hot (0-5, 6 dims)
    degree = min(atom.GetDegree(), 5)
    features += [1 if i == degree else 0 for i in range(6)]

    # 3. Aromaticity (1 dim)
    features.append(int(atom.GetIsAromatic()))

    # 4. Whether in ring (1 dim)
    features.append(int(atom.IsInRing()))

    # 5. Formal charge (normalized, 1 dim)
    features.append(float(atom.GetFormalCharge()) / 10.0)

    # 6. Connected hydrogen count one-hot (0-5, 6 dims)
    num_h = min(atom.GetTotalNumHs(), 5)
    features += [1 if i == num_h else 0 for i in range(6)]

    # 7. Hybridization type one-hot (5 dims)
    hybrid = atom.GetHybridization()
    features += [1 if hybrid == ht else 0 for ht in HYBRID_TYPES]

    # 8. Atomic mass (normalized, 1 dim)
    features.append(atom.GetMass() / 100.0)

    # 9. Whether chiral center (1 dim)
    features.append(int(atom.HasProp('_ChiralityPossible')))

    return features  # 35 dims total


def extract_node_features(mol):
    feats = [atom_features(atom) for atom in mol.GetAtoms()]
    return torch.tensor(feats, dtype=torch.float32)


def extract_edge_features(bond):
    bt = bond.GetBondType()
    is_arom = bond.GetIsAromatic()
    is_ring = bond.IsInRing()
    is_conj = bond.GetIsConjugated()
    single = 1.0 if bt == Chem.BondType.SINGLE and not is_arom else 0.0
    multi = 1.0 if (bt != Chem.BondType.SINGLE or is_arom) else 0.0
    return [single, multi, 1.0 if is_ring else 0.0, 1.0 if is_conj else 0.0]


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def mol_to_graph(mol, y, rwpe_transform, extra=None):
    x = extract_node_features(mol)
    edges, edge_feats = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ef = extract_edge_features(bond)
        edges.append([i, j]); edges.append([j, i])
        edge_feats.append(ef); edge_feats.append(ef)
    if len(edges) == 0:
        return None
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
    n = x.size(0)
    dce, mi = compute_pspe(edge_index, n)
    x = torch.cat([x, torch.tensor(dce).unsqueeze(-1), torch.tensor(mi).unsqueeze(-1)], dim=-1)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)
    try:
        data = rwpe_transform(data)
    except Exception:
        data.random_walk_pe = torch.zeros(n, RWPE_K)
    data.x = torch.cat([data.x, data.random_walk_pe], dim=-1)
    del data.random_walk_pe
    # is_backbone (append last column -> 70-dim nodes)
    ib = compute_is_backbone(data.edge_index, data.num_nodes)
    data.x = torch.cat([data.x, torch.tensor(ib).unsqueeze(-1)], dim=-1)
    data.y = torch.tensor([y], dtype=torch.float32)
    if extra is not None:
        for k, v in extra.items():
            setattr(data, k, v)
    return data


def build_copolymer_mol(smi_a, smi_b):
    """Join monomers A and B into a single connected molecule."""
    ma = Chem.MolFromSmiles(str(smi_a))
    mb = Chem.MolFromSmiles(str(smi_b))
    if ma is None or mb is None:
        return None

    def star_info(m):
        si = []
        for at in m.GetAtoms():
            if at.GetSymbol() == "*":
                neis = at.GetNeighbors()
                if len(neis) != 1:
                    return None
                b = m.GetBondBetweenAtoms(at.GetIdx(), neis[0].GetIdx())
                si.append({"star": at.GetIdx(), "nei": neis[0].GetIdx(), "bt": b.GetBondType()})
        if len(si) != 2:
            return None
        return si

    sa = star_info(ma); sb = star_info(mb)
    if sa is None or sb is None:
        return None
    if sa[0]["bt"] != sa[1]["bt"] or sb[0]["bt"] != sb[1]["bt"]:
        return None
    bt = sa[0]["bt"]
    na = ma.GetNumAtoms()
    combined = Chem.CombineMols(ma, mb)
    ed = Chem.EditableMol(combined)
    ed.AddBond(int(sa[1]["nei"]), int(sb[0]["nei"] + na), order=bt)
    ed.ReplaceAtom(int(sa[0]["star"]), Chem.Atom(1))
    ed.ReplaceAtom(int(sb[1]["star"] + na), Chem.Atom(1))
    for idx in sorted([int(sa[1]["star"]), int(sb[0]["star"] + na)], reverse=True):
        ed.RemoveAtom(idx)
    mol = ed.GetMol()
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.RemoveHs(mol)


def build_copolymer_graph(smi_a, smi_b, ra, rb, ctype, y, rwpe_transform):
    mol = build_copolymer_mol(smi_a, smi_b)
    if mol is None:
        return None
    type_oh = [1.0 if t == str(ctype) else 0.0 for t in COPOLYMER_TYPES]
    extra_vec = np.array([float(ra), float(rb)] + type_oh, dtype=np.float32)
    return mol_to_graph(mol, y, rwpe_transform,
                        extra={"u": torch.tensor(extra_vec).unsqueeze(0)})


# --------------------------------------------------------------------------- #
# Full-dataset featurization (cached)
# --------------------------------------------------------------------------- #
def _cache_path(dataset):
    return os.path.join(CACHE_DIR, f"{dataset}_{CACHE_VERSION}_full.pkl")


def featurize_full(dataset, max_rows=None, force=False):
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
    rwpe_transform = AddRandomWalkPE(walk_length=RWPE_K)

    graphs, valid_csv_idx = [], []
    n_total = len(df)
    for i in range(n_total):
        row = df.iloc[i]
        y = float(row[ycol])
        try:
            if kind == "single":
                smi = generate_multimer(str(row[meta["smi"]]), N_MER)
                if smi is None:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None or mol.GetNumAtoms() == 0:
                    continue
                g = mol_to_graph(mol, y, rwpe_transform)
            else:
                g = build_copolymer_graph(str(row["smiles_A"]), str(row["smiles_B"]),
                                          float(row["rA"]), float(row["rB"]),
                                          str(row["type"]), y, rwpe_transform)
            if g is None:
                continue
            graphs.append(g); valid_csv_idx.append(i)
        except Exception:
            continue

    u_dim = 2 + len(COPOLYMER_TYPES) if kind == "copolymer" else 0
    feat = dict(graphs=graphs, valid_csv_idx=valid_csv_idx, n_total=n_total,
                n_valid=len(graphs), kind=kind, u_dim=u_dim)
    if max_rows is None:
        with open(cpath, "wb") as f:
            pickle.dump(feat, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[featurize {dataset}] valid {len(graphs)}/{n_total} "
          f"({n_total - len(graphs)} dropped)  node_dim={graphs[0].x.shape[1]}", flush=True)
    return feat


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def load_given_split(dataset):
    p = os.path.join(DATASET_DIR, SPLIT_PKL[dataset])
    with open(p, "rb") as f:
        obj = pickle.load(f)
    assert isinstance(obj, (list, tuple)) and len(obj) == 3, \
        f"unexpected split format in {p}: {type(obj)}"
    return tuple(np.asarray(a) for a in obj)


def get_split(dataset, split_seed):
    """Return dict(train, val, test graph lists, y_mean, y_std, split_tag, counts, u_dim, kind)."""
    feat = featurize_full(dataset)
    graphs = feat["graphs"]
    valid = feat["valid_csv_idx"]
    csv2pos = {c: i for i, c in enumerate(valid)}

    if split_seed is None:
        tr_csv, va_csv, te_csv = load_given_split(dataset)
        if int(max(tr_csv.max(), va_csv.max(), te_csv.max())) >= feat["n_total"]:
            print(f"[WARN] {dataset}: split index exceeds CSV rows ({feat['n_total']}); "
                  f"check pkl provenance", flush=True)
        tr = [csv2pos[int(c)] for c in tr_csv if int(c) in csv2pos]
        va = [csv2pos[int(c)] for c in va_csv if int(c) in csv2pos]
        te = [csv2pos[int(c)] for c in te_csv if int(c) in csv2pos]
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
                u_dim=feat["u_dim"], kind=feat["kind"])


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else "tg"
    sp = get_split(ds, split_seed=None if ds in GIVEN_SPLIT_DATASETS else 71)
    g = sp["train"][0]
    print(f"{ds}: train/val/test = {sp['counts']}  y_mean={sp['y_mean']:.3f} "
          f"y_std={sp['y_std']:.3f}  u_dim={sp['u_dim']}")
    print(f"sample graph: x{tuple(g.x.shape)} edge_index{tuple(g.edge_index.shape)} "
          f"edge_attr{tuple(g.edge_attr.shape)} u={getattr(g, 'u', None) is not None}")
