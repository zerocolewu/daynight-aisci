#!/usr/bin/env python3
"""
IAPT v2 Improved: Virtual node + multi-scale pooling + global features
Target: Beat RMSE 30.8 on Tg dataset
"""

import os, sys, time, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GINEConv, GINConv, global_mean_pool, global_max_pool, global_add_pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings('ignore')

CSV_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
WORK_DIR = '/home/home_public/aisci/AIscientist/src/4_iapt_v2_intensive_aware'
CACHE_V2 = os.path.join(WORK_DIR, 'processed_data_v2.pkl')
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/alldataset/some_dataprocesscode')
from multimer import generate_multimer_smiles

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
MAX_EPOCHS = 500
PATIENCE = 40
LR = 5e-4
WD = 1e-4
HIDDEN = 256
N_GNN = 6
N_LG = 2
LG_HIDDEN = 128
DROP = 0.15

ATOM_TYPES = ['C','N','O','S','F','Si','P','Cl','Br','I','B','Se','Te','Ge','As','Sb','Bi','*']
AT_DICT = {t:i for i,t in enumerate(ATOM_TYPES)}
HYB_TYPES = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
             Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
             Chem.rdchem.HybridizationType.SP3D2]
HYB_DICT = {t:i for i,t in enumerate(HYB_TYPES)}
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BT_DICT = {t:i for i,t in enumerate(BOND_TYPES)}

def atom_feat(a):
    f = np.zeros(27, dtype=np.float32)
    s = a.GetSymbol()
    if s in AT_DICT: f[AT_DICT[s]] = 1.0
    f[18] = a.GetDegree()/6.0
    h = a.GetHybridization()
    if h in HYB_DICT: f[19+HYB_DICT[h]] = 1.0
    f[24] = float(a.GetIsAromatic())
    f[25] = a.GetFormalCharge()/4.0
    f[26] = float(a.IsInRing())
    return f

def bond_feat(b):
    f = np.zeros(6, dtype=np.float32)
    bt = b.GetBondType()
    f[BT_DICT.get(bt, 0)] = 1.0
    f[4] = float(b.GetIsConjugated())
    f[5] = float(b.IsInRing())
    return f

def gen_conformer(mol):
    try:
        if mol.GetNumAtoms() > 150: return None
        p = AllChem.ETKDGv3()
        p.randomSeed = 42; p.maxIterations = 5
        if AllChem.EmbedMolecule(mol, p) == -1: return None
        try:
            pr = AllChem.MMFFGetMoleculeProperties(mol)
            if pr:
                ff = AllChem.MMFFGetMoleculeForceField(mol, pr)
                if ff: ff.Minimize(maxIts=30); return mol
        except: pass
        try:
            ff = AllChem.UFFGetMoleculeForceField(mol)
            if ff: ff.Minimize(maxIts=30)
        except: pass
        return mol
    except: return None

def equiv_feat(mol3d):
    c = mol3d.GetConformer()
    n = mol3d.GetNumAtoms()
    pos = np.zeros((n,3), dtype=np.float32)
    for i in range(n):
        p = c.GetAtomPosition(i); pos[i] = [p.x, p.y, p.z]
    ctr = pos.mean(0); disp = pos - ctr
    norms = np.linalg.norm(disp, axis=1, keepdims=True)
    feats = np.zeros((n,3), dtype=np.float32)
    feats[:,0] = (norms/10.0).flatten()
    for i in range(n):
        nb = [j.GetIdx() for j in mol3d.GetAtomWithIdx(i).GetNeighbors()]
        if not nb: continue
        d = disp[i]; dn = norms[i,0]+1e-8; du = d/dn
        bd = [(pos[j]-pos[i])/(np.linalg.norm(pos[j]-pos[i])+1e-8) for j in nb]
        ab = np.mean(bd, axis=0)
        feats[i,1] = np.dot(du, ab)
        if len(bd)>1: feats[i,2] = np.std([np.dot(du,b) for b in bd])
    return feats

def compute_global_features(mol):
    """Compute RDKit molecular descriptors for the monomer."""
    feats = []
    try:
        feats.append(Descriptors.MolWt(mol) / 500.0)
        feats.append(Descriptors.NumRotatableBonds(mol) / 10.0)
        feats.append(rdMolDescriptors.CalcNumRings(mol) / 5.0)
        feats.append(Descriptors.NumHeteroatoms(mol) / mol.GetNumAtoms())
        feats.append(float(rdMolDescriptors.CalcNumAromaticRings(mol)) / 3.0)
        feats.append(float(mol.GetNumAtoms()) / 30.0)
        feats.append(float(mol.GetNumBonds()) / 30.0)
        feats.append(float(Descriptors.NumAliphaticRings(mol)) / 3.0)
        feats.append(float(Descriptors.FractionCSP3(mol)))
        feats.append(Descriptors.TPSA(mol) / 200.0)
        feats.append(Descriptors.MolLogP(mol) / 5.0)
        feats.append(float(Descriptors.NumHDonors(mol)) / 5.0)
        feats.append(float(Descriptors.NumHAcceptors(mol)) / 10.0)
    except:
        feats = [0.0] * 13
    return np.array(feats, dtype=np.float32)

def build_lg(ei):
    ne = ei.shape[1]
    if ne==0: return np.zeros((2,0), dtype=np.int64)
    n2e = {}
    for e in range(ne):
        u = int(ei[0,e]); n2e.setdefault(u,[]).append(e)
    edges = set()
    for _, el in n2e.items():
        for i in range(len(el)):
            for j in range(i+1, len(el)):
                edges.add((el[i],el[j])); edges.add((el[j],el[i]))
    return np.array(sorted(edges), dtype=np.int64).T if edges else np.zeros((2,0), dtype=np.int64)

def process_one(ps, tg):
    try:
        ts = generate_multimer_smiles(3, ps, replace_dummy_atoms=True)
        tm = Chem.MolFromSmiles(ts)
        if tm is None: return None
        na = tm.GetNumAtoms()

        # Global features from monomer
        mono = Chem.MolFromSmiles(ps.replace('*','[H]'))
        gf = compute_global_features(mono) if mono else np.zeros(13, dtype=np.float32)

        af = np.array([atom_feat(a) for a in tm.GetAtoms()], dtype=np.float32)
        el, efl = [], []
        for b in tm.GetBonds():
            i,j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bf = bond_feat(b)
            el.append([i,j]); el.append([j,i])
            efl.append(bf); efl.append(bf.copy())
        if not el: return None
        ei = np.array(el, dtype=np.int64).T
        ef = np.stack(efl)

        cok = False
        mc = Chem.Mol(tm)
        m3 = gen_conformer(mc)
        if m3 and m3.GetNumConformers()>0:
            cok = True; cf = m3.GetConformer()
            bl = np.zeros(len(el), dtype=np.float32)
            for b in tm.GetBonds():
                bi,bj = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                try:
                    pi = np.array(cf.GetAtomPosition(bi)); pj = np.array(cf.GetAtomPosition(bj))
                    l = np.linalg.norm(pi-pj)
                except: l = 1.5
                for idx,(u,v) in enumerate(el):
                    if u==bi and v==bj: bl[idx]=l
            ef = np.concatenate([ef, bl.reshape(-1,1)/2.0], axis=1)
            eq = equiv_feat(m3)
            af = np.concatenate([af, eq], axis=1)
            # Global 3D features
            positions = np.zeros((na,3), dtype=np.float32)
            for i in range(na):
                p = cf.GetAtomPosition(i); positions[i] = [p.x, p.y, p.z]
            # Radius of gyration
            ctr = positions.mean(0)
            rg = np.sqrt(np.mean(np.sum((positions - ctr)**2, axis=1)))
            gf = np.concatenate([gf, [rg/10.0]])
        else:
            ef = np.concatenate([ef, np.full((len(el),1), 0.75, dtype=np.float32)], axis=1)
            af = np.concatenate([af, np.zeros((na,3), dtype=np.float32)], axis=1)
            gf = np.concatenate([gf, [0.0]])

        # Periodic edges
        ends = [a.GetIdx() for a in tm.GetAtoms() if a.GetDegree()==1 and not a.IsInRing()]
        if len(ends)>=2:
            s1,s2 = ends[0], ends[-1]
            pe = np.zeros(ef.shape[1], dtype=np.float32); pe[0]=1.0
            ei = np.concatenate([ei, np.array([[s1,s2],[s2,s1]], dtype=np.int64)], axis=1)
            ef = np.concatenate([ef, np.stack([pe,pe])], axis=0)

        lg = build_lg(ei)
        return {'af':af, 'ei':ei, 'ef':ef, 'lg':lg, 'tg':float(tg), 'smi':ps,
                'na':na, 'cok':cok, 'gf':gf}
    except: return None

def load_data(seed=42):
    if os.path.exists(CACHE_V2):
        print(f"Loading cached v2 data from {CACHE_V2}")
        with open(CACHE_V2,'rb') as f: return pickle.load(f)
    print("Processing dataset v2...")
    df = pd.read_csv(CSV_PATH)
    np.random.seed(seed)
    idx = np.random.permutation(len(df))
    nt,nv = int(len(df)*0.72), int(len(df)*0.08)
    splits = {'train': idx[:nt], 'val': idx[nt:nt+nv], 'test': idx[nt+nv:]}
    data = {}
    for sn, si in splits.items():
        print(f"  {sn}: {len(si)}...")
        samp = []
        for c,i in enumerate(si):
            r = process_one(df.iloc[i]['Smiles'], df.iloc[i]['Tg'])
            if r: samp.append(r)
            if (c+1)%1000==0: print(f"    {c+1}/{len(si)} ok={len(samp)}")
        print(f"    Done: {len(samp)}")
        data[sn] = samp
    with open(CACHE_V2,'wb') as f: pickle.dump(data, f)
    return data

class PDataset(Dataset):
    def __init__(self, samples, use_eq=True, use_lg=True):
        self.samples = samples; self.use_eq = use_eq; self.use_lg = use_lg
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        af = s['af'].copy()
        if not self.use_eq and af.shape[1]>27: af = af[:,:27]
        lg = s['lg'] if self.use_lg else np.zeros((2,0), dtype=np.int64)
        return {'af':torch.from_numpy(af), 'ei':torch.from_numpy(s['ei']),
                'ef':torch.from_numpy(s['ef']), 'lg':torch.from_numpy(lg),
                'gf':torch.tensor(s['gf'], dtype=torch.float32),
                'tg':torch.tensor([s['tg']],dtype=torch.float32), 'na':s['na']}

def collate(batch):
    afs, eis, efs, lgs, gfs, tgs, bv = [],[],[],[],[],[],[]
    ao, eo = 0, 0
    for i,s in enumerate(batch):
        na,ne = s['na'], s['ei'].shape[1]
        afs.append(s['af']); eis.append(s['ei']+ao); efs.append(s['ef'])
        if s['lg'].shape[1]>0: lgs.append(s['lg']+eo)
        else: lgs.append(torch.zeros((2,0),dtype=torch.long))
        gfs.append(s['gf']); tgs.append(s['tg']); bv.append(torch.full((na,),i,dtype=torch.long))
        ao+=na; eo+=ne
    return {'af':torch.cat(afs,0), 'ei':torch.cat(eis,1), 'ef':torch.cat(efs,0),
            'lg':torch.cat(lgs,1) if lgs else torch.zeros((2,0),dtype=torch.long),
            'gf':torch.stack(gfs), 'tg':torch.cat(tgs,0), 'batch':torch.cat(bv,0)}


class ImprovedModel(nn.Module):
    def __init__(self, adim, edim, gfdim=14, hid=HIDDEN, ngnn=N_GNN, nlg=N_LG,
                 lgh=LG_HIDDEN, drop=DROP, use_lg=True, use_vn=True):
        super().__init__()
        self.hid = hid; self.use_lg = use_lg; self.use_vn = use_vn

        self.aemb = nn.Sequential(nn.Linear(adim,hid), nn.LayerNorm(hid), nn.GELU())
        self.eemb = nn.Sequential(nn.Linear(edim,hid), nn.LayerNorm(hid), nn.GELU())

        # Virtual node
        if use_vn:
            self.vn_emb = nn.Parameter(torch.zeros(1, hid))
            self.vn_mlp = nn.ModuleList([nn.Sequential(nn.Linear(hid,hid), nn.GELU())
                                          for _ in range(ngnn)])
            self.vn_proj = nn.ModuleList([nn.Linear(hid,hid) for _ in range(ngnn)])

        # GINE layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(ngnn):
            m = nn.Sequential(nn.Linear(hid,hid), nn.BatchNorm1d(hid), nn.GELU(),
                              nn.Dropout(drop), nn.Linear(hid,hid))
            self.convs.append(GINEConv(nn=m, eps=0.0, train_eps=True))
            self.norms.append(nn.LayerNorm(hid))

        # Line graph
        if use_lg:
            self.lemb = nn.Linear(edim, lgh)
            self.lconvs = nn.ModuleList()
            for _ in range(nlg):
                m = nn.Sequential(nn.Linear(lgh,lgh), nn.GELU(), nn.Linear(lgh,lgh))
                self.lconvs.append(GINConv(nn=m))
            self.lproj = nn.Sequential(nn.Linear(lgh,hid), nn.GELU())

        # Multi-scale attention pooling
        self.pool_query = nn.Parameter(torch.randn(1, hid))
        self.pool_key = nn.Linear(hid, hid)
        self.pool_value = nn.Linear(hid, hid)

        # Global features integration
        self.gf_emb = nn.Sequential(nn.Linear(gfdim, hid//2), nn.GELU(), nn.Linear(hid//2, hid//4))

        # Scale weights for jumping knowledge
        self.sw = nn.Parameter(torch.ones(ngnn)/ngnn)

        # Predictor with global features
        self.pred = nn.Sequential(
            nn.LayerNorm(hid + hid//4),
            nn.Linear(hid + hid//4, hid),
            nn.GELU(), nn.Dropout(drop),
            nn.Linear(hid, hid//2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hid//2, 1),
        )

    def forward(self, d):
        x = self.aemb(d['af']); ei = d['ei']; ea = self.eemb(d['ef'])
        batch = d['batch']
        bs = d['tg'].shape[0]

        # Virtual node initialization
        if self.use_vn:
            vn = self.vn_emb.expand(bs, -1)

        # Line graph
        if self.use_lg and hasattr(self,'lemb') and d['lg'].shape[1]>0:
            lx = self.lemb(d['ef'])
            for lc in self.lconvs: lx = lx + lc(lx, d['lg'])
            ea = ea + self.lproj(lx)

        # GINE layers with virtual node and jumping knowledge
        outs = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Virtual node -> node messages
            if self.use_vn:
                vn_to_node = self.vn_proj[i](vn)
                vn_per_node = vn_to_node[batch]
                x = x + vn_per_node

            if ei.shape[1]>0:
                x_new = conv(x, ei, ea)
            else:
                x_new = conv.nn(x)
            x = norm(x + x_new)
            x = F.gelu(x)
            outs.append(x)

            # Update virtual node
            if self.use_vn:
                vn_agg = global_add_pool(x, batch)
                vn = vn + self.vn_mlp[i](vn_agg)

        # Weighted combination
        w = F.softmax(self.sw, dim=0)
        x = sum(wi*xi for wi,xi in zip(w, outs))

        # Multi-scale pooling: combine mean, max, sum, attention
        p_mean = global_mean_pool(x, batch)
        p_max = global_max_pool(x, batch)
        p_sum = global_add_pool(x, batch) / (global_add_pool(torch.ones(x.shape[0],1,device=x.device), batch) + 1e-8)

        # Attention pooling
        keys = self.pool_key(x)
        vals = self.pool_value(x)
        scale = self.hid ** 0.5
        att_parts = []
        for b in range(bs):
            mask = (batch == b)
            if mask.sum() == 0:
                att_parts.append(torch.zeros(1, self.hid, device=x.device))
                continue
            k, v = keys[mask], vals[mask]
            q = self.pool_query.expand(k.shape[0], -1)
            scores = F.softmax((q * k).sum(-1) / scale, dim=0)
            att_parts.append((scores.unsqueeze(-1) * v).sum(0, keepdim=True))
        p_att = torch.cat(att_parts, 0)

        # Combined pooling
        pooled = (p_mean + p_max + p_sum + p_att) / 4.0

        # Global features
        gf_emb = self.gf_emb(d['gf'])

        # Concatenate and predict
        combined = torch.cat([pooled, gf_emb], dim=-1)
        return self.pred(combined)


def train_ep(model, ld, opt, crit, sc, dev):
    model.train(); tl=0; pa,ta=[],[]
    for b in ld:
        af,ei,ef,lg,gf,bv,tg = [b[k].to(dev) for k in ['af','ei','ef','lg','gf','batch','tg']]
        tn = torch.tensor(sc.transform(tg.cpu().numpy().reshape(-1,1)), dtype=torch.float32, device=dev)
        opt.zero_grad()
        pr = model({'af':af,'ei':ei,'ef':ef,'lg':lg,'gf':gf,'batch':bv,'tg':tg})
        loss = crit(pr,tn); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tl += loss.item()*pr.shape[0]
        pdn = sc.inverse_transform(pr.detach().cpu().numpy().reshape(-1,1)).flatten()
        pa.extend(pdn); ta.extend(tg.cpu().numpy().flatten())
    return tl/len(pa), np.sqrt(mean_squared_error(ta,pa))

@torch.no_grad()
def evaluate(model, ld, crit, sc, dev):
    model.eval(); tl=0; pa,ta=[],[]
    for b in ld:
        af,ei,ef,lg,gf,bv,tg = [b[k].to(dev) for k in ['af','ei','ef','lg','gf','batch','tg']]
        tn = torch.tensor(sc.transform(tg.cpu().numpy().reshape(-1,1)), dtype=torch.float32, device=dev)
        pr = model({'af':af,'ei':ei,'ef':ef,'lg':lg,'gf':gf,'batch':bv,'tg':tg})
        tl += crit(pr,tn).item()*pr.shape[0]
        pdn = sc.inverse_transform(pr.cpu().numpy().reshape(-1,1)).flatten()
        pa.extend(pdn); ta.extend(tg.cpu().numpy().flatten())
    return tl/len(pa), np.sqrt(mean_squared_error(ta,pa)), mean_absolute_error(ta,pa), r2_score(ta,pa), ta, pa


def run_multi_seed(data, seed_list=[42, 123, 456, 789, 2024]):
    """Run multiple seeds and take ensemble."""
    all_test_preds = []
    all_test_tgts = None
    per_seed_results = []

    for seed in seed_list:
        print(f"\n{'='*70}\nSeed: {seed}\n{'='*70}")

        # Re-split data with this seed
        df = pd.read_csv(CSV_PATH)
        np.random.seed(seed)
        idx = np.random.permutation(len(df))
        nt,nv = int(len(df)*0.72), int(len(df)*0.08)
        # We need to re-process or re-index - for simplicity, use same cache but different split
        # Since data is cached with seed=42 split, we'll just use that split for all seeds
        # This is a simplification - the important variation is model init seed

        torch.manual_seed(seed); np.random.seed(seed)

        tr = PDataset(data['train'])
        va = PDataset(data['val'])
        te = PDataset(data['test'])

        trl = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
        val = DataLoader(va, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
        tel = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

        s = tr[0]; ad=s['af'].shape[1]; ed=s['ef'].shape[1]
        gd = s['gf'].shape[0] if s['gf'].dim() == 1 else s['gf'].shape[-1]
        tgs = np.array([x['tg'] for x in data['train']]).reshape(-1,1)
        sc = StandardScaler(); sc.fit(tgs)

        model = ImprovedModel(ad, ed, gfdim=gd, hid=HIDDEN, ngnn=N_GNN, nlg=N_LG,
                              lgh=LG_HIDDEN, drop=DROP, use_lg=True, use_vn=True).to(device=DEVICE)
        np_ = sum(p.numel() for p in model.parameters())
        print(f"Params: {np_:,}")

        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        # Cosine annealing with warmup
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)
        crit = nn.HuberLoss(delta=1.0)

        bvr, bep, pc, bs = float('inf'), 0, 0, None
        for ep in range(MAX_EPOCHS):
            t0 = time.time()
            tl, tr = train_ep(model, trl, opt, crit, sc, DEVICE)
            vl, vr, vm, v2, _, _ = evaluate(model, val, crit, sc, DEVICE)
            sched.step()

            if (ep+1)%10==0 or ep==0:
                print(f"  Ep{ep+1:3d} Tr={tr:.2f} Va={vr:.2f} R2={v2:.4f} {time.time()-t0:.1f}s")
            if vr < bvr:
                bvr=vr; bep=ep+1; pc=0; bs={k:v.cpu().clone() for k,v in model.state_dict().items()}
            else: pc+=1
            if pc>=PATIENCE: print(f"  Early stop ep{ep+1}"); break

        if bs: model.load_state_dict({k:v.to(DEVICE) for k,v in bs.items()})
        _,_,tmae,tr2,ttg,tpr = evaluate(model, tel, crit, sc, DEVICE)
        trmse = np.sqrt(mean_squared_error(ttg, tpr))
        print(f"  Seed{seed}: Val={bvr:.2f}@ep{bep} Test: RMSE={trmse:.2f} MAE={tmae:.2f} R2={tr2:.4f}")

        per_seed_results.append({'seed':seed, 'test_rmse':trmse, 'test_mae':tmae,
                                  'test_r2':tr2, 'best_val':bvr, 'best_ep':bep})
        all_test_preds.append(np.array(tpr))
        if all_test_tgts is None: all_test_tgts = np.array(ttg)

        torch.cuda.empty_cache()

    # Ensemble: average predictions
    ens_preds = np.mean(all_test_preds, axis=0)
    ens_rmse = np.sqrt(mean_squared_error(all_test_tgts, ens_preds))
    ens_mae = mean_absolute_error(all_test_tgts, ens_preds)
    ens_r2 = r2_score(all_test_tgts, ens_preds)

    print(f"\n  ENSEMBLE ({len(seed_list)} seeds): RMSE={ens_rmse:.2f} MAE={ens_mae:.2f} R2={ens_r2:.4f}")

    return per_seed_results, ens_preds, all_test_tgts, ens_rmse, ens_mae, ens_r2


def main():
    print("="*70+"\nIAPT v2 IMPROVED: Virtual Node + Multi-Scale Pooling + Global Features")
    print("Target: Beat RMSE 30.8\n"+"="*70)
    print(f"Device: {DEVICE} Hidden={HIDDEN} GNN={N_GNN} BS={BATCH_SIZE}")
    os.makedirs(WORK_DIR, exist_ok=True)

    data = load_data(seed=42)
    for k,v in data.items(): print(f"  {k}: {len(v)}")

    per_seed, ens_preds, ens_tgts, ens_rmse, ens_mae, ens_r2 = run_multi_seed(data)

    # Run single best model for plots
    print(f"\n{'='*70}\nRunning final best model for plots...\n{'='*70}")
    torch.manual_seed(42); np.random.seed(42)
    tr = PDataset(data['train']); va = PDataset(data['val']); te = PDataset(data['test'])
    trl = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val = DataLoader(va, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    tel = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    s = tr[0]; ad=s['af'].shape[1]; ed=s['ef'].shape[1]
    gd = s['gf'].shape[0] if s['gf'].dim() == 1 else s['gf'].shape[-1]
    tgs = np.array([x['tg'] for x in data['train']]).reshape(-1,1)
    sc = StandardScaler(); sc.fit(tgs)

    model = ImprovedModel(ad, ed, gfdim=gd, hid=HIDDEN, ngnn=N_GNN, nlg=N_LG,
                          lgh=LG_HIDDEN, drop=DROP).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)
    crit = nn.HuberLoss(delta=1.0)
    bvr, bep, pc, bs = float('inf'), 0, 0, None
    tr_losses, vl_losses = [], []

    for ep in range(MAX_EPOCHS):
        t0 = time.time()
        tl, tr_ = train_ep(model, trl, opt, crit, sc, DEVICE)
        vl, vr, vm, v2, _, _ = evaluate(model, val, crit, sc, DEVICE)
        sched.step(); tr_losses.append(tl); vl_losses.append(vl)
        if (ep+1)%10==0:
            print(f"  Ep{ep+1:3d} Tr={tr_:.2f} Va={vr:.2f} R2={v2:.4f} {time.time()-t0:.1f}s")
        if vr < bvr:
            bvr=vr; bep=ep+1; pc=0; bs={k:v.cpu().clone() for k,v in model.state_dict().items()}
        else: pc+=1
        if pc>=PATIENCE: break
    if bs: model.load_state_dict({k:v.to(DEVICE) for k,v in bs.items()})
    _,_,tmae,tr2,ttg,tpr = evaluate(model, tel, crit, sc, DEVICE)
    trmse = np.sqrt(mean_squared_error(ttg, tpr))

    # Save results
    results = {
        'single_seed_rmse': trmse,
        'single_seed_mae': tmae,
        'single_seed_r2': tr2,
        'ensemble_rmse': ens_rmse,
        'ensemble_mae': ens_mae,
        'ensemble_r2': ens_r2,
        'best_val_rmse': bvr,
        'best_epoch': bep,
        'num_params': sum(p.numel() for p in model.parameters()),
        'n_seeds': len(per_seed),
    }
    for ps in per_seed:
        results[f"seed{ps['seed']}_rmse"] = ps['test_rmse']

    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(WORK_DIR, 'results_v2.csv'), index=False)

    # Also update the main results.csv with comparison
    ablation_df = pd.read_csv(os.path.join(WORK_DIR, 'results.csv'))
    improved_row = pd.DataFrame([{
        'variant': 'IAPT_v2_improved_ensemble',
        'test_rmse': ens_rmse, 'test_mae': ens_mae, 'test_r2': ens_r2,
        'best_val_rmse': bvr, 'best_epoch': bep,
        'num_params': results['num_params']
    }])
    ablation_df = pd.concat([ablation_df, improved_row], ignore_index=True)
    ablation_df.to_csv(os.path.join(WORK_DIR, 'results.csv'), index=False)

    # Plots
    pd_dir = os.path.join(WORK_DIR, 'plots'); os.makedirs(pd_dir, exist_ok=True)

    # Parity plot (ensemble)
    fig,ax=plt.subplots(figsize=(7,7))
    ax.scatter(ens_tgts, ens_preds, alpha=0.4, s=15)
    mn,mx=min(ens_tgts.min(),ens_preds.min()),max(ens_tgts.max(),ens_preds.max())
    ax.plot([mn,mx],[mn,mx],'r--',lw=2,label='Perfect')
    c=np.polyfit(ens_tgts,ens_preds,1); x=np.linspace(mn,mx,100)
    ax.plot(x,np.polyval(c,x),'b-',lw=2,label=f'y={c[0]:.2f}x+{c[1]:.1f}')
    ax.set_xlabel('True Tg');ax.set_ylabel('Pred Tg')
    ax.set_title(f'IAPT v2 Ensemble RMSE={ens_rmse:.2f}')
    ax.legend();ax.grid(True,alpha=0.3);fig.tight_layout()
    fig.savefig(os.path.join(pd_dir,'ensemble_parity.png'),dpi=150);plt.close()

    # Bar chart with all results
    fig,ax=plt.subplots(figsize=(12,6))
    names = list(ablation_df['variant']); rmses = list(ablation_df['test_rmse'])
    cols = ['green' if r<30.8 else 'steelblue' for r in rmses]
    bars = ax.bar(names, rmses, color=cols)
    ax.axhline(30.8, color='red', ls='--', lw=2, label='Target=30.8')
    ax.set_ylabel('Test RMSE'); ax.set_title('IAPT v2 Full Comparison')
    ax.legend(); plt.xticks(rotation=45, ha='right')
    for b,r in zip(bars,rmses): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,f'{r:.1f}',ha='center',fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(pd_dir,'full_comparison.png'),dpi=150);plt.close()

    # Training curves
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(tr_losses,'b-',label='Train'); ax.plot(vl_losses,'r-',label='Val')
    ax.set_xlabel('Epoch');ax.set_ylabel('Loss');ax.set_title('Training Curves')
    ax.legend();ax.grid(True,alpha=0.3);fig.tight_layout()
    fig.savefig(os.path.join(pd_dir,'improved_training.png'),dpi=150);plt.close()

    # Error distribution
    err = ens_preds - ens_tgts
    fig,ax=plt.subplots(figsize=(8,5))
    ax.hist(err,bins=50,edgecolor='black',alpha=0.7)
    ax.axvline(0,color='r',ls='--',lw=2)
    ax.set_xlabel('Error');ax.set_ylabel('Count');ax.set_title('Ensemble Error Distribution')
    ax.grid(True,alpha=0.3);fig.tight_layout()
    fig.savefig(os.path.join(pd_dir,'ensemble_error.png'),dpi=150);plt.close()

    # Summary
    print("\n"+"="*70+"\nFINAL RESULTS\n"+"="*70)
    print("\nAblation Results:")
    print(ablation_df.to_string(index=False))
    print(f"\nBest single model: RMSE={trmse:.2f}")
    print(f"Ensemble ({len(per_seed)} seeds): RMSE={ens_rmse:.2f}")
    print(f"Target: 30.8")
    if ens_rmse < 30.8:
        print(f"SUCCESS! Beat target by {30.8-ens_rmse:.2f}")
    else:
        print(f"Gap: {ens_rmse-30.8:.2f}")
    print(f"\nResults saved to {WORK_DIR}/results_v2.csv and {WORK_DIR}/results.csv")

if __name__=='__main__':
    main()
