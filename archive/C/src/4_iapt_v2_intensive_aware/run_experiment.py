#!/usr/bin/env python3
"""
IAPT v2: Intensive-Aware Polymer Transformer for Tg Prediction

GINE backbone with:
1. Monomer conformer features (bond lengths, angles, dihedrals)
2. 3 repeat units (trimer) with periodic edges
3. Equivariant L1 features (dipole-like)
4. Line graph for angular relationships
5. Pooling comparison: mean vs max vs attention vs sum

Ablation: IAPT_v2_mean, _max, _att, _sum, _no_eq, _no_lg
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
from rdkit.Chem import AllChem

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings('ignore')

# === PATHS ===
CSV_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
WORK_DIR = '/home/home_public/aisci/AIscientist/src/4_iapt_v2_intensive_aware'
CACHE_PATH = os.path.join(WORK_DIR, 'processed_data.pkl')
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/alldataset/some_dataprocesscode')
from multimer import generate_multimer_smiles

# === CONFIG ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
MAX_EPOCHS = 500
PATIENCE = 30
LR = 1e-3
WD = 1e-5
HIDDEN = 256
N_GNN = 5
N_LG = 2
LG_HIDDEN = 128
DROP = 0.1

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
    ctr = pos.mean(0)
    disp = pos - ctr
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

def build_lg(ei):
    ne = ei.shape[1]
    if ne==0: return np.zeros((2,0), dtype=np.int64)
    n2e = {}
    for e in range(ne):
        u = int(ei[0,e])
        n2e.setdefault(u,[]).append(e)
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
                    pi = np.array(cf.GetAtomPosition(bi))
                    pj = np.array(cf.GetAtomPosition(bj))
                    l = np.linalg.norm(pi-pj)
                except: l = 1.5
                for idx,(u,v) in enumerate(el):
                    if u==bi and v==bj: bl[idx]=l
            ef = np.concatenate([ef, bl.reshape(-1,1)/2.0], axis=1)
            eq = equiv_feat(m3)
            af = np.concatenate([af, eq], axis=1)
        else:
            ef = np.concatenate([ef, np.full((len(el),1), 0.75, dtype=np.float32)], axis=1)
            af = np.concatenate([af, np.zeros((na,3), dtype=np.float32)], axis=1)

        # Periodic edges: connect chain-end atoms
        ends = [a.GetIdx() for a in tm.GetAtoms() if a.GetDegree()==1 and not a.IsInRing()]
        if len(ends)>=2:
            s1,s2 = ends[0], ends[-1]
            pe = np.zeros(ef.shape[1], dtype=np.float32); pe[0]=1.0
            ei = np.concatenate([ei, np.array([[s1,s2],[s2,s1]], dtype=np.int64)], axis=1)
            ef = np.concatenate([ef, np.stack([pe,pe])], axis=0)

        lg = build_lg(ei)
        return {'af':af, 'ei':ei, 'ef':ef, 'lg':lg, 'tg':float(tg), 'smi':ps, 'na':na, 'cok':cok}
    except: return None

def load_data(seed=42):
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached data from {CACHE_PATH}")
        with open(CACHE_PATH,'rb') as f: return pickle.load(f)
    print("Processing dataset...")
    df = pd.read_csv(CSV_PATH)
    np.random.seed(seed)
    idx = np.random.permutation(len(df))
    nt,nv = int(len(df)*0.72), int(len(df)*0.08)
    splits = {'train': idx[:nt], 'val': idx[nt:nt+nv], 'test': idx[nt+nv:]}
    data = {}
    for sn, si in splits.items():
        print(f"\n  {sn}: {len(si)} samples...")
        samp, fail, cok = [], 0, 0
        for c,i in enumerate(si):
            r = process_one(df.iloc[i]['Smiles'], df.iloc[i]['Tg'])
            if r: samp.append(r); cok += r['cok']
            else: fail += 1
            if (c+1)%1000==0: print(f"    {c+1}/{len(si)} ok={len(samp)} fail={fail}")
        print(f"    Done: ok={len(samp)} fail={fail} conf={cok}/{len(samp)}")
        data[sn] = samp
    with open(CACHE_PATH,'wb') as f: pickle.dump(data, f)
    print(f"  Cached to {CACHE_PATH}")
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
                'tg':torch.tensor([s['tg']],dtype=torch.float32), 'na':s['na']}

def collate(batch):
    afs, eis, efs, lgs, tgs, bv = [],[],[],[],[],[]
    ao, eo = 0, 0
    for i,s in enumerate(batch):
        na,ne = s['na'], s['ei'].shape[1]
        afs.append(s['af']); eis.append(s['ei']+ao); efs.append(s['ef'])
        if s['lg'].shape[1]>0: lgs.append(s['lg']+eo)
        else: lgs.append(torch.zeros((2,0),dtype=torch.long))
        tgs.append(s['tg']); bv.append(torch.full((na,),i,dtype=torch.long))
        ao+=na; eo+=ne
    return {'af':torch.cat(afs,0), 'ei':torch.cat(eis,1), 'ef':torch.cat(efs,0),
            'lg':torch.cat(lgs,1) if lgs else torch.zeros((2,0),dtype=torch.long),
            'tg':torch.cat(tgs,0), 'batch':torch.cat(bv,0)}

class Model(nn.Module):
    def __init__(self, adim, edim, hid=HIDDEN, ngnn=N_GNN, nlg=N_LG, lgh=LG_HIDDEN,
                 drop=DROP, pool='mean', use_lg=True):
        super().__init__()
        self.hid = hid; self.pool = pool; self.use_lg = use_lg
        self.aemb = nn.Sequential(nn.Linear(adim,hid), nn.LayerNorm(hid), nn.GELU())
        self.eemb = nn.Sequential(nn.Linear(edim,hid), nn.LayerNorm(hid), nn.GELU())
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(ngnn):
            m = nn.Sequential(nn.Linear(hid,hid), nn.BatchNorm1d(hid), nn.GELU(),
                              nn.Dropout(drop), nn.Linear(hid,hid))
            self.convs.append(GINEConv(nn=m, eps=0.0, train_eps=True))
            self.norms.append(nn.LayerNorm(hid))
        if use_lg:
            self.lemb = nn.Linear(edim, lgh)
            self.lconvs = nn.ModuleList()
            for _ in range(nlg):
                m = nn.Sequential(nn.Linear(lgh,lgh), nn.GELU(), nn.Linear(lgh,lgh))
                self.lconvs.append(GINConv(nn=m))
            self.lproj = nn.Sequential(nn.Linear(lgh,hid), nn.GELU())
        if pool=='attention':
            self.aq = nn.Parameter(torch.randn(1,hid))
            self.ak = nn.Linear(hid,hid); self.av = nn.Linear(hid,hid)
        self.sw = nn.Parameter(torch.ones(ngnn)/ngnn)
        self.pred = nn.Sequential(nn.LayerNorm(hid), nn.Linear(hid,hid), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(hid,hid//2), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(hid//2,1))

    def forward(self, d):
        x = self.aemb(d['af']); ei = d['ei']; ea = self.eemb(d['ef'])
        if self.use_lg and hasattr(self,'lemb') and d['lg'].shape[1]>0:
            lx = self.lemb(d['ef'])
            for lc in self.lconvs: lx = lx + lc(lx, d['lg'])
            ea = ea + self.lproj(lx)
        outs = []
        for conv,norm in zip(self.convs, self.norms):
            if ei.shape[1]>0: x = norm(x + conv(x,ei,ea))
            else: x = norm(x + conv.nn(x))
            x = F.gelu(x); outs.append(x)
        w = F.softmax(self.sw, dim=0)
        x = sum(wi*xi for wi,xi in zip(w, outs))
        b = d['batch']
        if self.pool=='mean': p = global_mean_pool(x,b)
        elif self.pool=='max': p = global_max_pool(x,b)
        elif self.pool=='sum': p = global_add_pool(x,b)
        elif self.pool=='attention':
            ks,vs = self.ak(x), self.av(x)
            bs = d['tg'].shape[0]; sc = self.hid**0.5; pp=[]
            for bi in range(bs):
                m = (b==bi)
                if m.sum()==0: pp.append(torch.zeros(1,self.hid,device=x.device)); continue
                k,v = ks[m], vs[m]
                s = F.softmax((self.aq.expand(k.shape[0],-1)*k).sum(-1)/sc, dim=0)
                pp.append((s.unsqueeze(-1)*v).sum(0,keepdim=True))
            p = torch.cat(pp,0)
        else: p = global_mean_pool(x,b)
        return self.pred(p)

def train_ep(model, ld, opt, crit, sc, dev):
    model.train(); tl=0; pa,ta=[],[]
    for b in ld:
        af,ei,ef,lg,bv,tg = [b[k].to(dev) for k in ['af','ei','ef','lg','batch','tg']]
        tn = torch.tensor(sc.transform(tg.cpu().numpy().reshape(-1,1)), dtype=torch.float32, device=dev)
        opt.zero_grad()
        pr = model({'af':af,'ei':ei,'ef':ef,'lg':lg,'batch':bv,'tg':tg})
        loss = crit(pr,tn); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tl += loss.item()*pr.shape[0]
        pdn = sc.inverse_transform(pr.detach().cpu().numpy().reshape(-1,1)).flatten()
        pa.extend(pdn.flatten()); ta.extend(tg.cpu().numpy().flatten())
    return tl/len(pa), np.sqrt(mean_squared_error(ta,pa))

@torch.no_grad()
def evaluate(model, ld, crit, sc, dev):
    model.eval(); tl=0; pa,ta=[],[]
    for b in ld:
        af,ei,ef,lg,bv,tg = [b[k].to(dev) for k in ['af','ei','ef','lg','batch','tg']]
        tn = torch.tensor(sc.transform(tg.cpu().numpy().reshape(-1,1)), dtype=torch.float32, device=dev)
        pr = model({'af':af,'ei':ei,'ef':ef,'lg':lg,'batch':bv,'tg':tg})
        tl += crit(pr,tn).item()*pr.shape[0]
        pdn = sc.inverse_transform(pr.cpu().numpy().reshape(-1,1)).flatten()
        pa.extend(pdn.flatten()); ta.extend(tg.cpu().numpy().flatten())
    return tl/len(pa), np.sqrt(mean_squared_error(ta,pa)), mean_absolute_error(ta,pa), r2_score(ta,pa), ta, pa

def run_variant(name, pool, trl, val, tel, sc, ad, ed, dev, ulg=True, ueq=True):
    print(f"\n{'='*70}\nVariant: {name} pool={pool} lg={ulg} eq={ueq}\n{'='*70}")
    m = Model(ad, ed, pool=pool, use_lg=ulg).to(dev)
    np_ = sum(p.numel() for p in m.parameters())
    print(f"Params: {np_:,}")
    opt = optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt,'min',factor=0.5,patience=15)
    crit = nn.HuberLoss(delta=1.0)
    bvr, bep, pc, bs = float('inf'), 0, 0, None
    tl_list, vl_list = [], []

    for ep in range(MAX_EPOCHS):
        t0 = time.time()
        tl, tr = train_ep(m, trl, opt, crit, sc, dev)
        vl, vr, vm, v2, _, _ = evaluate(m, val, crit, sc, dev)
        sch.step(vr); tl_list.append(tl); vl_list.append(vl)
        if (ep+1)%10==0 or ep==0:
            print(f"  Ep{ep+1:3d} TrRMSE={tr:.2f} VaRMSE={vr:.2f} VaR2={v2:.4f} {time.time()-t0:.1f}s")
        if vr < bvr:
            bvr=vr; bep=ep+1; pc=0; bs={k:v.cpu().clone() for k,v in m.state_dict().items()}
        else: pc+=1
        if pc>=PATIENCE: print(f"  Early stop ep{ep+1}"); break

    if bs: m.load_state_dict({k:v.to(dev) for k,v in bs.items()})
    _,_,tmae,tr2,ttg,tpr = evaluate(m, tel, crit, sc, dev)
    trmse = np.sqrt(mean_squared_error(ttg, tpr))
    print(f"\n  BestVal={bvr:.2f}@ep{bep} Test: RMSE={trmse:.2f} MAE={tmae:.2f} R2={tr2:.4f}")

    torch.save(m.state_dict(), os.path.join(WORK_DIR, f'model_{name}.pt'))
    pd = os.path.join(WORK_DIR,'plots'); os.makedirs(pd, exist_ok=True)

    # Parity
    fig,ax=plt.subplots(figsize=(7,7))
    ax.scatter(ttg,tpr,alpha=0.4,s=15)
    mn,mx=min(min(ttg),min(tpr)),max(max(ttg),max(tpr))
    ax.plot([mn,mx],[mn,mx],'r--',lw=2,label='Perfect')
    c=np.polyfit(ttg,tpr,1); x=np.linspace(mn,mx,100)
    ax.plot(x,np.polyval(c,x),'b-',lw=2,label=f'y={c[0]:.2f}x+{c[1]:.1f}')
    ax.set_xlabel('True Tg');ax.set_ylabel('Pred Tg');ax.set_title(f'{name} RMSE={trmse:.2f}')
    ax.legend();ax.grid(True,alpha=0.3);fig.tight_layout()
    fig.savefig(os.path.join(pd,f'{name}_parity.png'),dpi=150);plt.close()

    # Error dist
    err=np.array(tpr)-np.array(ttg)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.hist(err,bins=50,edgecolor='black',alpha=0.7)
    ax.axvline(0,color='r',ls='--',lw=2);ax.set_xlabel('Error');ax.set_ylabel('Count')
    ax.set_title(f'{name} Error Dist');ax.grid(True,alpha=0.3);fig.tight_layout()
    fig.savefig(os.path.join(pd,f'{name}_err.png'),dpi=150);plt.close()

    # Training curves
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(tl_list,'b-',label='Train');ax.plot(vl_list,'r-',label='Val')
    ax.set_xlabel('Epoch');ax.set_ylabel('Loss');ax.set_title(f'{name} Training')
    ax.legend();ax.grid(True,alpha=0.3);fig.tight_layout()
    fig.savefig(os.path.join(pd,f'{name}_train.png'),dpi=150);plt.close()

    return {'variant':name,'test_rmse':trmse,'test_mae':tmae,'test_r2':tr2,
            'best_val_rmse':bvr,'best_epoch':bep,'num_params':np_}

def main():
    print("="*70+"\nIAPT v2: Intensive-Aware Polymer Transformer for Tg\nTarget: Beat RMSE 30.8\n"+"="*70)
    print(f"Device: {DEVICE} Hidden={HIDDEN} GNN={N_GNN} LG={N_LG} BS={BATCH_SIZE}")
    os.makedirs(WORK_DIR, exist_ok=True)

    data = load_data(seed=42)
    for k,v in data.items(): print(f"  {k}: {len(v)}")

    variants = [
        ('IAPT_v2_mean','mean',True,True),
        ('IAPT_v2_max','max',True,True),
        ('IAPT_v2_att','attention',True,True),
        ('IAPT_v2_sum','sum',True,True),
        ('IAPT_v2_no_eq','mean',True,False),
        ('IAPT_v2_no_lg','mean',False,True),
    ]

    results = []
    for name,pool,ulg,ueq in variants:
        tr = PDataset(data['train'],ueq,ulg)
        va = PDataset(data['val'],ueq,ulg)
        te = PDataset(data['test'],ueq,ulg)
        trl = DataLoader(tr,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate)
        val = DataLoader(va,batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate)
        tel = DataLoader(te,batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate)
        s = tr[0]; ad=s['af'].shape[1]; ed=s['ef'].shape[1]
        tgs = np.array([x['tg'] for x in data['train']]).reshape(-1,1)
        sc = StandardScaler(); sc.fit(tgs)
        r = run_variant(name,pool,trl,val,tel,sc,ad,ed,DEVICE,ulg,ueq)
        results.append(r); torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    rp = os.path.join(WORK_DIR,'results.csv'); df.to_csv(rp,index=False)

    print("\n"+"="*70+"\nFINAL RESULTS\n"+"="*70)
    print(f"\n{'Variant':<20} {'RMSE':<10} {'MAE':<10} {'R2':<10} {'BestEp':<10}")
    print("-"*60)
    for r in results:
        print(f"{r['variant']:<20} {r['test_rmse']:<10.2f} {r['test_mae']:<10.2f} "
              f"{r['test_r2']:<10.4f} {r['best_epoch']:<10}")

    best = min(results, key=lambda x:x['test_rmse'])
    print(f"\nBest: {best['variant']} RMSE={best['test_rmse']:.2f}")
    print(f"Target: 30.8")
    if best['test_rmse']<30.8:
        print(f"SUCCESS! Beat by {30.8-best['test_rmse']:.2f}")
    else:
        print(f"Gap: {best['test_rmse']-30.8:.2f}")

    # Bar chart
    fig,ax=plt.subplots(figsize=(10,6))
    ns=[r['variant'] for r in results]; rs=[r['test_rmse'] for r in results]
    cols=['green' if r<30.8 else 'steelblue' for r in rs]
    bars=ax.bar(ns,rs,color=cols)
    ax.axhline(30.8,color='red',ls='--',lw=2,label='Target=30.8')
    ax.set_ylabel('Test RMSE');ax.set_title('IAPT v2 Ablation')
    ax.legend();plt.xticks(rotation=45,ha='right')
    for b,r in zip(bars,rs): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,f'{r:.1f}',ha='center',fontsize=10)
    fig.tight_layout();fig.savefig(os.path.join(WORK_DIR,'plots','comparison.png'),dpi=150);plt.close()
    print(f"\nResults: {rp}\nDone!")

if __name__=='__main__':
    main()
