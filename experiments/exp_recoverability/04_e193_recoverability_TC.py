"""E193 -- TC replication of E143 + E144 + E192.

METHODOLOGY FROZEN. Transplanted verbatim from e143.py / e144.py / e192.
ONLY change: target region ET (seg==3) -> TC (seg==1 or seg==3).

BraTS 2023 labels: 1=NCR, 2=ED, 3=ET.  TC := NCR + ET = {1,3}.  WT := {1,2,3}.

Controls preserved (user-specified, pre-registered):
  1. Same 5-fold cross-fitting; observer never sees target subject's TC labels.
  2. Same Z (15 pre-specified vars) -- NOT redesigned for TC. Only the two
     ET-specific entries become their TC counterparts by direct substitution
     (log ET size -> log TC size; ET fraction -> TC fraction), because they are
     definitionally region-indexed. No new variables, no reselection.
  3. Same 200-shuffle permutation null.
  4. Same model set (all 6 e130 evals have dice_TC).
  5. ET and TC kept completely separate -- ET O_i is NOT an input here.
"""
import json, glob, os, sys
import numpy as np
from scipy import stats, ndimage
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import nibabel as nib
from pathlib import Path
import torch, torch.nn as nn

ROOT = Path('C:/Users/Rivan/Projects/Neuroscan')
D = ROOT/'Dataset/Training/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
OUT = ROOT/'experiments/exp_e12_eggo_m'

# ---------- subject selection: mirrors E143 exactly (>=200 target voxels) ----------
b = {x['subject_id']: x for x in json.load(open(
    OUT/'e130/E130_full_eval_E131_v5control_seed0_per_subject.json'))}
ids_all = [s for s in sorted(b) if b[s]['target_voxels_TC'] >= 200]
rng = np.random.default_rng(0)                       # same seed as E143
N = 90                                                # same N as E143
ids = list(rng.permutation(ids_all)[:N])
print(f'TC subjects with >=200 TC voxels: {len(ids_all)}; sampled {len(ids)}', flush=True)

FEAT = ['t1c','t1n','t2f','t2w']
def load(sid):
    seg = nib.load(str(D/sid/(sid+'-seg.nii.gz'))).get_fdata().astype(np.uint8)
    ch = []
    for m in FEAT:
        v = nib.load(str(D/sid/(sid+'-'+m+'.nii.gz'))).get_fdata().astype(np.float32)
        br = v > 0; ch.append((v-v[br].mean())/max(1e-6, v[br].std()))
    X = np.stack(ch,-1); tum = seg > 0
    tgt = (seg == 1) | (seg == 3)                    # <<< TC instead of ET
    idx = np.flatnonzero(tum.ravel())
    if len(idx) > 12000: idx = rng.choice(idx, 12000, replace=False)
    F = X.reshape(-1,4)[idx]; Y = tgt.ravel()[idx].astype(np.float32)
    d = (X[...,0]-X[...,1])
    lm = ndimage.uniform_filter(d,5)
    ls = np.sqrt(np.maximum(ndimage.uniform_filter(d*d,5)-lm*lm, 0))
    S = np.stack([lm.ravel()[idx], ls.ravel()[idx]],-1)
    return F, Y, S

print('loading...', flush=True)
DAT = {}
for i, s in enumerate(ids):
    try: DAT[s] = load(s)
    except Exception as e: print('  skip', s, e, flush=True)
    if (i+1) % 20 == 0: print(f'  {i+1}/{len(ids)}', flush=True)
ids = [s for s in ids if s in DAT]
print(f'loaded {len(ids)}', flush=True)

def dice(p,y): return float(2*(p*y).sum()/max(1e-6, p.sum()+y.sum()))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', dev, flush=True)

def fit_mlp(tr, hidden, epochs, use_spatial, seed=0):
    torch.manual_seed(seed)
    nin = 4+(2 if use_spatial else 0)
    layers=[]; prev=nin
    for h in hidden: layers += [nn.Linear(prev,h), nn.ReLU()]; prev=h
    layers += [nn.Linear(prev,1)]
    net = nn.Sequential(*layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xs=[]; Ys=[]
    for s in tr:
        F,Y,S = DAT[s]; A = np.hstack([F,S]) if use_spatial else F
        Xs.append(torch.tensor(A, device=dev)); Ys.append(torch.tensor(Y, device=dev))
    for _ in range(epochs):
        k = np.random.randint(len(Xs)); j = torch.randint(0, len(Xs[k]), (4096,), device=dev)
        l = nn.functional.binary_cross_entropy_with_logits(net(Xs[k][j]).squeeze(1), Ys[k][j])
        opt.zero_grad(); l.backward(); opt.step()
    return net

def eval_mlp(net, s, use_spatial):
    F,Y,S = DAT[s]; A = np.hstack([F,S]) if use_spatial else F
    with torch.no_grad():
        p = (torch.sigmoid(net(torch.tensor(A, device=dev)).squeeze(1)).cpu().numpy() >= 0.5).astype(np.float32)
    return dice(p, Y)

def fit_thresh(tr):
    best = (-1, 0.0)
    for t in np.linspace(-1,3,41):
        v = np.mean([dice((DAT[s][0][:,0]-DAT[s][0][:,1] >= t).astype(np.float32), DAT[s][1]) for s in tr])
        if v > best[0]: best = (v, t)
    return best[1]

OBS = [('simple_thresh', None),
       ('mlp_small',   dict(hidden=[8],     epochs=600,  use_spatial=False)),
       ('mlp_deep',    dict(hidden=[64,64], epochs=1500, use_spatial=False)),
       ('mlp_spatial', dict(hidden=[64,64], epochs=1500, use_spatial=True))]
K = 5
folds = np.array_split(np.array(ids), K)
O = {n:{} for n,_ in OBS}
for fi, te in enumerate(folds):
    tr = [s for s in ids if s not in set(te)]
    print(f'fold {fi+1}/{K}  train={len(tr)} test={len(te)}', flush=True)
    th = fit_thresh(tr)
    for s in te:
        O['simple_thresh'][s] = dice((DAT[s][0][:,0]-DAT[s][0][:,1] >= th).astype(np.float32), DAT[s][1])
    for name, cfg in OBS[1:]:
        net = fit_mlp(tr, **cfg)
        for s in te: O[name][s] = eval_mlp(net, s, cfg['use_spatial'])

json.dump({k:{s:float(v) for s,v in d.items()} for k,d in O.items()},
          open(OUT/'E193_recoverability_TC.json','w'), indent=1)

print()
print('='*70)
print(f'E193 -- CROSS-FITTED TC RECOVERABILITY, {len(ids)} subjects, {K}-fold')
print('='*70)
print('Per-observer held-out TC recoverability (mean):')
for n,_ in OBS: print(f'  {n:14s} {float(np.mean([O[n][s] for s in ids])):.4f}')
print()
print('OBSERVER-INDEPENDENCE (Spearman between observer subject rankings):')
names = [n for n,_ in OBS]
print('                 '+''.join(f'{n[:12]:>14s}' for n in names))
for a in names:
    row = f'  {a[:12]:14s}'
    for c in names:
        row += f'{stats.spearmanr([O[a][s] for s in ids],[O[c][s] for s in ids]).statistic:14.3f}'
    print(row)
