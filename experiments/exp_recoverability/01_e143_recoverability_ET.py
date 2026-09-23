"""E143 -- cross-fitted ET recoverability O_i.  [PROMOTED FROM SCRATCHPAD 2026-09-20]

Reconstructed verbatim from the original scratchpad `e143.py` (session
a8041820), which was deleted from the temp directory after E198 ran. The
stored artifact `E143_recoverability.json` was produced by this exact code;
its six pairwise observer correlations were independently recomputed on
2026-09-20 and matched the reported 0.786-0.974 to three decimals.

FIXES the two defects the audit found in E136:
  1. O_i was fitted AND evaluated on the same subject (leakage).
     -> here: 5-fold cross-fitting, every observer trained ONLY on other subjects.
  2. O_i came from ONE estimator, so it measured "recoverability according to f".
     -> here: FOUR deliberately different observers; we test whether the
        SUBJECT RANKING is stable across them (observer-independence).

Run from the repo root:  python experiments/exp_recoverability/01_e143_recoverability_ET.py
Output: experiments/exp_e12_eggo_m/E143_recoverability.json
"""
import sys, os, json, numpy as np
sys.path.insert(0,'.')
import nibabel as nib
from pathlib import Path
from scipy import stats, ndimage
import torch, torch.nn as nn

D=Path('Dataset/Training/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData')
b={x['subject_id']:x for x in json.load(open('experiments/exp_e12_eggo_m/e130/E130_full_eval_E131_v5control_seed0_per_subject.json'))}
ids=[s for s in sorted(b) if b[s]['target_voxels_ET']>=200]
rng=np.random.default_rng(0)
N=int(os.environ.get('E143_N','90'))
ids=list(rng.permutation(ids)[:N])
print('subjects: {}'.format(len(ids)),flush=True)

FEAT=['t1c','t1n','t2f','t2w']
def load(sid):
    seg=nib.load(str(D/sid/(sid+'-seg.nii.gz'))).get_fdata().astype(np.uint8)
    ch=[]
    for m in FEAT:
        v=nib.load(str(D/sid/(sid+'-'+m+'.nii.gz'))).get_fdata().astype(np.float32)
        br=v>0; ch.append((v-v[br].mean())/max(1e-6,v[br].std()))
    X=np.stack(ch,-1); tum=seg>0; et=(seg==3)
    idx=np.flatnonzero(tum.ravel())
    if len(idx)>12000: idx=rng.choice(idx,12000,replace=False)
    F=X.reshape(-1,4)[idx]; Y=et.ravel()[idx].astype(np.float32)
    # spatial features: local mean/std of delta in a 5^3 window
    d=(X[...,0]-X[...,1])
    lm=ndimage.uniform_filter(d,5); ls=np.sqrt(np.maximum(ndimage.uniform_filter(d*d,5)-lm*lm,0))
    S=np.stack([lm.ravel()[idx],ls.ravel()[idx]],-1)
    return F,Y,S

print('loading...',flush=True)
DAT={}
for i,s in enumerate(ids):
    try: DAT[s]=load(s)
    except Exception as e: print('  skip',s,e,flush=True)
    if (i+1)%20==0: print('  {}/{}'.format(i+1,len(ids)),flush=True)
ids=[s for s in ids if s in DAT]
print('loaded {}'.format(len(ids)),flush=True)

def dice(p,y): return float(2*(p*y).sum()/max(1e-6,p.sum()+y.sum()))
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fit_mlp(tr,hidden,epochs,use_spatial,seed=0):
    torch.manual_seed(seed)
    nin=4+(2 if use_spatial else 0)
    layers=[]; prev=nin
    for h in hidden: layers+=[nn.Linear(prev,h),nn.ReLU()]; prev=h
    layers+=[nn.Linear(prev,1)]
    net=nn.Sequential(*layers).to(dev)
    opt=torch.optim.Adam(net.parameters(),1e-3)
    Xs=[];Ys=[]
    for s in tr:
        F,Y,S=DAT[s]; A=np.hstack([F,S]) if use_spatial else F
        Xs.append(torch.tensor(A,device=dev)); Ys.append(torch.tensor(Y,device=dev))
    for _ in range(epochs):
        k=np.random.randint(len(Xs)); j=torch.randint(0,len(Xs[k]),(4096,),device=dev)
        l=nn.functional.binary_cross_entropy_with_logits(net(Xs[k][j]).squeeze(1),Ys[k][j])
        opt.zero_grad(); l.backward(); opt.step()
    return net

def eval_mlp(net,s,use_spatial):
    F,Y,S=DAT[s]; A=np.hstack([F,S]) if use_spatial else F
    with torch.no_grad():
        p=(torch.sigmoid(net(torch.tensor(A,device=dev)).squeeze(1)).cpu().numpy()>=0.5).astype(np.float32)
    return dice(p,Y)

def fit_thresh(tr):
    # f_simple: single global threshold on delta = t1c - t1n, fitted on train subjects
    best=(-1,0.0)
    for t in np.linspace(-1,3,41):
        v=np.mean([dice((DAT[s][0][:,0]-DAT[s][0][:,1]>=t).astype(np.float32),DAT[s][1]) for s in tr])
        if v>best[0]: best=(v,t)
    return best[1]

OBS=[('simple_thresh',None),('mlp_small',dict(hidden=[8],epochs=600,use_spatial=False)),
     ('mlp_deep',dict(hidden=[64,64],epochs=1500,use_spatial=False)),
     ('mlp_spatial',dict(hidden=[64,64],epochs=1500,use_spatial=True))]
K=5
folds=np.array_split(np.array(ids),K)
O={n:{} for n,_ in OBS}
for fi,te in enumerate(folds):
    tr=[s for s in ids if s not in set(te)]
    print('fold {}/{}  train={} test={}'.format(fi+1,K,len(tr),len(te)),flush=True)
    th=fit_thresh(tr)
    for s in te: O['simple_thresh'][s]=dice((DAT[s][0][:,0]-DAT[s][0][:,1]>=th).astype(np.float32),DAT[s][1])
    for name,cfg in OBS[1:]:
        net=fit_mlp(tr,**cfg)
        for s in te: O[name][s]=eval_mlp(net,s,cfg['use_spatial'])
json.dump({k:{s:float(v) for s,v in d.items()} for k,d in O.items()},
          open('experiments/exp_e12_eggo_m/E143_recoverability.json','w'),indent=1)
print()
print('='*66)
print('E143 -- CROSS-FITTED RECOVERABILITY, {} subjects, {}-fold'.format(len(ids),K))
print('='*66)
print()
print('Per-observer held-out recoverability (mean):')
for n,_ in OBS: print('  {:14s} {:.4f}'.format(n,float(np.mean([O[n][s] for s in ids]))))
print()
print('OBSERVER-INDEPENDENCE: Spearman between observers subject rankings')
names=[n for n,_ in OBS]
print('                 '+''.join('{:>14s}'.format(n[:12]) for n in names))
for a in names:
    row='  {:14s}'.format(a[:12])
    for c in names:
        r=stats.spearmanr([O[a][s] for s in ids],[O[c][s] for s in ids]).statistic
        row+='{:14.3f}'.format(r)
    print(row)
