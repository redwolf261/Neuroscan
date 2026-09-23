"""E198 -- DECISIVE: does the tail's ET rule TRANSFER, or is it subject-specific?

E197 (in-subject leakage) got 0.681 on the tail. E195's cross-fitted O_i on the
same population is ~0.279. E198 resolves which is right by removing the leakage
while keeping everything else identical.

ARMS (all use the SAME architecture/epochs as E197; only the TRAINING SET changes):
  1. in_subject   : train on subject i, test on i          (E197 replication, upper bound)
  2. loo_tail     : train on the OTHER tail subjects only   <-- THE TEST
  3. global_good  : train on good-stratum subjects only     (the population boundary)
  4. loo_tail+good: train on other-tail + good              (does tail data add anything?)

PRE-REGISTERED READING (fixed before running):
  loo_tail ~= 0.68  -> rule TRANSFERS within the tail -> real headroom -> E196 reopens
  loo_tail ~= 0.28  -> subject-specific boundary      -> E195/E196 stand, closure complete
  intermediate      -> report as intermediate, do NOT round toward the convenient end

Also reports global_good, because "tail rule differs from population rule" is the
E136 claim that E143 RETRACTED. If loo_tail >> global_good, that claim revives.
"""
import json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from scipy import ndimage
import nibabel as nib

ROOT = Path('C:/Users/Rivan/Projects/Neuroscan')
D = ROOT/'Dataset/Training/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
ev = {x['subject_id']: x for x in json.load(open(
    ROOT/'experiments/exp_e12_eggo_m/e130/E130_full_eval_E131_v5control_seed0_per_subject.json'))}

tail = sorted([s for s,x in ev.items() if x['dice_ET']<0.5 or x['dice_TC']<0.5])
good = sorted([s for s,x in ev.items() if not (x['dice_ET']<0.5 or x['dice_TC']<0.5)])
rng = np.random.default_rng(0)
good_tr = list(rng.permutation(good)[:30])      # fixed training pool from good stratum
print(f'tail n={len(tail)}  good-train pool n={len(good_tr)}')

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def dice(p,y): return float(2*(p*y).sum()/max(1e-6,p.sum()+y.sum()))

CACHE={}
def load(sid):
    if sid in CACHE: return CACHE[sid]
    seg = nib.load(str(D/sid/(sid+'-seg.nii.gz'))).get_fdata().astype(np.uint8)
    ch=[]
    for m in ['t1c','t1n','t2f','t2w']:
        v = nib.load(str(D/sid/(sid+'-'+m+'.nii.gz'))).get_fdata().astype(np.float32)
        br=v>0; ch.append((v-v[br].mean())/max(1e-6,v[br].std()))
    X=np.stack(ch,-1); tum=seg>0; et=(seg==3)
    idx=np.flatnonzero(tum.ravel())
    if len(idx)>12000: idx=rng.choice(idx,12000,replace=False)
    F=X.reshape(-1,4)[idx]; Y=et.ravel()[idx].astype(np.float32)
    d=(X[...,0]-X[...,1])
    lm=ndimage.uniform_filter(d,5); ls=np.sqrt(np.maximum(ndimage.uniform_filter(d*d,5)-lm*lm,0))
    S=np.stack([lm.ravel()[idx],ls.ravel()[idx]],-1)
    A=np.hstack([F,S]).astype(np.float32)        # all-4 + spatial (E197's best arm)
    CACHE[sid]=(A,Y)
    return CACHE[sid]

print('loading all subjects...', flush=True)
allids = tail + good_tr
for i,s in enumerate(allids):
    try: load(s)
    except Exception as e: print('  skip',s,e,flush=True)
    if (i+1)%10==0: print(f'  {i+1}/{len(allids)}',flush=True)
avail=[s for s in allids if s in CACHE]
tail=[s for s in tail if s in CACHE]; good_tr=[s for s in good_tr if s in CACHE]
print(f'loaded: tail={len(tail)} good={len(good_tr)}',flush=True)

def train(tr_ids, epochs=1500, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    Xs=[];Ys=[]
    for s in tr_ids:
        A,Y=CACHE[s]
        if Y.sum()<10: continue
        Xs.append(torch.tensor(A,device=dev)); Ys.append(torch.tensor(Y,device=dev))
    if not Xs: return None
    net=nn.Sequential(nn.Linear(Xs[0].shape[1],64),nn.ReLU(),
                      nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1)).to(dev)
    opt=torch.optim.Adam(net.parameters(),1e-3)
    for _ in range(epochs):
        k=np.random.randint(len(Xs)); j=torch.randint(0,len(Xs[k]),(4096,),device=dev)
        l=nn.functional.binary_cross_entropy_with_logits(net(Xs[k][j]).squeeze(1),Ys[k][j])
        opt.zero_grad(); l.backward(); opt.step()
    return net

def ev_net(net,s):
    A,Y=CACHE[s]
    if net is None or Y.sum()<10: return float('nan')
    with torch.no_grad():
        p=(torch.sigmoid(net(torch.tensor(A,device=dev)).squeeze(1)).cpu().numpy()>=0.5).astype(np.float32)
    return dice(p,Y)

print('\ntraining global_good model (shared across all tail subjects)...',flush=True)
net_good = train(good_tr)

res={'sid':[],'model':[],'in_subject':[],'loo_tail':[],'global_good':[],'loo_tail_good':[]}
print()
for i,s in enumerate(tail):
    others=[x for x in tail if x!=s]
    r_in  = ev_net(train([s]), s)
    r_loo = ev_net(train(others), s)
    r_gg  = ev_net(net_good, s)
    r_both= ev_net(train(others+good_tr), s)
    res['sid'].append(s); res['model'].append(ev[s]['dice_ET'])
    res['in_subject'].append(r_in); res['loo_tail'].append(r_loo)
    res['global_good'].append(r_gg); res['loo_tail_good'].append(r_both)
    print(f'  [{i+1}/{len(tail)}] {s}  model={ev[s]["dice_ET"]:.3f} | in={r_in:.3f} '
          f'LOO-tail={r_loo:.3f} good={r_gg:.3f} both={r_both:.3f}',flush=True)

f=lambda k: np.nanmean(res[k])
print()
print('='*80)
print('E198 -- DOES THE TAIL ET RULE TRANSFER?')
print('='*80)
print(f'  model (3D CNN)            : {f("model"):.3f}')
print(f'  in_subject   (LEAKY, E197): {f("in_subject"):.3f}   <- upper bound, not achievable')
print(f'  loo_tail     (HONEST)     : {f("loo_tail"):.3f}   <- THE TEST')
print(f'  global_good               : {f("global_good"):.3f}')
print(f'  loo_tail+good             : {f("loo_tail_good"):.3f}')
print()
print(f'  E195 cross-fitted O_i on low-O population was ~0.279')
print()
ev_all={x['subject_id']:x for x in json.load(open(ROOT/'experiments/exp_e12_eggo_m/e130/E130_full_eval_E131_v5control_seed0_per_subject.json'))}
et=np.array([x['dice_ET'] for x in ev_all.values()]);tc=np.array([x['dice_TC'] for x in ev_all.values()]);wt=np.array([x['dice_WT'] for x in ev_all.values()])
base=(et.mean()+tc.mean()+wt.mean())/3
ids=list(ev_all.keys())
for k in ['loo_tail','loo_tail_good','global_good']:
    lvl=f(k); e2=et.copy()
    for i,s in enumerate(ids):
        if s in tail: e2[i]=max(e2[i],lvl)
    nb=(e2.mean()+tc.mean()+wt.mean())/3
    print(f'  IF tail ET lifted to {k} ({lvl:.3f}): 3-region {base:.4f} -> {nb:.4f} = {100*(nb-base):+.3f} pp')
json.dump(res, open(ROOT/'experiments/exp_e12_eggo_m/E198_transfer.json','w'), indent=1)
print('\nsaved E198_transfer.json')
