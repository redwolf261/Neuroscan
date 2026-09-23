"""E197 -- Is the missing ET evidence INFERABLE from the present modalities?

The question (user-specified): for the 15 catastrophic-tail subjects, what
exactly is missing, and can it theoretically be inferred from the modalities
that ARE present?

This is an UPPER-BOUND / feasibility test, not a method. We give an observer
every advantage and ask whether the information is there AT ALL:

  ORACLE-IN-SUBJECT observers, trained AND tested on the SAME subject's labels
  (deliberate leakage -- this is an upper bound on inferability, not a
  generalization estimate):

    A. delta-only   : t1c - t1n           (the channel E142 says is dead)
    B. no-t1c       : t1n, t2f, t2w ONLY  (can the OTHERS carry ET?)
    C. all-4        : t1c,t1n,t2f,t2w
    D. all-4+spatial: + local mean/std of delta

If B is near 0 even with in-subject leakage, the ET/TC structure is NOT
inferable from the remaining acquisition -> new information required.
If B is substantial, the evidence IS present and a method could exist.
"""
import json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from scipy import ndimage
import nibabel as nib

ROOT = Path('C:/Users/Rivan/Projects/Neuroscan')
D = ROOT/'Dataset/Training/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
ev = {x['subject_id']: x for x in json.load(open(
    ROOT/'experiments/exp_e12_eggo_m/e130/E130_full_eval_E131_v5control_seed0_per_subject.json'))}

# the E139 tail definition, recomputed
tail = sorted([s for s,x in ev.items() if x['dice_ET']<0.5 or x['dice_TC']<0.5])
good = sorted([s for s,x in ev.items() if not (x['dice_ET']<0.5 or x['dice_TC']<0.5)])
rng = np.random.default_rng(0)
ctrl = list(rng.permutation(good)[:15])          # matched-size control group
print(f'tail n={len(tail)}   control n={len(ctrl)}')

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def dice(p,y): return float(2*(p*y).sum()/max(1e-6,p.sum()+y.sum()))

def load(sid):
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
    return F,Y,S

def oracle_fit(F,Y,cols,S=None,epochs=1500,seed=0):
    """Train AND test on the SAME subject -- deliberate upper bound."""
    torch.manual_seed(seed)
    A = F[:,cols] if S is None else np.hstack([F[:,cols],S])
    if Y.sum()<10 or Y.sum()==len(Y): return float('nan')
    X=torch.tensor(A,device=dev); Yt=torch.tensor(Y,device=dev)
    net=nn.Sequential(nn.Linear(A.shape[1],64),nn.ReLU(),
                      nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1)).to(dev)
    opt=torch.optim.Adam(net.parameters(),1e-3)
    for _ in range(epochs):
        j=torch.randint(0,len(X),(4096,),device=dev)
        l=nn.functional.binary_cross_entropy_with_logits(net(X[j]).squeeze(1),Yt[j])
        opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        p=(torch.sigmoid(net(X).squeeze(1)).cpu().numpy()>=0.5).astype(np.float32)
    return dice(p,Y)

rows={}
for grp,ids in [('TAIL',tail),('CONTROL',ctrl)]:
    res={'delta_only':[], 'no_t1c':[], 'all4':[], 'all4_spatial':[], 'model':[], 'et_frac':[]}
    print(f'\n--- {grp} ---', flush=True)
    for i,s in enumerate(ids):
        try: F,Y,S = load(s)
        except Exception as e:
            print('  skip',s,e); continue
        dlt = (F[:,0]-F[:,1]).reshape(-1,1)
        # A: delta only (as its own 1-col feature)
        Fd = np.hstack([dlt, dlt])   # duplicate to keep >=2 cols
        res['delta_only'].append(oracle_fit(Fd,Y,[0,1]))
        res['no_t1c'].append(oracle_fit(F,Y,[1,2,3]))       # t1n,t2f,t2w  (NO t1c)
        res['all4'].append(oracle_fit(F,Y,[0,1,2,3]))
        res['all4_spatial'].append(oracle_fit(F,Y,[0,1,2,3],S=S))
        res['model'].append(ev[s]['dice_ET'])
        res['et_frac'].append(float(Y.mean()))
        print(f'  [{i+1}/{len(ids)}] {s}  model={ev[s]["dice_ET"]:.3f}  '
              f'delta={res["delta_only"][-1]:.3f}  noT1c={res["no_t1c"][-1]:.3f}  '
              f'all4={res["all4"][-1]:.3f}  +sp={res["all4_spatial"][-1]:.3f}', flush=True)
    rows[grp]=res

print()
print('='*84)
print('E197 -- IN-SUBJECT ORACLE (deliberate leakage = UPPER BOUND on inferability)')
print('='*84)
print(f'{"group":9s} {"model ET":>9s} {"delta-only":>11s} {"NO t1c":>9s} {"all-4":>9s} {"all4+spatial":>13s}')
for grp in ['TAIL','CONTROL']:
    r=rows[grp]; f=lambda k: np.nanmean(r[k])
    print(f'{grp:9s} {f("model"):9.3f} {f("delta_only"):11.3f} {f("no_t1c"):9.3f} '
          f'{f("all4"):9.3f} {f("all4_spatial"):13.3f}')

print()
print('KEY QUESTION: on the TAIL, does dropping t1c matter?')
t=rows['TAIL']
print(f'  all-4 oracle      : {np.nanmean(t["all4"]):.3f}')
print(f'  NO-t1c oracle     : {np.nanmean(t["no_t1c"]):.3f}')
print(f'  difference        : {np.nanmean(t["all4"])-np.nanmean(t["no_t1c"]):+.3f}')
print(f'  mean ET fraction of tumour voxels (tail): {np.nanmean(t["et_frac"]):.4f}')
json.dump({g:{k:[float(x) for x in v] for k,v in r.items()} for g,r in rows.items()},
          open(ROOT/'experiments/exp_e12_eggo_m/E197_inferability.json','w'), indent=1)
print('\nsaved E197_inferability.json')
