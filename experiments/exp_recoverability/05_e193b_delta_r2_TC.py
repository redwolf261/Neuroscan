"""E193b -- TC delta-R^2 across all 6 models. Frozen transplant of e144 + e192."""
import json, glob, os
import numpy as np
from scipy import stats, ndimage
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import nibabel as nib
from pathlib import Path

ROOT = Path('C:/Users/Rivan/Projects/Neuroscan')
D = ROOT/'Dataset/Training/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
OUT = ROOT/'experiments/exp_e12_eggo_m'
O = json.load(open(OUT/'E193_recoverability_TC.json'))
ids = sorted(O['mlp_deep'])

evals = {}
for f in sorted(glob.glob(str(OUT/'e130/E130_full_eval_*_per_subject.json'))):
    name = os.path.basename(f).replace('E130_full_eval_','').replace('_per_subject.json','')
    evals[name] = {x['subject_id']: x for x in json.load(open(f))}
ids = [s for s in ids if all(s in e for e in evals.values())]
print('n =', len(ids), ' models =', len(evals))

# Z: SAME 15 vars, region-indexed entries substituted ET->TC by definition only
ZP = OUT/'E193_Z_TC_cache.json'
if ZP.exists():
    Zd = json.load(open(ZP)); print('Z cached')
else:
    Zd = {}
    for i,s in enumerate(ids):
        seg = nib.load(str(D/s/(s+'-seg.nii.gz'))).get_fdata().astype(np.uint8)
        tgt = (seg==1)|(seg==3); tum = seg>0
        com = np.array(ndimage.center_of_mass(tum)); ctr = np.array(seg.shape)/2
        lab,n = ndimage.label(tgt)
        sz = np.bincount(lab.ravel())[1:] if n else np.array([0])
        vols=[]
        for m in ['t1c','t1n','t2f','t2w']:
            v = nib.load(str(D/s/(s+'-'+m+'.nii.gz'))).get_fdata().astype(np.float32)
            br = v>0; v=(v-v[br].mean())/max(1e-6,v[br].std()); vols.append(v[tum])
        Zd[s] = [float(np.log1p(tgt.sum())), float(np.log1p(tum.sum())),
                 float(np.linalg.norm(com-ctr)), float(com[2]), float(n),
                 float(np.log1p(sz.max())), float(tgt.sum())/max(1.0,tum.sum())] \
                + [float(v.mean()) for v in vols] + [float(v.std()) for v in vols]
        if (i+1)%20==0: print(f'  Z {i+1}/{len(ids)}', flush=True)
    json.dump(Zd, open(ZP,'w'))

Z = np.array([Zd[s] for s in ids]); Z=(Z-Z.mean(0))/(Z.std(0)+1e-9)
Ov = np.array([O['mlp_deep'][s] for s in ids]).reshape(-1,1)

def cv_r2(X,y,seed=0):
    kf=KFold(5,shuffle=True,random_state=seed); pr=np.zeros_like(y)
    for tr,te in kf.split(X):
        m=RidgeCV(alphas=np.logspace(-3,3,25)).fit(X[tr],y[tr]); pr[te]=m.predict(X[te])
    return 1-((y-pr)**2).sum()/((y-y.mean())**2).sum()

print()
print('='*88)
print('E193b -- TC CROSS-MODEL delta-R^2  (compare against ET: dR2 +0.451..+0.563)')
print('='*88)
print(f'{"model":32s} {"meanTC":>8s} {"R2(Z)":>8s} {"R2(Z+O)":>9s} {"dR2":>8s} {"perm p":>8s} {"rho(O,Dice)":>12s}')
rows=[]
for name,ev in sorted(evals.items()):
    E=np.array([1.0-ev[s]['dice_TC'] for s in ids])
    r2Z=cv_r2(Z,E); r2ZO=cv_r2(np.hstack([Z,Ov]),E); d=r2ZO-r2Z
    rg=np.random.default_rng(0)
    null=np.array([cv_r2(np.hstack([Z,rg.permutation(Ov)]),E)-r2Z for _ in range(200)])
    p=float((null>=d).mean())
    rho=stats.spearmanr(Ov.ravel(),1.0-E).statistic
    mtc=float(np.mean([ev[s]['dice_TC'] for s in ids]))
    print(f'{name:32s} {mtc:8.4f} {r2Z:8.4f} {r2ZO:9.4f} {d:+8.4f} {p:8.4f} {rho:12.3f}')
    rows.append(dict(model=name,mean_tc=mtc,r2_Z=r2Z,r2_ZO=r2ZO,dR2=d,perm_p=p,rho=float(rho)))

# Regime III for TC
print()
lo=Ov.ravel()<np.quantile(Ov.ravel(),0.25)
ev=evals['E131_v5control_seed0']; E=np.array([1.0-ev[s]['dice_TC'] for s in ids])
print('REGIME CHECK (E131_v5control, TC):')
print(f'  low-O quartile : O={Ov.ravel()[lo].mean():.3f}  model TC={1-E[lo].mean():.3f}')
print(f'  rest           : O={Ov.ravel()[~lo].mean():.3f}  model TC={1-E[~lo].mean():.3f}')
r3=(Ov.ravel()>=0.75)&((1-E)<0.5)
print(f'  REGIME III (O>=0.75 but model TC<0.50): n = {int(r3.sum())} of {len(ids)}')
for s,o,d_ in zip(np.array(ids)[r3],Ov.ravel()[r3],(1-E)[r3]):
    print(f'     {s}  O={o:.3f}  TC={d_:.3f}')

# ET vs TC O_i relationship (kept separate as inputs; compared only post hoc)
try:
    OET=json.load(open(OUT/'E143_recoverability.json'))['mlp_deep']
    common=[s for s in ids if s in OET]
    if len(common)>10:
        r=stats.spearmanr([OET[s] for s in common],[O['mlp_deep'][s] for s in common]).statistic
        print(f'\nPOST-HOC ONLY: rho(O_ET, O_TC) over {len(common)} shared subjects = {r:.3f}')
except Exception as e:
    print('ET/TC compare skipped:', e)

json.dump(dict(n=len(ids),subjects=ids,rows=rows), open(OUT/'E193_crossmodel_TC.json','w'), indent=1)
print('\nsaved E193_crossmodel_TC.json')
