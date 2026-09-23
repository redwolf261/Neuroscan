"""E192 -- does O_i predict OTHER models' ET error, or only ours?

DECISIVE TEST for the ceiling interpretation.
If O_i only predicts E131_v5control (the model E144 used), it is a
model-specific difficulty correlate. If it predicts all models about
equally well, the ceiling interpretation survives.

NO new training. Reuses frozen O_i from E143 and 6 existing full evals.
"""
import json, glob, os, sys
import numpy as np
from scipy import stats, ndimage
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import nibabel as nib
from pathlib import Path

ROOT = Path('C:/Users/Rivan/Projects/Neuroscan')
D = ROOT/'Dataset/Training/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
O = json.load(open(ROOT/'experiments/exp_e12_eggo_m/E143_recoverability.json'))
ids = sorted(O['mlp_deep'])
print('O_i subjects (frozen from E143):', len(ids))

evals = {}
for f in sorted(glob.glob(str(ROOT/'experiments/exp_e12_eggo_m/e130/E130_full_eval_*_per_subject.json'))):
    name = os.path.basename(f).replace('E130_full_eval_','').replace('_per_subject.json','')
    evals[name] = {x['subject_id']: x for x in json.load(open(f))}
print('models found:', len(evals))

# all O_i subjects must be present in every eval
ids = [s for s in ids if all(s in e for e in evals.values())]
print('subjects present in ALL models:', len(ids))

# ---- rebuild Z exactly as E144 did (15 pre-specified difficulty vars) ----
ZP = ROOT/'experiments/exp_e12_eggo_m/E192_Z_cache.json'
if ZP.exists():
    Zd = json.load(open(ZP)); print('Z loaded from cache')
else:
    Zd = {}
    for i, s in enumerate(ids):
        seg = nib.load(str(D/s/(s+'-seg.nii.gz'))).get_fdata().astype(np.uint8)
        et = (seg == 3); tum = seg > 0
        com = np.array(ndimage.center_of_mass(tum)); ctr = np.array(seg.shape)/2
        lab, n = ndimage.label(et)
        sz = np.bincount(lab.ravel())[1:] if n else np.array([0])
        vols = []
        for m in ['t1c','t1n','t2f','t2w']:
            v = nib.load(str(D/s/(s+'-'+m+'.nii.gz'))).get_fdata().astype(np.float32)
            br = v > 0; v = (v-v[br].mean())/max(1e-6, v[br].std()); vols.append(v[tum])
        Zd[s] = [float(np.log1p(et.sum())), float(np.log1p(tum.sum())),
                 float(np.linalg.norm(com-ctr)), float(com[2]), float(n),
                 float(np.log1p(sz.max())), float(et.sum())/max(1.0, tum.sum())] \
                + [float(v.mean()) for v in vols] + [float(v.std()) for v in vols]
        if (i+1) % 20 == 0: print(f'  Z {i+1}/{len(ids)}', flush=True)
    json.dump(Zd, open(ZP,'w'))
    print('Z cached')

Z = np.array([Zd[s] for s in ids])
Z = (Z - Z.mean(0))/(Z.std(0)+1e-9)
Ov = np.array([O['mlp_deep'][s] for s in ids]).reshape(-1,1)

def cv_r2(X, y, seed=0):
    kf = KFold(5, shuffle=True, random_state=seed); pr = np.zeros_like(y)
    for tr, te in kf.split(X):
        m = RidgeCV(alphas=np.logspace(-3,3,25)).fit(X[tr], y[tr]); pr[te] = m.predict(X[te])
    return 1-((y-pr)**2).sum()/((y-y.mean())**2).sum()

print()
print('='*86)
print('E192 -- CROSS-MODEL TEST: does frozen O_i explain EVERY model\'s ET error?')
print('='*86)
print(f'{"model":32s} {"meanET":>8s} {"R2(Z)":>8s} {"R2(Z+O)":>9s} {"dR2":>8s} {"perm p":>8s} {"rho(O,Dice)":>12s}')
rows = []
for name, ev in sorted(evals.items()):
    E = np.array([1.0-ev[s]['dice_ET'] for s in ids])
    r2Z = cv_r2(Z, E); r2ZO = cv_r2(np.hstack([Z, Ov]), E); d = r2ZO-r2Z
    rg = np.random.default_rng(0)
    null = np.array([cv_r2(np.hstack([Z, rg.permutation(Ov)]), E)-r2Z for _ in range(200)])
    p = float((null >= d).mean())
    rho = stats.spearmanr(Ov.ravel(), 1.0-E).statistic
    mean_et = float(np.mean([ev[s]['dice_ET'] for s in ids]))
    print(f'{name:32s} {mean_et:8.4f} {r2Z:8.4f} {r2ZO:9.4f} {d:+8.4f} {p:8.4f} {rho:12.3f}')
    rows.append(dict(model=name, mean_et=mean_et, r2_Z=r2Z, r2_ZO=r2ZO,
                     dR2=d, perm_p=p, rho_O_dice=float(rho)))

print()
print('INTER-MODEL AGREEMENT (Spearman of per-subject ET Dice between models):')
names = sorted(evals)
for i, a in enumerate(names):
    for bb in names[i+1:]:
        va = [evals[a][s]['dice_ET'] for s in ids]; vb = [evals[bb][s]['dice_ET'] for s in ids]
        print(f'   {a[:26]:28s} x {bb[:26]:28s} rho={stats.spearmanr(va,vb).statistic:.3f}')

json.dump(dict(n=len(ids), subjects=ids, rows=rows),
          open(ROOT/'experiments/exp_e12_eggo_m/E192_crossmodel.json','w'), indent=1)
print()
print('saved E192_crossmodel.json')
