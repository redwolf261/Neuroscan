"""E144 -- does cross-fitted O_i explain RESIDUAL ET error beyond a pre-specified
difficulty model Z?  Out-of-sample Delta R^2.   [PROMOTED FROM SCRATCHPAD 2026-09-20]

Reconstructed verbatim from the original scratchpad `e144.py` (session a8041820),
which was deleted from the temp directory after E198 ran. Its stored run log
reported: Z alone 0.1243, Z+O_i 0.6874, dR2 +0.5631, perm p=0.0000, Regime III
n=0/90. Those numbers were independently reproduced on 2026-09-20 by E192's
re-implementation (dR2 +0.5631 on the same model), confirming this code is the
faithful generator.

Run from the repo root:  python experiments/exp_recoverability/02_e144_delta_r2_ET.py
Requires: experiments/exp_e12_eggo_m/E143_recoverability.json (script 01)
"""
import sys,json,numpy as np; sys.path.insert(0,'.')
import nibabel as nib
from pathlib import Path
from scipy import stats, ndimage
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
D=Path('Dataset/Training/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData')
b={x['subject_id']:x for x in json.load(open('experiments/exp_e12_eggo_m/e130/E130_full_eval_E131_v5control_seed0_per_subject.json'))}
O=json.load(open('experiments/exp_e12_eggo_m/E143_recoverability.json'))
ids=sorted(O['mlp_deep'])
print('n =',len(ids))
# ---- pre-specified difficulty variables Z (NO recoverability, NO contrast) ----
Z=[];E=[];Ov=[]
for s in ids:
    seg=nib.load(str(D/s/(s+'-seg.nii.gz'))).get_fdata().astype(np.uint8)
    et=(seg==3); tum=seg>0
    com=np.array(ndimage.center_of_mass(tum)); ctr=np.array(seg.shape)/2
    lab,n=ndimage.label(et)
    sz=np.bincount(lab.ravel())[1:] if n else np.array([0])
    vols=[]
    for m in ['t1c','t1n','t2f','t2w']:
        v=nib.load(str(D/s/(s+'-'+m+'.nii.gz'))).get_fdata().astype(np.float32)
        br=v>0; v=(v-v[br].mean())/max(1e-6,v[br].std()); vols.append(v[tum])
    z=[np.log1p(et.sum()),                      # size
       np.log1p(tum.sum()),                     # tumour size
       float(np.linalg.norm(com-ctr)),          # location: dist from centre
       com[2],                                  # location: superior-inferior
       float(n),                                # fragmentation
       np.log1p(sz.max()),                      # largest component
       float(et.sum())/max(1.0,tum.sum()),      # ET fraction of tumour
       ]+[float(v.mean()) for v in vols]+[float(v.std()) for v in vols]  # intensity descriptors
    Z.append(z); E.append(1.0-b[s]['dice_ET']); Ov.append(O['mlp_deep'][s])
Z=np.array(Z); E=np.array(E); Ov=np.array(Ov).reshape(-1,1)
Z=(Z-Z.mean(0))/(Z.std(0)+1e-9)
print('Z has {} pre-specified difficulty variables (size, location, fragmentation, intensity)'.format(Z.shape[1]))
print()
def cv_r2(X,y,seed=0):
    kf=KFold(5,shuffle=True,random_state=seed); pr=np.zeros_like(y)
    for tr,te in kf.split(X):
        m=RidgeCV(alphas=np.logspace(-3,3,25)).fit(X[tr],y[tr]); pr[te]=m.predict(X[te])
    return 1-((y-pr)**2).sum()/((y-y.mean())**2).sum(), pr
r2Z,prZ=cv_r2(Z,E); r2ZO,_=cv_r2(np.hstack([Z,Ov]),E)
print('OUT-OF-SAMPLE R^2 predicting ET ERROR (1 - Dice), 5-fold CV:')
print('  Z alone (difficulty model) : {:.4f}'.format(r2Z))
print('  Z + O_i (recoverability)   : {:.4f}'.format(r2ZO))
print('  DELTA R^2                  : {:+.4f}'.format(r2ZO-r2Z))
print()
R=E-prZ   # residual unexplained error
rho,p=stats.spearmanr(Ov.ravel(),R)
print('RESIDUAL TEST  rho(O_i, residual error) = {:+.3f}  p={:.2e}'.format(rho,p))
print()
# permutation null on delta R2
rs=[]
rg=np.random.default_rng(0)
for _ in range(200):
    rs.append(cv_r2(np.hstack([Z,rg.permutation(Ov)]),E)[0]-r2Z)
rs=np.array(rs)
print('PERMUTATION NULL (200 shuffles of O_i):')
print('  observed dR2 {:+.4f}   null mean {:+.4f}  sd {:.4f}   p = {:.4f}'.format(
      r2ZO-r2Z,rs.mean(),rs.std(),float((rs>=(r2ZO-r2Z)).mean())))
print()
lo=Ov.ravel()<np.quantile(Ov.ravel(),0.25)
print('REGIME CHECK on clean O_i:')
print('  low-O quartile  : O={:.3f}  model ET={:.3f}'.format(Ov.ravel()[lo].mean(),1-E[lo].mean()))
print('  rest            : O={:.3f}  model ET={:.3f}'.format(Ov.ravel()[~lo].mean(),1-E[~lo].mean()))
hiO_lowD=(Ov.ravel()>=0.75)&((1-E)<0.5)
print('  REGIME III (O>=0.75 but model ET<0.50): n = {} of {}'.format(int(hiO_lowD.sum()),len(ids)))
for s,o,d in zip(np.array(ids)[hiO_lowD],Ov.ravel()[hiO_lowD],(1-E)[hiO_lowD]):
    print('     {}  O={:.3f}  ET={:.3f}'.format(s,o,d))
