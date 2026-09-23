"""E189 analysis -- PREREGISTERED, written before results were seen.

PRIMARY: paired Delta_T vs Delta_Z on the 58 L components.
  Delta_T = z_T - z_I     (translation)
  Delta_Z = z_Z - z_I     (zero the same ROI)

H1 RELIEF   : Delta_Z >= Delta_T  -> translation merely removes suppressive
              skip content. KILL the registration-invariant operator branch.
H2 RECOVERY : Delta_T >> Delta_Z  -> translation supplies something zeroing
              cannot. PASS.
STRONGEST H2: Delta_T > 0 AND Delta_Z < 0.

Reported per spec: medians, means, counts of positive effects, T>Z vs Z>T,
and CROSSINGS (component mean logit z<0 -> z>0) under both T and Z, because
E186.1 established that ~1 logit of significant gain can yield ~no recovery.
"""
import csv
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'E189_relief.csv')))
for r in rows:
    for k in list(r):
        if k not in ('subject_id', 'population'):
            r[k] = float(r[k])
POPS = ['L', 'S', 'H']
cnt = {p: sum(1 for r in rows if r['population'] == p) for p in POPS}
print(f'rows={len(rows)}  pops={cnt}  '
      f'subjects={len({r["subject_id"] for r in rows})}\n')

for STAT, label in [('z_mean', 'MEAN component logit'),
                    ('z_med', 'MEDIAN component logit'),
                    ('dz', 'dz = median(comp) - median(shell)')]:
    print('=' * 84)
    print(f'{label}')
    print('=' * 84)
    print(f"{'pop':<5}{'n':>5}{'I':>10}{'T':>10}{'Z':>10}"
          f"{'dT':>10}{'dZ':>10}{'dT-dZ':>10}{'p(T vs Z)':>12}")
    for p in POPS:
        sel = [r for r in rows if r['population'] == p]
        if not sel:
            continue
        I = np.array([r[f'{STAT}_I'] for r in sel])
        T = np.array([r[f'{STAT}_T'] for r in sel])
        Z = np.array([r[f'{STAT}_Z'] for r in sel])
        dT, dZ = T - I, Z - I
        pv = stats.wilcoxon(dT, dZ)[1] if len(dT) >= 6 and np.any(dT != dZ) else np.nan
        print(f'{p:<5}{len(sel):>5}{np.median(I):>10.3f}{np.median(T):>10.3f}'
              f'{np.median(Z):>10.3f}{np.median(dT):>10.3f}{np.median(dZ):>10.3f}'
              f'{np.median(dT-dZ):>10.3f}{pv:>12.2e}')
    print()

# ---------------- the decisive L analysis ----------------
L = [r for r in rows if r['population'] == 'L']
I = np.array([r['z_mean_I'] for r in L])
T = np.array([r['z_mean_T'] for r in L])
Z = np.array([r['z_mean_Z'] for r in L])
dT, dZ = T - I, Z - I
print('=' * 84)
print(f'DECISIVE: L components (n={len(L)}), MEAN component logit')
print('=' * 84)
print(f'  median dT = {np.median(dT):+8.4f}    mean dT = {dT.mean():+8.4f}')
print(f'  median dZ = {np.median(dZ):+8.4f}    mean dZ = {dZ.mean():+8.4f}')
print(f'  n(dT>0)   = {int((dT>0).sum()):3d}/{len(dT)}'
      f'      n(dZ>0) = {int((dZ>0).sum()):3d}/{len(dZ)}')
print(f'  n(T>Z)    = {int((dT>dZ).sum()):3d}/{len(dT)}'
      f'      n(Z>T)  = {int((dZ>dT).sum()):3d}/{len(dZ)}')
if len(dT) >= 6 and np.any(dT != dZ):
    print(f'  Wilcoxon T vs Z: p = {stats.wilcoxon(dT, dZ)[1]:.3e}')

print('\n  CROSSINGS (component mean logit z<0 -> z>0):')
for tag, V in [('intact->T', T), ('intact->Z', Z)]:
    cross = int(((I < 0) & (V > 0)).sum())
    print(f'    {tag:<12} {cross:3d}/{int((I<0).sum())} negative components crossed')
print(f'    (reference: E186.1 pool-arm P2 gave 1/58)')

print('\n  RATIO R = dT/dZ  (unstable when dZ ~ 0; paired diffs are primary):')
ok = np.abs(dZ) > 0.05
if ok.sum() >= 4:
    R = dT[ok] / dZ[ok]
    print(f'    n={int(ok.sum())} with |dZ|>0.05, median R = {np.median(R):+.3f}')
else:
    print(f'    only {int(ok.sum())} components have |dZ|>0.05 -- ratio not reported')

# ---------------- verdict ----------------
print('\n' + '=' * 84)
print('PREREGISTERED VERDICT')
print('=' * 84)
mT, mZ = np.median(dT), np.median(dZ)
pv = stats.wilcoxon(dT, dZ)[1] if len(dT) >= 6 and np.any(dT != dZ) else 1.0
h1 = (mZ >= mT) or (pv > 0.05)
strong_h2 = (mT > 0) and (mZ < 0) and (pv < 0.05)
print(f'  median dT = {mT:+.4f}   median dZ = {mZ:+.4f}   p={pv:.3e}')
if strong_h2:
    v = 'H2 RECOVERY (STRONGEST FORM: T helps, Z hurts) -- PASS'
elif h1:
    v = 'H1 RELIEF -- zeroing explains it. KILL operator branch.'
else:
    v = 'H2 RECOVERY -- T significantly exceeds Z. PASS.'
print(f'\n  ==> {v}')
