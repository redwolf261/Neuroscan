"""E223 analysis -- exposure audit. Detected vs missed, raw and size/position-
controlled, per the preregistered plan (E_l, C_l, size, distance-to-WT-
centroid, isolation as confounds)."""
import csv
import numpy as np
from scipy import stats
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'E223_exposure.csv')))
for r in rows:
    for k in ('size', 'E_l', 'C_l', 'dist_to_wt_centroid', 'wt_volume'):
        r[k] = float(r[k])
    r['isolated_from_main_wt'] = int(r['isolated_from_main_wt'])
    r['detected'] = int(r['detected'])

det = np.array([r['detected'] for r in rows])
size = np.array([r['size'] for r in rows])
El = np.array([r['E_l'] for r in rows])
Cl = np.array([r['C_l'] for r in rows])
dist = np.array([r['dist_to_wt_centroid'] for r in rows])
iso = np.array([r['isolated_from_main_wt'] for r in rows])
wtvol = np.array([r['wt_volume'] for r in rows])
frac_size = size / np.maximum(wtvol, 1)

print(f'total components={len(rows)}  detected={det.sum()} ({100*det.mean():.1f}%)  '
      f'missed={len(rows)-det.sum()}\n')

print('=' * 90)
print('RAW: E_l and C_l, detected vs missed')
print('=' * 90)
for name, arr in [('E_l (inclusion)', El), ('C_l (centering)', Cl)]:
    d, m = arr[det == 1], arr[det == 0]
    u, p = stats.mannwhitneyu(d, m)
    print(f'  {name}')
    print(f'    detected: median={np.median(d):.4f}  mean={d.mean():.4f}  (n={len(d)})')
    print(f'    missed  : median={np.median(m):.4f}  mean={m.mean():.4f}  (n={len(m)})')
    print(f'    Mann-Whitney p={p:.2e}')

print('\n' + '=' * 90)
print('SIZE: does missed correlate with smaller lesions? (confound check)')
print('=' * 90)
d, m = size[det == 1], size[det == 0]
print(f'  detected size: median={np.median(d):.0f}  mean={d.mean():.0f}')
print(f'  missed size  : median={np.median(m):.0f}  mean={m.mean():.0f}')
print(f'  Mann-Whitney p={stats.mannwhitneyu(d, m)[1]:.2e}')
print(f'\n  Spearman(size, C_l) = {stats.spearmanr(size, Cl)[0]:.4f}  '
      f'p={stats.spearmanr(size, Cl)[1]:.2e}')
print(f'  Spearman(size, E_l) = {stats.spearmanr(size, El)[0]:.4f}  '
      f'p={stats.spearmanr(size, El)[1]:.2e}')

print('\n' + '=' * 90)
print('SIZE-MATCHED comparison: C_l for detected vs missed within size bins')
print('=' * 90)
bins = np.percentile(size, [0, 20, 40, 60, 80, 100])
for i in range(len(bins) - 1):
    lo, hi = bins[i], bins[i + 1]
    sel = (size >= lo) & (size <= hi)
    d = Cl[sel & (det == 1)]; m = Cl[sel & (det == 0)]
    if len(d) < 3 or len(m) < 3:
        print(f'  size [{lo:.0f},{hi:.0f}]: too few (det={len(d)}, missed={len(m)})')
        continue
    p = stats.mannwhitneyu(d, m)[1]
    print(f'  size [{lo:.0f},{hi:.0f}] (n={sel.sum()}): '
          f'detected C_l={np.median(d):.4f} (n={len(d)})  '
          f'missed C_l={np.median(m):.4f} (n={len(m)})  p={p:.3f}')

print('\n' + '=' * 90)
print('MULTIVARIATE: does "missed" predict C_l after controlling for size, distance, isolation?')
print('=' * 90)
log_size = np.log(size + 1)
X = np.column_stack([np.ones(len(rows)), log_size, dist, iso])
beta_size, *_ = np.linalg.lstsq(X, Cl, rcond=None)
resid_Cl = Cl - X @ beta_size
X_det = np.column_stack([np.ones(len(rows)), log_size, dist, iso])
beta_det, *_ = np.linalg.lstsq(X_det, det.astype(float), rcond=None)
resid_det = det.astype(float) - X_det @ beta_det
r_partial, p_partial = stats.pearsonr(resid_det, resid_Cl)
print(f'  partial corr(missed, C_l | log_size, dist, isolated) = {r_partial:+.4f}  p={p_partial:.2e}')

r_raw, p_raw = stats.pointbiserialr(det, Cl)
print(f'  raw corr(detected, C_l) = {r_raw:+.4f}  p={p_raw:.2e}')

print('\n' + '=' * 90)
print('ISOLATION: does spatial separation from main WT mass predict low C_l?')
print('=' * 90)
d_iso, d_conn = Cl[iso == 1], Cl[iso == 0]
print(f'  isolated (n={iso.sum()}): median C_l={np.median(d_iso):.4f}')
print(f'  connected (n={(iso==0).sum()}): median C_l={np.median(d_conn):.4f}')
if iso.sum() >= 3:
    print(f'  Mann-Whitney p={stats.mannwhitneyu(d_iso, d_conn)[1]:.3e}')
print(f'  fraction isolated among missed: {iso[det==0].mean():.3f}')
print(f'  fraction isolated among detected: {iso[det==1].mean():.3f}')

print('\n' + '=' * 90)
print('VERDICT')
print('=' * 90)
raw_gap = np.median(Cl[det == 0]) < np.median(Cl[det == 1])
survives_control = p_partial < 0.05 and r_partial < 0
El_ceiling = np.median(El) > 0.95
print(f'  E_l ceiling effect (median > 0.95)? {El_ceiling}  (median E_l = {np.median(El):.4f})')
print(f'  Missed lesions have lower raw C_l? {raw_gap}')
print(f'  Gap survives size/distance/isolation control? {survives_control}')
if El_ceiling and raw_gap and survives_control:
    print('\n  ==> HYPOTHESIS A (exposure bottleneck) SUPPORTED.')
    print('      E_l is saturated (inclusion is never the constraint), but C_l')
    print('      (centering) is significantly lower for missed lesions and the')
    print('      gap survives controlling for size/distance/isolation.')
    print('      -> Proceed to gradient-opportunity measurement, THEN design.')
elif raw_gap and not survives_control:
    print('\n  ==> CONFOUNDED. Missed lesions have lower C_l, but this is fully')
    print('      explained by size/distance/isolation, not "missedness" itself.')
else:
    print('\n  ==> HYPOTHESIS B (exposure adequate). Sampler is not the bottleneck.')
    print('      Kill this direction; investigate post-inclusion mechanisms instead.')
