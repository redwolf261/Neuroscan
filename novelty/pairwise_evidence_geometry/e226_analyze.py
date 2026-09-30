"""E226 analysis -- size-controlled test of the SHARPENED question:

    Does D4 target geometry explain persistent non-learning of ET
    components, AFTER CONTROLLING FOR LESION SIZE?

Joins E226_d4_geometry.csv (this experiment: R_l, N_l_D4, M_l_D4, S_l, Q_l
per lesion, Monte Carlo means over simulated crop-phase draws) against
E223_exposure.csv (REUSED detected/missed labels, same checkpoint/MIN_VOX/
component-labeling convention) on (subject_id, comp_id).

Follows this project's own established multivariate partial-correlation
method (same style as e223_analyze.py's multivariate section): regress
BOTH `missed` and each D4-geometry metric on log(size) [+ any other
controls available], then correlate the RESIDUALS. A real, size-independent
effect must survive this control; a spurious "small lesions have worse D4
geometry" effect will not.

DECISION TREE (preregistered, per this session's agreed framing):
  A. No difference after size control for ANY D4 metric -> KILL, close
     this mechanism. The old DS-family conclusion (D36/E37/E41) extends
     to the current v5 multimodal setup: D4 geometry doesn't explain non-
     learning beyond what size already explains.
  B. At least one D4 metric predicts missedness independently of size ->
     REAL, retained. Next step: gradient-contribution audit (NOT run by
     this script), per the explicit "target geometry != gradient
     mechanism" caution from this session's own discussion.
  C. Threshold/nonlinear effect (e.g. a real relationship only below some
     occupancy cutoff, invisible to a linear partial correlation) ->
     checked via a quintile-binned breakdown, same style as e223_analyze.

Also checks: does this differ from TC/WT? (deferred -- E226 only measured
ET; this is noted as a limitation, not silently glossed over.)
"""
import csv
import numpy as np
from scipy import stats
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---- load and join ----
d4_rows = list(csv.DictReader(open(HERE / 'E226_d4_geometry.csv')))
exp_rows = list(csv.DictReader(open(HERE / 'E223_exposure.csv')))

exp_by_key = {(r['subject_id'], r['comp_id']): r for r in exp_rows}

joined = []
unmatched = 0
for r in d4_rows:
    key = (r['subject_id'], r['comp_id'])
    e = exp_by_key.get(key)
    if e is None:
        unmatched += 1
        continue
    joined.append({
        'subject_id': r['subject_id'], 'comp_id': r['comp_id'],
        'size': float(r['size']),
        'n_incl_draws': int(r['n_incl_draws']),
        'R_l': float(r['R_l_mean']), 'N_l_D4': float(r['N_l_D4_mean']),
        'M_l_D4': float(r['M_l_D4_mean']), 'S_l': float(r['S_l_mean']),
        'Q_l': float(r['Q_l_mean']),
        'detected': int(e['detected']),
    })

print(f'E226 rows: {len(d4_rows)}  E223 rows: {len(exp_rows)}  '
      f'joined: {len(joined)}  unmatched (E226 lesion not in E223): {unmatched}\n')

if len(joined) < 20:
    print('TOO FEW JOINED ROWS for a meaningful statistical test -- stopping. '
          'Check E226/E223 comp_id numbering actually agree (same ndi.label '
          'convention, same subject ordering) before trusting any result below.')
    raise SystemExit(1)

size = np.array([r['size'] for r in joined])
det = np.array([r['detected'] for r in joined])
missed = 1 - det
log_size = np.log(size + 1)

metrics = {'R_l': 'retained mass ratio', 'N_l_D4': 'nonzero D4 cells touched',
          'M_l_D4': 'max D4 occupancy', 'S_l': 'absolute D4 supervision mass',
          'Q_l': 'mean D4 occupancy over lesion volume'}

print('=' * 100)
print('RAW: missed vs detected, per D4 metric (uncontrolled)')
print('=' * 100)
for key, label in metrics.items():
    arr = np.array([r[key] for r in joined])
    d, m = arr[det == 1], arr[det == 0]
    if len(d) < 3 or len(m) < 3:
        print(f'  {label:>35}: too few (det={len(d)}, missed={len(m)})')
        continue
    u, p = stats.mannwhitneyu(d, m)
    print(f'  {label:>35}: detected median={np.median(d):.4f}  missed median={np.median(m):.4f}  '
          f'Mann-Whitney p={p:.3e}')

print('\n' + '=' * 100)
print('SIZE CONFOUND CHECK: does each D4 metric correlate with size?')
print('=' * 100)
for key, label in metrics.items():
    arr = np.array([r[key] for r in joined])
    rho, p = stats.spearmanr(size, arr)
    print(f'  Spearman(size, {key}) = {rho:+.4f}  p={p:.3e}   ({label})')

print('\n' + '=' * 100)
print('THE DECISIVE TEST: partial correlation of missed vs each D4 metric, '
      'controlling for log(size)')
print('=' * 100)
X_size = np.column_stack([np.ones(len(joined)), log_size])
beta_det, *_ = np.linalg.lstsq(X_size, missed.astype(float), rcond=None)
resid_missed = missed.astype(float) - X_size @ beta_det

results = {}
for key, label in metrics.items():
    arr = np.array([r[key] for r in joined])
    beta_m, *_ = np.linalg.lstsq(X_size, arr, rcond=None)
    resid_metric = arr - X_size @ beta_m
    r_partial, p_partial = stats.pearsonr(resid_missed, resid_metric)
    results[key] = (r_partial, p_partial)
    flag = ' <-- SURVIVES size control (p<0.05)' if p_partial < 0.05 else ''
    print(f'  partial corr(missed, {key} | log_size) = {r_partial:+.4f}  '
          f'p={p_partial:.3e}   ({label}){flag}')

print('\n' + '=' * 100)
print('SIZE-MATCHED QUINTILE BREAKDOWN (checks for threshold/nonlinear effects '
      'a linear partial correlation could miss)')
print('=' * 100)
bins = np.percentile(size, [0, 20, 40, 60, 80, 100])
for key, label in metrics.items():
    arr = np.array([r[key] for r in joined])
    print(f'\n  -- {label} ({key}) --')
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        sel = (size >= lo) & (size <= hi)
        d = arr[sel & (det == 1)]; m = arr[sel & (det == 0)]
        if len(d) < 3 or len(m) < 3:
            print(f'    size [{lo:.0f},{hi:.0f}]: too few (det={len(d)}, missed={len(m)})')
            continue
        p = stats.mannwhitneyu(d, m)[1]
        print(f'    size [{lo:.0f},{hi:.0f}] (n={sel.sum()}): '
              f'detected median={np.median(d):.4f} (n={len(d)})  '
              f'missed median={np.median(m):.4f} (n={len(m)})  p={p:.3f}')

print('\n' + '=' * 100)
print('VERDICT')
print('=' * 100)
survivors = [k for k, (r, p) in results.items() if p < 0.05]
if not survivors:
    print('  ==> OUTCOME A: NO D4 metric predicts missedness after controlling for size.')
    print('      KILL. This mechanism is closed on the current v5 multimodal setup,')
    print('      extending E36/E37/E41\'s old-setup conclusion. D4 target geometry')
    print('      does not explain persistent ET non-learning beyond what lesion size')
    print('      already explains.')
else:
    print(f'  ==> OUTCOME B: {len(survivors)} D4 metric(s) predict missedness')
    print(f'      independently of size: {survivors}')
    print('      RETAINED. Per the pre-agreed caution, this is TARGET-GEOMETRY')
    print('      evidence only, NOT yet a gradient-mechanism finding -- next step')
    print('      would be a gradient-contribution audit, not a training modification.')
    print('      NOTE: this experiment only measured ET. Whether the effect is')
    print('      ET-specific (as E225 found the CC-DiceCE calibration effect to be)')
    print('      is UNTESTED here and would need a parallel TC/WT run to check.')
