"""E227 analysis -- tests the three pre-registered outcomes:
  (1) low Q_l -> low |G_l^D4| (magnitude starvation)
  (2) low Q_l -> no gradient difference (kill the D4-gradient hypothesis)
  (3) low Q_l -> gradient present, but wrong-directioned (cosine w/ seg loss)

Joins E227_gradient.csv against nothing external -- Q_l, detected, and the
gradient norms/cosine are all measured together per lesion in E227 itself
(unlike E226, which needed E223's cached labels; E227 computes detected
fresh via the same frozen checkpoint, same >=50% overlap convention).
"""
import csv
import numpy as np
from scipy import stats
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'E227_gradient.csv')))
for r in rows:
    for k in ('size', 'Q_l', 'G_head_norm', 'G_dec3_aux_norm', 'G_dec3_seg_norm',
             'cos_dec3_aux_vs_seg'):
        r[k] = float(r[k])
    r['detected'] = int(r['detected'])

print(f'E227 rows: {len(rows)}\n')
if len(rows) < 20:
    print('TOO FEW ROWS for a meaningful test -- stopping.')
    raise SystemExit(1)

size = np.array([r['size'] for r in rows])
Q_l = np.array([r['Q_l'] for r in rows])
det = np.array([r['detected'] for r in rows])
missed = 1 - det
G_head = np.array([r['G_head_norm'] for r in rows])
G_dec3_aux = np.array([r['G_dec3_aux_norm'] for r in rows])
G_dec3_seg = np.array([r['G_dec3_seg_norm'] for r in rows])
cos = np.array([r['cos_dec3_aux_vs_seg'] for r in rows])
log_size = np.log(size + 1)

print(f'detected={det.sum()} ({100*det.mean():.1f}%)  missed={len(rows)-det.sum()}\n')

print('=' * 100)
print('OUTCOME (1) TEST: does Q_l predict gradient MAGNITUDE, controlling for size?')
print('=' * 100)
X = np.column_stack([np.ones(len(rows)), log_size])
for name, arr in [('G_head_norm', G_head), ('G_dec3_aux_norm', G_dec3_aux)]:
    valid = np.isfinite(arr)
    beta_g, *_ = np.linalg.lstsq(X[valid], arr[valid], rcond=None)
    resid_g = arr[valid] - X[valid] @ beta_g
    beta_q, *_ = np.linalg.lstsq(X[valid], Q_l[valid], rcond=None)
    resid_q = Q_l[valid] - X[valid] @ beta_q
    r, p = stats.pearsonr(resid_q, resid_g)
    print(f'  partial corr(Q_l, {name} | log_size) = {r:+.4f}  p={p:.3e}')

print('\n' + '=' * 100)
print('OUTCOME (1) TEST, DIRECT: does gradient magnitude predict missedness, controlling for size?')
print('=' * 100)
beta_m, *_ = np.linalg.lstsq(X, missed.astype(float), rcond=None)
resid_m = missed.astype(float) - X @ beta_m
for name, arr in [('G_head_norm', G_head), ('G_dec3_aux_norm', G_dec3_aux)]:
    valid = np.isfinite(arr)
    beta_g, *_ = np.linalg.lstsq(X[valid], arr[valid], rcond=None)
    resid_g = arr[valid] - X[valid] @ beta_g
    r, p = stats.pearsonr(resid_m[valid], resid_g)
    flag = ' <-- SURVIVES (p<0.05)' if p < 0.05 else ''
    print(f'  partial corr(missed, {name} | log_size) = {r:+.4f}  p={p:.3e}{flag}')

print('\n' + '=' * 100)
print('OUTCOME (3) TEST: does Q_l predict gradient DIRECTION (cosine vs seg loss), '
      'controlling for size?')
print('=' * 100)
valid = np.isfinite(cos)
beta_c, *_ = np.linalg.lstsq(X[valid], cos[valid], rcond=None)
resid_c = cos[valid] - X[valid] @ beta_c
beta_q2, *_ = np.linalg.lstsq(X[valid], Q_l[valid], rcond=None)
resid_q2 = Q_l[valid] - X[valid] @ beta_q2
r, p = stats.pearsonr(resid_q2, resid_c)
print(f'  partial corr(Q_l, cos_dec3_aux_vs_seg | log_size) = {r:+.4f}  p={p:.3e}  (n={valid.sum()})')
print(f'  raw cosine: mean={cos[valid].mean():.4f}  median={np.median(cos[valid]):.4f}  '
      f'min={cos[valid].min():.4f}  max={cos[valid].max():.4f}')
print(f'  fraction with NEGATIVE cosine (actively opposed): {(cos[valid]<0).mean():.3f}')

beta_m2, *_ = np.linalg.lstsq(X[valid], missed[valid].astype(float), rcond=None)
resid_m2 = missed[valid].astype(float) - X[valid] @ beta_m2
r2, p2 = stats.pearsonr(resid_m2, resid_c)
print(f'  partial corr(missed, cos_dec3_aux_vs_seg | log_size) = {r2:+.4f}  p={p2:.3e}')

print('\n' + '=' * 100)
print('RAW group comparison: detected vs missed, all metrics')
print('=' * 100)
for name, arr in [('Q_l', Q_l), ('G_head_norm', G_head), ('G_dec3_aux_norm', G_dec3_aux),
                  ('G_dec3_seg_norm', G_dec3_seg), ('cos_dec3_aux_vs_seg', cos)]:
    valid = np.isfinite(arr)
    d, m = arr[valid & (det==1)], arr[valid & (det==0)]
    if len(d) < 3 or len(m) < 3:
        print(f'  {name:>25}: too few (det={len(d)}, missed={len(m)})')
        continue
    u, p = stats.mannwhitneyu(d, m)
    print(f'  {name:>25}: detected median={np.median(d):.5f}  missed median={np.median(m):.5f}  p={p:.3e}')

print('\n' + '=' * 100)
print('VERDICT')
print('=' * 100)
print('Read the three sections above against the pre-registered outcomes:')
print('  (1) magnitude starvation: check "OUTCOME (1) TEST, DIRECT" -- does gradient')
print('      magnitude predict missedness after size control?')
print('  (2) no relationship: if neither (1) nor (3) show a significant, sensibly-')
print('      signed effect, kill the D4-gradient hypothesis.')
print('  (3) direction, not magnitude: check "OUTCOME (3) TEST" -- does Q_l or')
print('      missedness predict gradient MISALIGNMENT with the main seg loss?')
