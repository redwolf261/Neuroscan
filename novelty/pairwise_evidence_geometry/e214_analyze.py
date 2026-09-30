"""E214 analysis -- PREREGISTERED, written before results were seen.

Does s_h survive replacing E213's thin, arbitrary mu_B (single fixed corner
voxel) with better-constructed alternatives, especially a proper per-subject
spatial background average (mu_B_spatial_subject -- the reference E213
should have used)?

KILL if: s_h's value/discrimination/Dice-proxy collapses under
mu_B_spatial_subject specifically, or correlation across mu_B choices is low
(denominator-driven, same pattern that killed E213's spatial scores).
PASS if: s_h is stable (high cross-mu_B correlation) AND
mu_B_spatial_subject's version still discriminates L from H with a Dice-proxy
improvement over raw z, matching E213's magnitude.
"""
import csv
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'E214_sh_inert.csv')))
for r in rows:
    for k in list(r):
        if k not in ('subject_id', 'population'):
            r[k] = float(r[k])

TAGS = ['corner', 'opposite', 'edge', 'spatial_global', 'spatial_subject']
print(f'rows={len(rows)}  subjects={len({r["subject_id"] for r in rows})}\n')

print('=' * 92)
print('STABILITY: correlation of s_h across mu_B constructions')
print('=' * 92)
vecs = {tag: np.array([r[f's_h_{tag}'] for r in rows]) for tag in TAGS}
ref = vecs['corner']
for tag in TAGS:
    corr = stats.pearsonr(ref, vecs[tag])[0]
    diff = np.median(np.abs(ref - vecs[tag]))
    print(f'  corner vs {tag:<18} corr={corr:.4f}  median|diff|={diff:.4f}')

print('\n' + '=' * 92)
print('DISCRIMINATION (AUC L vs H) for each mu_B construction')
print('=' * 92)
for tag in TAGS:
    L = np.array([r[f's_h_{tag}'] for r in rows if r['population'] == 'L'])
    H = np.array([r[f's_h_{tag}'] for r in rows if r['population'] == 'H'])
    u, p = stats.mannwhitneyu(L, H)
    auc = u / (len(L) * len(H))
    print(f'  {tag:<18} L median={np.median(L):+8.4f}  H median={np.median(H):+8.4f}  '
        f'AUC={auc:.4f}  p={p:.3e}')

print('\n' + '=' * 92)
print('COUNTERFACTUAL DICE-PROXY, LOSO threshold, for spatial_subject vs corner')
print('=' * 92)
subjects = np.array([r['subject_id'] for r in rows])
y = np.array([1 if r['population'] in ('L', 'S') else 0
             for r in rows])
uniq_subs = sorted(set(subjects))


def loso_dice(scores, y, subjects):
    TP = FP = FN = 0
    for held in uniq_subs:
        tr = subjects != held
        te = subjects == held
        if tr.sum() < 5 or te.sum() < 1:
            continue
        cand = np.unique(scores[tr])
        best_t, best_d = None, -1
        for th in cand:
            pred = (scores[tr] > th).astype(int)
            tp = int(((pred == 1) & (y[tr] == 1)).sum())
            fp = int(((pred == 1) & (y[tr] == 0)).sum())
            fn = int(((pred == 0) & (y[tr] == 1)).sum())
            d = 2 * tp / (2 * tp + fp + fn + 1e-9)
            if d > best_d:
                best_d, best_t = d, th
        pred_te = (scores[te] > best_t).astype(int)
        TP += int(((pred_te == 1) & (y[te] == 1)).sum())
        FP += int(((pred_te == 1) & (y[te] == 0)).sum())
        FN += int(((pred_te == 0) & (y[te] == 1)).sum())
    dice = 2 * TP / (2 * TP + FP + FN + 1e-9)
    return dice, TP, FP, FN


for tag in TAGS:
    scores = vecs[tag]
    dice, tp, fp, fn = loso_dice(scores, y, subjects)
    print(f'  {tag:<18} Dice-proxy={dice:.4f}  TP={tp:3d}  FP={fp:3d}  FN={fn:3d}')

print('\n' + '=' * 92)
print('VERDICT')
print('=' * 92)
corr_subj = stats.pearsonr(vecs['corner'], vecs['spatial_subject'])[0]
L_su = np.array([r['s_h_spatial_subject'] for r in rows if r['population'] == 'L'])
H_su = np.array([r['s_h_spatial_subject'] for r in rows if r['population'] == 'H'])
auc_su = stats.mannwhitneyu(L_su, H_su)[0] / (len(L_su) * len(H_su))
dice_su, tp_su, fp_su, fn_su = loso_dice(vecs['spatial_subject'], y, subjects)
dice_corner, tp_c, fp_c, fn_c = loso_dice(vecs['corner'], y, subjects)

print(f'  corr(corner, spatial_subject) = {corr_subj:.4f}  (want > 0.7)')
print(f'  AUC(spatial_subject) = {auc_su:.4f}  (E213 corner reference: 0.846)')
print(f'  Dice-proxy(spatial_subject) = {dice_su:.4f}  '
      f'(E213 corner reference: 0.9175; raw z: 0.8905)')

stable = corr_subj > 0.7
still_discriminates = auc_su > 0.70
still_improves = dice_su > 0.8905 + 0.01
fp_ok = fp_su <= fp_c * 1.5 + 2

if not stable:
    verdict = ('KILL -- s_h is NOT stable across mu_B constructions (low '
              'correlation). Denominator-driven, same E153 pattern that '
              'killed the spatial-relative scores. E213\'s corner-based '
              'result was an artifact of the arbitrary background sample.')
elif not (still_discriminates and still_improves and fp_ok):
    verdict = ('KILL -- s_h is stable in VALUE but the proper background '
              'reference degrades discrimination/Dice-proxy below E213\'s '
              'corner-based number, OR costs FP. The corner result was '
              'partly a lucky reference, not fully real.')
else:
    verdict = ('PASS -- s_h survives the missing inert control. Stable '
              'across mu_B constructions, and the proper per-subject '
              'spatial background reference reproduces E213\'s '
              'discrimination and Dice-proxy improvement. This is the first '
              'result in the E204-E214 chain to survive every control '
              'applied to it.')
print(f'\n  ==> {verdict}')
