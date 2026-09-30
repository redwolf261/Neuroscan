"""E213 analysis -- PREREGISTERED, written before results were seen.

MANDATORY FIRST: E153-style inert control. If q_rank/q_robust change
substantially when the reference population is resampled (same
boundary-distance band, z(x) held fixed), the relative score is
denominator-driven -- exactly E153's failure mode -- and everything below
is suspect regardless of how good it looks.

Then: does any relative score (spatial-rank, spatial-robust, subject-rank,
subject-robust, bottleneck s_h) discriminate L from H/S, and does swapping
the decision rule from z>0 to q>tau change the Dice-PROXY (2TP/(2TP+FP+FN))
on the fixed component set, LOSO-thresholded -- not just AUC.

KILL if: inert control shows q is denominator-driven; OR no score
discriminates L beyond z alone; OR Dice-proxy does not improve under LOSO
threshold swap; OR FP count explodes (H false-positive rate rises).
"""
import csv
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'E213_relative.csv')))
for r in rows:
    for k in list(r):
        if k not in ('subject_id', 'population'):
            r[k] = float(r[k])

POPS = ['L', 'S', 'H']
n = {p: sum(1 for r in rows if r['population'] == p) for p in POPS}
print(f'rows={len(rows)}  pops={n}  subjects={len({r["subject_id"] for r in rows})}\n')

print('=' * 92)
print('MANDATORY E153-STYLE INERT CONTROL')
print('=' * 92)
print('If q changes substantially when N is resampled at matched boundary-')
print('distance with z(x) held fixed, the score is denominator-driven.\n')
for qname, altname in [('q_rank_spatial', 'q_rank_inert_alt'),
                       ('q_robust_spatial', 'q_robust_inert_alt')]:
    q = np.array([r[qname] for r in rows])
    qa = np.array([r[altname] for r in rows])
    diff = q - qa
    corr = stats.pearsonr(q, qa)[0]
    print(f'  {qname}:')
    print(f'    corr(original, resampled) = {corr:.4f}  (want close to 1.0)')
    print(f'    median |diff| = {np.median(np.abs(diff)):.4f}  '
          f'mean |diff| = {np.mean(np.abs(diff)):.4f}')
inert_ok = all(
    stats.pearsonr([r[q] for r in rows], [r[a] for r in rows])[0] > 0.7
    for q, a in [('q_rank_spatial', 'q_rank_inert_alt'),
                ('q_robust_spatial', 'q_robust_inert_alt')])
print(f'\n  INERT CONTROL: {"PASSES (score is stable under resampling)" if inert_ok else "FAILS -- denominator-driven, treat everything below with suspicion"}')

print('\n' + '=' * 92)
print('DISCRIMINATION: does each score separate L from H?')
print('=' * 92)
SCORES = ['z', 'q_rank_spatial', 'q_robust_spatial', 'q_rank_subject',
         'q_robust_subject', 's_h']
for sc in SCORES:
    L = np.array([r[sc] for r in rows if r['population'] == 'L'])
    H = np.array([r[sc] for r in rows if r['population'] == 'H'])
    u, p = stats.mannwhitneyu(L, H)
    auc = u / (len(L) * len(H))
    print(f'  {sc:<20} L median={np.median(L):+8.4f}  H median={np.median(H):+8.4f}  '
        f'AUC(L>H)={auc:.4f}  p={p:.3e}')

print('\n' + '=' * 92)
print('COUNTERFACTUAL DICE-PROXY: z>0 vs each relative score, LOSO threshold')
print('=' * 92)
print('Dice-proxy = 2TP/(2TP+FP+FN) on the fixed L/S/H component set.')
print('Positive class = L or S (real lesion); negative = H.\n')

subjects = np.array([r['subject_id'] for r in rows])
y = np.array([1 if r['population'] in ('L', 'S') else 0 for r in rows])
uniq_subs = sorted(set(subjects))


def loso_dice(scores, y, subjects):
    """LOSO: for each subject, pick threshold maximizing Dice-proxy on the
    OTHER subjects, apply to held-out. Returns overall pooled TP/FP/FN."""
    TP = FP = FN = 0
    for held in uniq_subs:
        tr = subjects != held
        te = subjects == held
        if tr.sum() < 5 or te.sum() < 1:
            continue
        cand_thresh = np.unique(scores[tr])
        best_t, best_d = None, -1
        for th in cand_thresh:
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


for sc in SCORES:
    scores = np.array([r[sc] for r in rows])
    dice, tp, fp, fn = loso_dice(scores, y, subjects)
    print(f'  {sc:<20} Dice-proxy={dice:.4f}  TP={tp:3d}  FP={fp:3d}  FN={fn:3d}')

dice_z, tp_z, fp_z, fn_z = loso_dice(np.array([r['z'] for r in rows]), y, subjects)

print('\n' + '=' * 92)
print('VERDICT')
print('=' * 92)
best_alt = None
best_dice = dice_z
for sc in SCORES:
    if sc == 'z':
        continue
    scores = np.array([r[sc] for r in rows])
    dice, tp, fp, fn = loso_dice(scores, y, subjects)
    if dice > best_dice:
        best_dice = dice
        best_alt = (sc, dice, tp, fp, fn)

print(f'  baseline z Dice-proxy = {dice_z:.4f} (TP={tp_z} FP={fp_z} FN={fn_z})')
if best_alt:
    sc, dice, tp, fp, fn = best_alt
    fp_explosion = fp > fp_z * 1.5
    print(f'  best alternative: {sc} Dice-proxy={dice:.4f} (TP={tp} FP={fp} FN={fn})')
    print(f'  FP explosion (>1.5x baseline)? {fp_explosion}')
    improvement = dice - dice_z
    print(f'  Dice-proxy improvement = {improvement:+.4f}')
else:
    improvement = 0.0
    fp_explosion = False
    print('  no alternative score beat raw z on Dice-proxy')

if not inert_ok:
    verdict = 'KILL -- inert control fails, relative scores are denominator-driven (E153 pattern).'
elif improvement <= 0.01:
    verdict = 'KILL -- no relative score meaningfully improves Dice-proxy over raw z.'
elif fp_explosion:
    verdict = 'KILL -- improvement comes at the cost of FP explosion, not a real gain.'
else:
    verdict = (f'PASS -- {best_alt[0]} improves Dice-proxy by {improvement:+.4f} '
              f'without FP explosion, and survives the E153 inert control. '
              f'Worth investigating a trained version of the relative-ordering '
              f'objective.')
print(f'\n  ==> {verdict}')
