"""E211 analysis -- PREREGISTERED, written before results were seen.

ALL METRICS ON HELD-OUT (split='test') ROWS ONLY. Train rows exist purely to
have produced the proposal network; they are excluded from every reported
statistic here to avoid optimistic bias.

TEST 1 -- does the LEARNED PROPOSAL beat RANDOM on held-out folds?
  crossing rate, dz magnitude, prototype/random/proposal/anti_proposal
  compared at matched alpha, Mann-Whitney + Fisher exact.

TEST 2 -- THREE EVALUATORS (user's spec), same held-out folds:
  R : reactivity only
  C : reactivity + dz
  H : full transition features [z0, z1, dz, reactivity, alpha, a_gate,
      cos_to_prototype]
  Metric: predicting U>0 (AUC) and U itself (R^2), via LOGO (leave-one-fold-
  out, reusing the fold assignments already in the CSV -- no new splits).
  Delta R^2 = R^2_H - R^2_C is the decisive number.

TEST 3 -- does U's dependence on the hypothesis-relative signal survive
  conditioning on reactivity? Partial correlation(U, dz | reactivity),
  computed on proposal-condition test rows only (the condition that matters).

Small-n caveat: 58 components total, ~11-12 per held-out fold. Every
held-out statistic here is computed on a SMALL sample; this is stated
explicitly rather than glossed over.

KILL if: proposal doesn't beat random on held-out crossing/dz; OR
Delta R^2 is small/non-significant; OR U's dz-dependence vanishes after
the reactivity control.
"""
import csv
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'E211_hrs_lite.csv')))
for r in rows:
    for k in list(r):
        if k not in ('subject_id', 'condition', 'split'):
            r[k] = float(r[k])
    r['crossing'] = int(r['crossing'])
    r['fold'] = int(r['fold'])

test_rows = [r for r in rows if r['split'] == 'test']
n_test = len(test_rows)
n_comp_test = len({(r['subject_id'], r['comp_id']) for r in test_rows})
print(f'total rows={len(rows)}  held-out test rows={n_test}  '
      f'held-out components={n_comp_test}\n')
print('SMALL-N CAVEAT: ~11-12 components per held-out fold. Every statistic '
      'below is on a small sample; treat p-values as suggestive at this n, '
      'not as strong confirmatory evidence on their own.\n')

CONDS = ['prototype', 'random', 'proposal', 'anti_proposal']

print('=' * 92)
print('TEST 1: crossing rate & dz by condition (held-out test rows)')
print('=' * 92)
ALPHAS = sorted({r['alpha'] for r in test_rows})
print(f"{'condition':<16}" + ''.join(f'{a:>10}' for a in ALPHAS) + f"{'any-alpha':>12}")
comp_any = {}
for cond in CONDS:
    line = f'{cond:<16}'
    for a in ALPHAS:
        sel = [r for r in test_rows if r['condition'] == cond and r['alpha'] == a]
        rate = sum(r['crossing'] for r in sel) / max(len(sel), 1)
        line += f'{rate*100:>9.1f}%'
    keys = {(r['subject_id'], r['comp_id']) for r in test_rows if r['condition'] == cond}
    any_cross = sum(1 for k in keys if any(
        rr['crossing'] for rr in test_rows
        if rr['condition'] == cond and (rr['subject_id'], rr['comp_id']) == k))
    comp_any[cond] = (any_cross, len(keys))
    line += f'{100*any_cross/max(len(keys),1):>11.1f}%'
    print(line)

print('\nFisher exact, any-alpha crossing, proposal vs random:')
pa, na = comp_any['proposal']; ra, nb = comp_any['random']
table = [[pa, na - pa], [ra, nb - ra]]
_, p_fish = stats.fisher_exact(table)
print(f'  proposal {pa}/{na}  vs  random {ra}/{nb}   p={p_fish:.3f}')

print('\nMean/median dz by condition at alpha=1.0:')
for cond in CONDS:
    sel = [r['dz'] for r in test_rows if r['condition'] == cond and r['alpha'] == 1.0]
    if sel:
        print(f'  {cond:<16} median={np.median(sel):+8.4f}  mean={np.mean(sel):+8.4f}  n={len(sel)}')
sel_p = [r['dz'] for r in test_rows if r['condition'] == 'proposal' and r['alpha'] == 1.0]
sel_r = [r['dz'] for r in test_rows if r['condition'] == 'random' and r['alpha'] == 1.0]
if len(sel_p) >= 3 and len(sel_r) >= 3:
    pu = stats.mannwhitneyu(sel_p, sel_r)[1]
    print(f'  Mann-Whitney proposal vs random dz (alpha=1.0): p={pu:.3e}')

print('\nCosine of proposal direction to E210 prototype direction (diagnostic,')
print('NOT trained on -- shows whether the network reinvented the prototype):')
cos_vals = [r['cos_to_prototype'] for r in test_rows if r['condition'] == 'proposal']
print(f'  median cos = {np.median(cos_vals):+.4f}  mean = {np.mean(cos_vals):+.4f}')

print('\n' + '=' * 92)
print('TEST 2: R / C / H evaluators, leave-one-fold-out (reusing CSV fold ids)')
print('=' * 92)
FOLDS = sorted({r['fold'] for r in rows})
feat_sets = {
    'R': ['reactivity'],
    'C': ['reactivity', 'dz'],
    'H': ['z0', 'z1', 'dz', 'reactivity', 'alpha', 'a_gate', 'cos_to_prototype'],
}
results = {}
for name, feats in feat_sets.items():
    preds_auc, truths_auc = [], []
    preds_r2, truths_r2 = [], []
    for fo in FOLDS:
        tr = [r for r in rows if r['fold'] != fo]
        te = [r for r in rows if r['fold'] == fo and r['split'] == 'test']
        if len(te) < 3:
            continue
        Xtr = np.array([[r[f] for f in feats] for r in tr])
        ytr_auc = np.array([1 if r['U'] > 0 else 0 for r in tr])
        Xte = np.array([[r[f] for f in feats] for r in te])
        yte_auc = np.array([1 if r['U'] > 0 else 0 for r in te])
        if len(np.unique(ytr_auc)) >= 2:
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr_auc)
            p = clf.predict_proba(sc.transform(Xte))[:, 1]
            preds_auc.append(p); truths_auc.append(yte_auc)
        ytr_r2 = np.array([r['U'] for r in tr])
        yte_r2 = np.array([r['U'] for r in te])
        sc2 = StandardScaler().fit(Xtr)
        reg = LinearRegression().fit(sc2.transform(Xtr), ytr_r2)
        pr = reg.predict(sc2.transform(Xte))
        preds_r2.append(pr); truths_r2.append(yte_r2)
    p_all = np.concatenate(preds_auc); y_all = np.concatenate(truths_auc)
    auc = roc_auc_score(y_all, p_all) if len(np.unique(y_all)) >= 2 else float('nan')
    pr_all = np.concatenate(preds_r2); yr_all = np.concatenate(truths_r2)
    r2 = r2_score(yr_all, pr_all)
    results[name] = {'auc': auc, 'r2': r2}
    print(f'  {name} [{",".join(feats)}]')
    print(f'      held-out AUC(U>0) = {auc:.4f}   held-out R^2(U) = {r2:.4f}')

delta_r2 = results['H']['r2'] - results['C']['r2']
delta_auc = results['H']['auc'] - results['C']['auc']
print(f'\n  Delta R^2 (H - C) = {delta_r2:+.4f}')
print(f'  Delta AUC (H - C) = {delta_auc:+.4f}')

print('\n' + '=' * 92)
print('TEST 3: partial correlation(U, dz | reactivity) on PROPOSAL test rows')
print('=' * 92)
prop_test = [r for r in test_rows if r['condition'] == 'proposal']


def partial_corr(y, x, z):
    def resid(a, b):
        b1 = np.column_stack([np.ones_like(b), b])
        beta, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ beta
    ry = resid(y, z); rx = resid(x, z)
    return stats.pearsonr(rx, ry)


if len(prop_test) >= 8:
    U = np.array([r['U'] for r in prop_test])
    dz = np.array([r['dz'] for r in prop_test])
    reac = np.array([r['reactivity'] for r in prop_test])
    r_raw, p_raw = stats.pearsonr(dz, U)
    r_pc, p_pc = partial_corr(U, dz, reac)
    print(f'  n = {len(prop_test)}')
    print(f'  raw corr(U, dz)              r={r_raw:+.4f}  p={p_raw:.3e}')
    print(f'  partial corr | reactivity    r={r_pc:+.4f}  p={p_pc:.3e}'
          f'   {"SURVIVES" if p_pc < 0.10 else "DOES NOT SURVIVE"}')
else:
    print(f'  only {len(prop_test)} proposal test rows -- too few for a partial '
          f'correlation test at this n')
    p_pc = float('nan')

print('\n' + '=' * 92)
print('VERDICT')
print('=' * 92)
beats_random = p_fish < 0.10 and comp_any['proposal'][0] > comp_any['random'][0]
delta_r2_meaningful = delta_r2 > 0.02
survives_reactivity = (not np.isnan(p_pc)) and p_pc < 0.10

print(f'  proposal beats random (any-alpha crossing, p<0.10) : {beats_random}')
print(f'  Delta R^2 (H-C) > 0.02                              : {delta_r2_meaningful} '
      f'(actual {delta_r2:+.4f})')
print(f'  U-dz dependence survives reactivity control          : {survives_reactivity}')

if not beats_random:
    verdict = ('KILL -- learned proposal does not beat random on held-out '
              'folds. The bridge from E210 (precomputed prototype direction) '
              'to a TRAINED proposal has not been established.')
elif not delta_r2_meaningful:
    verdict = ('KILL (evaluator) -- hypothesis-relative features add no '
              'held-out predictive power beyond reactivity+dz. The full '
              'evaluator is not learning anything the simple C model lacks.')
elif not survives_reactivity:
    verdict = ('KILL (confound) -- U depends on dz only through reactivity; '
              'no hypothesis-specific value signal survives the control.')
else:
    verdict = ('PASS -- proposal beats random, evaluator adds real held-out '
              'information beyond reactivity, and the effect survives the '
              'reactivity control. HRS-Lite computational primitive is '
              'supported at small-n. Next: larger-n replication before any '
              'architecture/3-seed commitment.')
print(f'\n  ==> {verdict}')
