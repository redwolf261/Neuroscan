"""E209 analysis -- PREREGISTERED, written before results were seen.

Three tests (user's spec), run on L, with H as the negative control that
must NOT show the same relationship (else it's generic perturbation
magnitude, not hypothesis-value):
  1. Delta_E^cross vs Delta_E^noncross (magnitude + Mann-Whitney)
  2. subject-grouped AUC(Delta_E -> crossing)
  3. partial correlation(Delta_E, crossing | reactivity)

Crossing definition IDENTICAL to E206: component mean logit z<0 -> z>0,
here evaluated on z_region_Y0 -> z_region_Y1 from the SAME donor-splice
manipulation as E208.

KILL if: no separation, AUC~chance, association vanishes under reactivity
control, OR H shows the same Delta_E->crossing relationship as L.
"""
import csv
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'E209_crossing.csv')))
for r in rows:
    for k in list(r):
        if k not in ('subject_id', 'population', 'comp_id'):
            r[k] = float(r[k])
    r['crossing'] = int(r['crossing'])

POPS = ['L', 'H']


def partial_corr(y, x, z):
    def resid(a, b):
        b1 = np.column_stack([np.ones_like(b), b])
        beta, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ beta
    ry = resid(y, z); rx = resid(x, z)
    return stats.pearsonr(rx, ry)


def grouped_auc(X, y, groups, n_splits=5):
    if len(np.unique(y)) < 2:
        return float('nan')
    ng = len(np.unique(groups))
    if ng < 2:
        return float('nan')
    gkf = GroupKFold(n_splits=min(n_splits, ng))
    scores, truths = [], []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(sc.transform(X[tr]), y[tr])
        scores.append(clf.predict_proba(sc.transform(X[te]))[:, 1])
        truths.append(y[te])
    if not scores:
        return float('nan')
    p_all = np.concatenate(scores); y_all = np.concatenate(truths)
    if len(np.unique(y_all)) < 2:
        return float('nan')
    return roc_auc_score(y_all, p_all)


for pop in POPS:
    sel = [r for r in rows if r['population'] == pop]
    n = len(sel)
    ncross = sum(r['crossing'] for r in sel)
    print('=' * 90)
    print(f'POPULATION: {pop}   (n={n}, crossings={ncross}/{n} = {100*ncross/max(n,1):.1f}%)')
    print('=' * 90)
    if ncross == 0 or ncross == n:
        print('  degenerate (all-or-nothing crossing) -- skipping statistical tests\n')
        continue

    de = np.array([r['delta_E'] for r in sel])
    re_ = np.array([r['reactivity'] for r in sel])
    cross = np.array([r['crossing'] for r in sel])
    groups = np.array([r['subject_id'] for r in sel])

    print('\n1. MAGNITUDE: Delta_E^cross vs Delta_E^noncross')
    dc = de[cross == 1]; dn = de[cross == 0]
    print(f'   cross    (n={len(dc)}): median={np.median(dc):+.4f}  mean={dc.mean():+.4f}')
    print(f'   noncross (n={len(dn)}): median={np.median(dn):+.4f}  mean={dn.mean():+.4f}')
    if len(dc) >= 3 and len(dn) >= 3:
        pu = stats.mannwhitneyu(dc, dn)[1]
        print(f'   Mann-Whitney p = {pu:.3e}')
    else:
        pu = float('nan')
        print('   too few in one group for Mann-Whitney')

    print('\n2. subject-grouped AUC(Delta_E -> crossing)')
    auc = grouped_auc(de.reshape(-1, 1), cross, groups)
    print(f'   AUC = {auc:.4f}   (0.5 = chance)')

    print('\n3. partial correlation(Delta_E, crossing | reactivity)')
    r_raw, p_raw = stats.pointbiserialr(cross, de)
    r_pc, p_pc = partial_corr(cross.astype(float), de, re_)
    print(f'   raw corr     r={r_raw:+.4f}  p={p_raw:.3e}')
    print(f'   partial | reactivity   r={r_pc:+.4f}  p={p_pc:.3e}'
          f'   {"SURVIVES" if p_pc < 0.05 else "DOES NOT SURVIVE"}')

    globals()[f'result_{pop}'] = {
        'n': n, 'ncross': ncross, 'pu': pu, 'auc': auc,
        'r_raw': r_raw, 'p_raw': p_raw, 'r_pc': r_pc, 'p_pc': p_pc,
    }
    print()

print('=' * 90)
print('VERDICT')
print('=' * 90)
rL = globals().get('result_L')
rH = globals().get('result_H')
if rL is None:
    print('  L degenerate -- cannot evaluate. KILL (no usable crossing variance).')
else:
    l_sig = (rL['pu'] < 0.05 if not np.isnan(rL['pu']) else False)
    l_auc_ok = rL['auc'] > 0.65 if not np.isnan(rL['auc']) else False
    l_survives = rL['p_pc'] < 0.05
    print(f'  L: magnitude-separated={l_sig}  AUC={rL["auc"]:.3f} (>0.65={l_auc_ok})'
          f'  survives reactivity control={l_survives}')
    if rH is not None and not np.isnan(rH.get('p_pc', np.nan)):
        h_survives = rH['p_pc'] < 0.05
        h_auc_ok = rH['auc'] > 0.65 if not np.isnan(rH['auc']) else False
        print(f'  H: AUC={rH["auc"]:.3f} (>0.65={h_auc_ok})  '
              f'survives reactivity control={h_survives}')
        specific = l_survives and not h_survives
    else:
        print('  H: degenerate or insufficient data -- specificity check unavailable')
        specific = l_survives

    if not (l_sig and l_auc_ok and l_survives):
        verdict = 'KILL -- Delta_E does not predict crossing for L with adequate strength.'
    elif not specific:
        verdict = ('KILL (generic) -- H shows the same Delta_E->crossing relationship, '
                  'meaning this is perturbation-magnitude, not hypothesis-value, information.')
    else:
        verdict = ('PASS -- Delta_E predicts crossing specifically for L, survives '
                  'reactivity control, and does not generalise to H. Architecture '
                  'design is now licensed.')
    print(f'\n  ==> {verdict}')
