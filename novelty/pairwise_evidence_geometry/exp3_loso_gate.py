"""EXPERIMENT 3 -- LOSO physics gate on RAW fp32 logits.

Decisive population: L vs H (missed lesion vs shape/size-matched hard negative).
L vs A is reported for context only and is NOT the gate.

Preregistered decision rule (fixed before running):
  best single feature < 0.70            -> KILL
  best single feature 0.70-0.80         -> WEAK
  best single feature > 0.80            -> STRONG
  combined - best single  < 0.03        -> mostly redundant
  combined - best single >= 0.05        -> complementary mechanism

Features:
  F1 mu_C                       absolute level
  F2 sigma_C                    internal dispersion
  F3 IQR_C                      internal dispersion (robust)
  F4 dQ50 = Q50(C) - Q50(B)     local median contrast
  F5 dmu  = mu_C  - mu_B        local mean contrast
  F6 [mu, sigma, dQ50]
  F7 [mu, sigma, dQ50, IQR]

Controls: subject-preserving permutation null; size/location partials.
"""
import csv, json
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'EXP2_raw_regions.csv')))


def f(r, k, d=np.nan):
    try:
        return float(r.get(k, ''))
    except (TypeError, ValueError):
        return d


for r in rows:
    r['_dQ50'] = f(r, 'C_Q50') - f(r, 'B_Q50')
    r['_dmu'] = f(r, 'C_mu') - f(r, 'B_mu')


def auc_s(y, s):
    y = np.asarray(y); s = np.asarray(s)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def loso(X, y, g, seed=None):
    """Strict leave-one-SUBJECT-out. seed != None permutes labels within subject."""
    ys, ss = [], []
    yy = np.array(y, int)
    if seed is not None:
        rng = np.random.default_rng(seed); yy = yy.copy()
        for s in sorted(set(g)):
            m = np.array([x == s for x in g])
            v = yy[m].copy(); rng.shuffle(v); yy[m] = v
    for h in sorted(set(g)):
        tr = np.array([x != h for x in g]); te = ~tr
        if te.sum() == 0 or len(np.unique(yy[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        c = LogisticRegression(max_iter=5000, C=0.5).fit(sc.transform(X[tr]), yy[tr])
        ss += list(c.predict_proba(sc.transform(X[te]))[:, 1]); ys += list(yy[te])
    return auc_s(ys, ss)


def build(neg, keys):
    sel = [r for r in rows if r['population'] in ('LESION', neg)]
    X = np.array([[f(r, k, 0.0) if not k.startswith('_') else r[k] for k in keys] for r in sel],
                 dtype=float)
    y = np.array([1 if r['population'] == 'LESION' else 0 for r in sel])
    g = [r['subject_id'] for r in sel]
    return np.nan_to_num(X), y, g


SINGLE = [('F1 mu_C (absolute)', ['C_mu']),
          ('F2 sigma_C', ['C_sd']),
          ('F3 IQR_C', ['C_iqr']),
          ('F4 dQ50 (local median contrast)', ['_dQ50']),
          ('F5 dmu (local mean contrast)', ['_dmu'])]
COMBO = [('F6 [mu, sigma, dQ50]', ['C_mu', 'C_sd', '_dQ50']),
         ('F7 [mu, sigma, dQ50, IQR]', ['C_mu', 'C_sd', '_dQ50', 'C_iqr'])]

n_comp = len({(r['subject_id'], r['region'], r['comp_id']) for r in rows})
subs = sorted({r['subject_id'] for r in rows})
print(f'components={n_comp}  subjects={len(subs)}  rows={len(rows)}')
for p in ['LESION', 'ADJ', 'HARDNEG']:
    print(f'  {p:<9} n={sum(1 for r in rows if r["population"]==p)}')

results = {}
for neg, label in [('HARDNEG', 'DECISIVE GATE'), ('ADJ', 'context only')]:
    print('\n' + '=' * 88)
    print(f'LESION vs {neg}   [{label}]')
    print('=' * 88)
    print(f"{'feature':<36}{'raw AUC':>10}{'LOSO AUC':>11}")
    best_single, best_name = -1, None
    for nm, keys in SINGLE:
        X, y, g = build(neg, keys)
        ra = auc_s(y, X[:, 0]); ra = max(ra, 1 - ra)
        la = loso(X, y, g)
        results[f'{neg}|{nm}'] = float(la)
        print(f'  {nm:<34}{ra:>10.4f}{la:>11.4f}')
        if neg == 'HARDNEG' and la > best_single:
            best_single, best_name = la, nm
    print()
    best_combo = -1
    for nm, keys in COMBO:
        X, y, g = build(neg, keys)
        la = loso(X, y, g)
        results[f'{neg}|{nm}'] = float(la)
        print(f'  {nm:<34}{"":>10}{la:>11.4f}')
        if la > best_combo:
            best_combo = la
    if neg == 'HARDNEG':
        print(f'\n  best single: {best_name} = {best_single:.4f}')
        print(f'  best combined: {best_combo:.4f}   increment {best_combo-best_single:+.4f}')
        verdict = ('KILL' if best_single < 0.70 else
                   'WEAK' if best_single <= 0.80 else 'STRONG')
        inc = best_combo - best_single
        redun = ('mostly redundant' if inc < 0.03 else
                 'complementary' if inc >= 0.05 else 'intermediate')
        print(f'\n  >>> PREREGISTERED VERDICT: {verdict}   (combination: {redun})')
        results['verdict'] = verdict
        results['best_single'] = float(best_single)
        results['best_single_name'] = best_name
        results['best_combo'] = float(best_combo)

# ---- permutation null on the decisive gate ----
print('\n' + '=' * 88)
print('PERMUTATION NULL (subject-preserving), L vs H')
print('=' * 88)
for nm, keys in [('dQ50 alone', ['_dQ50']), ('sigma alone', ['C_sd']),
                 ('[mu,sigma,dQ50]', ['C_mu', 'C_sd', '_dQ50'])]:
    X, y, g = build('HARDNEG', keys)
    real = loso(X, y, g)
    null = np.array([loso(X, y, g, seed=s) for s in range(200)])
    null = null[~np.isnan(null)]
    z = (real - null.mean()) / (null.std() + 1e-12)
    print(f'  {nm:<18} real {real:.4f} | null {null.mean():.4f}+-{null.std():.4f} '
          f'| p={(null>=real).mean():.4f} | z={z:.2f}')

# ---- controls: size and location ----
print('\n' + '=' * 88)
print('CONTROLS (L vs H): does the signal survive size / location?')
print('=' * 88)
sel = [r for r in rows if r['population'] in ('LESION', 'HARDNEG')]
y = np.array([1 if r['population'] == 'LESION' else 0 for r in sel])
g = [r['subject_id'] for r in sel]
sz = np.log1p(np.array([f(r, 'C_n') for r in sel]))
print(f'  size (log n_vox) alone            LOSO {loso(sz.reshape(-1,1), y, g):.4f}')
for nm, keys in [('dQ50', ['_dQ50']), ('sigma', ['C_sd'])]:
    X, _, _ = build('HARDNEG', keys)
    Xs = np.hstack([X, sz.reshape(-1, 1)])
    print(f'  {nm} + size                        LOSO {loso(Xs, y, g):.4f}')
    # partial correlation of the feature with label, controlling size
    v = X[:, 0]
    A = np.vstack([sz, np.ones_like(sz)]).T
    resid = v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    a = auc_s(y, resid)
    print(f'    {nm} size-residualised raw AUC   {max(a,1-a):.4f}')

json.dump(results, open(HERE / 'EXP3_loso.json', 'w'), indent=2)
print('\nwrote EXP3_loso.json')
