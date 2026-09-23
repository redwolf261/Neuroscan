"""Local Ordering Geometry -- subject-level LOSO gate.

Decisive question: does L have a reproducible spatial ORDERING signature that H
lacks? AUC(L,H) ~ 0.5 => kill. Strong LOSO separation => phenomenon survives.
"""
import csv, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'ORDERING_features.csv')))
POPS = ['LESION', 'ADJ', 'HARDNEG']
NS = 6


def f(r, k, d=np.nan):
    try:
        return float(r.get(k, ''))
    except (TypeError, ValueError):
        return d


def auc_s(y, s):
    y = np.asarray(y); s = np.asarray(s)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = np.argsort(np.argsort(s)) + 1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def loso(X, y, g, seed=None):
    ys, ss = [], []
    yy = np.array(y, dtype=int)
    if seed is not None:
        rng = np.random.default_rng(seed)
        yy = yy.copy()
        for s in sorted(set(g)):
            m = np.array([x == s for x in g])
            v = yy[m].copy(); rng.shuffle(v); yy[m] = v
    for h in sorted(set(g)):
        tr = np.array([x != h for x in g]); te = ~tr
        if te.sum() == 0 or len(np.unique(yy[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        c = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(X[tr]), yy[tr])
        ss += list(c.predict_proba(sc.transform(X[te]))[:, 1]); ys += list(yy[te])
    return auc_s(ys, ss)


print(f'rows={len(rows)}  components={len({(r["subject_id"],r["region"],r["comp_id"]) for r in rows})}')
for p in POPS:
    print(f'  {p:<9} n={sum(1 for r in rows if r["population"]==p)}')

# ---- size/shape match verification (the confound this experiment must avoid) ----
print('\n' + '=' * 84)
print('MATCH VERIFICATION -- H must be size AND shape matched to L')
print('=' * 84)
for p in POPS:
    v = np.array([f(r, 'n_vox') for r in rows if r['population'] == p])
    print(f'  {p:<9} n_vox mean {v.mean():9.1f}  median {np.median(v):8.1f}')
lv = np.array([f(r, 'n_vox') for r in rows if r['population'] == 'LESION'])
hv = np.array([f(r, 'n_vox') for r in rows if r['population'] == 'HARDNEG'])
print(f'  L vs H identical sizes: {np.array_equal(np.sort(lv), np.sort(hv))}')

# ---- descriptive radial profiles ----
print('\n' + '=' * 84)
print('RADIAL PROFILE  m_i = mean logit in shell i  (S0 = region itself)')
print('=' * 84)
print(f"  {'pop':<9}" + ''.join(f'{"m"+str(i):>9}' for i in range(NS + 1)))
for p in POPS:
    line = f'  {p:<9}'
    for i in range(NS + 1):
        v = np.array([f(r, f'm{i}') for r in rows if r['population'] == p])
        line += f'{np.nanmean(v):>9.2f}'
    print(line)

print('\n' + '=' * 84)
print('SHAPE DESCRIPTORS (mean +- sd)')
print('=' * 84)
FEATS = ['mu', 'c_1', 'c_last', 'c_mean', 'c_last_norm', 'c_mean_norm', 'slope',
         'monotonic_frac', 'n_sign_changes', 'peak_shell', 'decay', 'coherence', 'local_rank']
print(f"  {'feature':<18}" + ''.join(f'{p:>22}' for p in POPS))
for k in FEATS:
    line = f'  {k:<18}'
    for p in POPS:
        v = np.array([f(r, k) for r in rows if r['population'] == p]); v = v[~np.isnan(v)]
        line += f'{v.mean():>14.3f}+-{v.std():<7.2f}' if len(v) else f'{"n/a":>22}'
    print(line)

# ---- single-feature AUC, L vs H ----
print('\n' + '=' * 84)
print('SINGLE-FEATURE AUC, LESION vs HARDNEG (no model, raw separability)')
print('=' * 84)
sel = [r for r in rows if r['population'] in ('LESION', 'HARDNEG')]
y = np.array([1 if r['population'] == 'LESION' else 0 for r in sel])
for k in FEATS + [f'm{i}' for i in range(NS + 1)]:
    v = np.array([f(r, k) for r in sel])
    if np.isnan(v).any():
        continue
    a = auc_s(y, v)
    flag = '  <<<' if max(a, 1 - a) > 0.75 else ''
    print(f'  {k:<18} AUC {max(a,1-a):.4f}  (raw {a:.4f}){flag}')

# ---- LOSO gate ----
print('\n' + '=' * 84)
print('LOSO GATE (subject-level)')
print('=' * 84)
SETS = {
    'mu only (absolute evidence)': ['mu'],
    'contrast only': ['c_1', 'c_last', 'c_mean'],
    'rank only': ['local_rank'],
    'SHAPE only (scale-free)': ['c_last_norm', 'c_mean_norm', 'slope', 'monotonic_frac',
                                'n_sign_changes', 'peak_shell', 'decay', 'coherence'],
    'raw profile m0..m6': [f'm{i}' for i in range(NS + 1)],
    'ALL ordering features': FEATS,
}
res = {}
for neg in ['HARDNEG', 'ADJ']:
    print(f'\n  --- LESION vs {neg} ---')
    sel = [r for r in rows if r['population'] in ('LESION', neg)]
    y = np.array([1 if r['population'] == 'LESION' else 0 for r in sel])
    g = [r['subject_id'] for r in sel]
    for name, fs in SETS.items():
        X = np.array([[f(r, k, 0.0) for k in fs] for r in sel])
        X = np.nan_to_num(X)
        a = loso(X, y, g)
        res[f'{neg}|{name}'] = a
        print(f'   {name:<34}{a:.4f}   (dim={X.shape[1]}, n={len(y)})')

# ---- permutation null on the decisive comparison ----
print('\n' + '=' * 84)
print('PERMUTATION NULL -- SHAPE-only features, LESION vs HARDNEG')
print('=' * 84)
sel = [r for r in rows if r['population'] in ('LESION', 'HARDNEG')]
y = np.array([1 if r['population'] == 'LESION' else 0 for r in sel])
g = [r['subject_id'] for r in sel]
fs = SETS['SHAPE only (scale-free)']
X = np.nan_to_num(np.array([[f(r, k, 0.0) for k in fs] for r in sel]))
real = loso(X, y, g)
null = np.array([loso(X, y, g, seed=s) for s in range(200)])
null = null[~np.isnan(null)]
print(f'  real {real:.4f} | null {null.mean():.4f}+-{null.std():.4f} '
      f'| p={(null>=real).mean():.4f} | z={(real-null.mean())/(null.std()+1e-12):.2f}')

json.dump(res, open(HERE / 'ORDERING_loso.json', 'w'), indent=2)
print('\nwrote ORDERING_loso.json')
