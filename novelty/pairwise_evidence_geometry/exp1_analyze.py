"""EXPERIMENT 1 analysis -- the four tests + preregistered kill criteria.

KILL if:
  - best defensible relational statistic has LOSO AUC < 0.70 (L vs H)
  - Pi(L) ~= Pi(H)
  - signal vanishes after controlling size / subject / mean logit / location
  - the statistic is merely equivalent to lowering the threshold
"""
import csv, json
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'EXP1_pairwise.csv')))
POPS = ['LESION', 'ADJ', 'HARDNEG']
QS = (0.5, 0.75, 0.9, 0.95)
STATS = ['Pi', 'M_tau1.0', 'M_tau2.0'] + [f'Pi_q{q}' for q in QS]


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
    r = stats.rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def loso(X, y, g, seed=None):
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
        c = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(X[tr]), yy[tr])
        ss += list(c.predict_proba(sc.transform(X[te]))[:, 1]); ys += list(yy[te])
    return auc_s(ys, ss)


n_comp = len({(r['subject_id'], r['region'], r['comp_id']) for r in rows})
print(f'rows={len(rows)}  components={n_comp}  subjects={len({r["subject_id"] for r in rows})}')
for p in POPS:
    print(f'  {p:<9} n={sum(1 for r in rows if r["population"]==p)}')

# ---------------- descriptive table ----------------
print('\n' + '=' * 96)
print('TABLE: L / A / H  x  Pi, M, Pi_q   (mean +- sd)')
print('=' * 96)
print(f"{'statistic':<14}" + ''.join(f'{p:>26}' for p in POPS))
desc = {}
for k in STATS:
    line = f'{k:<14}'
    desc[k] = {}
    for p in POPS:
        v = np.array([f(r, k) for r in rows if r['population'] == p]); v = v[~np.isnan(v)]
        desc[k][p] = (float(v.mean()), float(v.std()))
        line += f'{v.mean():>17.4f}+-{v.std():<7.4f}'
    print(line)

print(f"\n{'context':<14}" + ''.join(f'{p:>26}' for p in POPS))
for k in ['median_z', 'median_zb', 'D', 'R', 'n_vox']:
    line = f'{k:<14}'
    for p in POPS:
        v = np.array([f(r, k) for r in rows if r['population'] == p]); v = v[~np.isnan(v)]
        line += f'{v.mean():>17.4f}+-{v.std():<7.4f}'
    print(line)

# ---------------- tests 1 & 2: single-statistic AUC ----------------
print('\n' + '=' * 96)
print('TESTS 1-2: single-statistic separability (raw AUC, no model)')
print('=' * 96)
print(f"{'statistic':<14}{'L vs H':>12}{'L vs A':>12}")
for k in STATS:
    line = f'{k:<14}'
    for neg in ['HARDNEG', 'ADJ']:
        sel = [r for r in rows if r['population'] in ('LESION', neg)]
        y = [1 if r['population'] == 'LESION' else 0 for r in sel]
        s = [f(r, k) for r in sel]
        if np.isnan(s).any():
            line += f'{"n/a":>12}'; continue
        a = auc_s(y, s)
        line += f'{max(a,1-a):>12.4f}'
    print(line)

# ---------------- test 3: LOSO ----------------
print('\n' + '=' * 96)
print('TEST 3: LOSO (subject-level) -- the kill criterion is L vs H < 0.70')
print('=' * 96)
SETS = {
    'Pi alone': ['Pi'],
    'margin M alone': ['M_tau1.0', 'M_tau2.0'],
    'tail Pi_q alone': [f'Pi_q{q}' for q in QS],
    'ALL relational': STATS,
    'median_z alone (absolute)': ['median_z'],
    'ALL relational + median_z': STATS + ['median_z'],
}
res = {}
for neg in ['HARDNEG', 'ADJ']:
    print(f'\n  --- LESION vs {neg} ---')
    sel = [r for r in rows if r['population'] in ('LESION', neg)]
    y = np.array([1 if r['population'] == 'LESION' else 0 for r in sel])
    g = [r['subject_id'] for r in sel]
    for nm, fs in SETS.items():
        X = np.nan_to_num(np.array([[f(r, k, 0.0) for k in fs] for r in sel]))
        a = loso(X, y, g)
        res[f'{neg}|{nm}'] = float(a)
        flag = ''
        if neg == 'HARDNEG' and 'relational' in nm.lower() or nm in ('Pi alone',):
            flag = '  <-- KILL' if (neg == 'HARDNEG' and a < 0.70) else ''
        print(f'   {nm:<30}{a:.4f}{flag}')

# ---------------- control: is it just "lower the threshold"? ----------------
print('\n' + '=' * 96)
print('CONTROL: is the relational statistic just a restatement of absolute logit?')
print('=' * 96)
sel = [r for r in rows if r['population'] in ('LESION', 'HARDNEG')]
y = np.array([1 if r['population'] == 'LESION' else 0 for r in sel])
g = [r['subject_id'] for r in sel]
Xa = np.nan_to_num(np.array([[f(r, 'median_z', 0.0)] for r in sel]))
Xr = np.nan_to_num(np.array([[f(r, k, 0.0) for k in STATS] for r in sel]))
Xb = np.hstack([Xa, Xr])
a_abs, a_rel, a_both = loso(Xa, y, g), loso(Xr, y, g), loso(Xb, y, g)
print(f'  absolute only (median_z)      {a_abs:.4f}')
print(f'  relational only               {a_rel:.4f}')
print(f'  both                          {a_both:.4f}')
print(f'  INCREMENT of relational over absolute: {a_both - a_abs:+.4f}')

# correlation between Pi and median_z (if ~1, Pi is redundant)
pi = np.array([f(r, 'Pi') for r in sel]); mz = np.array([f(r, 'median_z') for r in sel])
sp = stats.spearmanr(pi, mz)
print(f'  Spearman(Pi, median_z) = {sp.statistic:+.3f} (p={sp.pvalue:.2e})')

# ---------------- test 4: D vs R ----------------
print('\n' + '=' * 96)
print('TEST 4: does relational evidence scale with absolute failure?  D vs R')
print('=' * 96)
for p in POPS:
    sub = [r for r in rows if r['population'] == p]
    D = np.array([f(r, 'D') for r in sub]); R = np.array([f(r, 'R') for r in sub])
    m = ~(np.isnan(D) | np.isnan(R))
    if m.sum() < 5:
        continue
    sp = stats.spearmanr(D[m], R[m])
    pe = stats.pearsonr(D[m], R[m])
    print(f'  {p:<9} n={m.sum():>3}  Spearman {sp.statistic:+.3f} (p={sp.pvalue:.2e})  '
          f'Pearson {pe[0]:+.3f}')
# size-controlled partial for LESION
sub = [r for r in rows if r['population'] == 'LESION']
D = np.array([f(r, 'D') for r in sub]); R = np.array([f(r, 'R') for r in sub])
S = np.log1p(np.array([f(r, 'n_vox') for r in sub]))
m = ~(np.isnan(D) | np.isnan(R) | np.isnan(S))
if m.sum() > 5:
    def resid(a, b):
        A = np.vstack([b, np.ones_like(b)]).T
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    sp = stats.spearmanr(resid(D[m], S[m]), resid(R[m], S[m]))
    print(f'  LESION, size-controlled partial: Spearman {sp.statistic:+.3f} (p={sp.pvalue:.2e})')

# ---------------- permutation null ----------------
print('\n' + '=' * 96)
print('PERMUTATION NULL (subject-preserving), relational-only, L vs H')
print('=' * 96)
null = np.array([loso(Xr, y, g, seed=s) for s in range(200)])
null = null[~np.isnan(null)]
print(f'  real {a_rel:.4f} | null {null.mean():.4f}+-{null.std():.4f} '
      f'| p={(null>=a_rel).mean():.4f} | z={(a_rel-null.mean())/(null.std()+1e-12):.2f}')

json.dump({'desc': desc, 'loso': res, 'abs': a_abs, 'rel': a_rel, 'both': a_both},
          open(HERE / 'EXP1_results.json', 'w'), indent=2)
print('\nwrote EXP1_results.json')
