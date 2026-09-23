"""E187 analysis -- PREREGISTERED, written before any results were seen.

THE GATE: does coherence surplus dC carry LESION-SPECIFIC information at
E2 -> pool2?

DECISIVE CONTRAST = L vs H. Hard negatives are shape- and size-matched and
sited at the max-logit background location, so they control for "any bright
blob has coherent support". If dC_L ~= dC_H, SEC++ dies here regardless of
how L compares to S.

L vs S is reported but is NOT the gate: S components are detected, so any
L-vs-S difference is confounded with detection itself.

KILL CRITERIA (fixed before seeing data):
  K1  dC is ~zero everywhere              -> |median dC| < 0.01 in all pops
  K2  dC does not separate L from H       -> LOSO AUC < 0.60 or perm p > 0.05
  K3  dC adds nothing over magnitude      -> dC AUC <= (p, S) AUC, i.e. the
                                             surplus is redundant with the
                                             winner and support magnitude

PASS requires: dC_L > dC_H with AUC >= 0.60, permutation p < 0.05, AND dC
surviving as a predictor after partialling out p and S. The third is the one
that matters: SEC++'s claim is that ORGANIZATION beyond MAGNITUDE carries
signal. If magnitude alone explains it, the null-calibration bought nothing
and plain SEC would do.
"""
import csv
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'E187_coherence.csv')))
NUM = ('p_med', 'S_med', 'S_mean', 'Cobs_med', 'Cnull_med',
       'dC_med', 'dC_mean', 'frac_dC_pos', 'delta', 'vox', 'nwin')
for r in rows:
    for k in NUM:
        r[k] = float(r[k])
POPS = ['L', 'S', 'H']
cnt = {p: sum(1 for r in rows if r['population'] == p) for p in POPS}
print(f'rows={len(rows)}  pops={cnt}  subjects={len({r["subject_id"] for r in rows})}')
print(f'delta = {rows[0]["delta"]:.4f}  (measured from E2, not tuned)\n')


def auc_s(y, s):
    y, s = np.asarray(y), np.asarray(s)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def loso(V, y, g):
    """Strict subject leave-one-out; V may be 1-D or 2-D."""
    V = np.atleast_2d(V.T).T if V.ndim == 1 else V
    y = np.asarray(y, int)
    ys, ss = [], []
    for h in sorted(set(g)):
        tr = np.array([x != h for x in g]); te = ~tr
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(V[tr])
        c = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(V[tr]), y[tr])
        ss += list(c.predict_proba(sc.transform(V[te]))[:, 1]); ys += list(y[te])
    return auc_s(ys, ss)


# ---------------- descriptive ----------------
print('=' * 80)
print('COHERENCE STATISTICS BY POPULATION (median over components)')
print('=' * 80)
print(f"{'stat':<14}" + ''.join(f'{p:>14}' for p in POPS) + f"{'L-H':>12}")
for k in ('p_med', 'S_med', 'Cobs_med', 'Cnull_med', 'dC_med', 'frac_dC_pos'):
    v = {p: np.median([r[k] for r in rows if r['population'] == p]) for p in POPS}
    print(f'{k:<14}' + ''.join(f'{v[p]:>14.4f}' for p in POPS)
          + f'{v["L"]-v["H"]:>12.4f}')

# ---------------- the gate: L vs H ----------------
print('\n' + '=' * 80)
print('GATE -- L vs H (decisive), subject-LOSO AUC + permutation null')
print('=' * 80)
sel = [r for r in rows if r['population'] in ('L', 'H')]
y = [1 if r['population'] == 'L' else 0 for r in sel]
g = [r['subject_id'] for r in sel]
print(f'  n_L={sum(y)}  n_H={len(y)-sum(y)}\n')
print(f"  {'predictor':<28}{'LOSO AUC':>10}{'perm p':>10}")
res = {}
FEATS = [('dC_med (THE CLAIM)', ['dC_med']),
         ('dC_mean', ['dC_mean']),
         ('frac_dC_pos', ['frac_dC_pos']),
         ('p_med (winner magnitude)', ['p_med']),
         ('S_med (support magnitude)', ['S_med']),
         ('Cobs_med (raw coherence)', ['Cobs_med']),
         ('p + S (magnitude only)', ['p_med', 'S_med']),
         ('p + S + dC (full)', ['p_med', 'S_med', 'dC_med'])]
for name, ks in FEATS:
    V = np.array([[r[k] for k in ks] for r in sel])
    a = loso(V, y, g)
    # subject-preserving permutation null: shuffle labels WITHIN subject
    null = []
    rng = np.random.default_rng(0)
    for _ in range(200):
        yp = np.array(y).copy()
        for h in set(g):
            m = np.array([x == h for x in g])
            yp[m] = rng.permutation(yp[m])
        null.append(loso(V, yp, g))
    null = np.array([x for x in null if np.isfinite(x)])
    p = float((null >= a).mean()) if len(null) else np.nan
    res[name] = (a, p)
    print(f'  {name:<28}{a:>10.4f}{p:>10.3f}')

# ---------------- K3: does dC add over magnitude? ----------------
print('\n' + '=' * 80)
print('K3 -- INCREMENT: does organization add over magnitude?')
print('=' * 80)
a_mag = res['p + S (magnitude only)'][0]
a_full = res['p + S + dC (full)'][0]
print(f'  magnitude only (p,S)      AUC = {a_mag:.4f}')
print(f'  magnitude + surplus       AUC = {a_full:.4f}')
print(f'  increment from dC             = {a_full-a_mag:+.4f}')
# partial correlation of dC with label, controlling p and S
V = np.array([[r['p_med'], r['S_med']] for r in sel])
d = np.array([r['dC_med'] for r in sel]); yy = np.array(y, float)
Vc = np.column_stack([np.ones(len(V)), V])
bd = np.linalg.lstsq(Vc, d, rcond=None)[0]; rd = d - Vc @ bd
by = np.linalg.lstsq(Vc, yy, rcond=None)[0]; ry = yy - Vc @ by
pr, pp = stats.pearsonr(rd, ry)
print(f'  partial corr(dC, label | p,S) = {pr:+.4f}  (p={pp:.2e})')

# ---------------- reported, not the gate ----------------
print('\n' + '=' * 80)
print('REPORTED (not the gate): L vs S -- confounded with detection itself')
print('=' * 80)
sel2 = [r for r in rows if r['population'] in ('L', 'S')]
y2 = [1 if r['population'] == 'L' else 0 for r in sel2]
g2 = [r['subject_id'] for r in sel2]
for name, ks in [('dC_med', ['dC_med']), ('p_med', ['p_med']), ('S_med', ['S_med'])]:
    V = np.array([[r[k] for k in ks] for r in sel2])
    print(f'  {name:<28}{loso(V, y2, g2):>10.4f}')

# ---------------- verdict ----------------
print('\n' + '=' * 80)
print('PREREGISTERED VERDICT')
print('=' * 80)
med = {p: np.median([r['dC_med'] for r in rows if r['population'] == p]) for p in POPS}
k1 = all(abs(med[p]) < 0.01 for p in POPS)
a_dc, p_dc = res['dC_med (THE CLAIM)']
k2 = (a_dc < 0.60) or (p_dc > 0.05)
k3 = (a_full - a_mag) <= 0 or pp > 0.05
print(f'  K1 dC ~ zero everywhere      : {"FIRED" if k1 else "passed"}'
      f'   (medians L={med["L"]:+.4f} S={med["S"]:+.4f} H={med["H"]:+.4f})')
print(f'  K2 dC does not separate L/H  : {"FIRED" if k2 else "passed"}'
      f'   (AUC={a_dc:.4f}, p={p_dc:.3f})')
print(f'  K3 dC redundant w/ magnitude : {"FIRED" if k3 else "passed"}'
      f'   (increment {a_full-a_mag:+.4f}, partial p={pp:.2e})')
print(f'\n  ==> {"KILL" if (k1 or k2 or k3) else "SURVIVES -- worth one seed"}')
