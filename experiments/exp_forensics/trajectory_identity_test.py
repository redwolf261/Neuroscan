"""TRAJECTORY IDENTITY TEST -- what is the discriminating quantity, really?

Steps 1-7 of the preregistered identity test. The question is NOT "does
coherence give high AUC" but "is there ordered computational history here, or
just a static multiscale phenotype?"

Test 4 (representation geometry) is the load-bearing one: the original T_* used
cos(dZ_a[:k], dZ_b[:k]) across stages of DIFFERENT widths, i.e. it assumed
channel index i at enc1 corresponds to channel index i at enc2. That assumption
is invalid. This script recomputes transition quantities using only
representation-INVARIANT statistics that never index-match across stages.
"""
import csv, json, pickle, itertools
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
STAGES = ['enc1', 'enc2', 'enc3', 'bottleneck', 'dec3', 'dec2', 'dec1']
TRANS = list(zip(STAGES[:-1], STAGES[1:]))

rows = list(csv.DictReader(open(HERE / 'TRAJECTORY_features.csv')))
vecs = pickle.load(open(HERE / 'TRAJECTORY_vectors.pkl', 'rb'))
key = lambda r: (r['subject_id'], r['region'], str(r['comp_id']), r['population'])
V = {key(v): {s: np.asarray(v['_dz'][s], dtype=np.float64) for s in STAGES} for v in vecs}
print(f'rows={len(rows)}  vectors={len(V)}')
print('channel widths:', {s: len(next(iter(V.values()))[s]) for s in STAGES})


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


def loso(X, y, groups, seed=None):
    """Strict leave-one-SUBJECT-out. seed!=None => permute labels within subject."""
    ys, ss = [], []
    subs = sorted(set(groups))
    yy = np.array(y, dtype=int)
    if seed is not None:
        rng = np.random.default_rng(seed)
        yy = yy.copy()
        for s in subs:
            m = np.array([g == s for g in groups])
            v = yy[m].copy(); rng.shuffle(v); yy[m] = v
    for h in subs:
        tr = np.array([g != h for g in groups]); te = ~tr
        if te.sum() == 0 or len(np.unique(yy[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(X[tr]), yy[tr])
        ss += list(clf.predict_proba(sc.transform(X[te]))[:, 1]); ys += list(yy[te])
    return auc_s(ys, ss)


# ---------- build representation-invariant transition features (TEST 4) ----------
def invariant_feats(dz):
    """Quantities that never index-match channels across stages."""
    m = {s: float(np.linalg.norm(dz[s])) for s in STAGES}
    out = {}
    for a, b in TRANS:
        # log magnitude ratio: how much does the displacement grow/shrink?
        out[f'R_{a}_{b}'] = float(np.log((m[b] + 1e-9) / (m[a] + 1e-9)))
        # participation-ratio change: how distributed is the displacement?
        def pr(v):
            p = v ** 2
            return float(p.sum() ** 2 / (np.sum(p ** 2) + 1e-30)) / len(v)
        out[f'PR_{a}_{b}'] = pr(dz[b]) - pr(dz[a])
        # kurtosis-like sparsity change (scale-free)
        def sp(v):
            av = np.abs(v)
            return float(av.max() / (av.mean() + 1e-12))
        out[f'SP_{a}_{b}'] = np.log((sp(dz[b]) + 1e-9) / (sp(dz[a]) + 1e-9))
    for s in STAGES:
        out[f'logM_{s}'] = float(np.log(m[s] + 1e-9))
    return out


INV = {k: invariant_feats(dz) for k, dz in V.items()}
INV_TRANS = [f'{p}_{a}_{b}' for p in ('R', 'PR', 'SP') for a, b in TRANS]
INV_MAG = [f'logM_{s}' for s in STAGES]

# ---------- assemble matrices ----------
def build(neg, feat_names, source='csv'):
    X, y, g = [], [], []
    for r in rows:
        if r['population'] not in ('LESION', neg):
            continue
        k = key(r)
        if source == 'inv' and k not in INV:
            continue
        src = INV[k] if source == 'inv' else r
        v = [f(src, n, 0.0) if source == 'csv' else float(src.get(n, 0.0)) for n in feat_names]
        if any(np.isnan(v)):
            continue
        X.append(v); y.append(1 if r['population'] == 'LESION' else 0); g.append(r['subject_id'])
    return np.nan_to_num(np.array(X)), np.array(y), g


BASE = ['logit_mean', 'logit_max', 'local_contrast']
T_ALL = [f'T_{a}_{b}' for a, b in TRANS]
M_ALL = [f'M_{s}' for s in STAGES]

print('\n' + '=' * 90)
print('TEST 1 -- TRANSITION LOCALIZATION: which transition separates L from H?')
print('=' * 90)
for neg in ['HARDNEG', 'ADJ']:
    print(f'\n  vs {neg}:')
    print(f"   {'transition':<24}{'T_ab AUC':>11}{'R_ab AUC':>11}{'PR_ab AUC':>11}{'SP_ab AUC':>11}")
    for a, b in TRANS:
        line = f'   {a+"->"+b:<24}'
        Xc, yc, gc = build(neg, [f'T_{a}_{b}'])
        line += f'{max(auc_s(yc, Xc[:,0]), 1-auc_s(yc, Xc[:,0])):>11.4f}'
        for p in ('R', 'PR', 'SP'):
            Xi, yi, gi = build(neg, [f'{p}_{a}_{b}'], 'inv')
            a_ = auc_s(yi, Xi[:, 0])
            line += f'{max(a_, 1-a_):>11.4f}'
        print(line)

print('\n' + '=' * 90)
print('TEST 2 -- ORDER TEST: does stage ORDER matter, or is this a static set?')
print('=' * 90)
print('  If shuffling which transition a feature belongs to changes nothing,')
print('  we are measuring a multiscale phenotype, not a trajectory.\n')
for neg in ['HARDNEG', 'ADJ']:
    X, y, g = build(neg, T_ALL)
    real = loso(X, y, g)
    # permute stage-order WITHIN each row (shuffle the transition features)
    rng = np.random.default_rng(0)
    perms = []
    for _ in range(200):
        Xp = np.array([row[rng.permutation(len(row))] for row in X])
        perms.append(loso(Xp, y, g))
    perms = np.array([p for p in perms if not np.isnan(p)])
    Xr = X[:, ::-1]
    rev = loso(Xr, y, g)
    # sorted (order-destroying but magnitude-preserving) representation
    Xs = np.sort(X, axis=1)
    srt = loso(Xs, y, g)
    print(f'  vs {neg}:  ordered {real:.4f} | reversed {rev:.4f} | sorted(order-free) {srt:.4f}')
    print(f'            stage-shuffled null: mean {perms.mean():.4f} sd {perms.std():.4f} '
          f'p(>=real)={ (perms>=real).mean():.3f}')

print('\n' + '=' * 90)
print('TEST 3 -- MAGNITUDE vs DYNAMICS DECOMPOSITION')
print('=' * 90)
for neg in ['HARDNEG', 'ADJ']:
    print(f'\n  vs {neg}:')
    combos = [
        ('base (logit+contrast)', BASE, 'csv'),
        ('unordered magnitudes (sorted)', M_ALL, 'csv'),
        ('ordered magnitudes', M_ALL, 'csv'),
        ('T_* (index-matched cos)  [SUSPECT]', T_ALL, 'csv'),
        ('INV transitions (R,PR,SP)', INV_TRANS, 'inv'),
        ('INV magnitudes only', INV_MAG, 'inv'),
        ('INV transitions + INV mags', INV_TRANS + INV_MAG, 'inv'),
    ]
    for name, fs, src in combos:
        X, y, g = build(neg, fs, src)
        if name.startswith('unordered'):
            X = np.sort(X, axis=1)
        print(f'   {name:<38}{loso(X, y, g):.4f}   (n={len(y)}, dim={X.shape[1]})')

print('\n' + '=' * 90)
print('TEST 4/5 -- INVARIANT FEATURES, STRICT LOSO, INCREMENT OVER BASE')
print('=' * 90)
for neg in ['HARDNEG', 'ADJ']:
    Xb, yb, gb = build(neg, BASE)
    base_auc = loso(Xb, yb, gb)
    # base + invariant transitions (must align rows across sources)
    keep = [r for r in rows if r['population'] in ('LESION', neg) and key(r) in INV]
    Xc = np.array([[f(r, n, 0.0) for n in BASE] + [INV[key(r)][n] for n in INV_TRANS]
                   for r in keep])
    yc = np.array([1 if r['population'] == 'LESION' else 0 for r in keep])
    gc = [r['subject_id'] for r in keep]
    comb = loso(np.nan_to_num(Xc), yc, gc)
    print(f'  vs {neg}: base {base_auc:.4f} -> base+INV {comb:.4f}   DELTA {comb-base_auc:+.4f}')

print('\n' + '=' * 90)
print('TEST 6 -- PERMUTATION NULL on the INVARIANT features (subject structure preserved)')
print('=' * 90)
for neg in ['HARDNEG', 'ADJ']:
    X, y, g = build(neg, INV_TRANS, 'inv')
    real = loso(X, y, g)
    null = np.array([loso(X, y, g, seed=s) for s in range(200)])
    null = null[~np.isnan(null)]
    z = (real - null.mean()) / (null.std() + 1e-12)
    print(f'  vs {neg}: real {real:.4f} | null {null.mean():.4f}+-{null.std():.4f} '
          f'| p={(null>=real).mean():.4f} | z={z:.2f}')

json.dump({'note': 'see stdout'}, open(HERE / 'IDENTITY_test.json', 'w'))
print('\ndone')
