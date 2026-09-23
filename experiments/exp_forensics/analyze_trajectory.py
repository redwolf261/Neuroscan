"""Steps 2-5 analysis: descriptive trajectory/persistence stats + LOSO incremental AUC.

The kill criterion is AUC(X3) - AUC(X2), i.e. does trajectory add anything over
logit + local contrast? Evaluated with strict leave-one-SUBJECT-out.
"""
import csv, json, itertools
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'TRAJECTORY_features.csv')))
STAGES = ['enc1', 'enc2', 'enc3', 'bottleneck', 'dec3', 'dec2', 'dec1']
SIGMAS = [0.0, 1.0, 2.0, 4.0, 8.0]
POPS = ['LESION', 'ADJ', 'HARDNEG']


def f(r, k, d=np.nan):
    v = r.get(k, '')
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


print(f"n rows = {len(rows)}   components = {len({(r['subject_id'],r['region'],r['comp_id']) for r in rows})}")
for p in POPS:
    print(f"   {p:<9} n={sum(1 for r in rows if r['population']==p)}")

# ---------------- Step 2: trajectory descriptive ----------------
print('\n' + '=' * 96)
print('STEP 2 -- REPRESENTATION TRAJECTORY (mean +/- sd by population)')
print('=' * 96)
print(f"{'feature':<22}" + ''.join(f'{p:>24}' for p in POPS))
for feat in ['M_enc1', 'M_enc3', 'M_bottleneck', 'M_dec1', 'A_dec1_seg', 'T_mean', 'turning']:
    line = f'{feat:<22}'
    for p in POPS:
        v = np.array([f(r, feat) for r in rows if r['population'] == p])
        v = v[~np.isnan(v)]
        line += f'{v.mean():>14.3f}+-{v.std():<8.3f}'
    print(line)

print('\n  magnitude profile ||dZ_s|| by stage:')
print(f"   {'stage':<12}" + ''.join(f'{p:>12}' for p in POPS))
for s in STAGES:
    line = f'   {s:<12}'
    for p in POPS:
        v = np.array([f(r, f'M_{s}') for r in rows if r['population'] == p])
        line += f'{np.nanmean(v):>12.2f}'
    print(line)

# ---------------- Step 3: spatial persistence ----------------
print('\n' + '=' * 96)
print('STEP 3 -- SPATIAL PERSISTENCE: logit advantage over patch background at scale sigma')
print('=' * 96)
print(f"   {'sigma':<8}" + ''.join(f'{p:>16}' for p in POPS) + f"{'LESION-HARDNEG':>18}")
for sg in SIGMAS:
    line = f'   {sg:<8.1f}'
    vals = {}
    for p in POPS:
        v = np.array([f(r, f'Lsig_{sg}_delta') for r in rows if r['population'] == p])
        vals[p] = np.nanmean(v)
        line += f'{vals[p]:>16.3f}'
    line += f"{vals['LESION']-vals['HARDNEG']:>18.3f}"
    print(line)
print('\n   (if LESION-HARDNEG grows with sigma => evidence is spatially coherent,')
print('    not merely a stronger point response)')

# ---------------- Steps 4+5: LOSO incremental AUC ----------------
print('\n' + '=' * 96)
print('STEPS 4-5 -- LOSO INCREMENTAL AUC (the kill criterion)')
print('=' * 96)

TRAJ = [f'M_{s}' for s in STAGES] + ['A_dec1_seg', 'T_mean', 'turning'] + \
       [f'T_{a}_{b}' for a, b in zip(STAGES[:-1], STAGES[1:])]
SETS = {
    'X1  L (logit only)': ['logit_mean', 'logit_max'],
    'X2  L + local contrast': ['logit_mean', 'logit_max', 'local_contrast'],
    'X2b L + contrast + scale': ['logit_mean', 'logit_max', 'local_contrast'] +
                                [f'Lsig_{sg}_delta' for sg in SIGMAS],
    'X3  L + contrast + trajectory': ['logit_mean', 'logit_max', 'local_contrast'] + TRAJ,
    'X3b full (all of the above)': ['logit_mean', 'logit_max', 'local_contrast'] +
                                   [f'Lsig_{sg}_delta' for sg in SIGMAS] + TRAJ,
    'TRAJ only': TRAJ,
}


def auc_score(y, s):
    y = np.asarray(y); s = np.asarray(s)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = np.argsort(np.argsort(s)) + 1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def loso(feat_names, neg_pop):
    sel = [r for r in rows if r['population'] in ('LESION', neg_pop)]
    subs = sorted({r['subject_id'] for r in sel})
    ys, ss = [], []
    for held in subs:
        tr = [r for r in sel if r['subject_id'] != held]
        te = [r for r in sel if r['subject_id'] == held]
        if not te or len({r['population'] for r in tr}) < 2:
            continue
        Xtr = np.array([[f(r, k, 0.0) for k in feat_names] for r in tr])
        ytr = np.array([1 if r['population'] == 'LESION' else 0 for r in tr])
        Xte = np.array([[f(r, k, 0.0) for k in feat_names] for r in te])
        yte = np.array([1 if r['population'] == 'LESION' else 0 for r in te])
        Xtr = np.nan_to_num(Xtr); Xte = np.nan_to_num(Xte)
        if len(np.unique(ytr)) < 2:
            continue
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(Xtr), ytr)
        ss += list(clf.predict_proba(sc.transform(Xte))[:, 1]); ys += list(yte)
    return auc_score(ys, ss), len(ys)


results = {}
for neg, label in [('ADJ', 'vs ADJACENT background (easy)'),
                   ('HARDNEG', 'vs HARD-NEGATIVE background (the real enemy)')]:
    print(f'\n  --- LESION {label} ---')
    print(f"   {'feature set':<34}{'LOSO AUC':>11}{'n':>7}")
    base = {}
    for name, fs in SETS.items():
        a, n = loso(fs, neg)
        base[name] = a
        print(f'   {name:<34}{a:>11.4f}{n:>7}')
    results[neg] = base
    d1 = base['X3  L + contrast + trajectory'] - base['X2  L + local contrast']
    d2 = base['X3b full (all of the above)'] - base['X2b L + contrast + scale']
    print(f"\n   DELTA AUC (X3 - X2)   = {d1:+.4f}   <-- primary kill criterion")
    print(f"   DELTA AUC (X3b - X2b) = {d2:+.4f}   <-- with scale features also controlled")

json.dump(results, open(HERE / 'TRAJECTORY_loso.json', 'w'), indent=2)
print('\nwrote TRAJECTORY_loso.json')
