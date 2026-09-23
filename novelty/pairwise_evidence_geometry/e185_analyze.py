"""E185 analysis -- the four preregistered gates.

Gate 1 IDENTITY      : is lesion info linearly encoded?  (L vs H, L vs S) LOSO
Gate 2 LOCALIZATION  : SCR > 1 ?
Gate 3 RETENTION     : where along E1->D1 does L diverge from S?
Gate 4 ACCESSIBILITY : can a POINTWISE 1x1x1 linear probe recover the mask
                       where the real seg head cannot?  (strict subject LOSO)
"""
import csv, json, pickle
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'E185_stages.csv')))
vecs = pickle.load(open(HERE / 'E185_vectors.pkl', 'rb'))
probe = pickle.load(open(HERE / 'E185_probe.pkl', 'rb'))
STAGES = ['E1', 'E2', 'E3', 'B', 'D3', 'D2', 'D1']
POPS = ['L', 'S', 'H']


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


def loso_vec(V, y, g):
    ys, ss = [], []
    y = np.array(y, int)
    for h in sorted(set(g)):
        tr = np.array([x != h for x in g]); te = ~tr
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(V[tr])
        c = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(V[tr]), y[tr])
        ss += list(c.predict_proba(sc.transform(V[te]))[:, 1]); ys += list(y[te])
    return auc_s(ys, ss)


cnt = {p: sum(1 for r in rows if r['population'] == p) for p in POPS}
print(f'rows={len(rows)}  populations={cnt}  subjects={len({r["subject_id"] for r in rows})}')

# ---------------- GATE 1: identity ----------------
print('\n' + '=' * 88)
print('GATE 1 -- IDENTITY: is lesion info linearly encoded in the channel-mean vector?')
print('=' * 88)
key = {(v['subject_id'], v['population'], v['comp_id']): v['_vecs'] for v in vecs}
print(f"{'stage':<7}{'L vs H':>12}{'L vs S':>12}{'S vs H':>12}")
ident = {}
for s in STAGES:
    line = f'{s:<7}'
    for a, b in [('L', 'H'), ('L', 'S'), ('S', 'H')]:
        sel = [v for v in vecs if v['population'] in (a, b)]
        if len(sel) < 6:
            line += f'{"n/a":>12}'; continue
        V = np.array([v['_vecs'][s] for v in sel])
        y = [1 if v['population'] == a else 0 for v in sel]
        g = [v['subject_id'] for v in sel]
        au = loso_vec(np.nan_to_num(V), y, g)
        ident[f'{s}|{a}{b}'] = float(au)
        line += f'{au:>12.4f}'
    print(line)

# ---------------- GATE 2/3: localization + retention ----------------
print('\n' + '=' * 88)
print('GATE 2/3 -- LOCALIZATION (SCR, >1 = lesion-focused) and DIFFUSION (entropy)')
print('=' * 88)
for metric, lbl in [('SCR10', 'spatial concentration ratio @top10%'),
                    ('H_norm', 'normalised spatial entropy'),
                    ('resp_in', 'mean response inside C')]:
    print(f'\n  {lbl}')
    print(f"  {'stage':<7}" + ''.join(f'{p:>16}' for p in POPS) + f"{'L-S gap':>12}")
    for s in STAGES:
        line = f'  {s:<7}'
        vals = {}
        for p in POPS:
            v = np.array([f(r, f'{s}_{metric}') for r in rows if r['population'] == p])
            v = v[~np.isnan(v)]
            vals[p] = v.mean() if len(v) else np.nan
            line += f'{vals[p]:>16.4f}'
        line += f'{vals["L"]-vals["S"]:>12.4f}'
        print(line)

# ---------------- GATE 3: retention curve ----------------
print('\n' + '=' * 88)
print('GATE 3 -- RETENTION: ||v_L - v_H|| and ||v_L - v_S||, normalised to E1')
print('=' * 88)
def meanvec(p, s):
    V = np.array([v['_vecs'][s] for v in vecs if v['population'] == p])
    return V.mean(0) if len(V) else None
print(f"  {'stage':<7}{'D(L,H)':>12}{'D(L,S)':>12}{'D(S,H)':>12}{'D(L,H)/E1':>12}{'D(L,S)/E1':>12}")
base = {}
for s in STAGES:
    vl, vs_, vh = meanvec('L', s), meanvec('S', s), meanvec('H', s)
    if vl is None or vh is None or vs_ is None:
        continue
    dlh = float(np.linalg.norm(vl - vh)); dls = float(np.linalg.norm(vl - vs_))
    dsh = float(np.linalg.norm(vs_ - vh))
    if s == 'E1':
        base['LH'], base['LS'] = dlh, dls
    print(f'  {s:<7}{dlh:>12.3f}{dls:>12.3f}{dsh:>12.3f}'
          f'{dlh/base["LH"]:>12.3f}{dls/base["LS"]:>12.3f}')

# ---------------- GATE 4: pointwise linear probe ----------------
print('\n' + '=' * 88)
print('GATE 4 -- ACCESSIBILITY: POINTWISE 1x1x1 linear probe, strict subject LOSO')
print('=' * 88)
print('  (probe fit on OTHER subjects only; recovers mask from channels at that voxel)')
print(f"  {'stage':<7}{'L: vox AUC':>13}{'L: Dice':>10}{'S: vox AUC':>13}{'S: Dice':>10}")
acc = {}
for s in STAGES:
    res = {}
    for pop in ['L', 'S']:
        recs = [d for d in probe if d['stage'] == s and d['pop'] == pop]
        if len(recs) < 6:
            res[pop] = (np.nan, np.nan); continue
        subs = sorted({d['sid'] for d in recs})
        ys, ss = [], []
        for h in subs:
            tr = [d for d in recs if d['sid'] != h]
            te = [d for d in recs if d['sid'] == h]
            if not te or not tr:
                continue
            Xtr = np.vstack([d['X'] for d in tr]); ytr = np.concatenate([d['y'] for d in tr])
            if len(np.unique(ytr)) < 2:
                continue
            # subsample for tractability, preserving class balance
            if len(ytr) > 60000:
                pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
                rng = np.random.default_rng(0)
                neg = rng.choice(neg, min(len(neg), 30000), replace=False)
                pos = rng.choice(pos, min(len(pos), 30000), replace=False)
                k = np.concatenate([pos, neg]); Xtr, ytr = Xtr[k], ytr[k]
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=2000, C=0.5,
                                     class_weight='balanced').fit(sc.transform(Xtr), ytr)
            for d in te:
                p = clf.predict_proba(sc.transform(d['X']))[:, 1]
                ss.append(p); ys.append(d['y'])
        if not ys:
            res[pop] = (np.nan, np.nan); continue
        Y = np.concatenate(ys); S = np.concatenate(ss)
        au = auc_s(Y, S)
        # Dice at the prevalence-matched operating point
        k = max(1, int(Y.sum()))
        thr = np.partition(S, -k)[-k]
        pred = S >= thr
        dice = 2 * float((pred & (Y == 1)).sum()) / max(1, pred.sum() + Y.sum())
        res[pop] = (float(au), float(dice))
    acc[s] = res
    print(f'  {s:<7}{res["L"][0]:>13.4f}{res["L"][1]:>10.4f}'
          f'{res["S"][0]:>13.4f}{res["S"][1]:>10.4f}')

print('\n  REFERENCE: the real seg head achieves Dice 0.0 on L by definition')
print('             (L = components with ZERO predicted overlap).')

json.dump({'identity': ident, 'accessibility': {k: {p: list(v) for p, v in r.items()}
                                                for k, r in acc.items()}},
          open(HERE / 'E185_results.json', 'w'), indent=2)
print('\nwrote E185_results.json')
