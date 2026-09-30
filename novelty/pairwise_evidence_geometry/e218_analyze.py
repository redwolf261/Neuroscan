"""E218 analysis -- PREREGISTERED (written before full results were seen).

Transitions: consecutive checkpoints 5->10 ... 30->35 (6).
  drop       : delta recall <= -0.05
  rest_up    : delta rest_dice >= +0.001
  SACRIFICE  : drop AND rest_up          (candidate-new, conditional event)
  FORGET     : drop                       (Toneva-style, OCCUPIED)

TEST 1 (existence): is SACRIFICE more frequent than expected if a
  component's recall changes were unrelated to its subject's rest changes?
  Null: for each component, permute the time order of its rest-deltas
  (keeps both marginals, breaks their pairing). 2000 permutations.
  Also P(rest_up | drop) vs P(rest_up) overall.
  KILL if observed <= null 95th percentile.

TEST 2: per-component sacrifice counts / conditional rates, L vs S vs M
  (final label at epoch 35). Mann-Whitney.

TEST 3 (collision gate vs forgetting): predict final L (recall35 == 0) from
  EARLY window only (transitions 5->...->25) among AT-RISK components
  (recall25 > 0). Baseline features: forget_early, mean recall early,
  recall25, log size. +sacrifice_early. Subject-grouped 5-fold CV, 20
  repeats, Delta AUC; plus full-fit sacrifice coefficient bootstrap CI.
  KILL if Delta AUC <= 0.01 OR coefficient CI includes 0.
"""
import csv, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
fn = sys.argv[1] if len(sys.argv) > 1 else 'E218_trajectory.csv'
rows = list(csv.DictReader(open(HERE / fn)))
traj = defaultdict(dict)
size = {}
for r in rows:
    k = (r['subject_id'], int(r['comp_id']))
    traj[k][int(r['epoch'])] = (float(r['recall']), float(r['rest_dice']))
    size[k] = int(r['size'])
EP = sorted({int(r['epoch']) for r in rows})
keys = [k for k in traj if len(traj[k]) == len(EP)]
DROP, UP = -0.05, 0.001

rec = np.array([[traj[k][e][0] for e in EP] for k in keys])
rst = np.array([[traj[k][e][1] for e in EP] for k in keys])
dR, dS = np.diff(rec, axis=1), np.diff(rst, axis=1)
drop, up = dR <= DROP, dS >= UP
sac = drop & up
final = rec[:, -1]
label = np.where(final == 0, 'L', np.where(final >= 0.5, 'S', 'M'))
subj = np.array([k[0] for k in keys])
print(f'components={len(keys)}  L={np.sum(label=="L")} S={np.sum(label=="S")} '
      f'M={np.sum(label=="M")}  transitions/comp={dR.shape[1]}\n')

print('=' * 90)
print('TEST 1: does SACRIFICE (recall drop while rest improves) exceed chance?')
print('=' * 90)
rng = np.random.default_rng(0)
for pop in ['ALL', 'L', 'S', 'M']:
    m = np.ones(len(keys), bool) if pop == 'ALL' else label == pop
    if m.sum() == 0:
        continue
    obs = int(sac[m].sum())
    null = []
    for _ in range(2000):
        perm_up = np.array([u[rng.permutation(len(u))] for u in up[m]])
        null.append(int((drop[m] & perm_up).sum()))
    null = np.array(null)
    p = float((null >= obs).mean())
    pu_given_d = up[m][drop[m]].mean() if drop[m].any() else np.nan
    print(f'  {pop:<4} n={m.sum():4d}  sacrifice obs={obs:4d}  null mean={null.mean():7.1f} '
          f'95th={np.percentile(null,95):6.1f}  p={p:.4f}  |  P(rest_up|drop)={pu_given_d:.3f}'
          f'  P(rest_up)={up[m].mean():.3f}  drops={int(drop[m].sum())}')
print()

print('=' * 90)
print('TEST 2: per-component counts by final label')
print('=' * 90)
P = sac.sum(1); Fg = drop.sum(1)
for pop in ['L', 'M', 'S']:
    m = label == pop
    if m.any():
        print(f'  {pop}: sacrifice mean={P[m].mean():.3f}  forget mean={Fg[m].mean():.3f}  '
              f'frac with >=2 sacrifices={np.mean(P[m]>=2):.3f}')
if (label == 'L').any() and (label == 'S').any():
    print(f'  Mann-Whitney sacrifice L vs S: p={stats.mannwhitneyu(P[label=="L"], P[label=="S"])[1]:.3e}')
    print(f'  Mann-Whitney forget    L vs S: p={stats.mannwhitneyu(Fg[label=="L"], Fg[label=="S"])[1]:.3e}')
print()

print('=' * 90)
print('TEST 3 (collision gate): does early SACRIFICE predict final miss beyond FORGETTING?')
print('=' * 90)
i25 = EP.index(25)
early = slice(0, i25)                      # transitions ending at or before epoch 25
at_risk = rec[:, i25] > 0
y = (final == 0).astype(int)[at_risk]
Xb = np.column_stack([drop[:, early].sum(1), rec[:, :i25 + 1].mean(1), rec[:, i25],
                      np.log(np.array([size[k] for k in keys]))])[at_risk]
xs = sac[:, early].sum(1)[at_risk]
g = subj[at_risk]
print(f'  at-risk components (recall25>0): {at_risk.sum()}, of which end missed: {y.sum()}')
if y.sum() >= 5 and (len(y) - y.sum()) >= 5 and len(set(g)) >= 5:
    aucs_b, aucs_f = [], []
    for rep in range(20):
        order = np.random.default_rng(rep).permutation(len(y))
        yb, Xbb, xsb, gb = y[order], Xb[order], xs[order], g[order]
        pb, pf, yt = [], [], []
        for tr, te in GroupKFold(5).split(Xbb, yb, gb):
            if len(set(yb[tr])) < 2:
                continue
            for X_, store in [(Xbb, pb), (np.column_stack([Xbb, xsb]), pf)]:
                sc = StandardScaler().fit(X_[tr])
                clf = LogisticRegression(max_iter=2000).fit(sc.transform(X_[tr]), yb[tr])
                store.append(clf.predict_proba(sc.transform(X_[te]))[:, 1])
            yt.append(yb[te])
        yt = np.concatenate(yt)
        aucs_b.append(roc_auc_score(yt, np.concatenate(pb)))
        aucs_f.append(roc_auc_score(yt, np.concatenate(pf)))
    da = np.array(aucs_f) - np.array(aucs_b)
    print(f'  AUC baseline (forget+recall+size) = {np.mean(aucs_b):.4f}')
    print(f'  AUC +sacrifice                    = {np.mean(aucs_f):.4f}')
    print(f'  Delta AUC = {da.mean():+.4f}  (repeats >0: {np.mean(da>0):.2f})')
    Xf = np.column_stack([Xb, xs])
    coefs = []
    brng = np.random.default_rng(1)
    usub = np.unique(g)
    for _ in range(1000):
        pick = brng.choice(usub, len(usub), replace=True)
        idx = np.concatenate([np.where(g == s)[0] for s in pick])
        if len(set(y[idx])) < 2:
            continue
        sc = StandardScaler().fit(Xf[idx])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xf[idx]), y[idx])
        coefs.append(clf.coef_[0][-1])
    lo, hi = np.percentile(coefs, [2.5, 97.5])
    print(f'  sacrifice coefficient (standardized): median {np.median(coefs):+.3f}  '
          f'95% CI [{lo:+.3f}, {hi:+.3f}]')
    gate = da.mean() > 0.01 and (lo > 0 or hi < 0)
    print(f'\n  COLLISION GATE: {"PASS" if gate else "FAIL -- sacrifice adds nothing beyond forgetting"}')
else:
    print('  too few at-risk outcomes for Test 3')
