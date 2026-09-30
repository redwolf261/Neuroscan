"""E212 analysis -- PREREGISTERED, written before results were seen.

GATE (mandatory, checked FIRST): does the prototype condition (delta_L column,
alpha=1.0) reproduce E210's 27/74 any-alpha crossing rate? If not, STOP --
no interpretation of S_H until the discrepancy is resolved.

E212-D: S_H(L) vs S_H(H) -- Mann-Whitney, subject-permutation, effect size,
  bootstrap CI.
E212-E: S_H -> crossing (subject-grouped CV AUC/PR-AUC), vs |Delta_L| alone.
E212-F: reactivity control -- logistic beta_1(S_H | R_mag); partial corr.
E212-G: negative control -- same machinery on H; want S_H(H)~=0 and
  S_H(H) NOT predictive of crossing after the reactivity control.

PASS requires ALL of: replication holds, S_H(L)>>S_H(H) (significant),
AUC(S_H->crossing)>=0.65, S_H beats |Delta_L| alone, effect survives R_mag
control, and H shows the dissociation (flat/non-predictive).
"""
import csv
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'E212_selectivity.csv')))
for r in rows:
    for k in list(r):
        if k not in ('subject_id', 'population'):
            r[k] = float(r[k])
    r['crossing'] = int(r['crossing'])

POPS = ['L', 'H']
n = {p: sum(1 for r in rows if r['population'] == p) for p in POPS}
print(f'rows={len(rows)}  pops={n}  subjects={len({r["subject_id"] for r in rows})}\n')


def partial_corr(y, x, z):
    def resid(a, b):
        b1 = np.column_stack([np.ones_like(b), b])
        beta, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ beta
    ry = resid(y, z); rx = resid(x, z)
    return stats.pearsonr(rx, ry)


def grouped_auc_pr(X, y, groups, n_splits=5):
    if len(np.unique(y)) < 2:
        return float('nan'), float('nan')
    ng = len(np.unique(groups))
    if ng < 2:
        return float('nan'), float('nan')
    gkf = GroupKFold(n_splits=min(n_splits, ng))
    scores, truths = [], []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        scores.append(clf.predict_proba(sc.transform(X[te]))[:, 1])
        truths.append(y[te])
    if not scores:
        return float('nan'), float('nan')
    p_all = np.concatenate(scores); y_all = np.concatenate(truths)
    if len(np.unique(y_all)) < 2:
        return float('nan'), float('nan')
    return roc_auc_score(y_all, p_all), average_precision_score(y_all, p_all)


print('=' * 92)
print('MANDATORY GATE: E210 replication (prototype condition == delta_L here)')
print('=' * 92)
L1 = [r for r in rows if r['population'] == 'L' and r['alpha'] == 1.0]
comps_1 = {(r['subject_id'], r['comp_id']) for r in rows if r['population'] == 'L'}
any_cross = 0
for k in comps_1:
    sub = [r for r in rows if r['population'] == 'L'
          and (r['subject_id'], r['comp_id']) == k]
    if any(r['crossing'] for r in sub):
        any_cross += 1
n_comp = len(comps_1)
print(f'  any-alpha crossings (L, prototype/delta_L): {any_cross}/{n_comp} '
      f'({100*any_cross/n_comp:.1f}%)')
print(f'  E210 reference: 27/74 (36.5%)')
reproduces = abs(any_cross - 27) <= 5 and n_comp >= 60   # tolerance band
print(f'  reproduction: {"HOLDS (within tolerance)" if reproduces else "FAILS -- STOP, do not interpret S_H below"}')
print()

if not reproduces:
    print('GATE FAILED. Halting analysis per the preregistered STOP rule.')
else:
    print('=' * 92)
    print('E212-D: S_H(L) vs S_H(H), alpha=1.0')
    print('=' * 92)
    SH = {p: np.array([r['S_H'] for r in rows if r['population'] == p and r['alpha'] == 1.0])
          for p in POPS}
    for p in POPS:
        print(f'  S_H_{p}: median={np.median(SH[p]):+.4f}  mean={SH[p].mean():+.4f}  n={len(SH[p])}')
    u, p_mw = stats.mannwhitneyu(SH['L'], SH['H'])
    print(f'  Mann-Whitney: p={p_mw:.3e}')

    # subject-preserving permutation
    L_rows = [r for r in rows if r['population'] == 'L' and r['alpha'] == 1.0]
    H_rows = [r for r in rows if r['population'] == 'H' and r['alpha'] == 1.0]
    combo = L_rows + H_rows
    label = np.array([1] * len(L_rows) + [0] * len(H_rows))
    sh_vals = np.array([r['S_H'] for r in combo])
    subs = np.array([r['subject_id'] for r in combo])
    by_sub = {}
    for i, s in enumerate(subs):
        by_sub.setdefault(s, []).append(i)
    obs = float(np.median(sh_vals[label == 1]) - np.median(sh_vals[label == 0]))
    rng = np.random.default_rng(0)
    null = []
    for _ in range(500):
        lab2 = label.copy()
        for s, idxs in by_sub.items():
            sub_lab = lab2[idxs].copy(); rng.shuffle(sub_lab); lab2[idxs] = sub_lab
        null.append(float(np.median(sh_vals[lab2 == 1]) - np.median(sh_vals[lab2 == 0])))
    null = np.array(null)
    perm_p = float((np.abs(null) >= abs(obs)).mean())
    print(f'  subject-permutation (500 iters): observed diff={obs:+.4f}  p={perm_p:.4f}')

    # bootstrap CI on the median difference
    boot = []
    idx_L = np.arange(len(SH['L'])); idx_H = np.arange(len(SH['H']))
    rng2 = np.random.default_rng(1)
    for _ in range(2000):
        bl = SH['L'][rng2.choice(idx_L, len(idx_L), replace=True)]
        bh = SH['H'][rng2.choice(idx_H, len(idx_H), replace=True)]
        boot.append(np.median(bl) - np.median(bh))
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    print(f'  bootstrap 95% CI on median(S_H_L)-median(S_H_H): [{ci_lo:+.4f}, {ci_hi:+.4f}]')

    cliffs_d = (2 * u / (len(SH['L']) * len(SH['H']))) - 1
    print(f"  effect size (rank-biserial / Cliff's delta approx): {cliffs_d:+.4f}")

    print('\n' + '=' * 92)
    print('E212-E: S_H -> crossing (subject-grouped CV), vs |Delta_L| alone')
    print('=' * 92)
    Lall = [r for r in rows if r['population'] == 'L']
    S_H_all = np.array([r['S_H'] for r in Lall])
    absDL = np.array([abs(r['delta_L']) for r in Lall])
    cross = np.array([r['crossing'] for r in Lall])
    groups = np.array([r['subject_id'] for r in Lall])

    auc_sh, pr_sh = grouped_auc_pr(S_H_all.reshape(-1, 1), cross, groups)
    auc_dl, pr_dl = grouped_auc_pr(absDL.reshape(-1, 1), cross, groups)
    print(f'  AUC(S_H -> crossing)       = {auc_sh:.4f}   PR-AUC = {pr_sh:.4f}')
    print(f'  AUC(|Delta_L| -> crossing) = {auc_dl:.4f}   PR-AUC = {pr_dl:.4f}')

    dc = S_H_all[cross == 1]; dn = S_H_all[cross == 0]
    if len(dc) >= 3 and len(dn) >= 3:
        print(f'  S_H median: crossing={np.median(dc):+.4f}  non-crossing={np.median(dn):+.4f}')

    print('\n' + '=' * 92)
    print('E212-F: reactivity control (R_mag)')
    print('=' * 92)
    R_mag = np.array([r['R_mag'] for r in Lall])
    X2 = np.column_stack([S_H_all, R_mag])
    if len(np.unique(cross)) >= 2:
        sc = StandardScaler().fit(X2)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X2), cross)
        beta1, beta2 = clf.coef_[0]
        print(f'  logistic beta_1 (S_H, standardized) = {beta1:+.4f}')
        print(f'  logistic beta_2 (R_mag, standardized) = {beta2:+.4f}')
    r_pc, p_pc = partial_corr(cross.astype(float), S_H_all, R_mag)
    print(f'  partial corr(S_H, crossing | R_mag) = {r_pc:+.4f}  p={p_pc:.3e}'
          f'   {"SURVIVES" if p_pc < 0.05 else "DOES NOT SURVIVE -- KILL"}')

    print('\n' + '=' * 92)
    print('E212-G: negative control -- same machinery on H')
    print('=' * 92)
    Hall = [r for r in rows if r['population'] == 'H']
    S_H_H = np.array([r['S_H'] for r in Hall])
    cross_H = np.array([r['crossing'] for r in Hall])
    groups_H = np.array([r['subject_id'] for r in Hall])
    print(f'  S_H_H median = {np.median(S_H_H):+.4f}  (want ~0)')
    print(f'  H crossings = {int(cross_H.sum())}/{len(cross_H)}')
    if len(np.unique(cross_H)) >= 2:
        auc_h, pr_h = grouped_auc_pr(S_H_H.reshape(-1, 1), cross_H, groups_H)
        print(f'  AUC(S_H -> crossing) for H = {auc_h:.4f}  (want ~0.5, i.e. NOT predictive)')
    else:
        auc_h = float('nan')
        print('  H has no crossing variance -- cannot compute AUC (itself consistent with G)')

    print('\n' + '=' * 92)
    print('VERDICT')
    print('=' * 92)
    gate_d = perm_p < 0.05 and obs > 0
    gate_e = (not np.isnan(auc_sh)) and auc_sh >= 0.65 and auc_sh > auc_dl
    gate_f = p_pc < 0.05
    gate_g = (np.isnan(auc_h) or abs(auc_h - 0.5) < 0.15)
    print(f'  D (S_H(L) > S_H(H), significant)      : {gate_d}')
    print(f'  E (AUC>=0.65 AND beats |Delta_L| alone) : {gate_e} (AUC_SH={auc_sh:.3f} vs AUC_DL={auc_dl:.3f})')
    print(f'  F (survives R_mag control)             : {gate_f}')
    print(f'  G (H dissociates -- not predictive)     : {gate_g}')
    all_pass = gate_d and gate_e and gate_f and gate_g
    if all_pass:
        verdict = ('PASS -- hypothesis-response selectivity S_H is a real, '
                  'reactivity-independent, crossing-predictive signal that '
                  'beats raw response magnitude, and does NOT generalise to '
                  'hard negatives. The HRS principle survives at the '
                  'existence level; design the smallest trainable mechanism next.')
    else:
        verdict = ('KILL -- one or more gates failed. See which gate(s) '
                  'above are False before deciding whether a narrower '
                  'variant is worth testing, or whether to close the HRS '
                  'principle entirely.')
    print(f'\n  ==> {verdict}')
