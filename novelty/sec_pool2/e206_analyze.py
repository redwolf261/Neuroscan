"""E206 analysis -- PREREGISTERED, written before results were seen.

Four required measurements (user's spec):
  1. DIRECTION    : dz_L(alpha=+small) > 0?
  2. DOSE-RESPONSE: dz_L(alpha) approximately monotonic?
  3. SPECIFICITY  : dz_L vs dz_S vs dz_H at matched alpha
  4. CROSSING     : how many L components move z<0 -> z>0 at ANY alpha?

Primary comparison: alpha=+0.25 (predicted-sign, moderate dose), with a
subject-preserving permutation control (200 iters) on dz_L vs dz_S there.

Interpretation ladder (user's spec):
  dz_L ~= 0                                -> K is diagnostic, not load-bearing. KILL.
  dz_L > 0 but tiny vs the ~18-logit deficit -> mechanistically real, insufficient.
  dz_L specific + monotonic + materially large -> earned right to test h'=h(1+aK).
"""
import csv
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'E206_inertness.csv')))
ALPHAS = [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]
for r in rows:
    for k in list(r):
        if k not in ('subject_id', 'population', 'comp_id'):
            r[k] = float(r[k])
POPS = ['L', 'S', 'H']
n = {p: sum(1 for r in rows if r['population'] == p) for p in POPS}
print(f'rows={len(rows)}  pops={n}  subjects={len({r["subject_id"] for r in rows})}\n')

print('=' * 92)
print('MEDIAN INTACT LOGIT by population (context for "materially large")')
print('=' * 92)
for p in POPS:
    v = [r['z_intact'] for r in rows if r['population'] == p]
    print(f'  {p}: median z_intact = {np.median(v):8.3f}   n={len(v)}')

print('\n' + '=' * 92)
print('1+2. DIRECTION & DOSE-RESPONSE: median dz(alpha) by population')
print('=' * 92)
print(f"{'alpha':>8}" + ''.join(f'{p:>12}' for p in POPS))
dz_by_pop_alpha = {p: {} for p in POPS}
for a in ALPHAS:
    line = f'{a:>+8.2f}'
    for p in POPS:
        v = np.array([r[f'z_a{a:+.2f}'] - r['z_intact']
                     for r in rows if r['population'] == p])
        dz_by_pop_alpha[p][a] = v
        line += f'{np.median(v):>12.4f}'
    print(line)

print('\n  Monotonicity check (L): is median dz_L(alpha) non-decreasing in alpha?')
medL = [np.median(dz_by_pop_alpha['L'][a]) for a in ALPHAS]
mono = all(medL[i] <= medL[i+1] + 1e-9 for i in range(len(medL)-1))
print(f'    {medL}')
print(f'    monotonic: {mono}')

print('\n' + '=' * 92)
print('3. SPECIFICITY at alpha=+0.25 (primary) and alpha=+0.50 (secondary)')
print('=' * 92)
for a in (0.25, 0.5):
    print(f'\n  alpha = {a:+.2f}')
    d = {p: dz_by_pop_alpha[p][a] for p in POPS}
    for p in POPS:
        print(f'    dz_{p}: median={np.median(d[p]):+.4f}  mean={d[p].mean():+.4f}'
              f'  n(dz>0)={int((d[p]>0).sum())}/{len(d[p])}')
    p_ls = stats.mannwhitneyu(d['L'], d['S'])[1]
    p_lh = stats.mannwhitneyu(d['L'], d['H'])[1]
    print(f'    Mann-Whitney L vs S: p={p_ls:.3e}   L vs H: p={p_lh:.3e}')

print('\n' + '=' * 92)
print('PERMUTATION CONTROL at alpha=+0.25 (subject-preserving, 200 iters)')
print('=' * 92)
LS = [r for r in rows if r['population'] in ('L', 'S')]
label = np.array([1.0 if r['population'] == 'L' else 0.0 for r in LS])
dzv = np.array([r['z_a+0.25'] - r['z_intact'] for r in LS])
by_sub = {}
for i, r in enumerate(LS):
    by_sub.setdefault(r['subject_id'], []).append(i)
obs = float(np.median(dzv[label == 1]) - np.median(dzv[label == 0]))
rng = np.random.default_rng(0)
null = []
for _ in range(200):
    lab2 = label.copy()
    for s, idxs in by_sub.items():
        sub = lab2[idxs].copy(); rng.shuffle(sub); lab2[idxs] = sub
    null.append(float(np.median(dzv[lab2 == 1]) - np.median(dzv[lab2 == 0])))
null = np.array(null)
pct = float((np.abs(null) >= abs(obs)).mean())
print(f'  observed median(dz_L) - median(dz_S) = {obs:+.4f}   perm p = {pct:.4f}')

print('\n' + '=' * 92)
print('4. CROSSINGS: L components with z_intact<0 that cross z>0 at ANY alpha')
print('=' * 92)
Lrows = [r for r in rows if r['population'] == 'L']
neg = [r for r in Lrows if r['z_intact'] < 0]
crossed = 0
for r in neg:
    if any(r[f'z_a{a:+.2f}'] > 0 for a in ALPHAS):
        crossed += 1
print(f'  {crossed}/{len(neg)} negative L components crossed at some alpha')
print('  (reference: E186.1 pool-arm P2 gave 1/58; E189 translation gave 5/57)')

print('\n' + '=' * 92)
print('VERDICT LADDER')
print('=' * 92)
best_a = max((0.1, 0.25, 0.5), key=lambda a: np.median(dz_by_pop_alpha['L'][a]))
best_med = np.median(dz_by_pop_alpha['L'][best_a])
med_intact_L = np.median([r['z_intact'] for r in Lrows])
frac_of_deficit = best_med / abs(med_intact_L) if med_intact_L != 0 else float('nan')
print(f'  best positive-alpha median dz_L = {best_med:+.4f}  (at alpha={best_a:+.2f})')
print(f'  as fraction of |median z_intact_L| ({med_intact_L:.2f}): {frac_of_deficit:.4f}'
      f'  ({frac_of_deficit*100:.2f}%)')
if abs(best_med) < 0.05:
    verdict = 'dz_L ~ 0 despite E205 existence signal -> K is DIAGNOSTIC, NOT LOAD-BEARING. KILL.'
elif frac_of_deficit < 0.05:
    verdict = 'dz_L > 0 but tiny relative to the ~18-logit deficit -> mechanistically real, PROBABLY INSUFFICIENT.'
else:
    verdict = 'dz_L specific + check monotonicity/crossings above -> may have earned the right to test h\'=h(1+alpha*K) properly.'
print(f'\n  ==> {verdict}')
