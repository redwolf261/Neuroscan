"""E208 analysis -- PREREGISTERED, written before results were seen.

Delta_E = z_shell_Y1 - z_shell_Y0 : does splicing hypothesis-consistent
  (donor tumor) texture into a candidate region shift the SURROUNDING
  shell's own prediction toward "this looks lesion-adjacent"?
reactivity = |z_region_Y1 - z_region_Y0| : how much does the region's OWN
  prediction move at all (the E107 flatness check).

PRIMARY: does Delta_E discriminate L from H?
E107 CHECK: is L's reactivity much smaller than S's (flatness reproducing)?
DECISIVE: does Delta_E survive partial correlation controlling for
  reactivity? If not, Delta_E is just "L is flat, so nothing moves anywhere"
  restated, not evidence of hypothesis-value discrimination.
"""
import csv
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'E208_hrs.csv')))
for r in rows:
    for k in list(r):
        if k not in ('subject_id', 'population', 'comp_id'):
            r[k] = float(r[k])
POPS = ['L', 'S', 'H']
n = {p: sum(1 for r in rows if r['population'] == p) for p in POPS}
print(f'rows={len(rows)}  pops={n}  subjects={len({r["subject_id"] for r in rows})}\n')

print('=' * 88)
print('DELTA_E (shell shift after hypothesis insertion) by population')
print('=' * 88)
DE = {p: np.array([r['delta_E'] for r in rows if r['population'] == p]) for p in POPS}
for p in POPS:
    print(f'  {p}: median={np.median(DE[p]):+.4f}  mean={DE[p].mean():+.4f}'
          f'  n(>0)={int((DE[p]>0).sum())}/{len(DE[p])}')

p_lh = stats.mannwhitneyu(DE['L'], DE['H'])[1]
p_ls = stats.mannwhitneyu(DE['L'], DE['S'])[1]
print(f'\n  Mann-Whitney L vs H: p={p_lh:.3e}   L vs S: p={p_ls:.3e}')

print('\n' + '=' * 88)
print('REACTIVITY (E107 flatness check) by population')
print('=' * 88)
RE = {p: np.array([r['reactivity'] for r in rows if r['population'] == p]) for p in POPS}
for p in POPS:
    print(f'  {p}: median={np.median(RE[p]):.4f}  mean={RE[p].mean():.4f}')
p_re_lh = stats.mannwhitneyu(RE['L'], RE['H'])[1]
p_re_ls = stats.mannwhitneyu(RE['L'], RE['S'])[1]
print(f'\n  Mann-Whitney reactivity L vs H: p={p_re_lh:.3e}   L vs S: p={p_re_ls:.3e}')
ratio = np.median(RE['L']) / (np.median(RE['S']) + 1e-9)
print(f'  median reactivity L / median reactivity S = {ratio:.4f}')
print(f'  (E107 reference: FN/TP logit-space ratio was 8.75/20.53 = 0.426)')

print('\n' + '=' * 88)
print('DECISIVE: does Delta_E survive controlling for reactivity? (L vs H)')
print('=' * 88)
LH = [r for r in rows if r['population'] in ('L', 'H')]
label = np.array([1.0 if r['population'] == 'L' else 0.0 for r in LH])
de = np.array([r['delta_E'] for r in LH])
re_ = np.array([r['reactivity'] for r in LH])

r_raw, p_raw = stats.pointbiserialr(label, de)
print(f'  raw corr(delta_E, label)        r={r_raw:+.4f}  p={p_raw:.3e}')


def partial_corr(y, x, z):
    def resid(a, b):
        b1 = np.column_stack([np.ones_like(b), b])
        beta, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ beta
    ry = resid(y, z); rx = resid(x, z)
    return stats.pearsonr(rx, ry)


r_pc, p_pc = partial_corr(label, de, re_)
print(f'  partial corr | reactivity       r={r_pc:+.4f}  p={p_pc:.3e}'
      f'   {"SURVIVES" if p_pc < 0.05 else "DOES NOT SURVIVE"}')

print('\n' + '=' * 88)
print('PERMUTATION CONTROL (subject-preserving, 200 iters) on Delta_E, L vs H')
print('=' * 88)
by_sub = {}
for i, r in enumerate(LH):
    by_sub.setdefault(r['subject_id'], []).append(i)
obs = float(np.median(de[label == 1]) - np.median(de[label == 0]))
rng = np.random.default_rng(0)
null = []
for _ in range(200):
    lab2 = label.copy()
    for s, idxs in by_sub.items():
        sub = lab2[idxs].copy(); rng.shuffle(sub); lab2[idxs] = sub
    null.append(float(np.median(de[lab2 == 1]) - np.median(de[lab2 == 0])))
null = np.array(null)
pct = float((np.abs(null) >= abs(obs)).mean())
print(f'  observed median(delta_E_L)-median(delta_E_H) = {obs:+.4f}   perm p = {pct:.4f}')

print('\n' + '=' * 88)
print('VERDICT')
print('=' * 88)
sig_raw = p_lh < 0.05
sig_perm = pct < 0.05
survives_reactivity = p_pc < 0.05
flatness = np.median(RE['L']) < 0.5 * np.median(RE['S'])
print(f'  Delta_E discriminates L from H (raw)         : {sig_raw}')
print(f'  Delta_E discriminates L from H (permutation) : {sig_perm}')
print(f'  Delta_E survives reactivity control           : {survives_reactivity}')
print(f'  E107 flatness reproduces (L reactivity << S)  : {flatness}')

if not (sig_raw and sig_perm):
    verdict = 'KILL -- Delta_E does not discriminate L from H at all.'
elif not survives_reactivity:
    verdict = ('KILL (explained) -- Delta_E discriminates L from H only because '
              'L is broadly unreactive (E107 flatness), not because hypothesis-'
              'insertion carries specific value information.')
elif flatness:
    verdict = ('PASS, but flatness ALSO present -- Delta_E adds signal beyond '
              'reactivity even though L is less reactive overall; worth a second '
              'falsification pass before building anything.')
else:
    verdict = 'PASS -- Delta_E is a real, reactivity-independent discriminator.'
print(f'\n  ==> {verdict}')
