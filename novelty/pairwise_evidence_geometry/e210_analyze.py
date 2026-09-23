"""E210 analysis -- PREREGISTERED, written before results were seen.

Does the ENDOGENOUS prototype hypothesis reproduce E209's crossing
phenomenon, beyond what a norm-matched RANDOM direction achieves?

KILL if prototype crossing rate is statistically indistinguishable from
random at every alpha (or a paired test shows no prototype-specific
advantage). PASS if prototype clears random at a meaningful alpha, ideally
with a magnitude near E209's own 24.1% crossing rate.
"""
import csv
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'E210_endogenous.csv')))
for r in rows:
    for k in list(r):
        if k not in ('subject_id', 'condition'):
            r[k] = float(r[k])
    r['crossing'] = int(r['crossing'])

ALPHAS = sorted({r['alpha'] for r in rows})
n_comp = len(rows) // 8
print(f'rows={len(rows)}  components={n_comp}  alphas={ALPHAS}\n')

print('=' * 90)
print('CROSSING RATE by condition x alpha')
print('=' * 90)
print(f"{'condition':<12}" + ''.join(f'{a:>10}' for a in ALPHAS))
for cond in ('prototype', 'random'):
    line = f'{cond:<12}'
    for a in ALPHAS:
        sel = [r for r in rows if r['condition'] == cond and r['alpha'] == a]
        rate = sum(r['crossing'] for r in sel) / len(sel)
        line += f'{rate*100:>9.1f}%'
    print(line)

print('\nFisher exact test, prototype vs random, per alpha:')
for a in ALPHAS:
    sel_p = [r['crossing'] for r in rows if r['condition'] == 'prototype' and r['alpha'] == a]
    sel_r = [r['crossing'] for r in rows if r['condition'] == 'random' and r['alpha'] == a]
    table = [[sum(sel_p), len(sel_p) - sum(sel_p)],
             [sum(sel_r), len(sel_r) - sum(sel_r)]]
    _, p = stats.fisher_exact(table)
    print(f'  alpha={a}: prototype {sum(sel_p)}/{len(sel_p)} vs random '
          f'{sum(sel_r)}/{len(sel_r)}   p={p:.3f}')

print('\n' + '=' * 90)
print('OVERALL (any alpha, per component): does EITHER condition cross at all?')
print('=' * 90)
comps = {}
for r in rows:
    key = (r['subject_id'], r['comp_id'])
    comps.setdefault(key, {'prototype': [], 'random': []})
    comps[key][r['condition']].append(r)
proto_any = sum(1 for c in comps.values() if any(r['crossing'] for r in c['prototype']))
rand_any = sum(1 for c in comps.values() if any(r['crossing'] for r in c['random']))
n = len(comps)
print(f'  prototype: {proto_any}/{n} components cross at SOME alpha ({100*proto_any/n:.1f}%)')
print(f'  random:    {rand_any}/{n} components cross at SOME alpha ({100*rand_any/n:.1f}%)')
print(f'  (E209 reference for comparison: 14/58 = 24.1% with real donor splicing)')

table = [[proto_any, n - proto_any], [rand_any, n - rand_any]]
_, p_any = stats.fisher_exact(table)
print(f'  Fisher exact (any-alpha crossing, prototype vs random): p={p_any:.3f}')

print('\n' + '=' * 90)
print('MAGNITUDE: z shift (region) by condition, at alpha=0.25 and 1.0')
print('=' * 90)
for a in (0.25, 1.0):
    print(f'  alpha={a}')
    dzs = {}
    for cond in ('prototype', 'random'):
        sel = [r for r in rows if r['condition'] == cond and r['alpha'] == a]
        dz = np.array([r['z_region_Y1'] - r['z_region_Y0'] for r in sel])
        dzs[cond] = dz
        print(f'    {cond:<12} median dz = {np.median(dz):+8.4f}  mean = {dz.mean():+8.4f}')
    pu = stats.mannwhitneyu(dzs['prototype'], dzs['random'])[1]
    print(f'    Mann-Whitney prototype vs random: p={pu:.3e}')

print('\n' + '=' * 90)
print('PAIRED: prototype crossing vs random crossing, SAME component, SAME alpha')
print('=' * 90)
for a in ALPHAS:
    sel_p = {(r['subject_id'], r['comp_id']): r['crossing']
             for r in rows if r['condition'] == 'prototype' and r['alpha'] == a}
    sel_r = {(r['subject_id'], r['comp_id']): r['crossing']
             for r in rows if r['condition'] == 'random' and r['alpha'] == a}
    keys = sorted(set(sel_p) & set(sel_r))
    p_only = sum(1 for k in keys if sel_p[k] == 1 and sel_r[k] == 0)
    r_only = sum(1 for k in keys if sel_p[k] == 0 and sel_r[k] == 1)
    both = sum(1 for k in keys if sel_p[k] == 1 and sel_r[k] == 1)
    neither = sum(1 for k in keys if sel_p[k] == 0 and sel_r[k] == 0)
    if p_only + r_only >= 3:
        pv = stats.binomtest(p_only, p_only + r_only, 0.5).pvalue
    else:
        pv = float('nan')
    print(f'  alpha={a}: prototype-only={p_only}  random-only={r_only}  '
          f'both={both}  neither={neither}  McNemar-style p={pv:.3f}')

print('\n' + '=' * 90)
print('VERDICT')
print('=' * 90)
best_a = max(ALPHAS, key=lambda a: sum(
    r['crossing'] for r in rows if r['condition'] == 'prototype' and r['alpha'] == a))
sel_p = [r['crossing'] for r in rows if r['condition'] == 'prototype' and r['alpha'] == best_a]
sel_r = [r['crossing'] for r in rows if r['condition'] == 'random' and r['alpha'] == best_a]
rate_p = sum(sel_p) / len(sel_p)
rate_r = sum(sel_r) / len(sel_r)
table = [[sum(sel_p), len(sel_p) - sum(sel_p)], [sum(sel_r), len(sel_r) - sum(sel_r)]]
_, p_best = stats.fisher_exact(table)
print(f'  best alpha for prototype = {best_a}: prototype={rate_p*100:.1f}% '
      f'vs random={rate_r*100:.1f}%   p={p_best:.3f}')
print(f'  overall any-alpha: prototype={100*proto_any/n:.1f}% vs '
      f'random={100*rand_any/n:.1f}%   p={p_any:.3f}')

specific = (p_best < 0.10 and rate_p > rate_r) or (p_any < 0.10 and proto_any > rand_any)
reproduces_e195 = (proto_any / n) > 0.15
if not specific:
    verdict = ('KILL -- endogenous prototype hypothesis is NOT distinguishable '
              'from a norm-matched random direction. The bottleneck locus does '
              'not support endogenous hypothesis construction; E209\'s '
              'phenomenon does not bridge to this representation-space '
              'formulation. Do not proceed to architecture design on this basis.')
elif not reproduces_e195:
    verdict = ('PARTIAL -- prototype beats random but crossing rate is far '
              'below E209\'s 24.1% reference; endogenous construction is '
              'directionally real but too weak to found an architecture on '
              'without further work characterizing why.')
else:
    verdict = ('PASS -- endogenous prototype hypothesis reproduces E209\'s '
              'phenomenon at a comparable rate, beating the random control. '
              'Bridge from forensic to differentiable mechanism holds.')
print(f'\n  ==> {verdict}')
