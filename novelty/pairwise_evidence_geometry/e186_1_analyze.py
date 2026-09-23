"""E186.1 analysis -- PREREGISTERED, written before results were seen.

Primary quantity: dz = median(z_component) - median(z_shell), the local ET
evidence contrast. Effect of a perturbation = dz(Pk) - dz(P0), paired within
the same (subject, component, population).

PRIMARY ARM = 'pool'. That is the only arm that isolates the stated hypothesis
(spatial arrangement of E3 as consumed by pool3 -> bottleneck). The 'skip' and
'both' arms are reported as decomposition, not as the test.

KILL CRITERIA (as preregistered by the user):
  K1  no causal effect on L                      -> |median effect| < 0.1 or p > 0.05
  K2  L, S, H all respond similarly              -> L-vs-S and L-vs-H both n.s.
  K5  effect smaller than 0.1pp-equivalent       -> handled via effect size

TARGET PATTERN: L: P0 > P1 > P2 > P3 (monotone damage), while S, H flat.
"""
import csv
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'E186_1_causal.csv')))
for r in rows:
    for k in ('dz', 'z_comp', 'z_shell', 'z_comp_mean', 'frac_pos', 'dmu_max', 'dsd_max'):
        r[k] = float(r[k])
POPS, PKS = ['L', 'S', 'H'], ['P1', 'P2', 'P3']

print(f'rows={len(rows)}  subjects={len({r["subject_id"] for r in rows})}')

# --- integrity: permutation must preserve activation statistics ---
dm = max((r['dmu_max'] for r in rows if r['perturb'] != 'P0'), default=0)
dsd = max((r['dsd_max'] for r in rows if r['perturb'] != 'P0'), default=0)
print(f'INTEGRITY  max|dmean|={dm:.2e}  max|dstd|={dsd:.2e}   '
      f'(must be ~1e-6; permutation preserves the multiset)')

base = {(r['subject_id'], r['comp_id'], r['population']): r['dz']
        for r in rows if r['perturb'] == 'P0'}
n_units = {p: len({k for k in base if k[2] == p}) for p in POPS}
print(f'paired units: {n_units}\n')


def paired(arm, pop, pk):
    out = []
    for r in rows:
        if r['arm'] != arm or r['population'] != pop or r['perturb'] != pk:
            continue
        k = (r['subject_id'], r['comp_id'], r['population'])
        if k in base:
            out.append(r['dz'] - base[k])
    return np.array(out)


for arm in ['pool', 'skip', 'both']:
    tag = '  <<< PRIMARY' if arm == 'pool' else ''
    print('=' * 84)
    print(f'ARM = {arm}{tag}')
    print('=' * 84)
    print(f"{'pop':<5}{'n':>5}{'dz(P0)':>10}" +
          ''.join(f'{"d"+k:>12}' for k in PKS) + f"{'p(P2)':>10}")
    eff = {}
    for pop in POPS:
        b = np.array([v for k, v in base.items() if k[2] == pop])
        line = f'{pop:<5}{len(b):>5}{np.median(b):>10.3f}'
        for pk in PKS:
            d = paired(arm, pop, pk)
            eff[(pop, pk)] = d
            line += f'{np.median(d):>12.3f}' if len(d) else f'{"n/a":>12}'
        d2 = eff[(pop, 'P2')]
        p = stats.wilcoxon(d2)[1] if len(d2) >= 6 and np.any(d2 != 0) else np.nan
        print(line + f'{p:>10.2e}')

    # population specificity: is L damaged MORE than S and H?
    print('\n  specificity (Mann-Whitney on the paired effect, L vs other):')
    for pk in PKS:
        dL = eff[('L', pk)]
        for other in ['S', 'H']:
            do = eff[(other, pk)]
            if len(dL) < 4 or len(do) < 4:
                continue
            u = stats.mannwhitneyu(dL, do, alternative='two-sided')[1]
            print(f'    {pk}  L({np.median(dL):+.3f}, n={len(dL)}) vs '
                  f'{other}({np.median(do):+.3f}, n={len(do)})   p={u:.2e}')
    # monotonicity within L
    dm_ = [np.median(eff[('L', k)]) for k in PKS if len(eff[('L', k)])]
    if len(dm_) == 3:
        mono = dm_[0] >= dm_[1] >= dm_[2]
        print(f'\n  L monotone damage P1>=P2>=P3?  {mono}   ({dm_[0]:+.3f} '
              f'{dm_[1]:+.3f} {dm_[2]:+.3f})')
    print()

# ---------------- VERDICT on the primary arm ----------------
print('=' * 84)
print('PREREGISTERED VERDICT (primary arm = pool)')
print('=' * 84)
dL = paired('pool', 'L', 'P2')
dS = paired('pool', 'S', 'P2')
dH = paired('pool', 'H', 'P2')
k1 = (len(dL) < 6 or abs(np.median(dL)) < 0.1 or
      stats.wilcoxon(dL)[1] > 0.05)
pLS = stats.mannwhitneyu(dL, dS)[1] if min(len(dL), len(dS)) >= 4 else 1.0
pLH = stats.mannwhitneyu(dL, dH)[1] if min(len(dL), len(dH)) >= 4 else 1.0
k2 = (pLS > 0.05 and pLH > 0.05)
print(f'  K1 no causal effect on L       : {"FIRED" if k1 else "passed"}'
      f'   (median {np.median(dL):+.3f}, n={len(dL)})')
print(f'  K2 effect not lesion-specific  : {"FIRED" if k2 else "passed"}'
      f'   (p_LS={pLS:.2e}, p_LH={pLH:.2e})')
print(f'\n  ==> {"KILL" if (k1 or k2) else "SURVIVES to E186.2"}')
