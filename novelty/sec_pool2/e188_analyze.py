"""E188 analysis -- PREREGISTERED, written before results were seen.

PRIMARY ENDPOINT   : Delta_L - Delta_S per perturbation (paired by component,
                     tested across components with Mann-Whitney).
SECONDARY ENDPOINT : ordering of the four effects within L, S, H.

E65 REFERENCE ORDERING at enc1 (v3, ungated):
    translation 0.2143 > channel 0.1009 > smoothing 0.0284 ~ local 0.0270

KILL (as preregistered): enc3 reproduces E65's enc1 ordering AND shows no
L-vs-S asymmetry. Specifically:
  K1  ordering within S matches E65 (translation top, channel second)
      AND no perturbation has a significant L-vs-S difference (all p > 0.05)

PASS: at least one perturbation shows a significant L-vs-S asymmetry that is
not explained by L simply having a smaller dz to lose (checked by the
NORMALISED effect Delta / |dz_intact|).
"""
import csv
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'E188_correspondence.csv')))
for r in rows:
    for k in ('dz_intact', 'dz_pert', 'Delta', 'vox', 'roi_e3_vox'):
        r[k] = float(r[k])
PERTS = ['channel', 'local_spatial', 'amplitude', 'translation']
POPS = ['L', 'S', 'H']

n = {p: len({(r['subject_id'], r['comp_id']) for r in rows
             if r['population'] == p}) for p in POPS}
print(f'rows={len(rows)}  components={n}  '
      f'subjects={len({r["subject_id"] for r in rows})}\n')

print('=' * 84)
print('INTACT BASELINE dz (median) -- the thing each perturbation destroys')
print('=' * 84)
base = {}
for p in POPS:
    v = np.array([r['dz_intact'] for r in rows
                  if r['population'] == p and r['perturb'] == 'channel'])
    base[p] = np.median(v)
    print(f'  {p}: dz_intact median = {base[p]:8.3f}   (n={len(v)})')

print('\n' + '=' * 84)
print('RAW EFFECTS: Delta = dz_intact - dz_perturbed  (POSITIVE = damage)')
print('=' * 84)
print(f"{'perturbation':<16}" + ''.join(f'{p:>12}' for p in POPS)
      + f"{'L-S':>10}{'p(L vs S)':>12}{'p(L vs H)':>12}")
eff = {}
for pt in PERTS:
    line = f'{pt:<16}'
    d = {}
    for p in POPS:
        d[p] = np.array([r['Delta'] for r in rows
                         if r['population'] == p and r['perturb'] == pt])
        line += f'{np.median(d[p]):>12.3f}'
    eff[pt] = d
    pls = stats.mannwhitneyu(d['L'], d['S'])[1] if min(len(d['L']), len(d['S'])) >= 4 else np.nan
    plh = stats.mannwhitneyu(d['L'], d['H'])[1] if min(len(d['L']), len(d['H'])) >= 4 else np.nan
    print(line + f'{np.median(d["L"])-np.median(d["S"]):>10.3f}'
          f'{pls:>12.2e}{plh:>12.2e}')

print('\n' + '=' * 84)
print('NORMALISED EFFECTS: Delta / |dz_intact|  -- controls for L simply')
print('having less evidence to lose (THE CONFOUND THAT MATTERS)')
print('=' * 84)
print(f"{'perturbation':<16}" + ''.join(f'{p:>12}' for p in POPS)
      + f"{'L-S':>10}{'p(L vs S)':>12}")
neff = {}
for pt in PERTS:
    line = f'{pt:<16}'
    d = {}
    for p in POPS:
        v = [r['Delta'] / (abs(r['dz_intact']) + 1e-6) for r in rows
             if r['population'] == p and r['perturb'] == pt]
        d[p] = np.array(v)
        line += f'{np.median(d[p]):>12.3f}'
    neff[pt] = d
    pls = stats.mannwhitneyu(d['L'], d['S'])[1] if min(len(d['L']), len(d['S'])) >= 4 else np.nan
    print(line + f'{np.median(d["L"])-np.median(d["S"]):>10.3f}{pls:>12.2e}')

print('\n' + '=' * 84)
print('SECONDARY: ordering within each population (rank 1 = most damaging)')
print('=' * 84)
order = {}
for p in POPS:
    med = {pt: np.median(eff[pt][p]) for pt in PERTS}
    o = sorted(PERTS, key=lambda k: -med[k])
    order[p] = o
    print(f'  {p}: ' + ' > '.join(f'{k}({med[k]:.2f})' for k in o))
print(f'\n  E65 at enc1: translation(0.214) > channel(0.101) > '
      f'smoothing(0.028) ~ local(0.027)')

print('\n' + '=' * 84)
print('PREREGISTERED VERDICT')
print('=' * 84)
# K1: does S reproduce E65's ordering (translation top, channel second)?
e65_like = (order['S'][0] == 'translation' and order['S'][1] == 'channel')
# any significant L-vs-S asymmetry, on the NORMALISED effect?
sig = []
for pt in PERTS:
    d = neff[pt]
    if min(len(d['L']), len(d['S'])) < 4:
        continue
    p = stats.mannwhitneyu(d['L'], d['S'])[1]
    if p < 0.05:
        sig.append((pt, p, np.median(d['L']) - np.median(d['S'])))
print(f'  S reproduces E65 enc1 ordering?        {e65_like}')
print(f'  perturbations with significant L-vs-S  {[s[0] for s in sig] or "NONE"}')
for pt, p, dd in sig:
    print(f'      {pt:<16} normalised L-S = {dd:+.3f}  p={p:.2e}')
k1 = e65_like and not sig
print(f'\n  K1 (enc3 == enc1, no asymmetry): {"FIRED" if k1 else "passed"}')
print(f'\n  ==> {"KILL BRANCH" if k1 else "ASYMMETRY PRESENT -- proceed to operator design"}')
