"""E205 analysis -- PREREGISTERED, written before results were seen.

K(x) = ||grad h||^2 / (h^2 + eps), evaluated at the lesion-component centroid,
at E3 and bottleneck, for L/S/H populations.

GATE (all required to PASS):
  G1  K_L vs K_S significantly different (Mann-Whitney), both loci
  G2  survives partial correlation controlling for boundary_dist (kills the
      E179 confound: K must not just be re-deriving "how close to the
      lesion boundary this voxel is")
  G3  survives partial correlation controlling for h_mag_sq (K must not just
      be re-deriving "small activations = noisy region")
  G4  subject-preserving permutation null (200 iters, shuffle L/S labels
      WITHIN subject) confirms G1 is not a component-count artifact

KILL if G1 fails at either locus, or if G2/G3 partial correlation drops
below significance (p>0.05) -- that is the E179 death, one layer earlier.
"""
import csv
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'E205_kinetic.csv')))
for r in rows:
    for k in ('K', 'grad_sq', 'h_mag_sq', 'boundary_dist', 'vox'):
        r[k] = float(r[k])
POPS = ['L', 'S', 'H']
LOCI = ['E3', 'bottleneck']


def partial_corr(y, x, z):
    """corr(y,x) controlling for z, via residualization."""
    def resid(a, b):
        b1 = np.column_stack([np.ones_like(b), b])
        beta, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ beta
    ry = resid(y, z)
    rx = resid(x, z)
    r, p = stats.pearsonr(rx, ry)
    return r, p


for locus in LOCI:
    print('=' * 90)
    print(f'LOCUS: {locus}')
    print('=' * 90)
    sel = {p: [r for r in rows if r['population'] == p and r['locus'] == locus]
           for p in POPS}
    n = {p: len(sel[p]) for p in POPS}
    print(f'n = {n}')

    K = {p: np.array([r['K'] for r in sel[p]]) for p in POPS}
    for p in POPS:
        print(f'  K_{p}: median={np.median(K[p]):.4f}  mean={K[p].mean():.4f}')

    print('\n  G1: K_L vs K_S (Mann-Whitney)')
    u, p_ls = stats.mannwhitneyu(K['L'], K['S'])
    print(f'    p = {p_ls:.3e}   {"PASS" if p_ls < 0.05 else "FAIL"}')

    u2, p_lh = stats.mannwhitneyu(K['L'], K['H'])
    print(f'  (secondary) K_L vs K_H: p = {p_lh:.3e}')

    # pooled L+S for partial-correlation tests (label = 1 for L, 0 for S)
    LS = sel['L'] + sel['S']
    label = np.array([1.0 if r['population'] == 'L' else 0.0 for r in LS])
    Kv = np.array([r['K'] for r in LS])
    bdist = np.array([r['boundary_dist'] for r in LS])
    hmag = np.array([r['h_mag_sq'] for r in LS])

    print('\n  G2: does K survive controlling for boundary_dist (E179 confound)?')
    r_raw, p_raw = stats.pearsonr(Kv, label)
    r_pc, p_pc = partial_corr(label, Kv, bdist)
    print(f'    raw corr(K,label)             r={r_raw:+.4f}  p={p_raw:.3e}')
    print(f'    partial corr | boundary_dist  r={r_pc:+.4f}  p={p_pc:.3e}'
          f'   {"PASS" if p_pc < 0.05 else "FAIL (E179 confound reproduces)"}')

    print('\n  G3: does K survive controlling for h_mag_sq (magnitude confound)?')
    r_pc2, p_pc2 = partial_corr(label, Kv, hmag)
    print(f'    partial corr | h_mag_sq       r={r_pc2:+.4f}  p={p_pc2:.3e}'
          f'   {"PASS" if p_pc2 < 0.05 else "FAIL (magnitude confound)"}')

    print('\n  G4: subject-preserving permutation null (200 iters)')
    subs = sorted({r['subject_id'] for r in LS})
    by_sub = {}
    for i, r in enumerate(LS):
        by_sub.setdefault(r['subject_id'], []).append(i)
    obs_stat = float(np.median(Kv[label == 1]) - np.median(Kv[label == 0]))
    rng = np.random.default_rng(0)
    null = []
    for _ in range(200):
        lab2 = label.copy()
        for s, idxs in by_sub.items():
            sub_lab = lab2[idxs].copy()
            rng.shuffle(sub_lab)
            lab2[idxs] = sub_lab
        null.append(float(np.median(Kv[lab2 == 1]) - np.median(Kv[lab2 == 0])))
    null = np.array(null)
    pct = float((np.abs(null) >= abs(obs_stat)).mean())
    print(f'    observed median diff = {obs_stat:+.4f}   perm p = {pct:.4f}'
          f'   {"PASS" if pct < 0.05 else "FAIL"}')

    g1 = p_ls < 0.05
    g2 = p_pc < 0.05
    g3 = p_pc2 < 0.05
    g4 = pct < 0.05
    verdict = g1 and g2 and g3 and g4
    print(f'\n  {locus} VERDICT: G1={g1} G2={g2} G3={g3} G4={g4}  ==> '
          f'{"PASS -- proceed toward h\' = h(1+alpha*K)" if verdict else "FAIL -- KILL at this locus"}')
    print()

print('=' * 90)
print('OVERALL')
print('=' * 90)
print('Both loci must PASS for the candidate to survive. If E3 fails but')
print('bottleneck passes (or vice versa), report as split result, not PASS.')
