"""E220 analysis -- PREREGISTERED, SIGNED gates (written before full results).

200 subject-level split-halves (seed 0). Each ARM picks its config
(p1, p2) + per-region thresholds on the train half (objective mean-of-3
Dice); baseline picks its own thresholds on the same train half; all scored
on the test half.

  G1  PSPDS test Delta(mean3) > 0 in >= 95% splits AND mean Delta(ET) > 0
  G2  PSPDS test mean3 > BEST CONTROL test mean3 (max over interp, global,
      dilate, boost, each with its own train-half selection) in >= 95% splits
  G3  PSPDS modal config, ET @ z>0: dTP > 0 AND >= 50% of dTP in missed comps
  G4  PSPDS mean Delta(WT) >= -0.2pp
"""
import csv, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent
PER = 12 * 3 * 5
rows = []
for p in sorted(HERE.glob('E220_counts*.csv')):
    rows += list(csv.DictReader(open(p)))
if len(sys.argv) > 1:
    rows = list(csv.DictReader(open(HERE / sys.argv[1])))
by = defaultdict(list)
for r in rows:
    by[r['subject_id']].append(r)
rows = []
for s, rs in by.items():
    seen, u = set(), []
    for r in rs:
        k = (r['arm'], r['p1'], r['p2'], r['region'], r['thresh'])
        if k not in seen:
            seen.add(k); u.append(r)
    if len(u) == PER:
        rows += u
subs = sorted({r['subject_id'] for r in rows})
cfgs = sorted({(r['arm'], float(r['p1']), float(r['p2'])) for r in rows})
ths = sorted({float(r['thresh']) for r in rows})
S, C, R, T = len(subs), len(cfgs), 3, len(ths)
si = {s: i for i, s in enumerate(subs)}; ci = {c: i for i, c in enumerate(cfgs)}
ti = {x: i for i, x in enumerate(ths)}
D = np.full((S, C, R, T), np.nan); extra = {}; ncore = {}
for r in rows:
    tp, fp, fn = int(r['tp']), int(r['fp']), int(r['fn'])
    d = (1.0 if fp == 0 else 0.0) if tp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    c = (r['arm'], float(r['p1']), float(r['p2']))
    D[si[r['subject_id']], ci[c], int(r['region']), ti[float(r['thresh'])]] = d
    ncore[r['subject_id']] = int(r['n_core'])
    if r['et_n_missed'] != '':
        extra[(r['subject_id'], c, float(r['thresh']))] = tuple(
            int(r[k]) for k in ('tp', 'fp', 'fn', 'et_tp_missed', 'et_n_missed',
                                'et_recovered', 'et_fp_comp'))
assert not np.isnan(D).any()
NONE = ci[('none', 0.0, 0.0)]
ARMS = ['pspds', 'interp', 'global', 'dilate', 'boost']
arm_c = {a: [ci[c] for c in cfgs if c[0] == a] for a in ARMS}
names = ['ET', 'TC', 'WT']
print(f'complete subjects={S}  no-core={sum(1 for s in subs if ncore[s] < 4)}\n')

rng = np.random.default_rng(0)
res = {a: [] for a in ARMS}; chosen = {a: [] for a in ARMS}
for _ in range(200):
    perm = rng.permutation(S); tr, te = perm[:S // 2], perm[S // 2:]
    bt = [int(np.argmax(D[tr, NONE, r, :].mean(0))) for r in range(R)]
    base = np.array([D[te, NONE, r, bt[r]].mean() for r in range(R)])
    for a in ARMS:
        best, bs = None, -1
        for c in arm_c[a]:
            tt = [int(np.argmax(D[tr, c, r, :].mean(0))) for r in range(R)]
            sc = np.mean([D[tr, c, r, tt[r]].mean() for r in range(R)])
            if sc > bs:
                bs, best = sc, (c, tt)
        c, tt = best
        res[a].append(np.array([D[te, c, r, tt[r]].mean() for r in range(R)]) - base)
        chosen[a].append(cfgs[c])
res = {a: np.array(v) * 100 for a, v in res.items()}

print('=' * 96)
print('SPLIT-HALF test-half Delta vs tuned baseline (pp), 200 splits')
print('=' * 96)
for a in ARMS:
    m3 = res[a].mean(1)
    lo, hi = np.percentile(m3, [2.5, 97.5])
    print(f'  {a.upper():<7} mean3 {m3.mean():+.3f} [{lo:+.3f},{hi:+.3f}] frac>0 {np.mean(m3>0):.3f}  ' +
          ' '.join(f'{names[r]} {res[a][:, r].mean():+.3f}' for r in range(R)) +
          f'  chosen {Counter(chosen[a]).most_common(1)}')

m3p = res['pspds'].mean(1)
best_ctrl = np.max(np.stack([res[a].mean(1) for a in ARMS if a != 'pspds']), 0)
g1 = np.mean(m3p > 0) >= 0.95 and res['pspds'][:, 0].mean() > 0
g2 = np.mean(m3p > best_ctrl) >= 0.95
g4 = res['pspds'][:, 2].mean() >= -0.2
modal = Counter(chosen['pspds']).most_common(1)[0][0]
b = np.array([extra[(s, ('none', 0.0, 0.0), 0.0)] for s in subs])
st = np.array([extra[(s, modal, 0.0)] for s in subs])
dtp = st[:, 0].sum() - b[:, 0].sum(); dtpm = st[:, 3].sum() - b[:, 3].sum()
g3 = dtp > 0 and dtpm / dtp >= 0.5

print('\n' + '=' * 96)
print(f'ET DECOMPOSITION @ z>0 (modal PSPDS {modal}) vs controls at their modal configs')
print('=' * 96)
for a in ARMS:
    mc = Counter(chosen[a]).most_common(1)[0][0]
    s2 = np.array([extra[(s, mc, 0.0)] for s in subs])
    dt = s2[:, 0].sum() - b[:, 0].sum(); dm = s2[:, 3].sum() - b[:, 3].sum()
    print(f'  {a:<7} {str(mc):<26} dTP={dt:+7d} in-missed {dm:+6d} '
          f'({(dm/dt if dt>0 else float("nan")):.1%})  dFP={s2[:,1].sum()-b[:,1].sum():+7d}  '
          f'recovered {s2[:,5].sum()}/{b[:,4].sum()}  FPcomp {b[:,6].sum()}->{s2[:,6].sum()}')
print(f'  PSPDS beats best control in {np.mean(m3p > best_ctrl):.1%} of splits')

print('\n' + '=' * 96)
print('VERDICT (signed gates)')
print('=' * 96)
for nm, v, desc in [('G1', g1, 'PSPDS mean3>0 in >=95% splits & ET>0'),
                    ('G2', g2, 'PSPDS > best control in >=95% splits'),
                    ('G3', g3, 'ET gain >0 and >=50% from missed lesions'),
                    ('G4', g4, 'WT >= -0.2pp')]:
    print(f'  {nm} {"PASS" if v else "FAIL"}  -- {desc}')
print(f'\n  ==> {"PASS: PS-PDS survives all signed gates" if all([g1,g2,g3,g4]) else "KILL"}')
