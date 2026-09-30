"""E219 analysis -- PREREGISTERED, SIGNED gates (written before full results).

Split-half protocol (200 subject-level splits, seed 0): each ARM chooses
(alpha, q) + per-region thresholds on train half (objective mean-of-3 Dice);
baseline ('none') chooses its own per-region thresholds on the same train
half; all scored on the test half.

GATES (all must hold, each with its predicted SIGN):
  G1  SELF: test Delta(mean3) > 0 in >= 95% of splits AND mean Delta(ET) > 0
  G2  SELF, modal chosen config, ET @ z>0: dTP > 0 AND dTP_missed/dTP >= 0.5
      (gain must come from MISSED lesions, not boundary growth -- E217 lesson)
  G3  SELF: mean Delta(WT) >= -0.2pp
  G4  SELF test mean3 > CROSS test mean3 in >= 95% of splits
  G5  SELF test mean3 > RANDOM test mean3 in >= 95% of splits
"""
import csv, sys
from pathlib import Path
from collections import Counter
import numpy as np

HERE = Path(__file__).resolve().parent
fn = sys.argv[1] if len(sys.argv) > 1 else 'E219_counts.csv'
if fn != 'E219_counts.csv':
    rows = list(csv.DictReader(open(HERE / fn)))
else:
    rows = []
    files = sorted(p for p in HERE.glob('E219_counts*.csv'))
    for p in files:
        rows += list(csv.DictReader(open(p)))
    # keep only subjects with exactly one COMPLETE block (195 rows); drop
    # partial blocks left by killed runs and any duplicate subject blocks
    from collections import defaultdict as _dd
    by = _dd(list)
    for r in rows:
        by[r['subject_id']].append(r)
    rows = []
    for s, rs in by.items():
        seen, uniq = set(), []
        for r in rs:
            k = (r['arm'], r['alpha'], r['q'], r['region'], r['thresh'])
            if k not in seen:
                seen.add(k); uniq.append(r)
        if len(uniq) == 195:
            rows += uniq
    print(f'merged {len(files)} files -> {len(rows)//195} complete subjects')
subs = sorted({r['subject_id'] for r in rows})
cfgs = sorted({(r['arm'], float(r['alpha']), float(r['q'])) for r in rows})
ths = sorted({float(r['thresh']) for r in rows})
S, C, R, T = len(subs), len(cfgs), 3, len(ths)
si = {s: i for i, s in enumerate(subs)}; ci = {c: i for i, c in enumerate(cfgs)}
ti = {x: i for i, x in enumerate(ths)}
D = np.full((S, C, R, T), np.nan)
extra = {}
ncore = {}
for r in rows:
    tp, fp, fnn = int(r['tp']), int(r['fp']), int(r['fn'])
    d = (1.0 if fp == 0 else 0.0) if tp + fnn == 0 else 2 * tp / (2 * tp + fp + fnn)
    c = (r['arm'], float(r['alpha']), float(r['q']))
    D[si[r['subject_id']], ci[c], int(r['region']), ti[float(r['thresh'])]] = d
    ncore[r['subject_id']] = int(r['n_core'])
    if r['et_n_missed'] != '':
        extra[(r['subject_id'], c, float(r['thresh']))] = (
            tp, fp, fnn, int(r['et_tp_missed']), int(r['et_n_missed']),
            int(r['et_recovered']), int(r['et_fp_comp']))
assert not np.isnan(D).any()
NONE = ci[('none', 0.0, 0.0)]
arm_cfgs = {a: [ci[c] for c in cfgs if c[0] == a] for a in ['self', 'cross', 'random']}
names = ['ET', 'TC', 'WT']
print(f'subjects={S}  no-core (fallback, unsteered) subjects='
      f'{sum(1 for s in subs if ncore[s] < 4)}\n')


def run_splits():
    rng = np.random.default_rng(0)
    res = {a: [] for a in arm_cfgs}; chosen = {a: [] for a in arm_cfgs}
    for _ in range(200):
        perm = rng.permutation(S); tr, te = perm[:S // 2], perm[S // 2:]
        bt = [int(np.argmax(D[tr, NONE, r, :].mean(0))) for r in range(R)]
        base = np.array([D[te, NONE, r, bt[r]].mean() for r in range(R)])
        for a, cl in arm_cfgs.items():
            best, bs = None, -1
            for c in cl:
                tt = [int(np.argmax(D[tr, c, r, :].mean(0))) for r in range(R)]
                sc = np.mean([D[tr, c, r, tt[r]].mean() for r in range(R)])
                if sc > bs:
                    bs, best = sc, (c, tt)
            c, tt = best
            res[a].append(np.array([D[te, c, r, tt[r]].mean() for r in range(R)]) - base)
            chosen[a].append(cfgs[c])
    return {a: np.array(v) * 100 for a, v in res.items()}, chosen


res, chosen = run_splits()
print('=' * 92)
print('SPLIT-HALF test-half Delta vs tuned baseline (pp), 200 splits')
print('=' * 92)
for a in ['self', 'cross', 'random']:
    m3 = res[a].mean(1)
    print(f'  {a.upper():<7} mean3 {m3.mean():+.3f} (frac>0 {np.mean(m3>0):.3f})  ' +
          '  '.join(f'{names[r]} {res[a][:, r].mean():+.3f}' for r in range(R)) +
          f'   chosen {Counter(chosen[a]).most_common(2)}')

m3s = res['self'].mean(1)
g1 = np.mean(m3s > 0) >= 0.95 and res['self'][:, 0].mean() > 0
g3 = res['self'][:, 2].mean() >= -0.2
g4 = np.mean(m3s > res['cross'].mean(1)) >= 0.95
g5 = np.mean(m3s > res['random'].mean(1)) >= 0.95

modal = Counter(chosen['self']).most_common(1)[0][0]
b = np.array([extra[(s, ('none', 0.0, 0.0), 0.0)] for s in subs])
st = np.array([extra[(s, modal, 0.0)] for s in subs])
dtp = st[:, 0].sum() - b[:, 0].sum()
dtpm = st[:, 3].sum() - b[:, 3].sum()
frac = dtpm / dtp if dtp > 0 else float('nan')
g2 = dtp > 0 and frac >= 0.5

print('\n' + '=' * 92)
print(f'G2 DECOMPOSITION, SELF modal config {modal}, ET @ z>0')
print('=' * 92)
print(f'  dTP={dtp:+d}  in missed comps {dtpm:+d} ({frac if dtp>0 else float("nan"):.2%})  '
      f'dFP={st[:,1].sum()-b[:,1].sum():+d}  recovered {b[:,5].sum()}->{st[:,5].sum()} '
      f'of {b[:,4].sum()}  FPcomp {b[:,6].sum()}->{st[:,6].sum()}')
for arm in ['cross', 'random']:
    c2 = (arm, modal[1], modal[2])
    s2 = np.array([extra[(s, c2, 0.0)] for s in subs])
    print(f'  [{arm} same cfg] dTP={s2[:,0].sum()-b[:,0].sum():+d} in-missed '
          f'{s2[:,3].sum()-b[:,3].sum():+d} recovered {s2[:,5].sum()} FPcomp {s2[:,6].sum()}')

print('\n' + '=' * 92)
print('VERDICT (signed gates)')
print('=' * 92)
for nm, v, desc in [('G1', g1, 'SELF mean3 >0 in >=95% splits & ET>0'),
                    ('G2', g2, 'ET gain positive & >=50% from missed lesions'),
                    ('G3', g3, 'WT >= -0.2pp'),
                    ('G4', g4, 'SELF > CROSS in >=95% splits'),
                    ('G5', g5, 'SELF > RANDOM in >=95% splits')]:
    print(f'  {nm} {"PASS" if v else "FAIL"}  -- {desc}')
print(f'\n  ==> {"PASS: Self-Prototype Steering survives all signed gates" if all([g1,g2,g3,g4,g5]) else "KILL"}')
