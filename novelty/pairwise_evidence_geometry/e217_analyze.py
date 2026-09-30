"""E217 analysis -- PREREGISTERED (written before full results were seen).

Dice per subject/region uses dice_per_region's convention (empty GT -> 1 if
empty prediction else 0).

PRIMARY: repeated subject-level split-half (200 splits, seed 0).
  Steering picks one steered config (alpha, pct) + per-region thresholds on
  the train half (objective: mean of ET/TC/WT Dice). Baseline picks its own
  per-region thresholds on the SAME train half. Both scored on test half.
  Reported: mean / 95% interval / fraction>0 of test-half Delta, for
  mean-of-3 and each region.
  PASS: mean Delta(mean3) > 0 with >= 95% of splits positive AND no region's
  mean Delta below -0.2pp.

SECONDARY: same with ET-only objective; fixed E216 config (a=0.5, p99)
threshold sweep for all regions; ET gain decomposition.
"""
import csv, sys
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
fn = sys.argv[1] if len(sys.argv) > 1 else 'E217_counts.csv'
rows = list(csv.DictReader(open(HERE / fn)))

subs = sorted({r['subject_id'] for r in rows})
cfgs = sorted({(float(r['alpha']), float(r['pct'])) for r in rows})
ths = sorted({float(r['thresh']) for r in rows})
S, C, R, T = len(subs), len(cfgs), 3, len(ths)
si = {s: i for i, s in enumerate(subs)}; ci = {c: i for i, c in enumerate(cfgs)}
ti = {x: i for i, x in enumerate(ths)}
D = np.full((S, C, R, T), np.nan)
extra = {}
for r in rows:
    tp, fp, fnn = int(r['tp']), int(r['fp']), int(r['fn'])
    d = (1.0 if fp == 0 else 0.0) if tp + fnn == 0 else 2 * tp / (2 * tp + fp + fnn)
    c = (float(r['alpha']), float(r['pct']))
    D[si[r['subject_id']], ci[c], int(r['region']), ti[float(r['thresh'])]] = d
    if r['et_n_missed'] != '':
        extra[(r['subject_id'], c, float(r['thresh']))] = (
            tp, fp, fnn, int(r['et_tp_missed']), int(r['et_n_missed']),
            int(r['et_recovered']), int(r['et_fp_comp']))
assert not np.isnan(D).any(), 'missing cells'
NONE = ci[(0.0, 0.0)]
steered = [i for i in range(C) if i != NONE]
names = ['ET', 'TC', 'WT']
print(f'subjects={S} configs={C} thresholds={ths}\n')


def select_eval(train, test, objective_regions):
    # baseline: per-region best threshold on train
    bt = [int(np.argmax(D[train, NONE, r, :].mean(0))) for r in range(R)]
    base_test = np.array([D[test, NONE, r, bt[r]].mean() for r in range(R)])
    best, bscore = None, -1
    for c in steered:
        tt = [int(np.argmax(D[train, c, r, :].mean(0))) for r in range(R)]
        score = np.mean([D[train, c, r, tt[r]].mean() for r in objective_regions])
        if score > bscore:
            bscore, best = score, (c, tt)
    c, tt = best
    st_test = np.array([D[test, c, r, tt[r]].mean() for r in range(R)])
    return st_test - base_test, c


for label, obj in [('PRIMARY (mean-of-3 objective)', [0, 1, 2]),
                   ('SECONDARY (ET-only objective)', [0])]:
    rng = np.random.default_rng(0)
    deltas, chosen = [], []
    for _ in range(200):
        perm = rng.permutation(S)
        train, test = perm[:S // 2], perm[S // 2:]
        dlt, c = select_eval(train, test, obj)
        deltas.append(dlt); chosen.append(cfgs[c])
    deltas = np.array(deltas) * 100
    m3 = deltas.mean(1)
    print('=' * 92)
    print(f'{label} -- 200 subject-level split-halves, test-half Delta (pp)')
    print('=' * 92)
    for nm, v in [('mean3', m3)] + [(names[r], deltas[:, r]) for r in range(R)]:
        lo, hi = np.percentile(v, [2.5, 97.5])
        print(f'  {nm:<6} mean {v.mean():+.3f}  95%int [{lo:+.3f}, {hi:+.3f}]  '
              f'frac>0 {np.mean(v > 0):.3f}')
    from collections import Counter
    print('  chosen configs:', Counter(chosen).most_common(4))
    if obj == [0, 1, 2]:
        ok = (m3.mean() > 0 and np.mean(m3 > 0) >= 0.95
              and all(deltas[:, r].mean() > -0.2 for r in range(R)))
        print(f'\n  PRIMARY VERDICT: {"PASS" if ok else "FAIL"}\n')

print('=' * 92)
print('FIXED E216 CONFIG (alpha=0.5, top-1% s_h): Delta vs none by region x threshold (pp)')
print('=' * 92)
cw = ci[(0.5, 99.0)]
print(f"{'thresh':>8}" + ''.join(f'{n:>22}' for n in names))
for x in ths:
    line = f'{x:>8}'
    for r in range(R):
        d = D[:, cw, r, ti[x]] - D[:, NONE, r, ti[x]]
        p = stats.wilcoxon(d)[1] if np.any(d != 0) else 1.0
        line += f'{d.mean()*100:>+12.3f} (p={p:.1e})'
    print(line)

print('\n' + '=' * 92)
print('ET GAIN DECOMPOSITION, fixed config (alpha=0.5, p99), summed over subjects')
print('=' * 92)
for x in [0.0, -4.02]:
    b = np.array([extra[(s, (0.0, 0.0), x)] for s in subs])
    st = np.array([extra[(s, (0.5, 99.0), x)] for s in subs])
    dtp = st[:, 0].sum() - b[:, 0].sum()
    dtp_m = st[:, 3].sum() - b[:, 3].sum()
    print(f'  z>{x}: dTP={dtp:+d} (in missed comps {dtp_m:+d}, in detected/other GT '
          f'{dtp-dtp_m:+d})  dFP={st[:,1].sum()-b[:,1].sum():+d}  '
          f'dFN={st[:,2].sum()-b[:,2].sum():+d}  recovered {b[:,5].sum()}->{st[:,5].sum()} '
          f'of {b[:,4].sum()}  FPcomp {b[:,6].sum()}->{st[:,6].sum()}')
