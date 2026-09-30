"""E216 analysis -- oracle vs deployable steering, whole-volume ET Dice.

Headline number: ORACLE GAP = (oracle dDice) - (best GT-free dDice).
Reported per threshold (z>0, production z>-4.02) and alpha, over all
subjects and over the TAIL (subjects with >=1 missed component under z>0).
Paired Wilcoxon vs 'none'. Recovery = recovered / total missed components.
"""
import csv, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
fn = sys.argv[1] if len(sys.argv) > 1 else 'E216_steering.csv'
rows = list(csv.DictReader(open(HERE / fn)))
for r in rows:
    for k in ('alpha', 'thresh', 'dice'):
        r[k] = float(r[k])
    for k in ('n_missed', 'recovered', 'fp_comp', 'steer_vox'):
        r[k] = int(r[k])

base = {(r['subject_id'], r['thresh']): r for r in rows if r['arm'] == 'none'}
subs = sorted({r['subject_id'] for r in rows})
tail = sorted({r['subject_id'] for r in rows if r['n_missed'] > 0})
print(f'subjects={len(subs)}  tail(>=1 missed ET comp)={len(tail)}  '
      f'total missed comps={sum(base[(s, 0.0)]["n_missed"] for s in subs)}\n')

ARMS = ['oracle', 'deploy_unc', 'deploy_sh', 'random']
summary = {}
for th in sorted({r['thresh'] for r in rows}, reverse=True):
    print('=' * 100)
    print(f'THRESHOLD z > {th}')
    print('=' * 100)
    b_all = np.array([base[(s, th)]['dice'] for s in subs])
    b_tail = np.array([base[(s, th)]['dice'] for s in tail])
    fp0 = sum(base[(s, th)]['fp_comp'] for s in subs)
    rec0 = sum(base[(s, th)]['recovered'] for s in subs)
    tot0 = sum(base[(s, th)]['n_missed'] for s in subs)
    print(f"{'arm':<12}{'alpha':>6}{'Dice all':>10}{'dDice':>9}{'p':>10}"
          f"{'Dice tail':>11}{'dTail':>9}{'recov':>11}{'FPcomp':>9}")
    print(f"{'none':<12}{'':>6}{b_all.mean():>10.4f}{'':>9}{'':>10}"
          f"{b_tail.mean():>11.4f}{'':>9}{f'{rec0}/{tot0}':>11}{fp0:>9}")
    for arm in ARMS:
        for a in sorted({r['alpha'] for r in rows if r['arm'] == arm}):
            sel = {r['subject_id']: r for r in rows
                   if r['arm'] == arm and r['alpha'] == a and r['thresh'] == th}
            d_all = np.array([sel[s]['dice'] for s in subs])
            d_tail = np.array([sel[s]['dice'] for s in tail])
            diff = d_all - b_all
            p = stats.wilcoxon(diff)[1] if np.any(diff != 0) else 1.0
            rec = sum(sel[s]['recovered'] for s in subs)
            tot = sum(sel[s]['n_missed'] for s in subs)
            fp = sum(sel[s]['fp_comp'] for s in subs)
            summary[(th, arm, a)] = (diff.mean(), (d_tail - b_tail).mean(), rec, tot, fp)
            print(f"{arm:<12}{a:>6.2f}{d_all.mean():>10.4f}{diff.mean()*100:>+8.3f}p"
                  f"{p:>10.2e}{d_tail.mean():>11.4f}{(d_tail-b_tail).mean()*100:>+8.3f}p"
                  f"{f'{rec}/{tot}':>11}{fp:>9}")
    print()

print('=' * 100)
print('ORACLE GAP  (oracle dDice minus best GT-free dDice; pp, all subjects / tail)')
print('=' * 100)
for th in sorted({r['thresh'] for r in rows}, reverse=True):
    for a in sorted({r['alpha'] for r in rows if r['arm'] == 'oracle'}):
        o_all, o_tail, o_rec, tot, _ = summary[(th, 'oracle', a)]
        best = max(['deploy_unc', 'deploy_sh'], key=lambda k: summary[(th, k, a)][0])
        d_all, d_tail, d_rec, _, _ = summary[(th, best, a)]
        r_all = summary[(th, 'random', a)][0]
        print(f'  z>{th:<6} alpha={a:<4} oracle {o_all*100:+.3f}pp (rec {o_rec}/{tot})  '
              f'best GT-free [{best}] {d_all*100:+.3f}pp (rec {d_rec}/{tot})  '
              f'random {r_all*100:+.3f}pp  ->  GAP {(o_all-d_all)*100:+.3f}pp all, '
              f'{(o_tail-d_tail)*100:+.3f}pp tail')
