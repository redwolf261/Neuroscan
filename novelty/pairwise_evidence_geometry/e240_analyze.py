"""E240 analysis -- H2 rerun at adequate power. Paired Wilcoxon on
sep_{stage} between missed/detected roles, per stage, matching E234's
own original analysis exactly, now at n~500 pairs instead of E234's
n=78."""
import csv
import numpy as np
from scipy import stats
from collections import defaultdict

STAGES = ('raw', 'E1', 'E2', 'E3', 'E4_BN', 'D3', 'D2', 'D1')

rows = list(csv.DictReader(open('E240_layerwise.csv')))
for r in rows:
    r['pair_id'] = int(r['pair_id'])
    r['size'] = int(r['size'])
    for s in STAGES:
        r[f'sep_{s}'] = float(r[f'sep_{s}']) if r[f'sep_{s}'] != '' else None

by_pair = defaultdict(dict)
for r in rows:
    by_pair[r['pair_id']][r['role']] = r

complete = [p for p, d in by_pair.items() if 'missed' in d and 'detected' in d]
print(f"Complete pairs (n={len(complete)}) -- E234 original had n=78")

print("\n" + "="*100)
print(f"{'Stage':10s} {'missed_mean':>12s} {'det_mean':>12s} {'diff':>10s} {'Cohen_d':>10s} {'Wilcoxon_p':>12s} {'n':>6s}")
print("="*100)
for s in STAGES:
    key = f'sep_{s}'
    a, b = [], []
    for p in complete:
        va, vb = by_pair[p]['missed'][key], by_pair[p]['detected'][key]
        if va is None or vb is None:
            continue
        a.append(va); b.append(vb)
    a, b = np.array(a), np.array(b)
    diff = a - b
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    cohen_d = diff.mean() / pooled_std if pooled_std > 0 else float('nan')
    if np.all(diff == 0):
        p_val = 1.0
    else:
        _, p_val = stats.wilcoxon(diff)
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
    print(f"{s:10s} {a.mean():12.4f} {b.mean():12.4f} {diff.mean():+10.4f} {cohen_d:+10.4f} {p_val:12.4e} {len(a):6d} {sig}")

print("\n" + "="*100)
print("COMPARISON TO E234's ORIGINAL (n=78) RESULT")
print("="*100)
print("E234 original: NO stage significant (p=0.07-0.82 all 8 stages); D1 weak hint (d=0.17, needed ~275 pairs)")
print("E240 (this run, n~%d): see table above" % len(complete))
