"""E244 Phase 1 analysis -- incremental differential test across the
D2->D1 stage sequence, G2-A only, per the user's exact spec:

  Delta_S_i = S(stage_i) - S(stage_{i-1})
  compare (Delta_S_i^miss - Delta_S_i^det) for each transition

The operation with the LARGEST NEGATIVE differential is the primary
target for Phase 2's replacement controls.
"""
import csv
import numpy as np
from scipy import stats
from collections import defaultdict

STAGE_ORDER = ['D2', 'S1_upconv1', 'S2_gate', 'S3_cat1', 'S4_dec1']
TRANSITIONS = list(zip(STAGE_ORDER[:-1], STAGE_ORDER[1:]))

rows = list(csv.DictReader(open('E244_stages.csv')))
for r in rows:
    r['pair_idx'] = int(r['pair_idx'])
    for s in STAGE_ORDER:
        r[s] = float(r[s])

by_pair = defaultdict(dict)
for r in rows:
    by_pair[r['pair_idx']][r['role']] = r

complete = [p for p, d in by_pair.items() if 'missed' in d and 'detected' in d]
print(f"Complete G2-A pairs: {len(complete)}")

print("\n" + "="*100)
print("ABSOLUTE SEPARABILITY AT EACH STAGE")
print("="*100)
for s in STAGE_ORDER:
    a = np.array([by_pair[p]['missed'][s] for p in complete])
    b = np.array([by_pair[p]['detected'][s] for p in complete])
    diff = a - b
    _, p_val = stats.wilcoxon(diff) if not np.all(diff == 0) else (0, 1)
    print(f"  {s:14s}: missed={a.mean():.4f}  detected={b.mean():.4f}  diff={diff.mean():+.4f}  p={p_val:.4e}")

print("\n" + "="*100)
print("INCREMENTAL DIFFERENTIAL PER TRANSITION: (Delta_S_missed - Delta_S_detected)")
print("="*100)
print(f"{'Transition':22s} {'DeltaS_miss':>12s} {'DeltaS_det':>12s} {'diff':>10s} {'Cohen_d':>10s} {'Wilcoxon_p':>12s}")
print("-"*100)

results = {}
for s_from, s_to in TRANSITIONS:
    dm = np.array([by_pair[p]['missed'][s_to] - by_pair[p]['missed'][s_from] for p in complete])
    dd = np.array([by_pair[p]['detected'][s_to] - by_pair[p]['detected'][s_from] for p in complete])
    diff = dm - dd
    pooled_std = np.sqrt((dm.var(ddof=1) + dd.var(ddof=1)) / 2)
    cohen_d = diff.mean() / pooled_std if pooled_std > 0 else float('nan')
    _, p_val = stats.wilcoxon(diff) if not np.all(diff == 0) else (0, 1)
    label = f"{s_from}->{s_to}"
    print(f"{label:22s} {dm.mean():12.4f} {dd.mean():12.4f} {diff.mean():+10.4f} {cohen_d:+10.4f} {p_val:12.4e}")
    results[label] = {'diff': diff.mean(), 'd': cohen_d, 'p': p_val}

print("\n" + "="*100)
print("TARGET IDENTIFICATION: largest NEGATIVE differential")
print("="*100)
sorted_results = sorted(results.items(), key=lambda kv: kv[1]['diff'])
for label, r in sorted_results:
    flag = ' <-- MOST NEGATIVE' if label == sorted_results[0][0] else ''
    sig = ' (p<0.05)' if r['p'] < 0.05 else ' (n.s.)'
    print(f"  {label}: diff={r['diff']:+.4f}  d={r['d']:+.4f}{sig}{flag}")

worst_label, worst = sorted_results[0]
print(f"\nPRIMARY TARGET: {worst_label} (diff={worst['diff']:+.4f}, d={worst['d']:+.4f}, p={worst['p']:.4e})")
if worst['p'] < 0.05:
    print("  This transition is BOTH the most negative AND statistically significant.")
    if worst_label == 'S1_upconv1->S2_gate':
        print("  ==> attn_gate1 IS the target. Proceed to Phase 2 replacement controls")
        print("      (bypass/constant/shuffled) to distinguish spatial-gating failure from")
        print("      mere magnitude change.")
    else:
        print(f"  ==> The target is the {worst_label} operation, NOT attn_gate1.")
        print("      Phase 2's attn_gate1-specific controls are NOT the right next step --")
        print("      would need controls specific to this operation instead.")
else:
    print("  This transition is the most negative but NOT statistically significant --")
    print("  no specific operation stands out. Per the decision tree: KILL the D1-operation")
    print("  hypothesis (no specific transformation identified).")
