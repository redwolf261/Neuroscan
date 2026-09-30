"""E245 analysis -- internal dec1 decomposition, G2-A only. Per explicit
user decision: BN transitions (conv->bn) are EXCLUDED from deficit-
localization interpretation, since E234's separability statistic is
invariant to BatchNorm's own per-channel affine transform (verified via
direct inspection during Phase 1 -- conv_out and bn_out give near-
identical separability values by measurement construction, not because
BN has no effect). Meaningful transitions: cat1->conv1 (feature mixing/
channel projection), conv1->relu1 (BN+ReLU combined -- nonlinear
suppression, can't isolate BN's own share), relu1->conv2 (second feature
mixing), conv2->relu2 (second BN+ReLU combined).
"""
import csv
import numpy as np
from scipy import stats
from collections import defaultdict

STAGE_ORDER = ['cat1', 'conv1_out', 'bn1_out', 'relu1_out', 'conv2_out', 'bn2_out', 'relu2_out']
MEANINGFUL_TRANSITIONS = [
    ('cat1', 'conv1_out'),      # feature mixing / channel projection (block 1)
    ('conv1_out', 'relu1_out'), # BN+ReLU combined -- nonlinear suppression
    ('relu1_out', 'conv2_out'), # feature mixing / channel projection (block 2)
    ('conv2_out', 'relu2_out'), # BN+ReLU combined -- nonlinear suppression
]
EXCLUDED_TRANSITIONS = [('conv1_out', 'bn1_out'), ('conv2_out', 'bn2_out')]

rows = list(csv.DictReader(open('E245_dec1_internal.csv')))
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
print("ABSOLUTE SEPARABILITY AT EACH INTERNAL STAGE")
print("="*100)
for s in STAGE_ORDER:
    a = np.array([by_pair[p]['missed'][s] for p in complete])
    b = np.array([by_pair[p]['detected'][s] for p in complete])
    diff = a - b
    _, p_val = stats.wilcoxon(diff) if not np.all(diff == 0) else (0, 1)
    excl = ' [BN stage, invariant statistic]' if s in ('bn1_out', 'bn2_out') else ''
    print(f"  {s:12s}: missed={a.mean():.4f}  detected={b.mean():.4f}  diff={diff.mean():+.4f}  p={p_val:.4e}{excl}")

print("\n" + "="*100)
print("MEANINGFUL TRANSITIONS ONLY (BN-transitions excluded, per user decision)")
print("="*100)
print(f"{'Transition':28s} {'DeltaS_miss':>12s} {'DeltaS_det':>12s} {'diff':>10s} {'Cohen_d':>10s} {'Wilcoxon_p':>12s}")
print("-"*100)

results = {}
for s_from, s_to in MEANINGFUL_TRANSITIONS:
    dm = np.array([by_pair[p]['missed'][s_to] - by_pair[p]['missed'][s_from] for p in complete])
    dd = np.array([by_pair[p]['detected'][s_to] - by_pair[p]['detected'][s_from] for p in complete])
    diff = dm - dd
    pooled_std = np.sqrt((dm.var(ddof=1) + dd.var(ddof=1)) / 2)
    cohen_d = diff.mean() / pooled_std if pooled_std > 0 else float('nan')
    _, p_val = stats.wilcoxon(diff) if not np.all(diff == 0) else (0, 1)
    label = f"{s_from}->{s_to}"
    print(f"{label:28s} {dm.mean():12.4f} {dd.mean():12.4f} {diff.mean():+10.4f} {cohen_d:+10.4f} {p_val:12.4e}")
    results[label] = {'diff': diff.mean(), 'd': cohen_d, 'p': p_val}

print("\n" + "="*100)
print("EXCLUDED (BN) TRANSITIONS -- reported for completeness, NOT interpreted as deficit-free")
print("="*100)
for s_from, s_to in EXCLUDED_TRANSITIONS:
    dm = np.array([by_pair[p]['missed'][s_to] - by_pair[p]['missed'][s_from] for p in complete])
    dd = np.array([by_pair[p]['detected'][s_to] - by_pair[p]['detected'][s_from] for p in complete])
    diff = dm - dd
    print(f"  {s_from}->{s_to}: diff={diff.mean():+.4f}  (statistic is BN-invariant by construction, not informative)")

print("\n" + "="*100)
print("TARGET IDENTIFICATION (among meaningful transitions only)")
print("="*100)
sorted_results = sorted(results.items(), key=lambda kv: kv[1]['diff'])
for label, r in sorted_results:
    flag = ' <-- MOST NEGATIVE' if label == sorted_results[0][0] else ''
    sig = ' (p<0.05)' if r['p'] < 0.05 else ' (n.s.)'
    print(f"  {label}: diff={r['diff']:+.4f}  d={r['d']:+.4f}{sig}{flag}")

worst_label, worst = sorted_results[0]
print(f"\nPRIMARY TARGET: {worst_label} (diff={worst['diff']:+.4f}, d={worst['d']:+.4f}, p={worst['p']:.4e})")
if worst['p'] < 0.05:
    if worst_label == 'cat1->conv1_out':
        print("  ==> FEATURE MIXING / CHANNEL PROJECTION: the first conv layer (processing the")
        print("      64-channel cat1 input) is where the deficit is introduced.")
    elif worst_label == 'conv1_out->relu1_out':
        print("  ==> NONLINEAR SUPPRESSION (BN+ReLU combined, block 1): cannot isolate BN's own")
        print("      share from ReLU's, but the combined block-1 normalization+activation step")
        print("      is where the deficit is introduced.")
    elif worst_label == 'relu1_out->conv2_out':
        print("  ==> SECOND FEATURE MIXING / CHANNEL PROJECTION: the second conv layer is where")
        print("      the deficit is introduced.")
    elif worst_label == 'conv2_out->relu2_out':
        print("  ==> NONLINEAR SUPPRESSION (BN+ReLU combined, block 2, final stage before D1).")
else:
    print("  No single internal operation shows a significant differential -- the deficit may")
    print("  be a PROGRESSIVE accumulation across the whole block rather than one operation.")
