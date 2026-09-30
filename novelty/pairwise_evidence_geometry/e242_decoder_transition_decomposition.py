"""E242 -- decoder transition decomposition, per the user's exact spec.
Reuses E240's own separability measurements UNCHANGED (no new forward
passes) -- this is a pure re-analysis of E240_layerwise.csv, asking a
sharper question than "are D1 features different": WHERE along the
decoder does the missed-vs-detected separability gap actually get
introduced.

Delta_{D3->D2} = S(D2) - S(D3)
Delta_{D2->D1} = S(D1) - S(D2)
(also Delta_{BN->D3} = S(D3) - S(E4_BN), the upstream transition, for
completeness -- the full decoder chain, not just the two the user named)

Critical prediction: if a SPECIFIC transition is responsible,
Delta_missed << Delta_detected for that transition specifically, not
uniformly across all transitions -- an operation-level target, not just
"D1 is different" (which E240 already established, aggregate).
"""
import csv
import numpy as np
from scipy import stats
from collections import defaultdict

STAGES = ('raw', 'E1', 'E2', 'E3', 'E4_BN', 'D3', 'D2', 'D1')
TRANSITIONS = [('E4_BN', 'D3'), ('D3', 'D2'), ('D2', 'D1')]

rows = list(csv.DictReader(open('E240_layerwise.csv')))
for r in rows:
    r['pair_id'] = int(r['pair_id'])
    for s in STAGES:
        r[f'sep_{s}'] = float(r[f'sep_{s}']) if r[f'sep_{s}'] != '' else None

by_pair = defaultdict(dict)
for r in rows:
    by_pair[r['pair_id']][r['role']] = r

complete = [p for p, d in by_pair.items() if 'missed' in d and 'detected' in d]
print(f"Complete pairs: {len(complete)} (same population as E240)")

print("\n" + "="*100)
print("PER-STAGE SEPARABILITY (reproduced from E240 for reference)")
print("="*100)
for s in STAGES:
    a = np.array([by_pair[p]['missed'][f'sep_{s}'] for p in complete
                 if by_pair[p]['missed'][f'sep_{s}'] is not None and by_pair[p]['detected'][f'sep_{s}'] is not None])
    b = np.array([by_pair[p]['detected'][f'sep_{s}'] for p in complete
                 if by_pair[p]['missed'][f'sep_{s}'] is not None and by_pair[p]['detected'][f'sep_{s}'] is not None])
    print(f"  {s:8s}: missed={a.mean():.4f}  detected={b.mean():.4f}")

print("\n" + "="*100)
print("TRANSITION DECOMPOSITION: Delta_S = S(stage_k) - S(stage_{k-1}), per lesion, paired")
print("="*100)
print(f"{'Transition':16s} {'missed_mean':>12s} {'det_mean':>12s} {'diff':>10s} {'Cohen_d':>10s} {'Wilcoxon_p':>12s} {'n':>6s}")
print("-"*100)

results = {}
for s_from, s_to in TRANSITIONS:
    delta_missed, delta_det = [], []
    for p in complete:
        m, d = by_pair[p]['missed'], by_pair[p]['detected']
        sm_from, sm_to = m[f'sep_{s_from}'], m[f'sep_{s_to}']
        sd_from, sd_to = d[f'sep_{s_from}'], d[f'sep_{s_to}']
        if None in (sm_from, sm_to, sd_from, sd_to):
            continue
        delta_missed.append(sm_to - sm_from)
        delta_det.append(sd_to - sd_from)
    delta_missed = np.array(delta_missed)
    delta_det = np.array(delta_det)
    diff = delta_missed - delta_det
    pooled_std = np.sqrt((delta_missed.var(ddof=1) + delta_det.var(ddof=1)) / 2)
    cohen_d = diff.mean() / pooled_std if pooled_std > 0 else float('nan')
    if np.all(diff == 0):
        p_val = 1.0
    else:
        _, p_val = stats.wilcoxon(diff)
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
    label = f"{s_from}->{s_to}"
    print(f"{label:16s} {delta_missed.mean():12.4f} {delta_det.mean():12.4f} {diff.mean():+10.4f} {cohen_d:+10.4f} {p_val:12.4e} {len(diff):6d} {sig}")
    results[label] = {'delta_missed': delta_missed, 'delta_det': delta_det, 'diff': diff, 'p': p_val, 'd': cohen_d}

print("\n" + "="*100)
print("WHICH TRANSITION IS THE OPERATION-LEVEL TARGET?")
print("="*100)
sig_transitions = [(k, v) for k, v in results.items() if v['p'] < 0.05]
if not sig_transitions:
    print("  NO transition shows a significant missed-vs-detected difference in Delta_S.")
    print("  The aggregate D1 gap (E240) is NOT concentrated at any single measured")
    print("  transition -- it may be a gradual accumulation across multiple small steps,")
    print("  or the transition-level test is underpowered relative to the (already small,")
    print("  |d|<0.15) aggregate D1 effect. Either way, there is no clean single")
    print("  operation-level target from this analysis alone.")
else:
    print("  Transition(s) reaching p<0.05:")
    for k, v in sig_transitions:
        direction = "missed LOSES more separability" if v['diff'].mean() < 0 else "missed GAINS more separability"
        print(f"    {k}: diff={v['diff'].mean():+.4f}  d={v['d']:+.4f}  p={v['p']:.4e}  ({direction})")
    best = min(sig_transitions, key=lambda kv: kv[1]['p'])
    print(f"\n  Strongest candidate: {best[0]} (p={best[1]['p']:.4e})")
    if best[0] == 'D2->D1':
        print("  This is the upconv1 -> attn_gate1(gate=bottleneck, skip=enc1) -> cat1 -> dec1")
        print("  transformation -- D1 = f(D2, skip_E1). Worth decomposing into its constituent")
        print("  operations (upconv, attention gate, concatenation, conv block) before any")
        print("  algorithm design.")
    elif best[0] == 'D3->D2':
        print("  This is the upconv2 -> cat2(dec3,enc2) -> dec2 transformation.")
    elif best[0] == 'E4_BN->D3':
        print("  This is the upconv3 -> cat3(bottleneck,enc3) -> dec3 transformation.")
