"""E250 analysis -- activation pattern vs channel identity, per the
user's exact decisive tests:
  1. Does destroying cross-channel structure (channel_shuffled) hurt
     G2-A's separability/recovery relative to real?
  2. Does destroying spatial arrangement (spatial_shuffled) hurt it?
  3. Does imposing DETECTED's own correlation structure
     (synthetic_matched) IMPROVE G2-A's separability/recovery, and does
     it beat covariance_randomized (same spectrum, wrong structure)?
The decisive comparison, per the user's own bar: synthetic_matched
RECOVERS lesions AND beats covariance_randomized.
"""
import csv
import numpy as np
from scipy import stats
from collections import defaultdict

rows = list(csv.DictReader(open('E250_pattern.csv')))
for r in rows:
    r['max_prob'] = float(r['max_prob'])
    r['mean_prob'] = float(r['mean_prob'])
    r['recovered'] = int(r['recovered'])
    r['sep_D1'] = float(r['sep_D1']) if r['sep_D1'] != '' else None
    r['fp_shell'] = float(r['fp_shell'])

g2a_rows = [r for r in rows if r['group'] == 'G2A']
g2b_real = [r for r in rows if r['group'] == 'G2B' and r['condition'] == 'real']
det_real = [r for r in rows if r['group'] == 'detected' and r['condition'] == 'real']

by_lesion = defaultdict(dict)
for r in g2a_rows:
    key = (r['subject_id'], r['comp_id'])
    by_lesion[key][r['condition']] = r

CONDITIONS = ['real', 'channel_shuffled', 'spatial_shuffled', 'synthetic_matched', 'covariance_randomized']
complete = [k for k, d in by_lesion.items() if all(c in d for c in CONDITIONS)]
print(f"Complete G2-A lesions (all 5 conditions): {len(complete)}")

print("\n" + "="*100)
print("CONDITION SUMMARIES (G2-A)")
print("="*100)
for metric in ['max_prob', 'sep_D1', 'recovered', 'fp_shell']:
    print(f"\n  --- {metric} ---")
    for cond in CONDITIONS:
        vals = np.array([by_lesion[k][cond][metric] for k in complete
                         if by_lesion[k][cond][metric] is not None])
        print(f"    {cond:22s}: mean={vals.mean():.4f}  median={np.median(vals):.4f}")

print(f"\n  --- reference: G2-B real (n={len(g2b_real)}), detected real (n={len(det_real)}) ---")
for label, rlist in [('G2-B real', g2b_real), ('detected real', det_real)]:
    for metric in ['max_prob', 'sep_D1']:
        vals = np.array([r[metric] for r in rlist if r[metric] is not None])
        print(f"    {label} {metric}: mean={vals.mean():.4f}")


def paired_compare(metric, cond_a, cond_b):
    a = np.array([by_lesion[k][cond_a][metric] for k in complete
                 if by_lesion[k][cond_a][metric] is not None and by_lesion[k][cond_b][metric] is not None])
    b = np.array([by_lesion[k][cond_b][metric] for k in complete
                 if by_lesion[k][cond_a][metric] is not None and by_lesion[k][cond_b][metric] is not None])
    diff = a - b
    if len(diff) < 2 or np.all(diff == 0):
        return diff.mean() if len(diff) else float('nan'), 1.0, len(diff)
    _, p = stats.wilcoxon(diff)
    return diff.mean(), p, len(diff)


print("\n" + "="*100)
print("DESTRUCTION TESTS: does breaking channel/spatial structure HURT G2-A?")
print("="*100)
for metric in ['max_prob', 'sep_D1']:
    print(f"\n  --- {metric} ---")
    for cond in ['channel_shuffled', 'spatial_shuffled']:
        diff, p, n = paired_compare(metric, cond, 'real')
        print(f"    {cond} - real: diff={diff:+.4f}  p={p:.4e}  n={n}")

print("\n" + "="*100)
print("DECISIVE TEST: synthetic_matched vs covariance_randomized (the user's own bar)")
print("="*100)
for metric in ['max_prob', 'mean_prob', 'recovered', 'sep_D1', 'fp_shell']:
    diff, p, n = paired_compare(metric, 'synthetic_matched', 'covariance_randomized')
    diff_real, p_real, _ = paired_compare(metric, 'synthetic_matched', 'real')
    sig = ' *' if p < 0.05 else ''
    print(f"  {metric}: synthetic-covrand diff={diff:+.4f} p={p:.4e}{sig}   "
          f"synthetic-real diff={diff_real:+.4f} p={p_real:.4e}")

print("\n" + "="*100)
print("VERDICT")
print("="*100)
diff_sm_cr, p_sm_cr, _ = paired_compare('max_prob', 'synthetic_matched', 'covariance_randomized')
diff_sm_real, p_sm_real, _ = paired_compare('max_prob', 'synthetic_matched', 'real')
diff_sm_sep, p_sm_sep, _ = paired_compare('sep_D1', 'synthetic_matched', 'covariance_randomized')

recovers_vs_real = p_sm_real < 0.05 and diff_sm_real > 0
beats_covrand = p_sm_cr < 0.05 and diff_sm_cr > 0
beats_covrand_sep = p_sm_sep < 0.05 and diff_sm_sep > 0

print(f"synthetic_matched recovers MORE than real (unmodified): {'YES' if recovers_vs_real else 'NO'} (p={p_sm_real:.2e})")
print(f"synthetic_matched beats covariance_randomized (max_prob): {'YES' if beats_covrand else 'NO'} (p={p_sm_cr:.2e})")
print(f"synthetic_matched beats covariance_randomized (separability): {'YES' if beats_covrand_sep else 'NO'} (p={p_sm_sep:.2e})")

if recovers_vs_real and beats_covrand:
    print("\n  ==> PATTERN MECHANISM CONFIRMED: imposing detected lesions' own cross-channel")
    print("      correlation structure (while preserving G2-A's own marginal statistics)")
    print("      produces recovery, AND this beats a same-spectrum random-structure control.")
    print("      The joint activation GEOMETRY, not individual channel identity, is causally")
    print("      implicated. This is now a defensible basis for an intervention design.")
else:
    print("\n  ==> PATTERN MECHANISM NOT CONFIRMED. Per the user's own explicit instruction:")
    print("      if E250 is also negative, STOP mining the ReLU/channel story and open a")
    print("      genuinely different mechanism rather than endlessly decomposing dec1.")
