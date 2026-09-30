"""E247 analysis -- channel-specific clipping, per the user's exact
question: are the SAME discriminative channels systematically clipped in
G2-A, and does this pattern reverse/vanish in G2-B (the critical
control)?

Primary test: per-lesion Spearman r(informativeness, clip_c), compared
across G2-A missed / matched detected / G2-B partial. A POSITIVE mean r
means informative channels are clipped MORE (the "wrong features
suppressed" pattern). If G2-A shows positive r, matched detected shows
~0 or negative r, and G2-B shows ~0 or negative r too -- that is the
user's predicted "feature-selective nonlinear bottleneck" signature.
"""
import csv
import numpy as np
from scipy import stats
from collections import defaultdict

rows = list(csv.DictReader(open('E247_channel_clipping.csv')))
for r in rows:
    r['pair_idx'] = int(r['pair_idx'])
    r['r_informativeness_vs_clip'] = float(r['r_informativeness_vs_clip'])
    r['r_informativeness_vs_deltaz'] = float(r['r_informativeness_vs_deltaz'])

g2a = [r for r in rows if r['group'] == 'G2A_missed']
det = [r for r in rows if r['group'] == 'matched_detected']
g2b = [r for r in rows if r['group'] == 'G2B_partial']

print("="*100)
print("GROUP SUMMARIES: r(informativeness, clip_c) -- POSITIVE means informative")
print("channels are clipped MORE (the 'wrong features suppressed' pattern)")
print("="*100)


def describe(name, group_rows, key):
    vals = np.array([r[key] for r in group_rows])
    t_stat, p_val = stats.ttest_1samp(vals, 0)
    w_stat, p_w = stats.wilcoxon(vals) if not np.all(vals == 0) else (0, 1)
    print(f"\n  {name} (n={len(vals)}):")
    print(f"    mean r={vals.mean():+.4f}  median={np.median(vals):+.4f}  std={vals.std():.4f}")
    print(f"    one-sample t-test vs 0: t={t_stat:.3f}  p={p_val:.4e}")
    print(f"    Wilcoxon vs 0: p={p_w:.4e}")
    return vals


print("\n--- r(informativeness, clip_c) ---")
vals_a_clip = describe("G2-A missed", g2a, 'r_informativeness_vs_clip')
vals_d_clip = describe("matched detected", det, 'r_informativeness_vs_clip')
vals_b_clip = describe("G2-B partial", g2b, 'r_informativeness_vs_clip')

print("\n--- r(informativeness, delta_z) [informative channels' OWN separability loss] ---")
vals_a_delta = describe("G2-A missed", g2a, 'r_informativeness_vs_deltaz')
vals_d_delta = describe("matched detected", det, 'r_informativeness_vs_deltaz')
vals_b_delta = describe("G2-B partial", g2b, 'r_informativeness_vs_deltaz')

print("\n" + "="*100)
print("PAIRED TEST: G2-A vs matched detected (r_clip)")
print("="*100)
by_pair = defaultdict(dict)
for r in rows:
    if r['group'] in ('G2A_missed', 'matched_detected'):
        role = 'missed' if r['group'] == 'G2A_missed' else 'detected'
        by_pair[r['pair_idx']][role] = r
complete = [p for p, d in by_pair.items() if 'missed' in d and 'detected' in d]
print(f"Complete pairs: {len(complete)}")

a_clip = np.array([by_pair[p]['missed']['r_informativeness_vs_clip'] for p in complete])
d_clip = np.array([by_pair[p]['detected']['r_informativeness_vs_clip'] for p in complete])
diff_clip = a_clip - d_clip
_, p_diff_clip = stats.wilcoxon(diff_clip)
print(f"\nr_clip: missed={a_clip.mean():+.4f}  detected={d_clip.mean():+.4f}  "
      f"diff={diff_clip.mean():+.4f}  Wilcoxon p={p_diff_clip:.4e}")

a_delta = np.array([by_pair[p]['missed']['r_informativeness_vs_deltaz'] for p in complete])
d_delta = np.array([by_pair[p]['detected']['r_informativeness_vs_deltaz'] for p in complete])
diff_delta = a_delta - d_delta
_, p_diff_delta = stats.wilcoxon(diff_delta)
print(f"r_delta: missed={a_delta.mean():+.4f}  detected={d_delta.mean():+.4f}  "
      f"diff={diff_delta.mean():+.4f}  Wilcoxon p={p_diff_delta:.4e}")

print("\n" + "="*100)
print("CRITICAL CONTROL: G2-B comparison (unpaired vs G2-A and vs detected)")
print("="*100)
u1, p_u1 = stats.mannwhitneyu(vals_a_clip, vals_b_clip)
print(f"G2-A vs G2-B (r_clip): Mann-Whitney p={p_u1:.4e}  "
      f"(G2-A mean={vals_a_clip.mean():+.4f}, G2-B mean={vals_b_clip.mean():+.4f})")
u2, p_u2 = stats.mannwhitneyu(vals_b_clip, vals_d_clip)
print(f"G2-B vs detected (r_clip): Mann-Whitney p={p_u2:.4e}  "
      f"(G2-B mean={vals_b_clip.mean():+.4f}, detected mean={vals_d_clip.mean():+.4f})")

print("\n" + "="*100)
print("VERDICT")
print("="*100)
g2a_positive = vals_a_clip.mean() > 0 and stats.ttest_1samp(vals_a_clip, 0)[1] < 0.05
g2a_gt_det = p_diff_clip < 0.05 and diff_clip.mean() > 0
g2b_differs_from_a = p_u1 < 0.05
g2b_like_detected = p_u2 > 0.05 or (vals_b_clip.mean() <= 0)

print(f"G2-A shows positive informativeness-clipping correlation: {'YES' if g2a_positive else 'NO'}")
print(f"G2-A > matched detected (paired): {'YES' if g2a_gt_det else 'NO'} (p={p_diff_clip:.2e})")
print(f"G2-B differs from G2-A: {'YES' if g2b_differs_from_a else 'NO'} (p={p_u1:.2e})")
print(f"G2-B resembles detected (not G2-A): {'YES' if g2b_like_detected else 'NO'}")

if g2a_positive and g2a_gt_det and g2b_differs_from_a and g2b_like_detected:
    print("\n  ==> FEATURE-SELECTIVE NONLINEAR BOTTLENECK CONFIRMED: G2-A's most")
    print("      discriminative channels are disproportionately clipped by ReLU1, this")
    print("      pattern is significantly stronger than matched detected lesions, and G2-B")
    print("      does NOT show the same pattern (resembles detected, not G2-A). The network's")
    print("      ReLU is suppressing the WRONG features specifically for the G2-A phenotype --")
    print("      a genuinely non-generic, phenotype-specific target for intervention design.")
elif not g2a_positive or not g2a_gt_det:
    print("\n  ==> NOT CONFIRMED: G2-A does not show a clear informativeness-biased clipping")
    print("      pattern beyond matched detected. The channel-selectivity hypothesis is not")
    print("      supported as tested -- do not design a feature-selective intervention on")
    print("      this basis.")
else:
    print("\n  ==> PARTIAL: G2-A shows the pattern but G2-B does not clearly differ as")
    print("      predicted -- report the exact numbers, do not force the clean story.")
