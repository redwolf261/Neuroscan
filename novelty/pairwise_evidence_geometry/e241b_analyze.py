"""E241b analysis -- H8 pre-screen verdict, CORRECTED sliding-window
version (matches the same inference method used to define detected/
missed labels, resolving the confound found in E241's original run)."""
import csv
import numpy as np
from scipy import stats
from collections import defaultdict

rows = list(csv.DictReader(open('E241b_h8_prescreen_sw.csv')))
for r in rows:
    r['lesion_max'] = float(r['lesion_max'])
    r['lesion_mean'] = float(r['lesion_mean'])
    r['bg_max'] = float(r['bg_max']) if r['bg_max'] else None
    r['bg_mean'] = float(r['bg_mean']) if r['bg_mean'] else None

by_group = defaultdict(list)
for r in rows:
    by_group[r['group']].append(r)

for g, rs in by_group.items():
    print(f"{g}: n={len(rs)}")

det = by_group['matched_detected']
g2 = by_group['high_contrast_missed']
g4 = by_group['ordinary_missed']

det_lesion_max = np.array([r['lesion_max'] for r in det])
det_bg_max = np.array([r['bg_max'] for r in det if r['bg_max'] is not None])
g2_lesion_max = np.array([r['lesion_max'] for r in g2])
g4_lesion_max = np.array([r['lesion_max'] for r in g4])

def describe(name, arr):
    print(f"  {name:35s}: n={len(arr):5d}  median={np.median(arr):.4f}  "
          f"IQR=[{np.percentile(arr,25):.4f}, {np.percentile(arr,75):.4f}]  mean={arr.mean():.4f}")

print("\n" + "="*90)
print("MAX_PROB DISTRIBUTIONS (sliding-window, matches detection-label convention)")
print("="*90)
describe("detected (lesion voxels)", det_lesion_max)
describe("detected (background/shell)", det_bg_max)
describe("high_contrast_missed (lesion voxels)", g2_lesion_max)
describe("ordinary_missed (lesion voxels)", g4_lesion_max)

print("\n" + "="*90)
print("BIMODALITY CHECK (does the SW-corrected result still show two modes?)")
print("="*90)
for name, arr in [('high_contrast_missed', g2_lesion_max), ('ordinary_missed', g4_lesion_max)]:
    frac_zero = np.mean(arr < 0.01)
    frac_high = np.mean(arr > 0.9)
    frac_mid = 1 - frac_zero - frac_high
    print(f"  {name}: near-0 (<0.01)={frac_zero:.3f}  near-1 (>0.9)={frac_high:.3f}  mid={frac_mid:.3f}")

print("\n" + "="*90)
print("KEY COMPARISON: displacement from background")
print("="*90)

def overlap_test(a, b, label):
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    n1, n2 = len(a), len(b)
    r_eff = 1 - (2*u) / (n1*n2)
    print(f"  {label}: Mann-Whitney p={p:.4e}  rank-biserial effect={r_eff:+.4f}")
    return p, r_eff

p1, e1 = overlap_test(g2_lesion_max, det_bg_max, "G2_lesion vs det_background")
p2, e2 = overlap_test(g2_lesion_max, det_lesion_max, "G2_lesion vs det_lesion")
p3, e3 = overlap_test(g4_lesion_max, det_bg_max, "G4_lesion vs det_background")
p4, e4 = overlap_test(g4_lesion_max, det_lesion_max, "G4_lesion vs det_lesion")
p5, e5 = overlap_test(g2_lesion_max, g4_lesion_max, "G2 vs G4")

bg_q25, bg_q75 = np.percentile(det_bg_max, 25), np.percentile(det_bg_max, 75)
frac_g2_bg_like = np.mean((g2_lesion_max >= bg_q25) & (g2_lesion_max <= bg_q75))
frac_g4_bg_like = np.mean((g4_lesion_max >= bg_q25) & (g4_lesion_max <= bg_q75))
print(f"\nFraction of high_contrast_missed inside detected-BACKGROUND's IQR: {frac_g2_bg_like:.3f}")
print(f"Fraction of ordinary_missed inside detected-BACKGROUND's IQR: {frac_g4_bg_like:.3f}")
print(f"detected-background IQR: [{bg_q25:.4f}, {bg_q75:.4f}]")

print("\n" + "="*90)
print("H8 PRE-SCREEN VERDICT (sliding-window corrected)")
print("="*90)
close_to_bg = frac_g2_bg_like > 0.5 or frac_g4_bg_like > 0.5
displaced_from_bg = p1 < 0.05 and abs(e1) > 0.2
print(f"Missed lesions mostly WITHIN background's typical range: {'YES' if close_to_bg else 'NO'}")
print(f"Missed lesions significantly DISPLACED from background: {'YES' if displaced_from_bg else 'NO'} (p={p1:.2e}, effect={e1:+.3f})")

if displaced_from_bg and not close_to_bg:
    print("\n  ==> Signal present, but check bimodality carefully before calling this")
    print("      'H8 earns an experiment' -- if the displacement is driven by a SUBSET")
    print("      of lesions near max_prob~1.0 (essentially detected by this metric) while")
    print("      the rest sit at background level, H8 may not be a single coherent")
    print("      phenomenon -- see bimodality check above.")
else:
    print("\n  ==> H8 KILLED per pre-screen: missed-lesion scores look statistically like")
    print("      background under the CORRECTED (label-consistent) measurement.")
