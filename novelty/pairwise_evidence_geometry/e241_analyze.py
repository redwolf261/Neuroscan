"""E241 analysis -- H8 pre-screen verdict, per the user's exact
instruction: "If there is a clear systematic displacement, then H8 earns
an experiment. If not, kill it immediately." """
import csv
import numpy as np
from scipy import stats
from collections import defaultdict

rows = list(csv.DictReader(open('E241_h8_prescreen.csv')))
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
det_lesion_mean = np.array([r['lesion_mean'] for r in det])
det_bg_max = np.array([r['bg_max'] for r in det if r['bg_max'] is not None])
det_bg_mean = np.array([r['bg_mean'] for r in det if r['bg_mean'] is not None])

g2_lesion_max = np.array([r['lesion_max'] for r in g2])
g2_lesion_mean = np.array([r['lesion_mean'] for r in g2])

g4_lesion_max = np.array([r['lesion_max'] for r in g4])
g4_lesion_mean = np.array([r['lesion_mean'] for r in g4])

def describe(name, arr):
    print(f"  {name:35s}: n={len(arr):5d}  median={np.median(arr):.4f}  "
          f"IQR=[{np.percentile(arr,25):.4f}, {np.percentile(arr,75):.4f}]  "
          f"mean={arr.mean():.4f}")

print("\n" + "="*90)
print("MAX_PROB DISTRIBUTIONS")
print("="*90)
describe("detected (lesion voxels)", det_lesion_max)
describe("detected (background/shell)", det_bg_max)
describe("high_contrast_missed (lesion voxels)", g2_lesion_max)
describe("ordinary_missed (lesion voxels)", g4_lesion_max)

print("\n" + "="*90)
print("MEAN_PROB DISTRIBUTIONS")
print("="*90)
describe("detected (lesion voxels)", det_lesion_mean)
describe("detected (background/shell)", det_bg_mean)
describe("high_contrast_missed (lesion voxels)", g2_lesion_mean)
describe("ordinary_missed (lesion voxels)", g4_lesion_mean)

print("\n" + "="*90)
print("KEY COMPARISON: do missed-lesion scores look like BACKGROUND, or like a")
print("DISTINCT displaced population (neither full detected-lesion-level nor")
print("full background-level)?")
print("="*90)

def overlap_test(a, b, label):
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    # effect size (rank-biserial)
    n1, n2 = len(a), len(b)
    r_eff = 1 - (2*u) / (n1*n2)
    print(f"  {label}: Mann-Whitney p={p:.4e}  rank-biserial effect={r_eff:+.4f}")
    return p, r_eff

print("\n[max_prob] high_contrast_missed vs detected_background (are missed lesions 'background-like'?):")
p1, e1 = overlap_test(g2_lesion_max, det_bg_max, "G2_lesion vs det_background")
print("\n[max_prob] high_contrast_missed vs detected_lesion (are missed lesions 'detected-like'?):")
p2, e2 = overlap_test(g2_lesion_max, det_lesion_max, "G2_lesion vs det_lesion")
print("\n[max_prob] ordinary_missed vs detected_background:")
p3, e3 = overlap_test(g4_lesion_max, det_bg_max, "G4_lesion vs det_background")
print("\n[max_prob] ordinary_missed vs detected_lesion:")
p4, e4 = overlap_test(g4_lesion_max, det_lesion_max, "G4_lesion vs det_lesion")
print("\n[max_prob] high_contrast_missed vs ordinary_missed:")
p5, e5 = overlap_test(g2_lesion_max, g4_lesion_max, "G2 vs G4")

# fraction of each missed group that is "indistinguishable from background"
# i.e. within background's own IQR
bg_q25, bg_q75 = np.percentile(det_bg_max, 25), np.percentile(det_bg_max, 75)
frac_g2_bg_like = np.mean((g2_lesion_max >= bg_q25) & (g2_lesion_max <= bg_q75))
frac_g4_bg_like = np.mean((g4_lesion_max >= bg_q25) & (g4_lesion_max <= bg_q75))
det_q25, det_q75 = np.percentile(det_lesion_max, 25), np.percentile(det_lesion_max, 75)
frac_g2_det_like = np.mean((g2_lesion_max >= det_q25) & (g2_lesion_max <= det_q75))
frac_g4_det_like = np.mean((g4_lesion_max >= det_q25) & (g4_lesion_max <= det_q75))

print(f"\nFraction of high_contrast_missed lesions with max_prob inside detected-BACKGROUND's IQR: {frac_g2_bg_like:.3f}")
print(f"Fraction of high_contrast_missed lesions with max_prob inside detected-LESION's IQR: {frac_g2_det_like:.3f}")
print(f"Fraction of ordinary_missed lesions with max_prob inside detected-BACKGROUND's IQR: {frac_g4_bg_like:.3f}")
print(f"Fraction of ordinary_missed lesions with max_prob inside detected-LESION's IQR: {frac_g4_det_like:.3f}")

print("\n" + "="*90)
print("H8 PRE-SCREEN VERDICT")
print("="*90)
# "clear systematic displacement" = missed lesions are NOT simply indistinguishable
# from background (would mean "H8 is nothing new, they're just background-level")
# AND NOT simply indistinguishable from detected (would mean no real gap to explain)
# but occupy a DISTINCT middle ground, OR the two missed groups (G2 vs G4) differ
# from each other in a way suggesting a genuine decision-boundary-adjacent population.
close_to_bg = frac_g2_bg_like > 0.5 or frac_g4_bg_like > 0.5
displaced_from_bg = p1 < 0.05 and abs(e1) > 0.2
print(f"Missed lesions mostly WITHIN background's typical range: {'YES' if close_to_bg else 'NO'}")
print(f"Missed lesions significantly DISPLACED from background: {'YES' if displaced_from_bg else 'NO'} (p={p1:.2e}, effect={e1:+.3f})")

if displaced_from_bg and not close_to_bg:
    print("\n  ==> H8 EARNS AN EXPERIMENT: missed-lesion scores are NOT simply background-")
    print("      level noise -- there is a clear systematic displacement above background,")
    print("      suggesting the model IS registering elevated (if sub-threshold) evidence")
    print("      at these locations, consistent with a decision-boundary-proximate")
    print("      population worth a dedicated causal test.")
else:
    print("\n  ==> H8 KILLED IMMEDIATELY per pre-screen: missed-lesion scores look")
    print("      statistically like background, not like a distinct displaced")
    print("      population. No clear systematic displacement -- do not design a")
    print("      threshold/calibration-fix experiment around this.")
