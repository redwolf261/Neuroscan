"""E255 analysis -- whole-volume FP cost of the max(p_prod, p_R1)
ensemble, the critical missing measurement from E251-E254 per this
session's own E237-informed discipline."""
import csv
import numpy as np
from scipy import stats

rows = list(csv.DictReader(open('E255_max_ensemble.csv')))
for r in rows:
    r['in_test_split'] = int(r['in_test_split'])
    r['fp_voxels_prod'] = int(r['fp_voxels_prod'])
    r['fp_components_prod'] = int(r['fp_components_prod'])
    r['fp_voxels_max'] = int(r['fp_voxels_max'])
    r['fp_components_max'] = int(r['fp_components_max'])

print(f"Total subjects evaluated: {len(rows)}")

fp_v_prod = np.array([r['fp_voxels_prod'] for r in rows])
fp_v_max = np.array([r['fp_voxels_max'] for r in rows])
fp_c_prod = np.array([r['fp_components_prod'] for r in rows])
fp_c_max = np.array([r['fp_components_max'] for r in rows])

print("\n" + "="*90)
print("WHOLE-VOLUME FALSE POSITIVE BURDEN: production vs max(prod, R1)")
print("="*90)
print(f"\nFP VOXELS:")
print(f"  production: mean={fp_v_prod.mean():.1f}  median={np.median(fp_v_prod):.0f}  "
      f"total={fp_v_prod.sum()}")
print(f"  max_ensemble: mean={fp_v_max.mean():.1f}  median={np.median(fp_v_max):.0f}  "
      f"total={fp_v_max.sum()}")
diff_v = fp_v_max - fp_v_prod
print(f"  per-subject increase: mean={diff_v.mean():+.1f}  median={np.median(diff_v):+.0f}")
pct_increase = 100 * (fp_v_max.sum() - fp_v_prod.sum()) / max(1, fp_v_prod.sum())
print(f"  AGGREGATE % increase in total FP voxels: {pct_increase:+.1f}%")

print(f"\nFP CONNECTED COMPONENTS:")
print(f"  production: mean={fp_c_prod.mean():.1f}  median={np.median(fp_c_prod):.0f}  total={fp_c_prod.sum()}")
print(f"  max_ensemble: mean={fp_c_max.mean():.1f}  median={np.median(fp_c_max):.0f}  total={fp_c_max.sum()}")
diff_c = fp_c_max - fp_c_prod
pct_increase_c = 100 * (fp_c_max.sum() - fp_c_prod.sum()) / max(1, fp_c_prod.sum())
print(f"  AGGREGATE % increase in total FP components: {pct_increase_c:+.1f}%")

_, p_wilcoxon_v = stats.wilcoxon(diff_v) if not np.all(diff_v == 0) else (0, 1)
_, p_wilcoxon_c = stats.wilcoxon(diff_c) if not np.all(diff_c == 0) else (0, 1)
print(f"\n  Wilcoxon (voxels, max vs prod): p={p_wilcoxon_v:.4e}")
print(f"  Wilcoxon (components, max vs prod): p={p_wilcoxon_c:.4e}")

print("\n" + "="*90)
print("PER-SUBJECT DISTRIBUTION: any subjects with a LARGE FP increase (not just average)?")
print("="*90)
# relative increase, guarding against div by zero
rel_increase = np.where(fp_v_prod > 0, diff_v / np.maximum(fp_v_prod, 1), np.where(diff_v > 0, np.inf, 0))
worst_idx = np.argsort(-diff_v)[:10]
print("\nTop 10 subjects by ABSOLUTE FP voxel increase:")
for i in worst_idx:
    print(f"  {rows[i]['subject_id']}: prod={fp_v_prod[i]}  max={fp_v_max[i]}  "
         f"increase={diff_v[i]:+d}  test_split={rows[i]['in_test_split']}")

n_zero_to_positive = int(((fp_v_prod == 0) & (fp_v_max > 0)).sum())
print(f"\nSubjects with ZERO FP under production but NONZERO under max_ensemble: {n_zero_to_positive}/{len(rows)}")

print("\n" + "="*90)
print("TEST-SPLIT-ONLY SUBSET (the genuinely held-out detected subjects)")
print("="*90)
test_rows_idx = [i for i, r in enumerate(rows) if r['in_test_split'] == 1]
if test_rows_idx:
    tv_prod = fp_v_prod[test_rows_idx]; tv_max = fp_v_max[test_rows_idx]
    print(f"  n={len(test_rows_idx)}  prod_mean={tv_prod.mean():.1f}  max_mean={tv_max.mean():.1f}  "
         f"increase_mean={(tv_max-tv_prod).mean():+.1f}")

print("\n" + "="*90)
print("VERDICT")
print("="*100)
large_relative_increase = pct_increase > 50
print(f"Aggregate FP voxel increase: {pct_increase:+.1f}%")
if large_relative_increase:
    print("\n  ==> SUBSTANTIAL FP COST: the naive max() ensemble meaningfully increases whole-")
    print("      volume false positives. A learned/gated mechanism is likely NECESSARY --")
    print("      the simplest possible intervention is not acceptable as-is.")
else:
    print("\n  ==> FP cost is modest in aggregate. Check per-subject worst cases above before")
    print("      concluding this is safe -- an average can hide a few badly-affected subjects.")
