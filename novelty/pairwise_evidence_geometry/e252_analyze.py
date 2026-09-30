"""E252 analysis -- directional vs calibration decomposition, per the
user's exact decisive question: is R1's +34.9pp advantage (E251) coming
from a DIFFERENT direction in D1-feature-space (production weighs
features differently) or the SAME direction with a different threshold/
bias (calibration)?
"""
import csv
import numpy as np
from scipy import stats

rows = list(csv.DictReader(open('E252_geometry.csv')))
for r in rows:
    r['cos_w'] = float(r['cos_w'])
    r['w_r1_norm'] = float(r['w_r1_norm'])
    r['b_r1'] = float(r['b_r1'])
    r['logit_prod_mean'] = float(r['logit_prod_mean'])
    r['logit_r1_mean'] = float(r['logit_r1_mean'])
    r['logit_diff_mean'] = float(r['logit_diff_mean'])
    r['aligned_component_mean'] = float(r['aligned_component_mean'])
    r['orthogonal_component_norm_mean'] = float(r['orthogonal_component_norm_mean'])

print(f"Total G2-A lesion-fold rows: {len(rows)}")

print("\n" + "="*100)
print("WEIGHT-VECTOR SIMILARITY: cos(w_prod, w_R1), per fold")
print("="*100)
by_fold = {}
for r in rows:
    by_fold[r['fold']] = r['cos_w']  # same value repeated per lesion in that fold
for fold, cos in sorted(by_fold.items()):
    print(f"  fold {fold}: cos(w_prod, w_R1) = {cos:+.4f}")
cos_vals = np.array(list(by_fold.values()))
print(f"\n  mean cos across folds: {cos_vals.mean():+.4f}  std: {cos_vals.std():.4f}")

print("\n" + "="*100)
print("LOGIT COMPARISON (mean per lesion, across all G2-A lesions)")
print("="*100)
logit_prod = np.array([r['logit_prod_mean'] for r in rows])
logit_r1 = np.array([r['logit_r1_mean'] for r in rows])
print(f"  logit_prod: mean={logit_prod.mean():.4f}  median={np.median(logit_prod):.4f}")
print(f"  logit_r1:   mean={logit_r1.mean():.4f}  median={np.median(logit_r1):.4f}")
print(f"  (logits are on DIFFERENT scales since w_prod and w_R1 have different norms --")
print(f"   the scale itself is not meaningful, only the SIGN and the aligned/orthogonal")
print(f"   decomposition below are directly comparable)")

print("\n" + "="*100)
print("ALIGNED vs ORTHOGONAL DECOMPOSITION (the decisive test)")
print("="*100)
aligned = np.array([r['aligned_component_mean'] for r in rows])
orth_norm = np.array([r['orthogonal_component_norm_mean'] for r in rows])
print(f"  aligned component (R1's contribution ALONG w_prod's own direction):")
print(f"    mean={aligned.mean():.4f}  median={np.median(aligned):.4f}  std={aligned.std():.4f}")
print(f"  orthogonal component norm (R1's contribution PERPENDICULAR to w_prod's direction):")
print(f"    mean={orth_norm.mean():.4f}  median={np.median(orth_norm):.4f}  std={orth_norm.std():.4f}")

ratio = orth_norm.mean() / (abs(aligned.mean()) + 1e-9)
print(f"\n  ratio |orthogonal| / |aligned|: {ratio:.4f}")
print(f"  (ratio >> 1 -> R1's advantage is mostly DIRECTIONAL (different feature weighting).")
print(f"   ratio << 1 -> R1's advantage is mostly ALONG production's own direction")
print(f"   (calibration/threshold-like, even though R1 is a fresh linear fit, not literally")
print(f"   'seg_head with a different bias').")

print("\n" + "="*100)
print("VERDICT")
print("="*100)
mean_cos = cos_vals.mean()
print(f"Mean cos(w_prod, w_R1) across folds: {mean_cos:+.4f}")

if abs(mean_cos) > 0.7:
    direction_verdict = "SAME DIRECTION (|cos|>0.7) -- points toward CALIBRATION/threshold as the primary issue."
elif abs(mean_cos) < 0.3:
    direction_verdict = "SUBSTANTIALLY DIFFERENT DIRECTION (|cos|<0.3) -- production seg_head is using the D1 feature dimensions DIFFERENTLY than R1, not just at a different threshold. This is a DIRECTIONAL/representational discrepancy, not primarily calibration."
else:
    direction_verdict = "PARTIALLY ALIGNED (0.3<=|cos|<=0.7) -- a mix of directional and calibration differences; report both components, do not force a single story."
print(f"\n{direction_verdict}")

if ratio > 2:
    print(f"\nOrthogonal/aligned ratio ({ratio:.2f}) confirms: R1's advantage over production is")
    print("PREDOMINANTLY DIRECTIONAL -- it comes from weighting D1 channels differently than")
    print("seg_head does, not from sitting at a different point along seg_head's own decision axis.")
elif ratio < 0.5:
    print(f"\nOrthogonal/aligned ratio ({ratio:.2f}) confirms: R1's advantage over production is")
    print("PREDOMINANTLY ALONG seg_head's own direction -- consistent with a calibration/")
    print("threshold story despite the low raw cos(w) value (a linear model can move mostly")
    print("along one direction even while its full weight vector isn't perfectly parallel).")
else:
    print(f"\nOrthogonal/aligned ratio ({ratio:.2f}) is intermediate -- both directional and")
    print("calibration components contribute meaningfully; do not force a single mechanism.")
