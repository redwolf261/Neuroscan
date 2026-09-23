import json
from pathlib import Path
import numpy as np
from scipy import stats

RES_DIR = Path(__file__).parent / "e26_response_phenotype_results"
data = json.load(open(RES_DIR / "e26_response_phenotype.json"))

subj = data["subject_records"]
comp = data["component_records"]
n = len(subj)

# ---------- SUBJECT LEVEL ----------
delta = np.array([r["delta_dice"] for r in subj])
features = {
    "dice_A (baseline)": np.array([r["dice_A"] for r in subj]),
    "mean_component_size": np.array([r["mean_component_size"] for r in subj]),
    "gt_lesion_volume": np.array([r["gt_lesion_volume"] for r in subj]),
    "n_gt_components": np.array([r["n_gt_components"] for r in subj]),
    "size_cv (fragmentation)": np.array([r["size_cv"] for r in subj]),
    "lesion_surface_to_volume": np.array([r["lesion_surface_to_volume"] for r in subj]),
    "mean_intensity": np.array([r["mean_intensity"] for r in subj]),
    "std_intensity": np.array([r["std_intensity"] for r in subj]),
    "a_comp_dice_variance": np.array([r["a_comp_dice_variance"] if r["a_comp_dice_variance"] is not None else np.nan for r in subj]),
}

print("=== SUBJECT-LEVEL: correlation of candidate features with delta_dice (D4only - A) ===")
print(f"(n={n})")
results_table = []
for name, vals in features.items():
    valid = ~np.isnan(vals) & ~np.isnan(delta)
    if valid.sum() < 10:
        continue
    r_p, p_p = stats.pearsonr(vals[valid], delta[valid])
    r_s, p_s = stats.spearmanr(vals[valid], delta[valid])
    results_table.append((name, r_p, p_p, r_s, p_s, valid.sum()))
    print(f"  {name:32s}: pearson r={r_p:+.3f} (p={p_p:.4f})  spearman rho={r_s:+.3f} (p={p_s:.4f})  n={valid.sum()}")

print("\n=== TOP-10 vs REMAINING-115: response phenotype comparison ===")
order = np.argsort(delta)[::-1]
top10_idx = order[:10]
rest_idx = order[10:]

for name, vals in features.items():
    valid_top = vals[top10_idx]
    valid_rest = vals[rest_idx]
    valid_top = valid_top[~np.isnan(valid_top)]
    valid_rest = valid_rest[~np.isnan(valid_rest)]
    if len(valid_top) < 3 or len(valid_rest) < 3:
        continue
    t_stat, p = stats.mannwhitneyu(valid_top, valid_rest, alternative="two-sided")
    print(f"  {name:32s}: top10 mean={valid_top.mean():.3f} (median={np.median(valid_top):.3f})  "
          f"rest115 mean={valid_rest.mean():.3f} (median={np.median(valid_rest):.3f})  MWU p={p:.4f}")

print(f"\nTop-10 subjects' delta_dice: {delta[top10_idx]}")
print(f"Top-10 subjects' dice_A (baseline): {features['dice_A (baseline)'][top10_idx]}")
print(f"Top-10 subjects' mean_component_size: {features['mean_component_size'][top10_idx]}")

# Partial correlation: does any feature predict delta_dice AFTER controlling for
# mean_component_size and dice_A (the two "obvious confounders")?
print("\n=== Partial correlations controlling for mean_component_size + dice_A ===")
def partial_corr(x, y, controls):
    # residualize x and y against controls via OLS, then correlate residuals
    X = np.column_stack([np.ones(len(x))] + [c for c in controls])
    def resid(v):
        coef, _, _, _ = np.linalg.lstsq(X, v, rcond=None)
        return v - X @ coef
    return stats.pearsonr(resid(x), resid(y))

controls = [features["mean_component_size"], features["dice_A (baseline)"]]
for name, vals in features.items():
    if name in ("mean_component_size", "dice_A (baseline)"):
        continue
    valid = ~np.isnan(vals) & ~np.isnan(delta)
    for c in controls:
        valid &= ~np.isnan(c)
    if valid.sum() < 15:
        continue
    ctrl_valid = [c[valid] for c in controls]
    r, p = partial_corr(vals[valid], delta[valid], ctrl_valid)
    print(f"  {name:32s}: partial r={r:+.3f} (p={p:.4f}), n={valid.sum()}")

# ---------- COMPONENT LEVEL ----------
print("\n\n=== COMPONENT-LEVEL: delta_comp_dice predictors (components detected by BOTH A and D4only) ===")
comp_both = [c for c in comp if c.get("delta_comp_dice") is not None]
print(f"n components detected by both = {len(comp_both)} / {len(comp)} total")

comp_delta = np.array([c["delta_comp_dice"] for c in comp_both])
comp_features = {
    "size": np.array([c["size"] for c in comp_both], dtype=float),
    "a_comp_dice (baseline difficulty)": np.array([c["a_comp_dice"] for c in comp_both]),
    "surface_to_volume": np.array([c["surface_to_volume"] if c["surface_to_volume"] is not None else np.nan for c in comp_both]),
    "mean_intensity": np.array([c["mean_intensity"] for c in comp_both]),
    "std_intensity": np.array([c["std_intensity"] for c in comp_both]),
    "dist_from_center": np.array([c["dist_from_center"] for c in comp_both]),
    "isolation": np.array([c["isolation"] if c["isolation"] is not None else np.nan for c in comp_both]),
}

for name, vals in comp_features.items():
    valid = ~np.isnan(vals) & ~np.isnan(comp_delta)
    if valid.sum() < 10:
        continue
    r_p, p_p = stats.pearsonr(vals[valid], comp_delta[valid])
    r_s, p_s = stats.spearmanr(vals[valid], comp_delta[valid])
    print(f"  {name:36s}: pearson r={r_p:+.3f} (p={p_p:.4f})  spearman rho={r_s:+.3f} (p={p_s:.4f})  n={valid.sum()}")

print("\n=== Component-level partial correlations controlling for size + a_comp_dice ===")
c_controls = [comp_features["size"], comp_features["a_comp_dice (baseline difficulty)"]]
for name, vals in comp_features.items():
    if name in ("size", "a_comp_dice (baseline difficulty)"):
        continue
    valid = ~np.isnan(vals) & ~np.isnan(comp_delta)
    for c in c_controls:
        valid &= ~np.isnan(c)
    if valid.sum() < 15:
        continue
    ctrl_valid = [c[valid] for c in c_controls]
    r, p = partial_corr(vals[valid], comp_delta[valid], ctrl_valid)
    print(f"  {name:36s}: partial r={r:+.3f} (p={p:.4f}), n={valid.sum()}")

# variance explained by best single predictor + size/dice_A jointly (R^2)
print("\n=== Joint linear model: delta_dice ~ mean_component_size + dice_A (R^2) ===")
X = np.column_stack([np.ones(n), features["mean_component_size"], features["dice_A (baseline)"]])
coef, _, _, _ = np.linalg.lstsq(X, delta, rcond=None)
pred = X @ coef
ss_res = np.sum((delta - pred) ** 2)
ss_tot = np.sum((delta - delta.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"  R^2 (size + baseline Dice only) = {r2:.4f}")
