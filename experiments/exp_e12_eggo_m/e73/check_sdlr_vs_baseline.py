"""
Phase E73: SDLR pilot vs matched MM baseline (E72's MM_baseline_pilot_seed0),
per-subject comparison + mechanism-consistency check (does the delta
correlate with the REAL per-subject error-burden the gate was meant to help,
i.e. subjects with more error voxels at baseline)?
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

E73_DIR = Path(__file__).parent
E72_DIR = E73_DIR.parent / "e72"

sdlr = json.load(open(E73_DIR / "runs" / "SDLR_seed0" / "result.json"))
base = json.load(open(E72_DIR / "runs" / "MM_baseline_pilot_seed0" / "result.json"))

sdlr_dice = sdlr["final_epoch_per_subject_dice"]
base_dice = base["final_epoch_per_subject_dice"]

common_ids = sorted(set(sdlr_dice.keys()) & set(base_dice.keys()))
print(f"Matched val subjects: {len(common_ids)} (SDLR n={len(sdlr_dice)}, base n={len(base_dice)})")

deltas = np.array([sdlr_dice[sid] - base_dice[sid] for sid in common_ids])
base_vals = np.array([base_dice[sid] for sid in common_ids])

print(f"\nMean per-subject delta (SDLR - baseline): {deltas.mean():+.4f}")
print(f"Median delta: {np.median(deltas):+.4f}")
print(f"Subjects improved: {(deltas > 0).sum()}/{len(deltas)} ({100*(deltas>0).mean():.1f}%)")

paired_t, p_t = stats.ttest_rel([sdlr_dice[s] for s in common_ids], [base_dice[s] for s in common_ids])
wilcoxon_stat, p_w = stats.wilcoxon([sdlr_dice[s] for s in common_ids], [base_dice[s] for s in common_ids])
print(f"\nPaired t-test: t={paired_t:.3f}, p={p_t:.4f}")
print(f"Wilcoxon signed-rank: p={p_w:.4f}")

# Mechanism check: does the gate help MORE on subjects with LOWER baseline
# Dice (i.e. harder/more error-prone subjects) -- the intended targeting?
rho, p_param = stats.spearmanr(-base_vals, deltas)  # -base_vals: higher = harder
rng = np.random.default_rng(0)
perm_rhos = np.empty(1000)
for i in range(1000):
    perm_y = rng.permutation(deltas)
    perm_rhos[i], _ = stats.spearmanr(-base_vals, perm_y)
p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())

print(f"\n=== MECHANISM-CONSISTENCY CHECK (delta vs baseline difficulty) ===")
print(f"Spearman(baseline difficulty [1-Dice], delta) = {rho:+.4f} (parametric p={p_param:.4e}, permutation p={p_perm:.4f})")
print("Hypothesis: POSITIVE rho -- SDLR should help harder (lower baseline Dice) subjects more.")

mean_improved = deltas.mean() > 0
significant = p_t < 0.05
mechanism_consistent = (rho > 0) and (p_perm < 0.05)

print(f"\n=== FINAL VERDICT ===")
print(f"Mean Dice change: {deltas.mean():+.4f} ({'significant' if significant else 'NOT significant'}, paired t p={p_t:.4f})")
print(f"Mechanism-consistent (helps harder subjects more): {'YES' if mechanism_consistent else 'NO'}")

if not significant and not mechanism_consistent:
    print("\nKILL: no detectable Dice effect and no mechanism-consistent targeting. Consistent with the "
          "pre-registered expectation that the weak spatial signal (ratio ~1.2x) would likely be "
          "insufficient to move Dice through this refinement mechanism.")
elif significant and mean_improved:
    print("\nDice improved significantly -- worth a 3-seed follow-up before any claim.")
elif not significant and mechanism_consistent:
    print("\nNo significant mean effect, but delta DOES track baseline difficulty as intended -- "
          "a directionally consistent but underpowered signal at 1 seed/12 epochs.")
else:
    print("\nMixed/inconclusive at this single-seed pilot scale.")

summary = {
    "n_matched": len(common_ids), "mean_delta": float(deltas.mean()),
    "paired_t": float(paired_t), "paired_t_p": float(p_t), "wilcoxon_p": float(p_w),
    "spearman_difficulty_delta": float(rho), "spearman_p_param": float(p_param), "spearman_p_perm": p_perm,
    "mean_improved": bool(mean_improved), "significant": bool(significant),
    "mechanism_consistent": bool(mechanism_consistent),
}
with open(E73_DIR / "E73_sdlr_vs_baseline_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved E73_sdlr_vs_baseline_summary.json")
