"""
Phase E72: mechanism-consistency check comparing FWL_seed0 vs
MM_baseline_pilot_seed0 (both 12 epochs, seed 0, identical recipe except
FWL's per-subject loss weighting), per the pre-declared decision rule in
PHASE_E72_FWL_DESIGN.md.

Pre-declared prediction: per-subject delta (FWL - baseline) should
correlate POSITIVELY with hat_d_i if the mechanism concentrates benefit
on causally-fragile subjects, regardless of what the mean does.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

OUT_DIR = Path(__file__).parent

fwl = json.load(open(OUT_DIR / "runs" / "FWL_seed0" / "result.json"))
base = json.load(open(OUT_DIR / "runs" / "MM_baseline_pilot_seed0" / "result.json"))
# The FWL pilot's own weight table only covers TRAINING subjects (that's what
# was actually weighted during training). For the mechanism-consistency check
# we need hat_d_i on the VAL subjects instead -- reusing E71's own persisted,
# validated table (same frozen scaled aux head, same held-out val population).
e71_val_table = json.load(open(OUT_DIR.parent / "e71" / "E71_val_per_subject_table_scaled.json"))
d_hat_val = {sid: v["predicted_d_i"] for sid, v in e71_val_table.items()}

fwl_dice = fwl["final_epoch_per_subject_dice"]
base_dice = base["final_epoch_per_subject_dice"]

common_ids = sorted(set(fwl_dice.keys()) & set(base_dice.keys()))
print(f"Matched val subjects: {len(common_ids)} (FWL n={len(fwl_dice)}, base n={len(base_dice)})")

deltas = np.array([fwl_dice[sid] - base_dice[sid] for sid in common_ids])
print(f"\nMean per-subject delta (FWL - baseline): {deltas.mean():+.4f}")
print(f"Median delta: {np.median(deltas):+.4f}")
print(f"Subjects improved: {(deltas > 0).sum()}/{len(deltas)} "
      f"({100*(deltas>0).mean():.1f}%)")

paired_t, p_t = stats.ttest_rel([fwl_dice[s] for s in common_ids], [base_dice[s] for s in common_ids])
wilcoxon_stat, p_w = stats.wilcoxon([fwl_dice[s] for s in common_ids], [base_dice[s] for s in common_ids])
print(f"\nPaired t-test: t={paired_t:.3f}, p={p_t:.4f}")
print(f"Wilcoxon signed-rank: p={p_w:.4f}")

val_ids_with_dhat = [sid for sid in common_ids if sid in d_hat_val]
print(f"\nVal subjects with a persisted hat_d_i (from E71's own scaled-predictor table): "
      f"{len(val_ids_with_dhat)}/{len(common_ids)}")

if len(val_ids_with_dhat) < 10:
    print("\n*** Insufficient overlap between the two val populations -- cannot run the "
          "mechanism-consistency check. Something is wrong (subject-id mismatch between "
          "E70/E71/E72 loaders) and needs investigation before trusting any comparison. ***")
else:
    d_arr = np.array([d_hat_val[sid] for sid in val_ids_with_dhat])
    delta_arr = np.array([fwl_dice[sid] - base_dice[sid] for sid in val_ids_with_dhat])

    rho, p_param = stats.spearmanr(d_arr, delta_arr)
    rng = np.random.default_rng(0)
    perm_rhos = np.empty(1000)
    for i in range(1000):
        perm_y = rng.permutation(delta_arr)
        perm_rhos[i], _ = stats.spearmanr(d_arr, perm_y)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())

    print(f"\n=== MECHANISM-CONSISTENCY CHECK ===")
    print(f"Spearman(hat_d_i, delta) = {rho:+.4f} (parametric p={p_param:.4e}, permutation p={p_perm:.4f})")
    print("Hypothesis (pre-declared): POSITIVE rho -- FWL should help causally-fragile "
          "(high hat_d_i) subjects specifically, regardless of what the mean does.")

    mechanism_consistent = (rho > 0) and (p_perm < 0.05)
    print(f"\nMechanism-consistency verdict: {'CONSISTENT' if mechanism_consistent else 'NOT CONSISTENT'}")

    mean_improved = deltas.mean() > 0
    print(f"\n=== FINAL PILOT VERDICT (per pre-declared rule in PHASE_E72_FWL_DESIGN.md) ===")
    if mean_improved and mechanism_consistent:
        print("Dice improved AND mechanism-consistent -- proceed to full 3-seed protocol.")
    elif not mean_improved and mechanism_consistent:
        print("Dice did NOT improve, but delta still tracks hat_d_i in the predicted direction -- "
              "the mechanism concentrates on fragile subjects as designed, but the training-time "
              "cost to easy subjects outweighs the benefit at this weighting strength/epoch count. "
              "This is a distinguishable failure mode from 'mechanism doesn't work' -- report as such.")
    elif mean_improved and not mechanism_consistent:
        print("Dice improved but NOT via the claimed mechanism -- report as an unexplained shift, "
              "NOT as FWL working as designed (same discipline that caught CAS's real vs intended mechanism).")
    else:
        print("Dice did NOT improve AND the mechanism is not consistent -- KILL. Report as a clean pilot null.")

    summary = {
        "n_matched": len(val_ids_with_dhat), "mean_delta": float(deltas.mean()),
        "paired_t": float(paired_t), "paired_t_p": float(p_t), "wilcoxon_p": float(p_w),
        "spearman_dhat_delta": float(rho), "spearman_p_param": float(p_param), "spearman_p_perm": p_perm,
        "mean_improved": bool(mean_improved), "mechanism_consistent": bool(mechanism_consistent),
    }
    with open(OUT_DIR / "E72_fwl_mechanism_consistency_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E72_fwl_mechanism_consistency_summary.json")
