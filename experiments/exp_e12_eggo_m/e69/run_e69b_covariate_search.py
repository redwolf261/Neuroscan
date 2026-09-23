"""
Phase E69b: Covariate search for E54's cross-seed-consistent subject effect.

CONTEXT: E69 found a real, non-noise, cross-seed-consistent subject-level
effect (41.6% of subjects improve across ALL 3 independently-trained E54
seeds vs. 25% chance, binomial p<0.0001) -- but the pooled size-
correlation that initially looked like an explanation does NOT survive a
direct check (consistently-improved subjects' median lesion size is
LARGER, not smaller, than consistently-degraded subjects', p=0.64, no
significant per-seed size correlation individually). This phase searches
OTHER already-computed subject-level covariates (zero new training,
zero new inference) to see whether something else explains which
subjects reliably benefit from A96 resolution.

PRE-REGISTERED CANDIDATE COVARIATES (declared before viewing results),
all reused verbatim from existing project artifacts, none re-derived:
  1. Baseline Dice (E48's dice_intact, seed0/v5 checkpoint) -- do subjects
     that are already hard for the baseline benefit more/less from A96?
  2. Bottleneck causal drop (E48's drop field) -- does A96 help subjects
     whose segmentation depends heavily on the coarse/bottleneck pathway?
  3. Lesion component count (E35's component roster, aggregated per
     subject) -- does fragmentation/multiplicity predict benefit?
  4. Native lesion size (already tested in E69, included here again for
     side-by-side comparison in one table).

Three-group comparison (consistently improved / consistently degraded /
mixed sign across the 3 E54 seeds, from E69's own grouping) via
Kruskal-Wallis (all 3 groups) + pairwise Mann-Whitney (improved vs.
degraded, the sharpest contrast), for each covariate independently.
"""
import json
from pathlib import Path
from collections import Counter

import numpy as np
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
OUT_DIR = Path(__file__).parent

E56_TABLE = project_root / "experiments" / "exp_e12_eggo_m" / "e56" / "E56_per_subject_rescoring.json"
SEED12_TABLE = project_root / "experiments" / "exp_e12_eggo_m" / "e56" / "E45_E54_seed12_rescoring.json"
E48_TABLE = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"
E35_ROSTER = project_root / "experiments" / "exp_e12_eggo_m" / "e35" / "E35_component_roster.json"


def main():
    with open(E56_TABLE) as f:
        e56 = json.load(f)
    with open(SEED12_TABLE) as f:
        seed12 = json.load(f)
    with open(E48_TABLE) as f:
        e48_records = json.load(f)
    with open(E35_ROSTER) as f:
        e35_roster = json.load(f)

    native_size = {r["subject_id"]: r["native_size"] for r in e48_records}
    dice_intact = {r["subject_id"]: r["dice_intact"] for r in e48_records}
    bottleneck_drop = {r["subject_id"]: r["drop"] for r in e48_records}

    component_count = Counter(r["subject_id"] for r in e35_roster)

    baseline = e56["baseline_A"]["per_subject_dice"]
    e54 = {0: e56["e54_A96"]["per_subject_dice"], 1: seed12["e54_A96_s1"]["per_subject_dice"],
           2: seed12["e54_A96_s2"]["per_subject_dice"]}
    subject_ids = sorted(baseline.keys())

    # Reconstruct E69's own groups
    improved, degraded, mixed = [], [], []
    for sid in subject_ids:
        d = [e54[s][sid] - baseline[sid] for s in (0, 1, 2)]
        signs = set(np.sign(d))
        if signs == {1.0}:
            improved.append(sid)
        elif signs == {-1.0}:
            degraded.append(sid)
        else:
            mixed.append(sid)

    print(f"Groups: improved={len(improved)}, degraded={len(degraded)}, mixed={len(mixed)}\n")

    covariates = {
        "native_size": native_size,
        "baseline_dice_intact": dice_intact,
        "bottleneck_causal_drop": bottleneck_drop,
        "component_count": {sid: component_count.get(sid, 0) for sid in subject_ids},
    }

    results = {}
    for cov_name, cov_dict in covariates.items():
        vals_improved = [cov_dict[sid] for sid in improved if sid in cov_dict]
        vals_degraded = [cov_dict[sid] for sid in degraded if sid in cov_dict]
        vals_mixed = [cov_dict[sid] for sid in mixed if sid in cov_dict]

        h_stat, kw_p = stats.kruskal(vals_improved, vals_degraded, vals_mixed)
        u_stat, mwu_p = stats.mannwhitneyu(vals_improved, vals_degraded, alternative="two-sided")

        print(f"=== {cov_name} ===")
        print(f"  improved: median={np.median(vals_improved):.4f} (n={len(vals_improved)})")
        print(f"  degraded: median={np.median(vals_degraded):.4f} (n={len(vals_degraded)})")
        print(f"  mixed:    median={np.median(vals_mixed):.4f} (n={len(vals_mixed)})")
        print(f"  Kruskal-Wallis (3 groups): p={kw_p:.4f}")
        print(f"  Mann-Whitney (improved vs degraded): p={mwu_p:.4f}\n")

        results[cov_name] = {
            "median_improved": float(np.median(vals_improved)), "median_degraded": float(np.median(vals_degraded)),
            "median_mixed": float(np.median(vals_mixed)),
            "kruskal_wallis_p": float(kw_p), "mannwhitney_improved_vs_degraded_p": float(mwu_p),
        }

    any_significant = any(r["kruskal_wallis_p"] < 0.05 or r["mannwhitney_improved_vs_degraded_p"] < 0.05
                           for r in results.values())
    print(f"=== Any pre-registered covariate significantly distinguishes the groups: {any_significant} ===")

    with open(OUT_DIR / "E69b_covariate_search_summary.json", "w") as f:
        json.dump({"n_improved": len(improved), "n_degraded": len(degraded), "n_mixed": len(mixed),
                    "covariate_results": results, "any_significant": any_significant}, f, indent=2)
    print("\nSaved E69b_covariate_search_summary.json")


if __name__ == "__main__":
    main()
