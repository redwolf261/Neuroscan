"""
Phase E35-C (prompt-labeled "E34-C"): critical small-lesion analysis.
NO TRAINING. Read-only analysis of E35_component_roster.json (E34-A) and
E35_target_family_stats.json (E34-B).

This is the most important section of the audit: does contextual support
survive at coarse decoder scales for the small-lesion population where
exact supervision becomes spatially impoverished?

Pre-declared groups (from the prompt, on NATIVE size, fixed before looking
at results):
    S1: native size 1-5
    S2: 6-20
    S3: 21-50
    S4: 51-150
    S5: >150

Also separately analyzed: the post-resize 64^3 size bins used in E25's own
4-way comparison (experiments/exp_e12_eggo_m/e25/analyze_4way_comparison.py,
SIZE_BINS = [0, 50, 150, 400, 1000, inf] on gt_size AT 64^3, SIZE_LABELS =
["1-50","50-150","150-400","400-1000",">1000"]) -- reused verbatim (not
re-invented) per the prompt's explicit instruction, applied here to this
audit's own `size_64` field.

For each group:
    1. Fraction of exact target surviving at D4 (mean support_nonzero / res^3,
       and separately: fraction of COMPONENTS with any nonzero exact support)
    2. Same at D2
    3. Fraction of contextual target MASS surviving (context mass relative to
       native mass -- see mass-survival definition below)
    4. Fraction of components with ZERO exact target voxels (at D4, D2)
    5. Fraction of those zero-exact components with NONZERO contextual support
    6. How much contextual support exists when the exact target has vanished
       (context mass/support restricted to the zero-exact subset)

The key quantity, the "contextual rescue ratio" = support(C_s) / support(Y_s),
is NOT computed as a raw ratio when support(Y_s) is at or near zero -- per
explicit instruction, this is reported SEPARATELY (as a rescue count/fraction
among zero-exact-support components) rather than manufactured via epsilon
division. A numerical floor (MIN_SUPPORT_FOR_RATIO = 1 voxel) is declared
before computing anything; below the floor, the ratio is reported as
"undefined (denominator below floor)", not silently substituted.

"RESCUE" DEFINITION -- REVISED AFTER A FIRST RESULT LOOKED SUSPICIOUSLY CLEAN
AND WAS INVESTIGATED BEFORE BEING TRUSTED (per the project's own standing
rule not to report a too-clean result without checking for a mundane
measurement artifact first):

  A first pass defined "rescued" as support_nonzero >= 1 voxel (i.e. any
  voxel with a value strictly greater than 0 anywhere after a Gaussian blur).
  This produced a 100% rescue rate in EVERY size bin at EVERY sigma with no
  exceptions -- checked directly rather than accepted, and found to be a
  measurement-threshold artifact, not a real finding: for a representative
  zero-exact S1 component, `support_gt_0p1` and `support_gt_0p01` (fields
  E35-B already computes) were BOTH ZERO at every sigma, while the max value
  anywhere in the field was ~0.0008 -- i.e. the entire "rescued" region was a
  numerical Gaussian tail with values near float precision, not a
  spatially meaningful contextual signal. A Gaussian kernel's support is
  mathematically infinite (never exactly zero except where hard-truncated),
  so "any nonzero value" was guaranteed to be satisfied almost everywhere,
  making the >=1-voxel threshold a trivially-true tautology rather than an
  informative measurement.

  Fixed by redefining "rescued" as support_gt_0p1 >= MIN_SUPPORT_FOR_RATIO --
  i.e. at least one voxel where the context field's value exceeds 0.1 (10%
  contribution, the same "effective support" threshold E35-B itself already
  computed and flagged as meaningful, one of three pre-declared effective-
  support levels: >0.01, >0.1, >0.5). This is a real, disclosed, non-outcome-
  driven threshold choice (chosen because it was already one of E35-B's own
  pre-declared summary statistics, not picked after seeing which threshold
  gave a nicer number).
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

OUT_DIR = Path(__file__).parent
ROSTER = OUT_DIR / "E35_component_roster.json"
TARGET_STATS = OUT_DIR / "E35_target_family_stats.json"

NATIVE_BINS = [(1, 5, "S1"), (6, 20, "S2"), (21, 50, "S3"), (51, 150, "S4"), (151, float("inf"), "S5")]

# Reused VERBATIM from e25/analyze_4way_comparison.py (SIZE_BINS/SIZE_LABELS),
# applied here to size_64 rather than gt_size (E27's own field name for the
# same quantity in its own resized-component convention).
SIZE64_BIN_EDGES = [0, 50, 150, 400, 1000, float("inf")]
SIZE64_BIN_LABELS = ["1-50", "50-150", "150-400", "400-1000", ">1000"]

MIN_SUPPORT_FOR_RATIO = 1  # voxels; declared BEFORE computing anything, per the "no epsilon rescue" rule
RES_VOL = {"D4": 16 ** 3, "D2": 32 ** 3, "D1": 64 ** 3}


def native_bin(size):
    for lo, hi, lab in NATIVE_BINS:
        if lo <= size <= hi:
            return lab
    return None


def size64_bin(size64):
    for lo, hi, lab in zip(SIZE64_BIN_EDGES[:-1], SIZE64_BIN_EDGES[1:], SIZE64_BIN_LABELS):
        if lo < size64 <= hi if lo > 0 else size64 <= hi:
            return lab
    return None


def main():
    roster = {(r["subject_id"], r["native_component_id"]): r for r in json.load(open(ROSTER))}
    stats = json.load(open(TARGET_STATS))

    # Index target-family stats by (key, scale)
    by_key_scale = {}
    for rec in stats:
        key = (rec["subject_id"], rec["native_component_id"])
        by_key_scale[(key, rec["scale"])] = rec

    results_native = {}
    results_size64 = {}

    for bin_scheme_name, bin_fn, bin_labels in [
        ("native", native_bin, [b[2] for b in NATIVE_BINS]),
        ("size64", size64_bin, SIZE64_BIN_LABELS),
    ]:
        results = {}
        for lab in bin_labels:
            results[lab] = {
                "n_components": 0,
                "D4": defaultdict(list), "D2": defaultdict(list), "D1": defaultdict(list),
            }

        for key, r in roster.items():
            group_val = r["native_size"] if bin_scheme_name == "native" else r["size_64"]
            lab = bin_fn(group_val)
            if lab is None:
                continue
            results[lab]["n_components"] += 1

            for scale in ("D4", "D2", "D1"):
                rec = by_key_scale.get((key, scale))
                if rec is None:
                    continue
                res_vol = RES_VOL[scale]
                exact_support = rec["exact"]["support_nonzero"]
                exact_mass = rec["exact"]["mass"]
                zero_exact = exact_support == 0

                d = results[lab][scale]
                d["exact_support_frac"].append(exact_support / res_vol)
                d["exact_nonzero"].append(not zero_exact)
                d["zero_exact"].append(zero_exact)

                # Per-sigma context stats (Gaussian only for the core rescue
                # question here; dilation-vs-Gaussian comparison is E35-E)
                for sg in rec["sigma_grid"]:
                    sigma = sg["sigma_target_vox"]
                    g = sg["gaussian"]
                    d[f"gauss_support_frac_sigma{sigma}"].append(g["support_nonzero"] / res_vol)
                    d[f"gauss_mass_sigma{sigma}"].append(g["mass"])
                    if zero_exact:
                        d[f"gauss_support_frac_sigma{sigma}_when_zero_exact"].append(g["support_nonzero"] / res_vol)
                        d[f"gauss_mass_sigma{sigma}_when_zero_exact"].append(g["mass"])
                        # Real rescue criterion (see module docstring): a
                        # meaningful (>0.1) context value must exist, not
                        # merely a nonzero float.
                        d[f"gauss_rescued_sigma{sigma}"].append(g["support_gt_0p1"] >= MIN_SUPPORT_FOR_RATIO)
                        # Debunked/trivial criterion, KEPT for transparency
                        # (shown alongside the real one in the report so the
                        # correction is auditable, not silently swapped).
                        d[f"gauss_rescued_trivial_sigma{sigma}"].append(g["support_nonzero"] >= MIN_SUPPORT_FOR_RATIO)
                        d[f"gauss_rescued_strict_sigma{sigma}"].append(g["support_gt_0p01"] >= MIN_SUPPORT_FOR_RATIO)

        # Summarize
        summary = {}
        for lab in bin_labels:
            n = results[lab]["n_components"]
            entry = {"n_components": n}
            for scale in ("D4", "D2", "D1"):
                d = results[lab][scale]
                if not d:
                    entry[scale] = None
                    continue
                n_zero_exact = sum(d["zero_exact"]) if "zero_exact" in d else None
                scale_entry = {
                    "mean_exact_support_frac": float(np.mean(d["exact_support_frac"])) if d.get("exact_support_frac") else None,
                    "frac_components_nonzero_exact": float(np.mean(d["exact_nonzero"])) if d.get("exact_nonzero") else None,
                    "frac_components_zero_exact": float(np.mean(d["zero_exact"])) if d.get("zero_exact") else None,
                    "n_zero_exact_components": int(n_zero_exact) if n_zero_exact is not None else None,
                    "sigma_grid": {},
                }
                for sigma in [0.5, 1.0, 1.5, 2.0]:
                    key_supp = f"gauss_support_frac_sigma{sigma}"
                    key_mass = f"gauss_mass_sigma{sigma}"
                    key_supp_ze = f"gauss_support_frac_sigma{sigma}_when_zero_exact"
                    key_mass_ze = f"gauss_mass_sigma{sigma}_when_zero_exact"
                    key_rescued = f"gauss_rescued_sigma{sigma}"
                    key_rescued_trivial = f"gauss_rescued_trivial_sigma{sigma}"
                    key_rescued_strict = f"gauss_rescued_strict_sigma{sigma}"

                    def rescue_stats(k):
                        lst = d.get(k, [])
                        n_r = int(sum(lst)) if lst else 0
                        n_t = len(lst)
                        return n_r, n_t, (n_r / n_t if n_t > 0 else None)

                    n_rescued, n_ze, frac_rescued = rescue_stats(key_rescued)
                    n_rescued_triv, _, frac_rescued_triv = rescue_stats(key_rescued_trivial)
                    n_rescued_strict, _, frac_rescued_strict = rescue_stats(key_rescued_strict)

                    scale_entry["sigma_grid"][str(sigma)] = {
                        "mean_gauss_support_frac_all": float(np.mean(d[key_supp])) if d.get(key_supp) else None,
                        "mean_gauss_mass_all": float(np.mean(d[key_mass])) if d.get(key_mass) else None,
                        "mean_gauss_support_frac_when_zero_exact": float(np.mean(d[key_supp_ze])) if d.get(key_supp_ze) else None,
                        "mean_gauss_mass_when_zero_exact": float(np.mean(d[key_mass_ze])) if d.get(key_mass_ze) else None,
                        # "Contextual rescue" reported as a COUNT/FRACTION among
                        # zero-exact-support components (denominator is n_ze,
                        # the count of GENUINELY zero-exact components, never
                        # a near-zero continuous quantity) -- NOT support(C)/support(Y)
                        # as a raw ratio, since support(Y)=0 there by construction
                        # (this is exactly the "don't manufacture a ratio via
                        # epsilon division" case the prompt warns against).
                        # PRIMARY criterion: support_gt_0p1 (meaningful >10% context value).
                        "n_zero_exact_with_nonzero_context": n_rescued,
                        "n_zero_exact_total": n_ze,
                        "frac_zero_exact_rescued_by_context": frac_rescued,
                        # DIAGNOSTIC criteria, kept for transparency (see module docstring):
                        "frac_rescued_TRIVIAL_support_nonzero": frac_rescued_triv,  # debunked -- near-tautological
                        "frac_rescued_STRICT_support_gt_0p01": frac_rescued_strict,  # stricter alternative threshold
                    }
                entry[scale] = scale_entry
            summary[lab] = entry

        if bin_scheme_name == "native":
            results_native = summary
        else:
            results_size64 = summary

    output = {
        "min_support_for_ratio_floor": MIN_SUPPORT_FOR_RATIO,
        "note": "Contextual rescue is reported as (count, fraction) of zero-exact-support "
                "components that gain >=1 voxel of nonzero Gaussian-context support, NOT as a "
                "raw support(C)/support(Y) ratio (undefined/manufactured when support(Y)=0). "
                "size64 bins reused verbatim from e25/analyze_4way_comparison.py's own "
                "SIZE_BINS/SIZE_LABELS, applied to size_64 instead of E27's gt_size field.",
        "native_size_bins": results_native,
        "size_64_bins": results_size64,
    }

    with open(OUT_DIR / "E35_critical_small_lesion_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {OUT_DIR / 'E35_critical_small_lesion_results.json'}")

    # Print a compact human-readable summary for the key S1-S4 rows at D4
    print("\n=== Native-size bins, D4 scale, key rescue numbers ===")
    for lab in [b[2] for b in NATIVE_BINS]:
        e = results_native[lab]["D4"]
        if e is None:
            print(f"{lab}: no data")
            continue
        print(f"{lab} (n={results_native[lab]['n_components']}): "
              f"frac_zero_exact={e['frac_components_zero_exact']:.3f} "
              f"({e['n_zero_exact_components']} components)")
        for sigma in ["0.5", "1.0", "1.5", "2.0"]:
            sg = e["sigma_grid"][sigma]
            if sg["n_zero_exact_total"] > 0:
                triv = sg["frac_rescued_TRIVIAL_support_nonzero"]
                strict = sg["frac_rescued_STRICT_support_gt_0p01"]
                print(f"    sigma={sigma}: rescued(support>0.1) {sg['n_zero_exact_with_nonzero_context']}/{sg['n_zero_exact_total']} "
                      f"({sg['frac_zero_exact_rescued_by_context']:.3f})  "
                      f"[trivial(>0) rate={triv:.3f}, strict(>0.01) rate={strict:.3f}]")


if __name__ == "__main__":
    main()
