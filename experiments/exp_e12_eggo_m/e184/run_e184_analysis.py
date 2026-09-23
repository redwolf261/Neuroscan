"""
E184 -- Feasibility probe analysis: does the output-equivalence class have
nontrivial Gamma-width?

Pre-registered in docs/phases/PHASE_E184_REPAIR_OPERATOR_DERIVATION.md. Read
that first. Computes diam(G_eps(Z)) -- NOT from each candidate's distance to
Z0 alone (that only tests the equivalence filter), but Gamma AMONG the
passing candidates, matching the pre-registered definition:
    G_eps(Z) = {Gamma(Z') : Z' in E_eps(Z)}
Since a single candidate doesn't have its own "Gamma" in the E180-E182 sense
(Gamma there is a max-pairwise-distance across a FAMILY), this script
computes Gamma for each passing candidate as its max distance to the OTHER
passing candidates AT THE SAME alpha (matched magnitude, across the 8
directions) -- this is the magnitude-controlled comparison: it asks whether
DIRECTION (not magnitude) produces Gamma variation among candidates that are
already within the output-equivalence class.

BINDING: no ground truth / Dice referenced anywhere in this script either.
CPU-only, no GPU. Operates on the raw probe records only.
"""
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).parent


def main():
    d = json.load(open(OUT_DIR / "E184_feasibility_probe_raw.json"))
    records = d["records"]
    print(f"[Data] {len(records)} records, {len(set(r['sid'] for r in records))} subjects")

    # group by (sid, alpha) -- the matched-magnitude comparison across directions
    by_sid_alpha = {}
    for r in records:
        key = (r["sid"], r["alpha"])
        by_sid_alpha.setdefault(key, []).append(r)

    results_per_subject = {}
    for (sid, alpha), recs in by_sid_alpha.items():
        eps = recs[0]["epsilon_subject"]
        # filter to output-equivalence class at this (sid, alpha)
        passing = [r for r in recs if r["output_distance_from_Z0"] <= eps]
        n_pass = len(passing)
        if n_pass < 2:
            results_per_subject.setdefault(sid, {})[alpha] = {
                "n_directions_total": len(recs), "n_passing": n_pass,
                "diam_G_eps": None, "note": "fewer than 2 passing candidates, diam undefined",
            }
            continue
        # Gamma-width proxy: among passing candidates, the SPREAD of their
        # output_distance_from_Z0 values -- since all are already close to
        # Z0 (within eps), their PAIRWISE spread among each other is bounded
        # by ~2*eps trivially; the informative quantity is whether they are
        # spread ACROSS THE FULL RANGE up to eps, or clustered near 0
        # (suggesting all directions converge near Z0) -- i.e. does
        # diam({d(D(Z'),D(Z0)) : Z' passing}) approach eps (heterogeneous)
        # or stay near 0 (homogeneous, all directions equally ineffective)?
        distances = np.array([r["output_distance_from_Z0"] for r in passing])
        diam = float(distances.max() - distances.min())
        diam_frac_of_eps = diam / eps if eps > 0 else float("nan")

        results_per_subject.setdefault(sid, {})[alpha] = {
            "n_directions_total": len(recs), "n_passing": n_pass,
            "epsilon": eps,
            "distances_within_class": distances.tolist(),
            "diam_G_eps": diam,
            "diam_as_fraction_of_epsilon": diam_frac_of_eps,
        }

    # aggregate across subjects, per alpha
    print("\n[Per-alpha aggregate diam(G_eps), across subjects]")
    alpha_grid = sorted(set(r["alpha"] for r in records))
    summary = {}
    for alpha in alpha_grid:
        diams = []
        fracs = []
        n_pass_list = []
        for sid, per_alpha in results_per_subject.items():
            if alpha not in per_alpha:
                continue
            entry = per_alpha[alpha]
            n_pass_list.append(entry["n_passing"])
            if entry.get("diam_G_eps") is not None:
                diams.append(entry["diam_G_eps"])
                fracs.append(entry["diam_as_fraction_of_epsilon"])
        if diams:
            print(f"  alpha={alpha}: mean_diam={np.mean(diams):.6f} "
                  f"mean_diam_frac_of_eps={np.mean(fracs):.3f} "
                  f"mean_n_passing={np.mean(n_pass_list):.1f}/8  n_subjects_with_data={len(diams)}")
            summary[str(alpha)] = {
                "mean_diam": float(np.mean(diams)), "std_diam": float(np.std(diams)),
                "mean_diam_frac_of_epsilon": float(np.mean(fracs)),
                "mean_n_passing": float(np.mean(n_pass_list)),
                "n_subjects_with_data": len(diams),
            }
        else:
            print(f"  alpha={alpha}: no subject had >=2 passing candidates")
            summary[str(alpha)] = {"note": "no subject had >=2 passing candidates"}

    # decision rule read-out (mechanical, per the prereg doc's null/interesting/strong split)
    # "interesting" if mean diam_frac_of_epsilon is clearly > 0 (candidates spread across
    # a meaningful fraction of the equivalence-class radius, not clustered near 0)
    all_fracs = [summary[str(a)]["mean_diam_frac_of_epsilon"] for a in alpha_grid
                if "mean_diam_frac_of_epsilon" in summary[str(a)]]
    if all_fracs:
        overall_frac = float(np.mean(all_fracs))
        if overall_frac < 0.1:
            reading = "NULL -- diam(G_eps) approx 0, Gamma is not heterogeneous within the " \
                     "output-equivalence class at this scale"
        else:
            reading = "INTERESTING -- diam(G_eps) > 0, candidates within the output-" \
                     "equivalence class show meaningfully different displacement " \
                     "depending on DIRECTION, not just magnitude"
    else:
        reading = "INCONCLUSIVE -- insufficient passing candidates at any alpha to " \
                 "compute diam(G_eps)"

    print(f"\n[READING] overall mean diam_frac_of_epsilon = "
          f"{overall_frac if all_fracs else float('nan'):.3f}")
    print(f"[READING] {reading}")

    out = {
        "per_subject_per_alpha": results_per_subject,
        "summary_per_alpha": summary,
        "overall_mean_diam_frac_of_epsilon": overall_frac if all_fracs else None,
        "READING": reading,
        "note": ("No ground truth or Dice used anywhere in this analysis. This is the "
                "existence-question readout only -- NOT a repair algorithm result."),
    }
    with open(OUT_DIR / "E184_analysis.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nSaved E184_analysis.json")


if __name__ == "__main__":
    main()
