"""
Phase E43: Representation Change Audit -- raw-effect analysis.

NO TRAINING. Reads E43_representation_change_table.json (2961 rows: 125
subjects x 3 conditions x 3 decoder stages x up to 3 regions).

Per the explicit instruction for this phase: report the RAW magnitude/
delta-norm/cosine numbers FIRST, inspect where the largest and most
D4-vs-D2-differentiated changes are, and only consider permutation testing
AFTER that raw inspection -- do not construct any new explanatory scalar
before this step.
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

OUT_DIR = Path(__file__).parent
TABLE_PATH = OUT_DIR / "E43_representation_change_table.json"


def main():
    records = json.load(open(TABLE_PATH))

    # ================= Step 1: raw summary table, subject-averaged =================
    print("=== Step 1: RAW summary (mean across 125 subjects), by condition x stage x region ===\n", flush=True)
    grouped = defaultdict(list)
    for r in records:
        key = (r["condition"], r["decoder_stage"], r["region"])
        grouped[key].append(r)

    summary = {}
    for key, rows in sorted(grouped.items()):
        cond, stage, region = key
        mag_b = np.array([r["mean_mag_baseline"] for r in rows])
        mag_c = np.array([r["mean_mag_cond"] for r in rows])
        delta = np.array([r["mean_delta_norm"] for r in rows])
        cos = np.array([r["mean_cosine"] for r in rows])
        n_subj = len(rows)

        # Relative change: delta norm relative to baseline's own magnitude,
        # since a raw delta-norm alone conflates "large change" with "region
        # naturally has large-magnitude features regardless of condition."
        rel_delta = delta / (mag_b + 1e-9)

        summary[key] = {
            "n_subjects": n_subj,
            "mean_mag_baseline": float(mag_b.mean()), "mean_mag_cond": float(mag_c.mean()),
            "mean_delta_norm": float(delta.mean()), "sd_delta_norm": float(delta.std()),
            "mean_relative_delta": float(rel_delta.mean()), "sd_relative_delta": float(rel_delta.std()),
            "mean_cosine": float(cos.mean()), "sd_cosine": float(cos.std()),
        }
        print(f"  [{cond:8s} {stage:5s} {region:10s}] n={n_subj:3d}  "
              f"mag_base={mag_b.mean():7.3f}  mag_cond={mag_c.mean():7.3f}  "
              f"|delta|={delta.mean():7.3f}(+/-{delta.std():.2f})  "
              f"rel_delta={rel_delta.mean():.3f}  cos={cos.mean():+.3f}(+/-{cos.std():.3f})", flush=True)

    # ================= Step 2: decisive comparison -- Delta_D4 vs Delta_D2 at boundary =================
    print("\n=== Step 2: DECISIVE comparison -- relative change (D4 vs D2), boundary region, dec3/dec2 ===\n", flush=True)
    decisive = {}
    for stage in ("dec3", "dec2"):
        key_d4 = ("D4only", stage, "boundary")
        key_d2 = ("D2only", stage, "boundary")
        rows_d4 = grouped[key_d4]
        rows_d2 = grouped[key_d2]

        # Paired by subject_id (same scan, both conditions) -- verified match before comparing.
        d4_by_subj = {r["subject_id"]: r for r in rows_d4}
        d2_by_subj = {r["subject_id"]: r for r in rows_d2}
        common = sorted(set(d4_by_subj) & set(d2_by_subj))
        print(f"  [{stage}] paired subjects with boundary region in BOTH D4only and D2only: {len(common)}", flush=True)

        rel_delta_d4 = np.array([d4_by_subj[s]["mean_delta_norm"] / (d4_by_subj[s]["mean_mag_baseline"] + 1e-9) for s in common])
        rel_delta_d2 = np.array([d2_by_subj[s]["mean_delta_norm"] / (d2_by_subj[s]["mean_mag_baseline"] + 1e-9) for s in common])
        cos_d4 = np.array([d4_by_subj[s]["mean_cosine"] for s in common])
        cos_d2 = np.array([d2_by_subj[s]["mean_cosine"] for s in common])

        w_stat_delta, w_p_delta = stats.wilcoxon(rel_delta_d4, rel_delta_d2)
        w_stat_cos, w_p_cos = stats.wilcoxon(cos_d4, cos_d2)

        print(f"    Relative change at boundary: D4={rel_delta_d4.mean():.3f} vs D2={rel_delta_d2.mean():.3f}  "
              f"(paired Wilcoxon: stat={w_stat_delta:.1f}, p={w_p_delta:.4e})", flush=True)
        print(f"    Direction (cosine) at boundary: D4={cos_d4.mean():+.3f} vs D2={cos_d2.mean():+.3f}  "
              f"(paired Wilcoxon: stat={w_stat_cos:.1f}, p={w_p_cos:.4e})", flush=True)

        decisive[stage] = {
            "n_paired": len(common),
            "rel_delta_D4_mean": float(rel_delta_d4.mean()), "rel_delta_D2_mean": float(rel_delta_d2.mean()),
            "wilcoxon_rel_delta": {"stat": float(w_stat_delta), "p": float(w_p_delta)},
            "cos_D4_mean": float(cos_d4.mean()), "cos_D2_mean": float(cos_d2.mean()),
            "wilcoxon_cos": {"stat": float(w_stat_cos), "p": float(w_p_cos)},
        }

    # ================= Step 3: is the D4-vs-baseline change LOCALIZED to boundary specifically? =================
    print("\n=== Step 3: is D4's OWN change (vs baseline) concentrated at boundary specifically, or uniform across regions? ===\n", flush=True)
    localization = {}
    for stage in ("dec3", "dec2", "dec1"):
        regions_here = ["interior", "boundary", "background"] if stage != "dec1" else ["interior", "background"]
        vals = {}
        for region in regions_here:
            key = ("D4only", stage, region)
            if key not in grouped:
                continue
            rows = grouped[key]
            rel_delta = np.array([r["mean_delta_norm"] / (r["mean_mag_baseline"] + 1e-9) for r in rows])
            vals[region] = float(rel_delta.mean())
        localization[stage] = vals
        print(f"  [{stage}] D4-only relative change by region: {vals}", flush=True)

    # Statistical test: is boundary's relative change greater than interior's
    # AND greater than background's, paired by subject, for dec3/dec2?
    print("\n  Paired test: is boundary's relative change greater than interior's (D4-only, same subjects)?", flush=True)
    for stage in ("dec3", "dec2"):
        key_bound = ("D4only", stage, "boundary")
        key_int = ("D4only", stage, "interior")
        b_by_subj = {r["subject_id"]: r["mean_delta_norm"] / (r["mean_mag_baseline"] + 1e-9) for r in grouped[key_bound]}
        i_by_subj = {r["subject_id"]: r["mean_delta_norm"] / (r["mean_mag_baseline"] + 1e-9) for r in grouped[key_int]}
        common = sorted(set(b_by_subj) & set(i_by_subj))
        b_vals = np.array([b_by_subj[s] for s in common])
        i_vals = np.array([i_by_subj[s] for s in common])
        if len(common) > 5:
            w_stat, w_p = stats.wilcoxon(b_vals, i_vals)
            frac_greater = float((b_vals > i_vals).mean())
            print(f"    [{stage}] n={len(common)}  boundary>interior in {frac_greater:.1%} of subjects  "
                  f"(Wilcoxon stat={w_stat:.1f}, p={w_p:.4e})", flush=True)
            localization[f"{stage}_boundary_vs_interior_test"] = {"n": len(common), "frac_greater": frac_greater, "p": float(w_p)}

    results = {
        "raw_summary": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in summary.items()},
        "decisive_D4_vs_D2_boundary": decisive,
        "localization_D4_by_region": localization,
    }
    with open(OUT_DIR / "E43_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved E43_analysis_results.json", flush=True)


if __name__ == "__main__":
    main()
