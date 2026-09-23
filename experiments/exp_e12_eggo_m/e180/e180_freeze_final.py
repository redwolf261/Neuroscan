"""
E180 final freeze -- Stages 0-11 complete and terminated per explicit user
directive ("freeze E180 now, do not touch the successful experiment").

Writes experiments/exp_e12_eggo_m/e180/FROZEN/MANIFEST.json following the
e169/FROZEN/MANIFEST.json convention: per-file sha256_16 + bytes + headline
summary + note. Copies every _full-suffixed result file (the decisive
125-subject-cohort artifacts) into FROZEN/.
"""
import json
import hashlib
import shutil
from pathlib import Path

OUT_DIR = Path(__file__).parent
FROZEN_DIR = OUT_DIR / "FROZEN"

# The decisive, full-cohort files -- what any future stage or write-up must
# read from FROZEN/, never regenerate.
FREEZE_FILES = [
    "E180_tile_ledger_full.json",
    "E180_1_sanity_summary_full.json",
    "E180_2_severity_calibration.json",
    "E180_2b_probe_calibration.json",
    "E180_gamma_per_subject_full.json",
    "E180_3_4_gamma_summary_full.json",
    "E180_delta_per_subject_full.json",
    "E180_5_delta_summary_full.json",
    "E180_5_5_pipeline_gate_summary.json",
    "E180_6_primary_test.json",
    "E180_7_competitor_features.json",
    "E180_8_incremental_r2.json",
    "E180_9_boundary_distance.json",
    "E180_9_boundary_control.json",
    "E180_10_dose_response.json",
    "E180_11_coupling_test.json",
]


def sha256_16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for fname in FREEZE_FILES:
        src = OUT_DIR / fname
        if not src.exists():
            print(f"[WARN] missing {fname}, skipping")
            continue
        dst = FROZEN_DIR / fname
        shutil.copy2(src, dst)
        manifest[fname] = {"sha256_16": sha256_16(src), "bytes": src.stat().st_size}
        print(f"  froze {fname}  ({manifest[fname]['bytes']} bytes, "
              f"{manifest[fname]['sha256_16']})")

    # headline result, read directly from the frozen files (not re-derived)
    stage6 = json.load(open(OUT_DIR / "E180_6_primary_test.json"))
    stage8 = json.load(open(OUT_DIR / "E180_8_incremental_r2.json"))
    stage9 = json.load(open(OUT_DIR / "E180_9_boundary_control.json"))
    stage11 = json.load(open(OUT_DIR / "E180_11_coupling_test.json"))

    headline = {
        "T1_rank": "KILLED at Stage 6 primary gate (full-125 p=0.112, E167-110 p=0.102)",
        "T4_spectral": {
            "stage6_full125": {"beta": stage6["A_full_125_primary"]["T4_spectral"]
                               ["primary_subject_fixed_effects"]["beta_gamma"],
                               "p": stage6["A_full_125_primary"]["T4_spectral"]
                               ["primary_subject_fixed_effects"]["p"]},
            "stage8_gamma_over_uncertainty_dR2": stage8["A_full_125"]["T4_spectral"]
                ["delta_R2_gamma_over_full_M2"],
            "stage9_classification": stage9["A_full_125"]["T4_spectral"]["CLASSIFICATION"],
            "stage11_verdict_full125": stage11["A_full_125"]["PREREGISTERED_VERDICT"],
            "stage11_verdict_E167_110": stage11["B_E167_110"]["PREREGISTERED_VERDICT"],
        },
        "T5_smooth": "HELD -- stratum-dependent (full-125 p=0.124, E167-110 p<0.0001), not advanced",
    }

    manifest_out = {
        "files": manifest,
        "HEADLINE": headline,
        "note": ("FROZEN 2026-09-17. E180 Stages 0-11 COMPLETE and TERMINATED per explicit "
                 "user directive. T4_spectral is the sole surviving family: demonstrated "
                 "graded, bidirectional, counterfactual coupling (Stage 10-11 'STRONGEST' "
                 "classification in both the full-125 cohort and the predefined E167 "
                 "110-subject stratum). T1_rank killed at Stage 6. T5_smooth held, "
                 "stratum-dependent. Novelty audit (PHASE_E180_NOVELTY_AUDIT.md) found the "
                 "instability-guided-refinement application shape occupied (TRUST, "
                 "CertainTTA); any future claim rests on Gamma's specific construction. "
                 "Stage 12 (an algorithm) is explicitly NOT authorized by this result -- next "
                 "phase is mechanism dissection of WHY spectral reshaping specifically "
                 "produces this coupling while rank truncation and smoothing do not. "
                 "Do NOT regenerate these files; read from FROZEN/."),
    }
    with open(FROZEN_DIR / "MANIFEST.json", "w") as f:
        json.dump(manifest_out, f, indent=1)
    print(f"\nWrote {FROZEN_DIR / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
