"""
Phase E49: CCABA calibration -- fits the FIXED, measured size-dependency
function used by UNet3D_v6's ccaba_w(), directly from E48's real causal
audit data (experiments/exp_e12_eggo_m/e48/E48_encoding_audit_table.json,
n=125 validation subjects, bottleneck-ablation Dice-drop vs. native_size).

This is the auditable, reproducible calibration step this project always
keeps separate from the architecture file itself (matching E45's own
calibrate_lambda_d8.py convention) -- the constants baked into
neuroscan_3d_v6.py (CCABA_B0, CCABA_B1, CCABA_W_MIN, CCABA_W_MAX) are
regenerated here and printed for verification; running this script
should reproduce those exact values.

FIT CHOICE: log-linear OLS (drop ~ b0 + b1*log(native_size+1)), chosen
over isotonic regression for two reasons, both disclosed:
  1. Isotonic regression's R^2 (0.359) was only marginally higher than
     log-linear's (0.260) on this data, and isotonic's step-function
     shape would need index lookups / non-differentiable clipping inside
     the model, whereas log-linear is a smooth closed form.
  2. The log-linear form is monotone-decreasing by construction over
     the entire positive domain (b1 < 0), matching the causally-observed
     direction (E48: rho=-0.454, smaller lesions -> larger drop) without
     any risk of a non-monotone artifact outside the observed data range
     (isotonic regression is only guaranteed monotone WITHIN the fitted
     range; log-linear extrapolates monotonically by construction).
Both fits are reported below for transparency -- the choice is
disclosed, not hidden.
"""
import json
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
E48_TABLE = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"

TOTAL_VOXELS = 240 * 240 * 155  # native BraTS volume size


def main():
    records = json.load(open(E48_TABLE))
    sizes = np.array([r["native_size"] for r in records], dtype=np.float64)
    drops = np.array([r["drop"] for r in records], dtype=np.float64)
    n = len(records)
    print(f"Loaded {n} subject records from E48's causal audit.")
    print(f"native_size range: [{sizes.min():.0f}, {sizes.max():.0f}]")
    print(f"drop range: [{drops.min():.4f}, {drops.max():.4f}]")

    # Log-linear OLS fit: drop = b0 + b1*log(size+1)
    log_size = np.log(sizes + 1.0)
    A = np.column_stack([np.ones_like(log_size), log_size])
    beta, *_ = np.linalg.lstsq(A, drops, rcond=None)
    b0, b1 = beta
    fitted = A @ beta
    ss_res = np.sum((drops - fitted) ** 2)
    ss_tot = np.sum((drops - drops.mean()) ** 2)
    r2_loglin = 1 - ss_res / ss_tot
    print(f"\nLog-linear fit: drop = {b0:.6f} + {b1:.6f} * log(size+1)   R^2={r2_loglin:.4f}")

    # Isotonic regression, for comparison/disclosure only (not used in the model)
    try:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(sizes, drops)
        fitted_iso = iso.predict(sizes)
        ss_res_iso = np.sum((drops - fitted_iso) ** 2)
        r2_iso = 1 - ss_res_iso / ss_tot
        print(f"Isotonic fit (comparison only, NOT used): R^2={r2_iso:.4f}")
    except ImportError:
        print("sklearn not available -- skipping isotonic comparison (non-critical, log-linear is what's used).")

    w_min = float(drops.min())
    w_max = float(drops.max())

    print(f"\n=== Constants to bake into neuroscan_3d_v6.py ===")
    print(f"CCABA_B0 = {b0:.6f}")
    print(f"CCABA_B1 = {b1:.6f}")
    print(f"CCABA_W_MIN = {w_min:.6f}")
    print(f"CCABA_W_MAX = {w_max:.6f}")
    print(f"CCABA_TOTAL_VOXELS = {TOTAL_VOXELS}")

    # Sanity table
    print(f"\n=== w(frac) at representative fractional occupancies ===")
    for f in [0.0008, 0.002, 0.005, 0.01, 0.02, 0.025]:
        s = f * TOTAL_VOXELS
        w = np.clip(b0 + b1 * np.log(s + 1), w_min, w_max)
        print(f"  frac={f:.4f} (size~{s:.0f}): w={w:.4f}")

    out = {
        "n_subjects": n,
        "loglinear_b0": float(b0), "loglinear_b1": float(b1), "loglinear_r2": float(r2_loglin),
        "w_min": w_min, "w_max": w_max, "total_voxels": TOTAL_VOXELS,
    }
    out_path = Path(__file__).parent / "E49_ccaba_calibration.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
