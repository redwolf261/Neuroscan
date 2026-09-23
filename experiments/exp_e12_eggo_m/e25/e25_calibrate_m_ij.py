"""
Phase E25, Section 12: m_ij calibration for SC-TAM (Candidate 6, C6-2),
per PHASE_E25_CANDIDATE6_SC_TAM_DESIGN.md Section 11, locked decision 3:
"m_ij calibration: required, using the identical E12e/E24 methodology
(fresh-init model, .train() mode -- never .eval(), same 10-30%-active-
hinge-at-init target, same percentile-based derivation)".

Directly adapted from e24_calibrate_delta_d_w.py, which itself replicates
e12e_calibrate_constants.py's methodology. The ONE substantive difference
from E24's delta_d_w calibration: SC-TAM's distance is SIGNED and FB-pair
scoped, and compute_margin_loss's sc_tam branch uses
    margin_target = delta_d          (i.e. m_ij, NOT 2*delta_d)
not the doubled target used by the euclidean/task_aligned branches (see
train_eggo_m.py: "margin_target = delta_d  # SC-TAM's own calibrated m_ij").
This is because the euclidean/task_aligned branches threshold an
UNSIGNED (absolute-value or norm) distance that is symmetric about 0, so
"2*delta_d" is really "delta_d on each side" collapsed into one absolute
scale; SC-TAM's distance is already signed and already computed with a
fixed, known-correct orientation (tumor - background, positive when
correctly separated), so its calibration target is a single scalar
threshold on the SIGNED value directly, no doubling.

Method: sample the SAME anchor construction as E24 (same VOXELS_PER_
VOLUME/MAX_PAIRS/seeds, same N_VAL_SUBJECTS, same fresh-init + .train()
discipline), restricted to FOREGROUND-BACKGROUND pairs only (SC-TAM's
own required scope, Section 11 decision 2), and compute the SIGNED
projected distance
    d^SC = (z_tumor . w_hat) - (z_bg . w_hat)
using seg_head's own freshly-initialized weight vector as w_hat (same
w_hat construction as E24's delta_d_w). Target: the same 20th-percentile
active-hinge criterion E12e/E24 used, i.e. m_ij is set so that ~20% of
sampled FB pairs have d^SC < m_ij at initialization (10-30% acceptable
range, per the same user-specified criterion carried through E12e ->
E24 -> here).

Note: at fresh initialization, d^SC is not guaranteed to be positive on
average (w_hat's orientation relative to "tumor" is arbitrary before any
training signal), so the raw distribution may be centered near 0 or even
negative-skewed. This calibration script reports the full percentile
range (including negative percentiles) rather than assuming positivity,
consistent with the project's discipline of computing before asserting.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

N_VAL_SUBJECTS = 20  # same as E12e/E24
VOXELS_PER_VOLUME = 2000  # same as E12e/E24
MAX_PAIRS = 5000  # same as E12e/E24
TARGET_PCT = 20  # same as E12e/E24 -- midpoint of the requested 10-30% active-hinge range


def sample_voxels_and_fb_pairs(model, val_dataset, device, n_use, seed=0):
    """Same sampling logic as E24's sample_voxels_and_pairs, restricted to
    signed FB-pair distance only (SC-TAM's required scope). Reuses the
    identical voxel-sampling loop and the identical pair-index draw
    (np.random.RandomState(1) for tumor, RandomState(2) for background)
    so the sampled PAIRS are the exact same pairs E24's delta_d_w
    calibration used -- only the distance FUNCTION applied to them
    differs (signed projection instead of absolute projection), keeping
    this calibration run apples-to-apples comparable to E24's own."""
    rng = np.random.RandomState(seed)
    all_features = []
    all_ground_truth = []

    with torch.no_grad():
        for idx in range(n_use):
            image, mask, _ = val_dataset[idx]
            image_b = image.unsqueeze(0).to(device)
            outputs = model(image_b)

            dec1_feat = outputs["dec1"].squeeze(0).cpu().numpy()
            gt = mask.squeeze(0).numpy().astype(np.float32)

            D, H, W = gt.shape
            total_voxels = D * H * W
            n_sample = min(VOXELS_PER_VOLUME, total_voxels)
            flat_idx = rng.choice(total_voxels, size=n_sample, replace=False)
            d_idx, h_idx, w_idx = np.unravel_index(flat_idx, (D, H, W))

            dec1_flat = dec1_feat.reshape(32, -1)
            for i in range(n_sample):
                vi, di, hi, wi = flat_idx[i], d_idx[i], h_idx[i], w_idx[i]
                all_features.append(dec1_flat[:, vi])
                all_ground_truth.append(bool(gt[di, hi, wi]))

    Z = np.array(all_features)
    ground_truth = np.array(all_ground_truth)

    tumor_idx = np.where(ground_truth)[0]
    bg_idx = np.where(~ground_truth)[0]
    signed_distances = np.array([])
    abs_distances_crosscheck = np.array([])

    if len(tumor_idx) > 0 and len(bg_idx) > 0:
        n_pairs = min(MAX_PAIRS, len(tumor_idx), len(bg_idx))
        t_sample = np.random.RandomState(1).choice(tumor_idx, size=n_pairs, replace=(n_pairs > len(tumor_idx)))
        b_sample = np.random.RandomState(2).choice(bg_idx, size=n_pairs, replace=(n_pairs > len(bg_idx)))

        with torch.no_grad():
            conv_layer = model.seg_head[0]
            w_raw = conv_layer.weight.detach().cpu().numpy().reshape(-1)  # (32,)
            w_hat = w_raw / (np.linalg.norm(w_raw) + 1e-8)

        # SIGNED, class-conditional distance: tumor_proj - bg_proj,
        # computed from KNOWN class identity (t_sample is always the
        # tumor voxel, b_sample always the background voxel) -- exactly
        # mirroring the fixed-sign-assignment convention locked into
        # compute_margin_loss's sc_tam branch (not loop-relative order).
        tumor_proj = Z[t_sample] @ w_hat
        bg_proj = Z[b_sample] @ w_hat
        signed_distances = tumor_proj - bg_proj  # (n_pairs,)

        # Cross-check only, not used for calibration: the absolute value,
        # to compare against E24's delta_d_w distribution as a sanity
        # check that the same underlying pairs/features are being used.
        abs_distances_crosscheck = np.abs(signed_distances)

    return signed_distances, abs_distances_crosscheck


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_use = min(N_VAL_SUBJECTS, len(val_dataset))

    print("=" * 70)
    print("E25 Section 12: SC-TAM m_ij calibration -- FRESH, randomly-initialized model")
    print("=" * 70)
    torch.manual_seed(0)  # same seed as E12e/E24's own calibration, for direct comparability
    fresh_model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    fresh_model.train()  # CRITICAL -- same E12e lesson: eval() on a fresh model is a ~68x-different, degenerate BN regime

    signed_dist, abs_dist_crosscheck = sample_voxels_and_fb_pairs(fresh_model, val_dataset, device, n_use)

    print(f"\nSigned FB-pair distance distribution at initialization (tumor_proj - bg_proj):")
    print(f"  mean={signed_dist.mean():.4f} std={signed_dist.std():.4f}")
    for p in (5, 10, 20, 30, 50, 70, 80, 90, 95):
        print(f"  p{p}={np.percentile(signed_dist, p):.4f}")
    frac_positive = float((signed_dist > 0).mean())
    print(f"  fraction with positive sign (tumor_proj > bg_proj) at fresh init: {frac_positive*100:.2f}%")
    print("  (not expected to be ~100% at fresh init -- w_hat's orientation relative to")
    print("   'tumor' is not yet meaningful before any training signal; this is expected")
    print("   and does not affect calibration validity, matching E24's own documented")
    print("   scope decision that w_hat's eventual meaningfulness is a separate, later question)")

    # m_ij is a threshold on the SIGNED distance. The hinge [m_ij - d^SC]_+^2
    # is active (positive) whenever d^SC < m_ij. To get ~20% of pairs
    # active at init (same 10-30% criterion as E12e/E24), m_ij is set to
    # the 20th percentile of the signed distribution directly (NOT
    # doubled -- see module docstring for why this differs from E24's
    # delta_d_w doubling convention).
    m_ij = np.percentile(signed_dist, TARGET_PCT)
    print(f"\nCalibrated m_ij (targeting ~{TARGET_PCT}% active hinge at init, same criterion as E12e/E24): {m_ij:.4f}")

    active_pct_at_new = (signed_dist < m_ij).mean() * 100
    print(f"  Verification: {active_pct_at_new:.2f}% of pairs violate the hinge at this m_ij (target: 10-30%)")
    assert 10.0 <= active_pct_at_new <= 30.0 + 1e-6, (
        f"FAILED: active hinge rate {active_pct_at_new:.2f}% outside the required 10-30% range -- "
        f"percentile-based construction should make this a near-tautology (~{TARGET_PCT}% by construction); "
        f"a violation here would indicate a bug in the percentile/threshold logic, not sampling noise"
    )
    print("  PASSED")

    # Cross-check against E24's delta_d_w run: the ABSOLUTE value of this
    # same signed distribution should be in a broadly similar range to
    # E24's task_aligned proj_dist (both derived from the same seg_head
    # w_hat construction and the same underlying dec1 features at
    # fresh init, differing only in FB-pair-only vs all-pair scope and
    # the specific RandomState draw sequence for pairing).
    e24_calib_path = Path(__file__).parent.parent / "e24" / "e24_results" / "delta_d_w_calibration.json"
    if e24_calib_path.exists():
        with open(e24_calib_path) as f:
            e24_calib = json.load(f)
        e24_proj_dist_mean_abs = e24_calib["projected_distance_percentiles_at_init"]["50"]  # p50 as a representative scale
        this_abs_p50 = float(np.percentile(abs_dist_crosscheck, 50))
        print(f"\nCross-check vs E24 delta_d_w calibration (both from fresh-init seg_head w_hat):")
        print(f"  E24 task_aligned |proj_dist| p50={e24_proj_dist_mean_abs:.4f}")
        print(f"  This SC-TAM |signed_dist| p50={this_abs_p50:.4f}")
        print(f"  (expected to be broadly similar order of magnitude, not identical -- different pair sampling, FB-only vs all-pairs)")
    else:
        print(f"\n(E24 delta_d_w_calibration.json not found at {e24_calib_path} -- skipping cross-check)")

    out = {
        "m_ij": float(m_ij),
        "active_pct_at_new_m_ij": float(active_pct_at_new),
        "fraction_positive_sign_at_init": frac_positive,
        "signed_distance_percentiles_at_init": {str(p): float(np.percentile(signed_dist, p)) for p in (5, 10, 20, 30, 50, 70, 80, 90, 95)},
        "target_pct": TARGET_PCT,
        "seed": 0,
        "n_val_subjects": n_use,
        "n_pairs": int(len(signed_dist)),
    }
    out_dir = Path(__file__).parent / "e25_results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "m_ij_calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_dir / 'm_ij_calibration.json'}")


if __name__ == "__main__":
    main()
