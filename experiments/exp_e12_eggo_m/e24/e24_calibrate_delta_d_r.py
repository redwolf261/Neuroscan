"""
Phase E24, Gate 6 pre-launch audit: delta_d_r calibration for the
random-projection control (condition E), d_r(z_i,z_j) = |r . (z_i-z_j)|.

REQUIRED FIX identified by the pre-launch audit: run_counterfactual.py's
ObjectiveConfig originally reused DELTA_D_W_CALIBRATED (0.2553, calibrated
for w_hat) for BOTH task_aligned AND random_projection modes. This is
wrong -- w_hat and r are DIFFERENT unit vectors with, in general,
different projected-distance distributions against the same fresh-init
dec1 geometry (w_hat comes from seg_head's own random init, r comes from
an independent seed). Reusing w's calibration for r's metric would mean
condition E is simultaneously testing "is projection onto a random axis
useful" AND "is a miscalibrated hinge threshold useful" -- confounding
the one comparison (B vs E) this gate's H4 specificity test exists to
isolate. Per the user's explicit instruction: E needs its OWN calibration
against its OWN frozen r, not a borrowed one.

Directly replicates e12e_calibrate_constants.py's / e24_calibrate_
delta_d_w.py's methodology exactly (same target criterion, same
percentile-based derivation, same fresh-init + model.train() discipline)
-- the only change is projecting onto r instead of w_hat.

CRITICAL ORDERING: r is fixed FIRST from its documented, training-RNG-
independent seed (999001, exactly matching run_counterfactual.py's
ObjectiveConfig(mode="random_projection") default), THEN delta_d_r is
calibrated against THAT SPECIFIC r -- not a fresh, different random draw
at calibration time. This ensures delta_d_r describes the metric
condition E will ACTUALLY train under, the same principle
delta_d_w's calibration already followed (calibrated against seg_head's
real weight, not a stand-in).
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

N_VAL_SUBJECTS = 20  # same as E12e / delta_d_w calibration
VOXELS_PER_VOLUME = 2000  # same as E12e
MAX_PAIRS = 5000  # same as E12e
TARGET_PCT = 20  # same as E12e -- midpoint of the requested 10-30% active-hinge range
RANDOM_SEED_FOR_R = 999001  # MUST match run_counterfactual.py's ObjectiveConfig default exactly


def construct_frozen_r(seed=RANDOM_SEED_FOR_R):
    """Byte-identical construction to run_counterfactual.py's
    ObjectiveConfig.__init__ for mode="random_projection" -- a fixed unit
    vector generated from a seed INDEPENDENT of the training/evaluation
    RNG. Extracted here as its own function so both this calibration
    script and run_counterfactual.py construct r the SAME way, verified
    identical below (not just asserted)."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(32, generator=g)
    return (r / r.norm()).detach()


def sample_voxels_and_pairs(model, val_dataset, device, n_use, r_hat, seed=0):
    """Same structure as e24_calibrate_delta_d_w.py's function of the
    same name, projecting onto r_hat instead of w_hat (r_hat is passed in,
    not recomputed from the model -- it does NOT depend on model state,
    unlike w_hat which reads seg_head's own weight)."""
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
    euclid_distances = np.array([])
    projected_distances = np.array([])

    if len(tumor_idx) > 0 and len(bg_idx) > 0:
        n_pairs = min(MAX_PAIRS, len(tumor_idx), len(bg_idx))
        t_sample = np.random.RandomState(1).choice(tumor_idx, size=n_pairs, replace=(n_pairs > len(tumor_idx)))
        b_sample = np.random.RandomState(2).choice(bg_idx, size=n_pairs, replace=(n_pairs > len(bg_idx)))
        diffs = Z[t_sample] - Z[b_sample]  # (n_pairs, 32)
        euclid_distances = np.linalg.norm(diffs, axis=1)

        r_np = r_hat.cpu().numpy()
        projected_distances = np.abs(diffs @ r_np)  # (n_pairs,)

    return euclid_distances, projected_distances


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_use = min(N_VAL_SUBJECTS, len(val_dataset))

    print("=" * 70)
    print("E24 pre-launch audit fix: delta_d_r calibration for condition E (random-projection control)")
    print("=" * 70)

    r_hat = construct_frozen_r()
    print(f"\nFrozen r (seed={RANDOM_SEED_FOR_R}): shape={tuple(r_hat.shape)}, norm={r_hat.norm().item():.6f}")

    # Cross-check against run_counterfactual.py's own construction, to
    # confirm this script and the real run will use the IDENTICAL r --
    # not just "a" random vector with the same seed number, but the
    # bit-identical tensor.
    sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e24"))
    from run_counterfactual import ObjectiveConfig
    real_objective = ObjectiveConfig(mode="random_projection", random_seed=RANDOM_SEED_FOR_R)
    real_r = real_objective._frozen_random_w_hat
    assert torch.equal(r_hat, real_r), (
        "FATAL: this script's r construction does NOT match run_counterfactual.py's ObjectiveConfig "
        "construction -- calibration would describe a DIFFERENT vector than the one condition E actually uses"
    )
    print("Verified: r matches run_counterfactual.py's ObjectiveConfig(mode='random_projection') exactly")

    torch.manual_seed(0)  # same seed as E12e's / delta_d_w's own calibration, for direct comparability
    fresh_model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    fresh_model.train()  # CRITICAL, same E12e-derived discipline as delta_d_w's own calibration

    euclid_dist, proj_dist = sample_voxels_and_pairs(fresh_model, val_dataset, device, n_use, r_hat)

    print(f"\nEuclidean distance distribution at initialization (cross-check, should match delta_d_w's own measurement exactly, same seed/model/data):")
    print(f"  mean={euclid_dist.mean():.4f} std={euclid_dist.std():.4f}")
    for p in (5, 10, 20, 30, 50):
        print(f"  p{p}={np.percentile(euclid_dist, p):.4f}")

    print(f"\nProjected (r) distance distribution at initialization:")
    print(f"  mean={proj_dist.mean():.4f} std={proj_dist.std():.4f}")
    for p in (5, 10, 20, 30, 50):
        print(f"  p{p}={np.percentile(proj_dist, p):.4f}")

    two_delta_d_r = np.percentile(proj_dist, TARGET_PCT)
    delta_d_r = two_delta_d_r / 2
    print(f"\nCalibrated delta_d_r (targeting ~{TARGET_PCT}% active hinge at init, same criterion as delta_d/delta_d_w): {delta_d_r:.4f}")
    print(f"  (i.e. 2*delta_d_r = {two_delta_d_r:.4f})")

    active_pct_at_new = (proj_dist < two_delta_d_r).mean() * 100
    print(f"  Verification: {active_pct_at_new:.2f}% of pairs violate the hinge at this delta_d_r (target: 10-30%)")

    ratio = proj_dist.mean() / euclid_dist.mean()
    print(f"\nSanity check: mean(proj_dist)/mean(euclid_dist) = {ratio:.4f} (must be <= 1.0 for a valid unit-vector projection)")
    assert ratio <= 1.0 + 1e-6, f"FAILED sanity check: projected distance exceeds Euclidean distance (ratio={ratio})"
    print("  PASSED")

    # Load delta_d_w's own results for a direct side-by-side comparison --
    # informative (not a pass/fail criterion): are the two calibrated
    # constants in the same ballpark, or does the random axis happen to
    # induce a very different distance distribution than w_hat's axis?
    w_calib_path = Path(__file__).parent / "e24_results" / "delta_d_w_calibration.json"
    if w_calib_path.exists():
        with open(w_calib_path) as f:
            w_calib = json.load(f)
        print(f"\nFor comparison, delta_d_w (task-aligned) = {w_calib['delta_d_w']:.4f}")
        print(f"delta_d_r (random-projection)              = {delta_d_r:.4f}")
        print(f"Ratio delta_d_r / delta_d_w = {delta_d_r / w_calib['delta_d_w']:.4f}")

    out = {
        "delta_d_r": float(delta_d_r),
        "two_delta_d_r": float(two_delta_d_r),
        "active_pct_at_new_delta_d_r": float(active_pct_at_new),
        "projected_distance_percentiles_at_init": {str(p): float(np.percentile(proj_dist, p)) for p in (5, 10, 20, 30, 50)},
        "euclid_distance_percentiles_at_init_crosscheck": {str(p): float(np.percentile(euclid_dist, p)) for p in (5, 10, 20, 30, 50)},
        "proj_to_euclid_mean_ratio": float(ratio),
        "target_pct": TARGET_PCT,
        "seed": 0,
        "random_seed_for_r": RANDOM_SEED_FOR_R,
        "r_hat": r_hat.tolist(),
        "n_val_subjects": n_use,
    }
    out_dir = Path(__file__).parent / "e24_results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "delta_d_r_calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_dir / 'delta_d_r_calibration.json'}")


if __name__ == "__main__":
    main()
