"""
Phase E24, Part 1: delta_d_w calibration for the task-aligned projected
margin loss (L_margin^w), selected and specified in PHASE_E23_
ALGORITHMIC_REDESIGN.md.

Directly replicates e12e_calibrate_constants.py's delta_d methodology
(same target criterion, same percentile-based derivation, same
fresh-init + model.train() discipline that E12e's own module docstring
documents was CRITICAL -- a first E12e attempt used model.eval() and got
a delta_d that produced 0.0% active hinge in real training, traced to a
~68x BatchNorm3d train/eval discrepancy at fresh initialization). This
script reuses that exact lesson: model.train() throughout, never eval(),
on a freshly-initialized model.

Scope decision, confirmed with user before writing this script:
calibrate on a FRESH, randomly-initialized model (matching E12e's own
convention and real training's actual starting regime), NOT after a
warmup period. This means w_hat (from seg_head's own random init) is
itself not yet a meaningful decision direction at calibration time --
an acknowledged, unresolved property of this calibration point, not
something calibration timing can fix. Calibrating later (e.g. after a
warmup) was considered and rejected: it would repeat exactly the mistake
E12d already diagnosed and E12e already fixed for tau_b (a value
calibrated at one snapshot going stale as training progresses) --
delta_d_w's job is only to set the hinge's INITIAL active-pair rate
correctly, mirroring delta_d's existing role exactly, not to encode any
claim about w_hat's own eventual meaningfulness (that is a separate,
already-named risk in PHASE_E23 Section 8 -- "w_hat stability", to be
checked empirically in a later phase, not resolved by calibration
timing).

Method: sample the SAME anchor/pair construction e12e_calibrate_
constants.py already uses (voxels sampled per validation volume, same
VOXELS_PER_VOLUME/MAX_PAIRS/seeds), but compute the PROJECTED distance
d_ij^w = |w_hat . (z_i - z_j)| instead of the 32-d Euclidean norm
||z_i - z_j||, using seg_head's own (freshly initialized) weight vector
as w_hat. Target: 2*delta_d_w at the same 20th-percentile criterion
E12e used for delta_d (10-30% active-hinge range at initialization, per
the same user-specified criterion), so delta_d_w plays an exactly
analogous role to delta_d, differing only in which distance function it
thresholds.
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

N_VAL_SUBJECTS = 20  # same as E12e
VOXELS_PER_VOLUME = 2000  # same as E12e
MAX_PAIRS = 5000  # same as E12e
TARGET_PCT = 20  # same as E12e -- midpoint of the requested 10-30% active-hinge range


def sample_voxels_and_pairs(model, val_dataset, device, n_use, seed=0):
    """Byte-identical sampling logic to e12e_calibrate_constants.py's own
    function of the same name, EXTENDED to also return w_hat (seg_head's
    own weight, at whatever state the passed-in model is in) so both the
    Euclidean and projected distance can be computed from the same
    sampled pairs for a direct, apples-to-apples comparison."""
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

        with torch.no_grad():
            conv_layer = model.seg_head[0]
            w_raw = conv_layer.weight.detach().cpu().numpy().reshape(-1)  # (32,)
            w_hat = w_raw / (np.linalg.norm(w_raw) + 1e-8)
        projected_distances = np.abs(diffs @ w_hat)  # (n_pairs,)

    return euclid_distances, projected_distances


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_use = min(N_VAL_SUBJECTS, len(val_dataset))

    print("=" * 70)
    print("E24 Part 1: delta_d_w calibration -- FRESH, randomly-initialized model")
    print("=" * 70)
    torch.manual_seed(0)  # same seed as E12e's own delta_d calibration, for direct comparability
    fresh_model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    fresh_model.train()  # CRITICAL -- see module docstring; eval() mode on a fresh model is a ~68x-different, degenerate BN regime per E12e's own finding

    euclid_dist, proj_dist = sample_voxels_and_pairs(fresh_model, val_dataset, device, n_use)

    print(f"\nEuclidean distance distribution at initialization (for cross-check against E12e's original delta_d=3.6659):")
    print(f"  mean={euclid_dist.mean():.4f} std={euclid_dist.std():.4f}")
    for p in (5, 10, 20, 30, 50):
        print(f"  p{p}={np.percentile(euclid_dist, p):.4f}")

    print(f"\nProjected (w_hat) distance distribution at initialization:")
    print(f"  mean={proj_dist.mean():.4f} std={proj_dist.std():.4f}")
    for p in (5, 10, 20, 30, 50):
        print(f"  p{p}={np.percentile(proj_dist, p):.4f}")

    two_delta_d_w = np.percentile(proj_dist, TARGET_PCT)
    delta_d_w = two_delta_d_w / 2
    print(f"\nCalibrated delta_d_w (targeting ~{TARGET_PCT}% active hinge at init, same criterion as E12e's delta_d): {delta_d_w:.4f}")
    print(f"  (i.e. 2*delta_d_w = {two_delta_d_w:.4f})")

    active_pct_at_new = (proj_dist < two_delta_d_w).mean() * 100
    print(f"  Verification: {active_pct_at_new:.2f}% of pairs violate the hinge at this delta_d_w (target: 10-30%)")

    # Sanity cross-check: ratio of projected to Euclidean distance, should
    # be < 1 always (a projection onto a unit vector can never exceed the
    # full-space distance) -- verifies the projection math is sane, not
    # just that the percentile machinery ran.
    ratio = proj_dist.mean() / euclid_dist.mean()
    print(f"\nSanity check: mean(proj_dist)/mean(euclid_dist) = {ratio:.4f} (must be <= 1.0 for a valid unit-vector projection)")
    assert ratio <= 1.0 + 1e-6, f"FAILED sanity check: projected distance exceeds Euclidean distance (ratio={ratio}), projection math is wrong"
    print("  PASSED")

    out = {
        "delta_d_w": float(delta_d_w),
        "two_delta_d_w": float(two_delta_d_w),
        "active_pct_at_new_delta_d_w": float(active_pct_at_new),
        "projected_distance_percentiles_at_init": {str(p): float(np.percentile(proj_dist, p)) for p in (5, 10, 20, 30, 50)},
        "euclid_distance_percentiles_at_init_crosscheck": {str(p): float(np.percentile(euclid_dist, p)) for p in (5, 10, 20, 30, 50)},
        "proj_to_euclid_mean_ratio": float(ratio),
        "target_pct": TARGET_PCT,
        "seed": 0,
        "n_val_subjects": n_use,
    }
    out_dir = Path(__file__).parent / "e24_results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "delta_d_w_calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_dir / 'delta_d_w_calibration.json'}")


if __name__ == "__main__":
    main()
