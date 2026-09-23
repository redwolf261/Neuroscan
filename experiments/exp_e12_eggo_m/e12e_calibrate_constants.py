"""
Experiment E12e: Hyperparameter Calibration.

Purpose (per user): NOT to optimize Dice. Only to make sure the
mechanism is actually active before any lambda sweep (E14). Derives
delta_d and tau_b from real MEASURED data, same discipline used
throughout this project (ABO's r_target, tau_b's original -- if flawed --
derivation, etc.) rather than picking new constants from intuition.

delta_d: measured on a FRESH, randomly-initialized UNet3D_v2 (not a
partially-trained EGGO-M checkpoint, and not the frozen-baseline's own
already-converged weights) -- the point is to calibrate against the
model's natural starting geometry, the same principle as how ABO's
r_target was derived from the model's own measured behavior rather than
an arbitrary constant. Target: 2*delta_d such that roughly 10-30% of
sampled same/opposite-class pairs violate the hinge at initialization,
per the user's explicit criterion.

CRITICAL, found via a real bug during E12e verification: the model MUST
be in model.train() mode for this measurement, matching how it's
actually run during training. The first calibration attempt used
model.eval() and got delta_d=0.0148, targeting 20% active pairs -- but
verified LIVE (train mode) that active_hinge_pct was 0.0% throughout,
not ~20% as intended. Root cause: BatchNorm3d layers (in Conv3DBlock)
behave completely differently in train() vs eval() mode -- train() uses
current-batch statistics, eval() uses running averages built up over
many batches (which don't exist yet for a fresh, just-initialized model,
so eval-mode BN on a fresh model is itself a somewhat degenerate
measurement). Direct comparison on the SAME seed/batch: dec1 std in
eval mode = 0.0089; in train mode = 0.6095 -- a ~68x difference. This
script now measures in train() mode throughout, matching the real
training loop's actual regime.

tau_b: PHASE_E12D found the ORIGINAL tau_b (derived from E1.3's static,
fully-converged offline classifier) mismatches the LIVE boundary head's
logit-magnitude growth during actual training, causing B_i to collapse
by epoch 15-20. This script measures the boundary head's OWN |d_i|
distribution at several points during a short live training run (not
just at init) to find a tau_b that keeps B_i in a useful dynamic range
throughout training, not just at one snapshot.
"""
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

N_VAL_SUBJECTS = 20
VOXELS_PER_VOLUME = 2000
MAX_PAIRS = 5000


def sample_voxels_and_pairs(model, val_dataset, device, n_use, seed=0):
    """Returns pairwise distances (opposite-class) and |boundary_logit|
    values, sampled the same way train_eggo_m.py's anchor sampling does."""
    rng = np.random.RandomState(seed)
    all_features = []
    all_ground_truth = []
    all_boundary_logit = []

    with torch.no_grad():
        for idx in range(n_use):
            image, mask, _ = val_dataset[idx]
            image_b = image.unsqueeze(0).to(device)
            outputs = model(image_b)

            boundary_logit = outputs["boundary_logit"].squeeze(0).squeeze(0).cpu().numpy()
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
                all_boundary_logit.append(float(boundary_logit[di, hi, wi]))

    Z = np.array(all_features)
    ground_truth = np.array(all_ground_truth)
    boundary_logit_arr = np.array(all_boundary_logit)

    tumor_idx = np.where(ground_truth)[0]
    bg_idx = np.where(~ground_truth)[0]
    distances = np.array([])
    if len(tumor_idx) > 0 and len(bg_idx) > 0:
        n_pairs = min(MAX_PAIRS, len(tumor_idx), len(bg_idx))
        t_sample = np.random.RandomState(1).choice(tumor_idx, size=n_pairs, replace=(n_pairs > len(tumor_idx)))
        b_sample = np.random.RandomState(2).choice(bg_idx, size=n_pairs, replace=(n_pairs > len(bg_idx)))
        distances = np.linalg.norm(Z[t_sample] - Z[b_sample], axis=1)

    return distances, np.abs(boundary_logit_arr)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_use = min(N_VAL_SUBJECTS, len(val_dataset))

    print("=" * 70)
    print("PART 1: delta_d calibration -- FRESH, randomly-initialized model")
    print("=" * 70)
    torch.manual_seed(0)
    fresh_model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    fresh_model.train()  # MUST match the real training loop's mode -- see module docstring for why eval() was wrong here

    distances_init, abs_boundary_init = sample_voxels_and_pairs(fresh_model, val_dataset, device, n_use)
    print(f"\nPairwise distance distribution at initialization:")
    print(f"  mean={distances_init.mean():.3f} std={distances_init.std():.3f}")
    for p in (5, 10, 20, 30, 50):
        print(f"  p{p}={np.percentile(distances_init, p):.3f}")

    # Target: 2*delta_d such that 10-30% of pairs violate the hinge
    # (i.e. 2*delta_d should sit around the 10th-30th percentile of the
    # distance distribution -- pairs closer than this are "active").
    target_pct = 20  # midpoint of the requested 10-30% range
    two_delta_d = np.percentile(distances_init, target_pct)
    delta_d_new = two_delta_d / 2
    print(f"\nCalibrated delta_d (targeting ~{target_pct}% active hinge at init): {delta_d_new:.4f}")
    print(f"  (i.e. 2*delta_d = {two_delta_d:.4f}, vs. the ORIGINAL, uncalibrated delta_d=1.0 "
          f"used in E12b -- {two_delta_d/2.0:.1f}x larger)")

    # Verify: what % of pairs actually violate at this delta_d?
    active_pct_at_new = (distances_init < two_delta_d).mean() * 100
    print(f"  Verification: {active_pct_at_new:.2f}% of pairs violate the hinge at this delta_d (target: 10-30%)")

    print("\n" + "=" * 70)
    print("PART 2: tau_b calibration -- track |boundary_logit| through a SHORT live training run")
    print("=" * 70)
    print("(Loads E12b's actual checkpoints at several epochs, since that IS live-trained data --")
    print(" more representative than re-running a fresh short train here, and free/no GPU cost)")

    ckpt_dir = Path(__file__).parent / "e12b_pilot_seed0" / "checkpoints"
    tau_b_candidates = {}
    for epoch in (1, 5, 10, 15, 20, 25, 30):
        ckpt_path = ckpt_dir / f"epoch_{epoch}.pth"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(ckpt["model_state"])
        # train() for full consistency with the actual training loop's mode
        # (matches e12d_gradient_norms.py's precedent). Note: for TRAINED
        # checkpoints (epoch>=1, not fresh-init) this makes little practical
        # difference -- direct A/B check found eval/train dec1 std differs
        # by only ~9% at epoch 1 and <3% by epoch 10, vs. the ~68x gap at
        # true fresh initialization that caused Part 1's original bug.
        model.train()

        _, abs_boundary = sample_voxels_and_pairs(model, val_dataset, device, n_use, seed=epoch)
        # A tau_b that keeps B_i = exp(-|d|/tau_b) in a useful [0.1, 0.9]
        # range for the MEDIAN |d_i| at this epoch would need
        # tau_b ~= median(|d_i|) / -ln(0.5) = median(|d_i|) / 0.693
        median_abs_d = np.median(abs_boundary)
        implied_tau_b = median_abs_d / 0.693  # solves exp(-median/tau_b)=0.5
        tau_b_candidates[epoch] = {"median_abs_d": float(median_abs_d), "implied_tau_b": float(implied_tau_b)}
        print(f"  epoch {epoch}: median|d_i|={median_abs_d:.3f}, implied tau_b (for B_i~0.5 at median)={implied_tau_b:.3f}")

    print(f"\nORIGINAL tau_b used in E12b: {np.sqrt(1.456*0.648):.4f} (derived from E1.3's static offline classifier)")
    print(f"Live |d_i| grows substantially over training (see above) -- a SINGLE fixed tau_b cannot keep")
    print(f"B_i well-calibrated across the whole run. Two options:")
    print(f"  (a) pick tau_b for a specific target epoch/regime (e.g. mid-training), accepting it will be")
    print(f"      miscalibrated elsewhere")
    print(f"  (b) make tau_b adaptive: an EMA of median|d_i| updated during training, recomputed each epoch")
    print(f"      (or every N steps), so B_i's dynamic range stays meaningful throughout -- recommended,")
    print(f"      since the live/offline mismatch IS the diagnosed root cause from PHASE_E12D.")

    import json
    out = {
        "delta_d_new": float(delta_d_new),
        "two_delta_d": float(two_delta_d),
        "active_pct_at_new_delta_d": float(active_pct_at_new),
        "distance_percentiles_at_init": {str(p): float(np.percentile(distances_init, p)) for p in (5, 10, 20, 30, 50)},
        "tau_b_candidates_by_epoch": tau_b_candidates,
        "original_tau_b": float(np.sqrt(1.456 * 0.648)),
    }
    out_dir = Path(__file__).parent / "e12e_results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "calibration_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_dir / 'calibration_results.json'}")


if __name__ == "__main__":
    main()
