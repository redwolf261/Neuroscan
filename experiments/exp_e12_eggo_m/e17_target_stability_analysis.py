"""
Phase E17: Margin Target Stability Analysis -- measurement-only, no
retraining, no changes to model/loss/optimizer/hyperparameters anywhere
in this script. Follow-up to E16's finding that the margin gradient is
consistently correctly-directed (Part C) yet net accumulated movement
along the margin direction is small and slightly negative (Part B/D).

Tests whether this is explained by a MOVING TARGET: if the opposite-
class centroid direction itself rotates checkpoint-to-checkpoint, then
"move toward separating today" and "move toward separating tomorrow"
point in genuinely different directions, and locally-correct per-step
gradients (E16 Part C) would fail to integrate into large net
displacement (E16 Part B/D) even with zero conflict (E14) and a
perfectly sensitive decoder (E15) -- a moving-target/representation-
drift explanation, distinct from (and not yet ruled out by) any prior
phase.

Uses the SAME E12f seed-0 checkpoints (epochs 1,5,10,15,20,25,30) and
the SAME fixed 20-subject validation set as E12b.5/E12d/E12f/E13/E14/
E15/E16.

Three measurements, per the user's spec, plus one additional cross-check
added to separate "target drift" from "representation rotation" (two of
the four candidate explanations raised, which are related but distinct
-- see module-level notes at each function):

  1. Centroid direction stability: cos(dir(t), dir(t+1)) for the
     tumor-vs-background separating direction, consecutive checkpoints.
  2. Margin GRADIENT stability: cos(g_margin(t), g_margin(t+1)), same
     idea applied to the actual loss gradient rather than the geometric
     centroid direction.
  3. Gradient transport: cos(g_margin(epoch5), g_margin(epoch30)) -- do
     early and late gradients still agree at all, or have they gone
     fully orthogonal (no long-range coherence)?
  4. (Added) Representation rotation check via Procrustes alignment: is
     the geometric RELATIONSHIP between tracked points changing
     (rotation/reflection of the whole point cloud), separate from
     whether any single direction (like the centroid axis) drifts --
     this can distinguish "the class-separating axis specifically
     rotates" from "the whole representation is being reorganized," per
     the user's own point that these are related but not identical
     phenomena.

BUG FOUND AND FIXED DURING DEVELOPMENT (Parts 2/3's anchor sampling):
first smoke test used a small (100-500/subject), UNIFORM RANDOM fixed
anchor set (matching Part 1/4's tracking-set design, appropriate for
pure displacement tracking) for the margin-GRADIENT measurement too --
this produced margin_loss=0.0 (and therefore a genuinely all-zero
gradient, not a sampling artifact of the READOUT) at multiple
checkpoints in the smoke test. Root cause: compute_margin_loss's hinge
term is only "active" (nonzero) for pairs closer together than 2*delta_d
-- per EVERY prior phase's own measurement (E12e-E17), only ~0.1-3.3%
of pairs are ever active, and this rate is achieved in real training
ONLY because of the stratified LOW-EVIDENCE (high-uncertainty) anchor
sampling compute_margin_loss's real callers use -- a small UNIFORM
RANDOM sample essentially never lands on an active pair. Fixed by using
the SAME stratified low-evidence sampling as real training
(sample_stratified_anchors, ANCHORS_PER_VOLUME=2000/volume, matching
train_eggo_m.py exactly) for Parts 2/3's fixed anchor set, while keeping
it FIXED (same indices reused across all 7 checkpoints, sampled once
using EACH CHECKPOINT'S OWN evidence at epoch 1 to decide inclusion --
see Part 2/3 setup code) so gradients remain directly comparable across
epochs, unlike E16 Part C which deliberately used a fresh live
stratified sample at every checkpoint (appropriate for a different
question -- see E16's own methodology notes on why that was correct
there but is NOT what Part 2/3 needs here).
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import orthogonal_procrustes

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import (  # noqa: E402
    sample_stratified_anchors, compute_margin_loss,
    DELTA_D_CALIBRATED, ANCHORS_PER_VOLUME, MAX_NEGATIVES_PER_ANCHOR,
    EVIDENCE_P99_DEFAULT, EMATauB,
)

CHECKPOINT_EPOCHS = [1, 5, 10, 15, 20, 25, 30]
SEED_DIR = "e12f_pilot_calibrated_seed0"
N_VAL_SUBJECTS = 20
N_TRACKED_VOXELS_PER_SUBJECT = 500  # same fixed-tracking-set size as E16, for Part 1 and the Procrustes check


def load_model(ckpt_dir, epoch, device):
    ckpt = torch.load(ckpt_dir / f"epoch_{epoch}.pth", map_location=device, weights_only=False)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model


def main():
    exp_dir = Path(__file__).parent
    ckpt_dir = exp_dir / SEED_DIR / "checkpoints"
    out_dir = exp_dir / "e17_target_stability_results"
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_use = min(N_VAL_SUBJECTS, len(val_dataset))
    print(f"Using {n_use} fixed validation subjects (same set as E12b.5/.../E16)")

    # =====================================================================
    # SETUP: fixed tracking voxel set, sampled once (reuses E16's exact
    # sampling seed/scheme so the SAME physical voxels are tracked --
    # enables direct cross-reference to E16's Part A/B numbers).
    # =====================================================================
    print("\nSampling fixed tracking voxel set (same scheme as E16)...")
    rng_sample = np.random.RandomState(42)  # identical seed to E16's tracking-set sampling
    model_ref = load_model(ckpt_dir, 1, device)
    model_ref.eval()

    tracked_subject_idx = []
    tracked_flat_idx = []
    tracked_gt_class = []

    with torch.no_grad():
        for subj_idx in range(n_use):
            image, mask, subject_id = val_dataset[subj_idx]
            image_b = image.unsqueeze(0).to(device)
            mask_b = mask.unsqueeze(0).to(device)
            outputs = model_ref(image_b)
            dec1 = outputs["dec1"]
            B, C, D, H, W = dec1.shape
            gt_flat = mask_b.reshape(-1) > 0.5
            total_voxels = D * H * W
            n_sample = min(N_TRACKED_VOXELS_PER_SUBJECT, total_voxels)
            idx = rng_sample.choice(total_voxels, size=n_sample, replace=False)
            tracked_subject_idx.extend([subj_idx] * n_sample)
            tracked_flat_idx.extend(idx.tolist())
            tracked_gt_class.extend(gt_flat[idx].cpu().numpy().tolist())
    tracked_gt_class = np.array(tracked_gt_class)
    del model_ref

    # =====================================================================
    # Extract z at every checkpoint for the tracked voxels (needed for
    # Part 1 centroid-direction and the Procrustes check).
    # =====================================================================
    print("\nExtracting dec1 at tracked voxels for all checkpoints...")
    z_by_epoch = {}
    for epoch in CHECKPOINT_EPOCHS:
        model = load_model(ckpt_dir, epoch, device)
        model.eval()
        z_this_epoch = []
        with torch.no_grad():
            for subj_idx in range(n_use):
                image, mask, subject_id = val_dataset[subj_idx]
                image_b = image.unsqueeze(0).to(device)
                outputs = model(image_b)
                dec1 = outputs["dec1"]
                B, C, D, H, W = dec1.shape
                dec1_flat = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
                this_subject_mask = np.array(tracked_subject_idx) == subj_idx
                this_subject_flat_idx = np.array(tracked_flat_idx)[this_subject_mask]
                z_this_epoch.append(dec1_flat[this_subject_flat_idx].cpu().numpy())
        z_by_epoch[epoch] = np.concatenate(z_this_epoch, axis=0)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("Done.")

    # =====================================================================
    # PART 1: Centroid direction stability
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 1: Centroid Direction Stability")
    print("=" * 70)

    centroid_dirs = {}
    for epoch in CHECKPOINT_EPOCHS:
        z = z_by_epoch[epoch]
        tumor_c = z[tracked_gt_class].mean(axis=0)
        bg_c = z[~tracked_gt_class].mean(axis=0)
        direction = tumor_c - bg_c
        dist = float(np.linalg.norm(direction))
        unit = direction / max(dist, 1e-8)
        centroid_dirs[epoch] = {"unit": unit, "dist": dist, "tumor_c": tumor_c, "bg_c": bg_c}

    part1_rows = []
    for i in range(len(CHECKPOINT_EPOCHS) - 1):
        e_t, e_t1 = CHECKPOINT_EPOCHS[i], CHECKPOINT_EPOCHS[i + 1]
        cos_consec = float(np.dot(centroid_dirs[e_t]["unit"], centroid_dirs[e_t1]["unit"]))
        row = {"epoch_t": e_t, "epoch_t1": e_t1, "cos_direction": cos_consec,
               "dist_t": centroid_dirs[e_t]["dist"], "dist_t1": centroid_dirs[e_t1]["dist"]}
        part1_rows.append(row)
        print(f"epoch {e_t:2d}->{e_t1:2d}: cos(dir_t, dir_t+1) = {cos_consec:+.4f} "
              f"(dist: {row['dist_t']:.2f} -> {row['dist_t1']:.2f})")

    # Also: direction relative to epoch 1 (long-range drift, not just consecutive)
    print("\nDirection relative to epoch 1 (long-range drift):")
    long_range_part1 = []
    for epoch in CHECKPOINT_EPOCHS:
        cos_vs_e1 = float(np.dot(centroid_dirs[1]["unit"], centroid_dirs[epoch]["unit"]))
        long_range_part1.append({"epoch": epoch, "cos_vs_epoch1": cos_vs_e1})
        print(f"  epoch {epoch:2d}: cos(dir_1, dir_{epoch}) = {cos_vs_e1:+.4f}")

    with open(out_dir / "part1_centroid_direction_stability.json", "w") as f:
        json.dump({"consecutive": part1_rows, "vs_epoch1": long_range_part1}, f, indent=2)

    # =====================================================================
    # PART 2 & 3: Margin gradient stability (consecutive) and transport
    # (epoch5 -> epoch30 long-range), computed on a FIXED, STRATIFIED
    # anchor set for comparability across checkpoints (unlike E16 Part C,
    # which deliberately used each checkpoint's OWN live stratified
    # sample -- here we need the SAME anchors at every checkpoint so that
    # "g_margin(t)" and "g_margin(t+1)" are directly comparable vectors
    # in the same voxel indexing, not two different anchor sets).
    #
    # Anchors are selected ONCE using epoch 1's own low-evidence
    # stratification (sample_stratified_anchors, matching real training's
    # sampling exactly), then the SAME indices are reused at every
    # subsequent checkpoint -- fixed identity (required for gradient
    # comparability) but stratified selection (required to land on
    # voxels where the hinge is actually active -- see module docstring's
    # bug note for why uniform random sampling fails here).
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 2/3: Margin Gradient Stability and Transport")
    print("=" * 70)
    print("(Using a FIXED, STRATIFIED (low-evidence) anchor set across all checkpoints -- "
          "distinct from E16 Part C's live per-checkpoint stratified sampling -- so gradients "
          "at different epochs are directly comparable as vectors over the SAME anchors, "
          "while still landing on voxels where the hinge loss is actually active.")

    print("\nSelecting fixed STRATIFIED anchor set (epoch 1's own low-evidence voxels)...")
    model_strat = load_model(ckpt_dir, 1, device)
    model_strat.eval()
    strat_subject_idx = []
    strat_flat_idx = []
    with torch.no_grad():
        for subj_idx in range(n_use):
            image, mask, subject_id = val_dataset[subj_idx]
            image_b = image.unsqueeze(0).to(device)
            outputs = model_strat(image_b)
            alpha, beta = outputs["alpha"], outputs["beta"]
            evidence_flat = (alpha + beta - 2.0).reshape(-1)
            rng_strat = np.random.RandomState(1000 + subj_idx)  # fixed, reproducible, independent of the training-time rng
            local_idx = sample_stratified_anchors(evidence_flat, ANCHORS_PER_VOLUME, rng_strat)
            strat_subject_idx.extend([subj_idx] * len(local_idx))
            strat_flat_idx.extend(local_idx.cpu().numpy().tolist())
    del model_strat
    strat_subject_idx = np.array(strat_subject_idx)
    strat_flat_idx = np.array(strat_flat_idx)
    print(f"Selected {len(strat_flat_idx)} fixed stratified anchors ({ANCHORS_PER_VOLUME}/subject x {n_use} subjects)")

    grad_by_epoch = {}  # epoch -> (n_strat_anchors, 32) gradient array, aligned to strat_flat_idx ordering

    for epoch in CHECKPOINT_EPOCHS:
        model = load_model(ckpt_dir, epoch, device)
        model.train()  # matches E14/E16's precedent: live gradient measurement needs training-time BatchNorm behavior
        tau_b_tracker = EMATauB()
        rng = np.random.RandomState(epoch)

        grads_this_epoch = np.full((len(strat_flat_idx), 32), np.nan, dtype=np.float32)
        offset = 0

        for subj_idx in range(n_use):
            image, mask, subject_id = val_dataset[subj_idx]
            image_b = image.unsqueeze(0).to(device)
            mask_b = mask.unsqueeze(0).to(device)

            outputs = model(image_b)
            dec1 = outputs["dec1"]
            alpha, beta = outputs["alpha"], outputs["beta"]
            boundary_logit = outputs["boundary_logit"]
            B, C, D, H, W = dec1.shape

            with torch.no_grad():
                evidence_full = alpha + beta - 2.0
            dec1_perm = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
            evidence_flat = evidence_full.reshape(-1)
            boundary_flat = boundary_logit.reshape(-1)
            gt_flat = mask_b.reshape(-1)

            this_subject_mask = strat_subject_idx == subj_idx
            this_subject_flat_idx = strat_flat_idx[this_subject_mask]
            n_this_subject = len(this_subject_flat_idx)
            anchor_idx = torch.from_numpy(this_subject_flat_idx).long().to(device)

            current_tau_b = tau_b_tracker.tau_b
            margin_loss, _, margin_diag = compute_margin_loss(
                dec1_perm, evidence_flat, boundary_flat, gt_flat,
                anchor_idx, current_tau_b, EVIDENCE_P99_DEFAULT, DELTA_D_CALIBRATED,
                MAX_NEGATIVES_PER_ANCHOR, rng, device,
            )
            tau_b_tracker.update(margin_diag["abs_boundary_logit"])

            if torch.isfinite(margin_loss) and margin_loss.item() != 0.0:
                g_margin, = torch.autograd.grad(margin_loss, dec1, retain_graph=False)
                g_margin_flat = g_margin.permute(0, 2, 3, 4, 1).reshape(-1, C)[anchor_idx]
                grads_this_epoch[offset:offset + n_this_subject] = g_margin_flat.detach().cpu().numpy()
            # else: leave as NaN (this subject's fixed anchors happened to
            # have no valid opposite-class pairing this epoch -- rare edge
            # case, excluded from that epoch's stats rather than
            # zero-filled, since a true zero gradient and "no valid pair"
            # are different things and should not be conflated)

            offset += n_this_subject

        grad_by_epoch[epoch] = grads_this_epoch
        n_valid = int(np.isfinite(grads_this_epoch).all(axis=1).sum())
        n_nonzero = int((np.linalg.norm(np.nan_to_num(grads_this_epoch), axis=1) > 1e-12).sum())
        print(f"epoch {epoch:2d}: computed gradients for {n_valid}/{len(strat_flat_idx)} fixed stratified anchors "
              f"({n_nonzero} nonzero)")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Part 2: consecutive-checkpoint gradient stability, per-anchor cosine
    part2_rows = []
    all_cos_consec_pooled = []
    for i in range(len(CHECKPOINT_EPOCHS) - 1):
        e_t, e_t1 = CHECKPOINT_EPOCHS[i], CHECKPOINT_EPOCHS[i + 1]
        g_t = grad_by_epoch[e_t]
        g_t1 = grad_by_epoch[e_t1]
        valid = np.isfinite(g_t).all(axis=1) & np.isfinite(g_t1).all(axis=1)
        norm_t = np.linalg.norm(g_t[valid], axis=1)
        norm_t1 = np.linalg.norm(g_t1[valid], axis=1)
        nonzero = (norm_t > 1e-12) & (norm_t1 > 1e-12)
        gt_v = g_t[valid][nonzero]
        gt1_v = g_t1[valid][nonzero]
        cos = np.sum(gt_v * gt1_v, axis=1) / (np.linalg.norm(gt_v, axis=1) * np.linalg.norm(gt1_v, axis=1))
        all_cos_consec_pooled.extend(cos.tolist())
        row = {"epoch_t": e_t, "epoch_t1": e_t1, "n": int(len(cos)),
               "mean_cos": float(cos.mean()), "median_cos": float(np.median(cos)), "std_cos": float(cos.std())}
        part2_rows.append(row)
        print(f"epoch {e_t:2d}->{e_t1:2d}: n={row['n']} mean_cos(g_margin_t, g_margin_t+1)={row['mean_cos']:+.4f} "
              f"median={row['median_cos']:+.4f} std={row['std_cos']:.4f}")

    all_cos_consec_pooled = np.array(all_cos_consec_pooled)
    with open(out_dir / "part2_gradient_stability.json", "w") as f:
        json.dump({"consecutive": part2_rows,
                    "pooled_mean": float(all_cos_consec_pooled.mean()),
                    "pooled_std": float(all_cos_consec_pooled.std()),
                    "pooled_n": int(len(all_cos_consec_pooled))}, f, indent=2)

    # Part 3: gradient transport, long-range (epoch 5 vs epoch 30, and
    # every epoch vs epoch 1 for the full trajectory)
    print("\nPart 3: Gradient transport (long-range)")
    part3_rows = []
    g5 = grad_by_epoch[5]
    g30 = grad_by_epoch[30]
    valid_5_30 = np.isfinite(g5).all(axis=1) & np.isfinite(g30).all(axis=1)
    norm5 = np.linalg.norm(g5[valid_5_30], axis=1)
    norm30 = np.linalg.norm(g30[valid_5_30], axis=1)
    nonzero_5_30 = (norm5 > 1e-12) & (norm30 > 1e-12)
    cos_5_30 = np.sum(g5[valid_5_30][nonzero_5_30] * g30[valid_5_30][nonzero_5_30], axis=1) / (
        np.linalg.norm(g5[valid_5_30][nonzero_5_30], axis=1) * np.linalg.norm(g30[valid_5_30][nonzero_5_30], axis=1))
    print(f"cos(g_margin(epoch5), g_margin(epoch30)): n={len(cos_5_30)} mean={cos_5_30.mean():+.4f} "
          f"median={np.median(cos_5_30):+.4f} std={cos_5_30.std():.4f}")

    long_range_part3 = []
    g1 = grad_by_epoch[1]
    for epoch in CHECKPOINT_EPOCHS:
        g_e = grad_by_epoch[epoch]
        valid = np.isfinite(g1).all(axis=1) & np.isfinite(g_e).all(axis=1)
        n1 = np.linalg.norm(g1[valid], axis=1)
        ne = np.linalg.norm(g_e[valid], axis=1)
        nz = (n1 > 1e-12) & (ne > 1e-12)
        if nz.sum() == 0:
            continue
        cos_vs_1 = np.sum(g1[valid][nz] * g_e[valid][nz], axis=1) / (n1[nz] * ne[nz])
        long_range_part3.append({"epoch": epoch, "n": int(nz.sum()), "mean_cos_vs_epoch1": float(cos_vs_1.mean())})
        print(f"  cos(g_margin(1), g_margin({epoch})): n={int(nz.sum())} mean={cos_vs_1.mean():+.4f}")

    with open(out_dir / "part3_gradient_transport.json", "w") as f:
        json.dump({"epoch5_vs_epoch30": {"n": int(len(cos_5_30)), "mean": float(cos_5_30.mean()),
                                          "median": float(np.median(cos_5_30)), "std": float(cos_5_30.std())},
                    "vs_epoch1_trajectory": long_range_part3}, f, indent=2)

    # =====================================================================
    # PART 4 (added): Representation rotation via orthogonal Procrustes
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 4 (added): Representation Rotation Check (Procrustes)")
    print("=" * 70)
    print("Distinguishes 'the whole point cloud is being rigidly reorganized' "
          "from 'only the specific class-separating axis drifts' (Part 1).")

    part4_rows = []
    for i in range(len(CHECKPOINT_EPOCHS) - 1):
        e_t, e_t1 = CHECKPOINT_EPOCHS[i], CHECKPOINT_EPOCHS[i + 1]
        Z_t = z_by_epoch[e_t] - z_by_epoch[e_t].mean(axis=0, keepdims=True)
        Z_t1 = z_by_epoch[e_t1] - z_by_epoch[e_t1].mean(axis=0, keepdims=True)

        # orthogonal_procrustes finds R minimizing ||Z_t @ R - Z_t1||_F,
        # i.e. the best rigid rotation/reflection mapping Z_t onto Z_t1.
        R, scale = orthogonal_procrustes(Z_t, Z_t1)
        # How close is R to the identity? trace(R)/dim = mean cosine of
        # the rotation's diagonal action -- 1.0 = no rotation (identity),
        # lower = more rotation. This is a standard summary of "how much"
        # an orthogonal matrix rotates (relates to the mean of cos(theta_i)
        # over its principal rotation angles).
        rotation_closeness_to_identity = float(np.trace(R) / R.shape[0])

        # Residual after best-fit rotation: how much of the change from
        # Z_t to Z_t1 is NOT explained by a pure rotation (i.e. genuine
        # reshaping, not just the point cloud spinning in place).
        Z_t_rotated = Z_t @ R
        residual_frac = float(np.linalg.norm(Z_t_rotated - Z_t1, "fro") / max(np.linalg.norm(Z_t1, "fro"), 1e-8))

        row = {"epoch_t": e_t, "epoch_t1": e_t1,
               "rotation_closeness_to_identity": rotation_closeness_to_identity,
               "residual_fraction_after_best_rotation": residual_frac}
        part4_rows.append(row)
        print(f"epoch {e_t:2d}->{e_t1:2d}: rotation closeness to identity (trace(R)/dim) = "
              f"{rotation_closeness_to_identity:+.4f} | residual after best rotation = {residual_frac:.4f}")

    with open(out_dir / "part4_procrustes_rotation.json", "w") as f:
        json.dump(part4_rows, f, indent=2)

    # =====================================================================
    # PLOTS
    # =====================================================================
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    p1_x = [r["epoch_t1"] for r in part1_rows]
    axes[0, 0].plot(p1_x, [r["cos_direction"] for r in part1_rows], marker="o", color="steelblue")
    axes[0, 0].axhline(0, color="gray", linewidth=0.8)
    axes[0, 0].axhline(1, color="green", linestyle=":", linewidth=1)
    axes[0, 0].set_xlabel("Epoch (consecutive pair ending here)")
    axes[0, 0].set_ylabel("cos(centroid_dir_t, centroid_dir_t+1)")
    axes[0, 0].set_title("Part 1: Centroid direction stability (consecutive)")
    axes[0, 0].set_ylim(-1.1, 1.1)

    axes[0, 1].plot([r["epoch"] for r in long_range_part1], [r["cos_vs_epoch1"] for r in long_range_part1], marker="o", color="darkorange")
    axes[0, 1].axhline(0, color="gray", linewidth=0.8)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("cos(centroid_dir_1, centroid_dir_t)")
    axes[0, 1].set_title("Part 1: Centroid direction vs. epoch 1 (long-range)")
    axes[0, 1].set_ylim(-1.1, 1.1)

    p2_x = [r["epoch_t1"] for r in part2_rows]
    p2_mean = [r["mean_cos"] for r in part2_rows]
    p2_std = [r["std_cos"] for r in part2_rows]
    axes[0, 2].errorbar(p2_x, p2_mean, yerr=p2_std, marker="o", color="crimson", capsize=3)
    axes[0, 2].axhline(0, color="gray", linewidth=0.8)
    axes[0, 2].set_xlabel("Epoch (consecutive pair ending here)")
    axes[0, 2].set_ylabel("cos(g_margin_t, g_margin_t+1)")
    axes[0, 2].set_title("Part 2: Margin gradient stability (consecutive)")
    axes[0, 2].set_ylim(-1.1, 1.1)

    axes[1, 0].plot([r["epoch"] for r in long_range_part3], [r["mean_cos_vs_epoch1"] for r in long_range_part3], marker="o", color="purple")
    axes[1, 0].axhline(0, color="gray", linewidth=0.8)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("cos(g_margin(1), g_margin(t))")
    axes[1, 0].set_title("Part 3: Gradient transport vs. epoch 1")
    axes[1, 0].set_ylim(-1.1, 1.1)

    axes[1, 1].hist(cos_5_30, bins=40, color="purple", edgecolor="none")
    axes[1, 1].axvline(0, color="gray", linewidth=0.8)
    axes[1, 1].axvline(cos_5_30.mean(), color="red", linestyle="--", linewidth=1)
    axes[1, 1].set_xlabel("cos(g_margin(epoch5), g_margin(epoch30))")
    axes[1, 1].set_ylabel(f"Count (n={len(cos_5_30)})")
    axes[1, 1].set_title("Part 3: Gradient transport, epoch5 vs epoch30")

    p4_x = [r["epoch_t1"] for r in part4_rows]
    axes[1, 2].plot(p4_x, [r["rotation_closeness_to_identity"] for r in part4_rows], marker="o", label="rotation closeness to identity", color="teal")
    axes[1, 2].plot(p4_x, [r["residual_fraction_after_best_rotation"] for r in part4_rows], marker="s", label="residual after best rotation", color="brown")
    axes[1, 2].axhline(1, color="gray", linestyle=":", linewidth=1)
    axes[1, 2].set_xlabel("Epoch (consecutive pair ending here)")
    axes[1, 2].set_ylabel("Value")
    axes[1, 2].set_title("Part 4: Representation rotation (Procrustes)")
    axes[1, 2].legend(fontsize=8)

    fig.suptitle("Phase E17: Margin Target Stability Analysis (EGGO-M seed 0, E12f checkpoints)")
    fig.tight_layout()
    fig.savefig(out_dir / "e17_plots.png", dpi=150)
    plt.close(fig)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
