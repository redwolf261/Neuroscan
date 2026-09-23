"""
Phase E16: Margin Reachability Analysis -- measurement-only, no
retraining, no hyperparameter/loss/architecture changes anywhere in this
script. Explains WHY E15 found large causal gains from artificial margin
widening while E12f/E13's real training only ever achieved ~0.2 units of
sustained margin growth.

All measurements are taken from EXISTING E12f seed-0 checkpoints
(epochs 1, 5, 10, 15, 20, 25, 30 -- the same set used throughout
E12b.5/E12d/E12f/E13/E14/E15) on the SAME fixed 20-subject validation
set used throughout the project.

IMPORTANT METHODOLOGICAL NOTE (limitation, stated up front not buried):
No true epoch-0 (pre-training) checkpoint was ever saved -- the earliest
available is epoch_1.pth, i.e. AFTER one full epoch of training. This
script uses epoch_1 as the practical z_0 reference point for all
"displacement from initialization" measurements (Part A). This is not
literal initialization, but the same window (epoch 1 -> epoch 30) is
what E12f/E13/E15 all measure their own margin trajectories over, so it
is the relevant window for the reachability question -- flagged as a
limitation in the report, not hidden.

VOXEL IDENTITY ACROSS CHECKPOINTS: BraTSDataset's __getitem__ has no
random augmentation on the validation split (deterministic resize only,
confirmed by reading Dataset/brats_dataset.py directly before assuming
this) -- so the same (subject_idx, d, h, w) triple refers to the same
physical tissue location at every checkpoint. Voxel indices for the
FIXED tracking set (Parts A, B, D) are sampled ONCE (at epoch 1) and
REUSED at every subsequent checkpoint, so "displacement" genuinely
tracks the same points over time, not a re-randomized sample each time.

Part C's gradient computation instead uses EACH CHECKPOINT'S OWN live
stratified (uncertainty-based) anchor sampling, exactly matching what
compute_margin_loss actually operates on during real training at that
point -- a deliberately different (not fixed) voxel set from Parts A/B/D,
because Part C's question ("what direction does the real training-time
gradient point in") requires using the real training-time sampling
mechanism, not an arbitrary fixed set.
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
Z0_EPOCH = 1  # practical initialization reference (see module docstring)
N_TRACKED_VOXELS_PER_SUBJECT = 500  # fixed tracking set for Parts A/B/D, sampled once at Z0_EPOCH
E15_REALISTIC_PUSH_UNITS = 14.0  # M_required, from PHASE_E15 (E12f's full observed mean_boundary_margin dynamic range, the "realistic ceiling" that produced +0.039 Dice)


def load_model(ckpt_dir, epoch, device):
    ckpt = torch.load(ckpt_dir / f"epoch_{epoch}.pth", map_location=device, weights_only=False)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model


def get_dec1_and_gt(model, image, mask, device, train_mode):
    """train_mode=True for gradient/Part-C measurements (matches live
    training-time BatchNorm behavior, per the E12e/E14 lesson); False
    (eval mode) for the pure displacement-tracking measurements in Parts
    A/B/D, where we want the model's stable, deployed-time representation
    of the SAME input at each checkpoint, not a batch-composition-
    dependent training-mode readout (each of the 20 volumes is run
    ONE AT A TIME here, so a single-sample "batch" in train() mode would
    use degenerate single-sample BatchNorm statistics -- eval() mode,
    using the checkpoint's own accumulated running stats, is the
    correct, stable choice for tracking a consistent trajectory across
    checkpoints when volumes are processed individually)."""
    if train_mode:
        model.train()
    else:
        model.eval()
    image_b = image.unsqueeze(0).to(device)
    mask_b = mask.unsqueeze(0).to(device)
    if train_mode:
        outputs = model(image_b)
    else:
        with torch.no_grad():
            outputs = model(image_b)
    return outputs, image_b, mask_b


def main():
    exp_dir = Path(__file__).parent
    ckpt_dir = exp_dir / SEED_DIR / "checkpoints"
    out_dir = exp_dir / "e16_margin_reachability_results"
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_use = min(N_VAL_SUBJECTS, len(val_dataset))
    print(f"Using {n_use} fixed validation subjects (same set as E12b.5/E12d/E12f/E13/E14/E15)")

    # =====================================================================
    # SETUP: sample a FIXED tracking voxel set at Z0_EPOCH, reused at
    # every subsequent checkpoint for Parts A/B/D.
    # =====================================================================
    print(f"\nSampling fixed tracking voxel set at epoch {Z0_EPOCH} (practical z_0)...")
    model_z0 = load_model(ckpt_dir, Z0_EPOCH, device)
    model_z0.eval()

    tracked_subject_idx = []   # which subject each tracked voxel belongs to
    tracked_flat_idx = []      # flat (d*H*W+h*W+w) index within that subject's volume
    tracked_gt_class = []      # ground-truth class (bool, tumor=True) of each tracked voxel
    z0_by_voxel = []           # (32,) dec1 vector at z0 for each tracked voxel

    rng_sample = np.random.RandomState(42)
    volume_shape = None

    with torch.no_grad():
        for subj_idx in range(n_use):
            image, mask, subject_id = val_dataset[subj_idx]
            outputs, image_b, mask_b = get_dec1_and_gt(model_z0, image, mask, device, train_mode=False)
            dec1 = outputs["dec1"]  # (1,32,D,H,W)
            B, C, D, H, W = dec1.shape
            volume_shape = (D, H, W)
            dec1_flat = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (D*H*W, 32)
            gt_flat = mask_b.reshape(-1) > 0.5

            total_voxels = D * H * W
            n_sample = min(N_TRACKED_VOXELS_PER_SUBJECT, total_voxels)
            idx = rng_sample.choice(total_voxels, size=n_sample, replace=False)

            tracked_subject_idx.extend([subj_idx] * n_sample)
            tracked_flat_idx.extend(idx.tolist())
            tracked_gt_class.extend(gt_flat[idx].cpu().numpy().tolist())
            z0_by_voxel.append(dec1_flat[idx].cpu().numpy())

    z0_by_voxel = np.concatenate(z0_by_voxel, axis=0)  # (N_total, 32)
    tracked_gt_class = np.array(tracked_gt_class)
    n_tracked = z0_by_voxel.shape[0]
    print(f"Tracking {n_tracked} fixed voxels across {n_use} subjects "
          f"({tracked_gt_class.sum()} tumor, {(~tracked_gt_class).sum()} background)")
    del model_z0

    # =====================================================================
    # PART A: Latent Drift Budget
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART A: Latent Drift Budget")
    print("=" * 70)

    part_a_rows = []
    z_by_epoch = {}  # epoch -> (N_total, 32) array, same voxel order as z0_by_voxel

    for epoch in CHECKPOINT_EPOCHS:
        model = load_model(ckpt_dir, epoch, device)
        model.eval()
        z_this_epoch = []
        with torch.no_grad():
            for subj_idx in range(n_use):
                image, mask, subject_id = val_dataset[subj_idx]
                outputs, image_b, mask_b = get_dec1_and_gt(model, image, mask, device, train_mode=False)
                dec1 = outputs["dec1"]
                B, C, D, H, W = dec1.shape
                dec1_flat = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)

                this_subject_mask = np.array(tracked_subject_idx) == subj_idx
                this_subject_flat_idx = np.array(tracked_flat_idx)[this_subject_mask]
                z_this_epoch.append(dec1_flat[this_subject_flat_idx].cpu().numpy())
        z_this_epoch = np.concatenate(z_this_epoch, axis=0)
        z_by_epoch[epoch] = z_this_epoch
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # A1: displacement from z0
        disp = np.linalg.norm(z_this_epoch - z0_by_voxel, axis=1)  # (N_total,)

        # A2: centroid drift
        tumor_centroid_t = z_this_epoch[tracked_gt_class].mean(axis=0)
        bg_centroid_t = z_this_epoch[~tracked_gt_class].mean(axis=0)
        tumor_centroid_0 = z0_by_voxel[tracked_gt_class].mean(axis=0)
        bg_centroid_0 = z0_by_voxel[~tracked_gt_class].mean(axis=0)
        tumor_centroid_drift = float(np.linalg.norm(tumor_centroid_t - tumor_centroid_0))
        bg_centroid_drift = float(np.linalg.norm(bg_centroid_t - bg_centroid_0))

        row = {
            "epoch": epoch,
            "disp_mean": float(disp.mean()), "disp_median": float(np.median(disp)), "disp_std": float(disp.std()),
            "tumor_centroid_drift": tumor_centroid_drift, "bg_centroid_drift": bg_centroid_drift,
        }
        part_a_rows.append(row)
        print(f"epoch {epoch:2d}: disp mean={row['disp_mean']:.3f} median={row['disp_median']:.3f} std={row['disp_std']:.3f} | "
              f"tumor_centroid_drift={tumor_centroid_drift:.3f} bg_centroid_drift={bg_centroid_drift:.3f}")

    # A3: margin evolution -- reuse E12f's own measured trajectory (already computed, per spec)
    with open(exp_dir / "e12f_mechanism_results" / "trajectory_results.json") as f:
        mech_traj = json.load(f)
    mech_by_epoch = {r["epoch"]: r for r in mech_traj}
    for row in part_a_rows:
        row["mean_boundary_margin"] = mech_by_epoch[row["epoch"]]["mean_boundary_margin"]
        row["val_dice"] = mech_by_epoch[row["epoch"]]["val_dice"]

    with open(out_dir / "part_a_drift_budget.json", "w") as f:
        json.dump(part_a_rows, f, indent=2)

    # =====================================================================
    # PART B: Direction Decomposition (parallel to margin direction vs orthogonal)
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART B: Direction Decomposition")
    print("=" * 70)

    part_b_rows = []
    for epoch in CHECKPOINT_EPOCHS:
        z_t = z_by_epoch[epoch]
        delta_z = z_t - z0_by_voxel  # (N_total, 32), the actual movement each voxel made

        # PRIMARY margin direction (per spec): FIXED, anchored at z0 --
        # "away from the opposite class's z0 centroid," held constant per
        # voxel across the whole trajectory. This isolates "did the
        # movement that happened go toward the separation direction that
        # existed at the start," without letting a shifting reference
        # frame silently redefine what "toward" means over time.
        tumor_centroid_0 = z0_by_voxel[tracked_gt_class].mean(axis=0)
        bg_centroid_0 = z0_by_voxel[~tracked_gt_class].mean(axis=0)

        margin_dir = np.zeros_like(z0_by_voxel)
        margin_dir[tracked_gt_class] = z0_by_voxel[tracked_gt_class] - bg_centroid_0
        margin_dir[~tracked_gt_class] = z0_by_voxel[~tracked_gt_class] - tumor_centroid_0
        margin_dir_norm = np.linalg.norm(margin_dir, axis=1, keepdims=True)
        margin_dir_norm[margin_dir_norm < 1e-8] = 1e-8
        margin_unit = margin_dir / margin_dir_norm

        delta_z_parallel_scalar = np.sum(delta_z * margin_unit, axis=1)  # signed projection onto FIXED margin direction
        delta_z_parallel = delta_z_parallel_scalar[:, None] * margin_unit
        delta_z_perp = delta_z - delta_z_parallel

        energy_parallel = np.sum(delta_z_parallel ** 2, axis=1)  # ||delta_z_parallel||^2 per voxel
        energy_perp = np.sum(delta_z_perp ** 2, axis=1)
        energy_total = np.sum(delta_z ** 2, axis=1)

        pct_positive_parallel = float((delta_z_parallel_scalar > 0).mean() * 100)

        # SECONDARY, cross-check direction: LIVE reference, recomputed at
        # THIS epoch's own opposite-class centroid (not z0's). Distinct
        # question from the primary metric: "is the voxel currently
        # farther from where the opposite class currently is," regardless
        # of which direction it originally needed to travel. A voxel can
        # score negatively on the PRIMARY (fixed) metric while its class's
        # bulk centroid still separates further, if there is substantial
        # within-class reshuffling alongside the bulk shift -- this
        # secondary check distinguishes that scenario from a genuine
        # reduction in class separability.
        tumor_centroid_t = z_t[tracked_gt_class].mean(axis=0)
        bg_centroid_t = z_t[~tracked_gt_class].mean(axis=0)
        live_dir = np.zeros_like(z_t)
        live_dir[tracked_gt_class] = z_t[tracked_gt_class] - bg_centroid_t
        live_dir[~tracked_gt_class] = z_t[~tracked_gt_class] - tumor_centroid_t
        live_dist = np.linalg.norm(live_dir, axis=1)
        z0_dist_from_opposite = np.zeros(len(z0_by_voxel))
        z0_dist_from_opposite[tracked_gt_class] = np.linalg.norm(z0_by_voxel[tracked_gt_class] - bg_centroid_0, axis=1)
        z0_dist_from_opposite[~tracked_gt_class] = np.linalg.norm(z0_by_voxel[~tracked_gt_class] - tumor_centroid_0, axis=1)
        live_separation_change = live_dist - z0_dist_from_opposite  # positive = this voxel is now farther from ITS CURRENT opposite-class centroid than it was from the z0 one

        row = {
            "epoch": epoch,
            "mean_energy_parallel": float(energy_parallel.mean()),
            "mean_energy_perp": float(energy_perp.mean()),
            "mean_energy_total": float(energy_total.mean()),
            "fraction_energy_parallel": float(energy_parallel.sum() / max(energy_total.sum(), 1e-12)),
            "mean_signed_parallel_projection": float(delta_z_parallel_scalar.mean()),
            "pct_positive_parallel": pct_positive_parallel,
            "mean_live_separation_change": float(live_separation_change.mean()),
            "pct_positive_live_separation_change": float((live_separation_change > 0).mean() * 100),
        }
        part_b_rows.append(row)
        print(f"epoch {epoch:2d}: E_parallel={row['mean_energy_parallel']:.4f} E_perp={row['mean_energy_perp']:.4f} "
              f"frac_parallel={row['fraction_energy_parallel']*100:.2f}% | "
              f"fixed_ref: mean_signed_proj={row['mean_signed_parallel_projection']:+.4f} %positive={pct_positive_parallel:.1f} | "
              f"live_ref: mean_change={row['mean_live_separation_change']:+.4f} %positive={row['pct_positive_live_separation_change']:.1f}")

    with open(out_dir / "part_b_direction_decomposition.json", "w") as f:
        json.dump(part_b_rows, f, indent=2)

    # =====================================================================
    # PART C: Gradient Alignment with E15 Push Direction
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART C: Gradient Alignment with E15 Push Direction")
    print("=" * 70)

    part_c_rows = []
    all_cos_pooled = []

    for epoch in CHECKPOINT_EPOCHS:
        model = load_model(ckpt_dir, epoch, device)
        model.train()  # matches E14's precedent: live gradient measurement needs training-time BatchNorm behavior

        rng = np.random.RandomState(epoch)
        tau_b_tracker = EMATauB()
        epoch_cos = []

        # Use small multi-volume batches, same pattern as E14, for a
        # representative sample of REAL training-time anchors per epoch.
        n_batches = 8
        subj_pool = list(range(n_use))
        rng.shuffle(subj_pool)
        batch_size = 2

        for b_start in range(0, min(n_batches * batch_size, len(subj_pool)), batch_size):
            subj_batch = subj_pool[b_start:b_start + batch_size]
            if len(subj_batch) == 0:
                break
            images, masks = [], []
            for si in subj_batch:
                image, mask, _ = val_dataset[si]
                images.append(image)
                masks.append(mask)
            images = torch.stack(images).to(device)
            masks = torch.stack(masks).to(device)

            outputs = model(images)
            dec1 = outputs["dec1"]
            alpha, beta = outputs["alpha"], outputs["beta"]
            boundary_logit = outputs["boundary_logit"]

            B, C, D, H, W = dec1.shape
            with torch.no_grad():
                evidence_full = alpha + beta - 2.0
            dec1_perm = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
            evidence_flat = evidence_full.reshape(-1)
            boundary_flat = boundary_logit.reshape(-1)
            gt_flat = masks.reshape(-1)

            voxels_per_vol = D * H * W
            anchor_idx_list = []
            for b in range(B):
                vol_evidence = evidence_flat[b * voxels_per_vol:(b + 1) * voxels_per_vol]
                local_idx = sample_stratified_anchors(vol_evidence, ANCHORS_PER_VOLUME, rng)
                anchor_idx_list.append(local_idx + b * voxels_per_vol)
            anchor_idx = torch.cat(anchor_idx_list)

            current_tau_b = tau_b_tracker.tau_b
            margin_loss, _, margin_diag = compute_margin_loss(
                dec1_perm, evidence_flat, boundary_flat, gt_flat,
                anchor_idx, current_tau_b, EVIDENCE_P99_DEFAULT, DELTA_D_CALIBRATED,
                MAX_NEGATIVES_PER_ANCHOR, rng, device,
            )
            tau_b_tracker.update(margin_diag["abs_boundary_logit"])

            if not torch.isfinite(margin_loss):
                continue

            g_margin, = torch.autograd.grad(margin_loss, dec1, retain_graph=False)
            g_margin_flat = g_margin.permute(0, 2, 3, 4, 1).reshape(-1, C)[anchor_idx]  # (n_anchor, 32)
            neg_grad_margin = -g_margin_flat  # -grad(L_margin) = the direction that DECREASES margin loss (per spec: "Margin gradient -grad(L_margin)")

            # E15's exact push direction, evaluated at THESE anchor voxels:
            # unit vector away from the SAME-BATCH opposite-class centroid.
            anchor_gt = gt_flat[anchor_idx] > 0.5
            anchor_z = dec1_perm[anchor_idx].detach()
            tumor_mask_a = anchor_gt
            bg_mask_a = ~anchor_gt
            if tumor_mask_a.sum() == 0 or bg_mask_a.sum() == 0:
                continue
            tumor_centroid = anchor_z[tumor_mask_a].mean(dim=0)
            bg_centroid = anchor_z[bg_mask_a].mean(dim=0)

            push_dir = torch.zeros_like(anchor_z)
            push_dir[tumor_mask_a] = anchor_z[tumor_mask_a] - bg_centroid.unsqueeze(0)
            push_dir[bg_mask_a] = anchor_z[bg_mask_a] - tumor_centroid.unsqueeze(0)
            push_unit = push_dir / push_dir.norm(dim=1, keepdim=True).clamp_min(1e-8)

            grad_norm = neg_grad_margin.norm(dim=1)
            valid = grad_norm > 1e-12
            if valid.sum() == 0:
                continue
            cos = F.cosine_similarity(neg_grad_margin[valid], push_unit[valid], dim=1)
            epoch_cos.extend(cos.detach().cpu().numpy().tolist())

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        epoch_cos = np.array(epoch_cos)
        if len(epoch_cos) == 0:
            print(f"epoch {epoch:2d}: WARNING no valid samples")
            continue
        all_cos_pooled.extend(epoch_cos.tolist())
        ci_lo, ci_hi = np.percentile(epoch_cos, [2.5, 97.5])
        row = {
            "epoch": epoch, "n": int(len(epoch_cos)),
            "mean_cos": float(epoch_cos.mean()), "std_cos": float(epoch_cos.std()),
            "ci_95_lo": float(ci_lo), "ci_95_hi": float(ci_hi),
        }
        part_c_rows.append(row)
        print(f"epoch {epoch:2d}: n={row['n']} mean_cos={row['mean_cos']:+.4f} std={row['std_cos']:.4f} "
              f"95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]")

    all_cos_pooled = np.array(all_cos_pooled)
    t_pooled, p_pooled = stats.ttest_1samp(all_cos_pooled, 0)
    print(f"\nPooled across all checkpoints: n={len(all_cos_pooled)} mean={all_cos_pooled.mean():+.4f} "
          f"std={all_cos_pooled.std():.4f} (t={t_pooled:.2f}, p={p_pooled:.2e})")

    with open(out_dir / "part_c_gradient_alignment.json", "w") as f:
        json.dump({"per_epoch": part_c_rows,
                    "pooled": {"n": int(len(all_cos_pooled)), "mean": float(all_cos_pooled.mean()),
                               "std": float(all_cos_pooled.std()), "t": float(t_pooled), "p": float(p_pooled)}},
                   f, indent=2)

    # =====================================================================
    # PART D: Reachability Ratio
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART D: Reachability Ratio")
    print("=" * 70)

    # M_actual: the ACTUAL mean parallel (margin-direction) displacement
    # achieved by the end of training (epoch 30), relative to z0 (epoch 1)
    # -- i.e. the real, signed, per-voxel projection onto the margin
    # direction, matching the units E15's push was defined in (dec1-space
    # Euclidean distance).
    final_row_b = [r for r in part_b_rows if r["epoch"] == 30][0]
    m_actual = final_row_b["mean_signed_parallel_projection"]

    # M_required: E15's realistic push ceiling (14 units) that produced
    # the +0.039 Dice gain.
    m_required = E15_REALISTIC_PUSH_UNITS

    reachability_ratio = m_actual / m_required
    print(f"M_actual (mean signed parallel displacement, epoch1->epoch30): {m_actual:.4f} units")
    print(f"M_required (E15's realistic push ceiling): {m_required:.1f} units")
    print(f"Reachability ratio R = M_actual / M_required = {reachability_ratio:.4f}")

    with open(out_dir / "part_d_reachability_ratio.json", "w") as f:
        json.dump({"m_actual": float(m_actual), "m_required": float(m_required), "R": float(reachability_ratio)}, f, indent=2)

    # =====================================================================
    # PART E: Optimization Budget / Margin Efficiency
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART E: Optimization Budget (Margin Efficiency)")
    print("=" * 70)

    part_e_rows = []
    for row in part_b_rows:
        efficiency = row["fraction_energy_parallel"] * 100
        part_e_rows.append({"epoch": row["epoch"], "margin_efficiency_pct": efficiency})
        print(f"epoch {row['epoch']:2d}: margin efficiency = {efficiency:.2f}% of total latent movement energy")

    with open(out_dir / "part_e_margin_efficiency.json", "w") as f:
        json.dump(part_e_rows, f, indent=2)

    # =====================================================================
    # PART F: Falsification attempt
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART F: Falsification -- does training move sufficiently by an ALTERNATE metric?")
    print("=" * 70)

    # Alternate margin proxies computed directly from the tracked voxel
    # set (not E12f's own random-pair mean_boundary_margin metric), to
    # check whether the specific metric used elsewhere might be
    # UNDERSTATING real margin growth:
    #  (i) centroid-to-centroid distance (a cleaner, lower-variance proxy
    #      than E12f's random same-batch pairwise sampling)
    #  (ii) minimum tumor-background distance (worst-case/closest-pair
    #       margin, which is what actually determines many boundary
    #       errors, arguably more relevant than the MEAN pairwise distance)
    falsification_rows = []
    for epoch in CHECKPOINT_EPOCHS:
        z_t = z_by_epoch[epoch]
        tumor_z = z_t[tracked_gt_class]
        bg_z = z_t[~tracked_gt_class]
        centroid_dist = float(np.linalg.norm(tumor_z.mean(axis=0) - bg_z.mean(axis=0)))

        # Min distance: sample-limited (full pairwise is expensive at this
        # n), use a random subset of 2000 pairs per epoch for a stable estimate.
        rng_f = np.random.RandomState(epoch + 1000)
        n_pairs = min(2000, len(tumor_z), len(bg_z))
        t_idx = rng_f.choice(len(tumor_z), size=n_pairs, replace=(n_pairs > len(tumor_z)))
        b_idx = rng_f.choice(len(bg_z), size=n_pairs, replace=(n_pairs > len(bg_z)))
        pair_dists = np.linalg.norm(tumor_z[t_idx] - bg_z[b_idx], axis=1)
        min_dist_estimate = float(pair_dists.min())
        mean_dist_estimate = float(pair_dists.mean())

        row = {"epoch": epoch, "centroid_to_centroid_dist": centroid_dist,
               "estimated_min_pairwise_dist": min_dist_estimate, "estimated_mean_pairwise_dist": mean_dist_estimate}
        falsification_rows.append(row)
        print(f"epoch {epoch:2d}: centroid_dist={centroid_dist:.3f} est_min_pairwise={min_dist_estimate:.3f} "
              f"est_mean_pairwise={mean_dist_estimate:.3f}")

    centroid_net_change = falsification_rows[-1]["centroid_to_centroid_dist"] - falsification_rows[0]["centroid_to_centroid_dist"]
    min_net_change = falsification_rows[-1]["estimated_min_pairwise_dist"] - falsification_rows[0]["estimated_min_pairwise_dist"]
    print(f"\nNet change (epoch1->epoch30), centroid-to-centroid: {centroid_net_change:+.4f}")
    print(f"Net change (epoch1->epoch30), estimated min pairwise: {min_net_change:+.4f}")
    print(f"(Compare to E12f's own mean_boundary_margin net change: "
          f"{mech_by_epoch[30]['mean_boundary_margin'] - mech_by_epoch[1]['mean_boundary_margin']:+.4f})")

    with open(out_dir / "part_f_falsification.json", "w") as f:
        json.dump(falsification_rows, f, indent=2)

    # =====================================================================
    # PLOTS
    # =====================================================================
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    epochs_arr = [r["epoch"] for r in part_a_rows]
    axes[0, 0].plot(epochs_arr, [r["disp_mean"] for r in part_a_rows], marker="o", label="mean ||z_t - z_0||", color="steelblue")
    axes[0, 0].plot(epochs_arr, [r["tumor_centroid_drift"] for r in part_a_rows], marker="s", label="tumor centroid drift", color="crimson")
    axes[0, 0].plot(epochs_arr, [r["bg_centroid_drift"] for r in part_a_rows], marker="^", label="bg centroid drift", color="darkgreen")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Displacement (dec1 units)")
    axes[0, 0].set_title("Part A: Latent Drift Budget")
    axes[0, 0].legend(fontsize=8)

    ax2 = axes[0, 1]
    ax2.plot(epochs_arr, [r["mean_boundary_margin"] for r in part_a_rows], marker="o", color="purple")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean boundary margin (E12f metric)")
    ax2.set_title("Part A3: Margin evolution (reused from E12f)")

    axes[0, 2].plot(epochs_arr, [r["fraction_energy_parallel"] * 100 for r in part_b_rows], marker="o", color="darkorange")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("% of movement energy parallel to margin direction")
    axes[0, 2].set_title("Part E: Margin efficiency")
    axes[0, 2].axhline(50, color="gray", linestyle=":", linewidth=1)

    c_epochs = [r["epoch"] for r in part_c_rows]
    c_means = [r["mean_cos"] for r in part_c_rows]
    c_lo = [r["ci_95_lo"] for r in part_c_rows]
    c_hi = [r["ci_95_hi"] for r in part_c_rows]
    axes[1, 0].plot(c_epochs, c_means, marker="o", color="steelblue")
    axes[1, 0].fill_between(c_epochs, c_lo, c_hi, alpha=0.2, color="steelblue")
    axes[1, 0].axhline(0, color="gray", linewidth=0.8)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("cos(-grad(L_margin), E15 push direction)")
    axes[1, 0].set_title("Part C: Gradient alignment with E15 push direction")

    axes[1, 1].hist(all_cos_pooled, bins=60, color="steelblue", edgecolor="none")
    axes[1, 1].axvline(0, color="gray", linewidth=0.8)
    axes[1, 1].set_xlabel("cos(-grad(L_margin), E15 push direction)")
    axes[1, 1].set_ylabel(f"Count (pooled, n={len(all_cos_pooled)})")
    axes[1, 1].set_title("Part C: Pooled cosine distribution")

    axes[1, 2].plot(epochs_arr, [r["centroid_to_centroid_dist"] for r in falsification_rows], marker="o", label="centroid-to-centroid", color="steelblue")
    axes[1, 2].plot(epochs_arr, [r["estimated_min_pairwise_dist"] for r in falsification_rows], marker="s", label="est. min pairwise", color="crimson")
    axes[1, 2].plot(epochs_arr, [r["estimated_mean_pairwise_dist"] for r in falsification_rows], marker="^", label="est. mean pairwise", color="darkgreen")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Distance")
    axes[1, 2].set_title("Part F: Alternate margin metrics (falsification check)")
    axes[1, 2].legend(fontsize=8)

    fig.suptitle("Phase E16: Margin Reachability Analysis (EGGO-M seed 0, E12f checkpoints)")
    fig.tight_layout()
    fig.savefig(out_dir / "e16_plots.png", dpi=150)
    plt.close(fig)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
