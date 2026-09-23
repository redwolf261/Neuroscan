"""
Phase E14: Gradient Conflict Analysis -- purely diagnostic, no algorithm
or hyperparameter changes. Determines whether L_seg and L_margin push the
shared dec1 representation in compatible or conflicting directions.

Method (per PHASE_E14 spec):
  For each of E12f seed 0's saved checkpoints (epochs 1,5,10,15,20,25,30),
  run several real training-set batches through the model (single forward
  pass per batch), then compute

    g_seg    = d L_seg    / d dec1
    g_margin = d L_margin / d dec1

  via two SEPARATE torch.autograd.grad(..., retain_graph=True) calls on
  the SAME dec1 tensor from the SAME forward pass. autograd.grad (unlike
  .backward()) does not write into .grad buffers or share any mutable
  state between calls, so the two gradient computations are naturally
  independent -- neither can influence the other.

  IMPORTANT METHODOLOGICAL NOTE (corrected during development): L_seg in
  this codebase (FocalTverskyLoss) is a GLOBAL, non-separable loss -- its
  tp/fp/fn terms are full-batch sums, not a mean of per-voxel losses, so
  there is no exact way to write "the L_seg contribution at voxel i" as a
  standalone scalar. This does NOT block the analysis: autograd.grad
  applied to the TRUE, UNMODIFIED seg_loss (exactly as computed in
  train_eggo_m.py, byte-for-byte) still produces a well-defined gradient
  TENSOR d(seg_loss)/d(dec1) with the same (B,32,D,H,W) shape as dec1 --
  this IS the correct per-voxel gradient contribution of L_seg, obtained
  the same way backprop always attributes a global scalar loss's gradient
  across all the tensor elements that fed into it. No per-voxel loss
  decomposition is needed or used; only the exact, real seg_loss and
  margin_loss scalars are differentiated, both taken directly from
  train_eggo_m.py's own loss construction so this analysis measures
  EXACTLY the objective the model was actually trained on, not an
  approximation of it.

  Both gradient tensors are then restricted to the SAME anchor voxel
  locations used by EGGO-M's own compute_margin_loss (same stratified
  sampling call), for an apples-to-apples cosine similarity -- comparing
  at the same physical voxels where L_margin actually acts, rather than
  diluting against the ~99.9% of voxels g_margin is exactly zero at by
  construction (anchors are sparse, seg gradient is dense).

Deliverables: PHASE_E14_GRADIENT_CONFLICT_ANALYSIS.md (written separately
from this script's output), plots, and the raw per-checkpoint /
per-batch cosine-similarity statistics saved to JSON.
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

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import (  # noqa: E402
    sample_stratified_anchors, compute_margin_loss,
    DELTA_D_CALIBRATED, ANCHORS_PER_VOLUME, MAX_NEGATIVES_PER_ANCHOR,
    EVIDENCE_P99_DEFAULT, EMATauB,
)

CHECKPOINT_EPOCHS = [1, 5, 10, 15, 20, 25, 30]
SEED_DIR = "e12f_pilot_calibrated_seed0"
N_BATCHES_PER_CHECKPOINT = 8  # real training-set batches, for a per-checkpoint distribution not a single point
FOCAL_WEIGHT = 0.5      # matches EGGOMExperiment.focal_weight
EVIDENTIAL_WEIGHT = 0.5  # matches EGGOMExperiment.evidential_weight


def analyze_checkpoint(epoch, ckpt_dir, loader, device, n_batches, focal_fn, evidential_fn):
    ckpt_path = ckpt_dir / f"epoch_{epoch}.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    # train() mode, matching training-time BatchNorm behavior -- per the
    # E12e lesson, eval() mode on a mid-training checkpoint gives a
    # MEANINGFULLY DIFFERENT dec1 than what the optimizer actually saw
    # (running stats vs live batch stats), which would make this
    # analysis measure a shifted, not-actually-encountered gradient
    # geometry rather than the true training-time one.
    model.train()

    rng = np.random.RandomState(epoch)  # per-checkpoint but reproducible across runs
    tau_b_tracker = EMATauB()

    batch_results = []
    it = iter(loader)

    for b_idx in range(n_batches):
        try:
            images, masks, _ = next(it)
        except StopIteration:
            break
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        probs = outputs["probs"]
        alpha, beta = outputs["alpha"], outputs["beta"]
        boundary_logit = outputs["boundary_logit"]
        dec1 = outputs["dec1"]

        B, C, D, H, W = dec1.shape
        with torch.no_grad():
            evidence_full = alpha + beta - 2.0
        dec1_perm = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)  # view of dec1, same graph node
        evidence_flat = evidence_full.reshape(-1)
        boundary_flat = boundary_logit.reshape(-1)
        gt_flat = masks.reshape(-1)

        # --- L_seg: EXACT same formula/weights as train_eggo_m.py ---
        focal_loss = focal_fn(probs, masks)
        evidential_loss = evidential_fn(alpha, beta, masks)
        seg_loss = FOCAL_WEIGHT * focal_loss + EVIDENTIAL_WEIGHT * evidential_loss

        # Same stratified anchor sampling EGGO-M itself uses, so g_seg and
        # g_margin are compared at IDENTICAL voxel locations.
        voxels_per_vol = D * H * W
        anchor_idx_list = []
        for b in range(B):
            vol_evidence = evidence_flat[b * voxels_per_vol:(b + 1) * voxels_per_vol]
            local_idx = sample_stratified_anchors(vol_evidence, ANCHORS_PER_VOLUME, rng)
            anchor_idx_list.append(local_idx + b * voxels_per_vol)
        anchor_idx = torch.cat(anchor_idx_list)

        # --- L_margin: EXACT same formula as train_eggo_m.py (unweighted
        # by lambda -- per spec, measuring conflict between the raw loss
        # terms, not the applied/weighted contributions) ---
        current_tau_b = tau_b_tracker.tau_b
        margin_loss, _, margin_diag = compute_margin_loss(
            dec1_perm, evidence_flat, boundary_flat, gt_flat,
            anchor_idx, current_tau_b, EVIDENCE_P99_DEFAULT, DELTA_D_CALIBRATED,
            MAX_NEGATIVES_PER_ANCHOR, rng, device,
        )
        tau_b_tracker.update(margin_diag["abs_boundary_logit"])

        if not (torch.isfinite(margin_loss) and torch.isfinite(seg_loss)):
            print(f"  WARNING epoch {epoch} batch {b_idx}: non-finite loss, skipping")
            continue

        # --- Independent gradients w.r.t. dec1, from the SAME forward pass ---
        g_seg, = torch.autograd.grad(seg_loss, dec1, retain_graph=True, allow_unused=False)
        g_margin, = torch.autograd.grad(margin_loss, dec1, retain_graph=False, allow_unused=False)

        # Restrict both gradients to the anchor voxels: g_margin is
        # exactly zero elsewhere by construction (only anchors enter
        # compute_margin_loss), so comparing at the full dense grid would
        # trivially bias cos()->0 from the ~99.9% zero-padded region --
        # comparing at the anchors is the correct, apples-to-apples test
        # of whether the two losses agree where L_margin actually acts.
        g_seg_flat = g_seg.permute(0, 2, 3, 4, 1).reshape(-1, C)[anchor_idx]       # (n_anchor, 32)
        g_margin_flat = g_margin.permute(0, 2, 3, 4, 1).reshape(-1, C)[anchor_idx]  # (n_anchor, 32)

        gt_anchor = gt_flat[anchor_idx] > 0.5

        def per_anchor_stats(gs, gm, mask=None):
            if mask is not None:
                gs = gs[mask]
                gm = gm[mask]
            if gs.shape[0] == 0:
                return None
            norm_s = gs.norm(dim=1)
            norm_m = gm.norm(dim=1)
            valid = (norm_s > 1e-12) & (norm_m > 1e-12)
            if valid.sum() == 0:
                return None
            cos = F.cosine_similarity(gs[valid], gm[valid], dim=1)
            ratio = norm_m[valid] / norm_s[valid].clamp_min(1e-12)
            return {
                "cos": cos.detach().cpu().numpy(),
                "norm_seg": norm_s[valid].detach().cpu().numpy(),
                "norm_margin": norm_m[valid].detach().cpu().numpy(),
                "ratio": ratio.detach().cpu().numpy(),
            }

        overall = per_anchor_stats(g_seg_flat, g_margin_flat)
        tumor = per_anchor_stats(g_seg_flat, g_margin_flat, gt_anchor)
        bg = per_anchor_stats(g_seg_flat, g_margin_flat, ~gt_anchor)

        if overall is None:
            continue

        batch_results.append({
            "epoch": epoch, "batch": b_idx,
            "overall": overall, "tumor": tumor, "bg": bg,
        })

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return batch_results


def main():
    exp_dir = Path(__file__).parent
    ckpt_dir = exp_dir / SEED_DIR / "checkpoints"
    out_dir = exp_dir / "e14_gradient_conflict_results"
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = create_brats_loaders(
        batch_size=2, num_workers=0,
        root_dir=str(project_root / "Dataset" / "Training"),
        val_split=0.1,
    )

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)

    all_results = {}
    for epoch in CHECKPOINT_EPOCHS:
        print(f"Analyzing epoch {epoch}...")
        batch_results = analyze_checkpoint(
            epoch, ckpt_dir, train_loader, device, N_BATCHES_PER_CHECKPOINT, focal_fn, evidential_fn
        )
        all_results[epoch] = batch_results
        if not batch_results:
            print(f"  WARNING: no valid batches for epoch {epoch}")
            continue
        n_anchors_total = sum(len(r["overall"]["cos"]) for r in batch_results)
        all_cos = np.concatenate([r["overall"]["cos"] for r in batch_results])
        print(f"  {len(batch_results)} batches, {n_anchors_total} anchor-level cosine samples, "
              f"mean cos={all_cos.mean():.4f}, median={np.median(all_cos):.4f}")

    # --- Save raw data ---
    serializable = {}
    for epoch, batch_results in all_results.items():
        serializable[str(epoch)] = []
        for r in batch_results:
            entry = {"epoch": r["epoch"], "batch": r["batch"]}
            for key in ["overall", "tumor", "bg"]:
                d = r[key]
                if d is None:
                    entry[key] = None
                else:
                    entry[key] = {k: v.tolist() for k, v in d.items()}
            serializable[str(epoch)].append(entry)
    with open(out_dir / "raw_gradient_conflict_data.json", "w") as f:
        json.dump(serializable, f)

    # --- Aggregate stats per epoch ---
    print("\n" + "=" * 70)
    print("TRAJECTORY: epoch -> cosine similarity stats")
    print("=" * 70)
    epoch_summary = []
    for epoch in CHECKPOINT_EPOCHS:
        batch_results = all_results[epoch]
        if not batch_results:
            continue
        all_cos = np.concatenate([r["overall"]["cos"] for r in batch_results])
        all_ratio = np.concatenate([r["overall"]["ratio"] for r in batch_results])
        all_norm_seg = np.concatenate([r["overall"]["norm_seg"] for r in batch_results])
        all_norm_margin = np.concatenate([r["overall"]["norm_margin"] for r in batch_results])

        tumor_cos = np.concatenate([r["tumor"]["cos"] for r in batch_results if r["tumor"] is not None])
        bg_cos = np.concatenate([r["bg"]["cos"] for r in batch_results if r["bg"] is not None])

        row = {
            "epoch": epoch,
            "mean_cos": float(all_cos.mean()), "median_cos": float(np.median(all_cos)),
            "std_cos": float(all_cos.std()), "min_cos": float(all_cos.min()), "max_cos": float(all_cos.max()),
            "mean_ratio": float(all_ratio.mean()), "median_ratio": float(np.median(all_ratio)),
            "mean_norm_seg": float(all_norm_seg.mean()), "mean_norm_margin": float(all_norm_margin.mean()),
            "tumor_mean_cos": float(tumor_cos.mean()) if len(tumor_cos) else None,
            "bg_mean_cos": float(bg_cos.mean()) if len(bg_cos) else None,
            # Sign-only breakdown (mutually exclusive, sums to 100 up to
            # the measure-zero cos==0 case): the headline conflict/
            # cooperation split.
            "pct_negative_cos": float((all_cos < 0).mean() * 100),
            "pct_positive_cos": float((all_cos > 0).mean() * 100),
            # Separate, THRESHOLDED, mutually-exclusive 3-way regime split
            # for the stacked-area plot (cos<=-0.1 / -0.1<cos<0.1 / cos>=0.1)
            # -- NOT additively combined with the sign-only numbers above,
            # which overlap with the near-zero band by construction.
            "pct_conflict_lt_neg0.1": float((all_cos <= -0.1).mean() * 100),
            "pct_near_zero_cos": float(((all_cos > -0.1) & (all_cos < 0.1)).mean() * 100),
            "pct_cooperative_gt_0.1": float((all_cos >= 0.1).mean() * 100),
            "n_samples": int(len(all_cos)),
        }
        epoch_summary.append(row)
        print(f"epoch {epoch:2d}: mean={row['mean_cos']:+.4f} median={row['median_cos']:+.4f} "
              f"std={row['std_cos']:.4f} min={row['min_cos']:+.4f} max={row['max_cos']:+.4f} | "
              f"tumor={row['tumor_mean_cos']:+.4f} bg={row['bg_mean_cos']:+.4f} | "
              f"sign: %neg={row['pct_negative_cos']:.1f} %pos={row['pct_positive_cos']:.1f} | "
              f"thresholded(0.1): %conflict={row['pct_conflict_lt_neg0.1']:.1f} %indep={row['pct_near_zero_cos']:.1f} %coop={row['pct_cooperative_gt_0.1']:.1f}")

    with open(out_dir / "epoch_summary.json", "w") as f:
        json.dump(epoch_summary, f, indent=2)

    # --- Load Dice/margin trajectory from E12f logs for the scatter plots ---
    with open(exp_dir / "e12f_mechanism_results" / "trajectory_results.json") as f:
        mech_traj = json.load(f)
    mech_by_epoch = {r["epoch"]: r for r in mech_traj}

    # --- Plots ---
    epochs_arr = [r["epoch"] for r in epoch_summary]
    mean_cos_arr = [r["mean_cos"] for r in epoch_summary]
    median_cos_arr = [r["median_cos"] for r in epoch_summary]
    ratio_arr = [r["mean_ratio"] for r in epoch_summary]
    dice_arr = [mech_by_epoch[e]["val_dice"] for e in epochs_arr if e in mech_by_epoch]
    margin_arr = [mech_by_epoch[e]["mean_boundary_margin"] for e in epochs_arr if e in mech_by_epoch]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(epochs_arr, mean_cos_arr, marker="o", label="mean", color="steelblue")
    axes[0, 0].plot(epochs_arr, median_cos_arr, marker="s", label="median", color="orange", linestyle="--")
    axes[0, 0].axhline(0, color="gray", linewidth=0.8)
    axes[0, 0].set_ylabel("cos(g_seg, g_margin)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_title("Trajectory: gradient cosine similarity vs epoch")
    axes[0, 0].legend()

    all_cos_pooled = np.concatenate([
        np.concatenate([r["overall"]["cos"] for r in all_results[e]]) for e in epochs_arr
    ])
    axes[0, 1].hist(all_cos_pooled, bins=60, color="steelblue", edgecolor="none")
    axes[0, 1].axvline(0, color="gray", linewidth=0.8)
    axes[0, 1].set_xlabel("cos(g_seg, g_margin)")
    axes[0, 1].set_ylabel("Count (all anchors, all checkpoints)")
    axes[0, 1].set_title(f"Histogram of cosine similarities (n={len(all_cos_pooled)})")

    axes[0, 2].scatter(ratio_arr, dice_arr, c=epochs_arr, cmap="viridis")
    for e, r, d in zip(epochs_arr, ratio_arr, dice_arr):
        axes[0, 2].annotate(str(e), (r, d), fontsize=8)
    axes[0, 2].set_xlabel("mean ||g_margin|| / ||g_seg||")
    axes[0, 2].set_ylabel("Val Dice")
    axes[0, 2].set_title("Gradient norm ratio vs. Dice")

    axes[1, 0].scatter(ratio_arr, margin_arr, c=epochs_arr, cmap="viridis")
    for e, r, m in zip(epochs_arr, ratio_arr, margin_arr):
        axes[1, 0].annotate(str(e), (r, m), fontsize=8)
    axes[1, 0].set_xlabel("mean ||g_margin|| / ||g_seg||")
    axes[1, 0].set_ylabel("Mean boundary margin")
    axes[1, 0].set_title("Gradient norm ratio vs. boundary margin")

    tumor_cos_traj = [r["tumor_mean_cos"] for r in epoch_summary]
    bg_cos_traj = [r["bg_mean_cos"] for r in epoch_summary]
    axes[1, 1].plot(epochs_arr, tumor_cos_traj, marker="o", label="tumor voxels", color="crimson")
    axes[1, 1].plot(epochs_arr, bg_cos_traj, marker="s", label="background voxels", color="darkgreen")
    axes[1, 1].axhline(0, color="gray", linewidth=0.8)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("mean cos(g_seg, g_margin)")
    axes[1, 1].set_title("Class-conditional cosine similarity")
    axes[1, 1].legend()

    # Mutually-exclusive thresholded bins only (NOT the sign-only pct_negative_cos/pct_positive_cos, which overlap the near-zero band)
    pct_neg = [r["pct_conflict_lt_neg0.1"] for r in epoch_summary]
    pct_zero = [r["pct_near_zero_cos"] for r in epoch_summary]
    pct_pos = [r["pct_cooperative_gt_0.1"] for r in epoch_summary]
    axes[1, 2].stackplot(epochs_arr, pct_neg, pct_zero, pct_pos,
                          labels=["cos<=-0.1 (conflict)", "-0.1<cos<0.1 (independent)", "cos>=0.1 (cooperative)"],
                          colors=["crimson", "gray", "steelblue"])
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("% of anchors")
    axes[1, 2].set_title("Regime breakdown per epoch")
    axes[1, 2].legend(loc="upper right", fontsize=8)

    fig.suptitle("Phase E14: Gradient Conflict Analysis (EGGO-M seed 0, E12f checkpoints)")
    fig.tight_layout()
    fig.savefig(out_dir / "gradient_conflict_plots.png", dpi=150)
    plt.close(fig)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
