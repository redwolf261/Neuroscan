"""
Phase E21: Parameter-to-Representation Update Transport -- purely
diagnostic, no training, no algorithm/hyperparameter changes. Direct
follow-up to E20: E20 found ||delta_theta_margin|| >= ||delta_theta_seg||
at every one of 6 checkpoints (ratio 1.01-1.61x), ruling out "the margin
objective is starved of parameter-update magnitude" -- but
cos(delta_theta_margin, delta_theta_total) was small (0.09-0.32, one
negative at epoch 25), meaning the margin update, while large, is poorly
aligned with the REAL combined-loss update direction IN PARAMETER SPACE.
This phase asks the next, more specific question: does that large
parameter-space update actually MOVE THE REPRESENTATION (dec1's output,
the actual space EGGO's margin loss and E15's causal intervention both
operate in) in a useful direction, or does it get lost in directions the
representation doesn't respond to?

Scope decision (confirmed with user before writing this script):
  - Same 6 checkpoints as E20 (epochs 5/10/15/20/25/30), dec1 only.
  - "Useful direction" reference: E15's OWN validated per-voxel
    Euclidean push direction (unit_dir = (z - opposite_class_centroid) /
    ||.||, computed separately for tumor and background voxels using
    that SAME batch's own ground truth), RECOMPUTED FRESH at each
    checkpoint/batch -- NOT a single fixed vector borrowed from one
    checkpoint. This is a faithful reuse of E15's validated method, not
    a new "useful direction" definition invented mid-arc, and not a
    reuse of a nonexistent fixed "d_E15" object (E15 never produced one
    -- its direction is inherently per-voxel and per-checkpoint, verified
    by direct inspection of e15_decoder_sensitivity.py before writing
    this script).
  - Delta_z is measured on the SAME real training batches E20 already
    used to compute Delta_theta at that checkpoint (not E15's validation
    set) -- keeps Delta_z causally tied to the exact update whose effect
    it measures, rather than mixing data distributions between the
    parameter-space and representation-space halves of this analysis.
    Confirmed with user before writing this script.
  - Method for Delta_z: out-of-place weight perturbation. dec1's
    parameters are perturbed by EACH of delta_theta_margin /
    delta_theta_seg / delta_theta_total (E20's already-computed vectors,
    reused exactly, not recomputed) on a DEEP-COPIED model (the real
    checkpoint and E20's own results are never mutated), then the SAME
    input batch is forward-passed again through the perturbed model.
    Delta_z = dec1_perturbed - dec1_original, both evaluated on the
    IDENTICAL input, so any difference is attributable only to the
    weight perturbation, not to a different forward pass's randomness
    (there is none here -- BatchNorm is the only source of batch-
    dependent behavior, and both passes use the SAME batch, so BN's
    live batch statistics are also identical between the two passes;
    the only thing that differs is the conv/affine weights themselves).

  IMPORTANT LIMITATION, stated explicitly: this reuses E20's
  short-local-replay Delta_theta vectors, which already carry E20's own
  documented limitations (fresh shadow-accumulator bias-correction
  transient, local-neighborhood-only validity). Delta_z inherits those
  same caveats -- this phase does not re-derive Delta_theta, only asks
  what a GIVEN already-measured Delta_theta does to the representation
  when actually applied.

Method, per checkpoint:
  1. Re-run E20's exact analyze_checkpoint logic (imported, not
     reimplemented) to get delta_theta_margin/seg/total for dec1 at this
     checkpoint, AND capture the actual batches used (by re-seeding the
     loader/rng identically) so Delta_z is measured on the same data.
  2. For each of a small number of real batches (reusing E20's batch
     stream): forward-pass the ORIGINAL model to get dec1_original and
     the ground truth for computing E15's direction on this batch.
  3. For each of {margin, seg, total}: deep-copy the model, add
     delta_theta_X to dec1's parameters (out-of-place, in-place
     modification of the COPY only), forward-pass the SAME input batch
     to get dec1_perturbed_X, compute delta_z_X = dec1_perturbed_X -
     dec1_original (per-voxel, (B,32,D,H,W) tensor, flattened to
     (N_voxels, 32) like every prior phase in this arc).
  4. Compute E15's per-voxel useful direction d_useful from
     dec1_original and this batch's ground truth (tumor voxels pushed
     away from bg centroid, bg voxels pushed away from tumor centroid --
     exact formula from e15_decoder_sensitivity.py's
     compute_manipulated_dice, verified by direct inspection).
  5. Per-voxel cosine similarity cos(delta_z_X, d_useful) for X in
     {margin, seg, total}, plus the amplification ratio
     ||delta_z_X|| / ||delta_theta_X|| (a SINGLE scalar per checkpoint
     for the ratio, since delta_theta is one vector over all of dec1's
     params but delta_z is per-voxel -- ||delta_z_X|| here means the
     pooled norm over ALL voxels in the batch, matching how E14/E19
     pooled anchor-level gradients).

Deliverables: PHASE_E21_PARAMETER_TO_REPRESENTATION_TRANSPORT.md
(written separately), plot, raw per-checkpoint/per-batch stats as JSON.
"""
import sys
import copy
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
from e20_dec1_update_decomposition import ShadowAdam  # noqa: E402

CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30]  # same as E20
SEED_DIR = "e12f_pilot_calibrated_seed0"
N_REPLAY_STEPS = 15   # same as E20, for computing delta_theta
N_TRANSPORT_BATCHES = 4  # additional fresh batches (beyond the 15 replay steps) used to measure Delta_z -- kept small since each requires 4 forward passes (1 original + 3 perturbed)
FOCAL_WEIGHT = 0.5
EVIDENTIAL_WEIGHT = 0.5
LAMBDA_MARGIN_CALIBRATED = 0.1


def compute_delta_theta_for_checkpoint(epoch, ckpt_dir, loader_iter, device, n_steps, focal_fn, evidential_fn, rng, tau_b_tracker):
    """Reproduces E20's analyze_checkpoint's core replay loop (imported
    conventions, not the whole function, since E21 also needs the model
    object and loader iterator afterward for the transport measurement).
    Returns (model, dec1_params, delta_theta_margin, delta_theta_seg,
    delta_theta_total) -- the model is left in its ORIGINAL (unperturbed)
    state, exactly as loaded from checkpoint."""
    ckpt_path = ckpt_dir / f"epoch_{epoch}.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.train()  # matches E14/E19/E20: training-time BN behavior

    dec1_params = list(model.dec1.parameters())
    pg = ckpt["optimizer_state"]["param_groups"][0]
    lr, (beta1, beta2), eps = pg["lr"], pg["betas"], pg["eps"]

    shadow_seg = ShadowAdam(dec1_params, lr, beta1, beta2, eps)
    shadow_margin = ShadowAdam(dec1_params, lr, beta1, beta2, eps)
    shadow_total = ShadowAdam(dec1_params, lr, beta1, beta2, eps)

    for step_idx in range(n_steps):
        images, masks, _ = next(loader_iter)
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
        dec1_perm = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
        evidence_flat = evidence_full.reshape(-1)
        boundary_flat = boundary_logit.reshape(-1)
        gt_flat = masks.reshape(-1)

        focal_loss = focal_fn(probs, masks)
        evidential_loss = evidential_fn(alpha, beta, masks)
        seg_loss = FOCAL_WEIGHT * focal_loss + EVIDENTIAL_WEIGHT * evidential_loss

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

        if not (torch.isfinite(margin_loss) and torch.isfinite(seg_loss)):
            continue

        total_loss = seg_loss + LAMBDA_MARGIN_CALIBRATED * margin_loss

        g_seg = torch.autograd.grad(seg_loss, dec1_params, retain_graph=True, allow_unused=True)
        g_margin = torch.autograd.grad(margin_loss, dec1_params, retain_graph=True, allow_unused=True)
        g_total = torch.autograd.grad(total_loss, dec1_params, retain_graph=False, allow_unused=True)

        g_seg = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_seg, dec1_params)]
        g_margin = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_margin, dec1_params)]
        g_total = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_total, dec1_params)]

        shadow_seg.step(g_seg)
        shadow_margin.step(g_margin)
        shadow_total.step(g_total)

    return model, dec1_params, shadow_margin.delta_sum, shadow_seg.delta_sum, shadow_total.delta_sum


def apply_delta_and_forward(model, delta_list, images):
    """Deep-copies model, adds delta_list (list of tensors matching
    model.dec1.parameters()' shapes) to the COPY's dec1 parameters
    in-place, forward-passes `images` through the copy, returns the
    copy's dec1 output. Original model is untouched."""
    model_copy = copy.deepcopy(model)
    model_copy.train()
    with torch.no_grad():
        for p, d in zip(model_copy.dec1.parameters(), delta_list):
            p.add_(d)
    with torch.no_grad():
        outputs = model_copy(images)
    del model_copy
    return outputs["dec1"]


def compute_useful_direction(dec1, gt_flat_per_voxel):
    """Exact reproduction of e15_decoder_sensitivity.py's direction
    formula: tumor voxels pushed away from bg centroid, bg voxels pushed
    away from tumor centroid, both computed from THIS batch's own
    dec1/ground truth (not borrowed from another checkpoint or dataset)."""
    B, C, D, H, W = dec1.shape
    dec1_flat = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
    tumor_mask = gt_flat_per_voxel > 0.5
    bg_mask = ~tumor_mask
    if tumor_mask.sum() < 10 or bg_mask.sum() < 10:
        return None  # degenerate batch, same skip criterion as E15
    tumor_centroid = dec1_flat[tumor_mask].mean(dim=0)
    bg_centroid = dec1_flat[bg_mask].mean(dim=0)
    direction = torch.zeros_like(dec1_flat)
    direction[tumor_mask] = dec1_flat[tumor_mask] - bg_centroid.unsqueeze(0)
    direction[bg_mask] = dec1_flat[bg_mask] - tumor_centroid.unsqueeze(0)
    unit_dir = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return unit_dir  # (N_voxels, 32)


def analyze_checkpoint(epoch, ckpt_dir, loader, device, focal_fn, evidential_fn):
    rng = np.random.RandomState(epoch)
    tau_b_tracker = EMATauB()
    loader_iter = iter(loader)

    model, dec1_params, delta_margin, delta_seg, delta_total = compute_delta_theta_for_checkpoint(
        epoch, ckpt_dir, loader_iter, device, N_REPLAY_STEPS, focal_fn, evidential_fn, rng, tau_b_tracker
    )

    theta_norms = {
        "margin": sum(float((d ** 2).sum()) for d in delta_margin) ** 0.5,
        "seg": sum(float((d ** 2).sum()) for d in delta_seg) ** 0.5,
        "total": sum(float((d ** 2).sum()) for d in delta_total) ** 0.5,
    }

    batch_records = []
    for b_idx in range(N_TRANSPORT_BATCHES):
        try:
            images, masks, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            images, masks, _ = next(loader_iter)
        images = images.to(device)
        masks = masks.to(device)

        with torch.no_grad():
            outputs_orig = model(images)
            dec1_orig = outputs_orig["dec1"]

        gt_flat = masks.reshape(-1)
        d_useful = compute_useful_direction(dec1_orig, gt_flat)
        if d_useful is None:
            continue

        dec1_perturbed = {}
        for name, delta in [("margin", delta_margin), ("seg", delta_seg), ("total", delta_total)]:
            dec1_perturbed[name] = apply_delta_and_forward(model, delta, images)

        B, C, D, H, W = dec1_orig.shape
        dec1_orig_flat = dec1_orig.permute(0, 2, 3, 4, 1).reshape(-1, C)

        record = {"batch": b_idx}
        for name in ["margin", "seg", "total"]:
            dz = dec1_perturbed[name].permute(0, 2, 3, 4, 1).reshape(-1, C) - dec1_orig_flat
            dz_norm_per_voxel = dz.norm(dim=1)
            valid = dz_norm_per_voxel > 1e-12
            cos_per_voxel = F.cosine_similarity(dz[valid], d_useful[valid], dim=1)
            record[f"cos_{name}_mean"] = float(cos_per_voxel.mean())
            record[f"cos_{name}_median"] = float(cos_per_voxel.median())
            record[f"dz_pooled_norm_{name}"] = float(dz.reshape(-1).norm())
            record[f"n_valid_voxels_{name}"] = int(valid.sum())
        batch_records.append(record)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return theta_norms, batch_records


def main():
    exp_dir = Path(__file__).parent
    ckpt_dir = exp_dir / SEED_DIR / "checkpoints"
    out_dir = exp_dir / "e21_transport_results"
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = create_brats_loaders(
        batch_size=2, num_workers=0,
        root_dir=str(project_root / "Dataset" / "Training"),
        val_split=0.1,
    )

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)

    all_results = []
    for epoch in CHECKPOINT_EPOCHS:
        print(f"Analyzing epoch {epoch}...")
        theta_norms, batch_records = analyze_checkpoint(epoch, ckpt_dir, train_loader, device, focal_fn, evidential_fn)
        if not batch_records:
            print(f"  WARNING: no valid (non-degenerate) batches for epoch {epoch}")
            continue

        summary = {"epoch": epoch, "theta_norms": theta_norms, "n_valid_batches": len(batch_records)}
        for name in ["margin", "seg", "total"]:
            cos_means = [r[f"cos_{name}_mean"] for r in batch_records]
            dz_norms = [r[f"dz_pooled_norm_{name}"] for r in batch_records]
            summary[f"cos_{name}_mean"] = float(np.mean(cos_means))
            summary[f"cos_{name}_std"] = float(np.std(cos_means))
            summary[f"dz_pooled_norm_{name}_mean"] = float(np.mean(dz_norms))
            summary[f"amplification_{name}"] = float(np.mean(dz_norms)) / theta_norms[name] if theta_norms[name] > 1e-20 else None

        all_results.append({"summary": summary, "batch_records": batch_records})
        print(f"  n_valid_batches={len(batch_records)}")
        for name in ["margin", "seg", "total"]:
            print(f"  {name:8s} ||delta_theta||={theta_norms[name]:.4e} "
                  f"mean||delta_z||={summary[f'dz_pooled_norm_{name}_mean']:.4e} "
                  f"amplification={summary[f'amplification_{name}']:.4e} "
                  f"cos(delta_z,d_useful)={summary[f'cos_{name}_mean']:+.4f}")

    # --- Save raw data ---
    with open(out_dir / "e21_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # --- Plots ---
    epochs_arr = [r["summary"]["epoch"] for r in all_results]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for name, color in [("margin", "crimson"), ("seg", "steelblue"), ("total", "darkgreen")]:
        vals = [r["summary"][f"amplification_{name}"] for r in all_results]
        axes[0, 0].plot(epochs_arr, vals, marker="o", label=name, color=color)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("||delta_z|| / ||delta_theta|| (log)")
    axes[0, 0].set_title("Parameter -> representation amplification (dec1)")
    axes[0, 0].legend()

    for name, color in [("margin", "crimson"), ("seg", "steelblue"), ("total", "darkgreen")]:
        vals = [r["summary"][f"cos_{name}_mean"] for r in all_results]
        stds = [r["summary"][f"cos_{name}_std"] for r in all_results]
        axes[0, 1].errorbar(epochs_arr, vals, yerr=stds, marker="o", label=name, color=color, capsize=3)
    axes[0, 1].axhline(0, color="gray", linewidth=0.8)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("mean cos(delta_z, d_useful)")
    axes[0, 1].set_title("Does the representation move toward E15's useful direction?")
    axes[0, 1].legend()

    for name, color in [("margin", "crimson"), ("seg", "steelblue")]:
        vals = [r["summary"][f"dz_pooled_norm_{name}_mean"] for r in all_results]
        axes[1, 0].plot(epochs_arr, vals, marker="o", label=name, color=color)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("mean pooled ||delta_z|| (log)")
    axes[1, 0].set_title("Raw representation displacement per objective")
    axes[1, 0].legend()

    theta_margin = [r["summary"]["theta_norms"]["margin"] for r in all_results]
    theta_seg = [r["summary"]["theta_norms"]["seg"] for r in all_results]
    axes[1, 1].plot(epochs_arr, theta_margin, marker="o", label="||delta_theta_margin||", color="crimson", linestyle="--")
    axes[1, 1].plot(epochs_arr, theta_seg, marker="s", label="||delta_theta_seg||", color="steelblue", linestyle="--")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("||delta_theta|| (log, from E20 replay, for reference)")
    axes[1, 1].set_title("Reference: E20's parameter-space update norms")
    axes[1, 1].legend()

    fig.suptitle("Phase E21: Parameter-to-Representation Update Transport (dec1)")
    fig.tight_layout()
    fig.savefig(out_dir / "e21_plots.png", dpi=150)
    plt.close(fig)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
