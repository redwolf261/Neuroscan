"""
Phase E15: Decoder Sensitivity Experiment -- causal intervention, not
another correlational diagnostic. Tests the assumption everything since
E8 has rested on: "wider latent margin -> better segmentation."

Method: freeze a trained checkpoint entirely (no gradient steps, no
retraining). Take real dec1 activations from real validation volumes.
Manually push voxel embeddings AWAY from the opposite class -- exactly
the effect EGGO-M's margin loss is trying to (and, per E12f/E13, fails
to meaningfully) achieve -- by realistic artificial amounts (absolute
unit offsets spanning E12f's own observed margin trajectory range, see
below). Feed the MODIFIED z through the frozen seg_head. Measure Dice.

  - If Dice increases with larger artificial margin -> the decoder CAN
    use better geometry; EGGO-M's failure is an optimization problem
    (it never actually achieved a big enough / well-placed margin).
  - If Dice is flat/unchanged -> the decoder is invariant to this
    degree of freedom; the "wider margin -> better Dice" hypothesis
    itself is false for this architecture, regardless of how well any
    optimizer could ever push the margin.

ONE discrete manipulation direction is used, not two: an earlier version
of this script also tried a "push along w (seg_head's weight vector),
sign chosen per-voxel by its OWN ground-truth class" manipulation, meant
to isolate whether the decoder cares about the one axis that provably
determines its output. This was WRONG and dropped after the first smoke
test exposed it: choosing the push SIGN using each voxel's true label is
an oracle intervention -- at a large enough magnitude it trivially drives
every voxel's sigmoid(w.z+b) to the "correct" saturated value BY
CONSTRUCTION (observed directly: Dice=1.0000+-0.0000 in the smoke test),
which tests nothing about decoder sensitivity, only that logistic
regression with the true label as the sign source can always separate
its own construction. Caught before trusting the result, not after.

What remains, and is what this script actually measures:

  (A) EUCLIDEAN manipulation: push each voxel directly away from the
      local (same-volume) opposite-class centroid, in raw dec1 space --
      this is literally the geometry EGGO-M's De Brabandere-style hinge
      loss optimizes (raw pairwise Euclidean distance), and does NOT use
      the label to pick a decoder-specific direction (only to determine
      which centroid is "opposite," the same information EGGO-M's own
      loss uses for its class-conditional pairing) -- so this remains a
      fair simulation of "what if the margin loss had actually
      succeeded further," not an oracle shortcut.
  (B) JACOBIAN sensitivity: d(probs)/dz analytically, projected onto
      both the Euclidean push direction and w itself. This is a LOCAL,
      LABEL-FREE, linear-approximation measure -- it asks "how much does
      a small nudge in this direction change the decoder's output,"
      without ever using ground truth to pick which way is "correct."
      It directly answers whether the decoder is sensitive to w (the
      only direction that mathematically CAN matter, since seg_head is
      a single 1x1x1 conv) versus the Euclidean push direction, as a
      continuous complement to (A)'s discrete result.

Push magnitudes for (A) are anchored to REALISTIC scale, not an
arbitrary multiple of the natural per-voxel dec1 norm: E12f's own fully-
observed mean_boundary_margin trajectory ranged 14.16-28.04 across all
30 epochs (a ~14-unit dynamic range, the largest swing this architecture
was ever seen to produce under real training, whether by design or
noise) while natural per-voxel ||dec1|| is only ~8 -- pushing by the RAW
centroid-to-centroid distance (order 23-28) at even a 20% fraction was
already 1.5x the natural voxel norm in an earlier attempt, an unrealistic
excursion far outside anything EGGO-M's loss could plausibly produce.
Push magnitudes here are instead defined as absolute unit offsets
(0, 1, 2, 4, 8, 14 units) spanning from "no change" up to "the full
E12f-observed dynamic range," directly comparable to that trajectory's
own scale rather than a fraction of an unrelated quantity.

If Dice increases meaningfully as the push grows toward the realistic
end of this range: the decoder CAN use better geometry, and EGGO-M's
failure to improve Dice is best read as an optimization/calibration
problem (the loss never achieved a big-enough, well-placed push).
If Dice stays flat even out to the full realistic range: the decoder is
effectively invariant to Euclidean margin widening at any scale EGGO-M's
loss could plausibly have produced -- the "wider margin -> better Dice"
hypothesis is not supported for this architecture, independent of
optimizer quality.
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
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

CHECKPOINT_EPOCH = 30
SEED_DIR = "e12f_pilot_calibrated_seed0"
N_VAL_SUBJECTS = 20  # same fixed subset used throughout E12b.5/E12d/E12f/E13/E14

# Absolute unit pushes, not fractions of an unrelated quantity. Anchored
# to E12f's own observed mean_boundary_margin trajectory: min=14.16,
# max=28.04 across all 30 real epochs (see module docstring) -- so
# PUSH_UNITS spans "no change" through "the full realistic dynamic range
# this architecture was ever seen to produce" (14 units), plus two points
# beyond it (20, 28 units -- roughly 1.4x and 2x the observed range) to
# check whether any effect emerges only at unrealistically large,
# beyond-what-training-could-produce scales (informative context, not
# the headline claim, which rests on the <=14-unit region).
PUSH_UNITS = [0.0, 1.0, 2.0, 4.0, 8.0, 14.0, 20.0, 28.0]
NATURAL_VOXEL_NORM_REFERENCE = 8.03  # measured directly on this checkpoint, epoch 30, subject 0 -- for the off-manifold sanity check only, not used in the push formula itself


def dice_score(pred_binary, gt):
    tp = (pred_binary * gt).sum()
    return (2 * tp / (pred_binary.sum() + gt.sum() + 1e-6)).item()


def compute_manipulated_dice(model, dec1, gt, push_units):
    """
    dec1: (B,32,D,H,W), gt: (B,1,D,H,W) -- already on device. Pure
    forward-only intervention, no backward pass anywhere in this function.

    Pushes each voxel by an ABSOLUTE distance of `push_units` directly
    away from the local (same-volume) opposite-class centroid -- the
    Euclidean manipulation (A) described in the module docstring. Uses
    the ground-truth label only to determine which centroid is
    "opposite" (the same information EGGO-M's own loss uses for its
    class-conditional pairing), never to pick a decoder-specific
    direction -- not an oracle intervention.
    """
    B, C, D, H, W = dec1.shape
    dec1_flat = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (B*D*H*W, 32)
    gt_flat = gt.reshape(-1)

    tumor_mask = gt_flat > 0.5
    bg_mask = ~tumor_mask
    if tumor_mask.sum() == 0 or bg_mask.sum() == 0:
        return None, None  # degenerate volume, caller already filters these out

    tumor_centroid = dec1_flat[tumor_mask].mean(dim=0)
    bg_centroid = dec1_flat[bg_mask].mean(dim=0)

    direction = torch.zeros_like(dec1_flat)
    direction[tumor_mask] = dec1_flat[tumor_mask] - bg_centroid.unsqueeze(0)
    direction[bg_mask] = dec1_flat[bg_mask] - tumor_centroid.unsqueeze(0)
    unit_dir = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)

    modified = dec1_flat + push_units * unit_dir  # absolute unit push, same magnitude for every voxel regardless of its current distance
    modified = modified.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)  # back to (B,32,D,H,W)

    with torch.no_grad():
        probs = model.seg_head(modified)
    pred_binary = (probs >= 0.5).float()
    dice = dice_score(pred_binary, gt)

    # Off-manifold magnitude check: push_units relative to the natural
    # per-voxel dec1 norm -- distinguishes "decoder ignores margin" from
    # "decoder broke because input became unrealistic."
    natural_norm = dec1_flat.norm(dim=1).mean().item()
    relative_push = push_units / max(natural_norm, 1e-8)

    return dice, relative_push


def compute_jacobian_sensitivity(model, dec1, gt, w_unit, n_sample=2000, rng=None):
    """
    Analytic d(probs)/d(z) projected onto (a) the Euclidean push
    direction (per-voxel, away from opposite-class centroid) and
    (b) w itself -- a continuous, small-perturbation-limit companion to
    the discrete manipulation above. Uses autograd on a freshly
    requires_grad-enabled copy of dec1; NOT part of any training loop,
    purely a forward+backward-mode measurement on frozen weights.
    """
    B, C, D, H, W = dec1.shape
    dec1_flat = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C).detach().clone()
    gt_flat = gt.reshape(-1)

    total = dec1_flat.shape[0]
    n_sample = min(n_sample, total)
    idx = torch.from_numpy(rng.choice(total, size=n_sample, replace=False)).to(dec1.device)

    z = dec1_flat[idx].clone().requires_grad_(True)  # (n_sample, 32)
    # seg_head expects (B,32,D,H,W); treat sampled voxels as a
    # (n_sample,32,1,1,1) "batch" of single-voxel volumes -- valid since
    # seg_head is a pure 1x1x1 conv (a per-voxel linear map with no
    # spatial mixing), so this is mathematically identical to running
    # them in-place in the real volume.
    z_vol = z.reshape(n_sample, C, 1, 1, 1)
    probs = model.seg_head(z_vol).reshape(n_sample)

    grad_z, = torch.autograd.grad(probs.sum(), z, retain_graph=False)  # (n_sample, 32), d(sum probs)/dz -- sum is fine since seg_head is per-voxel independent, this recovers the per-voxel gradient exactly

    tumor_mask_s = gt_flat[idx] > 0.5
    bg_mask_s = ~tumor_mask_s

    with torch.no_grad():
        if tumor_mask_s.sum() > 0 and bg_mask_s.sum() > 0:
            tumor_centroid = dec1_flat[idx][tumor_mask_s].mean(dim=0)
            bg_centroid = dec1_flat[idx][bg_mask_s].mean(dim=0)
            euclid_dir = torch.zeros_like(z)
            euclid_dir[tumor_mask_s] = z[tumor_mask_s] - bg_centroid
            euclid_dir[bg_mask_s] = z[bg_mask_s] - tumor_centroid
            euclid_dir = euclid_dir / euclid_dir.norm(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            euclid_dir = torch.zeros_like(z)

        sensitivity_euclid = (grad_z * euclid_dir).sum(dim=1)  # directional derivative along Euclidean push dir
        sensitivity_w = grad_z @ w_unit                          # directional derivative along w
        grad_norm = grad_z.norm(dim=1)

    return {
        "grad_norm_mean": float(grad_norm.mean().item()),
        "sensitivity_euclid_mean_abs": float(sensitivity_euclid.abs().mean().item()),
        "sensitivity_w_mean_abs": float(sensitivity_w.abs().mean().item()),
        "sensitivity_euclid_vs_w_ratio": float((sensitivity_euclid.abs().mean() / sensitivity_w.abs().mean().clamp_min(1e-12)).item()),
    }


def main():
    exp_dir = Path(__file__).parent
    ckpt_dir = exp_dir / SEED_DIR / "checkpoints"
    out_dir = exp_dir / "e15_decoder_sensitivity_results"
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_use = min(N_VAL_SUBJECTS, len(val_dataset))
    print(f"Using {n_use} fixed validation subjects (same set as E12b.5/E12d/E12f/E13/E14)")

    ckpt = torch.load(ckpt_dir / f"epoch_{CHECKPOINT_EPOCH}.pth", map_location=device, weights_only=False)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()  # frozen for inference -- no training-mode BatchNorm concerns here since NO gradient step is ever taken and we want the model's actual deployed/eval-time seg_head behavior, not a training-time approximation (unlike E14, which specifically needed train() to match live gradients)
    for p in model.parameters():
        p.requires_grad_(False)

    # seg_head is Sequential(Conv3d(32,1,kernel_size=1), Sigmoid()) --
    # extract w as a (32,) unit vector (out_channels=1, so weight shape
    # is (1,32,1,1,1)).
    conv_layer = model.seg_head[0]
    w_raw = conv_layer.weight.detach().reshape(-1)  # (32,)
    w_unit = w_raw / w_raw.norm().clamp_min(1e-8)
    print(f"seg_head weight norm: {w_raw.norm().item():.4f}")

    rng = np.random.RandomState(0)

    all_dice = {u: [] for u in PUSH_UNITS}
    all_relpush = {u: [] for u in PUSH_UNITS}
    jacobian_results = []
    n_used = 0

    with torch.no_grad():
        for idx in range(n_use):
            image, mask, subject_id = val_dataset[idx]
            image_b = image.unsqueeze(0).to(device)
            mask_b = mask.unsqueeze(0).to(device)
            outputs = model(image_b)
            dec1 = outputs["dec1"]  # (1,32,D,H,W)

            gt_flat = mask_b.reshape(-1)
            tumor_mask = gt_flat > 0.5
            bg_mask = ~tumor_mask
            if tumor_mask.sum() < 10 or bg_mask.sum() < 10:
                continue  # skip near-degenerate volumes (too few tumor voxels for a meaningful centroid)
            n_used += 1

            for units in PUSH_UNITS:
                d, rp = compute_manipulated_dice(model, dec1, mask_b, units)
                if d is not None:
                    all_dice[units].append(d)
                    all_relpush[units].append(rp)

    print(f"Used {n_used}/{n_use} volumes (skipped degenerate ones with <10 voxels of either class)")

    # Jacobian sensitivity needs grad -- separate pass
    for idx in range(n_use):
        image, mask, subject_id = val_dataset[idx]
        image_b = image.unsqueeze(0).to(device)
        mask_b = mask.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_b)
            dec1 = outputs["dec1"]
        jac = compute_jacobian_sensitivity(model, dec1, mask_b, w_unit, rng=rng)
        jacobian_results.append(jac)

    # --- Aggregate ---
    print("\n" + "=" * 70)
    print("RESULT: Dice vs. artificial margin push (absolute units)")
    print("=" * 70)
    summary = []
    for units in PUSH_UNITS:
        d = np.array(all_dice[units])
        rp = np.array(all_relpush[units])
        print(f"push={units:5.1f} units: dice={d.mean():.4f}+-{d.std():.4f} (n={len(d)}, relative_push={rp.mean():.3f}x natural norm)")
        summary.append({"push_units": units, "dice_mean": float(d.mean()), "dice_std": float(d.std()),
                         "relative_push": float(rp.mean()), "n": int(len(d))})

    # Statistical test: does Dice trend with push magnitude? Both across
    # the FULL range tested and restricted to the REALISTIC region
    # (<=14 units, E12f's actual observed dynamic range) separately,
    # since a trend that only appears beyond 14 units would not be
    # attributable to anything EGGO-M's loss could have plausibly done.
    from scipy import stats
    units_arr = np.array(PUSH_UNITS)
    dice_means = np.array([np.mean(all_dice[u]) for u in PUSH_UNITS])
    r_full, p_full = stats.pearsonr(units_arr, dice_means)

    realistic_mask = units_arr <= 14.0
    r_real, p_real = stats.pearsonr(units_arr[realistic_mask], dice_means[realistic_mask])
    print(f"\ncorr(push_units, dice), FULL range (0-28): r={r_full:+.4f} (p={p_full:.4f})")
    print(f"corr(push_units, dice), REALISTIC range (0-14, E12f's observed span): r={r_real:+.4f} (p={p_real:.4f})")

    # Paired test: dice at push=14 (max realistic) vs push=0 (baseline), per volume
    d0 = np.array(all_dice[0.0])
    d14 = np.array(all_dice[14.0])
    t_paired, p_paired = stats.ttest_rel(d14, d0)
    print(f"\nPaired t-test, push=14 vs push=0 (per-volume): mean delta={d14.mean()-d0.mean():+.4f}, t={t_paired:.3f}, p={p_paired:.4f}")

    # Jacobian
    jac_grad_norm = np.mean([j["grad_norm_mean"] for j in jacobian_results])
    jac_euclid = np.mean([j["sensitivity_euclid_mean_abs"] for j in jacobian_results])
    jac_w = np.mean([j["sensitivity_w_mean_abs"] for j in jacobian_results])
    print(f"\nJacobian (analytic, small-perturbation limit):")
    print(f"  mean |d(probs)/dz| norm: {jac_grad_norm:.6f}")
    print(f"  mean |directional deriv along Euclidean push|: {jac_euclid:.6f}")
    print(f"  mean |directional deriv along w|: {jac_w:.6f}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "summary": summary,
            "correlation_full_range": {"r": float(r_full), "p": float(p_full)},
            "correlation_realistic_range": {"r": float(r_real), "p": float(p_real)},
            "paired_ttest_push14_vs_0": {"mean_delta": float(d14.mean() - d0.mean()), "t": float(t_paired), "p": float(p_paired)},
            "jacobian": {
                "grad_norm_mean": float(jac_grad_norm),
                "sensitivity_euclid_mean_abs": float(jac_euclid),
                "sensitivity_w_mean_abs": float(jac_w),
            },
        }, f, indent=2)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    dice_means_plot = [s["dice_mean"] for s in summary]
    dice_stds_plot = [s["dice_std"] for s in summary]
    axes[0].errorbar(PUSH_UNITS, dice_means_plot, yerr=dice_stds_plot,
                      marker="o", label="Euclidean push (matches EGGO-M's loss geometry)", color="steelblue", capsize=3)
    axes[0].axvline(14.0, color="gray", linestyle="--", linewidth=1, label="E12f's full observed margin range (realistic ceiling)")
    axes[0].axhline(dice_means_plot[0], color="gray", linestyle=":", linewidth=1, label="_nolegend_")
    axes[0].set_xlabel("Artificial push (absolute units away from opposite-class centroid)")
    axes[0].set_ylabel("Val Dice (frozen decoder, no retraining)")
    axes[0].set_title("Decoder sensitivity to artificial margin widening")
    axes[0].legend(fontsize=8)

    relpush_plot = [s["relative_push"] for s in summary]
    axes[1].plot(PUSH_UNITS, relpush_plot, marker="o", color="steelblue")
    axes[1].axvline(14.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Push (absolute units)")
    axes[1].set_ylabel("Push magnitude / natural ||dec1|| (off-manifold check)")
    axes[1].set_title("How far off-manifold is the manipulation?")

    fig.suptitle(f"Phase E15: Decoder Sensitivity (EGGO-M seed 0, epoch {CHECKPOINT_EPOCH} checkpoint)")
    fig.tight_layout()
    fig.savefig(out_dir / "decoder_sensitivity_plots.png", dpi=150)
    plt.close(fig)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
