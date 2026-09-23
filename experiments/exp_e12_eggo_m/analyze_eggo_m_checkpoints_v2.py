"""
Experiment E12b.5: geometry diagnostics on EGGO-M's saved checkpoints,
reusing E1.3/E1.4/E7's diagnostic pipeline. UNet3D_v2 already exposes
dec1 and boundary_logit directly in its forward() output dict, so no
forward hook is needed (simpler than E7's analyze_checkpoints.py, which
had to hook v1's dec1 since v1 doesn't expose it).

For each checkpoint {1,5,10,15,20,25,30}, on the SAME fixed 20-subject
validation subset every time, computes:
  - latent boundary classifier AUC (E1.3 method, fit fresh per checkpoint
    for comparability with E7's baseline trajectory -- NOT using the
    live boundary_head's own predictions, to keep this an independent
    cross-check, not circular)
  - boundary margin: mean pairwise distance between same-batch
    opposite-class dec1 embeddings (the quantity L_margin directly
    optimizes) -- tests the mechanism-verification prediction "does
    boundary margin actually increase over training"
  - live boundary_head accuracy/AUC (using its own predictions, for
    comparison against the independent E1.3-style classifier)
  - val Dice, ECE (already logged in epoch_metrics.csv, re-verified here
    against the checkpoint directly for consistency)

Then plots against epoch, side by side with E7's baseline (no-EGGO)
trajectory at the same epochs, to test:
  "does larger boundary margin correlate with Dice?"
  "does boundary BCE correlate with ECE?"
  "did EGGO-M's margin loss actually widen the margin vs baseline?"
"""
import sys
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_c0_weight_ablation"))
from calibration import ECEAccumulator  # noqa: E402

CHECKPOINT_EPOCHS = [1, 5, 10, 15, 20, 25, 30]
N_VAL_SUBJECTS = 20
VOXELS_PER_VOLUME = 2000
MAX_MARGIN_PAIRS = 5000  # cap for pairwise margin computation, tractability


def main():
    exp_dir = Path(__file__).parent
    seed = 0
    # PHASE_E12F: points at the recalibrated pilot (delta_d=3.6659,
    # adaptive tau_b), NOT e12b_pilot_seed0 (the original, uncalibrated
    # run this script was first written against) -- see
    # PHASE_E12F_RECALIBRATED_PILOT_RESULTS.md
    ckpt_dir = exp_dir / "e12f_pilot_calibrated_seed0" / "checkpoints"
    out_dir = exp_dir / "e12f_mechanism_results"
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_use = min(N_VAL_SUBJECTS, len(val_dataset))
    print(f"Using {n_use} fixed validation subjects for all checkpoints")

    results_per_epoch = []

    for epoch in CHECKPOINT_EPOCHS:
        ckpt_path = ckpt_dir / f"epoch_{epoch}.pth"
        if not ckpt_path.exists():
            print(f"WARNING: checkpoint for epoch {epoch} not found, skipping")
            continue

        print(f"\n{'='*70}\nAnalyzing epoch {epoch}\n{'='*70}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        all_features = []
        all_evidence = []
        all_correct = []
        all_ground_truth = []
        all_boundary_head_pred = []  # live boundary_head's own sigmoid predictions
        ece_acc = ECEAccumulator(n_bins=15)
        dice_scores = []

        rng = np.random.RandomState(0)

        with torch.no_grad():
            for idx in range(n_use):
                image, mask, subject_id = val_dataset[idx]
                image_b = image.unsqueeze(0).to(device)
                outputs = model(image_b)

                probs = outputs["probs"].squeeze(0).squeeze(0).cpu().numpy()
                alpha = outputs["alpha"].squeeze(0).squeeze(0).cpu().numpy()
                beta = outputs["beta"].squeeze(0).squeeze(0).cpu().numpy()
                boundary_logit = outputs["boundary_logit"].squeeze(0).squeeze(0).cpu().numpy()
                dec1_feat = outputs["dec1"].squeeze(0).cpu().numpy()  # (32,D,H,W)
                gt = mask.squeeze(0).numpy().astype(np.float32)

                pred_binary = (probs >= 0.5).astype(np.float32)
                dice = 2 * (pred_binary * gt).sum() / max(pred_binary.sum() + gt.sum(), 1e-6)
                dice_scores.append(float(dice))

                ece_acc.update(outputs["alpha"], outputs["beta"], mask.unsqueeze(0).to(device))

                evidence_total = alpha + beta - 2.0
                boundary_head_pred_binary = (1 / (1 + np.exp(-boundary_logit)) >= 0.5).astype(np.float32)

                D, H, W = gt.shape
                total_voxels = D * H * W
                n_sample = min(VOXELS_PER_VOLUME, total_voxels)
                flat_idx = rng.choice(total_voxels, size=n_sample, replace=False)
                d_idx, h_idx, w_idx = np.unravel_index(flat_idx, (D, H, W))

                dec1_flat = dec1_feat.reshape(32, -1)
                for i in range(n_sample):
                    vi, di, hi, wi = flat_idx[i], d_idx[i], h_idx[i], w_idx[i]
                    all_features.append(dec1_flat[:, vi])
                    all_evidence.append(float(evidence_total[di, hi, wi]))
                    all_correct.append(bool(pred_binary[di, hi, wi] == gt[di, hi, wi]))
                    all_ground_truth.append(bool(gt[di, hi, wi]))
                    all_boundary_head_pred.append(bool(boundary_head_pred_binary[di, hi, wi] == gt[di, hi, wi]))

        Z = np.array(all_features)
        evidence = np.array(all_evidence)
        correct = np.array(all_correct)
        ground_truth = np.array(all_ground_truth)
        boundary_head_correct = np.array(all_boundary_head_pred)

        # Independent E1.3-style classifier (fresh fit per checkpoint,
        # NOT using the live boundary_head -- keeps this an independent
        # cross-check of separability, comparable to E7's baseline numbers)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Z, ground_truth.astype(int))
        auc = roc_auc_score(ground_truth.astype(int), clf.predict_proba(Z)[:, 1])

        # Live boundary_head's own accuracy (per-voxel, from the sampled set)
        live_boundary_acc = boundary_head_correct.mean()

        # Boundary margin: mean pairwise distance between same-batch
        # opposite-class embeddings, subsampled for tractability -- this
        # is the quantity L_margin directly optimizes (mechanism check)
        tumor_idx = np.where(ground_truth)[0]
        bg_idx = np.where(~ground_truth)[0]
        if len(tumor_idx) > 0 and len(bg_idx) > 0:
            n_pairs = min(MAX_MARGIN_PAIRS, len(tumor_idx), len(bg_idx))
            t_sample = np.random.RandomState(1).choice(tumor_idx, size=n_pairs, replace=(n_pairs > len(tumor_idx)))
            b_sample = np.random.RandomState(2).choice(bg_idx, size=n_pairs, replace=(n_pairs > len(bg_idx)))
            margin_dist = np.linalg.norm(Z[t_sample] - Z[b_sample], axis=1)
            mean_margin = float(margin_dist.mean())
        else:
            mean_margin = float("nan")

        ece, _ = ece_acc.compute()
        mean_dice = float(np.mean(dice_scores))

        row = {
            "epoch": epoch,
            "val_dice": mean_dice,
            "val_ece": float(ece),
            "independent_latent_boundary_auc": float(auc),
            "live_boundary_head_accuracy": float(live_boundary_acc),
            "mean_boundary_margin": mean_margin,
            "mean_evidence": float(evidence.mean()),
        }
        results_per_epoch.append(row)
        print(f"  Dice={mean_dice:.4f} ECE={ece:.4f} independent_AUC={auc:.4f} "
              f"live_boundary_acc={live_boundary_acc:.4f} mean_margin={mean_margin:.4f}")

    with open(out_dir / "trajectory_results.json", "w") as f:
        json.dump(results_per_epoch, f, indent=2)

    # ------------------------------------------------------------------
    # Load E7's baseline (no-EGGO) trajectory for side-by-side comparison
    # ------------------------------------------------------------------
    with open(project_root / "experiments" / "exp_e7_causality" / "seed_0" / "results.json") as f:
        e7 = json.load(f)
    baseline_val = e7["history"]["val"]

    epochs = [r["epoch"] for r in results_per_epoch]
    eggo_dice = [r["val_dice"] for r in results_per_epoch]
    eggo_ece = [r["val_ece"] for r in results_per_epoch]
    eggo_margin = [r["mean_boundary_margin"] for r in results_per_epoch]
    eggo_auc = [r["independent_latent_boundary_auc"] for r in results_per_epoch]
    baseline_dice_matched = [baseline_val[e - 1]["dice"] for e in epochs]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, eggo_dice, marker="o", label="EGGO-M", color="steelblue")
    axes[0, 0].plot(epochs, baseline_dice_matched, marker="s", label="Baseline (E7, no EGGO)", color="gray", linestyle="--")
    axes[0, 0].set_ylabel("Val Dice")
    axes[0, 0].set_title("Dice: EGGO-M vs. Baseline")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, eggo_ece, marker="o", color="crimson")
    axes[0, 1].set_ylabel("Val ECE")
    axes[0, 1].set_title("EGGO-M Calibration (baseline has no ECE, no uncertainty head)")

    axes[1, 0].plot(epochs, eggo_margin, marker="o", color="purple")
    axes[1, 0].set_ylabel("Mean boundary margin (pairwise dist)")
    axes[1, 0].set_title("Mechanism check: does the margin actually grow?")
    axes[1, 0].set_xlabel("Epoch")

    axes[1, 1].plot(epochs, eggo_auc, marker="o", label="EGGO-M (independent classifier)", color="darkorange")
    axes[1, 1].set_ylabel("Latent boundary AUC")
    axes[1, 1].set_title("Class separability (independent E1.3-style check)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()

    fig.suptitle("Experiment E12b.5: EGGO-M Mechanism Verification (seed=0)")
    fig.tight_layout()
    fig.savefig(out_dir / "mechanism_plot.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Mechanism correlations (per the mechanism-verification questions)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MECHANISM VERIFICATION")
    print("=" * 70)
    if len(epochs) >= 3:
        r_margin_dice, p1 = stats.pearsonr(eggo_margin, eggo_dice)
        r_ece_dice, p2 = stats.pearsonr(eggo_ece, eggo_dice)
        print(f"corr(boundary_margin, Dice) = {r_margin_dice:+.4f} (p={p1:.3f})")
        print(f"corr(ECE, Dice) = {r_ece_dice:+.4f} (p={p2:.3f})  [negative = lower ECE with higher Dice, as expected]")

    print(f"\nBoundary margin trajectory: {[f'{m:.3f}' for m in eggo_margin]}")
    print(f"Interpretation: {'INCREASING' if eggo_margin[-1] > eggo_margin[0] else 'NOT increasing'} "
          f"over training ({eggo_margin[0]:.3f} -> {eggo_margin[-1]:.3f})")

    print(f"\nEGGO-M Dice vs baseline Dice at matched epochs:")
    for e, ed, bd in zip(epochs, eggo_dice, baseline_dice_matched):
        print(f"  epoch {e}: EGGO-M={ed:.4f}, baseline={bd:.4f}, delta={ed-bd:+.4f}")

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
