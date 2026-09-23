"""
Experiment E12d: Mechanism Failure Investigation.

Post-hoc analysis (no retraining) on the 7 checkpoints already saved by
E12b's pilot run, computing the diagnostics needed to distinguish the 4
hypotheses for why the margin loss didn't grow the boundary margin:

  H1: the hinge is almost never active (most sampled pairs already
      satisfy dist > 2*delta_d, so the loss is 0 for them -- nothing to
      optimize)
  H2: lambda is too weak relative to L_seg's gradient magnitude
      (checked separately, requires live gradients -- see
      e12d_gradient_norms.py, NOT this script)
  H3: the boundary head/gate saturates immediately, so U_i*B_i collapses
      to near-0 or near-1 for almost all voxels, providing no useful
      per-voxel weighting signal
  H4: latent separation was already near-ceiling from the start (little
      room for the margin objective to act on, regardless of whether
      it's "active" in the H1 sense)

Computes, per checkpoint:
  1. Active hinge percentage: fraction of sampled same/opposite-class
     pairs with dist < 2*delta_d (i.e. where the hinge term is nonzero)
  2. Mean pairwise distance (already computed in analyze_eggo_m_checkpoints.py,
     recomputed here for a self-contained diagnostic run)
  3. Full pairwise distance histogram
  4. Margin violation histogram: the actual hinge value distribution,
     [2*delta_d - dist]_+^2, not just its mean
  5. (gradient norms: NOT here, see e12d_gradient_norms.py)
  6. Distribution histograms of U_hat, B_i, and U_hat*B_i (the actual
     per-voxel weights that would be applied, at delta_d=1.0, tau_b as
     used in the E12b run)
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

CHECKPOINT_EPOCHS = [1, 5, 10, 15, 20, 25, 30]
N_VAL_SUBJECTS = 20
VOXELS_PER_VOLUME = 2000
MAX_MARGIN_PAIRS = 5000
DELTA_D = 1.0  # matches the E12b run
TAU_B = float(np.sqrt(1.456 * 0.648))  # matches train_eggo_m.py's TAU_B
EVIDENCE_P99 = 23.25  # matches train_eggo_m.py's EVIDENCE_P99_DEFAULT


def main():
    exp_dir = Path(__file__).parent
    ckpt_dir = exp_dir / "e12b_pilot_seed0" / "checkpoints"
    out_dir = exp_dir / "e12d_results"
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_use = min(N_VAL_SUBJECTS, len(val_dataset))

    results_per_epoch = []
    all_distance_samples = {}
    all_weight_samples = {}

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
        all_boundary_logit = []
        all_ground_truth = []

        rng = np.random.RandomState(0)

        with torch.no_grad():
            for idx in range(n_use):
                image, mask, subject_id = val_dataset[idx]
                image_b = image.unsqueeze(0).to(device)
                outputs = model(image_b)

                alpha = outputs["alpha"].squeeze(0).squeeze(0).cpu().numpy()
                beta = outputs["beta"].squeeze(0).squeeze(0).cpu().numpy()
                boundary_logit = outputs["boundary_logit"].squeeze(0).squeeze(0).cpu().numpy()
                dec1_feat = outputs["dec1"].squeeze(0).cpu().numpy()
                gt = mask.squeeze(0).numpy().astype(np.float32)

                evidence_total = alpha + beta - 2.0

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
                    all_boundary_logit.append(float(boundary_logit[di, hi, wi]))
                    all_ground_truth.append(bool(gt[di, hi, wi]))

        Z = np.array(all_features)
        evidence = np.array(all_evidence)
        boundary_logit_arr = np.array(all_boundary_logit)
        ground_truth = np.array(all_ground_truth)

        # --- H3: U_hat, B_i, U_hat*B_i distributions ---
        U_hat = 1.0 - np.clip(evidence / EVIDENCE_P99, 0.0, 1.0)
        B_i = np.exp(-np.abs(boundary_logit_arr) / TAU_B)
        UB = U_hat * B_i

        print(f"  U_hat: mean={U_hat.mean():.4f} std={U_hat.std():.4f} "
              f"frac_near_0(<0.01)={np.mean(U_hat < 0.01):.4f} frac_near_1(>0.99)={np.mean(U_hat > 0.99):.4f}")
        print(f"  B_i:   mean={B_i.mean():.4f} std={B_i.std():.4f} "
              f"frac_near_0(<0.01)={np.mean(B_i < 0.01):.4f} frac_near_1(>0.99)={np.mean(B_i > 0.99):.4f}")
        print(f"  U*B:   mean={UB.mean():.4f} std={UB.std():.4f} "
              f"frac_near_0(<0.001)={np.mean(UB < 0.001):.4f}")

        # --- H1/H4: pairwise distances and hinge activity ---
        tumor_idx = np.where(ground_truth)[0]
        bg_idx = np.where(~ground_truth)[0]
        if len(tumor_idx) > 0 and len(bg_idx) > 0:
            n_pairs = min(MAX_MARGIN_PAIRS, len(tumor_idx), len(bg_idx))
            t_sample = np.random.RandomState(1).choice(tumor_idx, size=n_pairs, replace=(n_pairs > len(tumor_idx)))
            b_sample = np.random.RandomState(2).choice(bg_idx, size=n_pairs, replace=(n_pairs > len(bg_idx)))
            distances = np.linalg.norm(Z[t_sample] - Z[b_sample], axis=1)

            active_mask = distances < (2 * DELTA_D)
            active_pct = active_mask.mean()
            hinge_values = np.clip(2 * DELTA_D - distances, 0, None) ** 2
            mean_hinge = hinge_values.mean()

            print(f"  Pairwise distance: mean={distances.mean():.3f} std={distances.std():.3f} "
                  f"min={distances.min():.3f} p5={np.percentile(distances,5):.3f}")
            print(f"  Active hinge (dist < 2*delta_d={2*DELTA_D}): {active_pct*100:.4f}% of pairs")
            print(f"  Mean hinge value: {mean_hinge:.6f}")
        else:
            active_pct, mean_hinge, distances = float("nan"), float("nan"), np.array([])

        row = {
            "epoch": epoch,
            "U_hat_mean": float(U_hat.mean()), "U_hat_frac_near_0": float(np.mean(U_hat < 0.01)),
            "B_i_mean": float(B_i.mean()), "B_i_frac_near_0": float(np.mean(B_i < 0.01)),
            "UB_mean": float(UB.mean()), "UB_frac_near_0": float(np.mean(UB < 0.001)),
            "mean_pairwise_distance": float(distances.mean()) if len(distances) else float("nan"),
            "active_hinge_pct": float(active_pct) if not np.isnan(active_pct) else float("nan"),
            "mean_hinge_value": float(mean_hinge) if not np.isnan(mean_hinge) else float("nan"),
        }
        results_per_epoch.append(row)
        all_distance_samples[epoch] = distances.tolist()
        all_weight_samples[epoch] = {"U_hat": U_hat.tolist(), "B_i": B_i.tolist(), "UB": UB.tolist()}

    with open(out_dir / "mechanism_diagnosis_results.json", "w") as f:
        json.dump(results_per_epoch, f, indent=2)

    # ------------------------------------------------------------------
    # Dashboard: 6 panels (5 of the requested 6 -- gradient norm ratio
    # is in e12d_gradient_norms.py, a separate live-training script)
    # ------------------------------------------------------------------
    epochs = [r["epoch"] for r in results_per_epoch]
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    axes[0, 0].plot(epochs, [r["active_hinge_pct"] * 100 for r in results_per_epoch], marker="o", color="crimson")
    axes[0, 0].set_ylabel("Active hinge %")
    axes[0, 0].set_title("Panel 1: % of pairs with dist < 2*delta_d (hinge is active)")
    axes[0, 0].set_ylim(bottom=0)

    axes[0, 1].plot(epochs, [r["mean_pairwise_distance"] for r in results_per_epoch], marker="o", color="purple")
    axes[0, 1].axhline(2 * DELTA_D, color="gray", linestyle="--", label=f"2*delta_d={2*DELTA_D}")
    axes[0, 1].set_ylabel("Mean pairwise distance")
    axes[0, 1].set_title("Panel 2: Mean pair distance vs. hinge threshold")
    axes[0, 1].legend()

    # Panel 3: distance histograms, overlaid for first/mid/last epoch
    for e, color in zip([epochs[0], epochs[len(epochs)//2], epochs[-1]], ["lightblue", "orange", "darkred"]):
        if e in all_distance_samples:
            axes[1, 0].hist(all_distance_samples[e], bins=60, alpha=0.4, density=True, label=f"epoch {e}", color=color)
    axes[1, 0].axvline(2 * DELTA_D, color="black", linestyle="--", label=f"2*delta_d={2*DELTA_D}")
    axes[1, 0].set_xlabel("Pairwise distance")
    axes[1, 0].set_title("Panel 3: Distance histogram (early/mid/late epoch)")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, [r["mean_hinge_value"] for r in results_per_epoch], marker="o", color="darkgreen")
    axes[1, 1].set_ylabel("Mean hinge value [2*delta-d]+^2")
    axes[1, 1].set_title("Panel 4: Margin violation magnitude")
    axes[1, 1].set_yscale("log")

    # Panel 6: U_hat, B_i, UB distributions at the LAST epoch (most informative for "is the gate saturated by the end")
    last_epoch = epochs[-1]
    w = all_weight_samples[last_epoch]
    axes[2, 0].hist(w["U_hat"], bins=50, alpha=0.6, label="U_hat", color="teal")
    axes[2, 0].hist(w["B_i"], bins=50, alpha=0.6, label="B_i", color="orange")
    axes[2, 0].set_title(f"Panel 6a: U_hat / B_i distributions at epoch {last_epoch}")
    axes[2, 0].legend()
    axes[2, 0].set_yscale("log")

    axes[2, 1].hist(w["UB"], bins=50, color="darkviolet")
    axes[2, 1].set_title(f"Panel 6b: U_hat*B_i (actual applied weight) at epoch {last_epoch}")
    axes[2, 1].set_yscale("log")

    fig.suptitle("Experiment E12d: Mechanism Failure Investigation")
    fig.tight_layout()
    fig.savefig(out_dir / "mechanism_diagnosis_dashboard.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    last = results_per_epoch[-1]
    first = results_per_epoch[0]
    print(f"H1 (hinge rarely active): active_hinge_pct at epoch {last['epoch']} = {last['active_hinge_pct']*100:.4f}%")
    if last["active_hinge_pct"] < 0.05:
        print("  -> H1 SUPPORTED: hinge is active for <5% of pairs by the end of training. "
              "The loss has almost nothing left to optimize.")
    else:
        print(f"  -> H1 NOT clearly supported: {last['active_hinge_pct']*100:.1f}% of pairs still violate the margin.")

    print(f"\nH3 (gate saturation): UB_frac_near_0 at epoch {last['epoch']} = {last['UB_frac_near_0']*100:.2f}%")
    if last["UB_frac_near_0"] > 0.9:
        print("  -> H3 SUPPORTED: the combined gate U_hat*B_i is near-zero for >90% of sampled voxels by "
              "the end of training. Almost no anchors receive meaningful margin-loss weight.")
    else:
        print(f"  -> H3 NOT clearly supported: only {last['UB_frac_near_0']*100:.1f}% of weights are near-zero.")

    print(f"\nH4 (already near ceiling): mean pairwise distance at epoch 1 = {first['mean_pairwise_distance']:.2f} "
          f"vs. hinge threshold 2*delta_d = {2*DELTA_D}")
    if first["mean_pairwise_distance"] > 4 * DELTA_D:
        print(f"  -> H4 SUPPORTED: mean distance ({first['mean_pairwise_distance']:.2f}) is already far beyond "
              f"the hinge's active region ({2*DELTA_D}) even at epoch 1 -- delta_d=1.0 may be set far too small "
              f"relative to this feature space's natural scale, meaning the margin was 'already satisfied' from the start.")

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
