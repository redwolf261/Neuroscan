"""
Experiment E7 analysis: for each fixed-epoch checkpoint saved by
train_with_checkpoints.py, run the SAME diagnostic pipeline used in
E1.3/E1.4 (on a fixed validation subset, same subjects every checkpoint
for a fair trajectory comparison) and report:
  - latent decision-boundary classifier AUC (tumor vs background
    separability in dec1 space, E1.3 method)
  - latent boundary R^2 for predicting evidence (E1.3 method)
  - density Cohen's d (correct vs incorrect separation, E1.4 method, k=20)
  - mean evidence, mean confidence
  - val Dice, val ECE (standard eval metrics)

Then plots all of these against epoch number to see which comes first:
geometry structure (boundary AUC/R^2, density d) or calibration quality
(ECE) -- testing Hypothesis D from PHASE_E6_STRESS_TEST.md (is latent
geometry a plausible causal driver of uncertainty, or a downstream
consequence that only appears after calibration is already good?).

Fixed validation subset: same 20 subjects at every checkpoint (first 20
of the val split, deterministic), so trajectories are directly comparable
epoch-to-epoch -- not a different random sample each time.
"""
import sys
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import betaln, digamma
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, r2_score

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_fixed import UNet3D  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_c0_weight_ablation"))
from calibration import ECEAccumulator  # noqa: E402

STAGE_NAME = "dec1"
CHECKPOINT_EPOCHS = [1, 3, 5, 10, 20, 35, 50]
N_VAL_SUBJECTS = 20
K_DENSITY = 20


def register_dec1_hook(model):
    activations = {}

    def hook(module, inp, out):
        activations[STAGE_NAME] = out.detach()

    handle = model.dec1.register_forward_hook(hook)
    return activations, handle


def cohens_d(a, b):
    pooled_std = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0


def main():
    exp_dir = Path(__file__).parent
    seed = 0
    ckpt_dir = exp_dir / f"seed_{seed}" / "checkpoints"
    out_dir = exp_dir / "analysis_results"
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_use = min(N_VAL_SUBJECTS, len(val_dataset))
    print(f"Using {n_use} fixed validation subjects for all checkpoints (deterministic subset)")

    results_per_epoch = []

    for epoch in CHECKPOINT_EPOCHS:
        ckpt_path = ckpt_dir / f"epoch_{epoch}.pth"
        if not ckpt_path.exists():
            print(f"WARNING: checkpoint for epoch {epoch} not found at {ckpt_path}, skipping")
            continue

        print(f"\n{'='*70}\nAnalyzing epoch {epoch}\n{'='*70}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = UNet3D(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        activations, handle = register_dec1_hook(model)

        all_features = []
        all_evidence = []
        all_confidence = []
        all_correct = []
        all_ground_truth = []
        ece_acc = ECEAccumulator(n_bins=15)
        dice_scores = []

        rng = np.random.RandomState(0)
        VOXELS_PER_VOLUME = 2000

        with torch.no_grad():
            for idx in range(n_use):
                image, mask, subject_id = val_dataset[idx]
                image_b = image.unsqueeze(0).to(device)
                outputs = model(image_b)

                probs = outputs["probs"].squeeze(0).squeeze(0).cpu().numpy()
                alpha = outputs["alpha"].squeeze(0).squeeze(0).cpu().numpy()
                beta = outputs["beta"].squeeze(0).squeeze(0).cpu().numpy()
                gt = mask.squeeze(0).numpy().astype(np.float32)

                pred_binary = (probs >= 0.5).astype(np.float32)
                dice = 2 * (pred_binary * gt).sum() / max(pred_binary.sum() + gt.sum(), 1e-6)
                dice_scores.append(float(dice))

                ece_acc.update(outputs["alpha"], outputs["beta"], mask.unsqueeze(0).to(device))

                mu = alpha / (alpha + beta)
                confidence = np.where(pred_binary == 1, mu, 1 - mu)
                evidence_total = alpha + beta - 2.0

                dec1_feat = activations[STAGE_NAME].squeeze(0).cpu().numpy()  # (32,D,H,W)

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
                    all_confidence.append(float(confidence[di, hi, wi]))
                    all_correct.append(bool(pred_binary[di, hi, wi] == gt[di, hi, wi]))
                    all_ground_truth.append(bool(gt[di, hi, wi]))

        handle.remove()

        Z = np.array(all_features)
        evidence = np.array(all_evidence)
        confidence = np.array(all_confidence)
        correct = np.array(all_correct)
        ground_truth = np.array(all_ground_truth)

        # Latent boundary classifier (E1.3 method)
        if ground_truth.sum() >= 5 and (~ground_truth).sum() >= 5:
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(Z, ground_truth.astype(int))
            auc = roc_auc_score(ground_truth.astype(int), clf.predict_proba(Z)[:, 1])
            w = clf.coef_[0]
            b = clf.intercept_[0]
            latent_bd = (Z @ w + b) / np.linalg.norm(w)
            abs_latent_bd = np.abs(latent_bd)
            poly_r2 = {}
            for deg in (1, 2, 4):
                coeffs = np.polyfit(abs_latent_bd, evidence, deg)
                y_pred = np.polyval(coeffs, abs_latent_bd)
                poly_r2[deg] = float(r2_score(evidence, y_pred))
            best_r2 = max(poly_r2.values())
        else:
            auc, best_r2, abs_latent_bd = float("nan"), float("nan"), None
            print("  WARNING: insufficient class balance for latent boundary classifier at this epoch")

        # Density (E1.4 method, k=20)
        n_feat = Z.shape[0]
        k_use = min(K_DENSITY, n_feat - 1)
        nn = NearestNeighbors(n_neighbors=k_use + 1, n_jobs=-1)
        nn.fit(Z)
        distances, _ = nn.kneighbors(Z)
        rho = distances[:, 1:].mean(axis=1)

        if (~correct).sum() >= 2 and correct.sum() >= 2:
            d_density = cohens_d(rho[correct], rho[~correct])
            d_evidence = cohens_d(evidence[correct], evidence[~correct])
        else:
            d_density, d_evidence = float("nan"), float("nan")

        ece, _ = ece_acc.compute()
        mean_dice = float(np.mean(dice_scores))

        row = {
            "epoch": epoch,
            "val_dice": mean_dice,
            "val_ece": float(ece),
            "mean_evidence": float(evidence.mean()),
            "mean_confidence": float(confidence.mean()),
            "latent_boundary_auc": float(auc),
            "latent_boundary_r2_evidence": float(best_r2),
            "density_cohens_d": float(d_density),
            "evidence_cohens_d": float(d_evidence),
            "n_incorrect": int((~correct).sum()),
            "n_total": int(len(correct)),
        }
        results_per_epoch.append(row)
        print(f"  Dice={mean_dice:.4f} ECE={ece:.4f} latent_AUC={auc:.4f} "
              f"latent_R2={best_r2:.4f} density_d={d_density:.4f} evidence_d={d_evidence:.4f} "
              f"(n_incorrect={row['n_incorrect']}/{row['n_total']})")

    with open(out_dir / "trajectory_results.json", "w") as f:
        json.dump(results_per_epoch, f, indent=2)

    # ------------------------------------------------------------------
    # Plot trajectories
    # ------------------------------------------------------------------
    epochs = [r["epoch"] for r in results_per_epoch]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)

    axes[0, 0].plot(epochs, [r["val_dice"] for r in results_per_epoch], marker="o", color="steelblue")
    axes[0, 0].set_ylabel("Val Dice")
    axes[0, 0].set_title("Segmentation quality")

    axes[0, 1].plot(epochs, [r["val_ece"] for r in results_per_epoch], marker="o", color="crimson")
    axes[0, 1].set_ylabel("Val ECE")
    axes[0, 1].set_title("Calibration quality (lower = better)")

    axes[1, 0].plot(epochs, [r["latent_boundary_auc"] for r in results_per_epoch], marker="o", color="darkorange")
    axes[1, 0].set_ylabel("Latent boundary AUC")
    axes[1, 0].set_title("Geometry: class separability (E1.3)")

    axes[1, 1].plot(epochs, [r["latent_boundary_r2_evidence"] for r in results_per_epoch], marker="o", color="darkorange")
    axes[1, 1].set_ylabel("R^2 (latent boundary -> evidence)")
    axes[1, 1].set_title("Geometry: boundary predicts evidence (E1.3)")

    axes[2, 0].plot(epochs, [r["density_cohens_d"] for r in results_per_epoch], marker="o", color="purple")
    axes[2, 0].set_ylabel("Density Cohen's d (correct vs incorrect)")
    axes[2, 0].set_title("Geometry: density separates errors (E1.4)")
    axes[2, 0].set_xlabel("Epoch")

    axes[2, 1].plot(epochs, [r["mean_evidence"] for r in results_per_epoch], marker="o", color="teal")
    axes[2, 1].set_ylabel("Mean evidence")
    axes[2, 1].set_title("Evidential head output magnitude")
    axes[2, 1].set_xlabel("Epoch")

    fig.suptitle("Experiment E7: Geometry vs. Calibration vs. Segmentation Over Training (seed=0)")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_plot.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Correlation of epoch-rank with each quantity's "settling" pattern:
    # crude cross-correlation of when each series' derivative approaches 0
    # (numeric summary only; visual inspection of the plot is primary)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'epoch':<8}{'dice':<10}{'ece':<10}{'lat_auc':<10}{'lat_r2':<10}{'dens_d':<10}{'evid_d':<10}")
    for r in results_per_epoch:
        print(f"{r['epoch']:<8}{r['val_dice']:<10.4f}{r['val_ece']:<10.4f}{r['latent_boundary_auc']:<10.4f}"
              f"{r['latent_boundary_r2_evidence']:<10.4f}{r['density_cohens_d']:<10.4f}{r['evidence_cohens_d']:<10.4f}")

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
