"""
Experiment E13: mechanism verification across all 4 seeds (0,1,2,3),
generalizing analyze_eggo_m_checkpoints_v2.py (which only covered seed 0)
to loop over seeds and produce cross-seed mean+/-std for the mechanism
metrics, matching the same rigor as the headline Dice comparison.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
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
MAX_MARGIN_PAIRS = 5000

SEED_DIRS = {
    0: "e12f_pilot_calibrated_seed0",
    1: "e13_pilot_calibrated_seed1",
    2: "e13_pilot_calibrated_seed2",
    3: "e13_pilot_calibrated_seed3",
}


def analyze_seed(seed, val_dataset, device):
    exp_dir = Path(__file__).parent
    ckpt_dir = exp_dir / SEED_DIRS[seed] / "checkpoints"

    n_use = min(N_VAL_SUBJECTS, len(val_dataset))
    results_per_epoch = []

    for epoch in CHECKPOINT_EPOCHS:
        ckpt_path = ckpt_dir / f"epoch_{epoch}.pth"
        if not ckpt_path.exists():
            print(f"  WARNING seed {seed}: checkpoint epoch {epoch} not found, skipping")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        all_features, all_ground_truth, all_boundary_head_pred = [], [], []
        ece_acc = ECEAccumulator(n_bins=15)
        dice_scores = []
        rng = np.random.RandomState(0)

        with torch.no_grad():
            for idx in range(n_use):
                image, mask, subject_id = val_dataset[idx]
                image_b = image.unsqueeze(0).to(device)
                outputs = model(image_b)

                probs = outputs["probs"].squeeze(0).squeeze(0).cpu().numpy()
                boundary_logit = outputs["boundary_logit"].squeeze(0).squeeze(0).cpu().numpy()
                dec1_feat = outputs["dec1"].squeeze(0).cpu().numpy()
                gt = mask.squeeze(0).numpy().astype(np.float32)

                pred_binary = (probs >= 0.5).astype(np.float32)
                dice = 2 * (pred_binary * gt).sum() / max(pred_binary.sum() + gt.sum(), 1e-6)
                dice_scores.append(float(dice))

                ece_acc.update(outputs["alpha"], outputs["beta"], mask.unsqueeze(0).to(device))

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
                    all_ground_truth.append(bool(gt[di, hi, wi]))
                    all_boundary_head_pred.append(bool(boundary_head_pred_binary[di, hi, wi] == gt[di, hi, wi]))

        Z = np.array(all_features)
        ground_truth = np.array(all_ground_truth)
        boundary_head_correct = np.array(all_boundary_head_pred)

        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Z, ground_truth.astype(int))
        auc = roc_auc_score(ground_truth.astype(int), clf.predict_proba(Z)[:, 1])

        live_boundary_acc = boundary_head_correct.mean()

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
        }
        results_per_epoch.append(row)
        print(f"  seed {seed} epoch {epoch}: Dice={mean_dice:.4f} ECE={ece:.4f} "
              f"AUC={auc:.4f} margin={mean_margin:.4f}")

    return results_per_epoch


def main():
    exp_dir = Path(__file__).parent
    out_dir = exp_dir / "e13_mechanism_results_all_seeds"
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )

    all_seed_results = {}
    for seed in [0, 1, 2, 3]:
        print(f"\n{'='*70}\nSeed {seed}\n{'='*70}")
        all_seed_results[seed] = analyze_seed(seed, val_dataset, device)

    with open(out_dir / "all_seeds_trajectory.json", "w") as f:
        json.dump(all_seed_results, f, indent=2)

    # -------------------------------------------------------------
    # Cross-seed mechanism correlation + final-epoch margin/AUC stats
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CROSS-SEED MECHANISM VERIFICATION")
    print("=" * 70)

    final_margins, final_aucs, final_dices, corrs = [], [], [], []
    for seed, rows in all_seed_results.items():
        if len(rows) < 3:
            continue
        margins = [r["mean_boundary_margin"] for r in rows]
        dices = [r["val_dice"] for r in rows]
        r_val, p_val = stats.pearsonr(margins, dices)
        corrs.append(r_val)
        final_margins.append(margins[-1])
        final_aucs.append(rows[-1]["independent_latent_boundary_auc"])
        final_dices.append(dices[-1])
        print(f"seed {seed}: corr(margin, Dice) = {r_val:+.4f} (p={p_val:.3f}), "
              f"margin traj = {[f'{m:.2f}' for m in margins]}")

    print(f"\nFinal-epoch margin: mean={np.mean(final_margins):.3f} std={np.std(final_margins, ddof=1):.3f}")
    print(f"Final-epoch independent AUC: mean={np.mean(final_aucs):.4f} std={np.std(final_aucs, ddof=1):.4f}")
    print(f"Final-epoch Dice (this script's own recompute): mean={np.mean(final_dices):.4f} std={np.std(final_dices, ddof=1):.4f}")
    print(f"Cross-seed correlation coefficients (margin vs Dice): {[f'{c:+.4f}' for c in corrs]}")
    print(f"Mean correlation across seeds: {np.mean(corrs):+.4f}")

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
