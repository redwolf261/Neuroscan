"""
Phase E73 follow-up: does the PREDICTED (distilled) sensitivity map Ŝ_i(p)
-- not the real ablation map, which is expensive to compute at inference --
still correlate with the network's own segmentation errors, at above-chance
rate and beyond a boundary-proximity confound?

This is the decisive check before ever building a refinement branch: the
real map S_i(p) was already shown to correlate with error (22.6x ratio,
survives boundary control). The trained head was shown to predict S_i(p)
well (rho=0.39 pooled, 0.38 per-subject). This script closes the loop:
does the CHEAP, DISTILLED Ŝ_i(p) retain the property that made S worth
predicting in the first place? If yes, Ŝ is a genuinely useful,
inference-cheap substitute for the real (expensive) causal map. If the
correlation with error collapses even though Ŝ predicts S reasonably
well, that means the head is fitting the easy/generic part of S (e.g.
smooth spatial structure) while missing exactly the part that mattered.

DECISION RULE (pre-declared): PROCEED to a refinement-branch pilot only
if Ŝ (upsampled to 64^3 to align with voxel-level errors) correlates with
error voxels above chance AND that correlation survives controlling for
boundary distance, matching the bar the REAL map already cleared.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.ndimage import distance_transform_edt

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402
from train_spatial_sensitivity_predictor import SpatialSensitivityHead, decode_from_bottleneck  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_SAMPLE = 40  # matches the earlier E73 probe's sample size for direct comparability

MM_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e70" / "runs"
           / "MM_seed0" / "checkpoints" / "best.pth")
SPATIAL_HEAD_CKPT = OUT_DIR / "E73_spatial_head_state.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(str(MM_CKPT), map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    head = SpatialSensitivityHead(in_channels=256).to(device)
    head.load_state_dict(torch.load(str(SPATIAL_HEAD_CKPT), map_location=device))
    head.eval()
    print("Loaded frozen MM checkpoint + trained E73 spatial sensitivity head.")

    val_ds = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                    split="val", val_split=0.1, target_shape=(64, 64, 64))
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(val_ds), size=min(N_SAMPLE, len(val_ds)), replace=False)

    subj_raw_rhos = []
    subj_partial_rhos = []
    subj_error_ratio = []

    for count, idx in enumerate(sample_idx):
        img, msk, sid = val_ds[int(idx)]
        img_b = img.unsqueeze(0).to(device)
        target_bin = (msk.squeeze(0).numpy() > 0.5).astype(np.float32)

        with torch.no_grad():
            enc1 = model.enc1(img_b)
            pool1 = model.pool1(enc1); enc2 = model.enc2(pool1)
            pool2 = model.pool2(enc2); enc3 = model.enc3(pool2); pool3 = model.pool3(enc3)
            bottleneck = model.bottleneck(pool3)

            probs_intact = decode_from_bottleneck(model, bottleneck, enc1, enc2, enc3)
            probs_intact_np = probs_intact.squeeze(0).squeeze(0).cpu().numpy()

            # THE CHEAP INFERENCE-TIME PATH: predict Ŝ directly from the
            # bottleneck, no second (ablated) decode pass at all.
            S_hat_8 = head(bottleneck)  # (1, 8,8,8)
            S_hat_64 = F.interpolate(S_hat_8.unsqueeze(1), size=(64, 64, 64),
                                     mode="trilinear", align_corners=False)
            S_hat_np = S_hat_64.squeeze(0).squeeze(0).cpu().numpy()

        pred_bin = (probs_intact_np >= 0.5).astype(np.float32)
        error_mask = (pred_bin != target_bin).astype(np.float32)

        if target_bin.sum() == 0 or target_bin.sum() == target_bin.size:
            continue
        dist_out = distance_transform_edt(1 - target_bin)
        dist_in = distance_transform_edt(target_bin)
        boundary_dist = np.where(target_bin > 0, dist_in, dist_out)

        Sh_flat = S_hat_np.flatten()
        err_flat = error_mask.flatten()
        bd_flat = boundary_dist.flatten().astype(np.float64)

        n_vox = len(Sh_flat)
        vox_idx = rng.choice(n_vox, size=min(20000, n_vox), replace=False)
        Sh_s, err_s, bd_s = Sh_flat[vox_idx], err_flat[vox_idx], bd_flat[vox_idx]

        if err_s.std() > 0 and Sh_s.std() > 0:
            raw_rho, _ = stats.spearmanr(Sh_s, err_s)
        else:
            raw_rho = np.nan

        if bd_s.std() > 0:
            X = np.column_stack([np.ones(len(bd_s)), bd_s])
            beta_S, *_ = np.linalg.lstsq(X, Sh_s, rcond=None)
            resid_S = Sh_s - X @ beta_S
            beta_e, *_ = np.linalg.lstsq(X, err_s.astype(np.float64), rcond=None)
            resid_e = err_s.astype(np.float64) - X @ beta_e
            if resid_S.std() > 0 and resid_e.std() > 0:
                partial_rho, _ = stats.spearmanr(resid_S, resid_e)
            else:
                partial_rho = np.nan
        else:
            partial_rho = np.nan

        mean_at_error = Sh_flat[error_mask.flatten() > 0].mean() if error_mask.sum() > 0 else np.nan
        mean_at_correct = Sh_flat[error_mask.flatten() == 0].mean()
        ratio = mean_at_error / max(mean_at_correct, 1e-8)

        if not np.isnan(raw_rho):
            subj_raw_rhos.append(raw_rho)
        if not np.isnan(partial_rho):
            subj_partial_rhos.append(partial_rho)
        if not np.isnan(ratio):
            subj_error_ratio.append(ratio)

        if (count + 1) % 10 == 0:
            print(f"  processed {count+1}/{len(sample_idx)}", flush=True)

    subj_raw_rhos = np.array(subj_raw_rhos)
    subj_partial_rhos = np.array(subj_partial_rhos)
    subj_error_ratio = np.array(subj_error_ratio)

    print(f"\n=== PREDICTED map Ŝ vs. error: raw Spearman, n={len(subj_raw_rhos)} subjects ===")
    print(f"mean={subj_raw_rhos.mean():+.4f} std={subj_raw_rhos.std():.4f}")
    t_raw, p_raw = stats.ttest_1samp(subj_raw_rhos, 0.0)
    print(f"one-sample t-test vs 0: t={t_raw:.3f}, p={p_raw:.4e}")

    print(f"\nMean error/correct sensitivity ratio (predicted): mean={subj_error_ratio.mean():.3f} "
          f"(REAL map's earlier ratio was 22.6x -- for reference, not a like-for-like rescaling)")

    print(f"\n=== PREDICTED map Ŝ vs. error, CONTROLLING for boundary distance, n={len(subj_partial_rhos)} ===")
    print(f"mean={subj_partial_rhos.mean():+.4f} std={subj_partial_rhos.std():.4f}")
    t_p, p_p = stats.ttest_1samp(subj_partial_rhos, 0.0)
    w_p, p_w = stats.wilcoxon(subj_partial_rhos)
    print(f"one-sample t-test vs 0: t={t_p:.3f}, p={p_p:.4e}")
    print(f"Wilcoxon signed-rank vs 0: p={p_w:.4e}")

    survives = (subj_partial_rhos.mean() > 0) and (p_p < 0.05) and (p_w < 0.05)
    print(f"\n=== FINAL VERDICT: predicted-map error-correlation {'SURVIVES' if survives else 'DOES NOT SURVIVE'} "
          f"boundary-distance control ===")
    if survives:
        print("The cheap, distilled Ŝ (single forward pass, no ablation needed at inference) retains "
              "the property that made the real causal map worth predicting: it flags the network's own "
              "error-prone regions beyond simple boundary proximity. Worth building a refinement-branch pilot.")
    else:
        print("The distilled Ŝ does NOT reliably retain error-localizing information beyond boundary "
              "proximity -- the head is fitting the easy/generic part of S, missing the part that mattered. "
              "Do not proceed to a refinement branch on this basis.")

    summary = {
        "n_subjects": len(subj_raw_rhos),
        "raw_rho_mean": float(subj_raw_rhos.mean()), "raw_rho_p": float(p_raw),
        "error_ratio_mean": float(subj_error_ratio.mean()),
        "partial_rho_mean": float(subj_partial_rhos.mean()), "partial_rho_std": float(subj_partial_rhos.std()),
        "partial_ttest_p": float(p_p), "partial_wilcoxon_p": float(p_w),
        "survives_boundary_control": bool(survives),
    }
    with open(OUT_DIR / "E73_predicted_map_vs_error_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E73_predicted_map_vs_error_summary.json")


if __name__ == "__main__":
    main()
