"""
Phase E73 follow-up: does S's error-correlation (check b) survive controlling
for distance-to-lesion-boundary? Errors and high-S voxels could both simply
cluster near the lesion boundary independent of any deeper relationship --
this would make the E73 probe's "S correlates with error" result partly or
wholly a boundary-proximity confound, not evidence the signal is useful
beyond what's already known (boundaries are hard, unsurprising).

Reuses the per-subject S/target/pred arrays -- recomputed here since the
probe script didn't persist full volumes (only summary scalars), kept
identical in construction to check_spatial_sensitivity_probe.py.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from scipy.ndimage import distance_transform_edt

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_SAMPLE = 40

MM_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e70" / "runs"
           / "MM_seed0" / "checkpoints" / "best.pth")


def decode_from_bottleneck(model, bottleneck, enc1, enc2, enc3):
    up3 = model.upconv3(bottleneck)
    dec3 = model.dec3(torch.cat([up3, enc3], dim=1))
    up2 = model.upconv2(dec3)
    dec2 = model.dec2(torch.cat([up2, enc2], dim=1))
    up1 = model.upconv1(dec2)
    dec1 = model.dec1(torch.cat([up1, enc1], dim=1))
    return model.seg_head(dec1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(MM_CKPT), map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    val_ds = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                    split="val", val_split=0.1, target_shape=(64, 64, 64))
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(val_ds), size=min(N_SAMPLE, len(val_ds)), replace=False)

    # per-subject Spearman(S, boundary_distance) and partial correlation of
    # error-vs-S controlling for boundary distance
    subj_partial_rhos = []
    subj_raw_rhos = []

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
            probs_ablated = decode_from_bottleneck(model, torch.zeros_like(bottleneck), enc1, enc2, enc3)
            probs_intact_np = probs_intact.squeeze(0).squeeze(0).cpu().numpy()
            probs_ablated_np = probs_ablated.squeeze(0).squeeze(0).cpu().numpy()

        S = np.abs(probs_intact_np - probs_ablated_np)
        pred_bin = (probs_intact_np >= 0.5).astype(np.float32)
        error_mask = ((pred_bin != target_bin)).astype(np.float32)

        # distance to nearest lesion-boundary voxel (both inside and outside
        # the lesion contribute boundary proximity)
        if target_bin.sum() == 0 or target_bin.sum() == target_bin.size:
            continue  # degenerate, skip (no boundary to measure)
        dist_out = distance_transform_edt(1 - target_bin)  # distance from outside lesion
        dist_in = distance_transform_edt(target_bin)       # distance from inside lesion
        boundary_dist = np.where(target_bin > 0, dist_in, dist_out)  # signed-ish, always >=0, 0 AT boundary

        S_flat = S.flatten()
        err_flat = error_mask.flatten()
        bd_flat = boundary_dist.flatten().astype(np.float64)

        # subsample voxels for tractable correlation (64^3 = 262144 voxels/subject)
        n_vox = len(S_flat)
        vox_idx = rng.choice(n_vox, size=min(20000, n_vox), replace=False)
        S_s, err_s, bd_s = S_flat[vox_idx], err_flat[vox_idx], bd_flat[vox_idx]

        # raw correlation S vs error
        if err_s.std() > 0 and S_s.std() > 0:
            raw_rho, _ = stats.spearmanr(S_s, err_s)
        else:
            raw_rho = np.nan

        # partial correlation controlling for boundary distance (residualize
        # both S and error against boundary distance via rank regression proxy:
        # use boundary_dist as a covariate in a linear residualization on ranks)
        if bd_s.std() > 0:
            X = np.column_stack([np.ones(len(bd_s)), bd_s])
            beta_S, *_ = np.linalg.lstsq(X, S_s, rcond=None)
            resid_S = S_s - X @ beta_S
            beta_e, *_ = np.linalg.lstsq(X, err_s.astype(np.float64), rcond=None)
            resid_e = err_s.astype(np.float64) - X @ beta_e
            if resid_S.std() > 0 and resid_e.std() > 0:
                partial_rho, _ = stats.spearmanr(resid_S, resid_e)
            else:
                partial_rho = np.nan
        else:
            partial_rho = np.nan

        if not np.isnan(raw_rho):
            subj_raw_rhos.append(raw_rho)
        if not np.isnan(partial_rho):
            subj_partial_rhos.append(partial_rho)

        if (count + 1) % 10 == 0:
            print(f"  processed {count+1}/{len(sample_idx)}", flush=True)

    subj_raw_rhos = np.array(subj_raw_rhos)
    subj_partial_rhos = np.array(subj_partial_rhos)

    print(f"\n=== Raw Spearman(S, error) per subject, n={len(subj_raw_rhos)} ===")
    print(f"mean={subj_raw_rhos.mean():+.4f} std={subj_raw_rhos.std():.4f}")
    t_raw, p_raw = stats.ttest_1samp(subj_raw_rhos, 0.0)
    print(f"one-sample t-test vs 0: t={t_raw:.3f}, p={p_raw:.4e}")

    print(f"\n=== Partial Spearman(S, error | boundary_distance) per subject, n={len(subj_partial_rhos)} ===")
    print(f"mean={subj_partial_rhos.mean():+.4f} std={subj_partial_rhos.std():.4f}")
    t_p, p_p = stats.ttest_1samp(subj_partial_rhos, 0.0)
    print(f"one-sample t-test vs 0: t={t_p:.3f}, p={p_p:.4e}")

    survives = (subj_partial_rhos.mean() > 0) and (p_p < 0.05)
    print(f"\n=== VERDICT: S-error correlation {'SURVIVES' if survives else 'DOES NOT SURVIVE'} "
          f"controlling for boundary distance ===")
    if survives:
        print("The relationship is not simply 'both S and errors cluster near the lesion boundary' -- "
              "S carries information about error location beyond boundary proximity alone.")
    else:
        print("Once boundary proximity is controlled for, S's apparent correlation with error "
              "collapses -- the E73 probe's headline result is substantially a boundary-proximity "
              "confound, not new information.")


if __name__ == "__main__":
    main()
