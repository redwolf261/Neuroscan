"""
Phase E, Priorities 1/2/3/6/7/8: extract per-volume feature maps and
per-voxel statistics from the frozen baseline model over validation
volumes. Single extraction pass -- everything downstream (CKA, boundary
correlation, t-SNE, failure-case study) reads from these saved artifacts
rather than re-running inference.

Treats the trained baseline as a scientific instrument: hooks
intermediate activations (enc1/enc2/enc3/bottleneck/dec3/dec2/dec1) via
forward hooks so nothing in neuroscan_3d_fixed.py needs to change
(frozen per baseline_frozen_milestone -- extraction must not require
editing the model file).

Per volume, saves:
  features_{stage}.npy   for stage in (enc1, enc2, enc3, bottleneck, dec3, dec2, dec1)
                          -- dec1 is the shared trunk output (Priority 1:
                          "encoder output before branching" and "seg/evidential
                          head input" are literally the same tensor here,
                          since both heads are 1x1 convs applied directly to dec1)
  prediction.npy         -- probs, (D,H,W)
  ground_truth.npy       -- binary mask, (D,H,W)
  alpha.npy, beta.npy    -- evidential Beta params, (D,H,W) each
  boundary_distance.npy  -- per-voxel distance to nearest GT boundary, (D,H,W)

Per-voxel long-format table (all volumes concatenated) written to
per_voxel_stats.parquet with columns:
  subject_id, voxel_idx, dec1_feature (32-dim, stored as separate f0..f31
    columns), alpha, beta, evidence, entropy, confidence, mu,
  prediction, ground_truth, correct, boundary_distance, feature_norm

Usage: python extract_features.py --n_volumes 30 --seed 0
"""
import os
import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_fixed import UNet3D  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402


STAGE_NAMES = ["enc1", "enc2", "enc3", "bottleneck", "dec3", "dec2", "dec1"]


def register_hooks(model):
    """Attach forward hooks to capture intermediate activations without
    touching neuroscan_3d_fixed.py (frozen baseline, do not edit)."""
    activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach()
        return hook

    handles = []
    for name in STAGE_NAMES:
        module = getattr(model, name)
        handles.append(module.register_forward_hook(make_hook(name)))
    return activations, handles


def boundary_distance_map(binary_mask):
    """
    Per-voxel distance to nearest GT boundary, signed by inside/outside.
    Positive = inside tumor (distance to nearest background voxel),
    negative = outside tumor (distance to nearest tumor voxel), 0 at
    the boundary itself. Uses scipy's exact Euclidean distance transform.
    """
    mask = binary_mask.astype(bool)
    if mask.all() or (~mask).all():
        # degenerate case: no boundary exists (all-tumor or all-background
        # volume) -- distance is undefined in the usual sense; return a
        # large constant sentinel distance with correct sign so downstream
        # analysis can filter these voxels out explicitly rather than
        # silently getting nonsense (distance_transform_edt would return 0
        # everywhere here, which looks like "at the boundary" and is wrong).
        sign = 1.0 if mask.all() else -1.0
        return np.full(mask.shape, sign * 1e6, dtype=np.float32)
    dist_inside = distance_transform_edt(mask)
    dist_outside = distance_transform_edt(~mask)
    signed = np.where(mask, dist_inside, -dist_outside).astype(np.float32)
    return signed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_volumes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0,
                         help="Which frozen baseline seed's checkpoint to use")
    parser.add_argument("--checkpoint", default=None,
                         help="Override checkpoint path (default: frozen baseline seed_N best.pth)")
    parser.add_argument("--out_dir", default=None,
                         help="Output dir (default: experiments/exp_e_latent_analysis/extracted)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_volume_dir = out_dir / "per_volume"
    per_volume_dir.mkdir(exist_ok=True)

    ckpt_path = args.checkpoint or str(
        project_root / "experiments" / "exp00b_baseline_convergence"
        / f"seed_{args.seed}" / "checkpoints" / "best.pth"
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"  epoch={ckpt['epoch']} seed={ckpt['seed']} best_val_dice={ckpt['best_val_dice']:.4f}")

    model = UNet3D(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    activations, handles = register_hooks(model)

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_volumes = min(args.n_volumes, len(val_dataset))
    print(f"Extracting {n_volumes} of {len(val_dataset)} validation volumes")

    manifest_rows = []
    voxel_csv_path = out_dir / "per_voxel_stats.csv"
    voxel_f = open(voxel_csv_path, "w", newline="")
    voxel_writer = csv.writer(voxel_f)
    voxel_header = (
        ["subject_id", "voxel_idx", "d", "h", "w"]
        + [f"dec1_f{i}" for i in range(32)]
        + ["feature_norm", "alpha", "beta", "evidence", "entropy",
           "confidence", "mu", "prediction", "ground_truth", "correct",
           "boundary_distance"]
    )
    voxel_writer.writerow(voxel_header)

    rng = np.random.RandomState(0)
    VOXELS_PER_VOLUME_TO_LOG = 2000  # subsample for the per-voxel table; full arrays saved separately per volume

    with torch.no_grad():
        for idx in range(n_volumes):
            image, mask, subject_id = val_dataset[idx]
            subject_id = str(subject_id)
            print(f"[{idx+1}/{n_volumes}] {subject_id}")

            image_b = image.unsqueeze(0).to(device)  # (1, 1, D, H, W)
            outputs = model(image_b)

            probs = outputs["probs"].squeeze(0).squeeze(0).cpu().numpy()  # (D,H,W)
            alpha = outputs["alpha"].squeeze(0).squeeze(0).cpu().numpy()
            beta = outputs["beta"].squeeze(0).squeeze(0).cpu().numpy()
            gt = mask.squeeze(0).numpy().astype(np.float32)  # (D,H,W)

            pred_binary = (probs >= 0.5).astype(np.float32)
            mu = alpha / (alpha + beta)
            confidence = np.where(pred_binary == 1, mu, 1 - mu)
            evidence_total = alpha + beta - 2.0  # total evidence, Sensoy et al. convention (alpha,beta >= 1)
            # Beta distribution differential entropy (nats), standard formula:
            # H = ln B(a,b) - (a-1)psi(a) - (b-1)psi(b) + (a+b-2)psi(a+b)
            from scipy.special import betaln, digamma
            entropy = (
                betaln(alpha, beta)
                - (alpha - 1) * digamma(alpha)
                - (beta - 1) * digamma(beta)
                + (alpha + beta - 2) * digamma(alpha + beta)
            ).astype(np.float32)

            boundary_dist = boundary_distance_map(gt)

            dec1_feat = activations["dec1"].squeeze(0).cpu().numpy()  # (32, D, H, W)
            feature_norm = np.linalg.norm(dec1_feat, axis=0)  # (D,H,W)

            # Save full per-volume arrays (Priority 1/8: all stages, not just dec1)
            vol_dir = per_volume_dir / subject_id
            vol_dir.mkdir(exist_ok=True)
            for stage in STAGE_NAMES:
                np.save(vol_dir / f"features_{stage}.npy", activations[stage].squeeze(0).cpu().numpy())
            np.save(vol_dir / "prediction.npy", probs)
            np.save(vol_dir / "ground_truth.npy", gt)
            np.save(vol_dir / "alpha.npy", alpha)
            np.save(vol_dir / "beta.npy", beta)
            np.save(vol_dir / "evidence.npy", evidence_total)
            np.save(vol_dir / "entropy.npy", entropy)
            np.save(vol_dir / "confidence.npy", confidence)
            np.save(vol_dir / "boundary_distance.npy", boundary_dist)

            # Per-volume summary for manifest / failure-case ranking (Priority 7)
            dice = 2 * (pred_binary * gt).sum() / max(pred_binary.sum() + gt.sum(), 1e-6)
            manifest_rows.append({
                "subject_id": subject_id,
                "dice": float(dice),
                "mean_confidence": float(confidence.mean()),
                "mean_entropy": float(entropy.mean()),
                "mean_evidence": float(evidence_total.mean()),
                "tumor_voxel_count": int(gt.sum()),
                "mean_feature_norm": float(feature_norm.mean()),
            })

            # Subsample voxels for the per-voxel long-format table (full
            # arrays exist per-volume above for anything needing every voxel)
            D, H, W = gt.shape
            total_voxels = D * H * W
            n_sample = min(VOXELS_PER_VOLUME_TO_LOG, total_voxels)
            flat_idx = rng.choice(total_voxels, size=n_sample, replace=False)
            d_idx, h_idx, w_idx = np.unravel_index(flat_idx, (D, H, W))

            dec1_flat = dec1_feat.reshape(32, -1)  # (32, D*H*W)
            for i in range(n_sample):
                vi, di, hi, wi = flat_idx[i], d_idx[i], h_idx[i], w_idx[i]
                feat_vec = dec1_flat[:, vi]
                row = (
                    [subject_id, int(vi), int(di), int(hi), int(wi)]
                    + [float(x) for x in feat_vec]
                    + [
                        float(feature_norm[di, hi, wi]),
                        float(alpha[di, hi, wi]), float(beta[di, hi, wi]),
                        float(evidence_total[di, hi, wi]), float(entropy[di, hi, wi]),
                        float(confidence[di, hi, wi]), float(mu[di, hi, wi]),
                        float(pred_binary[di, hi, wi]), float(gt[di, hi, wi]),
                        float(pred_binary[di, hi, wi] == gt[di, hi, wi]),
                        float(boundary_dist[di, hi, wi]),
                    ]
                )
                voxel_writer.writerow(row)
            voxel_f.flush()

    voxel_f.close()
    for h in handles:
        h.remove()

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    manifest_rows.sort(key=lambda r: r["dice"])
    print("\n" + "=" * 70)
    print(f"Extraction complete: {n_volumes} volumes -> {out_dir}")
    print(f"Per-voxel table: {voxel_csv_path} ({n_volumes * VOXELS_PER_VOLUME_TO_LOG} rows)")
    print(f"Manifest: {manifest_path}")
    print("\nWorst 5 Dice (candidates for Priority 7 failure-case study):")
    for r in manifest_rows[:5]:
        print(f"  {r['subject_id']}: dice={r['dice']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
