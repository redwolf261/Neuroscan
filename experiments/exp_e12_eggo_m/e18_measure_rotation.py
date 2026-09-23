"""
Phase E18: Representation Rotation Source -- measurement script.

Runs E17's exact 4-part rotation diagnostic (centroid direction
stability, margin gradient stability, gradient transport, Procrustes
whole-representation rotation) PLUS a new Part 5 (principal-angle /
subspace-overlap analysis via PCA/SVD on dec1) across all 6 E18
training configs (baseline + 5 single-component ablations), so results
are directly comparable across configs using identical methodology.

Also collects the performance (Dice/HD95/ECE) and margin (mean margin,
active hinge %, margin loss) metrics requested, and produces the
rotation-magnitude summary table (epoch1->5, epoch5->10, epoch10->30).

No retraining, no changes to any model/loss/optimizer here -- this
script only loads existing checkpoints and measures them, exactly like
E14/E15/E16/E17.
"""
import sys
import json
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.linalg import orthogonal_procrustes, subspace_angles

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import (  # noqa: E402
    sample_stratified_anchors, compute_margin_loss,
    DELTA_D_CALIBRATED, ANCHORS_PER_VOLUME, MAX_NEGATIVES_PER_ANCHOR,
    EVIDENCE_P99_DEFAULT, EMATauB,
)

CHECKPOINT_EPOCHS = [1, 5, 10, 15, 20, 25, 30]
N_VAL_SUBJECTS = 20
N_TRACKED_VOXELS_PER_SUBJECT = 500
N_STRATIFIED_ANCHORS_PER_SUBJECT = ANCHORS_PER_VOLUME  # 2000, matches E17/real training exactly

CONFIGS = {
    "baseline": "e18_none_seed0",
    # freeze_bn never produced a clean, non-collapsed 30-epoch run across
    # 3 attempts (see PHASE_E18 doc) -- pointed at the 3rd (longest,
    # 10-epoch-warmup) attempt, stopped at epoch 25 after the collapse at
    # epoch 16 was confirmed non-recovering. Checkpoints 1/5/10/15 are
    # pre-collapse (healthy), 20/25 are post-collapse (degenerate) --
    # measuring rotation across this boundary is itself informative, not
    # a reason to exclude the config, though epoch 30 is unavailable and
    # results should be read with the collapse explicitly in mind.
    "freeze_bn": "e18_freeze_bn_ATTEMPT3_STOPPED_epoch25_seed0",
    "freeze_encoder": "e18_freeze_encoder_seed0",
    "freeze_decoder": "e18_freeze_decoder_seed0",
    "freeze_seg_head": "e18_freeze_seg_head_seed0",
    "lambda_zero": "e18_lambda_zero_seed0",
}


def load_model(ckpt_dir, epoch, device):
    ckpt = torch.load(ckpt_dir / f"epoch_{epoch}.pth", map_location=device, weights_only=False)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model


def measure_one_config(config_name, exp_dir, val_dataset, n_use, device):
    print(f"\n{'#'*70}\n# CONFIG: {config_name}\n{'#'*70}")
    ckpt_dir = exp_dir / CONFIGS[config_name] / "checkpoints"
    available_epochs = [e for e in CHECKPOINT_EPOCHS if (ckpt_dir / f"epoch_{e}.pth").exists()]
    if len(available_epochs) < 2:
        print(f"  SKIPPING {config_name}: fewer than 2 checkpoints available ({available_epochs})")
        return None
    if available_epochs != CHECKPOINT_EPOCHS:
        print(f"  WARNING: only {available_epochs} available (expected {CHECKPOINT_EPOCHS}), proceeding with what exists")

    results = {"config": config_name, "epochs": available_epochs}

    # =====================================================================
    # Fixed uniform tracking set (Parts 1, 4, 5) -- same seed/scheme as E17
    # =====================================================================
    rng_sample = np.random.RandomState(42)
    model_ref = load_model(ckpt_dir, available_epochs[0], device)
    model_ref.eval()
    tracked_subject_idx, tracked_flat_idx, tracked_gt_class = [], [], []
    with torch.no_grad():
        for subj_idx in range(n_use):
            image, mask, subject_id = val_dataset[subj_idx]
            image_b = image.unsqueeze(0).to(device)
            mask_b = mask.unsqueeze(0).to(device)
            outputs = model_ref(image_b)
            dec1 = outputs["dec1"]
            B, C, D, H, W = dec1.shape
            gt_flat = mask_b.reshape(-1) > 0.5
            total_voxels = D * H * W
            n_sample = min(N_TRACKED_VOXELS_PER_SUBJECT, total_voxels)
            idx = rng_sample.choice(total_voxels, size=n_sample, replace=False)
            tracked_subject_idx.extend([subj_idx] * n_sample)
            tracked_flat_idx.extend(idx.tolist())
            tracked_gt_class.extend(gt_flat[idx].cpu().numpy().tolist())
    tracked_gt_class = np.array(tracked_gt_class)
    tracked_subject_idx = np.array(tracked_subject_idx)
    tracked_flat_idx = np.array(tracked_flat_idx)
    del model_ref

    z_by_epoch = {}
    for epoch in available_epochs:
        model = load_model(ckpt_dir, epoch, device)
        model.eval()
        z_this_epoch = []
        with torch.no_grad():
            for subj_idx in range(n_use):
                image, mask, subject_id = val_dataset[subj_idx]
                image_b = image.unsqueeze(0).to(device)
                outputs = model(image_b)
                dec1 = outputs["dec1"]
                B, C, D, H, W = dec1.shape
                dec1_flat = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
                this_subject_mask = tracked_subject_idx == subj_idx
                this_subject_flat_idx = tracked_flat_idx[this_subject_mask]
                z_this_epoch.append(dec1_flat[this_subject_flat_idx].cpu().numpy())
        z_by_epoch[epoch] = np.concatenate(z_this_epoch, axis=0)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- Part 1: Centroid direction stability ---
    centroid_dirs = {}
    for epoch in available_epochs:
        z = z_by_epoch[epoch]
        tumor_c = z[tracked_gt_class].mean(axis=0)
        bg_c = z[~tracked_gt_class].mean(axis=0)
        direction = tumor_c - bg_c
        dist = float(np.linalg.norm(direction))
        unit = direction / max(dist, 1e-8)
        centroid_dirs[epoch] = {"unit": unit, "dist": dist}

    part1_consec = []
    for i in range(len(available_epochs) - 1):
        e_t, e_t1 = available_epochs[i], available_epochs[i + 1]
        cos_consec = float(np.dot(centroid_dirs[e_t]["unit"], centroid_dirs[e_t1]["unit"]))
        part1_consec.append({"epoch_t": e_t, "epoch_t1": e_t1, "cos_direction": cos_consec})
        print(f"  [Part1] {e_t:2d}->{e_t1:2d}: cos={cos_consec:+.4f}")

    e0 = available_epochs[0]
    part1_vs_first = [{"epoch": e, "cos_vs_first": float(np.dot(centroid_dirs[e0]["unit"], centroid_dirs[e]["unit"]))}
                       for e in available_epochs]

    results["part1_centroid_stability"] = {"consecutive": part1_consec, "vs_first": part1_vs_first}

    # --- Part 4: Procrustes whole-representation rotation ---
    part4_rows = []
    for i in range(len(available_epochs) - 1):
        e_t, e_t1 = available_epochs[i], available_epochs[i + 1]
        Z_t = z_by_epoch[e_t] - z_by_epoch[e_t].mean(axis=0, keepdims=True)
        Z_t1 = z_by_epoch[e_t1] - z_by_epoch[e_t1].mean(axis=0, keepdims=True)
        R, scale = orthogonal_procrustes(Z_t, Z_t1)
        rotation_closeness = float(np.trace(R) / R.shape[0])
        Z_t_rotated = Z_t @ R
        residual_frac = float(np.linalg.norm(Z_t_rotated - Z_t1, "fro") / max(np.linalg.norm(Z_t1, "fro"), 1e-8))
        part4_rows.append({"epoch_t": e_t, "epoch_t1": e_t1, "rotation_closeness_to_identity": rotation_closeness,
                            "residual_fraction_after_best_rotation": residual_frac})
        print(f"  [Part4] {e_t:2d}->{e_t1:2d}: rotation_closeness={rotation_closeness:+.4f} residual={residual_frac:.4f}")
    results["part4_procrustes"] = part4_rows

    # --- Part 5 (NEW): Principal-angle / subspace-overlap analysis ---
    # PCA on dec1 features at each checkpoint (top-5 components), then
    # compute PRINCIPAL ANGLES between epoch e0's top-5 subspace and
    # every other checkpoint's top-5 subspace -- distinguishes "the
    # entire latent basis is rotating" (large principal angles across
    # many/all components) from "only the specific margin-relevant axis
    # rotates" (small principal angles for most components, large only
    # for the direction Part 1 already tracks).
    print("  [Part5] Principal angles / subspace overlap (top-5 PCA components)...")
    part5_rows = []
    subspaces = {}
    for epoch in available_epochs:
        z = z_by_epoch[epoch]
        z_centered = z - z.mean(axis=0, keepdims=True)
        # SVD for PCA basis -- right singular vectors (Vt rows) are the
        # principal component directions in the 32-dim dec1 space.
        U, S, Vt = np.linalg.svd(z_centered, full_matrices=False)
        top5 = Vt[:5].T  # (32, 5), each column is a principal direction, orthonormal by construction
        subspaces[epoch] = top5
        explained_var_ratio = (S[:5] ** 2 / (S ** 2).sum()).tolist()
        part5_rows.append({"epoch": epoch, "explained_variance_ratio_top5": explained_var_ratio})

    part5_principal_angles = []
    for epoch in available_epochs:
        angles_rad = subspace_angles(subspaces[e0], subspaces[epoch])  # array of 5 principal angles, radians, sorted ascending
        angles_deg = np.degrees(angles_rad)
        # Subspace overlap: mean cos of the principal angles -- 1.0 = identical subspace, 0 = orthogonal subspaces
        mean_cos_overlap = float(np.mean(np.cos(angles_rad)))
        part5_principal_angles.append({
            "epoch": epoch, "principal_angles_deg": angles_deg.tolist(),
            "max_principal_angle_deg": float(angles_deg.max()),
            "mean_subspace_overlap_cos": mean_cos_overlap,
        })
        print(f"    epoch {epoch:2d} vs epoch {e0}: max principal angle={angles_deg.max():.1f} deg, "
              f"mean subspace overlap (cos)={mean_cos_overlap:+.4f}")
    results["part5_principal_angles"] = {"per_epoch_variance": part5_rows, "vs_first_epoch": part5_principal_angles}

    # =====================================================================
    # Parts 2/3: margin gradient stability -- SKIPPED for ablations where
    # lambda=0 (lambda_zero) since there is no margin gradient to measure
    # (would be trivially undefined/zero throughout) -- correctly
    # reported as N/A, not silently omitted.
    # =====================================================================
    lambda_active = (config_name != "lambda_zero")
    if lambda_active:
        print("  [Part2/3] Selecting fixed stratified anchor set...")
        model_strat = load_model(ckpt_dir, e0, device)
        model_strat.eval()
        strat_subject_idx, strat_flat_idx = [], []
        with torch.no_grad():
            for subj_idx in range(n_use):
                image, mask, subject_id = val_dataset[subj_idx]
                image_b = image.unsqueeze(0).to(device)
                outputs = model_strat(image_b)
                alpha, beta = outputs["alpha"], outputs["beta"]
                evidence_flat = (alpha + beta - 2.0).reshape(-1)
                rng_strat = np.random.RandomState(1000 + subj_idx)
                local_idx = sample_stratified_anchors(evidence_flat, N_STRATIFIED_ANCHORS_PER_SUBJECT, rng_strat)
                strat_subject_idx.extend([subj_idx] * len(local_idx))
                strat_flat_idx.extend(local_idx.cpu().numpy().tolist())
        del model_strat
        strat_subject_idx = np.array(strat_subject_idx)
        strat_flat_idx = np.array(strat_flat_idx)

        grad_by_epoch = {}
        for epoch in available_epochs:
            model = load_model(ckpt_dir, epoch, device)
            model.train()
            tau_b_tracker = EMATauB()
            rng = np.random.RandomState(epoch)
            grads_this_epoch = np.full((len(strat_flat_idx), 32), np.nan, dtype=np.float32)
            offset = 0
            for subj_idx in range(n_use):
                image, mask, subject_id = val_dataset[subj_idx]
                image_b = image.unsqueeze(0).to(device)
                mask_b = mask.unsqueeze(0).to(device)
                outputs = model(image_b)
                dec1 = outputs["dec1"]
                alpha, beta = outputs["alpha"], outputs["beta"]
                boundary_logit = outputs["boundary_logit"]
                B, C, D, H, W = dec1.shape
                with torch.no_grad():
                    evidence_full = alpha + beta - 2.0
                dec1_perm = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
                evidence_flat = evidence_full.reshape(-1)
                boundary_flat = boundary_logit.reshape(-1)
                gt_flat = mask_b.reshape(-1)
                this_subject_mask = strat_subject_idx == subj_idx
                this_subject_flat_idx = strat_flat_idx[this_subject_mask]
                n_this_subject = len(this_subject_flat_idx)
                anchor_idx = torch.from_numpy(this_subject_flat_idx).long().to(device)
                current_tau_b = tau_b_tracker.tau_b
                margin_loss, _, margin_diag = compute_margin_loss(
                    dec1_perm, evidence_flat, boundary_flat, gt_flat,
                    anchor_idx, current_tau_b, EVIDENCE_P99_DEFAULT, DELTA_D_CALIBRATED,
                    MAX_NEGATIVES_PER_ANCHOR, rng, device,
                )
                tau_b_tracker.update(margin_diag["abs_boundary_logit"])
                if torch.isfinite(margin_loss) and margin_loss.item() != 0.0:
                    g_margin, = torch.autograd.grad(margin_loss, dec1, retain_graph=False)
                    g_margin_flat = g_margin.permute(0, 2, 3, 4, 1).reshape(-1, C)[anchor_idx]
                    grads_this_epoch[offset:offset + n_this_subject] = g_margin_flat.detach().cpu().numpy()
                offset += n_this_subject
            grad_by_epoch[epoch] = grads_this_epoch
            n_nonzero = int((np.linalg.norm(np.nan_to_num(grads_this_epoch), axis=1) > 1e-12).sum())
            print(f"  [Part2/3] epoch {epoch:2d}: {n_nonzero} nonzero gradients / {len(strat_flat_idx)} anchors")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        part2_rows = []
        for i in range(len(available_epochs) - 1):
            e_t, e_t1 = available_epochs[i], available_epochs[i + 1]
            g_t, g_t1 = grad_by_epoch[e_t], grad_by_epoch[e_t1]
            valid = np.isfinite(g_t).all(axis=1) & np.isfinite(g_t1).all(axis=1)
            norm_t, norm_t1 = np.linalg.norm(g_t[valid], axis=1), np.linalg.norm(g_t1[valid], axis=1)
            nz = (norm_t > 1e-12) & (norm_t1 > 1e-12)
            if nz.sum() == 0:
                part2_rows.append({"epoch_t": e_t, "epoch_t1": e_t1, "n": 0, "mean_cos": None})
                continue
            gt_v, gt1_v = g_t[valid][nz], g_t1[valid][nz]
            cos = np.sum(gt_v * gt1_v, axis=1) / (np.linalg.norm(gt_v, axis=1) * np.linalg.norm(gt1_v, axis=1))
            part2_rows.append({"epoch_t": e_t, "epoch_t1": e_t1, "n": int(len(cos)),
                                "mean_cos": float(cos.mean()), "std_cos": float(cos.std())})
            print(f"  [Part2] {e_t:2d}->{e_t1:2d}: n={len(cos)} mean_cos={cos.mean():+.4f}")
        results["part2_gradient_stability"] = part2_rows

        g_first = grad_by_epoch[e0]
        part3_vs_first = []
        for epoch in available_epochs:
            g_e = grad_by_epoch[epoch]
            valid = np.isfinite(g_first).all(axis=1) & np.isfinite(g_e).all(axis=1)
            n1, ne = np.linalg.norm(g_first[valid], axis=1), np.linalg.norm(g_e[valid], axis=1)
            nz = (n1 > 1e-12) & (ne > 1e-12)
            if nz.sum() == 0:
                continue
            cos_vs_first = np.sum(g_first[valid][nz] * g_e[valid][nz], axis=1) / (n1[nz] * ne[nz])
            part3_vs_first.append({"epoch": epoch, "n": int(nz.sum()), "mean_cos_vs_first": float(cos_vs_first.mean())})
        results["part3_gradient_transport"] = {"vs_first_epoch": part3_vs_first}
        print(f"  [Part3] vs epoch {e0}: " + ", ".join(f"e{r['epoch']}={r['mean_cos_vs_first']:+.3f}" for r in part3_vs_first))
    else:
        print("  [Part2/3] SKIPPED (lambda=0, no margin gradient exists for this config)")
        results["part2_gradient_stability"] = None
        results["part3_gradient_transport"] = None

    # =====================================================================
    # Performance and margin metrics, straight from epoch_metrics.csv
    # =====================================================================
    csv_path = exp_dir / CONFIGS[config_name] / "epoch_metrics.csv"
    perf_rows = []
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                epoch_1indexed = int(row["epoch"]) + 1
                if epoch_1indexed in available_epochs:
                    perf_rows.append({
                        "epoch": epoch_1indexed,
                        "val_dice": float(row["val_dice"]), "val_hd95": float(row["val_hd95"]),
                        "val_ece": float(row["val_ece"]), "train_margin_loss": float(row["train_margin_loss"]),
                        "active_hinge_pct": float(row["active_hinge_pct"]),
                    })
    results["performance_margin_metrics"] = perf_rows

    # =====================================================================
    # Rotation magnitude summary (epoch1->5, 5->10, 10->30), per spec
    # =====================================================================
    def rotation_magnitude(e_a, e_b):
        """1 - mean_cos as a simple, bounded 'rotation magnitude' proxy
        (0 = no rotation, up to 2 = fully reversed) -- uses Part 1's
        centroid direction as the primary geometric summary, since it's
        available for every config (Part2/3 aren't, for lambda_zero)."""
        if e_a not in centroid_dirs or e_b not in centroid_dirs:
            return None
        cos_val = float(np.dot(centroid_dirs[e_a]["unit"], centroid_dirs[e_b]["unit"]))
        return 1.0 - cos_val

    rotation_summary = {
        "epoch1_to_5": rotation_magnitude(1, 5) if 1 in available_epochs and 5 in available_epochs else None,
        "epoch5_to_10": rotation_magnitude(5, 10) if 5 in available_epochs and 10 in available_epochs else None,
        "epoch10_to_30": rotation_magnitude(10, 30) if 10 in available_epochs and 30 in available_epochs else None,
    }
    results["rotation_magnitude_summary"] = rotation_summary
    print(f"  [Rotation summary] 1->5: {rotation_summary['epoch1_to_5']}, "
          f"5->10: {rotation_summary['epoch5_to_10']}, 10->30: {rotation_summary['epoch10_to_30']}")

    return results


def main():
    exp_dir = Path(__file__).parent
    out_dir = exp_dir / "e18_measurement_results"
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_use = min(N_VAL_SUBJECTS, len(val_dataset))

    all_results = {}
    for config_name in CONFIGS:
        existing_path = out_dir / f"{config_name}_results.json"
        if existing_path.exists():
            # Resume support: this script is expensive (~6 configs x 7
            # checkpoints x 20 volumes x 2 passes), and earlier runs were
            # interrupted by unrelated tool-call issues -- rather than
            # redo already-completed configs, load and reuse them.
            print(f"SKIPPING {config_name}: already measured (found {existing_path}), loading existing result")
            with open(existing_path) as f:
                all_results[config_name] = json.load(f)
            continue
        ckpt_dir = exp_dir / CONFIGS[config_name] / "checkpoints"
        if not ckpt_dir.exists():
            print(f"SKIPPING {config_name}: checkpoint dir not found ({ckpt_dir}) -- training likely still in progress")
            continue
        result = measure_one_config(config_name, exp_dir, val_dataset, n_use, device)
        if result is not None:
            all_results[config_name] = result
            with open(out_dir / f"{config_name}_results.json", "w") as f:
                json.dump(result, f, indent=2)

    with open(out_dir / "all_configs_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}\nMeasured {len(all_results)}/{len(CONFIGS)} configs: {list(all_results.keys())}\n{'='*70}")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
