"""
Phase E25, C6-2.6 (pre-check before trusting C6-2.5): sign/Jacobian
consistency audit, per the user's explicit instruction to verify the
localization experiment's perturbation sign BEFORE accepting the
"pathway-invariant inversion" finding.

Direct code inspection of run_c62_jacobian_localization.py already
confirmed the sign convention textually:
    delta = [-EPS_PROBE * g for g in g_masked]     (line ~246)
    apply_delta_and_forward: p.add_(d)              (line ~144)
  => theta_new = theta + delta = theta - eps*g  -- genuine gradient
     DESCENT, not ascent. The trivial "+eps*g used by mistake" bug the
     user's Branch A hypothesized is RULED OUT by this inspection alone.

But per the user's own, stronger point: reading the sign in the code is
not the same as verifying it EMPIRICALLY against the theoretical
constraint every gradient-descent step must satisfy to first order:

    cos(Delta_z, grad_z_L) <= 0

(since Delta_z ~= -eps * J J^T grad_z_L = -eps * K * grad_z_L, K=J J^T
is positive semidefinite, so grad_z_L . Delta_z = -eps * grad_z_L^T K
grad_z_L <= 0 for eps>0). This script measures this directly, for real
isolated anchors on real C6-2 checkpoints, with:
  - both +eps*g (gradient ASCENT, deliberately run as a built-in sign
    control -- should give the OPPOSITE-signed Delta_z from -eps*g to
    high precision at small eps) and -eps*g (gradient DESCENT, the
    condition C6-2.5 actually used)
  - an epsilon sweep (0.1, 0.01, 0.001, 0.0001) to confirm first-order
    (linear) behavior, not a large-step artifact
  - the KEY diagnostic: cos(Delta_z, grad_z_L_SC) for the DESCENT
    condition, which theory REQUIRES to be <=0 (up to numerical/finite-
    epsilon slack) regardless of voxel class, regardless of ANY
    directional claim SC-TAM's own design makes about w_hat. This is a
    property of gradient descent on ANY scalar loss, not specific to
    SC-TAM's sign convention -- it does NOT test "is Delta_z correctly
    signed relative to w_hat" (that's what C6-2.5 already tested); it
    tests "is Delta_z correctly signed relative to the loss's OWN
    gradient", a strictly more basic sanity check that must hold for
    the measurement pipeline to be trustworthy at all.
"""
import sys
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import sample_stratified_anchors, ANCHORS_PER_VOLUME, EMATauB  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e24"))
from run_counterfactual import ObjectiveConfig  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e25"))
from run_c62_isolation_checks import _sc_tam_per_anchor_losses, classify_voxels, CAT_NAMES  # noqa: E402

CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30]
N_PER_CAT = 2  # kept small -- this is a targeted consistency check, not a full statistical sweep
EPSILONS = [0.1, 0.01, 0.001, 0.0001]

C62_CHECKPOINT_DIR = (project_root / "experiments" / "exp_e12_eggo_m" / "e25" /
                       "c62_runs" / "C62_sc_tam_seed0" / "checkpoints")
OUT_DIR = Path(__file__).parent / "sign_consistency_results"


def apply_delta_and_forward(model_in, delta_list, images_in):
    """Verbatim from every prior phase's own mechanism -- UNCHANGED."""
    model_copy = copy.deepcopy(model_in)
    model_copy.train()
    with torch.no_grad():
        for p, d in zip(model_copy.parameters(), delta_list):
            p.add_(d)
    with torch.no_grad():
        out = model_copy(images_in)
    del model_copy
    return out["dec1"]


def expected_sign(category):
    return -1.0 if category in ("TP", "FN") else 1.0


def run_checkpoint(epoch, device, train_loader):
    ckpt = torch.load(C62_CHECKPOINT_DIR / f"epoch_{epoch}.pth", map_location=device, weights_only=False)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.train()

    objective = ObjectiveConfig(mode="sc_tam")
    w_hat = objective.get_w_hat(model, device)
    tau_b_tracker = EMATauB()
    rng = np.random.RandomState(epoch)

    images, masks, _ = next(iter(train_loader))
    images = images.to(device)
    masks = masks.to(device)

    with torch.enable_grad():
        outputs = model(images)
        dec1 = outputs["dec1"]
        alpha, beta = outputs["alpha"], outputs["beta"]
        with torch.no_grad():
            evidence_full = (alpha + beta - 2.0)
            probs = outputs["probs"]
            pred_binary = (probs >= 0.5).float()
        boundary_logit = outputs["boundary_logit"]

        B, C, D, H, W = dec1.shape
        dec1_perm = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
        evidence_flat = evidence_full.reshape(-1)
        boundary_flat = boundary_logit.reshape(-1)
        gt_flat = masks.reshape(-1)
        pred_flat = pred_binary.reshape(-1)
        cat_flat = classify_voxels(gt_flat, pred_flat)

        voxels_per_vol = D * H * W
        anchor_idx_list = []
        for b in range(B):
            vol_evidence = evidence_flat[b * voxels_per_vol:(b + 1) * voxels_per_vol]
            local_idx = sample_stratified_anchors(vol_evidence, ANCHORS_PER_VOLUME, rng)
            anchor_idx_list.append(local_idx + b * voxels_per_vol)
        anchor_idx = torch.cat(anchor_idx_list)
        anchors_cat = cat_flat[anchor_idx]

        torch.manual_seed(0)
        per_anchor_loss = _sc_tam_per_anchor_losses(
            dec1_perm, evidence_flat, boundary_flat, gt_flat,
            anchor_idx, tau_b_tracker.tau_b, w_hat, objective.delta_d, device,
        )

    all_params = list(model.parameters())
    axis_unit = w_hat / w_hat.norm().clamp_min(1e-8)

    with torch.no_grad():
        dec1_orig = model(images)["dec1"]

    records = []
    for cat_id in (0, 1, 2, 3):
        cat_local_idx = torch.where(anchors_cat == cat_id)[0]
        active_local = cat_local_idx[per_anchor_loss[cat_local_idx] > 0]
        if active_local.numel() == 0:
            continue
        chosen = active_local[torch.randperm(active_local.numel())[:N_PER_CAT]]

        for local_i in chosen.tolist():
            global_voxel_idx = int(anchor_idx[local_i].item())

            # --- grad_z L_SC: activation-space gradient at THIS voxel (Level 1's own quantity) ---
            g_z_full, = torch.autograd.grad(per_anchor_loss[local_i], dec1_perm, retain_graph=True)
            g_z_target = g_z_full[global_voxel_idx].detach().clone()  # (32,)
            g_z_norm = float(g_z_target.norm().item())

            # --- grad_theta L_SC: full-model parameter gradient (SAME quantity C6-2.5 used) ---
            g_theta = torch.autograd.grad(per_anchor_loss[local_i], all_params, retain_graph=True, allow_unused=True)
            g_theta = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_theta, all_params)]
            g_theta_norm = sum(float((g ** 2).sum()) for g in g_theta) ** 0.5
            if g_theta_norm < 1e-12 or g_z_norm < 1e-12:
                continue

            voxel_result = {
                "epoch": epoch, "category": CAT_NAMES[cat_id], "voxel_idx": global_voxel_idx,
                "g_z_norm": g_z_norm, "g_theta_norm": g_theta_norm,
                "cos_gz_what": float(F.cosine_similarity(g_z_target.unsqueeze(0), axis_unit.unsqueeze(0)).item()),
                "eps_sweep": [],
            }

            for eps in EPSILONS:
                # GRADIENT DESCENT: theta_new = theta - eps*g_theta (the condition C6-2.5 used)
                delta_gd = [-eps * g for g in g_theta]
                dec1_gd = apply_delta_and_forward(model, delta_gd, images)
                # GRADIENT ASCENT: theta_new = theta + eps*g_theta (built-in sign control)
                delta_ga = [eps * g for g in g_theta]
                dec1_ga = apply_delta_and_forward(model, delta_ga, images)

                with torch.no_grad():
                    dz_gd = (dec1_gd - dec1_orig).permute(0, 2, 3, 4, 1).reshape(-1, C)[global_voxel_idx]
                    dz_ga = (dec1_ga - dec1_orig).permute(0, 2, 3, 4, 1).reshape(-1, C)[global_voxel_idx]

                    dz_gd_norm = float(dz_gd.norm().item())
                    dz_ga_norm = float(dz_ga.norm().item())

                    # THE KEY THEORETICAL CHECK: cos(Delta_z_GD, grad_z_L) must be <=0
                    # for genuine first-order gradient descent, REGARDLESS of SC-TAM's
                    # own w_hat sign convention -- this tests consistency with the
                    # loss's OWN gradient, not with any external "correct" direction.
                    cos_dzgd_gz = (float(F.cosine_similarity(dz_gd.unsqueeze(0), g_z_target.unsqueeze(0)).item())
                                   if dz_gd_norm > 1e-12 else None)
                    cos_dzga_gz = (float(F.cosine_similarity(dz_ga.unsqueeze(0), g_z_target.unsqueeze(0)).item())
                                   if dz_ga_norm > 1e-12 else None)
                    # GD vs GA sign control: should be ~antiparallel at small eps
                    cos_gd_ga = (float(F.cosine_similarity(dz_gd.unsqueeze(0), dz_ga.unsqueeze(0)).item())
                                 if dz_gd_norm > 1e-12 and dz_ga_norm > 1e-12 else None)
                    # w_hat-relative sign (SAME quantity C6-2.5 reported)
                    cos_dzgd_what = (float(F.cosine_similarity(dz_gd.unsqueeze(0), axis_unit.unsqueeze(0)).item())
                                     if dz_gd_norm > 1e-12 else None)

                voxel_result["eps_sweep"].append({
                    "eps": eps,
                    "dz_gd_norm": dz_gd_norm, "dz_ga_norm": dz_ga_norm,
                    "cos_Delta_z_GD__grad_z_L": cos_dzgd_gz,
                    "cos_Delta_z_GA__grad_z_L": cos_dzga_gz,
                    "cos_Delta_z_GD__Delta_z_GA": cos_gd_ga,
                    "cos_Delta_z_GD__w_hat": cos_dzgd_what,
                })

            records.append(voxel_result)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = create_brats_loaders(
        batch_size=2, num_workers=0,
        root_dir=str(project_root / "Dataset" / "Training"),
        val_split=0.1,
    )

    all_records = []
    for epoch in CHECKPOINT_EPOCHS:
        print(f"\n=== Checkpoint epoch {epoch} ===")
        records = run_checkpoint(epoch, device, train_loader)
        all_records.extend(records)
        for r in records:
            print(f"  {r['category']} voxel={r['voxel_idx']} cos(g_z,w_hat)={r['cos_gz_what']:+.4f}")
            for e in r["eps_sweep"]:
                print(f"    eps={e['eps']:<8} cos(dz_GD,g_z)={e['cos_Delta_z_GD__grad_z_L']!s:>8} "
                      f"cos(dz_GA,g_z)={e['cos_Delta_z_GA__grad_z_L']!s:>8} "
                      f"cos(dz_GD,dz_GA)={e['cos_Delta_z_GD__Delta_z_GA']!s:>8} "
                      f"cos(dz_GD,w_hat)={e['cos_Delta_z_GD__w_hat']!s:>8}")

    json_path = OUT_DIR / "sign_consistency_C62.json"
    with open(json_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nSaved {len(all_records)} voxel records to {json_path}")


if __name__ == "__main__":
    main()
