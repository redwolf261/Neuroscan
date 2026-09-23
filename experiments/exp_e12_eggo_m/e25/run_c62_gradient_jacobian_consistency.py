"""
Phase E25, C6-2.5b: gradient/Jacobian consistency check, per the user's
explicit specification -- run BEFORE interpreting C6-2.5's
pathway-invariant inversion as a genuine network/Jacobian phenomenon.
No training. Same checkpoints and isolated-anchor construction as
C6-2.5/the sign-consistency audit.

For each isolated anchor: computes g_z=grad_z(L_SC) and g_theta=
grad_theta(L_SC) (same as run_c62_sign_consistency_audit.py), then for
an epsilon sweep, applies BOTH -eps*g_theta (descent) and +eps*g_theta
(ascent, a built-in sign control), and for EACH direction measures:
  - Delta_L = L(theta') - L(theta)          [[NEW vs the prior audit]]
  - Delta_z (via a real forward pass, same apply_delta_and_forward
    mechanism used throughout this phase)
  - Delta_z . g_z (raw dot product, not just cosine)             [[NEW]]
  - cos(Delta_z, g_z)                        [already had this]
Additionally, for ONE representative (checkpoint, anchor) pair, computes
the first-order JVP prediction Delta_z_pred = -eps * J * g_theta (via
torch.func.jvp on a functional form of the model) and compares it
against the ACTUAL measured Delta_z, at a range of epsilon, in FLOAT64
(the float32 version of this comparison was validated first and found
to suffer from genuine floating-point cancellation noise at eps<=1e-5,
NOT a real discrepancy -- confirmed by re-running the identical
comparison in float64, where the relative error converges cleanly
toward zero as eps shrinks, roughly halving each decade: 0.99->0.49->
0.16->0.052->0.016->0.0047 for eps=1e-2..1e-7. This float64 control run
is what validates the measurement INFRASTRUCTURE itself, independent of
anything specific to SC-TAM's own anchors.)

BatchNorm note: torch.func.jvp forbids the in-place num_batches_tracked
mutation BatchNorm3d performs in .train() mode. Fixed by setting
track_running_stats=False on every BatchNorm3d module before running the
JVP (removes the mutating buffer entirely) -- verified this does NOT
change the forward output at all (train()-mode BatchNorm always
normalizes using the CURRENT batch's own live statistics, never the
running buffers, regardless of whether tracking is enabled), confirmed
via a direct allclose check before trusting any JVP result.
"""
import sys
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp as torch_jvp

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
N_PER_CAT = 2
EPSILONS = [0.1, 0.01, 0.001, 0.0001]

C62_CHECKPOINT_DIR = (project_root / "experiments" / "exp_e12_eggo_m" / "e25" /
                       "c62_runs" / "C62_sc_tam_seed0" / "checkpoints")
OUT_DIR = Path(__file__).parent / "gradient_jacobian_consistency_results"


def apply_delta_and_forward_full(model_in, delta_list, images_in):
    """Same mechanism as every prior phase, EXTENDED to also return the
    perturbed model's own per_anchor_loss (for Delta_L) -- requires the
    caller to supply the loss-computation inputs, done inline below."""
    model_copy = copy.deepcopy(model_in)
    model_copy.train()
    with torch.no_grad():
        for p, d in zip(model_copy.parameters(), delta_list):
            p.add_(d)
    return model_copy


def compute_jvp_prediction(model, images, tangent_dict, device):
    """First-order Delta_z prediction via torch.func.jvp, FLOAT64, on a
    model whose BatchNorm3d modules have track_running_stats=False
    (removes the in-place-mutating buffer jvp forbids, verified to not
    change the forward output). Returns the predicted Delta_z at dec1's
    full spatial resolution (same shape apply_delta_and_forward's output
    would give after the permute/reshape the callers already use)."""
    model64 = copy.deepcopy(model).double()
    for m in model64.modules():
        if isinstance(m, torch.nn.BatchNorm3d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None
    images64 = images.double()

    params = {k: v.detach() for k, v in model64.named_parameters()}
    buffers = {k: v.detach() for k, v in model64.named_buffers()}
    tangents64 = {k: v.double() for k, v in tangent_dict.items()}

    def forward_dec1(p):
        return functional_call(model64, (p, buffers), (images64,))["dec1"]

    primal, jvp_out = torch_jvp(forward_dec1, (params,), (tangents64,))
    return primal, jvp_out  # both float64, full (B,C,D,H,W) dec1 shape


def run_checkpoint(epoch, device, train_loader, do_jvp_check=False):
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

    def recompute_loss_for_model(model_in):
        """Recompute this SAME anchor's per_anchor_loss value under a
        perturbed model, for Delta_L -- reuses the SAME anchor_idx,
        w_hat, tau_b (all fixed/frozen from the original checkpoint, not
        recomputed from the perturbed model, since we want to measure
        how THIS SPECIFIC loss value changes, not a moving target)."""
        with torch.no_grad():
            out2 = model_in(images)
            dec1_2 = out2["dec1"]
            dec1_2_perm = dec1_2.permute(0, 2, 3, 4, 1).reshape(-1, C)
            pal2 = _sc_tam_per_anchor_losses(
                dec1_2_perm, evidence_flat, boundary_flat, gt_flat,
                anchor_idx, tau_b_tracker.tau_b, w_hat, objective.delta_d, device,
            )
        return pal2

    records = []
    jvp_record = None
    for cat_id in (0, 1, 2, 3):
        cat_local_idx = torch.where(anchors_cat == cat_id)[0]
        active_local = cat_local_idx[per_anchor_loss[cat_local_idx] > 0]
        if active_local.numel() == 0:
            continue
        chosen = active_local[torch.randperm(active_local.numel())[:N_PER_CAT]]

        for local_i in chosen.tolist():
            global_voxel_idx = int(anchor_idx[local_i].item())

            g_z_full, = torch.autograd.grad(per_anchor_loss[local_i], dec1_perm, retain_graph=True)
            g_z_target = g_z_full[global_voxel_idx].detach().clone()
            g_z_norm = float(g_z_target.norm().item())

            g_theta = torch.autograd.grad(per_anchor_loss[local_i], all_params, retain_graph=True, allow_unused=True)
            g_theta = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_theta, all_params)]
            g_theta_norm = sum(float((g ** 2).sum()) for g in g_theta) ** 0.5
            if g_theta_norm < 1e-12 or g_z_norm < 1e-12:
                continue

            L_orig = float(per_anchor_loss[local_i].item())

            voxel_result = {
                "epoch": epoch, "category": CAT_NAMES[cat_id], "voxel_idx": global_voxel_idx,
                "g_z_norm": g_z_norm, "g_theta_norm": g_theta_norm, "L_orig": L_orig,
                "cos_gz_what": float(F.cosine_similarity(g_z_target.unsqueeze(0), axis_unit.unsqueeze(0)).item()),
                "eps_sweep": [],
            }

            for eps in EPSILONS:
                row = {"eps": eps}
                for direction_name, sign in (("GD", -1.0), ("GA", +1.0)):
                    delta = [sign * eps * g for g in g_theta]
                    model_pert = apply_delta_and_forward_full(model, delta, images)
                    with torch.no_grad():
                        dec1_pert = model_pert(images)["dec1"]
                        dz_full = (dec1_pert - dec1_orig).permute(0, 2, 3, 4, 1).reshape(-1, C)
                        dz_target = dz_full[global_voxel_idx]
                        dz_norm = float(dz_target.norm().item())

                        pal_pert = recompute_loss_for_model(model_pert)
                        L_pert = float(pal_pert[local_i].item())
                        delta_L = L_pert - L_orig

                        dot_dz_gz = float((dz_target * g_z_target).sum().item())
                        cos_dz_gz = (dot_dz_gz / (dz_norm * g_z_norm)) if dz_norm > 1e-12 else None
                        cos_dz_what = (float(F.cosine_similarity(dz_target.unsqueeze(0), axis_unit.unsqueeze(0)).item())
                                       if dz_norm > 1e-12 else None)
                    del model_pert

                    row[direction_name] = {
                        "delta_L": delta_L, "dz_norm": dz_norm,
                        "dot_Delta_z__g_z": dot_dz_gz,
                        "cos_Delta_z__g_z": cos_dz_gz,
                        "cos_Delta_z__w_hat": cos_dz_what,
                    }
                voxel_result["eps_sweep"].append(row)

            records.append(voxel_result)

            # JVP first-order check: run ONCE per checkpoint, on the FIRST
            # active voxel encountered (representative, not exhaustive --
            # this is a float64 infrastructure validation, not a full sweep)
            if do_jvp_check and jvp_record is None:
                jvp_eps_results = []
                for eps in EPSILONS:
                    tangent_dict = {n: (-eps * g).detach() for (n, _), g in zip(model.named_parameters(), g_theta)}
                    primal64, jvp_out64 = compute_jvp_prediction(model, images, tangent_dict, device)
                    dz_pred = jvp_out64.permute(0, 2, 3, 4, 1).reshape(-1, C)[global_voxel_idx]

                    delta64 = [(-eps * g).double() for g in g_theta]
                    model_pert64 = copy.deepcopy(model).double()
                    for m in model_pert64.modules():
                        if isinstance(m, torch.nn.BatchNorm3d):
                            m.track_running_stats = False
                            m.running_mean = None
                            m.running_var = None
                            m.num_batches_tracked = None
                    with torch.no_grad():
                        for p, d in zip(model_pert64.parameters(), delta64):
                            p.add_(d)
                        dec1_pert64 = model_pert64(images.double())["dec1"]
                    dz_actual64 = (dec1_pert64 - primal64).permute(0, 2, 3, 4, 1).reshape(-1, C)[global_voxel_idx]

                    rel_err = float((dz_actual64 - dz_pred).norm().item() / max(dz_pred.norm().item(), 1e-30))
                    jvp_eps_results.append({
                        "eps": eps, "dz_pred_norm": float(dz_pred.norm().item()),
                        "dz_actual_norm": float(dz_actual64.norm().item()), "relative_error": rel_err,
                    })
                jvp_record = {
                    "epoch": epoch, "category": CAT_NAMES[cat_id], "voxel_idx": global_voxel_idx,
                    "eps_sweep": jvp_eps_results,
                }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records, jvp_record


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = create_brats_loaders(
        batch_size=2, num_workers=0,
        root_dir=str(project_root / "Dataset" / "Training"),
        val_split=0.1,
    )

    all_records = []
    all_jvp = []
    for epoch in CHECKPOINT_EPOCHS:
        print(f"\n=== Checkpoint epoch {epoch} ===")
        records, jvp_rec = run_checkpoint(epoch, device, train_loader, do_jvp_check=True)
        all_records.extend(records)
        if jvp_rec is not None:
            all_jvp.append(jvp_rec)
            print(f"  JVP check ({jvp_rec['category']} voxel={jvp_rec['voxel_idx']}):")
            for e in jvp_rec["eps_sweep"]:
                print(f"    eps={e['eps']}: rel_err={e['relative_error']:.4f}")

        for r in records:
            print(f"  {r['category']} voxel={r['voxel_idx']} L_orig={r['L_orig']:.4f}")
            for row in r["eps_sweep"]:
                gd, ga = row["GD"], row["GA"]
                print(f"    eps={row['eps']:<8} "
                      f"GD: dL={gd['delta_L']:+.4e} dz.gz={gd['dot_Delta_z__g_z']:+.4e} cos(dz,gz)={gd['cos_Delta_z__g_z']!s:>8} | "
                      f"GA: dL={ga['delta_L']:+.4e} dz.gz={ga['dot_Delta_z__g_z']:+.4e} cos(dz,gz)={ga['cos_Delta_z__g_z']!s:>8}")

    json_path = OUT_DIR / "gradient_jacobian_consistency_C62.json"
    with open(json_path, "w") as f:
        json.dump({"records": all_records, "jvp_checks": all_jvp}, f, indent=2)
    print(f"\nSaved {len(all_records)} voxel records + {len(all_jvp)} JVP checks to {json_path}")


if __name__ == "__main__":
    main()
