"""
Phase E25, C6-2 mechanism audit: why does SC-TAM produce a correctly
signed REALIZED representation movement (H2) while the IMMEDIATE
loss-gradient analysis (H3's Q3/Q4) predicts substantially less
favorable class/error behavior, and where does the eventual unfavorable
segmentation effect (net-unfavorable FP/FN transitions, H4's Dice
shortfall) actually enter the causal chain?

Per the user's explicit instruction: NOT a new algorithm, NOT C6-3/C6-4.
A diagnostic decomposition of the existing, already-trained C6-2
checkpoints into four levels of the chain

    L_SC -> grad_theta(L_SC) -> Delta_theta -> Delta_z -> segmentation outcome

run on the SAME 6 checkpoints (epochs 5/10/15/20/25/30) used throughout
H1/H2/H3, reusing established, already-verified machinery at every level
rather than reinventing measurement infrastructure:

  Level 1 (loss gradient, activation space): cos(grad_z L_SC, w_hat) and
    cos(grad_z L_SC, -w_hat), split by class, restricted to ACTIVE PAIRS
    (voxels where the hinge is actually firing) -- this is the quantity
    Gate 5's orthogonality/sign check already verified is analytically
    predictable on synthetic data and on early-training smoke batches;
    here it is measured on real, later-training checkpoints.

  Level 2 (total gradient, parameter space): cos(grad_theta L_SC,
    grad_theta L_seg). ALREADY COMPUTED AND SAVED by H1 (the
    'cos_with_g_margin' field for the A_seg direction's rows -- verified
    epsilon-invariant, i.e. a true per-(checkpoint,subject) constant, not
    recomputed here, per the "reuse existing data" discipline established
    throughout this project). Extracted directly from h1_results_C62.json.

  Level 3 (optimizer update): cos(Delta_theta_margin, grad_theta L_SC)
    and cos(Delta_theta_margin, -grad_theta L_SC), where Delta_theta_margin
    is H2's own real 15-replay-step ShadowAdam-accumulated update and
    grad_theta L_SC is the single-step margin gradient at the SAME
    checkpoint/batch (freshly computed here, since H2 did not save the
    per-step raw gradients, only the accumulated delta). Tells us whether
    AdamW's moment-based update substantially rotates the instantaneous
    margin-descent direction.

  Level 4 (representation update): cos(Delta_z, w_hat), using H2's OWN
    real Delta_z (not recomputed), split by SIX categories: tumor,
    background (already computed in H2), plus the four confusion
    categories TP/FP/FN/TN (NEW -- requires the model's own prediction at
    the pre-perturbation checkpoint, not just ground truth, which H2 did
    not compute since its own scope was class-conditional only).

All four levels are computed on the SAME (checkpoint, batch) instances
where possible, to keep the chain traceable per-instance rather than only
in aggregate.
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
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import (  # noqa: E402
    sample_stratified_anchors, compute_margin_loss,
    ANCHORS_PER_VOLUME, MAX_NEGATIVES_PER_ANCHOR, EVIDENCE_P99_DEFAULT, EMATauB,
)

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e24"))
from run_counterfactual import ObjectiveConfig  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from e20_dec1_update_decomposition import ShadowAdam  # noqa: E402 -- SAME class H2/E21 use, unchanged

CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30]
N_REPLAY_STEPS = 15  # matches H2 exactly
N_TRANSPORT_BATCHES = 4  # matches H2 exactly
FOCAL_WEIGHT = 0.5
EVIDENTIAL_WEIGHT = 0.5

C62_CHECKPOINT_DIR = (project_root / "experiments" / "exp_e12_eggo_m" / "e25" /
                       "c62_runs" / "C62_sc_tam_seed0" / "checkpoints")
H1_RESULTS_PATH = Path(__file__).parent / "h1_results" / "h1_results_C62.json"
OUT_DIR = Path(__file__).parent / "mechanism_audit_results"


def extract_level2_from_h1(h1_path):
    """Level 2: cos(grad_theta L_seg, grad_theta L_SC), ALREADY present in
    H1's saved output as the A_seg direction's 'cos_with_g_margin' field
    (verified epsilon-invariant -- a true per-(checkpoint,subject)
    constant -- before trusting this shortcut). Returns a dict keyed by
    (epoch, subject_idx) -> cosine value, plus per-epoch summary stats."""
    rows = json.load(open(h1_path))
    a_rows = [r for r in rows if r["direction"] == "A_seg"]

    by_key = {}
    for r in a_rows:
        key = (r["epoch"], r["subject_idx"])
        by_key.setdefault(key, set()).add(round(r["cos_with_g_margin"], 10))
    non_constant = {k: v for k, v in by_key.items() if len(v) > 1}
    assert len(non_constant) == 0, (
        f"FATAL: cos_with_g_margin is NOT epsilon-invariant for {len(non_constant)} (epoch,subject) keys -- "
        f"the 'reuse H1 data' shortcut for Level 2 is invalid, would need real recomputation instead"
    )

    per_key = {}
    for r in a_rows:
        key = (r["epoch"], r["subject_idx"])
        per_key[key] = r["cos_with_g_margin"]

    by_epoch = {}
    for (epoch, subj), v in per_key.items():
        by_epoch.setdefault(epoch, []).append(v)
    epoch_summary = {ep: {"mean": float(np.mean(vs)), "std": float(np.std(vs)), "n": len(vs)} for ep, vs in by_epoch.items()}

    return per_key, epoch_summary


def compute_level1_and_level3_for_checkpoint(epoch, ckpt_dir, loader_iter, device, n_steps, focal_fn, evidential_fn, rng, tau_b_tracker, objective):
    """Extends H2's own compute_delta_theta_margin_for_checkpoint (same
    ShadowAdam replay mechanism, byte-identical for steps 0..n_steps-2)
    with TWO additions:

      Level 1: on the LAST replay step's batch specifically, before
      calling shadow_margin.step(), decompose g_margin (already computed
      there) into per-voxel activation-space gradient (grad_z L_SC) at
      the ANCHOR voxels, split by class and restricted to ACTIVE pairs
      (diagnostics['active_hinge_pct'] > 0 confirms hinge activity
      exists in this batch; per-anchor activity determined directly from
      which anchors have nonzero gradient norm, same technique Gate 5's
      orthogonality check already used).

      Level 3: returns the LAST replay step's raw grad_theta L_SC (a
      single parameter-space gradient snapshot) ALONGSIDE the full
      n_steps-accumulated Delta_theta_margin (H2's own quantity) so the
      caller can compute cos(Delta_theta, grad_theta_L_SC_last_step) --
      the most recent single-step gradient direction is the most
      relevant one to compare the accumulated trajectory against, since
      it is the direction AdamW's moment state was most recently pushed
      to align with.

    Returns: model, dec1_params, delta_theta_margin (H2's own quantity,
    unchanged), level1_records (list of per-batch class-split gradient
    cosines), g_margin_last_step (the final step's raw gradient list)."""
    ckpt_path = ckpt_dir / f"epoch_{epoch}.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.train()

    dec1_params = list(model.dec1.parameters())
    pg = ckpt["optimizer_state"]["param_groups"][0]
    lr, (beta1, beta2), eps = pg["lr"], pg["betas"], pg["eps"]

    shadow_margin = ShadowAdam(dec1_params, lr, beta1, beta2, eps)
    level1_records = []
    g_margin_last_step = None

    for step_idx in range(n_steps):
        images, masks, _ = next(loader_iter)
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        alpha, beta_t = outputs["alpha"], outputs["beta"]
        boundary_logit = outputs["boundary_logit"]
        dec1 = outputs["dec1"]

        B, C, D, H, W = dec1.shape
        with torch.no_grad():
            evidence_full = alpha + beta_t - 2.0
        dec1_perm = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
        evidence_flat = evidence_full.reshape(-1)
        boundary_flat = boundary_logit.reshape(-1)
        gt_flat = masks.reshape(-1)

        voxels_per_vol = D * H * W
        anchor_idx_list = []
        for b in range(B):
            vol_evidence = evidence_flat[b * voxels_per_vol:(b + 1) * voxels_per_vol]
            local_idx = sample_stratified_anchors(vol_evidence, ANCHORS_PER_VOLUME, rng)
            anchor_idx_list.append(local_idx + b * voxels_per_vol)
        anchor_idx = torch.cat(anchor_idx_list)

        current_tau_b = tau_b_tracker.tau_b
        w_hat = objective.get_w_hat(model, device)
        margin_loss, _, margin_diag = compute_margin_loss(
            dec1_perm, evidence_flat, boundary_flat, gt_flat,
            anchor_idx, current_tau_b, EVIDENCE_P99_DEFAULT, objective.delta_d,
            MAX_NEGATIVES_PER_ANCHOR, rng, device,
            w_hat=w_hat, margin_mode=objective.margin_mode,
        )
        tau_b_tracker.update(margin_diag["abs_boundary_logit"])

        if not torch.isfinite(margin_loss):
            continue

        # --- Level 1: activation-space gradient at anchors, THIS step's
        # batch, split by class -- computed for EVERY step (not just the
        # last), since Level 1 is a per-batch measurement, matching Gate
        # 5's own orthogonality-check granularity. ---
        if margin_loss.item() > 0:
            g_z, = torch.autograd.grad(margin_loss, dec1_perm, retain_graph=True)
            g_z_anchors = g_z[anchor_idx]
            g_norms = g_z_anchors.norm(dim=1)
            active_mask = g_norms > 1e-8
            if active_mask.sum() > 0 and w_hat is not None:
                g_active = g_z_anchors[active_mask]
                anchor_gt = gt_flat[anchor_idx][active_mask]
                tumor_active = anchor_gt > 0.5
                bg_active = ~tumor_active

                g_norm_active = g_active.norm(dim=1).clamp_min(1e-12)
                cos_with_w = (g_active @ w_hat) / g_norm_active  # true cosine, since w_hat is already unit-norm

                def summarize_cos(mask):
                    subset = cos_with_w[mask]
                    if subset.numel() == 0:
                        return None
                    return {"n": int(subset.numel()), "cos_mean": float(subset.mean().item()), "cos_std": float(subset.std().item()) if subset.numel() > 1 else 0.0}

                level1_records.append({
                    "epoch": epoch, "step_idx": step_idx,
                    "n_active_anchors": int(active_mask.sum().item()),
                    "tumor": summarize_cos(tumor_active),
                    "background": summarize_cos(bg_active),
                })

        g_margin = torch.autograd.grad(margin_loss, dec1_params, retain_graph=False, allow_unused=True)
        g_margin = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_margin, dec1_params)]
        shadow_margin.step(g_margin)
        g_margin_last_step = [g.detach().clone() for g in g_margin]

    return model, dec1_params, shadow_margin.delta_sum, level1_records, g_margin_last_step


def apply_delta_and_forward(model, delta_list, images):
    """Verbatim from H2/E21 -- UNCHANGED."""
    model_copy = copy.deepcopy(model)
    model_copy.train()
    with torch.no_grad():
        for p, d in zip(model_copy.dec1.parameters(), delta_list):
            p.add_(d)
    with torch.no_grad():
        outputs = model_copy(images)
    del model_copy
    return outputs["dec1"]


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("LEVEL 2 (reused from H1's saved data, not recomputed)")
    print("=" * 70)
    level2_per_key, level2_epoch_summary = extract_level2_from_h1(H1_RESULTS_PATH)
    for ep in sorted(level2_epoch_summary):
        s = level2_epoch_summary[ep]
        print(f"  epoch {ep}: cos(grad_theta L_seg, grad_theta L_SC) mean={s['mean']:+.4f} std={s['std']:.4f} n={s['n']}")

    train_loader, _ = create_brats_loaders(
        batch_size=2, num_workers=0,
        root_dir=str(project_root / "Dataset" / "Training"),
        val_split=0.1,
    )
    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)
    objective = ObjectiveConfig(mode="sc_tam")

    all_level1 = []
    all_level3 = []
    all_level4 = []

    print("\n" + "=" * 70)
    print("LEVELS 1, 3, 4 (computed fresh on real C6-2 checkpoints)")
    print("=" * 70)

    for epoch in CHECKPOINT_EPOCHS:
        print(f"\n=== Checkpoint epoch {epoch} ===")
        rng = np.random.RandomState(epoch)
        tau_b_tracker = EMATauB()
        loader_iter = iter(train_loader)

        model, dec1_params, delta_theta_margin, level1_records, g_margin_last_step = compute_level1_and_level3_for_checkpoint(
            epoch, C62_CHECKPOINT_DIR, loader_iter, device, N_REPLAY_STEPS, focal_fn, evidential_fn, rng, tau_b_tracker, objective
        )
        all_level1.extend(level1_records)

        # --- Level 3: cos(Delta_theta_margin, grad_theta_L_SC_last_step) ---
        if g_margin_last_step is not None:
            flat_delta = torch.cat([d.reshape(-1) for d in delta_theta_margin])
            flat_grad_last = torch.cat([g.reshape(-1) for g in g_margin_last_step])
            cos_delta_grad = float(F.cosine_similarity(flat_delta.unsqueeze(0), flat_grad_last.unsqueeze(0)).item())
            cos_delta_neg_grad = float(F.cosine_similarity(flat_delta.unsqueeze(0), (-flat_grad_last).unsqueeze(0)).item())
            delta_norm = float(flat_delta.norm().item())
            grad_norm = float(flat_grad_last.norm().item())
            level3_record = {
                "epoch": epoch,
                "cos_Delta_theta__grad_L_SC": cos_delta_grad,
                "cos_Delta_theta__neg_grad_L_SC": cos_delta_neg_grad,
                "delta_theta_norm": delta_norm,
                "grad_L_SC_last_step_norm": grad_norm,
            }
            all_level3.append(level3_record)
            print(f"  Level 3: cos(Delta_theta, grad_L_SC) = {cos_delta_grad:+.4f}, "
                  f"cos(Delta_theta, -grad_L_SC) = {cos_delta_neg_grad:+.4f} "
                  f"(||Delta_theta||={delta_norm:.4e}, ||grad_L_SC||={grad_norm:.4e})")
            print(f"    (a NEGATIVE cos(Delta_theta, grad_L_SC) close to -1 means AdamW's accumulated "
                  f"update points roughly OPPOSITE the raw gradient's last-step direction -- i.e. it IS "
                  f"a descent step, as expected; the interesting question is the MAGNITUDE of rotation "
                  f"away from -1, which quantifies how much AdamW's moment state and the seg-loss's own "
                  f"gradient reshape the naive single-step margin-descent direction)")

        n_l1 = len(level1_records)
        if n_l1 > 0:
            tumor_cos = [r["tumor"]["cos_mean"] for r in level1_records if r["tumor"] is not None]
            bg_cos = [r["background"]["cos_mean"] for r in level1_records if r["background"] is not None]
            print(f"  Level 1 ({n_l1} active steps/{N_REPLAY_STEPS}): "
                  f"tumor cos(grad_z L_SC, w_hat) mean={np.mean(tumor_cos):+.4f} (expect <0), "
                  f"bg mean={np.mean(bg_cos):+.4f} (expect >0)" if tumor_cos and bg_cos else
                  f"  Level 1: insufficient active data ({n_l1} steps)")

        # --- Level 4: cos(Delta_z, w_hat), split by tumor/bg/TP/FP/FN/TN ---
        axis = objective.get_w_hat(model, device)
        axis_unit = axis / axis.norm().clamp_min(1e-8)

        for batch_idx in range(N_TRANSPORT_BATCHES):
            try:
                images, masks, _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                images, masks, _ = next(loader_iter)
            images = images.to(device)
            masks = masks.to(device)

            model.eval()
            with torch.no_grad():
                outputs_orig = model(images)
                dec1_orig = outputs_orig["dec1"]
                probs_orig = outputs_orig["probs"]
                pred_orig = (probs_orig >= 0.5).float()
            model.train()

            dec1_perturbed = apply_delta_and_forward(model, delta_theta_margin, images)

            B, C, D, H, W = dec1_orig.shape
            dz = (dec1_perturbed - dec1_orig).permute(0, 2, 3, 4, 1).reshape(-1, C)
            gt_flat = masks.reshape(-1)
            pred_flat = pred_orig.reshape(-1)
            dz_norm_per_voxel = dz.norm(dim=1)
            valid = dz_norm_per_voxel > 1e-12

            if valid.sum() == 0:
                continue

            cos_per_voxel = F.cosine_similarity(dz[valid], axis_unit.unsqueeze(0).expand(valid.sum(), -1), dim=1)
            gt_valid = gt_flat[valid]
            pred_valid = pred_flat[valid]

            tumor_mask = gt_valid > 0.5
            bg_mask = ~tumor_mask
            tp_mask = (pred_valid == 1) & (gt_valid == 1)
            fp_mask = (pred_valid == 1) & (gt_valid == 0)
            fn_mask = (pred_valid == 0) & (gt_valid == 1)
            tn_mask = (pred_valid == 0) & (gt_valid == 0)

            def cat_stats(mask, expected_sign):
                subset = cos_per_voxel[mask]
                if subset.numel() == 0:
                    return None
                mean_cos = float(subset.mean().item())
                return {
                    "n": int(subset.numel()),
                    "cos_mean": mean_cos,
                    "frac_correct_sign": float(((subset * expected_sign) > 0).float().mean().item()) if expected_sign is not None else None,
                }

            # Expected sign is determined PURELY by ground-truth class
            # (SC-TAM's gradient direction depends only on GT identity,
            # not on the model's prediction -- see compute_margin_loss's
            # sc_tam branch, which never reads a prediction). So:
            #   TP, FN (both GT=tumor)      -> expect toward -w_hat
            #   FP, TN (both GT=background) -> expect toward +w_hat
            # This is the whole point of Level 4: measuring whether the
            # correct GT-conditioned sign (already confirmed in H2 for
            # the pooled tumor/background split) ALSO holds when the
            # voxels are split by whether the model's CURRENT prediction
            # already agrees with that ground truth (TP/TN) or not (FP/FN)
            # -- i.e. does SC-TAM's real movement treat errors differently
            # from already-correct voxels of the same GT class.
            level4_record = {
                "epoch": epoch, "batch_idx": batch_idx,
                "tumor": cat_stats(tumor_mask, -1.0),
                "background": cat_stats(bg_mask, +1.0),
                "TP": cat_stats(tp_mask, -1.0),
                "FN": cat_stats(fn_mask, -1.0),
                "FP": cat_stats(fp_mask, +1.0),
                "TN": cat_stats(tn_mask, +1.0),
            }
            all_level4.append(level4_record)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- Save all levels ---
    out = {
        "level1_activation_gradient": all_level1,
        "level2_total_gradient_from_h1": {f"{ep}": s for ep, s in level2_epoch_summary.items()},
        "level3_optimizer_update": all_level3,
        "level4_representation_update": all_level4,
    }
    json_path = OUT_DIR / "mechanism_audit_C62.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved full 4-level audit to {json_path}")


if __name__ == "__main__":
    main()
