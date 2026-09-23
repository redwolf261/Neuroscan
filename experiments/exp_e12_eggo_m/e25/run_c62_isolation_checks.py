"""
Phase E25, C6-2 mechanistic isolation experiments (pre-C6-3), per the
user's explicit instruction: investigate WHY a dominant, correctly-
sampled FN/FP loss contribution nonetheless produces wrong-signed
realized movement (Level 4 of PHASE_E25_C62_MECHANISM_AUDIT.md), before
designing anything. Three checks, in the user's own stated order:

  Check 3 (run FIRST here -- cheapest, and directly qualifies the prior
  anchor-composition result): the real compute_margin_loss weights each
  anchor's contribution by weight = U_hat * B (evidence-based uncertainty
  x boundary-confidence), which the prior anchor-composition diagnostic
  DELIBERATELY excluded (weight=1.0, to isolate raw class-conditional
  pair structure). This check recomputes loss shares WITH the real
  weight, on the same real checkpoint/batch data, to determine whether
  U_hat*B differentially suppresses or amplifies FN/FP relative to TP/TN
  -- if it does, the prior "FN/FP already dominate the loss" finding
  needs a real, quantified caveat, not just a note.

  Check 1: per-anchor gradient magnitude -- are FN/FP anchors' individual
  contributions to grad_theta(L_SC) correctly signed at the PARAMETER
  level (not just activation level, which Level 1 of the mechanism audit
  already proved is trivially always correct), but small relative to
  TP/TN's contributions, such that the SUMMED/aggregated grad_theta(L_SC)
  is dominated by TP/TN despite FN/FP's large LOSS share? This
  distinguishes "large loss value" from "large gradient contribution" --
  not necessarily the same thing once the loss's own derivative structure
  and the weight's own detached-but-nonuniform multiplier are accounted
  for.

  Check 2: network Jacobian -- for FN-anchor voxels specifically, does
  J_i = dz_i/d_theta (restricted to dec1's own parameters, matching every
  other phase's scope) transform a per-voxel grad_theta contribution that
  is itself correctly oriented (Check 1) into a wrong-signed contribution
  to the REALIZED per-voxel Delta_z once accumulated through AdamW and
  propagated back through the forward pass? This is measured by directly
  comparing, for the SAME FN voxels across the SAME checkpoint, (a) the
  sign of that voxel's own per-anchor grad_theta contribution (via a
  targeted single-anchor backward pass) against (b) the sign of that same
  voxel's REALIZED Delta_z (H2's own mechanism, restricted to FN voxels
  specifically rather than the whole batch) -- if (a) is correctly signed
  and (b) is not, the Jacobian/optimizer stage is where the sign flips.
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
from train_eggo_m import (  # noqa: E402
    sample_stratified_anchors, compute_margin_loss,
    ANCHORS_PER_VOLUME, MAX_NEGATIVES_PER_ANCHOR, EVIDENCE_P99_DEFAULT, EMATauB,
)

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e24"))
from run_counterfactual import ObjectiveConfig  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from e20_dec1_update_decomposition import ShadowAdam  # noqa: E402

CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30]
N_BATCHES_CHECK13 = 6  # matches anchor-composition diagnostic's own scope
N_REPLAY_STEPS = 15  # matches H2/mechanism-audit exactly
N_TRANSPORT_BATCHES_CHECK2 = 4  # matches H2's own scope

C62_CHECKPOINT_DIR = (project_root / "experiments" / "exp_e12_eggo_m" / "e25" /
                       "c62_runs" / "C62_sc_tam_seed0" / "checkpoints")
OUT_DIR = Path(__file__).parent / "isolation_check_results"

CAT_NAMES = {0: "TN", 1: "TP", 2: "FP", 3: "FN"}


def classify_voxels(gt, pred_binary):
    gt_pos = gt > 0.5
    pred_pos = pred_binary > 0.5
    tp = gt_pos & pred_pos
    tn = (~gt_pos) & (~pred_pos)
    fp = (~gt_pos) & pred_pos
    fn = gt_pos & (~pred_pos)
    cat = torch.zeros_like(gt, dtype=torch.int8)
    cat[tp] = 1
    cat[fp] = 2
    cat[fn] = 3
    return cat


# =====================================================================
# CHECK 3: real U_hat*B weighting effect on loss share by category
# =====================================================================

def check3_weighted_loss_share(dec1_flat, evidence_flat, boundary_logit_flat, gt_flat, cat_flat,
                                anchor_idx, tau_b, w_hat, delta_d, max_negatives, device):
    """Field-for-field copy of compute_margin_loss's sc_tam branch,
    INCLUDING the real U_hat*B weight this time (unlike the prior
    anchor-composition diagnostic), to directly measure whether the real
    weighting differentially suppresses/amplifies FN/FP vs TP/TN."""
    anchors_z = dec1_flat[anchor_idx]
    anchors_evidence = evidence_flat[anchor_idx].detach()
    anchors_boundary = boundary_logit_flat[anchor_idx].detach()
    anchors_gt = gt_flat[anchor_idx]
    anchors_cat = cat_flat[anchor_idx]

    U_hat = 1.0 - torch.clamp(anchors_evidence / EVIDENCE_P99_DEFAULT, 0.0, 1.0)
    B = torch.exp(-torch.abs(anchors_boundary) / tau_b)
    weight = U_hat * B  # REAL weight, unlike the prior diagnostic's weight=1.0

    n_anchor = anchors_z.shape[0]
    per_anchor_loss_unweighted = torch.zeros(n_anchor, device=device)
    per_anchor_loss_weighted = torch.zeros(n_anchor, device=device)

    tumor_mask = anchors_gt > 0.5
    bg_mask = ~tumor_mask
    tumor_local = torch.where(tumor_mask)[0]
    bg_local = torch.where(bg_mask)[0]

    with torch.no_grad():
        for pos_local, opp_local in ((tumor_local, bg_local), (bg_local, tumor_local)):
            if pos_local.numel() == 0 or opp_local.numel() == 0:
                continue
            n_pos = pos_local.numel()
            n_neg = min(max_negatives, opp_local.numel())
            rand_idx = torch.randint(0, opp_local.numel(), (n_pos, n_neg), device=device)
            neg_local = opp_local[rand_idx]

            zi_proj = (anchors_z[pos_local] @ w_hat).unsqueeze(1)
            zj_proj = anchors_z[neg_local] @ w_hat
            if pos_local is tumor_local:
                dist = zi_proj - zj_proj
            else:
                dist = zj_proj - zi_proj

            hinge = torch.clamp(delta_d - dist, min=0.0) ** 2
            per_anchor_loss = hinge.mean(dim=1)
            per_anchor_loss_unweighted[pos_local] = per_anchor_loss
            per_anchor_loss_weighted[pos_local] = weight[pos_local] * per_anchor_loss

    total_unweighted = float(per_anchor_loss_unweighted.sum().item())
    total_weighted = float(per_anchor_loss_weighted.sum().item())

    result = {}
    for cat_id in (0, 1, 2, 3):
        sel = anchors_cat == cat_id
        n_sel = int(sel.sum().item())
        raw_unw = float(per_anchor_loss_unweighted[sel].sum().item()) if n_sel > 0 else 0.0
        raw_w = float(per_anchor_loss_weighted[sel].sum().item()) if n_sel > 0 else 0.0
        mean_weight_this_cat = float(weight[sel].mean().item()) if n_sel > 0 else None
        result[CAT_NAMES[cat_id]] = {
            "n": n_sel,
            "share_unweighted": (raw_unw / total_unweighted) if total_unweighted > 1e-12 else None,
            "share_weighted": (raw_w / total_weighted) if total_weighted > 1e-12 else None,
            "mean_U_hat_B_weight": mean_weight_this_cat,
        }
    return result


# =====================================================================
# CHECK 1: per-anchor grad_theta(L_SC) sign, TP/TN vs FN/FP, aggregated
# vs individual
# =====================================================================

def _sc_tam_per_anchor_losses(dec1_flat_grad_enabled, evidence_flat, boundary_flat, gt_flat,
                               anchor_idx, tau_b, w_hat, delta_d, device):
    """Field-for-field reimplementation of compute_margin_loss's sc_tam
    branch, but returning the PER-ANCHOR weighted loss vector directly
    (n_anchor,) rather than only the scalar mean -- needed because Check
    1 must isolate ONE anchor's own contribution to grad_theta via
    autograd.grad on THAT SINGLE per-anchor loss term, while still
    drawing its negatives from the REAL, FULL anchor set (unlike a naive
    single-element anchor_idx call to the production function, which was
    smoke-tested and found to silently degenerate to a zero loss/no
    graph, since the opposite-class pool is drawn from OTHER anchors in
    the SAME anchor_idx -- a single-anchor anchor_idx has no opposite-
    class partner at all). This function keeps the FULL anchor pool
    (so real negatives exist) and returns each anchor's own per_anchor_loss
    row un-meaned, so the caller can isolate individual rows post hoc."""
    anchors_z = dec1_flat_grad_enabled[anchor_idx]
    anchors_evidence = evidence_flat[anchor_idx].detach()
    anchors_boundary = boundary_flat[anchor_idx].detach()
    anchors_gt = gt_flat[anchor_idx]

    U_hat = 1.0 - torch.clamp(anchors_evidence / EVIDENCE_P99_DEFAULT, 0.0, 1.0)
    B = torch.exp(-torch.abs(anchors_boundary) / tau_b)
    weight = U_hat * B

    n_anchor = anchors_z.shape[0]
    per_anchor_loss = torch.zeros(n_anchor, device=device)

    tumor_mask = anchors_gt > 0.5
    bg_mask = ~tumor_mask
    tumor_local = torch.where(tumor_mask)[0]
    bg_local = torch.where(bg_mask)[0]

    for pos_local, opp_local in ((tumor_local, bg_local), (bg_local, tumor_local)):
        if pos_local.numel() == 0 or opp_local.numel() == 0:
            continue
        n_pos = pos_local.numel()
        n_neg = min(MAX_NEGATIVES_PER_ANCHOR, opp_local.numel())
        rand_idx = torch.randint(0, opp_local.numel(), (n_pos, n_neg), device=device)
        neg_local = opp_local[rand_idx]

        zi_proj = (anchors_z[pos_local] @ w_hat).unsqueeze(1)
        zj_proj = anchors_z[neg_local] @ w_hat
        if pos_local is tumor_local:
            dist = zi_proj - zj_proj
        else:
            dist = zj_proj - zi_proj

        hinge = torch.clamp(delta_d - dist, min=0.0) ** 2
        pal = hinge.mean(dim=1)
        # weight[pos_local] is NOT detached from the graph via .detach()
        # here beyond what U_hat/B already are (both built from .detach()'d
        # inputs, matching the production function) -- assigning into a
        # fresh zeros tensor at pos_local indices preserves the
        # per-anchor gradient path exactly as compute_margin_loss's own
        # `losses[pos_local] = weight[pos_local] * per_anchor_loss` does.
        per_anchor_loss = per_anchor_loss.clone()
        per_anchor_loss[pos_local] = weight[pos_local] * pal

    return per_anchor_loss  # (n_anchor,), NOT meaned -- caller isolates individual rows


def check1_per_anchor_grad_theta(model, dec1_flat_grad_enabled, dec1_perm_shape_info, anchor_idx, anchors_cat,
                                  w_hat, delta_d, tau_b, evidence_flat, boundary_flat, gt_flat, rng, device):
    """For a SUBSET of individual anchors per category (a full per-anchor
    backward pass for every anchor would be prohibitively slow), measure
    that anchor's OWN isolated contribution to grad_theta(L_SC) by
    backpropagating from ONLY that anchor's own per_anchor_loss row
    (drawn from the REAL, full anchor pool's negative sampling -- see
    _sc_tam_per_anchor_losses's docstring for why a naive single-element
    anchor_idx call does not work), and record the L2 norm and cosine of
    that per-anchor parameter gradient against the FULL BATCH's
    aggregated grad_theta(L_SC)."""
    torch.manual_seed(0)
    per_anchor_loss_full = _sc_tam_per_anchor_losses(
        dec1_flat_grad_enabled, evidence_flat, boundary_flat, gt_flat,
        anchor_idx, tau_b, w_hat, delta_d, device,
    )
    margin_loss_full = per_anchor_loss_full.mean()
    dec1_params = list(model.dec1.parameters())
    g_full = torch.autograd.grad(margin_loss_full, dec1_params, retain_graph=True, allow_unused=True)
    g_full = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_full, dec1_params)]
    g_full_flat = torch.cat([g.reshape(-1) for g in g_full])
    g_full_norm = float(g_full_flat.norm().item())

    N_PER_CAT = 4
    per_anchor_records = []
    for cat_id in (0, 1, 2, 3):
        cat_anchor_local_idx = torch.where(anchors_cat == cat_id)[0]
        if cat_anchor_local_idx.numel() == 0:
            continue
        chosen = cat_anchor_local_idx[torch.randperm(cat_anchor_local_idx.numel())[:N_PER_CAT]]
        for local_i in chosen.tolist():
            if per_anchor_loss_full[local_i].item() == 0.0:
                continue  # this anchor was inactive (hinge not firing) -- no gradient to probe
            g_single = torch.autograd.grad(per_anchor_loss_full[local_i], dec1_params, retain_graph=True, allow_unused=True)
            g_single = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_single, dec1_params)]
            g_single_flat = torch.cat([g.reshape(-1) for g in g_single])
            g_single_norm = float(g_single_flat.norm().item())
            if g_single_norm < 1e-12:
                continue
            cos_with_full = float(F.cosine_similarity(g_single_flat.unsqueeze(0), g_full_flat.unsqueeze(0)).item())
            per_anchor_records.append({
                "category": CAT_NAMES[cat_id],
                "grad_norm": g_single_norm,
                "cos_with_full_batch_grad": cos_with_full,
            })

    return {
        "g_full_batch_norm": g_full_norm,
        "per_anchor_probes": per_anchor_records,
    }


# =====================================================================
# CHECK 2: Jacobian/optimizer stage -- does a correctly-signed per-anchor
# grad_theta contribution for FN voxels end up as a wrong-signed
# REALIZED Delta_z for those SAME voxels?
# =====================================================================

def check2_fn_realized_vs_gradient_sign(epoch, ckpt_dir, device, focal_fn=None, evidential_fn=None, seed_offset=0):
    """Reuses H2's exact ShadowAdam replay mechanism (15 real steps,
    full-batch aggregated gradient -- NOT the single-anchor isolation of
    Check 1, since Check 2 asks about the REALIZED, AdamW-accumulated
    effect, which is inherently a full-batch-trajectory quantity) to get
    the real Delta_theta_margin, then applies it and re-forward-passes,
    exactly as H2 did -- but THIS TIME, explicitly tracks, for a FIXED
    set of FN voxels identified at the LAST replay step's batch, both:
      (a) the sign of cos(Delta_z_voxel, w_hat) for those SPECIFIC voxels
          (H2's own quantity, but tracked for a fixed, identified voxel
          set rather than fresh voxels from a new batch), and
      (b) whether the network's OWN forward Jacobian direction (measured
          via a finite-difference probe: perturb theta by a SMALL step
          purely along the single-anchor gradient direction from Check 1
          for one of these same FN voxels, and see whether the resulting
          real Delta_z at that SAME voxel is correctly or wrongly signed)
    agrees or disagrees -- if (a)'s full realized displacement is wrong
    for FN but a small pure single-anchor-gradient-direction probe (b)
    is CORRECTLY signed, this localizes the flip specifically to the
    AGGREGATION/OPTIMIZER stage (many other anchors' gradients + AdamW's
    moment state), not to the Jacobian itself being adversarial for any
    individual voxel's own intended direction."""
    ckpt_path = ckpt_dir / f"epoch_{epoch}.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.train()

    dec1_params = list(model.dec1.parameters())
    pg = ckpt["optimizer_state"]["param_groups"][0]
    lr, (beta1, beta2), eps = pg["lr"], pg["betas"], pg["eps"]

    from Dataset.brats_dataset import create_brats_loaders
    train_loader, _ = create_brats_loaders(
        batch_size=2, num_workers=0,
        root_dir=str(project_root / "Dataset" / "Training"),
        val_split=0.1,
    )
    rng = np.random.RandomState(epoch * 1000 + seed_offset)  # seed_offset lets the caller retry with a different real batch sequence if this draw has 0 FN anchors
    tau_b_tracker = EMATauB()
    loader_iter = iter(train_loader)
    objective = ObjectiveConfig(mode="sc_tam")

    shadow_margin = ShadowAdam(dec1_params, lr, beta1, beta2, eps)
    last_batch_images = None
    last_batch_masks = None
    last_anchor_idx = None
    last_dec1_perm = None
    last_gt_flat = None
    last_evidence_flat = None
    last_boundary_flat = None

    for step_idx in range(N_REPLAY_STEPS):
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
            w_hat=w_hat, margin_mode="sc_tam",
        )
        tau_b_tracker.update(margin_diag["abs_boundary_logit"])

        if step_idx == N_REPLAY_STEPS - 1:
            last_batch_images = images
            last_batch_masks = masks
            last_anchor_idx = anchor_idx
            last_dec1_perm = dec1_perm
            last_gt_flat = gt_flat
            last_evidence_flat = evidence_flat
            last_boundary_flat = boundary_flat

        if not torch.isfinite(margin_loss):
            continue
        g_margin = torch.autograd.grad(margin_loss, dec1_params, retain_graph=(step_idx == N_REPLAY_STEPS - 1), allow_unused=True)
        g_margin = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_margin, dec1_params)]
        shadow_margin.step(g_margin)

    delta_theta_margin = shadow_margin.delta_sum

    # Identify FN voxels in the LAST batch (model's own prediction vs GT)
    with torch.no_grad():
        outputs_pred = model(last_batch_images)
        probs = outputs_pred["probs"]
        pred_binary = (probs >= 0.5).float()
        pred_flat = pred_binary.reshape(-1)
    cat_flat = classify_voxels(last_gt_flat, pred_flat)
    fn_local_idx = torch.where(cat_flat[last_anchor_idx] == 3)[0]  # FN anchors among the last batch's sampled anchors

    if fn_local_idx.numel() == 0:
        return {"epoch": epoch, "n_fn_anchors": 0, "note": "no FN anchors in last batch"}

    fn_global_idx = last_anchor_idx[fn_local_idx]

    # (a) H2's own realized-Delta_z mechanism, restricted to these SPECIFIC FN voxels
    def apply_delta_and_forward(model_in, delta_list, images_in):
        model_copy = copy.deepcopy(model_in)
        model_copy.train()
        with torch.no_grad():
            for p, d in zip(model_copy.dec1.parameters(), delta_list):
                p.add_(d)
        with torch.no_grad():
            outputs_out = model_copy(images_in)
        del model_copy
        return outputs_out["dec1"]

    with torch.no_grad():
        outputs_orig = model(last_batch_images)
        dec1_orig = outputs_orig["dec1"]
    dec1_perturbed = apply_delta_and_forward(model, delta_theta_margin, last_batch_images)

    B2, C2, D2, H2_, W2 = dec1_orig.shape
    dz_full = (dec1_perturbed - dec1_orig).permute(0, 2, 3, 4, 1).reshape(-1, C2)
    axis_unit = w_hat / w_hat.norm().clamp_min(1e-8)
    cos_full_realized = F.cosine_similarity(dz_full[fn_global_idx], axis_unit.unsqueeze(0).expand(fn_global_idx.numel(), -1), dim=1)
    realized_frac_correct = float((cos_full_realized < 0).float().mean().item())  # FN expects <0
    realized_cos_mean = float(cos_full_realized.mean().item())

    # (b) Single-anchor-ISOLATED gradient probe for a handful of these SAME
    # FN voxels: does that voxel's OWN isolated per-anchor loss row
    # (drawn from the REAL, full anchor pool for negative sampling -- see
    # _sc_tam_per_anchor_losses's docstring; a naive single-element
    # anchor_idx call to compute_margin_loss silently degenerates to a
    # zero loss with no graph, since the opposite-class pool would be
    # empty within a 1-anchor set), applied as a tiny standalone
    # perturbation (NOT the AdamW-accumulated, full-batch trajectory,
    # and NOT contaminated by any OTHER anchor's gradient), produce a
    # correctly-signed Delta_z at ITSELF?
    N_PROBE = min(6, fn_global_idx.numel())
    probe_records = []
    with torch.enable_grad():
        outputs_fresh = model(last_batch_images)
        dec1_fresh = outputs_fresh["dec1"]
        dec1_perm_fresh = dec1_fresh.permute(0, 2, 3, 4, 1).reshape(-1, C2)
        alpha_f, beta_f = outputs_fresh["alpha"], outputs_fresh["beta"]
        with torch.no_grad():
            evidence_fresh = (alpha_f + beta_f - 2.0).reshape(-1)
        boundary_fresh = outputs_fresh["boundary_logit"].reshape(-1)
        gt_fresh = last_batch_masks.reshape(-1)

        torch.manual_seed(0)
        per_anchor_loss_fresh = _sc_tam_per_anchor_losses(
            dec1_perm_fresh, evidence_fresh, boundary_fresh, gt_fresh,
            last_anchor_idx, tau_b_tracker.tau_b, w_hat, objective.delta_d, device,
        )
        fn_local_positions = fn_local_idx  # local indices within last_anchor_idx, already computed above

        for probe_i in range(N_PROBE):
            local_pos = int(fn_local_positions[probe_i].item())
            global_voxel_idx = int(last_anchor_idx[local_pos].item())
            if per_anchor_loss_fresh[local_pos].item() == 0.0:
                probe_records.append({"voxel_idx": global_voxel_idx, "note": "inactive (zero loss) for this anchor"})
                continue
            g_single = torch.autograd.grad(per_anchor_loss_fresh[local_pos], dec1_params, retain_graph=True, allow_unused=True)
            g_single = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_single, dec1_params)]
            g_single_norm = sum(float((g**2).sum()) for g in g_single) ** 0.5
            if g_single_norm < 1e-12:
                probe_records.append({"voxel_idx": global_voxel_idx, "note": "zero gradient norm"})
                continue
            # Apply a TINY step purely along -g_single (pure gradient descent on
            # this ONE anchor's own loss row, no AdamW, no other anchors) and
            # measure the resulting Delta_z sign at THIS SAME voxel.
            EPS_PROBE = 0.01
            delta_probe = [-EPS_PROBE * g for g in g_single]
            single_idx = torch.tensor([global_voxel_idx], device=device)
            dec1_probe_perturbed = apply_delta_and_forward(model, delta_probe, last_batch_images)
            with torch.no_grad():
                dz_probe = (dec1_probe_perturbed - dec1_orig).permute(0, 2, 3, 4, 1).reshape(-1, C2)
                cos_probe = float(F.cosine_similarity(dz_probe[single_idx], axis_unit.unsqueeze(0), dim=1).item())
            probe_records.append({
                "voxel_idx": int(single_idx.item()),
                "cos_isolated_probe": cos_probe,
                "correct_sign": cos_probe < 0,  # FN expects <0
            })

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "epoch": epoch,
        "n_fn_anchors_in_last_batch": int(fn_global_idx.numel()),
        "realized_full_trajectory": {
            "cos_mean": realized_cos_mean,
            "frac_correct_sign": realized_frac_correct,
        },
        "isolated_single_anchor_probes": probe_records,
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = create_brats_loaders(
        batch_size=2, num_workers=0,
        root_dir=str(project_root / "Dataset" / "Training"),
        val_split=0.1,
    )
    objective = ObjectiveConfig(mode="sc_tam")

    print("=" * 70)
    print("CHECK 3: real U_hat*B weighting effect on loss share by category")
    print("=" * 70)
    check3_all = []
    for epoch in CHECKPOINT_EPOCHS:
        ckpt = torch.load(C62_CHECKPOINT_DIR / f"epoch_{epoch}.pth", map_location=device, weights_only=False)
        model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        rng = np.random.RandomState(epoch)
        tau_b_tracker = EMATauB()
        loader_iter = iter(train_loader)
        w_hat = objective.get_w_hat(model, device)

        for batch_idx in range(N_BATCHES_CHECK13):
            try:
                images, masks, _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                images, masks, _ = next(loader_iter)
            images = images.to(device)
            masks = masks.to(device)
            with torch.no_grad():
                outputs = model(images)
                dec1 = outputs["dec1"]
                probs = outputs["probs"]
                pred_binary = (probs >= 0.5).float()
                alpha, beta = outputs["alpha"], outputs["beta"]
                evidence_full = alpha + beta - 2.0
                boundary_logit = outputs["boundary_logit"]

            B, C, D, H, W = dec1.shape
            dec1_flat = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
            gt_flat = masks.reshape(-1)
            pred_flat = pred_binary.reshape(-1)
            evidence_flat = evidence_full.reshape(-1)
            boundary_flat = boundary_logit.reshape(-1)
            cat_flat = classify_voxels(gt_flat, pred_flat)

            voxels_per_vol = D * H * W
            anchor_idx_list = []
            for b in range(B):
                vol_evidence = evidence_flat[b * voxels_per_vol:(b + 1) * voxels_per_vol]
                local_idx = sample_stratified_anchors(vol_evidence, ANCHORS_PER_VOLUME, rng)
                anchor_idx_list.append(local_idx + b * voxels_per_vol)
            anchor_idx = torch.cat(anchor_idx_list)

            result = check3_weighted_loss_share(
                dec1_flat, evidence_flat, boundary_flat, gt_flat, cat_flat,
                anchor_idx, tau_b_tracker.tau_b, w_hat, objective.delta_d, MAX_NEGATIVES_PER_ANCHOR, device,
            )
            result["epoch"] = epoch
            result["batch_idx"] = batch_idx
            check3_all.append(result)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  epoch {epoch} done")

    with open(OUT_DIR / "check3_weighted_loss_share.json", "w") as f:
        json.dump(check3_all, f, indent=2)
    print(f"Saved Check 3 to {OUT_DIR / 'check3_weighted_loss_share.json'}")

    print("\n" + "=" * 70)
    print("CHECK 1: per-anchor grad_theta(L_SC) sign vs full-batch aggregate")
    print("=" * 70)
    check1_all = []
    for epoch in CHECKPOINT_EPOCHS:
        ckpt = torch.load(C62_CHECKPOINT_DIR / f"epoch_{epoch}.pth", map_location=device, weights_only=False)
        model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.train()
        rng = np.random.RandomState(epoch)
        tau_b_tracker = EMATauB()
        loader_iter = iter(train_loader)
        w_hat = objective.get_w_hat(model, device)

        images, masks, _ = next(loader_iter)
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

            result = check1_per_anchor_grad_theta(
                model, dec1_perm, None, anchor_idx, anchors_cat,
                w_hat, objective.delta_d, tau_b_tracker.tau_b, evidence_flat, boundary_flat, gt_flat, rng, device,
            )
            result["epoch"] = epoch
            check1_all.append(result)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        n_probes = len(result["per_anchor_probes"])
        print(f"  epoch {epoch}: {n_probes} per-anchor probes, g_full_batch_norm={result['g_full_batch_norm']:.4e}")

    with open(OUT_DIR / "check1_per_anchor_gradient.json", "w") as f:
        json.dump(check1_all, f, indent=2)
    print(f"Saved Check 1 to {OUT_DIR / 'check1_per_anchor_gradient.json'}")

    print("\n" + "=" * 70)
    print("CHECK 2: Jacobian/optimizer stage -- FN realized vs isolated-probe sign")
    print("=" * 70)
    MAX_RETRIES_CHECK2 = 5  # retry with a different real batch sequence if a given draw happens to contain 0 FN anchors (FN is rare among sampled anchors -- ~4% per the anchor-composition diagnostic -- so this is expected to happen sometimes, not a bug)
    check2_all = []
    for epoch in CHECKPOINT_EPOCHS:
        result = None
        for attempt in range(MAX_RETRIES_CHECK2):
            result = check2_fn_realized_vs_gradient_sign(epoch, C62_CHECKPOINT_DIR, device, seed_offset=attempt)
            if result.get("n_fn_anchors_in_last_batch", 0) > 0:
                break
            print(f"  epoch {epoch} attempt {attempt}: 0 FN anchors, retrying with a different batch draw")
        check2_all.append(result)
        if "realized_full_trajectory" in result:
            rt = result["realized_full_trajectory"]
            probes = result["isolated_single_anchor_probes"]
            n_correct_probes = sum(1 for p in probes if p.get("correct_sign") is True)
            n_valid_probes = sum(1 for p in probes if "correct_sign" in p)
            print(f"  epoch {epoch}: n_FN={result['n_fn_anchors_in_last_batch']}, "
                  f"realized_frac_correct={rt['frac_correct_sign']:.3f}, "
                  f"isolated_probe_correct={n_correct_probes}/{n_valid_probes}")
        else:
            print(f"  epoch {epoch}: {result.get('note', 'no data')}")

    with open(OUT_DIR / "check2_jacobian_isolation.json", "w") as f:
        json.dump(check2_all, f, indent=2)
    print(f"Saved Check 2 to {OUT_DIR / 'check2_jacobian_isolation.json'}")


if __name__ == "__main__":
    main()
