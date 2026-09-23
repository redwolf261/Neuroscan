"""
Phase E24, Gate 6, Section 6 step 1: parameterized counterfactual
diagnostic, extracted from experiments/exp_e12_eggo_m/e22/
e22_counterfactual_geometry.py.

MECHANICAL EXTRACTION, NOT A REDESIGN. Per the user's explicit
constraint: same sampling, same RNG handling, same .train()/.eval()
conventions, same parameter scope (dec1), same epsilon definitions, same
subject handling, same geometry calculation, same aggregation. The
ORIGINAL e22_counterfactual_geometry.py is left completely untouched and
remains the reference implementation until this refactor's regression
gate (run_regression_gate.py) passes.

The ONLY thing objective_config changes: which distance function
compute_margin_loss uses (w_hat=None + delta_d=DELTA_D_CALIBRATED for
baseline Euclidean; w_hat=<live seg_head direction> + delta_d=delta_d_w
for task-aligned; w_hat=<frozen random unit vector> + delta_d=delta_d_w
for random-projection) -- applied UNIFORMLY to every compute_margin_loss
call site in a given run (gradient construction AND before/after scoring
for every direction), per PHASE_E24_GATE6_EXPERIMENTAL_SPECIFICATION.md
Section 5's correction: a run's reported L_margin values must all be the
SAME objective, consistently, for corr(ΔL_margin, ΔDice) to be a
coherent per-condition measurement. This mirrors exactly how the
ORIGINAL E22 script already applies DELTA_D_CALIBRATED uniformly across
its own 4 call sites (verified by direct inspection before writing this
file) -- objective_config generalizes that existing uniform-application
pattern, it does not introduce a new one.

Everything else -- anchor fixing, torch.manual_seed discipline (the fix
for compute_margin_loss's negative-sampling RNG issue that caused real,
documented finite-difference noise during E22's own development),
compute_useful_direction's formula, compute_geometry_metric's formula,
the epsilon sweeps, subject/checkpoint iteration, Direction D's
activation-space mechanism, Direction E's matched-norm random controls --
is copied verbatim from e22_counterfactual_geometry.py, not rewritten.
"""
import sys
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import (  # noqa: E402
    sample_stratified_anchors, compute_margin_loss,
    DELTA_D_CALIBRATED, ANCHORS_PER_VOLUME, MAX_NEGATIVES_PER_ANCHOR,
    EVIDENCE_P99_DEFAULT, EMATauB,
)

N_VAL_SUBJECTS = 20  # unchanged from E22
FOCAL_WEIGHT = 0.5  # unchanged from E22
EVIDENTIAL_WEIGHT = 0.5  # unchanged from E22
LAMBDA_MARGIN_CALIBRATED = 0.1  # unchanged from E22

EPSILON_SWEEP_PARAM = [0.25, 0.5, 1.0, 2.0]  # unchanged from E22
EPSILON_SWEEP_ACTIVATION = [1.0, 2.0, 4.0, 14.0]  # unchanged from E22

MAX_MARGIN_PAIRS = 5000  # unchanged from E22

DELTA_D_W_CALIBRATED = 0.2553  # PHASE_E24's calibration result, for task_aligned mode (w_hat = live seg_head direction)
DELTA_D_R_CALIBRATED = 0.2022  # PHASE_E24 pre-launch-audit fix: SEPARATE calibration for random_projection mode
                                # (w_hat = frozen random r, seed=999001) -- measured directly to differ from
                                # delta_d_w by ~21% (0.2022 vs 0.2553), confirming it is NOT safe to reuse
                                # delta_d_w for this mode. See e24_calibrate_delta_d_r.py and the
                                # ObjectiveConfig docstring below for the full rationale.
M_IJ_CALIBRATED = 0.3089  # PHASE_E25 Section 12 calibration result (e25_calibrate_m_ij.py), for sc_tam mode
                           # (w_hat = live seg_head direction, SAME construction as task_aligned -- SC-TAM's
                           # own signed FB-pair distance requires its own m_ij threshold, not delta_d_w, per
                           # e25_calibrate_m_ij.py's docstring: sc_tam's margin_target is used UN-doubled,
                           # unlike euclidean/task_aligned's 2*delta_d convention)


class ObjectiveConfig:
    """Selects which distance function compute_margin_loss uses for this
    entire run. Three modes, matching PHASE_E24_GATE6_EXPERIMENTAL_
    SPECIFICATION.md Section 2's condition matrix exactly:

      "baseline"          -> w_hat=None, delta_d=DELTA_D_CALIBRATED
                              (the ORIGINAL E22 behavior, byte-for-byte)
      "task_aligned"       -> w_hat=<live seg_head direction, recomputed
                              fresh per checkpoint from that checkpoint's
                              OWN seg_head.weight, detached>,
                              delta_d=DELTA_D_W_CALIBRATED
      "random_projection"  -> w_hat=<a FIXED, frozen-for-the-whole-run
                              random unit vector, generated from a seed
                              INDEPENDENT of the training/evaluation RNG,
                              per the audit's point 2>, delta_d=DELTA_D_R_CALIBRATED

    w_hat for "task_aligned" is recomputed per checkpoint (not frozen
    across checkpoints) because it IS seg_head's own live weight at that
    checkpoint -- this matches how B's real training run would compute it
    fresh each time (train_eggo_m.py's own w_hat construction, per
    PHASE_E24 Section 2.3), not a single frozen snapshot.

    CORRECTED PER PRE-LAUNCH AUDIT: "random_projection" originally reused
    DELTA_D_W_CALIBRATED (w_hat's own calibration), which is WRONG -- w_hat
    and r are different unit vectors with different projected-distance
    distributions against the same fresh-init geometry (measured directly:
    delta_d_w=0.2553 vs delta_d_r=0.2022, a real ~21% difference, NOT
    interchangeable). Reusing w's calibration for r would have confounded
    "is task-alignment useful" with "is a miscalibrated hinge threshold
    useful" -- defeating the H4 specificity comparison's whole purpose.
    Fixed by a dedicated calibration (e24_calibrate_delta_d_r.py),
    calibrated against THIS EXACT r (same seed=999001, verified
    bit-identical by that script's own cross-check before trusting the
    result).

    FOURTH MODE ADDED PER PHASE_E25 (C6-2): "sc_tam" -> w_hat=<live
    seg_head direction, SAME construction as task_aligned>,
    delta_d=M_IJ_CALIBRATED, margin_mode="sc_tam" (the ONLY mode where
    margin_mode differs from what compute_margin_loss's own w_hat-based
    auto-inference would pick -- auto-inference has no way to distinguish
    "task_aligned" from "sc_tam" since both pass a non-None w_hat, so this
    mode is the reason ObjectiveConfig now has an explicit margin_mode
    property/parameter threaded through every call site below, rather
    than continuing to rely on auto-inference as the first three modes
    always could)."""

    def __init__(self, mode, random_seed=999001):
        assert mode in ("baseline", "task_aligned", "random_projection", "sc_tam")
        self.mode = mode
        self.random_seed = random_seed  # independent of training RNG, per audit point 2
        self._frozen_random_w_hat = None
        if mode == "random_projection":
            g = torch.Generator().manual_seed(random_seed)
            r = torch.randn(32, generator=g)
            self._frozen_random_w_hat = (r / r.norm()).detach()

    @property
    def delta_d(self):
        if self.mode == "baseline":
            return DELTA_D_CALIBRATED
        elif self.mode == "task_aligned":
            return DELTA_D_W_CALIBRATED
        elif self.mode == "random_projection":
            return DELTA_D_R_CALIBRATED
        elif self.mode == "sc_tam":
            return M_IJ_CALIBRATED

    @property
    def margin_mode(self):
        """Explicit margin_mode for compute_margin_loss, threaded through
        every call site instead of relying on auto-inference -- required
        because "task_aligned" and "sc_tam" both pass a non-None w_hat and
        are otherwise indistinguishable to compute_margin_loss's own
        w_hat-based inference (margin_mode=None -> "task_aligned" whenever
        w_hat is not None). For baseline/task_aligned/random_projection
        this property returns EXACTLY what auto-inference would already
        have picked, so passing it explicitly is a no-op behavior change
        for those three modes (verified by the regression gate below)."""
        if self.mode == "baseline":
            return "euclidean"
        elif self.mode in ("task_aligned", "random_projection"):
            return "task_aligned"
        elif self.mode == "sc_tam":
            return "sc_tam"

    def get_w_hat(self, model, device):
        """Returns the w_hat to use for THIS checkpoint's compute_margin_loss
        calls, or None for baseline mode."""
        if self.mode == "baseline":
            return None
        elif self.mode in ("task_aligned", "sc_tam"):
            with torch.no_grad():
                w_raw = model.seg_head[0].weight.detach().reshape(-1)
                return (w_raw / w_raw.norm().clamp_min(1e-8)).detach().to(device)
        elif self.mode == "random_projection":
            return self._frozen_random_w_hat.to(device)


def dice_score(pred_binary, gt):
    tp = (pred_binary * gt).sum()
    return (2 * tp / (pred_binary.sum() + gt.sum() + 1e-6)).item()


def compute_geometry_metric(dec1_flat, gt_flat, seed_t=1, seed_b=2):
    """Verbatim copy from e22_counterfactual_geometry.py -- UNCHANGED."""
    tumor_idx = torch.where(gt_flat > 0.5)[0]
    bg_idx = torch.where(gt_flat <= 0.5)[0]
    if len(tumor_idx) == 0 or len(bg_idx) == 0:
        return None
    n_pairs = min(MAX_MARGIN_PAIRS, len(tumor_idx), len(bg_idx))
    t_rng = np.random.RandomState(seed_t)
    b_rng = np.random.RandomState(seed_b)
    t_sample = tumor_idx[torch.from_numpy(t_rng.choice(len(tumor_idx), size=n_pairs, replace=(n_pairs > len(tumor_idx))))]
    b_sample = bg_idx[torch.from_numpy(b_rng.choice(len(bg_idx), size=n_pairs, replace=(n_pairs > len(bg_idx))))]
    margin_dist = (dec1_flat[t_sample] - dec1_flat[b_sample]).norm(dim=1)
    return float(margin_dist.mean().item())


def compute_useful_direction(dec1_flat, gt_flat):
    """Verbatim copy from e22_counterfactual_geometry.py -- UNCHANGED.
    This is E15's direction formula and is NOT affected by objective_config
    -- Direction D always uses the true Euclidean-centroid useful direction,
    regardless of which margin objective is being tested, since D is the
    independently-validated reference point every objective_config is
    measured against."""
    tumor_mask = gt_flat > 0.5
    bg_mask = ~tumor_mask
    if tumor_mask.sum() < 10 or bg_mask.sum() < 10:
        return None
    tumor_centroid = dec1_flat[tumor_mask].mean(dim=0)
    bg_centroid = dec1_flat[bg_mask].mean(dim=0)
    direction = torch.zeros_like(dec1_flat)
    direction[tumor_mask] = dec1_flat[tumor_mask] - bg_centroid.unsqueeze(0)
    direction[bg_mask] = dec1_flat[bg_mask] - tumor_centroid.unsqueeze(0)
    unit_dir = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return unit_dir


def compute_losses_and_grads(model, image, mask, rng, tau_b_tracker, focal_fn, evidential_fn, device, torch_seed, objective):
    """Same as e22_counterfactual_geometry.py's function of the same name,
    EXTENDED ONLY to pass objective.get_w_hat()/objective.delta_d into
    compute_margin_loss instead of the hardcoded w_hat=None/DELTA_D_CALIBRATED.
    Every other line is unchanged."""
    model.train()
    torch.manual_seed(torch_seed)
    dec1_params = list(model.dec1.parameters())

    outputs = model(image)
    probs = outputs["probs"]
    alpha, beta = outputs["alpha"], outputs["beta"]
    boundary_logit = outputs["boundary_logit"]
    dec1 = outputs["dec1"]

    B, C, D, H, W = dec1.shape
    with torch.no_grad():
        evidence_full = alpha + beta - 2.0
    dec1_perm = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
    evidence_flat = evidence_full.reshape(-1)
    boundary_flat = boundary_logit.reshape(-1)
    gt_flat = mask.reshape(-1)

    focal_loss = focal_fn(probs, mask)
    evidential_loss = evidential_fn(alpha, beta, mask)
    seg_loss = FOCAL_WEIGHT * focal_loss + EVIDENTIAL_WEIGHT * evidential_loss

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

    if not (torch.isfinite(margin_loss) and torch.isfinite(seg_loss)):
        return None

    total_loss = seg_loss + LAMBDA_MARGIN_CALIBRATED * margin_loss

    g_seg = torch.autograd.grad(seg_loss, dec1_params, retain_graph=True, allow_unused=True)
    g_margin = torch.autograd.grad(margin_loss, dec1_params, retain_graph=True, allow_unused=True)
    g_total = torch.autograd.grad(total_loss, dec1_params, retain_graph=False, allow_unused=True)

    g_seg = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_seg, dec1_params)]
    g_margin = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_margin, dec1_params)]
    g_total = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_total, dec1_params)]

    return {
        "dec1_params": dec1_params,
        "g_seg": g_seg, "g_margin": g_margin, "g_total": g_total,
        "seg_loss": float(seg_loss.item()), "margin_loss": float(margin_loss.item()),
        "total_loss": float(total_loss.item()),
    }


def normalize_flat(grad_list):
    """Verbatim copy -- UNCHANGED."""
    total_norm = sum(float((g ** 2).sum()) for g in grad_list) ** 0.5
    if total_norm < 1e-20:
        return [torch.zeros_like(g) for g in grad_list], 0.0
    return [g / total_norm for g in grad_list], total_norm


def apply_param_perturbation_and_eval(model, dec1_params_template, delta_list, image, mask, focal_fn, evidential_fn, fixed_anchor_idx, tau_b_value, torch_seed, objective):
    """Same as e22_counterfactual_geometry.py's function of the same name,
    EXTENDED ONLY to use objective.get_w_hat()/objective.delta_d. Every
    other line, including the correctness-critical fixed_anchor_idx/
    torch_seed discipline documented in the original docstring, is
    unchanged."""
    model_copy = copy.deepcopy(model)
    with torch.no_grad():
        for p, d in zip(model_copy.dec1.parameters(), delta_list):
            p.add_(d)
    model_copy.eval()

    with torch.no_grad():
        outputs = model_copy(image)
        probs = outputs["probs"]
        alpha, beta = outputs["alpha"], outputs["beta"]
        dec1 = outputs["dec1"]
        pred_binary = (probs >= 0.5).float()
        dice = dice_score(pred_binary, mask)

        gt_flat = mask.reshape(-1)
        B, C, D, H, W = dec1.shape
        dec1_flat = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
        geometry = compute_geometry_metric(dec1_flat, gt_flat)

        focal_loss = focal_fn(probs, mask)
        evidential_loss = evidential_fn(alpha, beta, mask)
        seg_loss = float((FOCAL_WEIGHT * focal_loss + EVIDENTIAL_WEIGHT * evidential_loss).item())

        evidence_full = alpha + beta - 2.0
        boundary_logit = outputs["boundary_logit"]
        evidence_flat = evidence_full.reshape(-1)
        boundary_flat = boundary_logit.reshape(-1)
        torch.manual_seed(torch_seed)
        dummy_rng = np.random.RandomState(torch_seed)
        w_hat = objective.get_w_hat(model_copy, dec1.device)
        margin_loss, _, margin_diag = compute_margin_loss(
            dec1_flat, evidence_flat, boundary_flat, gt_flat,
            fixed_anchor_idx, tau_b_value, EVIDENCE_P99_DEFAULT, objective.delta_d,
            MAX_NEGATIVES_PER_ANCHOR, dummy_rng, dec1.device,
            w_hat=w_hat, margin_mode=objective.margin_mode,
        )
        margin_loss = float(margin_loss.item())

    del model_copy
    return {"dice": dice, "seg_loss": seg_loss, "margin_loss": margin_loss, "geometry": geometry}


def apply_direction_d_and_eval(model, image, mask, push_units, focal_fn, evidential_fn, fixed_anchor_idx, tau_b_value, torch_seed, objective):
    """Same as e22_counterfactual_geometry.py's function of the same name,
    EXTENDED ONLY to use objective.get_w_hat()/objective.delta_d for the
    margin_loss SCORING at the end (Direction D's own push mechanism --
    d_useful, the activation-space push itself -- is UNCHANGED, always
    E15's true Euclidean-centroid direction, per compute_useful_direction's
    own docstring above)."""
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        dec1 = outputs["dec1"]
        B, C, D, H, W = dec1.shape
        dec1_flat = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
        gt_flat = mask.reshape(-1)

        d_useful = compute_useful_direction(dec1_flat, gt_flat)
        if d_useful is None:
            return None

        dec1_perturbed_flat = dec1_flat + push_units * d_useful
        dec1_perturbed = dec1_perturbed_flat.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        probs = model.seg_head(dec1_perturbed)
        pred_binary = (probs >= 0.5).float()
        dice = dice_score(pred_binary, mask)

        evidential_raw = model.evidential_head(dec1_perturbed)
        alpha = F.softplus(evidential_raw[:, 0:1]) + 1.0
        beta = F.softplus(evidential_raw[:, 1:2]) + 1.0

        focal_loss = focal_fn(probs, mask)
        evidential_loss = evidential_fn(alpha, beta, mask)
        seg_loss = float((FOCAL_WEIGHT * focal_loss + EVIDENTIAL_WEIGHT * evidential_loss).item())

        geometry = compute_geometry_metric(dec1_perturbed_flat, gt_flat)

        evidence_full = alpha + beta - 2.0
        evidence_flat = evidence_full.reshape(-1)
        boundary_flat = outputs["boundary_logit"].reshape(-1)
        torch.manual_seed(torch_seed)
        dummy_rng = np.random.RandomState(torch_seed)
        w_hat = objective.get_w_hat(model, dec1.device)
        margin_loss, _, margin_diag = compute_margin_loss(
            dec1_perturbed_flat, evidence_flat, boundary_flat, gt_flat,
            fixed_anchor_idx, tau_b_value, EVIDENCE_P99_DEFAULT, objective.delta_d,
            MAX_NEGATIVES_PER_ANCHOR, dummy_rng, dec1.device,
            w_hat=w_hat, margin_mode=objective.margin_mode,
        )
        margin_loss = float(margin_loss.item())

    return {"dice": dice, "seg_loss": seg_loss, "margin_loss": margin_loss, "geometry": geometry, "d_useful": d_useful}


def run_one_subject(epoch, subject_idx, subject_id, model, image_b, mask_b, focal_fn, evidential_fn, device, objective):
    """Same as e22_counterfactual_geometry.py's function of the same name,
    EXTENDED ONLY to thread objective through every compute_losses_and_grads
    / apply_param_perturbation_and_eval / apply_direction_d_and_eval call.
    Every other line -- the anchor-fixing logic, the tau_b carry-forward,
    the epsilon loops, the deterministic random-control seed construction,
    the row dict field lists -- is unchanged."""
    all_rows = []
    rng = np.random.RandomState(epoch * 1000 + subject_idx)
    tau_b_tracker = EMATauB()
    SUBJECT_SEED_BASE = epoch * 1000 + subject_idx

    grad_result = compute_losses_and_grads(
        model, image_b, mask_b, rng, tau_b_tracker, focal_fn, evidential_fn, device,
        torch_seed=SUBJECT_SEED_BASE, objective=objective,
    )
    if grad_result is None:
        print(f"  WARNING: non-finite loss at epoch {epoch} subject {subject_idx}, skipping")
        return all_rows

    dec1_params = grad_result["dec1_params"]
    g_seg_hat, g_seg_norm = normalize_flat(grad_result["g_seg"])
    g_margin_hat, g_margin_norm = normalize_flat(grad_result["g_margin"])
    g_total_hat, g_total_norm = normalize_flat(grad_result["g_total"])

    def flat_cat(g_list):
        return torch.cat([g.reshape(-1) for g in g_list])
    fs, fm, ft = flat_cat(g_seg_hat), flat_cat(g_margin_hat), flat_cat(g_total_hat)
    cos_seg_margin = float(F.cosine_similarity(fs.unsqueeze(0), fm.unsqueeze(0)).item())
    cos_seg_total = float(F.cosine_similarity(fs.unsqueeze(0), ft.unsqueeze(0)).item())
    cos_margin_total = float(F.cosine_similarity(fm.unsqueeze(0), ft.unsqueeze(0)).item())
    cos_lookup = {
        "A_seg": {"cos_with_g_seg": 1.0, "cos_with_g_margin": cos_seg_margin},
        "B_margin": {"cos_with_g_seg": cos_seg_margin, "cos_with_g_margin": 1.0},
        "C_total": {"cos_with_g_seg": cos_seg_total, "cos_with_g_margin": cos_margin_total},
    }

    TORCH_SEED_THIS_CHECKPOINT = 100000 + SUBJECT_SEED_BASE
    tau_b_eval = EMATauB()
    tau_b_eval.ema_median_abs_d = tau_b_tracker.ema_median_abs_d
    tau_b_eval.step_count = tau_b_tracker.step_count
    tau_b_value = tau_b_eval.tau_b

    model.eval()
    with torch.no_grad():
        outputs0 = model(image_b)
        probs0 = outputs0["probs"]
        dec1_0 = outputs0["dec1"]
        pred0 = (probs0 >= 0.5).float()
        dice0 = dice_score(pred0, mask_b)
        gt_flat0 = mask_b.reshape(-1)
        B, C, D, H, W = dec1_0.shape
        dec1_0_flat = dec1_0.permute(0, 2, 3, 4, 1).reshape(-1, C)
        geometry0 = compute_geometry_metric(dec1_0_flat, gt_flat0)
        focal0 = focal_fn(probs0, mask_b)
        alpha0, beta0 = outputs0["alpha"], outputs0["beta"]
        evid0 = evidential_fn(alpha0, beta0, mask_b)
        seg_loss0 = float((FOCAL_WEIGHT * focal0 + EVIDENTIAL_WEIGHT * evid0).item())

        evidence_full0 = alpha0 + beta0 - 2.0
        evidence_flat0 = evidence_full0.reshape(-1)
        boundary_flat0 = outputs0["boundary_logit"].reshape(-1)
        baseline_anchor_rng = np.random.RandomState(TORCH_SEED_THIS_CHECKPOINT)
        voxels_per_vol = D * H * W
        anchor_idx_list = []
        for b in range(B):
            vol_evidence = evidence_flat0[b * voxels_per_vol:(b + 1) * voxels_per_vol]
            local_idx = sample_stratified_anchors(vol_evidence, ANCHORS_PER_VOLUME, baseline_anchor_rng)
            anchor_idx_list.append(local_idx + b * voxels_per_vol)
        FIXED_ANCHOR_IDX = torch.cat(anchor_idx_list)

        torch.manual_seed(TORCH_SEED_THIS_CHECKPOINT)
        dummy_rng0 = np.random.RandomState(TORCH_SEED_THIS_CHECKPOINT)
        w_hat0 = objective.get_w_hat(model, device)
        margin_loss0_t, _, _ = compute_margin_loss(
            dec1_0_flat, evidence_flat0, boundary_flat0, gt_flat0,
            FIXED_ANCHOR_IDX, tau_b_value, EVIDENCE_P99_DEFAULT, objective.delta_d,
            MAX_NEGATIVES_PER_ANCHOR, dummy_rng0, device,
            w_hat=w_hat0, margin_mode=objective.margin_mode,
        )
        margin_loss0 = float(margin_loss0_t.item())

    directions_param = [("A_seg", g_seg_hat), ("B_margin", g_margin_hat), ("C_total", g_total_hat)]

    for dir_name, g_hat in directions_param:
        for eps in EPSILON_SWEEP_PARAM:
            delta = [-eps * g for g in g_hat]
            delta_norm = sum(float((d ** 2).sum()) for d in delta) ** 0.5
            result = apply_param_perturbation_and_eval(
                model, dec1_params, delta, image_b, mask_b, focal_fn, evidential_fn,
                FIXED_ANCHOR_IDX, tau_b_value, TORCH_SEED_THIS_CHECKPOINT, objective,
            )
            row = {
                "epoch": epoch, "subject_idx": subject_idx, "subject_id": subject_id, "direction": dir_name, "epsilon": eps, "epsilon_space": "parameter",
                "delta_norm": delta_norm,
                "dice_before": dice0, "dice_after": result["dice"], "delta_dice": result["dice"] - dice0,
                "seg_loss_before": seg_loss0, "seg_loss_after": result["seg_loss"], "delta_seg_loss": result["seg_loss"] - seg_loss0,
                "margin_loss_before": margin_loss0, "margin_loss_after": result["margin_loss"], "delta_margin_loss": result["margin_loss"] - margin_loss0,
                "geometry_before": geometry0, "geometry_after": result["geometry"],
                "delta_geometry": (result["geometry"] - geometry0) if (result["geometry"] is not None and geometry0 is not None) else None,
                "cos_with_g_seg": cos_lookup[dir_name]["cos_with_g_seg"],
                "cos_with_g_margin": cos_lookup[dir_name]["cos_with_g_margin"],
                "cos_with_useful": None,
            }
            all_rows.append(row)

            dir_idx = {"A_seg": 1, "B_margin": 2, "C_total": 3}[dir_name]
            fixed_seed = (dir_idx * 1_000_003 + SUBJECT_SEED_BASE * 97 + int(eps * 1000)) % (2**31)
            torch.manual_seed(fixed_seed)
            rand_flat = [torch.randn_like(p) for p in dec1_params]
            rand_total_norm = sum(float((r ** 2).sum()) for r in rand_flat) ** 0.5
            rand_delta = [r * (delta_norm / rand_total_norm) for r in rand_flat] if rand_total_norm > 1e-20 else rand_flat
            rand_result = apply_param_perturbation_and_eval(
                model, dec1_params, rand_delta, image_b, mask_b, focal_fn, evidential_fn,
                FIXED_ANCHOR_IDX, tau_b_value, TORCH_SEED_THIS_CHECKPOINT, objective,
            )
            rand_delta_norm = sum(float((d ** 2).sum()) for d in rand_delta) ** 0.5
            row_e = {
                "epoch": epoch, "subject_idx": subject_idx, "subject_id": subject_id, "direction": f"E_random_matched_{dir_name}", "epsilon": eps, "epsilon_space": "parameter",
                "delta_norm": rand_delta_norm,
                "dice_before": dice0, "dice_after": rand_result["dice"], "delta_dice": rand_result["dice"] - dice0,
                "seg_loss_before": seg_loss0, "seg_loss_after": rand_result["seg_loss"], "delta_seg_loss": rand_result["seg_loss"] - seg_loss0,
                "margin_loss_before": margin_loss0, "margin_loss_after": rand_result["margin_loss"], "delta_margin_loss": rand_result["margin_loss"] - margin_loss0,
                "geometry_before": geometry0, "geometry_after": rand_result["geometry"],
                "delta_geometry": (rand_result["geometry"] - geometry0) if (rand_result["geometry"] is not None and geometry0 is not None) else None,
                "cos_with_g_seg": None, "cos_with_g_margin": None, "cos_with_useful": None,
            }
            all_rows.append(row_e)

    for push in EPSILON_SWEEP_ACTIVATION:
        result_d = apply_direction_d_and_eval(
            model, image_b, mask_b, push, focal_fn, evidential_fn,
            FIXED_ANCHOR_IDX, tau_b_value, TORCH_SEED_THIS_CHECKPOINT, objective,
        )
        if result_d is None:
            print(f"  D_useful push={push}: degenerate volume, skipped")
            continue
        row_d = {
            "epoch": epoch, "subject_idx": subject_idx, "subject_id": subject_id, "direction": "D_useful", "epsilon": push, "epsilon_space": "activation",
            "delta_norm": None,
            "dice_before": dice0, "dice_after": result_d["dice"], "delta_dice": result_d["dice"] - dice0,
            "seg_loss_before": seg_loss0, "seg_loss_after": result_d["seg_loss"], "delta_seg_loss": result_d["seg_loss"] - seg_loss0,
            "margin_loss_before": margin_loss0, "margin_loss_after": result_d["margin_loss"], "delta_margin_loss": result_d["margin_loss"] - margin_loss0,
            "geometry_before": geometry0, "geometry_after": result_d["geometry"],
            "delta_geometry": (result_d["geometry"] - geometry0) if (result_d["geometry"] is not None and geometry0 is not None) else None,
            "cos_with_g_seg": None, "cos_with_g_margin": None, "cos_with_useful": 1.0,
        }
        all_rows.append(row_d)

        fixed_seed_d = (4 * 1_000_003 + SUBJECT_SEED_BASE * 97 + int(push * 1000)) % (2**31)
        torch.manual_seed(fixed_seed_d)
        with torch.no_grad():
            outputs_r = model(image_b)
            dec1_r = outputs_r["dec1"]
            Bc, Cc, Dc, Hc, Wc = dec1_r.shape
            dec1_r_flat = dec1_r.permute(0, 2, 3, 4, 1).reshape(-1, Cc)
            rand_dir = torch.randn_like(dec1_r_flat)
            rand_dir = rand_dir / rand_dir.norm(dim=1, keepdim=True).clamp_min(1e-8)
            dec1_pert = dec1_r_flat + push * rand_dir
            dec1_pert_vol = dec1_pert.reshape(Bc, Dc, Hc, Wc, Cc).permute(0, 4, 1, 2, 3)
            probs_r = model.seg_head(dec1_pert_vol)
            pred_r = (probs_r >= 0.5).float()
            dice_r = dice_score(pred_r, mask_b)
            gt_flat_r = mask_b.reshape(-1)
            geom_r = compute_geometry_metric(dec1_pert, gt_flat_r)
        row_er = {
            "epoch": epoch, "subject_idx": subject_idx, "subject_id": subject_id, "direction": "E_random_matched_D_useful", "epsilon": push, "epsilon_space": "activation",
            "delta_norm": None,
            "dice_before": dice0, "dice_after": dice_r, "delta_dice": dice_r - dice0,
            "seg_loss_before": seg_loss0, "seg_loss_after": None, "delta_seg_loss": None,
            "margin_loss_before": margin_loss0, "margin_loss_after": None, "delta_margin_loss": None,
            "geometry_before": geometry0, "geometry_after": geom_r,
            "delta_geometry": (geom_r - geometry0) if (geom_r is not None and geometry0 is not None) else None,
            "cos_with_g_seg": None, "cos_with_g_margin": None, "cos_with_useful": None,
        }
        all_rows.append(row_er)

    return all_rows


def run_counterfactual(checkpoint_dir, objective_config, checkpoint_epochs, n_subjects_to_use, device=None, verbose=True):
    """Public entry point, matching the interface proposed in
    PHASE_E24_GATE6_EXPERIMENTAL_SPECIFICATION.md Section 6:
    run_counterfactual(checkpoint_dir, objective_config, seed, ...).

    checkpoint_dir: Path to a directory containing epoch_{N}.pth checkpoints
                    (e.g. an E12f-style seed-0 run's checkpoints/ folder).
    objective_config: an ObjectiveConfig instance ("baseline", "task_aligned",
                       or "random_projection").
    checkpoint_epochs: list of epoch numbers to load (e.g. [5,10,15,20,25,30]).
    n_subjects_to_use: how many of the fixed validation subjects to iterate
                        (E22's own original run used 8).

    Returns the list of row dicts (same schema as E22's own results.json),
    does NOT write files -- the caller decides where/whether to save.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)

    all_rows = []
    checkpoint_dir = Path(checkpoint_dir)

    for epoch in checkpoint_epochs:
        if verbose:
            print(f"\n=== Checkpoint epoch {epoch} (objective={objective_config.mode}) ===")
        ckpt = torch.load(checkpoint_dir / f"epoch_{epoch}.pth", map_location=device, weights_only=False)
        model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(ckpt["model_state"])

        for subject_idx in range(n_subjects_to_use):
            image, mask, subject_id = val_dataset[subject_idx]
            image_b = image.unsqueeze(0).to(device)
            mask_b = mask.unsqueeze(0).to(device)
            subject_rows = run_one_subject(
                epoch, subject_idx, subject_id, model, image_b, mask_b,
                focal_fn, evidential_fn, device, objective_config,
            )
            all_rows.extend(subject_rows)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return all_rows
