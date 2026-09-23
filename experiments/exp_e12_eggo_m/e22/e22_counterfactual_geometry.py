"""
Phase E22: Counterfactual Objective-Geometry Test -- diagnostic/
counterfactual intervention on existing trained checkpoints, no training,
no algorithm/hyperparameter changes. Direct follow-up to E21/E21.5: E21
found the real margin-induced update moves dec1 AWAY from E15's
empirically-useful direction (mean cosine consistently negative across
all 6 checkpoints); E21.5 independently re-verified this under two
alternate aggregations and found it strengthens, not weakens. E22 asks
the more direct question those phases could not: when we actually take a
LOCAL STEP that minimizes L_margin (not just observe the direction it
points), does Dice go up, down, or stay flat -- compared against the same
question for L_seg, L_total, E15's own validated useful direction, and a
matched-norm random control.

Scope decisions confirmed with user before writing this script (departures
from the original spec, each resolved because the spec's equations
contained a type/units ambiguity that would otherwise produce a silently
wrong result):

1. PARAMETER SCOPE (theta): Directions A/B/C/E perturb dec1's PARAMETERS
   ONLY (matching E19/E20/E21's own scope -- the direct site of L_margin
   and the shared trunk into all three heads), not the whole model. Keeps
   this phase's ||delta_theta|| numbers directly comparable to E20's real
   replay norms (0.16-1.25 across all 6 checkpoints), which is also the
   basis for this phase's own epsilon-sweep calibration (see EPSILON_SWEEP
   below).

2. DIRECTION D IS A DIFFERENT KIND OF INTERVENTION, DELIBERATELY: E15's
   d_useful lives in dec1's ACTIVATION space (per-voxel, shape
   (N_voxels,32)), not parameter space -- there is no way to add an
   activation-space vector to a parameter tensor; the spec's literal
   "theta' = theta + eps*d_useful" is a type mismatch once theta means
   "dec1's parameters" (per decision 1). Rather than inventing new,
   unvalidated machinery to force D into parameter space (e.g. some
   least-squares parameter update whose forward effect approximates
   d_useful), D is applied EXACTLY as E15 validated it: directly to dec1's
   OUTPUT activation (dec1_perturbed = dec1_original + eps*d_useful),
   feeding straight into seg_head, bypassing the rest of the forward pass
   entirely -- byte-identical to compute_manipulated_dice() in
   e15_decoder_sensitivity.py. This means D is NOT mechanistically
   comparable to A/B/C/E (activation push vs. parameter step) -- it IS
   outcome-comparable (Dice, L_seg, L_margin, geometry all still
   measurable for all 5 directions on the same footing). This asymmetry
   is deliberate and is called out explicitly in every place D's numbers
   are reported, not smoothed over.

3. EPSILON UNITS: A/B/C/E's epsilon is a PARAMETER-space norm applied to
   dec1's weights (confirmed well-matched to E20's real observed
   ||delta_theta|| range of 0.16-1.25 by direct inspection of
   e20_dec1_update_decomposition_results/e20_results.json before this
   script was written -- the suggested sweep {0.25,0.5,1.0,2.0} spans
   below-typical to ~2x the largest real update seen). Direction D uses
   E15's OWN calibrated activation-space push units instead (a subset of
   E15's original {0,1,2,4,8,14,20,28}, chosen for a comparable NUMBER of
   sweep points, not converted to the parameter-space numbers) -- reported
   in the SAME results table under the same "epsilon" column, but readers
   must not compare a parameter-space epsilon=1.0 to an activation-space
   epsilon=1.0 as if they were the same intervention size. Documented here
   and again in the report.

4. DATA SOURCE FOR DICE/GEOMETRY: E15's fixed 20-subject VALIDATION set
   (BraTSDataset(split="val")), not E20/E21's training batches -- required
   for "Dice" to mean what it meant in E15 (E20/E21 never computed Dice at
   all, only losses/gradients, since they used training data). g_seg/
   g_margin/g_total (needed for Directions A/B/C) are computed ON this
   same validation data, a straightforward extension of E20/E21's
   gradient-computation pattern to a different data source, consistent
   with E15's own precedent of running real backward passes on validation
   subjects when needed (E15's own Jacobian sensitivity pass did exactly
   this).

5. BATCHNORM MODE: gradients (g_seg/g_margin/g_total, for constructing
   Directions A/B/C) are computed with the model in .train() mode, matching
   E14/E19/E20/E21's own established and audited convention (E12e lesson:
   eval-mode gradients meaningfully differ from what real training
   experienced). The actual Dice-scoring forward pass (before AND after
   each intervention) uses .eval() mode, matching E15's own established
   convention (deployed/real inference behavior, running-stat BN). These
   are two deliberately different modes for two different purposes within
   the same script -- never blurred into one "mode" silently.

6. GEOMETRIC METRIC ("Geometry" column): reuses the EXACT formula already
   established in analyze_eggo_m_checkpoints_v2.py's own "mean_boundary_
   margin" computation (traced directly, not guessed, before writing this
   script) -- mean Euclidean distance between MAX_MARGIN_PAIRS randomly-
   sampled (fixed seeds 1/2) same-volume tumor/background dec1 embedding
   pairs. This is explicitly the quantity L_margin's own hinge term
   operates on (same distance metric, same class-conditional pairing
   logic), not a new invented metric, per the spec's Section 7 instruction
   not to introduce a new geometric metric unless necessary.

Method, per checkpoint x direction x epsilon:
  1. Load the real checkpoint's model_state onto a fresh UNet3D_v2.
  2. Compute g_seg, g_margin, g_total w.r.t. dec1's parameters, in
     .train() mode, on a REAL VALIDATION BATCH (single volume, batch=1,
     matching E15's per-volume validation loop) -- using the EXACT loss
     formulas from train_eggo_m.py (same FOCAL_WEIGHT/EVIDENTIAL_WEIGHT,
     same compute_margin_loss call, same stratified anchor sampling, same
     lambda_margin=0.1 for total).
  3. For A/B/C: normalize the relevant gradient, apply
     theta' = theta - eps * g_hat (a DESCENT step -- consistent with the
     spec's own equations, all written as theta - eps*grad/||grad||,
     i.e. minimizing that objective locally) to a DEEP-COPIED model's
     dec1 parameters, switch to .eval(), forward-pass, score Dice/L_seg/
     L_margin/geometry.
  4. For D: forward the ORIGINAL (unperturbed) model once to get
     dec1_original and this subject's d_useful (E15's exact formula),
     then apply dec1_perturbed = dec1_original + eps*d_useful directly to
     seg_head (bypassing the rest of the decoder, exactly E15's method)
     for Dice, AND separately feed dec1_perturbed through the REST of the
     forward computation (evidential_head only, since boundary_head reads
     dec1.detach() and is not needed here) to get L_margin/geometry at
     the perturbed activation -- L_seg is computed from the same probs
     used for Dice.
  5. For E: a random parameter-space (for A/B/C's matched controls) or
     activation-space (for D's matched control) perturbation of the SAME
     norm as the corresponding real direction at that epsilon, fixed seed.
  6. Record everything to results.json / results.csv per the spec's
     Section 11 field list.

Deliverables: PHASE_E22_COUNTERFACTUAL_OBJECTIVE_GEOMETRY.md (written
separately), results.json, results.csv.
"""
import sys
import copy
import json
import csv
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

CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30]
SEED_DIR = "e12f_pilot_calibrated_seed0"
N_VAL_SUBJECTS = 20  # same fixed subset as E15 (upper bound on the dataset's own fixed val split)
N_SUBJECTS_TO_USE = 8  # subset of N_VAL_SUBJECTS actually iterated per checkpoint, for Section 13's
                        # subject-level (not single-anecdote) Dice statistics -- 8 chosen as a
                        # runtime/statistical-power balance; not all 20 (would ~2.5x total runtime
                        # for a per-checkpoint grid already 24 rows/direction-variant x 6 checkpoints)
                        # but well beyond the original single-subject smoke-tested design.
FOCAL_WEIGHT = 0.5
EVIDENTIAL_WEIGHT = 0.5
LAMBDA_MARGIN_CALIBRATED = 0.1

# Parameter-space epsilon sweep for Directions A/B/C/E, confirmed
# well-matched to E20's real ||delta_theta|| range (0.16-1.25) by direct
# inspection before this script was written.
EPSILON_SWEEP_PARAM = [0.25, 0.5, 1.0, 2.0]

# Activation-space push units for Direction D, subset of E15's own
# calibrated {0,1,2,4,8,14,20,28} -- 4 points for parity with the
# parameter-space sweep's 4 points, spanning "small" to "E15's realistic
# ceiling" (14.0 = E12f's full observed mean_boundary_margin dynamic
# range, the exact value that produced E15's headline +0.039 Dice result).
EPSILON_SWEEP_ACTIVATION = [1.0, 2.0, 4.0, 14.0]

MAX_MARGIN_PAIRS = 5000  # matches analyze_eggo_m_checkpoints_v2.py exactly


def dice_score(pred_binary, gt):
    tp = (pred_binary * gt).sum()
    return (2 * tp / (pred_binary.sum() + gt.sum() + 1e-6)).item()


def compute_geometry_metric(dec1_flat, gt_flat, seed_t=1, seed_b=2):
    """EXACT reproduction of analyze_eggo_m_checkpoints_v2.py's
    mean_boundary_margin: mean Euclidean distance between MAX_MARGIN_PAIRS
    randomly-sampled (fixed seeds) same-volume tumor/background dec1
    embeddings -- the quantity L_margin's own hinge term operates on."""
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
    """Byte-identical to E15's / E21's direction formula."""
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


def compute_losses_and_grads(model, image, mask, rng, tau_b_tracker, focal_fn, evidential_fn, device, torch_seed):
    """.train()-mode forward pass + independent grad w.r.t. dec1's
    parameters for seg_loss, margin_loss (unweighted), and total_loss
    (seg + lambda*margin) -- matches E14/E19/E20/E21's exact loss
    construction and anchor-sampling convention.

    torch_seed: fixed via torch.manual_seed() immediately before the
    compute_margin_loss call. REQUIRED because compute_margin_loss's
    negative-pair sampling uses torch.randint on the GLOBAL torch RNG,
    NOT the passed-in numpy `rng` object (confirmed directly: repeated
    calls at the IDENTICAL point in parameter space, same numpy rng seed,
    produced margin_loss varying by ~10-20% run to run with no seed
    control -- this is the E21.5-flagged "rng parameter unused for
    negative sampling" issue, rated cosmetic there because no prior phase
    needed single-evaluation reproducibility at this precision; E22's
    before/after perturbation comparison DOES need it, since the
    first-order signal being measured is smaller than that noise floor at
    small epsilon). Pinning torch.manual_seed() to the SAME value for
    every before/after evaluation at a given checkpoint isolates the
    effect of the perturbation from the effect of a freshly-resampled
    negative set."""
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
    margin_loss, _, margin_diag = compute_margin_loss(
        dec1_perm, evidence_flat, boundary_flat, gt_flat,
        anchor_idx, current_tau_b, EVIDENCE_P99_DEFAULT, DELTA_D_CALIBRATED,
        MAX_NEGATIVES_PER_ANCHOR, rng, device,
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
    total_norm = sum(float((g ** 2).sum()) for g in grad_list) ** 0.5
    if total_norm < 1e-20:
        return [torch.zeros_like(g) for g in grad_list], 0.0
    return [g / total_norm for g in grad_list], total_norm


def apply_param_perturbation_and_eval(model, dec1_params_template, delta_list, image, mask, focal_fn, evidential_fn, fixed_anchor_idx, tau_b_value, torch_seed):
    """Deep-copies model, adds delta_list to the copy's dec1 params
    in-place, switches to .eval(), forward-passes, scores Dice + losses +
    geometry at the perturbed state. Original model untouched.

    CORRECTNESS-CRITICAL: fixed_anchor_idx (computed ONCE at the
    unperturbed baseline state) and torch_seed (pinned via
    torch.manual_seed() before the compute_margin_loss call) are both
    REQUIRED for a valid before/after comparison. Discovered during
    smoke-testing: (1) resampling stratified anchors fresh at each
    perturbed state means "before" and "after" margin_loss are measured
    at DIFFERENT physical voxels, conflating the perturbation's effect
    with the effect of comparing different anchors; (2)
    compute_margin_loss's negative-pair sampling uses the GLOBAL torch
    RNG (torch.randint), NOT the numpy `rng` argument it accepts --
    verified directly: repeated calls at the IDENTICAL point in parameter
    space varied by ~10-20% with no seed control, a noise floor LARGER
    than the true first-order signal at small epsilon. Both are fixed
    here by reusing the SAME anchor_idx and the SAME torch.manual_seed()
    value for every direction/epsilon at a given checkpoint, isolating
    the perturbation's effect from resampling noise."""
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

        # L_margin at the perturbed state, eval-mode forward, SAME FIXED
        # anchor set as the baseline (not a fresh resample) -- see
        # docstring. eval-mode margin_loss is a DIAGNOSTIC readout here,
        # not a training signal, consistent with using the checkpoint's
        # current tau_b, not updating it.
        evidence_full = alpha + beta - 2.0
        boundary_logit = outputs["boundary_logit"]
        evidence_flat = evidence_full.reshape(-1)
        boundary_flat = boundary_logit.reshape(-1)
        torch.manual_seed(torch_seed)
        dummy_rng = np.random.RandomState(torch_seed)  # unused by negative sampling internally, but the signature requires an rng object
        margin_loss, _, margin_diag = compute_margin_loss(
            dec1_flat, evidence_flat, boundary_flat, gt_flat,
            fixed_anchor_idx, tau_b_value, EVIDENCE_P99_DEFAULT, DELTA_D_CALIBRATED,
            MAX_NEGATIVES_PER_ANCHOR, dummy_rng, dec1.device,
        )
        margin_loss = float(margin_loss.item())

    del model_copy
    return {"dice": dice, "seg_loss": seg_loss, "margin_loss": margin_loss, "geometry": geometry}


def apply_direction_d_and_eval(model, image, mask, push_units, focal_fn, evidential_fn, fixed_anchor_idx, tau_b_value, torch_seed):
    """Direction D: activation-space push, exactly E15's mechanism.
    dec1_perturbed = dec1_original + push_units * d_useful, fed directly
    into seg_head for Dice (E15's exact method), and through
    evidential_head for L_seg's evidential term + geometry/L_margin at
    the perturbed activation.

    Uses the SAME fixed_anchor_idx/torch_seed convention as
    apply_param_perturbation_and_eval, for the same reason (see that
    function's docstring) -- a valid before/after margin_loss comparison
    requires the same physical anchor voxels and the same negative-sample
    draw, not fresh resamples at each perturbed state."""
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
        # boundary_logit is undefined at the perturbed activation since
        # boundary_head reads dec1.detach() from the ORIGINAL forward
        # pass only -- reuse the ORIGINAL boundary_logit (pre-perturbation)
        # for the anchor-selection U_hat/B weighting, consistent with
        # boundary_head never seeing this perturbation in the first place
        # (structurally correct per the E21.5 audit's confirmed autograd
        # graph: boundary_head is detached from dec1 entirely).
        boundary_flat = outputs["boundary_logit"].reshape(-1)
        torch.manual_seed(torch_seed)
        dummy_rng = np.random.RandomState(torch_seed)
        margin_loss, _, margin_diag = compute_margin_loss(
            dec1_perturbed_flat, evidence_flat, boundary_flat, gt_flat,
            fixed_anchor_idx, tau_b_value, EVIDENCE_P99_DEFAULT, DELTA_D_CALIBRATED,
            MAX_NEGATIVES_PER_ANCHOR, dummy_rng, dec1.device,
        )
        margin_loss = float(margin_loss.item())

    return {"dice": dice, "seg_loss": seg_loss, "margin_loss": margin_loss, "geometry": geometry, "d_useful": d_useful}


def main():
    exp_dir = Path(__file__).parent
    ckpt_dir = project_root / "experiments" / "exp_e12_eggo_m" / SEED_DIR / "checkpoints"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_use = min(N_VAL_SUBJECTS, len(val_dataset))
    print(f"Using {n_use} fixed validation subjects (same set as E15)")

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)

    all_rows = []

    for epoch in CHECKPOINT_EPOCHS:
        print(f"\n=== Checkpoint epoch {epoch} ===")
        ckpt = torch.load(ckpt_dir / f"epoch_{epoch}.pth", map_location=device, weights_only=False)
        model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(ckpt["model_state"])

        for subject_idx in range(N_SUBJECTS_TO_USE):
            image, mask, subject_id = val_dataset[subject_idx]
            image_b = image.unsqueeze(0).to(device)
            mask_b = mask.unsqueeze(0).to(device)
            print(f"\n  --- Subject {subject_idx} ({subject_id}) ---")
            subject_rows = run_one_subject(
                epoch, subject_idx, subject_id, model, image_b, mask_b,
                focal_fn, evidential_fn, device,
            )
            all_rows.extend(subject_rows)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- Save raw data ---
    with open(exp_dir / "results.json", "w") as f:
        json.dump(all_rows, f, indent=2)

    fieldnames = list(all_rows[0].keys())
    with open(exp_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows to {exp_dir / 'results.json'} and results.csv")


def run_one_subject(epoch, subject_idx, subject_id, model, image_b, mask_b, focal_fn, evidential_fn, device):
    """Runs the full direction x epsilon grid (A/B/C/D/E, all epsilons)
    for ONE validation subject at ONE checkpoint. Extracted from the
    original single-subject main() loop body unchanged (subject-loop
    added afterward per user decision -- Section 13's own requirement
    that Dice inference use subject/volume as the unit, not a single
    anecdote). Returns a list of result-row dicts, each tagged with
    subject_id."""
    all_rows = []
    rng = np.random.RandomState(epoch * 1000 + subject_idx)  # subject-specific but reproducible
    tau_b_tracker = EMATauB()
    SUBJECT_SEED_BASE = epoch * 1000 + subject_idx  # unique, reproducible per (epoch, subject) pair

    if True:
        grad_result = compute_losses_and_grads(
            model, image_b, mask_b, rng, tau_b_tracker, focal_fn, evidential_fn, device,
            torch_seed=SUBJECT_SEED_BASE,  # train-mode gradient-construction pass; independent of the eval-mode baseline's own fixed seed below
        )
        if grad_result is None:
            print(f"  WARNING: non-finite loss at epoch {epoch} subject {subject_idx}, skipping")
            return all_rows

        dec1_params = grad_result["dec1_params"]
        g_seg_hat, g_seg_norm = normalize_flat(grad_result["g_seg"])
        g_margin_hat, g_margin_norm = normalize_flat(grad_result["g_margin"])
        g_total_hat, g_total_norm = normalize_flat(grad_result["g_total"])
        print(f"  ||g_seg||={g_seg_norm:.4e} ||g_margin||={g_margin_norm:.4e} ||g_total||={g_total_norm:.4e}")

        # Pairwise cosines between the three PARAMETER-SPACE gradient
        # directions -- all comparable (same space, dec1 params). This is
        # the well-defined subset of Section 7/9's requested cosine
        # metrics: cos(intervention_direction, g_seg) and
        # cos(intervention_direction, g_margin) ARE meaningful for
        # Directions A/B/C (all parameter-space), computed here once per
        # checkpoint and attached to each row below. cos(A/B/C, d_useful)
        # is NOT computed -- d_useful lives in ACTIVATION space (per E21.5's
        # confirmed geometry finding), so a raw cosine against a
        # parameter-space delta would be a cross-space comparison with no
        # defined meaning (the same type mismatch already resolved for
        # Direction D's own construction, see module docstring decision 2).
        # This gap is reported explicitly, not silently left as a random
        # None -- see the report's Direction Comparison section.
        def flat_cat(g_list):
            return torch.cat([g.reshape(-1) for g in g_list])
        fs, fm, ft = flat_cat(g_seg_hat), flat_cat(g_margin_hat), flat_cat(g_total_hat)
        cos_seg_margin = float(F.cosine_similarity(fs.unsqueeze(0), fm.unsqueeze(0)).item())
        cos_seg_total = float(F.cosine_similarity(fs.unsqueeze(0), ft.unsqueeze(0)).item())
        cos_margin_total = float(F.cosine_similarity(fm.unsqueeze(0), ft.unsqueeze(0)).item())
        print(f"  cos(g_seg,g_margin)={cos_seg_margin:+.4f} cos(g_seg,g_total)={cos_seg_total:+.4f} cos(g_margin,g_total)={cos_margin_total:+.4f}")
        cos_lookup = {
            "A_seg": {"cos_with_g_seg": 1.0, "cos_with_g_margin": cos_seg_margin},
            "B_margin": {"cos_with_g_seg": cos_seg_margin, "cos_with_g_margin": 1.0},
            "C_total": {"cos_with_g_seg": cos_seg_total, "cos_with_g_margin": cos_margin_total},
        }

        # Baseline (epsilon=0) forward pass, EVAL mode -- this is the
        # reference every perturbed evaluation is compared against, so it
        # MUST use eval mode (matching apply_param_perturbation_and_eval/
        # apply_direction_d_and_eval's own mode) and a FIXED anchor set +
        # torch_seed that will be reused, unchanged, for every subsequent
        # perturbation at this checkpoint (see apply_param_perturbation_
        # and_eval's docstring for why this is correctness-critical, not
        # a style choice). This is intentionally a SEPARATE anchor sample
        # from the one inside compute_losses_and_grads (which is a
        # train-mode pass solely for constructing the gradient
        # DIRECTIONS, a different question from "what is the Dice/loss
        # baseline to compare perturbations against").
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
            FIXED_ANCHOR_IDX = torch.cat(anchor_idx_list)  # reused, unchanged, for EVERY perturbation below

            torch.manual_seed(TORCH_SEED_THIS_CHECKPOINT)
            dummy_rng0 = np.random.RandomState(TORCH_SEED_THIS_CHECKPOINT)
            margin_loss0_t, _, _ = compute_margin_loss(
                dec1_0_flat, evidence_flat0, boundary_flat0, gt_flat0,
                FIXED_ANCHOR_IDX, tau_b_value, EVIDENCE_P99_DEFAULT, DELTA_D_CALIBRATED,
                MAX_NEGATIVES_PER_ANCHOR, dummy_rng0, device,
            )
            margin_loss0 = float(margin_loss0_t.item())
        print(f"  Baseline: Dice={dice0:.4f} L_seg={seg_loss0:.4f} L_margin={margin_loss0:.4f} geometry={geometry0:.4f}")

        directions_param = [("A_seg", g_seg_hat), ("B_margin", g_margin_hat), ("C_total", g_total_hat)]

        for dir_name, g_hat in directions_param:
            for eps in EPSILON_SWEEP_PARAM:
                delta = [-eps * g for g in g_hat]  # descent step, per spec's theta - eps*grad/||grad||
                delta_norm = sum(float((d ** 2).sum()) for d in delta) ** 0.5
                result = apply_param_perturbation_and_eval(
                    model, dec1_params, delta, image_b, mask_b, focal_fn, evidential_fn,
                    FIXED_ANCHOR_IDX, tau_b_value, TORCH_SEED_THIS_CHECKPOINT,
                )
                cos_useful = None  # computed below after d_useful is available; parameter-space delta vs activation-space d_useful needs a forward-pass-derived Delta_z, added separately if needed
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
                    "cos_with_useful": None,  # cross-space (parameter vs activation) -- not defined, see module docstring
                }
                all_rows.append(row)
                print(f"  {dir_name:10s} eps={eps:.2f} ||delta||={delta_norm:.4f}  "
                      f"dDice={row['delta_dice']:+.4f} dL_seg={row['delta_seg_loss']:+.4f} "
                      f"dL_margin={row['delta_margin_loss']:+.4f} dGeom={row['delta_geometry']:+.4f}" if row['delta_geometry'] is not None else
                      f"  {dir_name:10s} eps={eps:.2f} ||delta||={delta_norm:.4f}  dDice={row['delta_dice']:+.4f}")

                # Direction E: matched-norm random control for this (direction,epsilon).
                # NOTE: Python's built-in hash() is RANDOMIZED per-process for
                # strings (PYTHONHASHSEED), confirmed directly (hash() on the
                # identical tuple gave two different values across two
                # process launches) -- using it here would silently break
                # Section 12 item 2/3's reproducibility requirement for the
                # random control specifically. Fixed with a deterministic
                # integer encoding instead (direction index * large prime +
                # epoch/epsilon-derived integers), stable across processes/
                # runs/machines.
                dir_idx = {"A_seg": 1, "B_margin": 2, "C_total": 3}[dir_name]
                fixed_seed = (dir_idx * 1_000_003 + SUBJECT_SEED_BASE * 97 + int(eps * 1000)) % (2**31)
                torch.manual_seed(fixed_seed)
                rand_flat = [torch.randn_like(p) for p in dec1_params]
                rand_total_norm = sum(float((r ** 2).sum()) for r in rand_flat) ** 0.5
                rand_delta = [r * (delta_norm / rand_total_norm) for r in rand_flat] if rand_total_norm > 1e-20 else rand_flat
                rand_result = apply_param_perturbation_and_eval(
                    model, dec1_params, rand_delta, image_b, mask_b, focal_fn, evidential_fn,
                    FIXED_ANCHOR_IDX, tau_b_value, TORCH_SEED_THIS_CHECKPOINT,
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

        # Direction D: activation-space push, E15's own mechanism
        for push in EPSILON_SWEEP_ACTIVATION:
            result_d = apply_direction_d_and_eval(
                model, image_b, mask_b, push, focal_fn, evidential_fn,
                FIXED_ANCHOR_IDX, tau_b_value, TORCH_SEED_THIS_CHECKPOINT,
            )
            if result_d is None:
                print(f"  D_useful push={push}: degenerate volume, skipped")
                continue
            row_d = {
                "epoch": epoch, "subject_idx": subject_idx, "subject_id": subject_id, "direction": "D_useful", "epsilon": push, "epsilon_space": "activation",
                "delta_norm": None,  # not directly comparable to parameter-space delta_norm
                "dice_before": dice0, "dice_after": result_d["dice"], "delta_dice": result_d["dice"] - dice0,
                "seg_loss_before": seg_loss0, "seg_loss_after": result_d["seg_loss"], "delta_seg_loss": result_d["seg_loss"] - seg_loss0,
                "margin_loss_before": margin_loss0, "margin_loss_after": result_d["margin_loss"], "delta_margin_loss": result_d["margin_loss"] - margin_loss0,
                "geometry_before": geometry0, "geometry_after": result_d["geometry"],
                "delta_geometry": (result_d["geometry"] - geometry0) if (result_d["geometry"] is not None and geometry0 is not None) else None,
                "cos_with_g_seg": None, "cos_with_g_margin": None, "cos_with_useful": 1.0,  # trivially 1.0, D is applied AS d_useful
            }
            all_rows.append(row_d)
            print(f"  D_useful   push={push:.2f} (activation-space)  "
                  f"dDice={row_d['delta_dice']:+.4f} dL_seg={row_d['delta_seg_loss']:+.4f} "
                  f"dL_margin={row_d['delta_margin_loss']:+.4f}")

            # Matched-norm random activation-space control for D. Same
            # PYTHONHASHSEED fix as Direction E's param-space control above.
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


if __name__ == "__main__":
    main()
