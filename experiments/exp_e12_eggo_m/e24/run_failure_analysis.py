"""
Phase E24, post-H4 failure-mechanism analysis: WHY did B increase
mean_boundary_margin (H4: p=0.030 vs A) without recovering local H1
usefulness or improving Dice (H3: null)? Diagnostic instrumentation only
-- no training, no algorithmic change, per the user's explicit "don't run
another training experiment yet" instruction.

Reuses run_counterfactual.py's EXACT machinery (compute_losses_and_grads
for the real g_margin direction, the SAME B_margin descent direction H1
already characterized statistically) at H1's OWN smallest epsilon (0.25)
-- confirmed with user as the right perturbation to represent "B's real
update" for this voxel-level drill-down, since it's exactly what H1's
rho_B/per-checkpoint correlations were computed from, not a new or
different measurement object.

Answers five questions, all from ONE consolidated forward-pass-per-
checkpoint-per-subject pipeline (not five separate scripts, confirmed
with user), on the SAME 6-checkpoint x 8-subject grid as H1/H2:

1. WRONG MAGNITUDE: re-sliced from H1's own existing delta_dice-vs-epsilon
   data for B_margin (no new computation -- see report, not this script).
2. WRONG SPATIAL LOCATIONS (boundary vs interior voxels): does the
   perturbation change predicted probability differently for voxels
   adjacent to a class boundary (a neighboring voxel of the opposite
   ground-truth class) vs. interior voxels (all same-class neighbors)?
3. WRONG SIGN PER CLASS: for each voxel, does the REAL displacement
   Delta_z's projection onto w_hat have the sign that should help that
   voxel's class (tumor voxels should move toward the POSITIVE side of
   seg_head's decision axis w -- i.e. toward higher logit/higher tumor
   probability; background voxels toward the NEGATIVE side)?
4. FALSE-POSITIVE / FALSE-NEGATIVE regions: does the perturbation shift
   voxels between TP/TN/FP/FN categories, and is that shift favorable
   (FP->TN, FN->TP) or unfavorable (TP->FN, TN->FP)?
5. DOES GEOMETRY CHANGE CORRESPOND TO ERROR-BOUNDARY CORRECTION: does the
   INCREASE in mean_boundary_margin specifically concentrate on originally-
   misclassified voxels (where correction would help Dice) or on already-
   correctly-classified voxels (where it cannot help Dice, per E15's own
   causal logic -- E15's intervention only helps because SOME voxels are
   near the decision boundary; widening margin for already-confident
   correct voxels is geometrically real but functionally inert)?

Run on conditions A, B, E (all three, so B's behavior can be read against
both the baseline AND the random-projection control, not in isolation).
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
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import (  # noqa: E402
    sample_stratified_anchors, compute_margin_loss,
    ANCHORS_PER_VOLUME, MAX_NEGATIVES_PER_ANCHOR, EVIDENCE_P99_DEFAULT, EMATauB,
)

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e24"))
from run_counterfactual import ObjectiveConfig, compute_losses_and_grads, normalize_flat  # noqa: E402

CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30]
N_SUBJECTS_TO_USE = 8
EPSILON = 0.25  # H1's smallest tested epsilon -- confirmed with user as the right perturbation to analyze
FOCAL_WEIGHT = 0.5
EVIDENTIAL_WEIGHT = 0.5

GATE6_RUNS_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e24" / "gate6_runs"
OUT_DIR = Path(__file__).parent / "failure_analysis_results"

CONDITIONS = [
    ("A", "baseline", GATE6_RUNS_DIR / "A_baseline_seed0" / "checkpoints"),
    ("B", "task_aligned", GATE6_RUNS_DIR / "B_task_aligned_seed0" / "checkpoints"),
    ("E", "random_projection", GATE6_RUNS_DIR / "E_random_projection_seed0" / "checkpoints"),
]


def compute_boundary_mask(gt_vol):
    """gt_vol: (D,H,W) binary ground truth. Returns (D,H,W) bool mask:
    True for voxels with at least one 6-connected neighbor of the
    OPPOSITE class (a genuine class boundary voxel), False for interior
    voxels (all present neighbors are same-class)."""
    gt = gt_vol
    D, H, W = gt.shape
    is_boundary = torch.zeros_like(gt, dtype=torch.bool)
    shifts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for dz, dy, dx in shifts:
        shifted = torch.roll(gt, shifts=(dz, dy, dx), dims=(0, 1, 2))
        # Zero out the wrapped-around edge (roll wraps, which would create
        # a false boundary at the volume edge) -- mask out the wrapped slice.
        valid = torch.ones_like(gt, dtype=torch.bool)
        if dz != 0:
            valid[0 if dz == -1 else -1, :, :] = False
        if dy != 0:
            valid[:, 0 if dy == -1 else -1, :] = False
        if dx != 0:
            valid[:, :, 0 if dx == -1 else -1] = False
        differs = (shifted != gt) & valid
        is_boundary = is_boundary | differs
    return is_boundary


def apply_delta_get_voxel_predictions(model, delta_list, image, mask):
    """Extends run_counterfactual.py's apply_param_perturbation_and_eval
    pattern (same deepcopy/perturb/eval structure) but returns FULL
    per-voxel probs/pred_binary/dec1, not just pooled Dice -- required for
    this voxel-level analysis, not present in the original function."""
    model_copy = copy.deepcopy(model)
    with torch.no_grad():
        for p, d in zip(model_copy.dec1.parameters(), delta_list):
            p.add_(d)
    model_copy.eval()
    with torch.no_grad():
        outputs = model_copy(image)
        probs = outputs["probs"]
        dec1 = outputs["dec1"]
        pred_binary = (probs >= 0.5).float()
    del model_copy
    return probs, pred_binary, dec1


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)

    for condition_label, mode, checkpoint_dir in CONDITIONS:
        print("\n" + "#" * 70)
        print(f"# Failure analysis: condition {condition_label} (objective={mode})")
        print("#" * 70)

        objective = ObjectiveConfig(mode=mode)
        all_records = []

        for epoch in CHECKPOINT_EPOCHS:
            print(f"\n=== Checkpoint epoch {epoch} ===")
            ckpt = torch.load(checkpoint_dir / f"epoch_{epoch}.pth", map_location=device, weights_only=False)
            model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
            model.load_state_dict(ckpt["model_state"])

            for subject_idx in range(N_SUBJECTS_TO_USE):
                image, mask, subject_id = val_dataset[subject_idx]
                image_b = image.unsqueeze(0).to(device)
                mask_b = mask.unsqueeze(0).to(device)

                rng = np.random.RandomState(epoch * 1000 + subject_idx)
                tau_b_tracker = EMATauB()
                SUBJECT_SEED_BASE = epoch * 1000 + subject_idx

                # Same construction as run_counterfactual.py's run_one_subject:
                # g_margin direction for THIS condition's own objective.
                grad_result = compute_losses_and_grads(
                    model, image_b, mask_b, rng, tau_b_tracker, focal_fn, evidential_fn, device,
                    torch_seed=SUBJECT_SEED_BASE, objective=objective,
                )
                if grad_result is None:
                    continue

                g_margin_hat, g_margin_norm = normalize_flat(grad_result["g_margin"])
                delta = [-EPSILON * g for g in g_margin_hat]  # SAME descent step as H1's B_margin direction at eps=0.25

                model.eval()
                with torch.no_grad():
                    outputs0 = model(image_b)
                    probs0 = outputs0["probs"]
                    dec1_0 = outputs0["dec1"]
                    pred0 = (probs0 >= 0.5).float()

                probs1, pred1, dec1_1 = apply_delta_get_voxel_predictions(model, delta, image_b, mask_b)

                gt = mask_b.squeeze(0).squeeze(0)  # (D,H,W)
                p0 = probs0.squeeze(0).squeeze(0)
                p1 = probs1.squeeze(0).squeeze(0)
                pred0_v = pred0.squeeze(0).squeeze(0)
                pred1_v = pred1.squeeze(0).squeeze(0)

                # --- Q2: boundary vs interior ---
                boundary_mask = compute_boundary_mask(gt)
                interior_mask = ~boundary_mask
                delta_prob = (p1 - p0)  # signed change in predicted probability
                boundary_delta_prob = delta_prob[boundary_mask].mean().item() if boundary_mask.sum() > 0 else None
                interior_delta_prob = delta_prob[interior_mask].mean().item() if interior_mask.sum() > 0 else None

                # --- Q3: correct sign per class (uses REAL dec1 displacement,
                # projected onto w_hat -- w_hat recomputed fresh for THIS
                # checkpoint's model, matching H2's own convention).
                #
                # SCOPE, matching H2's own already-established decision:
                # condition A (baseline Euclidean) has NO single constrained
                # axis -- objective.get_w_hat() returns None for mode=
                # "baseline" by design (A's margin gradient is free in all
                # 32 dims, there is no natural "w" to project onto). Q3 is
                # therefore UNDEFINED for A, exactly as H2 was scoped to B/E
                # only -- not computed for A, fields left as None rather than
                # silently defaulting to some arbitrary axis. Q2/Q4/Q5 below
                # do NOT depend on w_hat and ARE computed for all three
                # conditions (dz itself, and its norm, are well-defined
                # regardless of whether a constrained axis exists). ---
                tumor_mask = gt > 0.5
                bg_mask = ~tumor_mask
                dz = (dec1_1 - dec1_0).squeeze(0).permute(1, 2, 3, 0)  # (D,H,W,32) -- well-defined for ALL conditions

                w_hat = objective.get_w_hat(model, device)
                if w_hat is not None:
                    signed_proj = torch.einsum('dhwc,c->dhw', dz, w_hat)  # (D,H,W), signed scalar per voxel
                    # "Correct" sign: tumor voxels should move POSITIVE along w
                    # (same direction seg_head reads as "more tumor-like" --
                    # DIRECTLY VERIFIED, not assumed: a standalone check
                    # confirmed pushing an embedding along +w_hat increases
                    # seg_head's output probs, since logit=w.z+b and sigmoid
                    # is monotonic increasing).
                    frac_tumor_correct_sign = (signed_proj[tumor_mask] > 0).float().mean().item() if tumor_mask.sum() > 0 else None
                    frac_bg_correct_sign = (signed_proj[bg_mask] < 0).float().mean().item() if bg_mask.sum() > 0 else None
                else:
                    frac_tumor_correct_sign = None
                    frac_bg_correct_sign = None

                # --- Q4: TP/TN/FP/FN transition ---
                tp0 = (pred0_v == 1) & (gt == 1)
                tn0 = (pred0_v == 0) & (gt == 0)
                fp0 = (pred0_v == 1) & (gt == 0)
                fn0 = (pred0_v == 0) & (gt == 1)

                fp_to_tn = int(((fp0) & (pred1_v == 0)).sum().item())  # favorable
                fn_to_tp = int(((fn0) & (pred1_v == 1)).sum().item())  # favorable
                tp_to_fn = int(((tp0) & (pred1_v == 0)).sum().item())  # unfavorable
                tn_to_fp = int(((tn0) & (pred1_v == 1)).sum().item())  # unfavorable

                # --- Q5: does INCREASED geometry concentrate on originally-misclassified voxels? ---
                misclassified0 = (pred0_v != gt)
                correct0 = (pred0_v == gt)
                # Use |delta_z| magnitude (real displacement size) as the
                # "geometric movement" signal, split by original correctness.
                dz_norm = dz.norm(dim=-1)  # (D,H,W)
                mean_dz_norm_misclassified = dz_norm[misclassified0].mean().item() if misclassified0.sum() > 0 else None
                mean_dz_norm_correct = dz_norm[correct0].mean().item() if correct0.sum() > 0 else None

                record = {
                    "condition": condition_label, "epoch": epoch, "subject_idx": subject_idx, "subject_id": subject_id,
                    "epsilon": EPSILON,
                    "n_boundary_voxels": int(boundary_mask.sum().item()),
                    "n_interior_voxels": int(interior_mask.sum().item()),
                    "boundary_delta_prob_mean": boundary_delta_prob,
                    "interior_delta_prob_mean": interior_delta_prob,
                    "n_tumor_voxels": int(tumor_mask.sum().item()),
                    "n_bg_voxels": int(bg_mask.sum().item()),
                    "frac_tumor_correct_sign": frac_tumor_correct_sign,
                    "frac_bg_correct_sign": frac_bg_correct_sign,
                    "n_fp0": int(fp0.sum().item()), "n_fn0": int(fn0.sum().item()),
                    "n_tp0": int(tp0.sum().item()), "n_tn0": int(tn0.sum().item()),
                    "fp_to_tn": fp_to_tn, "fn_to_tp": fn_to_tp,
                    "tp_to_fn": tp_to_fn, "tn_to_fp": tn_to_fp,
                    "n_misclassified0": int(misclassified0.sum().item()),
                    "n_correct0": int(correct0.sum().item()),
                    "mean_dz_norm_misclassified": mean_dz_norm_misclassified,
                    "mean_dz_norm_correct": mean_dz_norm_correct,
                }
                all_records.append(record)

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            print(f"  epoch {epoch}: {len([r for r in all_records if r['epoch']==epoch])} subject records")

        json_path = OUT_DIR / f"failure_analysis_{condition_label}.json"
        with open(json_path, "w") as f:
            json.dump(all_records, f, indent=2)
        print(f"\nSaved {len(all_records)} records to {json_path}")

    print("\n" + "=" * 70)
    print("FAILURE-MECHANISM ANALYSIS: ALL CONDITIONS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
