"""
Phase E25, C6-2, H3: error-correction ratio, reusing
experiments/exp_e12_eggo_m/e24/run_failure_analysis.py's exact Q2/Q3/Q4/Q5
machinery -- per the design doc's own stated plan ("H3's own Q5 machinery
this design's own H3 reuses directly, unchanged").

Directly adapted for C6-2's checkpoints and objective (margin_mode=
"sc_tam", m_ij=0.3089). Runs all 5 questions (not just Q5) since the
underlying pipeline computes them together in one pass (matching E24's
own "one consolidated forward-pass-per-checkpoint-per-subject pipeline,
not five separate scripts" design) -- but per the design doc's own
framing, H3's DECISIVE question for C6-2 is Q5 (does SC-TAM's geometric
movement concentrate on originally-misclassified voxels, where it could
actually help Dice, or on already-correct voxels, where it is
functionally inert).

ONE REQUIRED CORRECTION vs the original E24 script's Q3 sign convention,
made explicit rather than silently inherited: E24's Q3 was written
assuming "tumor correct sign" means POSITIVE projection onto w_hat
(signed_proj > 0) and "background correct sign" means NEGATIVE
projection (signed_proj < 0) -- this was correct for the UNSIGNED/
task-aligned case, where "more tumor-like" was defined as the positive
side of w's decision axis (verified directly in that script: "pushing an
embedding along +w_hat increases seg_head's output probs"). SC-TAM's OWN
design makes the OPPOSITE prediction for its real displacement (verified
directly in H2 above, 24/24 checkpoint-batches confirm this holds in
practice): tumor voxels should move toward -w_hat, background toward
+w_hat (see compute_margin_loss's sc_tam docstring: "driving the tumor
voxel toward -w_hat and the background voxel toward +w_hat"). This is
NOT a contradiction -- SC-TAM's SIGNED distance is defined as
(z_tumor.w_hat - z_bg.w_hat), and its gradient pushes to WIDEN this gap
by making it MORE NEGATIVE from tumor's contribution and MORE POSITIVE
from bg's, which is a genuinely different convention from task_aligned's
absolute-distance hinge. This script's Q3 therefore flips the sign
convention to match SC-TAM's own design (tumor correct = negative
projection, background correct = positive projection), matching exactly
what H2 already measured and confirmed -- not a new, unverified
assumption.
"""
import sys
import copy
import json
from pathlib import Path

import numpy as np
import torch

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
from run_failure_analysis import compute_boundary_mask, apply_delta_get_voxel_predictions  # noqa: E402 -- REUSED UNCHANGED

CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30]
N_SUBJECTS_TO_USE = 8
EPSILON = 0.25  # H1's smallest tested epsilon -- SAME convention as E24's own failure analysis
FOCAL_WEIGHT = 0.5
EVIDENTIAL_WEIGHT = 0.5

C62_CHECKPOINT_DIR = (project_root / "experiments" / "exp_e12_eggo_m" / "e25" /
                       "c62_runs" / "C62_sc_tam_seed0" / "checkpoints")
OUT_DIR = Path(__file__).parent / "failure_analysis_results"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)

    print("\n" + "#" * 70)
    print("# H3/failure analysis: condition C6-2 (objective=sc_tam)")
    print("#" * 70)

    objective = ObjectiveConfig(mode="sc_tam")
    all_records = []

    for epoch in CHECKPOINT_EPOCHS:
        print(f"\n=== Checkpoint epoch {epoch} ===")
        ckpt = torch.load(C62_CHECKPOINT_DIR / f"epoch_{epoch}.pth", map_location=device, weights_only=False)
        model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(ckpt["model_state"])

        for subject_idx in range(N_SUBJECTS_TO_USE):
            image, mask, subject_id = val_dataset[subject_idx]
            image_b = image.unsqueeze(0).to(device)
            mask_b = mask.unsqueeze(0).to(device)

            rng = np.random.RandomState(epoch * 1000 + subject_idx)
            tau_b_tracker = EMATauB()
            SUBJECT_SEED_BASE = epoch * 1000 + subject_idx

            grad_result = compute_losses_and_grads(
                model, image_b, mask_b, rng, tau_b_tracker, focal_fn, evidential_fn, device,
                torch_seed=SUBJECT_SEED_BASE, objective=objective,
            )
            if grad_result is None:
                continue

            g_margin_hat, g_margin_norm = normalize_flat(grad_result["g_margin"])
            delta = [-EPSILON * g for g in g_margin_hat]  # SAME descent step convention as H1/E24's failure analysis

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
            delta_prob = (p1 - p0)
            boundary_delta_prob = delta_prob[boundary_mask].mean().item() if boundary_mask.sum() > 0 else None
            interior_delta_prob = delta_prob[interior_mask].mean().item() if interior_mask.sum() > 0 else None

            # --- Q3: correct sign per class, SC-TAM'S OWN CONVENTION
            # (flipped from E24's task_aligned convention -- see module
            # docstring): tumor correct = NEGATIVE projection onto w_hat,
            # background correct = POSITIVE projection. ---
            tumor_mask = gt > 0.5
            bg_mask = ~tumor_mask
            dz = (dec1_1 - dec1_0).squeeze(0).permute(1, 2, 3, 0)  # (D,H,W,32)

            w_hat = objective.get_w_hat(model, device)
            assert w_hat is not None, "sc_tam mode must always have a w_hat -- unlike baseline, this is not expected to be None"
            signed_proj = torch.einsum('dhwc,c->dhw', dz, w_hat)
            frac_tumor_correct_sign = (signed_proj[tumor_mask] < 0).float().mean().item() if tumor_mask.sum() > 0 else None  # SC-TAM: tumor -> -w_hat
            frac_bg_correct_sign = (signed_proj[bg_mask] > 0).float().mean().item() if bg_mask.sum() > 0 else None            # SC-TAM: bg -> +w_hat

            # --- Q4: TP/TN/FP/FN transition ---
            tp0 = (pred0_v == 1) & (gt == 1)
            tn0 = (pred0_v == 0) & (gt == 0)
            fp0 = (pred0_v == 1) & (gt == 0)
            fn0 = (pred0_v == 0) & (gt == 1)

            fp_to_tn = int(((fp0) & (pred1_v == 0)).sum().item())
            fn_to_tp = int(((fn0) & (pred1_v == 1)).sum().item())
            tp_to_fn = int(((tp0) & (pred1_v == 0)).sum().item())
            tn_to_fp = int(((tn0) & (pred1_v == 1)).sum().item())

            # --- Q5 (THE DECISIVE QUESTION FOR H3): does geometric
            # movement concentrate on originally-misclassified voxels? ---
            misclassified0 = (pred0_v != gt)
            correct0 = (pred0_v == gt)
            dz_norm = dz.norm(dim=-1)
            mean_dz_norm_misclassified = dz_norm[misclassified0].mean().item() if misclassified0.sum() > 0 else None
            mean_dz_norm_correct = dz_norm[correct0].mean().item() if correct0.sum() > 0 else None

            record = {
                "condition": "C6-2", "epoch": epoch, "subject_idx": subject_idx, "subject_id": subject_id,
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

    json_path = OUT_DIR / "failure_analysis_C62.json"
    with open(json_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nSaved {len(all_records)} records to {json_path}")

    print("\n" + "=" * 70)
    print("H3/FAILURE ANALYSIS: C6-2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
