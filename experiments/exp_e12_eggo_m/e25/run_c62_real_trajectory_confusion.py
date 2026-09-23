"""
Phase E25, C6-2.8 (real-trajectory version, per the user's explicit
redirect): does SC-TAM produce a net improvement or degradation in the
TP/FN/FP/TN populations across the REAL, actual C6-2 training
trajectory -- not a one-shot L_margin-only 15-step replay extrapolated
linearly across the whole population (the prior population-weighted
accounting, PHASE_E25_C62_POPULATION_WEIGHTED_ACCOUNTING.md, found that
extrapolation predicts a catastrophic Dice collapse that did NOT happen
in real training, and flagged this as evidence the one-shot mechanism
cannot be treated as a linear proxy for cumulative L_seg+L_margin
training dynamics).

Method: for BOTH condition A (baseline, no margin term) and C6-2
(SC-TAM), at every pair of CONSECUTIVE REAL SAVED CHECKPOINTS (1->5,
5->10, 10->15, 15->20, 20->25, 25->30 -- the actual checkpoint schedule
both runs share, not a synthetic replay), load the two real trained
models, run the SAME fixed validation subjects through BOTH, and
directly compare each voxel's TP/FN/FP/TN state at checkpoint t vs
t+1 -- NO perturbation, NO gradient replay, just two real forward
passes through two real trained models. This gives the REAL, cumulative
transition matrix the actual optimizer (AdamW, real L_seg+mu*L_boundary+
lambda*L_margin, real data, real 5-epoch gap between saves) produced,
sidestepping the one-shot-extrapolation problem entirely.

Full voxel population (not category-balanced subsampling) is used per
subject-checkpoint-pair -- the whole point of this experiment is to
measure the TRUE population-level net effect, which category-balanced
subsampling (used in C6-2.6) cannot answer directly (that experiment
measured PER-CATEGORY transition PROBABILITIES; this experiment measures
the REAL population-level transition COUNTS/matrix directly, no
extrapolation needed).
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

CHECKPOINT_PAIRS = [(1, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]
N_SUBJECTS = 8  # matches H1's own established scope for tractability

GATE6_A_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e24" / "gate6_runs" / "A_baseline_seed0" / "checkpoints"
C62_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "c62_runs" / "C62_sc_tam_seed0" / "checkpoints"
OUT_DIR = Path(__file__).parent / "real_trajectory_confusion_results"

CONDITIONS = [("A_baseline", GATE6_A_DIR), ("C62_sc_tam", C62_DIR)]


def classify_state(gt, pred):
    """Returns a (N,) int8 tensor: 0=TN, 1=TP, 2=FP, 3=FN."""
    gt_pos = gt > 0.5
    pred_pos = pred > 0.5
    tp = gt_pos & pred_pos
    tn = (~gt_pos) & (~pred_pos)
    fp = (~gt_pos) & pred_pos
    fn = gt_pos & (~pred_pos)
    cat = torch.zeros_like(gt, dtype=torch.int8)
    cat[tp] = 1
    cat[fp] = 2
    cat[fn] = 3
    return cat


STATE_NAMES = {0: "TN", 1: "TP", 2: "FP", 3: "FN"}


def get_predictions(ckpt_path, images, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()  # REAL prediction, matching how Dice/H4 itself is scored -- NOT train() mode (unlike the one-shot mechanism experiments, this is measuring actual deployed-model behavior)
    with torch.no_grad():
        probs = model(images)["probs"]
        pred = (probs >= 0.5).float()
    del model
    return pred


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )

    all_results = {}

    for condition_name, ckpt_dir in CONDITIONS:
        print("\n" + "#" * 70)
        print(f"# Condition: {condition_name}")
        print("#" * 70)
        condition_records = []

        for epoch_t, epoch_t1 in CHECKPOINT_PAIRS:
            print(f"\n=== {epoch_t} -> {epoch_t1} ===")
            # 4x4 transition matrix, aggregated across all subjects at this checkpoint pair
            matrix = torch.zeros(4, 4, dtype=torch.int64)  # rows=state at t, cols=state at t+1
            gt_all = None

            for subject_idx in range(N_SUBJECTS):
                image, mask, subject_id = val_dataset[subject_idx]
                image_b = image.unsqueeze(0).to(device)
                mask_b = mask.unsqueeze(0).to(device)
                gt_flat = mask_b.reshape(-1)

                pred_t = get_predictions(ckpt_dir / f"epoch_{epoch_t}.pth", image_b, device)
                pred_t1 = get_predictions(ckpt_dir / f"epoch_{epoch_t1}.pth", image_b, device)

                pred_t_flat = pred_t.reshape(-1)
                pred_t1_flat = pred_t1.reshape(-1)

                state_t = classify_state(gt_flat, pred_t_flat)
                state_t1 = classify_state(gt_flat, pred_t1_flat)

                # Accumulate into the 4x4 matrix (vectorized bincount)
                idx = state_t.long() * 4 + state_t1.long()
                counts = torch.bincount(idx, minlength=16).reshape(4, 4)
                matrix += counts.cpu()

            # Derived quantities
            tp_t = int(matrix[1, :].sum().item())
            fn_t = int(matrix[3, :].sum().item())
            fp_t = int(matrix[2, :].sum().item())
            tn_t = int(matrix[0, :].sum().item())

            tp_t1 = int(matrix[:, 1].sum().item())
            fn_t1 = int(matrix[:, 3].sum().item())
            fp_t1 = int(matrix[:, 2].sum().item())
            tn_t1 = int(matrix[:, 0].sum().item())

            def dice(tp, fp, fn):
                return 2 * tp / max(1, (2 * tp + fp + fn))

            dice_t = dice(tp_t, fp_t, fn_t)
            dice_t1 = dice(tp_t1, fp_t1, fn_t1)

            fn_to_tp = int(matrix[3, 1].item())
            fp_to_tn = int(matrix[2, 0].item())
            tp_to_fn = int(matrix[1, 3].item())
            tn_to_fp = int(matrix[0, 2].item())

            record = {
                "epoch_t": epoch_t, "epoch_t1": epoch_t1,
                "confusion_t": {"TP": tp_t, "FN": fn_t, "FP": fp_t, "TN": tn_t},
                "confusion_t1": {"TP": tp_t1, "FN": fn_t1, "FP": fp_t1, "TN": tn_t1},
                "dice_t": dice_t, "dice_t1": dice_t1, "delta_dice": dice_t1 - dice_t,
                "transition_matrix": matrix.tolist(),  # rows=state@t (TN,TP,FP,FN order 0-3), cols=state@t+1
                "FN_to_TP": fn_to_tp, "FP_to_TN": fp_to_tn,
                "TP_to_FN": tp_to_fn, "TN_to_FP": tn_to_fp,
            }
            condition_records.append(record)

            print(f"  confusion@{epoch_t}: TP={tp_t} FN={fn_t} FP={fp_t} TN={tn_t}  Dice={dice_t:.4f}")
            print(f"  confusion@{epoch_t1}: TP={tp_t1} FN={fn_t1} FP={fp_t1} TN={tn_t1}  Dice={dice_t1:.4f}")
            print(f"  delta_Dice={dice_t1-dice_t:+.4f}  FN->TP={fn_to_tp} FP->TN={fp_to_tn} TP->FN={tp_to_fn} TN->FP={tn_to_fp}")

        all_results[condition_name] = condition_records

    json_path = OUT_DIR / "real_trajectory_confusion_C62_vs_A.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {json_path}")


if __name__ == "__main__":
    main()
