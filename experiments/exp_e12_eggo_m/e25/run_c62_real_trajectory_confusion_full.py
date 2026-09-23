"""
Phase E25, C6-2.8 (real-trajectory version, FULL VALIDATION SET): extends
run_c62_real_trajectory_confusion.py's 8-subject version to the FULL
125-subject validation set, per explicit instruction: "Use all available
real checkpoint information, not another artificially balanced 8-subject
diagnostic for the primary conclusion. If full-volume inference over the
validation set is computationally manageable, use it."

Verified feasible before committing to the full run: a smoke test showed
~0.35s per (checkpoint, subject) forward pass, so 125 subjects x 2
checkpoints x 6 pairs x 2 conditions is on the order of 15-20 minutes of
pure compute -- well within budget, no need to fall back to a subset.

Two changes from the 8-subject version, both real fixes not just scale:
  1. N_SUBJECTS = full validation set (125), not 8.
  2. Checkpoints are loaded ONCE per (condition, pair) and reused across
     every subject, rather than reloaded fresh per subject (the 8-subject
     version's redundant-load pattern, harmless at n=8, wasteful at
     n=125).
  3. ALSO computes proper per-volume Dice, averaged across subjects,
     matching H4's own established methodology EXACTLY (not just the
     pooled-voxel confusion-matrix Dice the 8-subject version used,
     which was flagged there as producing a number in the OPPOSITE
     direction from H4's real result) -- this version reports BOTH
     statistics side by side, explicitly to resolve/explain that
     discrepancy rather than repeat it.
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

GATE6_A_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e24" / "gate6_runs" / "A_baseline_seed0" / "checkpoints"
C62_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "c62_runs" / "C62_sc_tam_seed0" / "checkpoints"
OUT_DIR = Path(__file__).parent / "real_trajectory_confusion_results"

CONDITIONS = [("A_baseline", GATE6_A_DIR), ("C62_sc_tam", C62_DIR)]


def classify_state(gt, pred):
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


def per_volume_dice(gt, pred):
    tp = (pred * gt).sum()
    return (2 * tp / (pred.sum() + gt.sum() + 1e-6)).item()


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_subjects = len(val_dataset)
    print(f"Full validation set: {n_subjects} subjects")

    all_results = {}

    for condition_name, ckpt_dir in CONDITIONS:
        print("\n" + "#" * 70)
        print(f"# Condition: {condition_name}  ({n_subjects} subjects)")
        print("#" * 70)
        condition_records = []

        for epoch_t, epoch_t1 in CHECKPOINT_PAIRS:
            print(f"\n=== {epoch_t} -> {epoch_t1} ===")
            matrix = torch.zeros(4, 4, dtype=torch.int64)
            per_volume_dice_t = []
            per_volume_dice_t1 = []

            model_t = load_model(ckpt_dir / f"epoch_{epoch_t}.pth", device)
            model_t1 = load_model(ckpt_dir / f"epoch_{epoch_t1}.pth", device)

            for subject_idx in range(n_subjects):
                image, mask, subject_id = val_dataset[subject_idx]
                image_b = image.unsqueeze(0).to(device)
                mask_b = mask.unsqueeze(0).to(device)
                gt_flat = mask_b.reshape(-1)

                with torch.no_grad():
                    pred_t = (model_t(image_b)["probs"] >= 0.5).float()
                    pred_t1 = (model_t1(image_b)["probs"] >= 0.5).float()

                pred_t_flat = pred_t.reshape(-1)
                pred_t1_flat = pred_t1.reshape(-1)

                per_volume_dice_t.append(per_volume_dice(gt_flat, pred_t_flat))
                per_volume_dice_t1.append(per_volume_dice(gt_flat, pred_t1_flat))

                state_t = classify_state(gt_flat, pred_t_flat)
                state_t1 = classify_state(gt_flat, pred_t1_flat)

                idx = state_t.long() * 4 + state_t1.long()
                counts = torch.bincount(idx, minlength=16).reshape(4, 4)
                matrix += counts.cpu()

            del model_t, model_t1
            if device.type == "cuda":
                torch.cuda.empty_cache()

            tp_t = int(matrix[1, :].sum().item())
            fn_t = int(matrix[3, :].sum().item())
            fp_t = int(matrix[2, :].sum().item())
            tn_t = int(matrix[0, :].sum().item())

            tp_t1 = int(matrix[:, 1].sum().item())
            fn_t1 = int(matrix[:, 3].sum().item())
            fp_t1 = int(matrix[:, 2].sum().item())
            tn_t1 = int(matrix[:, 0].sum().item())

            def pooled_dice(tp, fp, fn):
                return 2 * tp / max(1, (2 * tp + fp + fn))

            pooled_dice_t = pooled_dice(tp_t, fp_t, fn_t)
            pooled_dice_t1 = pooled_dice(tp_t1, fp_t1, fn_t1)
            mean_per_volume_dice_t = float(np.mean(per_volume_dice_t))
            mean_per_volume_dice_t1 = float(np.mean(per_volume_dice_t1))

            fn_to_tp = int(matrix[3, 1].item())
            fp_to_tn = int(matrix[2, 0].item())
            tp_to_fn = int(matrix[1, 3].item())
            tn_to_fp = int(matrix[0, 2].item())

            record = {
                "epoch_t": epoch_t, "epoch_t1": epoch_t1, "n_subjects": n_subjects,
                "confusion_t": {"TP": tp_t, "FN": fn_t, "FP": fp_t, "TN": tn_t},
                "confusion_t1": {"TP": tp_t1, "FN": fn_t1, "FP": fp_t1, "TN": tn_t1},
                "pooled_dice_t": pooled_dice_t, "pooled_dice_t1": pooled_dice_t1,
                "mean_per_volume_dice_t": mean_per_volume_dice_t,
                "mean_per_volume_dice_t1": mean_per_volume_dice_t1,
                "delta_pooled_dice": pooled_dice_t1 - pooled_dice_t,
                "delta_mean_per_volume_dice": mean_per_volume_dice_t1 - mean_per_volume_dice_t,
                "transition_matrix": matrix.tolist(),
                "FN_to_TP": fn_to_tp, "FP_to_TN": fp_to_tn,
                "TP_to_FN": tp_to_fn, "TN_to_FP": tn_to_fp,
            }
            condition_records.append(record)

            print(f"  confusion@{epoch_t}: TP={tp_t} FN={fn_t} FP={fp_t} TN={tn_t}")
            print(f"  confusion@{epoch_t1}: TP={tp_t1} FN={fn_t1} FP={fp_t1} TN={tn_t1}")
            print(f"  pooled_dice: {pooled_dice_t:.4f} -> {pooled_dice_t1:.4f}")
            print(f"  mean_per_volume_dice (matches H4 methodology): {mean_per_volume_dice_t:.4f} -> {mean_per_volume_dice_t1:.4f}")
            print(f"  FN->TP={fn_to_tp} FP->TN={fp_to_tn} TP->FN={tp_to_fn} TN->FP={tn_to_fp}")

        all_results[condition_name] = condition_records

    json_path = OUT_DIR / "real_trajectory_confusion_C62_vs_A_FULL.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {json_path}")


if __name__ == "__main__":
    main()
