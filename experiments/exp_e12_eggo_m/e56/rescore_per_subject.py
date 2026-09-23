"""
Phase E56, Step 2: per-subject re-scoring of already-trained checkpoints.

CONTEXT: PHASE_E27_PROJECT_AUDIT.md found pooled/batch Dice (what every
post-pivot mechanism, E44-E55, has been judged against) differs from
mean PER-SUBJECT Dice by 1.91pp for the SAME checkpoint (baseline A:
0.9063 pooled vs 0.8872 per-subject). PHASE_E25_STRUCTURAL_PIVOT_1B_
4WAY_MECHANISM_COMPARISON.md found NONE of A/D4-only/D2-only/Both are
statistically distinguishable on paired per-subject Dice (p=0.71 for
the largest comparison). This script re-scores every post-pivot
mechanism's own best checkpoint on PER-SUBJECT Dice (inference only, no
retraining) so they can be properly, pairwise statistically compared
against baseline A -- not just their pooled headline numbers.

CORRECTNESS CHECK (run first, before trusting this script on anything
else): baseline A's own recomputed per-subject mean should closely
match E27's own independently-reported 0.8872.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from neuroscan_3d_v4 import UNet3D_v4  # noqa: E402
from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from neuroscan_3d_v6 import UNet3D_v6  # noqa: E402
from neuroscan_3d_v7 import UNet3D_v7  # noqa: E402
from neuroscan_3d_v8 import UNet3D_v8  # noqa: E402
from neuroscan_3d_v9 import UNet3D_v9  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

BASE = Path(__file__).parent

# (name, checkpoint_path, model_class, target_shape, needs_native (for v9))
CHECKPOINTS = [
    ("baseline_A",   "e24/gate6_runs/A_baseline_seed0/checkpoints/best.pth", UNet3D_v2, (64, 64, 64), False),
    ("e45_D4D8",     "e45/runs/D4_D8_seed0/checkpoints/best.pth",             UNet3D_v4, (64, 64, 64), False),
    ("e46_AttnGate", "e46/runs/AttnGate_seed0/checkpoints/best.pth",          UNet3D_v5, (64, 64, 64), False),
    ("e49_CCABA_s0", "e49/runs/CCABA_seed0/checkpoints/best.pth",             UNet3D_v6, (64, 64, 64), False),
    ("e49_CCABA_s1", "e49/runs/CCABA_seed1/checkpoints/best.pth",             UNet3D_v6, (64, 64, 64), False),
    ("e49_CCABA_s2", "e49/runs/CCABA_seed2/checkpoints/best.pth",             UNet3D_v6, (64, 64, 64), False),
    ("e50_IECG_s0",  "e50/runs/IECG_seed0/checkpoints/best.pth",              UNet3D_v7, (64, 64, 64), False),
    ("e50_IECG_s1",  "e50/runs/IECG_seed1/checkpoints/best.pth",              UNet3D_v7, (64, 64, 64), False),
    ("e50_IECG_s2",  "e50/runs/IECG_seed2/checkpoints/best.pth",              UNet3D_v7, (64, 64, 64), False),
    ("e51_CCAG_s0",  "e51/runs/CCAG_seed0/checkpoints/best.pth",              UNet3D_v8, (64, 64, 64), False),
    ("e51_CCAG_s1",  "e51/runs/CCAG_seed1/checkpoints/best.pth",              UNet3D_v8, (64, 64, 64), False),
    ("e51_CCAG_s2",  "e51/runs/CCAG_seed2/checkpoints/best.pth",              UNet3D_v8, (64, 64, 64), False),
    ("e54_A96",      "e54/runs/A96_seed0/checkpoints/best.pth",               UNet3D_v3, (96, 96, 96), False),
    ("e55_DualRes",  "e55/runs/DualRes_seed0/checkpoints/best.pth",           UNet3D_v9, (64, 64, 64), True),
]


def dice_per_subject(pred_bin, gt_bin, eps=1e-6):
    """pred_bin, gt_bin: (D,H,W) numpy or torch binary arrays. Returns scalar Dice."""
    inter = (pred_bin * gt_bin).sum()
    return float((2 * inter + eps) / (pred_bin.sum() + gt_bin.sum() + eps))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {}

    for name, ckpt_rel, model_cls, target_shape, needs_native in CHECKPOINTS:
        ckpt_path = project_root / "experiments" / "exp_e12_eggo_m" / ckpt_rel
        if not ckpt_path.exists():
            print(f"[{name}] SKIP -- checkpoint not found: {ckpt_path}")
            continue

        print(f"\n[{name}] Loading {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = model_cls(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        val_ds = BraTSDataset(
            root_dir=str(project_root / "Dataset" / "Training"),
            split="val", val_split=0.1, target_shape=target_shape,
            return_native=needs_native,
        )

        per_subject_dice = {}
        with torch.no_grad():
            for idx in tqdm(range(len(val_ds)), desc=f"[{name}] scoring"):
                if needs_native:
                    image, mask, subject_id, native_image = val_ds[idx]
                    native_image = native_image.unsqueeze(0).to(device)
                else:
                    image, mask, subject_id = val_ds[idx]
                    native_image = None

                image = image.unsqueeze(0).to(device)
                mask_np = mask[0].numpy()

                if needs_native:
                    out = model(image, native_x=native_image)
                else:
                    out = model(image)
                probs = out["probs"] if isinstance(out, dict) else out
                pred_bin = (probs[0, 0].cpu().numpy() >= 0.5).astype(np.float32)
                gt_bin = (mask_np >= 0.5).astype(np.float32)

                per_subject_dice[subject_id] = dice_per_subject(pred_bin, gt_bin)

        dice_values = np.array(list(per_subject_dice.values()))
        mean_dice = float(dice_values.mean())
        std_dice = float(dice_values.std(ddof=1))
        pooled_dice_reported = float(ckpt.get("best_val_dice", float("nan")))

        print(f"[{name}] n_subjects={len(dice_values)} "
              f"mean_per_subject_dice={mean_dice:.4f} std={std_dice:.4f} "
              f"(reported pooled_dice={pooled_dice_reported:.4f}, gap={pooled_dice_reported-mean_dice:+.4f})")

        results[name] = {
            "checkpoint": ckpt_rel,
            "reported_pooled_dice": pooled_dice_reported,
            "mean_per_subject_dice": mean_dice,
            "std_per_subject_dice": std_dice,
            "n_subjects": len(dice_values),
            "per_subject_dice": per_subject_dice,
        }

        # Free GPU memory before loading the next checkpoint/model
        del model
        torch.cuda.empty_cache()

    with open(BASE / "E56_per_subject_rescoring.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to {BASE / 'E56_per_subject_rescoring.json'}")

    # ================= Correctness check =================
    if "baseline_A" in results:
        recomputed = results["baseline_A"]["mean_per_subject_dice"]
        expected = 0.8872  # PHASE_E27_PROJECT_AUDIT.md's own independently-reported value
        diff = abs(recomputed - expected)
        print(f"\n=== CORRECTNESS CHECK ===")
        print(f"Recomputed baseline_A mean per-subject Dice: {recomputed:.4f}")
        print(f"E27's own independently-reported value:      {expected:.4f}")
        print(f"Absolute difference: {diff:.4f}")
        check_pass = diff < 0.01  # 1pp tolerance -- different random val-set forward pass order, float precision, etc.
        print(f"CORRECTNESS CHECK PASS (diff < 0.01): {check_pass}")
        if not check_pass:
            print("!!! WARNING: recomputed value does not match E27's own reported baseline -- "
                  "investigate before trusting any other checkpoint's numbers in this run !!!")


if __name__ == "__main__":
    main()
