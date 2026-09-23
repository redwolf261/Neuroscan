"""
Phase E36-A2: multi-batch robustness check for E36-A's single-fixed-batch
gradient measurement.

WHY THIS EXISTS: E36-A's core finding (D4-only's D4-term gradient norm
relative to main climbs sharply late in training, 0.47x->2.42x by epoch 30,
while cosine similarity to main simultaneously DROPS 0.63->0.17 -- a
combination neither D2-only nor Both show) was measured on a single fixed
8-subject batch. Per this project's own standing practice (prefer the full
dataset over a small/single sample whenever affordable), this script re-runs
the IDENTICAL measurement across ALL 15 disjoint full batches of 8 subjects
that fit in the 125-subject validation set (120/125 subjects covered, the
remaining 5 dropped rather than padded with a partial/uneven batch) before
the E36-A pattern is trusted as a real, condition-distinguishing property
rather than a batch-composition artifact.

Reuses E36-A's exact loss-term definitions, layer groups, and model-loading
code (imported, not reimplemented, to guarantee identical measurement
semantics -- avoiding the classic bug of two "should be identical"
computations silently drifting apart).

Output: E36_gradient_measurements_multibatch.json (one record per
(condition, epoch, batch_idx), same schema as E36-A's output plus a
"batch_idx" field).
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from run_e36a_gradient_autopsy import (  # noqa: E402
    RUN_DIRS, EPOCHS, load_model, compute_term_gradient, flatten_grad,
    per_param_grad_norms, get_layer_group, cosine_and_ratio,
)

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
MU = 0.1
LAMBDA_DS3 = 0.9927
LAMBDA_DS2 = 1.0014
BATCH_SIZE = 8
N_FULL_BATCHES = 15  # floor(125/8) -- 120/125 subjects, remaining 5 dropped rather than padded


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_subjects = len(val_dataset)
    print(f"Validation set size: {n_subjects}, using {N_FULL_BATCHES} disjoint batches of {BATCH_SIZE} "
          f"({N_FULL_BATCHES * BATCH_SIZE}/{n_subjects} subjects covered)", flush=True)

    # Pre-load all batches once (images/masks), reused across every checkpoint.
    batches = []
    for b in range(N_FULL_BATCHES):
        idxs = list(range(b * BATCH_SIZE, (b + 1) * BATCH_SIZE))
        imgs, masks = [], []
        for idx in idxs:
            img, mask, _ = val_dataset[idx]
            imgs.append(img)
            masks.append(mask)
        batches.append((torch.stack(imgs).to(device), torch.stack(masks).to(device)))
    print(f"Loaded {len(batches)} batches.", flush=True)

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)
    boundary_criterion = torch.nn.BCEWithLogitsLoss()

    all_results = []
    for condition, run_dir in RUN_DIRS.items():
        for epoch in EPOCHS:
            ckpt_path = run_dir / "checkpoints" / f"epoch_{epoch}.pth"
            if not ckpt_path.exists():
                print(f"  MISSING: {ckpt_path} -- skipping", flush=True)
                continue
            model, ckpt = load_model(ckpt_path, device)
            model.eval()

            for batch_idx, (images, masks) in enumerate(batches):
                term_grads = {}
                term_losses = {}
                for term in ("main", "D2", "D4"):
                    loss_val = compute_term_gradient(
                        model, images, masks, focal_fn, evidential_fn, boundary_criterion,
                        LAMBDA_DS3, LAMBDA_DS2, term,
                    )
                    term_losses[term] = loss_val
                    term_grads[term] = flatten_grad(model).cpu()

                pair_stats = {}
                for t1, t2 in [("main", "D4"), ("main", "D2"), ("D4", "D2")]:
                    cos, ratio = cosine_and_ratio(term_grads[t1], term_grads[t2])
                    pair_stats[f"{t1}_vs_{t2}"] = {"cosine": cos, "norm_ratio_t1_over_t2": ratio}

                norms = {t: float(term_grads[t].norm().item()) for t in ("main", "D2", "D4")}

                all_results.append({
                    "condition": condition, "epoch": epoch, "batch_idx": batch_idx,
                    "best_val_dice_so_far": ckpt.get("best_val_dice"),
                    "term_losses": term_losses,
                    "term_grad_norms_full": norms,
                    "pairwise": pair_stats,
                })

            # Print per-checkpoint AGGREGATE (mean +/- SD across the 15 batches) immediately,
            # not just raw per-batch rows, so progress is legible.
            recs_here = [r for r in all_results if r["condition"] == condition and r["epoch"] == epoch]
            d4_ratio = [r["term_grad_norms_full"]["D4"] / r["term_grad_norms_full"]["main"] for r in recs_here]
            cos_main_d4 = [r["pairwise"]["main_vs_D4"]["cosine"] for r in recs_here]
            print(f"[{condition}] epoch={epoch}: "
                  f"D4/main ratio = {np.mean(d4_ratio):.3f} +/- {np.std(d4_ratio):.3f}  "
                  f"cos(main,D4) = {np.mean(cos_main_d4):+.3f} +/- {np.std(cos_main_d4):.3f}", flush=True)

    with open(OUT_DIR / "E36_gradient_measurements_multibatch.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} (condition, epoch, batch) records.", flush=True)


if __name__ == "__main__":
    main()
