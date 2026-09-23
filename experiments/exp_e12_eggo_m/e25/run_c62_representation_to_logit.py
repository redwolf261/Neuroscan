"""
Phase E25, C6-2.6: representation movement -> segmentation-logit
transition. Per the user's explicit direction: C6-2's mechanism is now
established (corrected Level 4: FN 98.0%, FP 82.3% correctly-signed
realized movement) -- the open question is no longer "is the mechanism
inverted" but "why does correctly-directed movement on FN/FP not
translate into enough Dice gain." This measures the missing link
Delta_z -> Delta_logit -> prediction transition directly, using H2's own
REALIZED trajectory mechanism (many-anchor, 15-step AdamW-accumulated
Delta_theta_margin -- NOT the single-anchor isolated probe used by
C6-2.5b/the sign-consistency audit, since this question is specifically
about what C6-2's ACTUAL training-realized update does to predictions,
matching Level 4's own scope).

For every voxel in a real batch (not just anchors): captures
pre-perturbation probability p, pre-sigmoid logit l (via
model.seg_head[0](dec1) -- the Conv3d alone, before Sigmoid, avoiding
saturation/compression near 0 and 1), applies H2's real Delta_theta_margin,
re-forward-passes, and captures p', l' at the SAME voxel. Computes:
  - Delta_logit = l' - l
  - Delta_z (H2's own quantity, reused unchanged)
  - cos(Delta_z, what) (H2's own quantity)
  - sensitivity S = |Delta_logit| / |Delta_z| (per voxel)
  - confidence tier (6 tiers: FN split into near-boundary/mid/deep,
    FP split into near-boundary/mid/deep, using the SAME p thresholds
    the user specified: FN tiers by p in [0.4,0.5)/[0.2,0.4)/[0,0.2),
    FP tiers by p in (0.5,0.6]/(0.6,0.8]/(0.8,1.0])
  - prediction transition: does p cross 0.5 in the corrective direction
    (FN: p'>=0.5; FP: p'<0.5)

Split by TP/TN/FP/FN throughout, matching the corrected Level 4
convention (verified against PHASE_E25_C62_SIGN_CONVENTION_CORRECTION.md:
FN/TP expect Delta_z toward +what under the corrected convention, FP/TN
toward -what -- but this script measures Delta_logit and transitions
directly from real probabilities/predictions, which do NOT depend on the
what-convention at all, only cos(Delta_z, what) does).
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
from train_eggo_m import EMATauB  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e24"))
from run_counterfactual import ObjectiveConfig  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e25"))
from run_h2_c62 import compute_delta_theta_margin_for_checkpoint, apply_delta_and_forward  # noqa: E402 -- REUSED, H2's own mechanism, unchanged
from run_c62_isolation_checks import classify_voxels, CAT_NAMES  # noqa: E402

CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30]
N_REPLAY_STEPS = 15  # matches H2 exactly
N_TRANSPORT_BATCHES = 4  # matches H2's own scope
N_VOXELS_PER_CAT_PER_BATCH = 300  # subsample per category per batch -- full volumes have millions of TN voxels, not all needed; FN/FP are naturally rare so this caps the COMMON categories without starving the rare ones

FOCAL_WEIGHT = 0.5
EVIDENTIAL_WEIGHT = 0.5

C62_CHECKPOINT_DIR = (project_root / "experiments" / "exp_e12_eggo_m" / "e25" /
                       "c62_runs" / "C62_sc_tam_seed0" / "checkpoints")
OUT_DIR = Path(__file__).parent / "representation_to_logit_results"


def confidence_tier(category, prob):
    """FN tiers (per the user's spec): near-boundary [0.4,0.5), mid
    [0.2,0.4), deep [0,0.2). FP tiers: near-boundary (0.5,0.6], mid
    (0.6,0.8], deep (0.8,1.0]. TP/TN: single tier each (not split,
    per the user's own scope -- the tiering question was specifically
    about ERROR depth, not about how confidently-correct a TP/TN is)."""
    if category == "FN":
        if prob >= 0.4:
            return "FN_near_boundary"
        elif prob >= 0.2:
            return "FN_mid"
        else:
            return "FN_deep"
    elif category == "FP":
        if prob <= 0.6:
            return "FP_near_boundary"
        elif prob <= 0.8:
            return "FP_mid"
        else:
            return "FP_deep"
    else:
        return category  # TP, TN -- single tier


def run_checkpoint(epoch, device, train_loader, focal_fn, evidential_fn):
    rng = np.random.RandomState(epoch)
    tau_b_tracker = EMATauB()
    loader_iter = iter(train_loader)
    objective = ObjectiveConfig(mode="sc_tam")

    model, dec1_params, delta_theta_margin = compute_delta_theta_margin_for_checkpoint(
        epoch, C62_CHECKPOINT_DIR, loader_iter, device, N_REPLAY_STEPS, focal_fn, evidential_fn, rng, tau_b_tracker, objective
    )
    axis = objective.get_w_hat(model, device)
    axis_unit = axis / axis.norm().clamp_min(1e-8)

    seg_conv = model.seg_head[0]  # Conv3d only, pre-Sigmoid -- avoids probability saturation near 0/1

    records = []
    for batch_idx in range(N_TRANSPORT_BATCHES):
        try:
            images, masks, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            images, masks, _ = next(loader_iter)
        images = images.to(device)
        masks = masks.to(device)

        with torch.no_grad():
            outputs_orig = model(images)
            dec1_orig = outputs_orig["dec1"]
            probs_orig = outputs_orig["probs"]
            pred_orig = (probs_orig >= 0.5).float()
            logit_orig = seg_conv(dec1_orig)

        dec1_perturbed = apply_delta_and_forward(model, delta_theta_margin, images)
        with torch.no_grad():
            probs_pert = model.seg_head(dec1_perturbed)
            logit_pert = seg_conv(dec1_perturbed)

        B, C, D, H, W = dec1_orig.shape
        dz = (dec1_perturbed - dec1_orig).permute(0, 2, 3, 4, 1).reshape(-1, C)
        gt_flat = masks.reshape(-1)
        pred_flat = pred_orig.reshape(-1)
        p_flat = probs_orig.reshape(-1)
        p_pert_flat = probs_pert.reshape(-1)
        l_flat = logit_orig.reshape(-1)
        l_pert_flat = logit_pert.reshape(-1)
        cat_flat = classify_voxels(gt_flat, pred_flat)

        dz_norm_per_voxel = dz.norm(dim=1)
        valid = dz_norm_per_voxel > 1e-12

        for cat_id in (0, 1, 2, 3):  # TN, TP, FP, FN
            cat_mask = (cat_flat == cat_id) & valid
            cat_idx = torch.where(cat_mask)[0]
            if cat_idx.numel() == 0:
                continue
            if cat_idx.numel() > N_VOXELS_PER_CAT_PER_BATCH:
                perm = torch.randperm(cat_idx.numel(), device=device)[:N_VOXELS_PER_CAT_PER_BATCH]
                cat_idx = cat_idx[perm]

            dz_sub = dz[cat_idx]
            dz_norm_sub = dz_norm_per_voxel[cat_idx]
            cos_what = F.cosine_similarity(dz_sub, axis_unit.unsqueeze(0).expand(cat_idx.numel(), -1), dim=1)

            p_before = p_flat[cat_idx]
            p_after = p_pert_flat[cat_idx]
            l_before = l_flat[cat_idx]
            l_after = l_pert_flat[cat_idx]
            delta_logit = l_after - l_before

            category = CAT_NAMES[cat_id]
            for i in range(cat_idx.numel()):
                pb = float(p_before[i].item())
                tier = confidence_tier(category, pb)
                dz_n = float(dz_norm_sub[i].item())
                dl = float(delta_logit[i].item())
                sensitivity = abs(dl) / dz_n if dz_n > 1e-12 else None

                if category == "FN":
                    transitioned = float(p_after[i].item()) >= 0.5
                elif category == "FP":
                    transitioned = float(p_after[i].item()) < 0.5
                else:
                    transitioned = None  # transition concept only meaningful for currently-wrong voxels

                records.append({
                    "epoch": epoch, "batch_idx": batch_idx, "category": category, "tier": tier,
                    "p_before": pb, "p_after": float(p_after[i].item()),
                    "delta_logit": dl, "dz_norm": dz_n,
                    "cos_dz_what": float(cos_what[i].item()),
                    "sensitivity": sensitivity,
                    "transitioned": transitioned,
                })

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss
    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)

    train_loader, _ = create_brats_loaders(
        batch_size=2, num_workers=0,
        root_dir=str(project_root / "Dataset" / "Training"),
        val_split=0.1,
    )

    all_records = []
    for epoch in CHECKPOINT_EPOCHS:
        print(f"\n=== Checkpoint epoch {epoch} ===")
        records = run_checkpoint(epoch, device, train_loader, focal_fn, evidential_fn)
        all_records.extend(records)
        for cat in ("FN", "FP", "TP", "TN"):
            cat_recs = [r for r in records if r["category"] == cat]
            if not cat_recs:
                continue
            trans = [r["transitioned"] for r in cat_recs if r["transitioned"] is not None]
            trans_rate = np.mean(trans) if trans else None
            mean_dl = np.mean([r["delta_logit"] for r in cat_recs])
            print(f"  {cat}: n={len(cat_recs)} mean_delta_logit={mean_dl:+.4f} "
                  + (f"transition_rate={trans_rate*100:.1f}%" if trans_rate is not None else ""))

    json_path = OUT_DIR / "representation_to_logit_C62.json"
    with open(json_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nSaved {len(all_records)} voxel records to {json_path}")


if __name__ == "__main__":
    main()
