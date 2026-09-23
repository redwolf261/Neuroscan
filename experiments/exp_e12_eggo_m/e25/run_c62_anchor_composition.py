"""
Phase E25, C6-2 pre-C6-3 diagnostic: active-anchor composition by
TP/TN/FP/FN, per the user's explicit instruction to run this BEFORE any
C6-3 design work -- "We need to determine: What does SC-TAM actually
sample?"

For every SAMPLED anchor and every ACTIVE (i,j) pair in a real C6-2
training batch, classify the ANCHOR (not the negative partner) by the
model's OWN CURRENT PREDICTION vs. ground truth (TP/TN/FP/FN), and report:

  1. fraction of SAMPLED anchors in each category (determined purely by
     sample_stratified_anchors' evidence-based top-k selection -- NOT
     dependent on the pair loop at all, computed once per anchor).
  2. fraction of ACTIVE pairs (dist < margin_target, i.e. hinge > 0)
     whose anchor is in each category.
  3. each category's RAW summed squared-hinge loss contribution
     (sum over that category's anchors of weight_i * per_anchor_loss_i,
     BEFORE the final losses.mean() division by n_anchor).
  4. each category's SHARE of the final mean loss (raw contribution /
     total summed loss) -- this is "how much of L_SC's actual value is
     attributable to voxels in each category," the gradient-allocation
     question the user asked for directly.

STRUCTURAL NOTE, confirmed by direct inspection of compute_margin_loss
before writing this script: anchors are sampled by sample_stratified_
anchors, which selects the n_sample voxels with LOWEST evidence (highest
uncertainty) -- this stratification is ENTIRELY orthogonal to whether the
model's current prediction is correct; it has no mechanism to prefer or
avoid TP/TN/FP/FN in any particular way. Any composition skew found here
is therefore a genuine EMERGENT property of high-uncertainty voxels'
prediction-correctness correlation, not an explicit design choice already
visible in the sampling code -- worth stating plainly since it means the
result is not something that could be predicted from the sampler's own
docstring alone.

Standalone instrumented reimplementation of ONLY the sc_tam branch's pair
logic (NOT a modification of the production compute_margin_loss, which
stays untouched and regression-locked) -- copied field-for-field from
train_eggo_m.py's own code (verified line-for-line against the real
function immediately before writing this file), extended ONLY to also
track TP/TN/FP/FN category membership per anchor and attribute loss/
activity to those categories.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import sample_stratified_anchors, ANCHORS_PER_VOLUME, MAX_NEGATIVES_PER_ANCHOR, EMATauB  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e24"))
from run_counterfactual import ObjectiveConfig  # noqa: E402

CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30]
N_BATCHES_PER_CHECKPOINT = 6  # real training batches sampled per checkpoint -- larger than H1/H2/H3's 4 since this is a pure forward-pass diagnostic (no backward/replay cost), affordable to sample more

C62_CHECKPOINT_DIR = (project_root / "experiments" / "exp_e12_eggo_m" / "e25" /
                       "c62_runs" / "C62_sc_tam_seed0" / "checkpoints")
OUT_DIR = Path(__file__).parent / "anchor_composition_results"


def classify_voxels(gt, pred_binary):
    """Returns a (N,) int8 tensor: 0=TN, 1=TP, 2=FP, 3=FN, matching
    train_eggo_m.py's own tumor_mask convention (gt > 0.5)."""
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
    # tn stays 0 by construction (already zeros), included for clarity in the assertion below
    assert (tp.int() + tn.int() + fp.int() + fn.int()).eq(1).all(), "voxel classification is not a partition -- bug"
    return cat  # 0=TN, 1=TP, 2=FP, 3=FN


CAT_NAMES = {0: "TN", 1: "TP", 2: "FP", 3: "FN"}


def instrumented_sc_tam_pass(dec1_flat, gt_flat, cat_flat, anchor_idx, w_hat, delta_d, max_negatives, rng, device):
    """Field-for-field copy of compute_margin_loss's sc_tam branch
    (pair construction, hinge, per_anchor_loss, active-pair tracking),
    verified line-for-line against train_eggo_m.py immediately before
    writing this file -- EXTENDED ONLY to also record each anchor's
    TP/TN/FP/FN category and attribute loss/activity to it. weight is
    fixed to 1.0 here (U_hat*B omitted) since this diagnostic asks about
    the RAW class-conditional pair structure and sampling, not the
    additional evidence/boundary reweighting layered on top -- kept
    OUT of scope deliberately, matching C6-2's own "no error/uncertainty
    weighting yet" locked scope (Section 11 decision 4)."""
    anchors_z = dec1_flat[anchor_idx]
    anchors_gt = gt_flat[anchor_idx]
    anchors_cat = cat_flat[anchor_idx]  # (n_anchor,) int8: 0=TN,1=TP,2=FP,3=FN

    n_anchor = anchors_z.shape[0]
    per_anchor_loss_full = torch.zeros(n_anchor, device=device)
    anchor_active_frac = torch.zeros(n_anchor, device=device)  # fraction of THIS anchor's own n_neg pairs that are active

    tumor_mask = anchors_gt > 0.5
    bg_mask = ~tumor_mask
    tumor_local = torch.where(tumor_mask)[0]
    bg_local = torch.where(bg_mask)[0]

    total_active_pairs = 0
    total_pairs = 0
    active_pair_cat_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # active pairs whose ANCHOR is in this category

    with torch.no_grad():  # pure diagnostic -- no gradient needed, matches the "forward-pass-only" scope
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

            margin_target = delta_d
            hinge = torch.clamp(margin_target - dist, min=0.0) ** 2
            per_anchor_loss = hinge.mean(dim=1)
            per_anchor_loss_full[pos_local] = per_anchor_loss  # weight=1.0, per this diagnostic's deliberate scope

            active_this = (dist < margin_target)  # (n_pos, n_neg) bool
            anchor_active_frac[pos_local] = active_this.float().mean(dim=1)

            total_active_pairs += int(active_this.sum().item())
            total_pairs += active_this.numel()

            # Attribute ACTIVE PAIRS to the ANCHOR's category (not the
            # negative's) -- for each pos_local anchor, count how many of
            # its n_neg pairs are active, grouped by that anchor's category.
            pos_cats = anchors_cat[pos_local]  # (n_pos,)
            active_counts_per_anchor = active_this.sum(dim=1)  # (n_pos,)
            for cat_id in (0, 1, 2, 3):
                cat_sel = pos_cats == cat_id
                if cat_sel.any():
                    active_pair_cat_counts[cat_id] += int(active_counts_per_anchor[cat_sel].sum().item())

    total_loss_sum = float(per_anchor_loss_full.sum().item())  # sum, NOT yet divided by n_anchor
    mean_loss = float(per_anchor_loss_full.mean().item())

    # Per-category: sampled count, active-pair count, raw loss sum, share of total loss sum
    sampled_counts = {cat_id: int((anchors_cat == cat_id).sum().item()) for cat_id in (0, 1, 2, 3)}
    raw_loss_sums = {cat_id: float(per_anchor_loss_full[anchors_cat == cat_id].sum().item()) if sampled_counts[cat_id] > 0 else 0.0 for cat_id in (0, 1, 2, 3)}
    loss_shares = {cat_id: (raw_loss_sums[cat_id] / total_loss_sum if total_loss_sum > 1e-12 else None) for cat_id in (0, 1, 2, 3)}

    return {
        "n_anchor": n_anchor,
        "sampled_counts": {CAT_NAMES[k]: v for k, v in sampled_counts.items()},
        "active_pair_counts": {CAT_NAMES[k]: v for k, v in active_pair_cat_counts.items()},
        "total_active_pairs": total_active_pairs,
        "total_pairs": total_pairs,
        "raw_loss_sums": {CAT_NAMES[k]: v for k, v in raw_loss_sums.items()},
        "loss_shares": {CAT_NAMES[k]: v for k, v in loss_shares.items()},
        "total_loss_sum": total_loss_sum,
        "mean_loss": mean_loss,
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
    all_records = []

    for epoch in CHECKPOINT_EPOCHS:
        print(f"\n=== Checkpoint epoch {epoch} ===")
        ckpt = torch.load(C62_CHECKPOINT_DIR / f"epoch_{epoch}.pth", map_location=device, weights_only=False)
        model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()  # PREDICTION for TP/TN/FP/FN classification uses eval-mode BN, matching how Dice/H4 itself is always scored (train_eggo_m.py's own validate()) -- a deliberate, distinct choice from H2's .train()-mode mechanism, since this diagnostic's category labels must match "the prediction Dice is actually computed from," not the training-time BN regime

        rng = np.random.RandomState(epoch)
        loader_iter = iter(train_loader)
        w_hat = objective.get_w_hat(model, device)

        for batch_idx in range(N_BATCHES_PER_CHECKPOINT):
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

            B, C, D, H, W = dec1.shape
            dec1_flat = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
            gt_flat = masks.reshape(-1)
            pred_flat = pred_binary.reshape(-1)
            evidence_flat = evidence_full.reshape(-1)
            cat_flat = classify_voxels(gt_flat, pred_flat)

            voxels_per_vol = D * H * W
            anchor_idx_list = []
            for b in range(B):
                vol_evidence = evidence_flat[b * voxels_per_vol:(b + 1) * voxels_per_vol]
                local_idx = sample_stratified_anchors(vol_evidence, ANCHORS_PER_VOLUME, rng)
                anchor_idx_list.append(local_idx + b * voxels_per_vol)
            anchor_idx = torch.cat(anchor_idx_list)

            result = instrumented_sc_tam_pass(
                dec1_flat, gt_flat, cat_flat, anchor_idx, w_hat, objective.delta_d,
                MAX_NEGATIVES_PER_ANCHOR, rng, device,
            )
            result["epoch"] = epoch
            result["batch_idx"] = batch_idx

            # Also record the VOLUME-WIDE (not just sampled-anchor)
            # TP/TN/FP/FN composition, for context -- how rare is each
            # category in the full data, vs. among sampled anchors.
            volume_counts = {CAT_NAMES[k]: int((cat_flat == k).sum().item()) for k in (0, 1, 2, 3)}
            result["volume_wide_counts"] = volume_counts

            all_records.append(result)

            sc = result["sampled_counts"]
            ls = result["loss_shares"]
            print(f"  batch {batch_idx}: sampled TP={sc['TP']} TN={sc['TN']} FP={sc['FP']} FN={sc['FN']} "
                  f"(n_anchor={result['n_anchor']}) | loss_share TP={ls['TP']} TN={ls['TN']} "
                  f"FP={ls['FP']} FN={ls['FN']} | active_pairs={result['total_active_pairs']}/{result['total_pairs']}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    json_path = OUT_DIR / "anchor_composition_C62.json"
    with open(json_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nSaved {len(all_records)} records to {json_path}")


if __name__ == "__main__":
    main()
