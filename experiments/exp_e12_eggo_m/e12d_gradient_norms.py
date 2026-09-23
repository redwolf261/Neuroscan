"""
Experiment E12d, Panel 5: gradient norm ratio ||grad_dec1 L_seg|| vs.
||grad_dec1 L_margin||. Requires a LIVE forward+backward pass (gradients
aren't saved in checkpoints) -- this is the one diagnostic that couldn't
be computed post-hoc from e12d_mechanism_diagnosis.py.

Loads the E12b checkpoints (same 7 epochs), runs ONE real training batch
through each, and measures the gradient each loss term ALONE would
produce on dec1 -- computed independently per term (not from the actual
combined backward pass) via torch.autograd.grad, so L_seg's and
L_margin's contributions can be compared in isolation without one
polluting the other's measurement.

No parameter updates happen -- this is read-only gradient measurement,
not additional training.
"""
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from train_eggo_m import compute_margin_loss, sample_stratified_anchors, ANCHORS_PER_VOLUME, MAX_NEGATIVES_PER_ANCHOR, TAU_B, EVIDENCE_P99_DEFAULT  # noqa: E402

CHECKPOINT_EPOCHS = [1, 5, 10, 15, 20, 25, 30]
DELTA_D = 1.0


def main():
    exp_dir = Path(__file__).parent
    ckpt_dir = exp_dir / "e12b_pilot_seed0" / "checkpoints"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading a fixed real training batch (same batch reused for every checkpoint, for a fair comparison)...")
    train_loader, _ = create_brats_loaders(
        batch_size=8, num_workers=0,
        root_dir=str(project_root / "Dataset" / "Training"), val_split=0.1,
    )
    images, masks, _ = next(iter(train_loader))
    images = images.to(device)
    masks = masks.to(device)

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)
    focal_weight, evidential_weight = 0.5, 0.5

    print(f"\n{'epoch':<8}{'||grad L_seg||':<18}{'||grad L_margin||':<20}{'ratio (margin/seg)':<22}")
    results = []
    for epoch in CHECKPOINT_EPOCHS:
        ckpt_path = ckpt_dir / f"epoch_{epoch}.pth"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.train()  # match training-mode BN behavior, consistent with how the loss was actually computed during training

        outputs = model(images)
        probs, alpha, beta = outputs["probs"], outputs["alpha"], outputs["beta"]
        boundary_logit, dec1 = outputs["boundary_logit"], outputs["dec1"]

        focal_loss = focal_fn(probs, masks)
        evidential_loss = evidential_fn(alpha, beta, masks)
        seg_loss = focal_weight * focal_loss + evidential_weight * evidential_loss

        # Gradient of L_seg w.r.t. dec1 ALONE (retain graph for the second call)
        grad_seg = torch.autograd.grad(seg_loss, dec1, retain_graph=True)[0]
        norm_seg = grad_seg.norm().item()

        # Margin loss, computed exactly as in training
        with torch.no_grad():
            evidence_full = (alpha + beta - 2.0)
        B, C, D, H, W = dec1.shape
        dec1_perm = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
        evidence_flat = evidence_full.reshape(-1)
        boundary_flat = boundary_logit.reshape(-1)
        gt_flat = masks.reshape(-1)

        voxels_per_vol = D * H * W
        anchor_idx_list = []
        for b in range(B):
            vol_evidence = evidence_flat[b * voxels_per_vol:(b + 1) * voxels_per_vol]
            local_idx = sample_stratified_anchors(vol_evidence, ANCHORS_PER_VOLUME, None)
            anchor_idx_list.append(local_idx + b * voxels_per_vol)
        anchor_idx = torch.cat(anchor_idx_list)

        margin_loss, _ = compute_margin_loss(
            dec1_perm, evidence_flat, boundary_flat, gt_flat, anchor_idx,
            TAU_B, EVIDENCE_P99_DEFAULT, DELTA_D, MAX_NEGATIVES_PER_ANCHOR, None, device,
        )

        if margin_loss.item() > 0:
            grad_margin = torch.autograd.grad(margin_loss, dec1, retain_graph=False)[0]
            norm_margin = grad_margin.norm().item()
        else:
            norm_margin = 0.0  # loss is exactly 0 -- gradient is exactly 0 too, no need to call autograd

        ratio = norm_margin / norm_seg if norm_seg > 0 else float("nan")
        print(f"{epoch:<8}{norm_seg:<18.6f}{norm_margin:<20.8f}{ratio:<22.8f}")
        results.append({"epoch": epoch, "norm_seg": norm_seg, "norm_margin": norm_margin, "ratio": ratio})

        model.zero_grad(set_to_none=True)

    import json
    with open(exp_dir / "e12d_results" / "gradient_norms.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nVERDICT (H2: is lambda too weak relative to L_seg's gradient magnitude?)")
    last = results[-1]
    if last["ratio"] < 0.01:
        print(f"  H2 SUPPORTED: at the final checkpoint, ||grad L_margin|| is only "
              f"{last['ratio']*100:.4f}% of ||grad L_seg||'s magnitude (BEFORE lambda is even applied). "
              f"Even lambda=1.0 would leave margin's gradient contribution tiny relative to segmentation's.")
    else:
        print(f"  H2 not clearly supported by gradient magnitude alone: ratio = {last['ratio']:.4f} "
              f"(pre-lambda). The current lambda=0.1 might still be a relevant factor -- "
              f"combine with lambda * ratio to see the ACTUAL applied contribution: "
              f"{0.1 * last['ratio']:.6f} vs seg's effective weight of 1.0.")


if __name__ == "__main__":
    main()
