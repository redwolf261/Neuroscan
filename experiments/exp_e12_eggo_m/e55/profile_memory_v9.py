"""
Phase E55, verification check #5: GPU memory profiling for UNet3D_v9
(dual-pathway), before committing to any training run. Follows
e54/profile_memory_96.py's own pattern.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v9 import UNet3D_v9  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402

GPU_BUDGET_GB = 8.15
SAFE_MARGIN_GB = 1.0


def profile_batch_size(batch_size, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(0)
    model = UNet3D_v9(in_channels=1, out_channels=1).to(device)
    model.train()
    with torch.no_grad():
        model.fusion_gate.fill_(0.5)  # nonzero, so the local pathway is genuinely exercised in this profile

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)
    boundary_criterion = nn.BCEWithLogitsLoss()

    train_loader, _ = create_brats_loaders(
        batch_size=batch_size, num_workers=0,
        root_dir=str(project_root / "Dataset" / "Training"),
        val_split=0.1, target_shape=(64, 64, 64), return_native=True,
    )
    images, masks, _, native_images = next(iter(train_loader))
    images = images.to(device)
    masks = masks.to(device)
    native_images = native_images.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5)
    optimizer.zero_grad(set_to_none=True)

    outputs = model(images, native_x=native_images)
    probs = outputs["probs"]
    alpha, beta = outputs["alpha"], outputs["beta"]
    boundary_logit = outputs["boundary_logit"]
    aux_probs3, aux_probs2 = outputs["aux_probs3"], outputs["aux_probs2"]

    focal_loss = focal_fn(probs, masks)
    evidential_loss = evidential_fn(alpha, beta, masks)
    seg_loss = 0.5 * focal_loss + 0.5 * evidential_loss
    boundary_loss = boundary_criterion(boundary_logit, masks)

    import torch.nn.functional as F
    mask_d4 = F.avg_pool3d(masks, kernel_size=4, stride=4)
    mask_d2 = F.avg_pool3d(masks, kernel_size=2, stride=2)
    aux3_loss = focal_fn(aux_probs3, mask_d4)
    aux2_loss = focal_fn(aux_probs2, mask_d2)

    total_loss = seg_loss + 0.1 * boundary_loss + 0.9927 * aux3_loss + 1.0014 * aux2_loss
    total_loss.backward()
    optimizer.step()

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    return peak_mem_gb


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: no CUDA device, memory profiling is meaningless on CPU.")
        return

    print(f"GPU budget: {GPU_BUDGET_GB}GB, safe margin: {SAFE_MARGIN_GB}GB")
    print()

    results = {}
    for bs in [8, 6, 4, 2]:
        try:
            peak_gb = profile_batch_size(bs, device)
            fits = peak_gb < (GPU_BUDGET_GB - SAFE_MARGIN_GB)
            results[bs] = peak_gb
            print(f"  batch={bs}: peak_mem={peak_gb:.3f}GB  {'OK (fits with margin)' if fits else 'TOO TIGHT / OOM RISK'}")
        except torch.cuda.OutOfMemoryError as e:
            print(f"  batch={bs}: OOM")
            results[bs] = None
            torch.cuda.empty_cache()

    print()
    chosen_bs = None
    for bs in [8, 6, 4, 2]:
        if results.get(bs) is not None and results[bs] < (GPU_BUDGET_GB - SAFE_MARGIN_GB):
            chosen_bs = bs
            break
    print(f"=== RECOMMENDED batch size for UNet3D_v9 training: {chosen_bs} ===")

    import json
    out = {"gpu_budget_gb": GPU_BUDGET_GB, "results_gb": results, "recommended_batch_size": chosen_bs}
    with open(Path(__file__).parent / "E55_memory_profile.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
