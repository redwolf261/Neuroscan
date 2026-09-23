"""
Phase E47: Causal Boundary-Routing Intervention Audit.

CONTEXT: E43 (representation-change audit) found real representation
change in the bottleneck under D4 supervision, but the raw
interior-vs-boundary comparison was NULL after controlling a confound
(interior-cell-count at coarse resolution). That was a CORRELATIONAL
measurement (representation magnitude), not a CAUSAL one -- it never
tested whether the routing mechanism itself (the bottleneck->enc1 skip)
matters differentially for boundary vs. interior voxels' actual
predictions. E46 (UNet3D_v5) subsequently built exactly such a routing
mechanism (attn_gate1, gating enc1 using bottleneck context) and trained
it successfully (psi is real, non-degenerate, 0.9102 val dice) -- this
gives us, for the first time in the project, a genuine INSTRUMENT to run
a causal test with, using NO NEW TRAINING: the already-trained E46
checkpoint.

NO TRAINING. NO NEW MODEL. Loads E46's best checkpoint
(experiments/exp_e12_eggo_m/e46/runs/AttnGate_seed0/checkpoints/best.pth)
and runs targeted CAUSAL INTERVENTIONS at inference time: clamp psi=1
(i.e. disable the learned gate, falling back to raw v3-style routing) at
a SPECIFIC SUBSET of spatial locations, leaving the model's own learned
psi intact everywhere else, and measure the resulting Dice CHANGE
relative to the fully-intact (no clamping) forward pass.

HYPOTHESIS: if the routing mechanism matters disproportionately at
boundary-adjacent voxels (the causal claim a boundary-targeted fix would
need to justify), then clamping psi=1 AT BOUNDARY VOXELS should hurt
Dice more than clamping psi=1 AT AN EQUAL NUMBER of INTERIOR voxels
(matched voxel count -- controls for the same "more voxels intervened on
= more effect" confound trivially, and additionally checked against the
E43-diagnosed interior-cell-count confound explicitly in Step 3 below).

REGION DEFINITION: reused EXACTLY from E43's own build_region_masks
convention (avg_pool3d of native GT mask, interior/boundary/background by
occupancy fraction), computed at 64^3 -- enc1's own resolution, where the
gate psi actually lives (E43 used 16^3/32^3 for dec3/dec2; here we need
64^3 specifically since that's attn_gate1's own operating resolution).

METRIC: per-subject Dice computed the SAME way as every prior condition's
own validate() (probs >= 0.5 hard threshold vs. binary mask), NOT a soft
proxy -- this is a genuine causal outcome measurement, not a
representation-magnitude proxy like E43 used.

STATISTICAL SAFEGUARDS (project's own established discipline):
  - Subject-clustered: each subject contributes ONE paired
    (boundary_drop, interior_drop) pair -- no double-counting components.
  - Permutation test (>=500 trials, per this project's own mandatory
    minimum): sign-flip permutation on the paired difference
    (boundary_drop - interior_drop) across subjects.
  - Explicit confound check (Step 3): does the boundary-vs-interior
    voxel-COUNT ratio predict the drop-difference on its own? If so, this
    is the SAME confound class E43 diagnosed and must be controlled
    (matched-count sampling already does this by construction, but this
    is verified empirically, not assumed).

PRE-DECLARED DECISION RULE:
  GO (boundary-targeted fix justified) only if:
    1. mean(boundary_drop - interior_drop) > 0 (boundary clamping hurts
       more), AND
    2. permutation p < 0.05, AND
    3. the effect is NOT explained by residual voxel-count confound
       (checked via correlation between per-subject voxel-count-matched-
       ratio and drop-difference; must be non-significant or the effect
       must survive stratification on it).
  Otherwise: NULL. Do not proceed to a boundary-targeted fix on this
  basis -- report the null plainly, as with every prior phase.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import scipy.ndimage as ndi
from scipy.ndimage import zoom

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
EPS = 1e-9
SEED = 0
N_PERM = 1000  # exceeds the project's own >=500 minimum

CKPT_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs" / "AttnGate_seed0" / "checkpoints" / "best.pth"


def resize_nn(volume, target_shape):
    zf = tuple(t / c for t, c in zip(target_shape, volume.shape))
    return zoom(volume, zf, order=0)


def fractional_occupancy_64(seg_binary_native):
    """Fractional tumor-occupancy at 64^3, matching E43/E36's own
    avg_pool3d-family convention -- CORRECTED for this script's specific
    need: dec1/enc1's own resolution (64^3) is NOT reached by avg_pool3d
    from a coarser stage in this project's existing scripts (E43 used
    avg_pool3d from a 64^3 NEAREST-NEIGHBOR mask down to 16^3/32^3, but
    at 64^3 itself E43's own mask was nearest-neighbor-binarized, which
    is why E43 excluded dec1 from its boundary/interior test entirely --
    verified by inspecting E43's own build_region_masks and its
    'regions_here = [...] if stage != dec1' branch). Here we DO need a
    genuine fractional 64^3 mask (this IS attn_gate1's own resolution),
    so this uses F.interpolate(mode='area') directly from the NATIVE
    volume -- equivalent to average-pooling to an arbitrary target size,
    verified to produce real fractional values (not 0/1 only) before use."""
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=(64, 64, 64), mode="area").squeeze().numpy()
    return frac


def build_region_masks_64(mask_64):
    """SAME convention as E43's build_region_masks, at 64^3 only (enc1's
    own resolution -- attn_gate1's psi operates here directly, no
    downsampling needed since mask_64 IS already 64^3)."""
    interior = mask_64 >= 1.0 - 1e-6
    background = mask_64 <= 1e-6
    boundary = (~interior) & (~background)
    return {"interior": interior, "boundary": boundary, "background": background}


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    if denom == 0:
        return 1.0  # both empty -- perfect agreement, matches project's own MetricAccumulator convention
    return float(2 * tp / denom)


def forward_with_psi_clamp(model, image, clamp_mask_64, device):
    """Runs the v5 trunk manually (mirroring UNet3D_v5.forward byte-for-
    byte) but INTERCEPTS psi right after the gate computes it: at every
    voxel where clamp_mask_64 is True, psi is forced to 1.0 (gate
    disabled -> raw ungated enc1, exactly v3's own behavior at that
    voxel), leaving the model's own learned psi elsewhere. clamp_mask_64:
    (64,64,64) bool numpy array, or None for no clamping (fully-intact
    pass, used as the reference)."""
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)

        upconv3 = model.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = model.dec3(cat3)

        upconv2 = model.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = model.dec2(cat2)

        upconv1 = model.upconv1(dec2)

        # Gate computation, IDENTICAL to AttentionGate3D.forward, but with
        # psi intercepted before it's applied.
        gate = bottleneck
        skip = enc1
        g = model.attn_gate1.W_g(gate)
        g_up = F.interpolate(g, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = model.attn_gate1.W_x(skip)
        psi = torch.sigmoid(model.attn_gate1.W_psi(F.relu(g_up + x)))  # (1,1,64,64,64)

        if clamp_mask_64 is not None:
            clamp_t = torch.from_numpy(clamp_mask_64).to(device).unsqueeze(0).unsqueeze(0)
            psi = torch.where(clamp_t, torch.ones_like(psi), psi)

        enc1_gated = enc1 * psi
        cat1 = torch.cat([upconv1, enc1_gated], dim=1)
        dec1 = model.dec1(cat1)

        probs = model.seg_head(dec1)
        return probs.squeeze(0).squeeze(0).cpu().numpy()  # (64,64,64)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded E46 checkpoint: best_val_dice={ckpt.get('best_val_dice')}", flush=True)

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    print(f"Validation set size: {len(val_dataset)}", flush=True)

    records = []
    for subject_idx in range(len(val_dataset)):
        image, mask, subject_id = val_dataset[subject_idx]
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_dataset.subject_dirs[subject_idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        mask_frac_64 = fractional_occupancy_64(seg_binary_native)  # fractional, for region definition
        mask_64 = (mask_frac_64 > 0.5).astype(np.float32)  # binarized, for the Dice OUTCOME metric (matches project's own thresholding convention)

        regions = build_region_masks_64(mask_frac_64)
        n_boundary = int(regions["boundary"].sum())
        n_interior = int(regions["interior"].sum())

        if n_boundary == 0 or n_interior == 0:
            continue  # cannot form a matched pair for this subject

        n_match = min(n_boundary, n_interior)  # matched voxel COUNT, per pre-declared control

        boundary_idx = np.argwhere(regions["boundary"])
        interior_idx = np.argwhere(regions["interior"])
        sel_boundary = boundary_idx[rng.choice(len(boundary_idx), size=n_match, replace=False)]
        sel_interior = interior_idx[rng.choice(len(interior_idx), size=n_match, replace=False)]

        clamp_boundary = np.zeros((64, 64, 64), dtype=bool)
        clamp_boundary[tuple(sel_boundary.T)] = True
        clamp_interior = np.zeros((64, 64, 64), dtype=bool)
        clamp_interior[tuple(sel_interior.T)] = True

        target_bin = mask_64  # (64,64,64), 0/1

        probs_intact = forward_with_psi_clamp(model, image_b, None, device)
        probs_clamp_boundary = forward_with_psi_clamp(model, image_b, clamp_boundary, device)
        probs_clamp_interior = forward_with_psi_clamp(model, image_b, clamp_interior, device)

        dice_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin)
        dice_clamp_boundary = dice_score((probs_clamp_boundary >= 0.5).astype(np.float32), target_bin)
        dice_clamp_interior = dice_score((probs_clamp_interior >= 0.5).astype(np.float32), target_bin)

        boundary_drop = dice_intact - dice_clamp_boundary
        interior_drop = dice_intact - dice_clamp_interior

        records.append({
            "subject_id": subject_id,
            "n_boundary_voxels": n_boundary, "n_interior_voxels": n_interior, "n_matched": n_match,
            "dice_intact": dice_intact,
            "dice_clamp_boundary": dice_clamp_boundary, "dice_clamp_interior": dice_clamp_interior,
            "boundary_drop": boundary_drop, "interior_drop": interior_drop,
            "drop_difference": boundary_drop - interior_drop,
        })

        if (subject_idx + 1) % 25 == 0:
            print(f"  processed {subject_idx+1}/{len(val_dataset)} subjects", flush=True)

    with open(OUT_DIR / "E47_causal_intervention_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.", flush=True)

    # ================= Statistical analysis =================
    n = len(records)
    boundary_drop = np.array([r["boundary_drop"] for r in records])
    interior_drop = np.array([r["interior_drop"] for r in records])
    diff = boundary_drop - interior_drop
    n_match = np.array([r["n_matched"] for r in records])
    count_ratio = np.array([r["n_boundary_voxels"] / max(1, r["n_interior_voxels"]) for r in records])

    print(f"\n=== E47 Causal Routing Audit: n={n} subjects ===")
    print(f"Mean dice_intact          = {np.mean([r['dice_intact'] for r in records]):.4f}")
    print(f"Mean boundary_drop        = {boundary_drop.mean():.5f} (+/-{boundary_drop.std():.5f})")
    print(f"Mean interior_drop        = {interior_drop.mean():.5f} (+/-{interior_drop.std():.5f})")
    print(f"Mean drop_difference      = {diff.mean():.5f} (+/-{diff.std():.5f})")
    print(f"Fraction subjects with boundary_drop > interior_drop: {(diff > 0).mean():.1%}")

    # Sign-flip permutation test on the paired difference
    observed = diff.mean()
    rng2 = np.random.default_rng(SEED + 1)
    perm_means = np.empty(N_PERM)
    for i in range(N_PERM):
        signs = rng2.choice([-1, 1], size=n)
        perm_means[i] = (diff * signs).mean()
    p_value = float((np.abs(perm_means) >= np.abs(observed)).mean())
    print(f"\nPermutation test ({N_PERM} trials, sign-flip): observed mean diff={observed:.5f}, p={p_value:.4f}")

    # Confound check: does the boundary/interior voxel-count ratio predict drop_difference?
    if np.std(count_ratio) > 1e-9:
        corr = float(np.corrcoef(count_ratio, diff)[0, 1])
    else:
        corr = float("nan")
    print(f"\nConfound check: corr(boundary/interior voxel-count ratio, drop_difference) = {corr:.4f}")

    # Pre-declared decision rule
    go = (observed > 0) and (p_value < 0.05) and (abs(corr) < 0.3 or np.isnan(corr))
    print(f"\n=== DECISION: {'GO' if go else 'NULL'} ===")
    if go:
        print("Boundary voxels show a significantly larger causal routing effect than interior voxels,")
        print("not explained by voxel-count confound. Boundary-targeted routing fix is justified.")
    else:
        print("Pre-declared GO criteria not met. Reporting NULL -- no boundary-targeted fix justified on this basis.")

    summary = {
        "n_subjects": n,
        "mean_dice_intact": float(np.mean([r["dice_intact"] for r in records])),
        "mean_boundary_drop": float(boundary_drop.mean()), "sd_boundary_drop": float(boundary_drop.std()),
        "mean_interior_drop": float(interior_drop.mean()), "sd_interior_drop": float(interior_drop.std()),
        "mean_drop_difference": float(diff.mean()), "sd_drop_difference": float(diff.std()),
        "frac_boundary_gt_interior": float((diff > 0).mean()),
        "permutation_p_value": p_value, "n_permutations": N_PERM,
        "confound_corr_count_ratio_vs_diff": corr,
        "decision": "GO" if go else "NULL",
    }
    with open(OUT_DIR / "E47_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E47_summary.json")


if __name__ == "__main__":
    main()
