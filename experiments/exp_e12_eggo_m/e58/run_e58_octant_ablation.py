"""
Phase E58 Stage 0-1: bottleneck spatial-subset (octant) ablation audit.

CONTEXT: E48 (full bottleneck zeroing, on E46's v5/attention-gate
checkpoint) found small lesions depend causally MORE on the bottleneck
than large lesions (rho(native_size, drop)=-0.454, p<0.001). E48 never
tested WHETHER that dependence is truly distributed across the whole
8^3x256ch bottleneck ("global") or concentrated in a spatially-local
subset of it ("coarse-local, physically small"). This phase answers
that specific, narrow question -- nothing else. Per explicit user
correction: fixed, unbiased spatial octants are tested FIRST (not
lesion-conditioned octants), to avoid conflating "does locality matter
at all" with "does it track lesion location specifically."

CHECKPOINT: UNet3D_v3 (D4-only, e25/deep_sup_runs/DeepSup_D4only_seed0),
NOT E46/E48's own v5 attention-gate checkpoint -- deliberately chosen to
avoid the gate-recomputes-from-ablated-bottleneck confound (the gate
reads whatever the bottleneck becomes, which is correct for E48's own
"sever the whole coarse pathway" question but not for isolating the
bottleneck's OWN spatial content, E58's actual question). Verified this
session: checkpoint's state_dict is an EXACT key match for UNet3D_v3
(no attn_gate keys present, has aux_head3/aux_head2), loads cleanly.

INTERVENTION: the 8^3 bottleneck grid is split into 8 octants (each a
4x4x4 corner block). For each subject, three conditions are run:
  1. intact (no ablation) -- reference.
  2. octant_A zeroed (a FIXED octant, e.g. index [0:4,0:4,0:4] -- same
     octant for every subject, not lesion-conditioned).
  3. octant_B zeroed (a FIXED, different octant, e.g. the diagonally
     opposite corner [4:8,4:8,4:8]) -- a second fixed reference point,
     so a null result isn't attributable to one arbitrary octant choice.
Manual trunk reimplementation, verified bit-for-bit against real
forward() before any ablated result is trusted (same discipline as
E47/E48's own scripts).

STATISTICS: for each octant condition, Spearman(native_size, drop) +
1000-trial permutation test, exactly as E48's own analysis, so the
resulting rho values are DIRECTLY comparable to E48's own -0.454
reference. Also reports mean Dice preserved per condition (octant
ablation should preserve much more Dice than full ablation, since only
1/8 of the bottleneck is zeroed -- this is an expected, disclosed
sanity check, not itself a hypothesis test).

PRE-DECLARED DECISION RULE (per the design agent's plan, confirmed by
user): "global" (distributed) is supported if BOTH octant conditions
show substantially attenuated rho relative to E48's full-ablation
reference (-0.454) -- i.e. no small spatial subset carries much of the
size-dependent signal. "Coarse-local" is supported if EITHER octant
condition reproduces a rho comparable in magnitude to -0.454 (bootstrap
CI overlap) despite ablating only 1/8 of the bottleneck.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000
N_BOOT = 1000

CKPT_PATH = (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs"
             / "DeepSup_D4only_seed0" / "checkpoints" / "best.pth")

# Two FIXED, unbiased octants of the 8^3 bottleneck grid -- same for
# every subject, chosen as diagonally opposite corners so a null result
# can't be attributed to one arbitrary/unlucky octant choice.
# DECISION (user-confirmed): the remaining 6 octants are NOT run --
# these two already answer the first-order question ("is the effect
# obviously concentrated in one arbitrary corner? No.") and running 6
# more arbitrary corners doesn't attack the sharper follow-up question
# (lesion-conditioned ablation, next) any better than 2 already do.
OCTANT_A = (slice(0, 4), slice(0, 4), slice(0, 4))
OCTANT_B = (slice(4, 8), slice(4, 8), slice(4, 8))

E48_REFERENCE_RHO = -0.454  # PHASE_E48_BOTTLENECK_ENCODING_AUDIT_REVERSED_FINDING.md


def fractional_occupancy_64(seg_binary_native):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=(64, 64, 64), mode="area").squeeze().numpy()
    return frac


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    if denom == 0:
        return 1.0
    return float(2 * tp / denom)


def forward_with_octant_ablation(model, image, octant_slice, device):
    """Manual v3 trunk reimplementation (verified bit-for-bit identical
    to model.forward() when octant_slice=None). If octant_slice is
    given, that spatial subset of the 8^3 bottleneck grid is zeroed
    (all 256 channels, at those spatial locations only) before
    upconv3 -- a PARTIAL, spatially-restricted version of E48's own
    complete-zeroing intervention."""
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)  # (B, 256, 8, 8, 8)

        if octant_slice is not None:
            bottleneck = bottleneck.clone()
            d_sl, h_sl, w_sl = octant_slice
            bottleneck[:, :, d_sl, h_sl, w_sl] = 0.0

        upconv3 = model.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = model.dec3(cat3)

        upconv2 = model.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = model.dec2(cat2)

        upconv1 = model.upconv1(dec2)
        cat1 = torch.cat([upconv1, enc1], dim=1)
        dec1 = model.dec1(cat1)

        probs = model.seg_head(dec1)
        return probs.squeeze(0).squeeze(0).cpu().numpy()


def spearman_with_permutation(x, y, n_perm, seed):
    rho, p_parametric = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    perm_rhos = np.empty(n_perm)
    for i in range(n_perm):
        perm_y = rng.permutation(y)
        perm_rhos[i], _ = stats.spearmanr(x, perm_y)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())
    return float(rho), float(p_parametric), p_perm


def bootstrap_rho_ci(x, y, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rhos[i], _ = stats.spearmanr(x[idx], y[idx])
    return float(np.percentile(boot_rhos, 2.5)), float(np.percentile(boot_rhos, 97.5))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded D4-only checkpoint: best_val_dice={ckpt.get('best_val_dice')}", flush=True)

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    print(f"Validation set size: {len(val_dataset)}", flush=True)

    # ================= Stage 0: bit-for-bit sanity check =================
    image0, _, _ = val_dataset[0]
    image0_b = image0.unsqueeze(0).to(device)
    with torch.no_grad():
        real = model(image0_b)["probs"].squeeze(0).squeeze(0).cpu().numpy()
    manual = forward_with_octant_ablation(model, image0_b, octant_slice=None, device=device)
    max_diff = float(np.abs(real - manual).max())
    print(f"\n[Stage 0] Sanity check (octant_slice=None vs real forward): max abs diff = {max_diff:.6e}")
    assert max_diff == 0.0, "Manual trunk reimplementation does not match real forward() -- STOP, bug present."
    print("[Stage 0] PASS -- manual trunk verified bit-for-bit identical to real forward().\n")

    # ================= Stage 1: octant ablation over all subjects =================
    records = []
    for subject_idx in range(len(val_dataset)):
        image, mask, subject_id = val_dataset[subject_idx]
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_dataset.subject_dirs[subject_idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_size = int(seg_binary_native.sum())

        mask_frac_64 = fractional_occupancy_64(seg_binary_native)
        target_bin = (mask_frac_64 > 0.5).astype(np.float32)

        probs_intact = forward_with_octant_ablation(model, image_b, None, device)
        probs_octA = forward_with_octant_ablation(model, image_b, OCTANT_A, device)
        probs_octB = forward_with_octant_ablation(model, image_b, OCTANT_B, device)

        dice_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin)
        dice_octA = dice_score((probs_octA >= 0.5).astype(np.float32), target_bin)
        dice_octB = dice_score((probs_octB >= 0.5).astype(np.float32), target_bin)

        records.append({
            "subject_id": subject_id, "native_size": native_size,
            "dice_intact": dice_intact,
            "dice_octA": dice_octA, "drop_octA": dice_intact - dice_octA,
            "dice_octB": dice_octB, "drop_octB": dice_intact - dice_octB,
        })

        if (subject_idx + 1) % 25 == 0:
            print(f"  processed {subject_idx+1}/{len(val_dataset)} subjects", flush=True)

    with open(OUT_DIR / "E58_octant_ablation_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.")

    # ================= Statistics =================
    native_size = np.array([r["native_size"] for r in records], dtype=np.float64)
    dice_intact = np.array([r["dice_intact"] for r in records])
    drop_octA = np.array([r["drop_octA"] for r in records], dtype=np.float64)
    drop_octB = np.array([r["drop_octB"] for r in records], dtype=np.float64)

    print(f"\n=== E58 Octant Ablation Audit: n={len(records)} subjects ===")
    print(f"Mean dice_intact = {dice_intact.mean():.4f}")
    print(f"Mean dice_octA   = {np.array([r['dice_octA'] for r in records]).mean():.4f} "
          f"(mean drop={drop_octA.mean():.4f})")
    print(f"Mean dice_octB   = {np.array([r['dice_octB'] for r in records]).mean():.4f} "
          f"(mean drop={drop_octB.mean():.4f})")
    print(f"\nReference: E48 full-bottleneck-ablation mean drop = 0.3205, rho = {E48_REFERENCE_RHO}")

    rho_A, p_A_param, p_A_perm = spearman_with_permutation(native_size, drop_octA, N_PERM, SEED)
    rho_B, p_B_param, p_B_perm = spearman_with_permutation(native_size, drop_octB, N_PERM, SEED + 1)
    ci_A = bootstrap_rho_ci(native_size, drop_octA, N_BOOT, SEED)
    ci_B = bootstrap_rho_ci(native_size, drop_octB, N_BOOT, SEED + 1)

    print(f"\nOctant A: Spearman(native_size, drop_octA) = {rho_A:+.4f} "
          f"(parametric p={p_A_param:.4e}, permutation p={p_A_perm:.4f}) 95% CI=[{ci_A[0]:+.4f}, {ci_A[1]:+.4f}]")
    print(f"Octant B: Spearman(native_size, drop_octB) = {rho_B:+.4f} "
          f"(parametric p={p_B_param:.4e}, permutation p={p_B_perm:.4f}) 95% CI=[{ci_B[0]:+.4f}, {ci_B[1]:+.4f}]")

    # ================= Pre-declared decision rule =================
    # "Coarse-local" supported if EITHER octant's CI overlaps E48's
    # reference rho (-0.454) despite only 1/8 of the bottleneck being
    # ablated. "Global/distributed" supported if BOTH octants show
    # substantially attenuated rho (CI does not come close to -0.454).
    octA_overlaps_ref = ci_A[0] <= E48_REFERENCE_RHO <= ci_A[1]
    octB_overlaps_ref = ci_B[0] <= E48_REFERENCE_RHO <= ci_B[1]
    coarse_local_supported = octA_overlaps_ref or octB_overlaps_ref
    print(f"\nOctant A 95% CI overlaps E48 reference rho ({E48_REFERENCE_RHO}): {octA_overlaps_ref}")
    print(f"Octant B 95% CI overlaps E48 reference rho ({E48_REFERENCE_RHO}): {octB_overlaps_ref}")

    print(f"\n=== DECISION: {'COARSE-LOCAL supported' if coarse_local_supported else 'GLOBAL/DISTRIBUTED supported (neither fixed octant alone reproduces the full-ablation signature)'} ===")

    summary = {
        "n_subjects": len(records),
        "mean_dice_intact": float(dice_intact.mean()),
        "octant_A": {
            "slice": "[0:4,0:4,0:4]", "mean_drop": float(drop_octA.mean()),
            "spearman_rho": rho_A, "parametric_p": p_A_param, "permutation_p": p_A_perm,
            "bootstrap_95ci": list(ci_A), "overlaps_e48_reference": octA_overlaps_ref,
        },
        "octant_B": {
            "slice": "[4:8,4:8,4:8]", "mean_drop": float(drop_octB.mean()),
            "spearman_rho": rho_B, "parametric_p": p_B_param, "permutation_p": p_B_perm,
            "bootstrap_95ci": list(ci_B), "overlaps_e48_reference": octB_overlaps_ref,
        },
        "e48_reference_rho": E48_REFERENCE_RHO,
        "decision": "coarse_local_supported" if coarse_local_supported else "global_distributed_supported",
    }
    with open(OUT_DIR / "E58_octant_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E58_octant_summary.json")


if __name__ == "__main__":
    main()
