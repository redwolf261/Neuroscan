"""
Phase E82 (numbered E82 in this project's ledger -- the user's message
called it "E81," but that name is already taken by the realignment
control phase just completed; keeping the ledger consistent):
Local Donor Identity Test (NO TRAINING, pure inference-time
intervention) -- per the user's design, following E81's correction
(relative encoder-decoder displacement is the real failure mode; global
coherent shift is largely equivariant and factors out).

QUESTION: when correspondence breaks, what does the decoder actually
need at a given location -- the EXACT direction u(p) (address-identity
dependency), or would a NEARBY u(q) work about as well (local
equivalence / correspondence-field structure)? And if nearby directions
are NOT interchangeable, is the damage isotropic (depends only on
distance |delta|) or anisotropic (depends on which specific direction
delta points)?

Two complementary measurements, both cheap (no training):

PART 1 -- GLOBAL DIRECTIONAL/RADIAL SWEEP (isotropy check):
  Reuses the translation machinery, but sweeps small offsets in MULTIPLE
  directions (not just the single +3,+3,+3 diagonal every prior phase
  used), at a few magnitudes. If damage depends mainly on ||delta||
  (offset magnitude) regardless of direction -> ISOTROPIC (positional
  distance is what matters). If damage varies substantially by which
  axis/direction is shifted -> ANISOTROPIC (a specific structural
  pattern, not a generic distance-decay).

PART 2 -- LOCAL DONOR SWAP PROBE (address-identity vs local equivalence):
  At a SUBSAMPLE of lesion-boundary and lesion-interior voxels (where
  E77 showed high magnitude / high salience concentrates), swap ONLY
  that single voxel's direction u(p) for a NEARBY voxel's direction
  u(q), q in a small neighborhood (6-connected + second-ring), r(p)
  held fixed at p's own value throughout. Multiple non-overlapping swap
  locations are batched into ONE forward pass per (subject, radius)
  combination (swaps at different, well-separated spatial locations
  don't interact within a single forward pass, so this is efficient --
  not one forward pass per voxel).
  Measures the local response curve C(delta) = mean Dice drop as a
  function of swap distance/direction, and specifically checks: does
  ANY nearby donor cause near-zero damage (local equivalence), or does
  EVERY nearby donor -- even the closest -- cause substantial damage
  (address-identity dependency)?

PRE-DECLARED READING:
  - C(delta) flat/near-zero at delta=1 (closest neighbors), rising with
    distance -> LOCAL EQUIVALENCE: nearby directions are largely
    interchangeable; a correspondence-field/soft-search mechanism could
    exploit this.
  - C(delta) already large even at delta=1, not much worse at delta=2,3
    -> ADDRESS-IDENTITY DEPENDENCY: even the closest possible donor is
    already "wrong" -- position specificity is sharp, not smoothly
    decaying. A soft local search would not obviously help; the
    decoder wants THIS voxel's own feature, not merely A nearby one.
  - Isotropic vs anisotropic (Part 1) further characterizes whichever
    case holds.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from scipy.ndimage import distance_transform_edt

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e74"))
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e78"))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402
from run_e74_spatial_dependence_audit import translate_volume  # noqa: E402
from run_e78_magnitude_direction_decomposition import decompose, recompose, forward_with_enc1  # noqa: E402

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 0
N_PERM = 1000
N_BOOT = 2000
N_SWAP_LOCATIONS_PER_SUBJECT = 12  # non-overlapping lesion-relevant voxels sampled per subject

MM_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e70" / "runs"
           / "MM_seed0" / "checkpoints" / "best.pth")

# Part 1: directional offsets at magnitude 3 (matching the project's own
# reference offset), covering axis-aligned and diagonal directions, plus
# a magnitude sweep along one axis for a distance reference.
DIRECTIONAL_OFFSETS = {
    "+x": (3, 0, 0), "-x": (-3, 0, 0),
    "+y": (0, 3, 0), "-y": (0, -3, 0),
    "+z": (0, 0, 3), "-z": (0, 0, -3),
    "+xyz_diag": (3, 3, 3),  # the project's own standard offset, reference point
}
MAGNITUDE_SWEEP = [1, 2, 3, 4, 5]  # along a single axis (+x), for a distance-decay reference

# Part 2: local swap radii (Chebyshev/max-norm distance in voxels)
SWAP_NEIGHBOR_OFFSETS = {
    1: [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)],  # 6-connected, distance 1
    2: [(2,0,0),(-2,0,0),(0,2,0),(0,-2,0),(0,0,2),(0,0,-2)],  # distance 2
    3: [(3,0,0),(-3,0,0),(0,3,0),(0,-3,0),(0,0,3),(0,0,-3)],  # distance 3
}


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    return 1.0 if denom == 0 else float(2 * tp / denom)


def shift_direction_only(u, offset):
    """u: (C,D,H,W) unit-direction tensor. offset: (dz,dy,dx) tuple."""
    return torch.roll(u, shifts=offset, dims=(1, 2, 3))


def local_swap_direction(u, locations, offset, D, H, W):
    """u: (C,D,H,W). locations: list of (z,y,x) voxel indices (assumed
    well-separated, non-overlapping under any tested offset radius <=3
    with N_SWAP_LOCATIONS_PER_SUBJECT small relative to volume size).
    offset: (dz,dy,dx) swap direction, wrapped at boundaries.
    Returns a COPY of u with each listed location's direction replaced
    by the direction at (location + offset)."""
    u_out = u.clone()
    dz, dy, dx = offset
    for (z, y, x) in locations:
        zs, ys, xs = (z + dz) % D, (y + dy) % H, (x + dx) % W
        u_out[:, z, y, x] = u[:, zs, ys, xs]
    return u_out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(MM_CKPT), map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded MM seed0 checkpoint: best_per_subject_dice={ckpt.get('best_per_subject_dice')}\n")

    val_ds = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                    split="val", val_split=0.1, target_shape=(64, 64, 64))
    rng = np.random.default_rng(SEED)

    directional_records = []
    swap_records = []

    for idx in range(len(val_ds)):
        img, msk, sid = val_ds[idx]
        img_b = img.unsqueeze(0).to(device)
        target_bin = (msk.squeeze(0).numpy() > 0.5).astype(np.float32)
        D, H, W = target_bin.shape

        with torch.no_grad():
            enc1_intact = model.enc1(img_b).squeeze(0)
        r_intact, u_intact = decompose(enc1_intact)

        probs_intact, _ = forward_with_enc1(model, img_b, enc1_intact, device)
        d_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin)

        # PART 1 skipped in this corrected re-run (already valid, no bug --
        # global Dice is a sound metric for a whole-volume translation, unlike
        # Part 2's sparse voxel swaps; reusing E82's original Part-1 results).

        # ---------------- PART 2: local donor swap probe (CORRECTED) ----------------
        if target_bin.sum() > 0 and target_bin.sum() < target_bin.size:
            dist_out = distance_transform_edt(1 - target_bin)
            dist_in = distance_transform_edt(target_bin)
            is_boundary = ((target_bin > 0) & (dist_in <= 2)) | ((target_bin == 0) & (dist_out <= 2))
            is_interior = (target_bin > 0) & (dist_in > 2)
            candidate_mask = is_boundary | is_interior
            candidate_voxels = np.argwhere(candidate_mask)
            # keep candidates at least 6 voxels from the volume edge (so all
            # tested swap radii stay in-bounds without wrap ambiguity) and
            # from each other (avoid overlapping swaps within one forward pass)
            margin = 6
            valid = ((candidate_voxels[:, 0] >= margin) & (candidate_voxels[:, 0] < D - margin) &
                    (candidate_voxels[:, 1] >= margin) & (candidate_voxels[:, 1] < H - margin) &
                    (candidate_voxels[:, 2] >= margin) & (candidate_voxels[:, 2] < W - margin))
            candidate_voxels = candidate_voxels[valid]
        else:
            candidate_voxels = np.empty((0, 3), dtype=int)

        if len(candidate_voxels) >= N_SWAP_LOCATIONS_PER_SUBJECT:
            # greedily pick well-separated locations (min pairwise distance >= 8)
            chosen = []
            perm = rng.permutation(len(candidate_voxels))
            for i in perm:
                cand = candidate_voxels[i]
                if all(np.abs(cand - c).max() >= 8 for c in chosen):
                    chosen.append(cand)
                if len(chosen) >= N_SWAP_LOCATIONS_PER_SUBJECT:
                    break
            locations = [tuple(int(v) for v in c) for c in chosen]

            swap_row = {"subject_id": sid, "n_locations": len(locations), "D_intact": d_intact}
            for radius, offsets in SWAP_NEIGHBOR_OFFSETS.items():
                # average over the 6 axis-aligned directions at this radius,
                # applying ALL locations' swaps simultaneously in one pass
                # per direction (locations are well-separated, so swaps don't interact)
                dice_per_direction = []
                local_prob_change_per_direction = []
                for offset in offsets:
                    u_swapped = local_swap_direction(u_intact, locations, offset, D, H, W)
                    z_swapped = recompose(r_intact, u_swapped)
                    probs_swapped, _ = forward_with_enc1(model, img_b, z_swapped, device)
                    d_swapped = dice_score((probs_swapped >= 0.5).astype(np.float32), target_bin)
                    dice_per_direction.append(d_intact - d_swapped)
                    # LOCAL metric: mean |probability change| AT the swapped
                    # voxels themselves -- global Dice is far too insensitive
                    # to a handful of isolated-voxel perturbations out of
                    # 262,144 total voxels (confirmed: only ~10 voxels/subject
                    # touched, ~0.004% of the volume), so this is the metric
                    # that actually answers the local-equivalence question.
                    local_changes = [abs(float(probs_swapped[loc] - probs_intact[loc])) for loc in locations]
                    local_prob_change_per_direction.append(float(np.mean(local_changes)))
                swap_row[f"S_radius{radius}"] = float(np.mean(dice_per_direction))
                swap_row[f"local_change_radius{radius}"] = float(np.mean(local_prob_change_per_direction))
            swap_records.append(swap_row)

        if (idx + 1) % 25 == 0:
            print(f"  processed {idx+1}/{len(val_ds)}", flush=True)

    with open(OUT_DIR / "E82b_local_swap_table_corrected.json", "w") as f:
        json.dump(swap_records, f, indent=2)
    print(f"\nSaved {len(swap_records)} local-swap records (corrected, with local-change metric).\n")

    # ---------------- Part 2 analysis: local equivalence vs address-identity ----------------
    print("\n=== PART 2: Local donor swap probe ===")
    if len(swap_records) > 0:
        radius_means = {}
        local_change_means = {}
        for radius in SWAP_NEIGHBOR_OFFSETS.keys():
            vals = np.array([r[f"S_radius{radius}"] for r in swap_records])
            radius_means[radius] = float(vals.mean())
            lc_vals = np.array([r[f"local_change_radius{radius}"] for r in swap_records])
            local_change_means[radius] = float(lc_vals.mean())
            n_touched = np.mean([r["n_locations"] for r in swap_records])
            frac_volume = n_touched / (64**3)
            print(f"  radius={radius}: mean GLOBAL Dice-drop S = {vals.mean():.6f} (std {vals.std():.6f}) "
                  f"-- INSENSITIVE, only ~{n_touched:.0f} voxels touched ({frac_volume*100:.4f}% of volume)")
            print(f"             mean LOCAL |prob change| AT swapped voxels = {lc_vals.mean():.4f} "
                  f"(std {lc_vals.std():.4f}, n={len(vals)} subjects) -- the metric that actually answers the question")

        print(f"\n  NOTE: the global Dice metric (S_radius*) is confirmed too insensitive to detect this "
              f"intervention -- swapping ~10 isolated, well-separated voxels out of 262,144 total per subject "
              f"(~0.004% of the volume) cannot move whole-volume Dice meaningfully even if every swapped voxel "
              f"flipped its prediction entirely. The LOCAL probability-change metric at the swap sites "
              f"themselves is the metric that actually tests local-equivalence vs address-identity.")

        lc1 = local_change_means.get(1, float("nan"))
        lc3 = local_change_means.get(3, float("nan"))
        near_zero_at_r1 = lc1 < 0.05
        steep_falloff = (lc3 > 0) and (lc1 / max(lc3, 1e-8) < 0.4)

        if near_zero_at_r1:
            local_reading = ("LOCAL EQUIVALENCE: even the closest (radius=1) donor swap causes near-zero "
                            "LOCAL probability change at the swapped voxel -- neighboring directions are "
                            "largely interchangeable at short range. A soft local-correspondence mechanism "
                            "could plausibly exploit this.")
        elif steep_falloff:
            local_reading = ("PARTIAL LOCAL EQUIVALENCE: radius=1 causes meaningfully less local change than "
                            "radius=3, suggesting SOME local tolerance, but not free -- a narrow-radius "
                            "correspondence search might help, not a fully generic one.")
        else:
            local_reading = ("ADDRESS-IDENTITY DEPENDENCY: even the closest possible donor (radius=1) "
                            "already causes substantial LOCAL probability change, comparable to farther "
                            "donors -- the decoder wants THIS voxel's own feature specifically, not merely a "
                            "nearby one. Local correspondence search is unlikely to help much; position "
                            "specificity is sharp.")
        print(f"\n{local_reading}")
    else:
        radius_means = {}
        local_change_means = {}
        local_reading = "No valid swap locations found across subjects -- cannot assess."
        print(local_reading)

    summary = {
        "n_swap_subjects": len(swap_records),
        "swap_radius_means": {str(k): v for k, v in radius_means.items()},
        "swap_local_change_means": {str(k): v for k, v in local_change_means.items()},
        "local_reading": local_reading,
    }
    with open(OUT_DIR / "E82b_local_donor_summary_corrected.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E82b_local_donor_summary_corrected.json")


if __name__ == "__main__":
    main()
