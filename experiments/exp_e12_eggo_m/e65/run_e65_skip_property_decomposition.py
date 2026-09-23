"""
Phase E65: Skip-Connection Causal Property Decomposition.

NO TRAINING. NO ARCHITECTURE CHANGE. Pure causal-intervention diagnostic.

CONTEXT: E64 corrected E62/E63's shared-tensor bug and confirmed the
real effect is a SKIP-CONNECTION representation effect, not a pooling
effect: Delta_pool = exactly 0.0, Delta_skip real/large/size-specific on
both a gated (v5) and ungated (v3) checkpoint, LARGER on the ungated one
(0.0270 vs 0.0226). This established WHERE the sensitivity lives (the
enc1 skip into dec1) but not WHY.

THIS PHASE decomposes "sensitivity to enc1's spatial arrangement" into
four candidate causal properties, using four independent, targeted
interventions on the skip tensor ONLY (E_pool held at the real,
un-permuted value throughout, using E64's validated split-forward
design -- so any measured effect is provably attributable to the skip
path alone, not pool1, per E64's own differential check).

FOUR INTERVENTIONS (each applied to E_skip only, E_pool always real):
  1. TRANSLATION (absolute correspondence): torch.roll the whole
     (32,64,64,64) enc1 tensor by a small fixed offset (+3 voxels along
     each spatial axis, wrapping), before the skip connection. Preserves
     every voxel's local neighborhood and the tensor's full value
     distribution exactly -- changes ONLY which upconv1 location each
     enc1 feature now lines up with. Tests: does the decoder need each
     spatial location's own corresponding feature, or tolerate a small
     global misregistration?
  2. LOCAL PERMUTATION (local relational structure): the EXACT same
     2x2x2 non-overlapping-cell derangement as E62/E64 (same
     permute_cells_derangement function, verified correct there),
     applied ONLY to the skip tensor -- this IS E64's Delta_skip,
     re-measured here as this phase's own reference point for
     "local arrangement matters," not a new intervention.
  3. CHANNEL PERMUTATION (semantic channel identity): a derangement of
     the 32 channel indices (not spatial positions) -- for every spatial
     location, channel c's value moves to a different channel slot,
     according to one shared random derangement (fixed across the whole
     volume for a given draw, since channel identity is not a
     per-location property). Preserves every voxel's own local spatial
     structure and every channel's own global marginal statistics;
     changes which channel index carries which feature at each location.
  4. SMOOTHING (frequency-scale / need for exact structure): replace
     each voxel with the mean of its 3x3x3 neighborhood (per channel,
     'same' padding via reflection to avoid introducing a boundary
     artifact), applied to the skip tensor. Removes high-frequency
     spatial detail while preserving the coarse/low-frequency spatial
     layout. No randomness -- deterministic given the input.

CHECKPOINT: v3/D4-only (ungated, same as E64's cleaner/larger-effect
checkpoint) ONLY -- per explicit sign-off, since E64 already established
the effect is not gate-dependent (present and larger without the gate),
and running all 4 arms on both checkpoints would double compute without
resolving a live question this phase asks.

STATISTICAL DISCIPLINE: identical to E47/E48/E58/E62/E63/E64 -- subject-
level paired t-test + Wilcoxon + sign-flip permutation test (>=1000
trials) + bootstrap 95% CI for each arm's mean Dice drop, plus
Spearman(native_size, drop) with its own permutation test for each arm's
size-dependence, matching E48's own significance convention throughout.

PRE-DECLARED INTERPRETATION MAPPING (not mutually exclusive; all four
Delta's reported and each tested independently -- the PATTERN across all
four determines the interpretation, not a single pass/fail gate):
  - Delta_translation significant & size-specific -> absolute spatial
    correspondence is causally necessary (candidate 1/4 from the
    write-up: "the concatenation treats corresponding positions as
    semantically aligned; a learned/registration-aware fix would be
    warranted").
  - Delta_local_perm significant but Delta_translation is NOT -> local
    relational structure matters, not absolute position (candidate 2).
  - Delta_channel_perm significant -> semantic channel identity at each
    location matters (candidate 3).
  - Delta_smoothing significant while Delta_local_perm is NOT (or is
    much smaller) -> only coarse/low-frequency content is needed, exact
    arrangement is not (a version of "not really candidate 1-3 at all").
  - Delta_smoothing significant AND Delta_local_perm ALSO significant
    -> both frequency content AND exact arrangement matter independently.
No single "GO/KILL" verdict is declared; this phase's deliverable is a
characterization, feeding a subsequent algorithm-design phase only if a
clear, size-specific pattern emerges.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
N_BOOT = 2000
N_DERANGEMENTS = 8  # independent draws per subject for the two randomized arms (local perm, channel perm)
TRANSLATION_OFFSET = 3  # voxels, fixed, each spatial axis

CKPT_V3_D4ONLY = (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs"
                   / "DeepSup_D4only_seed0" / "checkpoints" / "best.pth")
E48_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"


# ==================== shared utilities (reused verbatim from E62/E64 where applicable) ====================

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


def _batched_random_permutations(k, n_items, rng):
    keys = rng.random((k, n_items))
    return np.argsort(keys, axis=1)


def random_derangement_batch(n_items, k, rng):
    """Verbatim from E62/E64 (verified correct there)."""
    arange = np.arange(n_items)
    perms = _batched_random_permutations(k, n_items, rng)
    for _ in range(n_items):
        fixed_mask = perms == arange[None, :]
        if not fixed_mask.any():
            break
        rows_with_fixed = fixed_mask.any(axis=1)
        perms[rows_with_fixed] = np.roll(perms[rows_with_fixed], shift=1, axis=1)
    assert not (perms == arange[None, :]).any(), "Derangement construction failed to converge -- STOP."
    return perms


def random_derangement_single(n_items, rng):
    """A single derangement (n_items,) -- rejection loop, cheap at n=32."""
    idx = np.arange(n_items)
    while True:
        perm = rng.permutation(n_items)
        if not np.any(perm == idx):
            return perm


def permute_cells_derangement(enc1_np, rng):
    """Verbatim from E62/E64 (verified correct there): within every
    non-overlapping 2x2x2 cell, per channel, a random derangement of the
    8 values."""
    C, D, H, W = enc1_np.shape
    assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0
    nD, nH, nW = D // 2, H // 2, W // 2
    x = enc1_np.reshape(C, nD, 2, nH, 2, nW, 2)
    x = x.transpose(0, 1, 3, 5, 2, 4, 6)
    rows = x.reshape(-1, 8)
    n_rows = rows.shape[0]
    perms = random_derangement_batch(8, n_rows, rng)
    permuted_rows = np.take_along_axis(rows, perms, axis=1)
    out = permuted_rows.reshape(C, nD, nH, nW, 2, 2, 2)
    out = out.transpose(0, 1, 4, 2, 5, 3, 6)
    out = out.reshape(C, D, H, W)
    return out


def translate_volume(enc1_np, offset):
    """torch.roll-equivalent: shift the whole (C,D,H,W) volume by
    `offset` voxels along each of D,H,W, wrapping at boundaries (np.roll).
    Every voxel's own local neighborhood (and the tensor's full value
    distribution) is preserved exactly -- only spatial ADDRESS changes."""
    return np.roll(enc1_np, shift=(offset, offset, offset), axis=(1, 2, 3))


def permute_channels(enc1_np, rng):
    """Derange the 32 CHANNEL indices (not spatial positions): one
    shared random derangement of channel order, applied identically at
    every spatial location. Preserves every voxel's own spatial
    structure and every channel's global marginal statistics; changes
    which channel index carries which feature at each location."""
    C = enc1_np.shape[0]
    perm = random_derangement_single(C, rng)
    return enc1_np[perm]


def smooth_volume_3x3x3(enc1_np):
    """Per-channel 3x3x3 box average, reflection padding (avoids a
    boundary/edge artifact from zero-padding). Deterministic, no
    randomness."""
    t = torch.from_numpy(enc1_np).unsqueeze(0)  # (1,C,D,H,W)
    C = t.shape[1]
    kernel = torch.ones(C, 1, 3, 3, 3, dtype=t.dtype) / 27.0
    t_padded = F.pad(t, (1, 1, 1, 1, 1, 1), mode="reflect")
    out = F.conv3d(t_padded, kernel, groups=C)
    return out.squeeze(0).numpy()


# ==================== split forward (E64's validated design, v3/ungated) ====================

def forward_split_v3(model, e_pool, e_skip, device):
    """Identical to E64's forward_split_v3 (verified there): e_pool
    feeds pool1's input, e_skip feeds ONLY the skip concatenation into
    dec1. Architecturally independent from enc1 onward."""
    with torch.no_grad():
        pool1 = model.pool1(e_pool)
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
        cat1 = torch.cat([upconv1, e_skip], dim=1)
        dec1 = model.dec1(cat1)

        probs = model.seg_head(dec1)
        return probs.squeeze(0).squeeze(0).cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(CKPT_V3_D4ONLY), map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded v3/D4-only checkpoint: {CKPT_V3_D4ONLY}")
    print(f"best_val_dice={ckpt.get('best_val_dice')}\n")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n = len(val_dataset)
    print(f"Validation set size: {n}", flush=True)

    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}

    # ---------------- Sanity check: (real,real) reproduces model.forward() ----------------
    image0, _, _ = val_dataset[0]
    image0_b = image0.unsqueeze(0).to(device)
    with torch.no_grad():
        real = model(image0_b)["probs"].squeeze(0).squeeze(0).cpu().numpy()
        enc1_0 = model.enc1(image0_b)
    manual = forward_split_v3(model, enc1_0, enc1_0, device)
    max_diff = float(np.abs(real - manual).max())
    print(f"[Sanity check 1] forward_split(real,real) vs real forward(): max abs diff = {max_diff:.6e}")
    assert max_diff == 0.0, "Split-forward reimplementation mismatch -- STOP."

    # ---------------- Differential check: pool-path isolation (E64's own check, reused) ----------------
    rng_check = np.random.default_rng(999)
    enc1_0_np = enc1_0.squeeze(0).cpu().numpy()
    perm_check_np = permute_cells_derangement(enc1_0_np, rng_check)
    perm_check = torch.from_numpy(perm_check_np).unsqueeze(0).to(device)
    with torch.no_grad():
        pool1_real = model.pool1(enc1_0)
        pool1_perm = model.pool1(perm_check)
    pool1_diff = float(torch.abs(pool1_real - pool1_perm).max().item())
    print(f"[Differential check] pool1(real) vs pool1(perm): max abs diff = {pool1_diff:.6e} (MUST be 0.0)")
    assert pool1_diff == 0.0, "pool1 invariance violated -- STOP."
    probs_pool_only = forward_split_v3(model, perm_check, enc1_0, device)
    pool_only_diff = float(np.abs(probs_pool_only - real).max())
    print(f"[Differential check] pool-only-perm arm vs control: max abs diff = {pool_only_diff:.6e} (MUST be 0.0)")
    assert pool_only_diff == 0.0, "Pool-path not properly isolated -- STOP."
    print("[Differential check] PASS: pool-path isolation confirmed (E_pool held real throughout this phase).\n")

    # ---------------- Unit test: each intervention's invariant properties ----------------
    synth = np.random.default_rng(42).standard_normal((4, 8, 8, 8)).astype(np.float32)
    rng_u = np.random.default_rng(43)

    # Translation: exact multiset preserved (it's a pure roll)
    translated = translate_volume(synth, 3)
    assert np.allclose(np.sort(synth.flatten()), np.sort(translated.flatten())), "Translation changed value set -- STOP."
    assert not np.allclose(synth, translated), "Translation was a no-op -- STOP."

    # Local permutation: per-cell multiset preserved (verified pattern from E62/E64)
    local_perm = permute_cells_derangement(synth, rng_u)
    assert np.allclose(np.sort(synth[0, 0:2, 0:2, 0:2].flatten()), np.sort(local_perm[0, 0:2, 0:2, 0:2].flatten())), \
        "Local permutation: per-cell multiset changed -- STOP."

    # Channel permutation: each channel's own full spatial map is untouched, only channel ORDER changes.
    # The correct invariant is that the SET of per-channel spatial maps (each one compared as a whole,
    # not sorted per-voxel) is unchanged -- a per-row np.sort comparison is the wrong check here, since
    # each row now legitimately holds a DIFFERENT channel's data at that row index.
    chan_perm = permute_channels(synth, rng_u)
    assert not np.array_equal(synth, chan_perm), "Channel permutation was a no-op -- STOP."
    orig_channel_maps = sorted(synth[c].tobytes() for c in range(4))
    new_channel_maps = sorted(chan_perm[c].tobytes() for c in range(4))
    assert orig_channel_maps == new_channel_maps, "Channel permutation altered a channel's own spatial map -- STOP."

    # Smoothing: deterministic, reduces variance, does not change shape
    smoothed = smooth_volume_3x3x3(synth)
    assert smoothed.shape == synth.shape, "Smoothing changed tensor shape -- STOP."
    assert smoothed.std() < synth.std(), "Smoothing did not reduce variance -- STOP (unexpected for a box average)."
    print("[Unit tests] Translation/local-permutation/channel-permutation/smoothing invariant properties: PASS.\n")

    # ---------------- Main audit ----------------
    records = []
    for idx in range(n):
        image, mask, subject_id = val_dataset[idx]
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_dataset.subject_dirs[idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_size = int(seg_binary_native.sum())

        mask_frac_64 = fractional_occupancy_64(seg_binary_native)
        target_bin = (mask_frac_64 > 0.5).astype(np.float32)

        with torch.no_grad():
            enc1 = model.enc1(image_b)
        enc1_np = enc1.squeeze(0).cpu().numpy()

        probs_control = forward_split_v3(model, enc1, enc1, device)
        dice_control = dice_score((probs_control >= 0.5).astype(np.float32), target_bin)

        subj_rng = np.random.default_rng(SEED * 100000 + idx)

        # Arm 1: translation (deterministic, one draw)
        translated_np = translate_volume(enc1_np, TRANSLATION_OFFSET)
        translated_t = torch.from_numpy(translated_np).unsqueeze(0).to(device)
        probs_translation = forward_split_v3(model, enc1, translated_t, device)
        dice_translation = dice_score((probs_translation >= 0.5).astype(np.float32), target_bin)

        # Arm 2: local permutation (E64's Delta_skip, re-measured here), N_DERANGEMENTS draws
        dice_local_draws = []
        for _ in range(N_DERANGEMENTS):
            lp_np = permute_cells_derangement(enc1_np, subj_rng)
            lp_t = torch.from_numpy(lp_np).unsqueeze(0).to(device)
            probs_lp = forward_split_v3(model, enc1, lp_t, device)
            dice_local_draws.append(dice_score((probs_lp >= 0.5).astype(np.float32), target_bin))
        dice_local_perm = float(np.mean(dice_local_draws))

        # Arm 3: channel permutation, N_DERANGEMENTS draws
        dice_chan_draws = []
        for _ in range(N_DERANGEMENTS):
            cp_np = permute_channels(enc1_np, subj_rng)
            cp_t = torch.from_numpy(cp_np).unsqueeze(0).to(device)
            probs_cp = forward_split_v3(model, enc1, cp_t, device)
            dice_chan_draws.append(dice_score((probs_cp >= 0.5).astype(np.float32), target_bin))
        dice_channel_perm = float(np.mean(dice_chan_draws))

        # Arm 4: smoothing (deterministic, one draw)
        smoothed_np = smooth_volume_3x3x3(enc1_np)
        smoothed_t = torch.from_numpy(smoothed_np).unsqueeze(0).to(device)
        probs_smooth = forward_split_v3(model, enc1, smoothed_t, device)
        dice_smoothing = dice_score((probs_smooth >= 0.5).astype(np.float32), target_bin)

        e48r = e48_by_id.get(subject_id)

        records.append({
            "subject_id": subject_id, "native_size": native_size,
            "dice_control": dice_control,
            "dice_translation": dice_translation, "dice_local_perm": dice_local_perm,
            "dice_channel_perm": dice_channel_perm, "dice_smoothing": dice_smoothing,
            "drop_translation": dice_control - dice_translation,
            "drop_local_perm": dice_control - dice_local_perm,
            "drop_channel_perm": dice_control - dice_channel_perm,
            "drop_smoothing": dice_control - dice_smoothing,
            "e48_causal_drop": e48r["drop"] if e48r is not None else None,
        })

        if (idx + 1) % 25 == 0:
            print(f"  processed {idx+1}/{n} subjects", flush=True)

    with open(OUT_DIR / "E65_skip_decomposition_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.\n")

    # ================= Statistics =================
    native_size = np.array([r["native_size"] for r in records], dtype=np.float64)
    dice_control = np.array([r["dice_control"] for r in records], dtype=np.float64)
    arms = ["translation", "local_perm", "channel_perm", "smoothing"]
    drops = {a: np.array([r[f"drop_{a}"] for r in records], dtype=np.float64) for a in arms}

    def paired_tests(drop, label):
        t_stat, t_p = stats.ttest_1samp(drop, 0.0, alternative="greater")
        w_p = stats.wilcoxon(drop, alternative="greater")[1] if np.any(drop != 0) else 1.0
        rng = np.random.default_rng(SEED)
        perm_means = np.empty(N_PERM)
        for i in range(N_PERM):
            signs = rng.choice([-1, 1], size=len(drop))
            perm_means[i] = (drop * signs).mean()
        p_perm = float((perm_means >= drop.mean()).mean())
        boot_rng = np.random.default_rng(SEED + 1)
        boot_means = np.empty(N_BOOT)
        for i in range(N_BOOT):
            bs = boot_rng.choice(drop, size=len(drop), replace=True)
            boot_means[i] = bs.mean()
        ci = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))
        significant = bool(t_p < 0.05 and p_perm < 0.05 and drop.mean() > 0)
        print(f"  [{label:14s}] mean={drop.mean():+.5f} t_p={t_p:.4e} wilcoxon_p={w_p:.4e} "
              f"perm_p={p_perm:.4f} boot95CI={ci} significant={significant}")
        return {"mean": float(drop.mean()), "t_p": float(t_p), "wilcoxon_p": float(w_p),
                "perm_p": p_perm, "bootstrap_95ci": list(ci), "significant": significant}

    print(f"=== E65 Skip-Connection Property Decomposition: n={len(records)} subjects ===")
    print(f"Mean dice_control = {dice_control.mean():.4f}\n")
    print("Paired significance tests (H1: mean drop > 0), subject-level:")
    arm_stats = {a: paired_tests(drops[a], a) for a in arms}

    print("\n=== Size-dependence (Spearman(native_size, drop), each arm) ===")
    size_dep = {}
    for a in arms:
        rho, p_param = stats.spearmanr(native_size, drops[a])
        rng2 = np.random.default_rng(SEED + 2)
        perm_rhos = np.empty(N_PERM)
        for i in range(N_PERM):
            perm_y = rng2.permutation(drops[a])
            perm_rhos[i], _ = stats.spearmanr(native_size, perm_y)
        p_perm_rho = float((np.abs(perm_rhos) >= np.abs(rho)).mean())
        size_specific = bool(rho < 0 and p_perm_rho < 0.05)
        print(f"  [{a:14s}] rho={rho:+.4f} parametric_p={p_param:.4e} perm_p={p_perm_rho:.4f} "
              f"size_specific={size_specific}")
        size_dep[a] = {"rho": float(rho), "parametric_p": float(p_param), "permutation_p": p_perm_rho,
                        "size_specific": size_specific}

    # ================= Pre-declared interpretation mapping =================
    print("\n=== Interpretation (pattern across all four arms; not mutually exclusive) ===")
    sig = {a: arm_stats[a]["significant"] for a in arms}
    sizespec = {a: size_dep[a]["size_specific"] for a in arms}

    findings = []
    if sig["translation"]:
        tag = "SIZE-SPECIFIC" if sizespec["translation"] else "size-agnostic"
        findings.append(f"Absolute spatial correspondence IS causally necessary ({tag}, "
                         f"mean drop={arm_stats['translation']['mean']:+.5f}) -- supports candidate 1 "
                         f"(a registration/correspondence-preserving fix).")
    else:
        findings.append("Absolute spatial correspondence is NOT causally necessary "
                         f"(translation drop={arm_stats['translation']['mean']:+.5f}, not significant) "
                         "-- the decoder tolerates a small global misregistration.")

    if sig["local_perm"] and not sig["translation"]:
        findings.append("Local relational structure matters, NOT absolute position (local-perm significant, "
                         "translation is not) -- supports candidate 2.")
    elif sig["local_perm"] and sig["translation"]:
        findings.append("BOTH local arrangement and absolute correspondence matter independently.")

    if sig["channel_perm"]:
        tag = "SIZE-SPECIFIC" if sizespec["channel_perm"] else "size-agnostic"
        findings.append(f"Semantic channel identity at each location matters ({tag}, "
                         f"mean drop={arm_stats['channel_perm']['mean']:+.5f}) -- supports candidate 3.")
    else:
        findings.append("Semantic channel identity does NOT matter in isolation "
                         f"(channel-perm drop={arm_stats['channel_perm']['mean']:+.5f}, not significant).")

    if sig["smoothing"] and not sig["local_perm"]:
        findings.append("Only coarse/low-frequency content is needed; exact arrangement is not "
                         "(smoothing significant, local-perm is not) -- a version of 'not candidates 1-3.'")
    elif sig["smoothing"] and sig["local_perm"]:
        findings.append("BOTH frequency content and exact arrangement matter independently "
                         "(smoothing AND local-perm both significant).")
    elif not sig["smoothing"] and sig["local_perm"]:
        findings.append("Exact fine-grained arrangement matters beyond what coarse/smoothed content alone "
                         "provides (local-perm significant, smoothing is not).")

    for f in findings:
        print(f"  - {f}")

    summary = {
        "checkpoint": str(CKPT_V3_D4ONLY), "n_subjects": len(records),
        "mean_dice_control": float(dice_control.mean()),
        "translation_offset_voxels": TRANSLATION_OFFSET,
        "arm_stats": arm_stats, "size_dependence": size_dep,
        "interpretation": findings,
    }
    with open(OUT_DIR / "E65_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E65_summary.json")


if __name__ == "__main__":
    main()
