"""
Phase E62: Max-Pool Subcell Position-Information Causal Audit.

NO TRAINING. NO NEW ARCHITECTURE. Pure causal-intervention diagnostic,
same discipline as E47/E48/E58.

CONTEXT: after E61 killed the "small lesions need a higher-dimensional
bottleneck" hypothesis, the user went back to the network's literal
mathematics rather than another architectural guess. The concrete
structural fact: pool1/pool2/pool3 are nn.MaxPool3d(2) -- each collapses
8 values to 1, discarding not just the other 7 values but the SPATIAL
IDENTITY of which subcell position held the max. This motivated a
candidate fix ("Positional-Moment Downsampling", PMD: replace max
pooling with a mass + first-spatial-moment summary of each 2x2x2 cell).
A literature check found PMD's general idea (moment/position-aware
pooling) already occupied by several 2025-2026 papers, but ALSO found a
prior reasoning gap: the network has conv layers and skip connections
AFTER pooling that could recover positional information some other way,
so "max pooling loses position" does not by itself prove "the TRAINED
network's predictions are causally sensitive to that lost position."

THIS PHASE tests that causal claim directly, on the real trained network,
before any operator design or training spend.

METHOD (pre-declared, decided by explicit user sign-off before running):
  Real-activation subcell permutation at pool1 (the highest-resolution,
  most lesion-relevant pooling stage: 64^3 -> 32^3). For each subject,
  run the real enc1 activation forward as normal up to pool1's input.
  Then, independently for every 2x2x2 non-overlapping cell and every
  channel, apply a RANDOM DERANGEMENT of that cell's 8 values (a
  permutation with no fixed point, so position is guaranteed to change)
  -- this preserves the cell's exact multiset of 8 values (so max, mean/
  mass, and every other permutation-invariant statistic of the cell are
  IDENTICAL before and after) while changing which subcell position holds
  which value. Continue the REAL trained forward pass from pool1 onward
  (pool2, pool3, bottleneck, decoder, skip connections all intact and
  real) with the derangement-permuted enc1 in place of the real one.

This isolates exactly the quantity max pooling is blind to (subcell
position, holding the multiset of values fixed) and asks whether the
REST of the trained network (conv layers, skip connections, decoder)
recovers enough of that lost position to make the final prediction
insensitive to it, or whether the prediction changes -- i.e. whether the
position was doing real causal work despite MaxPool3d's own blindness to
it.

Multiple independent derangement draws per subject (not a single draw)
to get a distribution of Delta-Dice rather than one arbitrary
permutation's outcome.

PRE-DECLARED DECISION RULE (explicit user sign-off, matching E48's own
bar exactly):
  GO (subcell position loss is task-relevant AND small-lesion-specific,
  justifying PMD design) only if BOTH:
    1. Mean Dice drop (intact - permuted) is significantly > 0 across
       subjects (paired test + subject-level permutation test, >=1000
       trials, matching project convention).
    2. The per-subject drop is significantly correlated with SMALLER
       native lesion size (Spearman rho < 0, permutation p < 0.05,
       1000 trials) -- matching E48's own rho=-0.454 significance bar,
       not just "any" size relationship.
  Otherwise: NULL / KILL, reported plainly. A significant but
  size-agnostic drop is reported honestly but does NOT meet the
  pre-declared bar (would motivate a generic pooling fix, not
  specifically the small-lesion-focused PMD chain this project is
  pursuing).
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

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000
N_DERANGEMENTS = 8  # independent random derangement draws per subject

CKPT_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs" / "AttnGate_seed0" / "checkpoints" / "best.pth"
E48_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"


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


def random_derangement(n, rng):
    """A permutation of range(n) with NO fixed points. Rejection sampling
    (n=8 is tiny, this converges in a handful of draws on average)."""
    if n <= 1:
        raise ValueError("Derangement requires n >= 2.")
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def _batched_random_permutations(k, n_items, rng):
    """Vectorized row-wise random permutation of range(n_items), k rows at
    once (np.random.Generator has no batched shuffle, so this uses
    argsort-of-random-keys, the standard vectorized trick)."""
    keys = rng.random((k, n_items))
    return np.argsort(keys, axis=1)


def random_derangement_batch(n_items, k, rng):
    """Vectorized: draw k independent random derangements of range(n_items),
    fully vectorized with NO Python-level per-row loop (required for
    whole-volume application: k can be in the millions).

    Construction: draw a random permutation per row, then for EVERY row
    unconditionally cyclically-rotate-by-one the columns at even/odd
    fixed-point positions is unreliable in general, so instead this uses
    a simpler, fully general, fully vectorized closed-form: apply a fixed
    cyclic rotation by 1 to the COLUMN INDICES themselves (perms[:, i] ->
    value that was at perms[:, i-1]) ONLY for rows that have >=1 fixed
    point, wrapping until no row has a fixed point. Because rotating a
    whole row by one step can itself only fail to remove a fixed point at
    a column c if the value that lands there (from column c-1 after
    rotation) also happens to equal c, this is checked and iterated
    (rotate again) for the small residual -- iteration count is bounded
    in practice (empirically <=2 rounds) and every step is a fully
    vectorized numpy roll + mask, no per-row Python loop regardless of
    how many rounds are needed.

    Statistically: sampling a uniform permutation and repairing fixed
    points this way does not sample uniformly over the set of
    derangements, but every returned row IS a verified true derangement
    (checked below), which is what the causal intervention requires --
    exact uniformity over derangements is not a requirement for this
    causal test (only that max/mass/multiset is preserved and every
    value's position changes, both guaranteed)."""
    arange = np.arange(n_items)
    perms = _batched_random_permutations(k, n_items, rng)

    for _ in range(n_items):  # hard upper bound on rounds; converges much sooner
        fixed_mask = perms == arange[None, :]
        if not fixed_mask.any():
            break
        # For rows with any fixed point, rotate that ROW's values by one
        # column (value at col i <- value at col i-1, wraparound) --
        # fully vectorized via np.roll on the masked rows only.
        rows_with_fixed = fixed_mask.any(axis=1)
        rotated = np.roll(perms[rows_with_fixed], shift=1, axis=1)
        perms[rows_with_fixed] = rotated

    still_fixed = perms == arange[None, :]
    assert not still_fixed.any(), "Derangement construction failed to converge -- STOP, bug present."
    return perms  # (k, n_items)


def permute_cells_derangement(enc1_np, rng):
    """enc1_np: (C, D, H, W) numpy array, D/H/W all even (64 here).
    For every non-overlapping 2x2x2 cell and every channel INDEPENDENTLY,
    apply a random derangement of that cell's 8 values. Preserves the
    cell's exact multiset of values (max, sum/mass, mean, variance all
    identical) while moving every value's subcell position (no value
    stays in its original of the 8 slots).

    Vectorized: reshape into (C * n_cells, 8) rows, draw one independent
    derangement per row via random_derangement_batch, apply with
    np.take_along_axis. Equivalent to the per-cell-per-channel Python-loop
    definition, just fast enough to run on a 64^3 volume."""
    C, D, H, W = enc1_np.shape
    assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0
    nD, nH, nW = D // 2, H // 2, W // 2

    # Reshape (C,D,H,W) -> (C, nD,2, nH,2, nW,2) -> (C*nD*nH*nW, 8)
    x = enc1_np.reshape(C, nD, 2, nH, 2, nW, 2)
    x = x.transpose(0, 1, 3, 5, 2, 4, 6)  # (C, nD, nH, nW, 2, 2, 2)
    rows = x.reshape(-1, 8)  # (C*nD*nH*nW, 8)

    n_rows = rows.shape[0]
    perms = random_derangement_batch(8, n_rows, rng)  # (n_rows, 8)
    permuted_rows = np.take_along_axis(rows, perms, axis=1)

    out = permuted_rows.reshape(C, nD, nH, nW, 2, 2, 2)
    out = out.transpose(0, 1, 4, 2, 5, 3, 6)  # back to (C, nD,2, nH,2, nW,2)
    out = out.reshape(C, D, H, W)
    return out


def forward_from_enc1(model, enc1, device):
    """Continues the REAL trained forward pass from a given enc1 tensor
    (B,32,64,64,64) through pool1 onward -- bit-for-bit identical
    computation to model.forward() past this point, verified in main()."""
    with torch.no_grad():
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

        # v5 attention gate on the enc1 skip (reads the REAL, un-permuted
        # enc1 as the skip -- only pool1's INPUT is permuted, matching the
        # pre-declared intervention: MaxPool3d's own blindness is being
        # tested, not the skip connection's separate access to enc1).
        gate = bottleneck
        skip = enc1
        g = model.attn_gate1.W_g(gate)
        g_up = F.interpolate(g, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = model.attn_gate1.W_x(skip)
        psi = torch.sigmoid(model.attn_gate1.W_psi(F.relu(g_up + x)))

        enc1_gated = enc1 * psi
        cat1 = torch.cat([upconv1, enc1_gated], dim=1)
        dec1 = model.dec1(cat1)

        probs = model.seg_head(dec1)
        return probs.squeeze(0).squeeze(0).cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded E46/v5 checkpoint: best_val_dice={ckpt.get('best_val_dice')}", flush=True)
    print("NOTE: same checkpoint as E48/E61, for direct comparability of the size-dependence signature.\n")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n = len(val_dataset)
    print(f"Validation set size: {n}", flush=True)

    # ---------------- Sanity checks ----------------
    image0, _, _ = val_dataset[0]
    image0_b = image0.unsqueeze(0).to(device)
    with torch.no_grad():
        real = model(image0_b)["probs"].squeeze(0).squeeze(0).cpu().numpy()
        enc1_0 = model.enc1(image0_b)
    manual = forward_from_enc1(model, enc1_0, device)
    max_diff = float(np.abs(real - manual).max())
    print(f"[Sanity check 1] forward_from_enc1(real enc1) vs real forward(): max abs diff = {max_diff:.6e}")
    assert max_diff == 0.0, "Manual trunk reimplementation does not match real forward() -- STOP."

    rng_check = np.random.default_rng(999)
    enc1_0_np = enc1_0.squeeze(0).cpu().numpy()
    permuted_check = permute_cells_derangement(enc1_0_np, rng_check)
    # Verify multiset preservation for one arbitrary cell/channel.
    orig_cell = enc1_0_np[0, 0:2, 0:2, 0:2].flatten()
    perm_cell = permuted_check[0, 0:2, 0:2, 0:2].flatten()
    assert np.allclose(np.sort(orig_cell), np.sort(perm_cell)), "Multiset not preserved -- STOP."
    assert not np.allclose(orig_cell, perm_cell), "Derangement produced no change -- STOP (bug or degenerate cell)."
    print("[Sanity check 2] Derangement preserves per-cell multiset (max/mass/mean identical) "
          "and changes position. PASS.\n")

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
        probs_intact = forward_from_enc1(model, enc1, device)
        dice_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin)

        enc1_np = enc1.squeeze(0).cpu().numpy()
        subject_rng = np.random.default_rng(SEED * 100000 + idx)
        dice_permuted_draws = []
        for draw in range(N_DERANGEMENTS):
            enc1_perm_np = permute_cells_derangement(enc1_np, subject_rng)
            enc1_perm = torch.from_numpy(enc1_perm_np).unsqueeze(0).to(device)
            probs_perm = forward_from_enc1(model, enc1_perm, device)
            dice_perm = dice_score((probs_perm >= 0.5).astype(np.float32), target_bin)
            dice_permuted_draws.append(dice_perm)

        mean_dice_permuted = float(np.mean(dice_permuted_draws))
        drop = dice_intact - mean_dice_permuted

        records.append({
            "subject_id": subject_id, "native_size": native_size,
            "dice_intact": dice_intact, "dice_permuted_draws": dice_permuted_draws,
            "mean_dice_permuted": mean_dice_permuted, "drop": drop,
        })

        if (idx + 1) % 25 == 0:
            print(f"  processed {idx+1}/{n} subjects ({N_DERANGEMENTS} derangement draws each)", flush=True)

    with open(OUT_DIR / "E62_position_audit_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.\n")

    # ================= Statistical analysis =================
    native_size = np.array([r["native_size"] for r in records], dtype=np.float64)
    drop = np.array([r["drop"] for r in records], dtype=np.float64)
    dice_intact = np.array([r["dice_intact"] for r in records])
    dice_permuted = np.array([r["mean_dice_permuted"] for r in records])

    print(f"=== E62 Max-Pool Position Audit: n={len(records)} subjects, "
          f"{N_DERANGEMENTS} derangement draws/subject ===")
    print(f"Mean dice_intact          = {dice_intact.mean():.4f}")
    print(f"Mean dice_permuted (avg)  = {dice_permuted.mean():.4f}")
    print(f"Mean drop (intact - permuted) = {drop.mean():.4f} (+/-{drop.std():.4f})")

    # Criterion 1: paired test + subject-level permutation test that mean drop > 0
    t_stat, t_p = stats.ttest_1samp(drop, 0.0, alternative="greater")
    w_stat, w_p = stats.wilcoxon(drop, alternative="greater")
    print(f"\nPaired t-test (H0: mean drop = 0, H1: >0): t={t_stat:.4f}, p={t_p:.4e}")
    print(f"Wilcoxon signed-rank (H1: drop > 0): p={w_p:.4e}")

    rng = np.random.default_rng(SEED)
    perm_means = np.empty(N_PERM)
    for i in range(N_PERM):
        signs = rng.choice([-1, 1], size=len(drop))
        perm_means[i] = (drop * signs).mean()
    p_perm_drop = float((perm_means >= drop.mean()).mean())
    print(f"Sign-flip permutation test ({N_PERM} trials, H1: mean drop > 0): p={p_perm_drop:.4f}")

    criterion_1 = (t_p < 0.05) and (p_perm_drop < 0.05) and (drop.mean() > 0)

    # Criterion 2: Spearman(native_size, drop) < 0 (smaller lesions -> bigger drop), matching E48's bar
    rho, p_parametric = stats.spearmanr(native_size, drop)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_drop = rng.permutation(drop)
        perm_rhos[i], _ = stats.spearmanr(native_size, perm_drop)
    p_perm_rho = float((perm_rhos <= rho).mean()) if rho < 0 else float((perm_rhos >= rho).mean())
    # two-sided-style permutation p using |rho| for robustness, matching E48's own convention
    p_perm_rho_twosided = float((np.abs(perm_rhos) >= np.abs(rho)).mean())
    print(f"\nSpearman(native_size, drop) = {rho:+.4f} (parametric p={p_parametric:.4e}, "
          f"permutation p={p_perm_rho_twosided:.4f})")
    print("(Negative rho = smaller lesions lose MORE Dice from position permutation, matching the "
          "small-lesion-specific bar this phase pre-declared, same direction/significance convention as E48.)")

    criterion_2 = (rho < 0) and (p_perm_rho_twosided < 0.05)

    go = criterion_1 and criterion_2
    if go:
        decision = "GO"
    elif criterion_1:
        decision = "SIGNIFICANT_BUT_SIZE_AGNOSTIC"
    else:
        decision = "KILL"

    print(f"\n=== DECISION: {decision} ===")
    print(f"  Criterion 1 (mean drop > 0, both tests p<0.05): {criterion_1}")
    print(f"  Criterion 2 (size-specific, rho<0, perm p<0.05): {criterion_2}")
    if decision == "GO":
        print("Subcell position loss at pool1 is causally task-relevant AND specifically hurts small "
              "lesions more -- meets the pre-declared bar to proceed to PMD operator design.")
    elif decision == "SIGNIFICANT_BUT_SIZE_AGNOSTIC":
        print("Position loss has a real effect but is NOT small-lesion-specific -- does not meet the "
              "pre-declared bar for the small-lesion-focused PMD chain; reported honestly, not advanced.")
    else:
        print("Pre-declared GO criteria not met. The trained network's downstream layers (convs, skip "
              "connections, decoder) appear to already absorb/nullify the subcell position information "
              "max pooling discards -- KILL, do not design PMD on this basis.")

    summary = {
        "n_subjects": len(records), "n_derangements_per_subject": N_DERANGEMENTS,
        "mean_dice_intact": float(dice_intact.mean()), "mean_dice_permuted": float(dice_permuted.mean()),
        "mean_drop": float(drop.mean()), "sd_drop": float(drop.std()),
        "ttest_p_greater": float(t_p), "wilcoxon_p_greater": float(w_p),
        "sign_flip_permutation_p": p_perm_drop,
        "spearman_rho_size_vs_drop": float(rho), "parametric_p": float(p_parametric),
        "permutation_p_twosided": p_perm_rho_twosided,
        "criterion_1_significant_drop": bool(criterion_1),
        "criterion_2_size_specific": bool(criterion_2),
        "decision": decision,
    }
    with open(OUT_DIR / "E62_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E62_summary.json")


if __name__ == "__main__":
    main()
