"""
Phase E113: causal do(X) test of whether the DECODER ITSELF depends on the
bottleneck's channel-identity organization for necessity-related behavior,
or whether E110/E111's finding is only a passively-readable correlate.

BACKGROUND: E109 showed an EXTERNAL PROBE can linearly decode the network's
own future E48-style causal ablation-sensitivity (N_b) from its frozen
bottleneck, held out (rho=0.633-0.850 across runs). E110 showed this
readable signal is carried predominantly by CHANNEL IDENTITY (channel
permutation collapses it, spatial permutation only partially degrades it).
E111 showed the signal is DISTRIBUTED/REDUNDANT across many channels, not
concentrated or reliant on relational structure.

THE GAP THIS PHASE CLOSES: all of E109-E111 tested whether a SEPARATE,
freshly-TRAINED readout can recover N_b from the (possibly scrambled)
representation. None of them tested whether the network's OWN DECODER --
the actual segmentation pathway, never retrained -- causally DEPENDS on
this channel-identity structure for its real behavior. A signal can be
linearly decodable by an external probe while being causally inert to the
system that produced it (a real, common failure mode in interpretability
work -- "explanation correlation alone is insufficient," per this
project's own Section 7 causal-validation rule, already applied in
E106/E107 to a different candidate).

PRE-REGISTERED HYPOTHESIS: if channel-identity organization is something
the DECODER itself relies on for necessity-related behavior (not merely a
passively-readable correlate), then real, per-subject Dice damage from
CHANNEL-SHUFFLING the bottleneck before decoding should correlate with the
ALREADY-ESTABLISHED N_b (E48-style full-ablation Dice drop) MORE STRONGLY
than damage from SPATIAL-SHUFFLING it -- directly extending E110's own
probe-based readout finding (channel shuffle collapses external
decodability, spatial shuffle only partially degrades it) into the
CAUSAL/DECODER domain: does the REAL, never-retrained decoder's own
behavior show the same asymmetry?

CONTROL DESIGN (the causal core of this test) -- REVISED after a smoke-test
finding: the ORIGINAL design used magnitude-matched synthetic Gaussian
noise as the control (L2 perturbation norm matched to each subject's own
channel-shuffle draw). A 10-subject smoke test found this control was NOT
comparably damaging DESPITE matching L2 norm exactly (mean Delta_noise
approx 0.0008, essentially zero, vs mean Delta_shuffle approx 0.177) --
caught and NOT accepted at face value. Diagnosed cause: channel-shuffling
relocates large, structured, REAL activation values into decoder weight
slots tuned to expect different content, while random Gaussian noise of
the same aggregate L2 norm is diffuse and comparatively inert against
those same learned weights -- matching raw perturbation norm does not
equalize actual functional damage between a structured-but-misplaced
perturbation and an unstructured one. REPLACED with E110's own
already-validated SPATIAL_SHUFFLE operator as the comparison arm instead:
both channel_shuffle and spatial_shuffle relocate REAL, structured
activation values (never synthetic noise), making them a genuinely
comparable pair, and this reuses E110's own exact operators rather than
introducing a new, unvalidated control.

METHOD:
  1. Frozen canonical checkpoint (v5/E46, val_dice=0.9101624600589275,
     asserted). All 125 validation subjects.
  2. For each subject: compute N_b via the exact E48/E109/E110-verified
     ablation construction (bottleneck zeroed before BOTH upconv3 and the
     attention gate).
  3. Compute Delta_shuffle: real forward pass with the bottleneck's 256
     channels permuted (E110's channel_shuffle operator) fed to BOTH
     upconv3 and the gate (matching the ablation convention above, not
     just the decode path) -- Dice(intact) - Dice(shuffled). Averaged
     over N_DRAWS independent permutations per subject.
  4. Compute Delta_spatial: for an independently-drawn spatial
     permutation (E110's spatial_shuffle operator -- same shared
     permutation applied identically across all 256 channels, preserving
     cross-channel co-activation at each new position), fed the same way
     -- Dice(intact) - Dice(spatially-shuffled). Averaged over N_DRAWS.
  5. Compare Spearman(Delta_shuffle, N_b) vs Spearman(Delta_spatial, N_b),
     both raw and as PARTIAL correlations controlling for native_size
     (this project's own standard confound check throughout E48-E111).
     Test whether the DIFFERENCE in correlation strength is itself
     significant via a paired bootstrap over subjects (resampling
     subjects with replacement, recomputing both correlations each time,
     reporting what fraction of bootstrap draws show channel-shuffle
     winning).

PRE-DECLARED DECISION RULE (stated before running):
  - CHANNEL-IDENTITY CAUSALLY IMPLICATED: Spearman(Delta_shuffle, N_b) is
    significantly larger than Spearman(Delta_spatial, N_b) (bootstrap:
    >=95% of resamples favor channel-shuffle), AND this survives
    controlling for native_size (partial correlation gap in the same
    direction) -- directly mirroring E110's own probe-based asymmetry
    (channel destroys decodability, spatial only partially degrades it),
    now shown in the REAL decoder's own causal behavior, not just an
    external readout. This would mean the decoder's real behavior
    specifically depends on channel-identity organization in proportion
    to necessity.
  - PASSIVE CORRELATE ONLY (genuine, informative null): no significant
    difference between channel and spatial shuffle correlations with
    N_b -- both disrupt roughly in proportion to N_b regardless of which
    axis is scrambled, meaning the external-probe asymmetry found in
    E110 does NOT carry over to the decoder's own real causal behavior.
    E110/E111's finding would stand as a real, externally-decodable, but
    (for this specific causal question) inconclusive-or-inert
    representational fact for the network's own downstream use.
  - Report honestly regardless of outcome, per this project's own
    established discipline -- do not chain further hypotheses if this
    comes back null (matching the "one re-test" discipline already
    applied to FWL/E72 and others).
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
N_DRAWS = 5  # independent channel/spatial permutation draws per subject, averaged

CKPT_PATH = (project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs"
             / "AttnGate_seed0" / "checkpoints" / "best.pth")
EXPECTED_VAL_DICE = 0.9101624600589275


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    if denom == 0:
        return 1.0
    return float(2 * tp / denom)


def fractional_occupancy_64(seg_binary_native, shape=(64, 64, 64)):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=shape, mode="area").squeeze().numpy()
    return frac


def decode_from_bottleneck(model, bottleneck, enc1, enc2, enc3):
    """Shared decode logic (identical to E48/E109/E110's verified
    construction): the SAME bottleneck tensor is used for both upconv3
    AND the attention gate -- whatever intervention was applied to it
    (zeroing, shuffling, noising) is applied consistently to both use
    sites, not just the decode path."""
    upconv3 = model.upconv3(bottleneck)
    cat3 = torch.cat([upconv3, enc3], dim=1)
    dec3 = model.dec3(cat3)

    upconv2 = model.upconv2(dec3)
    cat2 = torch.cat([upconv2, enc2], dim=1)
    dec2 = model.dec2(cat2)

    upconv1 = model.upconv1(dec2)

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
    return probs


def get_trunk(model, image):
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)
    return enc1, enc2, enc3, bottleneck


def channel_shuffle(bottleneck, rng):
    """bottleneck: (1, 256, 8, 8, 8) tensor (batch dim=1). Returns a new
    tensor with a fresh random permutation of the channel axis, and the
    permutation used (so the exact same perturbation norm can be matched)."""
    c = bottleneck.shape[1]
    perm = rng.permutation(c)
    perm_t = torch.as_tensor(perm, dtype=torch.long, device=bottleneck.device)
    shuffled = bottleneck[:, perm_t]
    return shuffled, perm


def spatial_shuffle(bottleneck, rng):
    """bottleneck: (1, 256, 8, 8, 8). Flattens spatial dims, applies ONE
    shared random permutation to all channels identically, reshapes
    back -- preserves cross-channel co-activation at each new position.
    Identical operator to E110's own verified spatial_shuffle, adapted
    for a batch dim of 1."""
    b, c, d, h, w = bottleneck.shape
    flat = bottleneck.reshape(b, c, d * h * w)
    perm = rng.permutation(d * h * w)
    perm_t = torch.as_tensor(perm, dtype=torch.long, device=bottleneck.device)
    flat_shuffled = flat[:, :, perm_t]
    return flat_shuffled.reshape(b, c, d, h, w)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    val_dice = ckpt.get("best_val_dice") or ckpt.get("val_dice")
    print(f"Loaded canonical checkpoint: val_dice={val_dice}")
    assert abs(float(val_dice) - EXPECTED_VAL_DICE) < 1e-6, "Checkpoint identity check FAILED"
    print("[Sanity check] checkpoint identity PASS.\n")

    val_ds = BraTSDataset(root_dir="Dataset/Training", split="val", val_split=0.1,
                          target_shape=(64, 64, 64), normalize=True)
    print(f"Validation subjects: {len(val_ds)}")

    records = []
    rng_global = np.random.default_rng(SEED)

    for idx in range(len(val_ds)):
        image, mask, sid = val_ds[idx]
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_ds.subject_dirs[idx]
        seg_path = Path(subject_dir) / f"{sid}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_size = int(seg_binary_native.sum())
        target_bin = (fractional_occupancy_64(seg_binary_native) > 0.5).astype(np.float32)

        with torch.no_grad():
            enc1, enc2, enc3, bottleneck = get_trunk(model, image_b)

            probs_intact = decode_from_bottleneck(model, bottleneck, enc1, enc2, enc3)
            dice_intact = dice_score((probs_intact[0, 0].cpu().numpy() >= 0.5).astype(np.float32), target_bin)

            probs_ablated = decode_from_bottleneck(model, torch.zeros_like(bottleneck), enc1, enc2, enc3)
            dice_ablated = dice_score((probs_ablated[0, 0].cpu().numpy() >= 0.5).astype(np.float32), target_bin)
            n_b = dice_intact - dice_ablated

            shuffle_dices = []
            spatial_dices = []
            perturbation_norms = []
            for draw in range(N_DRAWS):
                draw_rng = np.random.default_rng(rng_global.integers(0, 2**31 - 1))
                shuffled_bn, perm = channel_shuffle(bottleneck, draw_rng)
                pert_norm = (shuffled_bn - bottleneck).norm().item()
                perturbation_norms.append(pert_norm)

                probs_shuffle = decode_from_bottleneck(model, shuffled_bn, enc1, enc2, enc3)
                dice_shuffle = dice_score((probs_shuffle[0, 0].cpu().numpy() >= 0.5).astype(np.float32), target_bin)
                shuffle_dices.append(dice_shuffle)

                spatial_rng = np.random.default_rng(rng_global.integers(0, 2**31 - 1))
                spatial_bn = spatial_shuffle(bottleneck, spatial_rng)
                probs_spatial = decode_from_bottleneck(model, spatial_bn, enc1, enc2, enc3)
                dice_spatial = dice_score((probs_spatial[0, 0].cpu().numpy() >= 0.5).astype(np.float32), target_bin)
                spatial_dices.append(dice_spatial)

        mean_dice_shuffle = float(np.mean(shuffle_dices))
        mean_dice_spatial = float(np.mean(spatial_dices))
        delta_shuffle = dice_intact - mean_dice_shuffle
        delta_spatial = dice_intact - mean_dice_spatial

        records.append({
            "subject_id": sid, "native_size": native_size,
            "dice_intact": dice_intact, "dice_ablated": dice_ablated, "n_b": n_b,
            "mean_dice_shuffle": mean_dice_shuffle, "delta_shuffle": delta_shuffle,
            "mean_dice_spatial": mean_dice_spatial, "delta_spatial": delta_spatial,
            "mean_perturbation_norm": float(np.mean(perturbation_norms)),
        })

        if (idx + 1) % 25 == 0:
            print(f"  labeled {idx+1}/{len(val_ds)}", flush=True)

    with open(OUT_DIR / "E113_per_subject_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved per-subject table ({len(records)} subjects).\n")

    # ================= Analysis =================
    n_b_arr = np.array([r["n_b"] for r in records])
    delta_shuffle_arr = np.array([r["delta_shuffle"] for r in records])
    delta_spatial_arr = np.array([r["delta_spatial"] for r in records])
    size_arr = np.array([r["native_size"] for r in records])

    print(f"Mean N_b: {n_b_arr.mean():.4f}  std: {n_b_arr.std():.4f}")
    print(f"Mean Delta_shuffle (channel): {delta_shuffle_arr.mean():.4f}  std: {delta_shuffle_arr.std():.4f}")
    print(f"Mean Delta_spatial: {delta_spatial_arr.mean():.4f}  std: {delta_spatial_arr.std():.4f}")
    print(f"(Both operators use REAL, structured activation values, unlike the "
          f"rejected synthetic-noise control -- means need not match exactly, "
          f"but both should be substantial, non-trivial disruptions.)\n")

    rho_shuffle, p_shuffle = stats.spearmanr(delta_shuffle_arr, n_b_arr)
    rho_spatial, p_spatial = stats.spearmanr(delta_spatial_arr, n_b_arr)
    print(f"Spearman(Delta_shuffle, N_b) = {rho_shuffle:+.4f} (p={p_shuffle:.4e})")
    print(f"Spearman(Delta_spatial, N_b) = {rho_spatial:+.4f} (p={p_spatial:.4e})")
    print(f"Raw difference (channel - spatial): {rho_shuffle - rho_spatial:+.4f}\n")

    # Partial correlations controlling for native_size (log-transformed,
    # matching this project's own established convention throughout E48-E111).
    def partial_spearman(x, y, control):
        rx, ry, rc = stats.rankdata(x), stats.rankdata(y), stats.rankdata(control)
        Xc = np.column_stack([np.ones(len(rc)), rc])
        bx, *_ = np.linalg.lstsq(Xc, rx, rcond=None); resid_x = rx - Xc @ bx
        by, *_ = np.linalg.lstsq(Xc, ry, rcond=None); resid_y = ry - Xc @ by
        r, p = stats.pearsonr(resid_x, resid_y)
        return r, p

    partial_rho_shuffle, partial_p_shuffle = partial_spearman(delta_shuffle_arr, n_b_arr, size_arr)
    partial_rho_spatial, partial_p_spatial = partial_spearman(delta_spatial_arr, n_b_arr, size_arr)
    print(f"Partial Spearman(Delta_shuffle, N_b | size) = {partial_rho_shuffle:+.4f} (p={partial_p_shuffle:.4e})")
    print(f"Partial Spearman(Delta_spatial, N_b | size) = {partial_rho_spatial:+.4f} (p={partial_p_spatial:.4e})")
    print(f"Partial difference (channel - spatial): {partial_rho_shuffle - partial_rho_spatial:+.4f}\n")

    # Paired bootstrap over subjects: resample subjects with replacement,
    # recompute BOTH correlations each time, report fraction of resamples
    # where channel-shuffle's correlation with N_b exceeds spatial-shuffle's.
    rng_boot = np.random.default_rng(SEED)
    n_subjects = len(records)
    boot_diffs = np.empty(N_PERM)
    for i in range(N_PERM):
        idx_boot = rng_boot.integers(0, n_subjects, size=n_subjects)
        rho_c, _ = stats.spearmanr(delta_shuffle_arr[idx_boot], n_b_arr[idx_boot])
        rho_s, _ = stats.spearmanr(delta_spatial_arr[idx_boot], n_b_arr[idx_boot])
        boot_diffs[i] = rho_c - rho_s
    frac_shuffle_wins = float((boot_diffs > 0).mean())
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    print(f"Bootstrap (n={N_PERM}): fraction of resamples where channel-shuffle's correlation "
          f"with N_b exceeds spatial-shuffle's: {frac_shuffle_wins:.4f}")
    print(f"Bootstrap 95% CI of (rho_channel - rho_spatial): [{ci_low:+.4f}, {ci_high:+.4f}]\n")

    # ================= Pre-declared verdict =================
    channel_identity_implicated = (frac_shuffle_wins >= 0.95) and (partial_rho_shuffle - partial_rho_spatial > 0)
    verdict = "CHANNEL_IDENTITY_CAUSALLY_IMPLICATED" if channel_identity_implicated else "PASSIVE_CORRELATE_ONLY_GENUINE_NULL"

    print(f"=== FINAL VERDICT: {verdict} ===")
    print("(Convenience label only -- inspect the actual rho/CI numbers above directly, "
          "per this project's own established convention of not trusting an "
          "auto-classifier's label without checking the underlying numbers.)")

    summary = {
        "checkpoint": str(CKPT_PATH), "val_dice_check": val_dice,
        "n_subjects": n_subjects, "n_draws": N_DRAWS,
        "mean_n_b": float(n_b_arr.mean()),
        "mean_delta_shuffle": float(delta_shuffle_arr.mean()),
        "mean_delta_spatial": float(delta_spatial_arr.mean()),
        "rho_shuffle_nb": float(rho_shuffle), "p_shuffle_nb": float(p_shuffle),
        "rho_spatial_nb": float(rho_spatial), "p_spatial_nb": float(p_spatial),
        "partial_rho_shuffle_nb": float(partial_rho_shuffle), "partial_p_shuffle_nb": float(partial_p_shuffle),
        "partial_rho_spatial_nb": float(partial_rho_spatial), "partial_p_spatial_nb": float(partial_p_spatial),
        "bootstrap_frac_shuffle_wins": frac_shuffle_wins,
        "bootstrap_ci_diff": [float(ci_low), float(ci_high)],
        "final_verdict_CONVENIENCE_LABEL_ONLY": verdict,
    }
    with open(OUT_DIR / "E113_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E113_summary.json")


if __name__ == "__main__":
    main()
