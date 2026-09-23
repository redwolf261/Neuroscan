"""
Phase E114: split-intervention correction of E113's channel-vs-spatial
causal test, isolating a real design confound the user correctly flagged
before trusting E113's result.

BACKGROUND: E113 found channel-shuffle damage correlates NEGATIVELY with
N_b (rho=-0.355, p=4.9e-05) while spatial-shuffle damage correlates
POSITIVELY (rho=+0.206, p=0.021) -- a significant sign reversal, opposite
E113's own pre-registered hypothesis, with bootstrap 0/1000 favoring
channel over spatial. Before interpreting this causally, audited the
design for a fault. Verified NOT the cause: both scrambling operators
(unit-tested directly, matching E110's own verified semantics), tensor
aliasing (verified neither operator's output shares memory with the
pristine bottleneck, so sequential channel-then-spatial calls each see a
clean original), and the decode reconstruction itself (bit-exact to the
real model, max abs diff 0.0).

THE REAL CONFOUND, IDENTIFIED: E113's `decode_from_bottleneck` feeds the
(possibly-shuffled) bottleneck to BOTH the decode path (upconv3) AND the
attention gate, matching E48/E109's own established ablation convention.
But the gate's own architecture applies TRILINEAR INTERPOLATION to
upsample the bottleneck-derived gate signal to enc1's resolution --
an operation that assumes spatial coherence (neighboring grid indices
correspond to physically nearby locations). Channel-shuffle preserves
this coherence (only WHICH channel's value sits at each position changes,
not the spatial grid itself); spatial-shuffle destroys it directly. This
makes the two conditions structurally asymmetric with respect to the
gate's interpolation step, independent of any real "channel-identity vs
position" informational difference -- a plausible, unaccounted-for driver
of E113's sign reversal that has nothing to do with the intended
hypothesis.

THE FIX: split the bottleneck's two use-sites. Run each intervention
(channel-shuffle, spatial-shuffle) on the DECODE PATH ONLY, while the
GATE receives the intact, unperturbed bottleneck -- removing the
interpolation-coherence confound entirely. Run BOTH the split (decode-only)
and original (both-intervened, E113's exact construction) versions in the
SAME script on the SAME subjects, so the two can be directly compared:
  - If the sign-reversal SURVIVES decode-only intervention: the effect is
    a real decode-path phenomenon, not a gate-interpolation artifact --
    E113's finding stands (with the gate-confound explicitly ruled out).
  - If the sign-reversal DISAPPEARS or REVERSES AGAIN under decode-only
    intervention: E113's result was driven by the gate's interpolation
    asymmetry, not a genuine causal fact about the decoder's use of
    channel-identity vs spatial information -- E113's causal claim is
    withdrawn, and only the (real, still-standing) E110/E111 readout-level
    finding remains.

N_b's OWN definition is kept EXACTLY as established throughout E48-E113
(full bottleneck zeroed for BOTH decode and gate simultaneously) --
changing it would break comparability with the entire project's causal-
necessity literature. Only the shuffle/spatial intervention arms are
split; N_b itself is the same already-validated quantity used everywhere
else.

PRE-DECLARED DECISION RULE (stated before running):
  For the DECODE-ONLY arm: does Spearman(Delta_shuffle_decodeonly, N_b)
  differ from Spearman(Delta_spatial_decodeonly, N_b) in the SAME
  direction as E113's both-intervened result (channel negative, spatial
  positive), with the same bootstrap-based significance test (>=95% or
  <=5% of resamples favoring one direction)?
  - SAME DIRECTION, SIGNIFICANT: E113's finding is a real decode-path
    causal fact, confound ruled out.
  - NO SIGNIFICANT DIFFERENCE, or REVERSED: E113's finding was (at least
    partly) a gate-interpolation artifact; withdraw the causal claim.
  Report the BOTH-INTERVENED arm's results too (recomputed fresh here,
  not reused from E113, to confirm reproducibility) as the reference
  point for comparison.
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
N_DRAWS = 5

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


def decode_split(model, bottleneck_decode, bottleneck_gate, enc1, enc2, enc3):
    """SPLIT reconstruction: the decode path (upconv3 onward) uses
    bottleneck_decode; the attention gate uses bottleneck_gate
    INDEPENDENTLY. Passing the SAME tensor for both reproduces E113's
    (and E48/E109's) original both-intervened construction exactly."""
    upconv3 = model.upconv3(bottleneck_decode)
    cat3 = torch.cat([upconv3, enc3], dim=1)
    dec3 = model.dec3(cat3)

    upconv2 = model.upconv2(dec3)
    cat2 = torch.cat([upconv2, enc2], dim=1)
    dec2 = model.dec2(cat2)

    upconv1 = model.upconv1(dec2)

    gate = bottleneck_gate
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


def channel_shuffle(bottleneck, rng):
    c = bottleneck.shape[1]
    perm = rng.permutation(c)
    perm_t = torch.as_tensor(perm, dtype=torch.long, device=bottleneck.device)
    return bottleneck[:, perm_t]


def spatial_shuffle(bottleneck, rng):
    b, c, d, h, w = bottleneck.shape
    flat = bottleneck.reshape(b, c, d * h * w)
    perm = rng.permutation(d * h * w)
    perm_t = torch.as_tensor(perm, dtype=torch.long, device=bottleneck.device)
    flat_shuffled = flat[:, :, perm_t]
    return flat_shuffled.reshape(b, c, d, h, w)


def analyze(name, delta_shuffle_arr, delta_spatial_arr, n_b_arr, size_arr):
    print(f"\n{'='*70}\n=== {name} ===\n{'='*70}")
    print(f"Mean Delta_shuffle: {delta_shuffle_arr.mean():.4f}  std: {delta_shuffle_arr.std():.4f}")
    print(f"Mean Delta_spatial: {delta_spatial_arr.mean():.4f}  std: {delta_spatial_arr.std():.4f}")

    rho_shuffle, p_shuffle = stats.spearmanr(delta_shuffle_arr, n_b_arr)
    rho_spatial, p_spatial = stats.spearmanr(delta_spatial_arr, n_b_arr)
    print(f"Spearman(Delta_shuffle, N_b) = {rho_shuffle:+.4f} (p={p_shuffle:.4e})")
    print(f"Spearman(Delta_spatial, N_b) = {rho_spatial:+.4f} (p={p_spatial:.4e})")
    print(f"Raw difference (channel - spatial): {rho_shuffle - rho_spatial:+.4f}")

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

    rng_boot = np.random.default_rng(SEED)
    n_subjects = len(n_b_arr)
    boot_diffs = np.empty(N_PERM)
    for i in range(N_PERM):
        idx_boot = rng_boot.integers(0, n_subjects, size=n_subjects)
        rho_c, _ = stats.spearmanr(delta_shuffle_arr[idx_boot], n_b_arr[idx_boot])
        rho_s, _ = stats.spearmanr(delta_spatial_arr[idx_boot], n_b_arr[idx_boot])
        boot_diffs[i] = rho_c - rho_s
    frac_shuffle_wins = float((boot_diffs > 0).mean())
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    print(f"Bootstrap (n={N_PERM}): fraction favoring channel over spatial: {frac_shuffle_wins:.4f}")
    print(f"Bootstrap 95% CI of (rho_channel - rho_spatial): [{ci_low:+.4f}, {ci_high:+.4f}]")

    return {
        "mean_delta_shuffle": float(delta_shuffle_arr.mean()),
        "mean_delta_spatial": float(delta_spatial_arr.mean()),
        "rho_shuffle_nb": float(rho_shuffle), "p_shuffle_nb": float(p_shuffle),
        "rho_spatial_nb": float(rho_spatial), "p_spatial_nb": float(p_spatial),
        "partial_rho_shuffle_nb": float(partial_rho_shuffle), "partial_p_shuffle_nb": float(partial_p_shuffle),
        "partial_rho_spatial_nb": float(partial_rho_spatial), "partial_p_spatial_nb": float(partial_p_spatial),
        "bootstrap_frac_channel_wins": frac_shuffle_wins,
        "bootstrap_ci_diff": [float(ci_low), float(ci_high)],
    }


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

    # ================= Sanity check: split reconstruction bit-exact when both args equal =================
    val_ds = BraTSDataset(root_dir="Dataset/Training", split="val", val_split=0.1,
                          target_shape=(64, 64, 64), normalize=True)
    image0, _, _ = val_ds[0]
    image0_b = image0.unsqueeze(0).to(device)
    real_out = model(image0_b)
    real_probs = real_out["probs"]
    enc1_0, enc2_0, enc3_0, bn_0 = get_trunk(model, image0_b)
    with torch.no_grad():
        manual_probs = decode_split(model, bn_0, bn_0, enc1_0, enc2_0, enc3_0)
    max_diff = (real_probs - manual_probs).abs().max().item()
    print(f"[Sanity check] split reconstruction (both args = intact bottleneck) vs real forward(): "
          f"max abs diff = {max_diff:.6e}")
    assert max_diff < 1e-5, "Split reconstruction mismatch -- STOP."
    print("[Sanity check] PASS.\n")

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

            probs_intact = decode_split(model, bottleneck, bottleneck, enc1, enc2, enc3)
            dice_intact = dice_score((probs_intact[0, 0].cpu().numpy() >= 0.5).astype(np.float32), target_bin)

            # N_b: EXACT established definition, both use-sites zeroed together.
            zero_bn = torch.zeros_like(bottleneck)
            probs_ablated = decode_split(model, zero_bn, zero_bn, enc1, enc2, enc3)
            dice_ablated = dice_score((probs_ablated[0, 0].cpu().numpy() >= 0.5).astype(np.float32), target_bin)
            n_b = dice_intact - dice_ablated

            both_shuffle_dices, both_spatial_dices = [], []
            decodeonly_shuffle_dices, decodeonly_spatial_dices = [], []

            for draw in range(N_DRAWS):
                ch_rng = np.random.default_rng(rng_global.integers(0, 2**31 - 1))
                sp_rng = np.random.default_rng(rng_global.integers(0, 2**31 - 1))
                shuffled_bn = channel_shuffle(bottleneck, ch_rng)
                spatial_bn = spatial_shuffle(bottleneck, sp_rng)

                # BOTH-INTERVENED arm (E113's exact construction, recomputed fresh here)
                probs_bs = decode_split(model, shuffled_bn, shuffled_bn, enc1, enc2, enc3)
                both_shuffle_dices.append(dice_score((probs_bs[0, 0].cpu().numpy() >= 0.5).astype(np.float32), target_bin))
                probs_bp = decode_split(model, spatial_bn, spatial_bn, enc1, enc2, enc3)
                both_spatial_dices.append(dice_score((probs_bp[0, 0].cpu().numpy() >= 0.5).astype(np.float32), target_bin))

                # DECODE-ONLY arm: gate receives the INTACT bottleneck always
                probs_ds = decode_split(model, shuffled_bn, bottleneck, enc1, enc2, enc3)
                decodeonly_shuffle_dices.append(dice_score((probs_ds[0, 0].cpu().numpy() >= 0.5).astype(np.float32), target_bin))
                probs_dp = decode_split(model, spatial_bn, bottleneck, enc1, enc2, enc3)
                decodeonly_spatial_dices.append(dice_score((probs_dp[0, 0].cpu().numpy() >= 0.5).astype(np.float32), target_bin))

        records.append({
            "subject_id": sid, "native_size": native_size,
            "dice_intact": dice_intact, "dice_ablated": dice_ablated, "n_b": n_b,
            "delta_both_shuffle": dice_intact - float(np.mean(both_shuffle_dices)),
            "delta_both_spatial": dice_intact - float(np.mean(both_spatial_dices)),
            "delta_decodeonly_shuffle": dice_intact - float(np.mean(decodeonly_shuffle_dices)),
            "delta_decodeonly_spatial": dice_intact - float(np.mean(decodeonly_spatial_dices)),
        })

        if (idx + 1) % 25 == 0:
            print(f"  labeled {idx+1}/{len(val_ds)}", flush=True)

    with open(OUT_DIR / "E114_per_subject_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved per-subject table ({len(records)} subjects).")

    n_b_arr = np.array([r["n_b"] for r in records])
    size_arr = np.array([r["native_size"] for r in records])

    both_result = analyze(
        "BOTH-INTERVENED (reproduces E113's exact construction)",
        np.array([r["delta_both_shuffle"] for r in records]),
        np.array([r["delta_both_spatial"] for r in records]),
        n_b_arr, size_arr,
    )
    decodeonly_result = analyze(
        "DECODE-ONLY (gate receives INTACT bottleneck -- confound removed)",
        np.array([r["delta_decodeonly_shuffle"] for r in records]),
        np.array([r["delta_decodeonly_spatial"] for r in records]),
        n_b_arr, size_arr,
    )

    print(f"\n{'='*70}\n=== COMPARISON: does the sign-reversal survive decode-only intervention? ===\n{'='*70}")
    both_direction = "channel<spatial" if both_result["rho_shuffle_nb"] < both_result["rho_spatial_nb"] else "channel>spatial"
    decode_direction = "channel<spatial" if decodeonly_result["rho_shuffle_nb"] < decodeonly_result["rho_spatial_nb"] else "channel>spatial"
    print(f"BOTH-INTERVENED direction: {both_direction} "
          f"(bootstrap frac channel-wins={both_result['bootstrap_frac_channel_wins']:.4f})")
    print(f"DECODE-ONLY direction:     {decode_direction} "
          f"(bootstrap frac channel-wins={decodeonly_result['bootstrap_frac_channel_wins']:.4f})")

    decode_significant = (decodeonly_result["bootstrap_frac_channel_wins"] <= 0.05
                          or decodeonly_result["bootstrap_frac_channel_wins"] >= 0.95)
    same_direction = (both_direction == decode_direction)

    if decode_significant and same_direction:
        verdict = "CONFOUND_RULED_OUT_EFFECT_REAL_IN_DECODE_PATH"
    elif not decode_significant:
        verdict = "CONFOUND_CONFIRMED_EFFECT_DISAPPEARS_DECODE_ONLY"
    else:
        verdict = "CONFOUND_CONFIRMED_EFFECT_REVERSES_DECODE_ONLY"

    print(f"\n=== FINAL VERDICT: {verdict} ===")
    print("(Convenience label only -- inspect the actual rho/bootstrap numbers above directly.)")

    summary = {
        "checkpoint": str(CKPT_PATH), "val_dice_check": val_dice,
        "n_subjects": len(records), "n_draws": N_DRAWS,
        "both_intervened": both_result,
        "decode_only": decodeonly_result,
        "final_verdict_CONVENIENCE_LABEL_ONLY": verdict,
    }
    with open(OUT_DIR / "E114_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E114_summary.json")


if __name__ == "__main__":
    main()
