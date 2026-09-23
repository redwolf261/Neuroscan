"""
Phase E80: Decoder-Readout Causality (NO TRAINING, pure inference-time
intervention) -- per the user's design, following E79's confirmation
that enc1 direction's translation vulnerability is absolute-coordinate
binding, not local noise, AND a targeted novelty search that ruled out
generic deformable/alignment mechanisms as the eventual algorithm
(occupied: deformable semantic alignment in medical segmentation skip
connections, cross-attention encoder-decoder fusion).

OPEN QUESTION: E65-E79 all intervened on the ENCODER side of the skip
concatenation (enc1 / its direction component) while leaving the
DECODER'S OWN upsampled feature (upconv1) untouched. This cannot
distinguish two structurally different explanations:
  (i) the vulnerability is an ENCODER REPRESENTATION property -- enc1's
      direction is "keyed" to a position in some absolute sense the
      decoder reads out, regardless of what's on the other side of the
      concatenation.
  (ii) the vulnerability is an ENCODER-DECODER INTERACTION (correspondence)
      property -- what actually matters is the RELATIVE alignment between
      enc1's direction and upconv1's own feature at concatenation; shifting
      either side alone breaks a correspondence the decoder is implicitly
      relying on, but shifting BOTH sides together (preserving their
      relative alignment) should NOT break it.

If (ii) is true, a coherent joint shift of both sides should look nearly
as good as the intact forward pass -- a strong, sharp, falsifiable
prediction that a pure encoder-property story (i) does NOT make (under
(i), shifting enc1's direction should hurt regardless of what upconv1 is
doing, since the "positional key" itself is corrupted independent of its
partner).

METHOD: same split-forward isolation as E64/E74/E75/E78 (E_pool always
real). Four arms, all operating on the DIRECTION component of enc1 only
(magnitude r held fixed throughout, matching E77-E79's own established
discipline) and on upconv1 (the decoder's own pre-concatenation feature,
which has no "magnitude/direction" decomposition of its own significance
here -- it is shifted as a whole tensor, since E78/E79's r/u distinction
was specific to enc1's role, not hypothesized to apply symmetrically to
the decoder path):

  Arm 1 (ENCODER-SIDE ONLY, reference = E78/E79's own S_u):
    translate u_enc1 by +3 voxels, upconv1 untouched.
  Arm 2 (DECODER-SIDE ONLY, new):
    u_enc1 untouched, translate upconv1 by +3 voxels.
  Arm 3 (COHERENT JOINT SHIFT, the decisive test):
    translate BOTH u_enc1 AND upconv1 by the SAME +3 voxel offset --
    preserves their RELATIVE alignment while moving both in absolute
    space.
  Arm 4 (MISMATCH SWEEP, characterizes the relationship if (ii) holds):
    translate u_enc1 by +3, upconv1 by a DIFFERENT offset in
    {0 (=Arm1), +1, +2, +3 (=Arm3), +4, +5} -- does damage scale with the
    RELATIVE mismatch (|3 - offset|) specifically, troughing at offset=3
    where relative alignment is restored?

PRE-DECLARED READING:
  - Arm 3 (coherent shift) recovers MOST of Arm 1's damage -> supports
    (ii), encoder-decoder correspondence/interaction property. The
    intervention should target the CORRESPONDENCE mechanism (how the
    decoder relates the two sides), not either representation alone.
  - Arm 3 does NOT recover Arm 1's damage (stays close to Arm 1 or worse)
    -> supports (i), encoder representation property. The intervention
    should target enc1's OWN positional binding independent of the
    decoder side.
  - Arm 4's damage should trough specifically at the offset matching
    Arm 3 (relative mismatch = 0) if (ii) holds, with damage increasing
    monotonically as |mismatch| grows in either direction -- this is the
    sharpest test, since it's a full curve, not a single point.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e74"))
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e78"))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402
from run_e74_spatial_dependence_audit import translate_volume  # noqa: E402
from run_e78_magnitude_direction_decomposition import decompose, recompose  # noqa: E402

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 0
N_PERM = 1000
N_BOOT = 2000
BASE_OFFSET = 3  # matches E65/E74/E75/E78/E79's own reference offset
MISMATCH_SWEEP_OFFSETS = [0, 1, 2, 3, 4, 5]  # upconv1's own offset; enc1-direction always shifted by BASE_OFFSET=3

MM_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e70" / "runs"
           / "MM_seed0" / "checkpoints" / "best.pth")


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    return 1.0 if denom == 0 else float(2 * tp / denom)


def forward_dual_shift(model, img_b, enc1_dir_offset, upconv1_offset, device):
    """enc1_dir_offset: voxel shift applied to enc1's DIRECTION component
    only (magnitude r held fixed, recombined via E78's recompose).
    upconv1_offset: voxel shift applied to the decoder's own upconv1
    feature (whole tensor, no r/u decomposition). offset=0 means
    untouched for either. E_pool always uses the network's own real
    enc1 throughout (E64's isolation discipline)."""
    with torch.no_grad():
        enc1_real = model.enc1(img_b)
        pool1 = model.pool1(enc1_real)
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
        upconv1 = model.upconv1(dec2)  # (1,32,64,64,64), decoder's own pre-concat feature

        r_enc1, u_enc1 = decompose(enc1_real.squeeze(0))
        if enc1_dir_offset != 0:
            u_enc1 = translate_volume(u_enc1, enc1_dir_offset)
        e_skip = recompose(r_enc1, u_enc1).unsqueeze(0)

        upconv1_used = upconv1
        if upconv1_offset != 0:
            upconv1_used = translate_volume(upconv1.squeeze(0), upconv1_offset).unsqueeze(0)

        cat1 = torch.cat([upconv1_used, e_skip], dim=1)
        dec1 = model.dec1(cat1)
        probs = model.seg_head(dec1)
    return probs.squeeze(0).squeeze(0).cpu().numpy()


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

    records = []
    for idx in range(len(val_ds)):
        img, msk, sid = val_ds[idx]
        img_b = img.unsqueeze(0).to(device)
        target_bin = (msk.squeeze(0).numpy() > 0.5).astype(np.float32)

        probs_intact = forward_dual_shift(model, img_b, 0, 0, device)
        d_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin)

        # Arm 1: encoder-side only (= E78/E79's S_u reference)
        probs_arm1 = forward_dual_shift(model, img_b, BASE_OFFSET, 0, device)
        d_arm1 = dice_score((probs_arm1 >= 0.5).astype(np.float32), target_bin)

        # Arm 2: decoder-side only
        probs_arm2 = forward_dual_shift(model, img_b, 0, BASE_OFFSET, device)
        d_arm2 = dice_score((probs_arm2 >= 0.5).astype(np.float32), target_bin)

        # Arm 3: coherent joint shift
        probs_arm3 = forward_dual_shift(model, img_b, BASE_OFFSET, BASE_OFFSET, device)
        d_arm3 = dice_score((probs_arm3 >= 0.5).astype(np.float32), target_bin)

        # Arm 4: mismatch sweep (enc1-direction fixed at BASE_OFFSET, upconv1 offset varies)
        sweep_dice = {}
        for upconv1_off in MISMATCH_SWEEP_OFFSETS:
            probs_sweep = forward_dual_shift(model, img_b, BASE_OFFSET, upconv1_off, device)
            sweep_dice[upconv1_off] = dice_score((probs_sweep >= 0.5).astype(np.float32), target_bin)

        records.append({
            "subject_id": sid,
            "D_intact": d_intact,
            "D_arm1_encoder_only": d_arm1, "S_arm1": d_intact - d_arm1,
            "D_arm2_decoder_only": d_arm2, "S_arm2": d_intact - d_arm2,
            "D_arm3_coherent": d_arm3, "S_arm3": d_intact - d_arm3,
            "sweep_dice": {str(k): v for k, v in sweep_dice.items()},
            "sweep_S": {str(k): d_intact - v for k, v in sweep_dice.items()},
        })

        if (idx + 1) % 25 == 0:
            print(f"  processed {idx+1}/{len(val_ds)}", flush=True)

    with open(OUT_DIR / "E80_decoder_readout_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.\n")

    S_arm1 = np.array([r["S_arm1"] for r in records])
    S_arm2 = np.array([r["S_arm2"] for r in records])
    S_arm3 = np.array([r["S_arm3"] for r in records])

    def paired_stats(drop, label):
        t_stat, p_t = stats.ttest_1samp(drop, 0.0)
        w_stat, p_w = stats.wilcoxon(drop)
        rng = np.random.default_rng(SEED)
        signs = rng.choice([-1, 1], size=(N_PERM, len(drop)))
        perm_means = (signs * drop[None, :]).mean(axis=1)
        p_perm = float((np.abs(perm_means) >= np.abs(drop.mean())).mean())
        boot_idx = rng.integers(0, len(drop), size=(N_BOOT, len(drop)))
        boot_means = drop[boot_idx].mean(axis=1)
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        print(f"=== {label}: mean = {drop.mean():.4f} (95% CI [{ci_lo:.4f}, {ci_hi:.4f}]) ===")
        print(f"  t-test p={p_t:.4e}, Wilcoxon p={p_w:.4e}, sign-flip permutation p={p_perm:.4f}\n")
        return {"mean": float(drop.mean()), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
                "ttest_p": float(p_t), "wilcoxon_p": float(p_w), "perm_p": p_perm}

    print("=== ARM RESULTS ===\n")
    stat1 = paired_stats(S_arm1, "Arm 1: encoder-side only (translate enc1 direction, upconv1 untouched)")
    stat2 = paired_stats(S_arm2, "Arm 2: decoder-side only (translate upconv1, enc1 direction untouched)")
    stat3 = paired_stats(S_arm3, "Arm 3: COHERENT joint shift (both translated together, relative alignment preserved)")

    recovery_frac = 1 - (S_arm3.mean() / S_arm1.mean()) if S_arm1.mean() > 0 else float("nan")
    print(f"=== Coherent-shift recovery: does Arm 3 rescue Arm 1's damage? ===")
    print(f"  S_arm1 (encoder-only) = {S_arm1.mean():.4f}")
    print(f"  S_arm3 (coherent)     = {S_arm3.mean():.4f}")
    print(f"  Recovery fraction: {recovery_frac*100:.1f}%\n")

    t_1v3, p_1v3 = stats.ttest_rel(S_arm1, S_arm3)
    w_1v3, pw_1v3 = stats.wilcoxon(S_arm1, S_arm3)
    print(f"  Arm1 vs Arm3 paired comparison: t={t_1v3:.3f}, p={p_1v3:.4e}, Wilcoxon p={pw_1v3:.4e}")

    # Arm 4: mismatch sweep -- mean S by relative mismatch
    print("\n=== Arm 4: mismatch sweep (relative offset = |BASE_OFFSET - upconv1_offset|) ===")
    sweep_means = {}
    for off in MISMATCH_SWEEP_OFFSETS:
        vals = np.array([r["sweep_S"][str(off)] for r in records])
        sweep_means[off] = float(vals.mean())
        rel_mismatch = abs(BASE_OFFSET - off)
        print(f"  upconv1_offset={off} (relative mismatch={rel_mismatch}): mean S = {vals.mean():.4f}")

    troughs_at_zero_mismatch = sweep_means[BASE_OFFSET] == min(sweep_means.values())
    print(f"\n  Troughs at zero relative mismatch (offset={BASE_OFFSET}): {troughs_at_zero_mismatch}")

    if recovery_frac > 0.5 and troughs_at_zero_mismatch:
        reading = ("(ii) ENCODER-DECODER INTERACTION property: coherent joint shift substantially recovers "
                  "the damage, and the mismatch sweep troughs at zero relative offset. The decoder relies on "
                  "RELATIVE correspondence between the two sides, not either side's absolute position alone. "
                  "The intervention should target the correspondence mechanism at the concatenation, not "
                  "enc1's representation in isolation.")
    elif recovery_frac < 0.2:
        reading = ("(i) ENCODER REPRESENTATION property: coherent joint shift does NOT meaningfully recover "
                  "the damage -- enc1's direction is corrupted by translation independent of what the decoder "
                  "side is doing. The intervention should target enc1's own positional binding, not the "
                  "encoder-decoder correspondence.")
    else:
        reading = "MIXED/PARTIAL: some recovery from coherent shift, but not clean -- both factors likely contribute."

    print(f"\n=== READING ===\n{reading}")

    summary = {
        "n_subjects": len(records), "base_offset": BASE_OFFSET,
        "arm1_encoder_only": stat1, "arm2_decoder_only": stat2, "arm3_coherent": stat3,
        "recovery_fraction": float(recovery_frac),
        "arm1_vs_arm3_ttest_p": float(p_1v3), "arm1_vs_arm3_wilcoxon_p": float(pw_1v3),
        "sweep_means": {str(k): v for k, v in sweep_means.items()},
        "troughs_at_zero_mismatch": bool(troughs_at_zero_mismatch),
        "reading": reading,
    }
    with open(OUT_DIR / "E80_decoder_readout_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E80_decoder_readout_summary.json")


if __name__ == "__main__":
    main()
