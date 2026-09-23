"""
Phase E76: enc1 magnitude operating-regime test (NO TRAINING, pure
inference-time intervention) -- per the user's design, following E75's
split result (enc1 magnitude causally confirmed; bottleneck killed/
reversed; cross-depth hypothesis killed).

E75 used a wide alpha range {0.25, 0.5, 1.0, 2.0} and found a real,
monotonic, causal enc1 result -- but the confound check showed the
INTACT (non-translated) network's own Dice already degrades substantially
at alpha=2.0 (0.902 -> 0.867), meaning the network may be pushed outside
its normal operating regime at the wide end of that range. This phase
asks the sharper, more decision-relevant question: does the SAME
relationship hold in a SMALL neighborhood around alpha=1 where the
intact segmentation is essentially undisturbed -- i.e. is
    dS/dalpha |_{alpha=1} > 0
a real LOCAL property of the network's own natural operating point, not
an artifact of pushing far out of distribution?

This is the regime any future training-time intervention would actually
operate in (small, learned magnitude adjustments near the network's
existing solution), so establishing the relationship holds there is a
precondition for designing one.

METHOD: identical construction to E75 (enc1 only this time), finer alpha
grid centered on 1.0: {0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30}. For
each alpha, per subject, measure BOTH:
  D_intact(alpha): Dice with magnitude rescaled, NO translation
  S(alpha): Dice(intact-alpha) - Dice(translated-alpha) [same construction
    as E75 -- how much the SAME 3-voxel translation hurts, at this alpha]

PRE-DECLARED DECISION RULE:
  PASS if:
    (a) D_intact(alpha) stays close to D_intact(1.0) across the WHOLE
        tested range (no large collapse -- a soft bound, not a formal
        equivalence test: flag if any alpha's mean D_intact drops by
        more than 0.02 from alpha=1.0's value, since that would mean
        the "operating regime" premise itself is violated even in this
        narrower range).
    (b) S(alpha) is monotonically increasing across the 7-point grid,
        with the LOCAL slope at alpha=1 (measured via S(1.10)-S(0.90),
        paired) significant and positive.
  FAIL/PARTIAL otherwise -- report exactly what the data show, do not
  round to a clean pass if the picture is mixed (matching this project's
  own standing discipline).
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
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e75"))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402
from run_e74_spatial_dependence_audit import translate_volume  # noqa: E402
from run_e75_magnitude_causality_test import rescale_magnitude, verify_direction_preserved  # noqa: E402

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 0
N_PERM = 1000
TRANSLATION_OFFSET = 3
ALPHAS = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]

MM_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e70" / "runs"
           / "MM_seed0" / "checkpoints" / "best.pth")


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    return 1.0 if denom == 0 else float(2 * tp / denom)


def forward_enc1_only(model, img_b, translate, alpha, device):
    """enc1-only version of E75's forward_full -- bottleneck untouched
    throughout, matching this phase's narrowed scope."""
    with torch.no_grad():
        enc1 = model.enc1(img_b)
        e_skip = rescale_magnitude(enc1.squeeze(0), alpha)
        if translate:
            e_skip = translate_volume(e_skip, TRANSLATION_OFFSET)
        e_skip = e_skip.unsqueeze(0)

        pool1 = model.pool1(enc1)  # E_pool always real, per E64/E74/E75's isolation design
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
    verify_direction_preserved(device)

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

        row = {"subject_id": sid}
        for alpha in ALPHAS:
            probs_no_t = forward_enc1_only(model, img_b, translate=False, alpha=alpha, device=device)
            dice_no_t = dice_score((probs_no_t >= 0.5).astype(np.float32), target_bin)

            probs_t = forward_enc1_only(model, img_b, translate=True, alpha=alpha, device=device)
            dice_t = dice_score((probs_t >= 0.5).astype(np.float32), target_bin)

            row[f"dice_intact_alpha{alpha}"] = dice_no_t
            row[f"dice_translated_alpha{alpha}"] = dice_t
            row[f"S_alpha{alpha}"] = dice_no_t - dice_t

        records.append(row)
        if (idx + 1) % 25 == 0:
            print(f"  processed {idx+1}/{len(val_ds)}", flush=True)

    with open(OUT_DIR / "E76_enc1_operating_regime_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.\n")

    # ---------------- (a) intact Dice stability across the range ----------------
    print("--- (a) Intact (no translation) Dice across alpha -- operating-regime check ---")
    intact_means = {}
    for alpha in ALPHAS:
        vals = np.array([r[f"dice_intact_alpha{alpha}"] for r in records])
        intact_means[alpha] = vals.mean()
        print(f"  alpha={alpha}: mean D_intact = {vals.mean():.4f}")

    baseline_d = intact_means[1.00]
    max_drop_in_range = baseline_d - min(intact_means.values())
    regime_stable = all(abs(intact_means[a] - baseline_d) <= 0.02 for a in ALPHAS)
    print(f"\n  Max deviation from alpha=1.0's D_intact: {max_drop_in_range:.4f} "
          f"(threshold: <=0.02 per alpha)")
    print(f"  Operating-regime stability: {'PASS' if regime_stable else 'FAIL'}")

    # ---------------- (b) S(alpha) monotonicity + local slope at alpha=1 ----------------
    print("\n--- (b) Translation-induced Dice drop S(alpha) across the narrow range ---")
    S_by_alpha = {alpha: np.array([r[f"S_alpha{alpha}"] for r in records]) for alpha in ALPHAS}
    for alpha in ALPHAS:
        v = S_by_alpha[alpha]
        print(f"  alpha={alpha}: mean S = {v.mean():.4f} (std {v.std():.4f})")

    means_in_order = [S_by_alpha[a].mean() for a in ALPHAS]
    monotonic = all(means_in_order[i] <= means_in_order[i+1] for i in range(len(means_in_order)-1))
    strictly_monotonic = all(means_in_order[i] < means_in_order[i+1] for i in range(len(means_in_order)-1))

    s_090, s_110 = S_by_alpha[0.90], S_by_alpha[1.10]
    t_local, p_local_t = stats.ttest_rel(s_110, s_090)
    w_local, p_local_w = stats.wilcoxon(s_110, s_090)
    local_slope = s_110.mean() - s_090.mean()

    print(f"\n  Monotonic (non-decreasing): {monotonic}")
    print(f"  Strictly monotonic: {strictly_monotonic}")
    print(f"\n  Local slope dS/dalpha near alpha=1 (S(1.10)-S(0.90)): {local_slope:+.4f}")
    print(f"  paired t-test: t={t_local:.3f}, p={p_local_t:.4e}")
    print(f"  Wilcoxon: p={p_local_w:.4e}")

    local_slope_significant_positive = (local_slope > 0) and (p_local_t < 0.05) and (p_local_w < 0.05)

    overall_pass = regime_stable and monotonic and local_slope_significant_positive
    print(f"\n=== E76 VERDICT ===")
    print(f"  (a) Operating-regime stable: {regime_stable}")
    print(f"  (b) S(alpha) monotonic: {monotonic}, local slope positive & significant: {local_slope_significant_positive}")
    print(f"  OVERALL: {'PASS -- clean local relationship in the natural operating regime' if overall_pass else 'PARTIAL/FAIL -- report exactly what the data show, see log'}")

    summary = {
        "alphas": ALPHAS, "n_subjects": len(records),
        "intact_means": {str(a): float(intact_means[a]) for a in ALPHAS},
        "regime_stable": bool(regime_stable), "max_drop_in_range": float(max_drop_in_range),
        "S_means": {str(a): float(S_by_alpha[a].mean()) for a in ALPHAS},
        "monotonic": bool(monotonic), "strictly_monotonic": bool(strictly_monotonic),
        "local_slope_alpha1": float(local_slope),
        "local_slope_ttest_p": float(p_local_t), "local_slope_wilcoxon_p": float(p_local_w),
        "local_slope_significant_positive": bool(local_slope_significant_positive),
        "overall_pass": bool(overall_pass),
    }
    with open(OUT_DIR / "E76_enc1_operating_regime_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E76_enc1_operating_regime_summary.json")


if __name__ == "__main__":
    main()
