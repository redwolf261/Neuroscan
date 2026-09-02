"""
Phase E71, smallest discriminating experiment for the CDCG gating
mechanism (Section 3.3 of PHASE_E71_CDCG_DESIGN.md), per the project's
Observe->Hypothesize->Attempt->Test->Adapt directive.

WHY THIS EXPERIMENT (not a full end-to-end training run):
Predictions 1 and 2 established that hat{d}_i (the aux head's own
prediction) is a real, non-degenerate signal that tracks the network's
measured causal bottleneck-ablation sensitivity. They did NOT establish
that ACTING on that signal (gating enc1's contribution to the decoder)
changes segmentation behavior in the intended direction. Training a full
gated model end-to-end would conflate three separate questions (does the
gate mechanism itself do the right thing; does joint training find a
good aux-loss weight; does the whole system improve Dice) into one
number, making a negative result undiagnosable. This experiment isolates
the FIRST question only, inference-only, no training.

HYPOTHESIS (falsifiable): if gating enc1's contribution by
(1 - sigma(hat{d}_i)) is a sensible mechanism, then artificially forcing
the gate toward "trust coarse more" (i.e. suppressing enc1) should hurt
Dice MORE for subjects with a LOW measured d_i (who don't actually
depend much on the bottleneck -- suppressing their fine detail should
be harmful) than for subjects with a HIGH measured d_i (who are already
bottleneck-dependent -- losing fine detail should matter less, since
they weren't relying on it as much anyway).

Formally: define the per-subject SLOPE of Dice against a swept
suppression strength s in [0,1] (s=0: no suppression, unchanged
architecture; s=1: enc1 fully zeroed, matching E48's own full-ablation
condition). Predict:
    slope_i is MORE NEGATIVE (larger Dice loss from suppression) for
    subjects with LOW measured d_i, and LESS NEGATIVE for subjects with
    HIGH measured d_i.
i.e. Spearman(measured d_i, slope_i) should be POSITIVE (higher d_i ->
less negative/steep slope -> more tolerant of enc1 suppression).

This is the mechanism's own prediction, independent of whether
hat{d}_i (the LEARNED predictor) is used at all -- it tests whether
suppressing enc1 in proportion to a subject's TRUE bottleneck-dependence
is even the right thing to do, before ever touching the learned gate.
If this fails, the whole CDCG mechanism's premise (that bottleneck-
dependent subjects can safely have their skip content down-weighted) is
wrong, independent of prediction fidelity.

DECISION RULE:
  PASS if Spearman(measured d_i, slope_i) > 0, permutation p<0.05 --
    the mechanism's core premise holds: subjects who depend more on the
    bottleneck tolerate skip-suppression better, exactly as the gating
    idea requires.
  FAIL otherwise -- the premise is wrong; do NOT proceed to training a
    full gated model, regardless of prediction 1/2's earlier results.
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
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000
SUPPRESSION_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]  # s: enc1 -> (1-s)*enc1

MM_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e70" / "runs"
           / "MM_seed0" / "checkpoints" / "best.pth")


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    return 1.0 if denom == 0 else float(2 * tp / denom)


def forward_with_enc1_suppression(model, image, suppression, device):
    """s=0: unchanged architecture (verified identity). s=1: enc1 fully
    zeroed, matching E48's own full-bottleneck-skip-ablation logic
    structurally (though E48 ablated the BOTTLENECK, not enc1 -- this
    ablates the SKIP, matching E62-E65's own locus)."""
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1); enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2); enc3 = model.enc3(pool2); pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)

        up3 = model.upconv3(bottleneck)
        dec3 = model.dec3(torch.cat([up3, enc3], dim=1))
        up2 = model.upconv2(dec3)
        dec2 = model.dec2(torch.cat([up2, enc2], dim=1))
        up1 = model.upconv1(dec2)

        enc1_suppressed = (1.0 - suppression) * enc1
        dec1 = model.dec1(torch.cat([up1, enc1_suppressed], dim=1))
        probs = model.seg_head(dec1)
    return probs.squeeze(0).squeeze(0).cpu().numpy()


def forward_get_bottleneck_and_ablated_dice(model, image, target_bin, device):
    """Reused from check_cdcg_prediction1.py (bottleneck ablation, for
    the measured d_i label) -- kept local/duplicated here rather than
    imported, since this script tests a DIFFERENT intervention locus
    (enc1 skip suppression) and importing could blur which construction
    is used where."""
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1); enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2); enc3 = model.enc3(pool2); pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)

        def decode(bn):
            up3 = model.upconv3(bn)
            dec3 = model.dec3(torch.cat([up3, enc3], dim=1))
            up2 = model.upconv2(dec3)
            dec2 = model.dec2(torch.cat([up2, enc2], dim=1))
            up1 = model.upconv1(dec2)
            dec1 = model.dec1(torch.cat([up1, enc1], dim=1))
            return model.seg_head(dec1)

        probs_intact = decode(bottleneck)
        probs_ablated = decode(torch.zeros_like(bottleneck))
        dice_intact = dice_score((probs_intact.squeeze(0).squeeze(0) >= 0.5).float().cpu().numpy(), target_bin)
        dice_ablated = dice_score((probs_ablated.squeeze(0).squeeze(0) >= 0.5).float().cpu().numpy(), target_bin)
    return dice_intact - dice_ablated


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(MM_CKPT), map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded MM seed0 checkpoint: best_per_subject_dice={ckpt.get('best_per_subject_dice')}")

    # ---------------- Sanity check: s=0 reproduces real forward() exactly ----------------
    val_ds = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                    split="val", val_split=0.1, target_shape=(64, 64, 64))
    img0, _, _ = val_ds[0]
    img0_b = img0.unsqueeze(0).to(device)
    with torch.no_grad():
        real = model(img0_b)["probs"]
    manual = forward_with_enc1_suppression(model, img0_b, 0.0, device)
    max_diff = float(np.abs(real.squeeze(0).squeeze(0).cpu().numpy() - manual).max())
    print(f"[Sanity check] s=0 vs real forward(): max abs diff = {max_diff:.6e}")
    assert max_diff < 1e-5, "s=0 does not reproduce real forward() -- STOP."
    print("[Sanity check] PASS.\n")

    # ---------------- Main sweep ----------------
    records = []
    for idx in range(len(val_ds)):
        img, msk, sid = val_ds[idx]
        img_b = img.unsqueeze(0).to(device)
        target_bin = (msk.squeeze(0).numpy() > 0.5).astype(np.float32)

        d_i = forward_get_bottleneck_and_ablated_dice(model, img_b, target_bin, device)

        dice_by_suppression = {}
        for s in SUPPRESSION_SWEEP:
            probs = forward_with_enc1_suppression(model, img_b, s, device)
            dice_by_suppression[s] = dice_score((probs >= 0.5).astype(np.float32), target_bin)

        # slope: simple linear fit of dice vs suppression strength
        xs = np.array(SUPPRESSION_SWEEP)
        ys = np.array([dice_by_suppression[s] for s in SUPPRESSION_SWEEP])
        slope = float(np.polyfit(xs, ys, 1)[0])  # negative = Dice drops as suppression increases

        records.append({"subject_id": sid, "d_i": d_i, "slope": slope,
                        "dice_by_suppression": {str(s): dice_by_suppression[s] for s in SUPPRESSION_SWEEP}})

        if (idx + 1) % 25 == 0:
            print(f"  processed {idx+1}/{len(val_ds)}", flush=True)

    with open(OUT_DIR / "E71_gate_sweep_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.\n")

    d_arr = np.array([r["d_i"] for r in records])
    slope_arr = np.array([r["slope"] for r in records])

    print(f"Mean d_i: {d_arr.mean():.4f}")
    print(f"Mean slope (Dice change per unit suppression): {slope_arr.mean():.4f}")
    print(f"(negative slope = Dice drops as enc1 is suppressed, expected on average)")

    rho, p_param = stats.spearmanr(d_arr, slope_arr)
    rng = np.random.default_rng(SEED)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng.permutation(slope_arr)
        perm_rhos[i], _ = stats.spearmanr(d_arr, perm_y)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())

    print(f"\n=== Spearman(measured d_i, slope) = {rho:+.4f} (parametric p={p_param:.4e}, permutation p={p_perm:.4f}) ===")
    print("Hypothesis: POSITIVE rho -- subjects with higher bottleneck-dependence (d_i) tolerate "
          "enc1 suppression BETTER (less negative/steep slope) -- the core premise the gating mechanism needs.")

    passes = (rho > 0) and (p_perm < 0.05)
    verdict = "PASS" if passes else "FAIL"
    print(f"\n=== GATE MECHANISM PREMISE VERDICT: {verdict} ===")
    if passes:
        print("The mechanism's core premise holds: bottleneck-dependent subjects tolerate skip "
              "suppression better. Proceeding to actually gate BY hat{d}_i (not just measured d_i) "
              "is justified as the next step.")
    else:
        print("The premise does NOT hold -- suppressing enc1 does not selectively spare "
              "bottleneck-dependent subjects. Do NOT proceed to training a full gated model; "
              "the mechanism as designed does not do what it needs to do, independent of prediction fidelity.")

    summary = {
        "n_subjects": len(records), "suppression_sweep": SUPPRESSION_SWEEP,
        "mean_d_i": float(d_arr.mean()), "mean_slope": float(slope_arr.mean()),
        "spearman_rho": float(rho), "parametric_p": float(p_param), "permutation_p": p_perm,
        "verdict": verdict,
    }
    with open(OUT_DIR / "E71_gate_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E71_gate_sweep_summary.json")


if __name__ == "__main__":
    main()
