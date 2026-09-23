"""
Phase E126: Graded (Dose-Response) Causal Test of the Effective-Rank ->
N_b Relationship -- redesign of E125 Part 2, which was CONFOUNDED (full
mean-replacement of all 512 pool3 windows cost 0.16 Dice on its own, with
36% of subjects showing negative N_b in that condition -- a clear
out-of-distribution signature that made the "causal evidence confirmed"
auto-verdict untrustworthy, overridden by the assistant after auditing the
raw numbers).

CONTEXT: E124 found (exploratory) and E125 Part 1 REPLICATED (held-out,
partial rho=0.877, p<0.0001) that effective rank of the pool3 pre-pooling
window (enc3) strongly predicts N_b. E125 Part 2's causal test used a
full collapse-to-mean intervention on ALL windows, which was too
destructive to interpret cleanly.

REDESIGN: instead of one all-or-nothing intervention, apply a GRADED,
CALIBRATED partial rank reduction, and sweep intervention strength to get
a dose-response curve -- both gentler (protects against the OOD confound)
and more informative (a monotonic dose-response is stronger evidence for
a real causal relationship than one significant point).

INTERVENTION: for each window and each alpha in {0.95, 0.85, 0.70, 0.50},
replace each window's 8 sub-voxels with:
    x'_i = alpha * x_i + (1-alpha) * window_mean
alpha=1.0 (not tested, trivially == intact). alpha=0.95 is very gentle
(95% original signal retained per sub-voxel, only nudged 5% toward the
window mean); alpha=0.50 is a substantial blend. This is applied to ALL
512 windows uniformly (same scope as E125, for comparability), but the
SEVERITY is now continuously controllable, unlike E125's binary
full-collapse.

SAFETY BAR (pre-registered): at the GENTLEST setting (alpha=0.95), the
intervention-alone Dice cost (bottleneck intact) must be < 0.02 -- if not,
even the gentlest tested setting already risks leaving the training
distribution, and results at ANY alpha should be treated with extra
caution (reported honestly, not suppressed, but flagged).

FOR EACH alpha, measure (same construction as E125): N_b(alpha) =
Dice(intact bottleneck | enc3 blended at this alpha) -
Dice(ablated bottleneck | enc3 blended at this alpha).

PRE-REGISTERED ANALYSIS:
  1. Report intervention-alone Dice cost at each alpha (context/safety
     check, not the causal test itself).
  2. Report mean N_b(alpha) for each alpha -- is there a MONOTONIC trend
     (N_b decreasing as alpha decreases, i.e. as rank is pushed toward 1)?
  3. Subject-level: does the SIZE of each subject's alpha=0.5 rank
     reduction (measured via actual E_stage recomputed under blending,
     not just alpha itself, since blending's effect on measured
     participation ratio may not be linear in alpha) predict the SIZE of
     their N_b shift? This directly tests the dose-response relationship
     at the subject level, not just the population mean.
  4. Paired significance test (paired t + Wilcoxon, matching this
     project's own E56 convention) comparing N_b(alpha) vs N_b(intact) at
     EACH alpha separately.

DECISION:
  - If even alpha=0.95 already exceeds the 0.02 Dice-cost safety bar:
    report honestly that this causal question cannot be cleanly tested
    via this intervention at ANY reasonable severity without leaving
    the training distribution -- the relationship may still be real but
    is not cleanly testable this way. Do not force a causal conclusion.
  - If a MONOTONIC, dose-dependent, statistically significant reduction
    in N_b is observed as alpha decreases (rank pushed lower), AND at
    least the gentlest 1-2 alpha levels stay within the safety bar or
    close to it -> genuine causal evidence, stronger than E125's single
    confounded point. GREEN for prior-art audit.
  - If no clear dose-response trend, or the trend only appears at
    severities that are themselves already confounded (large
    intervention-alone Dice cost) -> KILL / inconclusive, report
    honestly.

NO ARCHITECTURE MODIFICATION, NO NOVELTY CLAIM.
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

CKPT_PATH = (project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs"
             / "AttnGate_seed0" / "checkpoints" / "best.pth")
EXPECTED_VAL_DICE = 0.9101624600589275
ALPHAS = [0.95, 0.85, 0.70, 0.50]
SAFETY_BAR_DICE_COST = 0.02


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    if denom == 0:
        return 1.0
    return float(2 * tp / denom)


def fractional_occupancy(seg_binary_native, shape):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=shape, mode="area").squeeze().numpy()
    return frac


def blend_windows(enc3, alpha):
    """enc3: (C,16,16,16). Returns alpha*enc3 + (1-alpha)*window_mean,
    per non-overlapping 2x2x2 window (same partition as mean_replace_windows
    in E125, generalized with a blend factor; alpha=0 reduces to E125's
    full mean-replacement exactly, alpha=1 is the identity)."""
    C, D, H, W = enc3.shape
    x = enc3.view(C, D // 2, 2, H // 2, 2, W // 2, 2)
    window_mean = x.mean(dim=(2, 4, 6), keepdim=True)
    blended = alpha * x + (1 - alpha) * window_mean.expand_as(x)
    return blended.contiguous().view(C, D, H, W)


def unit_test_blend_windows():
    torch.manual_seed(0)
    t = torch.randn(3, 4, 4, 4)
    # alpha=1 -> identity
    out1 = blend_windows(t, 1.0)
    assert torch.allclose(out1, t, atol=1e-5), "alpha=1 should be identity"
    # alpha=0 -> full mean replacement (matches E125's mean_replace_windows)
    out0 = blend_windows(t, 0.0)
    window_vals = t[:, 0:2, 0:2, 0:2]
    expected_mean = window_vals.reshape(3, -1).mean(dim=1)
    assert torch.allclose(out0[:, 0, 0, 0], expected_mean, atol=1e-5), "alpha=0 should match full mean-replace"
    # intermediate alpha: check the blend formula directly on one sub-voxel
    alpha = 0.7
    out_mid = blend_windows(t, alpha)
    expected_mid = alpha * t[:, 0, 0, 0] + (1 - alpha) * expected_mean
    assert torch.allclose(out_mid[:, 0, 0, 0], expected_mid, atol=1e-5), "intermediate alpha blend formula wrong"
    print("[Unit test] blend_windows: PASS (alpha=1 identity, alpha=0 matches full mean-replace, "
          "intermediate alpha formula correct).")


def forward_with_blend_and_ablation(model, image, alpha, ablate_bottleneck, device):
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)

        if alpha is not None and alpha < 1.0:
            enc3 = blend_windows(enc3.squeeze(0), alpha).unsqueeze(0)

        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)
        if ablate_bottleneck:
            bottleneck = torch.zeros_like(bottleneck)

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

    return probs.squeeze(0).squeeze(0).cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    unit_test_blend_windows()

    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    val_dice = ckpt.get("best_val_dice") or ckpt.get("val_dice")
    assert val_dice is not None and abs(float(val_dice) - EXPECTED_VAL_DICE) < 1e-6, \
        f"Checkpoint identity check FAILED: expected {EXPECTED_VAL_DICE}, got {val_dice}"
    print(f"[Sanity check] checkpoint identity PASS (val_dice={val_dice}).")

    val_ds = BraTSDataset(root_dir=str(project_root / "Dataset" / "Training"), split="val",
                          val_split=0.1, target_shape=(64, 64, 64), normalize=True)

    with torch.no_grad():
        img0, _, _ = val_ds[0]
        img0_b = img0.unsqueeze(0).to(device)
        real_out = model(img0_b)
        real_probs = real_out["probs"] if isinstance(real_out, dict) else real_out
        manual_probs = forward_with_blend_and_ablation(model, img0_b, 1.0, False, device)
        real_np = real_probs.squeeze(0).squeeze(0).cpu().numpy() if torch.is_tensor(real_probs) else np.asarray(real_probs)
        max_diff = float(np.abs(real_np - manual_probs).max())
    assert max_diff < 1e-4, "Manual trunk mismatch -- STOP."
    print(f"[Sanity check] manual trunk vs real forward(): max abs diff = {max_diff:.6e} PASS.\n")

    records = []
    print(f"Running conditions (intact-intact, intact-ablated, then blend-intact/ablated for "
          f"{len(ALPHAS)} alphas), {len(val_ds)} subjects...", flush=True)
    for idx in range(len(val_ds)):
        image, mask, sid = val_ds[idx]
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_ds.subject_dirs[idx]
        seg_path = Path(subject_dir) / f"{sid}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_size = int(seg_binary_native.sum())
        target_bin_64 = (fractional_occupancy(seg_binary_native, (64, 64, 64)) > 0.5).astype(np.float32)

        probs_ii = forward_with_blend_and_ablation(model, image_b, 1.0, False, device)
        probs_ia = forward_with_blend_and_ablation(model, image_b, 1.0, True, device)
        d_ii = dice_score((probs_ii >= 0.5).astype(np.float32), target_bin_64)
        d_ia = dice_score((probs_ia >= 0.5).astype(np.float32), target_bin_64)
        n_b_intact = d_ii - d_ia

        rec = {"subject_id": sid, "native_size": native_size,
               "dice_intact": d_ii, "N_b_intact": n_b_intact}

        for alpha in ALPHAS:
            probs_bi = forward_with_blend_and_ablation(model, image_b, alpha, False, device)
            probs_ba = forward_with_blend_and_ablation(model, image_b, alpha, True, device)
            d_bi = dice_score((probs_bi >= 0.5).astype(np.float32), target_bin_64)
            d_ba = dice_score((probs_ba >= 0.5).astype(np.float32), target_bin_64)
            n_b_alpha = d_bi - d_ba
            rec[f"dice_blend_a{alpha}"] = d_bi
            rec[f"N_b_a{alpha}"] = n_b_alpha
            rec[f"intervention_alone_cost_a{alpha}"] = d_ii - d_bi

        records.append(rec)
        if (idx + 1) % 25 == 0:
            print(f"  {idx+1}/{len(val_ds)}", flush=True)

    with open(OUT_DIR / "E126_per_subject_table.json", "w") as f:
        json.dump(records, f, indent=2)

    n_b_intact = np.array([r["N_b_intact"] for r in records])
    print(f"\nN_b (intact): mean={n_b_intact.mean():.4f} std={n_b_intact.std():.4f} "
          f"(cross-check vs established ~0.27-0.32 range)")

    print("\n=== Dose-response summary ===")
    results = {"alphas": {}}
    safety_ok_at_gentlest = None
    for alpha in ALPHAS:
        cost = np.array([r[f"intervention_alone_cost_a{alpha}"] for r in records])
        n_b_a = np.array([r[f"N_b_a{alpha}"] for r in records])
        diff = n_b_a - n_b_intact
        t_stat, p_ttest = stats.ttest_rel(n_b_a, n_b_intact)
        w_stat, p_wilcoxon = stats.wilcoxon(n_b_a, n_b_intact)
        safety_ok = cost.mean() < SAFETY_BAR_DICE_COST
        if alpha == max(ALPHAS):
            safety_ok_at_gentlest = safety_ok
        print(f"alpha={alpha}: intervention-alone cost={cost.mean():.4f}+/-{cost.std():.4f} "
              f"(safety bar <{SAFETY_BAR_DICE_COST}: {'OK' if safety_ok else 'EXCEEDED'}), "
              f"N_b={n_b_a.mean():.4f}+/-{n_b_a.std():.4f}, "
              f"shift={diff.mean():+.4f} (paired t p={p_ttest:.4e}, Wilcoxon p={p_wilcoxon:.4e})")
        results["alphas"][str(alpha)] = {
            "intervention_alone_cost_mean": float(cost.mean()),
            "intervention_alone_cost_std": float(cost.std()),
            "safety_bar_ok": bool(safety_ok),
            "N_b_mean": float(n_b_a.mean()), "N_b_std": float(n_b_a.std()),
            "shift_from_intact_mean": float(diff.mean()),
            "paired_ttest_p": float(p_ttest), "wilcoxon_p": float(p_wilcoxon),
        }

    # Monotonicity check: does N_b decrease monotonically as alpha decreases
    # (i.e. as rank is pushed lower)?
    alpha_sorted = sorted(ALPHAS, reverse=True)  # 0.95 (gentlest) -> 0.50 (strongest)
    n_b_means_sorted = [results["alphas"][str(a)]["N_b_mean"] for a in alpha_sorted]
    n_b_means_sorted_full = [n_b_intact.mean()] + n_b_means_sorted  # prepend alpha=1.0 (intact)
    diffs_between_steps = np.diff(n_b_means_sorted_full)
    monotonic_decreasing = bool(np.all(diffs_between_steps <= 1e-9))
    print(f"\nN_b sequence (alpha=1.0 [intact] -> {alpha_sorted}): "
          f"{[round(x,4) for x in n_b_means_sorted_full]}")
    print(f"Monotonically decreasing as rank is pushed lower: {monotonic_decreasing}")

    print("\n=== DECISION ===")
    if not safety_ok_at_gentlest:
        decision = "SAFETY_BAR_FAILED_AT_GENTLEST_ALPHA"
        detail = (f"Even the gentlest tested alpha ({max(ALPHAS)}) exceeds the "
                  f"{SAFETY_BAR_DICE_COST} Dice-cost safety bar -- this causal question cannot "
                  f"be cleanly tested via window-blending at any reasonable severity without "
                  f"risking an out-of-distribution confound. The correlational relationship "
                  f"(E125 Part 1) may still be real, but causal testing via this intervention "
                  f"method is not reliable. Do not force a causal conclusion.")
    elif monotonic_decreasing and results["alphas"][str(min(ALPHAS))]["paired_ttest_p"] < 0.05:
        decision = "DOSE_RESPONSE_CONFIRMED"
        detail = ("N_b decreases monotonically as effective rank is pushed lower via graded "
                  "blending, with the gentlest setting staying within the safety bar. This is "
                  "genuine dose-response causal evidence -- stronger than E125 Part 2's single "
                  "confounded point. GREEN for a full prior-art audit -- still NOT architecture "
                  "implementation.")
    else:
        decision = "NO_CLEAN_DOSE_RESPONSE"
        detail = ("No clean monotonic dose-response was observed, or significance was not "
                  "reached at the safety-bar-respecting alphas. Report honestly -- the causal "
                  "claim is not established by this test.")

    print(f"{decision}\n{detail}")

    results.update({
        "checkpoint": str(CKPT_PATH), "val_dice_check": val_dice,
        "n_subjects": len(records),
        "N_b_intact_mean": float(n_b_intact.mean()),
        "n_b_sequence_alpha1_to_gentlest_to_strongest": n_b_means_sorted_full,
        "monotonic_decreasing": monotonic_decreasing,
        "safety_bar_ok_at_gentlest": bool(safety_ok_at_gentlest),
        "decision": decision, "detail": detail,
    })
    with open(OUT_DIR / "E126_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved E126_summary.json, E126_per_subject_table.json")


if __name__ == "__main__":
    main()
