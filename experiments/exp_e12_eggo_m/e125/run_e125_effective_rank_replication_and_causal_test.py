"""
Phase E125: Replication + Causal Test of E124's Unplanned Stage-3
Effective-Rank Finding.

CONTEXT: E124's PRIMARY hypothesis (effective rank explains E121's
N1~=N3>>N2 pattern) was killed (monotonic pattern, wrong shape). But an
UNPLANNED finding surfaced during audit of E124's unusually large pooled
Delta R^2: at the pool3/bottleneck transition (stage 3) SPECIFICALLY,
per-subject effective rank of the pre-pooling (enc3) window (E_N3,
participation-ratio-based, see E124's docstring for exact definition)
correlates strongly with N_b -- partial rho=0.832 (controlling lesion
size), permutation p<0.001, computed on ALL 125 subjects post-hoc (i.e.
exploratory, not pre-registered, discovered by looking at the data).

Per the user's explicit decision this session, this phase does TWO things
in sequence, per this project's own established discipline (never trust a
surprising exploratory result without (a) a clean replication check and
(b) causal, not just correlational, evidence):

PART 1 -- REPLICATION (pre-registered BEFORE looking at held-out numbers):
  Re-test Spearman(E_N3, N_3 | native_size) using ONLY the held-out
  probe-test split (every-4th subject in size-sorted order, IDENTICAL
  convention to E90-E124) -- i.e. the 32-subject subset that was never
  used to "discover" anything in E124's exploratory pooled regression
  (E124 itself used all 125 subjects with no train/test split for this
  specific stage-3-only relationship). This is the honest test of whether
  the exploratory finding holds on a properly held-out population, not
  just re-computing the same in-sample number.
  PRE-REGISTERED PASS: rho>0, permutation p<0.05, partial rho (controlling
  native_size) also >0 and p<0.05, on the held-out 32 subjects ALONE.
  FAIL -> KILL, do not proceed to Part 2 (causal test would be wasted
  compute on an unreplicated correlational artifact).

PART 2 -- CAUSAL TEST (only runs if Part 1 passes):
  Direct do(x)-style intervention: for EVERY one of the 512 pool3 windows
  in enc3 (ALL windows uniformly, matching how E_stage was itself defined
  as an average over ALL windows -- not just lesion-proximal ones, per
  explicit user decision), replace all 8 sub-voxels' channel vectors with
  their MEAN across the window. This collapses effective rank toward 1
  (maximally redundant) BEFORE pool3 runs -- MaxPool3d applied to 8
  identical sub-voxels trivially just outputs that mean value; no
  selection/competition occurs at all in the intervened condition.
  Measure the resulting N_b (via the EXACT same E48/E109/E118-E121
  ablation-based construction: zero the bottleneck, compare Dice) for
  BOTH the intact-enc3 and mean-replaced-enc3 conditions, and report the
  shift in N_b.
  PRE-REGISTERED PREDICTION (direction implied by the correlation, stated
  explicitly BEFORE running): if effective rank CAUSALLY drives necessity
  (not just correlates with it via some third factor), forcing rank to 1
  should REDUCE N_b (subjects should need the bottleneck LESS when their
  pre-pooling windows carry no genuinely distinct information for pooling
  to discard in the first place -- pooling becomes lossless in the
  intervened condition).
  DECISION: if N_b(mean-replaced) is significantly LOWER than
  N_b(intact) (paired test, same direction as predicted) -> causal
  evidence, GREEN for further investigation (prior-art audit next, still
  NOT architecture implementation). If no significant shift, or shift in
  the WRONG direction -> the correlation is likely non-causal (confounded
  by a third factor, e.g. general subject difficulty) -- KILL as
  correlational-only.

NO ARCHITECTURE MODIFICATION for the actual model going forward (the
mean-replacement is a diagnostic intervention on a frozen checkpoint's
forward pass only, exactly like every prior ablation in this project --
not a trained/deployed change). NO NOVELTY CLAIM.
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
E124_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e124" / "E124_per_subject_table.json"

CKPT_PATH = (project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs"
             / "AttnGate_seed0" / "checkpoints" / "best.pth")
EXPECTED_VAL_DICE = 0.9101624600589275


def permutation_test_corr(x, y, seed):
    rho, p_param = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng.permutation(y)
        perm_rhos[i], _ = stats.spearmanr(x, perm_y)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())
    return float(rho), float(p_param), p_perm


def partial_corr_test(x, y, log_size, seed):
    X1 = np.column_stack([np.ones(len(log_size)), log_size])
    beta_x, *_ = np.linalg.lstsq(X1, x, rcond=None)
    resid_x = x - X1 @ beta_x
    beta_y, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid_y = y - X1 @ beta_y
    rho, p_param = stats.spearmanr(resid_x, resid_y)
    rng = np.random.default_rng(seed)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y2 = rng.permutation(resid_y)
        perm_rhos[i], _ = stats.spearmanr(resid_x, perm_y2)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())
    return float(rho), float(p_param), p_perm


def part1_replication():
    print("=" * 70)
    print("PART 1: REPLICATION on held-out split (pre-registered before looking)")
    print("=" * 70)

    with open(E124_TABLE_PATH) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} subjects from E124's per-subject table.")

    # IDENTICAL split convention to E90-E124: every 4th subject in
    # size-sorted order held out.
    sorted_idx = sorted(range(len(records)), key=lambda i: records[i]["native_size"])
    test_flags = [(pos % 4 == 0) for pos in range(len(sorted_idx))]
    heldout_idx = [sorted_idx[pos] for pos, f in enumerate(test_flags) if f]
    print(f"Held-out subjects: {len(heldout_idx)} (IDENTICAL split rule to E90-E124)")

    heldout = [records[i] for i in heldout_idx]
    E_N3 = np.array([r["E_N_3"] for r in heldout])
    N_3 = np.array([r["N_3"] for r in heldout])
    native_size = np.array([r["native_size"] for r in heldout])
    log_size = np.log(native_size + 1)

    rho, p_param, p_perm = permutation_test_corr(E_N3, N_3, SEED)
    print(f"\nHeld-out marginal Spearman(E_N3, N_3): rho={rho:+.4f} "
          f"(param p={p_param:.4e}, perm p={p_perm:.4f})")

    rho_partial, p_partial_param, p_partial_perm = partial_corr_test(E_N3, N_3, log_size, SEED + 1)
    print(f"Held-out partial Spearman (controlling native_size): rho={rho_partial:+.4f} "
          f"(param p={p_partial_param:.4e}, perm p={p_partial_perm:.4f})")

    passes = (rho > 0) and (p_perm < 0.05) and (rho_partial > 0) and (p_partial_perm < 0.05)
    verdict = "REPLICATES" if passes else "DOES_NOT_REPLICATE"
    print(f"\n=== PART 1 VERDICT: {verdict} ===")

    result = {
        "n_heldout": len(heldout),
        "marginal_rho": rho, "marginal_perm_p": p_perm,
        "partial_rho_controlling_size": rho_partial, "partial_perm_p": p_partial_perm,
        "verdict": verdict,
    }
    with open(OUT_DIR / "E125_part1_replication.json", "w") as f:
        json.dump(result, f, indent=2)
    return passes, result


def windows_from_tensor(t):
    C = t.shape[0]
    x = t.unfold(1, 2, 2).unfold(2, 2, 2).unfold(3, 2, 2)
    return x  # (C, 8,8,8, 2,2,2) view -- keep unfolded shape for reconstruction


def mean_replace_windows(enc3):
    """enc3: (C, 16,16,16) tensor. Returns a NEW tensor of the SAME shape
    where every non-overlapping 2x2x2 window's 8 sub-voxels are replaced
    by their mean (per channel) -- forces effective rank toward 1 for
    EVERY window uniformly (per explicit user decision), matching how
    E_stage was itself defined as an average over ALL windows."""
    C, D, H, W = enc3.shape
    # reshape into windows: (C, D/2, 2, H/2, 2, W/2, 2)
    x = enc3.view(C, D // 2, 2, H // 2, 2, W // 2, 2)
    # mean over the three window-internal dims (2,4,6)
    window_mean = x.mean(dim=(2, 4, 6), keepdim=True)  # (C, D/2,1,H/2,1,W/2,1)
    x_replaced = window_mean.expand_as(x).contiguous()
    return x_replaced.view(C, D, H, W)


def unit_test_mean_replace():
    torch.manual_seed(0)
    t = torch.randn(3, 4, 4, 4)
    out = mean_replace_windows(t)
    assert out.shape == t.shape
    # check window (0,0,0) in enc3 coords -> voxels [0:2,0:2,0:2]
    window_vals = t[:, 0:2, 0:2, 0:2]
    expected_mean = window_vals.reshape(3, -1).mean(dim=1)
    actual = out[:, 0, 0, 0]
    assert torch.allclose(actual, expected_mean, atol=1e-5), f"{actual} vs {expected_mean}"
    # check every one of the 8 sub-voxels in that window got the SAME value
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                assert torch.allclose(out[:, dz, dy, dx], expected_mean, atol=1e-5)
    print("[Unit test] mean_replace_windows: PASS (window mean correct, all 8 sub-voxels identical).")


def forward_with_enc3_intervention_and_ablation(model, image, mean_replace, ablate_bottleneck, device):
    """Forward pass with two independently-controllable interventions:
    (1) mean_replace: if True, replace enc3 with its per-window mean
        (forces effective rank -> 1) BEFORE pool3 runs.
    (2) ablate_bottleneck: if True, zero the bottleneck tensor (E48-E121's
        N_b construction), applied AFTER (1).
    Both False = real, unmodified forward pass."""
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)

        if mean_replace:
            enc3 = mean_replace_windows(enc3.squeeze(0)).unsqueeze(0)

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


def part2_causal_test():
    print("\n" + "=" * 70)
    print("PART 2: CAUSAL TEST -- does forcing rank->1 reduce N_b?")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unit_test_mean_replace()

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
        manual_probs = forward_with_enc3_intervention_and_ablation(model, img0_b, False, False, device)
        real_np = real_probs.squeeze(0).squeeze(0).cpu().numpy() if torch.is_tensor(real_probs) else np.asarray(real_probs)
        max_diff = float(np.abs(real_np - manual_probs).max())
    assert max_diff < 1e-4, "Manual trunk mismatch -- STOP."
    print(f"[Sanity check] manual trunk vs real forward(): max abs diff = {max_diff:.6e} PASS.\n")

    records = []
    print(f"Running 4 conditions per subject (intact-intact, intact-ablated, "
          f"meanrepl-intact, meanrepl-ablated), {len(val_ds)} subjects...", flush=True)
    for idx in range(len(val_ds)):
        image, mask, sid = val_ds[idx]
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_ds.subject_dirs[idx]
        seg_path = Path(subject_dir) / f"{sid}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_size = int(seg_binary_native.sum())
        target_bin_64 = (fractional_occupancy(seg_binary_native, (64, 64, 64)) > 0.5).astype(np.float32)

        probs_intact_intact = forward_with_enc3_intervention_and_ablation(model, image_b, False, False, device)
        probs_intact_ablated = forward_with_enc3_intervention_and_ablation(model, image_b, False, True, device)
        probs_mr_intact = forward_with_enc3_intervention_and_ablation(model, image_b, True, False, device)
        probs_mr_ablated = forward_with_enc3_intervention_and_ablation(model, image_b, True, True, device)

        d_ii = dice_score((probs_intact_intact >= 0.5).astype(np.float32), target_bin_64)
        d_ia = dice_score((probs_intact_ablated >= 0.5).astype(np.float32), target_bin_64)
        d_mi = dice_score((probs_mr_intact >= 0.5).astype(np.float32), target_bin_64)
        d_ma = dice_score((probs_mr_ablated >= 0.5).astype(np.float32), target_bin_64)

        n_b_intact = d_ii - d_ia   # original N_b (enc3 untouched)
        n_b_meanrepl = d_mi - d_ma  # N_b when enc3's windows are forced to rank~1

        records.append({
            "subject_id": sid, "native_size": native_size,
            "dice_intact_enc3_intact_bottleneck": d_ii,
            "dice_meanrepl_enc3_intact_bottleneck": d_mi,
            "N_b_intact_enc3": n_b_intact,
            "N_b_meanreplaced_enc3": n_b_meanrepl,
        })
        if (idx + 1) % 25 == 0:
            print(f"  {idx+1}/{len(val_ds)}", flush=True)

    with open(OUT_DIR / "E125_part2_causal_table.json", "w") as f:
        json.dump(records, f, indent=2)

    n_b_intact = np.array([r["N_b_intact_enc3"] for r in records])
    n_b_mr = np.array([r["N_b_meanreplaced_enc3"] for r in records])
    dice_drop_from_mr = np.array([r["dice_intact_enc3_intact_bottleneck"] - r["dice_meanrepl_enc3_intact_bottleneck"] for r in records])

    print(f"\nN_b (intact enc3): mean={n_b_intact.mean():.4f} std={n_b_intact.std():.4f} "
          f"(cross-check vs established ~0.27-0.32 range)")
    print(f"N_b (mean-replaced enc3, rank forced ->1): mean={n_b_mr.mean():.4f} std={n_b_mr.std():.4f}")
    print(f"Mean-replacement itself costs (Dice drop from forcing rank->1, bottleneck intact): "
          f"{dice_drop_from_mr.mean():.4f} +/- {dice_drop_from_mr.std():.4f} "
          f"(context: how damaging is the rank-forcing intervention on its own)")

    diff = n_b_mr - n_b_intact
    t_stat, p_ttest = stats.ttest_rel(n_b_mr, n_b_intact)
    w_stat, p_wilcoxon = stats.wilcoxon(n_b_mr, n_b_intact)
    print(f"\nPaired shift (N_b_meanreplaced - N_b_intact): mean={diff.mean():+.4f} "
          f"(paired t p={p_ttest:.4e}, Wilcoxon p={p_wilcoxon:.4e}, n={len(records)})")

    predicted_direction = diff.mean() < 0  # mean-replacement should REDUCE N_b if causal
    significant = p_ttest < 0.05 and p_wilcoxon < 0.05

    if predicted_direction and significant:
        decision = "CAUSAL_EVIDENCE_CONFIRMED"
        detail = ("Forcing effective rank -> 1 SIGNIFICANTLY REDUCES N_b, in the predicted "
                   "direction -- causal evidence that pre-pooling window effective rank drives "
                   "bottleneck necessity, not just correlates with it via a third factor. GREEN "
                   "for further investigation -- prior-art audit next, still NOT architecture "
                   "implementation.")
    elif significant and not predicted_direction:
        decision = "WRONG_DIRECTION_KILL"
        detail = ("The shift is significant but in the OPPOSITE direction from predicted -- "
                   "the correlation does not reflect the causal story assumed. KILL as "
                   "correlational-only / mechanism mischaracterized.")
    else:
        decision = "NO_SIGNIFICANT_CAUSAL_EFFECT_KILL"
        detail = ("No significant shift in N_b from forcing rank->1. The strong correlation "
                   "found in E124 appears to be non-causal -- likely confounded by a third "
                   "factor (e.g. general subject difficulty) rather than reflecting a real "
                   "causal role for window effective rank. KILL.")

    print(f"\n=== PART 2 DECISION: {decision} ===\n{detail}")

    summary = {
        "checkpoint": str(CKPT_PATH), "val_dice_check": val_dice,
        "n_subjects": len(records),
        "N_b_intact_mean": float(n_b_intact.mean()), "N_b_intact_std": float(n_b_intact.std()),
        "N_b_meanreplaced_mean": float(n_b_mr.mean()), "N_b_meanreplaced_std": float(n_b_mr.std()),
        "dice_drop_from_meanreplacement_alone": float(dice_drop_from_mr.mean()),
        "paired_diff_mean": float(diff.mean()),
        "paired_ttest_p": float(p_ttest), "wilcoxon_p": float(p_wilcoxon),
        "predicted_direction_observed": bool(predicted_direction),
        "decision": decision, "detail": detail,
    }
    with open(OUT_DIR / "E125_part2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E125_part2_summary.json, E125_part2_causal_table.json")


def main():
    passes, part1_result = part1_replication()
    if not passes:
        print("\n" + "=" * 70)
        print("PART 1 FAILED TO REPLICATE -- STOPPING before Part 2 (causal test would be "
              "wasted compute on an unreplicated correlational artifact), per pre-registered rule.")
        print("=" * 70)
        return
    part2_causal_test()


if __name__ == "__main__":
    main()
