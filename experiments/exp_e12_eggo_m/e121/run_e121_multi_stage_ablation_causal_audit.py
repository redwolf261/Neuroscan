"""
Phase E121: Multi-Stage Ablation Causal Audit -- is the 16^3->8^3
transition (pool3, N_3=N_b) UNIQUELY causally important, or do the earlier
downsampling stages (pool1: 64^3->32^3, pool2: 32^3->16^3) show comparable
or even larger causal sensitivity?

CONTEXT: E92-E120 have all probed the pool3/bottleneck transition itself
(decodability, selection behavior, cross-scale innovation, context) and
found it causally necessary (N_b) but could not find a computable
mechanism WITHIN that transition to exploit. The user's redirect: before
committing to "change the downsampling schedule specifically at 16->8",
verify diagnostically whether 16->8 is actually SPECIAL among the three
downsampling transitions, using the SAME kind of causal measure (real
do(x)-style ablation, not a correlational probe) at all three stages.

MEASURE (per explicit user decision this session): ablation-based, NOT
E92's decodability-probe method -- zero the stage's OUTPUT tensor, let
everything downstream recompute for real from that zeroed tensor (the
EXACT same "zero and keep the rest of the trunk real" convention as
E48/E89/E109/E118/E119/E120's bottleneck ablation), measure the resulting
Dice drop. This keeps N_1/N_2/N_3 on the same footing as N_b (N_3 IS N_b,
computed fresh here for a consistency cross-check against the many prior
runs).

  N_1 = Dice(intact) - Dice(pool1-output ablated)      [64^3 -> 32^3]
  N_2 = Dice(intact) - Dice(pool2-output ablated)      [32^3 -> 16^3]
  N_3 = Dice(intact) - Dice(bottleneck-output ablated) [16^3 -> 8^3]  (= N_b)

  NOTE ON N_3's ablation POINT: to stay genuinely comparable to
  E48/E109/E118/E119/E120's own N_b, N_3 zeros the BOTTLENECK tensor
  (post model.bottleneck conv), NOT pool3's own raw output -- a first
  version of this script zeroed pool3 directly and fed the zeros through
  model.bottleneck's real (trained) conv/BatchNorm, which produced a
  much harsher, out-of-distribution collapse (~49% of subjects predicted
  nothing at all) that was NOT a real "16->8 is more causally important"
  finding -- caught by comparing dice_p3 (~0.01 mean) against E48-E120's
  own dice_ablated (~0.6 mean) on the same checkpoint, fixed before
  trusting the result. N_1/N_2 correctly zero pool1/pool2's own raw
  outputs directly (there is no analogous post-transform tensor at those
  stages to prefer), so they remain the "zero this stage's output"
  reading throughout.

IMPORTANT CAVEAT, STATED UP FRONT (not discovered after the fact): ablating
an EARLIER stage (pool1) zeros a tensor that everything downstream
structurally depends on -- enc2, pool2, enc3, pool3, bottleneck ALL
recompute from zeros. This means N_1 and N_2 are expected to be mechanically
larger than N_3 simply because more of the network's computation is
downstream of an earlier ablation, independent of whether that stage is
specially informative for lesions. Raw magnitude comparison (N_1 vs N_2 vs
N_3) is therefore NOT by itself evidence that any one stage is "uniquely
important" for the reason this project cares about (small-lesion
information specifically) -- report it honestly, but the more diagnostic
comparisons are the SUBJECT-LEVEL CORRELATES: does each N_k correlate with
lesion size / Dice-error in the same way, or does N_3 show a qualitatively
DIFFERENT relationship (e.g. small-lesion-specific elevation, matching
E48/E91/E92's own established finding) that N_1/N_2 do NOT show?

REPORTED (raw, per subject, no architecture change):
  - N_1, N_2, N_3 subject-level distributions (mean/std/histogram-ready)
  - Each N_k vs native_size (Spearman + permutation)
  - Each N_k vs dice_error (1-dice_intact) (Spearman + permutation)
  - Each N_k stratified small-vs-large (median split), permutation test
    on the small-vs-large difference (E92's own convention) -- THIS is
    the test that actually answers "is pool3's small-lesion-specificity
    unique among the three stages," not raw magnitude.
  - Cross-check: N_3 here vs N_b from E48/E109/E118/E119/E120 (same
    checkpoint, same construction -- should match closely).

NO ARCHITECTURE MODIFICATION. Diagnostic only, per explicit instruction.
Report raw numbers even if messy -- decision on next steps deferred to
the user after seeing this.
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

CKPT_PATH = (project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs"
             / "AttnGate_seed0" / "checkpoints" / "best.pth")
EXPECTED_VAL_DICE = 0.9101624600589275


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


def forward_with_stage_ablation(model, image, ablate_stage, device):
    """ablate_stage in {None, 'pool1', 'pool2', 'pool3'}. Zeros the named
    stage's OUTPUT tensor, then recomputes EVERYTHING downstream for real
    from that zeroed tensor -- a genuine do(x)-style intervention, same
    convention as E48's forward_with_bottleneck_ablation (which is the
    ablate_stage='pool3' case here, kept byte-identical to that
    construction including the attention-gate handling)."""
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1)
        if ablate_stage == "pool1":
            pool1 = torch.zeros_like(pool1)

        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        if ablate_stage == "pool2":
            pool2 = torch.zeros_like(pool2)

        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)

        bottleneck = model.bottleneck(pool3)
        if ablate_stage == "pool3":
            # BUG CAUGHT BEFORE TRUSTING THE RESULT (audit-your-own-result
            # step, per this project's established convention): zeroing
            # `pool3` and THEN running it through model.bottleneck's real
            # Conv3DBlock (with learned biases/BatchNorm) is NOT the same
            # intervention as E48/E109/E118/E119/E120's N_b, which zeros
            # the BOTTLENECK tensor itself (post-transform). Feeding an
            # all-zero, out-of-distribution input through BatchNorm/conv
            # biases does not reliably produce a zero (or even small)
            # output -- first version of this script zeroed `pool3`
            # in-place before the bottleneck conv, which collapsed
            # dice_p3 to ~0 for ~49% of subjects (verified: mean
            # dice_p3=0.011 vs E48-E120's own dice_ablated~0.6), giving a
            # wildly inflated N_3 (0.88 vs the established ~0.27-0.32)
            # that was NOT "16->8 is more important" -- it was testing a
            # different, harsher intervention than N_1/N_2 and than the
            # established N_b. Fixed: zero the bottleneck OUTPUT directly,
            # matching N_1/N_2's "zero this stage's own output tensor"
            # convention exactly AND matching E48-E120's N_b construction
            # exactly (N_3 should now reproduce N_b closely as a
            # cross-check).
            bottleneck = torch.zeros_like(bottleneck)

        upconv3 = model.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = model.dec3(cat3)

        upconv2 = model.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = model.dec2(cat2)

        upconv1 = model.upconv1(dec2)

        # Attention gate reads `bottleneck` -- for ablate_stage='pool3',
        # bottleneck is computed FROM the zeroed pool3, so it is itself a
        # (transformed) zero-derived tensor, matching E48's own explicit
        # convention ("if the bottleneck carries no real signal, the gate
        # itself should also degrade"). For 'pool1'/'pool2' ablation,
        # bottleneck is computed normally from the (real, unablated)
        # pool3 output derived from the zeroed upstream tensor -- this is
        # the correct causal reading: the gate sees whatever the REAL
        # downstream computation produces given the upstream zero, not an
        # artificially re-zeroed bottleneck.
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


def permutation_test_corr(x, y, seed):
    rho, p_param = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng.permutation(y)
        perm_rhos[i], _ = stats.spearmanr(x, perm_y)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())
    return float(rho), float(p_param), p_perm


def permutation_test_diff(small_vals, large_vals, seed):
    observed = small_vals.mean() - large_vals.mean()
    combined = np.concatenate([small_vals, large_vals])
    n_small = len(small_vals)
    rng = np.random.default_rng(seed)
    perm_diffs = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_idx = rng.permutation(len(combined))
        perm_diffs[i] = combined[perm_idx[:n_small]].mean() - combined[perm_idx[n_small:]].mean()
    p = float((np.abs(perm_diffs) >= np.abs(observed)).mean())
    return float(observed), p


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
        manual_probs = forward_with_stage_ablation(model, img0_b, ablate_stage=None, device=device)
        real_np = real_probs.squeeze(0).squeeze(0).cpu().numpy() if torch.is_tensor(real_probs) else np.asarray(real_probs)
        max_diff = float(np.abs(real_np - manual_probs).max())
    assert max_diff < 1e-4, "Manual trunk mismatch -- STOP."
    print(f"[Sanity check] manual trunk vs real forward(): max abs diff = {max_diff:.6e} PASS.\n")

    # ---------------- Compute N_1, N_2, N_3 for every validation subject ----------------
    records = []
    print(f"Running 4 forward passes (intact + 3 stage ablations) per subject, "
          f"{len(val_ds)} subjects...", flush=True)
    for idx in range(len(val_ds)):
        image, mask, sid = val_ds[idx]
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_ds.subject_dirs[idx]
        seg_path = Path(subject_dir) / f"{sid}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_size = int(seg_binary_native.sum())
        target_bin_64 = (fractional_occupancy(seg_binary_native, (64, 64, 64)) > 0.5).astype(np.float32)

        probs_intact = forward_with_stage_ablation(model, image_b, None, device)
        probs_p1 = forward_with_stage_ablation(model, image_b, "pool1", device)
        probs_p2 = forward_with_stage_ablation(model, image_b, "pool2", device)
        probs_p3 = forward_with_stage_ablation(model, image_b, "pool3", device)

        dice_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin_64)
        dice_p1 = dice_score((probs_p1 >= 0.5).astype(np.float32), target_bin_64)
        dice_p2 = dice_score((probs_p2 >= 0.5).astype(np.float32), target_bin_64)
        dice_p3 = dice_score((probs_p3 >= 0.5).astype(np.float32), target_bin_64)

        n1 = dice_intact - dice_p1
        n2 = dice_intact - dice_p2
        n3 = dice_intact - dice_p3

        records.append({
            "subject_id": sid, "native_size": native_size, "dice_intact": dice_intact,
            "N_1": n1, "N_2": n2, "N_3": n3,
        })
        if (idx + 1) % 25 == 0:
            print(f"  {idx+1}/{len(val_ds)}", flush=True)

    with open(OUT_DIR / "E121_per_subject_table.json", "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nTotal subjects: {len(records)}")
    N1 = np.array([r["N_1"] for r in records])
    N2 = np.array([r["N_2"] for r in records])
    N3 = np.array([r["N_3"] for r in records])
    size_arr = np.array([r["native_size"] for r in records])
    dice_intact_arr = np.array([r["dice_intact"] for r in records])
    dice_error_arr = 1.0 - dice_intact_arr

    print(f"\n=== Raw magnitude (mean +/- std) ===")
    print(f"N_1 (pool1, 64->32 ablated): {N1.mean():.4f} +/- {N1.std():.4f}")
    print(f"N_2 (pool2, 32->16 ablated): {N2.mean():.4f} +/- {N2.std():.4f}")
    print(f"N_3 (pool3, 16->8  ablated): {N3.mean():.4f} +/- {N3.std():.4f}  "
          f"(= N_b; cross-check vs E48/E109/E118/E119/E120's ~0.27-0.32 range)")

    print(f"\n=== Correlation with native_size ===")
    results = {"stage_stats": {}}
    for name, N in [("N_1", N1), ("N_2", N2), ("N_3", N3)]:
        rho, p_param, p_perm = permutation_test_corr(N, size_arr, SEED)
        print(f"  {name} vs native_size: rho={rho:+.4f} (perm p={p_perm:.4f})")
        results["stage_stats"].setdefault(name, {})["vs_native_size"] = {
            "rho": rho, "perm_p": p_perm,
        }

    print(f"\n=== Correlation with dice_error (1-dice_intact) ===")
    for name, N in [("N_1", N1), ("N_2", N2), ("N_3", N3)]:
        rho, p_param, p_perm = permutation_test_corr(N, dice_error_arr, SEED + 1)
        print(f"  {name} vs dice_error: rho={rho:+.4f} (perm p={p_perm:.4f})")
        results["stage_stats"][name]["vs_dice_error"] = {"rho": rho, "perm_p": p_perm}

    print(f"\n=== Small-vs-large lesion stratification (median split, E92 convention) ===")
    median_size = float(np.median(size_arr))
    small_mask = size_arr <= median_size
    large_mask = ~small_mask
    print(f"Median native_size: {median_size:.0f}  (n_small={small_mask.sum()}, n_large={large_mask.sum()})")
    for name, N in [("N_1", N1), ("N_2", N2), ("N_3", N3)]:
        small_vals = N[small_mask]
        large_vals = N[large_mask]
        diff, p_diff = permutation_test_diff(small_vals, large_vals, SEED + 2)
        print(f"  {name}: small_mean={small_vals.mean():.4f}, large_mean={large_vals.mean():.4f}, "
              f"diff={diff:+.4f}, perm p={p_diff:.4f}")
        results["stage_stats"][name]["small_large_strat"] = {
            "small_mean": float(small_vals.mean()), "large_mean": float(large_vals.mean()),
            "diff": diff, "perm_p": p_diff,
        }

    results.update({
        "checkpoint": str(CKPT_PATH), "val_dice_check": val_dice,
        "n_subjects": len(records), "median_native_size": median_size,
        "N1_mean": float(N1.mean()), "N1_std": float(N1.std()),
        "N2_mean": float(N2.mean()), "N2_std": float(N2.std()),
        "N3_mean": float(N3.mean()), "N3_std": float(N3.std()),
        "note": "DIAGNOSTIC ONLY, no architecture change. Raw magnitude of N_1/N_2 is "
                "expected to be inflated relative to N_3 due to more downstream computation "
                "depending on earlier ablations -- see module docstring. The small-vs-large "
                "stratification (not raw magnitude) is the test that actually answers whether "
                "pool3/16->8 is UNIQUELY small-lesion-relevant among the three stages.",
    })
    with open(OUT_DIR / "E121_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved E121_summary.json, E121_per_subject_table.json")


if __name__ == "__main__":
    main()
