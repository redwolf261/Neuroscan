"""
Phase E64: Corrected Split-Intervention Audit (pool-path vs. skip-path).

NO TRAINING. NO ARCHITECTURE CHANGE. Pure causal-intervention diagnostic.

CONTEXT: E62 and E63 are RETRACTED as evidence about MaxPool3d (see
docs/phases/PHASE_E62_E63_INVALIDATED_SKIP_CONNECTION_BUG.md). Both
scripts' forward_from_enc1() reused a SINGLE enc1 tensor for two roles --
pool1's input AND the decoder skip connection -- so the permuted tensor
reached both simultaneously. Since nn.MaxPool3d(2) is non-overlapping
(k=2,s=2), pool1(enc1) = pool1(pi(enc1)) EXACTLY for any within-cell
derangement pi (proven, not assumed): the entire pool1->pool2->pool3->
bottleneck->decoder-upsampling computation graph is bit-identical between
intact and permuted runs. Therefore E62/E63's entire measured Dice effect
necessarily came through the skip-connection path, not through anything
MaxPool3d discarded.

THIS PHASE corrects that by constructing TWO INDEPENDENT tensors:
    E_pool  -- feeds ONLY pool1's input (-> pool2 -> pool3 -> bottleneck
               -> upconv3 -> dec3 -> upconv2 -> dec2 -> upconv1)
    E_skip  -- feeds ONLY the decoder's skip connection into dec1 (v3:
               direct concatenation cat1=[upconv1, E_skip]; v5: additionally
               gates E_skip through the bottleneck-conditioned attention
               gate BEFORE concatenation, per v5's own architecture)

FOUR CONDITIONS per subject (2x2 factorial):
    (orig, orig)  -- control, must reproduce real forward() exactly
    (perm, orig)  -- PURE pool-path intervention (E_pool permuted, E_skip
                     real) -- expected effect ~=0 given pool1(enc1)=
                     pool1(pi(enc1)) exactly; this arm is really a
                     confirmatory control for the correction itself
    (orig, perm)  -- PURE skip-path intervention (E_pool real, E_skip
                     permuted) -- isolates the effect E62/E63 actually
                     measured
    (perm, perm)  -- total effect / replication of E62's original
                     (confounded) intervention, for direct comparison

CHECKPOINTS (both run, per explicit user request to separate the gate-
mediated path from a pure ungated skip effect):
  1. v5/E46 attention-gate checkpoint (same as E62/E63) -- skip path here
     is GATE-MEDIATED: E_skip feeds both model.attn_gate1's gating
     computation (as the "x" input to W_x) AND the final enc1_gated
     multiplication. A permuted-skip effect on this checkpoint cannot by
     itself distinguish "raw skip content matters" from "the gate's own
     computation over that content matters" -- reported as GATE-MEDIATED
     SKIP EFFECT, not a pure skip effect, per explicit instruction not to
     mislabel this.
  2. v3/D4-only checkpoint (same checkpoint E58/E59 used) -- NO attention
     gate; skip is a direct nn.cat([upconv1, E_skip], dim=1). This is the
     clean, gate-free control: any effect found here IS a pure
     skip-connection representation effect, unconfounded by any gating
     computation.

PRE-DECLARED DECISION RULE:
  If Delta_pool ~= 0 (not significant) AND Delta_skip > 0 (significant):
    the E62/E63 phenomenon is a SKIP-CONNECTION representation effect,
    not a pooling effect. PMD/RCD/TSQL and other pooling-derived
    proposals remain UNSUPPORTED. Redirect any future design work to the
    skip-connection locus (on whichever checkpoint(s) show the effect).
  If Delta_pool > 0 (significant):
    pool1 itself IS causally involved after all (would be surprising
    given the proven mathematical identity pool1(enc1)=pool1(pi(enc1)),
    and would itself indicate a DIFFERENT bug -- e.g. if E_pool's
    permutation were not actually reaching pool1 as intended -- so this
    outcome triggers a mandatory re-verification, not an immediate
    conclusion).
  If neither Delta_pool nor Delta_skip is significant:
    E62/E63's original effect does not replicate under the corrected,
    unconfounded design -- report plainly, do not force an explanation.
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
from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000
N_BOOT = 2000
N_DERANGEMENTS = 8  # independent derangement draws per subject per arm, matching E62's own convention

CKPT_V5 = project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs" / "AttnGate_seed0" / "checkpoints" / "best.pth"
CKPT_V3_D4ONLY = (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs"
                   / "DeepSup_D4only_seed0" / "checkpoints" / "best.pth")


# ==================== shared utilities ====================

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
    """Fully vectorized derangement construction (same as E62/E63's own,
    reused verbatim -- verified correct there, no change needed here)."""
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


def permute_cells_derangement(enc1_np, rng):
    """Identical construction to E62/E63's own (verified correct there):
    within every non-overlapping 2x2x2 cell, per channel, a random
    derangement of the 8 values -- multiset preserved, position changed."""
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


# ==================== corrected forward passes: TWO independent enc1 tensors ====================

def forward_split_v3(model, e_pool, e_skip, device):
    """UNet3D_v3 trunk (no attention gate), with e_pool feeding pool1's
    input and e_skip feeding ONLY the skip connection into dec1's
    concatenation. These are architecturally independent paths from
    enc1 onward -- verified identical to real forward() when
    e_pool is e_skip is the real enc1 (checked in main())."""
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
        cat1 = torch.cat([upconv1, e_skip], dim=1)  # PURE skip path, no gate
        dec1 = model.dec1(cat1)

        probs = model.seg_head(dec1)
        return probs.squeeze(0).squeeze(0).cpu().numpy()


def forward_split_v5(model, e_pool, e_skip, device):
    """UNet3D_v5 trunk (bottleneck-conditioned attention gate on the
    enc1 skip). e_pool feeds pool1's input; e_skip feeds BOTH the
    attention gate's own computation (W_x(e_skip)) and the final
    enc1_gated = e_skip * psi multiplication -- this is v5's actual
    architecture (the gate reads the skip tensor, it doesn't bypass it),
    so a permuted e_skip on this checkpoint measures a GATE-MEDIATED
    skip effect, not a pure ungated one. Verified identical to real
    forward() when e_pool is e_skip is the real enc1 (checked in
    main())."""
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

        gate = bottleneck
        skip = e_skip  # PURE skip tensor for this arm's intervention
        g = model.attn_gate1.W_g(gate)
        g_up = F.interpolate(g, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = model.attn_gate1.W_x(skip)
        psi = torch.sigmoid(model.attn_gate1.W_psi(F.relu(g_up + x)))

        enc1_gated = e_skip * psi
        cat1 = torch.cat([upconv1, enc1_gated], dim=1)
        dec1 = model.dec1(cat1)

        probs = model.seg_head(dec1)
        return probs.squeeze(0).squeeze(0).cpu().numpy()


def run_checkpoint_audit(ckpt_path, model_cls, forward_fn, label, val_dataset, e48_by_id, device):
    print(f"\n{'='*70}\nRunning split-intervention audit on: {label}\nCheckpoint: {ckpt_path}\n{'='*70}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = model_cls(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"best_val_dice={ckpt.get('best_val_dice')}")

    n = len(val_dataset)

    # ---------------- Sanity check: (orig,orig) reproduces real forward() ----------------
    image0, _, _ = val_dataset[0]
    image0_b = image0.unsqueeze(0).to(device)
    with torch.no_grad():
        real = model(image0_b)["probs"].squeeze(0).squeeze(0).cpu().numpy()
        enc1_0 = model.enc1(image0_b)
    manual = forward_fn(model, enc1_0, enc1_0, device)
    max_diff = float(np.abs(real - manual).max())
    print(f"[Sanity check] forward_split(real,real) vs real forward(): max abs diff = {max_diff:.6e}")
    assert max_diff == 0.0, "Split-forward reimplementation mismatch -- STOP."

    # ---------------- Differential check: THIS is the check that would have caught E62/E63's bug ----------------
    # Permute E_pool only, verify pool1's OWN output is unchanged (the
    # mathematical identity pool1(x)=pool1(pi(x))) -- and verify the
    # final probs DO change when skip is separately permuted, confirming
    # the two paths are genuinely independent in this implementation.
    rng_check = np.random.default_rng(999)
    enc1_0_np = enc1_0.squeeze(0).cpu().numpy()
    perm_check_np = permute_cells_derangement(enc1_0_np, rng_check)
    perm_check = torch.from_numpy(perm_check_np).unsqueeze(0).to(device)
    with torch.no_grad():
        pool1_real = model.pool1(enc1_0)
        pool1_perm = model.pool1(perm_check)
    pool1_diff = float(torch.abs(pool1_real - pool1_perm).max().item())
    print(f"[Differential check] pool1(real enc1) vs pool1(permuted enc1): max abs diff = {pool1_diff:.6e} "
          f"(MUST be 0.0 -- mathematical identity for non-overlapping MaxPool3d)")
    assert pool1_diff == 0.0, "pool1 output changed under a within-cell derangement -- mathematically impossible, STOP."

    probs_pool_perm_only = forward_fn(model, perm_check, enc1_0, device)
    probs_skip_perm_only = forward_fn(model, enc1_0, perm_check, device)
    pool_arm_diff = float(np.abs(probs_pool_perm_only - real).max())
    skip_arm_diff = float(np.abs(probs_skip_perm_only - real).max())
    print(f"[Differential check] max |probs| change, pool-only perm: {pool_arm_diff:.6e} (expected: near/exactly 0)")
    print(f"[Differential check] max |probs| change, skip-only perm: {skip_arm_diff:.6e} (expected: > 0 if a real skip effect exists)")
    assert pool_arm_diff == 0.0, "Pool-only permutation changed the final output -- pool-path is NOT properly isolated, STOP."
    print("[Differential check] PASS: pool-path isolation confirmed exact; skip-path is independently intervenable.\n")

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

        probs_control = forward_fn(model, enc1, enc1, device)
        dice_control = dice_score((probs_control >= 0.5).astype(np.float32), target_bin)

        subj_rng = np.random.default_rng(SEED * 100000 + idx)
        dice_pool_perm_draws = []
        dice_skip_perm_draws = []
        dice_both_perm_draws = []
        for draw in range(N_DERANGEMENTS):
            enc1_perm_np = permute_cells_derangement(enc1_np, subj_rng)
            enc1_perm = torch.from_numpy(enc1_perm_np).unsqueeze(0).to(device)

            probs_pool_perm = forward_fn(model, enc1_perm, enc1, device)
            probs_skip_perm = forward_fn(model, enc1, enc1_perm, device)
            probs_both_perm = forward_fn(model, enc1_perm, enc1_perm, device)

            dice_pool_perm_draws.append(dice_score((probs_pool_perm >= 0.5).astype(np.float32), target_bin))
            dice_skip_perm_draws.append(dice_score((probs_skip_perm >= 0.5).astype(np.float32), target_bin))
            dice_both_perm_draws.append(dice_score((probs_both_perm >= 0.5).astype(np.float32), target_bin))

        mean_dice_pool_perm = float(np.mean(dice_pool_perm_draws))
        mean_dice_skip_perm = float(np.mean(dice_skip_perm_draws))
        mean_dice_both_perm = float(np.mean(dice_both_perm_draws))

        drop_pool = dice_control - mean_dice_pool_perm
        drop_skip = dice_control - mean_dice_skip_perm
        drop_both = dice_control - mean_dice_both_perm

        e48r = e48_by_id.get(subject_id)

        records.append({
            "subject_id": subject_id, "native_size": native_size,
            "dice_control": dice_control,
            "dice_pool_perm": mean_dice_pool_perm, "dice_skip_perm": mean_dice_skip_perm, "dice_both_perm": mean_dice_both_perm,
            "drop_pool": drop_pool, "drop_skip": drop_skip, "drop_both": drop_both,
            "e48_causal_drop": e48r["drop"] if e48r is not None else None,
        })

        if (idx + 1) % 25 == 0:
            print(f"  processed {idx+1}/{n} subjects", flush=True)

    with open(OUT_DIR / f"E64_{label}_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records for {label}.\n")

    # ================= Statistics =================
    native_size = np.array([r["native_size"] for r in records], dtype=np.float64)
    drop_pool = np.array([r["drop_pool"] for r in records], dtype=np.float64)
    drop_skip = np.array([r["drop_skip"] for r in records], dtype=np.float64)
    drop_both = np.array([r["drop_both"] for r in records], dtype=np.float64)
    dice_control = np.array([r["dice_control"] for r in records], dtype=np.float64)

    def paired_tests(drop, tag):
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
        print(f"  [{tag}] mean={drop.mean():+.5f} t_p={t_p:.4e} wilcoxon_p={w_p:.4e} "
              f"sign_flip_perm_p={p_perm:.4f} boot95CI={ci}")
        return {"mean": float(drop.mean()), "t_p": float(t_p), "wilcoxon_p": float(w_p),
                "perm_p": p_perm, "bootstrap_95ci": list(ci)}

    print(f"=== E64 [{label}]: n={len(records)} subjects, {N_DERANGEMENTS} derangement draws/arm ===")
    print(f"Mean dice_control = {dice_control.mean():.4f}\n")
    print("Paired significance tests (H1: mean drop > 0), subject-level:")
    stats_pool = paired_tests(drop_pool, "Delta_pool (pure pool-path)")
    stats_skip = paired_tests(drop_skip, "Delta_skip (pure/gate-mediated skip-path)")
    stats_both = paired_tests(drop_both, "Delta_both (replicates original E62 confound)")

    pool_significant = stats_pool["t_p"] < 0.05 and stats_pool["perm_p"] < 0.05 and drop_pool.mean() > 0
    skip_significant = stats_skip["t_p"] < 0.05 and stats_skip["perm_p"] < 0.05 and drop_skip.mean() > 0

    if pool_significant:
        verdict = "POOL_PATH_INVOLVED_REQUIRES_REVERIFICATION"
    elif skip_significant:
        verdict = "SKIP_PATH_EFFECT_CONFIRMED"
    else:
        verdict = "NEITHER_ARM_SIGNIFICANT"
    print(f"\nDelta_pool significant: {pool_significant}  |  Delta_skip significant: {skip_significant}")
    print(f"=== VERDICT [{label}]: {verdict} ===\n")

    # Size-dependence of the skip effect, matching E62's own convention
    rho_size, p_size_param = stats.spearmanr(native_size, drop_skip)
    rng2 = np.random.default_rng(SEED + 2)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng2.permutation(drop_skip)
        perm_rhos[i], _ = stats.spearmanr(native_size, perm_y)
    p_size_perm = float((np.abs(perm_rhos) >= np.abs(rho_size)).mean())
    print(f"Spearman(native_size, drop_skip) = {rho_size:+.4f} (parametric p={p_size_param:.4e}, "
          f"permutation p={p_size_perm:.4f})")

    summary = {
        "label": label, "checkpoint": str(ckpt_path), "n_subjects": len(records),
        "mean_dice_control": float(dice_control.mean()),
        "stats_pool": stats_pool, "stats_skip": stats_skip, "stats_both": stats_both,
        "pool_significant": bool(pool_significant), "skip_significant": bool(skip_significant),
        "skip_size_dependence": {"rho": float(rho_size), "parametric_p": float(p_size_param), "permutation_p": p_size_perm},
        "verdict": verdict,
    }
    with open(OUT_DIR / f"E64_{label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved E64_{label}_summary.json")
    return summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    print(f"Validation set size: {len(val_dataset)}")

    E48_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"
    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}

    summary_v5 = run_checkpoint_audit(CKPT_V5, UNet3D_v5, forward_split_v5, "v5_gated", val_dataset, e48_by_id, device)
    summary_v3 = run_checkpoint_audit(CKPT_V3_D4ONLY, UNet3D_v3, forward_split_v3, "v3_ungated", val_dataset, e48_by_id, device)

    print("\n" + "=" * 70)
    print("=== E64 FINAL COMBINED REPORT ===")
    print("=" * 70)
    for s in (summary_v5, summary_v3):
        print(f"\n[{s['label']}] Delta_pool mean={s['stats_pool']['mean']:+.5f} (sig={s['pool_significant']}), "
              f"Delta_skip mean={s['stats_skip']['mean']:+.5f} (sig={s['skip_significant']}) "
              f"-> {s['verdict']}")

    with open(OUT_DIR / "E64_combined_summary.json", "w") as f:
        json.dump({"v5_gated": summary_v5, "v3_ungated": summary_v3}, f, indent=2)
    print("\nSaved E64_combined_summary.json")


if __name__ == "__main__":
    main()
