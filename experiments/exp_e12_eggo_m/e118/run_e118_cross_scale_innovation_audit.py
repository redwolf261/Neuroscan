"""
Phase E118: Cross-Scale Innovation Audit (the "E115" the user specified in
chat -- renumbered to E118 because experiments/exp_e12_eggo_m/e115/ is
already occupied by an unrelated, unstarted design doc,
PHASE_E115_CAUSALXNET_PCGRAD_DESIGN.md, that must not be overwritten or
confused with this phase).

HYPOTHESIS UNDER TEST (pre-registered, per this session's own convention):
    cross-scale innovation at the enc3(16^3) -> bottleneck(8^3) transition
    predicts causal bottleneck necessity N_b, beyond what lesion volume,
    Dice error, and gradient-allocation G_b already explain.

BACKGROUND / WHY THIS IS NOT JUST E109 RESTATED:
E109 tested whether the bottleneck is SELF-predictive (does a probe on
the bottleneck's own pooled channels predict N_b?) and found yes,
distributed, unexplained by magnitude/variance. THIS phase asks a
different, cross-scale question: how much of the bottleneck is NOT
predictable FROM THE PRECEDING SCALE (enc3), and does that specific
residual/innovation energy track N_b. These are related but distinct
claims -- E109 stays within one tensor, E118 crosses the enc3->bottleneck
transition E92 already causally localized as the loss site.

METHOD (deliberately boring per explicit instruction -- linear/1x1x1
predictor, NOT a neural network, so the predictor itself is not a
research contribution):
  1. Load canonical checkpoint (E46 AttnGate_seed0, val_dice=
     0.9101624600589275), assert identity, sanity-check manual trunk vs
     real forward() (E109's convention).
  2. For every validation subject (all 125, same set as E90-E92/E109):
     extract Z16 = enc3 (128ch @ 16^3) and Z8 = bottleneck (256ch @ 8^3)
     -- these are the exact tensors E92 causally localized the loss
     between. Also compute via E109's exact
     forward_with_bottleneck_ablation: N_b (bottleneck-zeroing Dice
     drop), dice_intact (for Dice-error control), native_size (lesion
     volume), and via E89's exact compute_allocation: G_b (gradient-norm
     allocation to bottleneck-path parameters).
  3. Fit Q: a LINEAR (1x1x1 conv, no bias-free assumption, no nonlinearity)
     map from avg-pooled Z16 (128-dim) to avg-pooled Z8 (256-dim), fit by
     least-squares on TRAINING subjects (same probe-train/probe-test
     split convention as E90-E92: every 4th subject in size-sorted order
     held out for test/val use here). This avoids any neural-network
     predictor and gives a closed-form, non-overfittable Q.
  4. On held-out validation subjects: R8 = Z8_pooled - Q(Z16_pooled).
     I_rel = ||R8||^2 / (||Z8_pooled||^2 + eps) -- scalar innovation score
     per subject.
  5. Test Spearman(I_rel, N_b), plus I_rel vs native_size, dice_error
     (1-dice_intact), G_b, and the pre-registered KEY test: partial
     Spearman(I_rel, N_b | native_size, dice_error, G_b) via residualized
     multi-covariate linear regression (same partial-correlation pattern
     as E71/E109, extended to 3 covariates).
  6. Spatial-innovation test: reconstruct full-resolution (not pooled)
     R8 map per-channel-averaged, downsample lesion mask to 8^3, compute
     I_lesion = energy of R8 inside lesion voxels / total R8 energy, vs.
     a SIZE-MATCHED random non-lesion control region (not just "outside
     lesion" generically, to control for the trivial fact that lesion
     regions are a small fraction of the volume).

PRE-DECLARED DECISION RULE (verbatim from the user's spec):
  Case A: I_rel does NOT correlate with N_b (permutation p>=0.05) ->
          KILL, no further steps.
  Case B: I_rel correlates with N_b, but the partial correlation
          controlling for {native_size, dice_error, G_b} drops to
          p>=0.05 or flips sign -> KILL as explained by generic
          difficulty, not a genuine cross-scale-innovation mechanism.
  Case C: partial correlation survives (p<0.05, same sign) but the
          lesion-localization test (step 6) does NOT show innovation
          concentrated in lesion voxels beyond the size-matched control
          -> report as an interesting representation-level phenomenon,
          but explicitly NOT sufficient grounds to propose an
          intervention yet.
  Case D ("jackpot"): partial correlation survives AND innovation is
          spatially concentrated in lesion regions specifically (more so
          for small lesions) -> STOP, do a full prior-art audit before
          any implementation, per explicit instruction not to build
          anything yet regardless of outcome here.

NO ARCHITECTURE CHANGE, NO TRAINING, NO NEW LOSS IN THIS SCRIPT. Read-only
diagnostic on the frozen canonical checkpoint.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
EPS = 1e-12

CKPT_PATH = (project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs"
             / "AttnGate_seed0" / "checkpoints" / "best.pth")
EXPECTED_VAL_DICE = 0.9101624600589275

# Bottleneck-path parameter name prefixes, IDENTICAL definition to E89's
# bottleneck_names convention (gradient allocated to the bottleneck
# Sequential block specifically), used for G_b.
BOTTLENECK_PARAM_PREFIXES = ("bottleneck.",)


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    if denom == 0:
        return 1.0
    return float(2 * tp / denom)


def dice_loss(probs, target):
    tp = (probs * target).sum()
    denom = probs.sum() + target.sum()
    return 1.0 - (2 * tp + 1.0) / (denom + 1.0)


def fractional_occupancy(seg_binary_native, shape):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=shape, mode="area").squeeze().numpy()
    return frac


def forward_with_bottleneck_ablation(model, image, ablate, device):
    """Bit-for-bit identical to E109's (itself copied from E48's verified
    construction) -- zero bottleneck BEFORE it feeds both upconv3 and the
    attention gate when ablate=True. Also returns enc3 (Z16) and the
    intact bottleneck (Z8) so this single forward pass serves both the
    N_b computation AND the innovation-audit feature extraction (avoids a
    second, possibly inconsistent, forward pass)."""
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck_intact = model.bottleneck(pool3)

        bottleneck = torch.zeros_like(bottleneck_intact) if ablate else bottleneck_intact

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

    return (probs.squeeze(0).squeeze(0).cpu().numpy(),
            enc3.squeeze(0), bottleneck_intact.squeeze(0))


def compute_g_b(model, image_b, target_bin_t, device):
    """IDENTICAL to E89's compute_allocation, restricted to bottleneck.*
    parameters -- fraction of total gradient L2-norm allocated to the
    bottleneck Sequential block for the real (unablated) forward pass."""
    model.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.requires_grad_(True)
    out = model(image_b)
    probs = out["probs"] if isinstance(out, dict) else out
    loss = dice_loss(probs, target_bin_t)
    loss.backward()

    bottleneck_sq_sum = 0.0
    total_sq_sum = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g_sq = float((p.grad ** 2).sum().item())
        total_sq_sum += g_sq
        if name.startswith(BOTTLENECK_PARAM_PREFIXES):
            bottleneck_sq_sum += g_sq

    model.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.requires_grad_(False)
    if total_sq_sum <= 0:
        return 0.0
    return (bottleneck_sq_sum ** 0.5) / (total_sq_sum ** 0.5)


def permutation_test_corr(x, y, seed):
    rho, p_param = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng.permutation(y)
        perm_rhos[i], _ = stats.spearmanr(x, perm_y)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())
    return float(rho), float(p_param), p_perm


def residualize(y, covariates):
    """y, covariates: 1D / 2D arrays over subjects. Returns residual of y
    after OLS regression on [1, covariates...]. Same pattern as E71/E109's
    single-covariate partial correlation, extended to multiple covariates."""
    X = np.column_stack([np.ones(len(y))] + [covariates[:, j] for j in range(covariates.shape[1])])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


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
    print(f"Loaded canonical checkpoint: val_dice={val_dice}")
    assert val_dice is not None and abs(float(val_dice) - EXPECTED_VAL_DICE) < 1e-6, \
        f"Checkpoint identity check FAILED: expected {EXPECTED_VAL_DICE}, got {val_dice}"
    print("[Sanity check] checkpoint identity PASS.")

    val_ds = BraTSDataset(root_dir=str(project_root / "Dataset" / "Training"), split="val",
                          val_split=0.1, target_shape=(64, 64, 64), normalize=True)

    with torch.no_grad():
        img0, _, _ = val_ds[0]
        img0_b = img0.unsqueeze(0).to(device)
        real_out = model(img0_b)
        real_probs = real_out["probs"] if isinstance(real_out, dict) else real_out
        manual_probs, _, _ = forward_with_bottleneck_ablation(model, img0_b, ablate=False, device=device)
        real_np = real_probs.squeeze(0).squeeze(0).cpu().numpy() if torch.is_tensor(real_probs) else np.asarray(real_probs)
        max_diff = float(np.abs(real_np - manual_probs).max())
    print(f"[Sanity check] manual trunk vs real forward(): max abs diff = {max_diff:.6e}")
    assert max_diff < 1e-4, "Manual trunk mismatch -- STOP."
    print("[Sanity check] PASS.\n")

    # ---------------- Step 2: extract per-subject data ----------------
    records = []
    print(f"Extracting {len(val_ds)} validation subjects (single forward pass each)...", flush=True)
    for idx in range(len(val_ds)):
        image, mask, sid = val_ds[idx]
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_ds.subject_dirs[idx]
        seg_path = Path(subject_dir) / f"{sid}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_size = int(seg_binary_native.sum())
        target_bin_64 = (fractional_occupancy(seg_binary_native, (64, 64, 64)) > 0.5).astype(np.float32)
        target_bin_t = torch.from_numpy(target_bin_64).unsqueeze(0).unsqueeze(0).to(device)

        probs_intact, enc3, bottleneck = forward_with_bottleneck_ablation(model, image_b, ablate=False, device=device)
        probs_ablated, _, _ = forward_with_bottleneck_ablation(model, image_b, ablate=True, device=device)
        dice_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin_64)
        dice_ablated = dice_score((probs_ablated >= 0.5).astype(np.float32), target_bin_64)
        n_b = dice_intact - dice_ablated

        g_b = compute_g_b(model, image_b, target_bin_t, device)

        # 8^3 lesion mask for the spatial-innovation test (step 6).
        lesion_mask_8 = (fractional_occupancy(seg_binary_native, (8, 8, 8)) > 0.5).astype(np.float32)

        records.append({
            "subject_id": sid, "enc3": enc3.cpu(), "bottleneck": bottleneck.cpu(),
            "native_size": native_size, "dice_intact": dice_intact, "n_b": n_b, "g_b": g_b,
            "lesion_mask_8": lesion_mask_8,
        })
        if (idx + 1) % 25 == 0:
            print(f"  extracted {idx+1}/{len(val_ds)}", flush=True)

    print(f"\nTotal subjects: {len(records)}")
    n_b_arr = np.array([r["n_b"] for r in records])
    print(f"N_b: mean={n_b_arr.mean():.4f} std={n_b_arr.std():.4f} "
          f"(sanity check vs E48/E109's ~0.32 mean on this exact ablation construction)")

    # ---------------- Step 3: split, fit boring linear Q ----------------
    # IDENTICAL split convention to E90-E92: every 4th subject in
    # size-sorted order held out (here used as the linear-Q fit/eval
    # split, since Q is closed-form least-squares, not a trained probe --
    # still keep a held-out set to avoid any in-sample leakage).
    sorted_idx = sorted(range(len(records)), key=lambda i: records[i]["native_size"])
    test_flags = [(pos % 4 == 0) for pos in range(len(sorted_idx))]
    fit_idx = [sorted_idx[pos] for pos, f in enumerate(test_flags) if not f]
    eval_idx = [sorted_idx[pos] for pos, f in enumerate(test_flags) if f]
    print(f"\nQ-fit: {len(fit_idx)} subjects, Q-eval (held-out): {len(eval_idx)} subjects "
          f"(IDENTICAL split rule to E90-E92)")

    def pooled_z16(r):
        return r["enc3"].mean(dim=(1, 2, 3)).numpy()  # (128,)

    def pooled_z8(r):
        return r["bottleneck"].mean(dim=(1, 2, 3)).numpy()  # (256,)

    Z16_fit = np.stack([pooled_z16(records[i]) for i in fit_idx])   # (n_fit, 128)
    Z8_fit = np.stack([pooled_z8(records[i]) for i in fit_idx])     # (n_fit, 256)
    X_fit = np.column_stack([np.ones(len(fit_idx)), Z16_fit])       # (n_fit, 129), affine linear

    # BUG CAUGHT BEFORE TRUSTING THE RESULT (audit-your-own-result step, per
    # this project's established convention): n_fit=93 < n_features=129, so
    # plain np.linalg.lstsq is UNDERDETERMINED and returns the minimum-norm
    # EXACT-INTERPOLATION solution (verified: in-sample R^2 was exactly
    # 1.0000). That is memorization, not a genuine predictable/innovation
    # split -- I_rel on held-out subjects would then reflect overfit-solution
    # geometry, not real cross-scale predictability. Fixed by switching to
    # ridge-regularized least squares (still linear/"boring" per spec, just
    # well-posed). Ridge strength chosen via k-fold CV on the fit set alone
    # (never touches eval subjects) so this stays a principled, not
    # hand-tuned, choice.
    from sklearn.linear_model import RidgeCV
    ridge_alphas = np.logspace(-1, 5, 25)
    ridge = RidgeCV(alphas=ridge_alphas, fit_intercept=False)  # intercept already in X_fit's ones column
    ridge.fit(X_fit, Z8_fit)
    Q_beta = ridge.coef_.T  # sklearn stores (n_targets, n_features) -> transpose to (129, 256)
    print(f"Fit RIDGE-regularized linear Q: {X_fit.shape} -> {Z8_fit.shape}, "
          f"beta shape {Q_beta.shape}, selected alpha={ridge.alpha_:.4g}")

    fit_pred = X_fit @ Q_beta
    fit_r2 = 1.0 - ((Z8_fit - fit_pred) ** 2).sum() / ((Z8_fit - Z8_fit.mean(axis=0)) ** 2).sum()
    print(f"Q in-sample R^2 (fit set, ridge): {fit_r2:.4f}")
    assert fit_r2 < 0.999, ("In-sample R^2 still ~1.0 after ridge regularization -- "
                             "predictor is still memorizing, STOP and investigate before trusting I_rel.")

    # ---------------- Step 4: I_rel on held-out subjects ----------------
    eval_records = [records[i] for i in eval_idx]
    for r in eval_records:
        z16 = pooled_z16(r)
        z8 = pooled_z8(r)
        x = np.concatenate([[1.0], z16])
        z8_hat = x @ Q_beta
        resid = z8 - z8_hat
        r["I_rel"] = float((resid ** 2).sum() / (float((z8 ** 2).sum()) + EPS))

    I_rel = np.array([r["I_rel"] for r in eval_records])
    N_b_eval = np.array([r["n_b"] for r in eval_records])
    native_size_eval = np.array([r["native_size"] for r in eval_records])
    dice_error_eval = 1.0 - np.array([r["dice_intact"] for r in eval_records])
    G_b_eval = np.array([r["g_b"] for r in eval_records])

    print(f"\nI_rel on held-out eval set: mean={I_rel.mean():.4f} std={I_rel.std():.4f}")

    # ---------------- Step 5: correlations ----------------
    print("\n=== Step 5: correlation tests (held-out eval subjects, n={}) ===".format(len(eval_records)))
    rho_nb, p_param_nb, p_perm_nb = permutation_test_corr(I_rel, N_b_eval, SEED)
    print(f"Spearman(I_rel, N_b)          = {rho_nb:+.4f} (param p={p_param_nb:.4e}, perm p={p_perm_nb:.4f})")

    rho_size, p_param_size, p_perm_size = permutation_test_corr(I_rel, native_size_eval, SEED + 1)
    print(f"Spearman(I_rel, native_size)  = {rho_size:+.4f} (param p={p_param_size:.4e}, perm p={p_perm_size:.4f})")

    rho_derr, p_param_derr, p_perm_derr = permutation_test_corr(I_rel, dice_error_eval, SEED + 2)
    print(f"Spearman(I_rel, dice_error)   = {rho_derr:+.4f} (param p={p_param_derr:.4e}, perm p={p_perm_derr:.4f})")

    rho_gb, p_param_gb, p_perm_gb = permutation_test_corr(I_rel, G_b_eval, SEED + 3)
    print(f"Spearman(I_rel, G_b)          = {rho_gb:+.4f} (param p={p_param_gb:.4e}, perm p={p_perm_gb:.4f})")

    # Case A check
    case_a_kill = not (rho_nb > 0 and p_perm_nb < 0.05)
    if case_a_kill:
        print("\n=== CASE A: I_rel does NOT significantly predict N_b -- KILL, stopping before partial correlation. ===")
        summary = {
            "checkpoint": str(CKPT_PATH), "val_dice_check": val_dice,
            "n_eval_subjects": len(eval_records), "q_fit_r2": float(fit_r2),
            "I_rel_mean": float(I_rel.mean()), "I_rel_std": float(I_rel.std()),
            "spearman_I_rel_vs_N_b": rho_nb, "perm_p_I_rel_vs_N_b": p_perm_nb,
            "spearman_I_rel_vs_native_size": rho_size, "spearman_I_rel_vs_dice_error": rho_derr,
            "spearman_I_rel_vs_G_b": rho_gb,
            "verdict": "CASE_A_KILL_NO_MARGINAL_CORRELATION",
        }
        with open(OUT_DIR / "E118_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\nSaved E118_summary.json")
        return

    # ---------------- Key test: partial correlation controlling all 3 covariates ----------------
    covariates = np.column_stack([
        np.log(native_size_eval + 1), dice_error_eval, G_b_eval,
    ])
    I_rel_resid = residualize(I_rel, covariates)
    N_b_resid = residualize(N_b_eval, covariates)
    rho_partial, p_partial_param = stats.spearmanr(I_rel_resid, N_b_resid)
    rng = np.random.default_rng(SEED + 4)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng.permutation(N_b_resid)
        perm_rhos[i], _ = stats.spearmanr(I_rel_resid, perm_y)
    p_partial_perm = float((np.abs(perm_rhos) >= np.abs(rho_partial)).mean())

    print(f"\n=== KEY TEST: partial Spearman(I_rel, N_b | native_size, dice_error, G_b) ===")
    print(f"rho_partial = {rho_partial:+.4f} (param p={p_partial_param:.4e}, perm p={p_partial_perm:.4f})")

    case_b_kill = not (rho_partial > 0 and p_partial_perm < 0.05)
    if case_b_kill:
        print("\n=== CASE B: partial correlation does not survive controls -- KILL as generic difficulty. ===")
        summary = {
            "checkpoint": str(CKPT_PATH), "val_dice_check": val_dice,
            "n_eval_subjects": len(eval_records), "q_fit_r2": float(fit_r2),
            "spearman_I_rel_vs_N_b": rho_nb, "perm_p_I_rel_vs_N_b": p_perm_nb,
            "spearman_I_rel_vs_native_size": rho_size, "spearman_I_rel_vs_dice_error": rho_derr,
            "spearman_I_rel_vs_G_b": rho_gb,
            "partial_rho": float(rho_partial), "partial_perm_p": p_partial_perm,
            "verdict": "CASE_B_KILL_EXPLAINED_BY_COVARIATES",
        }
        with open(OUT_DIR / "E118_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\nSaved E118_summary.json")
        return

    # ---------------- Step 6: spatial innovation / lesion localization ----------------
    print("\n=== Step 6: spatial innovation localization (survives-partial-correlation subjects only) ===")
    # Per-voxel (8^3) innovation energy: predict full spatial Z8 from a
    # PER-VOXEL linear map of the correspondingly-located Z16 2x2x2 block
    # mean-pooled to align grids (still linear/boring, now spatial instead
    # of globally pooled). Reuse Q_beta's channel-mixing structure applied
    # per-voxel for consistency with the global test.
    rng2 = np.random.default_rng(SEED + 5)
    lesion_fracs = []
    control_fracs = []
    small_median = float(np.median(native_size_eval))
    small_lesion_fracs, small_control_fracs = [], []
    for r in eval_records:
        enc3 = r["enc3"]  # (128, 16,16,16)
        bottleneck = r["bottleneck"]  # (256, 8,8,8)
        enc3_pooled_spatial = F.avg_pool3d(enc3.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)  # (128,8,8,8)
        z16_flat = enc3_pooled_spatial.reshape(128, -1).T.numpy()  # (512, 128)
        z8_flat = bottleneck.reshape(256, -1).T.numpy()            # (512, 256)
        x_flat = np.column_stack([np.ones(z16_flat.shape[0]), z16_flat])
        z8_hat_flat = x_flat @ Q_beta
        resid_flat = z8_flat - z8_hat_flat
        r8_energy_per_voxel = (resid_flat ** 2).sum(axis=1)  # (512,)

        lesion_mask_flat = r["lesion_mask_8"].reshape(-1)
        lesion_voxels = lesion_mask_flat > 0.5
        n_lesion_vox = int(lesion_voxels.sum())
        total_energy = r8_energy_per_voxel.sum() + EPS

        if n_lesion_vox == 0 or n_lesion_vox == 512:
            continue  # degenerate, skip (empty or full-volume lesion at 8^3)

        lesion_frac = float(r8_energy_per_voxel[lesion_voxels].sum() / total_energy)

        # Size-matched random control: same number of voxels, randomly
        # chosen from NON-lesion voxels, averaged over 20 draws per
        # subject to reduce noise.
        non_lesion_idx = np.where(~lesion_voxels)[0]
        draws = []
        for _ in range(20):
            ctrl_idx = rng2.choice(non_lesion_idx, size=n_lesion_vox, replace=False)
            draws.append(float(r8_energy_per_voxel[ctrl_idx].sum() / total_energy))
        control_frac = float(np.mean(draws))

        lesion_fracs.append(lesion_frac)
        control_fracs.append(control_frac)
        if r["native_size"] <= small_median:
            small_lesion_fracs.append(lesion_frac)
            small_control_fracs.append(control_frac)

    lesion_fracs = np.array(lesion_fracs)
    control_fracs = np.array(control_fracs)
    diff = lesion_fracs - control_fracs
    t_stat, p_ttest = stats.ttest_rel(lesion_fracs, control_fracs)
    print(f"Lesion-region innovation fraction: mean={lesion_fracs.mean():.4f}, "
          f"size-matched control: mean={control_fracs.mean():.4f}, "
          f"paired diff={diff.mean():+.4f} (paired t p={p_ttest:.4e}, n={len(lesion_fracs)})")

    small_diff = np.array(small_lesion_fracs) - np.array(small_control_fracs)
    if len(small_diff) > 5:
        t_stat_small, p_ttest_small = stats.ttest_rel(small_lesion_fracs, small_control_fracs)
        print(f"Small-lesion subset (n={len(small_diff)}): paired diff={small_diff.mean():+.4f} "
              f"(paired t p={p_ttest_small:.4e})")
    else:
        p_ttest_small = None
        print("Small-lesion subset too small for a separate test.")

    localization_confirmed = (diff.mean() > 0) and (p_ttest < 0.05)

    if localization_confirmed:
        decision = "CASE_D_JACKPOT_SURVIVES_PARTIAL_AND_LOCALIZED"
        detail = ("Partial correlation survives controls AND innovation energy is "
                   "significantly concentrated in lesion voxels vs size-matched control. "
                   "STOP HERE per instruction -- do a full prior-art audit before any "
                   "implementation. Do not build anything from this result yet.")
    else:
        decision = "CASE_C_PARTIAL_SURVIVES_NOT_LOCALIZED"
        detail = ("Partial correlation survives controls, but innovation energy is NOT "
                   "significantly concentrated in lesion voxels vs size-matched control. "
                   "Report as an interesting representation-level phenomenon (I_rel "
                   "predicts N_b beyond size/error/gradient-allocation) but NOT sufficient "
                   "grounds to propose an intervention -- the innovation is not "
                   "spatially where the lesion is, so a spatially-targeted mechanism is not "
                   "obviously motivated by this result alone.")

    print(f"\n=== DECISION: {decision} ===")
    print(detail)

    summary = {
        "checkpoint": str(CKPT_PATH), "val_dice_check": val_dice,
        "n_eval_subjects": len(eval_records), "q_fit_r2": float(fit_r2),
        "I_rel_mean": float(I_rel.mean()), "I_rel_std": float(I_rel.std()),
        "spearman_I_rel_vs_N_b": rho_nb, "perm_p_I_rel_vs_N_b": p_perm_nb,
        "spearman_I_rel_vs_native_size": rho_size, "spearman_I_rel_vs_dice_error": rho_derr,
        "spearman_I_rel_vs_G_b": rho_gb,
        "partial_rho_controlling_size_error_gb": float(rho_partial),
        "partial_perm_p": p_partial_perm,
        "lesion_innovation_fraction_mean": float(lesion_fracs.mean()),
        "control_innovation_fraction_mean": float(control_fracs.mean()),
        "lesion_vs_control_paired_diff": float(diff.mean()),
        "lesion_vs_control_paired_p": float(p_ttest),
        "small_lesion_subset_n": len(small_diff),
        "small_lesion_paired_diff": float(small_diff.mean()) if len(small_diff) > 5 else None,
        "small_lesion_paired_p": float(p_ttest_small) if p_ttest_small is not None else None,
        "decision": decision, "detail": detail,
    }
    with open(OUT_DIR / "E118_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E118_summary.json")


if __name__ == "__main__":
    main()
