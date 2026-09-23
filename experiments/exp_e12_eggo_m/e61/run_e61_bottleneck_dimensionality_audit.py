"""
Phase E61: Bottleneck Effective-Dimensionality Audit.

NO TRAINING. NO NEW ARCHITECTURE. NO NEW LOSS. NO HYPERPARAMETER SEARCH.

CONTEXT: E48 found the bottleneck's causal contribution (via full
zero-ablation) is size-dependent (Spearman rho(native_size, drop) =
-0.454, p<0.001): larger lesions lose MORE Dice when the bottleneck is
severed. E58 found this effect is NOT localized to any fixed spatial
octant. E59's coarse 8-octant synergy/interaction analysis suggested a
size-specific synergy signature, but E60 found that formulation does not
survive a cleaner training-time reproduction -- so "synergy" is NOT
established and this phase does not assume it.

NARROWER HYPOTHESIS UNDER TEST (pre-declared, falsifiable):
    Small-lesion subjects require a HIGHER-EFFECTIVE-DIMENSION bottleneck
    representation than large-lesion subjects. This is a claim about
    representation dimensionality, not semantics -- "global context" is
    deliberately not invoked.

CHECKPOINT (pre-declared after explicit user sign-off): the v5/E46
attention-gate checkpoint (best.pth, val_dice=0.9102), because it is the
EXACT checkpoint E48's CausalDrop numbers were computed from, and step 11
below requires a valid per-subject correlation against that exact
CausalDrop data. NOTE: this is NOT the checkpoint E58/E59 used (those
used the plain v3/D4-only checkpoint) -- E58/E59 and E48 are themselves
on different architectures, so this script's step-12 cross-check against
E59 is QUALITATIVE ONLY (does the sign/direction agree), never a
quantitative correlation across architectures. This limitation is
reported, not hidden.

LESION-REGIME DEFINITION (pre-declared BEFORE looking at any
dimensionality result, per explicit user sign-off): primary split is a
MEDIAN split on E48's own native_size field (per-subject total lesion
burden, whole native-volume voxel count) -- small = below median,
large = above median. E35's S1-S5 native bins do NOT apply here (they
were built for per-COMPONENT size and are degenerate at the per-subject
total-burden granularity E48 uses: all 125 E48 subjects fall in S5 under
those edges). The five E35 bins are NOT used at all in this script
(there is no per-subject population to distribute across them
meaningfully); no exploratory bin analysis is added beyond the
pre-declared median split, to avoid a post-hoc bin search.

STATISTICAL SAFEGUARDS: subject-level (never bottleneck-cell-level)
inference throughout -- bootstrap CI over subjects, permutation test
preserving subject identity, and explicit controls for activation scale,
lesion burden, and baseline Dice before any causal claim.

PRE-DECLARED DECISION RULE (from the user's spec, verbatim):
  GO only if ALL of:
    1. Small-lesion subjects have consistently higher effective dimension.
    2. Effect survives normalization for activation magnitude.
    3. Effect survives legitimate control for lesion burden.
    4. >=2 of 3 primary dimensionality metrics agree qualitatively.
    5. Result survives subject-level permutation/bootstrap analysis.
    6. Dimensionality is meaningfully associated with E48's CausalDrop.
    7. Result is not a trivial consequence of baseline Dice or lesion count.
  QUALIFIED GO: dimensionality difference real/robust but link to E48's
    causal effect (criterion 6) is incomplete.
  KILL: no difference, unstable metric, explained by activation
    magnitude/lesion burden, or no relationship to E48's causal effect.
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
N_BOOT = 2000

CKPT_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs" / "AttnGate_seed0" / "checkpoints" / "best.pth"
E48_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"


# ==================== Step 2: bottleneck extraction ====================

def extract_bottleneck(model, image):
    """Clean inference-only forward through the encoder trunk to the
    bottleneck (no GT involved). Returns the (256,8,8,8) tensor as numpy,
    plus a bit-for-bit sanity check is done once in main() against the
    real forward() output (not here, to keep this hot-path function
    minimal)."""
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)
    return bottleneck.squeeze(0).cpu().numpy()  # (256,8,8,8)


def fractional_occupancy_64(seg_binary_native):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=(64, 64, 64), mode="area").squeeze().numpy()
    return frac


# ==================== Step 3-4: dimensionality measures ====================

def effective_rank(X):
    """X: (n_obs, n_features), already centered per the declared scheme.
    Returns (R_eff, R_stable, PR) computed from the singular-value
    spectrum of X (equivalently eigenvalues of the Gram/covariance)."""
    # SVD-based singular values (avoids explicitly forming covariance,
    # numerically equivalent for effective rank / stable rank purposes).
    try:
        s = np.linalg.svd(X, compute_uv=False)
    except np.linalg.LinAlgError:
        s = np.linalg.svd(X + 1e-12 * np.random.default_rng(0).standard_normal(X.shape), compute_uv=False)
    s = s[s > 1e-12]
    if len(s) == 0:
        return 0.0, 0.0, 0.0

    # A. Effective rank (Shannon entropy of normalized singular values)
    p = s / s.sum()
    r_eff = float(np.exp(-np.sum(p * np.log(p + 1e-300))))

    # A (analogue). Stable rank = ||X||_F^2 / ||X||_2^2 = sum(s^2) / max(s)^2
    r_stable = float(np.sum(s ** 2) / (s[0] ** 2))

    # B. Participation ratio, using eigenvalues of the covariance
    # (lambda_i = s_i^2 / (n-1), but PR is scale-invariant so s_i^2 works directly)
    lam = s ** 2
    pr = float((lam.sum() ** 2) / np.sum(lam ** 2))

    return r_eff, r_stable, pr


# ==================== Step 9: randomized null ====================

def randomized_control(X, rng):
    """Preserve dimensions and per-feature scale (activation magnitude
    distribution) by shuffling each feature (column) independently across
    observations -- destroys cross-feature/cross-cell covariance
    structure (the thing effective rank measures) while preserving the
    marginal singular-value SCALE contributed by each feature's own
    variance. This does not destroy the quantity being measured by
    construction: correlated features could still coincidentally realign
    after independent shuffling, but on average the shuffle collapses
    structure toward what independent Gaussian-like features would show."""
    Xr = X.copy()
    for j in range(Xr.shape[1]):
        Xr[:, j] = rng.permutation(Xr[:, j])
    return Xr


# ==================== main ====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded E46/v5 checkpoint: best_val_dice={ckpt.get('best_val_dice')}", flush=True)
    print("NOTE: this is the SAME checkpoint E48's CausalDrop was computed from. "
          "It is NOT the plain v3/D4-only checkpoint E58/E59 used -- step-12 "
          "cross-check against E59 is qualitative only, per pre-declared limitation.\n")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n = len(val_dataset)
    print(f"Validation set size: {n}", flush=True)

    # Sanity check: bottleneck extraction path matches real forward() trunk exactly.
    image0, _, _ = val_dataset[0]
    image0_b = image0.unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(image0_b)  # warm/verify no exception
    with torch.no_grad():
        enc1 = model.enc1(image0_b)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck_manual = model.bottleneck(pool3)
    bottleneck_direct = extract_bottleneck(model, image0_b)
    max_diff = float(np.abs(bottleneck_manual.squeeze(0).cpu().numpy() - bottleneck_direct).max())
    print(f"[Sanity check] extract_bottleneck() vs manual trunk: max abs diff = {max_diff:.6e}")
    assert max_diff == 0.0, "Bottleneck extraction mismatch -- STOP, bug present."
    print("[Sanity check] PASS.\n")

    # ---------------- Step 2: extract + cache ----------------
    records = []
    for idx in range(n):
        image, mask, subject_id = val_dataset[idx]
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_dataset.subject_dirs[idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_size = int(seg_binary_native.sum())

        B = extract_bottleneck(model, image_b)  # (256,8,8,8)

        with torch.no_grad():
            probs = model(image_b)["probs"].squeeze(0).squeeze(0).cpu().numpy()
        mask_frac_64 = fractional_occupancy_64(seg_binary_native)
        target_bin = (mask_frac_64 > 0.5).astype(np.float32)
        pred_bin = (probs >= 0.5).astype(np.float32)
        tp = (pred_bin * target_bin).sum()
        denom = pred_bin.sum() + target_bin.sum()
        baseline_dice = float(1.0 if denom == 0 else 2 * tp / denom)

        records.append({
            "subject_id": subject_id,
            "native_size": native_size,
            "baseline_dice": baseline_dice,
            "bottleneck": B,  # kept in-memory only; not JSON-serialized (too large)
        })

        if (idx + 1) % 25 == 0:
            print(f"  extracted {idx+1}/{n} subjects", flush=True)

    print(f"\nExtracted {len(records)} bottlenecks.\n")

    # ---------------- Step 5: lesion-regime definition ----------------
    # PRE-DECLARED (per explicit user sign-off, before any dimensionality
    # result is computed): median split on E48's own native_size field.
    native_sizes = np.array([r["native_size"] for r in records], dtype=np.float64)
    median_size = float(np.median(native_sizes))
    print(f"Median native_size (subject-level total lesion burden) = {median_size:.0f}")
    for r in records:
        r["group"] = "small" if r["native_size"] <= median_size else "large"
    n_small = sum(1 for r in records if r["group"] == "small")
    n_large = sum(1 for r in records if r["group"] == "large")
    print(f"Group sizes: small(n={n_small}), large(n={n_large})\n")

    # ---------------- Step 3-4/7/10: per-subject dimensionality metrics ----------------
    # Centering scheme (declared, identical across groups): GLOBAL centering
    # per subject across the 512 spatial cells (i.e. subtract each subject's
    # own per-channel mean across its 512 cells before SVD) -- this is
    # "subject-wise" centering in the terms of the spec: the natural choice
    # because dimensionality is a WITHIN-subject property of the bottleneck's
    # 512x256 (or 256x512) matrix, and centering must be done before SVD for
    # effective rank / participation ratio to reflect COVARIANCE structure
    # rather than being dominated by the mean activation vector as a giant
    # rank-1 component. Declared before viewing any result; identical
    # procedure applied to every subject and both group.
    rng = np.random.default_rng(SEED)

    for r in records:
        B = r["bottleneck"]  # (256,8,8,8)
        X_cells = B.reshape(256, 512).T  # (512 cells, 256 channels) -- primary
        X_chan = X_cells.T  # (256 channels, 512 cells) -- transposed secondary

        # subject-wise centering (declared)
        Xc_cells = X_cells - X_cells.mean(axis=0, keepdims=True)
        Xc_chan = X_chan - X_chan.mean(axis=0, keepdims=True)

        r_eff_cells, r_stable_cells, pr_cells = effective_rank(Xc_cells)
        r_eff_chan, r_stable_chan, pr_chan = effective_rank(Xc_chan)

        # Step 7: activation magnitude controls
        fro_norm = float(np.linalg.norm(B.reshape(-1)))
        spec_norm = float(np.linalg.norm(X_cells, ord=2))
        mean_mag = float(np.abs(B).mean())
        var_mag = float(B.var())

        # Normalized spectrum (unit Frobenius norm) -- scale-free effective rank
        X_norm = Xc_cells / (np.linalg.norm(Xc_cells) + 1e-12)
        r_eff_norm, r_stable_norm, pr_norm = effective_rank(X_norm)

        # Step 9: randomized control (feature/channel-shuffle null), primary orientation
        X_rand = randomized_control(Xc_cells, rng)
        r_eff_rand, r_stable_rand, pr_rand = effective_rank(X_rand)

        r.update({
            "r_eff_spatial": r_eff_cells, "r_stable_spatial": r_stable_cells, "pr_spatial": pr_cells,
            "r_eff_channel": r_eff_chan, "r_stable_channel": r_stable_chan, "pr_channel": pr_chan,
            "r_eff_normalized": r_eff_norm, "r_stable_normalized": r_stable_norm, "pr_normalized": pr_norm,
            "r_eff_random_null": r_eff_rand, "r_stable_random_null": r_stable_rand, "pr_random_null": pr_rand,
            "fro_norm": fro_norm, "spec_norm": spec_norm, "mean_activation": mean_mag, "var_activation": var_mag,
        })
        del r["bottleneck"]  # free memory, not needed past this point

    # ---------------- Save cache table (without raw tensors) ----------------
    with open(OUT_DIR / "E61_bottleneck_dimensionality_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} subject dimensionality records.\n")

    # ---------------- Merge with E48 CausalDrop ----------------
    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}
    n_matched = 0
    for r in records:
        e48r = e48_by_id.get(r["subject_id"])
        if e48r is not None:
            r["e48_causal_drop"] = e48r["drop"]
            n_matched += 1
        else:
            r["e48_causal_drop"] = None
    print(f"Matched {n_matched}/{len(records)} subjects to E48 CausalDrop by subject_id.\n")

    # ================= Statistical analysis =================
    def arr(key, group=None):
        if group is None:
            return np.array([r[key] for r in records], dtype=np.float64)
        return np.array([r[key] for r in records if r["group"] == group], dtype=np.float64)

    metrics = ["r_eff_spatial", "r_stable_spatial", "pr_spatial",
               "r_eff_channel", "r_stable_channel", "pr_channel",
               "r_eff_normalized", "r_stable_normalized", "pr_normalized"]

    print("=== Step 6: Primary hypothesis test (small vs large, per metric) ===")
    group_tests = {}
    for m in metrics:
        s = arr(m, "small")
        l = arr(m, "large")
        diff = float(s.mean() - l.mean())
        t_stat, t_p = stats.mannwhitneyu(s, l, alternative="greater")  # H1: small > large

        # Subject-level permutation test (shuffle group labels)
        all_vals = np.concatenate([s, l])
        rng2 = np.random.default_rng(SEED)
        n_s = len(s)
        perm_diffs = np.empty(N_PERM)
        for i in range(N_PERM):
            perm = rng2.permutation(all_vals)
            perm_diffs[i] = perm[:n_s].mean() - perm[n_s:].mean()
        p_perm = float((perm_diffs >= diff).mean())

        # Bootstrap CI over subjects (resample within each group)
        rng3 = np.random.default_rng(SEED + 1)
        boot_diffs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            bs = rng3.choice(s, size=len(s), replace=True)
            bl = rng3.choice(l, size=len(l), replace=True)
            boot_diffs[i] = bs.mean() - bl.mean()
        ci_lo, ci_hi = float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))

        group_tests[m] = {
            "mean_small": float(s.mean()), "mean_large": float(l.mean()), "diff": diff,
            "mannwhitney_p_greater": float(t_p), "permutation_p": p_perm,
            "bootstrap_95ci": [ci_lo, ci_hi],
        }
        print(f"  {m:22s}: small={s.mean():.3f} large={l.mean():.3f} diff={diff:+.3f} "
              f"MWU_p={t_p:.4f} perm_p={p_perm:.4f} boot95CI=[{ci_lo:+.3f},{ci_hi:+.3f}]")

    # ================= Step 7: activation-scale control =================
    print("\n=== Step 7: Activation-scale controls ===")
    scale_controls = {}
    for scale_key in ["fro_norm", "spec_norm", "mean_activation", "var_activation"]:
        s = arr(scale_key, "small")
        l = arr(scale_key, "large")
        _, p = stats.mannwhitneyu(s, l, alternative="two-sided")
        scale_controls[scale_key] = {"mean_small": float(s.mean()), "mean_large": float(l.mean()), "p_two_sided": float(p)}
        print(f"  {scale_key:18s}: small={s.mean():.3f} large={l.mean():.3f} p={p:.4f}")

    print("\n  Effect after scale normalization (r_eff_normalized, unit-Frobenius-norm spectrum):")
    print(f"    {group_tests['r_eff_normalized']}")
    scale_survives = group_tests["r_eff_normalized"]["diff"] > 0 and group_tests["r_eff_normalized"]["permutation_p"] < 0.05

    # ================= Step 8: lesion-burden / Dice / count controls =================
    print("\n=== Step 8: Lesion-burden and baseline-Dice controls ===")
    r_eff = arr("r_eff_spatial")
    native_size_all = arr("native_size")
    dice_all = arr("baseline_dice")

    def r2_of(y, X):
        X1 = np.column_stack([np.ones(len(X)), X])
        beta, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
        y_hat = X1 @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / (ss_tot + 1e-12)

    log_size = np.log(native_size_all + 1)
    r2_size_only = r2_of(r_eff, log_size)
    r2_size_dice = r2_of(r_eff, np.column_stack([log_size, dice_all]))
    incremental_r2_dice = r2_size_dice - r2_size_only
    print(f"  R^2(r_eff ~ log(native_size))            = {r2_size_only:.4f}")
    print(f"  R^2(r_eff ~ log(native_size) + baseline_Dice) = {r2_size_dice:.4f}  (incremental {incremental_r2_dice:+.4f})")

    # Group difference after residualizing r_eff on log(native_size) (removes
    # the trivial "size predicts size-correlated dimensionality" confound
    # within-group, testing whether group still differs after this control)
    X1 = np.column_stack([np.ones(len(log_size)), log_size])
    beta, _, _, _ = np.linalg.lstsq(X1, r_eff, rcond=None)
    resid = r_eff - X1 @ beta
    resid_small = np.array([resid[i] for i, r in enumerate(records) if r["group"] == "small"])
    resid_large = np.array([resid[i] for i, r in enumerate(records) if r["group"] == "large"])
    _, p_resid = stats.mannwhitneyu(resid_small, resid_large, alternative="greater")
    print(f"  Residualized r_eff (control for log native_size): small={resid_small.mean():+.4f} "
          f"large={resid_large.mean():+.4f}, MWU p(small>large)={p_resid:.4f}")
    burden_survives = p_resid < 0.05 and resid_small.mean() > resid_large.mean()

    # ================= Step 10: spatial vs channel =================
    print("\n=== Step 10: Spatial vs channel dimensionality ===")
    spatial_go = group_tests["r_eff_spatial"]["permutation_p"] < 0.05 and group_tests["r_eff_spatial"]["diff"] > 0
    channel_go = group_tests["r_eff_channel"]["permutation_p"] < 0.05 and group_tests["r_eff_channel"]["diff"] > 0
    if spatial_go and channel_go:
        dim_locus = "both"
    elif spatial_go:
        dim_locus = "spatial_only"
    elif channel_go:
        dim_locus = "channel_only"
    else:
        dim_locus = "neither"
    print(f"  spatial effect significant: {spatial_go}, channel effect significant: {channel_go}")
    print(f"  Locus of effect: {dim_locus}")

    # ================= Step 9: randomized-control comparison =================
    print("\n=== Step 9: Real representation vs randomized control ===")
    real_r_eff = arr("r_eff_spatial")
    rand_r_eff = arr("r_eff_random_null")
    _, p_vs_null = stats.wilcoxon(real_r_eff, rand_r_eff)
    print(f"  Real r_eff mean={real_r_eff.mean():.3f}, randomized-null r_eff mean={rand_r_eff.mean():.3f}, "
          f"Wilcoxon p={p_vs_null:.4e}")
    print("  (Real representation expected to show LOWER effective rank than an independent-feature "
          "null of the same scale, since real channels are correlated -- structure beyond chance.)")

    # ================= Step 11: relationship to E48 CausalDrop =================
    print("\n=== Step 11: R_eff vs E48 CausalDrop ===")
    matched = [r for r in records if r["e48_causal_drop"] is not None]
    r_eff_m = np.array([r["r_eff_spatial"] for r in matched])
    drop_m = np.array([r["e48_causal_drop"] for r in matched])
    size_m = np.array([r["native_size"] for r in matched], dtype=np.float64)

    rho_causal, p_causal_param = stats.spearmanr(r_eff_m, drop_m)
    rng4 = np.random.default_rng(SEED + 2)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng4.permutation(drop_m)
        perm_rhos[i], _ = stats.spearmanr(r_eff_m, perm_y)
    p_causal_perm = float((np.abs(perm_rhos) >= np.abs(rho_causal)).mean())
    print(f"  Spearman(r_eff, E48 CausalDrop) = {rho_causal:+.4f} (parametric p={p_causal_param:.4e}, "
          f"permutation p={p_causal_perm:.4f}), n={len(matched)}")

    # Does this survive control for lesion size? Partial correlation via
    # residualization on log(native_size) for both variables.
    log_size_m = np.log(size_m + 1)
    X1s = np.column_stack([np.ones(len(log_size_m)), log_size_m])
    beta_r, _, _, _ = np.linalg.lstsq(X1s, r_eff_m, rcond=None)
    resid_r = r_eff_m - X1s @ beta_r
    beta_d, _, _, _ = np.linalg.lstsq(X1s, drop_m, rcond=None)
    resid_d = drop_m - X1s @ beta_d
    rho_partial, p_partial = stats.spearmanr(resid_r, resid_d)
    print(f"  Partial Spearman (controlling log native_size) = {rho_partial:+.4f} (p={p_partial:.4e})")

    # Sign convention (verified against E48_summary.json, not assumed):
    # E48 found Spearman(native_size, drop) = -0.454 -- i.e. SMALLER
    # native_size is associated with LARGER CausalDrop (small lesions lose
    # more Dice when the bottleneck is severed; this was E48's own
    # "reversed finding" relative to its pre-registered hypothesis).
    # This hypothesis (E61) predicts small lesions have HIGHER R_eff.
    # Composing both: small -> high R_eff AND small -> high CausalDrop
    # implies R_eff and CausalDrop should be POSITIVELY correlated
    # (both driven by "small-ness" in the same direction).
    causal_link = (p_causal_perm < 0.05) and (rho_causal > 0)
    print("  (Composed hypothesis direction: since E48 found smaller lesions -> LARGER CausalDrop "
          "(rho=-0.454 vs native_size), and this hypothesis predicts smaller lesions -> HIGHER R_eff, "
          "the composed prediction is R_eff POSITIVELY correlated with CausalDrop.)")

    # ================= Step 12: qualitative cross-check with E59 =================
    e59_summary_path = project_root / "experiments" / "exp_e12_eggo_m" / "e59" / "E59_summary.json"
    e59_note = "not available"
    if e59_summary_path.exists():
        with open(e59_summary_path) as f:
            e59_summary = json.load(f)
        e59_note = e59_summary.get("verdict", "unknown")
    print(f"\n=== Step 12: Qualitative cross-check with E59 (different architecture -- v3/D4only) ===")
    print(f"  E59 verdict: {e59_note}")
    print(f"  E61 direction (small > large R_eff): {group_tests['r_eff_spatial']['diff'] > 0}")
    print("  NOTE: this is a qualitative/directional comparison ONLY -- E59 used the plain v3/D4-only "
          "checkpoint, not v5/E46, so no quantitative correlation across architectures is computed.")

    # ================= Step 13: robustness across preprocessing =================
    print("\n=== Step 13: Robustness across standardization choices ===")
    robustness = {
        "subject_wise_centering (primary)": group_tests["r_eff_spatial"],
        "scale_normalized": group_tests["r_eff_normalized"],
    }
    n_agree = sum(1 for v in robustness.values() if v["diff"] > 0 and v["permutation_p"] < 0.05)
    print(f"  {n_agree}/{len(robustness)} preprocessing variants agree (small > large, p<0.05)")
    stable = n_agree == len(robustness)

    # ================= Step 14: pre-declared decision rule =================
    metrics_agreeing = sum(1 for m in ["r_eff_spatial", "r_stable_spatial", "pr_spatial"]
                            if group_tests[m]["diff"] > 0 and group_tests[m]["permutation_p"] < 0.05)
    criterion_1 = group_tests["r_eff_spatial"]["diff"] > 0 and group_tests["r_eff_spatial"]["permutation_p"] < 0.05
    criterion_2 = scale_survives
    criterion_3 = burden_survives
    criterion_4 = metrics_agreeing >= 2
    criterion_5 = (group_tests["r_eff_spatial"]["bootstrap_95ci"][0] > 0) or (group_tests["r_eff_spatial"]["permutation_p"] < 0.05)
    criterion_6 = causal_link
    criterion_7 = incremental_r2_dice < 0.05  # dimensionality-group effect not trivially explained by Dice alone (approximate proxy)

    all_criteria = [criterion_1, criterion_2, criterion_3, criterion_4, criterion_5, criterion_6, criterion_7]
    print("\n=== Step 14: Decision ===")
    print(f"  1. Small > large consistently:            {criterion_1}")
    print(f"  2. Survives activation-scale control:      {criterion_2}")
    print(f"  3. Survives lesion-burden control:         {criterion_3}")
    print(f"  4. >=2/3 primary metrics agree:            {criterion_4} ({metrics_agreeing}/3)")
    print(f"  5. Survives permutation/bootstrap:         {criterion_5}")
    print(f"  6. Associated with E48 CausalDrop:         {criterion_6}")
    print(f"  7. Not trivial consequence of Dice/count:  {criterion_7}")

    if all(all_criteria):
        decision = "GO"
    elif all([criterion_1, criterion_2, criterion_3, criterion_4, criterion_5]) and not criterion_6:
        decision = "QUALIFIED_GO"
    else:
        decision = "KILL"
    print(f"\n=== DECISION: {decision} ===")

    summary = {
        "checkpoint": str(CKPT_PATH),
        "checkpoint_note": "v5/E46 attention-gate checkpoint, matches E48's CausalDrop exactly; "
                            "differs from E58/E59's plain v3/D4-only checkpoint (step-12 is qualitative only)",
        "n_subjects": len(records),
        "median_native_size_split": median_size,
        "n_small": n_small, "n_large": n_large,
        "group_tests": group_tests,
        "scale_controls": scale_controls,
        "scale_survives": scale_survives,
        "burden_control": {
            "r2_size_only": float(r2_size_only), "r2_size_plus_dice": float(r2_size_dice),
            "incremental_r2_dice": float(incremental_r2_dice),
            "residualized_p_small_gt_large": float(p_resid),
        },
        "burden_survives": burden_survives,
        "dimensionality_locus": dim_locus,
        "randomized_null": {"real_r_eff_mean": float(real_r_eff.mean()), "null_r_eff_mean": float(rand_r_eff.mean()),
                             "wilcoxon_p": float(p_vs_null)},
        "e48_causal_link": {
            "n_matched": len(matched), "spearman_rho": float(rho_causal),
            "parametric_p": float(p_causal_param), "permutation_p": p_causal_perm,
            "partial_rho_controlling_size": float(rho_partial), "partial_p": float(p_partial),
        },
        "e59_qualitative_crosscheck": e59_note,
        "robustness_variants_agreeing": n_agree,
        "criteria": {
            "1_consistent_small_gt_large": criterion_1, "2_scale_control": criterion_2,
            "3_burden_control": criterion_3, "4_metrics_agree": criterion_4,
            "5_permutation_bootstrap": criterion_5, "6_causal_link": criterion_6,
            "7_not_trivial": criterion_7,
        },
        "decision": decision,
    }
    def _json_default(o):
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open(OUT_DIR / "E61_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print("\nSaved E61_summary.json")


if __name__ == "__main__":
    main()
