"""
Phase E120: Spatial Context Decodability Audit (deliberately NOT named
"receptive field sufficiency" yet -- that interpretation is earned only if
the pre-registered gates below pass; see E120-A/B/C/D decision tree).

CONTEXT: E119 killed the "competitive argmax selection" mechanism (real
oracle gap, mean 48%, but doesn't predict N_b). This phase asks a
different, complementary question, per the user's corrected framing:
is lesion information that distinguishes high-N_b subjects present in
spatial context AROUND a pool3 window, but weak/absent INSIDE the window
itself (the same 2x2x2 region E93/E116/E117/E119 all separately showed
cannot be rescued by any function of its own 8 values)?

FIXED OBJECTS (re-extracted fresh from the canonical checkpoint -- E92/E118
did not persist raw tensors to disk, only summary JSONs, so this follows
E109's own established convention of re-extracting rather than trusting
stale/absent artifacts):
  enc3          : (128, 16,16,16) per subject
  pool3_output  : (128, 8,8,8)    per subject (= MaxPool3d(2,2)(enc3))
  lesion_mask_8 : (8,8,8) fractional occupancy, downsampled from native seg
  N_b(s)        : E48/E109/E118/E119-style bottleneck-ablation Dice drop
  native_size(s): lesion volume in native voxels

CONTEXT SHELLS: for each of the 8^3=512 pool3 cells w, its native pool
window in enc3 coordinates is a fixed 2x2x2 block. Define concentric CUBES
in enc3 (16^3) coordinates, all centered on that same 2x2x2 block's center:
  C_0 = 2^3 (the pool window itself, exactly what E119 tested)
  C_1 = 4^3
  C_2 = 6^3
  C_3 = 8^3
enc3 is REFLECTION-PADDED by radius 3 before shell extraction so every one
of the 512 cells (including corners/edges) has a well-defined C_3 without
dropping edge cells or biasing toward interior lesions.

PROBE DESIGN (avoiding E118's overfitting bug by construction, not by
regularization after the fact): a 1x1x1-style linear probe with WEIGHT
SHARING ACROSS ALL 512 WINDOWS per subject -- i.e. a real Conv3d(C_in, 1,
kernel_size=k) applied convolutionally over the padded enc3 volume, trained
by gradient descent (E90-E92's exact optimizer/epoch/lr convention), NOT
per-subject closed-form least squares. This has a fixed, small parameter
count (channels_in * k^3 + 1) regardless of context size, so it cannot
memorize a subject-count-sized training set the way E118's underdetermined
lstsq did.

THREE TARGETS (pre-registered, all reported):
  A. local lesion occupancy m_w in [0,1] (regression, MSE probe + Pearson r
     as decodability, since this is continuous not binary)
  B. lesion presence 1[m_w>0] (classification, BCE probe + decodability
     Dice, E90-E92's exact convention)
  C. FRACTIONAL occupancy subset 0<m_w<1 only (same probe as B, evaluated
     restricted to fractional cells -- the regime that motivated this
     whole branch, per E119)

Decodability D(C_k) reported per target, per context size, on held-out
subjects (same size-sorted every-4th-held-out split as E90-E92/E119 --
subject-level, not voxel-level, per explicit instruction to avoid
E118-style leakage).

KEY SUBJECT-LEVEL QUANTITY: Delta_D(s) = D(C_3, s) - D(C_0, s), using
target B (presence, Dice) as the primary decodability metric (continuous
and interpretable per-subject, unlike a single pooled correlation).

HYPOTHESIS: Delta_D(s) predicts N_b(s), surviving a lesion-size control
(three nested models: N_b~Delta_D, N_b~size, N_b~Delta_D+size, per
user's Model 1/2/3 spec).

SECONDARY: context-saturation radius k*(s) = min{k : D(C_k,s) >=
D(C_3,s) - EPSILON}, EPSILON=0.02 Dice (pre-registered, small relative to
E91/E92's observed 0.02-0.11 decodability-delta range). Test rho(k*, N_b).

PRE-REGISTERED DECISION GATES (verbatim from user's spec):
  E120-A: Delta_D ~ 0 (mean held-out Delta_D < 0.02 Dice) -> KILL, wider
          context does not help decodability at all.
  E120-B: Delta_D > 0 but rho(Delta_D, N_b) not significant (perm p>=0.05)
          OR disappears in Model 3 (Delta_D coefficient sign flips or
          p>=0.05 after adding lesion size) -> KILL.
  E120-C: rho(Delta_D, N_b) > 0, perm p<0.05, AND survives Model 3 ->
          GREEN diagnostic result -- context gain predicts causal necessity
          beyond lesion size alone.
  E120-D: E120-C holds AND rho(k*, N_b) > 0 also survives a size control
          -> strongest result, justifies a MINIMAL pre-pool receptive-field
          intervention as the next experiment (not yet run here).

NO ARCHITECTURE MODIFICATION, NO TRAINING OF THE SEGMENTATION MODEL, NO
NOVELTY CLAIM IN THIS SCRIPT. Read-only diagnostic (only the small probes
are trained, exactly as in E90-E92/E119) on the frozen canonical
checkpoint.
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
N_PROBE_EPOCHS = 200
PROBE_LR = 1e-2
CONTEXT_SIZES = {"C0": 2, "C1": 4, "C2": 6, "C3": 8}  # cube edge length in enc3 (16^3) coords
PAD = 3  # reflection-pad radius, enough for C3=8 (needs +-3 around a 2-wide center on any edge cell)
K_STAR_EPSILON = 0.02

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


def forward_with_bottleneck_ablation(model, image, ablate, device):
    """Identical construction to E109/E118/E119."""
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

    return (probs.squeeze(0).squeeze(0).cpu().numpy(), enc3.squeeze(0), bottleneck_intact.squeeze(0))


class ContextProbe(nn.Module):
    """Convolutional linear probe with a kernel exactly matching a context
    cube size, applied with stride 2, padding=0, so it slides once per
    pool3 cell with centers aligned to pool3's own window centers by
    construction. The caller must feed an input already reflection-padded
    by extend(k)=(k-2)//2 on each side (verified below: for input size
    16+2*extend(k), stride=2, kernel=k, padding=0, output is EXACTLY 8^3
    for every context size in {2,4,6,8} -- checked numerically, NOT
    assumed, since the naive fixed-padding version was WRONG (gave 11^3,
    caught before running the full 125-subject extraction). Weight-shared
    across all 512 windows and all subjects -- fixed small parameter
    count, cannot memorize per-subject (avoids E118's overfitting bug by
    construction rather than by post-hoc regularization).
    """
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.extend = (kernel_size - 2) // 2
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=kernel_size, stride=2, padding=0)

    def crop(self, enc3_padded_maxextend):
        """enc3_padded_maxextend: (C, 16+2*PAD, 16+2*PAD, 16+2*PAD) padded
        by the GLOBAL max extend (PAD). Crop down to this probe's own
        required extend so centers still align (both paddings are
        symmetric around the same 16^3 core)."""
        e = self.extend
        if e == PAD:
            return enc3_padded_maxextend
        lo = PAD - e
        hi = lo + 16 + 2 * e
        return enc3_padded_maxextend[..., lo:hi, lo:hi, lo:hi]

    def forward(self, enc3_padded_maxextend):
        x = self.crop(enc3_padded_maxextend)
        return self.conv(x)


def unit_test_context_probe_shapes():
    """Verify each context size produces an exact (1,1,8,8,8) output when
    applied to the max-padded 16^3 input via crop+conv, confirming the
    padding/crop math is right BEFORE any real extraction."""
    for name, k in CONTEXT_SIZES.items():
        x = torch.randn(1, 128, 16 + 2 * PAD, 16 + 2 * PAD, 16 + 2 * PAD)
        probe = ContextProbe(128, k)
        out = probe(x)
        assert out.shape[2:] == (8, 8, 8), f"{name} (k={k}): got {out.shape}, expected 8^3"
    print("[Unit test] ContextProbe output shapes: PASS for all context sizes.")


def train_and_eval_context_probe(kernel_size, enc3_padded_train, targets_train,
                                  enc3_padded_eval, targets_eval, target_kind, device):
    """target_kind: 'regression' (MSE, target=occupancy fraction) or
    'classification' (BCE, target=binary presence). Returns per-subject,
    per-window decodability arrays for train diagnostics + eval (held-out).
    """
    torch.manual_seed(SEED)
    probe = ContextProbe(128, kernel_size).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    rng = np.random.default_rng(SEED)

    n_train = len(enc3_padded_train)
    for epoch in range(N_PROBE_EPOCHS):
        perm = rng.permutation(n_train)
        for i in perm:
            x = enc3_padded_train[i].unsqueeze(0).to(device)
            y = targets_train[i].unsqueeze(0).unsqueeze(0).to(device)
            logits = probe(x)
            if target_kind == "regression":
                pred = torch.sigmoid(logits)
                loss = F.mse_loss(pred, y)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    per_subject_decod = []
    with torch.no_grad():
        for i in range(len(enc3_padded_eval)):
            x = enc3_padded_eval[i].unsqueeze(0).to(device)
            logits = probe(x).squeeze(0).squeeze(0).cpu().numpy()  # (8,8,8)
            y = targets_eval[i].numpy()  # (8,8,8)
            if target_kind == "regression":
                pred = 1 / (1 + np.exp(-logits))
                # decodability = Pearson r between predicted and true occupancy, over all 512 cells
                r = np.corrcoef(pred.flatten(), y.flatten())[0, 1]
                per_subject_decod.append(float(r) if not np.isnan(r) else 0.0)
            else:
                pred_bin = (1 / (1 + np.exp(-logits)) >= 0.5).astype(np.float32)
                y_bin = (y > 0).astype(np.float32)
                per_subject_decod.append(dice_score(pred_bin, y_bin))
    return per_subject_decod, probe


def permutation_test_corr(x, y, seed):
    rho, p_param = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng.permutation(y)
        perm_rhos[i], _ = stats.spearmanr(x, perm_y)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())
    return float(rho), float(p_param), p_perm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    unit_test_context_probe_shapes()

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
        manual_probs, _, _ = forward_with_bottleneck_ablation(model, img0_b, ablate=False, device=device)
        real_np = real_probs.squeeze(0).squeeze(0).cpu().numpy() if torch.is_tensor(real_probs) else np.asarray(real_probs)
        max_diff = float(np.abs(real_np - manual_probs).max())
    assert max_diff < 1e-4, "Manual trunk mismatch -- STOP."
    print(f"[Sanity check] manual trunk vs real forward(): max abs diff = {max_diff:.6e} PASS.\n")

    # ---------------- Extract per-subject data ----------------
    records = []
    print(f"Extracting {len(val_ds)} validation subjects...", flush=True)
    for idx in range(len(val_ds)):
        image, mask, sid = val_ds[idx]
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_ds.subject_dirs[idx]
        seg_path = Path(subject_dir) / f"{sid}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_size = int(seg_binary_native.sum())
        target_bin_64 = (fractional_occupancy(seg_binary_native, (64, 64, 64)) > 0.5).astype(np.float32)
        lesion_frac_8 = fractional_occupancy(seg_binary_native, (8, 8, 8)).astype(np.float32)

        probs_intact, enc3, bottleneck = forward_with_bottleneck_ablation(model, image_b, ablate=False, device=device)
        probs_ablated, _, _ = forward_with_bottleneck_ablation(model, image_b, ablate=True, device=device)
        dice_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin_64)
        dice_ablated = dice_score((probs_ablated >= 0.5).astype(np.float32), target_bin_64)
        n_b = dice_intact - dice_ablated

        enc3_padded = F.pad(enc3.unsqueeze(0), (PAD,) * 6, mode="reflect").squeeze(0).cpu()

        records.append({
            "subject_id": sid, "enc3_padded": enc3_padded,
            "lesion_frac_8": torch.from_numpy(lesion_frac_8),
            "native_size": native_size, "n_b": n_b,
        })
        if (idx + 1) % 25 == 0:
            print(f"  extracted {idx+1}/{len(val_ds)}", flush=True)

    print(f"\nTotal subjects: {len(records)}")
    n_b_arr = np.array([r["n_b"] for r in records])
    print(f"N_b: mean={n_b_arr.mean():.4f} std={n_b_arr.std():.4f} (sanity check vs ~0.27-0.32 range)")

    # ---------------- Subject-level split (identical convention) ----------------
    sorted_idx = sorted(range(len(records)), key=lambda i: records[i]["native_size"])
    test_flags = [(pos % 4 == 0) for pos in range(len(sorted_idx))]
    train_idx = [sorted_idx[pos] for pos, f in enumerate(test_flags) if not f]
    eval_idx = [sorted_idx[pos] for pos, f in enumerate(test_flags) if f]
    print(f"Probe-train: {len(train_idx)}, Probe-eval (held-out): {len(eval_idx)} "
          f"(IDENTICAL split rule to E90-E92/E119)")

    enc3_train = [records[i]["enc3_padded"] for i in train_idx]
    enc3_eval = [records[i]["enc3_padded"] for i in eval_idx]
    occ_train = [records[i]["lesion_frac_8"] for i in train_idx]
    occ_eval = [records[i]["lesion_frac_8"] for i in eval_idx]
    presence_train = [(records[i]["lesion_frac_8"] > 0).float() for i in train_idx]
    presence_eval = [(records[i]["lesion_frac_8"] > 0).float() for i in eval_idx]

    # ---------------- Train probes for each context size / target ----------------
    results = {"A_regression": {}, "B_presence": {}}
    trained_probes_B = {}
    for name, k in CONTEXT_SIZES.items():
        print(f"\n=== Context {name} (cube={k}^3) ===", flush=True)
        decod_A, _ = train_and_eval_context_probe(k, enc3_train, occ_train, enc3_eval, occ_eval,
                                                    "regression", device)
        decod_B, probe_B = train_and_eval_context_probe(k, enc3_train, presence_train, enc3_eval, presence_eval,
                                                          "classification", device)
        results["A_regression"][name] = decod_A
        results["B_presence"][name] = decod_B
        trained_probes_B[name] = probe_B
        print(f"  Target A (occupancy, Pearson r): mean={np.mean(decod_A):.4f}")
        print(f"  Target B (presence, Dice):       mean={np.mean(decod_B):.4f}")

    # ---------------- Target C: fractional-occupancy-only subset, using Target B probes ----------------
    print("\n=== Target C: fractional-occupancy-cell decodability (using Target B probes) ===")
    results["C_fractional_only"] = {}
    for name, k in CONTEXT_SIZES.items():
        probe = trained_probes_B[name]
        probe.eval()
        frac_decods = []
        with torch.no_grad():
            for i in eval_idx:
                r = records[i]
                x = r["enc3_padded"].unsqueeze(0).to(device)
                logits = probe(x).squeeze(0).squeeze(0).cpu().numpy()
                pred_bin = (1 / (1 + np.exp(-logits)) >= 0.5).astype(np.float32)
                occ = r["lesion_frac_8"].numpy()
                frac_mask = (occ > 0) & (occ < 1)
                if frac_mask.sum() == 0:
                    continue
                y_bin_frac = (occ[frac_mask] > 0.5).astype(np.float32)
                pred_bin_frac = pred_bin[frac_mask]
                frac_decods.append(dice_score(pred_bin_frac, y_bin_frac))
        results["C_fractional_only"][name] = frac_decods
        print(f"  {name}: mean={np.mean(frac_decods):.4f} (n_subjects_with_fractional_cells={len(frac_decods)})")

    with open(OUT_DIR / "E120_context_decodability_table.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---------------- Subject-level Delta_D using Target B (primary) ----------------
    print("\n=== Subject-level Delta_D = D(C3) - D(C0), Target B (presence, Dice) ===")
    D_C0 = np.array(results["B_presence"]["C0"])
    D_C3 = np.array(results["B_presence"]["C3"])
    Delta_D = D_C3 - D_C0
    print(f"D(C0): mean={D_C0.mean():.4f}, D(C3): mean={D_C3.mean():.4f}, "
          f"Delta_D: mean={Delta_D.mean():.4f} std={Delta_D.std():.4f}")

    gate_A_kill = Delta_D.mean() < 0.02
    if gate_A_kill:
        print("\n=== E120-A: Delta_D ~ 0 -- KILL. Wider context does not help decodability at all. ===")
        summary = {
            "checkpoint": str(CKPT_PATH), "val_dice_check": val_dice,
            "n_eval_subjects": len(eval_idx),
            "context_decodability_means": {
                k2: {ck: float(np.mean(v)) for ck, v in v2.items()} for k2, v2 in results.items()
            },
            "delta_D_mean": float(Delta_D.mean()), "delta_D_std": float(Delta_D.std()),
            "verdict": "E120_A_KILL_NO_CONTEXT_GAIN",
        }
        with open(OUT_DIR / "E120_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\nSaved E120_summary.json")
        return

    N_b_eval = np.array([records[i]["n_b"] for i in eval_idx])
    native_size_eval = np.array([records[i]["native_size"] for i in eval_idx])

    rho, p_param, p_perm = permutation_test_corr(Delta_D, N_b_eval, SEED)
    print(f"\nSpearman(Delta_D, N_b) = {rho:+.4f} (param p={p_param:.4e}, perm p={p_perm:.4f})")

    # Model 1/2/3, per user's spec: OLS with permutation-based significance
    # on the Delta_D coefficient in the joint model.
    log_size = np.log(native_size_eval + 1)
    X_m1 = np.column_stack([np.ones(len(Delta_D)), Delta_D])
    X_m2 = np.column_stack([np.ones(len(Delta_D)), log_size])
    X_m3 = np.column_stack([np.ones(len(Delta_D)), Delta_D, log_size])

    def fit_and_report(X, y, label, target_col_idx):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        resid = y - pred
        ss_res = (resid ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        print(f"  {label}: R^2={r2:.4f}, coef[{target_col_idx}]={beta[target_col_idx]:+.4f}")
        return beta, r2

    print("\n=== Model 1/2/3 (N_b ~ Delta_D, N_b ~ size, N_b ~ Delta_D + size) ===")
    beta1, r2_1 = fit_and_report(X_m1, N_b_eval, "Model 1 (N_b ~ Delta_D)", 1)
    beta2, r2_2 = fit_and_report(X_m2, N_b_eval, "Model 2 (N_b ~ size)", 1)
    beta3, r2_3 = fit_and_report(X_m3, N_b_eval, "Model 3 (N_b ~ Delta_D + size)", 1)

    # Partial correlation (same convention as E118/E119) as the actual
    # significance test for "survives Model 3".
    X1 = np.column_stack([np.ones(len(log_size)), log_size])
    beta_d, *_ = np.linalg.lstsq(X1, Delta_D, rcond=None)
    resid_d = Delta_D - X1 @ beta_d
    beta_n, *_ = np.linalg.lstsq(X1, N_b_eval, rcond=None)
    resid_n = N_b_eval - X1 @ beta_n
    rho_partial, p_partial_param = stats.spearmanr(resid_d, resid_n)
    rng2 = np.random.default_rng(SEED + 1)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng2.permutation(resid_n)
        perm_rhos[i], _ = stats.spearmanr(resid_d, perm_y)
    p_partial_perm = float((np.abs(perm_rhos) >= np.abs(rho_partial)).mean())
    print(f"\nPartial Spearman (controlling native_size) = {rho_partial:+.4f} "
          f"(param p={p_partial_param:.4e}, perm p={p_partial_perm:.4f})")

    gate_c_pass = (rho > 0) and (p_perm < 0.05) and (rho_partial > 0) and (p_partial_perm < 0.05) \
        and (beta3[1] > 0)

    if not gate_c_pass:
        decision = "E120_B_KILL_NOT_ASSOCIATED_OR_SIZE_EXPLAINED"
        detail = ("Delta_D > 0 (context decodability gain is real) but does not significantly "
                   "predict N_b, or the association is explained by lesion size (does not "
                   "survive Model 3 / partial correlation).")
        print(f"\n=== {decision} ===\n{detail}")
        summary = {
            "checkpoint": str(CKPT_PATH), "val_dice_check": val_dice,
            "n_eval_subjects": len(eval_idx),
            "context_decodability_means": {
                k2: {ck: float(np.mean(v)) for ck, v in v2.items()} for k2, v2 in results.items()
            },
            "delta_D_mean": float(Delta_D.mean()), "delta_D_std": float(Delta_D.std()),
            "spearman_delta_D_vs_Nb": rho, "perm_p_delta_D_vs_Nb": p_perm,
            "partial_rho_controlling_size": float(rho_partial), "partial_perm_p": p_partial_perm,
            "model1_r2": float(r2_1), "model2_r2": float(r2_2), "model3_r2": float(r2_3),
            "model3_deltaD_coef": float(beta3[1]),
            "decision": decision, "detail": detail,
        }
        with open(OUT_DIR / "E120_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\nSaved E120_summary.json")
        return

    # ---------------- E120-C passed: compute k* and test E120-D ----------------
    print("\n=== E120-C PASSED. Computing context-saturation radius k*(s) for E120-D test. ===")
    order = ["C0", "C1", "C2", "C3"]
    order_vals = [CONTEXT_SIZES[o] for o in order]
    D_by_context = {name: np.array(results["B_presence"][name]) for name in order}
    k_star = np.zeros(len(eval_idx))
    for si in range(len(eval_idx)):
        d3 = D_by_context["C3"][si]
        chosen = order_vals[-1]
        for name, kv in zip(order, order_vals):
            if D_by_context[name][si] >= d3 - K_STAR_EPSILON:
                chosen = kv
                break
        k_star[si] = chosen

    rho_kstar, p_param_kstar, p_perm_kstar = permutation_test_corr(k_star, N_b_eval, SEED + 2)
    print(f"Spearman(k*, N_b) = {rho_kstar:+.4f} (param p={p_param_kstar:.4e}, perm p={p_perm_kstar:.4f})")

    beta_k, *_ = np.linalg.lstsq(X1, k_star, rcond=None)
    resid_k = k_star - X1 @ beta_k
    rho_kstar_partial, p_kstar_partial_param = stats.spearmanr(resid_k, resid_n)
    rng3 = np.random.default_rng(SEED + 3)
    perm_rhos_k = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng3.permutation(resid_n)
        perm_rhos_k[i], _ = stats.spearmanr(resid_k, perm_y)
    p_kstar_partial_perm = float((np.abs(perm_rhos_k) >= np.abs(rho_kstar_partial)).mean())
    print(f"Partial Spearman k* (controlling size) = {rho_kstar_partial:+.4f} "
          f"(param p={p_kstar_partial_param:.4e}, perm p={p_kstar_partial_perm:.4f})")

    gate_d_pass = (rho_kstar > 0) and (p_perm_kstar < 0.05) and (rho_kstar_partial > 0) and (p_kstar_partial_perm < 0.05)

    if gate_d_pass:
        decision = "E120_D_STRONGEST_KSTAR_SCALES_WITH_NB"
        detail = ("E120-C holds AND context-saturation radius k* also predicts N_b beyond "
                   "lesion size. Justifies a MINIMAL pre-pool receptive-field intervention as "
                   "the next experiment -- NOT yet run here, no architecture change made.")
    else:
        decision = "E120_C_GREEN_BUT_KSTAR_DOES_NOT_SCALE"
        detail = ("E120-C's core result holds (Delta_D predicts N_b beyond lesion size) but "
                   "the k* saturation-radius result did not additionally survive. Report the "
                   "core E120-C result as the finding; k* is a secondary null.")

    print(f"\n=== {decision} ===\n{detail}")

    summary = {
        "checkpoint": str(CKPT_PATH), "val_dice_check": val_dice,
        "n_eval_subjects": len(eval_idx),
        "context_decodability_means": {
            k2: {ck: float(np.mean(v)) for ck, v in v2.items()} for k2, v2 in results.items()
        },
        "delta_D_mean": float(Delta_D.mean()), "delta_D_std": float(Delta_D.std()),
        "spearman_delta_D_vs_Nb": rho, "perm_p_delta_D_vs_Nb": p_perm,
        "partial_rho_controlling_size": float(rho_partial), "partial_perm_p": p_partial_perm,
        "model1_r2": float(r2_1), "model2_r2": float(r2_2), "model3_r2": float(r2_3),
        "model3_deltaD_coef": float(beta3[1]),
        "k_star_mean": float(k_star.mean()),
        "spearman_kstar_vs_Nb": rho_kstar, "perm_p_kstar_vs_Nb": p_perm_kstar,
        "partial_rho_kstar_controlling_size": float(rho_kstar_partial),
        "partial_perm_p_kstar": p_kstar_partial_perm,
        "decision": decision, "detail": detail,
    }
    with open(OUT_DIR / "E120_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E120_summary.json")


if __name__ == "__main__":
    main()
