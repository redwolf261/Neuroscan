"""
Phase E127: Label-Blind Discarded-Residual Structure of the MaxPool
Transition -- diagnostic/falsification only, NO architecture change.

CONTEXT AND WHY THE ORIGINAL FRAMING WAS REJECTED:
The initial proposal was a "complementarity audit": decompose Z_16 into
Z_parallel + Z_perp where Z_perp = "information poorly represented after
the transition but still predictive of the target", then test whether
Z_perp is causally relevant. That framing was CIRCULAR: defining the
subspace by target-predictiveness guarantees that it predicts the target.
Rejected before any code was written.

E127 instead uses a strictly LABEL-BLIND, operator-derived construction.
The lesion mask, N_b, and Dice never enter the definition of the
quantities being tested -- only the frozen checkpoint's own activations
and MaxPool3d's own selection behaviour.

PER-CHANNEL CORRECTION (user's correction, accepted):
MaxPool3d selects independently PER CHANNEL. Within one 2x2x2 window,
channel c may select sub-voxel j_c while channel c' selects a different
j_{c'}. So there is no single "the argmax-selected sub-voxel" for a
window. The construction is therefore per (channel, location):

    j*_c      = argmax_{j in W} Z_16(j, c)          # per-channel winner
    Z_par(c)  = Z_16(j*_c, c)                       # what survives pool3
    R(j, c)   = Z_16(j, c) - Z_16(j*_c, c)   for j != j*_c

R is <= 0 by construction (the winner is the max). 7 x C residual values
per window.

THREE PRE-REGISTERED LABEL-BLIND SUMMARIES of R (chosen and fixed BEFORE
running; each is computed per window, then averaged over the 512 windows
to give one scalar per subject):

  S1 "magnitude"          = mean |R(j,c)| over all discarded (j,c).
      Raw amount of activation thrown away. MOST likely to be redundant
      with E126's effective rank -- included deliberately as the
      redundancy canary.

  S2 "spatial concentration" = per window, per channel, the normalized
      entropy of the discarded mass across the 7 discarded locations,
      averaged over channels (then over windows). Low entropy = one
      near-tie runner-up dominates; high entropy = all 7 uniformly far
      below the winner. Distinguishes the SHAPE of what was discarded,
      not just its size.
      Normalization: entropy / log(7), so S2 in [0, 1].

  S3 "channel-agreement"  = per window, the fraction of channels whose
      argmax lands on the MODAL sub-voxel (the location most channels
      agree on). Purely a property of WHICH voxel wins, per channel.
      This is the summary LEAST likely to duplicate E126, because
      participation ratio is computed from the 8 sub-voxel vectors'
      Gram spectrum and is INVARIANT to argmax identity -- PR cannot
      see which voxel won.

RELATION TO E119 (stated explicitly, not glossed over): E119 measured
argmax CORRECTNESS against the lesion mask (label-DEPENDENT) and was
killed -- selection mismatch did not predict N_b. S3 here measures argmax
CONSENSUS ACROSS CHANNELS (label-BLIND). Same operator family, different
quantity, different question. This closeness is disclosed rather than
hidden; if S3 survives, any later interpretation must address it.

THE DECISIVE TEST (pre-registered):
    Model A: N_b ~ size + dice_error + effective_rank          (E126 baseline)
    Model B: N_b ~ size + dice_error + effective_rank + R_perp (adds E127)

  SURVIVAL REQUIRES BOTH:
    (i)  Delta R^2 (B over A) > 0.01     -- the SAME magnitude gate that
         correctly killed E122/E123 (significant-but-trivial effects at
         large n), and
    (ii) permutation p < 0.05 on the added coefficient.

  Marginal correlation of R_perp with N_b is reported but is NOT
  sufficient for survival -- because R_perp is computed from the same 8
  values as E126's effective rank, only INCREMENTAL information counts.

  If Delta R^2 <= 0.01 or p >= 0.05 -> E127 = KILL, REDUNDANT WITH E126.

Each summary is tested individually (S1, S2, S3 each added alone) and
jointly (all three added together), so the redundancy verdict is
interpretable rather than hinging on one pre-run guess.

Stage 3 (enc3 16^3 -> pool3 8^3) only -- the transition where E124/E125/
E126 established the effective-rank result. Canonical checkpoint,
125 validation subjects, same conventions as E118-E126.

NO ARCHITECTURE MODIFICATION. NO TRAINING. NO NOVELTY CLAIM.
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
DELTA_R2_BAR = 0.01

E124_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e124" / "E124_per_subject_table.json"
CKPT_PATH = (project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs"
             / "AttnGate_seed0" / "checkpoints" / "best.pth")
EXPECTED_VAL_DICE = 0.9101624600589275

EPS = 1e-12


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    if denom == 0:
        return 1.0
    return float(2 * tp / denom)


def fractional_occupancy(seg_binary_native, shape):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=shape, mode="area").squeeze().numpy()


def forward_with_bottleneck_ablation(model, image, ablate, device):
    """Identical construction to E109/E118/E119/E121/E125/E126."""
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

    return probs.squeeze(0).squeeze(0).cpu().numpy(), enc3.squeeze(0)


def windows_from_tensor(t):
    """(C,R,R,R) -> (C, nW, 8), IDENTICAL partition/order to E119/E124."""
    C = t.shape[0]
    x = t.unfold(1, 2, 2).unfold(2, 2, 2).unfold(3, 2, 2)
    nW = (t.shape[1] // 2) * (t.shape[2] // 2) * (t.shape[3] // 2)
    return x.contiguous().view(C, nW, 8)


def residual_summaries(windows):
    """windows: (C, nW, 8) float tensor of enc3 activations.

    Returns (S1, S2, S3) scalars for this subject. ALL LABEL-BLIND --
    computed only from the activations and MaxPool3d's own per-channel
    argmax behaviour.
    """
    C, nW, _ = windows.shape

    # Per-channel, per-window argmax (what pool3 actually selects).
    argmax_idx = windows.argmax(dim=2)                       # (C, nW)
    winner_val = windows.max(dim=2).values                   # (C, nW)

    # R(j,c) = Z(j,c) - Z(j*_c, c); <= 0 by construction.
    R = windows - winner_val.unsqueeze(2)                    # (C, nW, 8)
    absR = R.abs()

    # Mask out the winner position itself (residual there is exactly 0
    # and is not a "discarded" location). scatter a False at argmax.
    keep = torch.ones_like(absR, dtype=torch.bool)
    keep.scatter_(2, argmax_idx.unsqueeze(2), False)
    discarded = absR[keep].view(C, nW, 7)                    # (C, nW, 7)

    # ---- S1: mean magnitude of discarded activation ----
    S1 = float(discarded.mean().item())

    # ---- S2: normalized entropy of discarded mass across the 7 slots ----
    # Per (c, window): distribute |R| over the 7 discarded locations,
    # normalize to a probability vector, take entropy / log(7).
    total = discarded.sum(dim=2, keepdim=True)               # (C, nW, 1)
    p = discarded / (total + EPS)
    ent = -(p * torch.log(p + EPS)).sum(dim=2)               # (C, nW)
    ent_norm = ent / float(np.log(7.0))
    # Windows where everything was identical (total ~ 0) are degenerate;
    # exclude them rather than let 0/0 drive the mean.
    valid = (total.squeeze(2) > 1e-8)
    if valid.any():
        S2 = float(ent_norm[valid].mean().item())
    else:
        S2 = float("nan")

    # ---- S3: channel-agreement on the modal argmax location ----
    # Per window: histogram the 8 possible winning locations across the C
    # channels, take the modal count / C. Invariant to activation scale;
    # depends ONLY on which voxel won per channel -- a property E126's
    # participation ratio is structurally blind to.
    onehot = F.one_hot(argmax_idx, num_classes=8).float()    # (C, nW, 8)
    counts = onehot.sum(dim=0)                               # (nW, 8)
    modal_share = counts.max(dim=1).values / float(C)        # (nW,)
    S3 = float(modal_share.mean().item())

    return S1, S2, S3


def unit_test_residual_summaries():
    """Verify the three summaries behave as intended on constructed inputs
    BEFORE trusting them on real activations."""
    # Case A: all 8 sub-voxels identical -> nothing discarded (R == 0),
    # and every channel's argmax ties at index 0 -> perfect agreement.
    w = torch.ones(4, 1, 8)
    S1, S2, S3 = residual_summaries(w)
    assert abs(S1) < 1e-6, f"identical window should discard nothing, S1={S1}"
    assert abs(S3 - 1.0) < 1e-6, f"identical window -> all channels tie at 0, S3={S3}"

    # Case B: every channel has a distinct clear winner at a DIFFERENT
    # location -> channel-agreement should be low (1/C for C<=8).
    C = 8
    w2 = torch.zeros(C, 1, 8)
    for c in range(C):
        w2[c, 0, c] = 10.0
    S1b, S2b, S3b = residual_summaries(w2)
    assert abs(S3b - 1.0 / C) < 1e-6, f"disjoint winners -> S3 should be 1/C={1/C}, got {S3b}"
    assert S1b > 0, "disjoint winners should discard nonzero magnitude"

    # Case C: all channels agree on the SAME winner -> S3 == 1.
    w3 = torch.zeros(C, 1, 8)
    w3[:, 0, 3] = 5.0
    _, _, S3c = residual_summaries(w3)
    assert abs(S3c - 1.0) < 1e-6, f"unanimous winner -> S3=1, got {S3c}"

    # Case D: S2 entropy bounds. One dominant near-tie runner-up (rest far
    # below) should give LOWER normalized entropy than 7 equal residuals.
    w4 = torch.zeros(1, 1, 8)
    w4[0, 0, 0] = 10.0   # winner
    w4[0, 0, 1] = 9.9    # near tie
    w4[0, 0, 2:] = -50.0  # far below
    _, S2_peaked, _ = residual_summaries(w4)
    w5 = torch.zeros(1, 1, 8)
    w5[0, 0, 0] = 10.0
    w5[0, 0, 1:] = 0.0   # all 7 equally below
    _, S2_flat, _ = residual_summaries(w5)
    assert S2_peaked < S2_flat, f"peaked residual should have lower entropy: {S2_peaked} vs {S2_flat}"
    assert 0.0 <= S2_flat <= 1.0 + 1e-6, f"S2 must be normalized to [0,1], got {S2_flat}"

    print(f"[Unit test] residual_summaries PASS "
          f"(identical->S1=0,S3=1; disjoint->S3=1/C; unanimous->S3=1; "
          f"peaked S2={S2_peaked:.3f} < flat S2={S2_flat:.3f}).")


def ols_r2(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = ((y - y.mean()) ** 2).sum()
    return (1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else 0.0), beta


def incremental_test(base_cols, add_cols, y, seed, label):
    """Model A = base, Model B = base + add. Returns delta R^2 and a
    permutation p-value on the added block (permuting the added columns'
    rows breaks their link to y while preserving the base fit)."""
    n = len(y)
    XA = np.column_stack([np.ones(n)] + base_cols)
    XB = np.column_stack([np.ones(n)] + base_cols + add_cols)
    r2A, _ = ols_r2(XA, y)
    r2B, betaB = ols_r2(XB, y)
    delta = r2B - r2A

    rng = np.random.default_rng(seed)
    perm_deltas = np.empty(N_PERM)
    add_mat = np.column_stack(add_cols)
    for i in range(N_PERM):
        perm_idx = rng.permutation(n)
        add_perm = [add_mat[perm_idx, j] for j in range(add_mat.shape[1])]
        XBp = np.column_stack([np.ones(n)] + base_cols + add_perm)
        r2Bp, _ = ols_r2(XBp, y)
        perm_deltas[i] = r2Bp - r2A
    p_perm = float((perm_deltas >= delta).mean())

    added_betas = betaB[-len(add_cols):]
    print(f"  {label:24s}: Delta R^2 = {delta:+.4f}, perm p = {p_perm:.4f}, "
          f"coef(s) = {np.round(added_betas, 4)}")
    return {"delta_r2": float(delta), "perm_p": p_perm,
            "coefficients": [float(b) for b in added_betas],
            "r2_base": float(r2A), "r2_full": float(r2B)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    unit_test_residual_summaries()

    with open(E124_TABLE_PATH) as f:
        e124 = json.load(f)
    e124_by_id = {r["subject_id"]: r for r in e124}
    print(f"Loaded E124 table ({len(e124)} subjects) for the effective-rank control.")

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
        manual_probs, _ = forward_with_bottleneck_ablation(model, img0_b, False, device)
        real_np = (real_probs.squeeze(0).squeeze(0).cpu().numpy()
                   if torch.is_tensor(real_probs) else np.asarray(real_probs))
        max_diff = float(np.abs(real_np - manual_probs).max())
    assert max_diff < 1e-4, "Manual trunk mismatch -- STOP."
    print(f"[Sanity check] manual trunk vs real forward(): max abs diff = {max_diff:.6e} PASS.\n")

    records = []
    print(f"Extracting label-blind residual summaries for {len(val_ds)} subjects...", flush=True)
    for idx in range(len(val_ds)):
        image, _, sid = val_ds[idx]
        if sid not in e124_by_id:
            continue
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_ds.subject_dirs[idx]
        seg_path = Path(subject_dir) / f"{sid}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_size = int(seg_binary_native.sum())
        target_bin_64 = (fractional_occupancy(seg_binary_native, (64, 64, 64)) > 0.5).astype(np.float32)

        probs_intact, enc3 = forward_with_bottleneck_ablation(model, image_b, False, device)
        probs_ablated, _ = forward_with_bottleneck_ablation(model, image_b, True, device)
        d_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin_64)
        d_ablated = dice_score((probs_ablated >= 0.5).astype(np.float32), target_bin_64)
        n_b = d_intact - d_ablated

        windows = windows_from_tensor(enc3)
        S1, S2, S3 = residual_summaries(windows)

        records.append({
            "subject_id": sid, "native_size": native_size,
            "dice_intact": d_intact, "N_b": n_b,
            "S1_magnitude": S1, "S2_spatial_concentration": S2, "S3_channel_agreement": S3,
            "E126_effective_rank": e124_by_id[sid]["E_N_3"],
        })
        if len(records) % 25 == 0:
            print(f"  {len(records)}/{len(val_ds)}", flush=True)

    with open(OUT_DIR / "E127_per_subject_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nTotal subjects: {len(records)}")

    N_b = np.array([r["N_b"] for r in records])
    size = np.log(np.array([r["native_size"] for r in records]) + 1)
    dice_err = 1.0 - np.array([r["dice_intact"] for r in records])
    eff_rank = np.array([r["E126_effective_rank"] for r in records])
    S1 = np.array([r["S1_magnitude"] for r in records])
    S2 = np.array([r["S2_spatial_concentration"] for r in records])
    S3 = np.array([r["S3_channel_agreement"] for r in records])

    print(f"\nN_b: mean={N_b.mean():.4f} std={N_b.std():.4f} "
          f"(cross-check vs established ~0.27 range)")
    for nm, v in [("S1 magnitude", S1), ("S2 spatial-concentration", S2),
                  ("S3 channel-agreement", S3), ("E126 effective rank", eff_rank)]:
        print(f"  {nm:26s}: mean={v.mean():.4f} std={v.std():.4f}")

    # --- Marginal correlations (reported for context; NOT the survival test) ---
    print("\n=== Marginal correlations with N_b (context only, NOT sufficient for survival) ===")
    marginals = {}
    for nm, v in [("S1", S1), ("S2", S2), ("S3", S3), ("effective_rank", eff_rank)]:
        rho, p_par = stats.spearmanr(v, N_b)
        print(f"  Spearman({nm}, N_b) = {rho:+.4f} (p={p_par:.4e})")
        marginals[nm] = {"rho": float(rho), "p": float(p_par)}

    # --- Redundancy with E126 (how collinear are these, really?) ---
    print("\n=== Collinearity of each summary with E126's effective rank ===")
    collin = {}
    for nm, v in [("S1", S1), ("S2", S2), ("S3", S3)]:
        rho, p_par = stats.spearmanr(v, eff_rank)
        print(f"  Spearman({nm}, effective_rank) = {rho:+.4f} (p={p_par:.4e})")
        collin[nm] = {"rho": float(rho), "p": float(p_par)}

    # --- THE DECISIVE PRE-REGISTERED TEST ---
    print(f"\n=== DECISIVE TEST: incremental over Model A "
          f"(N_b ~ size + dice_error + effective_rank) ===")
    base = [size, dice_err, eff_rank]
    r2A, _ = ols_r2(np.column_stack([np.ones(len(N_b))] + base), N_b)
    print(f"  Model A baseline R^2 = {r2A:.4f}")
    print(f"  Survival bar: Delta R^2 > {DELTA_R2_BAR} AND perm p < 0.05\n")

    tests = {}
    tests["S1_alone"] = incremental_test(base, [S1], N_b, SEED + 1, "S1 magnitude alone")
    tests["S2_alone"] = incremental_test(base, [S2], N_b, SEED + 2, "S2 spatial-conc alone")
    tests["S3_alone"] = incremental_test(base, [S3], N_b, SEED + 3, "S3 channel-agree alone")
    tests["all_three"] = incremental_test(base, [S1, S2, S3], N_b, SEED + 4, "all three jointly")

    survivors = [k for k, v in tests.items()
                 if v["delta_r2"] > DELTA_R2_BAR and v["perm_p"] < 0.05]

    print("\n=== DECISION ===")
    if not survivors:
        decision = "KILL_REDUNDANT_WITH_E126"
        detail = (f"No summary adds Delta R^2 > {DELTA_R2_BAR} with perm p < 0.05 over the "
                  f"E126 baseline. The label-blind discarded-residual structure carries no "
                  f"incremental information about N_b beyond effective rank, lesion size and "
                  f"Dice error. E127 = KILL, redundant with E126.")
    else:
        decision = "SURVIVES_INCREMENTAL_BAR"
        detail = (f"Survives for: {survivors}. These summaries add predictive information "
                  f"about N_b BEYOND E126's effective rank plus size and Dice error. This is "
                  f"evidence for a transition property distinct from effective rank. It is "
                  f"NOT yet 'complementary information' -- at this stage it is only "
                  f"label-blind discarded-residual structure. Next step is a prior-art audit, "
                  f"NOT architecture implementation.")
    print(f"{decision}\n{detail}")

    summary = {
        "checkpoint": str(CKPT_PATH), "val_dice_check": val_dice,
        "n_subjects": len(records),
        "delta_r2_bar": DELTA_R2_BAR,
        "model_A_r2_size_dice_effrank": float(r2A),
        "marginal_correlations_with_N_b": marginals,
        "collinearity_with_E126_effective_rank": collin,
        "incremental_tests": tests,
        "survivors": survivors,
        "decision": decision, "detail": detail,
    }
    with open(OUT_DIR / "E127_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E127_summary.json, E127_per_subject_table.json")


if __name__ == "__main__":
    main()
