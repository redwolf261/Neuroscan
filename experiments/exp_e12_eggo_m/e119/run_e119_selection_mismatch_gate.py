"""
Phase E119: Feature-Space Selection Mismatch Under Fractional Lesion
Occupancy (mechanistic precursor test, explicitly NOT called "operator
non-commutativity" yet, per the user's own naming discipline -- that
language is earned only if this test's full pre-registered sequence
survives: A (miss rate) -> B (lesion-restricted oracle gap) -> C (N_b
association), in that order, with each stage gating the next).

CONTEXT: E118 killed pooled cross-scale-innovation-energy as a mechanism
(experiments/exp_e12_eggo_m/e118/). The user redirected toward asking what
happens AT the pool3 operator itself: for a 2x2x2 window straddling a
lesion boundary (fractional occupancy), does MaxPool3d's per-channel
argmax systematically select a non-lesion sub-voxel because it is
competing against higher-response background locations? This is
deliberately narrower than "non-commutativity" -- it isolates ONE
candidate mechanism (competitive argmax selection under fractional
occupancy) before any operator-algebra framing is invoked.

DEFINITIONS (exactly as refined in chat):
  P = pool3's per-channel argmax sub-cell selection within a 2x2x2 window
      of enc3 (128ch @ 16^3) -- i.e. which of the 8 sub-voxels wins,
      per channel, exactly what MaxPool3d(2,2) computes internally.
  T = ground-truth lesion mask, restricted to that same window (voxel-
      level binary lesion/not-lesion at native resolution, downsampled to
      16^3 by nearest/majority -- see mask_to_16cubed below).
  m_W = fractional lesion occupancy of window W = (#lesion voxels in W)/8.

STAT A -- Selection miss rate, restricted to FRACTIONAL windows (0<m_W<1)
  only (a window that's all-lesion or all-background can't meaningfully
  "miss"). For each (window W, channel c): L_{W,c} = 1 if pool3's argmax
  sub-voxel for channel c lands on a lesion voxel, else 0.
  M_s (per subject) = 1 - mean(L_{W,c}) over all fractional (W,c) pairs.
  Reported (a) stratified by occupancy bin (0,0.125],(0.125,0.25],...,
  (0.875,1) -- NOT collapsed into one number, per spec -- and (b) with a
  SECONDARY restriction to the subject's own top-K most lesion-decodable
  channels (K=16, ranked by |LinearProbe conv weight| trained fresh on
  enc3 for THIS purpose, since no per-channel decodability ranking exists
  in any prior E9x/E11x artifact -- checked before writing this script).

STAT B -- Lesion-restricted oracle gap (the critical mechanistic control).
  Restricted further to windows with >=2 lesion sub-voxels (so the oracle
  argmax is a genuine choice, not a forced single-voxel pick; windows with
  0 or 1 lesion sub-voxels are excluded from B specifically, pre-
  registered before running, per the refined design).
  a_{W,c}  = ordinary MaxPool3d argmax (real network behavior).
  a*_{W,c} = argmax restricted to ONLY the lesion sub-voxels in W (a
             hypothetical "lesion-aware" pooling oracle).
  Oracle gap G_s (per subject) = P(a != a*) over qualifying (W,c) pairs
  -- how often ordinary MaxPool's choice differs from what a lesion-aware
  oracle would have picked. G_s=0 -> MaxPool already always picks the
  lesion-maximal sub-voxel when one exists -- the idea dies immediately.

STAT C -- Subject-level association with N_b (E48/E109-style bottleneck-
  ablation causal necessity). Only subjects with >=5 FRACTIONAL windows
  are included (pre-registered minimum, to avoid single-digit-window
  subjects injecting noise into M_s); excluded-subject count reported
  explicitly. Spearman(M_s, N_b) with permutation test, plus partial
  correlation controlling for native_size (log).

PRE-REGISTERED DECISION SEQUENCE (exact, per chat):
  1. If Stat B's oracle gap is ~0 (mean G_s < 0.02, i.e. MaxPool
     essentially always agrees with the lesion oracle when a real choice
     exists) -> KILL immediately, competitive-selection-under-fractional-
     -occupancy is not happening.
  2. Else if Stat C's M_s-vs-N_b correlation is not significant (perm
     p>=0.05) or the partial correlation controlling native_size does not
     survive -> KILL as unexplained/noise, not a real mechanism.
  3. Else (gap is real AND predicts N_b beyond lesion size) -> genuine
     mechanistic precursor result; ONLY THEN is it appropriate to start
     using operator non-commutativity language and design a further test.

NO TRAINING, NO ARCHITECTURE CHANGE. Read-only diagnostic on the frozen
canonical checkpoint (single forward pass per subject, reusing E92's
enc3 extraction convention). The one small piece of NEW learning in this
script is a per-channel LinearProbe trained on enc3 purely to RANK
channels by lesion-decodability for Stat A's secondary top-K view --
frozen model, standard E90-E92 probe conventions (same optimizer/epochs/
split), not part of the causal claim itself.
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
TOP_K_CHANNELS = 16
MIN_FRACTIONAL_WINDOWS = 5
MIN_LESION_SUBVOX_FOR_ORACLE = 2  # Stat B restriction, pre-registered

CKPT_PATH = (project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs"
             / "AttnGate_seed0" / "checkpoints" / "best.pth")
EXPECTED_VAL_DICE = 0.9101624600589275


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


def mask_to_16cubed(seg_binary_native):
    """Binary lesion mask at 16^3, majority-vote (>0.5 fractional
    occupancy) downsampling from native resolution -- same 'area' pooling
    convention as fractional_occupancy_64 elsewhere in this project, just
    at 16^3 so it aligns 1:1 with enc3's spatial grid (each 16^3 cell IS
    exactly one pool3 2x2x2-window's single input voxel neighborhood)."""
    frac = fractional_occupancy(seg_binary_native, (16, 16, 16))
    return (frac > 0.5).astype(np.float32)


def forward_with_bottleneck_ablation(model, image, ablate, device):
    """Identical construction to E109/E118 (copied from E48's verified
    version). Returns enc3 as well so a single pass covers N_b AND the
    pool3 argmax analysis."""
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


class LinearProbe(nn.Module):
    """Identical design to E90-E92's probe."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=1)

    def forward(self, z):
        logits_native = self.conv(z)
        return F.interpolate(logits_native, size=(64, 64, 64), mode="trilinear", align_corners=False)


def compute_argmax_indices(enc3_window_view):
    """enc3_window_view: (C, nW, 8) -- for each channel and window, the 8
    sub-voxel values in a fixed, consistent flatten order. Returns
    (C, nW) int argmax indices in [0,8)."""
    return enc3_window_view.argmax(dim=-1)


def windows_from_enc3(enc3):
    """enc3: (C,16,16,16) -> (C, 8*8*8, 8) sub-voxel values per 2x2x2
    window, using the EXACT same window partition MaxPool3d(2,2) uses
    (non-overlapping 2x2x2 blocks). Sub-voxel order within the last dim
    is (dz,dy,dx) each in {0,1}, consistent and arbitrary but FIXED so it
    aligns with the mask windowing below."""
    C = enc3.shape[0]
    x = enc3.unfold(1, 2, 2).unfold(2, 2, 2).unfold(3, 2, 2)  # (C,8,8,8,2,2,2)
    x = x.contiguous().view(C, 8 * 8 * 8, 8)
    return x


def windows_from_mask16(mask16):
    """mask16: (16,16,16) numpy -> (512, 8) binary sub-voxel membership per
    window, SAME window partition/order as windows_from_enc3 (verified by
    construction: identical unfold pattern)."""
    t = torch.from_numpy(mask16).unsqueeze(0)  # (1,16,16,16)
    x = t.unfold(1, 2, 2).unfold(2, 2, 2).unfold(3, 2, 2)
    x = x.contiguous().view(8 * 8 * 8, 8)
    return x.numpy()


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

    # ---------------- Extract per-subject data (single forward pass each) ----------------
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
        target_bin_t = torch.from_numpy(target_bin_64).unsqueeze(0).unsqueeze(0).to(device)
        mask16 = mask_to_16cubed(seg_binary_native)

        probs_intact, enc3, bottleneck = forward_with_bottleneck_ablation(model, image_b, ablate=False, device=device)
        probs_ablated, _, _ = forward_with_bottleneck_ablation(model, image_b, ablate=True, device=device)
        dice_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin_64)
        dice_ablated = dice_score((probs_ablated >= 0.5).astype(np.float32), target_bin_64)
        n_b = dice_intact - dice_ablated

        records.append({
            "subject_id": sid, "enc3": enc3.cpu(), "mask16": mask16,
            "native_size": native_size, "n_b": n_b, "target_bin_64": target_bin_64,
        })
        if (idx + 1) % 25 == 0:
            print(f"  extracted {idx+1}/{len(val_ds)}", flush=True)

    print(f"\nTotal subjects: {len(records)}")
    n_b_arr = np.array([r["n_b"] for r in records])
    print(f"N_b: mean={n_b_arr.mean():.4f} std={n_b_arr.std():.4f} (sanity check vs ~0.27-0.32 range from E48/E109/E118)")

    # ---------------- Train channel-ranking probe on enc3 (E90-E92 conventions) ----------------
    sorted_idx = sorted(range(len(records)), key=lambda i: records[i]["native_size"])
    test_flags = [(pos % 4 == 0) for pos in range(len(sorted_idx))]
    probe_train_idx = [sorted_idx[pos] for pos, f in enumerate(test_flags) if not f]

    print(f"\nTraining channel-ranking LinearProbe on enc3 ({len(probe_train_idx)} training subjects, "
          f"E90-E92 conventions)...", flush=True)
    torch.manual_seed(SEED)
    probe = LinearProbe(in_channels=128).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)
    rng = np.random.default_rng(SEED)
    N_PROBE_EPOCHS = 200
    for epoch in range(N_PROBE_EPOCHS):
        perm = rng.permutation(len(probe_train_idx))
        for pi in perm:
            r = records[probe_train_idx[pi]]
            enc3_b = r["enc3"].unsqueeze(0).to(device)
            target_t = torch.from_numpy(r["target_bin_64"]).unsqueeze(0).unsqueeze(0).to(device)
            logits = probe(enc3_b)
            loss = F.binary_cross_entropy_with_logits(logits, target_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    probe.eval()
    channel_importance = probe.conv.weight.detach().abs().squeeze().cpu().numpy()  # (128,)
    top_k_channels = np.argsort(-channel_importance)[:TOP_K_CHANNELS]
    print(f"Top-{TOP_K_CHANNELS} channels by |probe weight|: {sorted(top_k_channels.tolist())}")

    # ---------------- Stat A + B: per-window argmax analysis ----------------
    print("\n=== Stats A & B: per-subject selection-mismatch analysis ===", flush=True)
    occupancy_bins = [(0.0, 0.125), (0.125, 0.25), (0.25, 0.375), (0.375, 0.5),
                       (0.5, 0.625), (0.625, 0.75), (0.75, 0.875), (0.875, 1.0)]
    bin_hits = {b: [0, 0] for b in occupancy_bins}  # [miss_count, total_count]

    subject_stats = []
    for r in records:
        enc3_windows = windows_from_enc3(r["enc3"])           # (128, 512, 8)
        mask_windows = windows_from_mask16(r["mask16"])       # (512, 8)
        m_W = mask_windows.sum(axis=1) / 8.0                  # (512,)
        fractional = (m_W > 0) & (m_W < 1)
        n_fractional_windows = int(fractional.sum())

        argmax_idx = compute_argmax_indices(enc3_windows).cpu().numpy()  # (128, 512)

        # ---- Stat A: all-channel + top-K-channel miss rate, stratified ----
        all_L, topk_L = [], []
        for w in np.where(fractional)[0]:
            lesion_subvox = mask_windows[w].astype(bool)  # (8,)
            m_w_val = m_W[w]
            for c in range(128):
                a = argmax_idx[c, w]
                hit = bool(lesion_subvox[a])
                all_L.append(hit)
                if c in top_k_channels:
                    topk_L.append(hit)
                for lo, hi in occupancy_bins:
                    if lo < m_w_val <= hi or (lo == 0.0 and m_w_val == 0.0 and False):
                        bin_hits[(lo, hi)][0] += (0 if hit else 1)
                        bin_hits[(lo, hi)][1] += 1
                        break

        M_s_all = 1.0 - (np.mean(all_L) if all_L else np.nan)
        M_s_topk = 1.0 - (np.mean(topk_L) if topk_L else np.nan)

        # ---- Stat B: lesion-restricted oracle gap, windows with >=2 lesion subvox ----
        oracle_mismatches, oracle_total = 0, 0
        for w in np.where(fractional)[0]:
            lesion_subvox = mask_windows[w].astype(bool)
            n_lesion_subvox = int(lesion_subvox.sum())
            if n_lesion_subvox < MIN_LESION_SUBVOX_FOR_ORACLE:
                continue
            lesion_positions = np.where(lesion_subvox)[0]
            for c in range(128):
                vals = enc3_windows[c, w].cpu().numpy()
                a_real = int(np.argmax(vals))
                a_oracle = int(lesion_positions[np.argmax(vals[lesion_positions])])
                oracle_mismatches += int(a_real != a_oracle)
                oracle_total += 1
        G_s = (oracle_mismatches / oracle_total) if oracle_total > 0 else np.nan

        subject_stats.append({
            "subject_id": r["subject_id"], "native_size": r["native_size"], "n_b": r["n_b"],
            "n_fractional_windows": n_fractional_windows,
            "M_s_all_channel": M_s_all, "M_s_topk_channel": M_s_topk,
            "oracle_gap_G_s": G_s, "oracle_qualifying_pairs": oracle_total,
        })

    n_with_enough_windows = sum(1 for s in subject_stats if s["n_fractional_windows"] >= MIN_FRACTIONAL_WINDOWS)
    print(f"Subjects with >={MIN_FRACTIONAL_WINDOWS} fractional windows: {n_with_enough_windows}/{len(subject_stats)}")

    print("\nStat A -- selection miss rate by occupancy bin (all subjects pooled):")
    for (lo, hi), (miss, total) in bin_hits.items():
        rate = miss / total if total > 0 else float("nan")
        print(f"  m_W in ({lo:.3f},{hi:.3f}]: miss_rate={rate:.4f} (n={total})")

    valid_G = np.array([s["oracle_gap_G_s"] for s in subject_stats if not np.isnan(s["oracle_gap_G_s"])])
    print(f"\nStat B -- lesion-restricted oracle gap: mean G_s={valid_G.mean():.4f}, "
          f"median={np.median(valid_G):.4f}, n_subjects_with_qualifying_windows={len(valid_G)}")

    # ---------------- DECISION 1: oracle gap ~0? ----------------
    gap_near_zero = valid_G.mean() < 0.02
    if gap_near_zero:
        print("\n=== DECISION 1: Oracle gap ~0 -- KILL. MaxPool already agrees with the "
              "lesion-aware oracle whenever a real choice exists; competitive-selection-"
              "under-fractional-occupancy is not the mechanism. ===")
        summary = {
            "checkpoint": str(CKPT_PATH), "val_dice_check": val_dice,
            "n_subjects": len(subject_stats),
            "oracle_gap_mean": float(valid_G.mean()), "oracle_gap_median": float(np.median(valid_G)),
            "occupancy_bin_miss_rates": {f"{lo}-{hi}": (miss/total if total>0 else None)
                                          for (lo,hi),(miss,total) in bin_hits.items()},
            "verdict": "DECISION_1_KILL_ORACLE_GAP_NEAR_ZERO",
        }
        with open(OUT_DIR / "E119_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(OUT_DIR / "E119_subject_table.json", "w") as f:
            json.dump(subject_stats, f, indent=2)
        print("\nSaved E119_summary.json, E119_subject_table.json")
        return

    # ---------------- Stat C: N_b association ----------------
    print("\n=== Stat C: subject-level M_s vs N_b (min-window-filtered) ===")
    filtered = [s for s in subject_stats if s["n_fractional_windows"] >= MIN_FRACTIONAL_WINDOWS
                and not np.isnan(s["M_s_all_channel"])]
    n_excluded = len(subject_stats) - len(filtered)
    print(f"Included: {len(filtered)}, excluded (< {MIN_FRACTIONAL_WINDOWS} fractional windows or NaN): {n_excluded}")

    M_s_arr = np.array([s["M_s_all_channel"] for s in filtered])
    N_b_arr = np.array([s["n_b"] for s in filtered])
    size_arr = np.array([s["native_size"] for s in filtered])

    rho, p_param, p_perm = permutation_test_corr(M_s_arr, N_b_arr, SEED)
    print(f"Spearman(M_s, N_b) = {rho:+.4f} (param p={p_param:.4e}, perm p={p_perm:.4f})")

    log_size = np.log(size_arr + 1)
    X1 = np.column_stack([np.ones(len(log_size)), log_size])
    beta_m, *_ = np.linalg.lstsq(X1, M_s_arr, rcond=None)
    resid_m = M_s_arr - X1 @ beta_m
    beta_n, *_ = np.linalg.lstsq(X1, N_b_arr, rcond=None)
    resid_n = N_b_arr - X1 @ beta_n
    rho_partial, p_partial_param = stats.spearmanr(resid_m, resid_n)
    rng2 = np.random.default_rng(SEED + 1)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng2.permutation(resid_n)
        perm_rhos[i], _ = stats.spearmanr(resid_m, perm_y)
    p_partial_perm = float((np.abs(perm_rhos) >= np.abs(rho_partial)).mean())
    print(f"Partial Spearman (controlling native_size) = {rho_partial:+.4f} "
          f"(param p={p_partial_param:.4e}, perm p={p_partial_perm:.4f})")

    survives = (rho > 0) and (p_perm < 0.05) and (rho_partial > 0) and (p_partial_perm < 0.05)

    if not survives:
        decision = "DECISION_2_KILL_NOT_ASSOCIATED_WITH_NB"
        detail = ("Oracle gap is real (mean G_s={:.4f}) but selection miss rate M_s does not "
                   "significantly predict N_b, or the association does not survive controlling "
                   "for lesion size -- report as unexplained selection noise, do not proceed to "
                   "non-commutativity framing.").format(valid_G.mean())
    else:
        decision = "DECISION_3_SURVIVES_GENUINE_MECHANISTIC_PRECURSOR"
        detail = ("Oracle gap is real AND selection miss rate M_s predicts N_b beyond lesion "
                   "size -- genuine mechanistic precursor result. Per pre-registered sequence, "
                   "this is now (and only now) appropriate grounds to formulate an operator "
                   "non-commutativity hypothesis and design the next test -- NOT to implement "
                   "an intervention yet.")

    print(f"\n=== {decision} ===")
    print(detail)

    summary = {
        "checkpoint": str(CKPT_PATH), "val_dice_check": val_dice,
        "n_subjects_total": len(subject_stats),
        "n_subjects_min_window_filtered": len(filtered), "n_excluded": n_excluded,
        "oracle_gap_mean": float(valid_G.mean()), "oracle_gap_median": float(np.median(valid_G)),
        "occupancy_bin_miss_rates": {f"{lo}-{hi}": (miss/total if total>0 else None)
                                      for (lo,hi),(miss,total) in bin_hits.items()},
        "top_k_channels": top_k_channels.tolist(),
        "spearman_M_s_vs_N_b": rho, "perm_p_M_s_vs_N_b": p_perm,
        "partial_rho_controlling_size": float(rho_partial), "partial_perm_p": p_partial_perm,
        "decision": decision, "detail": detail,
    }
    with open(OUT_DIR / "E119_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(OUT_DIR / "E119_subject_table.json", "w") as f:
        json.dump(subject_stats, f, indent=2)
    print("\nSaved E119_summary.json, E119_subject_table.json")


if __name__ == "__main__":
    main()
