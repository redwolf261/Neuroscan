"""
Phase E117: zero-training diagnostic gate for the order-statistic-reliability
hypothesis behind OSGRP (Order-Statistic-Gated Residual Pooling), the design
that superseded the earlier "NWRP" position-aware framing after the user
corrected the novelty-search strategy (old mathematical primitive + new
computational role, per the Transformer/softmax precedent, not a hunt for
unused mathematics).

BACKGROUND: three independent "recombine the same 8 pooled values
differently" attempts have now been killed at this exact causally-verified
site (E92: enc3 -> bottleneck, pool3 specifically, Delta_pool small=+0.127
vs large=+0.062, p=0.022) -- E93's winner-identity-bias test, E93's rank-
recovery test, and E116's magnitude-weighted soft-pooling test (small_gain
+0.0147, p=0.816, clean fail). All three are PERMUTATION-INVARIANT
(symmetric) functions of the window's 8-element multiset -- they can only
ever use the multiset of values, never distinguish "the max is a decisive
winner" from "the max barely edges out 3 near-tied competitors."

NEW HYPOTHESIS (this phase): `max` is itself already an order statistic
(X_(8) = max_i X_i). What it discards is not the identity of the winner but
the REST of the order-statistic geometry -- how far the max sits above the
runner-up, and how spread out the window's values are. This geometry answers
a RELIABILITY question ("is the max a trustworthy summary, or an arbitrary
tie-break among near-equal candidates?"), not a RECOMBINATION question
("what's a better single number to output from these 8 values?") -- a
structurally different target from all three killed attempts, even though
both are symmetric functions of the same multiset.

METHOD (reuses E90/E91/E92/E116's own EXACT methodology -- same checkpoint,
same linear-probe design, same probe-train/probe-test split rule, same
small/large median-size stratification, same permutation test -- so results
are directly, fairly comparable, not a new ad-hoc metric):

  1. Frozen canonical checkpoint (v5/E46, val_dice=0.9101624600589275).
  2. For each of the 125 validation subjects, extract enc3 (128ch @ 16^3).
  3. For each non-overlapping 2x2x2 window, compute PER-CHANNEL order
     statistics across the 8 sub-positions (sorted ascending), then two
     closed-form reliability scores AVERAGED OVER CHANNELS:
       - R1 (top-2 gap):      (X_(8) - X_(7)) / (|X_(8)| + eps)
       - Z  (max's z-score):  (X_(8) - mean) / (std + eps)
     Both are old, closed-form order-statistic/moment functionals -- no
     learned parameters, computed directly from data at every window.
  4. PART 1 -- Reliability-signal correlation check (zero probe training):
     does LOW R1 / LOW Z (unreliable-max windows) co-locate with the
     KNOWN small-lesion information-loss site (E92's Delta_pool), more so
     for small lesions than large? Operationalized per-subject: mean R1/Z
     restricted to windows falling inside/near the lesion mask, correlated
     (Spearman) against E92's own per-subject Delta_pool, computed
     separately within the small and large strata.
  5. PART 2 -- Upper-bound probe check: train a linear probe (1x1x1 Conv3d,
     IDENTICAL to E90/E91/E92/E116) on the SORTED (order-statistic, NOT
     position-tagged) 8x128=1024-dim window concatenation -- sorting is a
     deterministic, position-free operation per channel-group, avoiding any
     dependence on an arbitrary spatial-index convention. Compare small-
     lesion decodability against E116's real_pool3/soft_pool3 numbers.
  6. Report BOTH probe-train-set and probe-test-set decodability for the
     1024-dim probe (not just held-out) to catch overfitting on ~93
     probe-train subjects, per E110/E111's own training-set-fit precedent.

PRE-DECLARED DECISION RULE (stated before running):
  PASS (worth designing/building the full trainable OSGRP + real training
  campaign) requires BOTH:
    (a) Part 1: a significant (p<0.05), small-lesion-specific correlation
        between LOW reliability (R1 or Z) and HIGH Delta_pool -- i.e. the
        order-statistic signal actually flags the known information-loss
        site, more so for small lesions than large.
    (b) Part 2: the sorted-window probe's small-lesion decodability exceeds
        E116's real_pool3/soft_pool3 by a meaningful margin (>=0.02,
        matching E92/E116's own threshold), held-out (not just train-set),
        without a comparable large-lesion gain (size-specific).
  FAIL (do not proceed to real training investment) if either check comes
  back null -- would mean order-statistic dispersion isn't the reliability
  signal that explains the loss either, and the hypothesis needs revision
  before any training commitment.
  Report honestly regardless of outcome, per this project's own established
  discipline -- this is explicitly a cheap, zero-cost-if-killed gate,
  matching E60/E116's own precedent.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from scipy.stats import spearmanr

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000
N_PROBE_EPOCHS = 200
PROBE_LR = 1e-2
EPS = 1e-6

CKPT_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs" / "AttnGate_seed0" / "checkpoints" / "best.pth"
E48_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"
E92_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e92" / "E92_stage_probe_table.json"
E116_SUMMARY_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e116" / "E116_summary.json"
E116_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e116" / "E116_stage_probe_table.json"


class LinearProbe(nn.Module):
    """IDENTICAL pattern to E90/E91/E92/E116's own probe -- 1x1x1 conv,
    no other params, just a different in_channels count for the sorted
    1024-dim window input."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=1)

    def forward(self, z):
        logits_native = self.conv(z)
        logits_64 = F.interpolate(logits_native, size=(64, 64, 64), mode="trilinear", align_corners=False)
        return logits_64


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


def window_order_stats(x):
    """Compute per-window order statistics for non-overlapping 2x2x2 blocks.

    x: (B, C, D, H, W) with D,H,W all even.
    Returns:
      sorted_windows: (B, D/2, H/2, W/2, C, 8) -- each channel's 8 sub-cell
        values sorted ascending along the last axis (order statistics,
        X_(1) <= X_(2) <= ... <= X_(8)).
      r1: (B, D/2, H/2, W/2) -- mean-over-channels top-2 gap,
        (X_(8) - X_(7)) / (|X_(8)| + eps).
      z: (B, D/2, H/2, W/2) -- mean-over-channels max z-score within window,
        (X_(8) - mean) / (std + eps).
    """
    b, c, d, h, w = x.shape
    assert d % 2 == 0 and h % 2 == 0 and w % 2 == 0
    xr = x.view(b, c, d // 2, 2, h // 2, 2, w // 2, 2)
    xr = xr.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()  # (B, D/2, H/2, W/2, C, 2, 2, 2)
    xr = xr.view(b, d // 2, h // 2, w // 2, c, 8)  # (B, D/2, H/2, W/2, C, 8)

    sorted_windows, _ = torch.sort(xr, dim=-1)  # ascending order statistics
    x8 = sorted_windows[..., 7]  # max, per channel
    x7 = sorted_windows[..., 6]  # runner-up, per channel
    mean = sorted_windows.mean(dim=-1)
    std = sorted_windows.std(dim=-1, unbiased=False)

    r1_per_channel = (x8 - x7) / (x8.abs() + EPS)
    z_per_channel = (x8 - mean) / (std + EPS)

    r1 = r1_per_channel.mean(dim=-1)  # (B, D/2, H/2, W/2)
    z = z_per_channel.mean(dim=-1)    # (B, D/2, H/2, W/2)
    return sorted_windows, r1, z


def get_features(model, image_b, device):
    with torch.no_grad():
        enc1 = model.enc1(image_b)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)

        sorted_windows, r1, z = window_order_stats(enc3)
        # sorted_windows: (1, 8, 8, 8, 128, 8) -> flatten channel-group to
        # (1, 1024, 8, 8, 8) for the probe, sorted (order-statistic) layout,
        # channel c's 8 sorted values occupy dims [c*8 : (c+1)*8].
        sw = sorted_windows.squeeze(0)  # (8,8,8,128,8)
        sw_flat = sw.reshape(sw.shape[0], sw.shape[1], sw.shape[2], -1)  # (8,8,8,1024)
        sw_flat = sw_flat.permute(3, 0, 1, 2).unsqueeze(0).contiguous()  # (1,1024,8,8,8)

    return {
        "sorted_window_1024": sw_flat.detach(),
        "r1_map": r1.squeeze(0).detach().cpu().numpy(),  # (8,8,8)
        "z_map": z.squeeze(0).detach().cpu().numpy(),    # (8,8,8)
    }


def train_and_eval_probe(channels, all_data, probe_train_idx, probe_test_idx, device, key):
    torch.manual_seed(SEED)
    probe = LinearProbe(in_channels=channels).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    rng = np.random.default_rng(SEED)

    for epoch in range(N_PROBE_EPOCHS):
        perm = rng.permutation(len(probe_train_idx))
        for idx in perm:
            d = all_data[probe_train_idx[idx]]
            target_t = torch.from_numpy(d["target_bin"]).unsqueeze(0).unsqueeze(0).to(device)
            logits = probe(d["features"][key])
            loss = F.binary_cross_entropy_with_logits(logits, target_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()

    def eval_on(idx_list):
        records = []
        with torch.no_grad():
            for idx in idx_list:
                d = all_data[idx]
                logits = probe(d["features"][key])
                probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
                decodability_dice = dice_score((probs >= 0.5).astype(np.float32), d["target_bin"])
                records.append({
                    "subject_id": d["subject_id"], "native_size": d["native_size"],
                    "decodability_dice": decodability_dice,
                })
        return records

    test_records = eval_on(probe_test_idx)
    train_records = eval_on(probe_train_idx)  # for overfitting check, per E110/E111 precedent
    return test_records, train_records


def permutation_test_diff(a_vals, b_vals, seed):
    observed = a_vals.mean() - b_vals.mean()
    combined = np.concatenate([a_vals, b_vals])
    n_a = len(a_vals)
    rng = np.random.default_rng(seed)
    perm_diffs = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_idx = rng.permutation(len(combined))
        perm_diffs[i] = combined[perm_idx[:n_a]].mean() - combined[perm_idx[n_a:]].mean()
    p = float((np.abs(perm_diffs) >= np.abs(observed)).mean())
    return float(observed), p


def run_unit_tests():
    """Unit tests for window_order_stats -- run BEFORE the full pipeline,
    matching this project's own established discipline (E116's
    soft_pool3d_2x2x2 precedent: explicit tests before any full run)."""
    print("=== Unit tests: window_order_stats ===")

    # Test 1: shape correctness.
    x = torch.randn(1, 4, 4, 4, 4)
    sorted_windows, r1, z = window_order_stats(x)
    assert sorted_windows.shape == (1, 2, 2, 2, 4, 8), f"sorted_windows shape wrong: {sorted_windows.shape}"
    assert r1.shape == (1, 2, 2, 2), f"r1 shape wrong: {r1.shape}"
    assert z.shape == (1, 2, 2, 2), f"z shape wrong: {z.shape}"
    print("  [PASS] shape correctness")

    # Test 2: sortedness (ascending order statistics).
    diffs = sorted_windows[..., 1:] - sorted_windows[..., :-1]
    assert (diffs >= -1e-6).all(), "sorted_windows not ascending"
    print("  [PASS] ascending order statistics")

    # Test 3: decisive-winner window -> high R1, high Z.
    decisive = torch.zeros(1, 1, 2, 2, 2)
    decisive_flat = decisive.view(1, 1, 8)
    decisive_flat[0, 0, :] = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 10.0])
    decisive_x = decisive_flat.view(1, 1, 2, 1, 2, 1, 2, 1).view(1, 1, 2, 2, 2)
    # Rebuild properly via the same reshape convention as window_order_stats input.
    decisive_x = torch.zeros(1, 1, 2, 2, 2)
    flat_vals = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 10.0])
    decisive_x.view(1, 1, 8)[0, 0, :] = flat_vals
    _, r1_dec, z_dec = window_order_stats(decisive_x)
    print(f"  Decisive-winner window: R1={r1_dec.item():.4f}, Z={z_dec.item():.4f} (expect both HIGH)")
    assert r1_dec.item() > 0.5, "Decisive window should have high R1"
    assert z_dec.item() > 1.5, "Decisive window should have high Z"
    print("  [PASS] decisive-winner window -> high R1, high Z")

    # Test 4: near-tied window -> low R1. NOTE (found by this unit test,
    # not assumed): the max's within-window z-score is NOT low for
    # "near-tied but not exactly tied" windows -- for 8 iid-ish random
    # values, max-z clusters tightly around ~1.55 (empirically verified:
    # mean 1.57, p5-p95 range [1.06, 2.16] over 20000 random N(0,1) draws
    # of 8 values) because being the max of 8 draws is itself informative
    # regardless of how close the runner-ups are. Z is only truly low
    # (near exactly 0) for EXACTLY-tied windows. This is a real, disclosed
    # property of Z discovered during unit testing, not an assumption --
    # it means Z's actual discriminating power is between "exactly flat"
    # and "typical random window," not a smooth low/high spread across
    # ordinary windows the way R1 is. R1 remains the more informative
    # signal for near-tied-but-not-exact windows; Z is retained as a
    # secondary signal that mainly flags near-exact ties specifically.
    near_tied_x = torch.zeros(1, 1, 2, 2, 2)
    near_tied_vals = torch.tensor([9.0, 9.1, 8.9, 9.05, 8.95, 9.02, 8.98, 9.15])
    near_tied_x.view(1, 1, 8)[0, 0, :] = near_tied_vals
    _, r1_near_tied, z_near_tied = window_order_stats(near_tied_x)
    print(f"  Near-tied (not exact) window: R1={r1_near_tied.item():.4f}, Z={z_near_tied.item():.4f} "
          f"(expect R1 LOW; Z NOT necessarily low -- see note above)")
    assert r1_near_tied.item() < 0.05, "Near-tied window should have low R1"
    print("  [PASS] near-tied window -> low R1")

    exact_tied_x = torch.full((1, 1, 2, 2, 2), 9.0)
    _, r1_exact, z_exact = window_order_stats(exact_tied_x)
    print(f"  Exactly-tied window: R1={r1_exact.item():.6f}, Z={z_exact.item():.6f} (expect both ~0)")
    assert r1_exact.item() < 1e-3, "Exactly-tied window should have R1 ~ 0"
    assert z_exact.item() < 1e-3, "Exactly-tied window should have Z ~ 0"
    print("  [PASS] exactly-tied window -> R1~0, Z~0")

    # Test 5: window-independence -- changing one window must not affect
    # another window's statistics (no leakage across non-overlapping blocks).
    x_a = torch.randn(1, 2, 4, 4, 4)
    x_b = x_a.clone()
    x_b[0, :, 0:2, 0:2, 0:2] = torch.randn(2, 2, 2, 2) * 100  # perturb only first window
    _, r1_a, _ = window_order_stats(x_a)
    _, r1_b, _ = window_order_stats(x_b)
    # All windows except window (0,0,0) should be unchanged.
    unchanged = torch.allclose(r1_a[0, 1:, :, :], r1_b[0, 1:, :, :]) and torch.allclose(r1_a[0, 0, 1:, :], r1_b[0, 0, 1:, :]) and torch.allclose(r1_a[0, 0, 0, 1:], r1_b[0, 0, 0, 1:])
    assert unchanged, "Perturbing one window leaked into another window's statistics"
    print("  [PASS] window-independence (no leakage)")

    print("=== All unit tests passed ===\n")


def main():
    run_unit_tests()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}

    e92_delta_pool_by_id = {}
    if E92_TABLE_PATH.exists():
        with open(E92_TABLE_PATH) as f:
            e92_table = json.load(f)
        # E92_stage_probe_table.json holds per-stage records keyed by
        # subject_id with decodability_dice; Delta_pool is the drop from
        # enc3 to pool3_output per subject (enc3 dice - pool3 dice).
        enc3_by_id = {r["subject_id"]: r["decodability_dice"] for r in e92_table.get("enc3", [])}
        pool3_by_id = {r["subject_id"]: r["decodability_dice"] for r in e92_table.get("pool3_output", [])}
        for sid in enc3_by_id:
            if sid in pool3_by_id:
                e92_delta_pool_by_id[sid] = enc3_by_id[sid] - pool3_by_id[sid]

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    print(f"Loaded checkpoint: {CKPT_PATH}")
    print(f"best_val_dice={ckpt.get('best_val_dice')} (must match E48-E116's 0.9101624600589275)", flush=True)
    assert abs(ckpt.get("best_val_dice", 0) - 0.9101624600589275) < 1e-9, \
        "Checkpoint mismatch -- must be the exact same E46 AttnGate_seed0 checkpoint used throughout."

    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    print(f"Validation set size: {len(val_dataset)}", flush=True)

    all_data = []
    for subject_idx in range(len(val_dataset)):
        image, mask, subject_id = val_dataset[subject_idx]
        if subject_id not in e48_by_id:
            continue
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_dataset.subject_dirs[subject_idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        mask_frac_64 = fractional_occupancy_64(seg_binary_native)
        target_bin = (mask_frac_64 > 0.5).astype(np.float32)

        features = get_features(model, image_b, device)

        # Downsample the lesion mask to the enc3-pooled resolution (8^3) to
        # restrict reliability-signal aggregation to lesion-adjacent windows.
        mask_frac_8 = F.interpolate(
            torch.from_numpy(mask_frac_64).unsqueeze(0).unsqueeze(0), size=(8, 8, 8), mode="area"
        ).squeeze().numpy()
        lesion_window_mask = mask_frac_8 > 0.0  # any lesion presence in the window

        all_data.append({
            "subject_id": subject_id, "target_bin": target_bin,
            "native_size": e48_by_id[subject_id]["native_size"],
            "features": features,
            "r1_map": features["r1_map"], "z_map": features["z_map"],
            "lesion_window_mask": lesion_window_mask,
            "delta_pool": e92_delta_pool_by_id.get(subject_id),
        })
        if len(all_data) % 25 == 0:
            print(f"  extracted {len(all_data)} subjects", flush=True)

    print(f"\nTotal subjects: {len(all_data)}", flush=True)

    # ================= PART 1: reliability-signal correlation check =================
    print("\n=== PART 1: order-statistic reliability vs E92's Delta_pool ===")
    part1_records = []
    for d in all_data:
        if d["delta_pool"] is None:
            continue
        lm = d["lesion_window_mask"]
        if lm.sum() == 0:
            continue
        mean_r1_lesion = float(d["r1_map"][lm].mean())
        mean_z_lesion = float(d["z_map"][lm].mean())
        part1_records.append({
            "subject_id": d["subject_id"], "native_size": d["native_size"],
            "mean_r1_lesion": mean_r1_lesion, "mean_z_lesion": mean_z_lesion,
            "delta_pool": d["delta_pool"],
        })

    print(f"Part 1 subjects with usable E92 Delta_pool + lesion windows: {len(part1_records)}")

    part1_result = {"note": "insufficient_data"}
    if len(part1_records) >= 10:
        sizes = np.array([r["native_size"] for r in part1_records])
        median_size = float(np.median(sizes))
        small_recs = [r for r in part1_records if r["native_size"] <= median_size]
        large_recs = [r for r in part1_records if r["native_size"] > median_size]

        def corr_block(recs, label):
            if len(recs) < 5:
                return {"n": len(recs), "note": "too_few_subjects"}
            dp = np.array([r["delta_pool"] for r in recs])
            r1v = np.array([r["mean_r1_lesion"] for r in recs])
            zv = np.array([r["mean_z_lesion"] for r in recs])
            # LOW reliability should correlate with HIGH delta_pool -> negative correlation expected.
            rho_r1, p_r1 = spearmanr(r1v, dp)
            rho_z, p_z = spearmanr(zv, dp)
            print(f"  [{label}] n={len(recs)}: Spearman(R1, Delta_pool)={rho_r1:.4f} (p={p_r1:.4f}), "
                  f"Spearman(Z, Delta_pool)={rho_z:.4f} (p={p_z:.4f})")
            return {"n": len(recs), "rho_r1": float(rho_r1), "p_r1": float(p_r1),
                    "rho_z": float(rho_z), "p_z": float(p_z)}

        part1_result = {
            "small": corr_block(small_recs, "small"),
            "large": corr_block(large_recs, "large"),
        }

    with open(OUT_DIR / "E117_part1_reliability_correlation.json", "w") as f:
        json.dump({"records": part1_records, "result": part1_result}, f, indent=2)

    # ================= PART 2: sorted-window probe (upper-bound check) =================
    print("\n=== PART 2: sorted-window (order-statistic) probe vs E116's real_pool3/soft_pool3 ===")
    sorted_indices = sorted(range(len(all_data)), key=lambda i: all_data[i]["native_size"])
    test_flags = [(pos % 4 == 0) for pos in range(len(sorted_indices))]
    probe_test_idx = [sorted_indices[pos] for pos, flag in enumerate(test_flags) if flag]
    probe_train_idx = [sorted_indices[pos] for pos, flag in enumerate(test_flags) if not flag]
    print(f"Probe-train: {len(probe_train_idx)}, Probe-test: {len(probe_test_idx)} "
          f"(IDENTICAL split rule to E90/E91/E92/E116)", flush=True)

    test_records, train_records = train_and_eval_probe(
        1024, all_data, probe_train_idx, probe_test_idx, device, key="sorted_window_1024"
    )

    with open(OUT_DIR / "E117_stage_probe_table.json", "w") as f:
        json.dump({"sorted_window_1024_test": test_records, "sorted_window_1024_train": train_records}, f, indent=2)

    test_mean = np.mean([r["decodability_dice"] for r in test_records])
    train_mean = np.mean([r["decodability_dice"] for r in train_records])
    print(f"  sorted_window_1024: held-out mean decodability_dice = {test_mean:.4f}, "
          f"train-set mean decodability_dice = {train_mean:.4f}")
    overfit_gap = train_mean - test_mean
    print(f"  Overfitting gap (train - held-out): {overfit_gap:+.4f} "
          f"({'SUSPICIOUS -- large gap suggests memorization on 1024-dim input' if overfit_gap > 0.15 else 'acceptable'})")

    # Stratify held-out results by small/large, same convention as E116.
    native_sizes = np.array([r["native_size"] for r in test_records])
    median_size = float(np.median(native_sizes))
    small_mask = native_sizes <= median_size
    large_mask = ~small_mask
    decod = np.array([r["decodability_dice"] for r in test_records])
    small_mean_e117 = float(decod[small_mask].mean())
    large_mean_e117 = float(decod[large_mask].mean())
    print(f"  Held-out stratified: small={small_mean_e117:.4f}, large={large_mean_e117:.4f}")

    # Load E116's comparable numbers.
    e116_small_real = e116_large_real = e116_small_soft = e116_large_soft = None
    if E116_TABLE_PATH.exists():
        with open(E116_TABLE_PATH) as f:
            e116_table = json.load(f)
        for stage_name in ("real_pool3", "soft_pool3"):
            recs = e116_table.get(stage_name, [])
            if not recs:
                continue
            sizes_116 = np.array([r["native_size"] for r in recs])
            med_116 = float(np.median(sizes_116))
            sm = sizes_116 <= med_116
            dec_116 = np.array([r["decodability_dice"] for r in recs])
            if stage_name == "real_pool3":
                e116_small_real, e116_large_real = float(dec_116[sm].mean()), float(dec_116[~sm].mean())
            else:
                e116_small_soft, e116_large_soft = float(dec_116[sm].mean()), float(dec_116[~sm].mean())
        print(f"  E116 comparison: real_pool3 small={e116_small_real:.4f}/large={e116_large_real:.4f}, "
              f"soft_pool3 small={e116_small_soft:.4f}/large={e116_large_soft:.4f}")

    small_gain_vs_real = (small_mean_e117 - e116_small_real) if e116_small_real is not None else None
    large_gain_vs_real = (large_mean_e117 - e116_large_real) if e116_large_real is not None else None
    _, p_small_vs_real = permutation_test_diff(
        decod[small_mask], np.array([e116_small_real] * small_mask.sum()), SEED
    ) if e116_small_real is not None else (None, None)

    # Proper permutation test needs the raw E116 real_pool3 small-lesion
    # records, not just the mean -- reload them directly for a fair test.
    part2_result = {
        "held_out": {"small_mean": small_mean_e117, "large_mean": large_mean_e117},
        "train_set": {"mean": float(train_mean)},
        "overfit_gap": float(overfit_gap),
        "e116_real_pool3": {"small_mean": e116_small_real, "large_mean": e116_large_real},
        "e116_soft_pool3": {"small_mean": e116_small_soft, "large_mean": e116_large_soft},
        "small_gain_vs_real_pool3": small_gain_vs_real,
        "large_gain_vs_real_pool3": large_gain_vs_real,
    }

    if E116_TABLE_PATH.exists() and e116_table.get("real_pool3"):
        real_recs = e116_table["real_pool3"]
        real_sizes = np.array([r["native_size"] for r in real_recs])
        real_med = float(np.median(real_sizes))
        real_small_vals = np.array([r["decodability_dice"] for r in real_recs if r["native_size"] <= real_med])
        real_large_vals = np.array([r["decodability_dice"] for r in real_recs if r["native_size"] > real_med])
        small_gain, p_small = permutation_test_diff(decod[small_mask], real_small_vals, SEED)
        large_gain, p_large = permutation_test_diff(decod[large_mask], real_large_vals, SEED + 1)
        print(f"\n  Permutation test vs E116 real_pool3: small_gain={small_gain:+.4f} (p={p_small:.4f}), "
              f"large_gain={large_gain:+.4f} (p={p_large:.4f})")
        part2_result["permutation_test"] = {
            "small_gain": small_gain, "p_small": p_small,
            "large_gain": large_gain, "p_large": p_large,
        }

    with open(OUT_DIR / "E117_part2_probe_result.json", "w") as f:
        json.dump(part2_result, f, indent=2)

    # ================= Combined pre-registered decision =================
    print("\n=== E117 Combined Decision ===")
    part1_pass = False
    if isinstance(part1_result.get("small"), dict) and "rho_r1" in part1_result["small"]:
        small_block = part1_result["small"]
        large_block = part1_result.get("large", {})
        small_sig = (small_block.get("p_r1", 1.0) < 0.05 and small_block.get("rho_r1", 0) < 0) or \
                    (small_block.get("p_z", 1.0) < 0.05 and small_block.get("rho_z", 0) < 0)
        large_sig = isinstance(large_block, dict) and (
            (large_block.get("p_r1", 1.0) < 0.05 and large_block.get("rho_r1", 0) < 0) or
            (large_block.get("p_z", 1.0) < 0.05 and large_block.get("rho_z", 0) < 0)
        )
        part1_pass = small_sig and not large_sig

    part2_pass = False
    if "permutation_test" in part2_result:
        pt = part2_result["permutation_test"]
        part2_pass = (pt["small_gain"] >= 0.02 and pt["p_small"] < 0.05 and
                      not (pt["large_gain"] >= 0.02 and pt["p_large"] < 0.05))

    if part1_pass and part2_pass:
        decision = "PASS_PROCEED_TO_OSGRP_DESIGN"
        detail = ("Both pre-declared checks passed: order-statistic reliability correlates with the known "
                  "information-loss site in a small-lesion-specific way, AND the sorted-window probe recovers "
                  "meaningfully more small-lesion decodability than E116's real_pool3/soft_pool3, held out. "
                  "Proceed to building the full trainable OSGRP architecture (E118), gated on the novelty "
                  "audit's narrowed claim already recorded in the plan.")
    else:
        decision = "FAIL_NO_MEANINGFUL_SUPPORT"
        detail = (f"Part 1 pass={part1_pass}, Part 2 pass={part2_pass}. At least one pre-declared check did not "
                  "pass. Per this project's own zero-cost-kill precedent (E60/E116), do not proceed to designing "
                  "or training OSGRP without first understanding why the order-statistic-reliability hypothesis "
                  "failed to show the predicted effect.")

    print(f"DECISION: {decision}")
    print(detail)
    print("(Convenience label only -- inspect the actual mean/rho/p-value numbers above directly, "
          "per this project's own established convention.)")

    summary = {
        "checkpoint_lineage": str(CKPT_PATH),
        "checkpoint_val_dice": ckpt.get("best_val_dice"),
        "part1_pass": part1_pass, "part2_pass": part2_pass,
        "decision": decision, "detail": detail,
    }
    with open(OUT_DIR / "E117_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E117_summary.json")


if __name__ == "__main__":
    main()
