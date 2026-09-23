"""
Phase E116: zero-training diagnostic gate for replacing MaxPool3d(2,2) at
the exact causally-verified information-loss site (E92: enc3 -> bottleneck,
pool3 specifically) with a non-destructive, content-adaptive alternative.

BACKGROUND: twelve prior interventions on this project's own bottleneck-
necessity signal (N_b) all worked DOWNSTREAM of pool3 -- amplifying/gating
the bottleneck (CCABA/IECG/CCAG), reweighting the loss (ASR), or bypassing/
recovering AFTER pooling (E93 rank-recovery, E97 linear residual bypass).
All twelve failed. None of them changed what pool3 itself COMPUTES at the
exact site E92 causally proved is where small-lesion information is
disproportionately destroyed (Delta_pool small=+0.127 vs large=+0.062,
p=0.022) -- versus the subsequent channel transformation, which showed no
comparable size-specific effect. This phase tests the ROOT-CAUSE-TARGETING
alternative directly: does REPLACING the destructive max-selection with a
non-destructive alternative, applied to the SAME frozen enc3 activations
(no retraining), recover more small-lesion decodability than real pool3
does -- BEFORE committing to designing a trainable, causally-conditioned
version or any real training campaign?

METHOD (reuses E90/E91/E92's own EXACT methodology -- same checkpoint, same
linear-probe design, same probe-train/probe-test split rule, same small/
large median-size stratification, same permutation test -- so results are
directly, fairly comparable to E92's own numbers, not a new ad-hoc metric):
  1. Frozen canonical checkpoint (v5/E46, val_dice=0.9101624600589275).
  2. For each of the 125 validation subjects, extract enc3 (128ch @ 16^3).
  3. Compute TWO alternative pool3 outputs from the SAME enc3, both
     (128ch @ 8^3), no retraining:
       (a) real_pool3 = model.pool3(enc3)  -- the actual, real MaxPool3d,
           for a same-run reproducibility check against E92's own numbers.
       (b) soft_pool3 = magnitude-weighted soft pooling: for each 2x2x2
           window, compute each of the 8 sub-cell positions' L2 norm
           across channels as a saliency score, softmax-normalize into
           weights, and take the WEIGHTED AVERAGE of the full 128-dim
           channel vectors (not per-channel-independent softpool -- this
           preserves cross-channel co-activation structure at each
           contributing position). This is a deliberately simple,
           established, non-adaptive-to-OUR-signal baseline (SoftPool,
           Stergiou et al. 2021) -- the point of THIS diagnostic is only
           to test whether ANY non-destructive alternative at this exact
           site helps, before designing anything causally-conditioned or
           novel. If this simplest alternative doesn't even clear the
           bar, a more sophisticated causally-conditioned version is not
           worth designing yet either.
  4. Train a linear probe (1x1x1 Conv3d, IDENTICAL to E90/E91/E92) on each
     of real_pool3 and soft_pool3, held out on the same probe-test split,
     stratified by the same small/large median-size split.
  5. Compare decodability(soft_pool3, small) vs decodability(real_pool3,
     small) -- does the alternative recover MORE small-lesion information
     at this exact site?

PRE-DECLARED DECISION RULE (stated before running):
  - PASS (worth designing a trainable/causally-conditioned version): mean
    decodability_dice(soft_pool3, small) > mean decodability_dice(real_
    pool3, small) by a MEANINGFUL margin (>= 0.02, matching E92's own
    threshold for calling a difference "elevated"), AND does not come at
    a comparable cost to LARGE-lesion decodability (a real small-lesion-
    specific gain, not a generic decodability shift that would show up
    for large lesions too).
  - FAIL (kill before any further design/training investment): no
    meaningful small-lesion-specific gain, or a gain that appears equally
    for large lesions too (not small-lesion-specific), or a real cost to
    large-lesion decodability.
  Report honestly regardless of outcome, per this project's own
  established discipline -- this is explicitly a cheap, zero-cost-if-
  killed gate, matching E60's own precedent (zero-training feasibility
  test before any training investment).
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000
N_PROBE_EPOCHS = 200
PROBE_LR = 1e-2

CKPT_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs" / "AttnGate_seed0" / "checkpoints" / "best.pth"
E48_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"
E92_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e92" / "E92_stage_probe_table.json"

STAGES = {
    "real_pool3": {"channels": 128, "resolution": 8},
    "soft_pool3": {"channels": 128, "resolution": 8},
}


class LinearProbe(nn.Module):
    """IDENTICAL to E90/E91/E92's own probe -- 1x1x1 conv, no other params."""
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


def soft_pool3d_2x2x2(x):
    """Magnitude-weighted soft pooling, kernel=2 stride=2, preserving
    cross-channel structure (NOT per-channel-independent SoftPool).

    x: (B, C, D, H, W) with D,H,W all even.
    For each non-overlapping 2x2x2 window, compute each of the 8
    sub-positions' L2 norm across channels as a saliency score, softmax-
    normalize across the 8 positions, and return the weighted average of
    the full C-dim channel vectors at each contributing position.
    """
    b, c, d, h, w = x.shape
    assert d % 2 == 0 and h % 2 == 0 and w % 2 == 0
    # Unfold into non-overlapping 2x2x2 blocks: (B, C, D/2, 2, H/2, 2, W/2, 2)
    xr = x.view(b, c, d // 2, 2, h // 2, 2, w // 2, 2)
    # Move the 3 "within-window" axes (each size 2) to the end, flatten to 8.
    xr = xr.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()  # (B, D/2, H/2, W/2, C, 2, 2, 2)
    xr = xr.view(b, d // 2, h // 2, w // 2, c, 8)  # (B, D/2, H/2, W/2, C, 8)

    # Saliency score per sub-position: L2 norm across channels.
    saliency = xr.norm(dim=4)  # (B, D/2, H/2, W/2, 8)
    weights = F.softmax(saliency, dim=-1)  # (B, D/2, H/2, W/2, 8)

    # Weighted average of the full channel vectors.
    weighted = (xr * weights.unsqueeze(4)).sum(dim=-1)  # (B, D/2, H/2, W/2, C)
    out = weighted.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D/2, H/2, W/2)
    return out


def get_pool3_variants(model, image_b, device):
    with torch.no_grad():
        enc1 = model.enc1(image_b)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)

        real_pool3 = model.pool3(enc3)          # actual MaxPool3d(2,2)
        soft_pool3 = soft_pool3d_2x2x2(enc3)     # alternative, same enc3 input
    return {"real_pool3": real_pool3.detach(), "soft_pool3": soft_pool3.detach()}


def train_and_eval_probe(stage_name, channels, all_data, probe_train_idx, probe_test_idx, device):
    torch.manual_seed(SEED)
    probe = LinearProbe(in_channels=channels).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    rng = np.random.default_rng(SEED)

    for epoch in range(N_PROBE_EPOCHS):
        perm = rng.permutation(len(probe_train_idx))
        for idx in perm:
            d = all_data[probe_train_idx[idx]]
            target_t = torch.from_numpy(d["target_bin"]).unsqueeze(0).unsqueeze(0).to(device)
            logits = probe(d["features"][stage_name])
            loss = F.binary_cross_entropy_with_logits(logits, target_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    records = []
    with torch.no_grad():
        for idx in probe_test_idx:
            d = all_data[idx]
            logits = probe(d["features"][stage_name])
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
            decodability_dice = dice_score((probs >= 0.5).astype(np.float32), d["target_bin"])
            records.append({
                "subject_id": d["subject_id"], "native_size": d["native_size"],
                "decodability_dice": decodability_dice,
            })
    return records


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    print(f"Loaded checkpoint: {CKPT_PATH}")
    print(f"best_val_dice={ckpt.get('best_val_dice')} (must match E48-E92's 0.9101624600589275)", flush=True)
    assert abs(ckpt.get("best_val_dice", 0) - 0.9101624600589275) < 1e-9, \
        "Checkpoint mismatch -- must be the exact same E46 AttnGate_seed0 checkpoint used throughout."

    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # ================= Sanity check: soft_pool3d_2x2x2 shape/behavior =================
    test_x = torch.randn(1, 4, 4, 4, 4)
    test_out = soft_pool3d_2x2x2(test_x)
    assert test_out.shape == (1, 4, 2, 2, 2), f"Shape mismatch: {test_out.shape}"
    # A window with one dominant-magnitude vector should output close to that vector.
    dominant = torch.zeros(1, 4, 2, 2, 2, 2, 2, 2)
    dominant[0, :, 0, 0, 0, 0, 0, 0] = torch.tensor([10.0, 10.0, 10.0, 10.0])
    dominant_flat = dominant.view(1, 4, 8, 8)[:, :, 0, :].view(1, 4, 2, 2, 2)
    out_dom = soft_pool3d_2x2x2(dominant_flat)
    print(f"[Sanity check] soft_pool3d_2x2x2 output shape correct: {test_out.shape == (1, 4, 2, 2, 2)}")
    print(f"[Sanity check] dominant-vector window output (expect close to [10,10,10,10] in one cell): "
          f"{out_dom[0, :, 0, 0, 0].tolist()}")
    print()

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

        features = get_pool3_variants(model, image_b, device)

        all_data.append({
            "subject_id": subject_id, "target_bin": target_bin,
            "native_size": e48_by_id[subject_id]["native_size"], "features": features,
        })
        if len(all_data) % 25 == 0:
            print(f"  extracted {len(all_data)} subjects", flush=True)

    print(f"\nTotal subjects: {len(all_data)}", flush=True)

    # IDENTICAL split rule to E90/E91/E92.
    sorted_indices = sorted(range(len(all_data)), key=lambda i: all_data[i]["native_size"])
    test_flags = [(pos % 4 == 0) for pos in range(len(sorted_indices))]
    probe_test_idx = [sorted_indices[pos] for pos, flag in enumerate(test_flags) if flag]
    probe_train_idx = [sorted_indices[pos] for pos, flag in enumerate(test_flags) if not flag]
    print(f"Probe-train: {len(probe_train_idx)}, Probe-test: {len(probe_test_idx)} "
          f"(IDENTICAL split rule to E90/E91/E92)", flush=True)

    stage_results = {}
    for stage_name, stage_info in STAGES.items():
        print(f"\n=== Training probe for stage: {stage_name} "
              f"({stage_info['channels']}ch @ {stage_info['resolution']}^3) ===", flush=True)
        records = train_and_eval_probe(
            stage_name, stage_info["channels"], all_data, probe_train_idx, probe_test_idx, device
        )
        stage_results[stage_name] = records
        print(f"  {stage_name}: mean decodability_dice = "
              f"{np.mean([r['decodability_dice'] for r in records]):.4f}", flush=True)

    with open(OUT_DIR / "E116_stage_probe_table.json", "w") as f:
        json.dump(stage_results, f, indent=2)

    # ================= Cross-check real_pool3 against E92's own pool3_output =================
    if E92_TABLE_PATH.exists():
        with open(E92_TABLE_PATH) as f:
            e92_table = json.load(f)
        e116_mean = np.mean([r["decodability_dice"] for r in stage_results["real_pool3"]])
        e92_mean = np.mean([r["decodability_dice"] for r in e92_table["pool3_output"]])
        print(f"\nCross-check real_pool3 vs E92's pool3_output: E116={e116_mean:.4f}, E92={e92_mean:.4f}, "
              f"diff={abs(e116_mean - e92_mean):.4f} (should be small -- identical construction)")

    # ================= Small vs large stratification =================
    print(f"\n=== E116 Adaptive-Pooling Diagnostic ===")
    per_stage_strat = {}
    for stage_name, records in stage_results.items():
        native_sizes = np.array([r["native_size"] for r in records])
        median_size = float(np.median(native_sizes))
        small_mask = native_sizes <= median_size
        large_mask = ~small_mask
        decod = np.array([r["decodability_dice"] for r in records])
        per_stage_strat[stage_name] = {
            "small": decod[small_mask], "large": decod[large_mask],
            "small_mean": float(decod[small_mask].mean()), "large_mean": float(decod[large_mask].mean()),
        }
        print(f"  {stage_name:12s}: small={decod[small_mask].mean():.4f}, large={decod[large_mask].mean():.4f}")

    small_gain, p_small = permutation_test_diff(
        per_stage_strat["soft_pool3"]["small"], per_stage_strat["real_pool3"]["small"], SEED
    )
    large_gain, p_large = permutation_test_diff(
        per_stage_strat["soft_pool3"]["large"], per_stage_strat["real_pool3"]["large"], SEED + 1
    )
    print(f"\nSmall-lesion gain (soft_pool3 - real_pool3): {small_gain:+.4f}, permutation p={p_small:.4f}")
    print(f"Large-lesion gain (soft_pool3 - real_pool3): {large_gain:+.4f}, permutation p={p_large:.4f}")

    meaningful_small_gain = (small_gain >= 0.02) and (p_small < 0.05)
    small_specific = meaningful_small_gain and not (large_gain >= 0.02 and p_large < 0.05)
    generic_gain = meaningful_small_gain and (large_gain >= 0.02 and p_large < 0.05)
    large_cost = (large_gain <= -0.02) and (p_large < 0.05)

    if small_specific:
        decision = "PASS_SMALL_LESION_SPECIFIC_GAIN"
        detail = ("Soft pooling recovers meaningfully more small-lesion decodability than real MaxPool3d, "
                  "without a comparable gain for large lesions -- a genuine, size-specific effect at the "
                  "exact causally-verified information-loss site. Worth designing a trainable, causally-"
                  "conditioned version and testing it with real training.")
    elif generic_gain:
        decision = "GENERIC_GAIN_NOT_SIZE_SPECIFIC"
        detail = ("Soft pooling recovers more decodability for BOTH small and large lesions -- a real "
                  "effect, but not the SIZE-SPECIFIC fix this project's own diagnosis (E92) motivated. "
                  "Report honestly; this does not confirm the size-specific hypothesis even if the overall "
                  "decodability change is real.")
    elif large_cost:
        decision = "COST_TO_LARGE_LESIONS"
        detail = ("Soft pooling costs large-lesion decodability -- a real tradeoff that would need to be "
                  "weighed, not just ignored, before any further design.")
    else:
        decision = "FAIL_NO_MEANINGFUL_GAIN"
        detail = ("No meaningful, size-specific small-lesion decodability gain from replacing MaxPool3d "
                  "with a simple magnitude-weighted soft pooling alternative at this exact site. Per this "
                  "project's own zero-cost-kill precedent (E60), this specific alternative should be killed "
                  "here -- do not proceed to a trainable/causally-conditioned version without first "
                  "understanding WHY this simplest alternative failed to help.")

    print(f"\n=== DECISION: {decision} ===")
    print(detail)
    print("(Convenience label only -- inspect the actual mean/p-value numbers above directly, "
          "per this project's own established convention.)")

    summary = {
        "checkpoint_lineage": str(CKPT_PATH),
        "checkpoint_val_dice": ckpt.get("best_val_dice"),
        "per_stage_strat_means": {k: {"small_mean": v["small_mean"], "large_mean": v["large_mean"]} for k, v in per_stage_strat.items()},
        "small_gain": small_gain, "small_gain_p": p_small,
        "large_gain": large_gain, "large_gain_p": p_large,
        "decision": decision, "detail": detail,
    }
    with open(OUT_DIR / "E116_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E116_summary.json")


if __name__ == "__main__":
    main()
