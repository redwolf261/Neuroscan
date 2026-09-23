"""
Phase E92: Pool-vs-Transform Decomposition of the enc3->bottleneck Loss.

CONTEXT: E91 localized a progressively-building small-lesion information
deficit that first reaches statistical significance at the bottleneck
(256ch @ 8^3), with enc1-enc3 showing an accumulating but individually
non-significant trend (diffs: +0.017, -0.030, -0.081, then -0.107 at
bottleneck, p=0.026 only at the bottleneck). User's correction: this
supports "progressive attrition, first STATISTICALLY DETECTABLE at the
bottleneck" -- NOT "the 16^3->8^3 pooling operation is proven to be the
cause." The enc3->bottleneck transition itself contains TWO distinct
operations that could each independently destroy small-lesion
information:

  (a) pool3: MaxPool3d(kernel_size=2, stride=2) -- PURE spatial
      downsampling, 16^3 -> 8^3, 128 channels unchanged, NO learnable
      parameters.
  (b) bottleneck Sequential: two Conv3DBlock layers (128->256->256
      channels, both stride=1, kernel=3, so NO further spatial change)
      -- pure CHANNEL/feature transformation on the already-pooled 8^3
      grid.

THIS PHASE decomposes the enc3->bottleneck decodability drop into these
two components, using the IDENTICAL linear-probe methodology as
E90/E91 (same subjects, same probe-train/probe-test split, same seed,
same optimizer/epochs, same small-vs-large stratification), by probing
an INTERMEDIATE tensor that did not exist as a named quantity in E91:

  Z'_3 = pool3(enc3)   -- (128ch, 8^3) -- pure pooling output, BEFORE
                           the bottleneck's channel transformation

Now we have FOUR points on the enc3->bottleneck path:
  enc3 (128ch @ 16^3)  ->  Z'_3 (128ch @ 8^3)  ->  bottleneck (256ch @ 8^3)
  [already have from E91]  [NEW, this phase]      [already have from E91]

Decodability at each lets us isolate:
  Delta_pool      = decodability(enc3) - decodability(Z'_3)
                    -- loss attributable to SPATIAL DOWNSAMPLING alone
                    (channel count unchanged, 128ch both sides)
  Delta_transform  = decodability(Z'_3) - decodability(bottleneck)
                    -- loss attributable to the CONVOLUTIONAL CHANNEL
                    TRANSFORMATION alone (spatial resolution unchanged,
                    8^3 both sides)

Computed SEPARATELY for small and large lesion strata, so we can ask:
  Is Delta_pool bigger for small lesions than large? (H1: pooling
  disproportionately destroys small-lesion information)
  Is Delta_transform bigger for small lesions than large? (H2: the
  channel transformation disproportionately destroys it)
  Or comparable magnitude for both? (H3: interaction/neither alone
  dominates)

PRE-DECLARED INTERPRETATION:
  - Delta_pool(small) >> Delta_pool(large), with Delta_transform(small)
    ~= Delta_transform(large): H1, pooling is the dominant mechanism --
    motivates a mechanism for SELECTIVE PRESERVATION OF SMALL-LESION
    INFORMATION ACROSS THE DOWNSAMPLING STEP specifically (e.g.
    attention-weighted or learned pooling, NOT naive resolution
    increase, which is the crowded/occupied direction the user
    explicitly wants to avoid).
  - Delta_transform(small) >> Delta_transform(large), with Delta_pool
    comparable: H2, the convolutional channel mixing is where small-
    lesion signal is preferentially discarded -- motivates a mechanism
    targeting the CONVOLUTION/CHANNEL-MIXING step, not pooling.
  - Both meaningfully elevated for small lesions, neither clearly
    dominant: H3, interaction -- would need a joint (not
    single-operation) intervention, and is the most likely case worth
    treating cautiously per this project's own repeated finding that
    "distributed/interaction effects" are harder to fix with a single
    targeted mechanism (cf. E65's own translation-dominates-but-not-
    exclusively finding).

Checkpoint lineage: SAME single checkpoint as E48/E85/E86/E88/E89/E90/E91
(E46 AttnGate_seed0/best.pth), asserted explicitly, not assumed.
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
E91_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e91" / "E91_stage_probe_table.json"

STAGES = {
    "enc3": {"channels": 128, "resolution": 16},
    "pool3_output": {"channels": 128, "resolution": 8},  # NEW intermediate point
    "bottleneck": {"channels": 256, "resolution": 8},
}


class LinearProbe(nn.Module):
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


def get_all_stage_features(model, image_b, device):
    """Single forward pass, now ALSO capturing Z'_3 = pool3(enc3) -- the
    exact intermediate point between pure spatial downsampling and the
    bottleneck's channel transformation."""
    with torch.no_grad():
        enc1 = model.enc1(image_b)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3_output = model.pool3(enc3)  # NEW: pure pooling output, 128ch @ 8^3
        bottleneck = model.bottleneck(pool3_output)
    return {
        "enc3": enc3.detach(),
        "pool3_output": pool3_output.detach(),
        "bottleneck": bottleneck.detach(),
    }


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

    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    print(f"Loaded checkpoint: {CKPT_PATH}")
    print(f"best_val_dice={ckpt.get('best_val_dice')} (must match E48-E91's 0.9101624600589275)", flush=True)
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

        features = get_all_stage_features(model, image_b, device)

        all_data.append({
            "subject_id": subject_id, "target_bin": target_bin,
            "native_size": e48_by_id[subject_id]["native_size"], "features": features,
        })
        if len(all_data) % 25 == 0:
            print(f"  extracted {len(all_data)} subjects", flush=True)

    print(f"\nTotal subjects: {len(all_data)}", flush=True)

    # IDENTICAL split rule to E90/E91.
    sorted_indices = sorted(range(len(all_data)), key=lambda i: all_data[i]["native_size"])
    test_flags = [(pos % 4 == 0) for pos in range(len(sorted_indices))]
    probe_test_idx = [sorted_indices[pos] for pos, flag in enumerate(test_flags) if flag]
    probe_train_idx = [sorted_indices[pos] for pos, flag in enumerate(test_flags) if not flag]
    print(f"Probe-train: {len(probe_train_idx)}, Probe-test: {len(probe_test_idx)} "
          f"(IDENTICAL split rule to E90/E91)", flush=True)

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

    with open(OUT_DIR / "E92_stage_probe_table.json", "w") as f:
        json.dump(stage_results, f, indent=2)

    # ================= Cross-check enc3/bottleneck against E91 =================
    if E91_TABLE_PATH.exists():
        with open(E91_TABLE_PATH) as f:
            e91_table = json.load(f)
        for stage in ["enc3", "bottleneck"]:
            e92_mean = np.mean([r["decodability_dice"] for r in stage_results[stage]])
            e91_mean = np.mean([r["decodability_dice"] for r in e91_table[stage]])
            print(f"\nCross-check {stage}: E92={e92_mean:.4f}, E91={e91_mean:.4f}, "
                  f"diff={abs(e92_mean - e91_mean):.4f} (should be small -- same methodology)")

    # ================= Decompose small vs large =================
    print(f"\n=== E92 Pool-vs-Transform Decomposition ===")
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
        print(f"  {stage_name:15s}: small={decod[small_mask].mean():.4f}, large={decod[large_mask].mean():.4f}")

    delta_pool_small = per_stage_strat["enc3"]["small"] - per_stage_strat["pool3_output"]["small"]
    delta_pool_large = per_stage_strat["enc3"]["large"] - per_stage_strat["pool3_output"]["large"]
    delta_transform_small = per_stage_strat["pool3_output"]["small"] - per_stage_strat["bottleneck"]["small"]
    delta_transform_large = per_stage_strat["pool3_output"]["large"] - per_stage_strat["bottleneck"]["large"]

    print(f"\nDelta_pool (enc3 -> pool3_output):      small={delta_pool_small.mean():+.4f}, "
          f"large={delta_pool_large.mean():+.4f}")
    print(f"Delta_transform (pool3_output -> bottleneck): small={delta_transform_small.mean():+.4f}, "
          f"large={delta_transform_large.mean():+.4f}")

    obs_pool_diff, p_pool_diff = permutation_test_diff(delta_pool_small, delta_pool_large, SEED)
    obs_transform_diff, p_transform_diff = permutation_test_diff(delta_transform_small, delta_transform_large, SEED + 1)

    print(f"\nDelta_pool small-vs-large difference: {obs_pool_diff:+.4f}, permutation p={p_pool_diff:.4f}")
    print(f"Delta_transform small-vs-large difference: {obs_transform_diff:+.4f}, permutation p={p_transform_diff:.4f}")

    pool_dominant = (obs_pool_diff > 0.02) and (p_pool_diff < 0.05) and not ((obs_transform_diff > 0.02) and (p_transform_diff < 0.05))
    transform_dominant = (obs_transform_diff > 0.02) and (p_transform_diff < 0.05) and not ((obs_pool_diff > 0.02) and (p_pool_diff < 0.05))
    both_elevated = (obs_pool_diff > 0.02) and (p_pool_diff < 0.05) and (obs_transform_diff > 0.02) and (p_transform_diff < 0.05)

    if pool_dominant:
        decision = "H1_POOLING_DOMINANT"
        detail = "Spatial downsampling (pool3) disproportionately destroys small-lesion information; the subsequent channel transformation does not show a comparable size-specific effect. Motivates selective-preservation-across-downsampling mechanisms, not bigger bottleneck channels."
    elif transform_dominant:
        decision = "H2_TRANSFORM_DOMINANT"
        detail = "The convolutional channel-mixing transformation disproportionately destroys small-lesion information; pooling alone does not show a comparable size-specific effect. Motivates a mechanism targeting the channel-mixing step, not pooling/resolution."
    elif both_elevated:
        decision = "H3_INTERACTION_BOTH_ELEVATED"
        detail = "Both pooling and channel transformation show significant size-specific loss -- no single operation dominates; a joint/distributed intervention would likely be needed, not a single-operation fix."
    else:
        decision = "NEITHER_CLEARLY_SIGNIFICANT"
        detail = "Neither component individually shows a clean, significant small-vs-large difference in its incremental loss -- the E91 bottleneck-level deficit may be diffuse/hard to attribute to a single operation with this decomposition. Report plainly, do not force a category."

    print(f"\n=== DECISION: {decision} ===")
    print(detail)

    summary = {
        "checkpoint_lineage": str(CKPT_PATH),
        "checkpoint_val_dice": ckpt.get("best_val_dice"),
        "per_stage_strat_means": {k: {"small_mean": v["small_mean"], "large_mean": v["large_mean"]} for k, v in per_stage_strat.items()},
        "delta_pool_small_mean": float(delta_pool_small.mean()),
        "delta_pool_large_mean": float(delta_pool_large.mean()),
        "delta_transform_small_mean": float(delta_transform_small.mean()),
        "delta_transform_large_mean": float(delta_transform_large.mean()),
        "delta_pool_diff": obs_pool_diff, "delta_pool_diff_p": p_pool_diff,
        "delta_transform_diff": obs_transform_diff, "delta_transform_diff_p": p_transform_diff,
        "decision": decision, "detail": detail,
    }
    with open(OUT_DIR / "E92_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E92_summary.json")


if __name__ == "__main__":
    main()
