"""
Phase E91: Encoder-Stage Information-Loss Localization.

CONTEXT: E90 found bottleneck decodability (linear-probe Dice) is
significantly lower for small lesions than large (0.652 vs 0.759,
p=0.019), ruling out H3 (decoder-extraction-limited) and supporting
H1 (bottleneck capacity-limited) or H2 (information already lost
upstream of the bottleneck) -- but could not distinguish H1 from H2 on
its own. This phase runs the IDENTICAL linear-probe methodology at
enc1 (32ch @ 64^3), enc2 (64ch @ 32^3), enc3 (128ch @ 16^3), and the
bottleneck (256ch @ 8^3, RE-RUN here rather than reusing E90's numbers,
so all four stages share IDENTICAL train/test split, probe capacity,
optimizer, seed, and epoch count -- avoiding any cross-experiment
methodology drift as a confound between stages), to find the FIRST
stage at which small-lesion decodability significantly drops relative
to large-lesion decodability.

CHECKPOINT LINEAGE (explicit, single, documented -- per user's
instruction and the project's own archived warning that v3/D4-only and
v5/attention-gated checkpoints are NOT interchangeable): this phase uses
ONLY experiments/exp_e12_eggo_m/e46/runs/AttnGate_seed0/checkpoints/
best.pth (UNet3D_v5, val_dice=0.9102) -- the SAME checkpoint used by
E48, E85, E86, E88, E89, and E90. No other checkpoint is loaded anywhere
in this script.

METHODOLOGY, matched exactly across all 4 stages (deliberately identical
to E90's design so cross-stage differences cannot be probe artifacts):
  - Same 125 subjects, same probe-train (75%) / probe-test (25%) split,
    stratified by native_size via the same "every 4th subject in
    size-sorted order" rule, same random seed (0).
  - Same probe architecture PATTERN: single 1x1x1 conv from the stage's
    own channel count -> 1 logit, upsampled via trilinear interpolation
    to 64^3 before loss/evaluation -- the ONLY thing that changes
    between stages is in_channels (32/64/128/256) and the probe's native
    spatial resolution (64/32/16/8), matching each stage's real shape.
  - Same optimizer (Adam, lr=1e-2), same N_PROBE_EPOCHS=200, same loss
    (BCEWithLogits), same evaluation metric (Dice on probe-test only).
  - Same decision-relevant statistic: decodability_dice, small vs large
    (median native_size split), permutation test on the difference.

PRE-DECLARED INTERPRETATION:
  - If enc1 (the shallowest, highest-resolution, first stage) ALREADY
    shows a significant small-vs-large decodability deficit comparable
    in magnitude to what E90 found at the bottleneck: H2 supported --
    information loss happens very early (plausibly at/before the 64^3
    resize itself, since enc1 operates directly on the resized input)
    and nothing later in the pipeline can be expected to recover it.
  - If enc1/enc2/enc3 show NO significant deficit (or a much smaller
    one) and the deficit only becomes significant at the bottleneck:
    H1 supported -- the bottleneck's own aggressive
    spatial/channel compression (16x16x16 -> 8x8x8 pooling into 256
    channels) is where small-lesion information specifically gets lost,
    not the earlier, higher-resolution encoder stages.
  - If the deficit GROWS PROGRESSIVELY at each stage (present but small
    at enc1, larger at enc2, larger still at enc3, largest at
    bottleneck): distributed attrition -- no single stage is uniquely
    responsible; the effect accumulates through the whole downsampling
    pathway. This would argue against a single-stage fix and toward a
    more distributed intervention (e.g. multi-scale supervision
    specifically for small lesions), a materially different design
    target from either H1 or H2 alone.
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

# SINGLE, EXPLICIT, DOCUMENTED checkpoint lineage -- v5/attention-gated,
# identical to E48/E85/E86/E88/E89/E90. Do not substitute any other
# checkpoint anywhere in this script.
CKPT_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs" / "AttnGate_seed0" / "checkpoints" / "best.pth"
E48_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"

STAGES = {
    "enc1": {"channels": 32, "resolution": 64},
    "enc2": {"channels": 64, "resolution": 32},
    "enc3": {"channels": 128, "resolution": 16},
    "bottleneck": {"channels": 256, "resolution": 8},
}


class LinearProbe(nn.Module):
    """Identical pattern to E90's probe -- single 1x1x1 conv, no hidden
    layers, upsampled to 64^3 before loss. Only in_channels varies."""
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
    """Single forward pass through the encoder trunk, returns enc1,
    enc2, enc3, and bottleneck tensors all at once -- reused across all
    4 stage-probes for a given subject, no redundant forward passes."""
    with torch.no_grad():
        enc1 = model.enc1(image_b)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)
    return {"enc1": enc1.detach(), "enc2": enc2.detach(), "enc3": enc3.detach(), "bottleneck": bottleneck.detach()}


def train_and_eval_probe(stage_name, channels, all_data, probe_train_idx, probe_test_idx, device):
    torch.manual_seed(SEED)  # reset seed per stage so each probe's init/training is independently comparable
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    print(f"Loaded checkpoint: {CKPT_PATH}")
    print(f"best_val_dice={ckpt.get('best_val_dice')} (must match E48/E85-E90's 0.9101624600589275)", flush=True)
    assert abs(ckpt.get("best_val_dice", 0) - 0.9101624600589275) < 1e-9, \
        "Checkpoint mismatch -- this MUST be the exact same E46 AttnGate_seed0 checkpoint used throughout E48-E90."

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
            print(f"  extracted {len(all_data)} subjects (all 4 stages)", flush=True)

    print(f"\nTotal subjects: {len(all_data)}", flush=True)

    # IDENTICAL split rule to E90: every 4th subject in size-sorted order.
    sorted_indices = sorted(range(len(all_data)), key=lambda i: all_data[i]["native_size"])
    test_flags = [(pos % 4 == 0) for pos in range(len(sorted_indices))]
    probe_test_idx = [sorted_indices[pos] for pos, flag in enumerate(test_flags) if flag]
    probe_train_idx = [sorted_indices[pos] for pos, flag in enumerate(test_flags) if not flag]
    print(f"Probe-train: {len(probe_train_idx)}, Probe-test: {len(probe_test_idx)} "
          f"(IDENTICAL split rule to E90)", flush=True)

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

    with open(OUT_DIR / "E91_stage_probe_table.json", "w") as f:
        json.dump(stage_results, f, indent=2)
    print(f"\nSaved per-stage probe-test tables.", flush=True)

    # ================= Stratified analysis per stage =================
    from scipy import stats as scipy_stats

    print(f"\n=== E91 Encoder-Stage Localization Summary ===")
    summary_per_stage = {}
    for stage_name, records in stage_results.items():
        native_sizes = np.array([r["native_size"] for r in records])
        median_size = float(np.median(native_sizes))
        small_mask = native_sizes <= median_size
        large_mask = ~small_mask

        decod = np.array([r["decodability_dice"] for r in records])
        decod_small = decod[small_mask]
        decod_large = decod[large_mask]

        observed_diff = decod_small.mean() - decod_large.mean()
        rng = np.random.default_rng(SEED)
        n_small = small_mask.sum()
        perm_diffs = np.empty(N_PERM)
        for i in range(N_PERM):
            perm_idx = rng.permutation(len(decod))
            perm_small = decod[perm_idx[:n_small]]
            perm_large = decod[perm_idx[n_small:]]
            perm_diffs[i] = perm_small.mean() - perm_large.mean()
        p_diff = float((np.abs(perm_diffs) >= np.abs(observed_diff)).mean())

        summary_per_stage[stage_name] = {
            "channels": STAGES[stage_name]["channels"], "resolution": STAGES[stage_name]["resolution"],
            "mean_decodability_small": float(decod_small.mean()),
            "mean_decodability_large": float(decod_large.mean()),
            "diff_small_minus_large": float(observed_diff),
            "permutation_p": p_diff,
            "significant_deficit": bool(observed_diff < -0.03 and p_diff < 0.05),
        }
        print(f"  {stage_name:12s} ({STAGES[stage_name]['channels']:3d}ch @ {STAGES[stage_name]['resolution']:2d}^3): "
              f"small={decod_small.mean():.4f}, large={decod_large.mean():.4f}, "
              f"diff={observed_diff:+.4f}, p={p_diff:.4f}, "
              f"{'DEFICIT' if summary_per_stage[stage_name]['significant_deficit'] else 'no sig. deficit'}")

    # ================= Localization decision =================
    stage_order = ["enc1", "enc2", "enc3", "bottleneck"]
    deficits = [summary_per_stage[s]["significant_deficit"] for s in stage_order]
    diffs = [summary_per_stage[s]["diff_small_minus_large"] for s in stage_order]

    if deficits[0]:
        localization = "H2_EARLY_INFORMATION_LOSS"
        detail = "enc1 (shallowest stage) already shows a significant small-lesion decodability deficit -- information loss happens very early, plausibly at/before the 64^3 resize itself."
    elif not any(deficits[:3]) and deficits[3]:
        localization = "H1_BOTTLENECK_COMPRESSION_LIMITED"
        detail = "enc1/enc2/enc3 show no significant deficit; the deficit appears ONLY at the bottleneck -- the bottleneck's own aggressive compression is where small-lesion information specifically gets lost."
    elif all(d2 <= d1 + 0.01 for d1, d2 in zip(diffs, diffs[1:])) and deficits[3] and (deficits[1] or deficits[2]):
        localization = "DISTRIBUTED_ATTRITION"
        detail = "Deficit grows progressively across stages -- no single stage is uniquely responsible; information is attritted throughout the downsampling pathway."
    else:
        localization = "MIXED_OR_UNCLEAR_PATTERN"
        detail = "Pattern does not cleanly match any of the three pre-declared outcomes -- report the raw per-stage numbers plainly rather than forcing a category."

    print(f"\n=== LOCALIZATION DECISION: {localization} ===")
    print(detail)

    summary = {
        "checkpoint_lineage": str(CKPT_PATH),
        "checkpoint_val_dice": ckpt.get("best_val_dice"),
        "per_stage": summary_per_stage,
        "localization": localization,
        "detail": detail,
    }
    with open(OUT_DIR / "E91_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E91_summary.json")


if __name__ == "__main__":
    main()
