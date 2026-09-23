"""
Phase E94: Neighborhood-Retention Audit (final diagnostic in the E92->E93->E94 chain).

CONTEXT: E92 localized small-lesion information loss to MaxPool3d(2,2)
at enc3->bottleneck specifically (not the subsequent convolution). E93
tested 3 within-cell mechanisms for WHY -- winner-identity bias (NOT
supported) and rank-recovery via mean-of-top-k within the SAME 2x2x2
cell (NOT supported) both failed; the lost information is not simply
sitting in the other 7 voxels of the same pooling cell. Per user's
correction, the defensible claim after E92-E93 is: "the enc3->bottleneck
MaxPool STAGE is causally associated with disproportionate small-lesion
information loss" -- NOT yet "the max operator itself, independent of
the 16^3->8^3 spatial-scale transition, is uniquely responsible."

THIS PHASE (final diagnostic before any decision to build an
intervention, per user's explicit one-shot stopping rule) asks whether
the missing small-lesion information exists in the SURROUNDING SPATIAL
NEIGHBORHOOD of enc3 (16^3, 128ch) -- i.e. is it destroyed by the local
2x2x2 aggregation specifically, or does the information genuinely not
exist locally anywhere nearby either?

METHODOLOGY: linear probes reading from progressively LARGER
neighborhoods of enc3, centered on each of the 8x8x8 output locations
(matching the pool3/bottleneck spatial grid for direct comparability to
E90-E93's numbers):
  - r=1 (kernel 2, stride 2): EXACTLY the original 2x2x2 pooling cell,
    fed through a linear probe -- this should closely reproduce E92's
    pool3_output decodability numbers as an internal consistency check.
  - r=2 (kernel 4, stride 2, padding 1): a 4x4x4 neighborhood centered
    on each cell, i.e. includes the original cell PLUS its immediate
    neighbors on each side.
  - r=3 (kernel 6, stride 2, padding 2): a 6x6x6 neighborhood.
  - r=4 (kernel 8, stride 2, padding 3): an 8x8x8 neighborhood -- half
    of the entire enc3 volume's extent in each dimension, a substantial
    fraction of the whole feature map.
All are SINGLE LINEAR layers (Conv3d with the given kernel/stride/
padding, no nonlinearity) -- "linearly decodable from a radius-r
neighborhood" is the controlled, comparable quantity across radii,
matching E90-E93's "linear probe" convention exactly. Larger kernels
naturally have more parameters (more input elements per output
location), which is an expected and appropriate confound of "testing a
larger receptive field," not a design flaw -- the quantity of interest
is INCREMENTAL decodability gain as radius grows, not an unconfounded
parameter-matched comparison.

Same 125 subjects, same probe-train/probe-test split (identical rule to
E90-E93), same seed, same optimizer, same epoch count, same small-vs-
large stratification (median native_size split). SAME single checkpoint
lineage as E48-E93 (asserted, not assumed).

PRE-DECLARED INTERPRETATION (user's 3-way outcome table):
  - LOCAL_IRRECOVERABLE: D(r=1) ~= D(r=2) ~= D(r=3) ~= D(r=4) for small
    lesions (no meaningful incremental gain with radius) -- information
    is already gone locally; STOP this branch entirely per the
    pre-declared one-shot rule. Report E48-E93 as a complete, honest
    causal-localization finding without a derived intervention.
  - SPATIALLY_DISTRIBUTED: decodability for small lesions increases
    monotonically and substantially as r grows (D(r=1) < D(r=2) <
    D(r=3) < D(r=4)), AND this growth is LARGER for small lesions than
    for large lesions (small lesions have MORE to gain from wider
    context, consistent with their information being "smeared" across
    a neighborhood that local 2x2x2 pooling artificially truncates) --
    this is the one outcome that motivates a genuinely new
    intervention: preserving/aggregating CROSS-CELL evidence before
    the pooling boundary is imposed, not merely changing the
    within-cell pooling statistic (which E93 already ruled out as
    sufficient).
  - BROAD_CONTEXT_DEPENDENT: decodability only jumps substantially at
    the LARGEST radius tested (r=4), with little gain at r=2/r=3 -- the
    problem is closer to a general context-aggregation/receptive-field
    limitation than a specific pooling-boundary artifact; a different
    class of intervention (e.g. larger effective receptive field
    generally) would be implied, distinct from a pooling-boundary fix.
  Decision uses permutation tests on the small-vs-large INCREMENTAL gain
  from r=1 to r=4 (the largest tested radius), matching E93's own gain-
  based decision criterion for consistency across this diagnostic chain.
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

CKPT_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs" / "AttnGate_seed0" / "checkpoints" / "best.pth"
E48_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"

# (radius_label, kernel_size, padding, learning_rate) -- stride is always
# 2, matching the original pool3 stride, so all radii produce the same
# 8x8x8 output grid, only the RECEPTIVE FIELD feeding each output
# location grows. LEARNING RATE IS TUNED PER KERNEL SIZE: a fixed
# lr=1e-2 (fine for kernel=2, 8 params/channel) was directly diagnosed
# as UNSTABLE for larger kernels (kernel=6: loss oscillated 0.07-0.20
# across 200 epochs, never converged -- confirmed by direct loss-curve
# inspection). lr=3e-4 was confirmed to converge smoothly and to a
# substantially lower final loss for kernel=6 (0.008 range, stable) --
# applying a similar reduction scaled to each kernel's larger parameter
# count (more input elements per output location = smaller stable step
# size, a standard and expected pattern, not an ad-hoc fix).
RADII = [
    ("r1_original_cell", 2, 0, 3e-3),
    ("r2_neighborhood4", 4, 1, 1e-3),
    ("r3_neighborhood6", 6, 2, 3e-4),
    ("r4_neighborhood8", 8, 3, 1e-4),
]


class NeighborhoodLinearProbe(nn.Module):
    """Single linear (Conv3d, no nonlinearity) layer reading a
    kernel_size^3 neighborhood of enc3 at stride 2, producing one logit
    per output location, upsampled to 64^3 before loss/eval -- same
    'linear probe' convention as E90-E93, generalized to variable
    receptive field."""
    def __init__(self, in_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=kernel_size, stride=2, padding=padding)

    def forward(self, enc3):
        logits_8 = self.conv(enc3)  # (B, 1, 8, 8, 8) for all radii by construction
        logits_64 = F.interpolate(logits_8, size=(64, 64, 64), mode="trilinear", align_corners=False)
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


def train_and_eval_probe(kernel_size, padding, lr, enc3_by_subject, target_by_subject,
                          probe_train_ids, probe_test_ids, device, radius_label):
    torch.manual_seed(SEED)
    probe = NeighborhoodLinearProbe(in_channels=128, kernel_size=kernel_size, padding=padding).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    rng = np.random.default_rng(SEED)

    # Verify output shape matches (8,8,8) exactly, given the padding formula:
    # out = floor((16 + 2*padding - kernel_size) / 2) + 1, checked once here.
    with torch.no_grad():
        test_out = probe.conv(enc3_by_subject[probe_train_ids[0]])
        assert test_out.shape[-3:] == (8, 8, 8), \
            f"kernel={kernel_size}, padding={padding} produced shape {test_out.shape[-3:]}, expected (8,8,8)"

    epoch_losses = []
    for epoch in range(N_PROBE_EPOCHS):
        perm = rng.permutation(len(probe_train_ids))
        epoch_loss = 0.0
        for idx in perm:
            sid = probe_train_ids[idx]
            target_t = torch.from_numpy(target_by_subject[sid]).unsqueeze(0).unsqueeze(0).to(device)
            logits = probe(enc3_by_subject[sid])
            loss = F.binary_cross_entropy_with_logits(logits, target_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_losses.append(epoch_loss / len(probe_train_ids))

    # Explicit convergence check: last 20% of epochs should be low-variance
    # relative to their mean (coefficient of variation), not oscillating --
    # this is the exact failure mode diagnosed for the unfixed r3 probe.
    tail = epoch_losses[-40:]
    tail_mean = float(np.mean(tail))
    tail_std = float(np.std(tail))
    tail_cv = tail_std / (tail_mean + 1e-9)
    converged = tail_cv < 0.25  # threshold chosen from the diagnosed contrast: unstable run had cv far above this, stable runs well below
    print(f"  [{radius_label}] final-tail loss: mean={tail_mean:.4f}, std={tail_std:.4f}, cv={tail_cv:.4f} "
          f"-> {'CONVERGED' if converged else 'NOT CONVERGED -- RESULT UNRELIABLE'}", flush=True)
    if not converged:
        print(f"  WARNING: [{radius_label}] probe did not converge cleanly -- treat its decodability number with caution.", flush=True)

    probe.eval()
    results = {}
    with torch.no_grad():
        for sid in probe_test_ids:
            logits = probe(enc3_by_subject[sid])
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
            results[sid] = dice_score((probs >= 0.5).astype(np.float32), target_by_subject[sid])
    return results, converged, tail_cv


def permutation_test_diff(a, b, seed):
    observed = a.mean() - b.mean()
    combined = np.concatenate([a, b])
    n_a = len(a)
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
    assert abs(ckpt.get("best_val_dice", 0) - 0.9101624600589275) < 1e-9, \
        "Checkpoint mismatch -- must match E48-E93's exact checkpoint."
    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded checkpoint, val_dice={ckpt.get('best_val_dice')}", flush=True)

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    print(f"Validation set size: {len(val_dataset)}", flush=True)

    enc3_by_subject = {}
    target64_by_subject = {}
    native_size_by_subject = {}

    for subject_idx in range(len(val_dataset)):
        image, mask, subject_id = val_dataset[subject_idx]
        if subject_id not in e48_by_id:
            continue
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_dataset.subject_dirs[subject_idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)

        with torch.no_grad():
            enc1 = model.enc1(image_b)
            pool1 = model.pool1(enc1)
            enc2 = model.enc2(pool1)
            pool2 = model.pool2(enc2)
            enc3 = model.enc3(pool2)  # (1, 128, 16, 16, 16)

        target_64 = (fractional_occupancy_64(seg_binary_native) > 0.5).astype(np.float32)

        enc3_by_subject[subject_id] = enc3.detach()
        target64_by_subject[subject_id] = target_64
        native_size_by_subject[subject_id] = e48_by_id[subject_id]["native_size"]

        if len(enc3_by_subject) % 25 == 0:
            print(f"  extracted {len(enc3_by_subject)} subjects", flush=True)

    subject_ids = list(enc3_by_subject.keys())
    print(f"\nTotal subjects: {len(subject_ids)}", flush=True)

    # IDENTICAL split rule to E90-E93.
    sorted_indices = sorted(subject_ids, key=lambda sid: native_size_by_subject[sid])
    test_flags = [(pos % 4 == 0) for pos in range(len(sorted_indices))]
    probe_test_ids = [sorted_indices[pos] for pos, flag in enumerate(test_flags) if flag]
    probe_train_ids = [sorted_indices[pos] for pos, flag in enumerate(test_flags) if not flag]
    print(f"Probe-train: {len(probe_train_ids)}, Probe-test: {len(probe_test_ids)} "
          f"(IDENTICAL split rule to E90-E93)", flush=True)

    median_size = float(np.median([native_size_by_subject[sid] for sid in probe_test_ids]))

    radius_results = {}
    any_not_converged = False
    for radius_label, kernel_size, padding, lr in RADII:
        print(f"\n=== Training probe: {radius_label} (kernel={kernel_size}, padding={padding}, lr={lr}) ===", flush=True)
        results, converged, tail_cv = train_and_eval_probe(
            kernel_size, padding, lr, enc3_by_subject, target64_by_subject,
            probe_train_ids, probe_test_ids, device, radius_label
        )
        if not converged:
            any_not_converged = True
        small_decod = np.array([results[sid] for sid in probe_test_ids if native_size_by_subject[sid] <= median_size])
        large_decod = np.array([results[sid] for sid in probe_test_ids if native_size_by_subject[sid] > median_size])
        radius_results[radius_label] = {
            "small_decodability": small_decod, "large_decodability": large_decod,
            "small_mean": float(small_decod.mean()), "large_mean": float(large_decod.mean()),
            "converged": converged, "tail_cv": tail_cv,
        }
        print(f"  {radius_label}: small={small_decod.mean():.4f}, large={large_decod.mean():.4f}", flush=True)

    # Cross-check r1 against E92's pool3_output decodability (should be close, same effective cell).
    e92_summary_path = project_root / "experiments" / "exp_e12_eggo_m" / "e92" / "E92_summary.json"
    if e92_summary_path.exists():
        with open(e92_summary_path) as f:
            e92_summary = json.load(f)
        e92_small = e92_summary["per_stage_strat_means"]["pool3_output"]["small_mean"]
        e92_large = e92_summary["per_stage_strat_means"]["pool3_output"]["large_mean"]
        print(f"\nCross-check r1 vs E92 pool3_output: "
              f"r1 small={radius_results['r1_original_cell']['small_mean']:.4f} vs E92={e92_small:.4f}, "
              f"r1 large={radius_results['r1_original_cell']['large_mean']:.4f} vs E92={e92_large:.4f} "
              f"(should be reasonably close -- same receptive field, different probe init/training run)")

    # ================= Incremental gain analysis =================
    print(f"\n=== E94 Incremental Decodability Gain (r1 -> r4) ===")
    for radius_label, _, _, _ in RADII:
        r = radius_results[radius_label]
        print(f"  {radius_label}: small={r['small_mean']:.4f}, large={r['large_mean']:.4f}")

    small_gain = radius_results["r4_neighborhood8"]["small_decodability"] - radius_results["r1_original_cell"]["small_decodability"]
    large_gain = radius_results["r4_neighborhood8"]["large_decodability"] - radius_results["r1_original_cell"]["large_decodability"]

    print(f"\nIncremental gain (r4 - r1): small={small_gain.mean():+.4f}, large={large_gain.mean():+.4f}")
    obs_diff, p_diff = permutation_test_diff(small_gain, large_gain, SEED)
    print(f"Small-vs-large gain difference: {obs_diff:+.4f}, permutation p={p_diff:.4f}")

    small_means = [radius_results[label]["small_mean"] for label, _, _, _ in RADII]
    monotonic_small = all(b >= a - 0.01 for a, b in zip(small_means, small_means[1:]))
    substantial_small_gain = small_gain.mean() > 0.03

    # Check where the jump happens: early (r1->r2) vs late (r3->r4)
    early_gain_small = radius_results["r2_neighborhood4"]["small_mean"] - radius_results["r1_original_cell"]["small_mean"]
    late_gain_small = radius_results["r4_neighborhood8"]["small_mean"] - radius_results["r3_neighborhood6"]["small_mean"]

    print(f"\nEarly gain (r1->r2) for small: {early_gain_small:+.4f}")
    print(f"Late gain (r3->r4) for small: {late_gain_small:+.4f}")

    # ================= Decision =================
    if any_not_converged:
        decision = "UNRELIABLE_CONVERGENCE_FAILURE"
        detail = ("At least one radius's probe did not pass the convergence check (final-tail loss "
                   "coefficient of variation >= 0.25). DO NOT trust the resulting decision -- fix "
                   "optimization for that radius and re-run before drawing any conclusion.")
        print(f"\n=== DECISION: {decision} ===")
        print(detail)
        summary = {
            "checkpoint_val_dice": ckpt.get("best_val_dice"),
            "per_radius": {label: {"small_mean": r["small_mean"], "large_mean": r["large_mean"],
                                    "converged": r["converged"], "tail_cv": r["tail_cv"]}
                           for label, r in radius_results.items()},
            "decision": decision, "detail": detail,
        }
        with open(OUT_DIR / "E94_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\nSaved E94_summary.json (UNRELIABLE)")
        return

    if not substantial_small_gain or p_diff >= 0.05:
        decision = "LOCAL_IRRECOVERABLE"
        detail = ("No substantial, significantly disproportionate gain for small lesions as radius grows. "
                   "Information is not simply sitting in the wider spatial neighborhood either. "
                   "Per pre-declared one-shot stopping rule: STOP this branch. Report E48-E94 as a complete "
                   "causal-localization finding (bottleneck -> pooling -> local cell -> neighborhood, all "
                   "tested) without a derived intervention.")
    elif monotonic_small and substantial_small_gain and p_diff < 0.05 and abs(early_gain_small) > abs(late_gain_small):
        decision = "SPATIALLY_DISTRIBUTED"
        detail = ("Decodability grows substantially and disproportionately for small lesions as radius "
                   "increases, with most of the gain at SMALLER radii (nearby cells) -- information exists "
                   "in immediate spatial context but is destroyed by local pooling. Motivates a mechanism "
                   "that aggregates cross-cell evidence BEFORE the pooling boundary, not a within-cell "
                   "pooling-statistic change (which E93 already ruled out as sufficient).")
    elif substantial_small_gain and p_diff < 0.05:
        decision = "BROAD_CONTEXT_DEPENDENT"
        detail = ("Decodability only jumps substantially at the LARGEST tested radius -- the problem is "
                   "closer to a general context-aggregation/receptive-field limitation than a specific "
                   "pooling-boundary artifact. A different class of intervention (larger effective receptive "
                   "field generally) would be implied.")
    else:
        decision = "MIXED_UNCLEAR"
        detail = "Pattern does not cleanly match a pre-declared outcome -- report raw numbers plainly."

    print(f"\n=== DECISION: {decision} ===")
    print(detail)

    summary = {
        "checkpoint_val_dice": ckpt.get("best_val_dice"),
        "per_radius": {label: {"small_mean": r["small_mean"], "large_mean": r["large_mean"]}
                       for label, r in radius_results.items()},
        "small_gain_r1_to_r4": float(small_gain.mean()),
        "large_gain_r1_to_r4": float(large_gain.mean()),
        "gain_diff_small_vs_large": obs_diff,
        "gain_diff_p": p_diff,
        "early_gain_small_r1_r2": float(early_gain_small),
        "late_gain_small_r3_r4": float(late_gain_small),
        "decision": decision,
        "detail": detail,
        "per_radius_convergence": {label: {"converged": bool(r["converged"]), "tail_cv": float(r["tail_cv"])}
                                     for label, r in radius_results.items()},
    }
    with open(OUT_DIR / "E94_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E94_summary.json")


if __name__ == "__main__":
    main()
