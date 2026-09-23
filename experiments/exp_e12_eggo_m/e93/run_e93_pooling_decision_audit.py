"""
Phase E93: Pooling-Decision Audit -- What Does MaxPool3d Throw Away?

CONTEXT: E92 found MaxPool3d(2,2) at enc3->pool3 disproportionately
destroys small-lesion decodable information (Delta_pool small=0.127 vs
large=0.062, p=0.022), while the subsequent convolution does not
(p=0.121). This phase does NOT propose an intervention yet -- per
user's explicit plan, it decomposes WHY max pooling loses small-lesion
information, distinguishing three mechanisms with different
implications:

  A. Winner identity: within a lesion-touching pooling cell, does the
     max-selected voxel/channel actually correspond to lesion-relevant
     activation, or does a background/non-lesion voxel win, suppressing
     the lesion's contribution entirely?
  B. Activation margin: even when the lesion-relevant voxel loses, is
     the margin small (lesion voxel activation close to the winner) or
     large (lesion signal is weak/noisy relative to the winner)? Small
     margins suggest "almost survived, just needs slightly more
     weight"; large margins suggest the lesion signal is genuinely weak
     at this stage, not merely outcompeted by chance.
  C. Multi-channel competition / rank-recovery: does retaining
     additional ranks (top-2, top-3, top-4 of the 8 voxels in each
     2x2x2 cell, mean-pooled) recover MORE decodable small-lesion
     information than the max alone -- i.e. is the signal "there" in
     lower-ranked voxels and simply discarded by winner-take-all, or is
     it just not present in that pooling cell at all regardless of rank?

METHODOLOGY (NO TRAINING of the main network -- frozen E48-lineage
checkpoint, same as E48/E85-E92; a rank-k pooling probe reuses the same
linear-probe methodology from E90-E92):

  1. Extract enc3 (128ch @ 16^3) for all 125 subjects (same checkpoint,
     same subjects as E90-E92).
  2. Build the ground-truth lesion mask at enc3's 16^3 resolution
     (fractional occupancy, matching this project's established
     convention -- reused from E43's own build_region_masks approach).
  3. For each subject, identify the set of pool3 output cells (8x8x8
     grid) whose corresponding 2x2x2 enc3 input block has ANY lesion
     occupancy (mask_16[cell] > 0) -- these are "lesion-touching cells."
  4. Winner-identity test (A): within each lesion-touching cell, compare
     the SPATIAL LOCATION of each channel's arg-max voxel against which
     of the 8 sub-voxels has higher lesion occupancy (using the
     finer-grained native lesion mask resampled to match enc3's spatial
     sub-structure) -- estimate P(winner voxel is also a
     higher-lesion-occupancy voxel) for small vs large lesion subjects.
  5. Activation-margin test (B): compute the gap between the winning
     activation and the 2nd-highest activation within each
     lesion-touching cell, separately for small vs large lesion
     subjects, and compare distributions.
  6. Rank-recovery probe test (C): build alternative pooled features
     using mean-of-top-k (k=1 [=max], 2, 3, 4) within each 2x2x2 cell
     per channel, train a linear probe (IDENTICAL methodology to
     E90-E92: same split, same seed, same optimizer/epochs) on each
     variant, and measure whether decodability for SMALL lesions
     recovers more from higher k than large lesions do (i.e. does
     retaining more of the pooling cell disproportionately help small
     lesions specifically, which would indicate the lost signal WAS
     present in lower ranks and is recoverable).

PRE-DECLARED INTERPRETATION:
  - If P(winner correlates with lesion location) is much lower for
    small-lesion cells than large-lesion cells: mechanism A dominates --
    winner-take-all is systematically selecting AWAY from small-lesion
    voxels. Motivates a mechanism that biases winner selection toward
    lesion-relevant locations (e.g. auxiliary-signal-guided pooling),
    not just "keep more information."
  - If activation margins in lesion-touching cells are systematically
    SMALL for small lesions (lesion signal is close to winning but
    loses): supports a "soft/near-max" pooling correction -- a gentler
    intervention than full rank retention.
  - If margins are LARGE (lesion signal genuinely weak, not a close
    call): the lesion signal may be inherently faint at this stage
    regardless of pooling operator -- pooling-operator fixes alone may
    not suffice; would need to look further upstream (contradicting the
    localization from E91, an important cross-check).
  - If rank-recovery probes show SMALL-lesion decodability increasing
    much more steeply with k than large-lesion decodability: mechanism
    C confirmed -- the information IS present in lower-ranked (rank
    2-4) voxels and specifically recoverable via multi-rank retention,
    a concrete, falsifiable basis for a size-selective retention
    mechanism (per user's own proposed direction) rather than a generic
    soft-pooling replacement.
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


class LinearProbe(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=1)

    def forward(self, z):
        logits_native = self.conv(z)
        logits_64 = F.interpolate(logits_native, size=(64, 64, 64), mode="trilinear", align_corners=False)
        return logits_64


def fractional_occupancy(seg_binary_native, size):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=(size, size, size), mode="area").squeeze().numpy()
    return frac


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    if denom == 0:
        return 1.0
    return float(2 * tp / denom)


def topk_mean_pool(enc3, k):
    """Non-overlapping 2x2x2 mean-of-top-k pooling, generalizing MaxPool3d
    (k=1 is exactly equivalent to MaxPool3d). enc3: (1, C, 16, 16, 16) ->
    (1, C, 8, 8, 8)."""
    B, C, D, H, W = enc3.shape
    # Unfold into (B, C, 8, 8, 8, 8) where last dim indexes the 8 voxels in each 2x2x2 cell.
    x = enc3.unfold(2, 2, 2).unfold(3, 2, 2).unfold(4, 2, 2)  # (B,C,8,8,8,2,2,2)
    x = x.contiguous().view(B, C, 8, 8, 8, 8)  # flatten the 2x2x2 window to 8
    topk_vals, _ = torch.topk(x, k=k, dim=-1)
    return topk_vals.mean(dim=-1)  # (B, C, 8, 8, 8)


def train_and_eval_probe(features_by_subject, target_by_subject, probe_train_ids, probe_test_ids,
                          channels, device):
    torch.manual_seed(SEED)
    probe = LinearProbe(in_channels=channels).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    rng = np.random.default_rng(SEED)

    for epoch in range(N_PROBE_EPOCHS):
        perm = rng.permutation(len(probe_train_ids))
        for idx in perm:
            sid = probe_train_ids[idx]
            target_t = torch.from_numpy(target_by_subject[sid]).unsqueeze(0).unsqueeze(0).to(device)
            logits = probe(features_by_subject[sid])
            loss = F.binary_cross_entropy_with_logits(logits, target_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    results = {}
    with torch.no_grad():
        for sid in probe_test_ids:
            logits = probe(features_by_subject[sid])
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
            results[sid] = dice_score((probs >= 0.5).astype(np.float32), target_by_subject[sid])
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    assert abs(ckpt.get("best_val_dice", 0) - 0.9101624600589275) < 1e-9, \
        "Checkpoint mismatch -- must match E48-E92's exact checkpoint."
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
    mask16_by_subject = {}
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

        mask_16 = fractional_occupancy(seg_binary_native, 16)  # lesion occupancy at enc3's own resolution
        target_64 = (fractional_occupancy(seg_binary_native, 64) > 0.5).astype(np.float32)

        enc3_by_subject[subject_id] = enc3.detach()
        mask16_by_subject[subject_id] = mask_16
        target64_by_subject[subject_id] = target_64
        native_size_by_subject[subject_id] = e48_by_id[subject_id]["native_size"]

        if len(enc3_by_subject) % 25 == 0:
            print(f"  extracted {len(enc3_by_subject)} subjects", flush=True)

    subject_ids = list(enc3_by_subject.keys())
    print(f"\nTotal subjects: {len(subject_ids)}", flush=True)

    native_sizes = np.array([native_size_by_subject[sid] for sid in subject_ids])
    median_size = float(np.median(native_sizes))
    small_ids = [sid for sid in subject_ids if native_size_by_subject[sid] <= median_size]
    large_ids = [sid for sid in subject_ids if native_size_by_subject[sid] > median_size]
    print(f"Small: {len(small_ids)} subjects, Large: {len(large_ids)} subjects (median={median_size:.0f})", flush=True)

    # ================= Test A + B: winner identity & activation margin =================
    print("\n=== Test A/B: winner identity and activation margin in lesion-touching cells ===")

    def analyze_cells(subject_ids_subset):
        winner_matches, margins = [], []
        for sid in subject_ids_subset:
            enc3 = enc3_by_subject[sid]  # (1, 128, 16, 16, 16)
            mask_16 = mask16_by_subject[sid]  # (16,16,16)

            # Reshape enc3 into (C, 8,8,8, 8) cells (2x2x2 flattened to 8).
            x = enc3.unfold(2, 2, 2).unfold(3, 2, 2).unfold(4, 2, 2).contiguous()  # (1,C,8,8,8,2,2,2)
            x = x.view(1, 128, 8, 8, 8, 8).squeeze(0).cpu().numpy()  # (128, 8,8,8, 8)

            mask_cells = mask_16.reshape(8, 2, 8, 2, 8, 2).transpose(0, 2, 4, 1, 3, 5).reshape(8, 8, 8, 8)
            # mask_cells: (8,8,8, 8) -- lesion occupancy of each of the 8 sub-voxels per cell

            lesion_touching = mask_cells.max(axis=-1) > 0  # (8,8,8) bool, cells with any lesion occupancy

            for i in range(8):
                for j in range(8):
                    for k in range(8):
                        if not lesion_touching[i, j, k]:
                            continue
                        cell_mask = mask_cells[i, j, k]  # (8,) lesion occupancy per sub-voxel
                        best_lesion_subvoxel = int(np.argmax(cell_mask))

                        # Average across channels: does the per-channel argmax
                        # tend to land on the sub-voxel with highest lesion occupancy?
                        cell_acts = x[:, i, j, k, :]  # (128, 8)
                        winner_subvoxels = np.argmax(cell_acts, axis=-1)  # (128,)
                        match_frac = float((winner_subvoxels == best_lesion_subvoxel).mean())
                        winner_matches.append(match_frac)

                        sorted_acts = np.sort(cell_acts, axis=-1)[:, ::-1]  # descending, (128, 8)
                        margin = (sorted_acts[:, 0] - sorted_acts[:, 1]).mean()
                        margins.append(float(margin))
        return np.array(winner_matches), np.array(margins)

    small_matches, small_margins = analyze_cells(small_ids)
    large_matches, large_margins = analyze_cells(large_ids)

    print(f"Small lesions: n_cells={len(small_matches)}, "
          f"mean P(winner matches best-lesion-subvoxel)={small_matches.mean():.4f}, "
          f"mean margin={small_margins.mean():.4f}")
    print(f"Large lesions: n_cells={len(large_matches)}, "
          f"mean P(winner matches best-lesion-subvoxel)={large_matches.mean():.4f}, "
          f"mean margin={large_margins.mean():.4f}")

    from scipy import stats as scipy_stats
    u_stat_match, p_match = scipy_stats.mannwhitneyu(small_matches, large_matches, alternative="two-sided")
    u_stat_margin, p_margin = scipy_stats.mannwhitneyu(small_margins, large_margins, alternative="two-sided")
    print(f"Mann-Whitney U test, winner-match small vs large: p={p_match:.4e}")
    print(f"Mann-Whitney U test, margin small vs large: p={p_margin:.4e}")

    # ================= Test C: rank-recovery probe =================
    print("\n=== Test C: rank-recovery (top-k mean pooling) probe ===")
    sorted_indices = sorted(subject_ids, key=lambda sid: native_size_by_subject[sid])
    test_flags = [(pos % 4 == 0) for pos in range(len(sorted_indices))]
    probe_test_ids = [sorted_indices[pos] for pos, flag in enumerate(test_flags) if flag]
    probe_train_ids = [sorted_indices[pos] for pos, flag in enumerate(test_flags) if not flag]

    rank_recovery_results = {}
    for k in [1, 2, 3, 4]:
        print(f"\n  Training probe on top-{k}-mean pooling...", flush=True)
        features_by_subject = {sid: topk_mean_pool(enc3_by_subject[sid], k) for sid in subject_ids}
        results = train_and_eval_probe(
            features_by_subject, target64_by_subject, probe_train_ids, probe_test_ids, channels=128, device=device
        )
        small_decod = np.array([results[sid] for sid in probe_test_ids if native_size_by_subject[sid] <= median_size])
        large_decod = np.array([results[sid] for sid in probe_test_ids if native_size_by_subject[sid] > median_size])
        rank_recovery_results[k] = {
            "small_mean": float(small_decod.mean()), "large_mean": float(large_decod.mean()),
        }
        print(f"  top-{k}: small_decodability={small_decod.mean():.4f}, large_decodability={large_decod.mean():.4f}")

    small_gain_k1_to_k4 = rank_recovery_results[4]["small_mean"] - rank_recovery_results[1]["small_mean"]
    large_gain_k1_to_k4 = rank_recovery_results[4]["large_mean"] - rank_recovery_results[1]["large_mean"]
    print(f"\nGain from k=1 (max) to k=4 (mean-of-top-4): small={small_gain_k1_to_k4:+.4f}, "
          f"large={large_gain_k1_to_k4:+.4f}")

    # ================= Decision =================
    mechanism_A = p_match < 0.05 and small_matches.mean() < large_matches.mean() - 0.02
    mechanism_B_small_margin = p_margin < 0.05 and small_margins.mean() < large_margins.mean()
    mechanism_C = (small_gain_k1_to_k4 > large_gain_k1_to_k4 + 0.02)

    print("\n=== DECISION ===")
    print(f"Mechanism A (winner-identity bias against small lesions): "
          f"{'SUPPORTED' if mechanism_A else 'not supported'} (p={p_match:.4e})")
    print(f"Mechanism B (small-lesion activation margins are smaller, i.e. close calls): "
          f"{'SUPPORTED' if mechanism_B_small_margin else 'not supported'} (p={p_margin:.4e})")
    print(f"Mechanism C (rank-recovery disproportionately helps small lesions): "
          f"{'SUPPORTED' if mechanism_C else 'not supported'} "
          f"(small_gain={small_gain_k1_to_k4:+.4f} vs large_gain={large_gain_k1_to_k4:+.4f})")

    summary = {
        "checkpoint_val_dice": ckpt.get("best_val_dice"),
        "n_small_cells": len(small_matches), "n_large_cells": len(large_matches),
        "mean_winner_match_small": float(small_matches.mean()), "mean_winner_match_large": float(large_matches.mean()),
        "p_winner_match": float(p_match),
        "mean_margin_small": float(small_margins.mean()), "mean_margin_large": float(large_margins.mean()),
        "p_margin": float(p_margin),
        "rank_recovery": rank_recovery_results,
        "small_gain_k1_to_k4": float(small_gain_k1_to_k4), "large_gain_k1_to_k4": float(large_gain_k1_to_k4),
        "mechanism_A_supported": bool(mechanism_A),
        "mechanism_B_supported": bool(mechanism_B_small_margin),
        "mechanism_C_supported": bool(mechanism_C),
    }
    with open(OUT_DIR / "E93_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E93_summary.json")


if __name__ == "__main__":
    main()
