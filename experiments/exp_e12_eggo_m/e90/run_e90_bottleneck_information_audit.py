"""
Phase E90: Bottleneck Information / Extractability Audit.

CONTEXT: E89 overturned E85-E88's "allocation mismatch" framing --
functional utilization U_b tracks causal necessity N_b almost perfectly
(rho up to 0.95). The bottleneck is NOT being ignored or misallocated.
Yet small lesions still fail disproportionately in Dice (E48's original
finding). User's synthesis: necessity != sufficiency. Three competing
explanations for why a correctly-utilized, causally-necessary bottleneck
still fails to produce adequate small-lesion segmentation:

  H1 (Capacity-limited): bottleneck used correctly, but its fixed
     channel/spatial capacity (256ch @ 8^3) cannot encode enough
     small-lesion information regardless of extraction quality.
  H2 (Information-limited encoder): the information needed for tiny
     lesions is already lost BEFORE the bottleneck (native MRI -> 64^3
     resize -> pooling -> 8^3), so nothing downstream can recover it.
  H3 (Decoder extraction-limited): the information IS present in the
     bottleneck and correctly utilized in proportion to necessity, but
     the decoder (dec3/dec2/dec1/seg_head) fails to extract enough of it
     into an accurate segmentation.

THIS PHASE trains linear probes on FROZEN bottleneck features (from
E48's checkpoint, no changes to the trained model) to directly measure
"how much lesion information is DECODABLE from the bottleneck alone,"
stratified by lesion size, then compares against the actual trained
decoder's own performance to compute an EXTRACTABILITY GAP.

METHODOLOGY, and its known pitfalls (explicitly guarded against):
  - Probe train/test split must be held out from what any of E48/E85-E89
    measured metrics were computed on to avoid circularity: the 125
    "validation" subjects here are split again into probe-train (75%)
    and probe-test (25%), and only probe-TEST performance is reported as
    the decodability metric. (Fitting AND evaluating a probe on the same
    subjects would silently inflate decodability -- a classic probing
    methodology error this design explicitly avoids.)
  - The probe is a SINGLE LINEAR layer (1x1x1 conv from 256 bottleneck
    channels -> 1 lesion-presence logit, upsampled to 64^3 via
    trilinear interpolation before the loss) -- deliberately minimal
    capacity, so "decodable" means "linearly decodable," a conservative
    lower bound on true information content (a stronger nonlinear probe
    could recover more; a weak linear probe finding LOW decodability is
    still informative as a florr, and finding HIGH decodability with
    just a linear probe is strong evidence the information is genuinely
    accessible, not requiring an elaborate readout to prove out).
  - Probe is trained ONLY on probe-train bottleneck features (extracted
    from the FROZEN, already-trained E48 checkpoint -- no backprop into
    the segmentation network itself, only into the new 1x1x1 probe
    layer), then evaluated on held-out probe-test subjects, stratified
    by native_size (E48's own established size metric) into small/large
    via median split -- avoiding an arbitrary absolute threshold.

METRICS PER SUBJECT (probe-test set only):
  - decodability_dice: Dice of the probe's OWN thresholded prediction
    (upsampled to 64^3) vs the ground-truth lesion mask -- directly
    comparable in scale to the real model's segmentation Dice.
  - actual_dice: the REAL trained model's own segmentation Dice (dice_
    intact, same as E48/E85-E89's convention) for the same subject.
  - extractability_gap = decodability_dice - actual_dice (the probe's
    own performance vs the full decoder's -- POSITIVE means the probe,
    despite being far simpler than the real decoder, still does at
    least as well from bottleneck features alone, suggesting the
    decoder is leaving recoverable information on the table; near-zero
    or negative means the decoder already extracts at least what a
    simple probe can, consistent with H1/H2 rather than H3).

PRE-DECLARED DECISION RULE (H1 vs H2 vs H3), evaluated SEPARATELY for
small-lesion and large-lesion strata:
  - H3 (decoder extraction-limited) supported for a stratum if:
    decodability_dice is HIGH (comparable to or exceeding large-lesion
    decodability) AND extractability_gap > 0 with permutation p < 0.05
    (the simple probe recovers real signal the actual decoder is not
    fully using).
  - H1/H2 (capacity- or information-limited, cannot be distinguished by
    this probe alone -- see Limitations) supported for a stratum if:
    decodability_dice is LOW in absolute terms AND/OR substantially
    lower than the large-lesion stratum's decodability_dice (permutation
    test on the small-vs-large decodability_dice difference, p<0.05) --
    i.e. even a probe with full access to the RAW bottleneck tensor
    cannot recover much lesion signal for small lesions specifically,
    meaning the information genuinely isn't well-represented there
    (regardless of whether it's a capacity or upstream-loss cause,
    which this specific probe design cannot separate -- flagged
    explicitly as a limitation, not glossed over).

LIMITATION (explicit): this probe design can distinguish H3 from
{H1,H2} cleanly (extraction-limited vs information/capacity-limited),
but cannot on its own separate H1 (capacity) from H2 (upstream
information loss) -- that would require probing intermediate encoder
stages (enc1/enc2/enc3) too, which is a natural follow-up if {H1,H2} is
supported here but out of scope for this specific audit.

NO CHANGES TO THE TRAINED SEGMENTATION MODEL. NO NEW TRAINING of the
main network. Only a small linear probe is trained, on frozen features.
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


class LinearProbe(nn.Module):
    """Minimal-capacity probe: single 1x1x1 conv, 256 bottleneck channels
    -> 1 logit per bottleneck spatial location (8^3), upsampled to 64^3
    before loss/evaluation. Deliberately no hidden layers, no
    nonlinearity beyond the final sigmoid -- 'linearly decodable' is
    the explicit, conservative claim being tested."""
    def __init__(self, in_channels=256):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=1)

    def forward(self, z_b):
        logits_8 = self.conv(z_b)  # (B, 1, 8, 8, 8)
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


def get_bottleneck_and_intact_probs(model, image_b, device):
    """Single forward pass, returns both the bottleneck tensor (for probe
    training) and the model's own real segmentation probs (actual_dice
    source) -- reuses one pass for both, no extra forward cost."""
    with torch.no_grad():
        out = model(image_b)
        probs = out["probs"].squeeze(0).squeeze(0).cpu().numpy()

        enc1 = model.enc1(image_b)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)  # (1, 256, 8, 8, 8)
    return bottleneck.detach(), probs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)

    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded E46/E48/E85-89 checkpoint (FROZEN): best_val_dice={ckpt.get('best_val_dice')}", flush=True)

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    print(f"Validation set size: {len(val_dataset)}", flush=True)

    # Pre-extract bottleneck features, targets, actual_dice, native_size for all subjects.
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

        bottleneck, probs_intact = get_bottleneck_and_intact_probs(model, image_b, device)
        actual_dice = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin)

        all_data.append({
            "subject_id": subject_id,
            "bottleneck": bottleneck,  # kept on device, (1,256,8,8,8)
            "target_bin": target_bin,
            "actual_dice": actual_dice,
            "native_size": e48_by_id[subject_id]["native_size"],
        })
        if len(all_data) % 25 == 0:
            print(f"  extracted {len(all_data)} subjects", flush=True)

    print(f"\nTotal subjects with valid data: {len(all_data)}", flush=True)

    # Held-out split: probe-train (75%) / probe-test (25%), stratified
    # roughly by native_size to keep both splits balanced across sizes.
    rng = np.random.default_rng(SEED)
    sorted_by_size = sorted(all_data, key=lambda d: d["native_size"])
    indices = np.arange(len(sorted_by_size))
    test_mask = (indices % 4 == 0)  # every 4th subject in size-sorted order -> balanced stratified holdout
    probe_train = [d for i, d in enumerate(sorted_by_size) if not test_mask[i]]
    probe_test = [d for i, d in enumerate(sorted_by_size) if test_mask[i]]
    print(f"Probe-train: {len(probe_train)} subjects, Probe-test: {len(probe_test)} subjects "
          f"(held out, decodability reported ONLY on this set)", flush=True)

    # ================= Train the linear probe on probe_train ONLY =================
    probe = LinearProbe(in_channels=256).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)

    print(f"\nTraining linear probe for {N_PROBE_EPOCHS} epochs on probe-train set...", flush=True)
    for epoch in range(N_PROBE_EPOCHS):
        epoch_loss = 0.0
        perm = rng.permutation(len(probe_train))
        for idx in perm:
            d = probe_train[idx]
            target_t = torch.from_numpy(d["target_bin"]).unsqueeze(0).unsqueeze(0).to(device)
            logits = probe(d["bottleneck"])
            loss = F.binary_cross_entropy_with_logits(logits, target_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}: mean BCE loss = {epoch_loss/len(probe_train):.4f}", flush=True)

    # ================= Evaluate on held-out probe_test =================
    print("\nEvaluating probe on HELD-OUT probe-test set...", flush=True)
    probe.eval()
    records = []
    with torch.no_grad():
        for d in probe_test:
            logits = probe(d["bottleneck"])
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
            decodability_dice = dice_score((probs >= 0.5).astype(np.float32), d["target_bin"])
            extractability_gap = decodability_dice - d["actual_dice"]
            records.append({
                "subject_id": d["subject_id"],
                "native_size": d["native_size"],
                "decodability_dice": decodability_dice,
                "actual_dice": d["actual_dice"],
                "extractability_gap": extractability_gap,
            })

    with open(OUT_DIR / "E90_probe_test_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} probe-test records.", flush=True)

    # ================= Stratified analysis: small vs large (median split) =================
    native_sizes = np.array([r["native_size"] for r in records])
    median_size = float(np.median(native_sizes))
    small_mask = native_sizes <= median_size
    large_mask = ~small_mask

    decod_small = np.array([r["decodability_dice"] for r in records])[small_mask]
    decod_large = np.array([r["decodability_dice"] for r in records])[large_mask]
    gap_small = np.array([r["extractability_gap"] for r in records])[small_mask]
    gap_large = np.array([r["extractability_gap"] for r in records])[large_mask]
    actual_small = np.array([r["actual_dice"] for r in records])[small_mask]
    actual_large = np.array([r["actual_dice"] for r in records])[large_mask]

    print(f"\n=== E90 Bottleneck Information Audit: n_test={len(records)} "
          f"(small={small_mask.sum()}, large={large_mask.sum()}, median_size={median_size:.0f}) ===")
    print(f"Small lesions: mean decodability_dice={decod_small.mean():.4f}, "
          f"mean actual_dice={actual_small.mean():.4f}, mean extractability_gap={gap_small.mean():+.4f}")
    print(f"Large lesions: mean decodability_dice={decod_large.mean():.4f}, "
          f"mean actual_dice={actual_large.mean():.4f}, mean extractability_gap={gap_large.mean():+.4f}")

    # Permutation test: decodability_dice small vs large
    rng2 = np.random.default_rng(SEED + 1)
    observed_decod_diff = decod_small.mean() - decod_large.mean()
    combined_decod = np.concatenate([decod_small, decod_large])
    n_small = len(decod_small)
    perm_diffs = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_idx = rng2.permutation(len(combined_decod))
        perm_small = combined_decod[perm_idx[:n_small]]
        perm_large = combined_decod[perm_idx[n_small:]]
        perm_diffs[i] = perm_small.mean() - perm_large.mean()
    p_decod_diff = float((np.abs(perm_diffs) >= np.abs(observed_decod_diff)).mean())
    print(f"\nDecodability small-vs-large difference: {observed_decod_diff:+.4f}, permutation p={p_decod_diff:.4f}")

    # Permutation test: extractability_gap > 0 for small lesions (sign-flip)
    rng3 = np.random.default_rng(SEED + 2)
    observed_gap_small = gap_small.mean()
    perm_gap_means = np.empty(N_PERM)
    for i in range(N_PERM):
        signs = rng3.choice([-1, 1], size=len(gap_small))
        perm_gap_means[i] = (gap_small * signs).mean()
    p_gap_small = float((perm_gap_means >= observed_gap_small).mean())  # one-sided: gap > 0
    print(f"Small-lesion extractability_gap > 0 test: mean={observed_gap_small:+.4f}, "
          f"one-sided permutation p={p_gap_small:.4f}")

    observed_gap_large = gap_large.mean()
    perm_gap_means_large = np.empty(N_PERM)
    rng4 = np.random.default_rng(SEED + 3)
    for i in range(N_PERM):
        signs = rng4.choice([-1, 1], size=len(gap_large))
        perm_gap_means_large[i] = (gap_large * signs).mean()
    p_gap_large = float((perm_gap_means_large >= observed_gap_large).mean())
    print(f"Large-lesion extractability_gap > 0 test: mean={observed_gap_large:+.4f}, "
          f"one-sided permutation p={p_gap_large:.4f}")

    # ================= Decision =================
    small_decodability_low = (decod_small.mean() < decod_large.mean() - 0.05) and (p_decod_diff < 0.05)
    small_gap_positive = (observed_gap_small > 0.02) and (p_gap_small < 0.05)

    if small_decodability_low:
        decision = "H1_H2_INFORMATION_OR_CAPACITY_LIMITED"
        detail = ("Decodability itself is significantly lower for small lesions -- even a simple "
                   "linear probe with full access to raw bottleneck features cannot recover much "
                   "small-lesion signal. Cannot distinguish H1 (capacity) from H2 (upstream information "
                   "loss) with this probe alone -- would need to probe enc1/enc2/enc3 as a follow-up.")
    elif small_gap_positive:
        decision = "H3_DECODER_EXTRACTION_LIMITED"
        detail = ("Decodability for small lesions is comparable to large lesions AND the simple probe "
                   "outperforms the actual trained decoder (positive extractability_gap) -- the "
                   "information IS present and reasonably decodable, but the real decoder is not "
                   "fully extracting it. Motivates an extraction-focused intervention, not more "
                   "bottleneck weight/capacity.")
    else:
        decision = "AMBIGUOUS_OR_NO_CLEAR_GAP"
        detail = "Neither condition met cleanly -- do not proceed to design an intervention from this result alone."

    print(f"\n=== DECISION: {decision} ===")
    print(detail)

    summary = {
        "n_test": len(records), "n_small": int(small_mask.sum()), "n_large": int(large_mask.sum()),
        "median_native_size": median_size,
        "small_mean_decodability_dice": float(decod_small.mean()),
        "large_mean_decodability_dice": float(decod_large.mean()),
        "small_mean_actual_dice": float(actual_small.mean()),
        "large_mean_actual_dice": float(actual_large.mean()),
        "small_mean_extractability_gap": float(gap_small.mean()),
        "large_mean_extractability_gap": float(gap_large.mean()),
        "decodability_small_vs_large_diff": float(observed_decod_diff),
        "decodability_diff_permutation_p": p_decod_diff,
        "small_gap_positive_permutation_p": p_gap_small,
        "large_gap_positive_permutation_p": p_gap_large,
        "decision": decision,
        "detail": detail,
    }
    with open(OUT_DIR / "E90_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E90_summary.json")


if __name__ == "__main__":
    main()
