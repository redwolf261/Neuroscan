"""
Phase E95: Zero-Training Diagnostic -- Does Causal Necessity Predict
Context Benefit?

CONTEXT: E48-E94 branch formally closed (real causal finding: small
lesions depend more on the bottleneck; no successful intervention
found -- see phase_e94_neighborhood_audit_inconclusive memory). New
candidate mechanism per user's redirected search: "causal bottleneck
necessity -> sample-specific context curriculum," motivated by PGPS
(Progressive Growing of Patch Size, arXiv 2510.23241 -- real, published,
+1.26% Dice on 15 datasets, NOT novel to claim but a validated existing
baseline). Novelty audit (see phase_e95_causal_curriculum_novelty_audit
memory) SURVIVED: the specific combination of causal ablation-measured
necessity driving a per-sample context-curriculum schedule was not
found in PGPS (fixed global schedule), APS (correlational
uncertainty/error, not causal), or causal-training literature
(constrains what's learned, not sampling schedule).

THIS PHASE is the mandatory gate before any training: does causal
bottleneck necessity N_b(x) (E48-style, already measured for 125
subjects) have ANY relationship to how much a subject's prediction
benefits from additional spatial context? If this relationship is
ABSENT, the entire causal-curriculum idea is killed per the user's
explicit pre-declared stopping rule -- no training is justified.

METHODOLOGY (NO TRAINING -- zero-training diagnostic on the existing
E48-lineage checkpoint):

Since PGPS's actual training-time mechanism (growing patch size across
epochs) cannot be tested without training a new model, this diagnostic
uses a zero-training PROXY for "context benefit" that is measurable on
a FIXED, already-trained checkpoint: compare the model's prediction
using its normal FULL 64^3 input (maximal available context, matching
what the model was actually trained on) against its prediction using a
DELIBERATELY CONTEXT-REDUCED input -- a tight crop centered on the
lesion, zoomed back up to 64^3 to match the network's expected input
size, removing surrounding anatomical context while preserving the
lesion's own local appearance at a comparable apparent scale.

  Delta_D_i(context) = Dice(full-context input) - Dice(context-reduced input)

A subject for whom Delta_D_i is LARGE is one whose prediction benefits
substantially from having surrounding context available -- i.e. this
model, at this checkpoint, NEEDS context for that subject. This is
DIFFERENT from testing whether MORE training with growing context would
help (that requires actual training, deferred to Step 4 only if this
gate passes) -- it is a legitimate, honest, zero-training proxy for
"does this subject's causal profile relate to context-sensitivity AT
ALL," which is precisely what the gate requires before spending any
compute on a training experiment.

CROP DESIGN: for each subject, extract a bounding box tightly containing
the lesion (from the ground-truth mask, at NATIVE resolution) padded by
a small fixed margin (25% of the lesion's own bounding-box extent per
axis, a modest fixed padding rather than a lesion-size-CONFOUNDED
absolute crop size, to avoid trivially reproducing E48's own
native_size correlation through the crop mechanism itself), then resize
that native-resolution crop to 64^3 (same resize convention as the
model's normal input pipeline) and run it through the SAME frozen
checkpoint used throughout E48-E94.

PRE-DECLARED DECISION RULE:
  HYPOTHESIS SUPPORTED (proceed to Step 4, build the minimal
  intervention) if:
    Spearman rho(N_b, Delta_D_context) > 0.3, AND permutation p < 0.05
    -- i.e. subjects with higher causal bottleneck necessity ALSO
    benefit more from additional spatial context on this fixed
    checkpoint, a non-trivial, testable, causally-motivated
    relationship.
  HYPOTHESIS KILLED (abandon the causal-curriculum idea entirely, per
  user's explicit stop condition) if the correlation is weak/null
  (|rho| < 0.3 or p >= 0.05).
  Also report: is Delta_D_context itself confounded by native_size
  directly (i.e. does this just reproduce "small lesions do worse"
  trivially)? Compute rho(native_size, Delta_D_context) separately and
  report the PARTIAL correlation of N_b vs Delta_D_context controlling
  for native_size, to check whether N_b carries information BEYOND
  size alone -- important since N_b and native_size are themselves
  correlated (E48's own rho=-0.454).

Checkpoint lineage: SAME single checkpoint as E48-E94 (asserted
explicitly).
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
CROP_PADDING_FRAC = 0.25  # fixed fraction of lesion bbox extent, per axis

CKPT_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs" / "AttnGate_seed0" / "checkpoints" / "best.pth"
E48_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    if denom == 0:
        return 1.0
    return float(2 * tp / denom)


def resize_volume_torch(volume_np, target_shape, mode):
    t = torch.from_numpy(volume_np).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(t, size=target_shape, mode=mode,
                             align_corners=False if mode != "nearest" else None)
    return resized.squeeze(0).squeeze(0).numpy()


def get_lesion_bbox_with_padding(seg_binary_native, padding_frac):
    """Native-resolution bounding box of the lesion, padded by a FIXED
    FRACTION of its own extent per axis (not an absolute voxel count --
    avoids trivially confounding this crop's information content with
    native_size in a mechanical way; a small lesion still gets a small
    absolute crop, a large lesion a large one, proportionally)."""
    coords = np.argwhere(seg_binary_native > 0)
    if len(coords) == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    extents = maxs - mins
    pad = np.maximum((extents * padding_frac).astype(int), 2)  # at least 2 voxels padding
    padded_mins = np.maximum(mins - pad, 0)
    padded_maxs = np.minimum(maxs + pad, seg_binary_native.shape)
    return padded_mins, padded_maxs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    assert abs(ckpt.get("best_val_dice", 0) - 0.9101624600589275) < 1e-9, \
        "Checkpoint mismatch -- must match E48-E94's exact checkpoint."
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

    records = []
    skipped_no_lesion = 0
    for subject_idx in range(len(val_dataset)):
        image, mask, subject_id = val_dataset[subject_idx]
        if subject_id not in e48_by_id:
            continue

        subject_dir = val_dataset.subject_dirs[subject_idx]
        flair_path = Path(subject_dir) / f"{subject_id}-t2f.nii.gz"
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        flair_data = nib.load(str(flair_path)).get_fdata().astype(np.float32)
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)

        bbox = get_lesion_bbox_with_padding(seg_binary_native, CROP_PADDING_FRAC)
        if bbox is None:
            skipped_no_lesion += 1
            continue
        mins, maxs = bbox

        # ---------------- Full-context input (standard pipeline) ----------------
        flair_min, flair_max = flair_data.min(), flair_data.max()
        flair_norm = (flair_data - flair_min) / (flair_max - flair_min) if flair_max > flair_min else np.zeros_like(flair_data)
        full_image_64 = resize_volume_torch(flair_norm, (64, 64, 64), mode="trilinear")
        full_target_64 = resize_volume_torch(seg_binary_native, (64, 64, 64), mode="nearest")
        full_target_bin = (full_target_64 > 0.5).astype(np.float32)

        # ---------------- Context-reduced input (tight lesion crop) ----------------
        crop_flair = flair_data[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
        crop_seg = seg_binary_native[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
        crop_min, crop_max = crop_flair.min(), crop_flair.max()
        crop_flair_norm = (crop_flair - crop_min) / (crop_max - crop_min) if crop_max > crop_min else np.zeros_like(crop_flair)
        crop_image_64 = resize_volume_torch(crop_flair_norm, (64, 64, 64), mode="trilinear")
        crop_target_64 = resize_volume_torch(crop_seg, (64, 64, 64), mode="nearest")
        crop_target_bin = (crop_target_64 > 0.5).astype(np.float32)

        with torch.no_grad():
            full_input_t = torch.from_numpy(full_image_64).unsqueeze(0).unsqueeze(0).to(device)
            full_probs = model(full_input_t)["probs"].squeeze(0).squeeze(0).cpu().numpy()
            dice_full = dice_score((full_probs >= 0.5).astype(np.float32), full_target_bin)

            crop_input_t = torch.from_numpy(crop_image_64).unsqueeze(0).unsqueeze(0).to(device)
            crop_probs = model(crop_input_t)["probs"].squeeze(0).squeeze(0).cpu().numpy()
            dice_crop = dice_score((crop_probs >= 0.5).astype(np.float32), crop_target_bin)

        delta_d_context = dice_full - dice_crop

        records.append({
            "subject_id": subject_id,
            "native_size": e48_by_id[subject_id]["native_size"],
            "N_b": e48_by_id[subject_id]["drop"],
            "dice_full_context": dice_full,
            "dice_reduced_context": dice_crop,
            "delta_D_context": delta_d_context,
        })

        if len(records) % 25 == 0:
            print(f"  processed {len(records)} subjects", flush=True)

    if skipped_no_lesion:
        print(f"Skipped {skipped_no_lesion} subjects with no lesion voxels.", flush=True)

    with open(OUT_DIR / "E95_context_benefit_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.", flush=True)

    # ================= Analysis =================
    n_b = np.array([r["N_b"] for r in records])
    delta_d = np.array([r["delta_D_context"] for r in records])
    native_size = np.array([r["native_size"] for r in records])

    print(f"\n=== E95 Necessity vs Context-Benefit Diagnostic: n={len(records)} ===")
    print(f"Mean dice_full_context={np.mean([r['dice_full_context'] for r in records]):.4f}, "
          f"mean dice_reduced_context={np.mean([r['dice_reduced_context'] for r in records]):.4f}")
    print(f"Mean delta_D_context={delta_d.mean():+.4f} (positive = full context helps)")

    rho, p_param = stats.spearmanr(n_b, delta_d)
    rng = np.random.default_rng(SEED)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_delta = rng.permutation(delta_d)
        perm_rhos[i], _ = stats.spearmanr(n_b, perm_delta)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())

    print(f"\nSpearman(N_b, delta_D_context) = {rho:+.4f} (parametric p={p_param:.4e}, permutation p={p_perm:.4f})")

    # Confound check: is this just native_size again?
    rho_size, p_size = stats.spearmanr(native_size, delta_d)
    print(f"Spearman(native_size, delta_D_context) = {rho_size:+.4f} (p={p_size:.4e}) -- confound check")

    # Partial correlation of N_b vs delta_D controlling for native_size
    # (rank-based partial correlation via residualized linear regression on ranks)
    def rank_partial_corr(x, y, z):
        rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
        beta_xz = np.polyfit(rz, rx, 1)
        resid_x = rx - np.polyval(beta_xz, rz)
        beta_yz = np.polyfit(rz, ry, 1)
        resid_y = ry - np.polyval(beta_yz, rz)
        return stats.pearsonr(resid_x, resid_y)

    partial_rho, partial_p = rank_partial_corr(n_b, delta_d, native_size)
    print(f"Partial correlation N_b vs delta_D_context, controlling for native_size: "
          f"rho={partial_rho:+.4f}, p={partial_p:.4e}")

    hypothesis_supported = (abs(rho) > 0.3) and (p_perm < 0.05)

    print(f"\n=== DECISION: {'HYPOTHESIS SUPPORTED -- proceed to Step 4' if hypothesis_supported else 'HYPOTHESIS KILLED -- abandon causal-curriculum idea'} ===")
    if hypothesis_supported:
        print("Causal bottleneck necessity IS related to context-sensitivity on this fixed checkpoint.")
        print("This provides a rational, evidence-based reason to proceed to a minimal training intervention.")
    else:
        print("No sufficiently strong relationship found between causal necessity and context benefit.")
        print("Per the pre-declared stop condition, the causal-curriculum idea should be ABANDONED here --")
        print("no training experiment is justified by this diagnostic.")

    summary = {
        "checkpoint_val_dice": ckpt.get("best_val_dice"),
        "n_subjects": len(records),
        "mean_delta_D_context": float(delta_d.mean()),
        "spearman_rho_Nb_deltaD": float(rho),
        "parametric_p": float(p_param),
        "permutation_p": p_perm,
        "spearman_rho_size_deltaD_confound_check": float(rho_size),
        "size_confound_p": float(p_size),
        "partial_rho_Nb_deltaD_given_size": float(partial_rho),
        "partial_p": float(partial_p),
        "hypothesis_supported": bool(hypothesis_supported),
    }
    with open(OUT_DIR / "E95_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E95_summary.json")


if __name__ == "__main__":
    main()
