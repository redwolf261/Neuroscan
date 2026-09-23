"""
Phase E105: Classical Hausdorff-Loss Gradient Pathology Audit (BEFORE any
implementation/training).

CONTEXT: E101-E104b explored 3 published-method-failure candidates
(graph-cut/E102, CC-DiceCE/E103, EDL/E104) plus a bridge experiment
(E104b), all closed with real evidence. User proposed a 4th candidate:
Hausdorff Distance Loss (HDL), motivated by a 2025 Scientific Reports
paper reporting HDL-alone giving only 64.75% Dice vs 87.72% for BCE
(arXiv/Nature 41598-025-33136-x). DIRECT READ of that paper found its
own math is INTERNALLY INCONSISTENT: their Eq.(1)/(7) define "HDL" as
an AVERAGE of directed nearest-neighbor distances, not the classical
MAX-based Hausdorff distance their own prose and Algorithm-1 pseudocode
describe ("take the maximum of these minimum distances"). Per an
explicit decision with the user, this candidate proceeds using the
mathematically CORRECT classical Hausdorff loss (max-based), NOT the
paper's inconsistent formulation -- the paper is treated as motivating
context only, not as a frozen technical spec.

Classical Hausdorff distance:
  H(P, G) = max( max_i min_j ||p_i - g_j||, max_j min_i ||g_j - p_i|| )
This is a MAX of MINs -- genuinely non-differentiable at the argmax
(subgradient exists but flows entirely through the single worst-
offending point in each direction). This is the actual mathematical
object the broader loss-function literature (not just the one flawed
paper) attributes real optimization pathologies to: gradient
concentration on a single outlier point, high sensitivity to a single
mislabeled/noisy voxel, and potential mismatch between "the point that
dominates Hausdorff distance" and "the points that matter for Dice."

THIS PHASE (zero-training, same frozen E48-E104 checkpoint) tests
whether this GENUINE mathematical pathology would actually manifest on
NeuroScan's own predictions, computed directly from the model's
existing (already-trained) boundary predictions -- NOT by training with
Hausdorff loss (deferred, per the user's own staged plan, until this
gate passes).

For each small-lesion subject (same convention as E102/E104):
  1. Extract predicted and ground-truth foreground boundary POINT SETS
     (surface voxels, via binary erosion difference -- a lesion's own
     boundary is used for the Hausdorff sub-gradient's implicit spatial
     support, as opposed to the DICE gradient's supportover ALL
     foreground/background voxels).
  2. Compute the classical (max-based) Hausdorff distance and identify
     the SPECIFIC argmax point (the single point whose nearest-neighbor
     distance is largest) in each direction -- this IS where 100% of
     the Hausdorff sub-gradient's signal would concentrate for that
     subject.
  3. Compute, for comparison, the Dice-loss gradient's spatial support
     (ALL false-positive and false-negative voxels contribute, not just
     one) -- and check what FRACTION of the boundary the single
     Hausdorff argmax point represents (1/N_boundary_voxels, by
     construction, always tiny -- but the INTERESTING quantity is
     whether that single point is a genuine, clinically-relevant error
     or a single noisy/outlier voxel).
  4. For the argmax point specifically, check: is it located in a
     region with LOW model confidence/high evidential uncertainty (a
     genuine, meaningful error) or does it correspond to an isolated,
     small, spurious prediction artifact (a true outlier, consistent
     with the classical, well-known Hausdorff pathology)?
  5. Test whether the outlier severity (Hausdorff distance value itself,
     normalized by the lesion's own native_size scale) is worse for
     SMALL lesions than LARGE lesions -- since a fixed absolute
     worst-point distance represents a much larger RELATIVE distortion
     for a small lesion.

PRE-DECLARED DECISION RULE:
  HAUSDORFF PATHOLOGY CONFIRMED (worth designing a repair) if:
    (a) the Hausdorff argmax point's local evidential uncertainty
        (total evidence S, reusing E104's already-computed alpha/beta
        machinery) is HIGH (i.e. the model is confident, not uncertain,
        at that single dominant point) for a meaningful fraction of
        subjects -- suggesting the argmax is often a spurious isolated
        artifact rather than a genuine boundary-confidence failure, AND
    (b) the normalized Hausdorff distance (raw distance / native_size^(1/3),
        a natural length-scale normalization) is SIGNIFICANTLY larger
        for small-lesion subjects than large-lesion subjects
        (permutation p<0.05) -- confirming the relative distortion is
        worse specifically in the regime this project's whole causal
        chain (E48-E104) has already shown is information-poor.
  KILL HAUSDORFF (do not implement, do not design a repair) if either
  condition fails.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000

CKPT_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs" / "AttnGate_seed0" / "checkpoints" / "best.pth"
E48_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"


def fractional_occupancy_64(seg_binary_native):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=(64, 64, 64), mode="area").squeeze().numpy()
    return frac


def get_surface_points(binary_mask):
    """Surface voxel coordinates (D,H,W) via binary erosion difference --
    the boundary point set P or G for the Hausdorff computation."""
    if binary_mask.sum() == 0:
        return np.zeros((0, 3), dtype=np.float64)
    eroded = ndi.binary_erosion(binary_mask, structure=np.ones((3, 3, 3)))
    surface = binary_mask & (~eroded)
    coords = np.argwhere(surface)
    if len(coords) == 0:
        coords = np.argwhere(binary_mask)  # fallback for tiny (<3 voxel) components with no interior
    return coords.astype(np.float64)


def classical_hausdorff_with_argmax(pred_points, gt_points):
    """Returns (H, argmax_point, argmax_direction) -- the classical
    max-of-mins Hausdorff distance and the SPECIFIC point where the
    subgradient's entire signal would concentrate."""
    if len(pred_points) == 0 or len(gt_points) == 0:
        return None, None, None

    tree_gt = cKDTree(gt_points)
    dist_pred_to_gt, _ = tree_gt.query(pred_points)
    idx_pg = int(np.argmax(dist_pred_to_gt))
    h_pg = float(dist_pred_to_gt[idx_pg])

    tree_pred = cKDTree(pred_points)
    dist_gt_to_pred, _ = tree_pred.query(gt_points)
    idx_gp = int(np.argmax(dist_gt_to_pred))
    h_gp = float(dist_gt_to_pred[idx_gp])

    if h_pg >= h_gp:
        return h_pg, pred_points[idx_pg], "pred_to_gt"
    else:
        return h_gp, gt_points[idx_gp], "gt_to_pred"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    assert abs(ckpt.get("best_val_dice", 0) - 0.9101624600589275) < 1e-9, \
        "Checkpoint mismatch -- must match E48-E104's exact checkpoint."
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

    native_sizes_all = np.array([r["native_size"] for r in e48_records])
    median_size = float(np.median(native_sizes_all))

    records = []
    for subject_idx in range(len(val_dataset)):
        image, mask, subject_id = val_dataset[subject_idx]
        if subject_id not in e48_by_id:
            continue

        image_b = image.unsqueeze(0).to(device)
        subject_dir = val_dataset.subject_dirs[subject_idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_size = e48_by_id[subject_id]["native_size"]
        target_bin = (fractional_occupancy_64(seg_binary_native) > 0.5)

        with torch.no_grad():
            out = model(image_b)
            probs = out["probs"].squeeze(0).squeeze(0).cpu().numpy()
            alpha = out["alpha"].squeeze(0).squeeze(0).cpu().numpy()
            beta = out["beta"].squeeze(0).squeeze(0).cpu().numpy()
        pred_bin = (probs >= 0.5)
        S = alpha + beta

        if pred_bin.sum() == 0 or target_bin.sum() == 0:
            continue  # Hausdorff undefined if either set is empty

        pred_surface = get_surface_points(pred_bin)
        gt_surface = get_surface_points(target_bin)
        if len(pred_surface) == 0 or len(gt_surface) == 0:
            continue

        h_dist, argmax_point, direction = classical_hausdorff_with_argmax(pred_surface, gt_surface)
        if h_dist is None:
            continue

        ap = tuple(int(round(c)) for c in argmax_point)
        ap = tuple(np.clip(ap, 0, 63))
        s_at_argmax = float(S[ap])
        s_mean_all_lesion_voxels = float(S[target_bin | pred_bin].mean())

        h_normalized = h_dist / (native_size ** (1 / 3) + 1e-6)

        records.append({
            "subject_id": subject_id,
            "native_size": native_size,
            "hausdorff_dist": h_dist,
            "hausdorff_normalized": h_normalized,
            "argmax_direction": direction,
            "S_at_argmax": s_at_argmax,
            "S_mean_lesion": s_mean_all_lesion_voxels,
            "argmax_is_high_confidence": bool(s_at_argmax > s_mean_all_lesion_voxels),
        })

        if len(records) % 25 == 0:
            print(f"  processed {len(records)} subjects", flush=True)

    with open(OUT_DIR / "E105_hausdorff_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.", flush=True)

    n = len(records)
    native_sizes = np.array([r["native_size"] for r in records])
    h_norm = np.array([r["hausdorff_normalized"] for r in records])
    s_argmax = np.array([r["S_at_argmax"] for r in records])
    s_mean = np.array([r["S_mean_lesion"] for r in records])
    frac_high_conf_argmax = np.mean([r["argmax_is_high_confidence"] for r in records])

    print(f"\n=== E105 Hausdorff Pathology Audit: n={n} subjects ===")
    print(f"Fraction of subjects where argmax point has ABOVE-average lesion confidence "
          f"(S_argmax > S_mean_lesion): {frac_high_conf_argmax:.3f}")
    print(f"Mean S_at_argmax={s_argmax.mean():.4f} vs mean S_mean_lesion={s_mean.mean():.4f}")

    # Condition (a): is the argmax often a high-confidence (spurious-artifact-consistent) point?
    condition_a = frac_high_conf_argmax > 0.5

    # Condition (b): is normalized Hausdorff distance worse for small lesions?
    median_size_local = float(np.median(native_sizes))
    small_mask = native_sizes <= median_size_local
    large_mask = ~small_mask
    h_norm_small = h_norm[small_mask]
    h_norm_large = h_norm[large_mask]

    observed_diff = h_norm_small.mean() - h_norm_large.mean()
    combined = np.concatenate([h_norm_small, h_norm_large])
    n_small = small_mask.sum()
    rng = np.random.default_rng(SEED)
    perm_diffs = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_idx = rng.permutation(len(combined))
        perm_diffs[i] = combined[perm_idx[:n_small]].mean() - combined[perm_idx[n_small:]].mean()
    p_size = float((np.abs(perm_diffs) >= np.abs(observed_diff)).mean())

    print(f"\nNormalized Hausdorff distance: small={h_norm_small.mean():.4f}, large={h_norm_large.mean():.4f}")
    print(f"Diff (small - large) = {observed_diff:+.4f}, permutation p = {p_size:.4f}")

    condition_b = (observed_diff > 0) and (p_size < 0.05)

    decision = "HAUSDORFF_PATHOLOGY_CONFIRMED" if (condition_a and condition_b) else "KILL_HAUSDORFF"

    print(f"\nCondition (a) [argmax often high-confidence/spurious]: {'MET' if condition_a else 'NOT MET'}")
    print(f"Condition (b) [normalized distortion worse for small lesions]: {'MET' if condition_b else 'NOT MET'}")
    print(f"\n=== DECISION: {decision} ===")

    if decision == "HAUSDORFF_PATHOLOGY_CONFIRMED":
        print("The classical Hausdorff subgradient's argmax point is often a high-confidence (likely spurious)")
        print("prediction artifact, AND the relative distortion is significantly worse for small lesions --")
        print("a genuine mathematical pathology relevant to NeuroScan's own information-poor small-lesion")
        print("regime. Worth a literature audit + cheap training smoke test before any full campaign.")
    else:
        print("Either the argmax point typically corresponds to a genuine low-confidence error (not a spurious")
        print("artifact), or the distortion is not significantly worse for small lesions. Per the pre-declared")
        print("rule: KILL Hausdorff-loss investigation before any implementation.")

    summary = {
        "checkpoint_val_dice": ckpt.get("best_val_dice"),
        "n_subjects": n,
        "frac_high_confidence_argmax": float(frac_high_conf_argmax),
        "mean_S_argmax": float(s_argmax.mean()), "mean_S_mean_lesion": float(s_mean.mean()),
        "h_norm_small_mean": float(h_norm_small.mean()), "h_norm_large_mean": float(h_norm_large.mean()),
        "h_norm_diff": float(observed_diff), "h_norm_diff_p": p_size,
        "condition_a_met": bool(condition_a), "condition_b_met": bool(condition_b),
        "decision": decision,
    }
    with open(OUT_DIR / "E105_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E105_summary.json")


if __name__ == "__main__":
    main()
