"""
Phase E87: Fit a cheap, mask-only necessity predictor N_hat_b(x).

CONTEXT: E85 confirmed a real necessity-allocation mismatch (bottleneck
gradient allocation A_b(x) does not track E48-style causal necessity
N_b(x), rho~0.03-0.06, consistently null). E86 ruled out a compensating-
circuit confound. The proposed fix (Necessity-Gated Gradient Rescaling,
NGGR) needs a necessity estimate N_hat_b(x) computable CHEAPLY at every
training step, without running a full ablation forward pass -- the real
N_b(x) requires TWO forward passes (intact + ablated) per subject, which
is too expensive to run every step for every sample in a batch.

THIS PHASE fits a small regressor from MASK-ONLY geometric features
(computable in <1ms from the ground-truth segmentation mask alone, no
model forward pass) to E48/E86's causal necessity measurements, using
the 125 validation subjects as the fitting set, WITH HELD-OUT evaluation
(not just in-sample R^2) since this predictor will be used to gate a
live training intervention and a purely in-sample-fit predictor would be
a real risk of overfitting to those exact 125 subjects.

FEATURES: an earlier version of this predictor used mask-only geometric
features (native_size, surface_to_volume, n_components,
max_component_frac) and FAILED its own pre-declared reliability check
(mean 5-fold CV R^2 = 0.069, one fold negative) -- mask geometry alone
does not carry enough signal to predict E48-style causal necessity.
Corrected here to ADD one forward-pass feature, computable from a single
ALREADY-HAPPENING (non-ablated) forward pass at train time -- no extra
model evaluation cost, unlike the true N_b(x) which needs TWO passes
(intact + ablated):

  5. pred_entropy: mean binary predictive entropy of the model's own
     (intact) segmentation probs over the lesion region --
     -p*log(p) - (1-p)*log(1-p), averaged over voxels where the ground-
     truth mask is positive. Directly measures "how uncertain is the
     model, right now, about this specific subject's lesion" -- this is
     motivated directly by E48's own mechanistic story (small lesions
     are locally ambiguous / hard to distinguish from noise), and unlike
     mask geometry, it reflects what the CURRENT model actually finds
     hard, not just a static shape property.

Original mask-only features retained alongside it:
  1. native_size: total lesion voxel count (E48's own established metric)
  2. surface_to_volume: lesion surface area / volume (a proxy for local
     ambiguity -- a highly fragmented/thin lesion has more boundary
     relative to its volume, meaning more of it is "locally ambiguous"
     even if total size is large)
  3. n_components: number of disconnected lesion components (multi-focal
     lesions may be individually small/ambiguous even if native_size is
     moderate)
  4. max_component_frac: the largest connected component's fraction of
     total lesion volume (low value = lesion is fragmented into many
     small pieces, none individually large/clear)

TARGET: E86's drop_bottleneck_alone (the FRESH, re-verified necessity
measurement from today's single-environment run -- NOT E48's original
stale table, per the E85/E86 re-verification finding).

VALIDATION: 5-fold cross-validated R^2 (not just in-sample), reported
honestly. If held-out R^2 is poor, this predictor should NOT be trusted
to gate a live training intervention -- report plainly and reconsider
before building NGGR on top of it.
"""
import sys
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

CKPT_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs" / "AttnGate_seed0" / "checkpoints" / "best.pth"

OUT_DIR = Path(__file__).parent
E86_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e86" / "E86_compensating_circuit_table.json"
SEED = 0


def compute_mask_features(seg_binary_native):
    """All features computable from the native-resolution binary mask
    alone, no model, no forward pass -- these must be cheap enough to
    compute at data-loading time for every training sample."""
    native_size = int(seg_binary_native.sum())

    if native_size == 0:
        return {
            "native_size": 0, "surface_to_volume": 0.0,
            "n_components": 0, "max_component_frac": 0.0,
        }

    # Surface voxels: mask voxels with at least one background 6-neighbor.
    eroded = ndi.binary_erosion(seg_binary_native, structure=np.ones((3, 3, 3)))
    surface_voxels = seg_binary_native.astype(bool) & (~eroded)
    surface_area = float(surface_voxels.sum())
    surface_to_volume = surface_area / native_size

    # Connected components (26-connectivity, matches this project's own
    # multi-focal-lesion handling convention elsewhere).
    labeled, n_components = ndi.label(seg_binary_native, structure=np.ones((3, 3, 3)))
    if n_components > 0:
        component_sizes = ndi.sum(seg_binary_native, labeled, index=range(1, n_components + 1))
        max_component_frac = float(np.max(component_sizes)) / native_size
    else:
        max_component_frac = 0.0

    return {
        "native_size": native_size,
        "surface_to_volume": surface_to_volume,
        "n_components": int(n_components),
        "max_component_frac": max_component_frac,
    }


def fractional_occupancy_64(seg_binary_native):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=(64, 64, 64), mode="area").squeeze().numpy()
    return frac


def compute_pred_entropy(probs_64, target_bin_64, eps=1e-7):
    """Mean binary predictive entropy over voxels where the ground-truth
    mask is positive -- 'how uncertain is the model about the lesion
    region it's supposed to find', computed from the SAME (intact)
    forward pass every training step already runs, no extra cost."""
    p = np.clip(probs_64, eps, 1 - eps)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    lesion_voxels = target_bin_64 > 0.5
    if lesion_voxels.sum() == 0:
        return 0.0
    return float(entropy[lesion_voxels].mean())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(E86_TABLE_PATH) as f:
        e86_records = json.load(f)
    e86_by_id = {r["subject_id"]: r["drop_bottleneck_alone"] for r in e86_records}
    print(f"Loaded {len(e86_records)} E86 subject records (fresh necessity targets).", flush=True)

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded E46/E48/E85/E86 checkpoint for pred_entropy feature: "
          f"best_val_dice={ckpt.get('best_val_dice')}", flush=True)

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    print(f"Validation set size: {len(val_dataset)}", flush=True)

    rows = []
    for subject_idx in range(len(val_dataset)):
        image, _, subject_id = val_dataset[subject_idx]
        if subject_id not in e86_by_id:
            continue

        subject_dir = val_dataset.subject_dirs[subject_idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)

        feats = compute_mask_features(seg_binary_native)

        image_b = image.unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(image_b)["probs"].squeeze(0).squeeze(0).cpu().numpy()
        mask_frac_64 = fractional_occupancy_64(seg_binary_native)
        target_bin_64 = (mask_frac_64 > 0.5).astype(np.float32)
        feats["pred_entropy"] = compute_pred_entropy(probs, target_bin_64)

        feats["subject_id"] = subject_id
        feats["N_b_target"] = e86_by_id[subject_id]
        rows.append(feats)

        if (len(rows)) % 25 == 0:
            print(f"  processed {len(rows)} subjects", flush=True)

    with open(OUT_DIR / "E87_feature_table.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {len(rows)} feature records.", flush=True)

    feature_names = ["native_size", "surface_to_volume", "n_components", "max_component_frac", "pred_entropy"]
    X = np.array([[r[f] for f in feature_names] for r in rows], dtype=np.float64)
    y = np.array([r["N_b_target"] for r in rows], dtype=np.float64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5-fold CV, HONEST held-out evaluation (this predictor will gate a
    # live training intervention -- in-sample R^2 alone would be
    # misleading and is explicitly not trusted here). Compare 3 model
    # classes: linear (Ridge, already shown to fail), and two non-linear
    # options with AGGRESSIVE regularization given only 125 samples --
    # shallow trees/few estimators to control overfitting risk explicitly.
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    models = {
        "ridge": Ridge(alpha=1.0),
        "gbr_shallow": GradientBoostingRegressor(
            n_estimators=30, max_depth=2, learning_rate=0.05,
            subsample=0.7, random_state=SEED,
        ),
        "rf_shallow": RandomForestRegressor(
            n_estimators=100, max_depth=3, min_samples_leaf=8,
            random_state=SEED,
        ),
    }

    results = {}
    print(f"\n=== E87 Necessity Predictor: model comparison, 5-fold CV R^2 ===")
    for name, mdl in models.items():
        cv_scores = cross_val_score(mdl, X_scaled, y, cv=kf, scoring="r2")
        results[name] = cv_scores
        print(f"{name:15s}: per-fold={np.round(cv_scores, 4)}, "
              f"mean={cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")

    best_name = max(results, key=lambda k: results[k].mean())
    best_cv_r2 = float(results[best_name].mean())
    print(f"\nBest model: {best_name} (mean CV R^2 = {best_cv_r2:.4f})")

    # Fit the best model on full data for the final deployed predictor.
    best_model = models[best_name]
    best_model.fit(X_scaled, y)
    y_pred_in_sample = best_model.predict(X_scaled)
    ss_res = np.sum((y - y_pred_in_sample) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_in_sample = 1 - ss_res / ss_tot
    print(f"In-sample R^2 of best model (reference only) = {r2_in_sample:.4f}")

    if hasattr(best_model, "feature_importances_"):
        print(f"\nFeature importances: "
              f"{dict(zip(feature_names, best_model.feature_importances_.round(4)))}")
    elif hasattr(best_model, "coef_"):
        print(f"\nFeature coefficients (standardized): "
              f"{dict(zip(feature_names, best_model.coef_.round(4)))}")

    # Decision: is this predictor good enough to gate a live intervention?
    predictor_usable = best_cv_r2 > 0.10  # weak but real predictive power threshold, pre-declared here

    print(f"\n=== DECISION: {'USABLE' if predictor_usable else 'NOT RELIABLE ENOUGH'} (best model: {best_name}) ===")
    if predictor_usable:
        print("Held-out R^2 shows real (if modest) predictive power. Proceed to use this predictor")
        print("in NGGR, but note its limitations explicitly in any write-up (small feature set,")
        print(f"modest R^2={best_cv_r2:.3f}, fit on only 125 subjects).")
    else:
        print("Held-out R^2 is too weak/unreliable across ALL THREE model classes tried (linear,")
        print("gradient-boosted trees, random forest) to trust for gating a live training")
        print("intervention. This is a real, informative negative result: E48-style causal necessity")
        print("does not appear to be cheaply predictable from available mask/forward-pass features")
        print("on this dataset size. Do NOT proceed with NGGR using a per-step learned predictor.")

    # Save results for all models tried, for the record.
    predictor_params = {
        "feature_names": feature_names,
        "n_fit_subjects": len(rows),
        "models_tried": {
            name: {"cv_r2_mean": float(scores.mean()), "cv_r2_std": float(scores.std()),
                   "cv_r2_per_fold": scores.tolist()}
            for name, scores in results.items()
        },
        "best_model": best_name,
        "best_cv_r2": best_cv_r2,
        "in_sample_r2_best": float(r2_in_sample),
        "usable": predictor_usable,
    }
    with open(OUT_DIR / "E87_necessity_predictor.json", "w") as f:
        json.dump(predictor_params, f, indent=2)
    print("\nSaved E87_necessity_predictor.json")


if __name__ == "__main__":
    main()
