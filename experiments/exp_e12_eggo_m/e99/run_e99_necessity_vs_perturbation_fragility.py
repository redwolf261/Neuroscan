"""
Phase E99: Zero-Training Diagnostic -- Does Causal Necessity Predict
Fragility to Lesion-Region Perturbation?

CONTEXT: E48-E97 branch formally closed (see phase_e97_preservation
_killed_gate1 memory). New candidate mechanism per full-pipeline sweep:
Necessity-Conditioned Lesion-Centric Augmentation (NC-LCA) -- LCA
(Lesion-Centric Augmentation, ISAIMS 2025) applies strong elastic/grid-
distortion augmentation to lesion regions UNIFORMLY across all training
samples and gets +1.63pp Dice; this project's differentiator is
conditioning augmentation STRENGTH on N_b(x) (E48-style causal
bottleneck necessity), an asset LCA does not have. Novelty audit found
no collision on this specific combination (checked against Causal-SAM-
LLM, which uses "causal" in a confounder/domain-generalization sense,
not ablation-derived, and does not condition augmentation by any
per-sample signal).

THIS PHASE is the mandatory zero-training gate before any training
code, per this project's own established discipline throughout E95-E97.
Since augmentation is inherently a training-time phenomenon, the
zero-training proxy tested here is: does the model's CURRENT fragility
to lesion-region elastic perturbation (the exact transform family LCA
uses) relate to N_b? If high-N_b subjects are ALREADY more fragile to
this specific perturbation type on the existing frozen checkpoint, that
is a rational, evidence-based reason to expect they would benefit more
from training-time exposure to that perturbation (LCA's actual
mechanism) -- directly analogous to E95's "does necessity predict
context benefit" logic, applied to perturbation-fragility instead.

METHODOLOGY (NO TRAINING, same frozen E48-E97 checkpoint):
  For each subject, apply a LESION-REGION-LOCALIZED elastic deformation
  (matching LCA's own described transform family: elastic deformation
  targeted at the lesion region, not the whole volume) to the FLAIR
  input, at a FIXED, MODERATE perturbation strength (same for every
  subject -- necessity-conditioning is what we're testing FOR, not
  assuming), then measure:

    Fragility_i = Dice(original input) - Dice(perturbed input)

  A subject with HIGH fragility is one whose current prediction is
  brittle to exactly the kind of deformation LCA trains against.

PRE-DECLARED DECISION RULE (mirrors E95's exact structure):
  HYPOTHESIS SUPPORTED (proceed to novelty-cleared design + 1-seed
  smoke test) if:
    Spearman rho(N_b, Fragility) > 0.3, permutation p < 0.05, AND
    the relationship survives a partial-correlation control for
    native_size (checking N_b carries information beyond size alone,
    exactly as E95 required).
  HYPOTHESIS KILLED (abandon NC-LCA, per explicit stop condition) if
  the correlation is weak/null, OR entirely explained by native_size.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import scipy.ndimage as ndi
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000
ELASTIC_ALPHA = 8.0   # deformation field magnitude (voxels, at 64^3 scale)
ELASTIC_SIGMA = 3.0   # Gaussian smoothing of the random field (controls smoothness)
DILATE_MARGIN = 4     # voxels of dilation around the lesion mask defining the "lesion region" to perturb

CKPT_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs" / "AttnGate_seed0" / "checkpoints" / "best.pth"
E48_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    if denom == 0:
        return 1.0
    return float(2 * tp / denom)


def elastic_deform_lesion_region(image_64, mask_64, rng, alpha, sigma, dilate_margin):
    """Applies a smooth random elastic deformation, MASKED to only affect
    the lesion region (dilated by a fixed margin) -- matching LCA's own
    'lesion-centric' design (augment the lesion region specifically, not
    the whole volume). image_64, mask_64: (64,64,64) numpy arrays."""
    shape = image_64.shape
    region = ndi.binary_dilation(mask_64 > 0.5, iterations=dilate_margin).astype(np.float32)
    if region.sum() == 0:
        return image_64.copy()

    # Random smooth displacement field, restricted to the lesion region.
    dx = ndi.gaussian_filter((rng.random(shape) * 2 - 1), sigma=sigma) * alpha
    dy = ndi.gaussian_filter((rng.random(shape) * 2 - 1), sigma=sigma) * alpha
    dz = ndi.gaussian_filter((rng.random(shape) * 2 - 1), sigma=sigma) * alpha
    dx *= region
    dy *= region
    dz *= region

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    indices = (x + dx, y + dy, z + dz)
    deformed = ndi.map_coordinates(image_64, indices, order=1, mode="reflect")
    return deformed.astype(np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    assert abs(ckpt.get("best_val_dice", 0) - 0.9101624600589275) < 1e-9, \
        "Checkpoint mismatch -- must match E48-E97's exact checkpoint."
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

    rng_global = np.random.default_rng(SEED)
    records = []
    for subject_idx in range(len(val_dataset)):
        image, mask, subject_id = val_dataset[subject_idx]
        if subject_id not in e48_by_id:
            continue

        image_64 = image.squeeze(0).numpy()
        mask_64 = mask.squeeze(0).numpy()
        target_bin = (mask_64 > 0.5).astype(np.float32)

        # Per-subject deterministic RNG (seeded from global + index) for reproducibility.
        rng = np.random.default_rng(SEED + subject_idx)
        perturbed_image = elastic_deform_lesion_region(
            image_64, mask_64, rng, ELASTIC_ALPHA, ELASTIC_SIGMA, DILATE_MARGIN
        )

        with torch.no_grad():
            def run(vol):
                t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
                probs = model(t)["probs"].squeeze(0).squeeze(0).cpu().numpy()
                return dice_score((probs >= 0.5).astype(np.float32), target_bin)

            dice_original = run(image_64)
            dice_perturbed = run(perturbed_image)

        fragility = dice_original - dice_perturbed

        records.append({
            "subject_id": subject_id,
            "native_size": e48_by_id[subject_id]["native_size"],
            "N_b": e48_by_id[subject_id]["drop"],
            "dice_original": dice_original,
            "dice_perturbed": dice_perturbed,
            "fragility": fragility,
        })

        if len(records) % 25 == 0:
            print(f"  processed {len(records)} subjects", flush=True)

    with open(OUT_DIR / "E99_fragility_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.", flush=True)

    n_b = np.array([r["N_b"] for r in records])
    fragility = np.array([r["fragility"] for r in records])
    native_size = np.array([r["native_size"] for r in records])

    print(f"\n=== E99 Necessity vs Perturbation-Fragility Diagnostic: n={len(records)} ===")
    print(f"Mean dice_original={np.mean([r['dice_original'] for r in records]):.4f}, "
          f"mean dice_perturbed={np.mean([r['dice_perturbed'] for r in records]):.4f}")
    print(f"Mean fragility={fragility.mean():+.4f}")

    rho, p_param = stats.spearmanr(n_b, fragility)
    rng = np.random.default_rng(SEED)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_rhos[i], _ = stats.spearmanr(n_b, rng.permutation(fragility))
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())

    print(f"\nSpearman(N_b, fragility) = {rho:+.4f} (parametric p={p_param:.4e}, permutation p={p_perm:.4f})")

    rho_size, p_size = stats.spearmanr(native_size, fragility)
    print(f"Spearman(native_size, fragility) = {rho_size:+.4f} (p={p_size:.4e}) -- confound check")

    def rank_partial_corr(x, y, z):
        rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
        beta_xz = np.polyfit(rz, rx, 1)
        resid_x = rx - np.polyval(beta_xz, rz)
        beta_yz = np.polyfit(rz, ry, 1)
        resid_y = ry - np.polyval(beta_yz, rz)
        return stats.pearsonr(resid_x, resid_y)

    partial_rho, partial_p = rank_partial_corr(n_b, fragility, native_size)
    print(f"Partial correlation N_b vs fragility, controlling for native_size: "
          f"rho={partial_rho:+.4f}, p={partial_p:.4e}")

    hypothesis_supported = (abs(rho) > 0.3) and (p_perm < 0.05) and (abs(partial_rho) > 0.2)

    print(f"\n=== DECISION: {'HYPOTHESIS SUPPORTED' if hypothesis_supported else 'HYPOTHESIS KILLED'} ===")
    if hypothesis_supported:
        print("Causal bottleneck necessity predicts fragility to lesion-region elastic perturbation,")
        print("beyond what native_size alone explains. Rational basis to design NC-LCA and proceed")
        print("to a 1-seed training smoke test.")
    else:
        print("No sufficiently strong, size-independent relationship found. Per pre-declared stop")
        print("condition, abandon NC-LCA -- no training experiment justified by this diagnostic.")

    summary = {
        "checkpoint_val_dice": ckpt.get("best_val_dice"),
        "n_subjects": len(records),
        "mean_fragility": float(fragility.mean()),
        "spearman_rho_Nb_fragility": float(rho), "parametric_p": float(p_param), "permutation_p": p_perm,
        "spearman_rho_size_fragility": float(rho_size), "size_confound_p": float(p_size),
        "partial_rho_Nb_fragility_given_size": float(partial_rho), "partial_p": float(partial_p),
        "hypothesis_supported": bool(hypothesis_supported),
    }
    with open(OUT_DIR / "E99_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E99_summary.json")


if __name__ == "__main__":
    main()
