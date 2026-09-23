"""
Phase E59: Bottleneck Interaction Audit.

CONTEXT (see docs/phases/PHASE_EVIDENCE_MAP.md): E48 established the
whole bottleneck is causally load-bearing (mean Dice drop 0.32 from
full ablation) while E58 Stage 1 found NEITHER of two fixed 1/8
octants reproduces anything close to that effect (mean drops 0.019 and
0.016 respectively -- individually almost negligible). E58 Stage 1b
found the same holds even for the lesion-containing octant specifically.

This phase asks the sharper, previously-untested mathematical question
those two facts jointly raise: is the bottleneck's causal contribution
ADDITIVE across its 8 spatial octant-groups (i.e. U(all) approx
sum of each octant's own marginal contribution, meaning it is simply
redundant/distributed information any large-enough subset recovers),
or SYNERGISTIC (U(all) >> sum of individual/pairwise contributions,
meaning the decoder needs COMBINATIONS of octants that no small subset
supplies)?

NO TRAINING. Loads the plain v3/D4-only checkpoint (same as E58 Stages
1/1b/2/3, chosen to avoid the E48/E46 attention-gate confound).

DEFINITIONS (exactly as pre-declared by the user):
  B = the (1,256,8,8,8) bottleneck tensor, partitioned into K=8 groups
      (the same 8 spatial octants used in E58 -- 4x4x4 blocks).
  For subset S subseteq {1..8}: B_S = B with every group NOT in S
      zeroed (M_S retains groups in S, zeros the rest) -- the same
      "ablate by zeroing" convention as every prior E47/E48/E58 script.
  U(S) = -L_seg(f(B_S), Y), using FocalTverskyLoss alone as L_seg
      (matching the outcome measure used by every prior E47/E48/E58
      causal script, for direct comparability -- NOT the full
      multi-term training loss).
  Second-order interaction: I_ij = U({i,j}) - U({i}) - U({j}) + U(emptyset).
      I_ij > 0 (systematically) => synergy (the pair carries information
      neither alone provides). I_ij approx 0 => additive/redundant.

SCOPE (K=8, matching E58's own octant partition -- NOT re-deriving a
new group definition): full audit requires, per subject:
  - U(emptyset)              (1 evaluation: B entirely zeroed)
  - U({i}) for i=1..8        (8 evaluations: only octant i kept)
  - U({i,j}) for all pairs   (28 evaluations: C(8,2))
  - U(all)                   (1 evaluation: intact, reused as sanity check)
Total: 38 forward passes/subject x 125 subjects = 4750 forward passes,
inference-only, no backward pass -- cheap, matching this project's own
"measured, not assumed" discipline for a question this specific.

PRE-DECLARED DECISION RULE:
  ADDITIVE/REDUNDANT supported if mean I_ij (across all 28 pairs, all
  subjects) is not significantly different from 0 (one-sample t-test,
  permutation test, 1000 trials) AND sum_i U({i}) - 7*U(emptyset)
  approximates U(all) reasonably well (small residual).
  SYNERGY supported if mean I_ij is significantly POSITIVE (p<0.05,
  both tests) AND/OR U(all) substantially exceeds the best achievable
  sum of any small subset's own marginal contributions.
  SIZE-SPECIFIC SYNERGY (the sharpest, most interesting sub-question):
  does mean I_ij (per subject) correlate with native lesion size,
  matching E48's own rho=-0.454 signature (Spearman + permutation,
  same 1000-trial discipline as every prior stage)?
"""
import sys
import json
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000

CKPT_PATH = (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs"
             / "DeepSup_D4only_seed0" / "checkpoints" / "best.pth")

# Same 8 fixed octants as E58 (4x4x4 blocks of the 8^3 bottleneck grid).
OCTANT_INDICES = list(itertools.product((0, 1), (0, 1), (0, 1)))  # 8 tuples
GROUP_IDS = list(range(8))  # 0..7, mapped 1:1 to OCTANT_INDICES


def octant_slice(idx_tuple):
    i, j, k = idx_tuple
    return (slice(i * 4, i * 4 + 4), slice(j * 4, j * 4 + 4), slice(k * 4, k * 4 + 4))


OCTANT_SLICES = [octant_slice(idx) for idx in OCTANT_INDICES]


def fractional_occupancy_64(seg_binary_native):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=(64, 64, 64), mode="area").squeeze().numpy()
    return frac


def forward_with_subset(model, image, subset, focal_fn, target):
    """subset: a set/list of group ids (0..7) to KEEP (zero everything
    else). Returns (probs_numpy, loss_value)."""
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)

        masked = torch.zeros_like(bottleneck)
        for gid in subset:
            d_sl, h_sl, w_sl = OCTANT_SLICES[gid]
            masked[:, :, d_sl, h_sl, w_sl] = bottleneck[:, :, d_sl, h_sl, w_sl]

        upconv3 = model.upconv3(masked)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = model.dec3(cat3)

        upconv2 = model.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = model.dec2(cat2)

        upconv1 = model.upconv1(dec2)
        cat1 = torch.cat([upconv1, enc1], dim=1)
        dec1 = model.dec1(cat1)

        probs = model.seg_head(dec1)
        loss = focal_fn(probs, target)
        return probs.squeeze(0).squeeze(0).cpu().numpy(), float(loss.item())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded D4-only checkpoint: best_val_dice={ckpt.get('best_val_dice')}", flush=True)

    focal_fn = FocalTverskyLoss()

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n = len(val_dataset)
    print(f"Validation set size: {n}", flush=True)

    # ================= Sanity check: subset=all 8 groups reproduces real forward() =================
    image0, mask0, _ = val_dataset[0]
    image0_b = image0.unsqueeze(0).to(device)
    target0 = mask0.unsqueeze(0).to(device)
    with torch.no_grad():
        real = model(image0_b)["probs"].squeeze(0).squeeze(0).cpu().numpy()
    manual_probs, _ = forward_with_subset(model, image0_b, set(GROUP_IDS), focal_fn, target0)
    max_diff = float(np.abs(real - manual_probs).max())
    print(f"\n[Sanity check] subset=all 8 groups vs real forward: max abs diff = {max_diff:.6e}")
    assert max_diff == 0.0, "Group-masking reimplementation does not match real forward() -- STOP."
    print("[Sanity check] PASS.\n")

    pairs = list(itertools.combinations(GROUP_IDS, 2))  # 28 pairs
    print(f"Auditing U(subset) for: emptyset, {len(GROUP_IDS)} singletons, {len(pairs)} pairs, "
          f"all(sanity) -- {2 + len(GROUP_IDS) + len(pairs)} evaluations/subject, {n} subjects.\n")

    records = []
    for idx in range(n):
        image, mask, subject_id = val_dataset[idx]
        image_b = image.unsqueeze(0).to(device)
        target = mask.unsqueeze(0).to(device)

        subject_dir = val_dataset.subject_dirs[idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_size = int(seg_binary_native.sum())

        # U(emptyset)
        _, L_empty = forward_with_subset(model, image_b, set(), focal_fn, target)
        U_empty = -L_empty

        # U({i}) for each singleton
        U_single = {}
        for gid in GROUP_IDS:
            _, L_i = forward_with_subset(model, image_b, {gid}, focal_fn, target)
            U_single[gid] = -L_i

        # U({i,j}) for each pair
        U_pair = {}
        for (i, j) in pairs:
            _, L_ij = forward_with_subset(model, image_b, {i, j}, focal_fn, target)
            U_pair[(i, j)] = -L_ij

        # U(all) -- sanity/reference
        _, L_all = forward_with_subset(model, image_b, set(GROUP_IDS), focal_fn, target)
        U_all = -L_all

        # Second-order interaction terms I_ij
        I_pair = {}
        for (i, j) in pairs:
            I_pair[(i, j)] = U_pair[(i, j)] - U_single[i] - U_single[j] + U_empty

        mean_I = float(np.mean(list(I_pair.values())))

        # Additive-model residual: how well does sum_i U({i}) - 7*U(emptyset) approximate U(all)?
        additive_estimate = sum(U_single.values()) - (len(GROUP_IDS) - 1) * U_empty
        additive_residual = U_all - additive_estimate

        records.append({
            "subject_id": subject_id, "native_size": native_size,
            "U_empty": U_empty, "U_all": U_all,
            "U_single": {str(k): v for k, v in U_single.items()},
            "U_pair": {f"{k[0]}_{k[1]}": v for k, v in U_pair.items()},
            "I_pair": {f"{k[0]}_{k[1]}": v for k, v in I_pair.items()},
            "mean_I_pair": mean_I,
            "additive_estimate_U_all": additive_estimate,
            "additive_residual": additive_residual,
        })

        if (idx + 1) % 10 == 0:
            print(f"  processed {idx+1}/{n} subjects", flush=True)

    with open(OUT_DIR / "E59_interaction_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.\n")

    # ================= Statistics =================
    native_size_arr = np.array([r["native_size"] for r in records], dtype=np.float64)
    mean_I_arr = np.array([r["mean_I_pair"] for r in records], dtype=np.float64)
    residual_arr = np.array([r["additive_residual"] for r in records], dtype=np.float64)
    U_all_arr = np.array([r["U_all"] for r in records], dtype=np.float64)
    U_empty_arr = np.array([r["U_empty"] for r in records], dtype=np.float64)

    print(f"=== E59 Bottleneck Interaction Audit: n={len(records)} subjects ===")
    print(f"Mean U(all)   = {U_all_arr.mean():.4f}  (higher = better, U = -loss)")
    print(f"Mean U(empty) = {U_empty_arr.mean():.4f}")
    print(f"Mean additive-model residual [U(all) - additive_estimate] = {residual_arr.mean():.4f} "
          f"(large positive => synergy: real U(all) beats what an additive model predicts)")

    # One-sample t-test: is mean_I_pair significantly different from 0?
    t_stat, t_p = stats.ttest_1samp(mean_I_arr, 0.0)
    print(f"\nMean I_ij across all subjects/pairs: {mean_I_arr.mean():+.4f} (std={mean_I_arr.std():.4f})")
    print(f"One-sample t-test (H0: mean I_ij = 0): t={t_stat:.4f}, p={t_p:.4e}")

    # Permutation test: sign-flip permutation on mean_I_pair (matching project convention)
    rng = np.random.default_rng(SEED)
    perm_means = np.empty(N_PERM)
    for i in range(N_PERM):
        signs = rng.choice([-1, 1], size=len(mean_I_arr))
        perm_means[i] = (mean_I_arr * signs).mean()
    p_perm = float((np.abs(perm_means) >= np.abs(mean_I_arr.mean())).mean())
    print(f"Sign-flip permutation test ({N_PERM} trials): p={p_perm:.4f}")

    # Size-dependence of the interaction/synergy signal (the sharpest question)
    rho_I, p_I_param = stats.spearmanr(native_size_arr, mean_I_arr)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng.permutation(mean_I_arr)
        perm_rhos[i], _ = stats.spearmanr(native_size_arr, perm_y)
    p_I_perm = float((np.abs(perm_rhos) >= np.abs(rho_I)).mean())
    print(f"\nSpearman(native_size, mean_I_pair) = {rho_I:+.4f} (parametric p={p_I_param:.4e}, "
          f"permutation p={p_I_perm:.4f})")
    print("(Negative rho would match E48's own size-dependence signature: MORE synergy for SMALLER lesions.)")

    rho_res, p_res_param = stats.spearmanr(native_size_arr, residual_arr)
    perm_rhos2 = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng.permutation(residual_arr)
        perm_rhos2[i], _ = stats.spearmanr(native_size_arr, perm_y)
    p_res_perm = float((np.abs(perm_rhos2) >= np.abs(rho_res)).mean())
    print(f"Spearman(native_size, additive_residual) = {rho_res:+.4f} (parametric p={p_res_param:.4e}, "
          f"permutation p={p_res_perm:.4f})")

    # ================= Pre-declared decision rule =================
    synergy_significant = (t_p < 0.05) and (p_perm < 0.05) and (mean_I_arr.mean() > 0)
    size_specific_synergy = (p_I_perm < 0.05) and (rho_I < 0)  # negative = MORE synergy for smaller lesions

    if synergy_significant and size_specific_synergy:
        verdict = "SIZE_SPECIFIC_SYNERGY_SUPPORTED"
    elif synergy_significant:
        verdict = "SYNERGY_SUPPORTED_NOT_SIZE_SPECIFIC"
    else:
        verdict = "ADDITIVE_REDUNDANT_SUPPORTED"

    print(f"\nSynergy significant (mean I_ij > 0, both tests p<0.05): {synergy_significant}")
    print(f"Size-specific (more synergy for smaller lesions, rho<0, perm p<0.05): {size_specific_synergy}")
    print(f"\n=== VERDICT: {verdict} ===")

    summary = {
        "n_subjects": len(records),
        "mean_U_all": float(U_all_arr.mean()), "mean_U_empty": float(U_empty_arr.mean()),
        "mean_additive_residual": float(residual_arr.mean()),
        "mean_I_pair": float(mean_I_arr.mean()), "std_I_pair": float(mean_I_arr.std()),
        "ttest_1samp": {"t": float(t_stat), "p": float(t_p)},
        "permutation_p": p_perm,
        "size_dependence_of_I": {"rho": float(rho_I), "parametric_p": float(p_I_param), "permutation_p": p_I_perm},
        "size_dependence_of_residual": {"rho": float(rho_res), "parametric_p": float(p_res_param), "permutation_p": p_res_perm},
        "synergy_significant": bool(synergy_significant),
        "size_specific_synergy": bool(size_specific_synergy),
        "verdict": verdict,
    }
    with open(OUT_DIR / "E59_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E59_summary.json")


if __name__ == "__main__":
    main()
