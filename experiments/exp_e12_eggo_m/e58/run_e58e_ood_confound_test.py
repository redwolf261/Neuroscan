"""
Phase E58 Stage 3: out-of-distribution ablation confound test.

CONTEXT: Stage 2/2b found cross-subject bottleneck DONOR substitution is
significantly MORE damaging than zeroing the bottleneck (worse on both
checkpoints tested, more strongly on E48's own). Before treating this
as a genuine "representation compatibility" phenomenon, this phase
tests the alternative, already-documented explanation from the
mechanistic-interpretability literature: the "out-of-distribution
ablation problem" (Li & Janson, NeurIPS 2024) -- resampling/substitution
ablations can push a model into activation regimes never seen during
training, so the resulting damage may reflect generic OOD confusion,
not a real "wrong information is actively misleading" effect specific
to cross-subject mismatch.

METHOD: MEAN ablation, the standard, cheap proxy used in the
interpretability literature to distinguish "missing information" from
"out-of-distribution input" -- replace each subject's bottleneck with
the MEAN bottleneck tensor across the full 125-subject validation set
(computed once, reused for every subject). This is NOT zero (an
extreme, definitely-OOD value the model likely never saw in training)
and NOT a real donor subject's bottleneck (a real, in-distribution
value, but for the WRONG subject) -- it is a smooth, "average/generic"
representation, in-distribution in the aggregate sense, carrying no
subject-specific signal at all. If mean ablation behaves like ZERO
(much less damaging than donor substitution), that is evidence FOR the
OOD-confound explanation: donor substitution's extra damage comes from
being a SPECIFIC, wrong, structured signal (an OOD-flavored confusion),
not from "wrongness" per se, since a generic/smooth non-informative
representation (mean) does comparatively little damage. If mean
ablation behaves like DONOR SUBSTITUTION (similarly damaging), that
argues AGAINST the OOD-confound explanation -- if even an unstructured,
"boring," in-distribution-aggregate representation is nearly as
damaging as a specific wrong subject's, the mechanism is more likely a
genuine sensitivity to non-null bottleneck content, not specifically to
donor-subject wrongness.

Reuses the SAME plain v3/D4-only checkpoint as Stage 2 (run_e58c) for
direct comparability -- not E48's own v5 checkpoint, to avoid
reintroducing the gate confound into this specific comparison.

STATISTICS: same discipline as every prior stage -- paired comparisons
(mean_ablation vs zero, mean_ablation vs donor_rand, from Stage 2's own
already-computed table, reused not rerun), Spearman(size, drop) +
permutation + bootstrap CI for the new mean_ablation condition.
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

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000
N_BOOT = 1000

CKPT_PATH = (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs"
             / "DeepSup_D4only_seed0" / "checkpoints" / "best.pth")

STAGE2_TABLE_PATH = OUT_DIR / "E58c_substitution_table.json"  # reused, not rerun


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


def compute_bottleneck(model, image, device):
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)
    return bottleneck, enc1, enc2, enc3


def forward_from_bottleneck(model, bottleneck, enc1, enc2, enc3):
    with torch.no_grad():
        upconv3 = model.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = model.dec3(cat3)

        upconv2 = model.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = model.dec2(cat2)

        upconv1 = model.upconv1(dec2)
        cat1 = torch.cat([upconv1, enc1], dim=1)
        dec1 = model.dec1(cat1)

        probs = model.seg_head(dec1)
        return probs.squeeze(0).squeeze(0).cpu().numpy()


def spearman_with_permutation(x, y, n_perm, seed):
    rho, p_parametric = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    perm_rhos = np.empty(n_perm)
    for i in range(n_perm):
        perm_y = rng.permutation(y)
        perm_rhos[i], _ = stats.spearmanr(x, perm_y)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())
    return float(rho), float(p_parametric), p_perm


def bootstrap_rho_ci(x, y, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rhos[i], _ = stats.spearmanr(x[idx], y[idx])
    return float(np.percentile(boot_rhos, 2.5)), float(np.percentile(boot_rhos, 97.5))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded D4-only checkpoint: best_val_dice={ckpt.get('best_val_dice')}", flush=True)

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n = len(val_dataset)
    print(f"Validation set size: {n}", flush=True)

    # ================= Sanity check =================
    image0, _, _ = val_dataset[0]
    image0_b = image0.unsqueeze(0).to(device)
    with torch.no_grad():
        real = model(image0_b)["probs"].squeeze(0).squeeze(0).cpu().numpy()
    bn0, e1_0, e2_0, e3_0 = compute_bottleneck(model, image0_b, device)
    manual = forward_from_bottleneck(model, bn0, e1_0, e2_0, e3_0)
    max_diff = float(np.abs(real - manual).max())
    print(f"\n[Sanity check] split forward vs real forward: max abs diff = {max_diff:.6e}")
    assert max_diff == 0.0, "Split reimplementation does not match real forward() -- STOP."
    print("[Sanity check] PASS.\n")

    # ================= Precompute cache + MEAN bottleneck =================
    print("Precomputing bottleneck/skip caches for all subjects...", flush=True)
    cache = {}
    native_sizes = {}
    target_bins = {}
    subject_ids_ordered = []
    bottleneck_sum = None
    for idx in range(n):
        image, mask, subject_id = val_dataset[idx]
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_dataset.subject_dirs[idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_size = int(seg_binary_native.sum())

        mask_frac_64 = fractional_occupancy_64(seg_binary_native)
        target_bin = (mask_frac_64 > 0.5).astype(np.float32)

        bn, e1, e2, e3 = compute_bottleneck(model, image_b, device)
        cache[subject_id] = {"bottleneck": bn, "enc1": e1, "enc2": e2, "enc3": e3}
        native_sizes[subject_id] = native_size
        target_bins[subject_id] = target_bin
        subject_ids_ordered.append(subject_id)

        bottleneck_sum = bn.clone() if bottleneck_sum is None else bottleneck_sum + bn

        if (idx + 1) % 25 == 0:
            print(f"  cached {idx+1}/{n}", flush=True)

    mean_bottleneck = bottleneck_sum / n
    print(f"Cached {len(cache)} subjects. Mean bottleneck computed, shape={tuple(mean_bottleneck.shape)}.\n")

    # ================= Main loop: zero / mean_ablation per subject =================
    records = []
    for sid in subject_ids_ordered:
        c = cache[sid]
        target_bin = target_bins[sid]
        native_size = native_sizes[sid]

        zero_bn = torch.zeros_like(c["bottleneck"])

        probs_intact = forward_from_bottleneck(model, c["bottleneck"], c["enc1"], c["enc2"], c["enc3"])
        probs_zero = forward_from_bottleneck(model, zero_bn, c["enc1"], c["enc2"], c["enc3"])
        probs_mean = forward_from_bottleneck(model, mean_bottleneck, c["enc1"], c["enc2"], c["enc3"])

        dice_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin)
        dice_zero = dice_score((probs_zero >= 0.5).astype(np.float32), target_bin)
        dice_mean = dice_score((probs_mean >= 0.5).astype(np.float32), target_bin)

        records.append({
            "subject_id": sid, "native_size": native_size,
            "dice_intact": dice_intact,
            "dice_zero": dice_zero, "drop_zero": dice_intact - dice_zero,
            "dice_mean": dice_mean, "drop_mean": dice_intact - dice_mean,
        })

    with open(OUT_DIR / "E58e_ood_confound_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} subject records.\n")

    # ================= Load Stage 2's own donor_rand results (reused, not rerun) =================
    with open(STAGE2_TABLE_PATH) as f:
        stage2_records = json.load(f)
    stage2_by_sid = {r["subject_id"]: r for r in stage2_records}
    drop_donor_rand = np.array([stage2_by_sid[sid]["drop_donor_rand"] for sid in subject_ids_ordered], dtype=np.float64)
    drop_donor_size = np.array([stage2_by_sid[sid]["drop_donor_size"] for sid in subject_ids_ordered], dtype=np.float64)

    # ================= Statistics =================
    native_size_arr = np.array([r["native_size"] for r in records], dtype=np.float64)
    dice_intact_arr = np.array([r["dice_intact"] for r in records])
    drop_zero = np.array([r["drop_zero"] for r in records], dtype=np.float64)
    drop_mean = np.array([r["drop_mean"] for r in records], dtype=np.float64)

    print(f"=== E58e OOD-Confound Test: n={len(records)} subjects (plain v3 checkpoint) ===")
    print(f"Mean dice_intact = {dice_intact_arr.mean():.4f}")
    print(f"Mean drop (zero)        = {drop_zero.mean():.4f}")
    print(f"Mean drop (mean_bottleneck) = {drop_mean.mean():.4f}")
    print(f"Mean drop (donor_size, from Stage 2)  = {drop_donor_size.mean():.4f}")
    print(f"Mean drop (donor_rand, from Stage 2)  = {drop_donor_rand.mean():.4f}")

    print("\n--- Paired comparisons (drop, lower = better preserved) ---")
    for name_a, arr_a, name_b, arr_b in [
        ("zero", drop_zero, "mean_bottleneck", drop_mean),
        ("mean_bottleneck", drop_mean, "donor_rand", drop_donor_rand),
        ("mean_bottleneck", drop_mean, "donor_size", drop_donor_size),
    ]:
        t_stat, t_p = stats.ttest_rel(arr_a, arr_b)
        w_stat, w_p = stats.wilcoxon(arr_a, arr_b)
        direction = "smaller" if arr_b.mean() < arr_a.mean() else "larger"
        print(f"  {name_a} vs {name_b}: mean_diff={arr_a.mean()-arr_b.mean():+.4f} "
              f"({name_b} drop is {direction}) | paired-t p={t_p:.4f}, Wilcoxon p={w_p:.4f}")

    rho_mean, p_mean_param, p_mean_perm = spearman_with_permutation(native_size_arr, drop_mean, N_PERM, SEED)
    ci_mean = bootstrap_rho_ci(native_size_arr, drop_mean, N_BOOT, SEED)
    print(f"\nmean_bottleneck: Spearman(native_size, drop) = {rho_mean:+.4f} "
          f"(perm p={p_mean_perm:.4f}) 95% CI=[{ci_mean[0]:+.4f}, {ci_mean[1]:+.4f}]")

    # ================= Decision =================
    t_mean_vs_zero, p_mean_vs_zero = stats.ttest_rel(drop_mean, drop_zero)
    t_mean_vs_randdonor, p_mean_vs_randdonor = stats.ttest_rel(drop_mean, drop_donor_rand)

    mean_close_to_zero = abs(drop_mean.mean() - drop_zero.mean()) < abs(drop_mean.mean() - drop_donor_rand.mean())
    mean_significantly_less_damaging_than_donor = (drop_mean.mean() < drop_donor_rand.mean()) and (p_mean_vs_randdonor < 0.05)

    print(f"\nMean-bottleneck damage is closer to ZERO than to donor_rand: {mean_close_to_zero}")
    print(f"Mean-bottleneck significantly LESS damaging than donor_rand: {mean_significantly_less_damaging_than_donor}")

    if mean_significantly_less_damaging_than_donor and mean_close_to_zero:
        verdict = "OOD_CONFOUND_SUPPORTED"
        interpretation = ("Mean ablation (a generic, non-informative, in-distribution-aggregate signal) does "
                           "comparatively little damage, similar to zero -- donor substitution's EXTRA damage "
                           "appears specific to injecting a SPECIFIC, structured, wrong signal, consistent with "
                           "the out-of-distribution ablation confound already documented in the interpretability "
                           "literature (Li & Janson 2024). The 'compatibility phenomenon' framing should be treated "
                           "with skepticism -- this looks more like a known measurement artifact than a new causal finding.")
    elif not mean_close_to_zero:
        verdict = "OOD_CONFOUND_NOT_SUPPORTED"
        interpretation = ("Mean ablation is nearly as damaging as donor substitution, despite carrying no "
                           "subject-specific WRONG signal -- if even a generic, smooth, non-informative "
                           "representation is comparably damaging, the effect is more likely a genuine sensitivity "
                           "to non-null bottleneck content per se, not specifically to cross-subject 'wrongness'. "
                           "This argues AGAINST the simple OOD-confound explanation.")
    else:
        verdict = "INCONCLUSIVE"
        interpretation = "Mixed result -- neither pre-declared pattern clearly holds; report plainly, do not force a conclusion."

    print(f"\n=== VERDICT: {verdict} ===")
    print(interpretation)

    summary = {
        "n_subjects": len(records),
        "mean_dice_intact": float(dice_intact_arr.mean()),
        "mean_drop": {
            "zero": float(drop_zero.mean()), "mean_bottleneck": float(drop_mean.mean()),
            "donor_size": float(drop_donor_size.mean()), "donor_rand": float(drop_donor_rand.mean()),
        },
        "mean_bottleneck_spearman": {"rho": rho_mean, "perm_p": p_mean_perm, "ci_95": list(ci_mean)},
        "mean_vs_zero_p": float(p_mean_vs_zero),
        "mean_vs_donor_rand_p": float(p_mean_vs_randdonor),
        "verdict": verdict,
        "interpretation": interpretation,
    }
    with open(OUT_DIR / "E58e_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E58e_summary.json")


if __name__ == "__main__":
    main()
