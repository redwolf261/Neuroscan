"""
Phase E58 Stage 2: cross-subject bottleneck substitution.

CONTEXT: Stage 1 + Stage 1b established the E48 size-dependent bottleneck
effect is (a) not concentrated in any single fixed spatial octant, and
(b) not preferentially tied to the lesion's own physical location within
the bottleneck grid -- i.e. distributed, not spatially localized in
absolute OR lesion-relative coordinates. This does NOT establish the
content is "global context" in the strong sense (a distributed
collection of local pieces can look the same way). Per explicit
instruction, this Stage tells us WHERE to search next, not itself a
novelty claim.

THIS PHASE asks a qualitatively different question: is the bottleneck's
causally-necessary content SUBJECT-SPECIFIC (this exact subject's own
spatial/anatomical configuration) or largely SUBJECT-INTERCHANGEABLE
(a generic tissue/context prior any plausible brain provides)?

METHOD: precompute and cache all 125 subjects' own intact bottleneck
tensors (one clean forward pass each). For each subject A, run THREE
conditions (in addition to A's own intact forward pass):
  1. zero      -- A's bottleneck replaced with zeros (E48's own
                  intervention, reference).
  2. donor_size -- A's bottleneck replaced with a DIFFERENT subject B's
                  own cached bottleneck, B chosen as the subject with
                  the CLOSEST native lesion size to A's (excluding A
                  itself), a plausible "similar difficulty" donor.
  3. donor_rand -- A's bottleneck replaced with a RANDOMLY chosen
                  different subject C's own cached bottleneck (seeded,
                  reproducible), disregarding size similarity.
Manual trunk reimplementation, verified bit-for-bit against real
forward() (both for the intact case AND for the "self-substitution"
case -- substituting a subject's OWN cached bottleneck for itself must
reproduce the real forward pass exactly, a stronger and more specific
check than Stage 1's own sanity check, since it verifies the caching/
substitution PATHWAY itself, not just the manual-trunk reimplementation).

STATISTICS: (a) mean Dice preserved per condition (zero vs donor_size vs
donor_rand vs intact) -- paired Wilcoxon/t-test, all vs. each other.
(b) Spearman(native_size, drop) per condition, same discipline as every
prior stage, for direct comparison against E48's -0.454 reference and
Stage 1/1b's own octant rhos. (c) donor_size vs donor_rand comparison
specifically -- does size-matching the donor matter, or is any donor
roughly interchangeable?

PRE-DECLARED DECISION RULE (subject-invariant vs subject-specific):
  - SUBJECT-INTERCHANGEABLE supported if: mean_drop(donor_size) and
    mean_drop(donor_rand) are BOTH significantly smaller than
    mean_drop(zero) (paired test, p<0.05), AND donor_size vs donor_rand
    are NOT significantly different from each other (i.e. any foreign
    context helps roughly equally, size-matching doesn't matter much).
  - SUBJECT-SPECIFIC supported if: mean_drop(donor_size) and
    mean_drop(donor_rand) are NOT significantly smaller than
    mean_drop(zero) (foreign context is about as bad as no context --
    wrong information isn't meaningfully better than none).
  - ACTIVELY MISLEADING (a third, more surprising possible outcome) if:
    mean_drop(donor_size) or mean_drop(donor_rand) is significantly
    LARGER than mean_drop(zero) (foreign context is WORSE than no
    context at all) -- reported plainly if observed, not something to
    explain away.
Per explicit instruction: NONE of these outcomes are themselves treated
as a novelty claim. They determine which specific causal phenomenon (if
any) is characterized precisely enough to become the target of a
SEPARATE, subsequent literature-novelty search -- not run here.
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

E48_REFERENCE_RHO = -0.454


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
    """Manual encoder-only pass, returns (bottleneck, enc1, enc2, enc3)
    -- all intermediate tensors needed to later run the decoder with a
    SUBSTITUTED bottleneck for this same subject's own enc1/enc2/enc3."""
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
    """Runs ONLY the decoder half of the trunk, given a (possibly
    substituted) bottleneck and THIS SUBJECT's own enc1/enc2/enc3 skip
    connections -- the skip connections are never substituted, only the
    bottleneck itself, isolating the bottleneck's own causal content
    from the (already fully causally attributed by E47's own null) skip
    pathway."""
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

    # ================= Sanity check 1: split forward reproduces real forward() =================
    image0, _, _ = val_dataset[0]
    image0_b = image0.unsqueeze(0).to(device)
    with torch.no_grad():
        real = model(image0_b)["probs"].squeeze(0).squeeze(0).cpu().numpy()
    bn0, e1_0, e2_0, e3_0 = compute_bottleneck(model, image0_b, device)
    manual = forward_from_bottleneck(model, bn0, e1_0, e2_0, e3_0)
    max_diff = float(np.abs(real - manual).max())
    print(f"\n[Sanity check 1] split forward (own bottleneck) vs real forward: max abs diff = {max_diff:.6e}")
    assert max_diff == 0.0, "Split encoder/decoder reimplementation does not match real forward() -- STOP."
    print("[Sanity check 1] PASS.\n")

    # ================= Precompute: cache every subject's own bottleneck + skip tensors + native size =================
    print("Precomputing bottleneck/skip caches for all subjects (one clean forward pass each)...", flush=True)
    cache = {}
    native_sizes = {}
    target_bins = {}
    subject_ids_ordered = []
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

        if (idx + 1) % 25 == 0:
            print(f"  cached {idx+1}/{n}", flush=True)

    print(f"Cached {len(cache)} subjects.\n")

    # ================= Sanity check 2: self-substitution reproduces real forward() =================
    sid0 = subject_ids_ordered[0]
    self_sub = forward_from_bottleneck(
        model, cache[sid0]["bottleneck"], cache[sid0]["enc1"], cache[sid0]["enc2"], cache[sid0]["enc3"]
    )
    max_diff2 = float(np.abs(real - self_sub).max())
    print(f"[Sanity check 2] self-substitution (cached own bottleneck) vs real forward: max abs diff = {max_diff2:.6e}")
    assert max_diff2 == 0.0, "Self-substitution pathway does not reproduce real forward() -- STOP."
    print("[Sanity check 2] PASS.\n")

    # ================= Donor selection (deterministic, seeded) =================
    rng = np.random.default_rng(SEED)
    size_donor = {}
    rand_donor = {}
    for sid in subject_ids_ordered:
        others = [s for s in subject_ids_ordered if s != sid]
        # closest native size, excluding self
        size_diffs = [(abs(native_sizes[s] - native_sizes[sid]), s) for s in others]
        size_diffs.sort(key=lambda t: t[0])
        size_donor[sid] = size_diffs[0][1]
        rand_donor[sid] = others[rng.integers(0, len(others))]

    # ================= Main loop: zero / donor_size / donor_rand per subject =================
    records = []
    for sid in subject_ids_ordered:
        c = cache[sid]
        target_bin = target_bins[sid]
        native_size = native_sizes[sid]

        zero_bn = torch.zeros_like(c["bottleneck"])
        donor_size_bn = cache[size_donor[sid]]["bottleneck"]
        donor_rand_bn = cache[rand_donor[sid]]["bottleneck"]

        probs_intact = forward_from_bottleneck(model, c["bottleneck"], c["enc1"], c["enc2"], c["enc3"])
        probs_zero = forward_from_bottleneck(model, zero_bn, c["enc1"], c["enc2"], c["enc3"])
        probs_donor_size = forward_from_bottleneck(model, donor_size_bn, c["enc1"], c["enc2"], c["enc3"])
        probs_donor_rand = forward_from_bottleneck(model, donor_rand_bn, c["enc1"], c["enc2"], c["enc3"])

        dice_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin)
        dice_zero = dice_score((probs_zero >= 0.5).astype(np.float32), target_bin)
        dice_donor_size = dice_score((probs_donor_size >= 0.5).astype(np.float32), target_bin)
        dice_donor_rand = dice_score((probs_donor_rand >= 0.5).astype(np.float32), target_bin)

        records.append({
            "subject_id": sid, "native_size": native_size,
            "size_donor_id": size_donor[sid], "size_donor_native_size": native_sizes[size_donor[sid]],
            "rand_donor_id": rand_donor[sid], "rand_donor_native_size": native_sizes[rand_donor[sid]],
            "dice_intact": dice_intact,
            "dice_zero": dice_zero, "drop_zero": dice_intact - dice_zero,
            "dice_donor_size": dice_donor_size, "drop_donor_size": dice_intact - dice_donor_size,
            "dice_donor_rand": dice_donor_rand, "drop_donor_rand": dice_intact - dice_donor_rand,
        })

    with open(OUT_DIR / "E58c_substitution_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} subject records.\n")

    # ================= Statistics =================
    native_size_arr = np.array([r["native_size"] for r in records], dtype=np.float64)
    dice_intact_arr = np.array([r["dice_intact"] for r in records])
    drop_zero = np.array([r["drop_zero"] for r in records], dtype=np.float64)
    drop_donor_size = np.array([r["drop_donor_size"] for r in records], dtype=np.float64)
    drop_donor_rand = np.array([r["drop_donor_rand"] for r in records], dtype=np.float64)

    print(f"=== E58c Bottleneck Substitution Audit: n={len(records)} subjects ===")
    print(f"Mean dice_intact = {dice_intact_arr.mean():.4f}")
    print(f"Mean drop (zero)        = {drop_zero.mean():.4f}  (reference: E48 full-ablation drop = 0.3205)")
    print(f"Mean drop (donor_size)  = {drop_donor_size.mean():.4f}")
    print(f"Mean drop (donor_rand)  = {drop_donor_rand.mean():.4f}")

    print("\n--- Paired comparisons (drop, lower = better preserved) ---")
    for name_a, arr_a, name_b, arr_b in [
        ("zero", drop_zero, "donor_size", drop_donor_size),
        ("zero", drop_zero, "donor_rand", drop_donor_rand),
        ("donor_size", drop_donor_size, "donor_rand", drop_donor_rand),
    ]:
        t_stat, t_p = stats.ttest_rel(arr_a, arr_b)
        w_stat, w_p = stats.wilcoxon(arr_a, arr_b)
        direction = "smaller" if arr_b.mean() < arr_a.mean() else "larger"
        print(f"  {name_a} vs {name_b}: mean_diff={arr_a.mean()-arr_b.mean():+.4f} "
              f"({name_b} drop is {direction}) | paired-t p={t_p:.4f}, Wilcoxon p={w_p:.4f}")

    print("\n--- Spearman(native_size, drop) per condition ---")
    results_rho = {}
    for cond_name, drop_arr in [("zero", drop_zero), ("donor_size", drop_donor_size), ("donor_rand", drop_donor_rand)]:
        rho, p_param, p_perm = spearman_with_permutation(native_size_arr, drop_arr, N_PERM, SEED)
        ci = bootstrap_rho_ci(native_size_arr, drop_arr, N_BOOT, SEED)
        results_rho[cond_name] = {"rho": rho, "perm_p": p_perm, "ci_95": list(ci)}
        print(f"  {cond_name}: rho={rho:+.4f} (perm p={p_perm:.4f}) 95% CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"  Reference (E48 full ablation): rho = {E48_REFERENCE_RHO}")

    # ================= Pre-declared decision rule =================
    t_zero_vs_size, p_zero_vs_size = stats.ttest_rel(drop_zero, drop_donor_size)
    t_zero_vs_rand, p_zero_vs_rand = stats.ttest_rel(drop_zero, drop_donor_rand)
    t_size_vs_rand, p_size_vs_rand = stats.ttest_rel(drop_donor_size, drop_donor_rand)

    donor_size_better_than_zero = (drop_donor_size.mean() < drop_zero.mean()) and (p_zero_vs_size < 0.05)
    donor_rand_better_than_zero = (drop_donor_rand.mean() < drop_zero.mean()) and (p_zero_vs_rand < 0.05)
    donor_size_worse_than_zero = (drop_donor_size.mean() > drop_zero.mean()) and (p_zero_vs_size < 0.05)
    donor_rand_worse_than_zero = (drop_donor_rand.mean() > drop_zero.mean()) and (p_zero_vs_rand < 0.05)
    size_vs_rand_differ = p_size_vs_rand < 0.05

    if (donor_size_better_than_zero and donor_rand_better_than_zero) and not size_vs_rand_differ:
        decision = "subject_interchangeable_supported"
    elif donor_size_worse_than_zero or donor_rand_worse_than_zero:
        decision = "actively_misleading_supported"
    elif not donor_size_better_than_zero and not donor_rand_better_than_zero:
        decision = "subject_specific_supported"
    else:
        decision = "mixed_inconclusive"

    print(f"\ndonor_size significantly better preserved than zero: {donor_size_better_than_zero}")
    print(f"donor_rand significantly better preserved than zero: {donor_rand_better_than_zero}")
    print(f"donor_size significantly WORSE than zero: {donor_size_worse_than_zero}")
    print(f"donor_rand significantly WORSE than zero: {donor_rand_worse_than_zero}")
    print(f"donor_size vs donor_rand significantly different: {size_vs_rand_differ}")
    print(f"\n=== DECISION: {decision} ===")
    print("NOTE per pre-declared instruction: this decision characterizes a causal phenomenon.")
    print("It is NOT itself a novelty claim -- a separate literature-novelty search is required")
    print("before any mechanism is designed around whichever branch this result supports.")

    summary = {
        "n_subjects": len(records),
        "mean_dice_intact": float(dice_intact_arr.mean()),
        "mean_drop": {"zero": float(drop_zero.mean()), "donor_size": float(drop_donor_size.mean()),
                      "donor_rand": float(drop_donor_rand.mean())},
        "paired_tests": {
            "zero_vs_donor_size": {"t": float(t_zero_vs_size), "p": float(p_zero_vs_size)},
            "zero_vs_donor_rand": {"t": float(t_zero_vs_rand), "p": float(p_zero_vs_rand)},
            "donor_size_vs_donor_rand": {"t": float(t_size_vs_rand), "p": float(p_size_vs_rand)},
        },
        "spearman_by_condition": results_rho,
        "e48_reference_rho": E48_REFERENCE_RHO,
        "decision": decision,
    }
    with open(OUT_DIR / "E58c_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E58c_summary.json")


if __name__ == "__main__":
    main()
