"""
Phase E58 Stage 2b: cross-subject bottleneck substitution, RE-RUN on
E48's own original checkpoint (E46's v5/attention-gate model), as a
checkpoint-artifact control for Stage 2's own result (run on the plain
v3/D4-only checkpoint instead, to avoid the gate confound for the
SPATIAL-localization questions in Stages 1/1b). Per explicit
instruction: before interpreting Stage 2's "wrong donor is worse than
zero" effect and its apparent non-size-dependence as meaningful, rule
out that it's an artifact of using a different checkpoint/architecture
than E48's own.

METHOD: IDENTICAL protocol to run_e58c_bottleneck_substitution.py --
same donor-selection logic (closest-size donor, random donor, both
seeded identically), same three conditions (zero, donor_size,
donor_rand), same statistics (paired t-test, Wilcoxon, Spearman +
permutation + bootstrap CI) -- the ONLY difference is the checkpoint
(E46/v5 instead of e25/D4-only/v3) and, consequently, the decoder path
must also recompute attn_gate1's own psi from the (zeroed/substituted)
bottleneck, exactly matching E48's OWN established convention (see
E48's own forward_with_bottleneck_ablation: "gate reads the (possibly
ablated) bottleneck too -- correct: if the bottleneck carries no real
signal, the gate itself should also degrade to whatever a zero-input
gate produces, which IS the intended full severing of the coarse
pathway's influence"). This is not a new methodological choice -- it
is the same rule E48 already established, applied consistently to the
donor-substitution case (a wrong-but-plausible bottleneck should also
drive the gate, exactly as a zeroed one does).

If Stage 2's own finding (random donor significantly worse than zero,
p=0.026; effect not size-dependent) REPRODUCES on this checkpoint too,
the checkpoint-artifact explanation is ruled out and the effect can be
treated as a genuine property of this architecture family, not one
specific trained model. If it does NOT reproduce, the original Stage 2
result must be treated as checkpoint-specific and NOT generalized.
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
N_BOOT = 1000

CKPT_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs" / "AttnGate_seed0" / "checkpoints" / "best.pth"

E48_REFERENCE_RHO = -0.454  # E48's own original full-ablation result, same checkpoint family


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
    """v5 decoder path -- includes attn_gate1, which reads the (possibly
    substituted) bottleneck as its own gating signal, matching E48's own
    established convention exactly (see module docstring)."""
    with torch.no_grad():
        upconv3 = model.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = model.dec3(cat3)

        upconv2 = model.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = model.dec2(cat2)

        upconv1 = model.upconv1(dec2)

        gate = bottleneck  # matches E48's own convention exactly
        skip = enc1
        g = model.attn_gate1.W_g(gate)
        g_up = F.interpolate(g, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = model.attn_gate1.W_x(skip)
        psi = torch.sigmoid(model.attn_gate1.W_psi(F.relu(g_up + x)))

        enc1_gated = enc1 * psi
        cat1 = torch.cat([upconv1, enc1_gated], dim=1)
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
    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded E46/v5 checkpoint: best_val_dice={ckpt.get('best_val_dice')}", flush=True)

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n = len(val_dataset)
    print(f"Validation set size: {n}", flush=True)

    # ================= Sanity check 1 =================
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

    # ================= Precompute cache =================
    print("Precomputing bottleneck/skip caches for all subjects...", flush=True)
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

    # ================= Sanity check 2 =================
    sid0 = subject_ids_ordered[0]
    self_sub = forward_from_bottleneck(
        model, cache[sid0]["bottleneck"], cache[sid0]["enc1"], cache[sid0]["enc2"], cache[sid0]["enc3"]
    )
    max_diff2 = float(np.abs(real - self_sub).max())
    print(f"[Sanity check 2] self-substitution vs real forward: max abs diff = {max_diff2:.6e}")
    assert max_diff2 == 0.0, "Self-substitution pathway does not reproduce real forward() -- STOP."
    print("[Sanity check 2] PASS.\n")

    # ================= Donor selection (IDENTICAL logic/seed to run_e58c) =================
    rng = np.random.default_rng(SEED)
    size_donor = {}
    rand_donor = {}
    for sid in subject_ids_ordered:
        others = [s for s in subject_ids_ordered if s != sid]
        size_diffs = [(abs(native_sizes[s] - native_sizes[sid]), s) for s in others]
        size_diffs.sort(key=lambda t: t[0])
        size_donor[sid] = size_diffs[0][1]
        rand_donor[sid] = others[rng.integers(0, len(others))]

    # ================= Main loop =================
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

    with open(OUT_DIR / "E58d_substitution_e48ckpt_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} subject records.\n")

    # ================= Statistics =================
    native_size_arr = np.array([r["native_size"] for r in records], dtype=np.float64)
    dice_intact_arr = np.array([r["dice_intact"] for r in records])
    drop_zero = np.array([r["drop_zero"] for r in records], dtype=np.float64)
    drop_donor_size = np.array([r["drop_donor_size"] for r in records], dtype=np.float64)
    drop_donor_rand = np.array([r["drop_donor_rand"] for r in records], dtype=np.float64)

    print(f"=== E58d Bottleneck Substitution Audit (E48/E46 checkpoint): n={len(records)} subjects ===")
    print(f"Mean dice_intact = {dice_intact_arr.mean():.4f}")
    print(f"Mean drop (zero)        = {drop_zero.mean():.4f}  (E48's own original full-ablation drop = 0.3205)")
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
    print(f"  Reference (E48's own original full ablation, SAME checkpoint): rho = {E48_REFERENCE_RHO}")

    t_zero_vs_size, p_zero_vs_size = stats.ttest_rel(drop_zero, drop_donor_size)
    t_zero_vs_rand, p_zero_vs_rand = stats.ttest_rel(drop_zero, drop_donor_rand)
    t_size_vs_rand, p_size_vs_rand = stats.ttest_rel(drop_donor_size, drop_donor_rand)

    donor_size_worse_than_zero = (drop_donor_size.mean() > drop_zero.mean()) and (p_zero_vs_size < 0.05)
    donor_rand_worse_than_zero = (drop_donor_rand.mean() > drop_zero.mean()) and (p_zero_vs_rand < 0.05)

    print(f"\ndonor_size significantly WORSE than zero: {donor_size_worse_than_zero} (p={p_zero_vs_size:.4f})")
    print(f"donor_rand significantly WORSE than zero: {donor_rand_worse_than_zero} (p={p_zero_vs_rand:.4f})")

    # ================= Checkpoint-artifact control verdict =================
    stage2_reproduces = bool(donor_rand_worse_than_zero)  # Stage 2's own headline finding, on the plain-v3 checkpoint
    print(f"\n=== CHECKPOINT-ARTIFACT CONTROL: Stage 2's finding "
          f"('random donor significantly worse than zero') REPRODUCES on E48's own checkpoint: {stage2_reproduces} ===")
    if stage2_reproduces:
        print("The 'wrong context is actively misleading' effect is NOT a checkpoint-specific artifact --")
        print("it holds on both a plain v3/D4-only model AND E46/E48's own v5/attention-gate model.")
    else:
        print("The effect does NOT reproduce on E48's own checkpoint -- Stage 2's original result must be")
        print("treated as checkpoint-specific, NOT generalized to this architecture family as a whole.")

    summary = {
        "checkpoint": "E46/v5 (E48's own original checkpoint)",
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
        "e48_reference_rho_same_checkpoint": E48_REFERENCE_RHO,
        "stage2_finding_reproduces_on_e48_checkpoint": stage2_reproduces,
    }
    with open(OUT_DIR / "E58d_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E58d_summary.json")


if __name__ == "__main__":
    main()
