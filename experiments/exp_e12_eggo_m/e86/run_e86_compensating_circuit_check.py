"""
Phase E86: Compensating-Circuit / Superadditivity Check on the Bottleneck.

CONTEXT: E85 found that the trained model's gradient allocation to the
bottleneck does NOT track E48's causal necessity measurement (rho=0.064,
p=0.475) -- a real, structured mismatch, motivating a training-dynamics
allocation-correction mechanism as the next design step (see
phase_e48_necessity_allocation_mismatch_reframe memory).

Before designing that mechanism, a prior-art check surfaced "Hidden
Heroes and Gradient Bloats" (arXiv 2602.01442, May 2026), which finds --
independently, in transformers on algorithmic tasks -- that gradient
attribution systematically diverges from causal importance BECAUSE of
REDUNDANT, COMPENSATING CIRCUITS: components whose individual ablation
looks harmless but whose JOINT ablation causes disproportionate
(14x-superadditive) damage, because other components compensate for each
one individually but cannot compensate for all simultaneously.

THIS PHASE tests whether the SAME confound applies to E48's bottleneck-
necessity measurement specifically: E48 only ever ablated the bottleneck
ALONE. If the attn_gate1 skip-connection pathway (enc1, gated by psi)
partially COMPENSATES for a severed bottleneck in some subjects, then
E48's per-subject "necessity" number may be partly an artifact of
how much compensation was AVAILABLE that subject, not a pure measure of
the bottleneck's own causal contribution -- which would mean designing
an allocation-correction mechanism straight from E48's raw N_b(x) values
risks fighting compensating-circuit dynamics rather than fixing a real
allocation gap (a more precise version of why E49/CCABA's forced
amplification may have failed).

INTERVENTIONS (reusing E48's bottleneck-zeroing and E47's psi-clamp
exactly, verified against their own sanity checks):
  (a) Bottleneck ablation ALONE (identical to E48: zero the bottleneck
      tensor before upconv3; the attn_gate1 skip pathway is left fully
      intact and free to compensate via its own learned psi).
  (b) Skip-compensation-disabled ALONE (clamp psi=1 EVERYWHERE, i.e.
      disable the learned gate entirely -- forces raw v3-style routing,
      removing the model's ability to lean on a bottleneck-conditioned
      gate as a compensation channel, while the bottleneck itself stays
      fully intact and reaches the decoder normally via upconv3).
  (c) JOINT: both (a) and (b) simultaneously -- bottleneck zeroed AND
      psi clamped to 1 everywhere.

SUPERADDITIVITY METRIC (mirrors arXiv 2602.01442's Table 4 exactly):
  individual_drop_sum(x) = drop_a(x) + drop_b(x)
  joint_drop(x)           = drop_c(x)
  superadditivity_ratio(x) = joint_drop(x) / individual_drop_sum(x)
      (ratio >> 1 indicates compensating-circuit structure: the pathways
      cover for each other individually but not jointly, exactly as
      arXiv 2602.01442 found for transformer components)

PRE-DECLARED DECISION RULE:
  COMPENSATION CONFOUND PRESENT (E48's N_b(x) is contaminated by
  available compensation, do NOT use it directly to design an allocation
  fix without correcting for this) if:
    1. median superadditivity_ratio > 1.5 (joint damage meaningfully
       exceeds the sum of individual damages), AND
    2. this median ratio is significantly > 1 by a one-sided Wilcoxon
       signed-rank test (p < 0.05) on log(joint_drop) - log(individual
       drop_sum) across subjects with well-defined ratios (both drops
       > 0).
  NO COMPENSATION CONFOUND (E48's N_b(x) is a clean, additive measure of
  bottleneck-specific necessity; proceed to design the allocation
  correction directly from E85's N_b/A_b values) if superadditivity is
  not detected under the above rule.
  AMBIGUOUS (investigate further before designing anything) if median
  ratio is elevated but not significant, or if too many subjects have
  degenerate (near-zero) individual drops to compute a stable ratio.

NO TRAINING. Same checkpoint, same subjects, same dataset ordering as
E47/E48/E85 (E46 AttnGate_seed0 best.pth, UNet3D_v5).
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

CKPT_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs" / "AttnGate_seed0" / "checkpoints" / "best.pth"
E48_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"


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


def forward_with_interventions(model, image, ablate_bottleneck, clamp_psi_full, device):
    """Runs the v5 trunk manually (same verified-bit-for-bit-identical
    pattern as E47/E48's own manual trunks). ablate_bottleneck: zero the
    bottleneck tensor before upconv3 (E48's exact intervention).
    clamp_psi_full: force psi=1.0 EVERYWHERE (E47's psi-clamp, applied to
    the full volume rather than a boundary/interior subset -- this is the
    'disable the compensation channel entirely' condition)."""
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)

        if ablate_bottleneck:
            bottleneck = torch.zeros_like(bottleneck)

        upconv3 = model.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = model.dec3(cat3)

        upconv2 = model.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = model.dec2(cat2)

        upconv1 = model.upconv1(dec2)

        gate = bottleneck  # reads the (possibly ablated) bottleneck, same as E48's convention
        skip = enc1
        g = model.attn_gate1.W_g(gate)
        g_up = F.interpolate(g, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = model.attn_gate1.W_x(skip)
        psi = torch.sigmoid(model.attn_gate1.W_psi(F.relu(g_up + x)))

        if clamp_psi_full:
            psi = torch.ones_like(psi)

        enc1_gated = enc1 * psi
        cat1 = torch.cat([upconv1, enc1_gated], dim=1)
        dec1 = model.dec1(cat1)

        probs = model.seg_head(dec1)
        return probs.squeeze(0).squeeze(0).cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}
    print(f"Loaded {len(e48_records)} E48 subject records (for cross-reference/sanity check).", flush=True)

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded E46/E48/E85 checkpoint: best_val_dice={ckpt.get('best_val_dice')}", flush=True)

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    print(f"Validation set size: {len(val_dataset)}", flush=True)

    # Sanity check: (ablate=False, clamp=False) must reproduce real forward() exactly.
    image0, _, _ = val_dataset[0]
    image0_b = image0.unsqueeze(0).to(device)
    with torch.no_grad():
        real = model(image0_b)["probs"].squeeze(0).squeeze(0).cpu().numpy()
    manual = forward_with_interventions(model, image0_b, False, False, device)
    max_diff = float(np.abs(real - manual).max())
    print(f"Sanity check (both interventions off vs real forward): max abs diff = {max_diff:.6e}", flush=True)
    assert max_diff == 0.0, "Manual trunk reimplementation does not match real forward() -- STOP, bug present."

    records = []
    missing_from_e48 = 0
    for subject_idx in range(len(val_dataset)):
        image, mask, subject_id = val_dataset[subject_idx]

        if subject_id not in e48_by_id:
            missing_from_e48 += 1
            continue

        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_dataset.subject_dirs[subject_idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)

        mask_frac_64 = fractional_occupancy_64(seg_binary_native)
        target_bin = (mask_frac_64 > 0.5).astype(np.float32)

        probs_intact = forward_with_interventions(model, image_b, False, False, device)
        probs_a = forward_with_interventions(model, image_b, True, False, device)   # bottleneck ablated alone
        probs_b = forward_with_interventions(model, image_b, False, True, device)   # psi clamp alone
        probs_c = forward_with_interventions(model, image_b, True, True, device)    # joint

        dice_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin)
        dice_a = dice_score((probs_a >= 0.5).astype(np.float32), target_bin)
        dice_b = dice_score((probs_b >= 0.5).astype(np.float32), target_bin)
        dice_c = dice_score((probs_c >= 0.5).astype(np.float32), target_bin)

        drop_a = dice_intact - dice_a
        drop_b = dice_intact - dice_b
        drop_c = dice_intact - dice_c
        individual_sum = drop_a + drop_b

        e48_drop = e48_by_id[subject_id]["drop"]

        records.append({
            "subject_id": subject_id,
            "dice_intact": dice_intact,
            "drop_bottleneck_alone": drop_a,
            "drop_psi_clamp_alone": drop_b,
            "drop_joint": drop_c,
            "individual_drop_sum": individual_sum,
            "e48_drop_crossref": e48_drop,  # should closely match drop_bottleneck_alone (same intervention as E48)
        })

        if (len(records)) % 25 == 0:
            print(f"  processed {len(records)} subjects", flush=True)

    if missing_from_e48 > 0:
        print(f"WARNING: {missing_from_e48} subjects not found in E48's table, excluded.", flush=True)

    with open(OUT_DIR / "E86_compensating_circuit_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.", flush=True)

    # ================= Cross-reference sanity check =================
    drop_a_arr = np.array([r["drop_bottleneck_alone"] for r in records])
    e48_arr = np.array([r["e48_drop_crossref"] for r in records])
    max_ref_diff = float(np.abs(drop_a_arr - e48_arr).max())
    print(f"\nCross-reference check: max|drop_bottleneck_alone - E48's own drop| = {max_ref_diff:.6e} "
          f"(should be ~0 -- same intervention, same checkpoint, same subjects)")

    # ================= Superadditivity analysis =================
    drop_b_arr = np.array([r["drop_psi_clamp_alone"] for r in records])
    drop_c_arr = np.array([r["drop_joint"] for r in records])
    individual_sum_arr = np.array([r["individual_drop_sum"] for r in records])

    print(f"\n=== E86 Compensating-Circuit Check: n={len(records)} subjects ===")
    print(f"Mean drop_bottleneck_alone = {drop_a_arr.mean():.4f}")
    print(f"Mean drop_psi_clamp_alone  = {drop_b_arr.mean():.4f}")
    print(f"Mean drop_joint            = {drop_c_arr.mean():.4f}")
    print(f"Mean individual_drop_sum   = {individual_sum_arr.mean():.4f}")

    # Well-defined ratio requires both individual drops to be meaningfully positive
    valid = (drop_a_arr > 1e-4) & (drop_b_arr > 1e-4) & (individual_sum_arr > 1e-4)
    n_valid = int(valid.sum())
    print(f"\nSubjects with well-defined superadditivity ratio (both individual drops > 1e-4): {n_valid}/{len(records)}")

    if n_valid >= 10:
        ratio = drop_c_arr[valid] / individual_sum_arr[valid]
        median_ratio = float(np.median(ratio))
        print(f"Median superadditivity_ratio (joint / individual_sum) = {median_ratio:.3f}")
        print(f"Mean superadditivity_ratio = {float(ratio.mean()):.3f}, "
              f"IQR = [{float(np.percentile(ratio,25)):.3f}, {float(np.percentile(ratio,75)):.3f}]")

        # One-sided Wilcoxon signed-rank test: log(joint) - log(individual_sum) > 0
        log_diff = np.log(drop_c_arr[valid] + 1e-6) - np.log(individual_sum_arr[valid] + 1e-6)
        try:
            stat, p_wilcoxon = stats.wilcoxon(log_diff, alternative="greater")
        except ValueError:
            stat, p_wilcoxon = float("nan"), float("nan")
        print(f"Wilcoxon signed-rank test (log_diff > 0, one-sided): stat={stat}, p={p_wilcoxon:.4f}")

        if median_ratio > 1.5 and p_wilcoxon < 0.05:
            decision = "COMPENSATION_CONFOUND_PRESENT"
        elif median_ratio <= 1.5 or p_wilcoxon >= 0.05:
            decision = "NO_COMPENSATION_CONFOUND"
        else:
            decision = "AMBIGUOUS"
    else:
        median_ratio, p_wilcoxon = float("nan"), float("nan")
        decision = "AMBIGUOUS"
        print("Too few subjects with well-defined individual drops to compute a stable ratio.")

    print(f"\n=== DECISION: {decision} ===")
    if decision == "COMPENSATION_CONFOUND_PRESENT":
        print("Joint ablation causes disproportionately more damage than the sum of individual")
        print("ablations predicts -- the bottleneck and skip-gate pathways compensate for each other")
        print("individually. E48's N_b(x) (bottleneck-alone drop) likely UNDERSTATES true necessity")
        print("in subjects where compensation was available, and OVERSTATES it as a 'pure' bottleneck")
        print("signal. Do NOT feed E85's raw N_b(x) directly into an allocation-correction mechanism")
        print("without first correcting for available compensation.")
    elif decision == "NO_COMPENSATION_CONFOUND":
        print("No significant superadditivity detected. E48's bottleneck-alone necessity measurement")
        print("is not meaningfully contaminated by skip-pathway compensation. E85's N_b(x)/A_b(x)")
        print("mismatch can be used directly to design the allocation-correction mechanism.")
    else:
        print("Ambiguous result -- investigate further (e.g. subject-level inspection of degenerate")
        print("drops) before using E85's raw values to design an intervention.")

    summary = {
        "n_subjects": len(records),
        "cross_reference_max_diff": max_ref_diff,
        "mean_drop_bottleneck_alone": float(drop_a_arr.mean()),
        "mean_drop_psi_clamp_alone": float(drop_b_arr.mean()),
        "mean_drop_joint": float(drop_c_arr.mean()),
        "n_valid_for_ratio": n_valid,
        "median_superadditivity_ratio": median_ratio,
        "wilcoxon_p_value": p_wilcoxon,
        "decision": decision,
    }
    with open(OUT_DIR / "E86_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E86_summary.json")


if __name__ == "__main__":
    main()
