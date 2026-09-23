"""
Phase E88: Temporal Necessity-Allocation Audit.

CONTEXT: E85 found rho(N_b, A_b) ~ 0.06 at a single checkpoint (E46's
final best.pth) -- a CROSS-SECTIONAL dissociation between causal
necessity (E48-style bottleneck ablation) and gradient allocation.
User-identified correction to the E85/E87 interpretation: a
near-zero cross-sectional correlation does NOT establish that allocation
is "wrong" or "suboptimal" -- it only shows the two are not statically
aligned at one point in training. The optimizer could still be TRACKING
necessity dynamically (responding to CHANGES in necessity as training
proceeds) even if the two are cross-sectionally uncorrelated at any
single snapshot. Also flagged: gradient-norm allocation A_b(x) may not
capture "functional consequence of the update" -- the same
gradient-magnitude-vs-useful-effect gap this project's own literature
audit found in Wu et al.'s RL work (see project2_candidate_b_survives
_audit memory). This phase addresses BOTH concerns in one design.

THIS PHASE re-measures N_b(x,t) and A_b(x,t) at MULTIPLE checkpoints
across E46's actual training trajectory (epochs 1, 5, 10, 15, 20, 25,
30 -- all real, already-trained checkpoints, NO NEW TRAINING), then asks
THREE questions instead of one:

  Q1 (cross-sectional, per checkpoint): rho_t = corr(N_b(x,t), A_b(x,t))
     -- replicates E85's single-checkpoint test at every checkpoint, to
     see if the null is stable across training or checkpoint-specific.

  Q2 (temporal/dynamic, the user's proposed test): does the CHANGE in
     allocation track the CHANGE in necessity between consecutive
     checkpoints? rho(Delta N_b, Delta A_b) computed pooling all
     consecutive checkpoint-pairs x subjects. This can be POSITIVE even
     when Q1's cross-sectional rho is near zero -- a real, distinct
     possibility the user is right to insist on testing before
     concluding anything about "misallocation."

  Q3 (validity check on A_b itself, addressing the "wrong quantity"
     concern directly): does gradient-norm allocation A_b(x,t) actually
     predict a FUNCTIONAL consequence -- specifically, does higher A_b
     at checkpoint t predict a LARGER subsequent improvement in that
     subject's own causal necessity-relevant Dice (dice_intact) by
     checkpoint t+1? If A_b has no relationship to actual subsequent
     improvement, that is direct evidence gradient-norm is the wrong
     proxy for "useful allocation" -- not just a caveat, but a testable
     claim.

PRE-DECLARED INTERPRETATION (three cases, as proposed):
  Case 1: rho_cross ~ 0 but rho(Delta N, Delta A) > 0, significant --
    optimizer is NOT statically allocation-matched but DOES track
    changing necessity dynamically. Revises E85's story substantially.
  Case 2: both rho_cross and rho(Delta N, Delta A) ~ 0 --
    the model is not adapting allocation to changing causal necessity
    either statically or dynamically. Strengthens E85's original
    "dissociation" framing (renamed from "under-allocation" per this
    session's correction).
  Case 3: rho(Delta N, Delta A) < 0, significant --
    the optimizer systematically reallocates AWAY from pathways as they
    become more necessary. A stronger, more interesting phenomenon than
    plain dissociation, would need its own follow-up.
  Q3 result interpreted separately: if A_b does not predict subsequent
  functional improvement, gradient-norm allocation should be considered
  a validity-questionable proxy regardless of which of cases 1-3 holds
  for it, and any future work should seek a better allocation metric
  before designing an intervention on top of this one.

NO TRAINING. Uses E46's real checkpoint sequence (7 checkpoints:
epoch_1, epoch_5, epoch_10, epoch_15, epoch_20, epoch_25, epoch_30/best)
and the same 125 validation subjects as E48/E85/E86/E87 (matched by
subject_id).
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

CKPT_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs" / "AttnGate_seed0" / "checkpoints"
CHECKPOINT_FILES = ["epoch_1.pth", "epoch_5.pth", "epoch_10.pth", "epoch_15.pth",
                     "epoch_20.pth", "epoch_25.pth", "epoch_30.pth"]


def dice_loss(probs, target_bin):
    probs_flat = probs.reshape(-1)
    target_flat = target_bin.reshape(-1)
    intersection = (probs_flat * target_flat).sum()
    denom = probs_flat.sum() + target_flat.sum()
    return 1.0 - (2.0 * intersection + 1e-6) / (denom + 1e-6)


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


def forward_with_bottleneck_ablation(model, image, ablate, device):
    """Identical to E48/E86's own verified manual trunk."""
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)

        if ablate:
            bottleneck = torch.zeros_like(bottleneck)

        upconv3 = model.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = model.dec3(cat3)

        upconv2 = model.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = model.dec2(cat2)

        upconv1 = model.upconv1(dec2)

        gate = bottleneck
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


def bottleneck_param_names(model):
    return [name for name, _ in model.named_parameters() if name.startswith("bottleneck.")]


def compute_allocation(model, image_b, target_bin_t, bottleneck_names, device):
    model.zero_grad(set_to_none=True)
    out = model(image_b)
    probs = out["probs"]
    loss = dice_loss(probs, target_bin_t)
    loss.backward()

    bottleneck_sq_sum = 0.0
    total_sq_sum = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g_sq = float((p.grad ** 2).sum().item())
        total_sq_sum += g_sq
        if name in bottleneck_names:
            bottleneck_sq_sum += g_sq

    model.zero_grad(set_to_none=True)
    if total_sq_sum <= 0:
        return 0.0
    return (bottleneck_sq_sum ** 0.5) / (total_sq_sum ** 0.5)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    print(f"Validation set size: {len(val_dataset)}", flush=True)

    # Pre-load all subjects' images/targets once (reused across all 7 checkpoints).
    subjects = []
    for subject_idx in range(len(val_dataset)):
        image, mask, subject_id = val_dataset[subject_idx]
        subject_dir = val_dataset.subject_dirs[subject_idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        mask_frac_64 = fractional_occupancy_64(seg_binary_native)
        target_bin = (mask_frac_64 > 0.5).astype(np.float32)
        subjects.append({
            "subject_id": subject_id, "image": image, "target_bin": target_bin,
        })
    print(f"Pre-loaded {len(subjects)} subjects.", flush=True)

    all_records = {}  # checkpoint_name -> list of per-subject records
    for ckpt_file in CHECKPOINT_FILES:
        ckpt_path = CKPT_DIR / ckpt_file
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(ckpt["model_state"])
        bottleneck_names = bottleneck_param_names(model)

        print(f"\nProcessing checkpoint {ckpt_file} (val_dice={ckpt.get('best_val_dice', 'n/a')})...", flush=True)

        records = []
        for s in subjects:
            image_b = s["image"].unsqueeze(0).to(device)
            target_bin = s["target_bin"]
            target_bin_t = torch.from_numpy(target_bin).unsqueeze(0).unsqueeze(0).to(device)

            model.eval()
            probs_intact = forward_with_bottleneck_ablation(model, image_b, ablate=False, device=device)
            probs_ablated = forward_with_bottleneck_ablation(model, image_b, ablate=True, device=device)
            dice_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin)
            dice_ablated = dice_score((probs_ablated >= 0.5).astype(np.float32), target_bin)
            n_b = dice_intact - dice_ablated

            model.train()
            a_b = compute_allocation(model, image_b, target_bin_t, bottleneck_names, device)

            records.append({
                "subject_id": s["subject_id"], "N_b": n_b, "A_b": a_b, "dice_intact": dice_intact,
            })

        all_records[ckpt_file] = records
        rho, p = stats.spearmanr([r["N_b"] for r in records], [r["A_b"] for r in records])
        print(f"  {ckpt_file}: cross-sectional rho(N_b, A_b) = {rho:+.4f} (p={p:.4e}), "
              f"mean N_b={np.mean([r['N_b'] for r in records]):.4f}, "
              f"mean dice_intact={np.mean([r['dice_intact'] for r in records]):.4f}")

    with open(OUT_DIR / "E88_temporal_table.json", "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nSaved temporal table across {len(CHECKPOINT_FILES)} checkpoints.", flush=True)

    # ================= Q1: cross-sectional rho per checkpoint =================
    print("\n=== Q1: Cross-sectional rho(N_b, A_b) per checkpoint ===")
    q1_results = {}
    for ckpt_file in CHECKPOINT_FILES:
        recs = all_records[ckpt_file]
        rho, p = stats.spearmanr([r["N_b"] for r in recs], [r["A_b"] for r in recs])
        q1_results[ckpt_file] = {"rho": float(rho), "p": float(p)}
        print(f"  {ckpt_file}: rho={rho:+.4f}, p={p:.4e}")

    # ================= Q2: temporal (delta) correlation =================
    print("\n=== Q2: Temporal rho(Delta N_b, Delta A_b), pooled across consecutive checkpoint pairs ===")
    delta_n, delta_a = [], []
    by_subject = {}
    for ckpt_file in CHECKPOINT_FILES:
        for r in all_records[ckpt_file]:
            by_subject.setdefault(r["subject_id"], {})[ckpt_file] = r

    for i in range(len(CHECKPOINT_FILES) - 1):
        ck_t, ck_t1 = CHECKPOINT_FILES[i], CHECKPOINT_FILES[i + 1]
        for sid, per_ckpt in by_subject.items():
            if ck_t in per_ckpt and ck_t1 in per_ckpt:
                delta_n.append(per_ckpt[ck_t1]["N_b"] - per_ckpt[ck_t]["N_b"])
                delta_a.append(per_ckpt[ck_t1]["A_b"] - per_ckpt[ck_t]["A_b"])

    delta_n = np.array(delta_n)
    delta_a = np.array(delta_a)
    rho_delta, p_delta = stats.spearmanr(delta_n, delta_a)
    print(f"n_pairs={len(delta_n)} (subjects x consecutive-checkpoint-transitions)")
    print(f"rho(Delta N_b, Delta A_b) = {rho_delta:+.4f} (p={p_delta:.4e})")

    rng = np.random.default_rng(SEED)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_delta_a = rng.permutation(delta_a)
        perm_rhos[i], _ = stats.spearmanr(delta_n, perm_delta_a)
    p_perm_delta = float((np.abs(perm_rhos) >= np.abs(rho_delta)).mean())
    print(f"Permutation test ({N_PERM} trials): p={p_perm_delta:.4f}")

    # ================= Q3: does A_b predict subsequent functional improvement? =================
    print("\n=== Q3: Does A_b(x,t) predict subsequent improvement in dice_intact(x,t+1)? ===")
    a_b_t, subsequent_dice_gain = [], []
    for i in range(len(CHECKPOINT_FILES) - 1):
        ck_t, ck_t1 = CHECKPOINT_FILES[i], CHECKPOINT_FILES[i + 1]
        for sid, per_ckpt in by_subject.items():
            if ck_t in per_ckpt and ck_t1 in per_ckpt:
                a_b_t.append(per_ckpt[ck_t]["A_b"])
                subsequent_dice_gain.append(per_ckpt[ck_t1]["dice_intact"] - per_ckpt[ck_t]["dice_intact"])

    a_b_t = np.array(a_b_t)
    subsequent_dice_gain = np.array(subsequent_dice_gain)
    rho_q3, p_q3 = stats.spearmanr(a_b_t, subsequent_dice_gain)
    print(f"rho(A_b(t), dice_intact(t+1) - dice_intact(t)) = {rho_q3:+.4f} (p={p_q3:.4e})")
    print("(If this is near zero/non-significant, gradient-norm allocation does not predict",
          "actual subsequent functional improvement -- direct evidence it may be the wrong proxy.)")

    # ================= Decision =================
    case = "UNDETERMINED"
    if abs(rho_delta) < 0.1 or p_perm_delta >= 0.05:
        case = "CASE_2_NEITHER_STATIC_NOR_DYNAMIC_TRACKING"
    elif rho_delta > 0.1 and p_perm_delta < 0.05:
        case = "CASE_1_DYNAMIC_TRACKING_DESPITE_STATIC_NULL"
    elif rho_delta < -0.1 and p_perm_delta < 0.05:
        case = "CASE_3_SYSTEMATIC_REALLOCATION_AWAY_FROM_NECESSITY"

    print(f"\n=== DECISION: {case} ===")
    a_b_validity = "QUESTIONABLE" if abs(rho_q3) < 0.1 or p_q3 >= 0.05 else "SUPPORTED"
    print(f"A_b construct validity (Q3): {a_b_validity} "
          f"(rho={rho_q3:+.4f}, p={p_q3:.4e} for A_b(t) predicting subsequent dice gain)")

    summary = {
        "q1_cross_sectional_per_checkpoint": q1_results,
        "q2_n_pairs": len(delta_n),
        "q2_rho_delta_N_delta_A": float(rho_delta),
        "q2_p_parametric": float(p_delta),
        "q2_p_permutation": p_perm_delta,
        "q3_rho_Ab_vs_subsequent_dice_gain": float(rho_q3),
        "q3_p": float(p_q3),
        "a_b_construct_validity": a_b_validity,
        "case": case,
    }
    with open(OUT_DIR / "E88_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E88_summary.json")


if __name__ == "__main__":
    main()
