"""
Phase E85: Necessity vs. Allocation Mismatch Diagnostic.

CONTEXT: E48 found (causally, via full bottleneck ablation) that SMALL
lesions depend MORE on the bottleneck than large lesions (Spearman
rho(native_size, drop) = -0.454, p<1e-7) -- the opposite of the originally
hypothesized direction, but a real, strong, significant effect. E49
(CCABA) built a bottleneck-amplification gate explicitly conditioned on
this finding and only reached a 3-seed mean Dice of 0.9094 (true effect
~+0.31pp vs D4-only, CI crossing zero) -- i.e. directly increasing the
bottleneck's forced contribution did NOT reliably improve Dice, despite
the bottleneck being causally necessary.

THIS PHASE asks a different, more precise question than "amplify harder":
does the TRAINED model's actual allocation of optimization capacity to
the bottleneck pathway track its measured causal NECESSITY on a
per-subject basis? These are conceptually distinct:

  N_b(x) = causal necessity of the bottleneck for subject x
           (E48's existing per-subject `drop` value: dice_intact -
           dice_ablated when the bottleneck is fully zeroed)

  A_b(x) = actual training/optimization allocation to the bottleneck for
           subject x (this phase's new measurement: the fraction of
           total-model gradient L2 norm attributable to the bottleneck
           block's parameters, computed via a single backward pass of
           the segmentation loss on that subject, on the SAME E48
           checkpoint N_b(x) was measured on -- no new training,
           no architecture change)

A pathway can be causally necessary (ablating it destroys information)
while the trained model still fails to allocate more optimization
attention to it where it matters most -- necessity != allocation.

HYPOTHESIS (falsifiable, pre-declared):
  If the model's allocation already tracks necessity reasonably well,
  Spearman(N_b(x), A_b(x)) should be POSITIVE and non-trivial in
  magnitude. In that case, CCABA's failure to convert necessity into a
  Dice gain is NOT explained by an allocation mismatch -- the necessity
  finding is real but not actionable through allocation-style
  interventions, and this diagnostic line should CLOSE.

  If allocation does NOT track necessity (weak/near-zero/negative
  correlation), that is a real, structured, previously unmeasured gap:
  the model under- or over-allocates gradient attention to the
  bottleneck independent of how much that subject actually needs it --
  which would motivate a genuinely new (not "steeper CCABA") allocation-
  correction mechanism as the next design step.

PRE-DECLARED DECISION RULE:
  MISMATCH CONFIRMED (proceed to design an allocation-correction
  mechanism) only if:
    1. |Spearman rho(N_b, A_b)| < 0.3 (weak-to-no tracking), AND
    2. permutation p > 0.05 for that (near-)null correlation (i.e. we
       cannot reject the no-tracking null), AND
    3. A_b(x) itself has real, non-degenerate variance across subjects
       (sd(A_b) / mean(A_b) > 0.05) -- otherwise a null correlation
       would be a measurement-floor artifact, not a real finding.
  NO MISMATCH (close this line, E48's finding stands as descriptive but
  not allocation-actionable) if rho is positive and non-trivial
  (rho > 0.3) with permutation p < 0.05.
  AMBIGUOUS (do not proceed to design anything) otherwise -- e.g. low
  A_b variance, or a significant but very weak/inconsistent correlation.

NO TRAINING. Loads E48's exact checkpoint (E46's best.pth, UNet3D_v5,
val_dice=0.9102) and the SAME 125 validation subjects E48 used (matched
by subject_id against E48_encoding_audit_table.json, not re-derived, to
guarantee N_b(x) and A_b(x) are computed on identical subjects in
identical order).
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


def dice_loss(probs, target_bin):
    """Soft Dice loss (differentiable), matched to this project's standard
    segmentation objective family -- used only to get a realistic gradient
    signal through the network, not as an evaluation metric (evaluation
    Dice is E48's job, already done)."""
    probs_flat = probs.reshape(-1)
    target_flat = target_bin.reshape(-1)
    intersection = (probs_flat * target_flat).sum()
    denom = probs_flat.sum() + target_flat.sum()
    return 1.0 - (2.0 * intersection + 1e-6) / (denom + 1e-6)


def fractional_occupancy_64(seg_binary_native):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=(64, 64, 64), mode="area").squeeze().numpy()
    return frac


def bottleneck_param_names(model):
    """Bottleneck block's own parameters only (matches E48's definition of
    'the bottleneck' as model.bottleneck -- the same module that was
    zeroed there)."""
    return [name for name, _ in model.named_parameters() if name.startswith("bottleneck.")]


def compute_allocation(model, image_b, target_bin_t, bottleneck_names, device):
    """Single backward pass of the segmentation loss on one subject.
    Returns A_b(x) = ||grad(bottleneck params)||_2 / ||grad(all params)||_2
    -- the fraction of total gradient L2 norm attributable to the
    bottleneck block, i.e. how much optimization 'attention' this subject
    is currently directing there, independent of E48's causal necessity
    measurement (which used a frozen, no-grad forward ablation)."""
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
        return 0.0, 0.0, 0.0
    bottleneck_norm = bottleneck_sq_sum ** 0.5
    total_norm = total_sq_sum ** 0.5
    return bottleneck_norm / total_norm, bottleneck_norm, total_norm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}
    print(f"Loaded {len(e48_records)} E48 subject records (N_b(x) source).", flush=True)

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.train()  # need grad graph; BN/dropout behavior matched to training-time allocation, not eval-time necessity
    print(f"Loaded E48/E46 checkpoint: best_val_dice={ckpt.get('best_val_dice')}", flush=True)

    bottleneck_names = bottleneck_param_names(model)
    print(f"Bottleneck parameter tensors: {bottleneck_names}", flush=True)
    assert len(bottleneck_names) > 0, "No bottleneck.* parameters found -- check model attribute naming."

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    print(f"Validation set size: {len(val_dataset)}", flush=True)

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
        target_bin_t = torch.from_numpy(target_bin).unsqueeze(0).unsqueeze(0).to(device)

        a_b, bottleneck_norm, total_norm = compute_allocation(
            model, image_b, target_bin_t, bottleneck_names, device
        )

        n_b = e48_by_id[subject_id]["drop"]
        native_size = e48_by_id[subject_id]["native_size"]

        records.append({
            "subject_id": subject_id,
            "native_size": native_size,
            "N_b": n_b,
            "A_b": a_b,
            "bottleneck_grad_norm": bottleneck_norm,
            "total_grad_norm": total_norm,
        })

        if (len(records)) % 25 == 0:
            print(f"  processed {len(records)} subjects", flush=True)

    if missing_from_e48 > 0:
        print(f"WARNING: {missing_from_e48} validation subjects not found in E48's table "
              f"(dataset split/order mismatch) -- excluded from this analysis.", flush=True)

    with open(OUT_DIR / "E85_necessity_allocation_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.", flush=True)

    # ================= Statistical analysis =================
    n_b = np.array([r["N_b"] for r in records], dtype=np.float64)
    a_b = np.array([r["A_b"] for r in records], dtype=np.float64)
    native_size = np.array([r["native_size"] for r in records], dtype=np.float64)

    print(f"\n=== E85 Necessity vs Allocation Mismatch: n={len(records)} subjects ===")
    print(f"N_b (necessity, from E48): mean={n_b.mean():.4f}, sd={n_b.std():.4f}")
    print(f"A_b (allocation, new):     mean={a_b.mean():.4f}, sd={a_b.std():.4f}, "
          f"cv={a_b.std()/ (abs(a_b.mean())+1e-12):.4f}")

    rho, p_parametric = stats.spearmanr(n_b, a_b)
    print(f"\nSpearman(N_b, A_b) = {rho:+.4f} (parametric p={p_parametric:.4e})")

    rng = np.random.default_rng(SEED)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_a_b = rng.permutation(a_b)
        perm_rhos[i], _ = stats.spearmanr(n_b, perm_a_b)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())
    print(f"Permutation test ({N_PERM} trials): p={p_perm:.4f}")

    a_b_cv = a_b.std() / (abs(a_b.mean()) + 1e-12)
    a_b_has_variance = a_b_cv > 0.05

    # Also report N_b vs native_size (E48's own relationship) and A_b vs
    # native_size for context -- helps interpret WHY any mismatch exists,
    # not just whether it does.
    rho_nb_size, p_nb_size = stats.spearmanr(native_size, n_b)
    rho_ab_size, p_ab_size = stats.spearmanr(native_size, a_b)
    print(f"\n[context] Spearman(native_size, N_b) = {rho_nb_size:+.4f} (p={p_nb_size:.4e}) "
          f"-- should match E48's own -0.454 finding (sanity check on subject alignment)")
    print(f"[context] Spearman(native_size, A_b) = {rho_ab_size:+.4f} (p={p_ab_size:.4e})")

    if abs(rho) < 0.3 and p_perm > 0.05 and a_b_has_variance:
        decision = "MISMATCH_CONFIRMED"
    elif rho > 0.3 and p_perm < 0.05:
        decision = "NO_MISMATCH"
    else:
        decision = "AMBIGUOUS"

    print(f"\n=== DECISION: {decision} ===")
    if decision == "MISMATCH_CONFIRMED":
        print("Allocation does NOT track necessity: the model does not direct more optimization")
        print("gradient toward the bottleneck for subjects where it is causally more necessary.")
        print("This is a real, structured, previously unmeasured gap -- motivates designing an")
        print("allocation-correction mechanism (NOT another size-conditioned amplification gate).")
    elif decision == "NO_MISMATCH":
        print("Allocation already tracks necessity reasonably well. CCABA's failure to convert")
        print("E48's necessity finding into a Dice gain is NOT explained by an allocation mismatch.")
        print("This diagnostic line should CLOSE -- E48's finding is real but not allocation-actionable.")
    else:
        print("Result is ambiguous under the pre-declared decision rule -- do NOT proceed to design")
        print("any intervention from this result. Investigate A_b's variance/measurement before retrying.")

    summary = {
        "n_subjects": len(records),
        "n_b_mean": float(n_b.mean()), "n_b_sd": float(n_b.std()),
        "a_b_mean": float(a_b.mean()), "a_b_sd": float(a_b.std()), "a_b_cv": float(a_b_cv),
        "spearman_rho_Nb_Ab": float(rho), "parametric_p": float(p_parametric),
        "permutation_p": p_perm, "n_permutations": N_PERM,
        "spearman_rho_size_Nb_sanity_check": float(rho_nb_size),
        "spearman_rho_size_Ab": float(rho_ab_size),
        "decision": decision,
    }
    with open(OUT_DIR / "E85_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E85_summary.json")


if __name__ == "__main__":
    main()
