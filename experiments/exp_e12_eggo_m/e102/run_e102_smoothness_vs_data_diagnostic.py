"""
Phase E102: Local Smoothness-vs-Data-Term Diagnostic (graph-cut failure
mechanism check, BEFORE any graph-cut implementation).

CONTEXT: Following E48-E97's formal closure and the augmentation-lever
kills (E99, E100), user proposed investigating alpha-expansion graph-cut
segmentation as a "known method with a documented BraTS failure"
(arXiv/MDPI 2026 paper reporting mean Jaccard ~0.776, 6.2% of cases
below Dice 0.70, small-lesion detection cited as a failure mode).
User's OWN pre-registered gate, explicitly required before any graph-cut
implementation: does this project's OWN network's behavior show the
signature graph-cut failure mechanism -- local spatial-smoothness
pressure overwhelming weak local data evidence specifically around
missed small lesions? If L_smooth >> L_data at missed small-lesion
locations, the mechanism is plausible and worth pursuing. If not, KILL
graph-cuts before implementing them.

Graph-cut's standard energy: E(y) = sum_i D_i(y_i) + lambda * sum_{ij}
V_ij(y_i, y_j) -- D_i is unary (data) evidence, V_ij is pairwise
(smoothness) cost. A missed small lesion under this framework would
occur when weak local D_i is overwhelmed by strong local V_ij pulling
toward the (background-dominated) neighborhood consensus.

NeuroScan's own UNet3D_v5/E48-lineage network does NOT literally
optimize a graph-cut energy, but its receptive-field-driven convolution
+ pooling structure has an ANALOGOUS implicit smoothness bias (nearby
predictions influence each other through shared feature computation).
This diagnostic operationalizes the SAME conceptual decomposition on
the network's actual output:

  D_i (local data evidence) approximated by the pre-sigmoid LOGIT
    magnitude at voxel i -- how strongly the network's own raw output
    argues for foreground/background at that exact location,
    independent of spatial context beyond what's already baked into
    the logit itself.
  V_i (local smoothness pressure) approximated by the DISAGREEMENT
    between voxel i's logit and the MEAN logit of its 6-connected
    neighbors -- how strongly the local neighborhood's consensus would
    pull this voxel's prediction toward a different value. This is a
    direct, standard discrete approximation of the pairwise smoothness
    term's local gradient (a total-variation-style local energy).

For MISSED small-lesion voxels (ground truth foreground, predicted
background, native_size below median) versus CORRECTLY-DETECTED
small-lesion voxels (ground truth foreground, predicted foreground,
same size stratum), compare:
  - D_i: is the raw data evidence itself weak at missed voxels (this
    would NOT support the graph-cut-style story -- it would mean the
    network simply never had good evidence there, a DIFFERENT failure
    mode than "smoothness overwhelmed good evidence")
  - The smoothness-to-data RATIO |neighbor_disagreement| / |D_i|: if
    this ratio is elevated specifically at missed voxels relative to
    correctly-detected ones, that IS the graph-cut-style signature --
    local consensus pressure is disproportionately large relative to
    the (possibly still non-trivial) direct evidence.

Then test whether N_b(x) (E48-style causal necessity) predicts the
MAGNITUDE of this smoothness-dominance signature per subject -- the
user's own proposed bridge between the graph-cut failure mode and this
project's causal machinery.

PRE-DECLARED DECISION RULE:
  GRAPH-CUT MECHANISM PLAUSIBLE (proceed to consider an intervention)
  if: the smoothness-to-data ratio is significantly HIGHER at missed
  small-lesion voxels than at correctly-detected small-lesion voxels
  (paired/grouped comparison, permutation p<0.05), AND this per-subject
  effect correlates with N_b (rho>0.3, p<0.05).
  KILL GRAPH-CUTS (do not implement, do not investigate further) if
  either condition fails -- specifically if missed voxels simply have
  weak RAW data evidence (not a smoothness-vs-data imbalance, but a
  data-evidence-absence problem, which graph-cut-style regularization
  changes would not fix).

NO TRAINING. Uses the same frozen E48-E97 checkpoint (E46 AttnGate_seed0,
UNet3D_v5), same 125 validation subjects, same native_size stratification
convention as every prior phase in this chain.
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


def neighbor_mean_6conn(logits):
    """Mean of 6-connected neighbors for each voxel, REPLICATE-padded at
    borders (F.pad mode='replicate') so border voxels are not spuriously
    pulled toward the volume's opposite face -- torch.roll wraps around,
    which would be a real bug here (lesion voxels are never at the
    absolute image border in this dataset, but correctness matters
    regardless). Computed via 3 separate shifted averages -- a standard
    discrete Laplacian-adjacent quantity, the natural local approximation
    of graph-cut's pairwise smoothness pull."""
    t = torch.from_numpy(logits).unsqueeze(0).unsqueeze(0)
    padded = F.pad(t, (1, 1, 1, 1, 1, 1), mode="replicate")
    shifts = [
        padded[:, :, 2:, 1:-1, 1:-1], padded[:, :, :-2, 1:-1, 1:-1],
        padded[:, :, 1:-1, 2:, 1:-1], padded[:, :, 1:-1, :-2, 1:-1],
        padded[:, :, 1:-1, 1:-1, 2:], padded[:, :, 1:-1, 1:-1, :-2],
    ]
    neighbor_sum = torch.stack(shifts, dim=0).sum(dim=0)
    neighbor_mean = (neighbor_sum / 6.0).squeeze(0).squeeze(0).numpy()
    return neighbor_mean


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
        if native_size > median_size:
            continue  # this diagnostic is specifically about SMALL lesions, per the graph-cut failure mode

        target_bin = (fractional_occupancy_64(seg_binary_native) > 0.5).astype(np.float32)

        with torch.no_grad():
            out = model(image_b)
            probs = out["probs"].squeeze(0).squeeze(0).cpu().numpy()
            # Recover pre-sigmoid logits for the D_i (data evidence) approximation.
            logits = np.log(np.clip(probs, 1e-7, 1 - 1e-7) / np.clip(1 - probs, 1e-7, 1 - 1e-7))

        pred_bin = (probs >= 0.5).astype(np.float32)
        neighbor_mean_logit = neighbor_mean_6conn(logits)
        smoothness_pull = np.abs(logits - neighbor_mean_logit)  # V_i: local disagreement with neighborhood consensus
        data_evidence = np.abs(logits)  # D_i: raw data evidence magnitude

        gt_fg = target_bin > 0.5
        missed_fn = gt_fg & (pred_bin < 0.5)          # false negatives: GT foreground, predicted background
        correct_tp = gt_fg & (pred_bin >= 0.5)         # true positives: GT foreground, predicted foreground

        n_fn = int(missed_fn.sum())
        n_tp = int(correct_tp.sum())
        if n_fn == 0 or n_tp == 0:
            continue  # need both categories present to compare within this subject

        d_fn = data_evidence[missed_fn]
        d_tp = data_evidence[correct_tp]
        s_fn = smoothness_pull[missed_fn]
        s_tp = smoothness_pull[correct_tp]

        ratio_fn = (s_fn / (d_fn + 1e-6)).mean()
        ratio_tp = (s_tp / (d_tp + 1e-6)).mean()

        records.append({
            "subject_id": subject_id,
            "native_size": native_size,
            "N_b": e48_by_id[subject_id]["drop"],
            "n_fn_voxels": n_fn, "n_tp_voxels": n_tp,
            "mean_data_evidence_fn": float(d_fn.mean()), "mean_data_evidence_tp": float(d_tp.mean()),
            "mean_smoothness_pull_fn": float(s_fn.mean()), "mean_smoothness_pull_tp": float(s_tp.mean()),
            "smooth_to_data_ratio_fn": float(ratio_fn), "smooth_to_data_ratio_tp": float(ratio_tp),
            "ratio_diff_fn_minus_tp": float(ratio_fn - ratio_tp),
        })

        if len(records) % 10 == 0:
            print(f"  processed {len(records)} small-lesion subjects", flush=True)

    with open(OUT_DIR / "E102_smoothness_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records (small-lesion subjects with both FN and TP voxels present).", flush=True)

    n = len(records)
    d_fn = np.array([r["mean_data_evidence_fn"] for r in records])
    d_tp = np.array([r["mean_data_evidence_tp"] for r in records])
    ratio_fn = np.array([r["smooth_to_data_ratio_fn"] for r in records])
    ratio_tp = np.array([r["smooth_to_data_ratio_tp"] for r in records])
    ratio_diff = ratio_fn - ratio_tp
    n_b = np.array([r["N_b"] for r in records])

    print(f"\n=== E102 Smoothness-vs-Data Diagnostic: n={n} small-lesion subjects ===")
    print(f"Mean raw data evidence: FN={d_fn.mean():.4f}, TP={d_tp.mean():.4f} "
          f"({'FN evidence weaker (data-absence story)' if d_fn.mean() < d_tp.mean() else 'comparable/FN not weaker'})")
    print(f"Mean smoothness/data ratio: FN={ratio_fn.mean():.4f}, TP={ratio_tp.mean():.4f}")
    print(f"Mean ratio_diff (FN - TP) = {ratio_diff.mean():+.4f}")

    # Sign-flip permutation test: is ratio_diff > 0 (smoothness dominance higher at missed voxels)?
    rng = np.random.default_rng(SEED)
    observed = ratio_diff.mean()
    perm_means = np.empty(N_PERM)
    for i in range(N_PERM):
        signs = rng.choice([-1, 1], size=n)
        perm_means[i] = (ratio_diff * signs).mean()
    p_ratio = float((perm_means >= observed).mean())  # one-sided: ratio_diff > 0
    print(f"\nSign-flip permutation test (ratio_diff > 0): p={p_ratio:.4f}")

    rho_nb, p_nb_param = stats.spearmanr(n_b, ratio_diff)
    perm_rhos = np.empty(N_PERM)
    rng2 = np.random.default_rng(SEED + 1)
    for i in range(N_PERM):
        perm_rhos[i], _ = stats.spearmanr(n_b, rng2.permutation(ratio_diff))
    p_nb_perm = float((np.abs(perm_rhos) >= np.abs(rho_nb)).mean())
    print(f"Spearman(N_b, ratio_diff) = {rho_nb:+.4f} (parametric p={p_nb_param:.4e}, permutation p={p_nb_perm:.4f})")

    ratio_elevated = (observed > 0.05) and (p_ratio < 0.05)
    nb_correlates = (abs(rho_nb) > 0.3) and (p_nb_perm < 0.05)
    data_absence_dominant = d_fn.mean() < 0.7 * d_tp.mean()  # FN evidence meaningfully weaker, not just noisier

    decision = "GRAPH_CUT_MECHANISM_PLAUSIBLE" if (ratio_elevated and nb_correlates) else "KILL_GRAPH_CUTS"

    print(f"\n=== DECISION: {decision} ===")
    if decision == "GRAPH_CUT_MECHANISM_PLAUSIBLE":
        print("Local smoothness pressure is significantly elevated relative to data evidence specifically at")
        print("missed small-lesion voxels, AND this effect correlates with N_b. The graph-cut-style failure")
        print("mechanism (smoothness overwhelming weak evidence) is plausible in THIS network's behavior too,")
        print("and connects to the causal necessity signal -- worth designing a causally-conditioned")
        print("regularization-relaxation intervention.")
    else:
        print("Either the smoothness-dominance signature is not significantly elevated at missed voxels,")
        print("or it doesn't correlate with N_b. Per the pre-declared rule: KILL graph-cut investigation")
        print("before any implementation.")
        if data_absence_dominant:
            print(f"NOTE: raw data evidence at missed voxels (mean={d_fn.mean():.4f}) is substantially weaker")
            print(f"than at correctly-detected voxels (mean={d_tp.mean():.4f}) -- suggests the real failure mode")
            print("is DATA-EVIDENCE ABSENCE, not smoothness-vs-data imbalance. A regularization-relaxation fix")
            print("would not address this -- the network simply lacks positive evidence there in the first place,")
            print("consistent with E90-E92's own information-loss finding (the evidence was never encoded).")

    summary = {
        "checkpoint_val_dice": ckpt.get("best_val_dice"),
        "n_subjects": n,
        "mean_data_evidence_fn": float(d_fn.mean()), "mean_data_evidence_tp": float(d_tp.mean()),
        "mean_smooth_to_data_ratio_fn": float(ratio_fn.mean()), "mean_smooth_to_data_ratio_tp": float(ratio_tp.mean()),
        "mean_ratio_diff": float(observed), "ratio_diff_permutation_p": p_ratio,
        "rho_Nb_ratio_diff": float(rho_nb), "p_Nb_ratio_diff_param": float(p_nb_param), "p_Nb_ratio_diff_perm": p_nb_perm,
        "ratio_elevated": bool(ratio_elevated), "nb_correlates": bool(nb_correlates),
        "data_absence_dominant": bool(data_absence_dominant),
        "decision": decision,
    }
    with open(OUT_DIR / "E102_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E102_summary.json")


if __name__ == "__main__":
    main()
