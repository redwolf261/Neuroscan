"""
Phase E104: Does This Project's OWN Existing Evidential Head Correctly
Flag Epistemic Evidence-Poverty at Missed Small Lesions?

CONTEXT: E48-E97 established (via multiple independent methods: linear
probing E90-E92, direct evidence-magnitude comparison E102) that small-
lesion information is genuinely UNDER-ENCODED in this architecture, not
merely mis-weighted. Following the closure of graph-cut (E102) and
CC-DiceCE (E103), user proposed investigating evidential/Dirichlet
uncertainty modeling as a candidate -- but with an important, ALREADY-
KNOWN fact this phase makes explicit: NeuroScan's UNet3D_v5 (the EXACT
E48-E97 checkpoint) already implements a real, trained Sensoy-style Beta
evidential head (EvidentialBetaLoss, in production since E45 -- alpha,
beta outputs on every checkpoint since E46/E48's own lineage). This is
NOT a new method to adopt; it is an assumption to AUDIT in a mechanism
already running.

The proposed EDL contradiction test (does the model's own evidential
framework correctly distinguish EPISTEMIC uncertainty -- genuine lack
of evidence -- from ordinary background confidence, specifically for
tiny lesions) can therefore be tested as a ZERO-TRAINING diagnostic on
the SAME frozen E48-E97 checkpoint, exactly like E95/E96/E99/E102 --
NOT a training-dependent audit like E103.

MATHEMATICS (Sensoy et al. 2018 Beta/Dirichlet evidential framework,
matching this project's own EvidentialBetaLoss docstring exactly):
  S = alpha + beta  (total evidence -- higher S = more confident/
      "certain" the model claims to be, regardless of correctness)
  mu = alpha / S    (predictive mean, the actual segmentation probability)
  A LOW S at a given voxel is the model's own admission of epistemic
  uncertainty (genuine lack of evidence) -- the theoretically correct
  behavior for evidence-poor regions.
  A HIGH S at a MISSED small lesion voxel would mean the model is
  CONFIDENTLY WRONG -- the evidential framework's core promise (know
  what you don't know) is violated exactly where E48-E97 already showed
  real information is missing.

THIS PHASE compares, for missed small-lesion voxels (false negatives,
same definition as E102: GT foreground, predicted background, subject's
native_size below median) against TWO reference categories:
  (a) Correctly-classified BACKGROUND voxels (GT background, predicted
      background) -- the "ordinary, uninteresting, correctly confident"
      case EDL should look similar to ONLY if evidence is genuinely
      absent in the same way.
  (b) Correctly-DETECTED small-lesion voxels (GT foreground, predicted
      foreground, same subjects) -- the "genuine positive evidence"
      case, for contrast.

PRE-DECLARED DECISION RULE:
  EDL ASSUMPTION VIOLATED (a real, testable, narrow finding worth
  designing an evidential correction around) if:
    Mean S at missed small-lesion voxels is NOT significantly lower
    than mean S at correctly-classified background voxels (i.e. the
    model is NOT less "confident" at its false negatives than at its
    ordinary correct background predictions) -- specifically, test
    whether S_FN >= S_background (one-sided, permutation p<0.05 for
    the NULL of no difference or S_FN higher). If the model's total
    evidence at missed lesions is statistically indistinguishable from
    (or higher than) its evidence at ordinary background, the
    evidential framework is NOT correctly signaling "I lack evidence
    here" -- it looks the same as ordinary confident background.
  EDL ASSUMPTION HOLDS (kill this candidate -- the model already
  correctly flags epistemic poverty, nothing to fix here) if S_FN is
  significantly LOWER than S_background (the model IS less confident
  at its mistakes than at ordinary correct predictions, exactly as EDL
  theory predicts) -- in this case, the failure is not evidential
  MISCALIBRATION but simply INSUFFICIENT downstream use of a correctly-
  computed uncertainty signal, a DIFFERENT (and separately testable)
  question from what's being asked here.
  Also report S_FN vs S_TP (correctly-detected lesion voxels) for
  context -- if S_FN < S_TP as well (uncertainty correctly separates
  missed vs. found lesions), that's a coherent, working uncertainty
  signal even if it doesn't (yet) drive better segmentation.
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    assert abs(ckpt.get("best_val_dice", 0) - 0.9101624600589275) < 1e-9, \
        "Checkpoint mismatch -- must match E48-E103's exact checkpoint."
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

    all_s_fn, all_s_bg, all_s_tp = [], [], []
    per_subject_records = []

    for subject_idx in range(len(val_dataset)):
        image, mask, subject_id = val_dataset[subject_idx]
        if subject_id not in e48_by_id:
            continue
        native_size = e48_by_id[subject_id]["native_size"]
        if native_size > median_size:
            continue  # small-lesion subjects only, matching E102's own scope

        image_b = image.unsqueeze(0).to(device)
        subject_dir = val_dataset.subject_dirs[subject_idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        target_bin = (fractional_occupancy_64(seg_binary_native) > 0.5).astype(np.float32)

        with torch.no_grad():
            out = model(image_b)
            probs = out["probs"].squeeze(0).squeeze(0).cpu().numpy()
            alpha = out["alpha"].squeeze(0).squeeze(0).cpu().numpy()
            beta = out["beta"].squeeze(0).squeeze(0).cpu().numpy()

        S = alpha + beta  # total evidence, per voxel
        pred_bin = (probs >= 0.5).astype(np.float32)

        gt_fg = target_bin > 0.5
        gt_bg = target_bin <= 0.5
        missed_fn = gt_fg & (pred_bin < 0.5)
        correct_bg = gt_bg & (pred_bin < 0.5)
        correct_tp = gt_fg & (pred_bin >= 0.5)

        n_fn, n_bg, n_tp = int(missed_fn.sum()), int(correct_bg.sum()), int(correct_tp.sum())
        if n_fn == 0:
            continue  # need at least some FN voxels to compare

        s_fn = S[missed_fn]
        s_bg = S[correct_bg] if n_bg > 0 else np.array([])
        s_tp = S[correct_tp] if n_tp > 0 else np.array([])

        all_s_fn.extend(s_fn.tolist())
        if n_bg > 0:
            all_s_bg.extend(s_bg.tolist())
        if n_tp > 0:
            all_s_tp.extend(s_tp.tolist())

        per_subject_records.append({
            "subject_id": subject_id, "native_size": native_size,
            "n_fn": n_fn, "n_bg": n_bg, "n_tp": n_tp,
            "mean_S_fn": float(s_fn.mean()),
            "mean_S_bg": float(s_bg.mean()) if n_bg > 0 else None,
            "mean_S_tp": float(s_tp.mean()) if n_tp > 0 else None,
        })

        if len(per_subject_records) % 10 == 0:
            print(f"  processed {len(per_subject_records)} small-lesion subjects", flush=True)

    with open(OUT_DIR / "E104_per_subject_table.json", "w") as f:
        json.dump(per_subject_records, f, indent=2)
    print(f"\nSaved {len(per_subject_records)} subject records.", flush=True)

    all_s_fn = np.array(all_s_fn)
    all_s_bg = np.array(all_s_bg)
    all_s_tp = np.array(all_s_tp)

    print(f"\n=== E104 Evidential Epistemic-Poverty Check: n_subjects={len(per_subject_records)} ===")
    print(f"Total voxels: FN={len(all_s_fn)}, background={len(all_s_bg)}, TP={len(all_s_tp)}")
    print(f"Mean total evidence S: FN={all_s_fn.mean():.4f}, background={all_s_bg.mean():.4f}, TP={all_s_tp.mean():.4f}")

    # Subsample for permutation test tractability (voxel counts can be huge).
    rng = np.random.default_rng(SEED)
    n_sub = min(20000, len(all_s_fn), len(all_s_bg))
    sub_fn = rng.choice(all_s_fn, size=n_sub, replace=False)
    sub_bg = rng.choice(all_s_bg, size=n_sub, replace=False)

    observed_diff = sub_fn.mean() - sub_bg.mean()
    combined = np.concatenate([sub_fn, sub_bg])
    perm_diffs = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_idx = rng.permutation(len(combined))
        perm_diffs[i] = combined[perm_idx[:n_sub]].mean() - combined[perm_idx[n_sub:]].mean()
    # One-sided: is FN evidence HIGHER than (or equal to) background? (violates EDL if so)
    p_violated = float((perm_diffs >= observed_diff).mean()) if observed_diff > 0 else 1.0
    # Also the two-sided significance of ANY difference, for context.
    p_two_sided = float((np.abs(perm_diffs) >= np.abs(observed_diff)).mean())

    print(f"\nS_FN - S_background = {observed_diff:+.4f}")
    print(f"Two-sided permutation p (any difference) = {p_two_sided:.4f}")

    edl_violated = (observed_diff >= -0.05) and (p_two_sided < 0.05 or abs(observed_diff) < 0.05)
    # Precise rule: violated if FN evidence is NOT meaningfully/significantly LOWER than background.
    edl_holds = (observed_diff < -0.05) and (p_two_sided < 0.05)
    decision = "EDL_ASSUMPTION_VIOLATED" if not edl_holds else "EDL_ASSUMPTION_HOLDS"

    n_sub_tp = min(20000, len(all_s_fn), len(all_s_tp))
    if n_sub_tp > 0:
        sub_fn2 = rng.choice(all_s_fn, size=n_sub_tp, replace=False)
        sub_tp = rng.choice(all_s_tp, size=n_sub_tp, replace=False)
        diff_fn_tp = sub_fn2.mean() - sub_tp.mean()
        print(f"S_FN - S_TP = {diff_fn_tp:+.4f} (context: does uncertainty separate missed vs found lesions?)")

    print(f"\n=== DECISION: {decision} ===")
    if decision == "EDL_ASSUMPTION_VIOLATED":
        print("The model's total evidence at MISSED small-lesion voxels is NOT meaningfully lower than at")
        print("ordinary correctly-classified background voxels -- the evidential framework is NOT correctly")
        print("flagging epistemic poverty specifically where E48-E97 already showed real information is")
        print("missing. This is a genuine, narrow, testable assumption violation worth investigating further")
        print("(though NOT yet a designed correction -- per this project's discipline, a full mechanism/")
        print("novelty audit and possibly a training-dependent follow-up would be needed before any fix).")
    else:
        print("The model's total evidence IS significantly lower at missed small-lesion voxels than at")
        print("ordinary background -- the evidential framework IS correctly signaling epistemic uncertainty.")
        print("Per the pre-declared rule: KILL this candidate as an 'evidential miscalibration' story --")
        print("the real issue (if any remains) would be insufficient DOWNSTREAM USE of an already-correct")
        print("uncertainty signal, a different and separately-testable question.")

    summary = {
        "checkpoint_val_dice": ckpt.get("best_val_dice"),
        "n_subjects": len(per_subject_records),
        "mean_S_fn": float(all_s_fn.mean()), "mean_S_bg": float(all_s_bg.mean()), "mean_S_tp": float(all_s_tp.mean()),
        "S_fn_minus_bg": float(observed_diff), "p_two_sided": p_two_sided,
        "decision": decision,
    }
    with open(OUT_DIR / "E104_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E104_summary.json")


if __name__ == "__main__":
    main()
