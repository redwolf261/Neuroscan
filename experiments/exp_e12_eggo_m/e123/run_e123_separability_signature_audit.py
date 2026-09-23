"""
Phase E123: Transition-Specific Signature Audit -- Lesion/Background
Separability (secondary property, per E122's decision tree Case A ->
proceed to targeted, non-shotgun transition-property characterization).

CONTEXT: E121 found N_1(64->32)~=N_3(16->8)>>N_2(32->16) in small-lesion
causal signature. E122 killed "relative lesion scale" as the explanation
(pooled cross-stage regression: q_k adds negligible R^2 beyond stage
identity + absolute size). This phase tests ONE pre-registered, targeted
property (per explicit instruction: no shotgun correlation matrix) --
lesion/background separability of the PRE-POOLING activations feeding
each of the three downsampling stages -- for a stage1~=stage3!=stage2
signature.

HYPOTHESIS: at stages where small-lesion causal necessity is high (1 and
3), the pre-pooling activations may ALREADY be poorly separated between
lesion and background voxels (so pooling has little "good" signal to
preserve regardless of how it combines the window) -- OR conversely, may
be WELL separated but pooling destroys that separability specifically at
those stages. Direction is not assumed; both are tested and reported.

MEASURE (per-channel AUC, the standard, assumption-light way to quantify
binary separability of a continuous activation without assuming a
particular threshold or distribution):
  For each stage's PRE-POOLING tensor (enc1: 32ch@64^3 feeding pool1,
  enc2: 64ch@32^3 feeding pool2, enc3: 128ch@16^3 feeding pool3):
    - Downsample the ground-truth lesion mask (native res) to that
      stage's OWN spatial resolution (64^3, 32^3, 16^3 respectively) via
      the same fractional_occupancy('area' interpolation) convention used
      throughout E90-E122, threshold >0.5 for a binary per-voxel label.
    - For each channel c, compute AUC(channel_c activations, binary lesion
      label) over all voxels in that subject's volume at that resolution.
    - Per-subject separability score S_stage = mean over channels of
      |AUC_c - 0.5| * 2 (rescaled to [0,1], 0=no separability, 1=perfect
      separability in EITHER direction -- a channel can be informative by
      being LOW inside lesions just as validly as HIGH).

CROSS-STAGE TEST (same pooled design as E122, to avoid E122's own
structural mistake of running per-stage-only regressions that cannot test
a cross-stage pattern): pool all (subject, stage) observations, test
whether S_stage (a genuine per-stage-varying quantity, unlike E122's
q_k which was actually a constant-rescaling artifact) explains N_k beyond
stage identity + lesion size, via the SAME pooled regression + within-
subject-permutation-test design as E122's corrected analysis.

ALSO reported directly (not just via the pooled test): per-stage mean
S_stage, to see by eye whether the raw pattern already looks like
stage1~=stage3!=stage2 (the signature we're looking for) BEFORE running
any regression -- per user's explicit request (Section 17) for a simple,
visible pattern match, not just a p-value.

PRE-REGISTERED DECISION GATES:
  Case A: S_stage does not show a visible stage1~=stage3!=stage2 pattern
          AND does not add explanatory power in the pooled regression
          (Delta R^2 < 0.01 or not significant) -> KILL.
  Case B: S_stage shows the visible pattern but the pooled regression
          shows it's redundant with stage identity + size (Delta R^2 <
          0.01 despite a visible raw pattern) -> KILL as non-differentiating
          once stage/size are accounted for.
  Case C: S_stage shows the visible pattern AND survives the pooled
          regression (Delta R^2 > 0.01, permutation p<0.05) -> interesting,
          proceed to a full prior-art audit before any architecture work.
  Case D: no clean property emerges -- report honestly, do not force one.
          This was explicitly permitted by the user (Section 17: "If no
          such property emerges, say so. Do not force one.").

NO ARCHITECTURE MODIFICATION, NO TRAINING, NO NOVELTY CLAIM.
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
E121_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e121" / "E121_per_subject_table.json"

CKPT_PATH = (project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs"
             / "AttnGate_seed0" / "checkpoints" / "best.pth")
EXPECTED_VAL_DICE = 0.9101624600589275

# Pre-pooling tensors: (name -> (attr, spatial_resolution))
STAGES = {
    "N_1": ("enc1", 64),  # feeds pool1
    "N_2": ("enc2", 32),  # feeds pool2
    "N_3": ("enc3", 16),  # feeds pool3
}


def fractional_occupancy(seg_binary_native, shape):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=shape, mode="area").squeeze().numpy()
    return frac


def fast_auc(scores, labels):
    """Rank-based AUC (Mann-Whitney U / n_pos*n_neg), avoids sklearn
    dependency and is exact, not approximated."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # handle ties by averaging ranks
    sorted_scores = scores[order]
    sorted_ranks = ranks[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = sorted_ranks[i:j + 1].mean()
            sorted_ranks[i:j + 1] = avg_rank
        i = j + 1
    ranks[order] = sorted_ranks
    sum_ranks_pos = ranks[labels.astype(bool)].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def permutation_test_corr(x, y, seed):
    rho, p_param = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng.permutation(y)
        perm_rhos[i], _ = stats.spearmanr(x, perm_y)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())
    return float(rho), float(p_param), p_perm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(E121_TABLE_PATH) as f:
        e121_records = json.load(f)
    e121_by_id = {r["subject_id"]: r for r in e121_records}
    print(f"Loaded {len(e121_records)} subjects from E121's per-subject table.")

    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    val_dice = ckpt.get("best_val_dice") or ckpt.get("val_dice")
    assert val_dice is not None and abs(float(val_dice) - EXPECTED_VAL_DICE) < 1e-6, \
        f"Checkpoint identity check FAILED: expected {EXPECTED_VAL_DICE}, got {val_dice}"
    print(f"[Sanity check] checkpoint identity PASS (val_dice={val_dice}).")

    val_ds = BraTSDataset(root_dir=str(project_root / "Dataset" / "Training"), split="val",
                          val_split=0.1, target_shape=(64, 64, 64), normalize=True)

    records = []
    print(f"\nExtracting pre-pooling activations + computing separability for "
          f"{len(val_ds)} subjects...", flush=True)
    for idx in range(len(val_ds)):
        image, mask, sid = val_ds[idx]
        if sid not in e121_by_id:
            continue
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_ds.subject_dirs[idx]
        seg_path = Path(subject_dir) / f"{sid}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)

        with torch.no_grad():
            enc1 = model.enc1(image_b)
            pool1 = model.pool1(enc1)
            enc2 = model.enc2(pool1)
            pool2 = model.pool2(enc2)
            enc3 = model.enc3(pool2)

        tensors = {"N_1": enc1.squeeze(0).cpu().numpy(),
                   "N_2": enc2.squeeze(0).cpu().numpy(),
                   "N_3": enc3.squeeze(0).cpu().numpy()}

        rec = dict(e121_by_id[sid])
        for stage, (attr, res) in STAGES.items():
            mask_frac = fractional_occupancy(seg_binary_native, (res, res, res))
            label = (mask_frac > 0.5).astype(np.float32).flatten()
            n_pos = int(label.sum())
            if n_pos == 0 or n_pos == len(label):
                rec[f"S_{stage}"] = None
                continue
            act = tensors[stage]  # (C, res, res, res)
            C = act.shape[0]
            aucs = []
            for c in range(C):
                a = fast_auc(act[c].flatten(), label)
                if a is not None:
                    aucs.append(abs(a - 0.5) * 2)
            rec[f"S_{stage}"] = float(np.mean(aucs)) if aucs else None
            rec[f"n_channels_{stage}"] = C

        records.append(rec)
        if len(records) % 25 == 0:
            print(f"  {len(records)}/{len(e121_records)}", flush=True)

    with open(OUT_DIR / "E123_per_subject_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nTotal subjects with valid separability scores: {len(records)}")

    # ---------------- Raw per-stage means (visible pattern check) ----------------
    print("\n=== Raw per-stage separability S_stage (mean |AUC-0.5|*2 over channels) ===")
    valid = {}
    for stage in STAGES:
        vals = np.array([r[f"S_{stage}"] for r in records if r.get(f"S_{stage}") is not None])
        valid[stage] = vals
        print(f"  S_{stage}: mean={vals.mean():.4f} std={vals.std():.4f} (n={len(vals)})")

    s1, s2, s3 = valid["N_1"].mean(), valid["N_2"].mean(), valid["N_3"].mean()
    visible_pattern = (abs(s1 - s3) < 0.5 * abs(s1 - s2) + 0.5 * abs(s3 - s2)) and \
                       (abs(s1 - s2) > 0.02 or abs(s3 - s2) > 0.02)
    print(f"\n  Visible pattern check (s1~=s3, both differ from s2): "
          f"s1={s1:.4f}, s2={s2:.4f}, s3={s3:.4f} -> {'YES' if visible_pattern else 'NO'}")

    # ---------------- Pooled cross-stage regression (E122's corrected design) ----------------
    print("\n=== Pooled cross-stage regression: does S_stage explain N beyond stage+size? ===")
    complete = [r for r in records if all(r.get(f"S_{s}") is not None for s in STAGES)]
    print(f"Complete-case subjects (all 3 stages valid): {len(complete)}")

    stage_dummy = {"N_1": (1, 0), "N_2": (0, 1), "N_3": (0, 0)}
    pooled_N, pooled_S, pooled_size, pooled_d1, pooled_d2 = [], [], [], [], []
    for r in complete:
        for stage in STAGES:
            pooled_N.append(r[stage])
            pooled_S.append(r[f"S_{stage}"])
            pooled_size.append(np.log(r["native_size"] + 1))
            d1, d2 = stage_dummy[stage]
            pooled_d1.append(d1)
            pooled_d2.append(d2)
    pooled_N, pooled_S = np.array(pooled_N), np.array(pooled_S)
    pooled_size, pooled_d1, pooled_d2 = np.array(pooled_size), np.array(pooled_d1), np.array(pooled_d2)

    X_p1 = np.column_stack([np.ones(len(pooled_N)), pooled_d1, pooled_d2, pooled_size])
    beta_p1, *_ = np.linalg.lstsq(X_p1, pooled_N, rcond=None)
    resid_p1 = pooled_N - X_p1 @ beta_p1
    r2_p1 = 1 - (resid_p1 ** 2).sum() / ((pooled_N - pooled_N.mean()) ** 2).sum()

    X_p2 = np.column_stack([np.ones(len(pooled_N)), pooled_d1, pooled_d2, pooled_size, pooled_S])
    beta_p2, *_ = np.linalg.lstsq(X_p2, pooled_N, rcond=None)
    resid_p2 = pooled_N - X_p2 @ beta_p2
    r2_p2 = 1 - (resid_p2 ** 2).sum() / ((pooled_N - pooled_N.mean()) ** 2).sum()
    s_coef = beta_p2[-1]

    print(f"Pooled n={len(pooled_N)} observations ({len(complete)} subjects x 3 stages)")
    print(f"Model P1 (N ~ stage + size):     R^2={r2_p1:.4f}")
    print(f"Model P2 (N ~ stage + size + S): R^2={r2_p2:.4f}, S coefficient={s_coef:+.4f}")
    print(f"Delta R^2 from adding S: {r2_p2 - r2_p1:+.4f}")

    rng = np.random.default_rng(SEED + 20)
    S_by_subject = pooled_S.reshape(len(complete), 3)
    perm_coefs = np.empty(N_PERM)
    for i in range(N_PERM):
        S_perm = np.array([rng.permutation(row) for row in S_by_subject]).reshape(-1)
        X_perm = np.column_stack([np.ones(len(pooled_N)), pooled_d1, pooled_d2, pooled_size, S_perm])
        beta_perm, *_ = np.linalg.lstsq(X_perm, pooled_N, rcond=None)
        perm_coefs[i] = beta_perm[-1]
    p_perm_s = float((np.abs(perm_coefs) >= np.abs(s_coef)).mean())
    print(f"Within-subject-permutation test on S's coefficient: p={p_perm_s:.4f}")

    # ---------------- Decision ----------------
    print("\n=== DECISION ===")
    delta_r2 = r2_p2 - r2_p1
    s_matters = (abs(delta_r2) > 0.01) and (p_perm_s < 0.05)

    if not visible_pattern and not s_matters:
        decision = "CASE_A_KILL_NO_PATTERN_NO_EXPLANATORY_POWER"
        detail = "No visible stage1~=stage3!=stage2 pattern in raw S_stage, and no explanatory power beyond stage+size in the pooled regression. KILL."
    elif visible_pattern and not s_matters:
        decision = "CASE_B_KILL_REDUNDANT_WITH_STAGE_SIZE"
        detail = "Raw S_stage shows the visible pattern, but this is redundant with stage identity + size once controlled (Delta R^2={:.4f}, perm p={:.4f}). KILL as non-differentiating.".format(delta_r2, p_perm_s)
    elif visible_pattern and s_matters:
        decision = "CASE_C_INTERESTING_SEPARABILITY_SIGNATURE"
        detail = "S_stage shows the visible stage1~=stage3!=stage2 pattern AND survives the pooled regression (Delta R^2={:+.4f}, perm p={:.4f}). Proceed to a full prior-art audit before any architecture work.".format(delta_r2, p_perm_s)
    else:
        decision = "CASE_D_NO_CLEAN_PROPERTY_UNEXPECTED"
        detail = "S_stage adds explanatory power in the pooled regression but does not show the expected visible raw pattern -- report honestly, do not force an interpretation."

    print(f"{decision}\n{detail}")

    summary = {
        "checkpoint": str(CKPT_PATH), "val_dice_check": val_dice,
        "n_subjects": len(records), "n_complete_case": len(complete),
        "S_stage_means": {s: float(valid[s].mean()) for s in STAGES},
        "S_stage_stds": {s: float(valid[s].std()) for s in STAGES},
        "visible_pattern": bool(visible_pattern),
        "pooled_r2_stage_size_only": float(r2_p1),
        "pooled_r2_with_S": float(r2_p2),
        "delta_r2_from_S": float(delta_r2),
        "S_coefficient": float(s_coef),
        "within_subject_permutation_p": p_perm_s,
        "decision": decision, "detail": detail,
    }
    with open(OUT_DIR / "E123_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E123_summary.json, E123_per_subject_table.json")


if __name__ == "__main__":
    main()
