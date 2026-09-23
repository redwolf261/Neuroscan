"""
Phase E122: Transition-Specific Signature Audit -- Relative Lesion Scale.

CONTEXT: E121 (corrected) found N_1(64->32) ~= N_3(16->8) >> N_2(32->16) in
small-lesion-specific causal signature, rejecting the "pool3 is uniquely
responsible" claim. This phase tests the PRIMARY hypothesis for why the
64->32 and 16->8 transitions behave alike while 32->16 does not: relative
lesion scale (lesion diameter relative to each stage's own pooling-cell
footprint), not raw lesion size, may govern per-stage necessity.

REUSES E121's per-subject N_1/N_2/N_3 table directly (no new forward
passes -- N_1/N_2/N_3 are already trustworthy, cross-checked against the
established N_b range after E121's own bug fix). This script ONLY adds
the relative-scale variable and the associated statistical tests.

DEFINITION OF q_{s,k} (pinned down explicitly, per user's choice this
session -- 64^3 RESIZED frame, not native voxel space, since that is the
ACTUAL grid the network's pooling stages operate on):
  1. lesion_voxels_64(s) = count of voxels with fractional occupancy > 0.5
     in the SAME 64^3 resized frame used everywhere in E90-E121 for
     target_bin_64 (fractional_occupancy(seg_binary_native, (64,64,64))).
  2. d_s = equivalent-sphere diameter of that lesion, IN 64^3-GRID UNITS:
       d_s = (6 * lesion_voxels_64(s) / pi) ** (1/3)
  3. cell_width_k = 64 / r_k, where r_k is stage k's OUTPUT spatial
     resolution (r_1=32, r_2=16, r_3=8) -- i.e. how many 64^3-grid units
     wide one pooling cell's receptive footprint is at that stage. This
     is a pure geometric constant (2, 4, 8), not extracted from the
     network.
  4. q_{s,k} = d_s / cell_width_k -- a dimensionless "how many pooling
     cells wide is this lesion, at this stage's resolution" ratio.

HYPOTHESIS: N_{s,k} = f(q_{s,k}), and this explains the N1~=N3>>N2 pattern
better than raw lesion size does, for the two DIFFERENTIATING stages
(1 and 3) specifically.

MODELS (per transition k, exactly as pre-registered):
  Model A: N_k ~ native_size          (already computed in E121, restated here)
  Model B: N_k ~ q_k
  Model C: N_k ~ q_k + native_size    (partial correlation convention, same
                                        as E118-E121's own size-control method)

CONTROLS: native_size (log), dice_error -- same covariates used throughout
E118-E121.

PRE-REGISTERED DECISION GATES:
  Case A: q_k does not explain N1/N2/N3 pattern any better than native_size
          alone (Model B's rho not significant for the differentiating
          stages 1&3, or no differential fit vs stage 2) -> KILL.
  Case B: q_k correlates with N_k but the relationship COLLAPSES (rho not
          significant / sign flips) in Model C after controlling
          native_size -> KILL as "just a re-expression of size."
  Case C: q_k's relationship with N_k SURVIVES Model C (partial rho
          significant, same sign) for stage 1 and/or stage 3, AND is
          weaker/absent for stage 2 -> interesting mechanistic candidate,
          proceed to a full prior-art audit before any architecture work.
  Case D: neither q_k nor native_size alone explains the N1~=N3>>N2
          pattern -> report as unexplained, recommend the secondary
          transition-property characterization (E122's own fallback,
          NOT run in this script) as the next diagnostic, not architecture
          work.

NO ARCHITECTURE MODIFICATION, NO TRAINING, NO NOVELTY CLAIM IN THIS
SCRIPT. Purely a closed-form geometric/statistical analysis of existing
per-subject N_1/N_2/N_3 values plus segmentation masks already on disk.
"""
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy import stats
import sys

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000
E121_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e121" / "E121_per_subject_table.json"

STAGE_RESOLUTIONS = {"N_1": 32, "N_2": 16, "N_3": 8}  # output resolution r_k


def fractional_occupancy(seg_binary_native, shape):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=shape, mode="area").squeeze().numpy()
    return frac


def permutation_test_corr(x, y, seed):
    rho, p_param = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng.permutation(y)
        perm_rhos[i], _ = stats.spearmanr(x, perm_y)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())
    return float(rho), float(p_param), p_perm


def partial_corr(x, y, control_log_size, seed):
    X1 = np.column_stack([np.ones(len(control_log_size)), control_log_size])
    beta_x, *_ = np.linalg.lstsq(X1, x, rcond=None)
    resid_x = x - X1 @ beta_x
    beta_y, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid_y = y - X1 @ beta_y
    rho, p_param = stats.spearmanr(resid_x, resid_y)
    rng = np.random.default_rng(seed)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y2 = rng.permutation(resid_y)
        perm_rhos[i], _ = stats.spearmanr(resid_x, perm_y2)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())
    return float(rho), float(p_param), p_perm


def main():
    with open(E121_TABLE_PATH) as f:
        e121_records = json.load(f)
    print(f"Loaded {len(e121_records)} subjects from E121's per-subject table.")

    val_ds = BraTSDataset(root_dir=str(project_root / "Dataset" / "Training"), split="val",
                          val_split=0.1, target_shape=(64, 64, 64), normalize=True)
    subject_dirs_by_id = {}
    for idx in range(len(val_ds)):
        subject_dirs_by_id[val_ds.subject_dirs[idx].name if hasattr(val_ds.subject_dirs[idx], "name")
                            else Path(val_ds.subject_dirs[idx]).name] = val_ds.subject_dirs[idx]

    records = []
    for r in e121_records:
        sid = r["subject_id"]
        subj_dir_candidates = [d for d in subject_dirs_by_id if sid in d or d in sid]
        subject_dir = subject_dirs_by_id.get(sid) or (subject_dirs_by_id[subj_dir_candidates[0]] if subj_dir_candidates else None)
        if subject_dir is None:
            # fallback: brute-force match by scanning val_ds directly
            subject_dir = None
            for idx in range(len(val_ds)):
                if val_ds.subject_dirs[idx].name == sid or str(val_ds.subject_dirs[idx]).endswith(sid):
                    subject_dir = val_ds.subject_dirs[idx]
                    break
        assert subject_dir is not None, f"Could not locate subject_dir for {sid}"

        seg_path = Path(subject_dir) / f"{sid}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        frac_64 = fractional_occupancy(seg_binary_native, (64, 64, 64))
        lesion_voxels_64 = int((frac_64 > 0.5).sum())
        d_s = (6.0 * lesion_voxels_64 / np.pi) ** (1.0 / 3.0) if lesion_voxels_64 > 0 else 0.0

        rec = dict(r)
        rec["lesion_voxels_64"] = lesion_voxels_64
        rec["d_s_64grid"] = d_s
        for stage, r_k in STAGE_RESOLUTIONS.items():
            cell_width_k = 64.0 / r_k
            rec[f"q_{stage}"] = d_s / cell_width_k
        records.append(rec)

    print(f"Matched {len(records)} subjects to segmentation files.")
    zero_lesion = sum(1 for r in records if r["lesion_voxels_64"] == 0)
    print(f"Subjects with lesion_voxels_64==0 (degenerate at 64^3): {zero_lesion}")
    records = [r for r in records if r["lesion_voxels_64"] > 0]
    print(f"Proceeding with {len(records)} non-degenerate subjects.\n")

    with open(OUT_DIR / "E122_per_subject_table.json", "w") as f:
        json.dump(records, f, indent=2)

    native_size = np.array([r["native_size"] for r in records])
    log_size = np.log(native_size + 1)
    d_s = np.array([r["d_s_64grid"] for r in records])
    print(f"d_s (equivalent-sphere diameter, 64^3-grid units): mean={d_s.mean():.3f} std={d_s.std():.3f} "
          f"min={d_s.min():.3f} max={d_s.max():.3f}")

    results = {"stages": {}}
    for stage in ["N_1", "N_2", "N_3"]:
        N_k = np.array([r[stage] for r in records])
        q_k = np.array([r[f"q_{stage}"] for r in records])
        r_k_res = STAGE_RESOLUTIONS[stage]
        print(f"\n=== Stage {stage} (output res {r_k_res}^3, cell_width={64.0/r_k_res:.1f} grid-units) ===")
        print(f"  q_{stage}: mean={q_k.mean():.3f} std={q_k.std():.3f} "
              f"(median={np.median(q_k):.3f}, frac q<1 [sub-cell-sized lesion]={float((q_k<1).mean()):.3f})")

        # Model A: N_k ~ native_size (restate E121's own result for this subset)
        rhoA, pA_param, pA_perm = permutation_test_corr(N_k, native_size, SEED)
        print(f"  Model A (N_k ~ native_size): rho={rhoA:+.4f} (perm p={pA_perm:.4f})")

        # Model B: N_k ~ q_k
        rhoB, pB_param, pB_perm = permutation_test_corr(N_k, q_k, SEED + 1)
        print(f"  Model B (N_k ~ q_k):         rho={rhoB:+.4f} (perm p={pB_perm:.4f})")

        # Model C: partial correlation of q_k with N_k, controlling native_size
        rhoC, pC_param, pC_perm = partial_corr(q_k, N_k, log_size, SEED + 2)
        print(f"  Model C (N_k ~ q_k | size):  partial_rho={rhoC:+.4f} (perm p={pC_perm:.4f})")

        # Does q_k explain MORE than size alone? Compare |rhoB| vs |rhoA|,
        # and whether q_k's partial (controlling size) is still significant
        # -- that's the actual test of "not just a re-expression of size."
        survives = (abs(rhoC) > 0) and (pC_perm < 0.05) and (np.sign(rhoC) == np.sign(rhoB) if rhoB != 0 else True)

        results["stages"][stage] = {
            "output_resolution": r_k_res, "cell_width_64grid": 64.0 / r_k_res,
            "q_mean": float(q_k.mean()), "q_std": float(q_k.std()),
            "modelA_rho_size": rhoA, "modelA_perm_p": pA_perm,
            "modelB_rho_q": rhoB, "modelB_perm_p": pB_perm,
            "modelC_partial_rho_q_given_size": rhoC, "modelC_perm_p": pC_perm,
            "q_survives_size_control": bool(survives),
        }
        print(f"  q_k survives size control (Model C significant, same sign as Model B): {survives}")

    # ---------------- BUG CAUGHT BEFORE TRUSTING THE PER-STAGE RESULT ----------------
    # Audit (before reporting anything to the user): q_{s,k} = d_s / cell_width_k,
    # and cell_width_k is a FIXED CONSTANT per stage (2, 4, or 8) -- identical
    # for every subject. Dividing by a positive constant does not change
    # Spearman rank correlation AT ALL, so within any single stage's
    # regression, Model B (N_k ~ q_k) is mathematically guaranteed to give
    # the SAME rho as N_k ~ d_s (confirmed above: Model A and Model B's rho
    # are nearly identical at every stage). Three separate single-stage
    # regressions are therefore STRUCTURALLY INCAPABLE of testing the real
    # cross-stage hypothesis ("relative scale differs across stages for the
    # SAME subject and that differential predicts the N1~=N3>>N2 pattern") --
    # caught before reporting the per-stage "Case C" verdict as real evidence.
    # FIX: pool all 3*n observations (subject, stage) into one regression
    # where q_k actually DOES vary across stages for a fixed subject (unlike
    # within a single stage), testing whether q predicts N beyond stage
    # identity (fixed effect) and lesion size.
    print("\n=== BUG CAUGHT: per-stage Models A/B/C cannot test the cross-stage "
          "hypothesis (q_k is a constant rescaling of d_s within any one stage, "
          "confirmed by Model A~=Model B rho above). Running the POOLED "
          "cross-stage regression instead. ===\n")

    stage_list = ["N_1", "N_2", "N_3"]
    stage_dummy = {"N_1": (1, 0), "N_2": (0, 1), "N_3": (0, 0)}  # N_3 is reference level

    pooled_N, pooled_q, pooled_size, pooled_d1, pooled_d2, pooled_subj = [], [], [], [], [], []
    for si, r in enumerate(records):
        for stage in stage_list:
            pooled_N.append(r[stage])
            pooled_q.append(r[f"q_{stage}"])
            pooled_size.append(np.log(r["native_size"] + 1))
            d1, d2 = stage_dummy[stage]
            pooled_d1.append(d1)
            pooled_d2.append(d2)
            pooled_subj.append(si)

    pooled_N = np.array(pooled_N)
    pooled_q = np.array(pooled_q)
    pooled_size = np.array(pooled_size)
    pooled_d1 = np.array(pooled_d1)
    pooled_d2 = np.array(pooled_d2)

    # Model P1: N ~ stage dummies + size (baseline: does stage identity +
    # size alone explain N, without q?)
    X_p1 = np.column_stack([np.ones(len(pooled_N)), pooled_d1, pooled_d2, pooled_size])
    beta_p1, *_ = np.linalg.lstsq(X_p1, pooled_N, rcond=None)
    resid_p1 = pooled_N - X_p1 @ beta_p1
    r2_p1 = 1 - (resid_p1 ** 2).sum() / ((pooled_N - pooled_N.mean()) ** 2).sum()

    # Model P2: N ~ stage dummies + size + q (does q add anything beyond
    # stage identity and size?)
    X_p2 = np.column_stack([np.ones(len(pooled_N)), pooled_d1, pooled_d2, pooled_size, pooled_q])
    beta_p2, *_ = np.linalg.lstsq(X_p2, pooled_N, rcond=None)
    resid_p2 = pooled_N - X_p2 @ beta_p2
    r2_p2 = 1 - (resid_p2 ** 2).sum() / ((pooled_N - pooled_N.mean()) ** 2).sum()
    q_coef = beta_p2[-1]

    print(f"Pooled n={len(pooled_N)} observations (125 subjects x 3 stages)")
    print(f"Model P1 (N ~ stage + size):     R^2={r2_p1:.4f}")
    print(f"Model P2 (N ~ stage + size + q): R^2={r2_p2:.4f}, q coefficient={q_coef:+.4f}")
    print(f"Delta R^2 from adding q: {r2_p2 - r2_p1:+.4f}")

    # Permutation test on q's coefficient: permute q WITHIN each subject
    # (i.e. shuffle which stage gets which q value, per subject) to break
    # the q<->stage link while preserving each subject's overall q level
    # and each stage's marginal N distribution -- a stricter, more
    # appropriate null than a global permutation, since it specifically
    # tests whether q's PATTERN ACROSS STAGES (not just its per-subject
    # level) explains N's pattern across stages.
    rng = np.random.default_rng(SEED + 10)
    observed_q_coef = q_coef
    perm_coefs = np.empty(N_PERM)
    q_by_subject = pooled_q.reshape(len(records), 3)
    for i in range(N_PERM):
        q_perm = np.array([rng.permutation(row) for row in q_by_subject]).reshape(-1)
        X_perm = np.column_stack([np.ones(len(pooled_N)), pooled_d1, pooled_d2, pooled_size, q_perm])
        beta_perm, *_ = np.linalg.lstsq(X_perm, pooled_N, rcond=None)
        perm_coefs[i] = beta_perm[-1]
    p_perm_q = float((np.abs(perm_coefs) >= np.abs(observed_q_coef)).mean())
    print(f"Within-subject-permutation test on q's coefficient: p={p_perm_q:.4f}")

    # ---------------- Decision (pooled test is now the real evidence) ----------------
    print("\n=== DECISION (based on pooled cross-stage test, not the flawed per-stage one) ===")
    q_matters = (abs(q_coef) > 1e-6) and (p_perm_q < 0.05) and (r2_p2 - r2_p1 > 0.01)

    if not q_matters:
        decision = "CASE_A_KILL_Q_DOES_NOT_EXPLAIN_PATTERN"
        detail = ("Pooled cross-stage regression: adding q_k on top of stage-identity + size "
                   "does not significantly improve fit (perm p={:.4f}, Delta R^2={:+.4f}). "
                   "Relative lesion scale does NOT explain the N1~=N3>>N2 pattern beyond what "
                   "stage identity and absolute size already capture. KILL this hypothesis."
                   ).format(p_perm_q, r2_p2 - r2_p1)
    else:
        decision = "CASE_C_INTERESTING_Q_ADDS_EXPLANATORY_POWER"
        detail = ("Pooled cross-stage regression: q_k significantly improves fit beyond stage "
                   "identity and size (perm p={:.4f}, Delta R^2={:+.4f}, coef={:+.4f}). "
                   "Relative lesion scale carries information about per-stage necessity beyond "
                   "what raw size and which-stage-it-is already explain. Still NOT yet a "
                   "validated mechanism -- proceed to a full prior-art audit before any "
                   "architecture work."
                   ).format(p_perm_q, r2_p2 - r2_p1, q_coef)

    print(f"{decision}\n{detail}")

    results.update({
        "n_subjects": len(records),
        "PER_STAGE_MODELS_ARE_STRUCTURALLY_FLAWED_SEE_COMMENT": (
            "q_k is a constant rescaling of d_s within one stage -- Model A/B/C per stage "
            "cannot test the cross-stage hypothesis. Use the pooled_cross_stage_test fields "
            "below as the real evidence."
        ),
        "pooled_cross_stage_test": {
            "n_observations": len(pooled_N),
            "model_p1_r2_stage_and_size_only": float(r2_p1),
            "model_p2_r2_with_q": float(r2_p2),
            "delta_r2_from_q": float(r2_p2 - r2_p1),
            "q_coefficient": float(q_coef),
            "within_subject_permutation_p": p_perm_q,
        },
        "decision": decision, "detail": detail,
    })
    with open(OUT_DIR / "E122_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved E122_summary.json, E122_per_subject_table.json")


if __name__ == "__main__":
    main()
