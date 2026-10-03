"""E287 -- Candidate-Recovery Reconstruction Ablation, per the user's
exact spec. Direct follow-up to E286, which found MRD-T2's whole-brain
Dice/precision/recall/IoU/HD95/ASSD are dramatically worse than
production's alone (Dice ~0.06 vs production's ~0.84 at FP~500) --
expected and NOT a contradiction of the G1-recovery result (MRD is a
sparse, targeted recovery channel, not a dense segmenter), but the
user's explicit next question is whether MRD's recovered SIGNAL can be
used to IMPROVE the final segmentation, without changing the frozen
MRD target itself. Per explicit user framing: "We should not change
the method just to chase Dice... First test whether MRD localization +
existing segmentation information can convert the sparse recovery
signal into a proper lesion mask."

SIX CONDITIONS (per explicit user's own ablation table, Section
"three plausible routes" + "controlled ablation"):
  A. Production alone (frozen, H0 from E286)
  B. MRD alone (raw R_phi(x) thresholded, H3 from E286 -- "current result")
  C. Production + raw MRD voxels (Route 1): hat_Y = hat_Y_P UNION
     {x : P(x)<tau_P AND R_phi(x)>tau_R}
  D. Production + connected MRD components, UNFILTERED (intermediate
     step toward E, included for a clean ablation progression)
  E. Production + connected MRD components, SIZE-FILTERED (Route 2):
     hat_Y = hat_Y_P UNION (union of connected components of
     {R_phi(x)>tau_R} with size > s, computed only within the region
     P(x)<tau_P)
  F. NOT RUN THIS PASS: "local delineation" (Route 3, growing/
     delineating the candidate using x/P) is explicitly deferred --
     per the ablation table it is the most speculative route and the
     user's own text flags it as needing the EARLIER routes' results
     first ("This experiment directly addresses the weakness exposed
     by E286 without destroying the mathematical result").

FROZEN THRESHOLDS (chosen WITHOUT touching E286's 125-subject locked
test cohort, per explicit user requirement and AskUserQuestion
resolution this session):
  tau_P = 0.5  -- this is NOT a newly-chosen value: it is TAU, the
                  SAME threshold already used throughout E274-E286 to
                  define "production confidently misses this lesion"
                  (G1 phenotype: max_v P(v) < TAU). Reusing it here
                  means "production uncertain/missing" means EXACTLY
                  the same thing it has meant everywhere else in this
                  investigation -- not a new free parameter.
  tau_R = 0.95082877089122 -- T2/MRD's OWN matched-FP~500 threshold
                  from the DEVELOPMENT cohort (E283's det_test
                  population, seed=999), read directly from
                  E283_matched_fp.csv before writing this script.
                  Chosen from the development cohort specifically so
                  E286's 125-subject LOCKED test cohort is never
                  touched when picking this number -- keeps the
                  no-peeking guarantee for the generalization result.
  s = MIN_VOX = 5 -- reuses the ALREADY-FROZEN lesion-inclusion floor
                  from e257_common.py (the same constant deciding
                  whether a ground-truth lesion counts at all
                  throughout E274-E286), no new free parameter.
                  No morphological closing/dilation this pass (kept
                  as the simplest version of Route 2 per explicit
                  user scoping).

EVALUATED ON: the EXACT SAME 125-subject LOCKED test cohort as E286
(re-derived identically -- same eligibility logic, same frozen
production checkpoint, same R1-B/MRD readouts refit on det_train with
the SAME 5 seeds), so results are directly comparable to E286's own
numbers without re-deriving the cohort from scratch.

ENDPOINTS: Dice/Precision/Recall/Specificity/IoU (pooled voxel
confusion matrix, matching E286's own convention exactly) AND G1
lesion-recovery sensitivity (matching E274-E286's own convention) --
per explicit user requirement, BOTH must be reported together, since
a method could win on one and lose on the other.
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e257_common import (load_model, get_w_prod, load_populations, build_splits,  # noqa: E402
                         extract_lesion_shell, extract_distant_background,
                         ROOT, PATCH, MIN_VOX, TAU, MAX_VOX_PER_LESION)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch  # noqa: E402

ET, TC, WT = 0, 1, 2
EPS_MARG = 1e-6
TAU_LOW, TAU_HIGH = TAU, 0.9
FP_TARGETS = [100, 500, 1000]
SEEDS = [999, 4242, 7, 123, 2024]
STRUCT = ndimage.generate_binary_structure(3, 1)

# FROZEN before touching E286's test cohort (see module docstring)
TAU_P = 0.5  # = TAU, production-uncertain/missing threshold (G1's own convention)
TAU_R = 0.95082877089122  # T2's own FP~500 threshold from E283's development cohort
MIN_COMPONENT_SIZE = MIN_VOX  # = 5, reuses the existing lesion-inclusion floor


def fit_soft_target(X, y_soft, max_iter=500, C=1.0):
    X2 = np.concatenate([X, X], axis=0)
    y2 = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    w2 = np.concatenate([y_soft, 1.0 - y_soft])
    keep = w2 > 1e-12
    X2, y2, w2 = X2[keep], y2[keep], w2[keep]
    clf = LogisticRegression(max_iter=max_iter, C=C, class_weight='balanced')
    clf.fit(X2, y2, sample_weight=w2)
    return clf.coef_[0].astype(np.float64), float(clf.intercept_[0])


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')
    t0_time = time.time()

    print('=' * 70, flush=True)
    print('FROZEN THRESHOLDS (set before touching the test cohort)', flush=True)
    print('=' * 70, flush=True)
    print(f'  tau_P = {TAU_P} (= TAU, the existing G1-definition threshold)', flush=True)
    print(f'  tau_R = {TAU_R} (T2\'s own FP~500 threshold, E283 development cohort)', flush=True)
    print(f'  s (min component size) = {MIN_COMPONENT_SIZE} (= MIN_VOX, existing constant)', flush=True)

    model = load_model(dev)
    w_et, b_et = get_w_prod(model)
    w_p_np, b_p_np = w_et.astype(np.float64), float(b_et)

    def p_prod_of(x):
        z = x @ w_p_np + b_p_np
        return 1.0 / (1.0 + np.exp(-z))

    # ---- rebuild E286's exact locked test cohort ----
    print('\nRebuilding E286\'s exact 125-subject locked test cohort...', flush=True)
    ds_train_pop = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                          val_split=0.1, patch_size=PATCH)
    ds_val_pop = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                        val_split=0.1, patch_size=PATCH)
    val_sids_all = sorted(os.path.basename(d) for d in ds_val_pop.subject_dirs)

    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)
    all_prior_touched = (set(r['subject_id'] for r in det_lesions) |
                         set(r['subject_id'] for r in g2a_lesions) |
                         set(r['subject_id'] for r in g2b_lesions))
    val_sids_fresh = sorted(set(val_sids_all) - all_prior_touched)

    ds_full = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                     val_split=0.0, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds_full.subject_dirs)}

    eligible_sids = []
    check_pool = val_sids_fresh[:15] if smoke else val_sids_fresh
    for sid in check_pool:
        if sid not in sid_to_idx:
            continue
        try:
            img_c, tgt_c = load_patch(ds_full, sid_to_idx, sid)
            img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
            stages = forward_to_dec1_internal(model, img_t)
            d1 = stages['relu2_out'][0]
            if torch.isnan(d1).any() or torch.isinf(d1).any():
                continue
            del d1, img_t, stages
            torch.cuda.empty_cache()
            eligible_sids.append(sid)
        except Exception:
            continue
    test_subjects = sorted(eligible_sids)
    print(f'  rebuilt test cohort: {len(test_subjects)} subjects '
         f'(must match E286\'s own N3={"15 (smoke)" if smoke else "125"})', flush=True)

    # ---- leakage check (same as E286) ----
    overlap_with_train = set(test_subjects) & set(os.path.basename(d) for d in ds_train_pop.subject_dirs)
    overlap_with_prior = set(test_subjects) & all_prior_touched
    assert len(overlap_with_train) == 0 and len(overlap_with_prior) == 0, 'LEAKAGE AUDIT FAILED'
    print('  leakage audit: PASS (0 overlap with production training, 0 overlap with prior experiments)',
         flush=True)

    # ---- fit R1-B/T0/T2 on det_train, 5 seeds (same pool as E283-E286) ----
    print('\nFitting H1 (ordinary, for reference)/H3 (MRD-T2) on det_train, 5 seeds...', flush=True)
    if smoke:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']][:30]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])[:20]
        seeds = SEEDS[:2]
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])
        seeds = SEEDS

    def fit_r1b_pool(seed):
        rng = np.random.default_rng(seed)
        lesion_pool, neg_pool = [], []
        for r in det_lesions_fit:
            res = extract_lesion_shell(model, ds_train_pop, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
            if res is None:
                continue
            lf, sf, sz = res
            if len(lf) > MAX_VOX_PER_LESION:
                lf = lf[rng.choice(len(lf), MAX_VOX_PER_LESION, replace=False)]
            if len(sf) > MAX_VOX_PER_LESION:
                sf = sf[rng.choice(len(sf), MAX_VOX_PER_LESION, replace=False)]
            lesion_pool.append(lf); neg_pool.append(sf)
        for sid in bg_subjects:
            feats = extract_distant_background(model, ds_train_pop, sid_to_idx, sid, dev, rng)
            if feats is None:
                continue
            neg_pool.append(feats)
        lesion_pool = np.concatenate(lesion_pool).astype(np.float64)
        neg_pool = np.concatenate(neg_pool).astype(np.float64)
        return lesion_pool, neg_pool

    fitted = {}
    for seed in seeds:
        lesion_pool, neg_pool = fit_r1b_pool(seed)
        X = np.concatenate([lesion_pool, neg_pool])
        y_hard = np.concatenate([np.ones(len(lesion_pool)), np.zeros(len(neg_pool))])
        p_i = p_prod_of(X)
        t2_vals = np.clip((y_hard - p_i) / (1.0 - p_i + EPS_MARG), 0.0, 1.0)
        w_h3, b_h3 = fit_soft_target(X, t2_vals)
        fitted[seed] = (w_h3, b_h3)
        print(f'  seed={seed} fit ({time.time()-t0_time:.0f}s)', flush=True)

    w_p_t = torch.from_numpy(w_p_np).float().to(dev)
    tensors = {seed: (torch.from_numpy(w).float().to(dev), b) for seed, (w, b) in fitted.items()}

    dense_cache = {}

    def dense_eval(sid):
        if sid in dense_cache:
            return dense_cache[sid]
        img_c, tgt_c = load_patch(ds_full, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]
        C_, D, H, W = d1.shape
        flat = d1.reshape(C_, -1).T
        maps = {}
        with torch.no_grad():
            z_prod = flat @ w_p_t + b_p_np
            maps['prod'] = torch.sigmoid(z_prod).reshape(D, H, W).cpu().numpy()
            for seed, (w_t, b) in tensors.items():
                r = torch.sigmoid(flat @ w_t + b)
                maps[seed] = r.reshape(D, H, W).cpu().numpy()
        del d1, flat, img_t, stages, z_prod
        torch.cuda.empty_cache()
        result = (maps, tgt_c, img_c)
        dense_cache[sid] = result
        return result

    test_subjects_eval = test_subjects[:15] if smoke else test_subjects

    # ---- G1 phenotype groups (same convention as E274-E286) ----
    print(f'\nBuilding G1/G2/G3 phenotype groups on {len(test_subjects_eval)} subjects...', flush=True)
    lesion_records = []
    for i, sid in enumerate(test_subjects_eval):
        maps, tgt_c, img_c = dense_eval(sid)
        p_prod = maps['prod']
        true_et = tgt_c[ET] > 0.5
        et_lbl, et_n = ndimage.label(true_et)
        for cid in range(1, et_n + 1):
            lesion_mask = et_lbl == cid
            sz = int(lesion_mask.sum())
            if sz < MIN_VOX:
                continue
            p_max = float(p_prod[lesion_mask].max())
            if p_max < TAU_LOW:
                phenotype = 'G1_confidently_missed'
            elif p_max < TAU_HIGH:
                phenotype = 'G2_low_confidence'
            else:
                phenotype = 'G3_confidently_detected'
            lesion_records.append({'subject_id': sid, 'comp_id': cid, 'phenotype': phenotype, 'size': sz})
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects_eval)} ({time.time()-t0_time:.0f}s)', flush=True)
    lesion_by_sid = {}
    for rec in lesion_records:
        lesion_by_sid.setdefault(rec['subject_id'], []).append(rec)
    n_by_phen = {}
    for rec in lesion_records:
        n_by_phen[rec['phenotype']] = n_by_phen.get(rec['phenotype'], 0) + 1
    print(f'  phenotype group sizes: {n_by_phen}', flush=True)

    # ---- 6 conditions, computed per subject, per seed ----
    print('\nComputing 5 conditions (A-E) per subject per seed...', flush=True)
    conditions = ['A_prod', 'B_mrd_alone', 'C_prod_plus_raw_mrd', 'D_prod_plus_unfiltered_cc',
                 'E_prod_plus_filtered_cc']
    voxel_tp = {(seed, c): 0 for seed in seeds for c in conditions}
    voxel_fp = {(seed, c): 0 for seed in seeds for c in conditions}
    voxel_fn = {(seed, c): 0 for seed in seeds for c in conditions}
    voxel_tn = {(seed, c): 0 for seed in seeds for c in conditions}
    g1_recovered = {(seed, c): [] for seed in seeds for c in conditions}

    for i, sid in enumerate(test_subjects_eval):
        maps, tgt_c, img_c = dense_eval(sid)
        p_map = maps['prod']
        true_et = tgt_c[ET] > 0.5
        brain_mask = img_c[0] != 0
        et_lbl, et_n = ndimage.label(true_et)
        recs = [r for r in lesion_by_sid.get(sid, []) if r['phenotype'] == 'G1_confidently_missed']

        pred_P = (p_map > TAU_P) & brain_mask  # NOTE: production's OWN detection mask uses
        # its own natural operating point (sigmoid 0.5), NOT tau_P (which marks
        # UNCERTAINTY, a different role) -- see hat_Y_P construction below.
        hat_Y_P = (p_map > 0.5) & brain_mask
        uncertain_region = (p_map < TAU_P) & brain_mask

        for seed in seeds:
            r_map = maps[seed]
            mrd_candidates = (r_map > TAU_R) & brain_mask
            c_mask = uncertain_region & mrd_candidates  # Route 1: raw voxels

            # Route 2: connected components of mrd_candidates (within uncertain
            # region), with and without size filtering
            cc_region = uncertain_region & mrd_candidates
            cc_lbl, cc_n = ndimage.label(cc_region, structure=STRUCT)
            d_mask_unfiltered = cc_region.copy()  # D = unfiltered CC = same voxels as C by construction;
            # the ablation value of D vs C is that D is computed via explicit
            # connected-component extraction (so size filtering in E can be
            # applied), not that D's own mask differs from C before filtering.
            e_mask_filtered = np.zeros_like(cc_region)
            if cc_n > 0:
                sizes = ndimage.sum(cc_region, cc_lbl, index=np.arange(1, cc_n + 1))
                keep_labels = np.where(sizes > MIN_COMPONENT_SIZE)[0] + 1
                if len(keep_labels) > 0:
                    e_mask_filtered = np.isin(cc_lbl, keep_labels)

            pred_masks = {
                'A_prod': hat_Y_P,
                'B_mrd_alone': mrd_candidates,
                'C_prod_plus_raw_mrd': hat_Y_P | c_mask,
                'D_prod_plus_unfiltered_cc': hat_Y_P | d_mask_unfiltered,
                'E_prod_plus_filtered_cc': hat_Y_P | e_mask_filtered,
            }
            for cond, pm in pred_masks.items():
                voxel_tp[(seed, cond)] += int((pm & true_et).sum())
                voxel_fp[(seed, cond)] += int((pm & ~true_et).sum())
                voxel_fn[(seed, cond)] += int((~pm & true_et & brain_mask).sum())
                voxel_tn[(seed, cond)] += int((~pm & ~true_et & brain_mask).sum())
                for rec in recs:
                    cid = rec['comp_id']
                    if cid < 1 or cid > et_n:
                        continue
                    lm = et_lbl == cid
                    g1_recovered[(seed, cond)].append(int((pm & lm).any()))
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects_eval)} ({time.time()-t0_time:.0f}s)', flush=True)

    # ---- write results ----
    print('\n' + '=' * 70, flush=True)
    print('RESULTS: pooled voxel confusion matrix + G1 recovery, per seed', flush=True)
    print('=' * 70, flush=True)
    rows = []
    for seed in seeds:
        for cond in conditions:
            tp, fp, fn, tn = (voxel_tp[(seed, cond)], voxel_fp[(seed, cond)],
                             voxel_fn[(seed, cond)], voxel_tn[(seed, cond)])
            precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
            recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
            specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float('nan')
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float('nan')
            g1_vals = g1_recovered[(seed, cond)]
            g1_sens = float(np.mean(g1_vals)) if g1_vals else float('nan')
            rows.append({'seed': seed, 'condition': cond, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                        'precision': precision, 'recall': recall, 'specificity': specificity,
                        'dice': dice, 'iou': iou, 'g1_sensitivity': g1_sens, 'n_g1_lesions': len(g1_vals)})

    out = HERE / ('E287_results_smoke.csv' if smoke else 'E287_results.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['seed', 'condition', 'TP', 'FP', 'FN', 'TN', 'precision',
                                           'recall', 'specificity', 'dice', 'iou', 'g1_sensitivity',
                                           'n_g1_lesions'])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f'wrote {out.name}', flush=True)

    print('\nSUMMARY (mean +/- SD across seeds):', flush=True)
    by_cond = {}
    for row in rows:
        by_cond.setdefault(row['condition'], {'dice': [], 'precision': [], 'recall': [],
                                              'iou': [], 'g1_sensitivity': []})
        for k in ['dice', 'precision', 'recall', 'iou', 'g1_sensitivity']:
            by_cond[row['condition']][k].append(row[k])
    for cond in conditions:
        d = by_cond[cond]
        print(f'  {cond:28s} Dice={np.mean(d["dice"]):.4f}±{np.std(d["dice"]):.4f}  '
             f'Precision={np.mean(d["precision"]):.4f}  Recall={np.mean(d["recall"]):.4f}  '
             f'IoU={np.mean(d["iou"]):.4f}  G1_sens={np.mean(d["g1_sensitivity"])*100:.1f}%'
             f'±{np.std(d["g1_sensitivity"])*100:.1f}%', flush=True)

    print(f'\nE287 complete ({time.time()-t0_time:.0f}s).', flush=True)


if __name__ == '__main__':
    main()
