"""E277 -- Local-Shell Radius Ablation, per the user's exact spec.
Direct follow-up to E276 (Case B: shell+distant COMBINATION drives
rescue, neither component alone sufficient). This experiment tests
whether the shell's EXACT radius matters, or whether "local hard
negatives + distant background" works across a range of spatial
scales -- isolating the one remaining uncertainty in R1-B's negative
construction.

COORDINATE SYSTEM NOTE (resolved with the user before building):
R1-B's existing shell uses ndimage.binary_dilation with iterations=
SHELL_DILATION=3 -- a VOXEL-radius dilation, not an explicit physical-
distance threshold. This BraTS pipeline's data loader does not
resample volumes, and BraTS's own standard distribution convention is
already 1mm isotropic spacing, so voxel-radius and mm-radius are
numerically IDENTICAL here. Per explicit user agreement, this means a
single voxel-radius sweep already satisfies both the "primary
coordinate system" and "physical-distance robustness check" the
original spec asked for -- no separate mm-aware implementation was
built, avoiding a redundant duplicate computation.

SHELL DEFINITION (per the user's own formula, implemented via
iterated binary dilation -- the same mechanism as R1-B's own existing
shell, just with a variable radius instead of the fixed SHELL_DILATION
=3): for radius r, S_r = {non-ET voxels within r dilation steps of the
ET lesion boundary} = dilate(ET_mask, iterations=r) & ~ET_mask &
brain_mask. This is CUMULATIVE (0 < d <= r), not annular -- the
annular-band secondary analysis is explicitly deferred per the user's
own staged ordering, contingent on this core result being interesting.

RADII TESTED: r in {1, 2, 3, 5, 10} (voxel/mm, per the coordinate-
system note above), plus R1-B's own ORIGINAL construction (r=3,
reused unchanged as R_orig -- NOT refit, since it is numerically
identical to the r=3 condition in this sweep; reported as a single
reference row to confirm the sweep reproduces it).

SAMPLE-COUNT EQUALIZATION (per explicit user "critical" requirement):
every radius condition uses the SAME total negative count N=14,895
(matching E276's own budget), split exactly 50% shell(r) + 50%
distant -- regardless of how many candidate voxels radius r actually
makes available. The distant-background candidate pool is IDENTICAL
across all radius conditions (built once, reused for every r), so any
difference between conditions can only come from the shell's own
radius, never from a change in the global distant-negative
population.

STAGED PLAN (per explicit user ordering): Phase 1 -- single seed (999)
across all 5 radii, to see the shape of the r -> G1-sensitivity curve
cheaply first. Phase 2 (deferred, contingent on Phase 1's shape):
3 seeds for only the 4 key conditions (original, best, smallest,
largest radius). Phase 1 ONLY is run in this script, per explicit
user agreement this session.

EVALUATION: IDENTICAL to E274/E275/E276 -- same non-circular G1/G2/G3
phenotype definition, same logit-spaced matched-FP threshold grid
(100/500/1000), same det_test subjects, same classifier/optimizer/
class-weighting/positive-sample construction. ONLY the shell radius
changes across conditions.

SHELL COMPOSITION (per explicit user secondary analysis): for each
radius, the fraction of shell voxels falling inside ground-truth WT
vs outside WT, to check whether radius is merely changing the
ANATOMICAL COMPOSITION of the negatives (not just their spatial
extent).
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
                         extract_distant_background,
                         ROOT, PATCH, MIN_VOX, TAU, MAX_VOX_PER_LESION)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch  # noqa: E402

ET, TC, WT = 0, 1, 2
STRUCT = ndimage.generate_binary_structure(3, 1)
EPS = 1e-12
TAU_LOW, TAU_HIGH = TAU, 0.9
FP_TARGETS = [100, 500, 1000]
N_NEGATIVE_TOTAL = 14895  # matches E276's own budget
RADII = [1, 2, 3, 5, 10]
PRIMARY_SEED = 999


def extract_lesion_and_shell_r(model, ds, sid_to_idx, sid, cid, dev, radius):
    """Radius-parameterized version of e257_common's extract_lesion_shell
    (which hardcodes SHELL_DILATION=3 via e245's get_masks). Returns
    (lesion_feats, shell_feats, shell_mask, true_wt_mask, lesion_size)
    or None. Cumulative shell: dilate(lesion, iterations=radius) &
    ~lesion & brain_mask & ~true_et (identical mechanism to R1-B's own
    shell, just with a variable radius)."""
    if sid not in sid_to_idx:
        return None
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    brain_mask = img_c[0] != 0
    et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
    if cid < 1 or cid > et_n:
        return None
    cm = et_lbl == cid
    if int(cm.sum()) < MIN_VOX:
        return None
    dilated = ndimage.binary_dilation(cm, structure=STRUCT, iterations=radius)
    true_et = tgt_c[ET] > 0.5
    shell = dilated & (~cm) & brain_mask & (~true_et)
    if int(shell.sum()) < MIN_VOX:
        return None
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    stages = forward_to_dec1_internal(model, img_t)
    d1 = stages['relu2_out'][0].cpu().numpy()
    lesion_feats = d1[:, cm].T
    shell_feats = d1[:, shell].T
    true_wt = tgt_c[WT] > 0.5
    frac_wt = float((shell & true_wt).sum() / max(1, int(shell.sum())))
    return lesion_feats, shell_feats, int(cm.sum()), frac_wt


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    model = load_model(dev)
    w_et, b_et = get_w_prod(model)
    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)

    if smoke:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']][:30]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])[:20]
        n_neg_total = 2000
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])
        n_neg_total = N_NEGATIVE_TOTAL

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    rng_global = np.random.default_rng(PRIMARY_SEED)
    t0 = time.time()

    # ---- distant-background candidate pool: built ONCE, reused for
    # EVERY radius condition, so it can never be the source of a
    # difference between conditions (per explicit user requirement) ----
    print('Building distant-background candidate pool (shared across ALL radii)...', flush=True)
    distant_pool_list = []
    for sid in bg_subjects:
        feats = extract_distant_background(model, ds, sid_to_idx, sid, dev, rng_global)
        if feats is None:
            continue
        distant_pool_list.append(feats)
    distant_candidate_pool = np.concatenate(distant_pool_list).astype(np.float64)
    print(f'  distant candidates: {len(distant_candidate_pool)} ({time.time()-t0:.0f}s)', flush=True)

    # ---- per-radius: ET-positive pool + shell candidate pool + shell
    # composition (fraction in WT) ----
    print('\nBuilding per-radius ET + shell candidate pools...', flush=True)
    et_pool_by_radius = {}
    shell_pool_by_radius = {}
    shell_frac_wt_by_radius = {}
    for radius in RADII:
        et_list, shell_list, frac_wt_list = [], [], []
        for r in det_lesions_fit:
            res = extract_lesion_and_shell_r(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev, radius)
            if res is None:
                continue
            lf, sf, sz, frac_wt = res
            if len(lf) > MAX_VOX_PER_LESION:
                lf = lf[rng_global.choice(len(lf), MAX_VOX_PER_LESION, replace=False)]
            if len(sf) > MAX_VOX_PER_LESION:
                sf = sf[rng_global.choice(len(sf), MAX_VOX_PER_LESION, replace=False)]
            et_list.append(lf); shell_list.append(sf); frac_wt_list.append(frac_wt)
        et_pool_by_radius[radius] = np.concatenate(et_list).astype(np.float64)
        shell_pool_by_radius[radius] = np.concatenate(shell_list).astype(np.float64)
        shell_frac_wt_by_radius[radius] = float(np.mean(frac_wt_list)) if frac_wt_list else float('nan')
        print(f'  r={radius}: ET={len(et_pool_by_radius[radius])} '
             f'shell_candidates={len(shell_pool_by_radius[radius])} '
             f'frac_WT={shell_frac_wt_by_radius[radius]:.4f} ({time.time()-t0:.0f}s)', flush=True)

    # ---- fit one readout per radius, N/2 shell + N/2 distant, matched budget ----
    print('\nFitting readouts (one per radius, matched negative budget)...', flush=True)
    fitted = {}
    for radius in RADII:
        rng = np.random.default_rng(PRIMARY_SEED)
        et_pool = et_pool_by_radius[radius]
        shell_pool = shell_pool_by_radius[radius]
        n_half = n_neg_total // 2
        n_s = min(n_half, len(shell_pool))
        n_d = min(n_neg_total - n_s, len(distant_candidate_pool))
        shell_neg = shell_pool[rng.choice(len(shell_pool), size=n_s, replace=False)]
        distant_neg = distant_candidate_pool[rng.choice(len(distant_candidate_pool), size=n_d, replace=False)]
        X = np.concatenate([et_pool, shell_neg, distant_neg])
        y = np.concatenate([np.ones(len(et_pool)), np.zeros(n_s + n_d)])
        clf = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X, y)
        fitted[radius] = (clf.coef_[0].astype(np.float64), float(clf.intercept_[0]))
        print(f'  r={radius}: n_shell_neg={n_s} n_distant_neg={n_d} total_neg={n_s+n_d} '
             f'({time.time()-t0:.0f}s)', flush=True)

    w_p_t = torch.from_numpy(w_et.astype(np.float64)).float().to(dev)
    b_p = float(b_et)
    variant_tensors = {'prod': (w_p_t, b_p)}
    for radius, (w, b) in fitted.items():
        variant_tensors[f'r{radius}'] = (torch.from_numpy(w).float().to(dev), b)

    dense_cache = {}

    def dense_eval(sid):
        if sid in dense_cache:
            return dense_cache[sid]
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]
        C, D, H, W = d1.shape
        flat = d1.reshape(C, -1).T
        maps = {}
        with torch.no_grad():
            for name, (w_t, b_v) in variant_tensors.items():
                p = torch.sigmoid(flat @ w_t + b_v).reshape(D, H, W).cpu().numpy()
                maps[name] = p
        del d1, flat, img_t, stages
        torch.cuda.empty_cache()
        result = (maps, tgt_c, img_c)
        dense_cache[sid] = result
        return result

    test_subjects = sorted(splits['det_test'])
    if smoke:
        test_subjects = test_subjects[:15]

    print(f'\nBuilding non-circular phenotype groups on {len(test_subjects)} TEST subjects...', flush=True)
    lesion_records = []
    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        maps, tgt_c, img_c = dense_eval(sid)
        p_prod = maps['prod']
        true_et = tgt_c[ET] > 0.5
        et_lbl, et_n = ndimage.label(true_et)
        for cid in range(1, et_n + 1):
            lesion_mask = et_lbl == cid
            if int(lesion_mask.sum()) < MIN_VOX:
                continue
            p_max = float(p_prod[lesion_mask].max())
            if p_max < TAU_LOW:
                phenotype = 'G1_confidently_missed'
            elif p_max < TAU_HIGH:
                phenotype = 'G2_low_confidence'
            else:
                phenotype = 'G3_confidently_detected'
            lesion_records.append({'subject_id': sid, 'comp_id': cid, 'phenotype': phenotype})
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)

    n_by_phen = {}
    for rec in lesion_records:
        n_by_phen[rec['phenotype']] = n_by_phen.get(rec['phenotype'], 0) + 1
    print(f'\nPhenotype group sizes: {n_by_phen}', flush=True)

    lesion_by_sid = {}
    for rec in lesion_records:
        lesion_by_sid.setdefault(rec['subject_id'], []).append(rec)

    print('\nSweeping thresholds (identical grid to E274-E276)...', flush=True)
    threshold_grid = np.concatenate([
        np.linspace(0.01, 0.1, 10),
        np.linspace(0.1, 0.9, 20),
        1 - 1 / (1 + np.exp(np.linspace(2, 30, 60))),
    ])
    threshold_grid = np.unique(np.round(threshold_grid, 8))

    variant_names = list(variant_tensors.keys())
    fp_lists = {v: {t: [] for t in threshold_grid} for v in variant_names}
    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        maps, tgt_c, img_c = dense_eval(sid)
        true_et = tgt_c[ET] > 0.5
        brain_mask = img_c[0] != 0
        for name, p_map in maps.items():
            for t in threshold_grid:
                mask = (p_map > t) & brain_mask
                fp_lists[name][t].append(int((mask & (~true_et)).sum()))
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)

    mean_fp_by_threshold = {v: {} for v in variant_names}
    for v in variant_names:
        for t in threshold_grid:
            fps = fp_lists[v][t]
            mean_fp_by_threshold[v][t] = float(np.mean(fps)) if fps else float('nan')

    def find_threshold_for_fp_target(v, target_fp):
        best_t, best_diff = None, np.inf
        for t in threshold_grid:
            diff = abs(mean_fp_by_threshold[v][t] - target_fp)
            if diff < best_diff:
                best_diff, best_t = diff, t
        return best_t

    matched_thresholds = {v: {target: find_threshold_for_fp_target(v, target) for target in FP_TARGETS}
                          for v in variant_names}

    def sensitivity_at_threshold(v, t, phenotype_filter=None):
        recs = []
        for sid in test_subjects:
            if sid not in sid_to_idx:
                continue
            maps, tgt_c, img_c = dense_eval(sid)
            p_map = maps[v]
            et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
            for rec in lesion_by_sid.get(sid, []):
                if phenotype_filter and rec['phenotype'] != phenotype_filter:
                    continue
                cid = rec['comp_id']
                if cid < 1 or cid > et_n:
                    continue
                lm = et_lbl == cid
                recs.append(int(((p_map > t) & lm).any()))
        return float(np.mean(recs)) if recs else float('nan')

    def dice_at_threshold(v, t):
        vals = []
        for sid in test_subjects:
            if sid not in sid_to_idx:
                continue
            maps, tgt_c, img_c = dense_eval(sid)
            p_map = maps[v]
            true_et = tgt_c[ET] > 0.5
            brain_mask = img_c[0] != 0
            mask = (p_map > t) & brain_mask
            ps, gs = mask.sum(), true_et.sum()
            if gs == 0:
                vals.append(1.0 if ps == 0 else np.nan)
            else:
                vals.append(2 * (mask & true_et).sum() / (ps + gs))
        return float(np.nanmean(vals)) if vals else float('nan')

    print('\nComputing matched-FP sensitivity per radius...', flush=True)
    results = []
    for v in variant_names:
        for target in FP_TARGETS:
            t_match = matched_thresholds[v][target]
            for phen in ['G1_confidently_missed', 'G2_low_confidence', 'G3_confidently_detected', None]:
                sens = sensitivity_at_threshold(v, t_match, phen)
                results.append({
                    'variant': v, 'fp_target': target, 'threshold': t_match,
                    'mean_fp_actual': mean_fp_by_threshold[v][t_match],
                    'phenotype': phen or 'ALL', 'sensitivity': sens,
                })
                print(f'  {v} FP~{target} (t={t_match:.4f}): {phen or "ALL":28s} sensitivity={sens:.4f}', flush=True)
        t_500 = matched_thresholds[v][500]
        dice500 = dice_at_threshold(v, t_500)
        results.append({'variant': v, 'fp_target': 500, 'threshold': t_500,
                        'mean_fp_actual': mean_fp_by_threshold[v][t_500], 'phenotype': 'DICE', 'sensitivity': dice500})

    out = HERE / ('E277_results_smoke.csv' if smoke else 'E277_results.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['variant', 'fp_target', 'threshold', 'mean_fp_actual',
                                           'phenotype', 'sensitivity'])
        w.writeheader()
        for row in results:
            w.writerow(row)

    composition_out = HERE / ('E277_shell_composition_smoke.csv' if smoke else 'E277_shell_composition.csv')
    with open(composition_out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['radius', 'mean_frac_wt_in_shell'])
        for radius in RADII:
            w.writerow([radius, shell_frac_wt_by_radius[radius]])

    print(f'\nE277 Phase 1 complete ({time.time()-t0:.0f}s). wrote {out.name}, {composition_out.name}', flush=True)


if __name__ == '__main__':
    main()
