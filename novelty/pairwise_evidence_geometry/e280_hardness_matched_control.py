"""E280 -- Hardness-Matched Negative Control, per the user's exact
spec. The final mechanism experiment in the E274-E280 chain. E279
found R1-B's shell negatives are systematically HARDER than SAM-
equivalent's local negatives on 4 axes (closer to ET, higher
production probability, closer in D1 feature space, more TC-
concentrated). But that result does NOT yet prove the lesion-
BOUNDARY-ANCHORING construction itself is causal -- a reviewer could
argue R1-B wins simply because its negatives happen to be hard on
those axes, and ANY similarly-hard negative (regardless of HOW it was
selected) would work just as well.

THE TEST: build a new negative pool, R_matched, that is hardness-
MATCHED to R1-B's own shell pool on the same 4 axes (p_P, d_D1, d_ET,
TC_fraction) but explicitly NOT constructed via lesion-boundary
anchoring. Per explicit user design decisions this session:
  - Candidates are drawn from a CROSS-LESION pool: WT voxels sampled
    from MANY DIFFERENT det_train lesions (not the lesion being
    matched), avoiding the circular case where "matching" just
    rediscovers the same lesion's own shell under a different name.
  - For each of R1-B's own shell negatives (per lesion), find the
    NEAREST available cross-lesion candidate in the 4-property
    z-scored distance space (rank-based matching, not exact matching
    -- exact matching across 4 continuous+1 categorical properties
    would be infeasible). Per explicit user choice, poor matches
    (above a z-distance tolerance) are DROPPED, and the drop rate is
    reported honestly rather than silently forcing bad matches.
  - The candidate pool itself oversamples from ground-truth WT
    regions (not uniform-random across the whole brain), per explicit
    user choice -- WT already includes TC and peritumoral tissue, so
    this pool has a realistic chance of containing hard (high p_P,
    close D1, TC-member) candidates; a uniform-random pool would
    almost certainly fail to match most of R1-B's harder negatives.

DECISION RULE (per explicit user pre-registration): if R1-B > matched
on G1 sensitivity at matched FP budgets, boundary-anchoring ITSELF is
responsible, beyond mere hardness -- the stronger, more specific
novelty claim. If R1-B ~= matched, the claim should instead be "R1-B
works because it creates a sufficiently hard negative distribution,"
not because of the boundary-anchored construction specifically.

Everything else identical to E274-E279: same positives, same total
negative count (N=14,895, 50% shell/matched + 50% distant -- the
matched condition's "local" half is replaced by R_matched, keeping
R1-B's own distant-background half UNCHANGED, per the spirit of
isolating ONLY the local-negative-construction variable, consistent
with E276's own ablation discipline), same classifier/optimizer/
class-weighting, same test subjects, same non-circular G1/G2/G3
phenotype definition, same logit-spaced matched-FP threshold grid.
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
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch, get_masks  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402

ET, TC, WT = 0, 1, 2
STRUCT = ndimage.generate_binary_structure(3, 1)
EPS = 1e-12
TAU_LOW, TAU_HIGH = TAU, 0.9
FP_TARGETS = [100, 500, 1000]
N_NEGATIVE_TOTAL = 14895
SEED = 999
MATCH_Z_TOLERANCE = 2.0  # max combined z-distance for an acceptable match
WT_CANDIDATES_PER_LESION = 15  # how many WT voxels to sample per OTHER lesion for the candidate pool


def extract_shell_with_properties(model, ds, sid_to_idx, sid, cid, dev, w_p, b_p):
    """Returns per-shell-voxel coords, D1 feats, and the 4 hardness
    properties (p_P, d_D1 to this lesion's own ET centroid, d_ET,
    TC membership), for one lesion's shell negatives."""
    if sid not in sid_to_idx:
        return None
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    brain_mask = img_c[0] != 0
    cm, shell = get_masks(tgt_c, cid, brain_mask)
    if cm is None:
        return None
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    stages = forward_to_dec1_internal(model, img_t)
    d1 = stages['relu2_out'][0].cpu().numpy()
    shell_coords = np.argwhere(shell)
    shell_feats = d1[:, shell].T
    et_feats_mean = d1[:, cm].T.mean(axis=0)

    true_et = tgt_c[ET] > 0.5
    dist_map = ndimage.distance_transform_edt(~true_et) if true_et.any() else None
    d_et = dist_map[shell_coords[:, 0], shell_coords[:, 1], shell_coords[:, 2]] if dist_map is not None else np.full(len(shell_coords), np.nan)
    p_p = 1.0 / (1.0 + np.exp(-(shell_feats @ w_p + b_p)))
    d_d1 = np.linalg.norm(shell_feats - et_feats_mean[None, :], axis=1)
    true_tc = tgt_c[TC] > 0.5
    tc_member = true_tc[shell_coords[:, 0], shell_coords[:, 1], shell_coords[:, 2]].astype(float)

    return {'coords': shell_coords, 'feats': shell_feats, 'p_p': p_p, 'd_d1': d_d1,
           'd_et': d_et, 'tc_member': tc_member}


def extract_wt_candidates(model, ds, sid_to_idx, sid, cid, dev, w_p, b_p, rng, n_sample):
    """Samples up to n_sample WT voxels (excluding ET) from ONE lesion's
    subject, with their OWN local hardness properties (p_P, d_D1 to
    THIS lesion's own centroid, d_ET to THIS lesion, TC membership) --
    used as cross-lesion candidates when matching a DIFFERENT lesion."""
    if sid not in sid_to_idx:
        return None
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    brain_mask = img_c[0] != 0
    true_et = tgt_c[ET] > 0.5
    et_lbl, et_n = ndimage.label(true_et)
    if cid < 1 or cid > et_n:
        return None
    cm = et_lbl == cid
    if int(cm.sum()) < MIN_VOX:
        return None
    true_wt = tgt_c[WT] > 0.5
    true_tc = tgt_c[TC] > 0.5
    wt_not_et = true_wt & (~true_et) & brain_mask
    wt_idx = np.where(wt_not_et)
    if len(wt_idx[0]) == 0:
        return None
    n = min(n_sample, len(wt_idx[0]))
    sel = rng.choice(len(wt_idx[0]), size=n, replace=False)
    coords = np.stack([wt_idx[0][sel], wt_idx[1][sel], wt_idx[2][sel]], axis=1)

    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    stages = forward_to_dec1_internal(model, img_t)
    d1 = stages['relu2_out'][0].cpu().numpy()
    feats = d1[:, coords[:, 0], coords[:, 1], coords[:, 2]].T
    et_feats_mean = d1[:, cm].T.mean(axis=0)

    dist_map = ndimage.distance_transform_edt(~true_et)
    d_et = dist_map[coords[:, 0], coords[:, 1], coords[:, 2]]
    p_p = 1.0 / (1.0 + np.exp(-(feats @ w_p + b_p)))
    d_d1 = np.linalg.norm(feats - et_feats_mean[None, :], axis=1)
    tc_member = true_tc[coords[:, 0], coords[:, 1], coords[:, 2]].astype(float)

    return {'feats': feats, 'p_p': p_p, 'd_d1': d_d1, 'd_et': d_et, 'tc_member': tc_member,
           'source_sid': sid, 'source_cid': cid}


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
    rng_global = np.random.default_rng(SEED)
    t0 = time.time()
    w_p = w_et.astype(np.float64)
    b_p = float(b_et)

    # ---- Step 1: extract per-lesion shell properties (R1-B's own
    # construction, WITH the 4 hardness properties for matching) ----
    print('Extracting R1-B shell negatives with hardness properties...', flush=True)
    per_lesion_shell = []
    et_pool_list = []
    for r in det_lesions_fit:
        sid, cid = r['subject_id'], int(r['comp_id'])
        res = extract_shell_with_properties(model, ds, sid_to_idx, sid, cid, dev, w_p, b_p)
        if res is None:
            continue
        if len(res['coords']) > MAX_VOX_PER_LESION:
            sel = rng_global.choice(len(res['coords']), MAX_VOX_PER_LESION, replace=False)
            for k in ['coords', 'feats', 'p_p', 'd_d1', 'd_et', 'tc_member']:
                res[k] = res[k][sel]
        res['subject_id'] = sid
        res['comp_id'] = cid
        per_lesion_shell.append(res)
        # ET positives, same as R1-B's own construction
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
        cm = et_lbl == cid
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0].cpu().numpy()
        ef = d1[:, cm].T
        if len(ef) > MAX_VOX_PER_LESION:
            ef = ef[rng_global.choice(len(ef), MAX_VOX_PER_LESION, replace=False)]
        et_pool_list.append(ef)
    et_pool = np.concatenate(et_pool_list).astype(np.float64)
    n_shell_total = sum(len(r['coords']) for r in per_lesion_shell)
    print(f'  {len(per_lesion_shell)} lesions, {n_shell_total} shell negatives, '
         f'{len(et_pool)} ET positives ({time.time()-t0:.0f}s)', flush=True)

    # ---- Step 2: build cross-lesion WT candidate pool (for matching) ----
    print('\nBuilding cross-lesion WT candidate pool...', flush=True)
    wt_candidates_by_lesion = {}
    for r in det_lesions_fit:
        sid, cid = r['subject_id'], int(r['comp_id'])
        res = extract_wt_candidates(model, ds, sid_to_idx, sid, cid, dev, w_p, b_p, rng_global,
                                    WT_CANDIDATES_PER_LESION)
        if res is None:
            continue
        wt_candidates_by_lesion[(sid, cid)] = res
    print(f'  {len(wt_candidates_by_lesion)} lesions contributed WT candidates ({time.time()-t0:.0f}s)', flush=True)

    # pool all candidates together with a lesion-id tag, for cross-lesion
    # exclusion. CRITICAL FIX (found via IndexError in smoke testing):
    # np.array([(sid,cid), ...], dtype=object) creates a 2D (n,2) array,
    # NOT a 1D array of tuples -- so `== (sid,cid)` compared per-COLUMN
    # elementwise rather than per-ROW tuple identity, producing a (n,2)
    # boolean mask instead of the intended (n,) mask. Fixed by using a
    # single string key ("sid::cid") per candidate instead of a tuple,
    # which numpy string-array equality compares correctly as (n,).
    all_cand_feats, all_cand_props, all_cand_lesion_key = [], [], []
    for (sid, cid), res in wt_candidates_by_lesion.items():
        n = len(res['feats'])
        all_cand_feats.append(res['feats'])
        all_cand_props.append(np.stack([res['p_p'], res['d_d1'], res['d_et'], res['tc_member']], axis=1))
        all_cand_lesion_key.extend([f'{sid}::{cid}'] * n)
    all_cand_feats = np.concatenate(all_cand_feats)
    all_cand_props = np.concatenate(all_cand_props)
    all_cand_lesion_key = np.array(all_cand_lesion_key)
    print(f'  total candidate pool: {len(all_cand_feats)} voxels ({time.time()-t0:.0f}s)', flush=True)

    # z-score the 4 properties across the FULL candidate pool (shared
    # scale for matching distance)
    prop_mean = all_cand_props.mean(axis=0)
    prop_std = all_cand_props.std(axis=0) + EPS

    # ---- Step 3: for each lesion's shell negatives, find nearest
    # cross-lesion match (excluding this lesion's own candidates) ----
    print('\nMatching each shell negative to its nearest cross-lesion candidate...', flush=True)
    matched_feats_list = []
    n_matched, n_dropped = 0, 0
    for lesion in per_lesion_shell:
        sid, cid = lesion['subject_id'], lesion['comp_id']
        exclude_mask = all_cand_lesion_key == f'{sid}::{cid}'
        avail_props = (all_cand_props[~exclude_mask] - prop_mean) / prop_std
        avail_feats = all_cand_feats[~exclude_mask]
        if len(avail_props) == 0:
            n_dropped += len(lesion['coords'])
            continue
        shell_props = np.stack([lesion['p_p'], lesion['d_d1'], lesion['d_et'], lesion['tc_member']], axis=1)
        shell_props_z = (shell_props - prop_mean) / prop_std
        for i in range(len(shell_props_z)):
            dists = np.linalg.norm(avail_props - shell_props_z[i:i+1], axis=1)
            best_idx = np.argmin(dists)
            if dists[best_idx] <= MATCH_Z_TOLERANCE:
                matched_feats_list.append(avail_feats[best_idx])
                n_matched += 1
            else:
                n_dropped += 1
    matched_pool = np.array(matched_feats_list) if matched_feats_list else np.zeros((0, 32))
    drop_rate = n_dropped / max(1, n_matched + n_dropped)
    print(f'  matched={n_matched} dropped={n_dropped} (drop_rate={drop_rate:.4f}) '
         f'({time.time()-t0:.0f}s)', flush=True)

    # ---- Step 4: fit readouts (R1-B unchanged, R_matched with the
    # matched local pool swapped in for the shell half, distant half
    # unchanged -- isolating ONLY the local-negative-construction axis) ----
    distant_pool_list = []
    for sid in bg_subjects:
        feats = extract_distant_background(model, ds, sid_to_idx, sid, dev, rng_global)
        if feats is None:
            continue
        distant_pool_list.append(feats)
    distant_candidate_pool = np.concatenate(distant_pool_list).astype(np.float64)

    shell_pool_full = np.concatenate([r['feats'] for r in per_lesion_shell]).astype(np.float64)

    n_half = n_neg_total // 2
    n_s = min(n_half, len(shell_pool_full))
    n_m = min(n_half, len(matched_pool))
    n_d = min(n_neg_total - n_s, len(distant_candidate_pool))
    n_d_matched = min(n_neg_total - n_m, len(distant_candidate_pool))

    rng1 = np.random.default_rng(SEED)
    shell_neg = shell_pool_full[rng1.choice(len(shell_pool_full), size=n_s, replace=False)]
    distant_neg_1 = distant_candidate_pool[rng1.choice(len(distant_candidate_pool), size=n_d, replace=False)]
    X_r1b = np.concatenate([et_pool, shell_neg, distant_neg_1])
    y_r1b = np.concatenate([np.ones(len(et_pool)), np.zeros(n_s + n_d)])
    clf_r1b = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_r1b, y_r1b)
    w_r1b, b_r1b = clf_r1b.coef_[0].astype(np.float64), float(clf_r1b.intercept_[0])
    print(f'R1-B fit: n_shell={n_s} n_distant={n_d} ({time.time()-t0:.0f}s)', flush=True)

    rng2 = np.random.default_rng(SEED)
    matched_neg = matched_pool[rng2.choice(len(matched_pool), size=n_m, replace=False)] if n_m > 0 else matched_pool
    distant_neg_2 = distant_candidate_pool[rng2.choice(len(distant_candidate_pool), size=n_d_matched, replace=False)]
    X_matched = np.concatenate([et_pool, matched_neg, distant_neg_2])
    y_matched = np.concatenate([np.ones(len(et_pool)), np.zeros(n_m + n_d_matched)])
    clf_matched = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_matched, y_matched)
    w_matched, b_matched = clf_matched.coef_[0].astype(np.float64), float(clf_matched.intercept_[0])
    print(f'R_matched fit: n_matched={n_m} n_distant={n_d_matched} ({time.time()-t0:.0f}s)', flush=True)

    # ---- dense evaluation (identical protocol to E274-E279) ----
    w_p_t = torch.from_numpy(w_p).float().to(dev)
    variant_tensors = {
        'H0_prod': (w_p_t, b_p),
        'H1_r1b': (torch.from_numpy(w_r1b).float().to(dev), b_r1b),
        'H5_matched': (torch.from_numpy(w_matched).float().to(dev), b_matched),
    }

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
        p_prod = maps['H0_prod']
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

    print('\nSweeping thresholds (identical grid to E274-E279)...', flush=True)
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

    print('\nComputing matched-FP sensitivity per head...', flush=True)
    results = []
    for v in variant_names:
        for target in FP_TARGETS:
            t_match = matched_thresholds[v][target]
            for phen in ['G1_confidently_missed', 'G2_low_confidence', 'G3_confidently_detected', None]:
                sens = sensitivity_at_threshold(v, t_match, phen)
                results.append({
                    'readout': v, 'fp_target': target, 'threshold': t_match,
                    'mean_fp_actual': mean_fp_by_threshold[v][t_match],
                    'phenotype': phen or 'ALL', 'sensitivity': sens,
                })
                print(f'  {v} FP~{target} (t={t_match:.4f}): {phen or "ALL":28s} sensitivity={sens:.4f}', flush=True)

    out = HERE / ('E280_results_smoke.csv' if smoke else 'E280_results.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['readout', 'fp_target', 'threshold', 'mean_fp_actual',
                                           'phenotype', 'sensitivity'])
        w.writeheader()
        for row in results:
            w.writerow(row)

    print(f'\nMatch quality: {n_matched} matched, {n_dropped} dropped (drop_rate={drop_rate:.4f})')
    print(f'E280 complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
