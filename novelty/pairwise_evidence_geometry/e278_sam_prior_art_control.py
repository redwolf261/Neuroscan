"""E278 -- Prior-Art-Equivalent Negative-Sampling Control, per the
user's exact spec. Direct follow-up to E276/E277 (Case B confirmed:
shell+distant combination drives rescue; Outcome A confirmed: the
mechanism is radius-insensitive within 1-10mm). The remaining open
question before any novelty claim: is R1-B's rescue effect simply the
consequence of a KNOWN negative-sampling principle (local-hard +
global/random negatives, as in SAM -- Yan et al., self-supervised
pixel-wise anatomical embeddings), or does R1-B's SPECIFIC formulation
add something beyond that known combination?

FOUR CONDITIONS on the SAME frozen D1, same positive ET pool, same
test subjects/phenotype definitions, same evaluation protocol as
E274-E277 (changed per explicit user design decisions this session,
documented below):
  H0 production:  frozen, no retraining (reused unchanged)
  H1 R1-B:         E257-B's own construction -- local SHELL (lesion-
                   boundary-anchored dilation, radius=3, within the
                   SAME subject) + distant background (SAME subject)
  H2/H4 SAM-equiv: per explicit user design -- SAM's actual negative
                   principle does not use a lesion-shape-anchored
                   shell (SAM has no "lesion," it is a self-supervised
                   point-correspondence task); the faithful SAM
                   translation is LOCAL = non-ET voxels at a random
                   SPATIAL OFFSET (magnitude 1-10 voxels, uniform
                   random direction) from each ET-positive voxel,
                   WITHIN THE SAME SUBJECT but NOT anatomically
                   shape-aware; GLOBAL = non-ET voxels from a
                   DIFFERENT, randomly chosen TRAINING SUBJECT entirely
                   (cross-subject, per explicit user choice -- the
                   real SAM principle: negatives from OTHER images,
                   not just distant locations in the same image).
                   Trained with the EXACT SAME LogisticRegression
                   binary-classifier objective as R1-B (NOT SAM's own
                   InfoNCE embedding loss -- per explicit user
                   instruction, reproducing the SAMPLING PRINCIPLE
                   only, to preserve causal attribution). H2 and H4
                   from the original spec collapse into this ONE
                   condition (confirmed with the user before building
                   -- H4's own description, "SAM sampling + NeuroScan
                   objective," is identical to H2's own description).
  H3 random:       uniform-random non-ET negatives (== E275/E276's
                   R0_random, reused unchanged)

SAMPLE-COUNT EQUALIZATION (per explicit user "critical" requirement):
every condition uses the SAME total negative count N=14,895 (matching
E276/E277's own budget), split 50/50 where the construction has two
components (H1: shell/distant; H2/H4: local/global).

NO LEAKAGE: all negative-sampling candidate pools are built from
det_train subjects ONLY; test subjects never enter any training or
negative-mining step. Phenotype groups (G1/G2/G3) are IDENTICAL to
E274 -- built from frozen production probability + ground truth ONLY,
never redefined per head (avoiding exactly the circularity risk the
user flagged).

EVALUATION: IDENTICAL E274-E277 protocol (matched-FP threshold sweep,
same logit-spaced grid, same 67 det_test subjects, G1 lesion
sensitivity as the primary endpoint, not Dice).
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
STRUCT = ndimage.generate_binary_structure(3, 1)
EPS = 1e-12
TAU_LOW, TAU_HIGH = TAU, 0.9
FP_TARGETS = [100, 500, 1000]
N_NEGATIVE_TOTAL = 14895
SEED = 999
LOCAL_OFFSET_MIN, LOCAL_OFFSET_MAX = 1, 10


def extract_et_and_sam_local(model, ds, sid_to_idx, sid, cid, dev, rng):
    """Returns (et_feats, sam_local_feats, lesion_size) for one lesion.
    sam_local: for each ET voxel, sample ONE non-ET voxel at a random
    offset of magnitude in [LOCAL_OFFSET_MIN, LOCAL_OFFSET_MAX] voxels,
    uniform-random 3D direction -- NOT anatomically shape-aware (no
    dilation/shell logic), per explicit user design for a faithful SAM
    translation. Retries with a new random direction if the offset
    voxel lands outside the brain or on another ET voxel."""
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
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    stages = forward_to_dec1_internal(model, img_t)
    d1 = stages['relu2_out'][0].cpu().numpy()
    D, H, W = cm.shape

    et_coords = np.argwhere(cm)
    # CRITICAL FIX (found via IndexError in smoke testing): et_coords and
    # local_coords must stay in 1:1 correspondence. The original version
    # silently SKIPPED et voxels whose offset sampling failed all 10
    # retries, shrinking local_coords below len(et_coords) without also
    # shrinking et_coords -- a real length-mismatch bug, not a rare edge
    # case (it triggered on the very first smoke-scale lesion). Fixed by
    # tracking which et_coords indices actually succeeded and indexing
    # BOTH arrays by that same successful-index list.
    local_coords = []
    successful_et_idx = []
    for i, (z, y, x) in enumerate(et_coords):
        for _ in range(10):  # up to 10 retries per voxel
            direction = rng.normal(size=3)
            direction /= (np.linalg.norm(direction) + EPS)
            magnitude = rng.uniform(LOCAL_OFFSET_MIN, LOCAL_OFFSET_MAX)
            offset = np.round(direction * magnitude).astype(int)
            nz, ny, nx = z + offset[0], y + offset[1], x + offset[2]
            if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W and brain_mask[nz, ny, nx] and not true_et[nz, ny, nx]:
                local_coords.append((nz, ny, nx))
                successful_et_idx.append(i)
                break
    if not local_coords:
        return np.zeros((0, 32)), np.zeros((0, 32)), int(cm.sum())
    local_coords = np.array(local_coords)
    local_feats = d1[:, local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]].T
    et_coords_matched = et_coords[successful_et_idx]
    et_feats = d1[:, et_coords_matched[:, 0], et_coords_matched[:, 1], et_coords_matched[:, 2]].T
    return et_feats, local_feats, int(cm.sum())


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
        conv_subjects = sorted(splits['det_train'])[:30]
        n_neg_total = 2000
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])
        conv_subjects = sorted(splits['det_train'])
        n_neg_total = N_NEGATIVE_TOTAL

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    rng_global = np.random.default_rng(SEED)
    t0 = time.time()

    # ---- H1 (R1-B): identical to E257-B/E274-E277 ----
    print('Building H1 (R1-B) pools: shell + distant...', flush=True)
    et_pool_list, shell_pool_list = [], []
    for r in det_lesions_fit:
        res = extract_lesion_shell(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
        if res is None:
            continue
        lf, sf, sz = res
        if len(lf) > MAX_VOX_PER_LESION:
            lf = lf[rng_global.choice(len(lf), MAX_VOX_PER_LESION, replace=False)]
        if len(sf) > MAX_VOX_PER_LESION:
            sf = sf[rng_global.choice(len(sf), MAX_VOX_PER_LESION, replace=False)]
        et_pool_list.append(lf); shell_pool_list.append(sf)
    et_pool = np.concatenate(et_pool_list).astype(np.float64)
    shell_candidate_pool = np.concatenate(shell_pool_list).astype(np.float64)

    distant_pool_list = []
    for sid in bg_subjects:
        feats = extract_distant_background(model, ds, sid_to_idx, sid, dev, rng_global)
        if feats is None:
            continue
        distant_pool_list.append(feats)
    distant_candidate_pool = np.concatenate(distant_pool_list).astype(np.float64)
    print(f'  ET={len(et_pool)} shell_cand={len(shell_candidate_pool)} '
         f'distant_cand={len(distant_candidate_pool)} ({time.time()-t0:.0f}s)', flush=True)

    n_half = n_neg_total // 2
    n_s = min(n_half, len(shell_candidate_pool))
    n_d = min(n_neg_total - n_s, len(distant_candidate_pool))
    shell_neg = shell_candidate_pool[rng_global.choice(len(shell_candidate_pool), size=n_s, replace=False)]
    distant_neg_h1 = distant_candidate_pool[rng_global.choice(len(distant_candidate_pool), size=n_d, replace=False)]
    X_h1 = np.concatenate([et_pool, shell_neg, distant_neg_h1])
    y_h1 = np.concatenate([np.ones(len(et_pool)), np.zeros(n_s + n_d)])
    clf_h1 = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_h1, y_h1)
    w_h1, b_h1 = clf_h1.coef_[0].astype(np.float64), float(clf_h1.intercept_[0])
    print(f'H1 (R1-B) fit: n_shell={n_s} n_distant={n_d} ({time.time()-t0:.0f}s)', flush=True)

    # ---- H3 (random): identical to E275/E276's conventional head ----
    print('\nBuilding H3 (random) pool...', flush=True)
    random_candidate_list = []
    for sid in conv_subjects:
        if sid not in sid_to_idx:
            continue
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0].cpu().numpy()
        true_et = tgt_c[ET] > 0.5
        brain_mask = img_c[0] != 0
        non_et_mask = brain_mask & (~true_et)
        non_et_idx = np.where(non_et_mask)
        if len(non_et_idx[0]) > 0:
            n_sample = min(500, len(non_et_idx[0]))
            sel = rng_global.choice(len(non_et_idx[0]), size=n_sample, replace=False)
            coords = (non_et_idx[0][sel], non_et_idx[1][sel], non_et_idx[2][sel])
            random_candidate_list.append(d1[:, coords[0], coords[1], coords[2]].T)
        del d1, img_t, stages
        torch.cuda.empty_cache()
    random_candidate_pool = np.concatenate(random_candidate_list).astype(np.float64)
    n_rand = min(n_neg_total, len(random_candidate_pool))
    random_neg = random_candidate_pool[rng_global.choice(len(random_candidate_pool), size=n_rand, replace=False)]
    X_h3 = np.concatenate([et_pool, random_neg])
    y_h3 = np.concatenate([np.ones(len(et_pool)), np.zeros(n_rand)])
    clf_h3 = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_h3, y_h3)
    w_h3, b_h3 = clf_h3.coef_[0].astype(np.float64), float(clf_h3.intercept_[0])
    print(f'H3 (random) fit: n_random={n_rand} ({time.time()-t0:.0f}s)', flush=True)

    # ---- H2/H4 (SAM-equivalent): local-offset (same subject, NOT
    # shape-aware) + global (DIFFERENT random subject, cross-subject) ----
    print('\nBuilding H2/H4 (SAM-equivalent) pools: local-offset + cross-subject global...', flush=True)
    sam_et_list, sam_local_list = [], []
    for r in det_lesions_fit:
        res = extract_et_and_sam_local(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev, rng_global)
        if res is None:
            continue
        ef, lf, sz = res
        if len(ef) == 0:
            continue
        if len(ef) > MAX_VOX_PER_LESION:
            sel = rng_global.choice(len(ef), MAX_VOX_PER_LESION, replace=False)
            ef, lf = ef[sel], lf[sel]
        sam_et_list.append(ef); sam_local_list.append(lf)
    sam_et_pool = np.concatenate(sam_et_list).astype(np.float64)
    sam_local_candidate_pool = np.concatenate(sam_local_list).astype(np.float64)
    print(f'  SAM ET pool={len(sam_et_pool)} (should match H1 ET pool exactly) '
         f'local_offset_candidates={len(sam_local_candidate_pool)} ({time.time()-t0:.0f}s)', flush=True)

    # cross-subject global negatives: per explicit user correction
    # (reusing H1's distant_candidate_pool here was found to be a real
    # design gap -- that pool is H1's OWN construction, so it would
    # have made H2/H4 differ from H1 ONLY in the local-negative
    # definition, not in both axes as planned). Fixed by using
    # random_candidate_pool instead (built for H3 above): uniform-
    # random non-ET voxels across det_train subjects with NO near-
    # shell exclusion zone (unlike extract_distant_background, which
    # explicitly excludes a 12-voxel boundary around every lesion) --
    # this is a genuinely SEPARATE pool from H1's distant negatives,
    # and faithfully reflects SAM's own principle that global negatives
    # come from elsewhere with no anatomical exclusion logic at all.
    sam_global_candidate_pool = random_candidate_pool

    n_half_sam = n_neg_total // 2
    n_l = min(n_half_sam, len(sam_local_candidate_pool))
    n_g = min(n_neg_total - n_l, len(sam_global_candidate_pool))
    local_neg = sam_local_candidate_pool[rng_global.choice(len(sam_local_candidate_pool), size=n_l, replace=False)]
    global_neg = sam_global_candidate_pool[rng_global.choice(len(sam_global_candidate_pool), size=n_g, replace=False)]
    X_h2 = np.concatenate([sam_et_pool, local_neg, global_neg])
    y_h2 = np.concatenate([np.ones(len(sam_et_pool)), np.zeros(n_l + n_g)])
    clf_h2 = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_h2, y_h2)
    w_h2, b_h2 = clf_h2.coef_[0].astype(np.float64), float(clf_h2.intercept_[0])
    print(f'H2/H4 (SAM-equivalent) fit: n_local={n_l} n_global={n_g} ({time.time()-t0:.0f}s)', flush=True)

    # ---- dense evaluation (identical protocol to E274-E277) ----
    w_p_t = torch.from_numpy(w_et.astype(np.float64)).float().to(dev)
    b_p = float(b_et)
    variant_tensors = {
        'H0_prod': (w_p_t, b_p),
        'H1_r1b': (torch.from_numpy(w_h1).float().to(dev), b_h1),
        'H2_sam_equiv': (torch.from_numpy(w_h2).float().to(dev), b_h2),
        'H3_random': (torch.from_numpy(w_h3).float().to(dev), b_h3),
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

    print(f'\nBuilding non-circular phenotype groups on {len(test_subjects)} TEST subjects '
         f'(IDENTICAL to E274, production + ground truth ONLY)...', flush=True)
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

    print('\nSweeping thresholds (identical grid to E274-E277)...', flush=True)
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
        t_500 = matched_thresholds[v][500]
        dice500 = dice_at_threshold(v, t_500)
        results.append({'readout': v, 'fp_target': 500, 'threshold': t_500,
                        'mean_fp_actual': mean_fp_by_threshold[v][t_500], 'phenotype': 'DICE', 'sensitivity': dice500})

    out = HERE / ('E278_results_smoke.csv' if smoke else 'E278_results.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['readout', 'fp_target', 'threshold', 'mean_fp_actual',
                                           'phenotype', 'sensitivity'])
        w.writeheader()
        for row in results:
            w.writerow(row)

    print(f'\nE278 complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
