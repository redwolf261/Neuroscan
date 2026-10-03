"""E273 -- readout-function decomposition, per the user's exact spec.
Direct follow-up to E272, which closed the subspace-geometry axis
(Outcome C: (w_P,w_R) is not special relative to 100 random 2D
subspaces at the removal-recovery level) while CONFIRMING a narrower,
still-valid finding: w_P's own 1D direction is a genuine, specific
bottleneck (P_P_only collapses to 2.78% vs D1_full's 58.3%). The user
explicitly redirected the question from "WHERE in D1 is the missing
information" (closed, E268-E272) to "WHAT does the independently
trained readout (R1-B, direction w_R) do differently from production
(direction w_P) that lets it access that information."

PART 1 ONLY (per explicit user scoping this session): the 2x2
direction-vs-bias decomposition and the systematic-positivity check.
The multi-classifier-type stability comparison (logistic/ridge/SVM/
MLP) is explicitly deferred to a follow-up, only if this core result
is clean enough to warrant checking it is not a logistic-regression-
specific artifact.

FOUR READOUTS (clean 2x2 factorial over direction x bias; A and D are
NOT new -- they are production and R1-B exactly as already evaluated
throughout E257-E272; B and C are the genuinely new cells):
                       b_P (production bias)   b_R (R1-B bias)
  w_P (production dir)         A                      C
  w_R (R1-B dir)                B                      D

  A: z = w_P^T x + b_P    (== production, reused unchanged)
  B: z = w_R^T x + b_P    (direction REPLACED, bias kept at production's)
  C: z = w_P^T x + b_R    (direction kept at production's, bias REPLACED)
  D: z = w_R^T x + b_R    (== R1-B, reused unchanged)

Per the user's own framing: E252 already found the production/R1-B
discrepancy is DIRECTIONAL not calibration (cos(w)=0.055, far from
aligned) -- so C (bias-only replacement) is expected to remain near
production's own near-zero recovery. The genuinely informative cell is
B: if w_R^T x + b_P (R1-B's direction, production's own bias) recovers
G2-A nearly as well as full R1-B (D), the mechanism is "direction
replacement is what matters," a cleaner and more specific statement
than "train an independent readout."

SECOND ANALYSIS: Delta_w = w_R - w_P. For G2-A (recovered and missed),
detected-ET, and background-FP voxels, test whether Delta_w^T x is
systematically positive specifically where it needs to be (G2-A) vs
elsewhere -- per the user's interpretation, if true this supports
"R1-B assigns positive decision weight to a feature combination
production largely ignores or suppresses," a specific mechanistic
reading rather than "R1-B found new information."

Same TRAIN/VAL/TEST split and R1-B fit as every prior experiment in
this chain (E257-E272); same fixed TAU=0.5 convention (per E272's own
caught-and-fixed lesson -- threshold comparability to all prior
reported numbers matters more than a literal per-variant calibration).
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e257_common import (load_model, get_w_prod, load_populations, build_splits,  # noqa: E402
                         extract_lesion_shell, extract_distant_background,
                         ROOT, PATCH, MIN_VOX, MAX_VOX_PER_LESION, TAU, NEAR_DILATION)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch  # noqa: E402

ET, TC, WT = 0, 1, 2
STRUCT = ndimage.generate_binary_structure(3, 1)
D1_DIM = 32
EPS = 1e-12


def dice_score(pred_mask, gt_mask):
    ps, gs = pred_mask.sum(), gt_mask.sum()
    if gs == 0:
        return 1.0 if ps == 0 else float('nan')
    return float(2 * (pred_mask & gt_mask).sum() / (ps + gs))


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    model = load_model(dev)
    w_et, b_et = get_w_prod(model)
    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)

    if smoke:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']][:30]
        g2a_lesions_test = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']][:15]
        det_lesions_test = [r for r in det_lesions if r['subject_id'] in splits['det_test']][:15]
        g2b_lesions_eval = g2b_lesions[:15]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])[:20]
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        g2a_lesions_test = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']]
        det_lesions_test = [r for r in det_lesions if r['subject_id'] in splits['det_test']]
        g2b_lesions_eval = g2b_lesions
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    rng = np.random.default_rng(999)
    t0 = time.time()

    print('Building TRAIN voxel pool and fitting R1-B...', flush=True)
    lesion_pool, neg_pool = [], []
    for r in det_lesions_fit:
        res = extract_lesion_shell(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
        if res is None:
            continue
        lf, sf, sz = res
        if len(lf) > MAX_VOX_PER_LESION:
            lf = lf[rng.choice(len(lf), MAX_VOX_PER_LESION, replace=False)]
        if len(sf) > MAX_VOX_PER_LESION:
            sf = sf[rng.choice(len(sf), MAX_VOX_PER_LESION, replace=False)]
        lesion_pool.append(lf); neg_pool.append(sf)
    for sid in bg_subjects:
        feats = extract_distant_background(model, ds, sid_to_idx, sid, dev, rng)
        if feats is None:
            continue
        neg_pool.append(feats)
    lesion_pool = np.concatenate(lesion_pool); neg_pool = np.concatenate(neg_pool)
    X_train = np.concatenate([lesion_pool, neg_pool]).astype(np.float64)
    y_train = np.concatenate([np.ones(len(lesion_pool)), np.zeros(len(neg_pool))])
    r1b = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_train, y_train)
    w_r = r1b.coef_[0].astype(np.float64)
    b_r = float(r1b.intercept_[0])
    w_p = w_et.astype(np.float64)
    b_p = float(b_et)
    print(f'R1-B fit ({time.time()-t0:.0f}s)', flush=True)

    cos_wp_wr = float(np.dot(w_p, w_r) / (np.linalg.norm(w_p) * np.linalg.norm(w_r) + EPS))
    print(f'cos(w_P, w_R) = {cos_wp_wr:.4f} (E252 reported 0.055 in a different geometry; '
         f'reused-unchanged R1-B direction here)', flush=True)

    # ---- 2x2 decomposition: (w_p,b_p)=A, (w_r,b_p)=B, (w_p,b_r)=C, (w_r,b_r)=D ----
    variants = {
        'A_production': (w_p, b_p),
        'B_direction_replaced': (w_r, b_p),
        'C_bias_replaced': (w_p, b_r),
        'D_full_R1B': (w_r, b_r),
    }
    variant_tensors = {name: (torch.from_numpy(w).float().to(dev), b) for name, (w, b) in variants.items()}

    def dense_eval(sid):
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
        d1_np = d1.cpu().numpy()
        del d1, flat, img_t, stages
        torch.cuda.empty_cache()
        return maps, d1_np, tgt_c, img_c

    test_subjects = sorted(set(r['subject_id'] for r in det_lesions_test) |
                          set(r['subject_id'] for r in g2a_lesions_test) |
                          set(r['subject_id'] for r in g2b_lesions_eval))
    if smoke:
        test_subjects = test_subjects[:15]

    print(f'\nEvaluating 2x2 decomposition on {len(test_subjects)} TEST subjects...', flush=True)

    per_subject_dice = {v: [] for v in variants}
    fp_total = {v: 0 for v in variants}
    g2a_recs = {v: [] for v in variants}
    det_recs = {v: [] for v in variants}
    g2b_recs = {v: [] for v in variants}

    delta_w = w_r - w_p
    dw_vals = {'recovered_G2A': [], 'missed_G2A': [], 'detected_ET': [], 'distant_bg': []}

    g2a_by_sid, det_by_sid, g2b_by_sid = {}, {}, {}
    for r in g2a_lesions_test:
        g2a_by_sid.setdefault(r['subject_id'], []).append(int(r['comp_id']))
    for r in det_lesions_test:
        det_by_sid.setdefault(r['subject_id'], []).append(int(r['comp_id']))
    for r in g2b_lesions_eval:
        g2b_by_sid.setdefault(r['subject_id'], []).append(int(r['comp_id']))

    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        maps, d1_np, tgt_c, img_c = dense_eval(sid)
        true_et = tgt_c[ET] > 0.5
        brain_mask = img_c[0] != 0
        et_lbl, et_n = ndimage.label(true_et)

        def valid_lesion_masks(cids):
            out_m = []
            for cid in cids:
                if cid < 1 or cid > et_n:
                    continue
                lm = et_lbl == cid
                if int(lm.sum()) < MIN_VOX:
                    continue
                out_m.append(lm)
            return out_m

        g2a_masks = valid_lesion_masks(g2a_by_sid.get(sid, []))
        det_masks = valid_lesion_masks(det_by_sid.get(sid, []))
        g2b_masks = valid_lesion_masks(g2b_by_sid.get(sid, []))

        for name, p_map in maps.items():
            mask = (p_map > TAU) & brain_mask
            per_subject_dice[name].append(dice_score(mask, true_et))
            fp_total[name] += int((mask & (~true_et)).sum())
            for lm in g2a_masks:
                g2a_recs[name].append(int((mask & lm).any()))
            for lm in det_masks:
                det_recs[name].append(int((mask & lm).any()))
            for lm in g2b_masks:
                g2b_recs[name].append(int((mask & lm).any()))

        # Delta_w^T x analysis, using D full R1-B's own recovery mask to
        # split G2A voxels into recovered vs missed sub-populations
        d_flat_vals = np.einsum('c,cdhw->dhw', delta_w, d1_np)
        r1b_mask = (maps['D_full_R1B'] > TAU) & brain_mask
        for lm in g2a_masks:
            recovered_part = lm & r1b_mask
            missed_part = lm & (~r1b_mask)
            if recovered_part.any():
                dw_vals['recovered_G2A'].extend(d_flat_vals[recovered_part].tolist())
            if missed_part.any():
                dw_vals['missed_G2A'].extend(d_flat_vals[missed_part].tolist())
        for lm in det_masks:
            dw_vals['detected_ET'].extend(d_flat_vals[lm].tolist())
        dilated = ndimage.binary_dilation(true_et, structure=STRUCT,
                                          iterations=NEAR_DILATION) if true_et.any() else np.zeros_like(true_et)
        local_shell = dilated & (~true_et) & brain_mask
        distant_bg_mask = brain_mask & (~true_et) & (~local_shell)
        distant_sample = np.where(distant_bg_mask)
        if len(distant_sample[0]) > 0:
            sel = rng.choice(len(distant_sample[0]), size=min(200, len(distant_sample[0])), replace=False)
            coords = (distant_sample[0][sel], distant_sample[1][sel], distant_sample[2][sel])
            dw_vals['distant_bg'].extend(d_flat_vals[coords].tolist())

        del maps, d1_np, d_flat_vals
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)

    print('\nComputing final metrics...', flush=True)
    results = {}
    for name in variants:
        results[name] = {
            'g2a_recovery': float(np.mean(g2a_recs[name])) if g2a_recs[name] else float('nan'),
            'det_recovery': float(np.mean(det_recs[name])) if det_recs[name] else float('nan'),
            'g2b_recovery': float(np.mean(g2b_recs[name])) if g2b_recs[name] else float('nan'),
            'et_dice_mean': float(np.nanmean(per_subject_dice[name])),
            'fp_total': fp_total[name],
        }

    print(f'\n=== E273 2x2 DECOMPOSITION RESULTS ===')
    for k, v in results.items():
        print(f'{k}: {v}')

    out = HERE / ('E273_decomposition_smoke.csv' if smoke else 'E273_decomposition.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['variant', 'g2a_recovery', 'det_recovery', 'g2b_recovery',
                                           'et_dice_mean', 'fp_total'])
        w.writeheader()
        for name, v in results.items():
            w.writerow({'variant': name, **v})

    print(f'\n=== Delta_w^T x ANALYSIS (Delta_w = w_R - w_P) ===')
    dw_summary = {}
    for pop, vals in dw_vals.items():
        arr = np.array(vals)
        dw_summary[pop] = {'n': len(arr), 'mean': float(arr.mean()) if len(arr) else float('nan'),
                           'median': float(np.median(arr)) if len(arr) else float('nan'),
                           'frac_positive': float((arr > 0).mean()) if len(arr) else float('nan')}
        print(f'{pop}: n={dw_summary[pop]["n"]} mean={dw_summary[pop]["mean"]:.4f} '
             f'median={dw_summary[pop]["median"]:.4f} frac_positive={dw_summary[pop]["frac_positive"]:.4f}')

    if dw_vals['recovered_G2A'] and dw_vals['distant_bg']:
        stat, p = mannwhitneyu(dw_vals['recovered_G2A'], dw_vals['distant_bg'], alternative='greater')
        print(f'MWU (recovered_G2A > distant_bg): p={p:.2e}')
    if dw_vals['missed_G2A'] and dw_vals['distant_bg']:
        stat, p2 = mannwhitneyu(dw_vals['missed_G2A'], dw_vals['distant_bg'], alternative='greater')
        print(f'MWU (missed_G2A > distant_bg): p={p2:.2e}')

    dw_out = HERE / ('E273_delta_w_smoke.csv' if smoke else 'E273_delta_w.csv')
    with open(dw_out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['population', 'n', 'mean', 'median', 'frac_positive'])
        for pop, s in dw_summary.items():
            w.writerow([pop, s['n'], s['mean'], s['median'], s['frac_positive']])

    print(f'\nE273 complete ({time.time()-t0:.0f}s). wrote {out.name}, {dw_out.name}', flush=True)


if __name__ == '__main__':
    main()
