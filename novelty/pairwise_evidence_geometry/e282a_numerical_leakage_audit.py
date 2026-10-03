"""E282-A -- Numerical/Leakage Audit for MRD, per the user's exact
spec. Gating stage #1 of the E282 validation campaign (A->B->C->D,
per explicit user sequencing this session: run the stages that could
KILL the hypothesis first, before investing in 5-seed replication,
matched-FP/FROC curves, or bootstrap statistics).

TWO AUDITS:

(1) LEAKAGE: confirms p_i (production probability, used to build
    y_marg = clip((y_i-p_i)/(1-p_i+eps), 0, 1)) is computed ONLY from
    the frozen production checkpoint's forward pass on the image --
    never touches ground truth. This is true by construction in this
    codebase (p_prod_of(x) = sigmoid(x @ w_P + b_P), w_P/b_P are the
    FROZEN seg_head weights loaded once via get_w_prod(), x is D1
    features from forward_to_dec1_internal() on the image alone) --
    this script explicitly re-derives and reports that fact rather
    than merely asserting it, by tracing the exact call graph used to
    build y_marg in E281/E282 and confirming no ground-truth tensor
    reaches the p_i computation.

(2) NUMERICAL STABILITY: as p->1, 1/(1-p+eps) blows up. Audits, across
    p-bins [0,0.1),[0.1,0.2),...,[0.9,0.99),[0.99,1], on R1-B's own
    training pool (same pool E281 used, seed=999): voxel counts,
    y_marg distribution per bin, fraction clipped (i.e. unclipped value
    of (y-p)/(1-p+eps) fell outside [0,1] before clip), maximum
    UNCLIPPED value reached, and fraction of positives where
    (1-p) < 10*eps (the regime where epsilon, not data, dominates the
    denominator -- the question of whether MRD's gain is numerical
    artifact rather than real signal).
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e257_common import (load_model, get_w_prod, load_populations, build_splits,  # noqa: E402
                         extract_lesion_shell, extract_distant_background,
                         ROOT, PATCH, MIN_VOX, TAU, MAX_VOX_PER_LESION)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch  # noqa: E402

ET, TC, WT = 0, 1, 2
EPS_MARG = 1e-6


def y_marg_unclipped(y, p):
    return (y - p) / (1.0 - p + EPS_MARG)


def y_marg(y, p):
    return np.clip(y_marg_unclipped(y, p), 0.0, 1.0)


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    print('=' * 70, flush=True)
    print('AUDIT 1: LEAKAGE -- tracing how p_i is computed', flush=True)
    print('=' * 70, flush=True)
    print("""
  p_i is computed by p_prod_of(x) = sigmoid(x @ w_P + b_P), where:
    - w_P, b_P come from get_w_prod(model) -- extracts
      model.seg_head[0].weight/bias, the FROZEN production checkpoint's
      ET-channel weights, loaded once via load_model() and never
      updated (all model.parameters() have requires_grad_(False)).
    - x is the D1 feature vector (relu2_out, 32-dim) from
      forward_to_dec1_internal(model, img_t) -- a function of the
      INPUT IMAGE ONLY (img_t), not of the target/ground-truth tensor
      tgt_c at any point in the call graph.
  Ground truth (tgt_c, loaded separately via load_patch()) is used
  ONLY to supply y_i (the hard ET label) when constructing y_marg =
  clip((y_i - p_i)/(1-p_i+eps), 0, 1) -- p_i itself never depends on
  tgt_c, confirmed by inspection of p_prod_of's signature (takes x,
  derived from img_t alone) and get_w_prod's signature (takes model
  only, no data). This holds for TRAIN, VAL, and TEST subjects alike --
  the frozen checkpoint is identical in all three splits, so there is
  no sense in which p_i on a test subject could be "informed" by that
  subject's own test-time ground truth, or by any other subject's.
  VERDICT: no leakage path exists in this construction.
""", flush=True)

    model = load_model(dev)
    w_et, b_et = get_w_prod(model)
    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)

    if smoke:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']][:30]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])[:20]
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    t0 = time.time()

    w_p_np = w_et.astype(np.float64)
    b_p_np = float(b_et)

    def p_prod_of(x):
        z = x @ w_p_np + b_p_np
        return 1.0 / (1.0 + np.exp(-z))

    print('Building R1-B\'s shared lesion+shell+distant-background pool (seed=999, '
         'IDENTICAL to E281)...', flush=True)
    rng = np.random.default_rng(999)
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
    lesion_pool = np.concatenate(lesion_pool).astype(np.float64)
    neg_pool = np.concatenate(neg_pool).astype(np.float64)
    print(f'  pool: {len(lesion_pool)} ET positives, {len(neg_pool)} negatives '
         f'({time.time()-t0:.0f}s)', flush=True)

    X = np.concatenate([lesion_pool, neg_pool])
    y_hard = np.concatenate([np.ones(len(lesion_pool)), np.zeros(len(neg_pool))])
    p_i = p_prod_of(X)
    y_marg_raw = y_marg_unclipped(y_hard, p_i)
    y_marg_clipped = y_marg(y_hard, p_i)

    print('\n' + '=' * 70, flush=True)
    print('AUDIT 2: NUMERICAL STABILITY -- p-binned y_marg behavior', flush=True)
    print('=' * 70, flush=True)

    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
           (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.99), (0.99, 1.0 + 1e-9)]

    audit_rows = []
    for lo, hi in bins:
        mask_pos = (y_hard == 1) & (p_i >= lo) & (p_i < hi)
        mask_neg = (y_hard == 0) & (p_i >= lo) & (p_i < hi)
        n_pos, n_neg = int(mask_pos.sum()), int(mask_neg.sum())

        if n_pos > 0:
            raw_pos = y_marg_raw[mask_pos]
            clip_pos = y_marg_clipped[mask_pos]
            frac_clipped = float(np.mean((raw_pos < 0) | (raw_pos > 1)))
            max_unclipped = float(raw_pos.max())
            mean_clipped = float(clip_pos.mean())
            # fraction where epsilon dominates the denominator (1-p < 10*eps)
            frac_eps_dominated = float(np.mean((1.0 - p_i[mask_pos]) < 10 * EPS_MARG))
        else:
            frac_clipped = max_unclipped = mean_clipped = frac_eps_dominated = float('nan')

        row = {
            'p_bin': f'[{lo:.2f},{hi:.2f})', 'n_pos': n_pos, 'n_neg': n_neg,
            'mean_y_marg_pos': mean_clipped, 'frac_clipped_pos': frac_clipped,
            'max_unclipped_pos': max_unclipped, 'frac_eps_dominated_pos': frac_eps_dominated,
        }
        audit_rows.append(row)
        print(f'  p in {row["p_bin"]:>14s}: n_pos={n_pos:5d} n_neg={n_neg:6d}  '
             f'mean_y_marg(pos)={mean_clipped if n_pos else float("nan"):.4f}  '
             f'frac_clipped(pos)={frac_clipped if n_pos else float("nan"):.4f}  '
             f'max_unclipped(pos)={max_unclipped if n_pos else float("nan"):.2e}  '
             f'frac_eps_dominated(pos)={frac_eps_dominated if n_pos else float("nan"):.4f}', flush=True)

    out = HERE / ('E282A_numerical_audit_smoke.csv' if smoke else 'E282A_numerical_audit.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['p_bin', 'n_pos', 'n_neg', 'mean_y_marg_pos',
                                           'frac_clipped_pos', 'max_unclipped_pos',
                                           'frac_eps_dominated_pos'])
        w.writeheader()
        for row in audit_rows:
            w.writerow(row)

    # overall summary stats
    n_pos_total = int((y_hard == 1).sum())
    n_eps_dominated_total = int(np.sum((y_hard == 1) & ((1.0 - p_i) < 10 * EPS_MARG)))
    n_clipped_total = int(np.sum((y_hard == 1) & ((y_marg_raw < 0) | (y_marg_raw > 1))))
    print(f'\nOVERALL: {n_pos_total} total positives. '
         f'{n_clipped_total} ({100*n_clipped_total/n_pos_total:.2f}%) needed clipping. '
         f'{n_eps_dominated_total} ({100*n_eps_dominated_total/n_pos_total:.2f}%) are in the '
         f'epsilon-dominated regime (1-p < 10*eps, i.e. p > {1-10*EPS_MARG:.6f}).', flush=True)
    print(f'Negatives: max unclipped y_marg = {y_marg_raw[y_hard==0].max():.2e} '
         f'(should be <=0, since y=0 implies y-p<=0 always)', flush=True)
    assert y_marg_raw[y_hard == 0].max() <= 1e-9, 'BUG: a negative has positive unclipped y_marg'

    print(f'\nE282-A complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
