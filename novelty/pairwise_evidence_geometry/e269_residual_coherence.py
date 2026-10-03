"""E269 -- Production-Orthogonal Residual Coherence, per the user's
exact spec. PHASE 1 of the full design (per explicit user scoping this
session): single neighborhood scale (3x3x3 / 26-neighbor), the three
negative controls (raw D1, random direction, R1 direction), voxel- AND
component-level analysis. Defers the 5x5x5/7x7x7 multi-scale sweep and
the combined-score (S_R) pipeline-comparison branches (A-E) to a
follow-up ONLY if this core result shows real signal.

This is the SECOND attempt at an "E269" in this session -- the first
(cumulative-k / band / random-subspace AUC localization design) was
abandoned after repeated OOM crashes around test subject 80/158 from
holding 121 simultaneous variant readouts in memory. This script is
unrelated in content (residual spatial coherence, not subspace-band
AUC) and is deliberately scoped to avoid the same failure mode: only
4 feature fields (production-orthogonal, raw D1, random-orthogonal,
R1-orthogonal) are ever computed per subject, each fully consumed
(reduced to small per-voxel/per-component summary arrays) before
moving to the next subject -- no dense per-variant volumes are
retained across subjects.

HYPOTHESIS: for the residual r_v = M_P @ x_v (x_v projected orthogonal
to production's own readout direction w_P, M_P = I - w_P w_P^T /
||w_P||^2), genuine recovered-G2A voxels have spatially MORE COHERENT
residual structure (higher mean cosine similarity C_mu to their 26
nearest neighbors' residuals, lower dispersion C_sigma) than R1-B's own
false positives -- and this coherence is SPECIFIC to the production-
orthogonal subspace, not a generic property of raw D1 or of removing
an arbitrary direction.

FOUR FIELDS computed identically (same C_mu/C_sigma/A_R machinery) for:
  1. r_v = M_P @ x_v            (production-orthogonal -- the hypothesis)
  2. r_v = x_v                   (raw D1 -- Control 1: is the projection needed at all?)
  3. r_v = M_rand @ x_v          (random-direction-orthogonal -- Control 2: is w_P specifically special?)
  4. r_v = M_R @ x_v             (R1-orthogonal -- Control 3: tied to the production-SUPPRESSED subspace specifically?)

M_rand uses ONE FIXED random direction (seeded, frozen before touching
any data) -- not re-randomized per subject, so the control is a fair,
reproducible single comparison, not noise-averaged across many random
draws (that extension is deferred to the follow-up if this core result
warrants it).

FOUR POPULATIONS (identical definitions to E266/E267/E268): recovered
G2A voxels/components, detected ET voxels/components, WT-not-TC FP,
distant-background FP. Component-level aggregation (mean C_mu, mean
C_sigma per R1-B positive connected component) directly addresses the
E267 pooling-artifact lesson (a single huge FP component must not
dominate a voxel-pooled statistic).
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
                         ROOT, PATCH, MIN_VOX, TAU, MAX_VOX_PER_LESION, NEAR_DILATION)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch  # noqa: E402

ET, TC, WT = 0, 1, 2
STRUCT = ndimage.generate_binary_structure(3, 1)
D1_DIM = 32
RNG_SEED_RANDOM_DIR = 9001
EPS = 1e-8

OFFSETS_3x3x3 = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if not (dz == 0 and dy == 0 and dx == 0)]


def projection_matrix(w):
    w = w.astype(np.float64)
    return np.eye(D1_DIM) - np.outer(w, w) / (np.dot(w, w) + EPS)


def coherence_fields(r_dense_t, dev):
    """r_dense_t: (32,D,H,W) float32 GPU tensor residual field. Returns
    numpy (A_R, C_mu, C_sigma) each (D,H,W), via the 26-neighbor
    shift-based cosine similarity computation. GPU (torch.roll) version
    of the original CPU numpy implementation -- verified exact to
    float32 precision (max diff 3e-8 for C_mu, 6e-8 for C_sigma) against
    it on a synthetic (32,20,20,20) test case before replacing it;
    ~15.6x faster at full 128^3 scale (23.8s/4 fields CPU vs 1.5s/4
    fields GPU), which was necessary -- the CPU version made this
    experiment's smoke-scale runtime project to ~92 minutes at full
    scale, prohibitively slow for what is only phase 1 of a larger
    planned design."""
    C, D, H, W = r_dense_t.shape
    A_R = torch.linalg.norm(r_dense_t, dim=0)
    norm_safe = torch.where(A_R < EPS, torch.full_like(A_R, EPS), A_R)
    r_unit = r_dense_t / norm_safe.unsqueeze(0)

    cos_sum = torch.zeros((D, H, W), dtype=torch.float64, device=dev)
    cos_sq_sum = torch.zeros((D, H, W), dtype=torch.float64, device=dev)
    count = torch.zeros((D, H, W), dtype=torch.int32, device=dev)
    for dz, dy, dx in OFFSETS_3x3x3:
        shifted = torch.roll(r_unit, shifts=(-dz, -dy, -dx), dims=(1, 2, 3))
        valid = torch.ones((D, H, W), dtype=torch.bool, device=dev)
        if dz == -1: valid[0, :, :] = False
        if dz == 1: valid[-1, :, :] = False
        if dy == -1: valid[:, 0, :] = False
        if dy == 1: valid[:, -1, :] = False
        if dx == -1: valid[:, :, 0] = False
        if dx == 1: valid[:, :, -1] = False
        cos = (r_unit * shifted).sum(dim=0)
        zeros64 = torch.zeros_like(cos, dtype=torch.float64)
        cos_sum += torch.where(valid, cos.double(), zeros64)
        cos_sq_sum += torch.where(valid, (cos ** 2).double(), zeros64)
        count += valid.int()
    count_safe = torch.clamp(count, min=1)
    C_mu = cos_sum / count_safe
    C_var = torch.clamp(cos_sq_sum / count_safe - C_mu ** 2, min=0)
    C_sigma = torch.sqrt(C_var)
    return (A_R.float().cpu().numpy(), C_mu.float().cpu().numpy(), C_sigma.float().cpu().numpy())


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
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])[:20]
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        g2a_lesions_test = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']]
        det_lesions_test = [r for r in det_lesions if r['subject_id'] in splits['det_test']]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    rng = np.random.default_rng(999)
    t0 = time.time()

    print('Fitting R1-B (identical to E257-B)...', flush=True)
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
    w_r1b = r1b.coef_[0].astype(np.float64)
    b_r1b = float(r1b.intercept_[0])
    print(f'R1-B fit ({time.time()-t0:.0f}s)', flush=True)

    w_p = w_et.astype(np.float64)
    rng_rand = np.random.default_rng(RNG_SEED_RANDOM_DIR)
    w_rand = rng_rand.normal(size=D1_DIM)

    M_P = projection_matrix(w_p)
    M_raw = np.eye(D1_DIM)  # Control 1: identity (no projection)
    M_rand = projection_matrix(w_rand)
    M_R = projection_matrix(w_r1b)

    fields = {
        'prod_orth': M_P,
        'raw_d1': M_raw,
        'rand_orth': M_rand,
        'r1b_orth': M_R,
    }
    field_tensors = {name: torch.from_numpy(M.T).float().to(dev) for name, M in fields.items()}

    w_r1b_t = torch.from_numpy(w_r1b).float().to(dev)

    def dense_residuals(sid):
        """Returns dict field_name -> (A_R,C_mu,C_sigma) each (D,H,W),
        plus p_r1b_map, tgt_c, img_c. Computed and consumed per-subject;
        nothing dense is retained after the caller extracts what it needs."""
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]  # (32,D,H,W)
        C, D, H, W = d1.shape
        flat = d1.reshape(C, -1).T  # (N,32)
        with torch.no_grad():
            p_r1b_flat = torch.sigmoid(flat @ w_r1b_t + b_r1b)
        p_r1b_map = p_r1b_flat.reshape(D, H, W).cpu().numpy()

        out = {}
        for name, M_t in field_tensors.items():
            with torch.no_grad():
                r_flat = flat @ M_t  # (N,32), M_t = M.T so r = x @ M.T == (M @ x^T)^T row-wise
                r_dense_t = r_flat.T.reshape(C, D, H, W)
            out[name] = coherence_fields(r_dense_t, dev)
            del r_flat, r_dense_t
        del d1, flat, img_t, stages
        torch.cuda.empty_cache()
        return out, p_r1b_map, tgt_c, img_c

    # ---- verify the M_t transpose convention is exact before trusting
    # any real data (x @ M.T gives the same result as M @ x for each
    # row x, since M is symmetric: M@x == (x@M.T) when M=M.T) ----
    _M_test = M_P
    _x_test = np.random.default_rng(0).normal(size=(5, D1_DIM))
    _direct = (_M_test @ _x_test.T).T
    _via_T = _x_test @ _M_test.T
    assert np.allclose(_direct, _via_T, atol=1e-10), 'projection convention mismatch'
    print('Projection convention verified exact.', flush=True)

    def et_lbl_for(tgt_c):
        return ndimage.label(tgt_c[ET] > 0.5)

    def collect_subject(sid, g2a_cids, det_cids):
        """Processes ONE subject fully: computes all 4 fields' dense
        coherence volumes, extracts voxel-level rows for the 4
        populations AND component-level aggregates (over R1-B's own
        positive-mask components), then returns compact lists -- no
        dense arrays survive this function."""
        fields_out, p_r1b_map, tgt_c, img_c = dense_residuals(sid)
        brain_mask = img_c[0] != 0
        true_et = tgt_c[ET] > 0.5
        true_tc = tgt_c[TC] > 0.5
        true_wt = tgt_c[WT] > 0.5
        et_lbl, et_n = et_lbl_for(tgt_c)

        dilated = ndimage.binary_dilation(true_et, structure=STRUCT,
                                          iterations=NEAR_DILATION) if true_et.any() else np.zeros_like(true_et)
        local_shell = dilated & (~true_et) & brain_mask
        distant_bg_mask = brain_mask & (~true_et) & (~local_shell)

        pos_mask = (p_r1b_map > TAU) & brain_mask
        pos_lbl, pos_n = ndimage.label(pos_mask, structure=STRUCT)

        def lesion_masks(cids):
            out = []
            for cid in cids:
                if cid < 1 or cid > et_n:
                    continue
                lm = et_lbl == cid
                if int(lm.sum()) < MIN_VOX:
                    continue
                out.append(lm)
            return out

        g2a_masks = lesion_masks(g2a_cids)
        det_masks = lesion_masks(det_cids)

        voxel_rows = []  # (population, field, A_R, C_mu, C_sigma)
        comp_rows = []   # (population, field, A_R_mean, C_mu_mean, C_sigma_mean, size)

        def add_voxel_pop(mask, pop_name, stride=1):
            idx = np.where(mask)
            if len(idx[0]) == 0:
                return
            if stride > 1:
                sel = np.arange(0, len(idx[0]), stride)
                idx = (idx[0][sel], idx[1][sel], idx[2][sel])
            for name, (A_R, C_mu, C_sigma) in fields_out.items():
                voxel_rows.append({
                    'population': pop_name, 'field': name,
                    'A_R_mean': float(A_R[idx].mean()), 'A_R_std': float(A_R[idx].std()),
                    'C_mu_mean': float(C_mu[idx].mean()), 'C_mu_std': float(C_mu[idx].std()),
                    'C_sigma_mean': float(C_sigma[idx].mean()), 'n_voxels': len(idx[0]),
                })

        for lm in g2a_masks:
            add_voxel_pop(lm, 'recovered_G2A')
        for lm in det_masks:
            add_voxel_pop(lm, 'detected_ET')
        wt_not_tc_fp_mask = pos_mask & (~true_et) & true_wt & (~true_tc)
        add_voxel_pop(wt_not_tc_fp_mask, 'WT_not_TC_FP', stride=5)
        distant_fp_mask = pos_mask & (~true_et) & distant_bg_mask
        add_voxel_pop(distant_fp_mask, 'distant_bg_FP', stride=5)

        # component-level: for each R1-B positive component, classify
        # by overlap (same convention as E262/E264: is_tp if it
        # overlaps true_et at all; else WT-not-TC vs distant by
        # majority-overlap, matching E259's convention)
        for cid in range(1, pos_n + 1):
            comp_mask = pos_lbl == cid
            size = int(comp_mask.sum())
            if size < 1:
                continue
            is_tp = bool((comp_mask & true_et).any())
            if is_tp:
                comp_pop = 'recovered_or_detected_component'
            else:
                frac_tc = (comp_mask & true_tc).sum() / size
                frac_wt_not_tc = (comp_mask & true_wt & (~true_tc)).sum() / size
                if frac_tc > 0.5 or frac_wt_not_tc > 0.5:
                    comp_pop = 'WT_not_TC_FP_component'
                else:
                    comp_pop = 'distant_bg_FP_component'
            for name, (A_R, C_mu, C_sigma) in fields_out.items():
                comp_rows.append({
                    'subject_id': sid, 'population': comp_pop, 'field': name,
                    'A_R_mean': float(A_R[comp_mask].mean()),
                    'C_mu_mean': float(C_mu[comp_mask].mean()),
                    'C_sigma_mean': float(C_sigma[comp_mask].mean()),
                    'size': size,
                })

        del fields_out, p_r1b_map, tgt_c, img_c, pos_mask, pos_lbl
        return voxel_rows, comp_rows

    # ---- run on TEST subjects (locked split, same as E264-E268) ----
    test_subjects = sorted(set(r['subject_id'] for r in det_lesions_test) |
                          set(r['subject_id'] for r in g2a_lesions_test))
    if smoke:
        test_subjects = test_subjects[:15]

    g2a_by_sid, det_by_sid = {}, {}
    for r in g2a_lesions_test:
        g2a_by_sid.setdefault(r['subject_id'], []).append(int(r['comp_id']))
    for r in det_lesions_test:
        det_by_sid.setdefault(r['subject_id'], []).append(int(r['comp_id']))

    print(f'\nProcessing {len(test_subjects)} TEST subjects...', flush=True)
    voxel_out = HERE / ('E269b_voxel_smoke.csv' if smoke else 'E269b_voxel.csv')
    comp_out = HERE / ('E269b_component_smoke.csv' if smoke else 'E269b_component.csv')
    fh_v = open(voxel_out, 'w', newline='')
    fh_c = open(comp_out, 'w', newline='')
    w_v = csv.DictWriter(fh_v, fieldnames=['population', 'field', 'A_R_mean', 'A_R_std',
                                           'C_mu_mean', 'C_mu_std', 'C_sigma_mean', 'n_voxels'])
    w_c = csv.DictWriter(fh_c, fieldnames=['subject_id', 'population', 'field',
                                           'A_R_mean', 'C_mu_mean', 'C_sigma_mean', 'size'])
    w_v.writeheader(); w_c.writeheader()

    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        g2a_cids = g2a_by_sid.get(sid, [])
        det_cids = det_by_sid.get(sid, [])
        voxel_rows, comp_rows = collect_subject(sid, g2a_cids, det_cids)
        for row in voxel_rows:
            w_v.writerow(row)
        for row in comp_rows:
            w_c.writerow(row)
        fh_v.flush(); fh_c.flush()
        if (i + 1) % 10 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)

    fh_v.close(); fh_c.close()
    print(f'\nE269 (residual coherence, phase 1) complete ({time.time()-t0:.0f}s). '
         f'wrote {voxel_out.name}, {comp_out.name}', flush=True)


if __name__ == '__main__':
    main()
