"""E263 -- Counterfactual Readout Curvature (CRC), per the user's exact
spec (corrected from an earlier proposal after noting that a linear
logit's second derivative is exactly zero -- curvature must be measured
in the PROBABILITY response, not the raw logit).

CRC operates entirely AFTER D1, using ONLY w_P (production's own ET
row of seg_head) and w_R (R1-B's independently-fit readout, refit
identically to E257-B/E258/E259/E260/E262). No new network, no new
training data beyond what already produced R1-B.

    d      = unit(unit(w_R) - unit(w_P))                  disagreement direction
    z^pm   = z +/- delta*d                                counterfactual probe
    p^pm   = sigmoid(w_P . z^pm + b_P)                    production response to probe
    G      = (p+ - p-) / (2*delta)                        directional response (1st deriv)
    C      = (p+ - 2p0 + p-) / delta^2                    response curvature (2nd deriv)
    E      = G / (1 + kappa*|C|)                            curvature-normalized evidence
    a      = |G|
    g      = a / (a + mu)                                  adaptive gate
    l_CRC  = l_P + lambda * g * tanh(E / tau)              bounded correction
    p_CRC  = sigmoid(l_CRC)

ABLATION LADDER (per explicit user table):
    production        -- baseline
    r1b               -- E257-B, strongest current baseline
    crc_no_curvature  -- l_P + lambda*g*tanh(G/tau)         (drop the 1+kappa|C| normalizer: E->G)
    crc_no_gating     -- l_P + lambda*tanh(E/tau)            (drop g_i, i.e. g=1 everywhere)
    crc_full          -- full module as specified above

HYPERPARAMETERS (delta, lambda, kappa, tau, mu): NOT given numeric
values in the user's spec. Per explicit user choice this session
("principled defaults + smoke-test tuning"), values are derived from
the data itself on the TRAIN population before any test-set contact:
  - delta: 0.1 * median(||z||) over the training lesion/shell/bg pool
           (a probe step that is a real but small fraction of typical
           D1 activation magnitude -- large enough to get a non-
           degenerate finite-difference signal, small enough to stay
           local).
  - lambda: the logit distance from the median NEGATIVE-population
            (shell+background) logit to the TAU=0.5 decision boundary
            (logit=0), i.e. lambda = |median(l_P over negatives)|.
            This bounds the correction to roughly "can move a typical
            negative voxel at most to the boundary, not past it by an
            unbounded amount" -- directly addressing the E255 lesson
            (dense exposure must be bounded).
  - kappa, tau, mu: set from smoke-scale percentiles of |C|, G, and
            |G| respectively (median of each), so the tanh and gate
            operate in their non-degenerate middle range rather than
            saturating everywh000 or nowhere. Printed and sanity-
            checked before the full run.

EVALUATION (identical protocol to E257-B/E260/E261): dense whole-volume
FP counts (total, WT-not-TC, outside-tumor per E259's convention) on
all held-out test subjects, lesion-level recovery for detected_test /
G2A_test / G2B, for all 5 ladder variants.
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e257_common import (load_model, get_w_prod, load_populations, build_splits,  # noqa: E402
                         extract_lesion_shell, extract_distant_background,
                         ROOT, PATCH, MIN_VOX, TAU, MAX_VOX_PER_LESION)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch  # noqa: E402

ET, TC, WT = 0, 1, 2


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

    # ---- build shared training voxel pools (identical to E257-B/E260/E261/E262) ----
    print('Building training voxel pools + fitting R1-B (== E257-B)...', flush=True)
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
    lesion_pool = np.concatenate(lesion_pool)
    neg_pool = np.concatenate(neg_pool)
    print(f'  lesion_pool n={len(lesion_pool)}  neg_pool n={len(neg_pool)} ({time.time()-t0:.0f}s)', flush=True)

    X_b = np.concatenate([lesion_pool, neg_pool])
    y_b = np.concatenate([np.ones(len(lesion_pool)), np.zeros(len(neg_pool))])
    r1b = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_b, y_b)
    w_r1b_np = r1b.coef_[0].astype(np.float64)
    b_r1b = float(r1b.intercept_[0])
    print(f'R1-B fit ({time.time()-t0:.0f}s)', flush=True)

    # ---- derive CRC direction d ----
    w_p_unit = w_et / (np.linalg.norm(w_et) + 1e-8)
    w_r_unit = w_r1b_np / (np.linalg.norm(w_r1b_np) + 1e-8)
    diff = w_r_unit - w_p_unit
    d_np = diff / (np.linalg.norm(diff) + 1e-8)
    print(f'cos(w_P,w_R)={np.dot(w_p_unit, w_r_unit):.4f}  ||d||={np.linalg.norm(d_np):.4f}', flush=True)

    # ---- derive hyperparameters from TRAIN population only ----
    delta = 0.1 * float(np.median(np.linalg.norm(X_b, axis=1)))
    logits_neg = neg_pool @ w_et + b_et
    lam = float(np.abs(np.median(logits_neg)))
    print(f'delta={delta:.4f}  lambda={lam:.4f} (from median ||z||={np.median(np.linalg.norm(X_b, axis=1)):.4f}, '
         f'median negative logit={np.median(logits_neg):.4f})', flush=True)

    w_r1b_t = torch.from_numpy(w_r1b_np).float().to(dev)
    w_p_d = torch.from_numpy(w_et).double().to(dev)
    d_d = torch.from_numpy(d_np).double().to(dev)
    b_p = float(b_et)

    def crc_components(z_flat):
        """z_flat: (N,32) torch tensor (any dtype). Computed in float64
        throughout to avoid float32 sigmoid-flattening for the many
        deeply-saturated (|logit|>15) voxels in a dense whole-volume
        pass (found necessary during hyperparameter derivation below --
        the pool's logits are heavily bimodal, most negatives sit deep
        in the saturated tail where float32 G/C round to exactly zero).
        Returns l0,G,C in float64."""
        z64 = z_flat.double()
        l0 = z64 @ w_p_d + b_p
        lp = ((z64 + delta * d_d) @ w_p_d) + b_p
        lm = ((z64 - delta * d_d) @ w_p_d) + b_p
        p0 = torch.sigmoid(l0); pp = torch.sigmoid(lp); pm = torch.sigmoid(lm)
        G = (pp - pm) / (2 * delta)
        C = (pp - 2 * p0 + pm) / (delta ** 2 + 1e-12)
        return l0, G, C

    # ---- percentile pass to set kappa, tau, mu ----
    # CRITICAL (caught in smoke test): pooling the WHOLE train population's
    # median G/C collapses to numerical zero because most negatives sit
    # deep in the sigmoid's saturated tail (|logit|>15), where G and C are
    # essentially zero by construction -- this made kappa blow up to ~1e7
    # and tau/mu collapse to 0, silently zeroing CRC's correction
    # everywhere as an artifact of this scale choice, not a real module
    # failure. CRC is meant to operate in the LOW-CONFIDENCE near-boundary
    # region (per the spec's own gating rationale in section 8), so
    # kappa/tau/mu are derived from the near-boundary subpopulation
    # (|logit_P|<10) where the sigmoid response is actually non-degenerate.
    probe_pool_np = np.concatenate([lesion_pool, neg_pool]).astype(np.float64)
    probe_logits = probe_pool_np @ w_et.astype(np.float64) + b_p
    near_mask = np.abs(probe_logits) < 10
    print(f'near-boundary probe subset: {near_mask.sum()}/{len(probe_logits)} '
         f'(|logit_P|<10)', flush=True)
    probe_near = torch.from_numpy(probe_pool_np[near_mask]).double().to(dev)

    with torch.no_grad():
        _, G_s, C_s = crc_components(probe_near)
    G_s = G_s.cpu().numpy(); C_s = C_s.cpu().numpy()
    kappa = 1.0 / (np.median(np.abs(C_s)) + 1e-8)
    tau = float(np.median(np.abs(G_s / (1 + kappa * np.abs(C_s)))) + 1e-8)
    mu = float(np.median(np.abs(G_s)) + 1e-8)
    print(f'kappa={kappa:.4f}  tau={tau:.6f}  mu={mu:.6f}  '
         f'(near-boundary median|G|={np.median(np.abs(G_s)):.6f} median|C|={np.median(np.abs(C_s)):.6f})', flush=True)

    def dense_eval(sid):
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]
        Cc, D, H, W = d1.shape
        flat = d1.reshape(Cc, -1).T
        with torch.no_grad():
            l_p, G, C = crc_components(flat)
            p_prod_flat = torch.sigmoid(l_p)
            p_r1b_flat = torch.sigmoid(flat @ w_r1b_t + b_r1b)

            E = G / (1 + kappa * torch.abs(C))
            a = torch.abs(G)
            g = a / (a + mu)

            l_no_curv = l_p + lam * g * torch.tanh(G / tau)
            l_no_gate = l_p + lam * torch.tanh(E / tau)
            l_full = l_p + lam * g * torch.tanh(E / tau)

            p_no_curv = torch.sigmoid(l_no_curv)
            p_no_gate = torch.sigmoid(l_no_gate)
            p_full = torch.sigmoid(l_full)

            maps = {}
            for name, t in [('prod', p_prod_flat), ('r1b', p_r1b_flat),
                            ('crc_no_curv', p_no_curv), ('crc_no_gate', p_no_gate),
                            ('crc_full', p_full)]:
                maps[name] = t.reshape(D, H, W).cpu().numpy()
        return maps, tgt_c, img_c

    test_subjects = sorted(set(r['subject_id'] for r in det_lesions_test) |
                          set(r['subject_id'] for r in g2a_lesions_test) |
                          set(r['subject_id'] for r in g2b_lesions_eval))

    print(f'\nDense whole-volume evaluation on {len(test_subjects)} held-out subjects...', flush=True)
    variants = ['prod', 'r1b', 'crc_no_curv', 'crc_no_gate', 'crc_full']

    out = HERE / ('E263_smoke.csv' if smoke else 'E263_crc.csv')
    fh = open(out, 'w', newline='')
    fields = ['subject_id']
    for v in variants:
        fields += [f'fp_total_{v}', f'fp_wt_not_tc_{v}', f'fp_outside_{v}']
    w_csv = csv.DictWriter(fh, fieldnames=fields)
    w_csv.writeheader()

    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        maps, tgt_c, img_c = dense_eval(sid)
        brain_mask = img_c[0] != 0
        true_et = tgt_c[ET] > 0.5
        true_tc = tgt_c[TC] > 0.5
        true_wt = tgt_c[WT] > 0.5

        def fp_breakdown(p_map):
            fp_mask = (p_map > TAU) & brain_mask & (~true_et)
            n_total = int(fp_mask.sum())
            n_wt_not_tc = int((fp_mask & true_wt & (~true_tc)).sum())
            n_outside = int((fp_mask & (~true_wt)).sum())
            return n_total, n_wt_not_tc, n_outside

        row = {'subject_id': sid}
        for v in variants:
            tot, wt, out_ = fp_breakdown(maps[v])
            row[f'fp_total_{v}'] = tot; row[f'fp_wt_not_tc_{v}'] = wt; row[f'fp_outside_{v}'] = out_
        w_csv.writerow(row)
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)
    fh.close()

    # ---- lesion-level recovery for all variants ----
    def lesion_recovery(lesion_list):
        recs = {v: [] for v in variants}
        for r in lesion_list:
            sid = r['subject_id']
            res = extract_lesion_shell(model, ds, sid_to_idx, sid, int(r['comp_id']), dev)
            if res is None:
                continue
            lf, sf, sz = res
            lf_t = torch.from_numpy(lf).float().to(dev)
            with torch.no_grad():
                l_p, G, C = crc_components(lf_t)
                p_prod_l = torch.sigmoid(l_p).cpu().numpy()
                p_r1b_l = torch.sigmoid(lf_t @ w_r1b_t + b_r1b).cpu().numpy()
                E = G / (1 + kappa * torch.abs(C))
                a = torch.abs(G)
                g = a / (a + mu)
                p_no_curv = torch.sigmoid(l_p + lam * g * torch.tanh(G / tau)).cpu().numpy()
                p_no_gate = torch.sigmoid(l_p + lam * torch.tanh(E / tau)).cpu().numpy()
                p_full = torch.sigmoid(l_p + lam * g * torch.tanh(E / tau)).cpu().numpy()
            for v, arr in [('prod', p_prod_l), ('r1b', p_r1b_l), ('crc_no_curv', p_no_curv),
                          ('crc_no_gate', p_no_gate), ('crc_full', p_full)]:
                recs[v].append(int(arr.max() > TAU))
        return {k: (np.mean(v) if v else float('nan')) for k, v in recs.items()}

    det_rec = lesion_recovery(det_lesions_test)
    g2a_rec = lesion_recovery(g2a_lesions_test)
    g2b_rec = lesion_recovery(g2b_lesions_eval)
    print(f'\ndetected_test: {det_rec}')
    print(f'G2A_test: {g2a_rec}')
    print(f'G2B: {g2b_rec}')

    print(f'\nE263 complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
