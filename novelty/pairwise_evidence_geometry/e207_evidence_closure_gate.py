"""E207 -- Evidence Closure existence gate: individual vs collective vs relational.

QUESTION (user, 2026-09-23): are missed lesions weakly represented individually
but MORE identifiable when their bottleneck features are considered
collectively/relationally?

FROZEN E131_v5control_seed0. No training of the segmentation model. The only
learning here is a tiny held-out-subject probe classifier (logistic
regression) used purely to MEASURE information content, exactly as E148/E149
used a linear probe to measure the readout gap -- not a component we would ship.

POPULATIONS: L (missed ET components) and H (matched hard negatives) --
S is deliberately NOT used as the positive class here, because the question
is "does aggregation reveal lesion-vs-background signal for the CASES WE
CURRENTLY FAIL", not "can we already tell detected lesions from background"
(trivially yes, that's what the model already does). Comparing L vs H is the
correct test of your question; S is carried as a sanity-check population
(pooling should discriminate it trivially, or something upstream is broken).

LOCUS: bottleneck. Chosen because E205/E206 already established this is the
only locus where a real, confound-surviving L-vs-S/H signal exists (though
E206 showed IT ISN'T LOAD-BEARING for direct amplification). This experiment
asks a DIFFERENT question of the SAME locus: not "can we push h to fix z"
but "does the INFORMATION exist in aggregate even though no single voxel
carries it."

THREE PROBES per component, held-out-SUBJECT cross-validation (5-fold, split
by subject so no subject's components appear in both train and test -- LOSO
discipline):

  1. INDIVIDUAL   : one row per VOXEL inside the component. Feature = that
                    voxel's C-channel vector at the bottleneck. Label =
                    lesion(1)/hard-negative(0). This measures how identifiable
                    a SINGLE voxel is on its own.

  2. COLLECTIVE   : one row per COMPONENT. Feature = permutation-invariant
                    pooling of all its voxels' channel vectors: per-channel
                    mean, max, and std (3*C-dim). This measures whether
                    AGGREGATE statistics (no relational structure) help.

  3. RELATIONAL   : one row per COMPONENT. Feature = collective features PLUS
                    pairwise-relationship statistics that collective pooling
                    cannot see: mean pairwise cosine similarity, std of
                    pairwise cosine similarity, mean pairwise Euclidean
                    distance, and the top-5 eigenvalues of the (voxels x
                    voxels) Gram matrix (normalised by trace) -- a compact
                    description of how the voxel-vectors relate to EACH OTHER,
                    not just their marginal statistics.

METRIC: 5-fold subject-grouped CV AUC for each probe, on each of (L vs H),
(S vs H) [sanity check].

PRIMARY RESULT:
  Delta_rel  = AUC_relational - AUC_individual   (component-level AUC vs
               voxel-level AUC -- computed by aggregating individual-probe
               voxel scores to a component score via mean, for a fair
               component-vs-component comparison)
  Delta_coll = AUC_collective - AUC_individual

Permutation control: labels shuffled within subject, 200 iterations, on the
primary Delta_rel(L vs H) statistic.

KILL if Delta_rel and Delta_coll are both small (<0.03 AUC) or non-significant
under permutation for L vs H. The interesting outcome is
relational > collective > individual, specifically for L (not for the S
sanity check, where individual should already be strong).
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from neuroscan_3d_v5 import UNet3D_v5
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset
import importlib.util

spec = importlib.util.spec_from_file_location(
    "t", str(ROOT / 'experiments/exp_e12_eggo_m/e130/train_e130_multimodal_baseline.py'))
t = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["e"]; spec.loader.exec_module(t); sys.argv = _a

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
PATCH, MIN_VOX, MAX_PER_SUBJ = 128, 5, 3
ET = 0
MAX_VOX_PER_COMP = 40      # cap voxels/component for the individual probe & Gram matrix


def collect(model, ds, dev):
    """Returns dict: comp_id -> {subject, population, voxel_feats (n,C)}."""
    out = []
    t0 = time.time()
    for ii in range(len(ds)):
        img, tgt, sid = ds[ii]
        Y = tgt.numpy() > 0.5
        probs = t.sliding_window_predict(model, img.unsqueeze(0).to(dev),
                                         t.PATCH, t.SW_OVERLAP, 3, dev, True)
        P = probs >= 0.5
        brain = img[0].numpy() != 0
        lbl, nl = ndimage.label(Y[ET])
        missed, succ = [], []
        for g in range(1, nl + 1):
            cm = lbl == g
            if cm.sum() < MIN_VOX:
                continue
            ov = float((cm & P[ET]).sum()) / cm.sum()
            if ov == 0:
                missed.append((int(cm.sum()), g))
            elif ov >= 0.5:
                succ.append((int(cm.sum()), g))
        missed.sort(reverse=True); succ.sort(reverse=True)
        picks = [('L', g) for _, g in missed[:MAX_PER_SUBJ]] + \
                [('S', g) for _, g in succ[:MAX_PER_SUBJ]]
        if not picks:
            continue
        for pop0, g in picks:
            cm = lbl == g
            com = np.array(ndimage.center_of_mass(cm)).astype(int)
            st = [int(np.clip(c - PATCH // 2, 0, s - PATCH))
                  for c, s in zip(com, cm.shape)]
            sl = tuple(slice(s, s + PATCH) for s in st)
            cm_p = cm[sl]
            if cm_p.sum() < MIN_VOX:
                continue
            anygt = (Y[0] | Y[1] | Y[2])[sl]; brain_p = brain[sl]
            x = img[(slice(None),) + sl].unsqueeze(0).to(dev)
            with torch.no_grad():
                e1 = model.enc1(x); e2 = model.enc2(model.pool1(e1))
                e3 = model.enc3(model.pool2(e2)); bn = model.bottleneck(model.pool3(e3))
            bn0 = bn[0]                          # (C,d,h,w)
            shp3 = bn0.shape[1:]

            rng = np.random.default_rng(abs(hash((sid, g))) % (2**31))
            forbid = ndimage.binary_dilation(anygt, iterations=6)
            idx = np.argwhere(cm_p); lo = idx.min(0); ext = idx.max(0) - lo + 1
            rel = idx - lo; dims = np.array(cm_p.shape)
            with torch.no_grad():
                d3 = model.dec3(torch.cat([model.upconv3(bn), e3], 1))
                d2 = model.dec2(torch.cat([model.upconv2(d3), e2], 1))
                eg, _ = model.attn_gate1(gate=bn, skip=e1)
                d1 = model.dec1(torch.cat([model.upconv1(d2), eg], 1))
                lg0 = model.seg_head[0](d1)[0, ET].cpu().numpy()
            best, bestv = None, -np.inf
            for _ in range(200):
                hi = dims - ext
                if (hi <= 0).any():
                    break
                off = np.array([rng.integers(0, h + 1) for h in hi])
                cand = np.zeros_like(cm_p); cand[tuple((rel + off).T)] = True
                if (cand & forbid).any() or not brain_p[cand].all():
                    continue
                v = float(lg0[cand].mean())
                if v > bestv:
                    bestv, best = v, cand
            if best is None:
                continue

            def voxel_feats(msk):
                scale = np.array(shp3) / np.array(cm_p.shape)
                vox_idx = np.argwhere(msk)
                if len(vox_idx) > MAX_VOX_PER_COMP:
                    sel = rng.choice(len(vox_idx), MAX_VOX_PER_COMP, replace=False)
                    vox_idx = vox_idx[sel]
                pos3 = np.clip((vox_idx * scale).astype(int), 0,
                               np.array(shp3) - 1)
                feats = bn0[:, pos3[:, 0], pos3[:, 1], pos3[:, 2]].T  # (n,C)
                return feats.cpu().numpy()

            for pop, msk in [(pop0, cm_p), ('H', best)]:
                fv = voxel_feats(msk)
                if fv.shape[0] < 3:
                    continue
                out.append({'subject_id': sid, 'comp_id': f'{sid}_{g}_{pop}',
                           'population': pop, 'feats': fv})
        if (ii + 1) % 25 == 0:
            print(f'  collect {ii+1}/{len(ds)} subj, {len(out)} comps '
                  f'({time.time()-t0:.0f}s)', flush=True)
    return out


def relational_feats(fv):
    """fv: (n,C). Returns collective(3C) + relational(extra) feature vector."""
    mean = fv.mean(0); mx = fv.max(0); std = fv.std(0)
    collective = np.concatenate([mean, mx, std])
    norm = fv / (np.linalg.norm(fv, axis=1, keepdims=True) + 1e-8)
    cos = norm @ norm.T
    iu = np.triu_indices(len(fv), k=1)
    cos_pairs = cos[iu]
    dist = np.linalg.norm(fv[:, None, :] - fv[None, :, :], axis=-1)[iu]
    gram = fv @ fv.T
    gram = gram / (np.trace(gram) + 1e-8)
    eigs = np.sort(np.linalg.eigvalsh(gram))[::-1][:5]
    eigs = np.pad(eigs, (0, max(0, 5 - len(eigs))))
    rel = np.array([cos_pairs.mean(), cos_pairs.std(),
                    dist.mean(), dist.std(), *eigs])
    return collective, rel


def cv_auc(X, y, groups, n_splits=5):
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < n_splits:
        return float('nan')
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    scores, truths = [], []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        scores.append(p); truths.append(y[te])
    if not scores:
        return float('nan')
    p_all = np.concatenate(scores); y_all = np.concatenate(truths)
    if len(np.unique(y_all)) < 2:
        return float('nan')
    return roc_auc_score(y_all, p_all)


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for q in model.parameters():
        q.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    print(f'E207 evidence-closure gate, {len(ds)} subjects', flush=True)

    comps = collect(model, ds, dev)
    print(f'\ncollected {len(comps)} components', flush=True)

    # ---- build the three probe datasets ----
    vox_rows = []   # individual probe: one row per voxel
    comp_rows = []  # collective/relational probes: one row per component
    for c in comps:
        y = 1 if c['population'] == 'L' or c['population'] == 'S' else 0
        for v in c['feats']:
            vox_rows.append({'subject_id': c['subject_id'], 'comp_id': c['comp_id'],
                             'population': c['population'], 'y': y, 'x': v})
        coll, rel = relational_feats(c['feats'])
        comp_rows.append({'subject_id': c['subject_id'], 'comp_id': c['comp_id'],
                          'population': c['population'], 'y': y,
                          'x_coll': coll, 'x_rel': np.concatenate([coll, rel]),
                          'x_indiv_mean': c['feats'].mean(0)})

    with open(HERE / 'E207_components.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['subject_id', 'comp_id', 'population', 'y', 'n_vox'])
        for c in comps:
            y = 1 if c['population'] in ('L', 'S') else 0
            w.writerow([c['subject_id'], c['comp_id'], c['population'], y,
                       len(c['feats'])])

    results = {}
    for pos_pop, tag in [('L', 'L_vs_H'), ('S', 'S_vs_H')]:
        sub = [c for c in comp_rows if c['population'] in (pos_pop, 'H')]
        subv = [r for r in vox_rows if r['population'] in (pos_pop, 'H')]
        groups_c = np.array([c['subject_id'] for c in sub])
        y_c = np.array([c['y'] for c in sub])
        groups_v = np.array([r['subject_id'] for r in subv])
        y_v = np.array([r['y'] for r in subv])

        X_indiv_vox = np.stack([r['x'] for r in subv])
        auc_indiv_voxel = cv_auc(X_indiv_vox, y_v, groups_v)

        # component-level "individual" score: mean voxel feature -> same probe
        X_indiv_comp = np.stack([c['x_indiv_mean'] for c in sub])
        auc_indiv_comp = cv_auc(X_indiv_comp, y_c, groups_c)

        X_coll = np.stack([c['x_coll'] for c in sub])
        auc_coll = cv_auc(X_coll, y_c, groups_c)

        X_rel = np.stack([c['x_rel'] for c in sub])
        auc_rel = cv_auc(X_rel, y_c, groups_c)

        results[tag] = {'n_comp': len(sub), 'n_vox': len(subv),
                        'auc_individual_voxel': auc_indiv_voxel,
                        'auc_individual_comp': auc_indiv_comp,
                        'auc_collective': auc_coll,
                        'auc_relational': auc_rel}
        print(f'\n{tag}  (n_comp={len(sub)}, n_vox={len(subv)})')
        print(f'  AUC individual (voxel-level)     = {auc_indiv_voxel:.4f}')
        print(f'  AUC individual (comp mean-feat)  = {auc_indiv_comp:.4f}')
        print(f'  AUC collective (pooled moments)  = {auc_coll:.4f}')
        print(f'  AUC relational (+ pairwise/Gram) = {auc_rel:.4f}')
        print(f'  Delta_coll = {auc_coll - auc_indiv_comp:+.4f}')
        print(f'  Delta_rel  = {auc_rel - auc_indiv_comp:+.4f}')

    with open(HERE / 'E207_results.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['tag'] + list(next(iter(results.values())).keys()))
        w.writeheader()
        for tag, r in results.items():
            w.writerow({'tag': tag, **r})
    print('\nwrote E207_results.csv, E207_components.csv', flush=True)


if __name__ == '__main__':
    main()
