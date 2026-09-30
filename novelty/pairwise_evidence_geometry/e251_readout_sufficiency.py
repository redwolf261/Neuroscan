"""E251 -- held-out readout sufficiency test, per the user's exact spec.
STANDS ON ITS OWN -- per explicit user correction, does NOT inherit
credibility from the old E148/E149/E150 "frontier is at the readout"
finding; this must independently establish (or fail to establish) that
D1 contains information the production seg_head fails to convert into a
decision.

R0 = the ACTUAL trained seg_head (production readout, unchanged).
R1 = an INDEPENDENT linear readout trained FRESH on D1 features, per
     explicit user decision: DETECTED-LESION-ONLY training (never sees
     ANY missed lesion during training, matching E236's own Experiment A
     cross-group design exactly -- the strongest possible test, since if
     R1 recovers G2-A without ever having seen a missed lesion, that is
     unambiguous evidence the INFORMATION was already there and only the
     PRODUCTION readout fails to use it).

PRIMARY ENDPOINT (per explicit user instruction: "AUC alone isn't
enough... lesion-level recovery, because E249/E250 taught us that
representation metrics can move without crossing the segmentation
threshold"): lesion-level recovery under R1 (max R1-probability over the
lesion's own voxels > tau=0.5, SAME convention as E237/E249/E250) vs R0's
own actual recovery status for the SAME lesions. Voxel-level AUC also
reported as a secondary/diagnostic metric, matching E236's own
convention, but NOT treated as sufficient on its own.

POPULATIONS (per explicit user requirement, tested SEPARATELY): detected
(held-out fold, R1 trained on the OTHER folds' detected lesions only),
G2-A rejected, G2-B partial.

Reuses e245's verified forward_to_dec1_internal (bit-exact, D1 =
relu2_out) and get_masks/load_patch UNCHANGED. Reuses E243's own
phenotype labels UNCHANGED.
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch, get_masks  # noqa: E402

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
ET = 0
PATCH = (128, 128, 128)
MIN_VOX = 5
N_FOLDS = 5
TAU = 0.5
MAX_VOX_PER_LESION = 30  # same cap as E235/E236, avoid large lesions dominating training


def extract_lesion_and_shell_D1(model, ds, sid_to_idx, sid, comp_id, dev):
    """Returns (lesion_feats, shell_feats, r0_lesion_probs, size) or None.
    lesion_feats/shell_feats: (n_vox_capped, 32) D1 (=relu2_out) values.
    r0_lesion_probs: the ACTUAL seg_head output at the lesion's own
    voxels (R0, the production readout -- computed from the SAME
    forward pass, not re-run separately, guaranteeing consistency)."""
    if sid not in sid_to_idx:
        return None
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    brain_mask = img_c[0] != 0
    cm, shell = get_masks(tgt_c, comp_id, brain_mask)
    if cm is None:
        return None
    sz = int(cm.sum())
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    stages = forward_to_dec1_internal(model, img_t)
    d1 = stages['relu2_out']  # (1,32,D,H,W)
    with torch.no_grad():
        r0_probs = model.seg_head(d1)[0, ET].cpu().numpy()  # (D,H,W)
    d1_np = d1[0].cpu().numpy()  # (32,D,H,W)

    lesion_feats = d1_np[:, cm].T  # (n_lesion, 32)
    shell_feats = d1_np[:, shell].T  # (n_shell, 32)
    r0_lesion = r0_probs[cm]

    rng = np.random.default_rng(hash((sid, comp_id)) % (2**32))
    if lesion_feats.shape[0] > MAX_VOX_PER_LESION:
        idx = rng.choice(lesion_feats.shape[0], size=MAX_VOX_PER_LESION, replace=False)
        lesion_feats = lesion_feats[idx]
    if shell_feats.shape[0] > MAX_VOX_PER_LESION:
        idx = rng.choice(shell_feats.shape[0], size=MAX_VOX_PER_LESION, replace=False)
        shell_feats = shell_feats[idx]

    return lesion_feats, shell_feats, r0_lesion, sz


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    phenotypes = {}
    for r in csv.DictReader(open(HERE / 'E243_phenotypes.csv')):
        phenotypes[(r['subject_id'], r['comp_id'])] = r['phenotype']

    e240 = list(csv.DictReader(open(HERE / 'E240_layerwise.csv')))
    seen = set()
    det_lesions, g2a_lesions, g2b_lesions = [], [], []
    for r in e240:
        key = (r['subject_id'], r['comp_id'])
        if r['role'] == 'detected':
            if key in seen:
                continue
            seen.add(key)
            det_lesions.append(r)
    seen2 = set()
    for r in e240:
        if r['role'] != 'missed':
            continue
        key = (r['subject_id'], r['comp_id'])
        if key in seen2:
            continue
        seen2.add(key)
        phen = phenotypes.get(key)
        if phen == 'G2A_rejected':
            g2a_lesions.append(r)
        elif phen == 'G2B_partial':
            g2b_lesions.append(r)

    print(f'E251: detected={len(det_lesions)} G2A={len(g2a_lesions)} G2B={len(g2b_lesions)}', flush=True)
    if smoke:
        det_lesions = det_lesions[:40]
        g2a_lesions = g2a_lesions[:15]
        g2b_lesions = g2b_lesions[:15]

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    # ---- subject-level folds over DETECTED subjects only (R1 training population) ----
    det_subjects = sorted(set(r['subject_id'] for r in det_lesions))
    rng = np.random.default_rng(2510)
    shuffled = rng.permutation(det_subjects)
    fold_of = {sid: i % N_FOLDS for i, sid in enumerate(shuffled)}

    # ---- extraction: cache per-lesion results ----
    t0 = time.time()
    print('Extracting D1 features for detected lesions...', flush=True)
    det_cache = {}
    for i, r in enumerate(det_lesions):
        res = extract_lesion_and_shell_D1(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
        det_cache[(r['subject_id'], r['comp_id'])] = res
        if (i + 1) % 50 == 0 or smoke:
            print(f'  detected {i+1}/{len(det_lesions)} ({time.time()-t0:.0f}s)', flush=True)

    print('Extracting D1 features for G2-A...', flush=True)
    g2a_cache = {}
    for i, r in enumerate(g2a_lesions):
        res = extract_lesion_and_shell_D1(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
        g2a_cache[(r['subject_id'], r['comp_id'])] = res
        if (i + 1) % 50 == 0 or smoke:
            print(f'  G2A {i+1}/{len(g2a_lesions)} ({time.time()-t0:.0f}s)', flush=True)

    print('Extracting D1 features for G2-B...', flush=True)
    g2b_cache = {}
    for i, r in enumerate(g2b_lesions):
        res = extract_lesion_and_shell_D1(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
        g2b_cache[(r['subject_id'], r['comp_id'])] = res
        if (i + 1) % 50 == 0 or smoke:
            print(f'  G2B {i+1}/{len(g2b_lesions)} ({time.time()-t0:.0f}s)', flush=True)

    print(f'\nExtraction complete ({time.time()-t0:.0f}s)', flush=True)

    out = HERE / ('E251_smoke.csv' if smoke else 'E251_readout.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=['group', 'subject_id', 'comp_id', 'size', 'fold',
                                       'r0_max_prob', 'r0_recovered', 'r1_max_prob', 'r1_recovered'])
    w.writeheader()

    voxel_level_rows = []  # for AUC computation

    for fold in range(N_FOLDS):
        # ---- train R1 on detected lesions NOT in this fold ----
        train_X, train_y = [], []
        for r in det_lesions:
            sid = r['subject_id']
            if fold_of.get(sid) == fold:
                continue
            key = (sid, r['comp_id'])
            res = det_cache.get(key)
            if res is None:
                continue
            lesion_feats, shell_feats, r0_lesion, sz = res
            train_X.append(lesion_feats); train_y.append(np.ones(len(lesion_feats)))
            train_X.append(shell_feats); train_y.append(np.zeros(len(shell_feats)))
        if not train_X:
            continue
        X_train = np.concatenate(train_X); y_train = np.concatenate(train_y)
        r1 = LogisticRegression(max_iter=500, C=1.0)
        r1.fit(X_train, y_train)

        # ---- evaluate on held-out detected lesions from THIS fold ----
        for r in det_lesions:
            sid = r['subject_id']
            if fold_of.get(sid) != fold:
                continue
            key = (sid, r['comp_id'])
            res = det_cache.get(key)
            if res is None:
                continue
            lesion_feats, shell_feats, r0_lesion, sz = res
            r1_probs_lesion = r1.predict_proba(lesion_feats)[:, 1]
            r1_probs_shell = r1.predict_proba(shell_feats)[:, 1]
            w.writerow({'group': 'detected', 'subject_id': sid, 'comp_id': r['comp_id'], 'size': sz,
                       'fold': fold, 'r0_max_prob': float(r0_lesion.max()),
                       'r0_recovered': int(r0_lesion.max() > TAU),
                       'r1_max_prob': float(r1_probs_lesion.max()),
                       'r1_recovered': int(r1_probs_lesion.max() > TAU)})
            voxel_level_rows.append(('detected', np.concatenate([r1_probs_lesion, r1_probs_shell]),
                                     np.concatenate([np.ones(len(r1_probs_lesion)), np.zeros(len(r1_probs_shell))])))

        # ---- evaluate G2-A and G2-B lesions belonging to subjects in THIS fold ----
        # (G2-A/G2-B subjects are NOT detected subjects necessarily -- use the SAME
        # r1 fitted on this fold's training set for ALL G2-A/G2-B lesions whose
        # subject falls in this fold's mod-N_FOLDS assignment, extended to non-
        # detected subjects via the same hash so every lesion is evaluated by
        # EXACTLY one fold's r1, never by an r1 that could have seen it)
        for grp_name, cache, lesions in [('G2A', g2a_cache, g2a_lesions), ('G2B', g2b_cache, g2b_lesions)]:
            for r in lesions:
                sid = r['subject_id']
                # assign every G2A/G2B subject to a fold via the SAME hashing
                # convention as detected subjects would get if they'd been
                # included -- deterministic, subject-level, no leakage since
                # G2A/G2B subjects never appear in R1's OWN training data
                # regardless of fold (R1 only ever trains on detected lesions)
                sid_fold = fold_of.get(sid)
                if sid_fold is None:
                    sid_fold = abs(hash(sid)) % N_FOLDS
                if sid_fold != fold:
                    continue
                key = (sid, r['comp_id'])
                res = cache.get(key)
                if res is None:
                    continue
                lesion_feats, shell_feats, r0_lesion, sz = res
                r1_probs_lesion = r1.predict_proba(lesion_feats)[:, 1]
                r1_probs_shell = r1.predict_proba(shell_feats)[:, 1]
                w.writerow({'group': grp_name, 'subject_id': sid, 'comp_id': r['comp_id'], 'size': sz,
                           'fold': fold, 'r0_max_prob': float(r0_lesion.max()),
                           'r0_recovered': int(r0_lesion.max() > TAU),
                           'r1_max_prob': float(r1_probs_lesion.max()),
                           'r1_recovered': int(r1_probs_lesion.max() > TAU)})
                voxel_level_rows.append((grp_name, np.concatenate([r1_probs_lesion, r1_probs_shell]),
                                         np.concatenate([np.ones(len(r1_probs_lesion)), np.zeros(len(r1_probs_shell))])))
    fh.close()

    # ---- voxel-level AUC per group (secondary/diagnostic, per spec) ----
    print('\nVoxel-level R1 AUC by group (secondary metric):')
    from collections import defaultdict
    by_grp = defaultdict(lambda: ([], []))
    for grp, probs, labels in voxel_level_rows:
        by_grp[grp][0].append(probs); by_grp[grp][1].append(labels)
    for grp in by_grp:
        p = np.concatenate(by_grp[grp][0]); l = np.concatenate(by_grp[grp][1])
        if len(np.unique(l)) < 2:
            continue
        auc = roc_auc_score(l, p)
        print(f'  {grp}: AUC={auc:.4f}  n_voxels={len(l)}')

    print(f'\nE251 complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
