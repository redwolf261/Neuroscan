"""E236 -- D2 information recoverability test, per the user's exact spec.
NOT "can a probe distinguish detected from missed" (E235 already answered
that) -- "can the D2 representation still identify the LESION ITSELF for
high-contrast-missed lesions, even though the production segmentation
head fails to use it."

Reuses E235's exact feature-extraction machinery (extract_lesion_voxels,
itself built on E234's verified-bit-exact resumed_forward), restricted to
D2 ONLY (not all 8 stages -- cuts extraction cost ~8x vs E235's full run)
since this is a targeted follow-up, not a fresh layerwise sweep.

TWO EXPERIMENTS, per explicit user design:

EXPERIMENT A -- cross-group generalization. Train a D2 probe ONLY on
  DETECTED lesions vs their own background (never sees a missed lesion
  during training). Evaluate its held-out AUC on high_contrast_missed
  lesions vs THEIR OWN background.
    AUC_miss >> 0.5  -> the D2 representation of a missed lesion still
                        looks lesion-like to a probe that only learned
                        what DETECTED lesions look like -- information
                        EXISTS, segmentation pathway fails to use it.
    AUC_miss ~= 0.5  -> a detected-trained probe cannot recognize missed
                        lesions at D2 at all -- weaker evidence for
                        pure non-utilization (though see Experiment B).

EXPERIMENT B -- within-group recoverability (the stronger test). Train a
  probe using ONLY missed-lesion representations vs their own background
  (5-fold CV WITHIN the missed population, subject-level split, same
  leakage guard as E235), see if the missed lesions' OWN statistics allow
  a linear readout to separate them from background at all.
    Recovers (AUC >> 0.5)      -> information exists, segmentation
                                   pathway ignores/fails to route it --
                                   a DECISION/UTILIZATION problem.
    Does not recover (AUC~0.5) -> representation genuinely lacks usable
                                   lesion identity information at D2 for
                                   this population -- a REPRESENTATION
                                   TRANSFORMATION problem.

Per explicit user correction: this experiment does NOT use the word
"saturation" -- E235 showed DIVERGENCE (a measured AUC gap), not a
capacity ceiling, which was never actually measured. This experiment is
about RECOVERABILITY, a different and better-defined question.
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
from e234_layerwise_separability import resumed_forward, downsample_mask  # noqa: E402
from e235_linear_probe_layerwise import build_groups, extract_lesion_voxels  # noqa: E402

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
STAGE = 'D2'
N_FOLDS = 5


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    det, high_contrast_missed, matched_missed, t1c_median = build_groups()
    print(f'E236 D2 recoverability: detected={len(det)} '
          f'high_contrast_missed={len(high_contrast_missed)} '
          f'(t1c split at {t1c_median:.4f})', flush=True)

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=(128, 128, 128))
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    if smoke:
        det = det[:10]; high_contrast_missed = high_contrast_missed[:8]

    # ---- extraction: D2 ONLY (via extract_lesion_voxels, then discard
    # every stage except D2 to save memory -- re-running the SAME verified
    # extraction, not a new implementation) ----
    t0 = time.time()
    print('Extracting D2 features (only stage needed)...', flush=True)
    lesion_D2 = {}  # (sid, comp) -> (lesion_feats, shell_feats) or None
    all_needed = [(r, 'detected') for r in det] + [(r, 'high_contrast_missed') for r in high_contrast_missed]
    n_done = 0
    for r, group in all_needed:
        key = (r['subject_id'], r['comp_id'])
        if key in lesion_D2:
            continue
        if r['subject_id'] not in sid_to_idx:
            lesion_D2[key] = None
            continue
        full_res = extract_lesion_voxels(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
        lesion_D2[key] = full_res[STAGE] if full_res is not None else None
        n_done += 1
        if n_done % 100 == 0 or smoke:
            print(f'  extracted {n_done}/{len(all_needed)} ({time.time()-t0:.0f}s)', flush=True)
    print(f'Extraction complete: {n_done} lesions ({time.time()-t0:.0f}s)', flush=True)

    # subject-level folds (SAME convention as E235)
    unique_subjects = sorted(set(r['subject_id'] for r, _ in all_needed))
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(unique_subjects)
    fold_of = {sid: i % N_FOLDS for i, sid in enumerate(shuffled)}

    out = HERE / ('E236_smoke.csv' if smoke else 'E236_d2_recoverability.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=['experiment', 'fold', 'auc', 'n_pos', 'n_neg'])
    w.writeheader()

    # ---- EXPERIMENT A: train ONLY on detected, test ONLY on high_contrast_missed ----
    print('\nExperiment A: detected-trained probe -> high_contrast_missed', flush=True)
    for fold in range(N_FOLDS):
        train_X, train_y = [], []
        for r in det:
            sid = r['subject_id']
            if fold_of.get(sid) == fold:
                continue
            res = lesion_D2.get((sid, r['comp_id']))
            if res is None:
                continue
            lesion_feats, shell_feats = res
            train_X.append(lesion_feats); train_y.append(np.ones(len(lesion_feats)))
            train_X.append(shell_feats); train_y.append(np.zeros(len(shell_feats)))
        if not train_X:
            continue
        X_train = np.concatenate(train_X); y_train = np.concatenate(train_y)
        if len(np.unique(y_train)) < 2:
            continue
        probe = LogisticRegression(max_iter=500, C=1.0)
        probe.fit(X_train, y_train)

        test_X, test_y = [], []
        for r in high_contrast_missed:
            sid = r['subject_id']
            if fold_of.get(sid) != fold:
                continue
            res = lesion_D2.get((sid, r['comp_id']))
            if res is None:
                continue
            lesion_feats, shell_feats = res
            test_X.append(lesion_feats); test_y.append(np.ones(len(lesion_feats)))
            test_X.append(shell_feats); test_y.append(np.zeros(len(shell_feats)))
        if not test_X:
            continue
        X_test = np.concatenate(test_X); y_test = np.concatenate(test_y)
        if len(np.unique(y_test)) < 2:
            continue
        probs = probe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        w.writerow({'experiment': 'A_cross_group', 'fold': fold, 'auc': auc,
                   'n_pos': int(y_test.sum()), 'n_neg': int((1-y_test).sum())})
        print(f'  fold {fold}: AUC={auc:.4f} (n_pos={int(y_test.sum())}, n_neg={int((1-y_test).sum())})', flush=True)

    # ---- EXPERIMENT B: train and test ONLY on high_contrast_missed (5-fold CV within) ----
    print('\nExperiment B: within-group (missed-only) probe recoverability', flush=True)
    for fold in range(N_FOLDS):
        train_X, train_y = [], []
        for r in high_contrast_missed:
            sid = r['subject_id']
            if fold_of.get(sid) == fold:
                continue
            res = lesion_D2.get((sid, r['comp_id']))
            if res is None:
                continue
            lesion_feats, shell_feats = res
            train_X.append(lesion_feats); train_y.append(np.ones(len(lesion_feats)))
            train_X.append(shell_feats); train_y.append(np.zeros(len(shell_feats)))
        if not train_X:
            continue
        X_train = np.concatenate(train_X); y_train = np.concatenate(train_y)
        if len(np.unique(y_train)) < 2:
            continue
        probe = LogisticRegression(max_iter=500, C=1.0)
        probe.fit(X_train, y_train)

        test_X, test_y = [], []
        for r in high_contrast_missed:
            sid = r['subject_id']
            if fold_of.get(sid) != fold:
                continue
            res = lesion_D2.get((sid, r['comp_id']))
            if res is None:
                continue
            lesion_feats, shell_feats = res
            test_X.append(lesion_feats); test_y.append(np.ones(len(lesion_feats)))
            test_X.append(shell_feats); test_y.append(np.zeros(len(shell_feats)))
        if not test_X:
            continue
        X_test = np.concatenate(test_X); y_test = np.concatenate(test_y)
        if len(np.unique(y_test)) < 2:
            continue
        probs = probe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        w.writerow({'experiment': 'B_within_group', 'fold': fold, 'auc': auc,
                   'n_pos': int(y_test.sum()), 'n_neg': int((1-y_test).sum())})
        print(f'  fold {fold}: AUC={auc:.4f} (n_pos={int(y_test.sum())}, n_neg={int((1-y_test).sum())})', flush=True)

    fh.close()
    print(f'\nwrote {out.name} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
