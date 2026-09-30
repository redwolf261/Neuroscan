"""E237 Stage 1 -- train the D2-resolution ET-evidence probe, per the
user's exact spec. Frozen E131 checkpoint, NO retraining of the UNet
itself anywhere in E237 (explicit user requirement, so we never confuse
"a new model can learn this" with "the existing model has it but doesn't
use it").

DIFFERENCE FROM E236: E236's probe was trained on lesion-vs-shell VOXEL
SAMPLES pooled across many subjects (a per-voxel classifier, but not
materialized as a full spatial evidence MAP). E237 needs w (R^64) applied
to EVERY D2 voxel across the WHOLE patch, producing a genuine spatial
evidence map S_D2(x) = w^T h_D2(x) + b, which is then upsampled and
injected into D1. Same underlying linear-probe idea as E236 (same
features, same voxel-level supervision), just packaged as a reusable
weight vector instead of a one-off sklearn fit whose weights were
discarded.

TRAINING POPULATION: G1 (detected) lesion voxels vs their OWN local shell
voxels, pooled across TRAINING-FOLD subjects only (5-fold subject-level
CV, same convention as E235/E236) -- the probe never sees G2 (high-
contrast missed) or G3 (ordinary missed) lesions during training, exactly
matching E236's own Experiment A design (which is what gave AUC=0.681 on
G2 -- Stage 1 reproduces THAT probe, not a new one).

Saves, per fold: the fitted (w, b) as a .npz file, for Stage 2 (injection)
to load and apply as a genuine forward-pass operation, not just an
offline AUC score.
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e234_layerwise_separability import resumed_forward, downsample_mask  # noqa: E402
from e235_linear_probe_layerwise import build_groups, extract_lesion_voxels  # noqa: E402

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
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
    print(f'E237 Stage 1: training D2 probe on G1 (detected, n={len(det)}) '
          f'only, {N_FOLDS}-fold subject-level CV', flush=True)

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=(128, 128, 128))
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    if smoke:
        det = det[:15]

    # ---- extraction: D2 only, G1 (detected) only ----
    t0 = time.time()
    lesion_D2 = {}
    n_done = 0
    for r in det:
        key = (r['subject_id'], r['comp_id'])
        if r['subject_id'] not in sid_to_idx:
            lesion_D2[key] = None
            continue
        full_res = extract_lesion_voxels(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
        lesion_D2[key] = full_res['D2'] if full_res is not None else None
        n_done += 1
        if n_done % 100 == 0 or smoke:
            print(f'  extracted {n_done}/{len(det)} ({time.time()-t0:.0f}s)', flush=True)
    print(f'Extraction complete: {n_done} G1 lesions ({time.time()-t0:.0f}s)', flush=True)

    # ---- ALL subjects (det + high_contrast_missed + matched_missed) get the
    # SAME fold assignment, so Stage 2 can reuse it consistently ----
    all_subjects = sorted(set(r['subject_id'] for r in det) |
                          set(r['subject_id'] for r in high_contrast_missed) |
                          set(r['subject_id'] for r in matched_missed))
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(all_subjects)
    fold_of = {sid: i % N_FOLDS for i, sid in enumerate(shuffled)}
    np.savez(HERE / ('E237_folds_smoke.npz' if smoke else 'E237_folds.npz'),
            subjects=np.array(list(fold_of.keys())), folds=np.array(list(fold_of.values())))

    out_dir = HERE / ('E237_probes_smoke' if smoke else 'E237_probes')
    out_dir.mkdir(exist_ok=True)

    for fold in range(N_FOLDS):
        train_X, train_y = [], []
        for r in det:
            sid = r['subject_id']
            if fold_of.get(sid) == fold:
                continue  # held out
            res = lesion_D2.get((sid, r['comp_id']))
            if res is None:
                continue
            lesion_feats, shell_feats = res
            train_X.append(lesion_feats); train_y.append(np.ones(len(lesion_feats)))
            train_X.append(shell_feats); train_y.append(np.zeros(len(shell_feats)))
        if not train_X:
            print(f'  fold {fold}: no training data, skipping')
            continue
        X_train = np.concatenate(train_X); y_train = np.concatenate(train_y)
        probe = LogisticRegression(max_iter=500, C=1.0)
        probe.fit(X_train, y_train)
        w = probe.coef_[0].astype(np.float32)  # (64,)
        b = float(probe.intercept_[0])
        np.savez(out_dir / f'fold{fold}.npz', w=w, b=b)
        print(f'  fold {fold}: probe fit, ||w||={np.linalg.norm(w):.4f}, b={b:.4f}, '
              f'n_train={len(y_train)}', flush=True)

    print(f'\nStage 1 complete ({time.time()-t0:.0f}s). Probes saved to {out_dir}/', flush=True)


if __name__ == '__main__':
    main()
