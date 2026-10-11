"""Demo asset extraction: runs the EXACT frozen E293 pipeline (B0
production baseline vs B3 complete integrated system) on a handful of
cherry-picked locked-test subjects, and saves per-subject PNG slice
comparisons (MRI, ground truth, production mask, E293 mask) plus a
JSON of per-subject metrics, for an interactive demo artifact.

Does NOT change any frozen weight, threshold, or spec -- reuses
e291's compute_subject_maps_with_tta, e293's exact B0/B3 construction
logic, and E290's cached MRD weights / acceptance model unchanged.
"""
import sys, os, json
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT_REPO / 'experiments' / 'exp_e12_eggo_m' / 'e131'))

from e257_common import load_model, load_populations, build_splits, ROOT, PATCH, MIN_VOX
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch
from e291_dual_zone_precision_integration import extract_27_features, FEATS_A2, compute_subject_maps_with_tta

ET, TC, WT = 0, 1, 2
TAU_LOW = 0.50
TAU_ET_LOCKED = 0.15
TAU_WT_LOCKED = 0.50
TAU_R_BASE = 0.95082877089122
TAU_ACCEPTANCE = 0.80
DIST_ZONE_BOUNDARY = 2.0
SEEDS = [999, 4242, 7, 123, 2024]
STRUCT = ndimage.generate_binary_structure(3, 1)

DEMO_SUBJECTS = ['BraTS-GLI-00404-000', 'BraTS-GLI-00299-000', 'BraTS-GLI-01482-000']

OUT_DIR = HERE / 'demo_assets'
OUT_DIR.mkdir(exist_ok=True)


def to_uint8(img_slice):
    lo, hi = np.percentile(img_slice, [1, 99])
    s = np.clip((img_slice - lo) / max(hi - lo, 1e-6), 0, 1)
    return (s * 255).astype(np.uint8)


def save_png(arr_uint8, path):
    from PIL import Image
    Image.fromarray(arr_uint8).save(path)


def main():
    dev = torch.device('cuda')
    model = load_model(dev)
    conv_w = model.seg_head[0].weight.detach().cpu().numpy()
    conv_b = model.seg_head[0].bias.detach().cpu().numpy()
    w_p, b_p = conv_w[ET].reshape(32).astype(np.float64), float(conv_b[ET])
    w_tc, b_tc = conv_w[TC].reshape(32).astype(np.float64), float(conv_b[TC])
    w_wt, b_wt = conv_w[WT].reshape(32).astype(np.float64), float(conv_b[WT])

    det_lesions, g2a_lesions, g2b_lesions = load_populations()

    mrd_cache_path = HERE / 'E290_mrd_cache.pt'
    cached_mrd = torch.load(mrd_cache_path, map_location='cpu', weights_only=False)
    fitted_mrd = {s: cached_mrd[s] for s in SEEDS}

    train_cache = HERE / 'E290_train_records.pt'
    train_records = torch.load(train_cache, map_location='cpu', weights_only=False)
    X_train = np.stack([r['feat'][FEATS_A2] for r in train_records])
    y_train = np.array([r['label'] for r in train_records], dtype=int)
    acceptance_model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', random_state=290))
    ])
    acceptance_model.fit(X_train, y_train)

    ds_full = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train', val_split=0.0, patch_size=PATCH)
    full_sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds_full.subject_dirs)}

    demo_data = []

    for sid in DEMO_SUBJECTS:
        if sid not in full_sid_to_idx:
            print(f'SKIP {sid}: not found'); continue
        print(f'Processing {sid}...', flush=True)

        raw_maps, calib_maps, tgt_c = compute_subject_maps_with_tta(
            model, ds_full, full_sid_to_idx, sid, w_p, b_p, w_tc, b_tc, w_wt, b_wt, fitted_mrd, dev)
        p_et_raw, p_tc_raw, p_wt_raw, r_maps_raw = raw_maps
        p_et_cal, p_tc_cal, p_wt_cal, r_maps_cal = calib_maps
        true_et = tgt_c[ET] > 0.5

        img_c, _ = load_patch(ds_full, full_sid_to_idx, sid)

        # G1 lesions (confidently missed by raw production)
        et_lbl_gt, n_et_gt = ndimage.label(true_et, structure=STRUCT)
        g1_masks = []
        for lid in range(1, n_et_gt + 1):
            lm = et_lbl_gt == lid
            if lm.sum() >= MIN_VOX and p_et_raw[lm].max() < TAU_LOW:
                g1_masks.append(lm)

        # B0: production baseline
        pred_b0 = p_et_raw >= 0.50

        # B2 core + B3 complete (reuses e293's exact construction)
        core_b2 = (p_et_cal >= TAU_ET_LOCKED) & (p_wt_cal >= TAU_WT_LOCKED)
        edt_et_b3 = ndimage.distance_transform_edt(~core_b2) if core_b2.any() else None
        edt_wt_b3 = ndimage.distance_transform_edt(~(p_wt_cal >= 0.50)) if (p_wt_cal >= 0.50).any() else None

        seed_b3_preds = []
        for seed in fitted_mrd.keys():
            r_seed = r_maps_cal[seed]
            mrd_s_pos = (r_seed > TAU_R_BASE) & (~core_b2)
            cc_lbl, cc_n = ndimage.label(mrd_s_pos, structure=STRUCT)
            admitted_s = np.zeros_like(core_b2)
            for cid in range(1, cc_n + 1):
                cm = cc_lbl == cid
                feat = extract_27_features(cm, p_et_cal, p_tc_cal, p_wt_cal, r_seed, edt_et_b3, edt_wt_b3, TAU_R_BASE)
                prob = acceptance_model.predict_proba(feat[FEATS_A2].reshape(1, -1))[0, 1]
                if prob >= TAU_ACCEPTANCE and feat[21] > DIST_ZONE_BOUNDARY:
                    admitted_s |= cm
            seed_b3_preds.append(core_b2 | admitted_s)
        pred_b3 = (np.stack(seed_b3_preds).sum(axis=0) >= (len(fitted_mrd) / 2.0))

        # metrics
        def dice(pred, true):
            tp = int((pred & true).sum()); fp = int((pred & ~true).sum()); fn = int((~pred & true).sum())
            return (2 * tp) / max(2 * tp + fp + fn, 1e-8)

        g1_rec_b0 = sum(1 for lm in g1_masks if (pred_b0 & lm).any())
        g1_rec_b3 = sum(1 for lm in g1_masks if (pred_b3 & lm).any())

        # pick the axial slice with the most ground-truth ET (or most B3-recovered G1 if available)
        recovered_masks = [lm for lm in g1_masks if (pred_b3 & lm).any() and not (pred_b0 & lm).any()]
        if recovered_masks:
            target_mask = recovered_masks[0]
            slice_idx = int(np.argmax(target_mask.sum(axis=(1, 2))))
        else:
            slice_idx = int(np.argmax(true_et.sum(axis=(1, 2))))

        t1c = img_c[0]  # MODALITIES = ("t1c", "t1n", "t2f", "t2w") -- index 0 is t1c (contrast-enhanced)
        mri_slice = to_uint8(t1c[slice_idx])
        gt_slice = (true_et[slice_idx] * 255).astype(np.uint8)
        b0_slice = (pred_b0[slice_idx] * 255).astype(np.uint8)
        b3_slice = (pred_b3[slice_idx] * 255).astype(np.uint8)

        save_png(mri_slice, OUT_DIR / f'{sid}_mri.png')
        save_png(gt_slice, OUT_DIR / f'{sid}_gt.png')
        save_png(b0_slice, OUT_DIR / f'{sid}_b0.png')
        save_png(b3_slice, OUT_DIR / f'{sid}_b3.png')

        demo_data.append({
            'subject_id': sid,
            'slice_idx': slice_idx,
            'dice_b0': dice(pred_b0, true_et),
            'dice_b3': dice(pred_b3, true_et),
            'g1_total': len(g1_masks),
            'g1_recovered_b0': g1_rec_b0,
            'g1_recovered_b3': g1_rec_b3,
        })
        print(f'  slice={slice_idx} Dice B0={demo_data[-1]["dice_b0"]:.4f} B3={demo_data[-1]["dice_b3"]:.4f} '
             f'G1 {g1_rec_b0}->{g1_rec_b3} of {len(g1_masks)}', flush=True)

        del raw_maps, calib_maps, tgt_c
        torch.cuda.empty_cache()

    with open(OUT_DIR / 'demo_data.json', 'w') as f:
        json.dump(demo_data, f, indent=2)
    print(f'\nWrote {OUT_DIR / "demo_data.json"}')


if __name__ == '__main__':
    main()
