"""E241b -- H8 pre-screen, CORRECTED for a methodological confound found
while reviewing E241's own results: E241 (and every prior experiment in
the E234-E240 family) measures the model's output on a single fixed
128^3 center-crop, ONE forward pass. But the 'detected'/'missed' label
itself (E223_exposure.csv's own 'detected' column, which e235's
build_groups() and every downstream population definition are built on)
was computed using train_e130_multimodal_baseline.py's own
sliding_window_predict -- Gaussian-weighted, OVERLAPPING tile inference
across the WHOLE native volume (SW_OVERLAP=0.5), not a single crop.

DISCOVERED VIA: E241's raw output showed high_contrast_missed lesions
were strongly BIMODAL in max_prob (44% near 0, consistent with E232's
'confidently absent'; but 31% >0.9, which looks like a essentially a
correctly-segmented lesion by peak confidence). Checked whether this was
crop truncation (lesion partially outside the fixed 128^3 patch) --
verified NOT the case, 0/40 sampled lesions were truncated. The remaining
explanation is the inference-method mismatch: a lesion's single-crop
prediction (this investigation's convention throughout) can differ
substantially from its sliding-window multi-tile-averaged prediction (the
convention that actually DEFINES 'missed' in E223), simply because the
two methods see/weight the lesion's voxels differently.

THIS SCRIPT: reuses sliding_window_predict UNCHANGED (imported directly
from train_e130_multimodal_baseline.py, exactly as e223_exposure_audit.py
itself does) so H8's score distributions are computed with the SAME
inference method that defines the group labels being compared -- no
label/measurement mismatch. This is more expensive per subject (many
tiles vs one crop) but correctly apples-to-apples.
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
import importlib.util

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e235_linear_probe_layerwise import build_groups  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "t", str(ROOT / 'experiments/exp_e12_eggo_m/e130/train_e130_multimodal_baseline.py'))
t = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["e"]; spec.loader.exec_module(t); sys.argv = _a

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
ET = 0
PATCH = t.PATCH
SW_OVERLAP = t.SW_OVERLAP
MIN_VOX = 5
SHELL_DILATION = 3


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    det, high_contrast_missed, matched_missed, t1c_median = build_groups()
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    if smoke:
        det = det[:6]; high_contrast_missed = high_contrast_missed[:6]
        matched_missed = matched_missed[:6]

    print(f'E241b H8 pre-screen (SLIDING-WINDOW, matches detection-label convention): '
          f'detected={len(det)} high_contrast_missed={len(high_contrast_missed)} '
          f'ordinary_missed={len(matched_missed)}', flush=True)

    out = HERE / ('E241b_smoke.csv' if smoke else 'E241b_h8_prescreen_sw.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=['group', 'subject_id', 'comp_id', 'size',
                                       'lesion_max', 'lesion_mean', 'bg_max', 'bg_mean'])
    w.writeheader()

    t0 = time.time()
    n_done = 0
    subject_probs_cache = {}
    struct = ndimage.generate_binary_structure(3, 1)

    def get_sw_probs(sid):
        if sid not in subject_probs_cache:
            image, target, _ = ds._load_subject(ds.subject_dirs[sid_to_idx[sid]])
            brain = image[0] != 0
            x = torch.from_numpy(image).unsqueeze(0).to(dev)
            probs = t.sliding_window_predict(model, x, PATCH, SW_OVERLAP, 3, dev, True)
            subject_probs_cache[sid] = (probs, target, brain)
            if len(subject_probs_cache) > 3:  # bound memory: keep at most a few subjects cached
                subject_probs_cache.pop(next(iter(subject_probs_cache)))
        return subject_probs_cache[sid]

    groups = [('matched_detected', det, True), ('high_contrast_missed', high_contrast_missed, False),
             ('ordinary_missed', matched_missed, False)]
    for gname, rows, want_bg in groups:
        gn = 0
        for r in rows:
            sid = r['subject_id']
            if sid not in sid_to_idx:
                continue
            probs, target, brain = get_sw_probs(sid)
            p = probs[ET]
            et_lbl, et_n = ndimage.label(target[ET] > 0.5)
            cid = int(r['comp_id'])
            if cid < 1 or cid > et_n:
                continue
            cm = et_lbl == cid
            sz = int(cm.sum())
            if sz < MIN_VOX:
                continue
            lesion_max = float(p[cm].max())
            lesion_mean = float(p[cm].mean())
            bg_max, bg_mean = None, None
            if want_bg:
                dilated = ndimage.binary_dilation(cm, structure=struct, iterations=SHELL_DILATION)
                shell = dilated & (~cm) & brain & (~(target[ET] > 0.5))
                if shell.sum() >= MIN_VOX:
                    bg_max = float(p[shell].max())
                    bg_mean = float(p[shell].mean())
            w.writerow({'group': gname, 'subject_id': sid, 'comp_id': cid, 'size': sz,
                       'lesion_max': lesion_max, 'lesion_mean': lesion_mean,
                       'bg_max': bg_max if bg_max is not None else '',
                       'bg_mean': bg_mean if bg_mean is not None else ''})
            gn += 1; n_done += 1
            if n_done % 20 == 0 or smoke:
                print(f'  {gname}: {gn}/{len(rows)} ({time.time()-t0:.0f}s)', flush=True)
        fh.flush()
        print(f'  {gname} done: {gn} lesions', flush=True)
    fh.close()
    print(f'\nE241b complete: {n_done} rows ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
