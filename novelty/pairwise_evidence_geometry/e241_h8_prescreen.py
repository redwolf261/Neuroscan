"""E241 -- H8 pre-screen, per the user's exact sequencing instruction
("don't build H8 yet... first extract the actual segmentation-head score
distributions... if there is a clear systematic displacement, then H8
earns an experiment. If not, kill it immediately"). Runs IN PARALLEL with
E240 (H2 rerun), not blocking on it, per explicit instruction.

MEASUREMENT ONLY. Frozen E131 checkpoint, no training/gradient/injection
anywhere. Extracts the production seg_head's own output (probs[ET], the
model's REAL, unmodified prediction -- not a probe, not an injected
direction) for four populations:
  1. matched_detected     -- from E234's original matched pairs (kept
                              separate from E240's larger t1c/size-only
                              match, since "matched detected" here just
                              needs a representative detected population,
                              not the H2-specific strict pairing)
  2. high_contrast_missed  -- e235's build_groups() G2
  3. ordinary_missed       -- e235's build_groups() matched_missed (full)
  4. matched_background    -- the SAME shell voxels used throughout this
                              investigation (E233/E234/E238's 3-voxel
                              dilation ring around each lesion, excluding
                              other ET tissue) for a DETECTED lesion (so
                              "background" here means "the tissue right
                              around a real, correctly-segmented lesion",
                              the most relevant background population to
                              compare against, not generic random brain
                              tissue)

For each lesion/region, records: max_prob, mean_prob (over that
region's own voxels), using the model's real probs output at the
lesion's own native patch (same 128^3 centered-crop convention used
throughout this investigation).
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e235_linear_probe_layerwise import build_groups  # noqa: E402

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
ET = 0
PATCH = (128, 128, 128)
MIN_VOX = 5
SHELL_DILATION = 3


def load_patch(ds, sid_to_idx, sid):
    image, target, _ = ds._load_subject(ds.subject_dirs[sid_to_idx[sid]])
    D, H, W = image.shape[1:]
    pd_, ph_, pw_ = PATCH
    cd, ch, cw = D // 2, H // 2, W // 2
    sd_, ed_ = max(0, cd - pd_//2), min(D, cd + pd_//2)
    sh_, eh_ = max(0, ch - ph_//2), min(H, ch + ph_//2)
    sw_, ew_ = max(0, cw - pw_//2), min(W, cw + pw_//2)
    img_c = image[:, sd_:ed_, sh_:eh_, sw_:ew_]
    tgt_c = target[:, sd_:ed_, sh_:eh_, sw_:ew_]
    if img_c.shape[1:] != (pd_, ph_, pw_):
        ip = np.zeros((4, pd_, ph_, pw_), dtype=np.float32)
        tp_ = np.zeros((3, pd_, ph_, pw_), dtype=np.float32)
        ip[:, :img_c.shape[1], :img_c.shape[2], :img_c.shape[3]] = img_c
        tp_[:, :tgt_c.shape[1], :tgt_c.shape[2], :tgt_c.shape[3]] = tgt_c
        img_c, tgt_c = ip, tp_
    return img_c, tgt_c


def component_mask(tgt_c, comp_id):
    et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
    if comp_id < 1 or comp_id > et_n:
        return None
    return et_lbl == comp_id, et_lbl, et_n


def measure(model, ds, sid_to_idx, sid, comp_id, dev, want_background=False):
    if sid not in sid_to_idx:
        return None
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    res = component_mask(tgt_c, comp_id)
    if res is None:
        return None
    cm, et_lbl, et_n = res
    sz = int(cm.sum())
    if sz < MIN_VOX:
        return None

    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    with torch.no_grad():
        out = model(img_t)
    p = out['probs'][0, ET].cpu().numpy()

    lesion_max = float(p[cm].max())
    lesion_mean = float(p[cm].mean())

    bg_max, bg_mean = None, None
    if want_background:
        struct = ndimage.generate_binary_structure(3, 1)
        dilated = ndimage.binary_dilation(cm, structure=struct, iterations=SHELL_DILATION)
        brain_mask = img_c[0] != 0
        shell = dilated & (~cm) & brain_mask & (~(tgt_c[ET] > 0.5))
        if shell.sum() >= MIN_VOX:
            bg_max = float(p[shell].max())
            bg_mean = float(p[shell].mean())

    return {'sid': sid, 'comp_id': comp_id, 'size': sz,
           'lesion_max': lesion_max, 'lesion_mean': lesion_mean,
           'bg_max': bg_max, 'bg_mean': bg_mean}


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
        det = det[:15]; high_contrast_missed = high_contrast_missed[:15]
        matched_missed = matched_missed[:15]

    print(f'E241 H8 pre-screen: detected={len(det)} high_contrast_missed={len(high_contrast_missed)} '
          f'ordinary_missed={len(matched_missed)}', flush=True)

    out = HERE / ('E241_smoke.csv' if smoke else 'E241_h8_prescreen.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=['group', 'subject_id', 'comp_id', 'size',
                                       'lesion_max', 'lesion_mean', 'bg_max', 'bg_mean'])
    w.writeheader()

    t0 = time.time()
    n_done = 0
    groups = [('matched_detected', det, True), ('high_contrast_missed', high_contrast_missed, False),
             ('ordinary_missed', matched_missed, False)]
    for gname, rows, want_bg in groups:
        gn = 0
        for r in rows:
            res = measure(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev, want_background=want_bg)
            if res is None:
                continue
            w.writerow({'group': gname, 'subject_id': res['sid'], 'comp_id': res['comp_id'],
                       'size': res['size'], 'lesion_max': res['lesion_max'], 'lesion_mean': res['lesion_mean'],
                       'bg_max': res['bg_max'] if res['bg_max'] is not None else '',
                       'bg_mean': res['bg_mean'] if res['bg_mean'] is not None else ''})
            gn += 1; n_done += 1
            if n_done % 50 == 0 or smoke:
                print(f'  {gname}: {gn}/{len(rows)} ({time.time()-t0:.0f}s)', flush=True)
        fh.flush()
        print(f'  {gname} done: {gn} lesions', flush=True)
    fh.close()
    print(f'\nE241 complete: {n_done} rows ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
