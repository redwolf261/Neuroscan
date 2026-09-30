"""E229 -- cheap, no-training feasibility diagnostic for the ADRC-inspired
"cross-scale disagreement as disturbance" candidate.

BEFORE building anything: does a coarse-vs-fine DISAGREEMENT signal predict
missedness independently of everything already tested (size, isolation,
distance, Q_l, cos_dec3_aux_vs_seg)? If not, kill this candidate before any
architecture work -- same discipline as E226/E227/E228-A's own feasibility
checks before committing to expensive builds.

DISAGREEMENT DEFINITION (the proposed "disturbance" signal): for each
lesion l,
    D_l = Q_l - P_l
where Q_l is the coarse D4 evidence (E227's OWN cached value, mean pooled
occupancy the lesion's own voxels experience at D4 resolution -- ALREADY
COMPUTED, reused unchanged) and P_l is the FINE-resolution decoder's own
mean predicted ET probability over the lesion's own voxels (measured HERE,
via a single forward pass on the SAME frozen E131 checkpoint, SAME
stratified 298-lesion sample E227 already selected, SAME lesion-finding/
patch-cropping logic reused from e227_d4_gradient_audit.py to guarantee
identical (subject_id, comp_id, patch) triples -- not a new sample, not a
new selection).

D_l > 0 means: the coarse D4 head has evidence for this lesion that the
fine decoder is NOT expressing -- exactly the "stalled actuator with an
observable disturbance" signature the ADRC transplant is built on. D_l ~ 0
means no exploitable disagreement exists (either both resolutions agree
the lesion is absent, or both agree it's present) -- in EITHER case there
is nothing for a disturbance-observer mechanism to correct.

NO GRADIENTS computed here (unlike E227) -- pure forward-pass inference,
much cheaper. NO TRAINING.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e223_exposure_audit import simulate_patch_draws, FG_BIAS, verify_fg_bias  # noqa: E402
import importlib.util

spec = importlib.util.spec_from_file_location(
    "t", str(ROOT / 'experiments/exp_e12_eggo_m/e130/train_e130_multimodal_baseline.py'))
t = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["e"]; spec.loader.exec_module(t); sys.argv = _a

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
MIN_VOX = 5
ET = 0
K = 4
N_SIM_FOR_INCLUSION = 50  # SAME as E227, so the SAME lesions resolve to a usable draw


def main():
    verify_fg_bias()
    dev = torch.device('cuda')
    smoke = '--smoke' in sys.argv

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=t.PATCH, fg_bias=FG_BIAS)

    sample_path = HERE / 'E227_sample_keys.csv'
    sample_keys = set()
    with open(sample_path) as f:
        for row in csv.DictReader(f):
            sample_keys.add((row['subject_id'], row['comp_id']))
    sample_subjects = set(sid for sid, _ in sample_keys)
    print(f'E229 disagreement feasibility diagnostic: {len(sample_keys)} lesions '
          f'(SAME stratified sample as E227)', flush=True)

    n_total = 5 if smoke else len(ds)

    out = HERE / ('E229_smoke.csv' if smoke else 'E229_disagreement.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=['subject_id', 'comp_id', 'size', 'P_l_fine'])
    w.writeheader()

    t0 = time.time()
    pd, ph, pw = t.PATCH
    n_measured = 0
    import os
    for ii in range(n_total):
        sid_probe = os.path.basename(ds.subject_dirs[ii])
        if sid_probe not in sample_subjects:
            continue

        starts, centres, image, target, sid = simulate_patch_draws(
            ds, ii, t.PATCH, N_SIM_FOR_INCLUSION, rng_seed_base=30_000 + ii * 100_000)
        # SAME rng_seed_base formula as E227, so the SAME candidate draws
        # are generated -- the "first usable draw" selection below will
        # find the IDENTICAL patch E227 used for gradient measurement.
        Y = target > 0.5
        et_lbl, et_n = ndimage.label(Y[ET])
        starts_arr = np.array(starts)

        for g in range(1, et_n + 1):
            cm = et_lbl == g
            sz = int(cm.sum())
            if sz < MIN_VOX:
                continue
            if (sid, str(g)) not in sample_keys:
                continue
            lz, ly, lx = np.where(cm)
            l_lo = np.array([lz.min(), ly.min(), lx.min()])
            l_hi = np.array([lz.max(), ly.max(), lx.max()])
            voxels_native = np.stack([lz, ly, lx], axis=1)

            p_lo = starts_arr; p_hi = starts_arr + np.array([pd, ph, pw]) - 1
            bbox_ok = np.where(np.all(p_lo <= l_hi, axis=1) & np.all(p_hi >= l_lo, axis=1))[0]
            chosen = None
            for k in bbox_ok:
                start = starts_arr[k]
                local = voxels_native - start[None, :]
                inside = np.all((local >= 0) & (local < np.array([pd, ph, pw])), axis=1)
                if inside.any():
                    chosen = (k, local[inside])
                    break
            if chosen is None:
                continue
            k, local_voxels = chosen
            start = starts_arr[k]

            z0, y0, x0 = start
            img_c = image[:, z0:z0+pd, y0:y0+ph, x0:x0+pw]
            if img_c.shape[1:] != (pd, ph, pw):
                img_p = np.zeros((img_c.shape[0], pd, ph, pw), dtype=np.float32)
                img_p[:, :img_c.shape[1], :img_c.shape[2], :img_c.shape[3]] = img_c
                img_c = img_p
            img_t = torch.from_numpy(img_c).unsqueeze(0).to(dev)

            with torch.no_grad():
                out_ = model(img_t)
                probs_fine = out_['probs'][0, ET].cpu().numpy()  # (pd,ph,pw)

            mask = np.zeros((pd, ph, pw), dtype=bool)
            mask[local_voxels[:, 0], local_voxels[:, 1], local_voxels[:, 2]] = True
            P_l = float(probs_fine[mask].mean())

            w.writerow({'subject_id': sid, 'comp_id': g, 'size': sz, 'P_l_fine': P_l})
            n_measured += 1
        fh.flush()
        if (ii + 1) % 100 == 0 or smoke:
            print(f'  {ii+1}/{n_total} subj, {n_measured} lesions measured '
                  f'({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'wrote {out.name}: {n_measured} lesions ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
