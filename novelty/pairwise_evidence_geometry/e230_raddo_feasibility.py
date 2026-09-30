"""E230 -- RADDO (Region-Adaptive Depth-Domain Disturbance Observer)
feasibility diagnostic. NO TRAINING. Pure forward-pass measurement on the
frozen E131 checkpoint, SAME 298-lesion stratified sample as E227/E229
(identical (subject_id,comp_id,patch) triples, reusing simulate_patch_draws
with the SAME rng_seed_base formula, so results are directly comparable).

CIRCULARITY GUARD (the exact bug caught and fixed in E229 -- see
[[e229_adrc_candidate_killed_circularity_caught]]): `detected`/`missed` is
defined from e_2 (the FINAL stage's error, dec1/probs vs GT) via the
project's own >=50%-overlap convention. THEREFORE e_2 MUST NEVER be an
input to D_i/P_i/C_i/A_i -- only e_0 (D4/aux_probs3) and e_1 (D2/
aux_probs2, UNSUPERVISED, flagged as exploratory per explicit user
decision) are used to construct predictors. e_2 is computed ONLY to
define the (frozen, pre-existing) outcome variable, exactly as E227/E229
already did.

PER-LESION QUANTITIES (region-level, i.e. pooled over the lesion's own
GT voxels at each stage's own native resolution -- not voxel-level):
  e_{i,0} = mean_{v in lesion, D4-resolution} |aux_probs3(v) - t_d4(v)|
  e_{i,1} = mean_{v in lesion, D2-resolution} |aux_probs2(v) - t_d2(v)|
  (e_{i,2} = mean_{v in lesion, full-res}      |probs(v) - t(v)|  -- NOT
   used as a predictor, computed only for the detected/missed label,
   which is ALREADY CACHED in E227_gradient.csv and reused unchanged)

DERIVED (predictors, k in {0,1} only):
  D_i = |e_{i,0}| + |e_{i,1}|                       -- persistent magnitude
  P_i = |e_{i,1} - e_{i,0}|                          -- "velocity" (2-point
                                                         first difference,
                                                         per this session's
                                                         explicit scoping:
                                                         2-3 points only,
                                                         acknowledged short)
  C_i = 1 - |e_{i,1} - e_{i,0}| / (e_{i,0}+e_{i,1}+eps)  -- consistency
                                                         (high when e_0~e_1,
                                                         low when they
                                                         diverge)
  A_i = D_i                                          -- the CONTROL signal
                                                         is deliberately
                                                         IDENTICAL to D_i's
                                                         formula (simple
                                                         sum) -- RADDO's
                                                         claimed advantage
                                                         must come from
                                                         P_i/C_i's
                                                         INCREMENTAL power
                                                         over D_i/A_i, not
                                                         from D_i itself
                                                         (which is already
                                                         the "simplest
                                                         alternative" the
                                                         user specified)
"""
import sys, csv, time, os
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
K4 = 4
K2 = 2
N_SIM_FOR_INCLUSION = 50  # SAME as E227/E229


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
    print(f'E230 RADDO feasibility diagnostic: {len(sample_keys)} lesions '
          f'(SAME stratified sample as E227/E229)', flush=True)

    n_total = 5 if smoke else len(ds)
    out = HERE / ('E230_smoke.csv' if smoke else 'E230_raddo.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=['subject_id', 'comp_id', 'size', 'e0_d4', 'e1_d2'])
    w.writeheader()

    t0 = time.time()
    pd, ph, pw = t.PATCH
    n_measured = 0
    for ii in range(n_total):
        sid_probe = os.path.basename(ds.subject_dirs[ii])
        if sid_probe not in sample_subjects:
            continue

        starts, centres, image, target, sid = simulate_patch_draws(
            ds, ii, t.PATCH, N_SIM_FOR_INCLUSION, rng_seed_base=30_000 + ii * 100_000)
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
            tgt_c = target[:, z0:z0+pd, y0:y0+ph, x0:x0+pw]
            if img_c.shape[1:] != (pd, ph, pw):
                img_p = np.zeros((img_c.shape[0], pd, ph, pw), dtype=np.float32)
                tgt_p = np.zeros((tgt_c.shape[0], pd, ph, pw), dtype=np.float32)
                img_p[:, :img_c.shape[1], :img_c.shape[2], :img_c.shape[3]] = img_c
                tgt_p[:, :tgt_c.shape[1], :tgt_c.shape[2], :tgt_c.shape[3]] = tgt_c
                img_c, tgt_c = img_p, tgt_p
            img_t = torch.from_numpy(img_c).unsqueeze(0).to(dev)
            tgt_t = torch.from_numpy(tgt_c).unsqueeze(0).to(dev)

            with torch.no_grad():
                out_ = model(img_t)
                aux_probs3 = out_['aux_probs3']  # (1,3,D/4,H/4,W/4)
                aux_probs2 = out_['aux_probs2']  # (1,3,D/2,H/2,W/2)

            t_d4 = F.avg_pool3d(tgt_t, kernel_size=K4, stride=K4)
            t_d2 = F.avg_pool3d(tgt_t, kernel_size=K2, stride=K2)

            # region-level e_0: lesion's own D4-cell footprint, error vs GT
            cz4 = local_voxels[:, 0] // K4; cy4 = local_voxels[:, 1] // K4; cx4 = local_voxels[:, 2] // K4
            cells4 = np.unique(np.stack([cz4, cy4, cx4], axis=1), axis=0)
            p4 = aux_probs3[0, ET, cells4[:, 0], cells4[:, 1], cells4[:, 2]].cpu().numpy()
            g4 = t_d4[0, ET, cells4[:, 0], cells4[:, 1], cells4[:, 2]].cpu().numpy()
            e0 = float(np.mean(np.abs(p4 - g4)))

            cz2 = local_voxels[:, 0] // K2; cy2 = local_voxels[:, 1] // K2; cx2 = local_voxels[:, 2] // K2
            cells2 = np.unique(np.stack([cz2, cy2, cx2], axis=1), axis=0)
            p2 = aux_probs2[0, ET, cells2[:, 0], cells2[:, 1], cells2[:, 2]].cpu().numpy()
            g2 = t_d2[0, ET, cells2[:, 0], cells2[:, 1], cells2[:, 2]].cpu().numpy()
            e1 = float(np.mean(np.abs(p2 - g2)))

            w.writerow({'subject_id': sid, 'comp_id': g, 'size': sz, 'e0_d4': e0, 'e1_d2': e1})
            n_measured += 1
        fh.flush()
        if (ii + 1) % 100 == 0 or smoke:
            print(f'  {ii+1}/{n_total} subj, {n_measured} lesions measured '
                  f'({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'wrote {out.name}: {n_measured} lesions ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
