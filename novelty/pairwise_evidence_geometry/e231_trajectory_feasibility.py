"""E231 -- RADDO kill criterion 3: does the D4-stage disturbance predict
SUBSEQUENT recovery (D4 -> final), not just final status? Per explicit
user instruction, aux_probs2 (D2, unsupervised) is EXCLUDED entirely --
only the two genuinely-supervised stages are used: D4 (aux_probs3) and
final (probs). This collapses the original 3-point P_i/C_i design to a
single transition; per user's own follow-up decision, P_i/C_i are DROPPED
here and the test is e_0 (signed D4 error) -> Delta q, including the
explicit DIRECTION test (sign(e_0) -> sign(Delta q)) the user asked for.

CIRCULARITY GUARD: `detected` (used only for context/secondary reporting,
NEVER as a predictor here) is thresholded from q_1 (=P_l_fine, final-stage
confidence) alone. The OUTCOME here is Delta q = q_1 - q_0, a genuinely
DIFFERENT continuous quantity, not `detected` itself and not q_1 alone.
The PREDICTOR e_0 is computed entirely from stage k=0 (D4), never touching
q_1 or anything derived from it. This is a clean predictor/outcome
separation -- distinct from E229's bug (which mixed q_1 directly into
the predictor D_l=Q_l-P_l).

MEASURED (signed, unlike E230's |e_0| unsigned error):
  q_0 = mean, over lesion's own D4 cells, of aux_probs3 (D4-resolution
        confidence itself -- NOT compared to GT here, this is the
        network's own D4-stage BELIEF, used as one endpoint of Delta q)
  e_0 = q_0 - t_d4_mean  (signed disturbance: D4 confidence minus D4
        ground-truth occupancy, same cells as q_0 -- positive means the
        D4 head over-believes relative to what's actually there)
  q_1 = P_l_fine (final-stage confidence, REUSED from E229's own cached
        value where the (subject,comp) keys match -- not recomputed,
        avoiding redundant inference; recomputed fresh here otherwise for
        completeness/robustness against any E229 gaps)
  Delta q = q_1 - q_0
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
N_SIM_FOR_INCLUSION = 50  # SAME as E227/E229/E230


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
    print(f'E231 trajectory feasibility (kill criterion 3): {len(sample_keys)} lesions '
          f'(SAME stratified sample), aux_probs2 EXCLUDED', flush=True)

    n_total = 5 if smoke else len(ds)
    out = HERE / ('E231_smoke.csv' if smoke else 'E231_trajectory.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=['subject_id', 'comp_id', 'size', 'q0', 'e0_signed', 'q1'])
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
                aux_probs3 = out_['aux_probs3']  # (1,3,32,32,32)
                probs = out_['probs']            # (1,3,128,128,128)

            t_d4 = F.avg_pool3d(tgt_t, kernel_size=K4, stride=K4)

            cz4 = local_voxels[:, 0] // K4; cy4 = local_voxels[:, 1] // K4; cx4 = local_voxels[:, 2] // K4
            cells4 = np.unique(np.stack([cz4, cy4, cx4], axis=1), axis=0)
            q0_cells = aux_probs3[0, ET, cells4[:, 0], cells4[:, 1], cells4[:, 2]].cpu().numpy()
            g4_cells = t_d4[0, ET, cells4[:, 0], cells4[:, 1], cells4[:, 2]].cpu().numpy()
            q0 = float(np.mean(q0_cells))
            e0_signed = float(np.mean(q0_cells - g4_cells))  # SIGNED, unlike E230's |e0|

            mask = np.zeros((pd, ph, pw), dtype=bool)
            mask[local_voxels[:, 0], local_voxels[:, 1], local_voxels[:, 2]] = True
            probs_fine = probs[0, ET].cpu().numpy()
            q1 = float(probs_fine[mask].mean())

            w.writerow({'subject_id': sid, 'comp_id': g, 'size': sz,
                       'q0': q0, 'e0_signed': e0_signed, 'q1': q1})
            n_measured += 1
        fh.flush()
        if (ii + 1) % 100 == 0 or smoke:
            print(f'  {ii+1}/{n_total} subj, {n_measured} lesions measured '
                  f'({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'wrote {out.name}: {n_measured} lesions ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
