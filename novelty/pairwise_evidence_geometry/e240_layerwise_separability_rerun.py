"""E240 -- H2 rerun at adequate power, per the user's exact sequencing
instruction ("finish H2 properly... repeat the matched representation
experiment at roughly the required sample size (~275 pairs), with the
same strict matching and held-out probe protocol... do not modify loss/
optimizer/architecture/decoder/threshold/sampling").

Per explicit user answer to this turn's disambiguation question: this
reruns E234's own frozen Mahalanobis-style layerwise separability
statistic UNCHANGED (that was the genuinely underpowered test -- n=78,
Cohen's d=0.17, needed ~275 pairs; E235's trained-probe design was
already at full power and is not being rerun here) -- just on a larger,
still-rigorously-matched population.

POPULATION CHANGE FROM E234: E234's original 5D match (log_size,
isolated, dist_to_wt_centroid, contrast_t1c, contrast_t2f) capped out at
n=117 for strict simultaneous equalization -- verified via
e234b_sweep_threshold.py's own sweep, no threshold reaches ~275 pairs
without t1c equalization breaking (p=0.03 at n=188, monotonically worse
beyond). Per explicit user choice, e240_build_matches_t1c_only.py
rebuilt the match using only the TWO features that actually matter most
(log_size, the DOMINANT raw confound with detection at r=0.63 -- checked
explicitly, not assumed, before finalizing this design -- and
contrast_t1c, E233's own strongest confound at r=-0.264), dropping the
three weaker/redundant features (isolated, dist, t2f) that were over-
constraining E234's original match for little benefit (per E234's own
match-quality report, these added little discriminating power). This
2D match reaches n=502 pairs with STRONGER equalization than E234's
original 117 (size p=0.55, t1c p=0.38 vs E234's p=0.40/0.40) -- both
better power AND at least as tight a match, not a trade-off.

This script does NOT reimplement E234's separability statistic,
resumed_forward, or downsample_mask -- it imports them UNCHANGED from
e234_layerwise_separability.py and simply points the matched-pairs input
at E240_matches.csv instead of E234_matches.csv, with its own output
filename (E240_layerwise.csv) so E234's original result is never
overwritten.
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
from e234_layerwise_separability import (  # noqa: E402
    resumed_forward, downsample_mask, separability,
    STAGES, STAGE_DOWNSAMPLE, MIN_VOX, ET, SHELL_DILATION, CKPT,
)


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    matches_path = HERE / 'E240_matches.csv'
    matches = list(csv.DictReader(open(matches_path)))
    if smoke:
        matches = matches[:15]
    print(f'E240 layerwise separability (H2 rerun, adequate power): {len(matches)} matched pairs', flush=True)

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=(128, 128, 128))
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    out = HERE / ('E240_smoke.csv' if smoke else 'E240_layerwise.csv')
    fh = open(out, 'w', newline='')
    fieldnames = ['pair_id', 'role', 'subject_id', 'comp_id', 'size'] + [f'sep_{s}' for s in STAGES]
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()

    subject_cache = {}
    struct = ndimage.generate_binary_structure(3, 1)

    def get_stage_tensors_and_masks(sid, comp_id):
        if sid not in subject_cache:
            image, target, _ = ds._load_subject(ds.subject_dirs[sid_to_idx[sid]])
            D, H, W = image.shape[1:]
            pd, ph, pw = 128, 128, 128
            cd, ch, cw = D // 2, H // 2, W // 2
            sd_, ed_ = max(0, cd - pd//2), min(D, cd + pd//2)
            sh_, eh_ = max(0, ch - ph//2), min(H, ch + ph//2)
            sw_, ew_ = max(0, cw - pw//2), min(W, cw + pw//2)
            img_c = image[:, sd_:ed_, sh_:eh_, sw_:ew_]
            tgt_c = target[:, sd_:ed_, sh_:eh_, sw_:ew_]
            if img_c.shape[1:] != (pd, ph, pw):
                ip = np.zeros((4, pd, ph, pw), dtype=np.float32)
                tp_ = np.zeros((3, pd, ph, pw), dtype=np.float32)
                ip[:, :img_c.shape[1], :img_c.shape[2], :img_c.shape[3]] = img_c
                tp_[:, :tgt_c.shape[1], :tgt_c.shape[2], :tgt_c.shape[3]] = tgt_c
                img_c, tgt_c = ip, tp_
            img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
            with torch.no_grad():
                stages = resumed_forward(model, img_t)
            stages_np = {k: v[0].cpu().numpy() for k, v in stages.items() if k != 'probs'}
            et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
            brain_mask = img_c[0] != 0
            subject_cache[sid] = (stages_np, et_lbl, et_n, brain_mask, tgt_c)
        return subject_cache[sid]

    t0 = time.time()
    n_ok = 0
    for pi, m in enumerate(matches):
        for role, sid, cid in [('missed', m['missed_subject'], m['missed_comp']),
                               ('detected', m['detected_subject'], m['detected_comp'])]:
            if sid not in sid_to_idx:
                continue
            stages_np, et_lbl, et_n, brain_mask, tgt_c = get_stage_tensors_and_masks(sid, int(cid))
            cid_i = int(cid)
            if cid_i < 1 or cid_i > et_n:
                continue
            cm = et_lbl == cid_i
            sz = int(cm.sum())
            if sz < MIN_VOX:
                continue
            dilated = ndimage.binary_dilation(cm, structure=struct, iterations=SHELL_DILATION)
            shell = dilated & (~cm) & brain_mask & (~(tgt_c[ET] > 0.5))
            if shell.sum() < MIN_VOX:
                continue

            row = {'pair_id': pi, 'role': role, 'subject_id': sid, 'comp_id': cid_i, 'size': sz}
            for s in STAGES:
                factor = STAGE_DOWNSAMPLE[s]
                lesion_s = downsample_mask(cm, factor)
                shell_s = downsample_mask(shell, factor)
                sep = separability(stages_np[s], lesion_s, shell_s)
                row[f'sep_{s}'] = sep if sep is not None else ''
            w.writerow(row)
            n_ok += 1
        if (pi + 1) % 20 == 0 or smoke:
            print(f'  {pi+1}/{len(matches)} pairs, {n_ok} rows ({time.time()-t0:.0f}s)', flush=True)
        subject_cache.clear()
    fh.close()
    print(f'wrote {out.name}: {n_ok} rows ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
