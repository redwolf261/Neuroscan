"""E243 -- stratified D2->D1 replication, per the user's exact spec.

Splits E240's own matched-missed population (n=446 unique lesions, the
SAME population E242's transition analysis was computed on) into two
phenotypes, using E241b's own sliding-window max_prob/mean_prob
convention (reused unchanged, not reimplemented):

  G2-A (near-zero / rejected): D1 sliding-window max_prob < 0.01
  G2-B (partial-coverage): D1 sliding-window max_prob > 0.9 AND
                            mean_prob < 0.5 (per E241b's own finding that
                            the near-1.0-max subset has mean_prob 0.12-0.48
                            -- i.e. genuinely partial, not a full but
                            differently-thresholded detection)
  (lesions in neither bucket, e.g. mid-range max_prob, are EXCLUDED from
  this stratified test -- not a clean phenotype per the user's own two-
  group design, would dilute rather than clarify the comparison)

STAGE 1 (this script): phenotype every E240 missed lesion via a FRESH
sliding-window forward pass (reusing sliding_window_predict UNCHANGED
from train_e130_multimodal_baseline.py, exact E241b convention) --
E240's own separability CSV doesn't contain probs output, so this must
be computed fresh, but the separability values themselves (sep_D1,
sep_D2 etc.) are reused UNCHANGED from E240_layerwise.csv, not
recomputed.

STAGE 2 (e243_analyze.py): repeats E242's exact transition-decomposition
analysis (Delta_S_D2->D1 = S(D1)-S(D2), paired against MATCHED
DETECTED lesions from the SAME E240 pairs) separately within G2-A and
G2-B.

PRE-REGISTERED DECISION GATE (per explicit user instruction, stated
BEFORE running the analysis): E243 survives only if the D2->D1 effect
replicates (same direction, comparable or larger magnitude) within AT
LEAST ONE predefined subgroup and remains directionally coherent against
matched controls. If it disappears in both subgroups, KILL the D1-
transition hypothesis. If it survives strongly in ONE subgroup, that
subgroup's own D2->D1 transformation becomes the next target -- NOT an
architecture change yet.
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

spec = importlib.util.spec_from_file_location(
    "t", str(ROOT / 'experiments/exp_e12_eggo_m/e130/train_e130_multimodal_baseline.py'))
t = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["e"]; spec.loader.exec_module(t); sys.argv = _a

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
ET = 0
PATCH = t.PATCH
SW_OVERLAP = t.SW_OVERLAP
MIN_VOX = 5


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    e240_rows = list(csv.DictReader(open(HERE / 'E240_layerwise.csv')))
    missed_lesions = [(r['subject_id'], int(r['comp_id'])) for r in e240_rows if r['role'] == 'missed']
    missed_lesions = sorted(set(missed_lesions))
    if smoke:
        missed_lesions = missed_lesions[:12]
    print(f'E243 phenotyping: {len(missed_lesions)} unique missed lesions from E240', flush=True)

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    out = HERE / ('E243_smoke.csv' if smoke else 'E243_phenotypes.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=['subject_id', 'comp_id', 'size', 'max_prob', 'mean_prob', 'phenotype'])
    w.writeheader()

    t0 = time.time()
    n_done = 0
    subj_cache = {}
    for sid, cid in missed_lesions:
        if sid not in sid_to_idx:
            continue
        if sid not in subj_cache:
            image, target, _ = ds._load_subject(ds.subject_dirs[sid_to_idx[sid]])
            x = torch.from_numpy(image).unsqueeze(0).to(dev)
            probs = t.sliding_window_predict(model, x, PATCH, SW_OVERLAP, 3, dev, True)
            subj_cache[sid] = (probs, target)
            if len(subj_cache) > 3:
                subj_cache.pop(next(iter(subj_cache)))
        probs, target = subj_cache[sid]
        p = probs[ET]
        et_lbl, et_n = ndimage.label(target[ET] > 0.5)
        if cid < 1 or cid > et_n:
            continue
        cm = et_lbl == cid
        sz = int(cm.sum())
        if sz < MIN_VOX:
            continue
        max_p = float(p[cm].max())
        mean_p = float(p[cm].mean())
        if max_p < 0.01:
            phen = 'G2A_rejected'
        elif max_p > 0.9 and mean_p < 0.5:
            phen = 'G2B_partial'
        else:
            phen = 'mid_excluded'
        w.writerow({'subject_id': sid, 'comp_id': cid, 'size': sz,
                   'max_prob': max_p, 'mean_prob': mean_p, 'phenotype': phen})
        n_done += 1
        if n_done % 50 == 0 or smoke:
            print(f'  {n_done}/{len(missed_lesions)} ({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'\nE243 phenotyping complete: {n_done} lesions ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
