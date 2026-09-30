"""E246 -- ReLU1 clipping -> separability mediation test, per the user's
exact spec. G2-A only. Does not modify the model; measurement only.

E245 found the dominant deficit at conv1_out->relu1_out (BN1+ReLU1
combined, d=-0.235, p=3.3e-6), but per the user's own caution, the
BN-invariant separability statistic cannot isolate BN's contribution
from ReLU's. This experiment tests the specific mechanistic claim:
"negative pre-activation -> ReLU clipping -> loss of discriminative
information" DIRECTLY, rather than inferring it from the BN-invariance
argument alone.

PART 1 -- clipping fraction r_0, per lesion, at bn1_out (the tensor
ReLU1 will act on):
  r_0 = #(bn1_out < 0) / (#lesion_voxels * C)
  measured SEPARATELY for lesion voxels and shell voxels (the shell
  control matters per the user's explicit point -- otherwise a finding
  of "G2-A has higher r_0" could just mean G2-A sits in a region where
  the WHOLE local neighborhood's representation is more negative, not
  something specific to the lesion itself). Compared across G2-A missed,
  matched detected, and G2-B partial (three-way comparison, per spec).

PART 2 -- mediation test, per lesion:
  Delta_S_ReLU = S(relu1_out) - S(bn1_out)   [already computable from
    E245's own per-lesion CSV -- reused, not recomputed]
  Delta_S_full = S(dec1) - S(cat1)            [E244's own per-lesion
    measurement -- reused, not recomputed]
  Test: does Delta_S_ReLU correlate with Delta_S_full across G2-A
  lesions? (Pearson + Spearman, since the relationship may be
  monotonic-but-nonlinear). A high correlation would support "ReLU
  clipping specifically drives the overall cat1->D1 separability
  collapse" as a mediating mechanism, not just a co-occurring stage.
  Also tests whether r_0 (clipping fraction) ITSELF correlates with
  Delta_S_full -- the more direct "clipping amount predicts collapse"
  version of the same claim.
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage, stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch, get_masks  # noqa: E402

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
ET = 0
PATCH = (128, 128, 128)
MIN_VOX = 5


def measure_clipping(model, ds, sid_to_idx, sid, comp_id, dev):
    if sid not in sid_to_idx:
        return None
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    brain_mask = img_c[0] != 0
    cm, shell = get_masks(tgt_c, comp_id, brain_mask)
    if cm is None:
        return None
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    stages = forward_to_dec1_internal(model, img_t)
    bn1 = stages['bn1_out'][0].cpu().numpy()  # (32, D, H, W)

    lesion_vals = bn1[:, cm]   # (32, n_lesion_vox)
    shell_vals = bn1[:, shell]  # (32, n_shell_vox)

    r0_lesion = float((lesion_vals < 0).mean())
    r0_shell = float((shell_vals < 0).mean())

    return {'r0_lesion': r0_lesion, 'r0_shell': r0_shell, 'r0_diff': r0_lesion - r0_shell}


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    phenotypes = {}
    for r in csv.DictReader(open(HERE / 'E243_phenotypes.csv')):
        phenotypes[(r['subject_id'], r['comp_id'])] = r['phenotype']

    e240 = list(csv.DictReader(open(HERE / 'E240_layerwise.csv')))
    by_pair = {}
    for r in e240:
        pid = int(r['pair_id'])
        by_pair.setdefault(pid, {})[r['role']] = r

    # G2-A pairs (missed=G2A_rejected) and their matched detected
    g2a_pairs = []
    for pid, d in by_pair.items():
        if 'missed' not in d or 'detected' not in d:
            continue
        key = (d['missed']['subject_id'], d['missed']['comp_id'])
        if phenotypes.get(key) == 'G2A_rejected':
            g2a_pairs.append((d['missed'], d['detected']))

    # G2-B lesions (for the 3-way r0 comparison, per spec) -- no detected
    # partner needed here, just the missed lesion's own r0
    g2b_lesions = []
    for pid, d in by_pair.items():
        if 'missed' not in d:
            continue
        key = (d['missed']['subject_id'], d['missed']['comp_id'])
        if phenotypes.get(key) == 'G2B_partial':
            g2b_lesions.append(d['missed'])

    print(f'E246: {len(g2a_pairs)} G2-A pairs, {len(g2b_lesions)} G2-B lesions', flush=True)
    if smoke:
        g2a_pairs = g2a_pairs[:8]
        g2b_lesions = g2b_lesions[:8]

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    out = HERE / ('E246_smoke.csv' if smoke else 'E246_clipping.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=['pair_idx', 'group', 'subject_id', 'comp_id',
                                       'r0_lesion', 'r0_shell', 'r0_diff'])
    w.writeheader()

    t0 = time.time()
    n_done = 0
    for i, (m, d) in enumerate(g2a_pairs):
        res_m = measure_clipping(model, ds, sid_to_idx, m['subject_id'], int(m['comp_id']), dev)
        res_d = measure_clipping(model, ds, sid_to_idx, d['subject_id'], int(d['comp_id']), dev)
        for grp, res, row_src in [('G2A_missed', res_m, m), ('matched_detected', res_d, d)]:
            if res is None:
                continue
            row = {'pair_idx': i, 'group': grp, 'subject_id': row_src['subject_id'], 'comp_id': row_src['comp_id']}
            row.update(res)
            w.writerow(row)
        n_done += 1
        if n_done % 10 == 0 or smoke:
            print(f'  G2-A pairs: {n_done}/{len(g2a_pairs)} ({time.time()-t0:.0f}s)', flush=True)
    fh.flush()

    for i, lesion in enumerate(g2b_lesions):
        res = measure_clipping(model, ds, sid_to_idx, lesion['subject_id'], int(lesion['comp_id']), dev)
        if res is None:
            continue
        row = {'pair_idx': -1, 'group': 'G2B_partial', 'subject_id': lesion['subject_id'], 'comp_id': lesion['comp_id']}
        row.update(res)
        w.writerow(row)
        if (i + 1) % 10 == 0 or smoke:
            print(f'  G2-B lesions: {i+1}/{len(g2b_lesions)} ({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'\nE246 complete: wrote {out.name} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
