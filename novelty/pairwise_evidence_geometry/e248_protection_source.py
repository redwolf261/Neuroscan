"""E248 -- source of the protective clipping bias, per the user's exact
spec. Tests H8-A (fixed channel bias, encoded in conv1/BN1 parameters,
same channels protected across unrelated lesions) vs H8-B (lesion-
adaptive: detected lesions' own activation patterns happen to push
useful channels positive, not a fixed per-channel property).

Measurement only, no model modification.

PER LESION, PER CHANNEL (at bn1_out, before ReLU):
  mu_c   = mean(bn1_out[c] over lesion voxels)
  sigma_c = std(bn1_out[c] over lesion voxels)
  p_pos_c = P(bn1_out[c] > 0 | lesion voxels)  (fraction NOT clipped --
    the direct complement of E246/E247's own clip_c, kept as p_pos here
    to match the user's own P(BN1_c>0|lesion) notation)

POPULATION: ALL available detected lesions from E240 (n=413, NOT just
the 199 matched to G2-A -- maximizes power for the cross-lesion
consistency test, which needs many INDEPENDENT lesions, not a matched
subset), plus G2-A missed (n=199) and G2-B partial (n=112) for the
three-way comparison.

CROSS-LESION CONSISTENCY TEST (the decisive test, per the user's exact
design): split DETECTED lesions into two random halves (fixed seed).
For each channel c, compute its mean p_pos_c across each half separately
-> two vectors of 32 per-channel means. Correlate them (Pearson +
Spearman) across channels. HIGH correlation (same channels protected in
both halves) supports H8-A (fixed channel bias). LOW/near-zero
correlation supports H8-B (lesion-adaptive, no stable per-channel
identity). Repeated with multiple random splits (not just one) for a
robust estimate, not a single lucky/unlucky split.

BN PARAMETERS (explanatory only, per explicit user caution -- NOT
treated as causal): extracts block1.bn's own gamma (weight) and beta
(bias) per channel, reports correlation with the observed protection
score, but explicitly flagged as descriptive, not a causal claim.
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
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


def measure_lesion_protection(model, ds, sid_to_idx, sid, comp_id, dev):
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

    C = bn1.shape[0]
    mu_c = np.zeros(C); sigma_c = np.zeros(C); p_pos_c = np.zeros(C)
    for c in range(C):
        vals = bn1[c][cm]
        mu_c[c] = vals.mean()
        sigma_c[c] = vals.std()
        p_pos_c[c] = float((vals > 0).mean())

    return {'mu_c': mu_c, 'sigma_c': sigma_c, 'p_pos_c': p_pos_c}


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
    seen_detected = set()
    detected_lesions = []
    for r in e240:
        if r['role'] != 'detected':
            continue
        key = (r['subject_id'], r['comp_id'])
        if key in seen_detected:
            continue
        seen_detected.add(key)
        detected_lesions.append(r)

    by_pair = {}
    for r in e240:
        pid = int(r['pair_id'])
        by_pair.setdefault(pid, {})[r['role']] = r
    g2a_missed = []
    seen_g2a = set()
    for pid, d in by_pair.items():
        if 'missed' not in d:
            continue
        key = (d['missed']['subject_id'], d['missed']['comp_id'])
        if phenotypes.get(key) == 'G2A_rejected' and key not in seen_g2a:
            seen_g2a.add(key)
            g2a_missed.append(d['missed'])
    g2b_lesions = []
    seen_g2b = set()
    for pid, d in by_pair.items():
        if 'missed' not in d:
            continue
        key = (d['missed']['subject_id'], d['missed']['comp_id'])
        if phenotypes.get(key) == 'G2B_partial' and key not in seen_g2b:
            seen_g2b.add(key)
            g2b_lesions.append(d['missed'])

    print(f'E248: detected={len(detected_lesions)} G2A={len(g2a_missed)} G2B={len(g2b_lesions)}', flush=True)
    if smoke:
        detected_lesions = detected_lesions[:20]
        g2a_missed = g2a_missed[:8]
        g2b_lesions = g2b_lesions[:8]

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    out = HERE / ('E248_smoke.npz' if smoke else 'E248_protection.npz')
    all_results = {'detected': [], 'G2A': [], 'G2B': []}
    all_keys = {'detected': [], 'G2A': [], 'G2B': []}

    t0 = time.time()
    for grp, lesions in [('detected', detected_lesions), ('G2A', g2a_missed), ('G2B', g2b_lesions)]:
        n_done = 0
        for r in lesions:
            res = measure_lesion_protection(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
            if res is None:
                continue
            all_results[grp].append(res['p_pos_c'])
            all_keys[grp].append(f"{r['subject_id']}|{r['comp_id']}")
            n_done += 1
            if n_done % 20 == 0 or smoke:
                print(f'  {grp}: {n_done}/{len(lesions)} ({time.time()-t0:.0f}s)', flush=True)
        print(f'  {grp} done: {n_done} lesions', flush=True)

    # ---- BN1 parameters (explanatory only) ----
    block1 = model.dec1[0]
    gamma = block1.bn.weight.detach().cpu().numpy()
    beta = block1.bn.bias.detach().cpu().numpy()

    np.savez(out,
             detected=np.array(all_results['detected']), G2A=np.array(all_results['G2A']),
             G2B=np.array(all_results['G2B']), gamma=gamma, beta=beta,
             detected_keys=np.array(all_keys['detected']), G2A_keys=np.array(all_keys['G2A']),
             G2B_keys=np.array(all_keys['G2B']))
    print(f'\nE248 complete: wrote {out.name} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
