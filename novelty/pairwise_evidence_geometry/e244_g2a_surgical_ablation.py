"""E244 -- surgical ablation of the D2->D1 transition, restricted to
G2-A ONLY, per the user's exact spec. Measures REPRESENTATION
SEPARABILITY at each intermediate stage (not Dice/recovery), per the
explicit E237-motivated discipline: an intervention can appear to
"recover" lesions by pushing logits across a threshold without doing
anything representationally meaningful -- E244 must never fall into that
trap.

STAGE SEQUENCE (exact D2->D1 transition, verified against
neuroscan_3d_v5.py lines 151-156):
  D2       = dec2 (baseline, already measured in E240/E243)
  S1       = upconv1(D2)                                    (32ch, 128^3)
  S2       = attn_gate1(gate=bottleneck, skip=enc1) -> enc1_gated (32ch, 128^3)
  S3 (cat1)= torch.cat([S1, S2], dim=1)                      (64ch, 128^3)
  S4 (dec1)= self.dec1(cat1)                                 (32ch, 128^3, = D1)

INCREMENTAL DIFFERENTIAL, per explicit user formula:
  Delta_S_i = S(stage_i) - S(stage_{i-1})   (frozen separability
    statistic, UNCHANGED from e234_layerwise_separability.py's own
    `separability()`, reused not reimplemented)
  compared as (Delta_S_i^miss - Delta_S_i^det) for G2-A missed lesions
  vs their E240-matched detected partner -- the operation with the
  LARGEST NEGATIVE differential is the primary target.

REPLACEMENT CONTROLS on attn_gate1 specifically (run ONLY if attn_gate1's
own stage shows the largest negative differential, since these are
expensive and only meaningful as a follow-up to that finding):
  normal    -- psi as actually computed (baseline S2)
  bypass    -- enc1_gated = enc1 directly (psi==1 everywhere, magnitude
               AND spatial pattern both removed)
  constant  -- enc1_gated = enc1 * mean(psi) (SAME average magnitude as
               the real gate, but spatially UNIFORM -- isolates whether
               the gate's SPATIAL PATTERN matters, or only its average
               suppressive strength)
  shuffled  -- enc1_gated = enc1 * spatially-permuted psi (same psi
               VALUES, scrambled arrangement -- isolates whether the
               gate's spatial LOCATION relative to the lesion matters,
               or just its value distribution -- mirrors E237's own
               shuffle-control logic exactly)
These 4 conditions test the "genuine spatial gating failure vs mere
magnitude change" distinction the user explicitly asked for.

POPULATION: G2-A ONLY (per explicit user instruction: "do not mix G2-B
into this experiment... it is empirically a different failure
phenotype"), i.e. E243's own G2A_rejected-labeled lesions, paired against
their E240 matched-detected partner (same pairing, same population,
reused unchanged).
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
from e234_layerwise_separability import separability, downsample_mask  # noqa: E402

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
ET = 0
PATCH = (128, 128, 128)
MIN_VOX = 5
SHELL_DILATION = 3
RNG = np.random.default_rng(2440)


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


def get_masks(tgt_c, comp_id, brain_mask):
    et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
    if comp_id < 1 or comp_id > et_n:
        return None, None
    cm = et_lbl == comp_id
    if cm.sum() < MIN_VOX:
        return None, None
    struct = ndimage.generate_binary_structure(3, 1)
    dilated = ndimage.binary_dilation(cm, structure=struct, iterations=SHELL_DILATION)
    shell = dilated & (~cm) & brain_mask & (~(tgt_c[ET] > 0.5))
    if shell.sum() < MIN_VOX:
        return None, None
    return cm, shell


def forward_to_stages(model, img_t, gate_mode='normal'):
    """Runs the model up through dec1, returning EVERY intermediate stage
    tensor of the D2->D1 transition, plus D2 itself for reference.
    gate_mode controls how attn_gate1's output is computed (for the
    replacement controls) -- 'normal' reproduces the real forward()
    EXACTLY (verified structurally against neuroscan_3d_v5.py)."""
    with torch.no_grad():
        enc1 = model.enc1(img_t); pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1); pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2); pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)
        upconv3 = model.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = model.dec3(cat3)
        upconv2 = model.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = model.dec2(cat2)  # D2

        upconv1 = model.upconv1(dec2)  # S1

        # ---- attn_gate1, with replaceable gate_mode ----
        g = model.attn_gate1.W_g(bottleneck)
        g_up = F.interpolate(g, size=enc1.shape[2:], mode="trilinear", align_corners=False)
        x = model.attn_gate1.W_x(enc1)
        psi_real = torch.sigmoid(model.attn_gate1.W_psi(F.relu(g_up + x)))

        if gate_mode == 'normal':
            psi = psi_real
        elif gate_mode == 'bypass':
            psi = torch.ones_like(psi_real)
        elif gate_mode == 'constant':
            psi = torch.full_like(psi_real, psi_real.mean().item())
        elif gate_mode == 'shuffled':
            flat = psi_real.reshape(-1).cpu().numpy()
            RNG.shuffle(flat)
            psi = torch.from_numpy(flat.reshape(psi_real.shape)).to(psi_real.device)
        else:
            raise ValueError(gate_mode)

        enc1_gated = enc1 * psi  # S2

        cat1 = torch.cat([upconv1, enc1_gated], dim=1)  # S3
        dec1 = model.dec1(cat1)  # S4 = D1

    return {'D2': dec2, 'S1_upconv1': upconv1, 'S2_gate': enc1_gated,
           'S3_cat1': cat1, 'S4_dec1': dec1}


STAGE_ORDER = ['D2', 'S1_upconv1', 'S2_gate', 'S3_cat1', 'S4_dec1']
STAGE_DOWNSAMPLE = {'D2': 2, 'S1_upconv1': 1, 'S2_gate': 1, 'S3_cat1': 1, 'S4_dec1': 1}


def measure_lesion(model, ds, sid_to_idx, sid, comp_id, dev, gate_mode='normal'):
    if sid not in sid_to_idx:
        return None
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    brain_mask = img_c[0] != 0
    cm, shell = get_masks(tgt_c, comp_id, brain_mask)
    if cm is None:
        return None
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    stages = forward_to_stages(model, img_t, gate_mode=gate_mode)
    result = {}
    for name in STAGE_ORDER:
        factor = STAGE_DOWNSAMPLE[name]
        lesion_s = downsample_mask(cm, factor)
        shell_s = downsample_mask(shell, factor)
        tensor_np = stages[name][0].cpu().numpy()
        sep = separability(tensor_np, lesion_s, shell_s)
        result[name] = sep
    return result


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- G2-A population: E243 phenotypes joined to E240 matched pairs ----
    phenotypes = {}
    for r in csv.DictReader(open(HERE / 'E243_phenotypes.csv')):
        phenotypes[(r['subject_id'], r['comp_id'])] = r['phenotype']

    e240 = list(csv.DictReader(open(HERE / 'E240_layerwise.csv')))
    by_pair = {}
    for r in e240:
        pid = int(r['pair_id'])
        by_pair.setdefault(pid, {})[r['role']] = r

    g2a_pairs = []
    for pid, d in by_pair.items():
        if 'missed' not in d or 'detected' not in d:
            continue
        key = (d['missed']['subject_id'], d['missed']['comp_id'])
        if phenotypes.get(key) == 'G2A_rejected':
            g2a_pairs.append((d['missed'], d['detected']))

    print(f'E244: {len(g2a_pairs)} G2-A matched pairs (missed vs detected)', flush=True)
    if smoke:
        g2a_pairs = g2a_pairs[:10]

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    # ---- PHASE 1: normal gate, full stage sequence, missed + detected ----
    out1 = HERE / ('E244_smoke_stages.csv' if smoke else 'E244_stages.csv')
    fh1 = open(out1, 'w', newline='')
    w1 = csv.DictWriter(fh1, fieldnames=['pair_idx', 'role', 'subject_id', 'comp_id'] + STAGE_ORDER)
    w1.writeheader()

    t0 = time.time()
    n_done = 0
    for i, (m, d) in enumerate(g2a_pairs):
        res_m = measure_lesion(model, ds, sid_to_idx, m['subject_id'], int(m['comp_id']), dev, gate_mode='normal')
        res_d = measure_lesion(model, ds, sid_to_idx, d['subject_id'], int(d['comp_id']), dev, gate_mode='normal')
        for role, res, row_src in [('missed', res_m, m), ('detected', res_d, d)]:
            if res is None:
                continue
            row = {'pair_idx': i, 'role': role, 'subject_id': row_src['subject_id'], 'comp_id': row_src['comp_id']}
            row.update(res)
            w1.writerow(row)
        n_done += 1
        if n_done % 10 == 0 or smoke:
            print(f'  phase1 {n_done}/{len(g2a_pairs)} ({time.time()-t0:.0f}s)', flush=True)
    fh1.close()
    print(f'Phase 1 (normal gate) complete: wrote {out1.name} ({time.time()-t0:.0f}s)', flush=True)

    print('\nPhase 2 (replacement controls) DEFERRED -- run e244_analyze.py first to', flush=True)
    print('confirm attn_gate1 is actually the largest-negative-differential stage', flush=True)
    print('before spending compute on the controls, per the script\'s own design note.', flush=True)


if __name__ == '__main__':
    main()
