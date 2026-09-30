"""E245 -- internal decomposition of dec1, G2-A only, per the user's
exact spec. E244 localized the D2->D1 deficit to the cat1->dec1
transition specifically (d=-0.379, p=3.2e-7), ruling OUT attn_gate1
(d=-0.134, n.s.) as the primary target. dec1 itself is a BLOCK, not a
single operation -- this experiment decomposes it into its ACTUAL
internal operations (verified against the real implementation, not
assumed) to find WHICH internal transformation introduces the deficit.

VERIFIED ACTUAL STRUCTURE (neuroscan_3d_fixed.py lines 16-25, 72-75 --
UNet3D_v5 inherits this UNCHANGED through v2/v3, neither of which
touches self.dec1, checked directly before writing this):
  dec1 = nn.Sequential(Conv3DBlock(64, 32), Conv3DBlock(32, 32))
  Conv3DBlock(x) = ReLU(BatchNorm3d(Conv3d(x)))

So the full internal sequence from cat1 (64ch) to D1 (32ch) is SIX
operations across two blocks:
  cat1 -> Conv3d_1 -> BN_1 -> ReLU_1 -> [mid, 32ch] -> Conv3d_2 -> BN_2 -> ReLU_2 -> D1

STAGES MEASURED (separability, E234's frozen statistic, UNCHANGED):
  cat1        (= E244's S3, already measured, reused as the baseline)
  conv1_out   (after Conv3d_1, before BN_1)
  bn1_out     (after BN_1, before ReLU_1)
  relu1_out   (after ReLU_1 = mid, block 1's output)
  conv2_out   (after Conv3d_2, before BN_2)
  bn2_out     (after BN_2, before ReLU_2)
  relu2_out   (after ReLU_2 = D1, = E244's S4, already measured)

INTERPRETATION (per explicit user framework):
  deficit at conv1_out    -> feature mixing / channel projection
  deficit at bn1/bn2_out  -> feature-statistics transformation
  deficit at relu1/relu2  -> nonlinear feature suppression
  deficit spread evenly   -> progressive feature compression/transformation

CRITICAL: BatchNorm3d in eval() mode uses the frozen running_mean/
running_var (NOT batch statistics) -- this is automatically correct
since the model is already in .eval() mode throughout this entire
investigation, but flagged explicitly since BN's eval-vs-train behavior
difference is a classic source of silent bugs in exactly this kind of
manual layer-by-layer replay.
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

STAGE_ORDER = ['cat1', 'conv1_out', 'bn1_out', 'relu1_out', 'conv2_out', 'bn2_out', 'relu2_out']


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


def forward_to_dec1_internal(model, img_t):
    """Runs the model up through dec1, but replays dec1's OWN two
    Conv3DBlocks operation-by-operation (conv, bn, relu each retained
    separately) instead of calling model.dec1(cat1) as one black box.
    Uses the model's own layer objects (model.dec1[0].conv,
    model.dec1[0].bn, model.dec1[0].relu, model.dec1[1].*) directly --
    NOT reimplemented, so this is guaranteed bit-identical to the real
    dec1(cat1) call at the final relu2_out stage (verified below)."""
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
        dec2 = model.dec2(cat2)
        upconv1 = model.upconv1(dec2)
        enc1_gated, _ = model.attn_gate1(gate=bottleneck, skip=enc1)
        cat1 = torch.cat([upconv1, enc1_gated], dim=1)

        # ---- dec1 internals, operation-by-operation ----
        # CRITICAL: Conv3DBlock.relu is nn.ReLU(inplace=True) (verified in
        # neuroscan_3d_fixed.py line 22) -- calling module.relu(x) MUTATES
        # x's own storage in place, REGARDLESS of whether x was itself a
        # fresh .clone() -- the clone just delays which tensor gets
        # mutated, it doesn't protect the value stored in a dict pointing
        # at that same clone. FIRST FIX ATTEMPT (single .clone() before
        # the relu call) was insufficient -- verified via direct object-
        # identity check (bn1_out IS relu1_out afterward) even with that
        # clone in place, because the clone itself was what got passed
        # into (and mutated by) the in-place relu. REAL FIX: use
        # functional F.relu (NOT the module's own inplace=True instance)
        # for this decomposition's OWN measurement forward pass, which
        # returns a genuinely new tensor. This does not change the
        # computed VALUES at all (same weights, same nonlinearity) --
        # only avoids the in-place aliasing that was corrupting the
        # bn*_out measurement.
        block1 = model.dec1[0]  # Conv3DBlock(64,32)
        block2 = model.dec1[1]  # Conv3DBlock(32,32)

        conv1_out = block1.conv(cat1)
        bn1_out = block1.bn(conv1_out)
        relu1_out = F.relu(bn1_out)

        conv2_out = block2.conv(relu1_out)
        bn2_out = block2.bn(conv2_out)
        relu2_out = F.relu(bn2_out)

    return {'cat1': cat1, 'conv1_out': conv1_out, 'bn1_out': bn1_out,
           'relu1_out': relu1_out, 'conv2_out': conv2_out, 'bn2_out': bn2_out,
           'relu2_out': relu2_out}


def measure_lesion(model, ds, sid_to_idx, sid, comp_id, dev):
    if sid not in sid_to_idx:
        return None
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    brain_mask = img_c[0] != 0
    cm, shell = get_masks(tgt_c, comp_id, brain_mask)
    if cm is None:
        return None
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    stages = forward_to_dec1_internal(model, img_t)
    result = {}
    for name in STAGE_ORDER:
        tensor_np = stages[name][0].cpu().numpy()
        sep = separability(tensor_np, cm, shell)  # all stages native 128^3, no downsample
        result[name] = sep
    return result


def verify_bitexact(model, img_t):
    """Sanity check: relu2_out from the manual replay must exactly match
    model.dec1(cat1) computed the normal way, and the full model's own
    D1 output."""
    with torch.no_grad():
        full_out = model(img_t)
        real_d1 = full_out['dec1']
        stages = forward_to_dec1_internal(model, img_t)
        manual_d1 = stages['relu2_out']
        max_diff = (real_d1 - manual_d1).abs().max().item()
    return max_diff


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

    g2a_pairs = []
    for pid, d in by_pair.items():
        if 'missed' not in d or 'detected' not in d:
            continue
        key = (d['missed']['subject_id'], d['missed']['comp_id'])
        if phenotypes.get(key) == 'G2A_rejected':
            g2a_pairs.append((d['missed'], d['detected']))

    print(f'E245: {len(g2a_pairs)} G2-A matched pairs', flush=True)
    if smoke:
        g2a_pairs = g2a_pairs[:8]

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    # ---- verify bit-exact before trusting anything ----
    first_sid = g2a_pairs[0][0]['subject_id']
    if first_sid in sid_to_idx:
        img_c, _ = load_patch(ds, sid_to_idx, first_sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        max_diff = verify_bitexact(model, img_t)
        print(f'[verify] manual dec1 replay vs real model(): max diff = {max_diff:.2e}', flush=True)
        assert max_diff < 1e-4, 'dec1 internal replay does NOT match real forward pass -- DO NOT TRUST RESULTS'

    out = HERE / ('E245_smoke.csv' if smoke else 'E245_dec1_internal.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=['pair_idx', 'role', 'subject_id', 'comp_id'] + STAGE_ORDER)
    w.writeheader()

    t0 = time.time()
    n_done = 0
    for i, (m, d) in enumerate(g2a_pairs):
        res_m = measure_lesion(model, ds, sid_to_idx, m['subject_id'], int(m['comp_id']), dev)
        res_d = measure_lesion(model, ds, sid_to_idx, d['subject_id'], int(d['comp_id']), dev)
        for role, res, row_src in [('missed', res_m, m), ('detected', res_d, d)]:
            if res is None:
                continue
            row = {'pair_idx': i, 'role': role, 'subject_id': row_src['subject_id'], 'comp_id': row_src['comp_id']}
            row.update(res)
            w.writerow(row)
        n_done += 1
        if n_done % 10 == 0 or smoke:
            print(f'  {n_done}/{len(g2a_pairs)} ({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'\nE245 complete: wrote {out.name} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
