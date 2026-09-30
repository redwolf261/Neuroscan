"""E249 -- channel rescue experiment, per the user's exact spec. First
genuinely CAUSAL test in the dec1-block1-channel thread (E244-E248 were
all correlational/associational, explicitly flagged as such throughout).

INTERVENTION (per explicit user decision on this turn's disambiguation
question -- the MINIMAL version, not a magnitude-injection): for the
target channel set, BYPASS ReLU1 entirely (identity: relu1_out[c] =
bn1_out[c], negatives pass through unclipped) instead of the normal
max(0, bn1_out[c]). No value is injected, no forced positivity -- this
purely RELAXES the clipping constraint for those channels, the most
defensible "minimal intervention" per the user's own framing and this
session's E237-informed discipline against generic magnitude effects.

TARGET CHANNELS (from E248's own cross-lesion-consistent measurement,
top-8 by mean p_pos_c across all 413 detected lesions):
  {30, 27, 20, 25, 15, 12, 4, 16}  (p_pos_c range 0.81-0.97)

FIVE CONDITIONS (per explicit user spec):
  1. baseline        -- unmodified model, normal ReLU everywhere
  2. target_rescue    -- ReLU bypassed at the 8 TARGET channels
  3. random_rescue     -- ReLU bypassed at 8 RANDOMLY chosen channels
                          (fixed draw per lesion via seeded RNG, but a
                          FRESH random draw -- not the same 8 across all
                          lesions -- to avoid one lucky/unlucky specific
                          random set dominating the result)
  4. dead_rescue       -- ReLU bypassed at the 8 MOST RELIABLY-DEAD
                          channels from E248 (bottom-8 by mean p_pos_c:
                          {26, 11, 5, 18, 14, 23, 8, 17}, p_pos_c range
                          0.00002-0.006) -- tests whether "any 8-channel
                          bypass" helps regardless of which channels, using
                          channels that carry essentially NO signal for
                          ANY lesion (a stronger null than random)
  5. shuffled_rescue   -- ReLU bypassed at the TARGET channels, but the
                          bn1_out values used for the bypass are
                          SPATIALLY PERMUTED within the lesion mask first
                          (same values, same channels, scrambled voxel-
                          to-voxel correspondence) -- preserves per-
                          channel activation STATISTICS while destroying
                          spatial correspondence to the lesion's own
                          geometry, per the user's explicit E237-informed
                          requirement to rule out a magnitude-only effect

POPULATION: G2-A (primary, where the deficit was found) AND G2-B
(secondary, per explicit user requirement to check "whether G2-B is
unaffected or behaves differently" -- G2-B's OWN D2->D1 effect was
already found to run in the OPPOSITE direction in E243, so this is a
genuine specificity check, not an afterthought).

MEASUREMENT (per explicit user instruction: NOT simply Dice/recovery --
must include voxel-level separability, matching this session's own
E237-motivated discipline against magnitude-only "recovery" that isn't
genuine representational change):
  - max_prob, mean_prob, recovered (tau=0.5) -- SAME convention as E237
  - separability at relu2_out (=D1, the final dec1 output) -- does the
    intervention change the REPRESENTATION, not just the final
    probability
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
from e234_layerwise_separability import separability  # noqa: E402
from e245_dec1_internal_decomposition import load_patch, get_masks  # noqa: E402

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
ET = 0
PATCH = (128, 128, 128)
MIN_VOX = 5
TAU = 0.5

TARGET_CHANNELS = [30, 27, 20, 25, 15, 12, 4, 16]
DEAD_CHANNELS = [26, 11, 5, 18, 14, 23, 8, 17]
N_CHANNELS_TOTAL = 32
RNG = np.random.default_rng(2490)


def forward_with_rescue(model, img_t, rescue_channels=None, shuffle_mask=None):
    """Full forward pass, with an OPTIONAL channel-selective ReLU1 bypass.
    rescue_channels: list of channel indices to bypass (identity instead
    of ReLU) at block1's ReLU1. shuffle_mask: if given (D,H,W) boolean
    lesion mask, the RESCUED channels' bn1_out values are spatially
    permuted WITHIN that mask before substitution (the shuffled-rescue
    condition). Returns dict with 'probs' and 'relu2_out' (=D1)."""
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

        block1 = model.dec1[0]
        block2 = model.dec1[1]

        conv1_out = block1.conv(cat1)
        bn1_out = block1.bn(conv1_out)

        if rescue_channels is None:
            relu1_out = F.relu(bn1_out)
        else:
            relu1_out = F.relu(bn1_out).clone()
            bn1_np = bn1_out[0].cpu().numpy()
            for c in rescue_channels:
                if shuffle_mask is not None:
                    vals = bn1_np[c][shuffle_mask].copy()
                    RNG.shuffle(vals)
                    rescued = bn1_np[c].copy()
                    rescued[shuffle_mask] = vals
                    relu1_out[0, c] = torch.from_numpy(rescued).to(relu1_out.device)
                else:
                    relu1_out[0, c] = bn1_out[0, c]  # identity: bypass ReLU for this channel

        conv2_out = block2.conv(relu1_out)
        bn2_out = block2.bn(conv2_out)
        relu2_out = F.relu(bn2_out)  # = D1

        probs = model.seg_head(relu2_out)

    return {'probs': probs, 'relu2_out': relu2_out}


def measure_condition(model, img_t, cm, shell, rescue_channels=None, shuffle_mask=None):
    out = forward_with_rescue(model, img_t, rescue_channels, shuffle_mask)
    p = out['probs'][0, ET].cpu().numpy()
    max_p = float(p[cm].max())
    mean_p = float(p[cm].mean())
    recovered = int(max_p > TAU)
    d1_np = out['relu2_out'][0].cpu().numpy()
    sep = separability(d1_np, cm, shell)
    return {'max_prob': max_p, 'mean_prob': mean_p, 'recovered': recovered,
           'sep_D1': sep if sep is not None else ''}


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
    seen = set()
    g2a_lesions, g2b_lesions = [], []
    for r in e240:
        if r['role'] != 'missed':
            continue
        key = (r['subject_id'], r['comp_id'])
        if key in seen:
            continue
        seen.add(key)
        phen = phenotypes.get(key)
        if phen == 'G2A_rejected':
            g2a_lesions.append(r)
        elif phen == 'G2B_partial':
            g2b_lesions.append(r)

    print(f'E249: G2-A={len(g2a_lesions)} G2-B={len(g2b_lesions)}', flush=True)
    if smoke:
        g2a_lesions = g2a_lesions[:8]
        g2b_lesions = g2b_lesions[:8]

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    out = HERE / ('E249_smoke.csv' if smoke else 'E249_rescue.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=['group', 'subject_id', 'comp_id', 'size', 'condition',
                                       'max_prob', 'mean_prob', 'recovered', 'sep_D1'])
    w.writeheader()

    t0 = time.time()
    n_done = 0
    verified_baseline = False
    for grp, lesions in [('G2A', g2a_lesions), ('G2B', g2b_lesions)]:
        for r in lesions:
            sid, cid = r['subject_id'], int(r['comp_id'])
            if sid not in sid_to_idx:
                continue
            img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
            brain_mask = img_c[0] != 0
            cm, shell = get_masks(tgt_c, cid, brain_mask)
            if cm is None:
                continue
            sz = int(cm.sum())
            img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)

            # ---- verify baseline matches the REAL unmodified model() call once ----
            if not verified_baseline:
                with torch.no_grad():
                    real_out = model(img_t)
                base_res = measure_condition(model, img_t, cm, shell, rescue_channels=None)
                real_max = float(real_out['probs'][0, ET][torch.from_numpy(cm).to(dev)].max().item())
                assert abs(real_max - base_res['max_prob']) < 1e-4, \
                    f'baseline mismatch: {real_max} vs {base_res["max_prob"]} -- DO NOT TRUST RESULTS'
                print(f'  [verify] baseline replay matches real model() (max_prob diff '
                      f'{abs(real_max - base_res["max_prob"]):.2e})', flush=True)
                verified_baseline = True

            remaining = [c for c in range(N_CHANNELS_TOTAL) if c not in TARGET_CHANNELS]
            random_channels = list(RNG.choice(remaining, size=8, replace=False))

            conditions = {
                'baseline': (None, None),
                'target_rescue': (TARGET_CHANNELS, None),
                'random_rescue': (random_channels, None),
                'dead_rescue': (DEAD_CHANNELS, None),
                'shuffled_rescue': (TARGET_CHANNELS, cm),
            }
            for cond_name, (rescue_ch, shuf_mask) in conditions.items():
                res = measure_condition(model, img_t, cm, shell, rescue_channels=rescue_ch, shuffle_mask=shuf_mask)
                w.writerow({'group': grp, 'subject_id': sid, 'comp_id': cid, 'size': sz,
                           'condition': cond_name, **res})
            n_done += 1
            if n_done % 10 == 0 or smoke:
                print(f'  {grp}: {n_done} lesions done ({time.time()-t0:.0f}s)', flush=True)
        fh.flush()
    fh.close()
    print(f'\nE249 complete: {n_done} lesions ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
