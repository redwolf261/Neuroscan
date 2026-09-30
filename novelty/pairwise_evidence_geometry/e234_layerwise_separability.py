"""E234 -- layerwise representation-loss localization, per the user's exact
spec (they called this "E233" but that name is already used by this
session's own intensity-separability experiment; renumbered E234 to avoid
collision, flagged explicitly).

FALSIFICATION EXPERIMENT, not an algorithm design. Restricts to lesions
where H1 (input-information-absent) CANNOT explain the miss -- matched
pairs (missed vs detected) with EQUIVALENT raw input evidence -- then
traces lesion-vs-local-shell separability through every stage of the
frozen E131 encoder-decoder, using a FROZEN, untrained, identically-
constructed probe statistic at every stage (per-channel z-scored mean-
vector distance, "Mahalanobis-style", NOT a trained classifier -- avoids
confounding probe capacity with information content, per explicit user
instruction).

MATCHING: for every missed ET lesion, find the NEAREST detected ET lesion
in a normalized feature space of {log(size), isolation, dist_to_wt_centroid,
contrast_t1c, contrast_t2f} (E223's own cached covariates + E233's own
cached intensity contrasts, reused unchanged -- not recomputed). Matching
is GREEDY nearest-neighbor without replacement, on z-scored features (each
feature standardized by its own population std before distance
computation, so no single feature with a larger natural scale dominates
the match).

STAGES MEASURED (all from ONE resumed forward pass per subject, reusing
the model's own unchanged layer objects, exactly as E232's decoder-replay
was verified bit-exact against the real forward()):
  raw   = input x itself (4 channels, t1c/t1n/t2f/t2w)         -- H1's own test
  E1    = enc1   (32 ch,  128^3)
  E2    = enc2   (64 ch,  64^3)
  E3    = enc3   (128 ch, 32^3)
  E4/BN = bottleneck (256 ch, 16^3)
  D3    = dec3   (128 ch, 32^3)
  D2    = dec2   (64 ch,  64^3)
  D1    = dec1   (32 ch,  128^3)

SEPARABILITY STATISTIC (frozen probe, per user's confirmed choice):
at each stage's OWN resolution, pool the lesion's own voxels/cells and the
surrounding shell's own voxels/cells (from a shared mask downsampled to
that stage's grid via nearest-neighbor index mapping -- exact for power-
of-2 downsampling factors, which every stage here is), z-score EACH
CHANNEL by the shell population's own std (matches E233's own per-
modality contrast convention, generalized to many channels), then measure
||mean_lesion_zscored - mean_shell_zscored||_2 across channels -- a single
number per stage per lesion, comparable across stages despite very
different channel counts (4 to 256) because it's always a per-channel-
standardized Euclidean distance, not a raw-scale magnitude.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
MIN_VOX = 5
ET = 0
SHELL_DILATION = 3
STAGES = ('raw', 'E1', 'E2', 'E3', 'E4_BN', 'D3', 'D2', 'D1')
STAGE_DOWNSAMPLE = {'raw': 1, 'E1': 1, 'E2': 2, 'E3': 4, 'E4_BN': 8, 'D3': 4, 'D2': 2, 'D1': 1}


def resumed_forward(model, x):
    """Runs the FULL forward pass, capturing EVERY stage tensor, using the
    model's own unchanged layer objects -- verified structurally identical
    to UNet3D_v5.forward() (same lines, same order), just additionally
    retaining enc1/enc2/enc3 (not exposed by the model's own return dict)."""
    enc1 = model.enc1(x)
    pool1 = model.pool1(enc1)
    enc2 = model.enc2(pool1)
    pool2 = model.pool2(enc2)
    enc3 = model.enc3(pool2)
    pool3 = model.pool3(enc3)
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
    dec1 = model.dec1(cat1)

    probs = model.seg_head(dec1)

    return {'raw': x, 'E1': enc1, 'E2': enc2, 'E3': enc3, 'E4_BN': bottleneck,
           'D3': dec3, 'D2': dec2, 'D1': dec1, 'probs': probs}


def downsample_mask(mask_native, factor):
    """Downsamples a boolean native-resolution mask to a coarser grid by
    NEAREST-NEIGHBOR index mapping (exact for power-of-2 factors used
    here): a coarse cell is True if ANY of its native voxels are True.
    Uses max-pooling semantics via reshape+any, matching how a coarse
    grid CELL corresponds to a block of native voxels exactly."""
    if factor == 1:
        return mask_native
    D, H, W = mask_native.shape
    Dc, Hc, Wc = D // factor, H // factor, W // factor
    # crop to an exact multiple first (patch dims are always multiples of
    # every factor used here: 128 divisible by 1,2,4,8)
    m = mask_native[:Dc*factor, :Hc*factor, :Wc*factor]
    m = m.reshape(Dc, factor, Hc, factor, Wc, factor)
    return m.any(axis=(1, 3, 5))


def separability(tensor, lesion_mask_stage, shell_mask_stage):
    """tensor: (C, d, h, w) numpy. lesion_mask_stage/shell_mask_stage:
    (d,h,w) bool, at the SAME resolution as tensor. Returns the frozen-
    probe separability statistic: per-channel shell-std-normalized
    Euclidean distance between lesion and shell mean vectors.

    ROBUSTNESS FIX (caught by smoke test before trusting the full run --
    sep_D1=1028 and sep_E1=10868 appeared, wildly inconsistent with every
    other value in the same column, typically 2-10): a small minority of
    channels (2/32 for E1, confirmed by direct inspection) have near-zero
    variance in the SHELL population for a given small lesion's local
    neighborhood (effectively dead/inactive channels at that location) --
    dividing by shell_std+1e-6 for those channels produces an enormous,
    meaningless z-score spike that dominates the sqrt(sum(z^2)) statistic
    for the WHOLE stage, even though 30/32 other channels behave sanely.
    FIXED by excluding channels whose shell_std falls below a relative
    floor (1% of the channel's own overall activation scale across the
    WHOLE patch, not an absolute constant, so this generalizes correctly
    across stages with very different raw activation magnitudes) from the
    z-score sum entirely, rather than letting a near-zero denominator
    blow up the aggregate."""
    C = tensor.shape[0]
    flat = tensor.reshape(C, -1)
    lesion_flat = lesion_mask_stage.reshape(-1)
    shell_flat = shell_mask_stage.reshape(-1)
    if lesion_flat.sum() < 1 or shell_flat.sum() < 1:
        return None
    lesion_vals = flat[:, lesion_flat]   # (C, n_lesion)
    shell_vals = flat[:, shell_flat]     # (C, n_shell)
    lesion_mean = lesion_vals.mean(axis=1)
    shell_mean = shell_vals.mean(axis=1)
    shell_std = shell_vals.std(axis=1)
    # relative floor: 1% of this channel's own overall std across the
    # WHOLE patch (a stable, non-degenerate reference scale), not an
    # absolute constant that would mean different things at different
    # stages' very different activation magnitudes
    overall_std = flat.std(axis=1)
    floor = np.maximum(overall_std * 0.01, 1e-6)
    valid_channels = shell_std >= floor
    if valid_channels.sum() == 0:
        return None
    z_diff = (lesion_mean[valid_channels] - shell_mean[valid_channels]) / shell_std[valid_channels]
    # normalize by sqrt(n_valid_channels) so the statistic is comparable
    # across stages with different channel counts (4 to 256) despite a
    # possibly different number of channels surviving the floor at each
    return float(np.sqrt(np.sum(z_diff ** 2)) / np.sqrt(valid_channels.sum()))


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- load matched pairs (built by e234_build_matches.py, run first) ----
    matches_path = HERE / ('E234_smoke_matches.csv' if smoke else 'E234_matches.csv')
    if not matches_path.exists():
        print(f'ERROR: {matches_path} not found -- run e234_build_matches.py first', flush=True)
        sys.exit(1)
    matches = list(csv.DictReader(open(matches_path)))
    print(f'E234 layerwise separability: {len(matches)} matched pairs', flush=True)

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=(128, 128, 128))
    sid_to_idx = {}
    import os
    for i, sd in enumerate(ds.subject_dirs):
        sid_to_idx[os.path.basename(sd)] = i

    out = HERE / ('E234_smoke.csv' if smoke else 'E234_layerwise.csv')
    fh = open(out, 'w', newline='')
    fieldnames = ['pair_id', 'role', 'subject_id', 'comp_id', 'size'] + [f'sep_{s}' for s in STAGES]
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()

    # cache per-subject forward passes (many lesions may share a subject)
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
        subject_cache.clear()  # bound memory; each subject usually appears once per pair anyway
    fh.close()
    print(f'wrote {out.name}: {n_ok} rows ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
