"""E232 -- modality-completion intersection test, per the user's exact
spec. NO NEW TRAINING. Pure inference on the frozen E131 checkpoint,
testing whether uncertainty over PLAUSIBLE LATENT PERTURBATIONS (NOT
actual missing-modality completion, per the user's own explicit
"important correction" against leaking the true missing modality)
predicts which lesions the model struggles with.

WHY BOTTLENECK PERTURBATION, NOT MODALITY MASKING: this project's
checkpoints (E131 and everything built on it this session) take all 4
MRI modalities simultaneously and were NEVER trained with modality
dropout -- zeroing an input channel would push the forward pass
out-of-distribution, confounding "uncertainty" with a genuine OOD
artifact rather than epistemic ambiguity. Per explicit user confirmation,
z_k perturbs the BOTTLENECK activation (B,256,D/8,H/8,W/8) -- the
network's actual information bottleneck, upstream of EVERYTHING
downstream including the attention gate -- with K stochastic MC-dropout-
style perturbations (channel dropout + small Gaussian noise), keeping the
REAL 4-modality input x_O unperturbed and in-distribution throughout.

MECHANICS: the encoder (enc1..bottleneck) runs ONCE per subject (cheap,
deterministic, identical every k). For each of K perturbations, the
PERTURBED bottleneck is replayed through the EXACT SAME decoder path the
real forward() uses (upconv3->dec3->upconv2->dec2->upconv1->attn_gate1->
dec1->seg_head), reusing the model's OWN layer objects unchanged -- not a
reimplementation, a resumed/replayed forward pass, verified against the
real forward() to be bit-identical when k=0 (zero perturbation).

MEASURED PER VOXEL, PER SUBJECT (ET channel):
  mu(x)    = mean_k p_k(x)
  A(x)     = Var_k[p_k(x)]                 -- the "ambiguity" signal
  p_cap(x) = 1[min_k p_k(x) > tau]          -- stable-consensus prediction
  p_0(x)   = the REAL, UNPERTURBED prediction (k=0 exactly, i.e. the
             checkpoint's own genuine forward pass -- this IS what E223/
             E226/E227/E228-A/E229/E230/E231 all call `detected`)

FOUR MEASUREMENTS (per the user's table): Dice(p_cap) vs Dice(p_0), FN,
FP, and the CRITICAL one -- does A(x) at GT lesion locations predict
missed/detected status, controlling for size/isolation (reusing E223's
own cached covariates).
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
import importlib.util

spec = importlib.util.spec_from_file_location(
    "t", str(ROOT / 'experiments/exp_e12_eggo_m/e130/train_e130_multimodal_baseline.py'))
t = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["e"]; spec.loader.exec_module(t); sys.argv = _a

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
MIN_VOX = 5
ET = 0
K_PERT = 20          # number of stochastic latent perturbations
TAU = 0.5            # consensus threshold for p_cap
DROPOUT_P = 0.15     # bottleneck channel dropout rate for perturbation
NOISE_STD = 0.05     # Gaussian noise std, relative to bottleneck's own std


def encoder_forward(model, x):
    """Runs ONLY the encoder + bottleneck, matching UNet3D_v5.forward()'s
    own code exactly (verified against the real source), stopping before
    the decoder. Returns everything the decoder replay needs."""
    enc1 = model.enc1(x)
    pool1 = model.pool1(enc1)
    enc2 = model.enc2(pool1)
    pool2 = model.pool2(enc2)
    enc3 = model.enc3(pool2)
    pool3 = model.pool3(enc3)
    bottleneck = model.bottleneck(pool3)
    return enc1, enc2, enc3, bottleneck


def decoder_forward(model, enc1, enc2, enc3, bottleneck_k):
    """Replays the EXACT decoder path from UNet3D_v5.forward() (verified
    against the real source line-by-line), starting from a (possibly
    perturbed) bottleneck tensor, reusing the model's own layer objects
    unchanged. Returns probs (B,3,D,H,W)."""
    upconv3 = model.upconv3(bottleneck_k)
    cat3 = torch.cat([upconv3, enc3], dim=1)
    dec3 = model.dec3(cat3)

    upconv2 = model.upconv2(dec3)
    cat2 = torch.cat([upconv2, enc2], dim=1)
    dec2 = model.dec2(cat2)

    upconv1 = model.upconv1(dec2)
    enc1_gated, _ = model.attn_gate1(gate=bottleneck_k, skip=enc1)
    cat1 = torch.cat([upconv1, enc1_gated], dim=1)
    dec1 = model.dec1(cat1)

    probs = model.seg_head(dec1)
    return probs


def perturb_bottleneck(bottleneck, rng_seed):
    """K-th stochastic perturbation: MC-dropout-style channel dropout +
    small Gaussian noise scaled to the bottleneck's OWN std (so the
    perturbation magnitude is calibrated to this specific checkpoint's
    actual activation scale, not an arbitrary absolute constant)."""
    g = torch.Generator(device=bottleneck.device).manual_seed(rng_seed)
    keep_mask = (torch.rand(bottleneck.shape[1], generator=g, device=bottleneck.device)
                > DROPOUT_P).float()
    keep_mask = keep_mask.view(1, -1, 1, 1, 1)
    noise = torch.randn(bottleneck.shape, generator=g, device=bottleneck.device)
    std = bottleneck.std().item()
    perturbed = bottleneck * keep_mask + noise * (std * NOISE_STD)
    return perturbed


def verify_k0_matches_real_forward(model, img_t):
    """SANITY CHECK before trusting the replay mechanism: with NO
    perturbation, decoder_forward(encoder_forward(x)) must be BIT-
    IDENTICAL to model(x)'s own real probs output. Run once at startup,
    not per-subject (deterministic, would always pass/fail the same way)."""
    with torch.no_grad():
        real_out = model(img_t)
        real_probs = real_out['probs']
        enc1, enc2, enc3, bottleneck = encoder_forward(model, img_t)
        replayed_probs = decoder_forward(model, enc1, enc2, enc3, bottleneck)
    max_diff = (real_probs - replayed_probs).abs().max().item()
    assert max_diff < 1e-5, f'Replay mismatch: max diff {max_diff} -- DO NOT TRUST RESULTS'
    print(f'  [verify] replay mechanism matches real forward() exactly '
          f'(max diff {max_diff:.2e})', flush=True)


def main():
    dev = torch.device('cuda')
    smoke = '--smoke' in sys.argv

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    n_total = 3 if smoke else len(ds)
    k_pert = 5 if smoke else K_PERT
    print(f'E232 intersection-consensus test: {n_total} VAL subjects, K={k_pert} '
          f'perturbations, tau={TAU}', flush=True)

    # one-time sanity check on the FIRST val subject
    image0, target0, _ = ds[0]
    D0, H0, W0 = image0.shape[1:]
    pd, ph, pw = t.PATCH
    cd, ch, cw = D0 // 2, H0 // 2, W0 // 2
    sd, ed = max(0, cd - pd // 2), min(D0, cd + pd // 2)
    sh, eh = max(0, ch - ph // 2), min(H0, ch + ph // 2)
    sw, ew = max(0, cw - pw // 2), min(W0, cw + pw // 2)
    img0_c = image0[:, sd:ed, sh:eh, sw:ew]
    if img0_c.shape[1:] != (pd, ph, pw):
        pad = torch.zeros((img0_c.shape[0], pd, ph, pw), dtype=img0_c.dtype)
        pad[:, :img0_c.shape[1], :img0_c.shape[2], :img0_c.shape[3]] = img0_c
        img0_c = pad
    img0_t = img0_c.unsqueeze(0).to(dev)
    verify_k0_matches_real_forward(model, img0_t)

    out = HERE / ('E232_smoke.csv' if smoke else 'E232_ambiguity.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=[
        'subject_id', 'comp_id', 'size', 'mean_A', 'max_A', 'detected_p0',
        'detected_pcap', 'dice_p0', 'dice_pcap', 'fn_p0', 'fn_pcap',
        'fp_p0_total', 'fp_pcap_total'])
    w.writeheader()

    t0 = time.time()
    for ii in range(n_total):
        image, target, sid = ds[ii]
        D, H, W = image.shape[1:]
        cd_, ch_, cw_ = D // 2, H // 2, W // 2
        sd_, ed_ = max(0, cd_ - pd // 2), min(D, cd_ + pd // 2)
        sh_, eh_ = max(0, ch_ - ph // 2), min(H, ch_ + ph // 2)
        sw_, ew_ = max(0, cw_ - pw // 2), min(W, cw_ + pw // 2)
        img_c = image[:, sd_:ed_, sh_:eh_, sw_:ew_]
        tgt_c = target[:, sd_:ed_, sh_:eh_, sw_:ew_]
        if img_c.shape[1:] != (pd, ph, pw):
            ip = torch.zeros((img_c.shape[0], pd, ph, pw), dtype=img_c.dtype)
            tp_ = torch.zeros((tgt_c.shape[0], pd, ph, pw), dtype=tgt_c.dtype)
            ip[:, :img_c.shape[1], :img_c.shape[2], :img_c.shape[3]] = img_c
            tp_[:, :tgt_c.shape[1], :tgt_c.shape[2], :tgt_c.shape[3]] = tgt_c
            img_c, tgt_c = ip, tp_
        img_t = img_c.unsqueeze(0).to(dev)
        tgt_np = tgt_c.numpy()

        with torch.no_grad():
            enc1, enc2, enc3, bottleneck = encoder_forward(model, img_t)

            # k=0: the REAL, unperturbed prediction (p_0)
            probs_0 = decoder_forward(model, enc1, enc2, enc3, bottleneck)[0, ET].cpu().numpy()

            all_probs = [probs_0]
            for k in range(1, k_pert):
                bn_k = perturb_bottleneck(bottleneck, rng_seed=1000 * ii + k)
                probs_k = decoder_forward(model, enc1, enc2, enc3, bn_k)[0, ET].cpu().numpy()
                all_probs.append(probs_k)

        all_probs = np.stack(all_probs, axis=0)  # (K, D, H, W)
        mu = all_probs.mean(axis=0)
        A = all_probs.var(axis=0)
        p_cap = (all_probs.min(axis=0) > TAU).astype(np.float32)
        p0_bin = (probs_0 > TAU).astype(np.float32)

        gt_et = (tgt_np[ET] > 0.5)

        def dice(pred, gt):
            inter = (pred * gt).sum()
            den = pred.sum() + gt.sum()
            return float(2 * inter / den) if den > 0 else 1.0

        dice_p0 = dice(p0_bin, gt_et.astype(np.float32))
        dice_pcap = dice(p_cap, gt_et.astype(np.float32))
        fn_p0 = int(((p0_bin == 0) & gt_et).sum())
        fn_pcap = int(((p_cap == 0) & gt_et).sum())
        fp_p0 = int(((p0_bin == 1) & ~gt_et).sum())
        fp_pcap = int(((p_cap == 1) & ~gt_et).sum())

        et_lbl, et_n = ndimage.label(gt_et)
        for g in range(1, et_n + 1):
            cm = et_lbl == g
            sz = int(cm.sum())
            if sz < MIN_VOX:
                continue
            mean_A = float(A[cm].mean())
            max_A = float(A[cm].max())
            det_p0 = int((p0_bin[cm].sum() / sz) >= 0.5)
            det_pcap = int((p_cap[cm].sum() / sz) >= 0.5)
            w.writerow({
                'subject_id': sid, 'comp_id': g, 'size': sz,
                'mean_A': mean_A, 'max_A': max_A,
                'detected_p0': det_p0, 'detected_pcap': det_pcap,
                'dice_p0': dice_p0, 'dice_pcap': dice_pcap,
                'fn_p0': fn_p0, 'fn_pcap': fn_pcap,
                'fp_p0_total': fp_p0, 'fp_pcap_total': fp_pcap,
            })
        fh.flush()
        if (ii + 1) % 25 == 0 or smoke:
            print(f'  {ii+1}/{n_total} subj ({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'wrote {out.name} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
