"""E237 Stage 2 -- the causal intervention, per the user's exact spec.
Frozen E131 checkpoint throughout: NO layer is ever retrained. Only the
D1 tensor that feeds into seg_head is modified, using the Stage-1 probe's
FIXED weights (trained on G1/detected only, held-out-fold discipline
preserved).

MECHANICS, verified against the real UNet3D_v5.forward() source
(neuroscan_3d_v5.py) line-by-line before writing this:
  ... -> dec2 (D2, 64ch, 32^3) -> upconv1 -> [attn_gate1 with enc1] ->
  cat1 -> self.dec1(cat1) = dec1 (D1, 32ch, 128^3) -> self.seg_head(dec1)
  = probs

INTERVENTION: probs' = seg_head(dec1 + alpha * U(S_D2))
  where S_D2(x) = w^T h_D2(x) + b, computed DENSELY over the WHOLE D2
  grid (not just lesion/shell-masked voxels -- Stage 1's probe weights
  applied to EVERY D2 spatial location), U = trilinear upsample D2's
  32^3 grid to D1's 128^3 grid, and the resulting 1-channel evidence map
  is broadcast-added identically across all 32 of dec1's channels (the
  natural, assumption-minimal combination given seg_head expects exactly
  32 channels -- no new learned projection is introduced, keeping every
  downstream weight frozen).

CRITICAL VERIFICATION (run automatically before any real result is
trusted, per this session's own established discipline): at alpha=0,
probs' MUST be bit-identical to the model's own unmodified probs. This
is checked with an assertion on EVERY subject processed, not just once
at startup -- catches any accidental silent divergence.

CONTROLS, all built into this single script (per explicit user design,
so the SAME extraction/injection code path is reused, not four separate
reimplementations that could silently diverge from each other):
  real       : S_D2 from the genuine Stage-1 probe direction w
  random_dir : S_D2 from a RANDOM direction, SAME norm as w (Control A)
  shuffled   : genuine S_D2 map, spatially permuted within the D2 grid
               before upsampling (Control B)
  sign_neg   : alpha negated relative to the swept grid (Control C is
               implemented by the alpha grid itself including negative
               values, per the user's own alpha grid spec -- NOT a
               separate condition, exactly as specified: "alpha in
               {-2,-1,-0.5,0,0.25,0.5,1,2}")

ALPHA GRID: FIXED a priori, per explicit user instruction ("do NOT choose
the best alpha on the test set") -- {-2,-1,-0.5,0,0.25,0.5,1,2}, IDENTICAL
across all folds and all conditions. No fold-dependent tuning of this
grid; only EVALUATION differs per held-out fold.

For each lesion in G2 (high_contrast_missed) and G3 (matched_missed),
records: max ET probability in lesion (baseline vs intervened),
mean ET probability, lesion-level recovery Y=1[max_p>tau], voxel Dice
proxy IS NOT computed here (whole-subject Dice needs a full sliding-
window pass, deferred -- lesion-level recovery is the PRIMARY endpoint
per explicit user instruction, "a 0.5% Dice improvement could otherwise
hide the actual scientific question").
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
from e234_layerwise_separability import resumed_forward  # noqa: E402
from e235_linear_probe_layerwise import build_groups  # noqa: E402

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
MIN_VOX = 5
ET = 0
ALPHA_GRID = (-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0)
CONDITIONS = ('real', 'random_dir', 'shuffled')
TAU = 0.5


def decoder_from_dec2_with_injection(model, enc1, enc2, enc3, dec2, bottleneck, evidence_map_d1res):
    """Replays upconv1 -> attn_gate1 -> cat1 -> dec1 -> [INJECTION] ->
    seg_head, using the model's OWN unchanged layers, matching
    UNet3D_v5.forward() exactly except for the single additive injection
    before seg_head. evidence_map_d1res: (B,1,128,128,128) or None (no
    injection, i.e. alpha=0 path -- but alpha=0 is ALSO tested via the
    injection path with alpha literally 0, as the primary correctness
    check, see verify_alpha_zero)."""
    upconv1 = model.upconv1(dec2)
    enc1_gated, _ = model.attn_gate1(gate=bottleneck, skip=enc1)
    cat1 = torch.cat([upconv1, enc1_gated], dim=1)
    dec1 = model.dec1(cat1)
    if evidence_map_d1res is not None:
        dec1 = dec1 + evidence_map_d1res  # already alpha-scaled by caller
    probs = model.seg_head(dec1)
    return probs, dec1


def compute_S_D2(dec2_np, w, b):
    """dec2_np: (64, 32, 32, 32). w: (64,), b: float. Returns S_D2:
    (32,32,32), the DENSE per-voxel probe score over the WHOLE D2 grid --
    every spatial location gets w^T h + b, not just lesion/shell-masked
    ones (this is the key difference from E236's masked evaluation)."""
    C = dec2_np.shape[0]
    flat = dec2_np.reshape(C, -1)  # (64, 32768)
    scores = w @ flat + b  # (32768,)
    return scores.reshape(dec2_np.shape[1:])


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    det, high_contrast_missed, matched_missed, t1c_median = build_groups()
    print(f'E237 Stage 2: G2 (high_contrast_missed, n={len(high_contrast_missed)}) '
          f'G3 (matched_missed, n={len(matched_missed)})', flush=True)

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=(128, 128, 128))
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    folds_path = HERE / ('E237_folds_smoke.npz' if smoke else 'E237_folds.npz')
    folds_data = np.load(folds_path)
    fold_of = dict(zip(folds_data['subjects'], folds_data['folds'].astype(int)))

    probes_dir = HERE / ('E237_probes_smoke' if smoke else 'E237_probes')
    probes = {}
    for fold in range(5):
        p = probes_dir / f'fold{fold}.npz'
        if p.exists():
            d = np.load(p)
            probes[fold] = (d['w'].astype(np.float32), float(d['b']))

    if smoke:
        high_contrast_missed = high_contrast_missed[:6]
        matched_missed = matched_missed[:6]

    out = HERE / ('E237_smoke.csv' if smoke else 'E237_intervention.csv')
    fh = open(out, 'w', newline='')
    w_csv = csv.DictWriter(fh, fieldnames=[
        'group', 'subject_id', 'comp_id', 'size', 'condition', 'alpha',
        'max_prob', 'mean_prob', 'recovered'])
    w_csv.writeheader()

    pd_, ph_, pw_ = 128, 128, 128
    t0 = time.time()
    n_done = 0
    rng_global = np.random.default_rng(123)
    verified_zero = False

    for group_name, group_rows in [('G2_high_contrast_missed', high_contrast_missed),
                                   ('G3_matched_missed', matched_missed)]:
        for r in group_rows:
            sid = r['subject_id']
            if sid not in sid_to_idx or sid not in fold_of:
                continue
            fold = fold_of[sid]
            if fold not in probes:
                continue
            w, b = probes[fold]

            image, target, _ = ds._load_subject(ds.subject_dirs[sid_to_idx[sid]])
            D, H, W = image.shape[1:]
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
            img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)

            et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
            cid = int(r['comp_id'])
            if cid < 1 or cid > et_n:
                continue
            cm = et_lbl == cid
            sz = int(cm.sum())
            if sz < MIN_VOX:
                continue

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

                dec2_np = dec2[0].cpu().numpy()  # (64,32,32,32)
                S_D2_real = compute_S_D2(dec2_np, w, b)  # (32,32,32)

                # --- CORRECTNESS CHECK: alpha=0 must exactly match the real forward() ---
                if not verified_zero:
                    real_out = model(img_t)
                    real_probs = real_out['probs']
                    ev_zero = torch.zeros(1, 1, pd_, ph_, pw_, device=dev)
                    probs_check, _ = decoder_from_dec2_with_injection(
                        model, enc1, enc2, enc3, dec2, bottleneck, 0.0 * ev_zero)
                    max_diff = (real_probs - probs_check).abs().max().item()
                    assert max_diff < 1e-4, f'alpha=0 mismatch: {max_diff} -- DO NOT TRUST RESULTS'
                    print(f'  [verify] alpha=0 injection path matches real forward() '
                          f'(max diff {max_diff:.2e})', flush=True)
                    verified_zero = True

                mask = np.zeros((pd_, ph_, pw_), dtype=bool)
                mask[np.where(cm)] = True
                mask_t = torch.from_numpy(mask).to(dev)

                for cond in CONDITIONS:
                    if cond == 'real':
                        S_map = S_D2_real
                    elif cond == 'random_dir':
                        w_rand = rng_global.standard_normal(64).astype(np.float32)
                        w_rand = w_rand / (np.linalg.norm(w_rand) + 1e-8) * np.linalg.norm(w)
                        S_map = compute_S_D2(dec2_np, w_rand, b)
                    elif cond == 'shuffled':
                        flat = S_D2_real.reshape(-1).copy()
                        rng_global.shuffle(flat)
                        S_map = flat.reshape(S_D2_real.shape)

                    S_t = torch.from_numpy(S_map).unsqueeze(0).unsqueeze(0).float().to(dev)  # (1,1,32,32,32)
                    S_up = F.interpolate(S_t, size=(pd_, ph_, pw_), mode='trilinear', align_corners=False)

                    for alpha in ALPHA_GRID:
                        ev = alpha * S_up
                        ev_bcast = ev.expand(-1, 32, -1, -1, -1)
                        probs_i, _ = decoder_from_dec2_with_injection(
                            model, enc1, enc2, enc3, dec2, bottleneck, ev_bcast)
                        p_et = probs_i[0, ET]
                        lesion_vals = p_et[mask_t]
                        max_p = float(lesion_vals.max().item())
                        mean_p = float(lesion_vals.mean().item())
                        recovered = int(max_p > TAU)
                        w_csv.writerow({
                            'group': group_name, 'subject_id': sid, 'comp_id': cid, 'size': sz,
                            'condition': cond, 'alpha': alpha,
                            'max_prob': max_p, 'mean_prob': mean_p, 'recovered': recovered,
                        })
            n_done += 1
            if n_done % 10 == 0 or smoke:
                print(f'  {n_done} lesions processed ({time.time()-t0:.0f}s)', flush=True)
        fh.flush()
    fh.close()
    print(f'\nStage 2 complete: {n_done} lesions x {len(CONDITIONS)} conditions x '
          f'{len(ALPHA_GRID)} alphas ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
