"""
Phase E70 gate-map analysis: what did CAS actually learn, and does it
agree with E65's independent causal measurement?

This is the part of E70 that is genuinely ours. DCU and related deformable
skip modules apply an always-on correction and expose no readable
statement about WHERE correspondence is broken. CAS's gate g in [0,1] is
exactly such a statement: g -> 0 means "this junction was already
aligned, leave it", g -> 1 means "resample here".

Three analyses, inference-only (no training, no gradient):

  A. Gate/offset descriptive statistics -- did CAS learn a non-trivial
     correction at all, or did it collapse to the identity? (If the
     latter, MM_CAS == MM and the ablation result is explained.)

  B. Spatial agreement with E65. E65 measured, per voxel, how much the
     prediction degrades when the skip tensor is translated. If CAS is
     doing what it was designed to do, its learned gate should be
     ELEVATED where E65's intervention showed the network is most
     sensitive to correspondence. Tested here as a per-subject
     correlation between the CAS gate map and a per-voxel
     translation-sensitivity map recomputed with E65's own construction.

  C. Lesion-region concentration -- is the gate elevated inside/near
     lesions relative to background? Reported per subject with a
     paired test across the validation set.

Analysis B is the key one: it asks whether a module trained ONLY on the
segmentation objective independently rediscovers the correspondence
structure that a separate causal-intervention experiment measured. That
is a falsifiable prediction, and it can fail.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v10 import UNet3D_v10  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
RUNS = OUT_DIR / "runs"
TRANSLATION_VOXELS = 3          # E65's own tested displacement
N_PERM = 1000


@torch.no_grad()
def translation_sensitivity_map(model, image, shift=TRANSLATION_VOXELS):
    """Per-voxel |Δprediction| when the enc1 skip is translated by a known
    amount -- E65's own intervention, localised. Uses the CAS model with
    the gate forced OFF (gate=0 -> raw skip), so this measures the
    UNDERLYING architecture's correspondence sensitivity, not CAS's own
    behaviour. Returns (D,H,W)."""
    enc1 = model.enc1(image)
    pool1 = model.pool1(enc1); enc2 = model.enc2(pool1)
    pool2 = model.pool2(enc2); enc3 = model.enc3(pool2)
    pool3 = model.pool3(enc3); bott = model.bottleneck(pool3)
    up3 = model.upconv3(bott); dec3 = model.dec3(torch.cat([up3, enc3], 1))
    up2 = model.upconv2(dec3); dec2 = model.dec2(torch.cat([up2, enc2], 1))
    up1 = model.upconv1(dec2)

    def head(skip):
        d1 = model.dec1(torch.cat([up1, skip], 1))
        return model.seg_head(d1)

    p_ref = head(enc1)
    enc1_shift = torch.roll(enc1, shifts=(shift, shift, shift), dims=(2, 3, 4))
    p_shift = head(enc1_shift)
    return (p_ref - p_shift).abs().squeeze(0).squeeze(0)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck_path = RUNS / f"MM_CAS_seed{a.seed}" / "checkpoints" / "best.pth"
    if not ck_path.exists():
        print(f"Checkpoint not found (training may still be running): {ck_path}")
        return
    ck = torch.load(str(ck_path), map_location=device, weights_only=False)
    model = UNet3D_v10(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    ds = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                split="val", val_split=0.1, target_shape=(64, 64, 64))

    gate_means, gate_maxes, off_means, off_maxes = [], [], [], []
    corr_gate_vs_sensitivity, gate_lesion, gate_bg = [], [], []

    for i in range(len(ds)):
        img, msk, sid = ds[i]
        img = img.unsqueeze(0).to(device); msk = msk.unsqueeze(0).to(device)

        out = model(img, return_cas_aux=True)
        gate = out["cas_gate"].squeeze(0).squeeze(0)          # (D,H,W)
        delta = out["cas_delta"].squeeze(0)                   # (3,D,H,W)
        dmag = delta.norm(dim=0)                              # (D,H,W)

        gate_means.append(gate.mean().item()); gate_maxes.append(gate.max().item())
        off_means.append(dmag.mean().item()); off_maxes.append(dmag.max().item())

        # --- B: agreement with E65's translation-sensitivity ---
        sens = translation_sensitivity_map(model, img)
        g_flat = gate.flatten().cpu().numpy()
        s_flat = sens.flatten().cpu().numpy()
        # Subsample for tractable rank correlation (fixed stride, deterministic)
        idx = np.arange(0, g_flat.size, 37)
        r, _ = stats.spearmanr(g_flat[idx], s_flat[idx])
        if np.isfinite(r):
            corr_gate_vs_sensitivity.append(float(r))

        # --- C: lesion vs background gate ---
        m = msk.squeeze(0).squeeze(0).bool()
        if m.any() and (~m).any():
            gate_lesion.append(gate[m].mean().item())
            gate_bg.append(gate[~m].mean().item())

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(ds)}", flush=True)

    gm, gx = np.array(gate_means), np.array(gate_maxes)
    om, ox = np.array(off_means), np.array(off_maxes)
    print("\n=== A. Did CAS learn a non-trivial correction? ===")
    print(f"  gate   mean over subjects : {gm.mean():.4f}  (0 = pure identity, 1 = always resample)")
    print(f"  gate   max  over subjects : {gx.mean():.4f}")
    print(f"  |offset| mean (voxels)    : {om.mean():.4f}")
    print(f"  |offset| max  (voxels)    : {ox.mean():.4f}   (bound = 4.0·sqrt(3) = 6.93)")
    collapsed = bool(gm.mean() < 0.02 or om.mean() < 0.02)
    print(f"  -> collapsed to identity? {collapsed}")

    res = {"seed": a.seed, "gate_mean": float(gm.mean()), "gate_max_mean": float(gx.mean()),
           "offset_mean_vox": float(om.mean()), "offset_max_mean_vox": float(ox.mean()),
           "collapsed_to_identity": collapsed}

    print("\n=== B. Does the learned gate agree with E65's causal sensitivity map? ===")
    if corr_gate_vs_sensitivity:
        c = np.array(corr_gate_vs_sensitivity)
        t_stat, t_p = stats.ttest_1samp(c, 0.0)
        try:
            w_p = stats.wilcoxon(c)[1]
        except ValueError:
            w_p = float("nan")
        print(f"  per-subject Spearman(gate, translation-sensitivity), n={len(c)}")
        print(f"  mean rho = {c.mean():+.4f}   median = {np.median(c):+.4f}")
        print(f"  t-test p = {t_p:.3e}   Wilcoxon p = {w_p:.3e}")
        print(f"  frac subjects with rho>0 : {(c > 0).mean():.3f}")
        res["gate_vs_sensitivity"] = {"mean_rho": float(c.mean()), "median_rho": float(np.median(c)),
                                      "t_p": float(t_p), "wilcoxon_p": float(w_p),
                                      "frac_positive": float((c > 0).mean()), "n": int(len(c))}

    print("\n=== C. Is the gate concentrated on lesion regions? ===")
    if gate_lesion:
        gl, gb = np.array(gate_lesion), np.array(gate_bg)
        d = gl - gb
        t_stat, t_p = stats.ttest_rel(gl, gb)
        print(f"  gate in lesion : {gl.mean():.4f}")
        print(f"  gate in bg     : {gb.mean():.4f}")
        print(f"  paired delta   : {d.mean():+.4f}  (t p = {t_p:.3e})")
        res["gate_lesion_vs_bg"] = {"lesion": float(gl.mean()), "background": float(gb.mean()),
                                    "delta": float(d.mean()), "t_p": float(t_p)}

    with open(OUT_DIR / f"E70_cas_gate_analysis_seed{a.seed}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved E70_cas_gate_analysis_seed{a.seed}.json")


if __name__ == "__main__":
    main()
