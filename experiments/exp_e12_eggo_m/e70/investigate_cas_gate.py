"""
Phase E70 follow-up: what IS the CAS gate actually tracking, if not
E65's correspondence-sensitivity structure?

The gate diagnostic (analyze_cas_gates.py) found:
  - gate is high in background (0.72), low in lesion (0.49)
  - gate anti-correlates with E65's translation-sensitivity map (rho=-0.39,
    unanimous across 125/125 subjects)

The most parsimonious alternative hypothesis: the gate is not tracking
correspondence need at all, but simply tracking local activation
STATISTICS of enc1 -- e.g. resampling more where the skip tensor is
low-magnitude/homogeneous (background, "safe" to blend) and less where
it is high-magnitude/high-variance (lesion boundaries, informative,
"risky" to blend). This would make CAS a magnitude-gated smoothing-style
operator, not a correspondence-correction operator, and would explain
both observed directions (B and C) with a single simple mechanism.

Tests, per subject, all inference-only:
  1. Spearman(gate, local |enc1| magnitude) -- is gate higher where
     activation magnitude is LOWER?
  2. Spearman(gate, local enc1 variance in a 3x3x3 neighborhood) -- is
     gate higher where local content is more homogeneous?
  3. Partial check: does the anti-correlation with E65's sensitivity
     map (test B) survive controlling for magnitude? If the magnitude
     story fully explains it, the partial correlation should shrink
     toward 0.
"""
import sys
import json
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
TRANSLATION_VOXELS = 3


@torch.no_grad()
def translation_sensitivity_map(model, image, shift=TRANSLATION_VOXELS):
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
    return (p_ref - p_shift).abs().squeeze(0).squeeze(0), enc1


def local_variance_3x3x3(x):
    """x: (C,D,H,W). Returns (D,H,W) mean-over-channels local variance
    via a 3x3x3 box filter (E[x^2]-E[x]^2), reflect-padded."""
    C = x.shape[0]
    xin = x.unsqueeze(0)  # (1,C,D,H,W)
    kernel = torch.ones(C, 1, 3, 3, 3, device=x.device, dtype=x.dtype) / 27.0
    xpad = F.pad(xin, (1, 1, 1, 1, 1, 1), mode="reflect")
    mean = F.conv3d(xpad, kernel, groups=C)
    x2pad = F.pad((xin ** 2), (1, 1, 1, 1, 1, 1), mode="reflect")
    mean2 = F.conv3d(x2pad, kernel, groups=C)
    var = (mean2 - mean ** 2).clamp(min=0)
    return var.squeeze(0).mean(dim=0)  # (D,H,W), averaged over channels


@torch.no_grad()
def main(seed=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(str(RUNS / f"MM_CAS_seed{seed}" / "checkpoints" / "best.pth"),
                    map_location=device, weights_only=False)
    model = UNet3D_v10(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    ds = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                split="val", val_split=0.1, target_shape=(64, 64, 64))

    rho_mag, rho_var, rho_partial = [], [], []
    stride = 41  # deterministic subsample for tractable rank correlation

    for i in range(len(ds)):
        img, msk, sid = ds[i]
        img = img.unsqueeze(0).to(device)

        out = model(img, return_cas_aux=True)
        gate = out["cas_gate"].squeeze(0).squeeze(0)  # (D,H,W)

        sens, enc1 = translation_sensitivity_map(model, img)
        enc1_c = enc1.squeeze(0)  # (C,D,H,W)
        mag = enc1_c.abs().mean(dim=0)  # (D,H,W)
        var = local_variance_3x3x3(enc1_c)

        g = gate.flatten().cpu().numpy()
        m = mag.flatten().cpu().numpy()
        v = var.flatten().cpu().numpy()
        s = sens.flatten().cpu().numpy()

        idx = np.arange(0, g.size, stride)
        g_s, m_s, v_s, s_s = g[idx], m[idx], v[idx], s[idx]

        r1, _ = stats.spearmanr(g_s, m_s)
        r2, _ = stats.spearmanr(g_s, v_s)
        if np.isfinite(r1):
            rho_mag.append(float(r1))
        if np.isfinite(r2):
            rho_var.append(float(r2))

        # Partial Spearman(gate, sensitivity | magnitude): rank-transform all
        # three, regress out magnitude linearly from ranks, correlate residuals.
        def rank(a):
            return stats.rankdata(a)
        rg, rs, rm = rank(g_s), rank(s_s), rank(m_s)
        # residualize rg and rs on rm via simple OLS
        A = np.vstack([rm, np.ones_like(rm)]).T
        beta_g, *_ = np.linalg.lstsq(A, rg, rcond=None)
        beta_s, *_ = np.linalg.lstsq(A, rs, rcond=None)
        resid_g = rg - A @ beta_g
        resid_s = rs - A @ beta_s
        if resid_g.std() > 1e-9 and resid_s.std() > 1e-9:
            rp = np.corrcoef(resid_g, resid_s)[0, 1]
            if np.isfinite(rp):
                rho_partial.append(float(rp))

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(ds)}", flush=True)

    rho_mag = np.array(rho_mag); rho_var = np.array(rho_var); rho_partial = np.array(rho_partial)

    print("\n=== Gate vs local |enc1| magnitude ===")
    print(f"  mean rho = {rho_mag.mean():+.4f}  median = {np.median(rho_mag):+.4f}  "
          f"frac<0 = {(rho_mag < 0).mean():.3f}  n={len(rho_mag)}")
    t, p = stats.ttest_1samp(rho_mag, 0.0)
    print(f"  t-test p = {p:.3e}")

    print("\n=== Gate vs local 3x3x3 variance ===")
    print(f"  mean rho = {rho_var.mean():+.4f}  median = {np.median(rho_var):+.4f}  "
          f"frac<0 = {(rho_var < 0).mean():.3f}  n={len(rho_var)}")
    t2, p2 = stats.ttest_1samp(rho_var, 0.0)
    print(f"  t-test p = {p2:.3e}")

    print("\n=== Partial correlation: gate vs E65-sensitivity, CONTROLLING for magnitude ===")
    print(f"  mean partial rho = {rho_partial.mean():+.4f}  median = {np.median(rho_partial):+.4f}  n={len(rho_partial)}")
    t3, p3 = stats.ttest_1samp(rho_partial, 0.0)
    print(f"  t-test p = {p3:.3e}")
    print(f"  (original unpartialled rho was -0.388; if this partial value is much closer to 0, "
          f"magnitude explains most of the anti-correlation)")

    result = {
        "seed": seed,
        "gate_vs_magnitude": {"mean_rho": float(rho_mag.mean()), "median_rho": float(np.median(rho_mag)),
                              "frac_negative": float((rho_mag < 0).mean()), "p": float(p)},
        "gate_vs_local_variance": {"mean_rho": float(rho_var.mean()), "median_rho": float(np.median(rho_var)),
                                   "frac_negative": float((rho_var < 0).mean()), "p": float(p2)},
        "gate_vs_sensitivity_partial_on_magnitude": {"mean_rho": float(rho_partial.mean()),
                                                      "median_rho": float(np.median(rho_partial)), "p": float(p3)},
        "original_unpartialled_rho": -0.3877,
    }
    with open(OUT_DIR / f"E70_cas_gate_investigation_seed{seed}.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved E70_cas_gate_investigation_seed0.json")


if __name__ == "__main__":
    main()
