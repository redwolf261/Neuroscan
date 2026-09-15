"""
E160 / Assumption L -- Does the E48 small-lesion -> bottleneck-dependence
relationship survive in Regime 2, where preprocessing has NOT destroyed the
small lesions?

THE QUESTION (pre-registered, single, narrow):

  E48 established Spearman(native_lesion_size, N_b) = -0.454 (p<0.001): SMALL
  lesions depend MORE on the bottleneck. That result, and the entire E48->E97
  chain (~50 experiments) built on it, was measured ENTIRELY in Regime 1:
      FLAIR-only, 1 channel, binary whole-tumour, 64^3 WHOLE-VOLUME RESIZE.
  E29 proved that under that 64^3 resize the MEDIAN native lesion component
  vanishes to ZERO voxels. So the relationship may be an artifact of
  information destruction the project itself introduced in preprocessing,
  rather than a property of the model.

  Regime 2 (this script) has NO resize: the multimodal validation loader
  returns the FULL NATIVE volume at 1mm and evaluates by sliding window
  (Dataset/brats_multimodal_dataset.py: "Patch sampling preserves native 1mm
  resolution"; val items are "the full native volume"). So a small lesion is
  still a small lesion when the model sees it.

  If the size->necessity relationship reproduces here, it is a real property
  of the model. If it vanishes, a large part of E48->E97 is confounded.

WHAT THIS SCRIPT DOES NOT DO (per explicit instruction):
  - no architecture change, no retraining, no new intervention
  - no attempt to improve Dice
  - ONE checkpoint, ONE inference pass, ONE predefined question

N_b CONSTRUCTION -- follows E48's own convention EXACTLY (verified against
e48/run_e48_bottleneck_encoding_audit.py:137-140 and E126's reimplementation):
the attention gate receives the SAME possibly-ablated bottleneck, NOT an
always-intact one. E109 got a false DOES_NOT_REPLICATE by getting this wrong
and was only caught via a 14x N_b scale mismatch. Do not "fix" this.

    N_b(subject) = Dice(intact) - Dice(bottleneck zeroed)

computed per REGION (ET/TC/WT) and as the 3-region mean, since Regime 2 is a
3-region task.

PRE-REGISTERED DECISION RULE (fixed before running):
  Primary test: Spearman(native WT lesion voxel count, N_b_mean) over all
  evaluated subjects, with a permutation p-value (10k shuffles).

  L = ARTIFACT   if rho >= -0.15 (i.e. the negative relationship is gone)
                 or the permutation p > 0.05
  L = SURVIVES   if rho <= -0.30 AND permutation p < 0.05
                 (-0.30 is a deliberately lenient two-thirds of E48's -0.454;
                  Regime 2 differs in modality/label/resolution so an exact
                  magnitude match is not required, only the same direction
                  and a real effect)
  L = AMBIGUOUS  otherwise -- report, do not adjudicate, name the confound.

  Secondary (reported, NOT part of the rule): the same correlation per region,
  and a small-vs-large split at the median.

OUTPUT: per-subject raw table (JSON + CSV) so the result can be adjudicated
from the raw numbers, not from this script's verdict.
"""
import sys
import json
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders, REGIONS  # noqa: E402

OUT_DIR = Path(__file__).parent
PATCH = (128, 128, 128)
OVERLAP = 0.5

# Regime-2 v5 4-in/3-out control. Structurally identical to E141's arms;
# this is the plain control arm (no evidence weighting).
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694


# ------------------------------------------------------------ N_b forward
def forward_with_optional_ablation(model, image, ablate_bottleneck):
    """Manual unroll of UNet3D_v5.forward().

    E48 CONVENTION (do not change): when the bottleneck is ablated, the
    attention gate is conditioned on the ABLATED bottleneck too -- the
    intended full severing of the coarse pathway's influence.
    """
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)

        if ablate_bottleneck:
            bottleneck = torch.zeros_like(bottleneck)

        upconv3 = model.upconv3(bottleneck)
        dec3 = model.dec3(torch.cat([upconv3, enc3], dim=1))
        upconv2 = model.upconv2(dec3)
        dec2 = model.dec2(torch.cat([upconv2, enc2], dim=1))
        upconv1 = model.upconv1(dec2)

        gate = bottleneck
        g = model.attn_gate1.W_g(gate)
        g_up = F.interpolate(g, size=enc1.shape[2:], mode="trilinear", align_corners=False)
        xg = model.attn_gate1.W_x(enc1)
        psi = torch.sigmoid(model.attn_gate1.W_psi(F.relu(g_up + xg)))
        enc1_gated = enc1 * psi

        dec1 = model.dec1(torch.cat([upconv1, enc1_gated], dim=1))
        logits = model.seg_head(dec1)
        probs = torch.sigmoid(logits) if logits.min() < 0 or logits.max() > 1 else logits
    return probs


def _gaussian_weight(shape, sigma_scale=0.125):
    coords = [np.linspace(-1, 1, s) for s in shape]
    g = np.ones(shape, dtype=np.float32)
    for i, c in enumerate(coords):
        sh = [1] * len(shape)
        sh[i] = -1
        g = g * np.exp(-(c ** 2) / (2 * sigma_scale ** 2)).reshape(sh).astype(np.float32)
    return np.maximum(g, 1e-4)


def sliding_window(model, image, ablate, device, n_out=3, amp=True):
    _, _, D, H, W = image.shape
    pd, ph, pw = PATCH
    stride = [max(1, int(p * (1 - OVERLAP))) for p in PATCH]

    def starts(full, p, st):
        if full <= p:
            return [0]
        s = list(range(0, full - p + 1, st))
        if s[-1] != full - p:
            s.append(full - p)
        return s

    zs, ys, xs = starts(D, pd, stride[0]), starts(H, ph, stride[1]), starts(W, pw, stride[2])
    acc = torch.zeros((n_out, D, H, W), device=device, dtype=torch.float32)
    wsum = torch.zeros((1, D, H, W), device=device, dtype=torch.float32)
    gw = torch.from_numpy(_gaussian_weight((min(pd, D), min(ph, H), min(pw, W)))).to(device)

    for z in zs:
        for y in ys:
            for x in xs:
                zc, yc, xc = min(pd, D), min(ph, H), min(pw, W)
                tile = image[:, :, z:z + zc, y:y + yc, x:x + xc]
                if tile.shape[2:] != (pd, ph, pw):
                    tile = F.pad(tile, (0, pw - tile.shape[4], 0, ph - tile.shape[3],
                                        0, pd - tile.shape[2]))
                with torch.amp.autocast("cuda", enabled=amp):
                    pr = forward_with_optional_ablation(model, tile, ablate)
                pr = pr.float()[:, :, :zc, :yc, :xc].squeeze(0)
                acc[:, z:z + zc, y:y + yc, x:x + xc] += pr * gw
                wsum[:, z:z + zc, y:y + yc, x:x + xc] += gw
    return (acc / wsum.clamp(min=1e-6)).cpu().numpy()


def dice_per_region(pred_bin, target_bin):
    """E141's exact convention, including the empty-target rule."""
    out = []
    for r in range(pred_bin.shape[0]):
        p, t = pred_bin[r], target_bin[r]
        ps, ts = p.sum(), t.sum()
        if ts == 0:
            out.append(1.0 if ps == 0 else 0.0)
        else:
            out.append(float(2.0 * (p * t).sum() / (ps + ts)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only the first N val subjects (0 = all)")
    ap.add_argument("--no_amp", action="store_true")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    got = ckpt.get("best_mean_dice")
    assert got is not None and abs(float(got) - EXPECTED_DICE) < 1e-9, \
        f"Checkpoint identity FAILED: expected {EXPECTED_DICE}, got {got}"
    print(f"[Sanity] checkpoint identity PASS (best_mean_dice={got}).")

    model = UNet3D_v5(in_channels=4, out_channels=3).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)
    print(f"[Data] val subjects: {len(val_loader.dataset)} (full native volumes, NO resize)")

    # Sanity: manual unroll (ablate=False) must match model.forward() exactly.
    img0, tgt0, sid0 = val_loader.dataset[0]
    tile = img0[:, :128, :128, :128].unsqueeze(0).to(device)
    with torch.no_grad():
        real = model(tile)
        real = real["probs"] if isinstance(real, dict) else real
        manual = forward_with_optional_ablation(model, tile, False)
        md = float((real - manual).abs().max())
    assert md < 1e-4, f"Manual trunk mismatch {md:.3e} -- STOP."
    print(f"[Sanity] manual trunk vs forward(): max abs diff {md:.3e} PASS.\n")

    amp = not a.no_amp
    records = []
    n = len(val_loader.dataset)
    if a.limit:
        n = min(n, a.limit)

    for i in range(n):
        image, target, sid = val_loader.dataset[i]
        image_b = image.unsqueeze(0).to(device)
        tgt = target.numpy()

        p_intact = sliding_window(model, image_b, False, device, amp=amp)
        p_abl = sliding_window(model, image_b, True, device, amp=amp)

        d_i = dice_per_region((p_intact > 0.5).astype(np.float32), tgt)
        d_a = dice_per_region((p_abl > 0.5).astype(np.float32), tgt)
        nb = [d_i[r] - d_a[r] for r in range(3)]

        rec = {
            "sid": sid,
            "native_shape": list(tgt.shape[1:]),
            "size_ET": int(tgt[0].sum()),
            "size_TC": int(tgt[1].sum()),
            "size_WT": int(tgt[2].sum()),
            "dice_intact": {REGIONS[r]: d_i[r] for r in range(3)},
            "dice_ablated": {REGIONS[r]: d_a[r] for r in range(3)},
            "N_b": {REGIONS[r]: nb[r] for r in range(3)},
            "dice_intact_mean": float(np.mean(d_i)),
            "dice_ablated_mean": float(np.mean(d_a)),
            "N_b_mean": float(np.mean(nb)),
        }
        records.append(rec)
        print(f"[{i+1}/{n}] {sid} size_WT={rec['size_WT']:>7d} "
              f"intact={rec['dice_intact_mean']:.4f} abl={rec['dice_ablated_mean']:.4f} "
              f"N_b={rec['N_b_mean']:+.4f}", flush=True)

    # ------------------------------------------------- pre-registered test
    size = np.array([r["size_WT"] for r in records], dtype=float)
    nbm = np.array([r["N_b_mean"] for r in records], dtype=float)

    rho, p_param = stats.spearmanr(size, nbm)
    rng = np.random.default_rng(0)
    null = np.array([stats.spearmanr(size, rng.permutation(nbm)).statistic
                     for _ in range(10000)])
    p_perm = float((np.abs(null) >= abs(rho)).mean())

    per_region = {}
    for r, name in enumerate(REGIONS):
        v = np.array([rec["N_b"][name] for rec in records], dtype=float)
        s = np.array([rec[f"size_{name}"] for rec in records], dtype=float)
        rr, pp = stats.spearmanr(s, v)
        per_region[name] = {"rho_size_vs_Nb": float(rr), "p": float(pp),
                            "N_b_mean": float(v.mean()), "N_b_std": float(v.std())}

    med = float(np.median(size))
    small = nbm[size <= med]
    large = nbm[size > med]

    if rho >= -0.15 or p_perm > 0.05:
        verdict = "ARTIFACT"
    elif rho <= -0.30 and p_perm < 0.05:
        verdict = "SURVIVES"
    else:
        verdict = "AMBIGUOUS"

    summary = {
        "question": "Does E48's size->N_b relationship survive without the 64^3 resize?",
        "regime": "Regime 2: 4-modality, 3-region, FULL NATIVE resolution, sliding window",
        "checkpoint": str(CKPT),
        "checkpoint_dice_verified": float(got),
        "n_subjects": len(records),
        "E48_reference_rho": -0.454,
        "primary": {
            "spearman_rho_sizeWT_vs_NbMean": float(rho),
            "parametric_p": float(p_param),
            "permutation_p_10k": p_perm,
        },
        "per_region": per_region,
        "median_split": {
            "median_size_WT": med,
            "small_N_b_mean": float(small.mean()), "small_n": int(small.size),
            "large_N_b_mean": float(large.mean()), "large_n": int(large.size),
        },
        "N_b_mean_overall": float(nbm.mean()),
        "N_b_std_overall": float(nbm.std()),
        "dice_intact_mean_overall": float(np.mean([r["dice_intact_mean"] for r in records])),
        "PREREGISTERED_VERDICT": verdict,
        "decision_rule": "ARTIFACT if rho>=-0.15 or perm_p>0.05; SURVIVES if rho<=-0.30 and perm_p<0.05; else AMBIGUOUS",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "E160_L_per_subject.json", "w") as f:
        json.dump(records, f, indent=1)
    with open(OUT_DIR / "E160_L_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    with open(OUT_DIR / "E160_L_per_subject.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "size_ET", "size_TC", "size_WT",
                    "dice_intact_ET", "dice_intact_TC", "dice_intact_WT",
                    "dice_abl_ET", "dice_abl_TC", "dice_abl_WT",
                    "N_b_ET", "N_b_TC", "N_b_WT", "N_b_mean"])
        for r in records:
            w.writerow([r["sid"], r["size_ET"], r["size_TC"], r["size_WT"],
                        *[r["dice_intact"][k] for k in REGIONS],
                        *[r["dice_ablated"][k] for k in REGIONS],
                        *[r["N_b"][k] for k in REGIONS], r["N_b_mean"]])

    print("\n" + "=" * 70)
    print(json.dumps(summary, indent=1))
    print("=" * 70)
    print(f"\nRaw per-subject table written to {OUT_DIR / 'E160_L_per_subject.csv'}")


if __name__ == "__main__":
    main()
