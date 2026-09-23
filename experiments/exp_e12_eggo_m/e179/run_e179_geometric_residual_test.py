"""
E179 -- Geometric residual existence test.

Pre-registered in docs/phases/PHASE_E179_GEOMETRIC_RESIDUAL_EXISTENCE_TEST_PREREG.md.
Read that first. Purely diagnostic: NO training, NO new loss, NO correction
network, NO architecture change, ONE frozen checkpoint. Tests whether

    R(x) = ||grad p(x)|| / (|p(x)-tau| + eps)

identifies locations with recoverable segmentation error, BEYOND what
boundary-proximity-to-the-model's-OWN-prediction (S1) already tells you --
the central trap this design is built to close, not merely acknowledge.

Four tests, four gates, evaluated separately per region (ET/TC/WT) and per
subject (never pooled across subjects for the headline statistic -- pooling
would inflate significance via voxel non-independence, same failure mode
E180's subject-FE correction was built to fix elsewhere in this project).

BINDING RULES:
  - GT used ONLY for: (a) Test 1's error/distance targets (already the norm
    throughout this project), (b) Test 3's oracle correction's construction,
    NEVER to select which candidate voxels get tested (selection is by R(x)
    decile within d_pred bins, computed with zero GT involvement).
  - tau=0.5, the pipeline's own existing threshold convention (E167 onward).
  - boundary_distance() reused verbatim from E167 -- same interface
    definition, no reinterpretation.
  - Gate thresholds fixed BEFORE running: Gate 2 requires S3>S1 WITHIN
    matched d_pred bins; Gate 4 requires >=80% same-sign subjects.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy import stats

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders, REGIONS  # noqa: E402

OUT_DIR = Path(__file__).parent
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694
PATCH = (128, 128, 128)
OVERLAP = 0.5
TAU = 0.5           # LOCKED, pipeline's own threshold convention
EPS = 1e-3           # LOCKED, fixed before running
ORACLE_RADIUS = 2    # LOCKED, matches E167's STRATA convention
D_PRED_BINS = [0, 1, 2, 3]  # LOCKED, matches E167's STRATA
N_CANDIDATES_PER_CELL = 8   # LOCKED, per (subject, region, d_pred bin, R decile)
N_DECILES = 10
SEED = 0


def _gaussian_weight(shape, sigma_scale=0.125):
    coords = [np.linspace(-1, 1, s) for s in shape]
    g = np.ones(shape, dtype=np.float32)
    for i, c in enumerate(coords):
        sh = [1] * len(shape)
        sh[i] = -1
        g = g * np.exp(-(c ** 2) / (2 * sigma_scale ** 2)).reshape(sh).astype(np.float32)
    return np.maximum(g, 1e-4)


def sliding_window(model, image, device, n_out=3, amp=True):
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

    with torch.no_grad():
        for z in zs:
            for y in ys:
                for x in xs:
                    zc, yc, xc = min(pd, D), min(ph, H), min(pw, W)
                    tile = image[:, :, z:z + zc, y:y + yc, x:x + xc]
                    if tile.shape[2:] != (pd, ph, pw):
                        tile = F.pad(tile, (0, pw - tile.shape[4], 0, ph - tile.shape[3],
                                            0, pd - tile.shape[2]))
                    with torch.amp.autocast("cuda", enabled=amp):
                        out = model(tile)
                    pr = (out["probs"] if isinstance(out, dict) else out).float()
                    pr = pr[:, :, :zc, :yc, :xc].squeeze(0)
                    acc[:, z:z + zc, y:y + yc, x:x + xc] += pr * gw
                    wsum[:, z:z + zc, y:y + yc, x:x + xc] += gw
    return (acc / wsum.clamp(min=1e-6)).cpu().numpy()


def boundary_distance(bin_mask):
    """Verbatim from E167 -- interface-to-voxel Euclidean distance transform."""
    g = bin_mask.astype(bool)
    if not g.any():
        return None
    inner = g & ~binary_erosion(g, iterations=1, border_value=0)
    if not inner.any():
        inner = g
    return distance_transform_edt(~inner)


def gradient_magnitude(p):
    """Central-difference 3D gradient magnitude, no smoothing."""
    gz, gy, gx = np.gradient(p.astype(np.float64))
    return np.sqrt(gz ** 2 + gy ** 2 + gx ** 2).astype(np.float32)


def dice_whole(pred_bin, gt_bin):
    ps, gs = pred_bin.sum(), gt_bin.sum()
    if ps == 0 and gs == 0:
        return 1.0
    return float(2.0 * (pred_bin & gt_bin).sum() / (ps + gs))


def ball_mask(shape, center, radius):
    """Boolean mask of voxels within `radius` of `center` (Euclidean, sphere).
    Only the local bounding-box neighborhood is computed to avoid a full-
    volume distance calculation per candidate voxel (this is called many
    times per subject)."""
    z0, y0, x0 = center
    zlo, zhi = max(0, z0 - radius), min(shape[0], z0 + radius + 1)
    ylo, yhi = max(0, y0 - radius), min(shape[1], y0 + radius + 1)
    xlo, xhi = max(0, x0 - radius), min(shape[2], x0 + radius + 1)
    mask = np.zeros(shape, dtype=bool)
    zzf, yyf, xxf = np.meshgrid(np.arange(zlo, zhi), np.arange(ylo, yhi),
                                np.arange(xlo, xhi), indexing="ij")
    d2 = (zzf - z0) ** 2 + (yyf - y0) ** 2 + (xxf - x0) ** 2
    mask[zlo:zhi, ylo:yhi, xlo:xhi] = d2 <= radius ** 2
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--limit_subjects", type=int, default=0)
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

    amp = not a.no_amp
    n_target = a.limit_subjects if a.limit_subjects else len(val_loader.dataset)
    n = min(n_target, len(val_loader.dataset))
    print(f"[Data] {n} validation subjects")

    rng = np.random.default_rng(SEED)

    test1_records = []   # per subject/region: decile -> median d_GT, error rate
    test2_records = []   # per subject/region/d_pred_bin: rho(S1,err), rho(S3,err)
    test3_records = []   # per subject/region/d_pred_bin/decile: mean delta_dice

    for si in range(n):
        image, target, sid = val_loader.dataset[si]
        image_b = image.unsqueeze(0).to(device)
        tgt = target.numpy()

        probs = sliding_window(model, image_b, device, amp=amp)
        pred = (probs > TAU).astype(np.float32)

        for r, region in enumerate(REGIONS):
            gt = tgt[r].astype(bool)
            pr_bin = pred[r].astype(bool)
            p = probs[r]
            if not gt.any():
                continue

            R = gradient_magnitude(p) / (np.abs(p - TAU) + EPS)
            S1 = 1.0 / (np.abs(p - TAU) + EPS)
            S2 = gradient_magnitude(p)

            d_gt = boundary_distance(gt)
            d_pred = boundary_distance(pr_bin) if pr_bin.any() else None
            if d_gt is None or d_pred is None:
                continue

            err = np.logical_xor(gt, pr_bin)

            # ---------------- population restriction (Tests 1/2 share this)
            # R(x) is ~0 almost everywhere in the huge confident-background
            # region (measured: 99.6% of voxels have p~0, collapsing whole-
            # volume deciles into a degenerate couple of bins -- caught
            # during smoke-testing). Restrict to the SAME predicted-boundary
            # -proximal population Test 3 already uses (d_pred<=3), so all
            # three tests operate on one consistent, non-degenerate
            # candidate population -- not a post-hoc adjustment to flatter
            # the result, since Test 3's own design already made this choice.
            candidate_mask = d_pred <= D_PRED_BINS[-1]
            R_flat = R[candidate_mask]
            d_gt_flat = d_gt[candidate_mask]
            err_flat = err[candidate_mask]

            # ---------------- Test 1: R decile -> median d_GT, error rate
            deciles = np.percentile(R_flat, np.linspace(0, 100, N_DECILES + 1))
            bin_idx = np.digitize(R_flat, deciles[1:-1])
            t1 = []
            for k in range(N_DECILES):
                m = bin_idx == k
                if not m.any():
                    continue
                t1.append({
                    "decile": k, "median_d_gt": float(np.median(d_gt_flat[m])),
                    "error_rate": float(err_flat[m].mean()),
                })
            test1_records.append({"sid": sid, "region": region, "deciles": t1})

            # ---------------- Test 2: S3 vs S1 vs S2, within matched d_pred bins
            # NOTE: since candidate_mask already restricts to d_pred<=3, the
            # bins below partition [0,3] exactly (no separate "interior"
            # bin here -- that would be vacuous given the restriction, a
            # design inconsistency caught while wiring this up. The
            # whole-volume/interior comparison remains available from Test
            # 1's own d_gt/error-rate numbers if needed later).
            d_pred_flat = d_pred[candidate_mask]
            S1_flat = S1[candidate_mask]
            S2_flat = S2[candidate_mask]
            S3_flat = R_flat
            t2 = []
            for b in D_PRED_BINS:
                m = (d_pred_flat > (b - 1) if b > 0 else d_pred_flat >= 0) & (d_pred_flat <= b)
                if m.sum() < 50:
                    continue
                e = err_flat[m].astype(float)
                if e.std() == 0:
                    continue
                rho_s1, _ = stats.spearmanr(S1_flat[m], e)
                rho_s2, _ = stats.spearmanr(S2_flat[m], e)
                rho_s3, _ = stats.spearmanr(S3_flat[m], e)
                t2.append({"d_pred_bin": b, "n": int(m.sum()),
                          "rho_S1": float(rho_s1), "rho_S2": float(rho_s2),
                          "rho_S3": float(rho_s3)})
            test2_records.append({"sid": sid, "region": region, "bins": t2})

            # ---------------- Test 3: oracle local correction, matched d_pred bins
            baseline_dice = dice_whole(pr_bin, gt)
            t3 = []
            for b in D_PRED_BINS:
                boundary_mask = d_pred <= b if b == 0 else (d_pred > (b - 1)) & (d_pred <= b)
                cand_idx = np.argwhere(boundary_mask)
                if len(cand_idx) < N_DECILES * N_CANDIDATES_PER_CELL:
                    continue
                cand_R = R[boundary_mask]
                dec_edges = np.percentile(cand_R, np.linspace(0, 100, N_DECILES + 1))
                cand_decile = np.digitize(cand_R, dec_edges[1:-1])
                for k in range(N_DECILES):
                    pool = cand_idx[cand_decile == k]
                    if len(pool) == 0:
                        continue
                    take = min(N_CANDIDATES_PER_CELL, len(pool))
                    sel = pool[rng.choice(len(pool), size=take, replace=False)]
                    deltas = []
                    for center in sel:
                        m = ball_mask(gt.shape, tuple(center), ORACLE_RADIUS)
                        corrected = pr_bin.copy()
                        corrected[m] = gt[m]
                        deltas.append(dice_whole(corrected, gt) - baseline_dice)
                    t3.append({"d_pred_bin": b, "decile": k, "n": take,
                              "mean_delta_dice": float(np.mean(deltas))})
            test3_records.append({"sid": sid, "region": region, "cells": t3})

        print(f"[{si+1}/{n}] {sid}  done", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "E179_test1_raw.json", "w") as f:
        json.dump(test1_records, f, indent=1)
    with open(OUT_DIR / "E179_test2_raw.json", "w") as f:
        json.dump(test2_records, f, indent=1)
    with open(OUT_DIR / "E179_test3_raw.json", "w") as f:
        json.dump(test3_records, f, indent=1)
    print(f"\nSaved E179_test{{1,2,3}}_raw.json ({n} subjects)")


if __name__ == "__main__":
    main()
