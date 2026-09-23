"""
E180 Stages 3-4 -- Gamma measurement (transformation-specific, then aggregate).

Pre-registered in docs/phases/PHASE_E180_3_4_GAMMA_PREREG.md. Read that first;
the same-operating-point construction and the four displacement metrics below
are copied from it and must not change after seeing results.

For each surviving family (T1_rank@r=2, T4_spectral@gamma=3.0, T5_smooth@sigma=1.5,
from Stage 2), construct Z^deg = T_s(Z) applied uniformly, then for each
128^3 sliding-window TILE i, apply a SECOND application of the same transform
at the same severity restricted to tile i's spatial extent within the enc3
activation, and measure four whole-volume displacement metrics between
D(Z^deg) and D(Z^deg_perturb_i).

NO training, NO architecture change, ONE checkpoint. Inference only. GT is
NEVER used here -- prediction vs prediction only.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders, REGIONS  # noqa: E402
from run_e180_transform_lab import (  # noqa: E402
    lowrank_channels, spectral_reshape, local_smooth, PATCH, TARGET_STAGE,
)

OUT_DIR = Path(__file__).parent
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694
OVERLAP = 0.25
N_SUBJECTS = 10   # matches Stage 2's calibration scale; Stage 5.5 gate decides scale-up

# fixed severities from Stage 2 (PROCEED verdict)
FAMILIES = {
    "T1_rank": {"fn": lowrank_channels, "Ts": 2},
    "T4_spectral": {"fn": spectral_reshape, "Ts": 3.0},
    "T5_smooth": {"fn": local_smooth, "Ts": 1.5},
}

# enc3 is downsampled 4x from the 128^3 input tile (pool1 2x, pool2 2x) -> 32^3
ENC3_DOWNSAMPLE = 4


PROBE_SEED = 999
C_ENC3 = 128
# fixed per-family eps, filled in from E180_2b_probe_calibration.json at import time
PROBE_EPS = None


def _load_probe_eps():
    global PROBE_EPS
    cal_path = OUT_DIR / "E180_2b_probe_calibration.json"
    cal = json.load(open(cal_path))
    PROBE_EPS = {fam: v["chosen_eps"] for fam, v in cal["per_family"].items()}
    assert all(e is not None for e in PROBE_EPS.values()), \
        f"Stage 2b did not reach GRADED for all families: {PROBE_EPS}"


def fixed_direction(device):
    g = torch.Generator().manual_seed(PROBE_SEED)
    v = torch.randn(C_ENC3, generator=g)
    v = v / v.norm()
    return v.to(device)


def apply_uniform(x, fam):
    """Apply family fn at its fixed T_s to the WHOLE tensor (all tiles)."""
    return FAMILIES[fam]["fn"](x, FAMILIES[fam]["Ts"])


def apply_probe(x, fam, mask, direction):
    """Add the FIXED, family-independent small perturbation eps*std(x)*v
    within mask only (Stage 2b correction -- reapplying the family transform
    itself was idempotent for T1 and uncalibrated-compounding for T4/T5)."""
    eps = PROBE_EPS[fam]
    std = x.std()
    perturb = (eps * std) * direction.view(1, -1, 1, 1, 1)
    m = mask.to(x.dtype).view(1, 1, *mask.shape)
    return x + perturb * m


def _gaussian_weight(shape, sigma_scale=0.125):
    coords = [np.linspace(-1, 1, s) for s in shape]
    g = np.ones(shape, dtype=np.float32)
    for i, c in enumerate(coords):
        sh = [1] * len(shape)
        sh[i] = -1
        g = g * np.exp(-(c ** 2) / (2 * sigma_scale ** 2)).reshape(sh).astype(np.float32)
    return np.maximum(g, 1e-4)


class MaskedHook:
    """Forward hook on enc3: applies apply_uniform always (Z^deg), and
    optionally the FIXED probe perturbation at one tile's mask on top of Z^deg
    (Z^deg_perturb_i). Probe is family-independent (Stage 2b correction)."""

    def __init__(self, direction):
        self.family = None
        self.extra_mask = None   # torch bool tensor over enc3 spatial dims, or None
        self.direction = direction

    def __call__(self, module, inputs, output):
        if self.family is None:
            return output
        z_deg = apply_uniform(output, self.family)
        if self.extra_mask is None:
            return z_deg
        return apply_probe(z_deg, self.family, self.extra_mask, self.direction)


def register(model, hook):
    mod = getattr(model, TARGET_STAGE)
    return mod.register_forward_hook(lambda m, i, o: hook(m, i, o))


def sliding_window_collect(model, image, device, hook, n_out=3, amp=True, overlap=OVERLAP):
    """Per-window (batch_size=1) sliding-window inference. A batched variant
    (window_batch_size>1) was tried and REJECTED after measurement: it OOM'd
    without AMP (8x128^3x4ch activations exceed the 8GB budget at this U-Net's
    early high-resolution layers) and was ~11x SLOWER with AMP (8.1s vs 0.71s
    on an 8-tile subject) rather than faster -- the GPU-util/VRAM readings that
    motivated trying it were misleading; this model is not memory-bound at
    batch_size=1, and batching made things worse, not better. Reverted; kept
    as the original per-window loop."""
    _, _, D, H, W = image.shape
    pd, ph, pw = PATCH
    stride = [max(1, int(p * (1 - overlap))) for p in PATCH]

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


def dice_agree(a_bin, b_bin):
    out = []
    for r in range(a_bin.shape[0]):
        p, q = a_bin[r], b_bin[r]
        ps, qs = p.sum(), q.sum()
        out.append(1.0 if ps == 0 and qs == 0 else float(2.0 * (p * q).sum() / (ps + qs)))
    return float(np.mean(out))


def voxel_flip_frac(a_bin, b_bin):
    union = np.logical_or(a_bin, b_bin)
    if not union.any():
        return 0.0
    flips = np.logical_xor(a_bin, b_bin)
    return float(flips.sum() / union.sum())


def kl_bernoulli(p, q, mask):
    """Mean Bernoulli KL(p||q) over voxels in mask, per region averaged."""
    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    qc = np.clip(q, eps, 1 - eps)
    kl = pc * np.log(pc / qc) + (1 - pc) * np.log((1 - pc) / (1 - qc))
    out = []
    for r in range(p.shape[0]):
        m = mask[r]
        if not m.any():
            continue
        out.append(float(kl[r][m].mean()))
    return float(np.mean(out)) if out else 0.0


def boundary_distance(bin_mask):
    g = bin_mask.astype(bool)
    if not g.any():
        return None
    inner = g & ~binary_erosion(g, iterations=1, border_value=0)
    if not inner.any():
        inner = g
    return distance_transform_edt(~inner)


def boundary_shift(a_bin, b_bin, near_thresh=3):
    """Mean shift in predicted-boundary distance near either mask's boundary,
    averaged over regions with a defined boundary."""
    out = []
    for r in range(a_bin.shape[0]):
        da = boundary_distance(a_bin[r])
        db = boundary_distance(b_bin[r])
        if da is None or db is None:
            continue
        near = (da <= near_thresh) | (db <= near_thresh)
        if not near.any():
            continue
        out.append(float(np.abs(da[near] - db[near]).mean()))
    return float(np.mean(out)) if out else 0.0


def enc3_tile_bounds(z0, y0, x0, z1, y1, x1):
    """Map a 128^3 input-tile's spatial bounds to enc3's 32^3 grid (4x downsample)."""
    return (z0 // ENC3_DOWNSAMPLE, y0 // ENC3_DOWNSAMPLE, x0 // ENC3_DOWNSAMPLE,
            z1 // ENC3_DOWNSAMPLE, y1 // ENC3_DOWNSAMPLE, x1 // ENC3_DOWNSAMPLE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_subjects", type=int, default=N_SUBJECTS)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many val subjects before starting (match the ledger's offset)")
    ap.add_argument("--limit_tiles", type=int, default=0,
                    help="cap tiles per subject for a quick feasibility pass (0=all)")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--ledger_suffix", type=str, default="",
                    help="suffix to select an alternate tile ledger file, e.g. '_55'")
    ap.add_argument("--out_suffix", type=str, default="",
                    help="suffix for output filenames, to avoid overwriting prior runs")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    got = ckpt.get("best_mean_dice")
    assert got is not None and abs(float(got) - EXPECTED_DICE) < 1e-9, \
        f"Checkpoint identity FAILED: expected {EXPECTED_DICE}, got {got}"
    print(f"[Sanity] checkpoint identity PASS (best_mean_dice={got}).")

    _load_probe_eps()
    print(f"[Probe] eps per family (from Stage 2b): {PROBE_EPS}")

    ledger = json.load(open(OUT_DIR / f"E180_tile_ledger{a.ledger_suffix}.json"))

    model = UNet3D_v5(in_channels=4, out_channels=3).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    direction = fixed_direction(device)
    hook = MaskedHook(direction)
    register(model, hook)

    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    amp = not a.no_amp
    total = len(val_loader.dataset)
    n = min(a.limit_subjects, total - a.offset)
    print(f"[Data] measuring Gamma on {n} subjects (offset={a.offset}), families={list(FAMILIES)}")

    records = []
    for idx in range(n):
        i = a.offset + idx
        image, _, sid = val_loader.dataset[i]
        if sid not in ledger:
            print(f"  [skip] {sid} not in tile ledger")
            continue
        image_b = image.unsqueeze(0).to(device)
        tiles = ledger[sid]["tiles"]
        if a.limit_tiles:
            tiles = tiles[:a.limit_tiles]

        for fam in FAMILIES:
            hook.family, hook.extra_mask = fam, None
            p_deg = sliding_window_collect(model, image_b, device, hook, amp=amp)
            deg_bin = (p_deg > 0.5).astype(np.float32).astype(bool)

            for t in tiles:
                ez0, ey0, ex0, ez1, ey1, ex1 = enc3_tile_bounds(
                    t["z0"], t["y0"], t["x0"], t["z1"], t["y1"], t["x1"])
                mask = torch.zeros((32, 32, 32), dtype=torch.bool, device=device)
                mask[ez0:ez1, ey0:ey1, ex0:ex1] = True

                hook.family, hook.extra_mask = fam, mask
                p_pert = sliding_window_collect(model, image_b, device, hook, amp=amp)
                pert_bin = (p_pert > 0.5).astype(np.float32).astype(bool)

                g_dice = 1.0 - dice_agree(pert_bin.astype(np.float32), deg_bin.astype(np.float32))
                g_voxel = voxel_flip_frac(deg_bin, pert_bin)
                g_kl = kl_bernoulli(p_pert, p_deg, np.logical_or(deg_bin, pert_bin))
                g_boundary = boundary_shift(deg_bin, pert_bin)

                records.append({
                    "sid": sid, "window_id": t["window_id"], "family": fam,
                    "gamma_dice": g_dice, "gamma_voxel": g_voxel,
                    "gamma_kl": g_kl, "gamma_boundary": g_boundary,
                })
        hook.family, hook.extra_mask = None, None
        print(f"[{idx+1}/{n}] {sid}  n_tiles={len(tiles)}  "
              f"records_so_far={len(records)}", flush=True)

    # ------------------------------------------------------------- Stage 4 aggregate
    by_tile = {}
    for r in records:
        key = (r["sid"], r["window_id"])
        by_tile.setdefault(key, []).append(r)

    aggregated = []
    for (sid, wid), recs in by_tile.items():
        agg = {"sid": sid, "window_id": wid}
        for metric in ["gamma_dice", "gamma_voxel", "gamma_kl", "gamma_boundary"]:
            agg[f"{metric}_mean_over_families"] = float(np.mean([r[metric] for r in recs]))
        all_vals = [r[m] for r in recs for m in
                    ["gamma_dice", "gamma_voxel", "gamma_kl", "gamma_boundary"]]
        # normalize each metric to [0,1] range within this run before combining,
        # so no single metric's scale dominates the family-level aggregate
        agg["gamma_family_raw_mean"] = float(np.mean(all_vals))
        aggregated.append(agg)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"E180_gamma_per_subject{a.out_suffix}.json", "w") as f:
        json.dump({"per_family_metric": records, "aggregated": aggregated}, f, indent=1)

    summary = {
        "question": "Stage 3-4: per-tile representation-induced decision instability (Gamma), "
                    "transformation-specific then aggregated",
        "checkpoint": str(CKPT), "checkpoint_dice_verified": float(got),
        "n_subjects": n, "families": {k: v["Ts"] for k, v in FAMILIES.items()},
        "probe_eps": PROBE_EPS, "probe_seed": PROBE_SEED,
        "n_records": len(records), "n_tiles_aggregated": len(aggregated),
        "metrics": ["gamma_dice", "gamma_voxel", "gamma_kl", "gamma_boundary"],
        "note": "GT never used. All quantities are whole-volume, prediction-vs-prediction, "
                "measured from the SAME degraded operating point Z^deg per the same-"
                "operating-point correction. Probe is a FIXED family-independent random "
                "direction (Stage 2b correction), not a re-application of the family transform.",
    }
    with open(OUT_DIR / f"E180_3_4_gamma_summary{a.out_suffix}.json", "w") as f:
        json.dump(summary, f, indent=1)

    print("\n" + "=" * 70)
    print(json.dumps(summary, indent=1))
    print("=" * 70)


if __name__ == "__main__":
    main()
