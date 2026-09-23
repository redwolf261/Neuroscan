"""
E180 Stage 1 -- Transformation laboratory + tile ledger.

Pre-registered in docs/phases/PHASE_E180_1_TRANSFORMATION_LAB_PREREG.md. Read that
first; the transform definitions, identity settings, and sanity gates below are
copied from it and must not change after seeing results.

Builds:
  1. A deterministic tile-coordinate ledger (subject_id, window_id, z0/y0/x0,
     z1/y1/x1, enc3_shape) from the existing sliding_window() start-index logic.
     Reused verbatim by every later E180 stage.
  2. Five hook-swappable representation-space transformations (T1-T5) on the
     enc3 stage output, with their identity/no-op sanity checks.

NO training, NO architecture change, ONE checkpoint. Inference only.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders  # noqa: E402

OUT_DIR = Path(__file__).parent
PATCH = (128, 128, 128)
OVERLAP = 0.25
TARGET_STAGE = "enc3"

CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694

# ----------------------------------------------------------------- transforms
T1_RANK_GRID = [1, 2, 4, 8, 16, 32, 64, 128]
T2_SEED = 12345
T3_BETAS = [0.0, 0.05, 0.10, 0.20]
T4_GAMMAS = [1.0, 1.5, 2.0, 3.0]
T5_SIGMAS = [0.0, 0.5, 1.0, 1.5]


def lowrank_channels(x, r):
    """T1. x: (1,C,D,H,W). Mean-centred SVD truncation over the channel axis.
    Identical to run_e165_per_stage_rank.py:lowrank_channels -- do not diverge."""
    b, c, d, h, w = x.shape
    if r >= c:
        return x
    m = x.reshape(c, -1).float()
    mu = m.mean(dim=1, keepdim=True)
    mc = m - mu
    U, S, Vh = torch.linalg.svd(mc, full_matrices=False)
    approx = (U[:, :r] * S[:r]) @ Vh[:r, :]
    return (approx + mu).reshape(b, c, d, h, w).to(x.dtype)


def channel_permute(x, perm):
    """T2. x: (1,C,D,H,W). Permute the channel axis. DESTRUCTIVE CONTROL ONLY --
    never part of the eventual mechanism (arbitrary channel-semantic destruction)."""
    if perm is None:
        return x
    return x[:, perm, :, :, :]


def local_mix(x, beta):
    """T3. x: (1,C,D,H,W). Blend each voxel with its 6-neighbour spatial mean."""
    if beta <= 0:
        return x
    kernel = torch.zeros((1, 1, 3, 3, 3), device=x.device, dtype=x.dtype)
    kernel[0, 0, 1, 1, 0] = 1.0
    kernel[0, 0, 1, 1, 2] = 1.0
    kernel[0, 0, 1, 0, 1] = 1.0
    kernel[0, 0, 1, 2, 1] = 1.0
    kernel[0, 0, 0, 1, 1] = 1.0
    kernel[0, 0, 2, 1, 1] = 1.0
    kernel = kernel / 6.0
    b, c, d, h, w = x.shape
    xr = x.reshape(b * c, 1, d, h, w)
    neigh = F.conv3d(xr, kernel, padding=1).reshape(b, c, d, h, w)
    return (1 - beta) * x + beta * neigh


def spectral_reshape(x, gamma):
    """T4. x: (1,C,D,H,W). SVD over the channel axis (same decomposition as T1),
    reconstruct with S' = S^gamma, preserving U and V (the leading subspace)."""
    if gamma == 1.0:
        return x
    b, c, d, h, w = x.shape
    m = x.reshape(c, -1).float()
    mu = m.mean(dim=1, keepdim=True)
    mc = m - mu
    U, S, Vh = torch.linalg.svd(mc, full_matrices=False)
    S_pow = S.clamp(min=0).pow(gamma)
    # rescale to preserve total energy (sum of squares) so gamma isolates
    # SHAPE of the spectrum, not overall magnitude
    scale = S.pow(2).sum().sqrt() / S_pow.pow(2).sum().sqrt().clamp(min=1e-12)
    S_pow = S_pow * scale
    approx = (U * S_pow) @ Vh
    return (approx + mu).reshape(b, c, d, h, w).to(x.dtype)


def local_smooth(x, sigma):
    """T5. x: (1,C,D,H,W). Per-channel Gaussian spatial smoothing."""
    if sigma <= 0:
        return x
    radius = max(1, int(3 * sigma))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float32)
    g1 = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1 = (g1 / g1.sum()).to(x.dtype)
    b, c, d, h, w = x.shape
    xr = x.reshape(b * c, 1, d, h, w)
    kz = g1.view(1, 1, -1, 1, 1)
    ky = g1.view(1, 1, 1, -1, 1)
    kx = g1.view(1, 1, 1, 1, -1)
    xr = F.conv3d(xr, kz, padding=(radius, 0, 0))
    xr = F.conv3d(xr, ky, padding=(0, radius, 0))
    xr = F.conv3d(xr, kx, padding=(0, 0, radius))
    return xr.reshape(b, c, d, h, w)


TRANSFORMS = {
    "T1_rank": lowrank_channels,
    "T2_permute": channel_permute,
    "T3_mix": local_mix,
    "T4_spectral": spectral_reshape,
    "T5_smooth": local_smooth,
}
IDENTITY_PARAM = {
    "T1_rank": 128,
    "T2_permute": None,
    "T3_mix": 0.0,
    "T4_spectral": 1.0,
    "T5_smooth": 0.0,
}


class StageTransformer:
    """Forward hook applying one named transform with a given param to enc3's output."""

    def __init__(self):
        self.name = None
        self.param = None

    def __call__(self, module, inputs, output):
        if self.name is None:
            return output
        fn = TRANSFORMS[self.name]
        return fn(output, self.param)


def register(model, transformer):
    mod = getattr(model, TARGET_STAGE)
    return mod.register_forward_hook(
        lambda m, i, o: transformer(m, i, o))


# --------------------------------------------------------------- tile ledger
def sliding_window_starts(full, p, st):
    if full <= p:
        return [0]
    s = list(range(0, full - p + 1, st))
    if s[-1] != full - p:
        s.append(full - p)
    return s


TILE_LEDGER_OVERLAP = 0.0   # sparse, non-autocorrelated tile SELECTION for Gamma/Delta
                            # (independent of the 0.25 overlap used for inference/reassembly
                            # quality) -- see docs/phases/PHASE_E180_1_TRANSFORMATION_LAB_PREREG.md
                            # "tile ledger overlap correction"
ENC3_DOWNSAMPLE = 4


def build_tile_ledger(image_shape, patch=PATCH, overlap=TILE_LEDGER_OVERLAP):
    """image_shape: (D,H,W) of the full native volume. Returns list of tile dicts,
    DEDUPLICATED at enc3-mask granularity: adjacent 128^3 input tiles whose offset
    is smaller than ENC3_DOWNSAMPLE (4 voxels) map to the IDENTICAL enc3 mask after
    the //4 downsample and are dropped, keeping only the first occurrence. This
    also requires overlap=0 by default (sparse, non-overlapping tile selection)
    to keep adjacent SURVIVING tiles from sharing most of their enc3 extent --
    found necessary after overlap=0.25 tiles showed ~87.5% enc3-mask overlap,
    producing artificially duplicated/autocorrelated Gamma/Delta pairs."""
    D, H, W = image_shape
    pd, ph, pw = patch
    stride = [max(1, int(p * (1 - overlap))) for p in patch]
    zs = sliding_window_starts(D, pd, stride[0])
    ys = sliding_window_starts(H, ph, stride[1])
    xs = sliding_window_starts(W, pw, stride[2])
    tiles = []
    seen_enc3_masks = set()
    wid = 0
    n_dropped = 0
    for z in zs:
        for y in ys:
            for x in xs:
                zc, yc, xc = min(pd, D), min(ph, H), min(pw, W)
                z1, y1, x1 = z + zc, y + yc, x + xc
                enc3_key = (z // ENC3_DOWNSAMPLE, y // ENC3_DOWNSAMPLE, x // ENC3_DOWNSAMPLE,
                            z1 // ENC3_DOWNSAMPLE, y1 // ENC3_DOWNSAMPLE, x1 // ENC3_DOWNSAMPLE)
                if enc3_key in seen_enc3_masks:
                    n_dropped += 1
                    continue
                seen_enc3_masks.add(enc3_key)
                tiles.append({
                    "window_id": wid,
                    "z0": z, "y0": y, "x0": x,
                    "z1": z1, "y1": y1, "x1": x1,
                })
                wid += 1
    if n_dropped:
        print(f"    [dedup] dropped {n_dropped} tile(s) with duplicate enc3 mask")
    return tiles


def enc3_box_iou(t_a, t_b):
    """Intersection-over-union of two tiles' enc3-space bounding boxes."""
    def box(t):
        return (t["z0"] // ENC3_DOWNSAMPLE, t["y0"] // ENC3_DOWNSAMPLE, t["x0"] // ENC3_DOWNSAMPLE,
                t["z1"] // ENC3_DOWNSAMPLE, t["y1"] // ENC3_DOWNSAMPLE, t["x1"] // ENC3_DOWNSAMPLE)
    az0, ay0, ax0, az1, ay1, ax1 = box(t_a)
    bz0, by0, bx0, bz1, by1, bx1 = box(t_b)
    iz = max(0, min(az1, bz1) - max(az0, bz0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    inter = iz * iy * ix
    vol_a = (az1 - az0) * (ay1 - ay0) * (ax1 - ax0)
    vol_b = (bz1 - bz0) * (by1 - by0) * (bx1 - bx0)
    union = vol_a + vol_b - inter
    return inter / union if union > 0 else 0.0


def annotate_pairwise_overlap(tiles):
    """For each tile, record its MAX enc3-box IoU against any other tile in the
    same subject -- a covariate for how much a tile's Gamma/Delta might be
    autocorrelated with a neighbor, per the accepted structural-overlap finding."""
    for i, t in enumerate(tiles):
        max_iou = 0.0
        for j, t2 in enumerate(tiles):
            if i == j:
                continue
            max_iou = max(max_iou, enc3_box_iou(t, t2))
        t["max_neighbor_enc3_iou"] = max_iou
    return tiles


def enc3_tile_shape(model, device, patch=PATCH):
    """Run one dummy tile through enc3 to record the activation shape."""
    dummy = torch.zeros((1, 4) + patch, device=device)
    captured = {}

    def hook(m, i, o):
        captured["shape"] = list(o.shape)

    h = model.enc3.register_forward_hook(hook)
    with torch.no_grad():
        model(dummy)
    h.remove()
    return captured["shape"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5,
                    help="subjects for the ledger + sanity check pass (default 5, "
                         "per Stage 5.5's pipeline-gate discipline)")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many subjects before starting (used to select FRESH "
                         "subjects for Stage 5.5, not overlapping Stage 2/2b/3/4/5's 10)")
    ap.add_argument("--out_suffix", type=str, default="",
                    help="suffix for output filenames, e.g. '_55' for Stage 5.5's separate "
                         "ledger so it doesn't overwrite the 10-subject diagnostic ledger")
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

    transformer = StageTransformer()
    register(model, transformer)

    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    # ------------------------------------------------------- sanity gates
    img0, _, _ = val_loader.dataset[0]
    tile = img0[:, :128, :128, :128].unsqueeze(0).to(device)

    transformer.name, transformer.param = None, None
    with torch.no_grad():
        ref = model(tile)["probs"].float()
        again = model(tile)["probs"].float()
    diff_dormant = float((ref - again).abs().max())
    assert diff_dormant == 0.0, "dormant hook is not a no-op -- STOP."
    print(f"[Sanity] dormant hook exact no-op PASS (diff={diff_dormant:.3e}).")

    gate_results = {}
    for name in TRANSFORMS:
        transformer.name, transformer.param = name, IDENTITY_PARAM[name]
        with torch.no_grad():
            out = model(tile)["probs"].float()
        d = float((ref - out).abs().max())
        transformer.name, transformer.param = None, None
        tol = 1e-3 if name in ("T1_rank", "T4_spectral") else 0.0
        passed = d <= tol if tol > 0 else d == 0.0
        gate_results[name] = {"identity_param": IDENTITY_PARAM[name],
                              "max_abs_diff": d, "tolerance": tol, "pass": passed}
        status = "PASS" if passed else "FAIL"
        print(f"[Sanity] {name} identity ({IDENTITY_PARAM[name]}) {status} "
              f"(max diff={d:.3e}, tol={tol:.1e})")
        assert passed, f"{name} identity gate FAILED -- STOP."

    # ------------------------------------------------------- enc3 tile shape
    shape = enc3_tile_shape(model, device)
    print(f"[Info] enc3 activation shape for a full 128^3 tile: {shape}")

    # ------------------------------------------------------- tile ledger
    total = len(val_loader.dataset)
    n = min(a.limit, total - a.offset) if a.limit else (total - a.offset)
    print(f"[Ledger] building tile-coordinate ledger for {n} subjects "
          f"(offset={a.offset})...")
    ledger = {}
    for idx in range(n):
        i = a.offset + idx
        image, _, sid = val_loader.dataset[i]
        _, D, H, W = image.shape
        tiles = build_tile_ledger((D, H, W))
        for t in tiles:
            t["enc3_shape"] = shape
        tiles = annotate_pairwise_overlap(tiles)
        max_iou_overall = max((t["max_neighbor_enc3_iou"] for t in tiles), default=0.0)
        ledger[sid] = {"native_shape": [D, H, W], "n_tiles": len(tiles), "tiles": tiles}
        print(f"  [{idx+1}/{n}] {sid}  native_shape={[D,H,W]}  n_tiles={len(tiles)}  "
              f"max_neighbor_enc3_iou={max_iou_overall:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"E180_tile_ledger{a.out_suffix}.json", "w") as f:
        json.dump(ledger, f, indent=1)

    summary = {
        "question": "Stage 1: transformation laboratory sanity + tile ledger",
        "checkpoint": str(CKPT), "checkpoint_dice_verified": float(got),
        "target_stage": TARGET_STAGE,
        "transforms": list(TRANSFORMS.keys()),
        "identity_gate_results": gate_results,
        "dormant_hook_diff": diff_dormant,
        "enc3_tile_shape_full": shape,
        "n_subjects_ledger": n,
        "overlap": OVERLAP, "patch": list(PATCH),
        "ALL_GATES_PASS": all(g["pass"] for g in gate_results.values()) and diff_dormant == 0.0,
    }
    with open(OUT_DIR / f"E180_1_sanity_summary{a.out_suffix}.json", "w") as f:
        json.dump(summary, f, indent=1)

    print("\n" + "=" * 70)
    print(json.dumps(summary, indent=1))
    print("=" * 70)


if __name__ == "__main__":
    main()
