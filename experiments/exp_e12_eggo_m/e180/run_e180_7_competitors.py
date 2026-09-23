"""
E180 Stage 7 -- Competitor ladder.

Pre-registered in docs/phases/PHASE_E180_6_9_ANALYSIS_PREREG.md and
docs/phases/PHASE_E180_7_COMPETITOR_LADDER_NOTE.md. Read both first.

Ladder: Magnitude -> Perturbation magnitude -> Uncertainty -> VJP sensitivity
-> Gamma. Each evaluated under the SAME subject fixed-effects framework as
Stage 6, not pooled correlation.

Uncertainty is elevated per the novelty audit (TRUST/CertainTTA-style
entropy-based TTA-selection signal) -- the literature-matched baseline any
future novelty claim must be shown not to reduce to.

Requires ONE additional GPU pass per subject (intact forward, for magnitude/
uncertainty/perturbation-magnitude at each tile's enc3 region) plus ONE VJP
per subject per tile (sensitivity), following E172's proven feasibility
pattern. No training, no architecture change, same checkpoint.
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
sys.path.insert(0, str(Path(__file__).parent))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders  # noqa: E402
from run_e180_gamma import FAMILIES, ENC3_DOWNSAMPLE, enc3_tile_bounds  # noqa: E402
from run_e180_transform_lab import PATCH, TARGET_STAGE  # noqa: E402

OUT_DIR = Path(__file__).parent
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694
OVERLAP = 0.25


def _gaussian_weight(shape, sigma_scale=0.125):
    coords = [np.linspace(-1, 1, s) for s in shape]
    g = np.ones(shape, dtype=np.float32)
    for i, c in enumerate(coords):
        sh = [1] * len(shape)
        sh[i] = -1
        g = g * np.exp(-(c ** 2) / (2 * sigma_scale ** 2)).reshape(sh).astype(np.float32)
    return np.maximum(g, 1e-4)


def single_tile_forward(model, image, z0, y0, x0, device, amp=True):
    """Run ONE 128^3 tile (not the full sliding-window volume) through the
    model, capturing enc3's intact activation via a hook, and the tile-local
    probability map. Used for per-tile competitor features (cheaper than a
    full-subject sliding-window pass since we only need one tile's enc3
    region and its own local prediction, matching how Gamma/Delta's tile
    definition already ties one 128^3 window to one enc3 mask)."""
    captured = {}

    def hook(m, i, o):
        captured["enc3"] = o.detach()

    h = getattr(model, TARGET_STAGE).register_forward_hook(hook)
    pd, ph, pw = PATCH
    _, C, D, H, W = image.shape
    zc, yc, xc = min(pd, D - z0), min(ph, H - y0), min(pw, W - x0)
    tile = image[:, :, z0:z0 + zc, y0:y0 + yc, x0:x0 + xc]
    if tile.shape[2:] != (pd, ph, pw):
        tile = F.pad(tile, (0, pw - tile.shape[4], 0, ph - tile.shape[3],
                            0, pd - tile.shape[2]))
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(tile)
    pr = (out["probs"] if isinstance(out, dict) else out).float()
    h.remove()
    return captured["enc3"], pr[:, :, :zc, :yc, :xc]


def vjp_sensitivity(model, image, z0, y0, x0, device):
    """VJP of total predicted foreground mass (label-free functional, per
    E172's precedent) w.r.t. the enc3 activation of ONE tile. Returns the
    gradient restricted to tile i's own enc3 mask region. Model parameters
    stay frozen (requires_grad_(False), per every other E180 stage) -- grad
    tracking is enabled only on the input tensor for this VJP, matching
    E172's inference-only, no-training discipline."""
    pd, ph, pw = PATCH
    _, C, D, H, W = image.shape
    zc, yc, xc = min(pd, D - z0), min(ph, H - y0), min(pw, W - x0)
    tile = image[:, :, z0:z0 + zc, y0:y0 + yc, x0:x0 + xc]
    if tile.shape[2:] != (pd, ph, pw):
        tile = F.pad(tile, (0, pw - tile.shape[4], 0, ph - tile.shape[3],
                            0, pd - tile.shape[2]))
    tile = tile.clone().requires_grad_(True)

    captured = {}

    def hook(m, i, o):
        o.retain_grad()
        captured["enc3"] = o

    h = getattr(model, TARGET_STAGE).register_forward_hook(hook)
    out = model(tile)
    pr = out["probs"] if isinstance(out, dict) else out
    total_mass = pr.sum()
    total_mass.backward()
    h.remove()
    grad = captured["enc3"].grad
    return grad.detach()


def enc3_tile_local_bounds(zc, yc, xc):
    """For a tile computed via single_tile_forward, its OWN enc3 mask is the
    full local grid (0,0,0)-(32,32,32) if the tile is a full 128^3 crop --
    but for per-tile competitor features we want the SAME global tile-i
    definition used by Gamma/Delta (enc3_tile_bounds on the ledger's stored
    z0..x1), not a re-derivation. This helper is unused; kept for clarity
    that competitor features use the ledger's own bounds directly."""
    raise NotImplementedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_subjects", type=int, default=125)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    got = ckpt.get("best_mean_dice")
    assert got is not None and abs(float(got) - EXPECTED_DICE) < 1e-9, \
        f"Checkpoint identity FAILED: expected {EXPECTED_DICE}, got {got}"
    print(f"[Sanity] checkpoint identity PASS (best_mean_dice={got}).")

    ledger = json.load(open(OUT_DIR / "E180_tile_ledger_full.json"))

    model = UNet3D_v5(in_channels=4, out_channels=3).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    amp = not a.no_amp
    total = len(val_loader.dataset)
    n = min(a.limit_subjects, total - a.offset)
    print(f"[Data] computing competitor features for {n} subjects (offset={a.offset})")

    records = []
    for idx in range(n):
        i = a.offset + idx
        image, _, sid = val_loader.dataset[i]
        if sid not in ledger:
            continue
        image_b = image.unsqueeze(0).to(device)
        tiles = ledger[sid]["tiles"]

        for t in tiles:
            z0, y0, x0 = t["z0"], t["y0"], t["x0"]
            ez0, ey0, ex0, ez1, ey1, ex1 = enc3_tile_bounds(
                t["z0"], t["y0"], t["x0"], t["z1"], t["y1"], t["x1"])

            enc3_intact, pr_local = single_tile_forward(model, image_b, z0, y0, x0, device, amp=amp)
            tile_mask_region = enc3_intact[:, :, ez0:ez1, ey0:ey1, ex0:ex1]

            # Level 1: magnitude
            magnitude = float(tile_mask_region.abs().mean())

            # Level 2: perturbation magnitude (norm of uniform-degradation effect,
            # recomputed per family since T_s differs per family)
            pert_mag = {}
            for fam, spec in FAMILIES.items():
                z_deg_region = spec["fn"](enc3_intact, spec["Ts"])[:, :, ez0:ez1, ey0:ey1, ex0:ex1]
                pert_mag[fam] = float((tile_mask_region - z_deg_region).norm())

            # Level 3: uncertainty (predictive entropy of the tile-local prediction)
            p = pr_local.clamp(1e-6, 1 - 1e-6)
            entropy = float((-(p * p.log() + (1 - p) * (1 - p).log())).mean())

            # Level 4: VJP sensitivity, requires its own forward+backward
            grad = vjp_sensitivity(model, image_b, z0, y0, x0, device)
            grad_region = grad[:, :, ez0:ez1, ey0:ey1, ex0:ex1]
            sensitivity = float(grad_region.norm())

            records.append({
                "sid": sid, "window_id": t["window_id"],
                "magnitude": magnitude, "entropy": entropy, "sensitivity": sensitivity,
                **{f"pert_mag_{fam}": v for fam, v in pert_mag.items()},
            })
        print(f"[{idx+1}/{n}] {sid}  n_tiles={len(tiles)}", flush=True)

    with open(OUT_DIR / "E180_7_competitor_features.json", "w") as f:
        json.dump(records, f, indent=1)
    print(f"\nSaved E180_7_competitor_features.json ({len(records)} records)")


if __name__ == "__main__":
    main()
