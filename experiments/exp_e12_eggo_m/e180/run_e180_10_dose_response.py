"""
E180 Stage 10 -- T4_spectral dose-response measurement.

Pre-registered in docs/phases/PHASE_E180_10_11_T4_DOSE_RESPONSE_PREREG.md.
Read that first. T4_spectral ONLY (T1_rank killed at Stage 6, T5_smooth held/
mixed -- not advanced). Transform, probe eps, subject set, and tile-selection
criterion are LOCKED, not re-tuned.

For each tile i, constructs graded restoration doses
    Z_i^(k) = Z^deg + alpha_k * (Z_intact - Z^deg),  alpha in {0,.25,.5,.75,1.0}
and measures, at each dose:
  - Delta_i^(k): whole-volume Dice change vs GT (extends Stage 5; alpha=0 and
    alpha=1.0 reproduce Stage 5's existing Delta_i^(0)=0 and Delta_i exactly)
  - Gamma_i^(k): the SAME fixed-probe instability measurement from Stage 3-4,
    applied at each dose (does the tile's own instability shrink as it is
    restored?)

NO training, NO architecture change, ONE checkpoint. Inference only. GT used
ONLY to score Delta, never to construct Gamma or select doses.
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
from run_e180_transform_lab import PATCH, TARGET_STAGE  # noqa: E402
import run_e180_gamma as gamma_mod  # noqa: E402
from run_e180_gamma import (  # noqa: E402
    FAMILIES, ENC3_DOWNSAMPLE, enc3_tile_bounds, _gaussian_weight,
    fixed_direction, _load_probe_eps, dice_agree,
)

OUT_DIR = Path(__file__).parent
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694
OVERLAP = 0.25
FAM = "T4_spectral"          # LOCKED -- only T4 advances to Stage 10-11
ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.0]


class DoseHook:
    """Forward hook on enc3: Z^deg = T_s(Z_intact) always. Within tile i's
    mask, blends toward the intact value by dose alpha:
        Z_i^(alpha) = Z^deg + alpha*(Z_intact - Z^deg)
    Outside the mask, always Z^deg (matches Delta's construction). If
    probe_mask/probe_direction are also set, ADDS the fixed Stage 3-4 probe
    perturbation on top (for measuring Gamma^(k) at this dose) -- mutually
    exclusive per-call with pure dose measurement, selected by which fields
    are set when the hook fires."""

    def __init__(self, direction):
        self.dose_mask = None
        self.alpha = None
        self.probe_mask = None       # additional Stage-3-style probe mask, or None
        self.direction = direction

    def __call__(self, module, inputs, output):
        z_intact = output
        z_deg = FAMILIES[FAM]["fn"](z_intact, FAMILIES[FAM]["Ts"])
        if self.dose_mask is None:
            z = z_deg
        else:
            m = self.dose_mask.to(z_deg.dtype).view(1, 1, *self.dose_mask.shape)
            z = z_deg * (1 - m) + (z_deg + self.alpha * (z_intact - z_deg)) * m
        if self.probe_mask is not None:
            eps = gamma_mod.PROBE_EPS[FAM]
            std = z.std()
            perturb = (eps * std) * self.direction.view(1, -1, 1, 1, 1)
            pm = self.probe_mask.to(z.dtype).view(1, 1, *self.probe_mask.shape)
            z = z + perturb * pm
        return z


def register(model, hook):
    mod = getattr(model, TARGET_STAGE)
    return mod.register_forward_hook(lambda m, i, o: hook(m, i, o))


def sliding_window_collect(model, image, device, hook, n_out=3, amp=True, overlap=OVERLAP):
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


def dice_vs_gt(pred_bin, target_bin):
    out = []
    for r in range(pred_bin.shape[0]):
        p, t = pred_bin[r], target_bin[r]
        ps, ts = p.sum(), t.sum()
        out.append(1.0 if ps == 0 and ts == 0 else float(2.0 * (p * t).sum() / (ps + ts)))
    return float(np.mean(out))


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

    _load_probe_eps()
    print(f"[Probe] eps for {FAM} (locked from Stage 2b): {gamma_mod.PROBE_EPS[FAM]}")

    ledger = json.load(open(OUT_DIR / "E180_tile_ledger_full.json"))

    model = UNet3D_v5(in_channels=4, out_channels=3).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    direction = fixed_direction(device)
    hook = DoseHook(direction)
    register(model, hook)

    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    amp = not a.no_amp
    total = len(val_loader.dataset)
    n = min(a.limit_subjects, total - a.offset)
    print(f"[Data] dose-response for {n} subjects (offset={a.offset}), family={FAM}, "
          f"alphas={ALPHAS}")

    records = []
    for idx in range(n):
        i = a.offset + idx
        image, target, sid = val_loader.dataset[i]
        if sid not in ledger:
            continue
        image_b = image.unsqueeze(0).to(device)
        tgt = target.numpy()
        tiles = ledger[sid]["tiles"]

        for t in tiles:
            ez0, ey0, ex0, ez1, ey1, ex1 = enc3_tile_bounds(
                t["z0"], t["y0"], t["x0"], t["z1"], t["y1"], t["x1"])
            dose_mask = torch.zeros((32, 32, 32), dtype=torch.bool, device=device)
            dose_mask[ez0:ez1, ey0:ey1, ex0:ex1] = True

            for alpha in ALPHAS:
                # ---- Delta^(k): dose-restored prediction vs GT ----
                hook.dose_mask, hook.alpha, hook.probe_mask = dose_mask, alpha, None
                p_dose = sliding_window_collect(model, image_b, device, hook, amp=amp)
                dose_bin = (p_dose > 0.5).astype(np.float32)
                dice_dose = dice_vs_gt(dose_bin, tgt)

                # ---- Gamma^(k): instability probe applied ON TOP of this dose ----
                hook.dose_mask, hook.alpha, hook.probe_mask = dose_mask, alpha, dose_mask
                p_probe = sliding_window_collect(model, image_b, device, hook, amp=amp)
                probe_bin = (p_probe > 0.5).astype(np.float32)
                gamma_dose = 1.0 - dice_agree(probe_bin, dose_bin)

                records.append({
                    "sid": sid, "window_id": t["window_id"], "alpha": alpha,
                    "dice_at_dose": dice_dose, "gamma_at_dose": gamma_dose,
                })
        hook.dose_mask, hook.alpha, hook.probe_mask = None, None, None
        print(f"[{idx+1}/{n}] {sid}  n_tiles={len(tiles)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "E180_10_dose_response.json", "w") as f:
        json.dump(records, f, indent=1)

    # dice_at_dose(alpha=0) is the dice_deg baseline; delta_i^(k) = dice(alpha=k) - dice(alpha=0)
    by_tile = {}
    for r in records:
        by_tile.setdefault((r["sid"], r["window_id"]), {})[r["alpha"]] = r
    n_tiles = len(by_tile)
    print(f"\nSaved E180_10_dose_response.json ({len(records)} records, {n_tiles} tiles x "
          f"{len(ALPHAS)} doses)")


if __name__ == "__main__":
    main()
