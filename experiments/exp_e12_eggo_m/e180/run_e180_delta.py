"""
E180 Stage 5 -- Delta counterfactual (corrected, same-operating-point design).

Pre-registered in docs/phases/PHASE_E180_5_DELTA_PREREG.md. Read that first;
the masked-restoration construction below is copied from it and must not
change after seeing results.

For each surviving family (T1_rank@r=2, T4_spectral@gamma=3.0, T5_smooth@sigma=1.5,
same T_s as Stage 3), construct Z^deg = T_s(Z) uniformly, then for each tile i
restore ONLY tile i's enc3 activation to its intact value (everywhere else
stays degraded), and measure the WHOLE-VOLUME Dice change against ground
truth. GT is used ONLY here, ONLY to evaluate -- never to construct Gamma or
to select which tile to restore.

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
from Dataset.brats_multimodal_dataset import create_multimodal_loaders, REGIONS  # noqa: E402
from run_e180_transform_lab import (  # noqa: E402
    lowrank_channels, spectral_reshape, local_smooth, PATCH, TARGET_STAGE,
)
from run_e180_gamma import (  # noqa: E402
    FAMILIES, ENC3_DOWNSAMPLE, enc3_tile_bounds, _gaussian_weight, N_SUBJECTS,
)

OUT_DIR = Path(__file__).parent
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694
OVERLAP = 0.25


class RestoreHook:
    """Forward hook on enc3: applies uniform degradation T_s always (Z^deg).
    If restore_mask is set, that region's activation is spliced back to the
    INTACT (pre-degradation) value: Z^cf_i = Z^deg + M_i*(Z - Z^deg)."""

    def __init__(self):
        self.family = None
        self.restore_mask = None   # torch bool tensor over enc3 spatial dims, or None

    def __call__(self, module, inputs, output):
        if self.family is None:
            return output
        z_intact = output
        z_deg = FAMILIES[self.family]["fn"](z_intact, FAMILIES[self.family]["Ts"])
        if self.restore_mask is None:
            return z_deg
        m = self.restore_mask.to(z_deg.dtype).view(1, 1, *self.restore_mask.shape)
        return z_deg * (1 - m) + z_intact * m


def register(model, hook):
    mod = getattr(model, TARGET_STAGE)
    return mod.register_forward_hook(lambda m, i, o: hook(m, i, o))


def sliding_window_collect(model, image, device, hook, n_out=3, amp=True, overlap=OVERLAP):
    """Per-window (batch_size=1) sliding-window inference. A batched variant was
    tried and REJECTED after measurement -- see run_e180_gamma.py's identical
    function docstring: OOM'd without AMP, ~11x SLOWER with AMP. Reverted."""
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
    ap.add_argument("--limit_subjects", type=int, default=N_SUBJECTS)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many val subjects before starting (match the ledger's offset)")
    ap.add_argument("--limit_tiles", type=int, default=0)
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

    ledger = json.load(open(OUT_DIR / f"E180_tile_ledger{a.ledger_suffix}.json"))

    model = UNet3D_v5(in_channels=4, out_channels=3).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hook = RestoreHook()
    register(model, hook)

    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    amp = not a.no_amp
    total = len(val_loader.dataset)
    n = min(a.limit_subjects, total - a.offset)
    print(f"[Data] measuring Delta on {n} subjects (offset={a.offset}), families={list(FAMILIES)}")

    records = []
    intact_dice_context = []
    for idx in range(n):
        i = a.offset + idx
        image, target, sid = val_loader.dataset[i]
        if sid not in ledger:
            print(f"  [skip] {sid} not in tile ledger")
            continue
        image_b = image.unsqueeze(0).to(device)
        tgt = target.numpy()
        tiles = ledger[sid]["tiles"]
        if a.limit_tiles:
            tiles = tiles[:a.limit_tiles]

        for fam in FAMILIES:
            hook.family, hook.restore_mask = fam, None
            p_deg = sliding_window_collect(model, image_b, device, hook, amp=amp)
            deg_bin = (p_deg > 0.5).astype(np.float32)
            dice_deg = dice_vs_gt(deg_bin, tgt)

            for t in tiles:
                ez0, ey0, ex0, ez1, ey1, ex1 = enc3_tile_bounds(
                    t["z0"], t["y0"], t["x0"], t["z1"], t["y1"], t["x1"])
                mask = torch.zeros((32, 32, 32), dtype=torch.bool, device=device)
                mask[ez0:ez1, ey0:ey1, ex0:ex1] = True

                hook.family, hook.restore_mask = fam, mask
                p_cf = sliding_window_collect(model, image_b, device, hook, amp=amp)
                cf_bin = (p_cf > 0.5).astype(np.float32)
                dice_cf = dice_vs_gt(cf_bin, tgt)

                records.append({
                    "sid": sid, "window_id": t["window_id"], "family": fam,
                    "dice_deg": dice_deg, "dice_cf_i": dice_cf,
                    "delta_i_global": dice_cf - dice_deg,
                })
        hook.family, hook.restore_mask = None, None
        intact_dice_context.append({"sid": sid, "dice_deg_by_family":
                                    {r["family"]: r["dice_deg"] for r in records
                                     if r["sid"] == sid}})
        print(f"[{idx+1}/{n}] {sid}  n_tiles={len(tiles)}  "
              f"records_so_far={len(records)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"E180_delta_per_subject{a.out_suffix}.json", "w") as f:
        json.dump(records, f, indent=1)

    deltas = np.array([r["delta_i_global"] for r in records])
    dice_deg_vals = np.array([r["dice_deg"] for r in records])
    summary = {
        "question": "Stage 5: whole-volume Dice benefit of restoring one tile's enc3 "
                    "activation from the same degraded operating point Gamma was measured from",
        "checkpoint": str(CKPT), "checkpoint_dice_verified": float(got),
        "n_subjects": n, "families": {k: v["Ts"] for k, v in FAMILIES.items()},
        "n_records": len(records),
        "delta_distribution": {
            "mean": float(deltas.mean()), "std": float(deltas.std()),
            "median": float(np.median(deltas)), "max": float(deltas.max()),
            "min": float(deltas.min()),
            "frac_positive": float((deltas > 1e-4).mean()),
            "frac_near_zero": float((np.abs(deltas) <= 1e-4).mean()),
        },
        "degradation_cost_context": {
            "dice_deg_mean": float(dice_deg_vals.mean()),
            "dice_deg_std": float(dice_deg_vals.std()),
            "checkpoint_intact_dice_reference": 0.8929357248544694,
            "note": "This is the FIRST stage where GT enters -- dice_deg vs the checkpoint's "
                    "known intact Dice is the first real cost check of T_s (Stage 2 only "
                    "measured prediction-vs-prediction displacement, not GT cost).",
        },
        "note": "GT used ONLY to score Delta. Gamma (Stage 3-4) never used GT.",
    }
    with open(OUT_DIR / f"E180_5_delta_summary{a.out_suffix}.json", "w") as f:
        json.dump(summary, f, indent=1)

    print("\n" + "=" * 70)
    print(json.dumps(summary, indent=1))
    print("=" * 70)


if __name__ == "__main__":
    main()
