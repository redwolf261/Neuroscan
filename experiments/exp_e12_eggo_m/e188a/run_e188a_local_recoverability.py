"""
E188-A -- Local latent recoverability map.

Pre-registered in docs/phases/PHASE_E188A_LOCAL_RECOVERABILITY_MAP_PREREG.md.
Read that first.

THE QUESTION: is E15 finding a structured region of better latent states, or
merely a favorable direction? Answered by comparing the E15 direction against
N=32 norm-matched random directions at each of eps in {4,8,12,14}, at dec1,
on the frozen current checkpoint.

CONSTRUCTION MATCHED VERBATIM TO e172/smoke_oracle_e15.py, which is the only
verified reproduction of E15 on THIS checkpoint, and whose oracle numbers
(+0.0139 at a8, peak; +0.0035 at a14, declining) are the reference this
experiment is calibrated against:
  - site        : dec1 (32ch, immediately before seg_head)
  - crop        : SINGLE centered 128^3 crop, NOT sliding-window
  - fg          : binary union of ET/TC/WT
  - direction   : PER-VOXEL unit, z - mu_opp normalized per voxel
  - alpha       : ABSOLUTE units, not fractions of ||z||
Changing any of these breaks comparability with the reference anchor.

THE STRICT NULL: random directions are PER-VOXEL unit random, scaled by the
same eps -- identical per-voxel displacement magnitude AND identical total
tensor norm to the E15 direction. Only the DIRECTION differs. This isolates
"is E15's choice of direction special" from "does moving every voxel by eps
help at all".

NO training, NO new network, NO loss, NO optimizer. Inference only.
GT is used to construct the E15 direction (inherited from E15's own design,
acknowledged oracle) and to score Dice. Random directions use no GT.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders  # noqa: E402

OUT_DIR = Path(__file__).parent
CKPT = (root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694

EPS_GRID = [4.0, 8.0, 12.0, 14.0]   # LOCKED; brackets the measured peak (8), 14 for continuity
N_RANDOM = 32                        # LOCKED
N_SUBJECTS = 10                      # LOCKED
RNG_SEED = 188_000                   # distinct from every prior probe seed in this project


def dice3(pred, gt):
    """Per-region Dice, same convention as e172/smoke_oracle_e15.py."""
    out = []
    for r in range(pred.shape[0]):
        p, t = pred[r], gt[r]
        ps, ts = p.sum(), t.sum()
        out.append(1.0 if ts == 0 and ps == 0 else
                   (0.0 if ts == 0 else float(2 * (p * t).sum() / (ps + ts))))
    return out


def get_dec1(model, image):
    """Run the model, capture dec1 via forward hook. Verbatim from E172."""
    cache = {}
    h = model.dec1.register_forward_hook(lambda mo, i, o: cache.__setitem__("z", o))
    with torch.no_grad():
        model(image)
    h.remove()
    return cache["z"].detach()


def score(model, z, gt_np):
    """Push z through the frozen seg_head, return mean per-region Dice."""
    with torch.no_grad():
        out = model.seg_head(z).float()
    pred = (out > 0.5).float().squeeze(0).cpu().numpy()
    return float(np.mean(dice3(pred, gt_np))), out


def e15_direction(z, tgt):
    """PER-VOXEL unit direction away from the same-volume opposite-class
    centroid. Binary fg = union of ET/TC/WT. Verbatim from E172's
    reproduction -- GT selects only WHICH centroid is opposite."""
    C = z.shape[1]
    fg = (tgt.sum(1, keepdim=True) > 0).float()          # (1,1,D,H,W)
    zf = z.reshape(1, C, -1)
    w = fg.reshape(1, 1, -1)
    nf = w.sum().clamp(min=1.0)
    nb = (1 - w).sum().clamp(min=1.0)
    mu_f = (zf * w).sum(2) / nf                          # (1,C)
    mu_b = (zf * (1 - w)).sum(2) / nb                    # (1,C)
    mu_opp = (w * mu_b.unsqueeze(2) + (1 - w) * mu_f.unsqueeze(2)).reshape_as(z)
    diff = z - mu_opp
    return diff / diff.norm(dim=1, keepdim=True).clamp(min=1e-8)


def random_direction(shape, device, generator):
    """PER-VOXEL unit random -- the strict null. Same per-voxel norm and same
    total tensor norm as the E15 direction; only the direction differs."""
    g = torch.randn(shape, device=device, generator=generator)
    return g / g.norm(dim=1, keepdim=True).clamp(min=1e-8)


def center_crop_128(image, target):
    """Single centered 128^3 crop, matching E172's reproduction exactly."""
    D, H, W = image.shape[1:]
    pad = [0, 0, 0, 0, 0, 0]
    for ax, sz in enumerate((D, H, W)):
        if sz < 128:
            pad[(2 - ax) * 2 + 1] = 128 - sz
    if any(pad):
        image = F.pad(image.unsqueeze(0), pad).squeeze(0)
        target = F.pad(target.unsqueeze(0), pad).squeeze(0)
    D, H, W = image.shape[1:]
    z0, y0, x0 = (D - 128) // 2, (H - 128) // 2, (W - 128) // 2
    return (image[:, z0:z0 + 128, y0:y0 + 128, x0:x0 + 128],
            target[:, z0:z0 + 128, y0:y0 + 128, x0:x0 + 128])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_subjects", type=int, default=N_SUBJECTS)
    ap.add_argument("--n_random", type=int, default=N_RANDOM)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    got = ck.get("best_mean_dice")
    assert got is not None and abs(float(got) - EXPECTED_DICE) < 1e-9, \
        f"Checkpoint identity FAILED: expected {EXPECTED_DICE}, got {got}"
    print(f"[Sanity] checkpoint identity PASS (best_mean_dice={got}).")

    model = UNet3D_v5(4, 3).to(dev)
    model.load_state_dict(ck["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    _, val = create_multimodal_loaders(
        root_dir=str(root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=(128, 128, 128), seed=0)

    gen = torch.Generator(device=dev).manual_seed(RNG_SEED)

    print(f"[Data] {a.limit_subjects} subjects, eps={EPS_GRID}, "
          f"N_random={a.n_random} per (subject,eps)")
    print("[Ref] E172 oracle on this checkpoint: a4=+0.0106 a8=+0.0139(peak) a14=+0.0035")

    records = []
    for i in range(a.limit_subjects):
        image, target, sid = val.dataset[i]
        image, target = center_crop_128(image, target)
        img = image.unsqueeze(0).to(dev)
        tgt = target.unsqueeze(0).to(dev)
        gt_np = tgt.squeeze(0).cpu().numpy()

        z = get_dec1(model, img)                      # (1,32,128,128,128)
        native_dice, native_out = score(model, z, gt_np)

        d_e15 = e15_direction(z, tgt)

        rec = {"sid": sid, "native_dice": native_dice, "eps": {}}
        for eps in EPS_GRID:
            z_e15 = z + eps * d_e15
            e15_dice, e15_out = score(model, z_e15, gt_np)
            e15_disp = float((e15_out - native_out).norm().item())
            e15_pnorm = float((eps * d_e15).norm().item())

            rand_dices, rand_disps, rand_pnorms = [], [], []
            for j in range(a.n_random):
                d_r = random_direction(z.shape, dev, gen)
                z_r = z + eps * d_r
                d_dice, r_out = score(model, z_r, gt_np)
                rand_dices.append(d_dice)
                rand_disps.append(float((r_out - native_out).norm().item()))
                rand_pnorms.append(float((eps * d_r).norm().item()))

            rand_dices = np.array(rand_dices)
            # percentile of E15 among the random draws (fraction it beats)
            pct = float((rand_dices < e15_dice).mean() * 100.0)
            rec["eps"][str(eps)] = {
                "e15_dice": e15_dice,
                "e15_delta": e15_dice - native_dice,
                "rand_mean": float(rand_dices.mean()),
                "rand_median": float(np.median(rand_dices)),
                "rand_max": float(rand_dices.max()),
                "rand_min": float(rand_dices.min()),
                "rand_std": float(rand_dices.std()),
                "e15_percentile": pct,
                "headroom_R": float(rand_dices.max() - native_dice),
                "e15_perturb_norm": e15_pnorm,
                "rand_perturb_norm_mean": float(np.mean(rand_pnorms)),
                "e15_output_disp": e15_disp,
                "rand_output_disp_mean": float(np.mean(rand_disps)),
                "n_random": int(a.n_random),
            }
            print(f"  [{sid}] eps={eps:4.1f}  native={native_dice:.4f}  "
                  f"E15={e15_dice:.4f} (d={e15_dice-native_dice:+.4f})  "
                  f"rand_mean={rand_dices.mean():.4f} rand_max={rand_dices.max():.4f}  "
                  f"E15_pct={pct:5.1f}  R={rand_dices.max()-native_dice:+.4f}", flush=True)

        records.append(rec)
        print(f"[{i+1}/{a.limit_subjects}] {sid} done", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "E188A_raw.json", "w") as f:
        json.dump({"eps_grid": EPS_GRID, "n_random": a.n_random,
                   "rng_seed": RNG_SEED, "records": records}, f, indent=1)
    print(f"\nSaved E188A_raw.json ({len(records)} subjects)")


if __name__ == "__main__":
    main()
