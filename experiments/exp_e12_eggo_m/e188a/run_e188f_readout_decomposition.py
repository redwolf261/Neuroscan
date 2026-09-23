"""
E188-F -- Readout-vs-representation decomposition of E15.

Pre-registered in docs/phases/PHASE_E188F_READOUT_DECOMPOSITION_PREREG.md.
Read that first.

THE QUESTION: does E15 contain information beyond what the frozen readout
already explains? E188-A showed E15's direction is not arbitrary (100th
percentile in 38/40 cells) but that no local neighborhood headroom exists
(R<0 in 40/40). So the live hypothesis is that E15 is simply steering the
readout.

seg_head is Conv3d(32,3,k=1) -- RANK 3, not rank 1 (rows ET/TC/WT have
pairwise cosines 0.67-0.85; singular values [5.13,1.87,1.18], top direction
= 84.4% of energy). Projecting onto a single w would leak 15.6% of the
readout's linear action into the "orthogonal" arm. So we project onto the
FULL 3D row space:

    P_W = pinv(W) @ W          (32x32 orthogonal projector)
    d_par  = d_E @ P_W.T       (visible to readout)
    d_perp = d_E - d_par       (W @ d_perp == 0 EXACTLY -- verified at runtime)

Any Dice effect from d_perp provably cannot be a first-order logit shift.

Setup matched verbatim to E188-A / e172/smoke_oracle_e15.py: dec1 site,
single centered 128^3 crop, binary fg = union of ET/TC/WT, per-voxel
direction, absolute eps. NO training, NO optimizer, NO random sweep.
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
EPS_GRID = [4.0, 8.0, 12.0, 14.0]
N_SUBJECTS = 10
REGIONS = ("ET", "TC", "WT")


def dice3(pred, gt):
    out = []
    for r in range(pred.shape[0]):
        p, t = pred[r], gt[r]
        ps, ts = p.sum(), t.sum()
        out.append(1.0 if ts == 0 and ps == 0 else
                   (0.0 if ts == 0 else float(2 * (p * t).sum() / (ps + ts))))
    return out


def get_dec1(model, image):
    cache = {}
    h = model.dec1.register_forward_hook(lambda mo, i, o: cache.__setitem__("z", o))
    with torch.no_grad():
        model(image)
    h.remove()
    return cache["z"].detach()


def score(model, z, gt_np):
    with torch.no_grad():
        out = model.seg_head(z).float()
    pred = (out > 0.5).float().squeeze(0).cpu().numpy()
    return float(np.mean(dice3(pred, gt_np))), dice3(pred, gt_np)


def e15_direction(z, tgt):
    """Per-voxel unit direction away from the same-volume opposite-class
    centroid. Binary fg = union of ET/TC/WT. Verbatim from E188-A/E172."""
    C = z.shape[1]
    fg = (tgt.sum(1, keepdim=True) > 0).float()
    zf = z.reshape(1, C, -1)
    w = fg.reshape(1, 1, -1)
    nf = w.sum().clamp(min=1.0)
    nb = (1 - w).sum().clamp(min=1.0)
    mu_f = (zf * w).sum(2) / nf
    mu_b = (zf * (1 - w)).sum(2) / nb
    mu_opp = (w * mu_b.unsqueeze(2) + (1 - w) * mu_f.unsqueeze(2)).reshape_as(z)
    diff = z - mu_opp
    return diff / diff.norm(dim=1, keepdim=True).clamp(min=1e-8)


def project_rowspace(delta, P):
    """delta: (1,C,D,H,W); P: (C,C) projector onto W's row space."""
    C = delta.shape[1]
    flat = delta.reshape(1, C, -1)                 # (1,C,N)
    par = torch.einsum("ij,bjn->bin", P, flat)     # (1,C,N)
    return par.reshape_as(delta)


def logit_disp(W, delta):
    """Mean |W @ delta| per region -- the first-order readout displacement."""
    C = delta.shape[1]
    flat = delta.reshape(C, -1)                    # (C,N)
    dl = W @ flat                                  # (3,N)
    return [float(dl[r].abs().mean().item()) for r in range(dl.shape[0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_subjects", type=int, default=N_SUBJECTS)
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

    W = model.seg_head[0].weight.detach().reshape(3, 32)          # (3,32)
    P = torch.linalg.pinv(W) @ W                                   # (32,32) projector
    # verify the projector is idempotent and that W kills the complement
    idem = float((P @ P - P).abs().max().item())
    print(f"[Check] ||P@P - P||_max = {idem:.3e} (projector idempotent)")
    s = torch.linalg.svdvals(W)
    print(f"[Check] W singular values: {[round(float(x),4) for x in s]}  "
          f"top-dir energy={float(s[0]**2/(s**2).sum()):.4f}")

    _, val = create_multimodal_loaders(
        root_dir=str(root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=(128, 128, 128), seed=0)

    records = []
    for i in range(a.limit_subjects):
        image, target, sid = val.dataset[i]
        D, H, Wd = image.shape[1:]
        pad = [0, 0, 0, 0, 0, 0]
        for ax, sz in enumerate((D, H, Wd)):
            if sz < 128:
                pad[(2 - ax) * 2 + 1] = 128 - sz
        if any(pad):
            image = F.pad(image.unsqueeze(0), pad).squeeze(0)
            target = F.pad(target.unsqueeze(0), pad).squeeze(0)
        D, H, Wd = image.shape[1:]
        z0, y0, x0 = (D - 128) // 2, (H - 128) // 2, (Wd - 128) // 2
        img = image[:, z0:z0 + 128, y0:y0 + 128, x0:x0 + 128].unsqueeze(0).to(dev)
        tgt = target[:, z0:z0 + 128, y0:y0 + 128, x0:x0 + 128].unsqueeze(0).to(dev)
        gt_np = tgt.squeeze(0).cpu().numpy()

        z = get_dec1(model, img)
        native, native_pr = score(model, z, gt_np)
        u = e15_direction(z, tgt)          # per-voxel unit

        rec = {"sid": sid, "native_dice": native, "native_per_region": native_pr, "eps": {}}
        for eps in EPS_GRID:
            d_E = eps * u
            d_par = project_rowspace(d_E, P)
            d_perp = d_E - d_par

            # HARD CHECK: the orthogonal component must be invisible to W
            leak = float((W @ d_perp.reshape(32, -1)).abs().max().item())
            scale = float((W @ d_E.reshape(32, -1)).abs().max().item())

            n_E = float(d_E.norm().item())
            n_par = float(d_par.norm().item())
            n_perp = float(d_perp.norm().item())

            arms = {
                "full":        z + d_E,
                "par_nat":     z + d_par,
                "perp_nat":    z + d_perp,
                "par_renorm":  z + d_par * (n_E / max(n_par, 1e-8)),
                "perp_renorm": z + d_perp * (n_E / max(n_perp, 1e-8)),
                "reversed":    z - d_E,
            }
            cell = {
                "norm_full": n_E, "norm_par": n_par, "norm_perp": n_perp,
                "frac_energy_par": float((n_par ** 2) / max(n_E ** 2, 1e-12)),
                "readout_leak_perp_max": leak, "readout_scale_full_max": scale,
                "logit_disp_full": logit_disp(W, d_E),
                "logit_disp_par": logit_disp(W, d_par),
                "logit_disp_perp": logit_disp(W, d_perp),
                "arms": {},
            }
            for name, zz in arms.items():
                dsc, per_r = score(model, zz, gt_np)
                cell["arms"][name] = {"dice": dsc, "delta": dsc - native,
                                      "per_region": per_r}
            rec["eps"][str(eps)] = cell

            A = cell["arms"]
            print(f"  [{sid}] eps={eps:4.1f} nat={native:.4f} | "
                  f"full={A['full']['delta']:+.4f} par={A['par_nat']['delta']:+.4f} "
                  f"perp={A['perp_nat']['delta']:+.4f} | "
                  f"parR={A['par_renorm']['delta']:+.4f} perpR={A['perp_renorm']['delta']:+.4f} "
                  f"rev={A['reversed']['delta']:+.4f} | "
                  f"E_par={cell['frac_energy_par']:.3f} leak={leak:.2e}", flush=True)

        records.append(rec)
        print(f"[{i+1}/{a.limit_subjects}] {sid} done", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "E188F_raw.json", "w") as f:
        json.dump({"eps_grid": EPS_GRID, "records": records,
                   "W_singular_values": [float(x) for x in s]}, f, indent=1)
    print(f"\nSaved E188F_raw.json ({len(records)} subjects)")


if __name__ == "__main__":
    main()
