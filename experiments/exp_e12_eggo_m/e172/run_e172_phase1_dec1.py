"""
E172 Phase 1 -- POSITIVE CONTROL at dec1.

QUESTION: can a LABEL-FREE direction recover any of the causal steering that
E15's oracle demonstrates at this site?

WHY dec1 FIRST. The harness has been validated here and only here: the
per-voxel oracle reproduces E15's qualitative signature on this checkpoint
(0.8796 -> 0.8936 peak at alpha=8, then turning over; +1.40pp). So a negative
label-free result at dec1 is EVIDENCE, not a harness failure. That was not true
of the first implementation, which used enc1 + a globally broadcast direction +
fractional alpha -- three mismatches against E15's spec. Those numbers are void.

WHAT THIS PHASE IS NOT. It does not test E171's premise. E170 locates the
capacity deficit at enc1 (84% of subjects), not dec1. Phase 2 tests that site
separately, precisely so we cannot silently convert E171 from "repair the
capacity deficit at enc1" into "steer wherever steering happens to work."

THE INVIOLABLE RULE
    GT may EVALUATE a direction. GT may NOT CONSTRUCT one.
Enforced structurally: build_labelfree_directions() takes no target argument.
The oracle is built by a separate, explicitly-named function.

MATCHED NORM. Every arm is unit-normalised per voxel and applied at the same
absolute alpha grid, so arms are compared at identical perturbation energy.
Alpha grid is local to THIS checkpoint (peak near 8), not imported from E15.

PRIMARY ENDPOINT -- not Dice >= 1pp:
    Does a label-free direction produce SELECTIVE CORRECTIVE movement
    (repairs wrong voxels without breaking right ones)
    rather than generic prediction distortion?
"""
import sys
import json
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root))
from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders  # noqa: E402

OUT = Path(__file__).parent
CKPT = root / "experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth"
ALPHAS = [0, 1, 2, 4, 8, 14]          # local grid: oracle peaks near 8 here
N_SUBJ = 20                            # same smoke set as the oracle validation


def get_dec1(m, image):
    c = {}
    h = m.dec1.register_forward_hook(lambda mo, i, o: c.__setitem__("z", o))
    with torch.no_grad():
        m(image)
    h.remove()
    return c["z"].detach()


def _vjp_at_dec1(m, z, scalar_fn):
    """grad of a scalar functional of the OUTPUT w.r.t. dec1. seg_head only."""
    zz = z.clone().requires_grad_(True)
    out = m.seg_head(zz)
    scalar_fn(out).backward()
    g = zz.grad.detach().clone()
    del zz, out
    torch.cuda.empty_cache()
    return g


def unit_per_voxel(d):
    """Per-voxel unit normalisation -- matches E15's ||.|| over the channel axis."""
    return d / d.norm(dim=1, keepdim=True).clamp(min=1e-8)


def build_labelfree_directions(m, z, seed):
    """NO target argument. This signature is the GT firewall."""
    # d_jac: VJP of total predicted foreground mass (a functional of Y_hat only)
    d_jac = _vjp_at_dec1(m, z, lambda o: o.sum())

    # d_conf: VJP of prediction entropy -- the sharpening control
    def ent(o):
        p = o.clamp(1e-6, 1 - 1e-6)
        return -(p * p.log() + (1 - p) * (1 - p).log()).sum()
    d_conf = _vjp_at_dec1(m, z, ent)

    g = torch.Generator(device=z.device).manual_seed(seed)
    d_rand = torch.randn(z.shape, generator=g, device=z.device, dtype=z.dtype)

    return {"jac": unit_per_voxel(d_jac),
            "conf": unit_per_voxel(d_conf),
            "rand": unit_per_voxel(d_rand)}


def build_ORACLE_uses_GT(z, tgt):
    """E15 reference arm. USES GROUND TRUTH BY DESIGN. Never a candidate.

    z' = z + alpha * (z - mu_opp)/||z - mu_opp||, mu_opp per voxel.
    """
    C = z.shape[1]
    fg = (tgt.sum(1, keepdim=True) > 0).float()
    zf = z.reshape(1, C, -1)
    w = fg.reshape(1, 1, -1)
    mu_f = (zf * w).sum(2) / w.sum().clamp(min=1.0)
    mu_b = (zf * (1 - w)).sum(2) / (1 - w).sum().clamp(min=1.0)
    mu_opp = (w * mu_b.unsqueeze(2) + (1 - w) * mu_f.unsqueeze(2)).reshape_as(z)
    return unit_per_voxel(z - mu_opp)


def dice3(pred, gt):
    out = []
    for r in range(pred.shape[0]):
        p, t = pred[r], gt[r]
        ps, ts = p.sum(), t.sum()
        out.append(1.0 if ts == 0 and ps == 0 else
                   (0.0 if ts == 0 else float(2 * (p * t).sum() / (ps + ts))))
    return out


def main():
    dev = torch.device("cuda")
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    assert abs(ck["best_mean_dice"] - 0.8929357248544694) < 1e-9
    print("[Sanity] checkpoint identity PASS")
    m = UNet3D_v5(4, 3).to(dev)
    m.load_state_dict(ck["model_state"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)

    _, val = create_multimodal_loaders(root_dir=str(root / "Dataset" / "Training"),
                                       batch_size=1, num_workers=0, val_split=0.1,
                                       patch_size=(128, 128, 128), seed=0)
    recs = []
    for i in range(N_SUBJ):
        image, target, sid = val.dataset[i]
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
        img = image[:, z0:z0 + 128, y0:y0 + 128, x0:x0 + 128].unsqueeze(0).to(dev)
        tgt = target[:, z0:z0 + 128, y0:y0 + 128, x0:x0 + 128].unsqueeze(0).to(dev)

        z = get_dec1(m, img)
        # ---- LABEL-FREE ZONE ----
        dirs = build_labelfree_directions(m, z, seed=i)
        # ---- oracle, explicitly GT-using ----
        dirs["oracle_GT"] = build_ORACLE_uses_GT(z, tgt)

        with torch.no_grad():
            base = m.seg_head(z).float()
        base_bin = (base > 0.5).float()
        gt_np = tgt.squeeze(0).cpu().numpy()
        base_dice = float(np.mean(dice3(base_bin.squeeze(0).cpu().numpy(), gt_np)))
        wrong0 = (base_bin != tgt).float()
        right0 = 1.0 - wrong0

        rec = {"sid": sid, "base_dice": base_dice, "arms": {}}
        for name, d in dirs.items():
            arm = []
            for al in ALPHAS:
                if al == 0:
                    continue
                with torch.no_grad():
                    out = m.seg_head(z + al * d).float()
                    pb = (out > 0.5).float()
                    dc = float(np.mean(dice3(pb.squeeze(0).cpu().numpy(), gt_np)))
                    rep = float(((pb == tgt).float() * wrong0).sum() / wrong0.sum().clamp(min=1))
                    brk = float(((pb != tgt).float() * right0).sum() / right0.sum().clamp(min=1))
                    flip = float((pb != base_bin).float().mean())
                arm.append({"alpha": al, "dice": dc, "d_dice": dc - base_dice,
                            "repaired": rep, "broke": brk, "flipped": flip,
                            "selectivity": rep / max(brk, 1e-9)})
            rec["arms"][name] = arm
        recs.append(rec)
        o = rec["arms"]["oracle_GT"][3]   # alpha=8
        j = rec["arms"]["jac"][3]
        print(f"[{i+1}/{N_SUBJ}] {sid} base={base_dice:.4f} a=8: "
              f"oracle{o['d_dice']:+.4f} jac{j['d_dice']:+.4f}", flush=True)

    agg = {}
    for name in ["oracle_GT", "jac", "conf", "rand"]:
        agg[name] = {}
        for k, al in enumerate([a for a in ALPHAS if a > 0]):
            v = [r["arms"][name][k] for r in recs]
            agg[name][f"a{al}"] = {
                "mean_d_dice": float(np.mean([x["d_dice"] for x in v])),
                "frac_improved": float(np.mean([x["d_dice"] > 0 for x in v])),
                "mean_repaired": float(np.mean([x["repaired"] for x in v])),
                "mean_broke": float(np.mean([x["broke"] for x in v])),
                "mean_selectivity": float(np.median([x["selectivity"] for x in v])),
                "mean_flipped": float(np.mean([x["flipped"] for x in v])),
            }

    best = {n: max(agg[n].items(), key=lambda kv: kv[1]["mean_d_dice"]) for n in agg}
    oracle_best = best["oracle_GT"][1]["mean_d_dice"]
    jac_best = best["jac"][1]["mean_d_dice"]
    verdict = ("A_LABELFREE_STEERING_WORKS" if jac_best > 0.005
               else "B_LABELFREE_STEERING_FAILS_oracle_works"
               if oracle_best > 0.005 else "HARNESS_SUSPECT")

    summary = {
        "phase": "1 -- positive control at dec1",
        "site": "dec1 (E15's site; harness validated here)",
        "n_subjects": N_SUBJ, "alphas": ALPHAS,
        "harness_validation": "oracle reproduces E15 signature: 0.8796 -> 0.8936 peak a=8",
        "aggregate": agg,
        "best_alpha_per_arm": {n: {"alpha": k, **v} for n, (k, v) in best.items()},
        "PREREGISTERED_VERDICT": verdict,
        "primary_endpoint": ("selective corrective movement (repaired/broke), "
                             "NOT Dice >= 1pp"),
    }
    json.dump(recs, open(OUT / "E172_phase1_per_subject.json", "w"), indent=1)
    json.dump(summary, open(OUT / "E172_phase1_summary.json", "w"), indent=1)
    with open(OUT / "E172_phase1_arms.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "alpha", "mean_d_dice", "frac_improved", "repaired", "broke",
                    "selectivity", "flipped"])
        for n, d in agg.items():
            for al, v in d.items():
                w.writerow([n, al, round(v["mean_d_dice"], 5), round(v["frac_improved"], 3),
                            round(v["mean_repaired"], 5), round(v["mean_broke"], 5),
                            round(v["mean_selectivity"], 3), round(v["mean_flipped"], 5)])
    print("\n" + json.dumps(summary["best_alpha_per_arm"], indent=1))
    print("\nVERDICT:", verdict)


if __name__ == "__main__":
    main()
