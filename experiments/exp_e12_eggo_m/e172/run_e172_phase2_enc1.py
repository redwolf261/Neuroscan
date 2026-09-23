"""
E172 Phase 2 -- the E171 PREMISE TEST, at enc1.

This is NOT a continuation of Phase 1. Phase 1 established a clean negative for
the dec1 route: the oracle steers (+1.39pp at alpha=8, 85% of subjects improve)
while every model-derived direction fails (jac +0.05pp at best, collapsing to
-0.79 at the alpha where the oracle peaks; conf improves 0% of subjects).
Conclusion: E15 cannot be converted into a label-free DECODER-steering method.

Phase 2 asks a different question, at E171's actual intervention site:

  H1: there exists d_enc1(X, Y_hat) producing SELECTIVE CORRECTIVE change
  H0: model-derived enc1 directions produce no correction beyond matched random

Why enc1 is not prejudged by Phase 1: at dec1 the readout is rank-1
(grad_z D = D(1-D) w), so every functional yields +-w and only GT supplies the
sign. At enc1 the map runs through the whole decoder and is NOT rank-1 --
measured cosines between VJP directions: cos(ET,TC)=+0.9858 but
cos(ET,entropy)=+0.1556, cos(TC,entropy)=+0.2701.

FIVE CONTROLS (Dice is NOT the only gate):
  1. nontrivial output sensitivity     ||dY|| > 0
  2. direction specificity             dY(d_cand) != dY(d_rand)
  3. correction specificity            repaired / broke / dDice, GT AFTER construction
  4. norm matching                     per-voxel unit, identical alpha grid
  5. SIGN SYMMETRY                     test +d AND -d
     Phase 1 showed the failing ingredient is the SIGN, not the magnitude.
     If +d and -d are equally (un)helpful, the direction carries no corrective
     orientation and the route is dead regardless of magnitude tuning.

HARD STOP, fixed in advance: if candidates approximate random, or predominantly
damage already-correct voxels, KILL the label-free steering route. Do NOT
respond by adding more direction constructions.

GT FIREWALL: build_labelfree_directions_enc1() takes no target argument.
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
N_SUBJ = 20
# enc1 activations have a different scale from dec1, so alpha is a FRACTION of
# the per-voxel norm here. Grid spans 3 orders of magnitude to locate any peak.
ALPHAS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
SIGNS = [+1, -1]


def rest_from_enc1(m, z1):
    """enc1 -> seg logits. Manual unroll; attention gate reads the same
    (possibly perturbed) enc1, matching the E48 convention used in e160/e165."""
    p1 = m.pool1(z1)
    e2 = m.enc2(p1)
    p2 = m.pool2(e2)
    e3 = m.enc3(p2)
    p3 = m.pool3(e3)
    b = m.bottleneck(p3)
    u3 = m.upconv3(b)
    d3 = m.dec3(torch.cat([u3, e3], 1))
    u2 = m.upconv2(d3)
    d2 = m.dec2(torch.cat([u2, e2], 1))
    u1 = m.upconv1(d2)
    g = m.attn_gate1.W_g(b)
    gu = F.interpolate(g, size=z1.shape[2:], mode="trilinear", align_corners=False)
    psi = torch.sigmoid(m.attn_gate1.W_psi(F.relu(gu + m.attn_gate1.W_x(z1))))
    return m.seg_head(m.dec1(torch.cat([u1, z1 * psi], 1)))


def get_enc1(m, image):
    c = {}
    h = m.enc1.register_forward_hook(lambda mo, i, o: c.__setitem__("z", o))
    with torch.no_grad():
        m(image)
    h.remove()
    return c["z"].detach()


def _vjp(m, z, fn):
    zz = z.clone().requires_grad_(True)
    out = rest_from_enc1(m, zz)
    fn(out).backward()
    g = zz.grad.detach().clone()
    del zz, out
    torch.cuda.empty_cache()
    return g


def unit_pv(d):
    return d / d.norm(dim=1, keepdim=True).clamp(min=1e-8)


def build_labelfree_directions_enc1(m, z, seed):
    """NO target argument -- the GT firewall."""
    d_jac = _vjp(m, z, lambda o: o.sum())          # predicted foreground mass

    def ent(o):
        p = o.clamp(1e-6, 1 - 1e-6)
        return -(p * p.log() + (1 - p) * (1 - p).log()).sum()
    d_conf = _vjp(m, z, ent)

    g = torch.Generator(device=z.device).manual_seed(seed)
    d_rand = torch.randn(z.shape, generator=g, device=z.device, dtype=z.dtype)
    return {"jac": unit_pv(d_jac), "conf": unit_pv(d_conf), "rand": unit_pv(d_rand)}


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

        z = get_enc1(m, img)
        dirs = build_labelfree_directions_enc1(m, z, seed=i)     # label-free zone
        zpv = z.norm(dim=1, keepdim=True)                        # per-voxel scale

        with torch.no_grad():
            base = rest_from_enc1(m, z).float()
        base_bin = (base > 0.5).float()
        gt_np = tgt.squeeze(0).cpu().numpy()
        base_dice = float(np.mean(dice3(base_bin.squeeze(0).cpu().numpy(), gt_np)))
        wrong0 = (base_bin != tgt).float()
        right0 = 1.0 - wrong0

        rec = {"sid": sid, "base_dice": base_dice, "arms": {}}
        for name, d in dirs.items():
            for sg in SIGNS:
                key = f"{name}{'+' if sg > 0 else '-'}"
                arm = []
                for al in ALPHAS:
                    if al == 0.0:
                        continue
                    with torch.no_grad():
                        out = rest_from_enc1(m, z + sg * al * zpv * d).float()
                        pb = (out > 0.5).float()
                        dc = float(np.mean(dice3(pb.squeeze(0).cpu().numpy(), gt_np)))
                        rep = float(((pb == tgt).float() * wrong0).sum()
                                    / wrong0.sum().clamp(min=1))
                        brk = float(((pb != tgt).float() * right0).sum()
                                    / right0.sum().clamp(min=1))
                        dY = float((out - base).abs().mean())
                        flip = float((pb != base_bin).float().mean())
                    arm.append({"alpha": al, "dice": dc, "d_dice": dc - base_dice,
                                "repaired": rep, "broke": brk,
                                "dY": dY, "flipped": flip})
                rec["arms"][key] = arm
        recs.append(rec)
        jp = rec["arms"]["jac+"][2]
        jm = rec["arms"]["jac-"][2]
        print(f"[{i+1}/{N_SUBJ}] {sid} base={base_dice:.4f} a=0.1: "
              f"jac+{jp['d_dice']:+.4f} jac-{jm['d_dice']:+.4f}", flush=True)

    keys = [f"{n}{s}" for n in ["jac", "conf", "rand"] for s in ["+", "-"]]
    agg = {}
    for name in keys:
        agg[name] = {}
        for k, al in enumerate([a for a in ALPHAS if a > 0]):
            v = [r["arms"][name][k] for r in recs]
            agg[name][f"a{al}"] = {
                "mean_d_dice": float(np.mean([x["d_dice"] for x in v])),
                "frac_improved": float(np.mean([x["d_dice"] > 0 for x in v])),
                "mean_repaired": float(np.mean([x["repaired"] for x in v])),
                "mean_broke": float(np.mean([x["broke"] for x in v])),
                "mean_dY": float(np.mean([x["dY"] for x in v])),
                "mean_flipped": float(np.mean([x["flipped"] for x in v])),
            }

    from scipy import stats as st
    # control 2 + 3 at the alpha where |dY| is comparable: use a=0.1 (index 2)
    k = 2
    jac_p = np.array([r["arms"]["jac+"][k]["d_dice"] for r in recs])
    jac_m = np.array([r["arms"]["jac-"][k]["d_dice"] for r in recs])
    rnd_p = np.array([r["arms"]["rand+"][k]["d_dice"] for r in recs])
    best_jac = max(jac_p.mean(), jac_m.mean())
    p_vs_rand = float(st.wilcoxon(np.maximum(jac_p, jac_m), rnd_p).pvalue)
    sign_asym = abs(jac_p.mean() - jac_m.mean())

    if best_jac > 0.005 and p_vs_rand < 0.05 and sign_asym > 0.002:
        verdict = "H1_SUPPORTED_labelfree_enc1_direction_exists"
    else:
        verdict = "H0_KILL_labelfree_steering_route"

    summary = {
        "phase": "2 -- E171 premise test at enc1",
        "hypothesis": "exists d_enc1(X,Y_hat) producing selective corrective change",
        "why_not_prejudged_by_phase1": ("dec1 readout is rank-1 so only GT supplies the sign; "
                                        "enc1 map is NOT rank-1 (cos to entropy 0.16/0.27)"),
        "n_subjects": N_SUBJ, "alphas_frac_of_per_voxel_norm": ALPHAS,
        "controls": ["output sensitivity", "direction specificity vs random",
                     "correction specificity", "norm matching", "SIGN SYMMETRY (+d and -d)"],
        "aggregate": agg,
        "test_at_alpha_0.1": {
            "jac_plus_mean_d_dice": float(jac_p.mean()),
            "jac_minus_mean_d_dice": float(jac_m.mean()),
            "rand_plus_mean_d_dice": float(rnd_p.mean()),
            "sign_asymmetry": float(sign_asym),
            "wilcoxon_bestjac_vs_rand_p": p_vs_rand,
        },
        "phase1_reference": {"oracle_dec1_best": 0.01392, "jac_dec1_best": 0.00048},
        "PREREGISTERED_VERDICT": verdict,
        "hard_stop": ("if candidates ~ random or damage correct voxels, KILL the route; "
                      "do NOT add more direction constructions"),
    }
    json.dump(recs, open(OUT / "E172_phase2_per_subject.json", "w"), indent=1)
    json.dump(summary, open(OUT / "E172_phase2_summary.json", "w"), indent=1)
    with open(OUT / "E172_phase2_arms.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "alpha", "mean_d_dice", "frac_improved", "repaired",
                    "broke", "mean_dY", "flipped"])
        for n, d in agg.items():
            for al, v in d.items():
                w.writerow([n, al, round(v["mean_d_dice"], 5), round(v["frac_improved"], 3),
                            round(v["mean_repaired"], 5), round(v["mean_broke"], 5),
                            round(v["mean_dY"], 6), round(v["mean_flipped"], 5)])
    print("\n" + json.dumps(summary["test_at_alpha_0.1"], indent=1))
    print("\nVERDICT:", verdict)


if __name__ == "__main__":
    main()
