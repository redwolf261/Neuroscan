"""
E172 -- Label-Free Corrective Direction Test.

PRE-REGISTERED in docs/phases/PHASE_E172_LABEL_FREE_DIRECTION_PREREG.md.

QUESTION: does there exist d_l(X, Y_hat) -- constructed WITHOUT ground truth --
such that z_enc1 + alpha*d corrects segmentation errors?

HARD REQUIREMENT (from E133): when the model is confidently wrong, a useful
direction must CHANGE the prediction, not merely sharpen it. 5/8 ET failures
predict exactly zero voxels at max probability 0.0000. Sharpening zero is zero.

THE INVIOLABLE RULE
    GT may EVALUATE d_l.  GT may NOT CONSTRUCT d_l.

Enforced STRUCTURALLY here, not by convention: build_directions() never receives
the target tensor. It is called with (model, image) only; the target stays in
the caller's scope and is first touched in score_against_gt(), after every
direction and every perturbed prediction already exists. d_GT is the single
exception -- an explicit oracle REFERENCE arm, never a candidate -- and it is
built in a separate function whose name says so.

PRE-MEASURED GEOMETRY (verified before writing this):
    cos(d_ET, d_TC)      = +0.9858   region selectivity is ~rank-1
    cos(d_ET, d_entropy) = +0.1556   mass-change and confidence-change are
    cos(d_TC, d_entropy) = +0.2701   near-orthogonal => NOT globally rank-1
So the experiment is well posed: d_conf is a real control, not a strawman.

PRE-REGISTERED DECISION RULE (fixed before running):
  A  d_Jac raises Dice significantly above BOTH d_rand and d_conf at matched
     norm, AND the gain concentrates in initially-WRONG voxels
     -> label-free corrective steering exists; E171 proceeds to rank unification
  B  d_Jac indistinguishable from d_rand, or moves predictions without
     preferentially repairing errors
     -> E171 DIES CLEANLY; E15 steering requires oracle information
  C  d_conf raises confidence while confident errors remain errors
     -> confirms the E133 objection experimentally (reported regardless of A/B)

PRE-RECORDED EXPECTATION (so results cannot be read as confirmation):
  d_conf sharpens without correcting -- outcome C near-certain.
  d_Jac genuinely uncertain: orthogonality to entropy says it is a DIFFERENT
  operation, but different does not imply corrective.
"""
import sys
import json
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders, REGIONS  # noqa: E402

OUT_DIR = Path(__file__).parent
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694
PATCH = (128, 128, 128)

# matched-norm sweep: alpha is a FRACTION of ||z_enc1||, so all directions are
# compared at identical perturbation energy. Chosen before seeing any result.
ALPHAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]


def rest_of_net(m, z1):
    """enc1 output -> seg probabilities. Manual unroll of UNet3D_v5.forward().

    Mirrors e160/e165 exactly, including the attention gate reading the same
    (possibly perturbed) enc1 -- the E48 convention.
    """
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
    z = {}
    h = m.enc1.register_forward_hook(lambda mod, i, o: z.__setitem__("z", o))
    with torch.no_grad():
        m(image)
    h.remove()
    return z["z"].detach()


def _vjp(m, z1, scalar_fn):
    """grad of a scalar functional of the OUTPUT w.r.t. enc1. One backward."""
    zz = z1.clone().requires_grad_(True)
    out = rest_of_net(m, zz)
    scalar_fn(out).backward()
    g = zz.grad.detach().clone()
    del zz, out
    torch.cuda.empty_cache()
    return g


def build_directions(m, image, seed):
    """LABEL-FREE directions. Deliberately takes NO target argument.

    This signature is the GT firewall: the target tensor is not in scope, so it
    cannot enter the VJP, the construction, or the normalisation.
    """
    z1 = get_enc1(m, image)

    # d_Jac: VJP of total predicted foreground mass -- a functional of Y_hat only
    d_jac = _vjp(m, z1, lambda o: o.sum())

    # d_conf: VJP of prediction entropy -- the sharpening control
    def ent(o):
        p = o.clamp(1e-6, 1 - 1e-6)
        return -(p * p.log() + (1 - p) * (1 - p).log()).sum()
    d_conf = _vjp(m, z1, ent)

    # d_rand: matched-norm Gaussian noise floor
    gen = torch.Generator(device=z1.device).manual_seed(seed)
    d_rand = torch.randn(z1.shape, generator=gen, device=z1.device, dtype=z1.dtype)

    def unit(d):
        n = d.norm()
        return d / n.clamp(min=1e-12)

    return z1, {"jac": unit(d_jac), "conf": unit(d_conf), "rand": unit(d_rand)}


def build_ORACLE_direction_uses_GT(m, z1, target):
    """E15-style reference arm. USES GROUND TRUTH BY DESIGN.

    Never a candidate. Establishes the scale a deployable direction would have
    to reach. Named so it cannot be confused with build_directions().
    """
    C = z1.shape[1]
    fg = (target.sum(1, keepdim=True) > 0).float()            # GT foreground
    fg_d = F.interpolate(fg, size=z1.shape[2:], mode="trilinear", align_corners=False)
    m_fg = (fg_d > 0.5).float()
    zf = z1.reshape(1, C, -1)
    w = m_fg.reshape(1, 1, -1)
    n_f = w.sum().clamp(min=1.0)
    n_b = (1 - w).sum().clamp(min=1.0)
    mu_f = (zf * w).sum(2) / n_f
    mu_b = (zf * (1 - w)).sum(2) / n_b
    d = (mu_f - mu_b).reshape(1, C, 1, 1, 1).expand_as(z1).contiguous()
    return d / d.norm().clamp(min=1e-12)


def dice_per_region(pred, gt):
    out = []
    for r in range(pred.shape[0]):
        p, t = pred[r], gt[r]
        ps, ts = p.sum(), t.sum()
        out.append(1.0 if ts == 0 and ps == 0 else
                   (0.0 if ts == 0 else float(2 * (p * t).sum() / (ps + ts))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    got = ck.get("best_mean_dice")
    assert got is not None and abs(float(got) - EXPECTED_DICE) < 1e-9, \
        f"checkpoint identity FAILED: {got}"
    print(f"[Sanity] checkpoint identity PASS ({got})")

    m = UNet3D_v5(in_channels=4, out_channels=3).to(dev)
    m.load_state_dict(ck["model_state"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)

    _, val = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    n = len(val.dataset) if not a.limit else min(a.limit, len(val.dataset))
    records = []

    for i in range(n):
        image, target, sid = val.dataset[i]
        # centre patch: VJP through a full 240^3 volume does not fit in 8GB
        D, H, W = image.shape[1:]
        # centre crop to 128^3; PAD first if any axis is short (some BraTS
        # volumes are thinner than 128 in z -- caught in smoke test).
        img_t, tgt_t = image, target
        pad = [0, 0, 0, 0, 0, 0]
        for ax, sz in enumerate((D, H, W)):
            if sz < 128:
                pad[(2 - ax) * 2 + 1] = 128 - sz
        if any(pad):
            img_t = F.pad(img_t.unsqueeze(0), pad).squeeze(0)
            tgt_t = F.pad(tgt_t.unsqueeze(0), pad).squeeze(0)
        D, H, W = img_t.shape[1:]
        z0, y0, x0 = (D - 128) // 2, (H - 128) // 2, (W - 128) // 2
        img = img_t[:, z0:z0 + 128, y0:y0 + 128, x0:x0 + 128].unsqueeze(0).to(dev)
        tgt = tgt_t[:, z0:z0 + 128, y0:y0 + 128, x0:x0 + 128].unsqueeze(0).to(dev)

        # ---- LABEL-FREE ZONE: target is not passed in ----
        z1, dirs = build_directions(m, img, seed=a.seed + i)
        # ---- oracle reference arm, explicitly GT-using ----
        dirs["gt_ORACLE"] = build_ORACLE_direction_uses_GT(m, z1, tgt)

        zn = float(z1.norm())
        with torch.no_grad():
            base = torch.sigmoid(rest_of_net(m, z1)) if False else rest_of_net(m, z1)
            base = base.float()
        base_bin = (base > 0.5).float()
        gt_np = tgt.squeeze(0).cpu().numpy()
        base_dice = dice_per_region(base_bin.squeeze(0).cpu().numpy(), gt_np)
        wrong0 = (base_bin != tgt).float()          # GT used for SCORING only

        rec = {"sid": sid, "base_dice": base_dice,
               "base_dice_mean": float(np.mean(base_dice)),
               "z_norm": zn, "arms": {}}

        for name, d in dirs.items():
            arm = []
            for al in ALPHAS:
                if al == 0.0:
                    continue
                with torch.no_grad():
                    zp = z1 + (al * zn) * d
                    out = rest_of_net(m, zp).float()
                    pb = (out > 0.5).float()
                    dice = dice_per_region(pb.squeeze(0).cpu().numpy(), gt_np)
                    flipped = float((pb != base_bin).float().mean())
                    dmass = float(pb.sum() - base_bin.sum())
                    dlogit = float((out - base).abs().mean())
                    # repair: of voxels initially WRONG, how many became right?
                    now_right = ((pb == tgt).float() * wrong0).sum()
                    repaired = float(now_right / wrong0.sum().clamp(min=1))
                    # damage: of voxels initially RIGHT, how many broke?
                    right0 = 1 - wrong0
                    broke = float(((pb != tgt).float() * right0).sum() / right0.sum().clamp(min=1))
                arm.append({"alpha": al, "dice": dice, "dice_mean": float(np.mean(dice)),
                            "d_dice": float(np.mean(dice)) - float(np.mean(base_dice)),
                            "frac_flipped": flipped, "d_mass": dmass,
                            "mean_abs_dprob": dlogit,
                            "repaired_frac": repaired, "broke_frac": broke})
            rec["arms"][name] = arm

        records.append(rec)
        j = rec["arms"]["jac"][1]
        c = rec["arms"]["conf"][1]
        g = rec["arms"]["gt_ORACLE"][1]
        print(f"[{i+1}/{n}] {sid} base={rec['base_dice_mean']:.3f} "
              f"| a=0.10 dDice jac={j['d_dice']:+.4f} conf={c['d_dice']:+.4f} "
              f"ORACLE={g['d_dice']:+.4f}", flush=True)

    # ------------------------------------------------ aggregate + verdict
    agg = {}
    for name in ["jac", "conf", "rand", "gt_ORACLE"]:
        agg[name] = {}
        for k, al in enumerate([x for x in ALPHAS if x > 0]):
            dd = np.array([r["arms"][name][k]["d_dice"] for r in records])
            agg[name][f"alpha_{al}"] = {
                "mean_d_dice": float(dd.mean()), "std": float(dd.std()),
                "frac_improved": float((dd > 0).mean()),
                "mean_repaired": float(np.mean([r["arms"][name][k]["repaired_frac"] for r in records])),
                "mean_broke": float(np.mean([r["arms"][name][k]["broke_frac"] for r in records])),
                "mean_frac_flipped": float(np.mean([r["arms"][name][k]["frac_flipped"] for r in records])),
                "mean_abs_dprob": float(np.mean([r["arms"][name][k]["mean_abs_dprob"] for r in records])),
            }

    from scipy import stats as st
    best_a = f"alpha_{[x for x in ALPHAS if x>0][1]}"   # alpha=0.01, fixed a priori
    jac = np.array([r["arms"]["jac"][1]["d_dice"] for r in records])
    rnd = np.array([r["arms"]["rand"][1]["d_dice"] for r in records])
    cnf = np.array([r["arms"]["conf"][1]["d_dice"] for r in records])
    p_vs_rand = float(st.wilcoxon(jac, rnd).pvalue) if len(jac) > 5 else float("nan")
    p_vs_conf = float(st.wilcoxon(jac, cnf).pvalue) if len(jac) > 5 else float("nan")
    beats = (jac.mean() > rnd.mean() and jac.mean() > cnf.mean()
             and p_vs_rand < 0.05 and p_vs_conf < 0.05 and jac.mean() > 0)
    verdict = "A_CORRECTIVE_DIRECTION_EXISTS" if beats else "B_NO_CORRECTIVE_DIRECTION"

    summary = {
        "question": "Does a label-free direction at enc1 correct segmentation errors?",
        "gt_firewall": "build_directions() takes no target; GT enters only at scoring",
        "premeasured_geometry": {"cos_ET_TC": 0.9858, "cos_ET_entropy": 0.1556,
                                 "cos_TC_entropy": 0.2701},
        "n_subjects": len(records), "alphas_frac_of_z_norm": ALPHAS,
        "aggregate": agg,
        "test_at_alpha_0.10": {
            "jac_mean_d_dice": float(jac.mean()),
            "rand_mean_d_dice": float(rnd.mean()),
            "conf_mean_d_dice": float(cnf.mean()),
            "wilcoxon_jac_vs_rand_p": p_vs_rand,
            "wilcoxon_jac_vs_conf_p": p_vs_conf,
        },
        "PREREGISTERED_VERDICT": verdict,
        "decision_rule": ("A if d_Jac > BOTH d_rand and d_conf, significant, and positive; "
                          "else B. C (conf sharpens without correcting) reported separately."),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(OUT_DIR / "E172_per_subject.json", "w"), indent=1)
    json.dump(summary, open(OUT_DIR / "E172_summary.json", "w"), indent=1)
    with open(OUT_DIR / "E172_arms.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "alpha", "mean_d_dice", "frac_improved", "mean_repaired",
                    "mean_broke", "mean_frac_flipped", "mean_abs_dprob"])
        for nm, d in agg.items():
            for al, v in d.items():
                w.writerow([nm, al, round(v["mean_d_dice"], 5), round(v["frac_improved"], 3),
                            round(v["mean_repaired"], 5), round(v["mean_broke"], 5),
                            round(v["mean_frac_flipped"], 5), round(v["mean_abs_dprob"], 5)])

    print("\n" + "=" * 70)
    print(json.dumps(summary, indent=1))
    print("=" * 70)


if __name__ == "__main__":
    main()
