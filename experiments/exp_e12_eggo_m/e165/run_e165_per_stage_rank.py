"""
E165 -- Per-stage task-relevant transformation rank R*_l.

THE QUESTION (pre-registered, single, narrow):

  E147 measured R*_i = the minimum PCA rank of the BOTTLENECK (256ch @ 8^3)
  retaining >=90% agreement with the subject's OWN undegraded prediction.
  Result: 74/88 subjects served by rank <=4 of 256; ZERO need >32; mean
  R* = 4.95 = 1.94% of channel capacity.

  That confirms "a transformation may be computationally high-dimensional
  while its task-relevant effect is low-dimensional" -- AT THE BOTTLENECK.

  Every rank measurement in this project (E147, E124-E126) is at that ONE
  site. R*_l has NEVER been measured at enc1/enc2/enc3/dec1. E121 shows the
  encoder stages carry the causal necessity (N_1=0.653, N_2=0.614 vs
  N_3=0.272), so if a genuine task-relevant-rank DEFICIT exists anywhere,
  it is there -- and a compute argument only becomes interesting at a site
  that is actually expensive.

PRE-REGISTERED DECISION RULE (fixed before running):

  CLOSE          if all stages mirror the bottleneck (R*/C <~ 5%).
                 The proposition is universal and uninteresting: every
                 stage is over-provisioned, nothing to exploit. Consistent
                 with E128 (3.03x params -> +0.139pp).
  RANK-LIMITED   if some stage shows R*/C high (>50%). That stage is
                 genuinely rank-limited -- an unmeasured fact, and the only
                 place a replacement argument could have teeth. Audit THEN.
  MIXED          otherwise. Report honestly, do not adjudicate.

  EXPECTED: CLOSE.

CONFOUND GUARD (essential, inherited from E147 verbatim): agreement is
measured against the subject's OWN UNDEGRADED PREDICTION, never ground
truth. Otherwise a subject at Dice 0.95 has more room to fall than one at
0.30, manufacturing a correlation with baseline difficulty.

METHOD: forward hook on each stage's output tensor. Per subject, per stage,
flatten to (C, N_voxels), take PCA over the channel axis, reconstruct with
rank r, run the REST of the network normally, and measure agreement
(binary Dice over the 3 regions) against the intact prediction.

NO training. NO architecture change. ONE checkpoint. Inference only.
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
PATCH = (128, 128, 128)

CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694

# E147's exact grid and threshold -- do not change, results must be comparable.
RANK_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256]
AGREE_THRESHOLD = 0.90

# stage name -> channel count (verified from the checkpoint's conv weights)
STAGES = {"enc1": 32, "enc2": 64, "enc3": 128, "bottleneck": 256, "dec1": 32}


def lowrank_channels(x, r):
    """x: (1,C,D,H,W). PCA over the channel axis, reconstruct with rank r.

    Mean-centred across voxels (so rank r means r principal DIRECTIONS on top
    of the mean), matching the standard PCA-truncation reading used in E147.
    """
    b, c, d, h, w = x.shape
    if r >= c:
        return x
    m = x.reshape(c, -1).float()                      # (C, N)
    mu = m.mean(dim=1, keepdim=True)
    mc = m - mu
    # economy SVD over the channel axis
    U, S, Vh = torch.linalg.svd(mc, full_matrices=False)
    approx = (U[:, :r] * S[:r]) @ Vh[:r, :]
    return (approx + mu).reshape(b, c, d, h, w).to(x.dtype)


class StageTruncator:
    """Forward hook that low-rank-truncates one stage's output."""

    def __init__(self):
        self.stage = None
        self.rank = None

    def __call__(self, module, inputs, output):
        if self.stage is None or self.rank is None:
            return output
        return lowrank_channels(output, self.rank)


def register(model, trunc):
    handles = {}
    for name in STAGES:
        mod = getattr(model, name)
        handles[name] = mod.register_forward_hook(
            lambda m, i, o, n=name: trunc(m, i, o) if trunc.stage == n else o)
    return handles


def dice_agree(a_bin, b_bin):
    """Binary agreement Dice per region, averaged. Both are PREDICTIONS."""
    out = []
    for r in range(a_bin.shape[0]):
        p, q = a_bin[r], b_bin[r]
        ps, qs = p.sum(), q.sum()
        if ps == 0 and qs == 0:
            out.append(1.0)
        elif ps + qs == 0:
            out.append(1.0)
        else:
            out.append(float(2.0 * (p * q).sum() / (ps + qs)))
    return float(np.mean(out))


def _gaussian_weight(shape, sigma_scale=0.125):
    coords = [np.linspace(-1, 1, s) for s in shape]
    g = np.ones(shape, dtype=np.float32)
    for i, c in enumerate(coords):
        sh = [1] * len(shape)
        sh[i] = -1
        g = g * np.exp(-(c ** 2) / (2 * sigma_scale ** 2)).reshape(sh).astype(np.float32)
    return np.maximum(g, 1e-4)


def sliding_window(model, image, device, n_out=3, amp=True, overlap=0.5):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--overlap", type=float, default=0.25,
                    help="lower overlap = faster; this is an AGREEMENT measure, "
                         "both arms use the identical setting so it cancels")
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

    trunc = StageTruncator()
    register(model, trunc)

    # SANITY: with the hook dormant, output must be bit-identical to forward().
    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)
    img0, _, _ = val_loader.dataset[0]
    tile = img0[:, :128, :128, :128].unsqueeze(0).to(device)
    trunc.stage, trunc.rank = None, None
    with torch.no_grad():
        ref = model(tile)["probs"].float()
        again = model(tile)["probs"].float()
    assert float((ref - again).abs().max()) == 0.0, "dormant hook is not a no-op -- STOP."
    # and a full-rank truncation must also be a near-no-op
    trunc.stage, trunc.rank = "bottleneck", 256
    with torch.no_grad():
        full = model(tile)["probs"].float()
    d_full = float((ref - full).abs().max())
    trunc.stage, trunc.rank = None, None
    print(f"[Sanity] dormant hook exact no-op PASS; rank=C truncation max diff {d_full:.3e}")
    assert d_full < 1e-3, "rank=C truncation should be ~identity -- STOP."

    n = len(val_loader.dataset)
    if a.limit:
        n = min(n, a.limit)
    amp = not a.no_amp
    print(f"[Data] evaluating {n} subjects, overlap={a.overlap}, stages={list(STAGES)}\n")

    records = []
    for i in range(n):
        image, _, sid = val_loader.dataset[i]
        image_b = image.unsqueeze(0).to(device)

        trunc.stage, trunc.rank = None, None
        p_intact = sliding_window(model, image_b, device, amp=amp, overlap=a.overlap)
        intact_bin = (p_intact > 0.5).astype(np.float32)

        rec = {"sid": sid, "stages": {}}
        for stage, C in STAGES.items():
            curve, rstar = [], None
            for r in RANK_GRID:
                if r > C:
                    curve.append(None)
                    continue
                trunc.stage, trunc.rank = stage, r
                p_r = sliding_window(model, image_b, device, amp=amp, overlap=a.overlap)
                ag = dice_agree((p_r > 0.5).astype(np.float32), intact_bin)
                curve.append(ag)
                if rstar is None and ag >= AGREE_THRESHOLD:
                    rstar = r
                    break          # first crossing == E147's rule; stop early
            trunc.stage, trunc.rank = None, None
            if rstar is None:
                rstar = C
            rec["stages"][stage] = {"C": C, "R_star": rstar,
                                    "R_over_C": rstar / C, "agree_curve": curve}
        records.append(rec)
        summ = "  ".join(f"{s}:{rec['stages'][s]['R_star']}/{STAGES[s]}" for s in STAGES)
        print(f"[{i+1}/{n}] {sid}  {summ}", flush=True)

    # ------------------------------------------------- pre-registered verdict
    per_stage = {}
    for stage, C in STAGES.items():
        rs = np.array([r["stages"][stage]["R_star"] for r in records], float)
        frac = rs / C
        per_stage[stage] = {
            "C": C,
            "R_star_mean": float(rs.mean()), "R_star_median": float(np.median(rs)),
            "R_star_max": int(rs.max()),
            "R_over_C_mean": float(frac.mean()), "R_over_C_median": float(np.median(frac)),
            "frac_subjects_at_or_below_4": float((rs <= 4).mean()),
            "frac_subjects_needing_over_half": float((frac > 0.5).mean()),
        }

    maxfrac = max(v["R_over_C_mean"] for v in per_stage.values())
    if maxfrac <= 0.05:
        verdict = "CLOSE"
    elif maxfrac > 0.50:
        verdict = "RANK_LIMITED"
    else:
        verdict = "MIXED"

    summary = {
        "question": "Is the low-task-relevant-rank property universal across stages, or specific to the bottleneck?",
        "checkpoint": str(CKPT), "checkpoint_dice_verified": float(got),
        "n_subjects": len(records), "rank_grid": RANK_GRID,
        "agreement_threshold": AGREE_THRESHOLD,
        "confound_guard": "agreement vs subject's OWN undegraded prediction, never ground truth",
        "E147_bottleneck_reference": {"mean_R_star": 4.95, "R_over_C": 0.0194,
                                      "frac_at_or_below_4": 0.84, "n": 88},
        "per_stage": per_stage,
        "max_R_over_C_mean": maxfrac,
        "PREREGISTERED_VERDICT": verdict,
        "decision_rule": "CLOSE if all stages R*/C<=0.05; RANK_LIMITED if any >0.50; else MIXED",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "E165_per_subject.json", "w") as f:
        json.dump(records, f, indent=1)
    with open(OUT_DIR / "E165_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    with open(OUT_DIR / "E165_per_subject.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid"] + [f"R_star_{s}" for s in STAGES] + [f"R_over_C_{s}" for s in STAGES])
        for r in records:
            w.writerow([r["sid"]]
                       + [r["stages"][s]["R_star"] for s in STAGES]
                       + [round(r["stages"][s]["R_over_C"], 4) for s in STAGES])

    print("\n" + "=" * 70)
    print(json.dumps(summary, indent=1))
    print("=" * 70)


if __name__ == "__main__":
    main()
