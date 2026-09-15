"""
E167 Part 2 -- Where does the residual error of the GOOD subjects live?

THE QUESTION (pre-registered, single, narrow):

  Seven prior-art audits returned OCCUPIED. The seventh's parting claim is the
  last unfalsified assumption in the whole search:

    "At ET 0.900 the residual error is dominated by inter-rater annotation
     ambiguity and partial-volume voxels. A +2pp gain there would require
     agreeing with the annotators MORE than the annotators agree with each
     other."

  If true, route (b) -- improve the 110 good subjects by +2pp each, which
  yields +1.14pp overall -- is closed for the same reason route (a) is: the
  information is not in the data.

  THE BAR THIS ADJUDICATES: the 110 good subjects sit at ET 0.8995 /
  TC 0.9380. Residual to perfect is 0.1005 / 0.0620. Route (b) needs +0.02
  on each, i.e. eliminating 20% of the ET residual and 32% of the TC
  residual on subjects that are ALREADY GOOD.

METHOD (inference only, one checkpoint, no training, no intervention):

  For each of the 110 good subjects, at NATIVE 1mm resolution:
    - run the model, threshold at 0.5, per region
    - compute the Euclidean distance transform of the GT mask boundary
    - assign every error voxel (FP or FN) to a distance stratum:
        d <= 1, d <= 2, d <= 3, or interior (d > 3)
    - report error mass per stratum as a FRACTION OF TOTAL RESIDUAL
      (so the strata sum to 1 by construction)

  Also report, as context, the fraction of GT voxels that are themselves
  boundary-adjacent -- a thin structure is boundary-dominated by geometry
  alone, and that must not be mistaken for an annotation-noise signature.

PRE-REGISTERED DECISION RULE (fixed before running):

  CLOSED     >=80% of residual error within <=2 voxels of the GT boundary.
             Residual is a boundary-localisation problem, consistent with the
             annotation-noise ceiling. Combined with a Part-1 published
             ceiling near 0.90-0.92 for ET => route (b) CLOSED on evidence.
  OPEN       <=50% boundary-adjacent, i.e. substantial INTERIOR error. The
             residual contains real recoverable structure that is NOT
             annotation ambiguity. Route (b) stays open, and this is the
             target seven audits have never aimed at.
  AMBIGUOUS  otherwise. Report honestly, do not adjudicate, name what would
             discriminate.

  The GEOMETRY CONTROL is decisive for reading CLOSED: if the GT itself is
  >=80% boundary-adjacent, then "error is boundary-adjacent" is trivially
  true and carries no information about annotation noise. Report the ratio
  (error boundary fraction) / (GT boundary fraction) as the real statistic.

SUBJECT SET: the 110 are defined BY ID against E160's stored table (the
complement of the 15-subject ET/TC bottom-decile union), never recomputed.
"""
import sys
import json
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders, REGIONS  # noqa: E402

OUT_DIR = Path(__file__).parent
E160_TABLE = (project_root / "experiments" / "exp_e12_eggo_m" / "e160"
              / "E160_L_per_subject.json")
PATCH = (128, 128, 128)
OVERLAP = 0.5

CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694

STRATA = [1, 2, 3]   # distance thresholds in voxels; beyond the last = interior


def good_subject_ids():
    """The 110, defined BY ID from E160's stored table -- not recomputed."""
    recs = json.load(open(E160_TABLE))
    det = np.array([r["dice_intact"]["ET"] for r in recs])
    dtc = np.array([r["dice_intact"]["TC"] for r in recs])
    k = 13
    bad = set(np.argsort(det)[:k].tolist()) | set(np.argsort(dtc)[:k].tolist())
    good = [recs[i]["sid"] for i in range(len(recs)) if i not in bad]
    assert len(good) == len(recs) - len(bad), "subject bookkeeping mismatch"
    return set(good), len(bad)


def _gaussian_weight(shape, sigma_scale=0.125):
    coords = [np.linspace(-1, 1, s) for s in shape]
    g = np.ones(shape, dtype=np.float32)
    for i, c in enumerate(coords):
        sh = [1] * len(shape)
        sh[i] = -1
        g = g * np.exp(-(c ** 2) / (2 * sigma_scale ** 2)).reshape(sh).astype(np.float32)
    return np.maximum(g, 1e-4)


def sliding_window(model, image, device, n_out=3, amp=True):
    _, _, D, H, W = image.shape
    pd, ph, pw = PATCH
    stride = [max(1, int(p * (1 - OVERLAP))) for p in PATCH]

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


def boundary_distance(gt_bin):
    """Distance (in voxels) from every voxel to the GT boundary surface.

    The boundary is the set of GT voxels with at least one non-GT 6-neighbour,
    plus the non-GT voxels adjacent to it -- i.e. the interface. Distance is
    measured to that interface, so both FP (outside) and FN (inside) voxels
    get a meaningful small distance when they sit near the edge.
    """
    g = gt_bin.astype(bool)
    if not g.any():
        return None
    inner = g & ~binary_erosion(g, iterations=1, border_value=0)
    if not inner.any():
        inner = g
    # EDT of the complement of the interface == distance to the interface
    return distance_transform_edt(~inner)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    got = ckpt.get("best_mean_dice")
    assert got is not None and abs(float(got) - EXPECTED_DICE) < 1e-9, \
        f"Checkpoint identity FAILED: expected {EXPECTED_DICE}, got {got}"
    print(f"[Sanity] checkpoint identity PASS (best_mean_dice={got}).")

    good, n_bad = good_subject_ids()
    print(f"[Subjects] {len(good)} good (excluded {n_bad} catastrophic-tail), by ID from E160.")

    model = UNet3D_v5(in_channels=4, out_channels=3).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    amp = not a.no_amp
    records = []
    idxs = [i for i in range(len(val_loader.dataset))]
    done = 0

    for i in idxs:
        image, target, sid = val_loader.dataset[i]
        if sid not in good:
            continue
        if a.limit and done >= a.limit:
            break
        image_b = image.unsqueeze(0).to(device)
        tgt = target.numpy()

        probs = sliding_window(model, image_b, device, amp=amp)
        pred = (probs > 0.5).astype(np.float32)

        rec = {"sid": sid, "regions": {}}
        for r, name in enumerate(REGIONS):
            gt, pr = tgt[r].astype(bool), pred[r].astype(bool)
            if not gt.any():
                continue
            dist = boundary_distance(gt)
            err = np.logical_xor(gt, pr)          # FP + FN
            n_err = int(err.sum())
            if n_err == 0:
                continue
            d_err = dist[err]
            strat = {}
            prev = 0.0
            for t in STRATA:
                strat[f"d<={t}"] = float((d_err <= t).mean())
            strat["interior(d>%d)" % STRATA[-1]] = float((d_err > STRATA[-1]).mean())

            # GEOMETRY CONTROL: how boundary-adjacent is the GT itself?
            d_gt = dist[gt]
            gt_strat = {f"d<={t}": float((d_gt <= t).mean()) for t in STRATA}

            dice = float(2 * (gt & pr).sum() / (gt.sum() + pr.sum()))
            rec["regions"][name] = {
                "dice": dice, "n_err": n_err,
                "n_gt": int(gt.sum()), "n_pred": int(pr.sum()),
                "err_strata": strat, "gt_strata": gt_strat,
                "err_frac_within2": strat["d<=2"],
                "gt_frac_within2": gt_strat["d<=2"],
                "enrichment_within2": float(strat["d<=2"] / max(gt_strat["d<=2"], 1e-9)),
            }
        records.append(rec)
        done += 1
        et = rec["regions"].get("ET", {})
        print(f"[{done}] {sid}  ET dice={et.get('dice',float('nan')):.3f} "
              f"err<=2vox={et.get('err_frac_within2',float('nan')):.3f} "
              f"(GT<=2vox={et.get('gt_frac_within2',float('nan')):.3f})", flush=True)

    # ------------------------------------------------- pre-registered verdict
    per_region = {}
    for name in REGIONS:
        vals = [r["regions"][name] for r in records if name in r["regions"]]
        if not vals:
            continue
        # error-mass weighted, so big-error subjects count proportionally
        w = np.array([v["n_err"] for v in vals], float)
        e2 = np.array([v["err_frac_within2"] for v in vals])
        g2 = np.array([v["gt_frac_within2"] for v in vals])
        per_region[name] = {
            "n_subjects": len(vals),
            "mean_dice": float(np.mean([v["dice"] for v in vals])),
            "err_frac_within2_mean": float(e2.mean()),
            "err_frac_within2_weighted": float((e2 * w).sum() / w.sum()),
            "gt_frac_within2_mean": float(g2.mean()),
            "enrichment_within2_mean": float(np.mean([v["enrichment_within2"] for v in vals])),
            "err_strata_mean": {k: float(np.mean([v["err_strata"][k] for v in vals]))
                                for k in vals[0]["err_strata"]},
        }

    et = per_region.get("ET", {})
    f2 = et.get("err_frac_within2_weighted", float("nan"))
    if f2 >= 0.80:
        verdict = "CLOSED"
    elif f2 <= 0.50:
        verdict = "OPEN"
    else:
        verdict = "AMBIGUOUS"

    summary = {
        "question": "Does the residual error of the 110 good subjects sit at the GT boundary (annotation-noise signature) or in the interior (recoverable structure)?",
        "checkpoint": str(CKPT), "checkpoint_dice_verified": float(got),
        "n_subjects": len(records),
        "bar_being_adjudicated": {
            "the_110_at": {"ET": 0.8995, "TC": 0.9380},
            "route_b_needs": "+0.02 each = 20% of ET residual, 32% of TC residual",
        },
        "per_region": per_region,
        "PREREGISTERED_VERDICT_on_ET": verdict,
        "decision_rule": "CLOSED if ET err_frac_within2>=0.80; OPEN if <=0.50; else AMBIGUOUS",
        "geometry_control_note": "enrichment_within2 = err_frac / gt_frac. If ~1.0 the error is "
                                 "merely as boundary-adjacent as the GT itself (geometry, not a "
                                 "boundary-specific failure). >1 means error CONCENTRATES at the "
                                 "boundary beyond what geometry alone implies.",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "E167_per_subject.json", "w") as f:
        json.dump(records, f, indent=1)
    with open(OUT_DIR / "E167_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    with open(OUT_DIR / "E167_per_subject.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "region", "dice", "n_err", "err_frac_within2",
                    "gt_frac_within2", "enrichment_within2"])
        for r in records:
            for name, v in r["regions"].items():
                w.writerow([r["sid"], name, round(v["dice"], 4), v["n_err"],
                            round(v["err_frac_within2"], 4),
                            round(v["gt_frac_within2"], 4),
                            round(v["enrichment_within2"], 4)])

    print("\n" + "=" * 70)
    print(json.dumps(summary, indent=1))
    print("=" * 70)


if __name__ == "__main__":
    main()
