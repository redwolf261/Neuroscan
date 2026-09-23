"""
E133 -- is the ET/TC contagion CAUSAL, and where does it live?

BACKGROUND. Decomposing E131's v5-control evaluation showed the 0.8597 mean
is destroyed by ~7 subjects scoring near zero, and that those failures are
COUPLED: ET Dice predicts TC Dice at rho=+0.797 (p=1.1e-28) after
controlling BOTH log ET size and log NCR size. On those subjects WT is
segmented at 0.79-0.96 while ET and TC are EXACTLY zero, with max
probability 0.0000 over the whole volume.

WHY THE NAIVE CAUSAL TEST WOULD BE MEANINGLESS. Verified by reading the
architecture: seg_head is a SINGLE Conv3d(32, 3, kernel_size=1) over dec1.
The three regions are INDEPENDENT LINEAR READOUTS of the same shared
32-channel feature map. ET does not feed TC at inference -- there is no
downstream. "Clamping the ET channel" would change nothing and would
trivially, misleadingly, show "no effect".

So the coupling, if causal, must live in the SHARED REPRESENTATION dec1 or
in the learned readout WEIGHTS. That is a decidable question with genuinely
different consequences:

  H1  REPRESENTATION FAILURE. dec1 does not encode the ET/TC distinction on
      these subjects. No readout could recover it, and the fix must change
      what the network LEARNS (loss weighting, sampling, synthetic data).

  H2  READOUT FAILURE. dec1 DOES encode it, but the learned 1x1x1 weights
      fail to extract it. Then a fresh readout on a FROZEN trunk recovers
      it, and the fix is local and cheap.

THE TEST. Freeze the network. Fit a fresh probe of EXACTLY seg_head's
functional form (a per-voxel linear classifier, i.e. a 1x1x1 conv) on the
GOOD subjects only, and evaluate it on the FAILING subjects it never saw.

CONTROLS, because a probe trained on the failures would be circular:
  - the probe NEVER sees a failing subject during fitting;
  - WT is fitted identically as a POSITIVE CONTROL. WT works fine, so its
    probe must too -- if the WT probe also fails, the methodology is broken
    and neither verdict is trustworthy;
  - probe capacity matches seg_head EXACTLY (linear, 32->1), so a win
    cannot come from extra capacity;
  - dec2 and dec3 are probed too, to localise how deep the information
    persists.

TWO ENGINEERING CHECKS DONE BEFORE RUNNING, NOT AFTER CRASHING:
  1. Caching dec1/2/3 for 125 subjects would need 44 GB in fp32 (dec1 alone
     is 33.6 GB). Not viable.
  2. Recomputing features per probe step would need 3 depths x 3 regions x
     60 epochs x 116 subjects = 62,640 forward passes ~ 1.7 hours.
  RESOLUTION: the probe is a 1x1x1 conv, i.e. a PER-VOXEL linear
  classifier. Fitting it on a balanced VOXEL SUBSAMPLE is mathematically
  the same problem. So each subject gets ONE forward pass, from which a
  balanced sample of voxels is retained (0.3-1.3 GB total), and the probe
  is fitted on those. Dice is then evaluated on FULL volumes.

NO TRAINING OF THE SEGMENTATION MODEL. Frozen checkpoint throughout.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset, REGIONS  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
PATCH = (128, 128, 128)
VOX_PER_SUBJECT = 20000     # balanced foreground/background sample for fitting
PROBE_STEPS = 3000
PROBE_LR = 1e-2

# Catastrophic-failure subjects from the E131 v5-control decomposition
# (ET dice < 0.10 or TC dice < 0.10).
FAIL_IDS = [
    "BraTS-GLI-00021-000", "BraTS-GLI-00731-001", "BraTS-GLI-01169-000",
    "BraTS-GLI-01176-000", "BraTS-GLI-01314-000", "BraTS-GLI-01530-000",
    "BraTS-GLI-00525-001", "BraTS-GLI-01103-000", "BraTS-GLI-01293-000",
]


def dice(pred_bin, target_bin):
    ps, ts = pred_bin.sum(), target_bin.sum()
    if ts == 0:
        return 1.0 if ps == 0 else 0.0
    return float(2.0 * (pred_bin * target_bin).sum() / (ps + ts))


def centre_patch(t, size=PATCH):
    c = [s // 2 for s in t.shape[1:]]
    sl = tuple(slice(max(0, c[j] - size[j] // 2),
                     max(0, c[j] - size[j] // 2) + size[j]) for j in range(3))
    return t[(slice(None),) + sl]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(
        project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth"))
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev)
    model.load_state_dict(ck["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("frozen checkpoint: epoch {}, mean dice {:.4f}".format(
        ck["epoch"] + 1, ck["best_mean_dice"]))

    ds = BraTSMultimodalDataset(str(project_root / "Dataset" / "Training"), "val",
                                val_split=0.1, patch_size=PATCH)

    rng = np.random.default_rng(SEED)
    ids, own_dice, keep = [], [], []
    samples = {1: [], 2: [], 3: []}
    labels = {1: [], 2: [], 3: []}

    print("")
    print("one forward pass per subject; sampling voxels for probe fitting...", flush=True)
    for i in range(len(ds)):
        image, target, sid = ds[i]
        x = centre_patch(image)
        y = centre_patch(target)
        if tuple(x.shape[1:]) != PATCH:
            continue
        with torch.no_grad():
            o = model(x.unsqueeze(0).to(dev))
        pr = o["probs"][0].cpu().numpy()
        yb = y.numpy()
        own_dice.append([dice((pr[r] >= 0.5).astype(np.float32),
                              (yb[r] > 0.5).astype(np.float32)) for r in range(3)])

        for depth in (1, 2, 3):
            f = o["dec{}".format(depth)][0]
            fac = {1: 1, 2: 2, 3: 4}[depth]
            t = torch.from_numpy(yb).unsqueeze(0)
            if fac > 1:
                t = F.avg_pool3d(t, fac, fac)
            t = (t[0] > 0.5).float()
            C = f.shape[0]
            fv = f.reshape(C, -1).T.cpu().numpy()
            tv = t.reshape(3, -1).T.numpy()
            pos = np.flatnonzero(tv[:, 2] > 0.5)
            neg = np.flatnonzero(tv[:, 2] <= 0.5)
            k = VOX_PER_SUBJECT // 2
            parts = []
            if len(pos):
                parts.append(rng.choice(pos, size=min(k, len(pos)), replace=False))
            if len(neg):
                parts.append(rng.choice(neg, size=min(k, len(neg)), replace=False))
            sel = np.concatenate(parts) if parts else np.array([], dtype=int)
            samples[depth].append(fv[sel].astype(np.float32))
            labels[depth].append(tv[sel].astype(np.float32))

        ids.append(sid)
        keep.append(i)
        if len(ids) % 25 == 0:
            print("  {}".format(len(ids)), flush=True)

    own_dice = np.array(own_dice)
    fail_pos = [k for k, s in enumerate(ids) if s in FAIL_IDS]
    good_pos = [k for k, s in enumerate(ids) if s not in FAIL_IDS]
    print("")
    print("{} good (probe TRAIN) / {} failing (probe TEST, unseen)".format(
        len(good_pos), len(fail_pos)))
    print("")
    print("the frozen model's OWN dice on these centred patches:")
    print("{:>8} {:>8} {:>8} {:>8}".format("group", "ET", "TC", "WT"))
    for lab, pos in [("good", good_pos), ("FAIL", fail_pos)]:
        d = own_dice[pos]
        print("{:>8} {:8.4f} {:8.4f} {:8.4f}".format(lab, d[:, 0].mean(), d[:, 1].mean(), d[:, 2].mean()))

    results = {}
    for depth in (1, 2, 3):
        Xtr = torch.from_numpy(np.concatenate([samples[depth][k] for k in good_pos])).to(dev)
        Ytr = torch.from_numpy(np.concatenate([labels[depth][k] for k in good_pos])).to(dev)
        C = Xtr.shape[1]
        fac = {1: 1, 2: 2, 3: 4}[depth]
        print("")
        print("=== dec{}: {} channels, {:,} training voxels ===".format(depth, C, Xtr.shape[0]), flush=True)
        results["dec{}".format(depth)] = {}

        for r, reg in enumerate(REGIONS):
            torch.manual_seed(SEED)
            w = nn.Linear(C, 1).to(dev)
            opt = torch.optim.Adam(w.parameters(), lr=PROBE_LR)
            g = torch.Generator().manual_seed(SEED)
            n = Xtr.shape[0]
            for _ in range(PROBE_STEPS):
                idx = torch.randint(0, n, (8192,), generator=g).to(dev)
                loss = F.binary_cross_entropy_with_logits(w(Xtr[idx]).squeeze(1), Ytr[idx, r])
                opt.zero_grad()
                loss.backward()
                opt.step()

            out = {}
            with torch.no_grad():
                for lab, pos in [("good", good_pos), ("fail", fail_pos)]:
                    dd = []
                    for k in pos:
                        image, target, _ = ds[keep[k]]
                        x = centre_patch(image).unsqueeze(0).to(dev)
                        o = model(x)
                        f = o["dec{}".format(depth)][0]
                        Cc, dd_, hh, ww = f.shape
                        logit = w(f.reshape(Cc, -1).T).squeeze(1).reshape(dd_, hh, ww)
                        pred = (torch.sigmoid(logit).cpu().numpy() >= 0.5).astype(np.float32)
                        t = centre_patch(target).unsqueeze(0)
                        if fac > 1:
                            t = F.avg_pool3d(t, fac, fac)
                        dd.append(dice(pred, (t[0, r].numpy() > 0.5).astype(np.float32)))
                    out[lab] = float(np.mean(dd))
            results["dec{}".format(depth)][reg] = out
            print("  {}: probe dice  good={:.4f}  FAILING={:.4f}".format(reg, out["good"], out["fail"]),
                  flush=True)

    own_fail = own_dice[fail_pos].mean(0)
    print("")
    print("=" * 70)
    print("VERDICT")
    print("{:14} {:>12} {:>12} {:>10}".format("", "model head", "dec1 probe", "recovery"))
    verdict = {}
    for r, reg in enumerate(REGIONS):
        p = results["dec1"][reg]["fail"]
        print("  {:12} {:12.4f} {:12.4f} {:+10.4f}".format(reg, own_fail[r], p, p - own_fail[r]))
        verdict[reg] = {"model_head_fail": float(own_fail[r]), "dec1_probe_fail": p,
                        "recovery": float(p - own_fail[r])}

    wt_ok = results["dec1"]["WT"]["good"] > 0.7
    et_rec, tc_rec = verdict["ET"]["recovery"], verdict["TC"]["recovery"]
    print("")
    if not wt_ok:
        v = "PROBE_METHOD_BROKEN"
        detail = ("The WT positive-control probe failed on GOOD subjects, so the probe "
                  "methodology is unsound and neither verdict can be trusted.")
    elif et_rec > 0.2 or tc_rec > 0.2:
        v = "H2_READOUT_FAILURE"
        detail = ("dec1 CONTAINS the ET/TC information on the failing subjects -- a fresh "
                  "linear probe of identical capacity, never trained on them, recovers it. "
                  "The trained seg_head weights fail to extract what the representation "
                  "already encodes. The fix is local: readout and/or loss weighting, NOT a "
                  "change to what the encoder learns.")
    else:
        v = "H1_REPRESENTATION_FAILURE"
        detail = ("dec1 does NOT encode the ET/TC distinction on the failing subjects -- an "
                  "identical-capacity probe cannot recover it either, so no readout could. "
                  "The fix must change what the network LEARNS (loss weighting, sampling, "
                  "synthetic data).")
    print(v)
    print(detail)

    out = {"checkpoint": a.ckpt, "n_good": len(good_pos), "n_fail": len(fail_pos),
           "fail_ids": [ids[k] for k in fail_pos],
           "model_head_on_failing": {reg: float(own_fail[r]) for r, reg in enumerate(REGIONS)},
           "probes": results, "verdict": v, "detail": detail}
    with open(OUT_DIR / "E133_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("")
    print("Saved E133_summary.json")


if __name__ == "__main__":
    main()
