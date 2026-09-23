"""
E130 FULL EVALUATION -- all 125 validation subjects, per-region Dice + HD95,
on a specified checkpoint.

WHY THIS EXISTS SEPARATELY FROM THE TRAINING SCRIPT: the E130 run suffered a
training collapse at epoch 39 (train_dice 0.919 -> 0.878 -> 0.491, val WT
0.92 -> 0.31 and stuck; almost certainly an AMP loss-scale event, with the
cosine LR by then too low to re-escape the basin -- ET and TC partially
recovered, WT did not). The training script's own end-of-run full validation
would therefore have measured the COLLAPSED model. The best checkpoint was
saved at epoch 32, BEFORE the collapse, verified intact (all parameters
finite, best_mean_dice 0.8926). This script evaluates that checkpoint
properly.

It also fixes a second limitation: training-time validation used
fast_val=25 subjects to keep per-epoch cost sane, so every intermediate
number carries real sampling noise. This evaluates all 125.

Reported per region (ET, TC, WT):
  Dice           mean +/- std, plus median and IQR
  HD95           mean +/- std over subjects where it is defined
  empty-ET count how many subjects genuinely have no enhancing tumour, since
                 the Dice convention for those (1.0 if correctly predicted
                 empty, else 0.0) materially affects the ET mean and should
                 be reported, not hidden.
Also saves a per-subject table so later analyses (E131 failure
characterisation, the causal re-measurement) can join on subject_id without
recomputing.
"""
import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from neuroscan_3d_v13 import UNet3D_v13, UNet3D_v14  # noqa: E402
from neuroscan_3d_v16 import UNet3D_v16  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset, REGIONS  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "e130_train", str(Path(__file__).parent / "train_e130_multimodal_baseline.py"))
_t = importlib.util.module_from_spec(_spec)
_saved_argv = sys.argv
sys.argv = ["eval"]
_spec.loader.exec_module(_t)
sys.argv = _saved_argv

OUT_DIR = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default=str(OUT_DIR / "runs" / "E130_baseline_seed0" / "checkpoints" / "best.pth"))
    ap.add_argument("--tag", type=str, default="E130_baseline_seed0_ep32")
    ap.add_argument("--amp", type=int, default=1)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    print(f"checkpoint: {a.ckpt}")
    print(f"  epoch={ck['epoch']+1} best_mean_dice={ck.get('best_mean_dice', ck.get('mean_dice', float('nan'))):.4f} "
          f"arch={ck.get('arch')} task={ck.get('task')}")
    assert ck.get("arch") in ("v5_4in3out", "v5", "v13", "v14", "v16"), f"unexpected arch {ck.get('arch')}"

    _arch = ck.get("arch")
    if _arch == "v16":
        model = UNet3D_v16(in_channels=4, out_channels=3).to(device)
    elif _arch == "v14":
        model = UNet3D_v14(in_channels=4, out_channels=3).to(device)
    elif _arch == "v13":
        model = UNet3D_v13(in_channels=4, out_channels=3).to(device)
    else:
        model = UNet3D_v5(in_channels=4, out_channels=3).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ds = BraTSMultimodalDataset(str(project_root / "Dataset" / "Training"), "val",
                                val_split=0.1, patch_size=_t.PATCH)
    print(f"evaluating {len(ds)} validation subjects (full volumes, sliding window)\n", flush=True)

    recs = []
    t0 = time.time()
    for i in range(len(ds)):
        image, target, sid = ds[i]
        img = image.unsqueeze(0).to(device)
        probs = _t.sliding_window_predict(model, img, _t.PATCH, _t.SW_OVERLAP, 3, device, bool(a.amp))
        pb = (probs >= 0.5).astype(np.float32)
        tb = target.numpy()
        d = _t.dice_per_region(pb, tb)
        h = _t.hd95_per_region(pb, tb)
        recs.append({
            "subject_id": sid,
            **{f"dice_{r}": float(d[j]) for j, r in enumerate(REGIONS)},
            **{f"hd95_{r}": float(h[j]) for j, r in enumerate(REGIONS)},
            **{f"target_voxels_{r}": int(tb[j].sum()) for j, r in enumerate(REGIONS)},
            **{f"pred_voxels_{r}": int(pb[j].sum()) for j, r in enumerate(REGIONS)},
        })
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(ds)}  ({el:.0f}s, {el/(i+1)*(len(ds)-i-1):.0f}s left)", flush=True)

    with open(OUT_DIR / f"E130_full_eval_{a.tag}_per_subject.json", "w") as f:
        json.dump(recs, f, indent=2)

    print(f"\n{'='*66}")
    print(f"FULL VALIDATION -- {a.tag}  (n={len(recs)})")
    print(f"{'='*66}")
    print(f"{'region':>6} {'Dice mean':>11} {'std':>7} {'median':>8} {'HD95 mean':>10} {'std':>7} {'n_hd95':>7}")
    summary = {}
    for r in REGIONS:
        dv = np.array([x[f"dice_{r}"] for x in recs])
        hv = np.array([x[f"hd95_{r}"] for x in recs])
        hvv = hv[~np.isnan(hv)]
        print(f"{r:>6} {dv.mean():11.4f} {dv.std():7.4f} {np.median(dv):8.4f} "
              f"{(hvv.mean() if len(hvv) else float('nan')):10.3f} "
              f"{(hvv.std() if len(hvv) else float('nan')):7.3f} {len(hvv):7d}")
        summary[r] = {"dice_mean": float(dv.mean()), "dice_std": float(dv.std()),
                      "dice_median": float(np.median(dv)),
                      "hd95_mean": float(hvv.mean()) if len(hvv) else None,
                      "hd95_std": float(hvv.std()) if len(hvv) else None,
                      "n_hd95_defined": int(len(hvv))}
    mean_d = float(np.mean([summary[r]["dice_mean"] for r in REGIONS]))
    print(f"\n  MEAN Dice across regions: {mean_d:.4f}")

    # Honest reporting of the empty-ET population.
    n_empty = {r: int(sum(1 for x in recs if x[f"target_voxels_{r}"] == 0)) for r in REGIONS}
    print(f"\n  subjects with EMPTY ground-truth region: "
          + ", ".join(f"{r}={n_empty[r]}" for r in REGIONS))
    for r in REGIONS:
        if n_empty[r]:
            dv = np.array([x[f"dice_{r}"] for x in recs if x[f"target_voxels_{r}"] == 0])
            dvn = np.array([x[f"dice_{r}"] for x in recs if x[f"target_voxels_{r}"] > 0])
            print(f"    {r}: empty-subject Dice mean={dv.mean():.4f} (n={len(dv)}) | "
                  f"non-empty Dice mean={dvn.mean():.4f} (n={len(dvn)})")

    out = {"tag": a.tag, "checkpoint": a.ckpt, "epoch": int(ck["epoch"]) + 1,
           "n_subjects": len(recs), "per_region": summary, "mean_dice": mean_d,
           "n_empty_gt": n_empty,
           "note": ("Evaluated the epoch-32 checkpoint saved BEFORE the epoch-39 training "
                    "collapse. Training-time val used only 25 subjects; this uses all 125.")}
    with open(OUT_DIR / f"E130_full_eval_{a.tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved E130_full_eval_{a.tag}.json + per-subject table")


if __name__ == "__main__":
    main()
