"""
Phase E70 evaluation: per-subject Dice + paired statistics.

Produces the three numbers the paper needs, each against an explicitly
named reference so no comparison is ambiguous:

  1. MM      vs. FLAIR-only baseline   -> gain from modality expansion
  2. MM_CAS  vs. MM                    -> gain attributable to CAS ALONE
                                          (the ablation that isolates the
                                          novel component)
  3. MM_CAS  vs. FLAIR-only baseline   -> total system gain

Statistics follow this project's own established protocol (E56/E48):
per-subject Dice as the primary endpoint, paired t-test AND Wilcoxon
signed-rank on the per-subject deltas, plus bootstrap 95% CI over
subjects. Pooled Dice is also reported for continuity with the older
pre-E56 numbers, clearly labelled as the secondary endpoint.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from neuroscan_3d_v10 import UNet3D_v10  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
RUNS = OUT_DIR / "runs"
E56_TABLE = project_root / "experiments" / "exp_e12_eggo_m" / "e56" / "E56_per_subject_rescoring.json"
N_BOOT = 10000


def dice(tp, fp, fn, eps=1e-6):
    return float((2 * tp) / (2 * tp + fp + fn + eps))


@torch.no_grad()
def score(condition, seed, device):
    ck_path = RUNS / f"{condition}_seed{seed}" / "checkpoints" / "best.pth"
    ck = torch.load(str(ck_path), map_location=device, weights_only=False)
    cls = UNet3D_v3 if condition == "MM" else UNet3D_v10
    model = cls(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    ds = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                split="val", val_split=0.1, target_shape=(64, 64, 64))
    per_subject, TP, FP, FN = {}, 0.0, 0.0, 0.0
    for i in range(len(ds)):
        img, msk, sid = ds[i]
        img = img.unsqueeze(0).to(device)
        msk = msk.unsqueeze(0).to(device)
        pred = (model(img)["probs"] >= 0.5).float()
        tp = (pred * msk).sum().item()
        fp = (pred * (1 - msk)).sum().item()
        fn = ((1 - pred) * msk).sum().item()
        TP += tp; FP += fp; FN += fn
        per_subject[sid] = dice(tp, fp, fn)
    del model
    torch.cuda.empty_cache()
    return {"per_subject": per_subject, "pooled": dice(TP, FP, FN),
            "per_subject_mean": float(np.mean(list(per_subject.values()))),
            "epoch_of_best": ck.get("epoch")}


def paired(a, b, ids, label):
    """a, b: dict sid->dice. Tests a - b (positive = a better)."""
    d = np.array([a[s] - b[s] for s in ids])
    t_stat, t_p = stats.ttest_rel([a[s] for s in ids], [b[s] for s in ids])
    try:
        w_stat, w_p = stats.wilcoxon([a[s] for s in ids], [b[s] for s in ids])
    except ValueError:
        w_stat, w_p = float("nan"), 1.0
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(N_BOOT)])
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    res = {"comparison": label, "n": len(ids), "mean_delta": float(d.mean()),
           "median_delta": float(np.median(d)), "std_delta": float(d.std()),
           "frac_improved": float((d > 0).mean()),
           "t_p": float(t_p), "wilcoxon_p": float(w_p), "bootstrap_95ci": list(ci)}
    print(f"\n--- {label} ---")
    print(f"  mean delta      : {d.mean():+.4f}  ({d.mean()*100:+.2f} pp)")
    print(f"  median delta    : {np.median(d):+.4f}")
    print(f"  bootstrap 95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
          f"{'EXCLUDES 0' if ci[0] > 0 or ci[1] < 0 else 'includes 0'}")
    print(f"  paired t p      : {t_p:.3e}")
    print(f"  Wilcoxon p      : {w_p:.3e}")
    print(f"  frac improved   : {(d > 0).mean():.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(E56_TABLE) as f:
        e56 = json.load(f)
    flair_baseline = e56["baseline_A"]["per_subject_dice"]
    flair_pooled = e56["baseline_A"]["reported_pooled_dice"]
    flair_mean = e56["baseline_A"]["mean_per_subject_dice"]
    print(f"FLAIR-only reference baseline (e24/A_baseline_seed0):")
    print(f"  pooled={flair_pooled:.4f}  per-subject-mean={flair_mean:.4f}")

    results = {"flair_baseline": {"pooled": flair_pooled, "per_subject_mean": flair_mean},
               "seeds": {}, "comparisons": []}

    for seed in a.seeds:
        print(f"\n{'='*62}\nSEED {seed}\n{'='*62}")
        mm = score("MM", seed, device)
        cas = score("MM_CAS", seed, device)
        ids = sorted(set(mm["per_subject"]) & set(cas["per_subject"]) & set(flair_baseline))
        print(f"\nMM      : pooled={mm['pooled']:.4f}  per-subj={mm['per_subject_mean']:.4f}")
        print(f"MM_CAS  : pooled={cas['pooled']:.4f}  per-subj={cas['per_subject_mean']:.4f}")
        print(f"(n={len(ids)} subjects common to all three)")

        results["seeds"][str(seed)] = {"MM": {k: v for k, v in mm.items() if k != "per_subject"},
                                       "MM_CAS": {k: v for k, v in cas.items() if k != "per_subject"}}
        results["comparisons"].append({
            "seed": seed,
            "MM_vs_FLAIR": paired(mm["per_subject"], flair_baseline, ids, f"[s{seed}] MM vs FLAIR-only  (modality gain)"),
            "CAS_vs_MM": paired(cas["per_subject"], mm["per_subject"], ids, f"[s{seed}] MM_CAS vs MM     (CAS ablation)"),
            "CAS_vs_FLAIR": paired(cas["per_subject"], flair_baseline, ids, f"[s{seed}] MM_CAS vs FLAIR  (total system)"),
        })

    with open(OUT_DIR / "E70_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved E70_evaluation.json")


if __name__ == "__main__":
    main()
