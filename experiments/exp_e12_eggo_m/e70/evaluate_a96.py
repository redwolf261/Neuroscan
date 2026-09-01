"""
Evaluate MM_A96 (multimodal, 96^3) with paired statistics against MM
(multimodal, 64^3) and the FLAIR-only baseline.

Same statistical protocol as evaluate_e70.py: per-subject Dice primary,
paired t-test + Wilcoxon on per-subject deltas, bootstrap 95% CI.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
RUNS = OUT_DIR / "runs"
E56_TABLE = project_root / "experiments" / "exp_e12_eggo_m" / "e56" / "E56_per_subject_rescoring.json"
N_BOOT = 10000


def dice(tp, fp, fn, eps=1e-6):
    return float((2 * tp) / (2 * tp + fp + fn + eps))


@torch.no_grad()
def score_a96(seed, device):
    ck_path = RUNS / f"MM_A96_seed{seed}" / "checkpoints" / "best.pth"
    ck = torch.load(str(ck_path), map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    ds = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                split="val", val_split=0.1, target_shape=(96, 96, 96))
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
    return per_subject, dice(TP, FP, FN)


def paired(a, b, ids, label):
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
           "median_delta": float(np.median(d)), "bootstrap_95ci": list(ci),
           "t_p": float(t_p), "wilcoxon_p": float(w_p), "frac_improved": float((d > 0).mean())}
    print(f"\n--- {label} ---")
    print(f"  mean delta      : {d.mean():+.4f} ({d.mean()*100:+.2f} pp)")
    print(f"  bootstrap 95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}] {'EXCLUDES 0' if ci[0] > 0 or ci[1] < 0 else 'includes 0'}")
    print(f"  paired t p      : {t_p:.3e}   Wilcoxon p: {w_p:.3e}")
    print(f"  frac improved   : {(d > 0).mean():.3f}")
    return res


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(E56_TABLE) as f:
        e56 = json.load(f)
    flair_baseline = e56["baseline_A"]["per_subject_dice"]

    mm_eval_path = OUT_DIR / "MM_seed0_per_subject.json"
    if mm_eval_path.exists():
        with open(mm_eval_path) as f:
            mm_per_subject = json.load(f)
    else:
        mm_per_subject = None
        print("WARNING: MM_seed0_per_subject.json not found -- skipping MM_A96 vs MM comparison.")

    a96_per_subject, a96_pooled = score_a96(0, device)
    print(f"MM_A96 (seed 0): pooled={a96_pooled:.4f} per-subj-mean={np.mean(list(a96_per_subject.values())):.4f}")

    results = {"a96_pooled": a96_pooled,
               "a96_per_subject_mean": float(np.mean(list(a96_per_subject.values()))),
               "comparisons": []}

    ids_flair = sorted(set(a96_per_subject) & set(flair_baseline))
    results["comparisons"].append(paired(a96_per_subject, flair_baseline, ids_flair,
                                         "MM_A96 vs FLAIR-only (total gain: modality + resolution)"))

    if mm_per_subject is not None:
        ids_mm = sorted(set(a96_per_subject) & set(mm_per_subject))
        results["comparisons"].append(paired(a96_per_subject, mm_per_subject, ids_mm,
                                             "MM_A96 vs MM (resolution-alone contribution, given modality already present)"))

    with open(OUT_DIR / "E70_A96_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved E70_A96_evaluation.json")


if __name__ == "__main__":
    main()
