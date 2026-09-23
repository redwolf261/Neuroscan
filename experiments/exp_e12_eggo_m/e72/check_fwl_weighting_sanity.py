"""
Phase E72, Fragility-Weighted Loss (FWL) -- cheapest, first sanity check
BEFORE any training run, per the Observe->Hypothesize->Attempt->Test->Adapt
directive.

CANDIDATE IDEA: use hat{d}_i (the E71 scaled aux head's OWN prediction of
a subject's causal bottleneck-ablation sensitivity, already validated:
Spearman(predicted, measured) = 0.866 held-out, partial rho=0.855
controlling for size) as a PER-SUBJECT LOSS WEIGHT during segmentation
training -- NOT for routing/gating (killed at E71's gate-sweep test) and
NOT for extra compute (occupied by MAGICORE / test-time-compute-scaling
literature). This is a different verb: reweight supervision by a
CAUSALLY-DERIVED (not loss- or uncertainty-derived) fragility signal.

WHY THIS SCRIPT (not straight to a training run):
Training-time interventions can't be tested purely at inference like the
CDCG gate was (E71's check_cdcg_gate_sweep.py). But the WEIGHTING SCHEME
ITSELF -- does it produce a sane, non-degenerate distribution of weights
that actually concentrates on subjects hat{d}_i identifies as fragile,
without collapsing onto a handful of subjects or just re-deriving lesion
size -- CAN and SHOULD be checked cheaply first, using the already-trained
aux head and NO new model training.

THIS SCRIPT DOES NOT TRAIN ANYTHING. It only:
  1. Loads the E71 scaled aux head (already trained, validated).
  2. Computes hat{d}_i for a sample of TRAINING subjects (the population
     that would actually be reweighted).
  3. Converts hat{d}_i -> per-subject loss weight under 2 candidate
     schemes (softmax-temperature and linear-normalized), and reports
     the resulting weight distribution (a degenerate scheme is one where
     a tiny fraction of subjects absorb almost all the weight mass).
  4. Checks Spearman(weight, hat{d}_i) (should be strongly positive, by
     construction) AND Spearman(weight, native_size) (should be WEAKER
     than the measured d_i-vs-size relationship, i.e. the weighting
     should not just be re-deriving "upweight small lesions").

DECISION RULE:
  PASS (proceed to a short pilot training run) if:
    (a) no single subject holds > 5% of total weight mass (non-degenerate)
    (b) effective sample size (1 / sum(w_i^2) for normalized w) > 50% of N
    (c) |Spearman(weight, native_size)| is materially weaker than
        |Spearman(measured_d_i, native_size)| = 0.454 (E48's reference),
        confirming the weighting is not simply a size proxy in disguise
  FAIL otherwise -- do not proceed to a pilot training run; the weighting
  scheme itself is broken/degenerate and needs a fix or the idea is dead.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e71"))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402
from check_cdcg_prediction1 import AuxHead, forward_get_bottleneck_and_ablated_dice  # noqa: E402

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 0
N_SAMPLE = 250  # subset of the 1126 training subjects -- cheap sanity check, not full labeling
E48_SIZE_RHO_REFERENCE = -0.454

MM_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e70" / "runs"
           / "MM_seed0" / "checkpoints" / "best.pth")
AUX_HEAD_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e71"
                  / "E71_aux_head_state_scaled.pt")


def softmax_weights(d, temperature):
    z = d / temperature
    z = z - z.max()
    w = np.exp(z)
    return w / w.sum()


def linear_normalized_weights(d, floor=0.3):
    # map d (can be negative) into [floor, 1] via min-max, then normalize to mean 1
    d_shift = (d - d.min()) / (d.max() - d.min() + 1e-8)
    w = floor + (1 - floor) * d_shift
    return w / w.mean()


def effective_sample_size_fraction(w_normalized_to_sum1, n):
    ess = 1.0 / np.sum(w_normalized_to_sum1 ** 2)
    return ess / n


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not AUX_HEAD_CKPT.exists():
        print(f"Scaled aux head not found at {AUX_HEAD_CKPT} -- cannot run. Exiting.")
        return

    ckpt = torch.load(str(MM_CKPT), map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    aux = AuxHead(in_channels=256).to(device)
    aux.load_state_dict(torch.load(str(AUX_HEAD_CKPT), map_location=device))
    aux.eval()
    print(f"Loaded MM seed0 checkpoint + E71 scaled aux head (hô rho=0.866 held-out, from prior validation).")

    train_ds = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                      split="train", val_split=0.1, target_shape=(64, 64, 64))

    rng = np.random.default_rng(SEED)
    n_total = len(train_ds)
    sample_idx = rng.choice(n_total, size=min(N_SAMPLE, n_total), replace=False)
    print(f"Sampling {len(sample_idx)}/{n_total} training subjects for the sanity check "
          f"(hat{{d}}_i only -- no real ablation needed, using the trained predictor directly).")

    import nibabel as nib

    pred_d = []
    native_sizes = []
    subject_ids = []
    for count, idx in enumerate(sample_idx):
        img, msk, sid = train_ds[int(idx)]
        img_b = img.unsqueeze(0).to(device)
        with torch.no_grad():
            enc1 = model.enc1(img_b)
            pool1 = model.pool1(enc1); enc2 = model.enc2(pool1)
            pool2 = model.pool2(enc2); enc3 = model.enc3(pool2); pool3 = model.pool3(enc3)
            bottleneck = model.bottleneck(pool3)
            d_hat = aux(bottleneck).item()

        subject_dir = train_ds.subject_dirs[int(idx)]
        seg_path = Path(subject_dir) / f"{sid}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        native_size = int((seg_data > 0).sum())

        pred_d.append(d_hat)
        native_sizes.append(native_size)
        subject_ids.append(sid)

        if (count + 1) % 50 == 0:
            print(f"  processed {count+1}/{len(sample_idx)}", flush=True)

    pred_d = np.array(pred_d)
    native_sizes = np.array(native_sizes)
    print(f"\nhat{{d}}_i over sample: mean={pred_d.mean():.4f} std={pred_d.std():.4f} "
          f"range=[{pred_d.min():.4f}, {pred_d.max():.4f}]")

    results = {}
    for scheme_name, w in [
        ("softmax_T1.0", softmax_weights(pred_d, temperature=1.0)),
        ("softmax_T0.5", softmax_weights(pred_d, temperature=0.5)),
        ("linear_normalized_floor0.3", linear_normalized_weights(pred_d, floor=0.3) / len(pred_d)),
    ]:
        w_sum1 = w / w.sum()
        max_frac = float(w_sum1.max())
        ess_frac = effective_sample_size_fraction(w_sum1, len(w_sum1))
        rho_w_d, p_w_d = stats.spearmanr(w_sum1, pred_d)
        rho_w_size, p_w_size = stats.spearmanr(w_sum1, native_sizes)

        non_degenerate = max_frac <= 0.05
        ess_ok = ess_frac >= 0.5
        weaker_than_size_ref = abs(rho_w_size) < abs(E48_SIZE_RHO_REFERENCE)

        scheme_verdict = "PASS" if (non_degenerate and ess_ok and weaker_than_size_ref) else "FAIL"

        print(f"\n--- Scheme: {scheme_name} ---")
        print(f"  Max single-subject weight fraction: {max_frac:.4f} (want <= 0.05)")
        print(f"  Effective sample size fraction: {ess_frac:.4f} (want >= 0.50)")
        print(f"  Spearman(weight, hat_d_i) = {rho_w_d:+.4f} (p={p_w_d:.2e}) [expected strongly positive]")
        print(f"  Spearman(weight, native_size) = {rho_w_size:+.4f} (p={p_w_size:.2e}) "
              f"[want weaker than E48 ref |{E48_SIZE_RHO_REFERENCE}|]")
        print(f"  Scheme verdict: {scheme_verdict}")

        results[scheme_name] = {
            "max_weight_fraction": max_frac, "ess_fraction": ess_frac,
            "rho_weight_vs_predicted_d": float(rho_w_d), "p_weight_vs_predicted_d": float(p_w_d),
            "rho_weight_vs_size": float(rho_w_size), "p_weight_vs_size": float(p_w_size),
            "non_degenerate": bool(non_degenerate), "ess_ok": bool(ess_ok),
            "weaker_than_size_reference": bool(weaker_than_size_ref),
            "verdict": scheme_verdict,
        }

    overall_pass = any(r["verdict"] == "PASS" for r in results.values())
    print(f"\n=== OVERALL SANITY CHECK: {'PASS (at least one scheme viable)' if overall_pass else 'FAIL (all schemes degenerate/unsuitable)'} ===")
    if overall_pass:
        best = max(results.items(), key=lambda kv: kv[1]["verdict"] == "PASS")
        print("Proceed to a short pilot training run using a PASSing scheme "
              "(prefer the one with the best ESS fraction among PASSing schemes).")
    else:
        print("Do NOT proceed to a pilot training run. Weighting scheme(s) as designed are degenerate "
              "or reduce to a size proxy -- revisit the weighting function before any training.")

    summary = {
        "n_sampled": len(sample_idx), "e48_size_rho_reference": E48_SIZE_RHO_REFERENCE,
        "schemes": results, "overall_pass": bool(overall_pass),
    }
    with open(OUT_DIR / "E72_fwl_weighting_sanity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E72_fwl_weighting_sanity_summary.json")


if __name__ == "__main__":
    main()
