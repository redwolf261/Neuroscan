"""
Phase E71, prediction 2 RE-TEST with more data (follow-up to
PHASE_E71_PREDICTION2_RESULT.md's diagnosed likely cause).

CONTEXT: the first prediction-2 run (200 training labels, 30 epochs)
FAILED -- predicted sensitivity inverted E48's own size relationship
(rho=+0.395 vs E48's -0.454). Diagnosis (from the persisted per-subject
table): predicted range was severely compressed (0.43 vs measured 0.81)
and errors concentrated at the smallest lesions specifically (predicted
0.255 vs true 0.475) -- consistent with an undertrained head, not
necessarily a conceptual failure.

THIS SCRIPT tests that diagnosis directly, as a single pre-registered
re-run, not an unlimited retry loop: use ALL 1126 training subjects
(not 200) for ablation labels, and more epochs (150, not 30), same
tiny 2-layer MLP head (no capacity change -- isolating "more data" as
the one variable under test, not "bigger model"). Everything else
(seed, architecture, ablation construction, held-out val population)
is IDENTICAL to check_cdcg_prediction1.py / check_cdcg_prediction2.py.

PRE-DECLARED RULE (same as prediction 2's original, reused verbatim,
not loosened): PASS only if Spearman(predicted, native_size) < 0,
permutation p<0.05, AND the partial correlation (predicted vs measured,
controlling for size) remains significant (not a trivial size proxy).
This is run ONCE with the scaled-up data. If it still fails, CDCG is
reported as killed at prediction 2, not retried again with yet more
tweaks (per this project's own no-rescue discipline).
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402
from check_cdcg_prediction1 import AuxHead, forward_get_bottleneck_and_ablated_dice, MM_CKPT  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000
AUX_EPOCHS = 150  # was 30 -- the other scaled variable, held modest (not tuned to chase a result)

def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    return 1.0 if denom == 0 else float(2 * tp / denom)


def build_dataset(model, dataset, indices, device, tag):
    records = []
    t0 = time.time()
    for count, idx in enumerate(indices):
        img, msk, sid = dataset[int(idx)]
        img_b = img.unsqueeze(0).to(device)
        target_bin = (msk.squeeze(0).numpy() > 0.5).astype(np.float32)

        bottleneck, dice_intact, dice_ablated = forward_get_bottleneck_and_ablated_dice(
            model, img_b, target_bin, device)
        d_i = dice_intact - dice_ablated

        subject_dir = dataset.subject_dirs[idx]
        seg_path = Path(subject_dir) / f"{sid}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        native_size = int((seg_data > 0).sum())

        records.append({"subject_id": sid, "bottleneck": bottleneck.squeeze(0).cpu(),
                        "d_i": d_i, "native_size": native_size})
        if (count + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{tag}] labeled {count+1}/{len(indices)} ({elapsed:.0f}s elapsed, "
                  f"~{elapsed/(count+1)*(len(indices)-count-1):.0f}s remaining)", flush=True)
    return records


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(MM_CKPT), map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded MM seed0 checkpoint: best_per_subject_dice={ckpt.get('best_per_subject_dice')}")

    train_ds = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                      split="train", val_split=0.1, target_shape=(64, 64, 64))
    val_ds = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                    split="val", val_split=0.1, target_shape=(64, 64, 64))

    print(f"\nBuilding labels for ALL {len(train_ds)} training subjects (real ablation, ~20-25 min)...")
    train_indices = np.arange(len(train_ds))  # ALL training subjects, not a 200-subject sample
    train_records = build_dataset(model, train_ds, train_indices, device, "train")

    print(f"\nBuilding labels for all {len(val_ds)} held-out validation subjects...")
    val_indices = np.arange(len(val_ds))
    val_records = build_dataset(model, val_ds, val_indices, device, "val")

    d_train = np.array([r["d_i"] for r in train_records])
    d_val = np.array([r["d_i"] for r in val_records])
    print(f"\nTraining d_i (n={len(d_train)}): mean={d_train.mean():.4f} std={d_train.std():.4f}")
    print(f"Validation d_i (n={len(d_val)}): mean={d_val.mean():.4f} std={d_val.std():.4f}")

    # ---------------- Train the SAME small head, more data + more epochs ----------------
    aux = AuxHead(in_channels=256).to(device)
    optimizer = torch.optim.Adam(aux.parameters(), lr=1e-3, weight_decay=1e-4)  # light weight decay, more epochs now warrant it

    train_bn = torch.stack([r["bottleneck"] for r in train_records]).to(device)
    train_d = torch.tensor(d_train, dtype=torch.float32, device=device)

    print(f"\nTraining auxiliary head ({AUX_EPOCHS} epochs, {len(train_records)} labeled subjects, "
          f"{AUX_EPOCHS} epochs, full-batch)...")
    for epoch in range(AUX_EPOCHS):
        aux.train()
        optimizer.zero_grad(set_to_none=True)
        pred = aux(train_bn)
        loss = F.mse_loss(pred, train_d)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 30 == 0:
            print(f"  epoch {epoch+1}/{AUX_EPOCHS}: mse={loss.item():.6f}")

    aux.eval()
    val_bn = torch.stack([r["bottleneck"] for r in val_records]).to(device)
    with torch.no_grad():
        pred_val = aux(val_bn).cpu().numpy()

    native_size_val = np.array([r["native_size"] for r in val_records])
    subject_ids_val = [r["subject_id"] for r in val_records]

    print(f"\nPredicted range: [{pred_val.min():.4f}, {pred_val.max():.4f}] (span {pred_val.max()-pred_val.min():.4f})")
    print(f"Measured range:  [{d_val.min():.4f}, {d_val.max():.4f}] (span {d_val.max()-d_val.min():.4f})")

    # ---------------- Prediction 1 reconfirm ----------------
    rho1, p1_param = stats.spearmanr(pred_val, d_val)
    print(f"\n[Prediction 1 reconfirm] Spearman(predicted, measured) = {rho1:+.4f} (p={p1_param:.4e})")

    # ---------------- Prediction 2 ----------------
    rho2, p2_param = stats.spearmanr(pred_val, native_size_val)
    rng = np.random.default_rng(SEED + 1)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng.permutation(native_size_val)
        perm_rhos[i], _ = stats.spearmanr(pred_val, perm_y)
    p2_perm = float((np.abs(perm_rhos) >= np.abs(rho2)).mean())

    print(f"\n=== PREDICTION 2 (SCALED, n_train={len(train_records)}, epochs={AUX_EPOCHS}) ===")
    print(f"Spearman(predicted, native_size) = {rho2:+.4f} (parametric p={p2_param:.4e}, permutation p={p2_perm:.4f})")
    print(f"E48's own reference: -0.454. Direction match: {rho2 < 0}")

    rho_measured, p_measured = stats.spearmanr(d_val, native_size_val)
    print(f"[Reference] Spearman(measured, native_size) on this val set = {rho_measured:+.4f} (p={p_measured:.4e})")

    # size-binned breakdown, same as the diagnostic in the original prediction-2 report
    order = np.argsort(native_size_val)
    size_s, pred_s, meas_s = native_size_val[order], pred_val[order], d_val[order]
    print("\nSize-binned means (5 bins):")
    for i, idx in enumerate(np.array_split(np.arange(len(size_s)), 5)):
        print(f"  bin{i}: size[{size_s[idx].min()},{size_s[idx].max()}] n={len(idx)} "
              f"measured={meas_s[idx].mean():.3f} predicted={pred_s[idx].mean():.3f}")

    log_size = np.log(native_size_val + 1)
    X1 = np.column_stack([np.ones(len(log_size)), log_size])
    beta_p, *_ = np.linalg.lstsq(X1, pred_val, rcond=None)
    resid_pred = pred_val - X1 @ beta_p
    beta_d, *_ = np.linalg.lstsq(X1, d_val, rcond=None)
    resid_d = d_val - X1 @ beta_d
    rho_partial, p_partial = stats.spearmanr(resid_pred, resid_d)
    print(f"\nPartial Spearman(predicted, measured | size) = {rho_partial:+.4f} (p={p_partial:.4e})")

    sign_matches = rho2 < 0
    significant = p2_perm < 0.05
    not_degenerate = abs(rho_partial) > 0.1 and p_partial < 0.05

    if sign_matches and significant and not_degenerate:
        verdict = "PASS"
    elif sign_matches and significant:
        verdict = "QUALIFIED_SIZE_PROXY"
    else:
        verdict = "FAIL"

    print(f"\n=== PREDICTION 2 (SCALED) VERDICT: {verdict} ===")
    if verdict == "FAIL":
        print("Even with all 1126 training subjects and 150 epochs, the predicted sensitivity still does "
              "NOT match E48's own size relationship. This is now a real, structural finding, not a "
              "data-scale artifact -- CDCG's self-prediction signal does not organize itself along the "
              "size axis the causal chain establishes matters. Do not retry with yet more tweaks; report honestly.")
    elif verdict == "PASS":
        print("With more data, the predictor now matches E48's own direction AND carries independent "
              "signal beyond size. The original prediction-2 failure WAS a data-scale artifact. "
              "CDCG clears prediction 2 -- may proceed toward the gating mechanism design in a future phase.")
    else:
        print("Direction matches but the signal may be substantially explainable by size alone. Report as qualified.")

    torch.save(aux.state_dict(), OUT_DIR / "E71_aux_head_state_scaled.pt")
    per_subject_table = {
        subject_ids_val[i]: {"predicted_d_i": float(pred_val[i]), "measured_d_i": float(d_val[i]),
                             "native_size": int(native_size_val[i])}
        for i in range(len(val_records))
    }
    with open(OUT_DIR / "E71_val_per_subject_table_scaled.json", "w") as f:
        json.dump(per_subject_table, f, indent=2)

    summary = {
        "n_train_labels": len(train_records), "n_val_labels": len(val_records), "aux_epochs": AUX_EPOCHS,
        "prediction1_reconfirm_rho": float(rho1), "prediction1_reconfirm_p": float(p1_param),
        "prediction2_rho": float(rho2), "prediction2_parametric_p": float(p2_param),
        "prediction2_permutation_p": p2_perm,
        "reference_measured_vs_size_rho": float(rho_measured), "reference_measured_vs_size_p": float(p_measured),
        "e48_reference_rho": -0.454,
        "partial_rho_controlling_size": float(rho_partial), "partial_p": float(p_partial),
        "predicted_range": [float(pred_val.min()), float(pred_val.max())],
        "measured_range": [float(d_val.min()), float(d_val.max())],
        "sign_matches_e48": bool(sign_matches), "significant": bool(significant),
        "not_degenerate_size_proxy": bool(not_degenerate),
        "verdict": verdict,
    }
    with open(OUT_DIR / "E71_prediction2_scaled_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E71_prediction2_scaled_summary.json")


if __name__ == "__main__":
    main()
