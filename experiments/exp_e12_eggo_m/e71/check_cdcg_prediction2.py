"""
Phase E71, prediction 2 (Section 5 of PHASE_E71_CDCG_DESIGN.md):
mechanism check -- does the predicted sensitivity hat{d}_i behave
sensibly relative to E48's own independently-established finding?

CONTEXT: prediction 1 (check_cdcg_prediction1.py) PASSED cleanly
overnight -- held-out Spearman(predicted, measured) = +0.70 (p<0.001),
partial correlation controlling for native_size = +0.90 (p=4e-47). That
established the auxiliary head CAN predict its own causal ablation
sensitivity. It does NOT establish that the predicted signal is doing
so for a sensible reason rather than some other, unrelated regularity.

PRE-DECLARED TEST (verbatim from the design doc, Section 5, item 2):
  Does hat{d}_i correlate with native lesion size in the SAME DIRECTION
  as E48's own established finding? E48 found Spearman(native_size,
  bottleneck-ablation drop) = -0.454 (p<0.001) -- SMALLER lesions have
  LARGER causal drop, i.e. depend MORE on the bottleneck. If CDCG's
  hat{d}_i is a sensible, non-degenerate signal, it should show the
  SAME negative relationship with native_size on the same population.

This is explicitly NOT the same question as prediction 1's partial
correlation (which asked whether hat{d}_i predicts d_i BEYOND what size
alone would predict). Prediction 2 asks whether hat{d}_i's OWN raw
relationship with size matches the established causal finding's sign
and rough magnitude -- a sanity/mechanism check, not a novel test.

PRE-DECLARED DECISION RULE:
  PASS if Spearman(hat{d}_i, native_size) < 0, permutation p < 0.05,
  AND the relationship is not degenerate (hat{d}_i is not literally
  constant or a trivial monotonic transform of size alone -- checked
  by confirming prediction 1's already-established partial correlation,
  controlling for size, remains significant; re-verified here rather
  than assumed).
  FAIL / QUALIFIED if the sign is wrong, non-significant, or hat{d}_i
  turns out to be explainable ENTIRELY by size (partial correlation
  collapses toward 0) -- report honestly either way, matching the
  discipline that correctly killed CAS's own gate-vs-sensitivity claim.

Reuses check_cdcg_prediction1.py's own labeling/training pipeline
(SAME seed, SAME train/val subject split, SAME ablation construction)
rather than reimplementing it, and additionally PERSISTS the trained
auxiliary head + per-subject predictions/labels/sizes to disk, so
future analyses (e.g. a size-quartile breakdown) do not need to
re-run the (moderately expensive) ablation-labeling step again.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402
from check_cdcg_prediction1 import (  # noqa: E402
    AuxHead, forward_get_bottleneck_and_ablated_dice, MM_CKPT,
    SEED, N_PERM, N_TRAIN_SUBJECTS_FOR_LABELS, AUX_EPOCHS,
)

OUT_DIR = Path(__file__).parent


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not MM_CKPT.exists():
        print(f"MM seed0 checkpoint not found at {MM_CKPT} -- cannot run. Exiting.")
        return

    ckpt = torch.load(str(MM_CKPT), map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded MM seed0 checkpoint: best_per_subject_dice={ckpt.get('best_per_subject_dice')}")

    import nibabel as nib

    def dice_score(pred_bin, target_bin):
        tp = (pred_bin * target_bin).sum()
        denom = pred_bin.sum() + target_bin.sum()
        return 1.0 if denom == 0 else float(2 * tp / denom)

    val_ds = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                    split="val", val_split=0.1, target_shape=(64, 64, 64))
    train_ds = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                      split="train", val_split=0.1, target_shape=(64, 64, 64))

    def build_dataset(dataset, n_limit, tag):
        records = []
        rng = np.random.default_rng(SEED)
        indices = rng.choice(len(dataset), size=min(n_limit, len(dataset)), replace=False)
        for count, idx in enumerate(indices):
            img, msk, sid = dataset[idx]
            img_b = img.unsqueeze(0).to(device)
            target_bin = (msk.squeeze(0).numpy() > 0.5).astype(np.float32)

            bottleneck, dice_intact, dice_ablated = forward_get_bottleneck_and_ablated_dice(
                model, img_b, target_bin, device)
            d_i = dice_intact - dice_ablated

            subject_dir = dataset.subject_dirs[idx]
            seg_path = Path(subject_dir) / f"{sid}-seg.nii.gz"
            seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
            native_size = int((seg_data > 0).sum())

            records.append({
                "subject_id": sid, "bottleneck": bottleneck.squeeze(0).cpu(),
                "d_i": d_i, "native_size": native_size,
            })
            if (count + 1) % 50 == 0:
                print(f"  [{tag}] labeled {count+1}/{len(indices)}", flush=True)
        return records

    print(f"Rebuilding SAME train/val label sets as prediction 1 (identical seed, identical split)...")
    train_records = build_dataset(train_ds, N_TRAIN_SUBJECTS_FOR_LABELS, "train")
    val_records = build_dataset(val_ds, len(val_ds), "val")

    d_train = np.array([r["d_i"] for r in train_records])
    d_val = np.array([r["d_i"] for r in val_records])

    aux = AuxHead(in_channels=256).to(device)
    optimizer = torch.optim.Adam(aux.parameters(), lr=1e-3)
    train_bn = torch.stack([r["bottleneck"] for r in train_records]).to(device)
    train_d = torch.tensor(d_train, dtype=torch.float32, device=device)

    print(f"\nRetraining auxiliary head (identical recipe to prediction 1)...")
    for epoch in range(AUX_EPOCHS):
        aux.train()
        optimizer.zero_grad(set_to_none=True)
        pred = aux(train_bn)
        loss = F.mse_loss(pred, train_d)
        loss.backward()
        optimizer.step()
    print(f"  final train mse: {loss.item():.6f}")

    aux.eval()
    val_bn = torch.stack([r["bottleneck"] for r in val_records]).to(device)
    with torch.no_grad():
        pred_val = aux(val_bn).cpu().numpy()

    native_size_val = np.array([r["native_size"] for r in val_records])
    subject_ids_val = [r["subject_id"] for r in val_records]

    # ---------------- Reconfirm prediction 1 (should match overnight result) ----------------
    rho1, p1_param = stats.spearmanr(pred_val, d_val)
    print(f"\n[Reconfirm] Spearman(predicted d_i, measured d_i) = {rho1:+.4f} (p={p1_param:.4e}) "
          f"-- overnight was +0.7008, small difference expected from re-training stochasticity")

    # ---------------- Prediction 2: hat{d}_i vs native_size, matching E48's own direction ----------------
    rho2, p2_param = stats.spearmanr(pred_val, native_size_val)
    rng = np.random.default_rng(SEED + 1)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng.permutation(native_size_val)
        perm_rhos[i], _ = stats.spearmanr(pred_val, perm_y)
    p2_perm = float((np.abs(perm_rhos) >= np.abs(rho2)).mean())

    print(f"\n=== PREDICTION 2: Spearman(predicted d_i, native_size), n={len(val_records)} ===")
    print(f"rho = {rho2:+.4f} (parametric p={p2_param:.4e}, permutation p={p2_perm:.4f})")
    print(f"E48's own reference finding: Spearman(native_size, MEASURED drop) = -0.454 (p<0.001)")
    print(f"Direction match (both negative): {(rho2 < 0)}")

    # Also report the MEASURED d_i vs native_size on this exact held-out population,
    # as the direct, apples-to-apples comparison to E48's original number.
    rho_measured, p_measured = stats.spearmanr(d_val, native_size_val)
    print(f"\n[Reference check] Spearman(native_size, MEASURED d_i) on this same 125-subject val set "
          f"= {rho_measured:+.4f} (p={p_measured:.4e}) -- compare directly to E48's own -0.454")

    # Re-verify prediction 1's partial correlation (controlling for size) still holds
    log_size = np.log(native_size_val + 1)
    X1 = np.column_stack([np.ones(len(log_size)), log_size])
    beta_p, *_ = np.linalg.lstsq(X1, pred_val, rcond=None)
    resid_pred = pred_val - X1 @ beta_p
    beta_d, *_ = np.linalg.lstsq(X1, d_val, rcond=None)
    resid_d = d_val - X1 @ beta_d
    rho_partial, p_partial = stats.spearmanr(resid_pred, resid_d)
    print(f"\n[Re-verify] Partial Spearman(predicted, measured | size) = {rho_partial:+.4f} (p={p_partial:.4e}) "
          f"-- overnight was +0.9036")

    # ---------------- Decision ----------------
    sign_matches = rho2 < 0
    significant = p2_perm < 0.05
    not_degenerate = abs(rho_partial) > 0.1 and p_partial < 0.05  # signal survives controlling for size

    if sign_matches and significant and not_degenerate:
        verdict = "PASS"
    elif sign_matches and significant and not not_degenerate:
        verdict = "QUALIFIED_SIZE_PROXY"  # matches E48's direction but may just be re-deriving size
    else:
        verdict = "FAIL"

    print(f"\n=== PREDICTION 2 VERDICT: {verdict} ===")
    if verdict == "PASS":
        print("hat{d}_i matches E48's own established direction (smaller lesions -> higher predicted "
              "sensitivity) AND is not reducible to a trivial size proxy (partial correlation with the "
              "measured label survives controlling for size). CDCG's signal behaves sensibly.")
    elif verdict == "QUALIFIED_SIZE_PROXY":
        print("hat{d}_i matches E48's direction, but the partial-correlation check suggests it may be "
              "substantially explainable by size alone rather than carrying independent causal signal. "
              "Report honestly as a qualified pass, not a full pass.")
    else:
        print("hat{d}_i does NOT match E48's own established direction, or the match is not significant. "
              "This is a real inconsistency -- report honestly, do not proceed to implementing the gating "
              "mechanism (Section 3.3) without resolving this.")

    # ---------------- Persist everything for future analyses ----------------
    torch.save(aux.state_dict(), OUT_DIR / "E71_aux_head_state.pt")
    per_subject_table = {
        subject_ids_val[i]: {
            "predicted_d_i": float(pred_val[i]), "measured_d_i": float(d_val[i]),
            "native_size": int(native_size_val[i]),
        } for i in range(len(val_records))
    }
    with open(OUT_DIR / "E71_val_per_subject_table.json", "w") as f:
        json.dump(per_subject_table, f, indent=2)

    summary = {
        "reconfirm_prediction1_rho": float(rho1), "reconfirm_prediction1_p": float(p1_param),
        "prediction2_rho_pred_vs_size": float(rho2), "prediction2_parametric_p": float(p2_param),
        "prediction2_permutation_p": p2_perm,
        "reference_measured_d_vs_size_rho": float(rho_measured), "reference_measured_d_vs_size_p": float(p_measured),
        "e48_reference_rho": -0.454,
        "partial_rho_controlling_size_reverify": float(rho_partial), "partial_p_reverify": float(p_partial),
        "sign_matches_e48": bool(sign_matches), "significant": bool(significant),
        "not_degenerate_size_proxy": bool(not_degenerate),
        "verdict": verdict,
    }
    with open(OUT_DIR / "E71_prediction2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E71_prediction2_summary.json, E71_val_per_subject_table.json, E71_aux_head_state.pt")


if __name__ == "__main__":
    main()
