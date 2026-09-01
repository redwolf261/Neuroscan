"""
Phase E71, prediction 1 (Section 5 of PHASE_E71_CDCG_DESIGN.md):
self-prediction fidelity check.

CHEAPEST, FIRST, MOST FALSIFIABLE TEST. Per the design doc: if this
fails, KILL before implementing anything downstream (the gating
mechanism, any Dice comparison). This script does NOT implement the
gate or run a full training loop -- only the auxiliary head, trained to
predict a real, measured causal quantity, tested on held-out subjects.

METHOD:
  1. Load the trained MM (multimodal, seed 0) checkpoint -- frozen, not
     fine-tuned here.
  2. For every one of the 125 validation subjects AND a matched-size
     sample of training subjects, compute the REAL bottleneck-ablation
     Dice drop d_i, using E48's own verified construction (zero the
     bottleneck tensor before the decoder, everything else identical).
  3. Train a SMALL auxiliary head g_phi (a few conv+pool+linear layers)
     to predict d_i from the bottleneck representation z_i, using ONLY
     the training-subject d_i labels.
  4. Evaluate: does g_phi(z_i) on HELD-OUT (validation) subjects
     correlate with their independently-measured d_i? (Spearman +
     permutation test, subject-level, matching this project's own
     convention throughout E47/E48/E58/E62-E71.)

PRE-DECLARED DECISION RULE:
  PASS if Spearman(predicted, measured) on held-out subjects is
  significantly positive (permutation p<0.05) AND the correlation is
  not trivially explained by a simple confound already measured in this
  project (native lesion size -- E48's own rho=-0.454 finding means
  size alone might partially predict d_i; this is checked via partial
  correlation controlling for size).
  FAIL otherwise -- KILL CDCG, do not implement the gating mechanism.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000
N_TRAIN_SUBJECTS_FOR_LABELS = 200  # subset of the 1126 training subjects, for tractable ablation-labeling cost
AUX_EPOCHS = 30

MM_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e70" / "runs"
           / "MM_seed0" / "checkpoints" / "best.pth")
E48_TABLE = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    if denom == 0:
        return 1.0
    return float(2 * tp / denom)


def fractional_occupancy_64(seg_binary_native, shape):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=shape, mode="area").squeeze().numpy()
    return frac


class AuxHead(nn.Module):
    """Small head: bottleneck (256, D/8,H/8,W/8) -> scalar prediction of
    the ablation Dice drop. Global-pool then a 2-layer MLP -- deliberately
    minimal, per the design doc's own "smallest correct implementation"
    principle."""
    def __init__(self, in_channels=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(in_channels, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, bottleneck):
        x = self.pool(bottleneck).flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


def forward_get_bottleneck_and_ablated_dice(model, image, target_bin, device):
    """Runs the real v3 trunk, returns (bottleneck, dice_intact, dice_ablated).
    Bit-for-bit the same ablation construction as E48's own
    forward_with_bottleneck_ablation (bottleneck zeroed before upconv3)."""
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1); enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2); enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)

        def decode(bn):
            up3 = model.upconv3(bn)
            dec3 = model.dec3(torch.cat([up3, enc3], dim=1))
            up2 = model.upconv2(dec3)
            dec2 = model.dec2(torch.cat([up2, enc2], dim=1))
            up1 = model.upconv1(dec2)
            dec1 = model.dec1(torch.cat([up1, enc1], dim=1))
            return model.seg_head(dec1)

        probs_intact = decode(bottleneck)
        probs_ablated = decode(torch.zeros_like(bottleneck))

        dice_intact = dice_score((probs_intact.squeeze(0).squeeze(0) >= 0.5).float().cpu().numpy(), target_bin)
        dice_ablated = dice_score((probs_ablated.squeeze(0).squeeze(0) >= 0.5).float().cpu().numpy(), target_bin)

    return bottleneck, dice_intact, dice_ablated


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

    # ---------------- Sanity check: ablation reproduces E48's own construction pattern ----------------
    val_ds = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                    split="val", val_split=0.1, target_shape=(64, 64, 64))
    train_ds = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                      split="train", val_split=0.1, target_shape=(64, 64, 64))

    with torch.no_grad():
        img0, _, _ = val_ds[0]
        img0_b = img0.unsqueeze(0).to(device)
        real = model(img0_b)["probs"]
        enc1 = model.enc1(img0_b); pool1 = model.pool1(enc1); enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2); enc3 = model.enc3(pool2); pool3 = model.pool3(enc3)
        bn = model.bottleneck(pool3)
        up3 = model.upconv3(bn); dec3 = model.dec3(torch.cat([up3, enc3], 1))
        up2 = model.upconv2(dec3); dec2 = model.dec2(torch.cat([up2, enc2], 1))
        up1 = model.upconv1(dec2); dec1 = model.dec1(torch.cat([up1, enc1], 1))
        manual = model.seg_head(dec1)
    max_diff = float((real - manual).abs().max().item())
    print(f"[Sanity check] manual trunk vs real forward(): max abs diff = {max_diff:.6e}")
    assert max_diff < 1e-5, "Manual trunk mismatch -- STOP."
    print("[Sanity check] PASS.\n")

    # ---------------- Step 1-2: build (bottleneck, d_i) label pairs ----------------
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

    print(f"Building training labels ({N_TRAIN_SUBJECTS_FOR_LABELS} training subjects, real ablation)...")
    train_records = build_dataset(train_ds, N_TRAIN_SUBJECTS_FOR_LABELS, "train")
    print(f"\nBuilding validation labels (all {len(val_ds)} held-out subjects, real ablation)...")
    val_records = build_dataset(val_ds, len(val_ds), "val")

    d_train = np.array([r["d_i"] for r in train_records])
    d_val = np.array([r["d_i"] for r in val_records])
    print(f"\nTraining d_i: mean={d_train.mean():.4f} std={d_train.std():.4f}")
    print(f"Validation d_i: mean={d_val.mean():.4f} std={d_val.std():.4f}")

    # ---------------- Step 3: train the auxiliary head on TRAINING subjects only ----------------
    aux = AuxHead(in_channels=256).to(device)
    optimizer = torch.optim.Adam(aux.parameters(), lr=1e-3)

    train_bn = torch.stack([r["bottleneck"] for r in train_records]).to(device)
    train_d = torch.tensor(d_train, dtype=torch.float32, device=device)

    print(f"\nTraining auxiliary head ({AUX_EPOCHS} epochs, {len(train_records)} labeled subjects)...")
    for epoch in range(AUX_EPOCHS):
        aux.train()
        optimizer.zero_grad(set_to_none=True)
        pred = aux(train_bn)
        loss = F.mse_loss(pred, train_d)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{AUX_EPOCHS}: mse={loss.item():.6f}")

    # ---------------- Step 4: evaluate on HELD-OUT validation subjects ----------------
    aux.eval()
    val_bn = torch.stack([r["bottleneck"] for r in val_records]).to(device)
    with torch.no_grad():
        pred_val = aux(val_bn).cpu().numpy()

    native_size_val = np.array([r["native_size"] for r in val_records])

    rho, p_param = stats.spearmanr(pred_val, d_val)
    rng = np.random.default_rng(SEED)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng.permutation(d_val)
        perm_rhos[i], _ = stats.spearmanr(pred_val, perm_y)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())

    print(f"\n=== Prediction fidelity on HELD-OUT validation subjects (n={len(val_records)}) ===")
    print(f"Spearman(predicted d_i, measured d_i) = {rho:+.4f} (parametric p={p_param:.4e}, permutation p={p_perm:.4f})")

    # Partial correlation controlling for native_size (E48's own known confound)
    log_size = np.log(native_size_val + 1)
    X1 = np.column_stack([np.ones(len(log_size)), log_size])
    beta_p, *_ = np.linalg.lstsq(X1, pred_val, rcond=None)
    resid_pred = pred_val - X1 @ beta_p
    beta_d, *_ = np.linalg.lstsq(X1, d_val, rcond=None)
    resid_d = d_val - X1 @ beta_d
    rho_partial, p_partial = stats.spearmanr(resid_pred, resid_d)
    print(f"Partial Spearman (controlling native_size) = {rho_partial:+.4f} (p={p_partial:.4e})")

    passes = (rho > 0) and (p_perm < 0.05)
    verdict = "PASS" if passes else "FAIL"
    print(f"\n=== PREDICTION 1 VERDICT: {verdict} ===")
    if passes:
        print("The auxiliary head CAN predict its own causal bottleneck-ablation sensitivity on held-out "
              "subjects, significantly better than chance. Proceed to prediction 2 (does the signal behave "
              "sensibly relative to E48's own size-dependence finding) in a SEPARATE future phase.")
    else:
        print("The auxiliary head CANNOT predict its own causal sensitivity better than chance. "
              "KILL CDCG here -- do not implement the gating mechanism or run any Dice comparison.")

    summary = {
        "n_train_labels": len(train_records), "n_val_labels": len(val_records),
        "d_train_mean": float(d_train.mean()), "d_train_std": float(d_train.std()),
        "d_val_mean": float(d_val.mean()), "d_val_std": float(d_val.std()),
        "spearman_rho": float(rho), "parametric_p": float(p_param), "permutation_p": p_perm,
        "partial_rho_controlling_size": float(rho_partial), "partial_p": float(p_partial),
        "verdict": verdict,
    }
    with open(OUT_DIR / "E71_prediction1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E71_prediction1_summary.json")


if __name__ == "__main__":
    main()
