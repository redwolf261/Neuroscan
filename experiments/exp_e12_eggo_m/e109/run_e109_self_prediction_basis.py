"""
Phase E109: fresh replication + mechanism probe for E71 prediction-1's
"bottleneck self-predictive signal" anomaly, flagged as proven-but-
unexplained in the Research Knowledge Map (docs/RESEARCH_KNOWLEDGE_MAP.md,
anomaly #4).

WHY A FRESH REPLICATION (not reuse of E71's own saved artifacts):
E71 used UNet3D_v3 + the E70 "MM" (multimodal) checkpoint
(experiments/exp_e12_eggo_m/e70/runs/MM_seed0/checkpoints/best.pth).
That checkpoint file no longer exists on disk (verified via `find`
before writing this script) -- so E71's original aux-head weights
cannot be paired with real forward-pass activations from the model that
produced them. Rather than analyze stale weights against nothing, this
phase replicates the CORE finding (a small probe on the frozen
bottleneck predicts the network's own E48-style causal ablation
sensitivity, held out) fresh on THIS project's canonical checkpoint
(UNet3D_v5, experiments/exp_e12_eggo_m/e46/runs/AttnGate_seed0/
checkpoints/best.pth, val_dice=0.9101624600589275 -- same checkpoint as
E85-E108), which also serves as an independent generalization check:
does the self-predictive signal hold on a different architecture/
checkpoint family, not just the one E71 originally used?

MECHANISM QUESTION (the actual anomaly): once the probe is trained and
confirmed to predict N_b on held-out subjects, WHAT does it key on?
Candidates:
  (a) magnitude-coded: probe weight concentrates on channels with high
      mean activation magnitude (per-channel).
  (b) variance-coded: probe weight concentrates on channels with high
      cross-subject variance (channels that vary a lot are more
      informative for predicting a per-subject quantity).
  (c) neither -- distributed across many channels with no single
      dominant statistical correlate (a genuine "still unexplained"
      residual, reported honestly as such).

METHOD:
  1. Load canonical v5 checkpoint, frozen.
  2. For N_TRAIN training subjects + all 125 val subjects: forward pass,
     extract the (256, 8, 8, 8) bottleneck tensor, and compute N_b via
     the SAME bottleneck-zeroing ablation construction as E48/E89
     (decode with intact vs zeroed bottleneck, Dice drop = N_b).
  3. Train a small aux head (global-avg-pool -> 256 -> 64 -> 1) on
     TRAINING subjects only to predict N_b from the pooled bottleneck.
  4. Evaluate held-out Spearman(predicted, measured) on the 125 val
     subjects, with a permutation test and a partial correlation
     controlling for native_size (matching E71/E48's own convention).
  5. If (and only if) step 4 passes: compute an empirical per-channel
     sensitivity score for the trained aux head (gradient of predicted
     N_b w.r.t. each of the 256 pooled bottleneck channels, averaged
     over the training subjects -- exact, since it's a real gradient,
     not an approximation through the ReLU). Correlate this per-channel
     sensitivity against two per-channel statistics computed from the
     SAME forward passes: (a) mean activation magnitude across subjects
     and channels' spatial extent, (b) cross-subject variance of the
     pooled activation. Report which (if either) explains the probe's
     channel weighting.

PRE-DECLARED DECISION RULE:
  Step 4 PASS condition: same as E71 original -- held-out Spearman
  significantly positive (permutation p<0.05), not fully explained by
  native_size alone (partial correlation controlling size still
  positive and nominally significant).
  If step 4 FAILS: the anomaly does not replicate on this architecture;
  report as a genuine non-replication, do not proceed to step 5.
  If step 4 PASSES: step 5's channel-sensitivity vs magnitude/variance
  correlation is reported honestly regardless of outcome -- a null
  result here (neither magnitude nor variance explains the weighting)
  is itself informative (rules out the two simplest hypotheses) and
  will be reported as such, not suppressed.
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

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000
N_TRAIN_SUBJECTS_FOR_LABELS = 200
AUX_EPOCHS = 30

CKPT_PATH = (project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs"
             / "AttnGate_seed0" / "checkpoints" / "best.pth")
EXPECTED_VAL_DICE = 0.9101624600589275


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    if denom == 0:
        return 1.0
    return float(2 * tp / denom)


def fractional_occupancy_64(seg_binary_native, shape=(64, 64, 64)):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=shape, mode="area").squeeze().numpy()
    return frac


class AuxHead(nn.Module):
    """Same minimal design as E71's original: global-avg-pool -> 2-layer
    MLP. Kept deliberately identical so the step-5 channel-sensitivity
    analysis is comparable in spirit to what E71 would have measured."""
    def __init__(self, in_channels=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(in_channels, 64)
        self.fc2 = nn.Linear(64, 1)

    def pooled(self, bottleneck):
        return self.pool(bottleneck).flatten(1)

    def forward_from_pooled(self, pooled):
        x = F.relu(self.fc1(pooled))
        return self.fc2(x).squeeze(-1)

    def forward(self, bottleneck):
        return self.forward_from_pooled(self.pooled(bottleneck))


def forward_with_bottleneck_ablation(model, image, ablate, device):
    """Bit-for-bit the SAME construction as E48's own (verified) ablation
    (experiments/exp_e12_eggo_m/e48/run_e48_bottleneck_encoding_audit.py,
    forward_with_bottleneck_ablation, lines 105-152) -- NOT the version
    originally written for this script, which incorrectly assumed the
    attention gate should read the intact bottleneck. E48's own explicit
    reasoning (kept verbatim as a NOTE there): "if the bottleneck carries
    no real signal, the gate itself should also degrade to whatever a
    zero-input gate produces, which IS the intended full severing of the
    coarse pathway's influence." So when ablate=True, bottleneck is
    zeroed BEFORE being used for both upconv3 AND the gate -- the
    ORIGINAL version of this function (which read `gate=bottleneck` from
    a variable that was never reassigned to the ablated tensor) was a
    real bug, caught by comparing this run's freshly-computed N_b
    distribution (mean=0.0223, std=0.054) against E48's own stored table
    on the SAME 125 validation subjects/checkpoint (mean=0.3205,
    std=0.174) -- a 14x mean discrepancy too large to be provenance
    noise (cf. the much smaller E85/E86 provenance gap), diagnosed and
    fixed before trusting the DOES_NOT_REPLICATE verdict it produced."""
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)

        if ablate:
            bottleneck = torch.zeros_like(bottleneck)

        upconv3 = model.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = model.dec3(cat3)

        upconv2 = model.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = model.dec2(cat2)

        upconv1 = model.upconv1(dec2)

        gate = bottleneck  # possibly-ablated, matching E48's own construction exactly
        skip = enc1
        g = model.attn_gate1.W_g(gate)
        g_up = F.interpolate(g, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = model.attn_gate1.W_x(skip)
        psi = torch.sigmoid(model.attn_gate1.W_psi(F.relu(g_up + x)))

        enc1_gated = enc1 * psi
        cat1 = torch.cat([upconv1, enc1_gated], dim=1)
        dec1 = model.dec1(cat1)
        probs = model.seg_head(dec1)

    return probs.squeeze(0).squeeze(0).cpu().numpy(), bottleneck.squeeze(0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    val_dice = ckpt.get("best_val_dice") or ckpt.get("val_dice")
    print(f"Loaded canonical checkpoint: val_dice={val_dice}")
    if val_dice is not None:
        assert abs(float(val_dice) - EXPECTED_VAL_DICE) < 1e-6, \
            f"Checkpoint identity check FAILED: expected {EXPECTED_VAL_DICE}, got {val_dice}"
        print("[Sanity check] checkpoint identity PASS.")

    # ---------------- Sanity check: manual trunk matches real forward() ----------------
    val_ds = BraTSDataset(root_dir="Dataset/Training", split="val", val_split=0.1,
                          target_shape=(64, 64, 64), normalize=True)
    train_ds = BraTSDataset(root_dir="Dataset/Training", split="train", val_split=0.1,
                            target_shape=(64, 64, 64), normalize=True)

    with torch.no_grad():
        img0, _, _ = val_ds[0]
        img0_b = img0.unsqueeze(0).to(device)
        real_out = model(img0_b)
        real_probs = real_out["probs"] if isinstance(real_out, dict) else real_out
        manual_probs, _ = forward_with_bottleneck_ablation(model, img0_b, ablate=False, device=device)
        real_np = real_probs.squeeze(0).squeeze(0).cpu().numpy() if torch.is_tensor(real_probs) else np.asarray(real_probs)
        max_diff = float(np.abs(real_np - manual_probs).max())
    print(f"[Sanity check] manual trunk vs real forward(): max abs diff = {max_diff:.6e}")
    assert max_diff < 1e-4, "Manual trunk mismatch -- STOP."
    print("[Sanity check] PASS.\n")

    # ---------------- Step 1-2: build (bottleneck, N_b) label pairs ----------------
    def build_dataset(dataset, n_limit, tag):
        records = []
        rng = np.random.default_rng(SEED)
        indices = rng.choice(len(dataset), size=min(n_limit, len(dataset)), replace=False)
        for count, idx in enumerate(indices):
            image, mask, sid = dataset[idx]
            image_b = image.unsqueeze(0).to(device)

            subject_dir = dataset.subject_dirs[idx]
            seg_path = Path(subject_dir) / f"{sid}-seg.nii.gz"
            seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
            seg_binary_native = (seg_data > 0).astype(np.float32)
            native_size = int(seg_binary_native.sum())
            target_bin = (fractional_occupancy_64(seg_binary_native) > 0.5).astype(np.float32)

            probs_intact, bottleneck = forward_with_bottleneck_ablation(model, image_b, ablate=False, device=device)
            probs_ablated, _ = forward_with_bottleneck_ablation(model, image_b, ablate=True, device=device)
            dice_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin)
            dice_ablated = dice_score((probs_ablated >= 0.5).astype(np.float32), target_bin)
            n_b = dice_intact - dice_ablated

            records.append({
                "subject_id": sid, "bottleneck": bottleneck.cpu(),
                "n_b": n_b, "native_size": native_size,
            })
            if (count + 1) % 50 == 0:
                print(f"  [{tag}] labeled {count+1}/{len(indices)}", flush=True)
        return records

    print(f"Building training labels ({N_TRAIN_SUBJECTS_FOR_LABELS} training subjects, real ablation)...")
    train_records = build_dataset(train_ds, N_TRAIN_SUBJECTS_FOR_LABELS, "train")
    print(f"\nBuilding validation labels (all {len(val_ds)} held-out subjects, real ablation)...")
    val_records = build_dataset(val_ds, len(val_ds), "val")

    n_train = np.array([r["n_b"] for r in train_records])
    n_val = np.array([r["n_b"] for r in val_records])
    print(f"\nTraining N_b: mean={n_train.mean():.4f} std={n_train.std():.4f}")
    print(f"Validation N_b: mean={n_val.mean():.4f} std={n_val.std():.4f}")

    # ---------------- Step 3: train the aux head on TRAINING subjects only ----------------
    aux = AuxHead(in_channels=256).to(device)
    optimizer = torch.optim.Adam(aux.parameters(), lr=1e-3)

    train_bn = torch.stack([r["bottleneck"] for r in train_records]).to(device)
    train_n = torch.tensor(n_train, dtype=torch.float32, device=device)

    print(f"\nTraining auxiliary head ({AUX_EPOCHS} epochs, {len(train_records)} labeled subjects)...")
    for epoch in range(AUX_EPOCHS):
        aux.train()
        optimizer.zero_grad(set_to_none=True)
        pred = aux(train_bn)
        loss = F.mse_loss(pred, train_n)
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

    rho, p_param = stats.spearmanr(pred_val, n_val)
    rng = np.random.default_rng(SEED)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng.permutation(n_val)
        perm_rhos[i], _ = stats.spearmanr(pred_val, perm_y)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())

    print(f"\n=== Prediction fidelity on HELD-OUT validation subjects (n={len(val_records)}) ===")
    print(f"Spearman(predicted N_b, measured N_b) = {rho:+.4f} (parametric p={p_param:.4e}, permutation p={p_perm:.4f})")

    log_size = np.log(native_size_val + 1)
    X1 = np.column_stack([np.ones(len(log_size)), log_size])
    beta_p, *_ = np.linalg.lstsq(X1, pred_val, rcond=None)
    resid_pred = pred_val - X1 @ beta_p
    beta_n, *_ = np.linalg.lstsq(X1, n_val, rcond=None)
    resid_n = n_val - X1 @ beta_n
    rho_partial, p_partial = stats.spearmanr(resid_pred, resid_n)
    print(f"Partial Spearman (controlling native_size) = {rho_partial:+.4f} (p={p_partial:.4e})")

    passes = (rho > 0) and (p_perm < 0.05) and (rho_partial > 0) and (p_partial < 0.05)
    verdict = "REPLICATES" if passes else "DOES_NOT_REPLICATE"
    print(f"\n=== STEP 4 VERDICT: {verdict} ===")

    summary = {
        "checkpoint": str(CKPT_PATH), "val_dice_check": val_dice,
        "n_train_labels": len(train_records), "n_val_labels": len(val_records),
        "n_train_mean": float(n_train.mean()), "n_train_std": float(n_train.std()),
        "n_val_mean": float(n_val.mean()), "n_val_std": float(n_val.std()),
        "spearman_rho": float(rho), "parametric_p": float(p_param), "permutation_p": p_perm,
        "partial_rho_controlling_size": float(rho_partial), "partial_p": float(p_partial),
        "step4_verdict": verdict,
    }

    if not passes:
        print("\nDoes not replicate on this checkpoint/architecture -- STOPPING before step 5, "
              "per pre-declared decision rule.")
        with open(OUT_DIR / "E109_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return

    # ---------------- Step 5: per-channel sensitivity vs magnitude/variance ----------------
    print("\n=== Step 5: channel-sensitivity mechanism probe ===")

    # Exact empirical gradient of the aux head's output w.r.t. each of the
    # 256 pooled input channels, averaged over training subjects (real
    # autograd gradient through fc1->ReLU->fc2, not an approximation).
    train_pooled = aux.pool(train_bn).flatten(1).detach().clone().requires_grad_(True)
    pred_from_pooled = aux.forward_from_pooled(train_pooled)
    pred_from_pooled.sum().backward()
    channel_sensitivity = train_pooled.grad.abs().mean(dim=0).cpu().numpy()  # (256,)

    pooled_np = aux.pool(train_bn).flatten(1).detach().cpu().numpy()  # (n_train, 256)
    channel_mean_magnitude = np.abs(pooled_np).mean(axis=0)  # (256,)
    channel_cross_subject_variance = pooled_np.var(axis=0)   # (256,)

    rho_mag, p_mag = stats.spearmanr(channel_sensitivity, channel_mean_magnitude)
    rho_var, p_var = stats.spearmanr(channel_sensitivity, channel_cross_subject_variance)

    print(f"Spearman(channel_sensitivity, mean |activation|) = {rho_mag:+.4f} (p={p_mag:.4e})")
    print(f"Spearman(channel_sensitivity, cross-subject variance) = {rho_var:+.4f} (p={p_var:.4e})")

    # Concentration check: what fraction of total sensitivity is carried
    # by the top-10% most-sensitive channels? (distributed vs concentrated)
    order = np.argsort(-channel_sensitivity)
    top10pct_n = max(1, len(channel_sensitivity) // 10)
    top10_fraction = float(channel_sensitivity[order[:top10pct_n]].sum() / channel_sensitivity.sum())
    print(f"Fraction of total sensitivity in top 10% of channels ({top10pct_n}/{len(channel_sensitivity)}): {top10_fraction:.4f}")

    if abs(rho_mag) > 0.3 and p_mag < 0.05:
        mechanism_reading = "MAGNITUDE_CODED"
    elif abs(rho_var) > 0.3 and p_var < 0.05:
        mechanism_reading = "VARIANCE_CODED"
    else:
        mechanism_reading = "NEITHER_CLEAN_UNEXPLAINED"
    print(f"\n=== STEP 5 READING: {mechanism_reading} ===")
    print("(Note: this classification is a convenience label only -- read the actual "
          "rho/p values above directly, do not trust this label alone, per this "
          "project's own established convention of catching auto-classifier mistakes.)")

    summary.update({
        "channel_sensitivity_vs_magnitude_rho": float(rho_mag), "channel_sensitivity_vs_magnitude_p": float(p_mag),
        "channel_sensitivity_vs_variance_rho": float(rho_var), "channel_sensitivity_vs_variance_p": float(p_var),
        "top10pct_sensitivity_fraction": top10_fraction,
        "mechanism_reading_CONVENIENCE_LABEL_ONLY": mechanism_reading,
    })
    with open(OUT_DIR / "E109_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E109_summary.json")


if __name__ == "__main__":
    main()
