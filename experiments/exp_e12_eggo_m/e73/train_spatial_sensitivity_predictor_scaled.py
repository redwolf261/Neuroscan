"""
Phase E73 SCALED re-test: same spatial-predictor task, ALL 1126 training
subjects (not 250) and 150 epochs (not 60) -- single pre-registered
scale-up, isolating "more data" as the only variable, matching how E71's
prediction-2 undertrained-head diagnosis was resolved earlier tonight.

Motivation: the 250-label/60-epoch predictor passed both design-time
checks (pooled rho=+0.39, per-subject rho=+0.38) but its DOWNSTREAM
error-correlation was much weaker than the real map's (partial rho=+0.09
vs the real map's +0.88; error/correct ratio 1.07x vs 22.6x) -- consistent
with, though not proof of, an undertrained-head ceiling rather than a true
low ceiling on the signal itself. This is the single re-test to check.

Everything else (architecture, seed, val population, decision rule) is
IDENTICAL to train_spatial_sensitivity_predictor.py. Per the no-rescue
discipline: if the downstream error-correlation is still weak after this,
report honestly -- it is a real ceiling, not a data-scale artifact.

Phase E73: Self-Diagnostic Error Localization -- can the frozen bottleneck
predict WHERE the network is causally vulnerable, not just how much?

Follows the established ladder (per the user's own framing):
  E48    : representation dependence exists (causal ablation, scalar)
  CDCG-1 : bottleneck can predict its own scalar ablation sensitivity (rho=0.87)
  E71    : can't use it to route between pathways (killed)
  E72    : can't use it as a subject-level loss weight (killed)
  E73 probe (done): the SPATIAL sensitivity map S_i(p) is non-trivial
    (not the lesion mask) and correlates with the network's own errors
    far above chance (22.6x ratio, survives controlling for boundary
    distance) -- PROCEED verdict per the pre-declared rule.

THIS SCRIPT: train a small spatial head H(Z_i) -> Ŝ_i to predict the REAL
causal sensitivity map S_i(p) = |P_intact(p) - P_ablated(p)| from the
frozen bottleneck representation alone (256 channels, 8^3 spatial -- the
bottleneck's own native resolution, avoiding upsampling ambiguity). Test
on held-out validation subjects: does Ŝ correlate with the REAL S at the
voxel level, pooled across subjects, and is it not degenerate (not just
predicting the same map regardless of subject)?

NOVELTY POSITION (per tonight's search): the MAP itself (model-space
ablation -> voxel difference) is an occupied post-hoc explanation
technique (Discriminative Attribution from Counterfactuals, Counterfactual
-based Saliency Maps, Contrast-CAM family). What was NOT found: training a
head to predict that map from frozen features as a SELF-SUPERVISED
TRAINING-TIME SIGNAL, distilling what would otherwise require a second
forward+ablation pass into a single cheap forward pass at inference. This
script tests ONLY that distillation step -- no refinement branch yet,
per the "smallest correct implementation" / one-step-at-a-time discipline
that has governed every phase tonight.

PRE-DECLARED DECISION RULE:
  PASS if:
    (a) pooled voxel-level Spearman(predicted, measured) on held-out
        subjects is significantly positive (permutation p<0.05)
    (b) NOT degenerate: per-subject Spearman(predicted, measured), computed
        WITHIN each subject's own volume (controls for the head just
        learning "always predict high near a generic boundary shape"
        rather than something subject-specific), is also significantly
        positive on average across subjects.
  FAIL otherwise -- kill before ever building a refinement branch.
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

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000
N_TRAIN_SUBJECTS_FOR_LABELS = 1126  # ALL training subjects -- the scale-up variable under test
SPATIAL_HEAD_EPOCHS = 150
BOTTLENECK_SPATIAL = 8  # native bottleneck grid at 64^3 input, 3 pool stages

MM_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e70" / "runs"
           / "MM_seed0" / "checkpoints" / "best.pth")


class SpatialSensitivityHead(nn.Module):
    """Small 3D conv head: bottleneck (256, 8,8,8) -> predicted sensitivity
    map (1, 8,8,8). Deliberately minimal -- a couple of 3x3x3 convs, no
    downsampling (already at the target resolution), sigmoid output since
    S in [0,1] (it's an absolute difference of two probability maps)."""
    def __init__(self, in_channels=256):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, bottleneck):
        x = F.relu(self.bn1(self.conv1(bottleneck)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.sigmoid(self.conv3(x))
        return x.squeeze(1)  # (B, 8,8,8)


def decode_from_bottleneck(model, bottleneck, enc1, enc2, enc3):
    up3 = model.upconv3(bottleneck)
    dec3 = model.dec3(torch.cat([up3, enc3], dim=1))
    up2 = model.upconv2(dec3)
    dec2 = model.dec2(torch.cat([up2, enc2], dim=1))
    up1 = model.upconv1(dec2)
    dec1 = model.dec1(torch.cat([up1, enc1], dim=1))
    return model.seg_head(dec1)


def compute_sensitivity_map_8cubed(model, img_b, device):
    """Real causal sensitivity map S_i(p), computed at full 64^3 resolution
    (matching E73 probe's construction exactly), then average-pooled down
    to the bottleneck's own native 8^3 grid -- the prediction target."""
    with torch.no_grad():
        enc1 = model.enc1(img_b)
        pool1 = model.pool1(enc1); enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2); enc3 = model.enc3(pool2); pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)

        probs_intact = decode_from_bottleneck(model, bottleneck, enc1, enc2, enc3)
        probs_ablated = decode_from_bottleneck(model, torch.zeros_like(bottleneck), enc1, enc2, enc3)

        S_full = torch.abs(probs_intact - probs_ablated)  # (1,1,64,64,64)
        S_8 = F.avg_pool3d(S_full, kernel_size=8, stride=8)  # (1,1,8,8,8)
    return bottleneck.squeeze(0), S_8.squeeze(0).squeeze(0)  # (256,8,8,8), (8,8,8)


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

    rng = np.random.default_rng(SEED)
    train_sample_idx = rng.choice(len(train_ds), size=min(N_TRAIN_SUBJECTS_FOR_LABELS, len(train_ds)), replace=False)
    print(f"Labeling {len(train_sample_idx)} training subjects for spatial sensitivity maps...")

    train_bn, train_S = [], []
    for count, idx in enumerate(train_sample_idx):
        img, _, sid = train_ds[int(idx)]
        img_b = img.unsqueeze(0).to(device)
        bn, S = compute_sensitivity_map_8cubed(model, img_b, device)
        train_bn.append(bn.cpu())
        train_S.append(S.cpu())
        if (count + 1) % 50 == 0:
            print(f"  labeled {count+1}/{len(train_sample_idx)}", flush=True)

    train_bn = torch.stack(train_bn).to(device)   # (N, 256, 8,8,8)
    train_S = torch.stack(train_S).to(device)     # (N, 8,8,8)
    print(f"Train tensors: bottleneck {train_bn.shape}, S {train_S.shape}")
    print(f"Train S: mean={train_S.mean().item():.4f} std={train_S.std().item():.4f}")

    print(f"\nLabeling {len(val_ds)} held-out validation subjects...")
    val_bn, val_S, val_ids = [], [], []
    for count, idx in enumerate(range(len(val_ds))):
        img, _, sid = val_ds[idx]
        img_b = img.unsqueeze(0).to(device)
        bn, S = compute_sensitivity_map_8cubed(model, img_b, device)
        val_bn.append(bn.cpu())
        val_S.append(S.cpu())
        val_ids.append(sid)
        if (count + 1) % 25 == 0:
            print(f"  labeled {count+1}/{len(val_ds)}", flush=True)

    val_bn = torch.stack(val_bn).to(device)
    val_S = torch.stack(val_S).to(device)
    print(f"Val tensors: bottleneck {val_bn.shape}, S {val_S.shape}")

    # ---------------- Train the spatial head ----------------
    head = SpatialSensitivityHead(in_channels=256).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"\nTraining spatial sensitivity head ({SPATIAL_HEAD_EPOCHS} epochs, "
          f"{train_bn.shape[0]} labeled subjects, full-batch)...")
    for epoch in range(SPATIAL_HEAD_EPOCHS):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        pred = head(train_bn)
        loss = F.mse_loss(pred, train_S)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 15 == 0:
            print(f"  epoch {epoch+1}/{SPATIAL_HEAD_EPOCHS}: mse={loss.item():.6f}")

    head.eval()
    with torch.no_grad():
        pred_val_S = head(val_bn)  # (n_val, 8,8,8)

    # ---------------- Check (a): pooled voxel-level correlation, held-out ----------------
    pred_flat = pred_val_S.cpu().numpy().flatten()
    meas_flat = val_S.cpu().numpy().flatten()
    rho_pooled, p_pooled_param = stats.spearmanr(pred_flat, meas_flat)

    rng2 = np.random.default_rng(SEED + 1)
    n_perm_sub = 200  # subsample for permutation tractability (8^3 * 125 = 64000 points)
    perm_rhos = np.empty(n_perm_sub)
    for i in range(n_perm_sub):
        perm_y = rng2.permutation(meas_flat)
        perm_rhos[i], _ = stats.spearmanr(pred_flat, perm_y)
    p_pooled_perm = float((np.abs(perm_rhos) >= np.abs(rho_pooled)).mean())

    print(f"\n=== CHECK (a): pooled voxel-level Spearman(predicted, measured), held-out ===")
    print(f"rho = {rho_pooled:+.4f} (parametric p={p_pooled_param:.4e}, permutation p={p_pooled_perm:.4f}, n_perm={n_perm_sub})")

    # ---------------- Check (b): per-subject (within-volume) correlation, not degenerate ----------------
    per_subject_rhos = []
    for i in range(val_bn.shape[0]):
        p_i = pred_val_S[i].cpu().numpy().flatten()
        m_i = val_S[i].cpu().numpy().flatten()
        if m_i.std() > 1e-6 and p_i.std() > 1e-6:
            r, _ = stats.spearmanr(p_i, m_i)
            if not np.isnan(r):
                per_subject_rhos.append(r)
    per_subject_rhos = np.array(per_subject_rhos)

    t_stat, p_ttest = stats.ttest_1samp(per_subject_rhos, 0.0)
    w_stat, p_wilcoxon = stats.wilcoxon(per_subject_rhos - 0.0) if len(per_subject_rhos) > 0 else (np.nan, np.nan)

    print(f"\n=== CHECK (b): per-subject (within-volume) Spearman, not degenerate ===")
    print(f"n subjects with valid within-volume correlation: {len(per_subject_rhos)}/{val_bn.shape[0]}")
    print(f"mean per-subject rho = {per_subject_rhos.mean():+.4f} std={per_subject_rhos.std():.4f}")
    print(f"one-sample t-test vs 0: t={t_stat:.3f}, p={p_ttest:.4e}")
    print(f"Wilcoxon signed-rank vs 0: p={p_wilcoxon:.4e}")

    check_a_pass = (rho_pooled > 0) and (p_pooled_perm < 0.05)
    check_b_pass = (per_subject_rhos.mean() > 0) and (p_ttest < 0.05) and (p_wilcoxon < 0.05)

    print(f"\n=== FINAL VERDICT ===")
    print(f"Check (a) pooled correlation: {'PASS' if check_a_pass else 'FAIL'}")
    print(f"Check (b) not degenerate (per-subject): {'PASS' if check_b_pass else 'FAIL'}")
    overall_pass = check_a_pass and check_b_pass
    print(f"\nOVERALL: {'PASS -- the bottleneck can be trained to predict WHERE it is causally vulnerable' if overall_pass else 'FAIL -- do not proceed to a refinement branch'}")

    summary = {
        "n_train_labels": int(train_bn.shape[0]), "n_val": int(val_bn.shape[0]),
        "spatial_head_epochs": SPATIAL_HEAD_EPOCHS,
        "pooled_rho": float(rho_pooled), "pooled_p_param": float(p_pooled_param), "pooled_p_perm": p_pooled_perm,
        "per_subject_mean_rho": float(per_subject_rhos.mean()), "per_subject_std_rho": float(per_subject_rhos.std()),
        "per_subject_ttest_p": float(p_ttest), "per_subject_wilcoxon_p": float(p_wilcoxon),
        "n_subjects_valid": int(len(per_subject_rhos)),
        "check_a_pass": bool(check_a_pass), "check_b_pass": bool(check_b_pass),
        "overall_pass": bool(overall_pass),
    }
    with open(OUT_DIR / "E73_spatial_predictor_summary_scaled.json", "w") as f:
        json.dump(summary, f, indent=2)

    per_subject_table = {val_ids[i]: float(per_subject_rhos[i]) for i in range(len(per_subject_rhos))
                         if i < len(val_ids)}
    with open(OUT_DIR / "E73_spatial_predictor_per_subject_scaled.json", "w") as f:
        json.dump(per_subject_table, f, indent=2)

    torch.save(head.state_dict(), OUT_DIR / "E73_spatial_head_state_scaled.pt")
    print("\nSaved E73_spatial_predictor_summary_scaled.json, E73_spatial_predictor_per_subject_scaled.json, E73_spatial_head_state_scaled.pt")


if __name__ == "__main__":
    main()
