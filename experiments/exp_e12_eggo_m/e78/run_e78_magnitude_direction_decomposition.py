"""
Phase E78: Magnitude-Direction Causal Decomposition (NO TRAINING, pure
inference-time intervention) -- per the user's design, following E77's
corrected A+B reading (high enc1 magnitude is simultaneously lesion
salience AND translation vulnerability, ruling out any magnitude-
suppression intervention).

OPEN QUESTION E75/E76 DID NOT ANSWER: those phases established that
RESCALING magnitude (z -> alpha*z, direction u=z/||z|| held exactly
fixed) causally changes translation sensitivity. They did NOT establish
WHERE the actual positional information that gets disrupted by
translation lives -- in the magnitude channel r(p)=||z(p)||, the
direction channel u(p)=z(p)/||z(p)||, or their interaction. It is
possible large r(p) marks lesion importance while the POSITIONAL
structure translation disrupts is actually carried in u(p), with large r
merely amplifying the downstream consequence.

METHOD: decompose z(p) = r(p) * u(p) at every enc1 voxel, for both the
INTACT tensor and the TRANSLATED tensor (E65/E74's own 3-voxel roll).
Construct three controlled hybrid tensors that mix components from the
two, isolating each component's own contribution to the translation
effect:

  z_intact       = r_intact * u_intact               [reference: untouched]
  z_full_translated = r_translated * u_translated     [reference: E65's own
                       intervention -- both components come from the
                       translated tensor, since translation acts on the
                       whole vector jointly]
  z_r_only       = r_translated * u_intact            [ONLY magnitude is
                       translated -- takes each voxel's translated-tensor
                       MAGNITUDE but keeps the ORIGINAL (intact) direction
                       at that same spatial location]
  z_u_only       = r_intact * u_translated            [ONLY direction is
                       translated -- takes each voxel's ORIGINAL magnitude
                       but the TRANSLATED tensor's direction at that
                       location]

For each, decode and measure Dice, giving:
  D_intact, D_full, D_r_only, D_u_only
  S_full   = D_intact - D_full     (E65's own quantity, the whole effect)
  S_r      = D_intact - D_r_only   (magnitude-translation-only effect)
  S_u      = D_intact - D_u_only   (direction-translation-only effect)

PRE-DECLARED READING (a characterization across arms, matching E65's own
"pattern across arms" convention, not a single pass/fail gate):
  - S_u >> S_r -> direction carries the positional structure; magnitude
    mainly amplifies the downstream consequence (candidate: "preserve
    magnitude, regularize direction's positional dependence").
  - S_r >> S_u -> magnitude itself carries positional information beyond
    amplification (candidate: "magnitude-conditioned positional
    robustness," a more complex mechanism than pure direction
    regularization).
  - S_r + S_u approx S_full -> components act roughly additively/
    independently.
  - S_r + S_u far from S_full (either direction) -> a real interaction
    effect between magnitude and direction, neither component alone
    explains the whole.

Reuses E65/E74/E75/E76's own translate_volume and split-forward
machinery verbatim.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e74"))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402
from run_e74_spatial_dependence_audit import translate_volume  # noqa: E402

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 0
N_PERM = 1000
N_BOOT = 2000
TRANSLATION_OFFSET = 3
EPS = 1e-8

MM_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e70" / "runs"
           / "MM_seed0" / "checkpoints" / "best.pth")


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    return 1.0 if denom == 0 else float(2 * tp / denom)


def decompose(t):
    """t: (C,D,H,W). Returns (r: (D,H,W), u: (C,D,H,W)) with u having
    unit norm at every spatial location (except where r~0, guarded)."""
    r = t.norm(dim=0)  # (D,H,W)
    u = t / r.clamp_min(EPS).unsqueeze(0)
    return r, u


def recompose(r, u):
    """Inverse of decompose: z(p) = r(p) * u(p)."""
    return r.unsqueeze(0) * u


def forward_with_enc1(model, img_b, enc1_used, device):
    """Standard v3 forward from a GIVEN enc1 tensor (batch dim added
    internally) -- E_pool always uses the network's OWN real enc1
    (computed fresh inside), matching E64/E74/E75/E76's split-forward
    isolation: only the SKIP path is ever intervened on."""
    with torch.no_grad():
        enc1_real = model.enc1(img_b)
        pool1 = model.pool1(enc1_real)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)

        upconv3 = model.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = model.dec3(cat3)
        upconv2 = model.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = model.dec2(cat2)
        upconv1 = model.upconv1(dec2)
        cat1 = torch.cat([upconv1, enc1_used.unsqueeze(0)], dim=1)
        dec1 = model.dec1(cat1)
        probs = model.seg_head(dec1)
    return probs.squeeze(0).squeeze(0).cpu().numpy(), enc1_real


def verify_decomposition_identity(device):
    """Sanity check: r*u must reconstruct the original tensor exactly
    (up to float precision), and z_full (r_translated*u_translated) must
    exactly equal translate_volume(z) -- since translating a tensor
    commutes with the r/u decomposition (translating then decomposing
    gives the same r,u as decomposing then translating, both being
    pointwise-then-spatial operations)."""
    torch.manual_seed(0)
    z = torch.randn(8, 6, 6, 6, device=device) + 0.5
    r, u = decompose(z)
    z_recon = recompose(r, u)
    diff_recon = (z - z_recon).abs().max().item()
    assert diff_recon < 1e-4, f"Decomposition/recomposition not identity: {diff_recon}"

    z_t = translate_volume(z, 2)
    r_t_direct, u_t_direct = decompose(z_t)
    r_of_z, u_of_z = decompose(z)
    r_t_via_translate = translate_volume(r_of_z.unsqueeze(0), 2).squeeze(0)
    u_t_via_translate = translate_volume(u_of_z, 2)
    diff_r = (r_t_direct - r_t_via_translate).abs().max().item()
    diff_u = (u_t_direct - u_t_via_translate).abs().max().item()
    assert diff_r < 1e-4 and diff_u < 1e-4, f"Translate/decompose does not commute: r_diff={diff_r}, u_diff={diff_u}"
    print(f"[Sanity check] Decomposition identity verified (recon diff {diff_recon:.2e}), "
          f"translate/decompose commutativity verified (r_diff {diff_r:.2e}, u_diff {diff_u:.2e}). PASS.\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    verify_decomposition_identity(device)

    ckpt = torch.load(str(MM_CKPT), map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded MM seed0 checkpoint: best_per_subject_dice={ckpt.get('best_per_subject_dice')}\n")

    val_ds = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                    split="val", val_split=0.1, target_shape=(64, 64, 64))

    records = []
    for idx in range(len(val_ds)):
        img, msk, sid = val_ds[idx]
        img_b = img.unsqueeze(0).to(device)
        target_bin = (msk.squeeze(0).numpy() > 0.5).astype(np.float32)

        with torch.no_grad():
            enc1_intact = model.enc1(img_b).squeeze(0)
        enc1_translated = translate_volume(enc1_intact, TRANSLATION_OFFSET)

        r_intact, u_intact = decompose(enc1_intact)
        r_translated, u_translated = decompose(enc1_translated)

        z_intact = enc1_intact
        z_full = enc1_translated
        z_r_only = recompose(r_translated, u_intact)
        z_u_only = recompose(r_intact, u_translated)

        probs_intact, _ = forward_with_enc1(model, img_b, z_intact, device)
        probs_full, _ = forward_with_enc1(model, img_b, z_full, device)
        probs_r_only, _ = forward_with_enc1(model, img_b, z_r_only, device)
        probs_u_only, _ = forward_with_enc1(model, img_b, z_u_only, device)

        d_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin)
        d_full = dice_score((probs_full >= 0.5).astype(np.float32), target_bin)
        d_r_only = dice_score((probs_r_only >= 0.5).astype(np.float32), target_bin)
        d_u_only = dice_score((probs_u_only >= 0.5).astype(np.float32), target_bin)

        records.append({
            "subject_id": sid,
            "D_intact": d_intact, "D_full": d_full, "D_r_only": d_r_only, "D_u_only": d_u_only,
            "S_full": d_intact - d_full, "S_r": d_intact - d_r_only, "S_u": d_intact - d_u_only,
        })

        if (idx + 1) % 25 == 0:
            print(f"  processed {idx+1}/{len(val_ds)}", flush=True)

    with open(OUT_DIR / "E78_decomposition_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.\n")

    S_full = np.array([r["S_full"] for r in records])
    S_r = np.array([r["S_r"] for r in records])
    S_u = np.array([r["S_u"] for r in records])

    def paired_stats(drop, label):
        t_stat, p_t = stats.ttest_1samp(drop, 0.0)
        w_stat, p_w = stats.wilcoxon(drop)
        rng = np.random.default_rng(SEED)
        signs = rng.choice([-1, 1], size=(N_PERM, len(drop)))
        perm_means = (signs * drop[None, :]).mean(axis=1)
        p_perm = float((np.abs(perm_means) >= np.abs(drop.mean())).mean())
        boot_idx = rng.integers(0, len(drop), size=(N_BOOT, len(drop)))
        boot_means = drop[boot_idx].mean(axis=1)
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        print(f"=== {label}: mean = {drop.mean():.4f} (95% CI [{ci_lo:.4f}, {ci_hi:.4f}]) ===")
        print(f"  t-test p={p_t:.4e}, Wilcoxon p={p_w:.4e}, sign-flip permutation p={p_perm:.4f}\n")
        return {"mean": float(drop.mean()), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
                "ttest_p": float(p_t), "wilcoxon_p": float(p_w), "perm_p": p_perm}

    print("=== MAGNITUDE-DIRECTION DECOMPOSITION RESULTS ===\n")
    stat_full = paired_stats(S_full, "S_full (both r and u translated -- E65's own quantity)")
    stat_r = paired_stats(S_r, "S_r (ONLY magnitude translated, direction held intact)")
    stat_u = paired_stats(S_u, "S_u (ONLY direction translated, magnitude held intact)")

    # component comparison
    t_ru, p_ru = stats.ttest_rel(S_r, S_u)
    w_ru, pw_ru = stats.wilcoxon(S_r, S_u)
    print(f"=== S_r vs S_u (which component dominates?) ===")
    print(f"  mean S_r={S_r.mean():.4f}, mean S_u={S_u.mean():.4f}, diff={S_r.mean()-S_u.mean():+.4f}")
    print(f"  paired t-test: t={t_ru:.3f}, p={p_ru:.4e}, Wilcoxon p={pw_ru:.4e}\n")

    additivity_gap = S_full.mean() - (S_r.mean() + S_u.mean())
    print(f"=== Additivity check: S_full vs S_r + S_u ===")
    print(f"  S_full mean = {S_full.mean():.4f}")
    print(f"  S_r + S_u   = {S_r.mean() + S_u.mean():.4f}")
    print(f"  Gap (S_full - (S_r+S_u)) = {additivity_gap:+.4f} "
          f"({'super-additive/interaction' if additivity_gap > 0.02 else 'sub-additive/interaction' if additivity_gap < -0.02 else 'roughly additive'})")

    if S_r.mean() > S_u.mean() * 1.5 and p_ru < 0.05:
        reading = "MAGNITUDE dominates: S_r >> S_u -- magnitude itself carries positional information beyond amplification"
    elif S_u.mean() > S_r.mean() * 1.5 and p_ru < 0.05:
        reading = "DIRECTION dominates: S_u >> S_r -- direction carries the positional structure; magnitude mainly amplifies the downstream consequence"
    elif p_ru >= 0.05:
        reading = "S_r and S_u are NOT significantly different -- magnitude and direction contribute comparably"
    else:
        reading = "Mixed/intermediate pattern -- see full numbers"

    print(f"\n=== READING ===\n{reading}")

    summary = {
        "n_subjects": len(records),
        "S_full": stat_full, "S_r": stat_r, "S_u": stat_u,
        "S_r_vs_S_u_ttest_p": float(p_ru), "S_r_vs_S_u_wilcoxon_p": float(pw_ru),
        "additivity_gap": float(additivity_gap),
        "reading": reading,
    }
    with open(OUT_DIR / "E78_decomposition_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E78_decomposition_summary.json")


if __name__ == "__main__":
    main()
