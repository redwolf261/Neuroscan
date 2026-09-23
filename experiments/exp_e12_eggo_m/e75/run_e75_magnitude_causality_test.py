"""
Phase E75: Magnitude-Positional Sensitivity CAUSALITY test (NO TRAINING,
pure inference-time intervention) -- per the user's design, testing
whether E74's magnitude-translation-sensitivity CORRELATION is causal.

E74 found: at both enc1 and the bottleneck, higher feature magnitude
||z(p)|| correlates with larger Dice drop from translating that tensor,
controlling for lesion size (partial rho +0.30 at enc1, +0.53 at the
bottleneck). This is a correlation across SUBJECTS/VOXELS with naturally
varying magnitude -- it does NOT establish that magnitude CAUSES the
sensitivity (vs. both being downstream of some third property, e.g.
"this is lesion tissue" -- the tautological confound the user explicitly
flagged: "large activations are important features" is nearly definitional
and would NOT be an interesting finding on its own).

CAUSAL TEST (per the user's design): decompose each feature vector at
every spatial location into magnitude and direction,
    z(p) = ||z(p)|| * u(p),   u(p) = z(p) / ||z(p)||
then ARTIFICIALLY RESCALE ONLY THE MAGNITUDE by a fixed multiplier alpha,
leaving direction u(p) exactly unchanged:
    z_alpha(p) = alpha * z(p)     [trivially preserves direction, since
                                    scaling by a positive scalar leaves
                                    z/||z|| unchanged -- this holds exactly,
                                    verified numerically below]
For alpha in {0.25, 0.5, 1.0, 2.0}, on the SAME subject and SAME feature
map, measure BOTH:
  (a) I(alpha) = Dice(intact-but-rescaled) vs Dice(intact, alpha=1) --
      "does the network's OWN prediction degrade just from magnitude
      rescaling alone, before any translation?" (this is a confound
      check, not the main test -- distinguishes "the network is fragile
      to magnitude changes generally" from the causal question below).
  (b) S(alpha) = Dice(rescaled-then-translated) vs Dice(rescaled,
      NOT translated) -- the actual causal test: does the SAME
      translation hurt MORE when magnitude is inflated (alpha>1) and
      LESS when magnitude is shrunk (alpha<1), on the identical feature
      DIRECTIONS throughout?

PRE-DECLARED DECISION RULE (per the user's own framing):
  PASS (magnitude is causally responsible, not just correlated) if
    S(alpha) is monotonically increasing in alpha, subject-level paired
    tests confirm S(2.0) > S(1.0) > S(0.25) significantly, at BOTH enc1
    and bottleneck independently.
  FAIL (magnitude is not causal, or the earlier correlation was
    confounded by something else, e.g. lesion identity/size) otherwise --
    kill the magnitude hypothesis, do not design a decoupling intervention.

Reuses E74's own translate_volume and split-forward machinery verbatim.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
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
ALPHAS = [0.25, 0.5, 1.0, 2.0]

MM_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e70" / "runs"
           / "MM_seed0" / "checkpoints" / "best.pth")


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    return 1.0 if denom == 0 else float(2 * tp / denom)


def rescale_magnitude(t, alpha):
    """z(p) -> alpha * z(p), per spatial location, all channels. Trivially
    preserves direction u(p) = z(p)/||z(p)|| for alpha > 0 (verified
    numerically in main() before any subject is processed)."""
    return alpha * t


def forward_full(model, img_b, locus, translate, alpha, device):
    """locus in {'enc1','bottleneck'}. translate: bool. alpha: magnitude
    multiplier applied to the SAME locus tensor, applied BEFORE any
    translation (order: rescale magnitude, then optionally translate --
    translation is a spatial permutation and does not interact with the
    per-voxel magnitude rescale, so order does not affect the rescaled
    tensor's per-voxel magnitudes, only their spatial arrangement)."""
    with torch.no_grad():
        enc1 = model.enc1(img_b)

        if locus == "enc1":
            e_skip = rescale_magnitude(enc1.squeeze(0), alpha)
            if translate:
                e_skip = translate_volume(e_skip, TRANSLATION_OFFSET)
            e_skip = e_skip.unsqueeze(0)
        else:
            e_skip = enc1  # untouched when locus == 'bottleneck'

        pool1 = model.pool1(enc1)  # E_pool always real, per E64/E74's isolation design
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)

        if locus == "bottleneck":
            bn_used = rescale_magnitude(bottleneck.squeeze(0), alpha)
            if translate:
                bn_used = translate_volume(bn_used, TRANSLATION_OFFSET)
            bn_used = bn_used.unsqueeze(0)
        else:
            bn_used = bottleneck

        upconv3 = model.upconv3(bn_used)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = model.dec3(cat3)

        upconv2 = model.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = model.dec2(cat2)

        upconv1 = model.upconv1(dec2)
        cat1 = torch.cat([upconv1, e_skip], dim=1)
        dec1 = model.dec1(cat1)

        probs = model.seg_head(dec1)
    return probs.squeeze(0).squeeze(0).cpu().numpy()


def verify_direction_preserved(device):
    """Numerical check (not just claimed): rescaling by a positive scalar
    leaves z/||z|| unchanged. Run once before any subject processing."""
    torch.manual_seed(0)
    z = torch.randn(8, 4, 4, 4, device=device) + 0.1  # avoid exact-zero norm
    norm = z.norm(dim=0, keepdim=True).clamp_min(1e-8)
    u_before = z / norm
    for alpha in ALPHAS:
        z_scaled = alpha * z
        norm_scaled = z_scaled.norm(dim=0, keepdim=True).clamp_min(1e-8)
        u_after = z_scaled / norm_scaled
        max_diff = (u_before - u_after).abs().max().item()
        assert max_diff < 1e-5, f"Direction NOT preserved at alpha={alpha}: max diff {max_diff}"
    print("[Sanity check] Direction preservation verified numerically for all alphas (max diff < 1e-5). PASS.\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    verify_direction_preserved(device)

    ckpt = torch.load(str(MM_CKPT), map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded MM seed0 checkpoint: best_per_subject_dice={ckpt.get('best_per_subject_dice')}\n")

    val_ds = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                    split="val", val_split=0.1, target_shape=(64, 64, 64))

    records = {"enc1": [], "bottleneck": []}

    for idx in range(len(val_ds)):
        img, msk, sid = val_ds[idx]
        img_b = img.unsqueeze(0).to(device)
        target_bin = (msk.squeeze(0).numpy() > 0.5).astype(np.float32)

        for locus in ["enc1", "bottleneck"]:
            row = {"subject_id": sid}
            for alpha in ALPHAS:
                probs_no_t = forward_full(model, img_b, locus, translate=False, alpha=alpha, device=device)
                dice_no_t = dice_score((probs_no_t >= 0.5).astype(np.float32), target_bin)

                probs_t = forward_full(model, img_b, locus, translate=True, alpha=alpha, device=device)
                dice_t = dice_score((probs_t >= 0.5).astype(np.float32), target_bin)

                row[f"dice_intact_alpha{alpha}"] = dice_no_t
                row[f"dice_translated_alpha{alpha}"] = dice_t
                row[f"S_alpha{alpha}"] = dice_no_t - dice_t  # translation-induced drop AT this alpha

            records[locus].append(row)

        if (idx + 1) % 25 == 0:
            print(f"  processed {idx+1}/{len(val_ds)}", flush=True)

    for locus in ["enc1", "bottleneck"]:
        with open(OUT_DIR / f"E75_magnitude_causality_table_{locus}.json", "w") as f:
            json.dump(records[locus], f, indent=2)
    print(f"\nSaved subject records for both loci.\n")

    # ---------------- Analysis per locus ----------------
    summary = {}
    for locus in ["enc1", "bottleneck"]:
        recs = records[locus]
        print(f"\n{'='*20} LOCUS: {locus} {'='*20}")

        # Confound check (a): does magnitude rescaling ALONE (no translation)
        # degrade the network's own prediction? Report but this is not the
        # main test.
        print(f"\n--- (a) Confound check: intact-but-rescaled Dice, no translation ---")
        for alpha in ALPHAS:
            vals = np.array([r[f"dice_intact_alpha{alpha}"] for r in recs])
            print(f"  alpha={alpha}: mean Dice = {vals.mean():.4f}")

        # Main causal test (b): S(alpha) = translation-induced drop at each alpha
        S_by_alpha = {alpha: np.array([r[f"S_alpha{alpha}"] for r in recs]) for alpha in ALPHAS}
        print(f"\n--- (b) MAIN TEST: translation-induced Dice drop S(alpha) ---")
        for alpha in ALPHAS:
            v = S_by_alpha[alpha]
            print(f"  alpha={alpha}: mean S = {v.mean():.4f} (std {v.std():.4f})")

        # monotonicity + pairwise significance
        s_025, s_05, s_1, s_2 = S_by_alpha[0.25], S_by_alpha[0.5], S_by_alpha[1.0], S_by_alpha[2.0]
        monotonic = (s_025.mean() < s_05.mean() < s_1.mean() < s_2.mean())

        t_2v1, p_2v1 = stats.ttest_rel(s_2, s_1)
        w_2v1, pw_2v1 = stats.wilcoxon(s_2, s_1)
        t_1v025, p_1v025 = stats.ttest_rel(s_1, s_025)
        w_1v025, pw_1v025 = stats.wilcoxon(s_1, s_025)
        t_2v025, p_2v025 = stats.ttest_rel(s_2, s_025)
        w_2v025, pw_2v025 = stats.wilcoxon(s_2, s_025)

        print(f"\n  Monotonic (mean S increases with alpha): {monotonic}")
        print(f"  S(2.0) vs S(1.0): paired t p={p_2v1:.4e}, Wilcoxon p={pw_2v1:.4e}, "
              f"mean diff={s_2.mean()-s_1.mean():+.4f}")
        print(f"  S(1.0) vs S(0.25): paired t p={p_1v025:.4e}, Wilcoxon p={pw_1v025:.4e}, "
              f"mean diff={s_1.mean()-s_025.mean():+.4f}")
        print(f"  S(2.0) vs S(0.25): paired t p={p_2v025:.4e}, Wilcoxon p={pw_2v025:.4e}, "
              f"mean diff={s_2.mean()-s_025.mean():+.4f}")

        all_significant = (p_2v1 < 0.05 and pw_2v1 < 0.05 and
                           p_1v025 < 0.05 and pw_1v025 < 0.05 and
                           p_2v025 < 0.05 and pw_2v025 < 0.05)
        locus_pass = monotonic and all_significant
        print(f"\n  LOCUS VERDICT: {'PASS' if locus_pass else 'FAIL'}")

        summary[locus] = {
            "mean_S_by_alpha": {str(a): float(S_by_alpha[a].mean()) for a in ALPHAS},
            "monotonic": bool(monotonic),
            "S2_vs_S1_p_ttest": float(p_2v1), "S2_vs_S1_p_wilcoxon": float(pw_2v1),
            "S1_vs_S025_p_ttest": float(p_1v025), "S1_vs_S025_p_wilcoxon": float(pw_1v025),
            "S2_vs_S025_p_ttest": float(p_2v025), "S2_vs_S025_p_wilcoxon": float(pw_2v025),
            "all_pairwise_significant": bool(all_significant),
            "locus_pass": bool(locus_pass),
        }

    overall_pass = summary["enc1"]["locus_pass"] and summary["bottleneck"]["locus_pass"]
    print(f"\n{'='*50}")
    print(f"=== OVERALL E75 VERDICT: {'PASS -- magnitude is causally responsible for translation sensitivity at BOTH loci' if overall_pass else 'FAIL -- magnitude is NOT confirmed causal at both loci; do not design a decoupling intervention on this basis alone'} ===")
    summary["overall_pass"] = bool(overall_pass)

    with open(OUT_DIR / "E75_magnitude_causality_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E75_magnitude_causality_summary.json")


if __name__ == "__main__":
    main()
