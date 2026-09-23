"""
Phase E74: Representation Spatial-Dependence Audit (cheap, offline, NO
TRAINING) -- per the user's reframe after E71-E73: stop trying to act on
self-diagnostic information directly (routing/reweighting/gating all
killed); instead measure WHAT REPRESENTATION PROPERTY makes a tensor
translation-sensitive in the first place, before designing any fix.

BACKGROUND:
  E65 (FLAIR-only checkpoint): translating enc1 by a small fixed offset
    before the skip connection causes a MUCH larger Dice drop (0.214) than
    local permutation (0.027) or smoothing (0.028) -- absolute spatial
    correspondence dominates over local arrangement or frequency content,
    specifically at the enc1-to-dec1 skip junction.
  E73: causal vulnerability (bottleneck-ablation sensitivity) has real,
    non-trivial SPATIAL structure that correlates with the network's own
    errors.

THIS PHASE asks a NEW, narrower question than either: is translation-
sensitivity a property specific to the enc1 SKIP junction (E65's own
locus), or does it appear more generally at OTHER depths too (bottleneck)?
And does the MAGNITUDE of a subject's own translation-sensitivity track a
measurable property of that subject's OWN representation -- specifically,
spatial-frequency content and feature magnitude, the two candidate
properties named in the reframe -- rather than lesion size alone (E65's
already-known confound, controlled for here as it was throughout E48-E65).

METHOD (reusing E65's own validated intervention machinery verbatim,
translate_volume, and E64's split-forward design):
  1. TRANSLATE enc1 by the same fixed 3-voxel offset as E65, on the
     MULTIMODAL (4-channel) MM-seed0 checkpoint (E65 used the FLAIR-only
     checkpoint; E74 deliberately re-targets the checkpoint this project's
     later work, E70-E73, has been building on, for continuity).
  2. TRANSLATE the bottleneck tensor by the same 3-voxel offset (adapted
     to its own 8^3 spatial resolution: a 3-voxel shift there is a much
     larger FRACTION of the tensor's extent than 3/64 at enc1 -- this is
     flagged explicitly in the results, not glossed over, since it is not
     a directly comparable magnitude, only a directly comparable
     INTERVENTION TYPE).
  3. For each subject and each locus (enc1, bottleneck), measure:
       - Dice drop from the translation (paired, same statistical
         discipline as E65: t-test, Wilcoxon, permutation, bootstrap CI)
       - the INTACT tensor's own high-frequency spatial-energy fraction
         (Laplacian-filtered variance / total variance, per subject)
       - the INTACT tensor's own mean feature magnitude (L2 norm per
         voxel, averaged)
  4. Test: does per-subject Dice drop correlate with (a) high-frequency
     energy fraction, (b) feature magnitude -- CONTROLLING for lesion
     size (partial Spearman, matching E48/E65's own confound-control
     convention) -- at EACH locus independently.

PRE-DECLARED READING (not a single pass/fail gate, a characterization,
matching E65's own "pattern across arms" convention):
  - If high-frequency energy fraction predicts translation-sensitivity
    (partial rho significant, same sign at both loci) -> supports the
    "coordinate-dependent / high-frequency representations are what's
    fragile" hypothesis -- a concrete, measurable property to target.
  - If feature magnitude predicts it instead (or as well) -> a different,
    still concrete, candidate (large-magnitude features may encode more
    positionally-specific information).
  - If NEITHER predicts it beyond lesion size -> the vulnerability is not
    explained by either simple candidate property tested here; report
    honestly, this narrows rather than closes the search.
  - If translation-sensitivity is present at enc1 but NOT the bottleneck
    (or vice versa) -> the phenomenon is locus-specific, not general
    across depth -- itself an informative, reportable finding.

This script trains NOTHING and modifies NO architecture. Pure measurement.
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

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 0
N_PERM = 1000
N_BOOT = 2000
TRANSLATION_OFFSET = 3  # voxels, same as E65, applied at each locus's own resolution

MM_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e70" / "runs"
           / "MM_seed0" / "checkpoints" / "best.pth")


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    return 1.0 if denom == 0 else float(2 * tp / denom)


def translate_volume(t, offset):
    """torch tensor (C,D,H,W), roll by `offset` voxels along each spatial
    axis, wrapping -- identical construction to E65's translate_volume,
    ported to torch (GPU-resident tensors here vs E65's numpy)."""
    return torch.roll(t, shifts=(offset, offset, offset), dims=(1, 2, 3))


def high_freq_energy_fraction(t):
    """Laplacian-filtered variance / total variance, per-channel-averaged.
    A simple, standard proxy for how much of a tensor's spatial energy
    lives at high spatial frequency (fine detail) vs low (coarse
    structure) -- deliberately simple/interpretable over a full spectral
    decomposition, matching this project's own "smallest correct
    implementation" convention for a first-pass diagnostic."""
    C = t.shape[0]
    lap_kernel = torch.tensor([
        [[0, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, -1, 0], [-1, 6, -1], [0, -1, 0]],
        [[0, 0, 0], [0, -1, 0], [0, 0, 0]],
    ], dtype=t.dtype, device=t.device).unsqueeze(0).unsqueeze(0)
    kernel = lap_kernel.repeat(C, 1, 1, 1, 1)
    t_b = t.unsqueeze(0)
    t_padded = F.pad(t_b, (1, 1, 1, 1, 1, 1), mode="reflect")
    lap = F.conv3d(t_padded, kernel, groups=C)
    hf_var = lap.var().item()
    total_var = t.var().item()
    return hf_var / (total_var + 1e-8)


def mean_feature_magnitude(t):
    """Mean L2 norm across channels, averaged over all spatial locations."""
    return t.pow(2).sum(dim=0).sqrt().mean().item()


def forward_full_with_translation(model, img_b, locus, offset, device):
    """locus in {'none','enc1','bottleneck'}. Returns binarized prediction."""
    with torch.no_grad():
        enc1 = model.enc1(img_b)
        e_skip = translate_volume(enc1.squeeze(0), offset).unsqueeze(0) if locus == "enc1" else enc1

        pool1 = model.pool1(enc1)  # E_pool always uses the REAL enc1, per E64's split-forward isolation
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)
        bn_used = translate_volume(bottleneck.squeeze(0), offset).unsqueeze(0) if locus == "bottleneck" else bottleneck

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
    return probs.squeeze(0).squeeze(0).cpu().numpy(), enc1.squeeze(0), bottleneck.squeeze(0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    subject_dirs_by_order = []
    for idx in range(len(val_ds)):
        img, msk, sid = val_ds[idx]
        subject_dirs_by_order.append(val_ds.subject_dirs[idx])
        img_b = img.unsqueeze(0).to(device)
        target_bin = (msk.squeeze(0).numpy() > 0.5).astype(np.float32)

        probs_intact, enc1_intact, bn_intact = forward_full_with_translation(
            model, img_b, "none", TRANSLATION_OFFSET, device)
        dice_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin)

        probs_enc1_t, _, _ = forward_full_with_translation(model, img_b, "enc1", TRANSLATION_OFFSET, device)
        dice_enc1_t = dice_score((probs_enc1_t >= 0.5).astype(np.float32), target_bin)

        probs_bn_t, _, _ = forward_full_with_translation(model, img_b, "bottleneck", TRANSLATION_OFFSET, device)
        dice_bn_t = dice_score((probs_bn_t >= 0.5).astype(np.float32), target_bin)

        hf_enc1 = high_freq_energy_fraction(enc1_intact)
        hf_bn = high_freq_energy_fraction(bn_intact)
        mag_enc1 = mean_feature_magnitude(enc1_intact)
        mag_bn = mean_feature_magnitude(bn_intact)

        records.append({
            "subject_id": sid,
            "dice_intact": dice_intact,
            "dice_enc1_translated": dice_enc1_t, "drop_enc1": dice_intact - dice_enc1_t,
            "dice_bn_translated": dice_bn_t, "drop_bn": dice_intact - dice_bn_t,
            "hf_energy_frac_enc1": hf_enc1, "hf_energy_frac_bn": hf_bn,
            "mean_magnitude_enc1": mag_enc1, "mean_magnitude_bn": mag_bn,
        })

        if (idx + 1) % 25 == 0:
            print(f"  processed {idx+1}/{len(val_ds)}", flush=True)

    # real native lesion size, from the actual seg file (replacing the rough proxy above)
    for i, r in enumerate(records):
        subject_dir = subject_dirs_by_order[i]
        seg_path = Path(subject_dir) / f"{r['subject_id']}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        r["native_size"] = int((seg_data > 0).sum())

    with open(OUT_DIR / "E74_spatial_dependence_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.\n")

    # ---------------- Statistics per locus ----------------
    drop_enc1 = np.array([r["drop_enc1"] for r in records])
    drop_bn = np.array([r["drop_bn"] for r in records])
    size = np.array([r["native_size"] for r in records])
    hf_enc1 = np.array([r["hf_energy_frac_enc1"] for r in records])
    hf_bn = np.array([r["hf_energy_frac_bn"] for r in records])
    mag_enc1 = np.array([r["mean_magnitude_enc1"] for r in records])
    mag_bn = np.array([r["mean_magnitude_bn"] for r in records])

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
        print(f"\n=== {label}: mean Dice drop = {drop.mean():.4f} (95% CI [{ci_lo:.4f}, {ci_hi:.4f}]) ===")
        print(f"  t-test p={p_t:.4e}, Wilcoxon p={p_w:.4e}, sign-flip permutation p={p_perm:.4f}")
        return {"mean_drop": float(drop.mean()), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
                "ttest_p": float(p_t), "wilcoxon_p": float(p_w), "perm_p": p_perm}

    stat_enc1 = paired_stats(drop_enc1, "ENC1 translation (skip junction, matches E65's own locus)")
    stat_bn = paired_stats(drop_bn, "BOTTLENECK translation (new locus)")

    def partial_corr(y, x_of_interest, control):
        log_control = np.log(control + 1)
        X = np.column_stack([np.ones(len(log_control)), log_control])
        beta_y, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid_y = y - X @ beta_y
        beta_x, *_ = np.linalg.lstsq(X, x_of_interest, rcond=None)
        resid_x = x_of_interest - X @ beta_x
        rho, p = stats.spearmanr(resid_y, resid_x)
        return rho, p

    print("\n=== Does translation-sensitivity track high-frequency energy or feature magnitude, "
          "controlling for lesion size? ===")
    for locus_name, drop, hf, mag in [("enc1", drop_enc1, hf_enc1, mag_enc1),
                                        ("bottleneck", drop_bn, hf_bn, mag_bn)]:
        rho_size, p_size = stats.spearmanr(size, drop)
        rho_hf_partial, p_hf_partial = partial_corr(drop, hf, size)
        rho_mag_partial, p_mag_partial = partial_corr(drop, mag, size)
        print(f"\n  [{locus_name}] Spearman(size, drop) = {rho_size:+.4f} (p={p_size:.4e}) [reference, matches E48/E65 direction]")
        print(f"  [{locus_name}] Partial Spearman(drop, hf_energy | size) = {rho_hf_partial:+.4f} (p={p_hf_partial:.4e})")
        print(f"  [{locus_name}] Partial Spearman(drop, magnitude | size) = {rho_mag_partial:+.4f} (p={p_mag_partial:.4e})")

    summary = {
        "translation_offset": TRANSLATION_OFFSET, "n_subjects": len(records),
        "enc1": stat_enc1, "bottleneck": stat_bn,
    }
    for locus_name, drop, hf, mag in [("enc1", drop_enc1, hf_enc1, mag_enc1),
                                        ("bottleneck", drop_bn, hf_bn, mag_bn)]:
        rho_size, p_size = stats.spearmanr(size, drop)
        rho_hf_partial, p_hf_partial = partial_corr(drop, hf, size)
        rho_mag_partial, p_mag_partial = partial_corr(drop, mag, size)
        summary[f"{locus_name}_size_rho"] = float(rho_size)
        summary[f"{locus_name}_size_p"] = float(p_size)
        summary[f"{locus_name}_hf_partial_rho"] = float(rho_hf_partial)
        summary[f"{locus_name}_hf_partial_p"] = float(p_hf_partial)
        summary[f"{locus_name}_mag_partial_rho"] = float(rho_mag_partial)
        summary[f"{locus_name}_mag_partial_p"] = float(p_mag_partial)

    with open(OUT_DIR / "E74_spatial_dependence_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E74_spatial_dependence_summary.json")


if __name__ == "__main__":
    main()
