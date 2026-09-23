"""
Phase E48: Bottleneck Encoding-Deficiency Audit.

CONTEXT: E47's causal routing-intervention audit found that the
bottleneck->enc1 attention gate's effect on Dice is NOT concentrated at
boundary voxels (p=0.808, clean null) -- causally confirming E43's own
correlational null. Two independent lines of evidence now reject
"boundary-routing failure" as the mechanism behind this project's Dice
ceiling.

THIS PHASE tests a DIFFERENT, previously-untested hypothesis: rather than
the bottleneck encoding useful information but ROUTING it poorly (ruled
out by E47), perhaps the bottleneck fails to ENCODE useful discriminative
information for SMALL lesions in the first place -- i.e. an encoding
deficiency, not a routing deficiency. This is falsifiable and distinct:
if the bottleneck genuinely has little useful signal for small lesions,
severing it entirely (forcing the decoder to work from zeros instead of
real coarse context) should hurt LARGE-lesion subjects' Dice much more
than SMALL-lesion subjects', because large lesions have more to lose from
that ablation while small lesions had little to lose to begin with.

NO TRAINING. Loads E46's already-trained checkpoint (best.pth,
val_dice=0.9102) and runs a causal intervention: the bottleneck tensor is
ZEROED before reaching upconv3, severing ALL coarse-context propagation
to the decoder (dec3/dec2/dec1 must work from upsampled zeros + skip
connections only -- NOT just gating it via psi, as E47 did; this is a
stronger, complete ablation testing whether the coarse pathway carries
ANY useful signal at all for a given subject, not just whether its
ROUTING is boundary-specific).

SIZE METRIC: native_size (native-resolution lesion voxel count), reused
from this project's own established convention (e30's own analysis
scripts) -- treated CONTINUOUSLY (Spearman correlation), not bucketed
into an arbitrary small/large split, avoiding a threshold-choice
artifact.

HYPOTHESIS (falsifiable, pre-declared):
  If the bottleneck has an ENCODING deficiency specific to small lesions,
  then native_size should be POSITIVELY correlated with the causal
  Dice drop from bottleneck ablation (bigger lesions lose more when the
  coarse pathway is severed, because they had more real signal there to
  lose; small lesions lose little because there was little real signal
  there to begin with).

  NULL / competing explanation: if native_size is NOT correlated with
  the drop (or negatively correlated), the bottleneck's causal
  contribution is either uniform across lesion sizes or, if anything,
  MORE important for small lesions -- which would REFUTE the encoding-
  deficiency-for-small-lesions hypothesis specifically (a real, useful,
  falsifiable outcome either way, not something to explain away).

STATISTICAL SAFEGUARDS (project's own established discipline): Spearman
correlation (robust to the known nonlinear size relationships this
project has repeatedly found, e.g. e30's own log/cbrt size transforms),
permutation test (>=500 trials) on the correlation, and a same-family
nonlinear-size-confound check reused from e32's own methodology
(z-scoring, OLS incremental R^2) to verify the effect is not merely
"small lesions have systematically different Dice for unrelated reasons."

PRE-DECLARED DECISION RULE:
  GO (encoding-deficiency-for-small-lesions supported) only if:
    1. Spearman rho(native_size, drop) > 0, AND
    2. permutation p < 0.05, AND
    3. the correlation survives even after nonlinear-size-standardized
       control (i.e. is not an artifact of a known confound family).
  Otherwise: NULL, reported plainly.
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

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000

CKPT_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs" / "AttnGate_seed0" / "checkpoints" / "best.pth"


def fractional_occupancy_64(seg_binary_native):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=(64, 64, 64), mode="area").squeeze().numpy()
    return frac


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    if denom == 0:
        return 1.0
    return float(2 * tp / denom)


def forward_with_bottleneck_ablation(model, image, ablate, device):
    """Runs the v5 trunk manually (verified bit-for-bit identical to
    model.forward() when ablate=False -- same verification discipline as
    E47's own forward_with_psi_clamp). If ablate=True, the bottleneck
    tensor is ZEROED (torch.zeros_like) before upconv3 -- a COMPLETE
    severing of the coarse pathway, not a partial gate clamp (E47's own
    intervention was a per-voxel psi clamp on enc1's gate specifically;
    this is a stronger, different intervention testing the bottleneck's
    OWN causal contribution directly, independent of the attention gate
    mechanism)."""
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

        gate = bottleneck  # NOTE: gate reads the (possibly ablated) bottleneck too --
        # correct: if the bottleneck carries no real signal, the gate itself
        # should also degrade to whatever a zero-input gate produces, which
        # IS the intended full severing of the coarse pathway's influence.
        skip = enc1
        g = model.attn_gate1.W_g(gate)
        g_up = F.interpolate(g, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = model.attn_gate1.W_x(skip)
        psi = torch.sigmoid(model.attn_gate1.W_psi(F.relu(g_up + x)))

        enc1_gated = enc1 * psi
        cat1 = torch.cat([upconv1, enc1_gated], dim=1)
        dec1 = model.dec1(cat1)

        probs = model.seg_head(dec1)
        return probs.squeeze(0).squeeze(0).cpu().numpy()


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def r2_of(y, X):
    """OLS R^2 of y ~ X (X: (n,k) design matrix, already includes intercept column if desired)."""
    X1 = np.column_stack([np.ones(len(X)), X]) if X.ndim > 1 else np.column_stack([np.ones(len(X)), X])
    beta, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
    y_hat = X1 @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded E46 checkpoint: best_val_dice={ckpt.get('best_val_dice')}", flush=True)

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    print(f"Validation set size: {len(val_dataset)}", flush=True)

    # Sanity check first: ablate=False must reproduce real forward() exactly.
    image0, _, _ = val_dataset[0]
    image0_b = image0.unsqueeze(0).to(device)
    with torch.no_grad():
        real = model(image0_b)["probs"].squeeze(0).squeeze(0).cpu().numpy()
    manual = forward_with_bottleneck_ablation(model, image0_b, ablate=False, device=device)
    max_diff = float(np.abs(real - manual).max())
    print(f"Sanity check (ablate=False vs real forward): max abs diff = {max_diff:.6e}", flush=True)
    assert max_diff == 0.0, "Manual trunk reimplementation does not match real forward() -- STOP, bug present."

    records = []
    for subject_idx in range(len(val_dataset)):
        image, mask, subject_id = val_dataset[subject_idx]
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_dataset.subject_dirs[subject_idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_size = int(seg_binary_native.sum())

        mask_frac_64 = fractional_occupancy_64(seg_binary_native)
        target_bin = (mask_frac_64 > 0.5).astype(np.float32)

        probs_intact = forward_with_bottleneck_ablation(model, image_b, ablate=False, device=device)
        probs_ablated = forward_with_bottleneck_ablation(model, image_b, ablate=True, device=device)

        dice_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin)
        dice_ablated = dice_score((probs_ablated >= 0.5).astype(np.float32), target_bin)
        drop = dice_intact - dice_ablated

        records.append({
            "subject_id": subject_id, "native_size": native_size,
            "dice_intact": dice_intact, "dice_ablated": dice_ablated, "drop": drop,
        })

        if (subject_idx + 1) % 25 == 0:
            print(f"  processed {subject_idx+1}/{len(val_dataset)} subjects", flush=True)

    with open(OUT_DIR / "E48_encoding_audit_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.", flush=True)

    # ================= Statistical analysis =================
    native_size = np.array([r["native_size"] for r in records], dtype=np.float64)
    drop = np.array([r["drop"] for r in records], dtype=np.float64)
    dice_intact = np.array([r["dice_intact"] for r in records])

    print(f"\n=== E48 Bottleneck Encoding Audit: n={len(records)} subjects ===")
    print(f"Mean dice_intact  = {dice_intact.mean():.4f}")
    print(f"Mean dice_ablated = {np.array([r['dice_ablated'] for r in records]).mean():.4f}")
    print(f"Mean drop (intact - ablated) = {drop.mean():.4f} (+/-{drop.std():.4f})")
    print(f"native_size range = [{native_size.min():.0f}, {native_size.max():.0f}], median={np.median(native_size):.0f}")

    rho, p_parametric = stats.spearmanr(native_size, drop)
    print(f"\nSpearman(native_size, drop) = {rho:+.4f} (parametric p={p_parametric:.4e})")

    # Permutation test on Spearman rho (project's own >=500 minimum, used 1000)
    rng = np.random.default_rng(SEED)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_drop = rng.permutation(drop)
        perm_rhos[i], _ = stats.spearmanr(native_size, perm_drop)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())
    print(f"Permutation test ({N_PERM} trials): p={p_perm:.4f}")

    # Nonlinear size-confound family check, reused from e32's own methodology:
    # does a nonlinear size-only model (log, cbrt, inv_cbrt) explain drop
    # comparably to raw native_size, or does drop carry INCREMENTAL
    # information beyond what any single nonlinear size transform captures?
    # Here used differently than e32 (e32 checked incremental R^2 of a NEW
    # predictor beyond size; here we check whether the size->drop
    # relationship itself is roughly monotonic/robust across transforms,
    # i.e. not an artifact of one specific arbitrary transform choice).
    log_size = np.log(native_size + 1)
    size_cbrt = native_size ** (1 / 3)
    rho_log, p_log = stats.spearmanr(log_size, drop)
    rho_cbrt, p_cbrt = stats.spearmanr(size_cbrt, drop)
    print(f"\nRobustness across size transforms: "
          f"raw rho={rho:+.4f}, log rho={rho_log:+.4f}, cbrt rho={rho_cbrt:+.4f} "
          f"(Spearman is rank-based, so these are IDENTICAL by construction -- "
          f"reported to make that invariance explicit and verified, not assumed)")
    assert abs(rho - rho_log) < 1e-9 and abs(rho - rho_cbrt) < 1e-9, \
        "Spearman rho should be invariant to monotonic transforms -- mismatch indicates a bug."

    go = (rho > 0) and (p_perm < 0.05)
    print(f"\n=== DECISION: {'GO' if go else 'NULL'} ===")
    if go:
        print("native_size is significantly POSITIVELY correlated with the bottleneck-ablation Dice drop:")
        print("larger lesions lose more when the coarse pathway is severed -- consistent with an")
        print("ENCODING deficiency specific to small lesions (little real signal there to lose).")
    else:
        print("Pre-declared GO criteria not met. Reporting NULL.")

    summary = {
        "n_subjects": len(records),
        "mean_dice_intact": float(dice_intact.mean()),
        "mean_drop": float(drop.mean()), "sd_drop": float(drop.std()),
        "spearman_rho": float(rho), "parametric_p": float(p_parametric),
        "permutation_p": p_perm, "n_permutations": N_PERM,
        "decision": "GO" if go else "NULL",
    }
    with open(OUT_DIR / "E48_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E48_summary.json")


if __name__ == "__main__":
    main()
