"""
Phase E96: Timing Diagnostic -- Is the Necessity/Context-Benefit
Relationship Strongest at Low Resolution?

CONTEXT: E95 found rho(N_b, Delta_D_context) = +0.436 (p<1e-6), a real,
confound-checked relationship between causal bottleneck necessity and
benefit from spatial context, using a FIXED 64^3-trained checkpoint.
User's critical caution before writing any training code: E95 shows
necessity predicts context benefit, but does NOT yet show that EARLY
access to more context during training is the relevant mechanism --
that is an assumption PGPS-NC's design depends on.

Architecture decision (resolved with user): NeuroScan has no patch-based
training (whole-volume resize to 64^3 only, no spatial crops) -- PGPS's
literal patch-size curriculum does not apply directly. Adapting PGPS's
PRINCIPLE (class balance / detail availability grows over training) to
this pipeline: reinterpret "context" as INPUT RESOLUTION rather than
patch/crop size. Curriculum becomes: train at low resolution early
(more background-dominated per-voxel, less fine detail, similar
class-imbalance-mitigation rationale to PGPS's own patch-size argument),
grow to full 64^3 later.

THIS PHASE (still zero-training, using the SAME fixed checkpoint) tests
whether the N_b/context-benefit relationship is STRONGEST at LOW
resolution specifically -- i.e. does necessity predict resolution
benefit MORE when resolution is low (simulating early-training
conditions) than when it's already high (simulating late-training,
where PGPS's own schedule would already have provided full detail)?

METHODOLOGY: for each subject, compute Dice at THREE resolutions
(16^3, 32^3, 64^3 -- all inputs resized via the SAME resize convention
as the standard pipeline, then run through the SAME fixed 64^3-trained
checkpoint by upsampling the LOW-res input back to 64^3 before feeding
the network, since the network's conv/pool structure requires a fixed
64^3 input size -- this simulates "how much true detail is available"
while keeping the network's own input contract unchanged, directly
analogous to E95's own crop-then-resize-to-64^3 method).

  Delta_D_low  = Dice(64^3 native detail) - Dice(16^3 detail, upsampled)
  Delta_D_mid  = Dice(64^3 native detail) - Dice(32^3 detail, upsampled)

Then test:
  rho(N_b, Delta_D_low)  -- necessity vs benefit from resolution, LOW-res gap
  rho(N_b, Delta_D_mid)  -- necessity vs benefit from resolution, MID-res gap

PRE-DECLARED INTERPRETATION:
  - If rho(N_b, Delta_D_low) > rho(N_b, Delta_D_mid) meaningfully (e.g.
    by >0.1, with both individually significant), and both are
    positive: the necessity/context-benefit relationship IS stronger
    when resolution is more severely reduced -- consistent with
    "high-N_b subjects specifically need EARLY access to more detail,"
    supporting PGPS-NC's core design assumption (advance high-N_b
    subjects to higher resolution earlier).
  - If the relationship is roughly EQUAL in strength at both resolution
    gaps, or STRONGER at the smaller (mid) gap: necessity predicts
    context benefit in general, but not specifically an EARLY-TRAINING-
    TIMING effect -- conditioning the TIMING of resolution growth on
    N_b would not be well-motivated by this data; a different
    intervention (e.g. simply training high-N_b subjects at higher
    resolution THROUGHOUT, not on a different timing schedule) would be
    better supported instead.
  - If either relationship is weak/null: reconsider whether the E95
    finding generalizes across resolution gaps at all, or is specific
    to the particular crop-based context reduction E95 used.

Checkpoint lineage: SAME single checkpoint as E48-E95 (asserted).
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
E48_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    if denom == 0:
        return 1.0
    return float(2 * tp / denom)


def resize_np(volume_np, size, mode):
    t = torch.from_numpy(volume_np).unsqueeze(0).unsqueeze(0).float()
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    resized = F.interpolate(t, size=size, mode=mode, **kwargs)
    return resized.squeeze(0).squeeze(0).numpy()


def make_resolution_variant(flair_norm_64, target_res):
    """Downsample the standard 64^3 input to target_res (simulating less
    available detail at that resolution), then upsample BACK to 64^3
    (the network's fixed required input size) -- the resulting tensor
    has genuinely less true spatial detail than the native 64^3 input,
    even though its shape is unchanged, exactly analogous to E95's
    crop-then-resize approach for context reduction."""
    down = resize_np(flair_norm_64, (target_res, target_res, target_res), mode="trilinear")
    back_up = resize_np(down, (64, 64, 64), mode="trilinear")
    return back_up


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    assert abs(ckpt.get("best_val_dice", 0) - 0.9101624600589275) < 1e-9, \
        "Checkpoint mismatch -- must match E48-E95's exact checkpoint."
    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded checkpoint, val_dice={ckpt.get('best_val_dice')}", flush=True)

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    print(f"Validation set size: {len(val_dataset)}", flush=True)

    records = []
    for subject_idx in range(len(val_dataset)):
        image, mask, subject_id = val_dataset[subject_idx]
        if subject_id not in e48_by_id:
            continue

        flair_64 = image.squeeze(0).numpy()  # already normalized, 64^3, standard pipeline convention
        target_bin = mask.squeeze(0).numpy()
        target_bin = (target_bin > 0.5).astype(np.float32)

        variant_16 = make_resolution_variant(flair_64, 16)
        variant_32 = make_resolution_variant(flair_64, 32)

        with torch.no_grad():
            def run(vol):
                t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
                probs = model(t)["probs"].squeeze(0).squeeze(0).cpu().numpy()
                return dice_score((probs >= 0.5).astype(np.float32), target_bin)

            dice_64 = run(flair_64)
            dice_32 = run(variant_32)
            dice_16 = run(variant_16)

        delta_low = dice_64 - dice_16
        delta_mid = dice_64 - dice_32

        records.append({
            "subject_id": subject_id,
            "native_size": e48_by_id[subject_id]["native_size"],
            "N_b": e48_by_id[subject_id]["drop"],
            "dice_64": dice_64, "dice_32": dice_32, "dice_16": dice_16,
            "delta_D_low": delta_low, "delta_D_mid": delta_mid,
        })

        if len(records) % 25 == 0:
            print(f"  processed {len(records)} subjects", flush=True)

    with open(OUT_DIR / "E96_timing_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.", flush=True)

    n_b = np.array([r["N_b"] for r in records])
    delta_low = np.array([r["delta_D_low"] for r in records])
    delta_mid = np.array([r["delta_D_mid"] for r in records])

    print(f"\n=== E96 Timing Diagnostic: n={len(records)} ===")
    print(f"Mean dice_64={np.mean([r['dice_64'] for r in records]):.4f}, "
          f"dice_32={np.mean([r['dice_32'] for r in records]):.4f}, "
          f"dice_16={np.mean([r['dice_16'] for r in records]):.4f}")
    print(f"Mean delta_D_low (64-16) = {delta_low.mean():+.4f}, "
          f"Mean delta_D_mid (64-32) = {delta_mid.mean():+.4f}")

    def perm_test(x, y, seed):
        rho, p_param = stats.spearmanr(x, y)
        rng = np.random.default_rng(seed)
        perm_rhos = np.empty(N_PERM)
        for i in range(N_PERM):
            perm_rhos[i], _ = stats.spearmanr(x, rng.permutation(y))
        p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())
        return float(rho), float(p_param), p_perm

    rho_low, p_low_param, p_low_perm = perm_test(n_b, delta_low, SEED)
    rho_mid, p_mid_param, p_mid_perm = perm_test(n_b, delta_mid, SEED + 1)

    print(f"\nSpearman(N_b, delta_D_low)  = {rho_low:+.4f} (p_param={p_low_param:.4e}, p_perm={p_low_perm:.4f})")
    print(f"Spearman(N_b, delta_D_mid)  = {rho_mid:+.4f} (p_param={p_mid_param:.4e}, p_perm={p_mid_perm:.4f})")

    both_significant = (p_low_perm < 0.05) and (p_mid_perm < 0.05)
    low_stronger = (rho_low - rho_mid) > 0.1

    if both_significant and low_stronger and rho_low > 0 and rho_mid > 0:
        verdict = "TIMING_SUPPORTED"
        detail = ("Necessity/context-benefit relationship is meaningfully STRONGER at low resolution "
                   "than mid resolution -- supports PGPS-NC's core design: high-N_b subjects specifically "
                   "benefit from EARLY access to more detail, motivating necessity-conditioned TIMING of "
                   "resolution growth (not just a static resolution difference).")
    elif both_significant and not low_stronger:
        verdict = "TIMING_NOT_SUPPORTED_STATIC_EFFECT_LIKELY"
        detail = ("Necessity predicts context benefit roughly EQUALLY at both resolution gaps -- this is a "
                   "general 'high-N_b subjects need more detail' effect, NOT specifically an early-training-"
                   "timing effect. Conditioning curriculum TIMING on N_b is not well-motivated by this data; "
                   "training high-N_b subjects at higher resolution THROUGHOUT (not a different schedule) "
                   "would be the better-supported intervention instead.")
    else:
        verdict = "WEAK_OR_INCONSISTENT"
        detail = "One or both relationships are weak/non-significant -- reconsider before designing PGPS-NC."

    print(f"\n=== DECISION: {verdict} ===")
    print(detail)

    summary = {
        "checkpoint_val_dice": ckpt.get("best_val_dice"),
        "mean_delta_D_low": float(delta_low.mean()), "mean_delta_D_mid": float(delta_mid.mean()),
        "rho_Nb_deltaD_low": rho_low, "p_param_low": p_low_param, "p_perm_low": p_low_perm,
        "rho_Nb_deltaD_mid": rho_mid, "p_param_mid": p_mid_param, "p_perm_mid": p_mid_perm,
        "verdict": verdict, "detail": detail,
    }
    with open(OUT_DIR / "E96_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E96_summary.json")


if __name__ == "__main__":
    main()
