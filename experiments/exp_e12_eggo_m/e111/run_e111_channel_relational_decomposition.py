"""
Phase E111: does the bottleneck self-predictive signal (E71/E109/E110)
depend on INDIVIDUAL channel-slot content (H1), or on the JOINT/RELATIONAL
configuration across channels within a subject (H2)?

BACKGROUND: E110's full within-subject channel permutation collapsed the
FLATTEN readout's held-out signal (rho 0.850 -> 0.097, p=0.628). But per
the CNN channel-permutation-symmetry literature (raised explicitly by the
user, citing e.g. ICLR 2024's discussion of permutation symmetries), a
full permutation confounds two distinct things: (a) individual channel
IDENTITY/content might matter (H1), and (b) the JOINT, within-subject
RELATIONAL structure across channel pairs might matter (H2) -- both are
destroyed by any full permutation, since a full permutation necessarily
scrambles which pairs of channels co-occur together for a given subject.

THIS PHASE'S DESIGN (pre-registered before running): the MARGINAL-
PRESERVING RESHUFFLE. For each subject i and a fraction f of the 256
channel-slots (chosen uniformly at random per subject per draw), replace
slot c's content with a DIFFERENT, independently-and-uniformly-drawn
subject j's real content at that SAME slot c (donor drawn fresh per
corrupted slot, not one donor for the whole subject). Critically:
  - Each channel-slot's ACROSS-SUBJECT marginal distribution is exactly
    preserved (slot c's content is always some real subject's real slot-c
    content -- nothing is invented, only resampled).
  - The label used for training/evaluation is ALWAYS subject i's own
    real, independently-measured N_b -- never a synthesized label. This
    tests: how much of subject i's own real channel-slot content can be
    replaced by unrelated donors' content before N_b(i) becomes
    unpredictable?
  - At f=0: identical to ORIGINAL (E110's baseline).
  - At f=1: every slot is donor content -- but unlike E110's
    CHANNEL_SHUFFLE (which reorders subject i's OWN content across
    slots, preserving i's own per-channel value SET), f=1 here means
    NONE of subject i's own bottleneck content survives at all -- the
    label (N_b(i)) is being predicted from an entirely different,
    independently-sampled representation. This is a DIFFERENT, stronger
    corruption than E110's full shuffle, included as the f=1 anchor.

SWEEP: f in {0, 0.125, 0.25, 0.5, 0.75, 1.0}, FLATTEN readout only
(POOLED is uninformative here: global-average-pooling only cares about
which VALUES are present per channel-slot on average, not which subject
they came from, so donor substitution barely perturbs POOLED's input
distribution at any f -- this was true by construction, not found
"surprising" after the fact, so POOLED is excluded from this phase to
avoid wasting compute on a readout known in advance not to discriminate
here).

PRE-DECLARED INTERPRETATION (stated before running):
  - GRADED decay (rho falls off smoothly and roughly proportionally to
    f, e.g. still >half of rho(f=0) at f=0.5): individual per-slot
    content dominates and is roughly ADDITIVE/redundant across slots --
    H1-leaning, a DISTRIBUTED code where many slots individually carry
    usable signal, no strong dependence on their joint configuration.
  - SHARP drop at LOW f (e.g. rho already collapses by f=0.125-0.25):
    the signal is fragile to even minor cross-subject content mixing --
    H2-leaning, implicates the JOINT/relational configuration across
    channels being load-bearing, not merely which subject's content
    happens to sit in each slot.
  - FLAT then cliff (rho stays near rho(f=0) until some f*, then drops
    sharply): a genuinely distributed/redundant code up to a capacity
    limit -- reported as its own distinct outcome, not forced into H1 or
    H2.
  A "sharp drop" is defined the same way as E110: rho falling below half
  of rho(f=0), AND losing permutation-test significance (p>0.05).

Reuses E110's verified labeling pipeline (bit-for-bit identical
bottleneck-ablation construction, same checkpoint, same train/val split)
and E110's FLATTEN readout (fixed random projection to 512, since direct
131072-dim training was found in E110 to collapse under any
subject-mixing corruption regardless of weight-decay/LR tuning) and
training-set-fit + degenerate-output diagnostics (both bugs caught and
fixed in E110 before any result there was trusted).
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
N_SCRAMBLE_DRAWS = 5  # independent random-donor draws per subject per fraction,
                       # averaged, to avoid a single unlucky/lucky donor assignment driving the result

CKPT_PATH = (project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs"
             / "AttnGate_seed0" / "checkpoints" / "best.pth")
EXPECTED_VAL_DICE = 0.9101624600589275

FRACTIONS = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]
N_CHANNELS = 256


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


# ==================== Verified-identical-to-E48/E109/E110 ablation construction ====================
def forward_with_bottleneck_ablation(model, image, ablate, device):
    """Bit-for-bit identical to E109/E110's verified construction."""
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

        gate = bottleneck
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


# ==================== Marginal-preserving reshuffle ====================
def marginal_preserving_reshuffle(subject_idx, fraction, all_bottlenecks, rng):
    """all_bottlenecks: (N, 256, 8, 8, 8) tensor of ALL subjects in the
    pool (the donor population). Returns subject_idx's bottleneck with a
    `fraction` of its 256 channel-slots replaced by a FRESH,
    independently-drawn OTHER subject's content at that same slot
    (donor drawn per corrupted slot, with replacement, excluding
    subject_idx itself as its own donor to guarantee genuine corruption
    at every selected slot)."""
    n_pool = all_bottlenecks.shape[0]
    original = all_bottlenecks[subject_idx].clone()
    n_corrupt = int(round(fraction * N_CHANNELS))
    if n_corrupt == 0:
        return original
    corrupt_slots = rng.choice(N_CHANNELS, size=n_corrupt, replace=False)
    donor_pool = [j for j in range(n_pool) if j != subject_idx]
    donor_indices = rng.choice(donor_pool, size=n_corrupt, replace=True)
    for slot, donor_idx in zip(corrupt_slots, donor_indices):
        original[slot] = all_bottlenecks[donor_idx, slot]
    return original


# ==================== Readout (FLATTEN only, per design rationale above) ====================
class FlattenHead(nn.Module):
    """Identical to E110's fixed-random-projection design (verified
    there to avoid the capacity collapse of direct 131072-dim training)."""
    PROJ_DIM = 512

    def __init__(self, in_dim=256 * 8 * 8 * 8, hidden=32, proj_seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(proj_seed)
        proj = torch.randn(in_dim, self.PROJ_DIM, generator=gen) / (in_dim ** 0.5)
        self.register_buffer("proj", proj)
        self.fc1 = nn.Linear(self.PROJ_DIM, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, bottleneck):
        x = bottleneck.flatten(1) @ self.proj
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    val_dice = ckpt.get("best_val_dice") or ckpt.get("val_dice")
    print(f"Loaded canonical checkpoint: val_dice={val_dice}")
    assert abs(float(val_dice) - EXPECTED_VAL_DICE) < 1e-6, "Checkpoint identity check FAILED"
    print("[Sanity check] checkpoint identity PASS.\n")

    val_ds = BraTSDataset(root_dir="Dataset/Training", split="val", val_split=0.1,
                          target_shape=(64, 64, 64), normalize=True)
    train_ds = BraTSDataset(root_dir="Dataset/Training", split="train", val_split=0.1,
                            target_shape=(64, 64, 64), normalize=True)

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
    print(f"Validation N_b: mean={n_val.mean():.4f} std={n_val.std():.4f}\n")

    train_bn_orig = torch.stack([r["bottleneck"] for r in train_records])  # (200,256,8,8,8) CPU
    val_bn_orig = torch.stack([r["bottleneck"] for r in val_records])

    # Donor pools: training subjects corrupted using OTHER TRAINING subjects
    # as donors (never val); validation subjects corrupted using OTHER
    # VALIDATION subjects as donors -- keeps train/val strictly separate,
    # consistent with how the readout itself is trained/evaluated.
    results = {}
    for fraction in FRACTIONS:
        print(f"=== FRACTION={fraction} ===", flush=True)
        cell_rhos = []
        cell_ps = []
        cell_capacity_limited = []
        cell_train_fit_rhos = []
        n_draws = N_SCRAMBLE_DRAWS if fraction != 0.0 else 1
        for draw in range(n_draws):
            draw_rng = np.random.default_rng(SEED * 1000 + draw)

            train_bn_cond = torch.stack([
                marginal_preserving_reshuffle(i, fraction, train_bn_orig, draw_rng)
                for i in range(len(train_records))
            ]).to(device)
            val_bn_cond = torch.stack([
                marginal_preserving_reshuffle(i, fraction, val_bn_orig, draw_rng)
                for i in range(len(val_records))
            ]).to(device)

            torch.manual_seed(SEED)
            aux = FlattenHead().to(device)
            optimizer = torch.optim.Adam(aux.parameters(), lr=1e-3, weight_decay=1e-4)
            train_n_t = torch.tensor(n_train, dtype=torch.float32, device=device)

            loss_history = []
            for epoch in range(AUX_EPOCHS):
                aux.train()
                optimizer.zero_grad(set_to_none=True)
                pred = aux(train_bn_cond)
                loss = F.mse_loss(pred, train_n_t)
                loss.backward()
                optimizer.step()
                loss_history.append(float(loss.item()))

            aux.eval()
            with torch.no_grad():
                pred_val = aux(val_bn_cond).cpu().numpy()
                pred_train = aux(train_bn_cond).cpu().numpy()

            pred_std = float(np.std(pred_val))
            pred_train_std = float(np.std(pred_train))

            train_rho_fit = float("nan")
            is_train_degenerate = (pred_train_std < 1e-4) or not np.isfinite(pred_train).all()
            if not is_train_degenerate:
                train_rho_fit, _ = stats.spearmanr(pred_train, n_train)
                if not np.isfinite(train_rho_fit):
                    is_train_degenerate = True
            if draw == 0:
                print(f"    [draw 0 diagnostic] loss: epoch1={loss_history[0]:.6f} "
                      f"epoch{AUX_EPOCHS//2}={loss_history[AUX_EPOCHS//2-1]:.6f} "
                      f"epoch{AUX_EPOCHS}={loss_history[-1]:.6f} | "
                      f"pred_val std={pred_std:.6f} (n_val std={n_val.std():.6f}) | "
                      f"pred_train std={pred_train_std:.6f}, TRAIN-SET fit "
                      f"rho={train_rho_fit:+.4f}", flush=True)
            if is_train_degenerate:
                print(f"    [CAPACITY LIMIT] draw {draw}: readout collapsed to a "
                      f"near-constant prediction on its OWN TRAINING data "
                      f"(pred_train std={pred_train_std:.2e}) -- optimization/capacity "
                      f"failure, NOT evidence this fraction destroys signal.", flush=True)
                cell_capacity_limited.append(True)
                continue
            cell_capacity_limited.append(False)
            cell_train_fit_rhos.append(train_rho_fit)

            is_val_degenerate = (pred_std < 1e-4) or not np.isfinite(pred_val).all()
            if not is_val_degenerate:
                rho_check, _ = stats.spearmanr(pred_val, n_val)
                if not np.isfinite(rho_check):
                    is_val_degenerate = True
            if is_val_degenerate:
                print(f"    [WARNING] draw {draw}: pred_val (held-out) is NaN/constant "
                      f"despite training fit rho={train_rho_fit:+.4f} -- complete "
                      f"generalization failure; recording as rho=0.0 explicitly.", flush=True)
                cell_rhos.append(0.0)
                cell_ps.append(1.0)
                continue

            rho, _ = stats.spearmanr(pred_val, n_val)
            cell_rhos.append(rho)

            rng_perm = np.random.default_rng(SEED)
            perm_rhos = np.empty(N_PERM)
            for i in range(N_PERM):
                perm_y = rng_perm.permutation(n_val)
                perm_rhos[i], _ = stats.spearmanr(pred_val, perm_y)
            p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())
            cell_ps.append(p_perm)

        n_valid_draws = len(cell_rhos)
        n_capacity_limited = sum(cell_capacity_limited)
        mean_train_fit_rho = float(np.mean(cell_train_fit_rhos)) if cell_train_fit_rhos else None
        if n_valid_draws == 0:
            print(f"  ALL {n_draws} draws degenerate (capacity_limited={n_capacity_limited}/{n_draws}) "
                  f"-- this fraction's comparison is UNDEFINED.\n", flush=True)
            results[str(fraction)] = {
                "rho_per_draw": [], "mean_rho": None, "worst_case_p": None,
                "n_valid_draws": 0, "n_requested_draws": n_draws,
                "n_capacity_limited": n_capacity_limited,
                "mean_train_fit_rho": mean_train_fit_rho,
                "UNDEFINED_ALL_DRAWS_DEGENERATE": True,
            }
            continue

        mean_rho = float(np.mean(cell_rhos))
        max_p = float(np.max(cell_ps))
        print(f"  rho draws: {[f'{r:+.3f}' for r in cell_rhos]} "
              f"({n_valid_draws}/{n_draws} valid, {n_capacity_limited} capacity-limited)")
        print(f"  mean rho={mean_rho:+.4f}, worst-case permutation p={max_p:.4f}, "
              f"mean TRAIN-SET fit rho={mean_train_fit_rho:+.4f}\n", flush=True)

        results[str(fraction)] = {
            "rho_per_draw": [float(r) for r in cell_rhos],
            "mean_rho": mean_rho, "worst_case_p": max_p,
            "n_valid_draws": n_valid_draws, "n_requested_draws": n_draws,
            "n_capacity_limited": n_capacity_limited,
            "mean_train_fit_rho": mean_train_fit_rho,
        }

    # ---------------- Pre-declared interpretation ----------------
    print("\n" + "=" * 70)
    print("PRE-DECLARED INTERPRETATION")
    print("=" * 70)

    def fmt(f):
        r = results[str(f)]["mean_rho"]
        return f"{r:+.4f}" if r is not None else "UNDEFINED"

    rho_by_f = {f: results[str(f)]["mean_rho"] for f in FRACTIONS}
    p_by_f = {f: results[str(f)]["worst_case_p"] for f in FRACTIONS}
    print("Sweep: " + ", ".join(f"f={f}: rho={fmt(f)}" for f in FRACTIONS))

    if any(v is None for v in rho_by_f.values()):
        verdict = "INCONCLUSIVE_DEGENERATE_TRAINING"
        print("\n[STOP] At least one fraction produced only degenerate output -- "
              "cannot draw the graded/sharp-drop distinction from this run.")
    else:
        rho0 = abs(rho_by_f[0.0])
        half = 0.5 * rho0
        # Find the first fraction (>0) at which the sharp-drop criterion is met.
        first_drop_f = None
        for f in FRACTIONS[1:]:
            if abs(rho_by_f[f]) < half and p_by_f[f] > 0.05:
                first_drop_f = f
                break
        if first_drop_f is None:
            verdict = "NO_SHARP_DROP_GRADED_OR_ROBUST"
        elif first_drop_f <= 0.25:
            verdict = "SHARP_EARLY_DROP_H2_RELATIONAL_LEANING"
        elif first_drop_f >= 0.75:
            verdict = "FLAT_THEN_CLIFF_DISTRIBUTED_CODE"
        else:
            verdict = "GRADED_DECAY_H1_INDIVIDUAL_CONTENT_LEANING"
        print(f"\nFirst fraction meeting sharp-drop criterion (rho<{half:.4f} AND p>0.05): "
              f"{first_drop_f}")

    print(f"\n=== FINAL VERDICT: {verdict} ===")
    print("(Convenience label only -- inspect the actual rho/p sweep above directly, "
          "per this project's own established convention of not trusting an "
          "auto-classifier's label without checking the underlying numbers.)")

    summary = {
        "checkpoint": str(CKPT_PATH), "val_dice_check": val_dice,
        "n_train_labels": len(train_records), "n_val_labels": len(val_records),
        "n_scramble_draws": N_SCRAMBLE_DRAWS, "fractions": FRACTIONS,
        "sweep_results": results,
        "final_verdict_CONVENIENCE_LABEL_ONLY": verdict,
    }
    with open(OUT_DIR / "E111_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E111_summary.json")


if __name__ == "__main__":
    main()
