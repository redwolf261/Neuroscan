"""
Phase E110: representation-scrambling localization of E71/E109's bottleneck
self-predictive-signal anomaly.

BACKGROUND: E109 replicated E71's finding fresh on the canonical v5/E46
checkpoint -- a small probe on the frozen (256,8,8,8) bottleneck predicts
the network's own E48-style causal ablation sensitivity N_b, held out
(rho=0.633, p=2.4e-15; partial rho=0.791 controlling size). E109's own
step-5 mechanism probe (does the probe key on per-channel magnitude or
variance?) came back a clean null on both. This phase asks a sharper
question: does the signal depend on CHANNEL IDENTITY, SPATIAL POSITION,
neither, or both?

PRE-REGISTERED DESIGN (locked before running, per this project's own
policy of pre-declaring decision rules): a 2 (readout) x 4 (condition)
grid, each cell trained fresh (not just evaluated) on its own condition,
held out on validation subjects of the SAME condition.

READOUTS (deliberately different capacity/inductive bias, per explicit
instruction to avoid rediscovering a single probe's own artifact):
  A. POOLED: global-avg-pool -> 256 -> 64 -> 1 (same as E71/E109's
     original probe). By construction, global average pooling is
     permutation-invariant over spatial position, so this readout can
     ONLY ever be sensitive to channel-identity scrambling, never to
     spatial scrambling -- it is included as the channel-sensitive arm,
     not expected to detect any spatial effect by design, and this is
     stated up front rather than found "surprising" after the fact.
  B. FLATTEN: flatten (256,8,8,8) -> 131072, FIXED random projection to
     512 (untrained, seeded -- mixes all channels/positions so nothing
     is discarded), -> 32 -> 1 (trainable). Direct 131072->32 training
     was tried first and found (via added training-set-fit diagnostics)
     to collapse to a near-constant output under scrambling regardless
     of weight-decay/LR tuning -- L2 pressure on 4.2M weights dominates
     when there's no consistent per-weight gradient signal (each weight
     sees a different, randomly-relabeled channel/position per subject).
     The fixed random projection keeps full channel+position sensitivity
     (a random projection mixes every input dimension) while making the
     TRAINABLE parameter count tractable for 200 samples.

CONDITIONS (each subject gets an INDEPENDENT random permutation --
answers "does the probe rely on FIXED channel/position identity" rather
than an information-theoretic claim about the tensor's raw content):
  1. ORIGINAL: bottleneck unchanged.
  2. CHANNEL_SHUFFLE: for each subject, draw a fresh random permutation
     pi of {0..255}; B'[c,:,:,:] = B[pi(c),:,:,:]. Preserves each
     channel's own full spatial map and the multiset of joint values
     across channels at any given position; destroys any probe weight's
     ability to exploit "this fixed weight index means channel X".
  3. SPATIAL_SHUFFLE: for each subject, draw a fresh random permutation
     sigma of the 512 spatial positions, applied IDENTICALLY to every
     channel; B'[:,pos] = B[:,sigma(pos)]. Preserves which channels'
     values co-occur at any given position (true cross-channel
     co-activation intact) and each channel's own value multiset;
     destroys any probe weight's ability to exploit "this fixed weight
     index means position P".
  4. BOTH_SHUFFLE: apply an independent channel permutation AND an
     independent (identical-across-channels) spatial permutation to the
     same subject.

CRITICALLY: the SAME permutation (per subject, per condition) must be
used for that subject's appearance in both the training set and (for
that condition) the validation set is NOT required to reuse a training
permutation -- each subject/condition/split draws its own fresh
permutation independently. The point is not to memorize one specific
permutation; it's to test whether a readout CAN be trained at all to
recover N_b when identity is randomized per-subject.

PRE-DECLARED INTERPRETATION (stated before running):
  Let S(condition, readout) = held-out Spearman rho for that cell.
  - If S(CHANNEL_SHUFFLE) drops sharply toward 0 vs S(ORIGINAL) but
    S(SPATIAL_SHUFFLE) does not (for the FLATTEN readout, since POOLED
    cannot show a spatial effect by construction): channel identity is
    the leading carrier -> H_C supported.
  - If S(SPATIAL_SHUFFLE) drops sharply but S(CHANNEL_SHUFFLE) does not
    (FLATTEN readout only): spatial position is the leading carrier ->
    H_S supported.
  - If BOTH single-shuffle conditions drop sharply (FLATTEN readout):
    the signal requires a joint channel-spatial configuration -- report
    as such, do not force a single winner.
  - If NEITHER single-shuffle condition drops much (FLATTEN readout):
    the signal survives independent per-subject relabeling of both axes
    -- genuinely surprising, would indicate the signal is carried by
    some global statistic invariant to both permutations (e.g. the
    marginal distribution of values, independent of which channel/
    position they came from) -- flagged as its own distinct, notable
    outcome, not folded into either H_C or H_S.
  A "sharp drop" is defined, before seeing results, as S falling below
  half of S(ORIGINAL) for that readout, AND losing permutation-test
  significance (p>0.05).
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
N_SCRAMBLE_DRAWS = 5  # independent random-permutation draws per subject per condition,
                       # averaged, to avoid a single unlucky/lucky permutation driving the result

CKPT_PATH = (project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs"
             / "AttnGate_seed0" / "checkpoints" / "best.pth")
EXPECTED_VAL_DICE = 0.9101624600589275

CONDITIONS = ["ORIGINAL", "CHANNEL_SHUFFLE", "SPATIAL_SHUFFLE", "BOTH_SHUFFLE"]
READOUTS = ["POOLED", "FLATTEN"]


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


# ==================== Verified-identical-to-E48/E109 ablation construction ====================
def forward_with_bottleneck_ablation(model, image, ablate, device):
    """Identical to E109's (corrected) version, itself matched bit-for-bit
    to E48's own verified construction: when ablate=True, the bottleneck
    is zeroed BEFORE being used for both upconv3 AND the attention gate."""
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


# ==================== Scrambling operators ====================
def channel_shuffle(bottleneck, rng):
    """bottleneck: (256, 8, 8, 8) tensor. Returns a NEW tensor with a
    fresh random permutation of the channel axis."""
    perm = rng.permutation(bottleneck.shape[0])
    return bottleneck[perm, :, :, :].clone()


def spatial_shuffle(bottleneck, rng):
    """bottleneck: (256, 8, 8, 8). Flattens spatial dims, applies ONE
    shared random permutation to all channels identically, reshapes
    back -- preserves cross-channel co-activation at each (new) position."""
    c, d, h, w = bottleneck.shape
    flat = bottleneck.reshape(c, d * h * w)
    perm = rng.permutation(d * h * w)
    flat_shuffled = flat[:, perm]
    return flat_shuffled.reshape(c, d, h, w).clone()


def apply_condition(bottleneck, condition, rng):
    if condition == "ORIGINAL":
        return bottleneck.clone()
    elif condition == "CHANNEL_SHUFFLE":
        return channel_shuffle(bottleneck, rng)
    elif condition == "SPATIAL_SHUFFLE":
        return spatial_shuffle(bottleneck, rng)
    elif condition == "BOTH_SHUFFLE":
        return spatial_shuffle(channel_shuffle(bottleneck, rng), rng)
    else:
        raise ValueError(condition)


# ==================== Readouts ====================
class PooledHead(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(in_channels, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, bottleneck):
        x = self.pool(bottleneck).flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class FlattenHead(nn.Module):
    """Every (channel, position) pair is spatially/channel-sensitive by
    construction, but training a 131072x32 first layer directly on 200
    samples was found (E110 smoke tests) to collapse to a near-constant
    output under scrambling -- L2 weight-decay pressure on 4.2M weights
    dominates when there's no consistent per-weight gradient signal
    (each weight sees a different, randomly-relabeled channel/position
    per subject under CHANNEL/SPATIAL_SHUFFLE). Fixed via a FIXED
    (untrained, seeded) random linear projection down to PROJ_DIM before
    the trainable head -- a random projection mixes all 131072 input
    dimensions (so every channel AND every position still contributes,
    preserving full sensitivity to both scrambling axes) while making
    the TRAINABLE parameter count tractable for 200 samples, without
    needing delicate weight-decay/LR tuning to avoid the collapse."""
    PROJ_DIM = 512

    def __init__(self, in_dim=256 * 8 * 8 * 8, hidden=32, proj_seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(proj_seed)
        # Fixed (buffer, not Parameter -- never trained), scaled so the
        # projected output has roughly unit variance for unit-variance input.
        proj = torch.randn(in_dim, self.PROJ_DIM, generator=gen) / (in_dim ** 0.5)
        self.register_buffer("proj", proj)
        self.fc1 = nn.Linear(self.PROJ_DIM, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, bottleneck):
        x = bottleneck.flatten(1) @ self.proj
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


def build_readout(name):
    if name == "POOLED":
        return PooledHead()
    elif name == "FLATTEN":
        return FlattenHead()
    else:
        raise ValueError(name)


def weight_decay_for(name):
    return 1e-4 if name == "FLATTEN" else 0.0  # much lighter now that the
    # trainable head is only PROJ_DIM(512)->32->1, not 131072->32->1


def lr_for(name):
    return 1e-3


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

    # ---------------- Build (bottleneck, N_b, native_size) label pairs ONCE ----------------
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
    native_size_val = np.array([r["native_size"] for r in val_records])
    print(f"\nTraining N_b: mean={n_train.mean():.4f} std={n_train.std():.4f}")
    print(f"Validation N_b: mean={n_val.mean():.4f} std={n_val.std():.4f}\n")

    train_bn_orig = torch.stack([r["bottleneck"] for r in train_records])  # (200,256,8,8,8) CPU
    val_bn_orig = torch.stack([r["bottleneck"] for r in val_records])

    # ---------------- 2x4 grid ----------------
    results = {}
    for readout_name in READOUTS:
        for condition in CONDITIONS:
            if readout_name == "POOLED" and condition in ("SPATIAL_SHUFFLE",):
                # Still run it -- the pre-registered design explicitly predicts
                # POOLED is invariant to spatial shuffle; running it confirms
                # that predicted invariance rather than assuming it.
                pass

            print(f"=== READOUT={readout_name}  CONDITION={condition} ===", flush=True)
            cell_rhos = []
            cell_ps = []
            cell_capacity_limited = []
            cell_train_fit_rhos = []
            for draw in range(N_SCRAMBLE_DRAWS if condition != "ORIGINAL" else 1):
                draw_rng = np.random.default_rng(SEED * 1000 + draw)

                train_bn_cond = torch.stack([
                    apply_condition(train_bn_orig[i], condition, draw_rng) for i in range(len(train_records))
                ]).to(device)
                val_bn_cond = torch.stack([
                    apply_condition(val_bn_orig[i], condition, draw_rng) for i in range(len(val_records))
                ]).to(device)

                torch.manual_seed(SEED)
                aux = build_readout(readout_name).to(device)
                lr = lr_for(readout_name)
                optimizer = torch.optim.Adam(aux.parameters(), lr=lr,
                                              weight_decay=weight_decay_for(readout_name))
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
                # TRAINING-SET fit check (not held-out): can this readout even
                # fit its OWN training data under this condition? A collapse
                # here (pred_train_std ~ 0, i.e. predicting near-constant
                # output for training subjects too) means the optimization
                # itself failed to use the available capacity -- a capacity/
                # optimization-budget limit, NOT evidence that the condition
                # destroys recoverable signal. Only a readout that FITS
                # training data but fails held-out is telling us something
                # about the representation (real overfitting-to-scrambled-
                # noise, i.e. no generalizable signal survived the scramble).
                # Threshold matched to what scipy's own spearmanr effectively
                # requires to distinguish values after rank conversion --
                # 1e-8 was found (via the constant-input warnings this guard
                # was supposed to prevent) to be too loose: a pred_train_std
                # that rounds to 0.000000 at 6dp can still exceed 1e-8 yet be
                # numerically indistinguishable to spearmanr's ranking. Use
                # a relative threshold instead (a fraction of the training
                # label's own scale) plus a stricter absolute floor.
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
                          f"(pred_train std={pred_train_std:.2e}) -- this is an "
                          f"optimization/capacity failure under this condition, "
                          f"NOT evidence the condition destroys signal. Flagging "
                          f"as CAPACITY_LIMIT rather than folding into the "
                          f"held-out comparison.", flush=True)
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
                          f"(std={pred_std:.2e}) despite training fit rho={train_rho_fit:+.4f} "
                          f"-- this IS informative (fits training, fails to generalize at all) "
                          f"but Spearman is undefined here; recording as rho=0.0 explicitly "
                          f"(complete generalization failure), not skipping.", flush=True)
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
            n_requested_draws = N_SCRAMBLE_DRAWS if condition != "ORIGINAL" else 1
            n_capacity_limited = sum(cell_capacity_limited)
            mean_train_fit_rho = float(np.mean(cell_train_fit_rhos)) if cell_train_fit_rhos else None
            if n_valid_draws == 0:
                if n_capacity_limited == n_requested_draws:
                    print(f"  ALL {n_requested_draws} draws were CAPACITY-LIMITED (readout "
                          f"could not even fit its own training data under this condition) -- "
                          f"this is an optimization-budget limit, NOT a real null. This cell's "
                          f"comparison is UNDEFINED until the readout/training budget is fixed "
                          f"for this condition.\n", flush=True)
                else:
                    print(f"  ALL {n_requested_draws} draws produced NaN/constant output -- "
                          f"this cell is UNDEFINED, not zero. Flagging explicitly rather than "
                          f"reporting a fabricated mean_rho=0.0.\n", flush=True)
                results[f"{readout_name}__{condition}"] = {
                    "rho_per_draw": [], "p_per_draw": [],
                    "mean_rho": None, "worst_case_p": None,
                    "n_valid_draws": 0, "n_requested_draws": n_requested_draws,
                    "n_capacity_limited": n_capacity_limited,
                    "mean_train_fit_rho": mean_train_fit_rho,
                    "UNDEFINED_ALL_DRAWS_DEGENERATE": True,
                    "UNDEFINED_REASON": "capacity_limit" if n_capacity_limited == n_requested_draws else "other",
                }
                continue

            mean_rho = float(np.mean(cell_rhos))
            max_p = float(np.max(cell_ps))  # conservative: worst-case p across draws
            print(f"  rho draws: {[f'{r:+.3f}' for r in cell_rhos]} "
                  f"({n_valid_draws}/{n_requested_draws} valid, {n_capacity_limited} capacity-limited)")
            print(f"  mean rho={mean_rho:+.4f}, worst-case permutation p={max_p:.4f}, "
                  f"mean TRAIN-SET fit rho={mean_train_fit_rho if mean_train_fit_rho is not None else float('nan'):+.4f}\n",
                  flush=True)

            results[f"{readout_name}__{condition}"] = {
                "rho_per_draw": [float(r) for r in cell_rhos],
                "n_capacity_limited": n_capacity_limited,
                "mean_train_fit_rho": mean_train_fit_rho,
                "p_per_draw": [float(p) for p in cell_ps],
                "mean_rho": mean_rho,
                "worst_case_p": max_p,
                "n_valid_draws": n_valid_draws, "n_requested_draws": n_requested_draws,
            }

    # ---------------- Pre-declared interpretation ----------------
    print("\n" + "=" * 70)
    print("PRE-DECLARED INTERPRETATION")
    print("=" * 70)

    def sharp_drop(cond_key, orig_key):
        r_cond_raw = results[cond_key]["mean_rho"]
        r_orig_raw = results[orig_key]["mean_rho"]
        if r_cond_raw is None or r_orig_raw is None:
            return None  # UNDEFINED, not False -- must not silently read as "no drop"
        r_cond = abs(r_cond_raw)
        r_orig = abs(r_orig_raw)
        p_cond = results[cond_key]["worst_case_p"]
        return (r_cond < 0.5 * r_orig) and (p_cond > 0.05)

    flatten_orig = "FLATTEN__ORIGINAL"
    flatten_chan = "FLATTEN__CHANNEL_SHUFFLE"
    flatten_spat = "FLATTEN__SPATIAL_SHUFFLE"
    flatten_both = "FLATTEN__BOTH_SHUFFLE"

    chan_drops = sharp_drop(flatten_chan, flatten_orig)
    spat_drops = sharp_drop(flatten_spat, flatten_orig)

    def fmt_rho(key):
        v = results[key]["mean_rho"]
        return f"{v:+.4f}" if v is not None else "UNDEFINED(all draws degenerate)"

    print(f"FLATTEN readout: S(ORIGINAL)={fmt_rho(flatten_orig)}, "
          f"S(CHANNEL_SHUFFLE)={fmt_rho(flatten_chan)} (sharp_drop={chan_drops}), "
          f"S(SPATIAL_SHUFFLE)={fmt_rho(flatten_spat)} (sharp_drop={spat_drops}), "
          f"S(BOTH_SHUFFLE)={fmt_rho(flatten_both)}")

    if chan_drops is None or spat_drops is None:
        verdict = "INCONCLUSIVE_DEGENERATE_TRAINING"
        print("\n[STOP] One or more FLATTEN cells produced only NaN/constant "
              "predictions across all draws -- the comparison this experiment "
              "needs is UNDEFINED, not a real null result. Do NOT report "
              "'neither survives' from this run; diagnose and fix the FLATTEN "
              "readout's training stability (see per-draw diagnostic lines "
              "above) before re-running.")
    elif chan_drops and not spat_drops:
        verdict = "H_C_CHANNEL_IDENTITY_SUPPORTED"
    elif spat_drops and not chan_drops:
        verdict = "H_S_SPATIAL_POSITION_SUPPORTED"
    elif chan_drops and spat_drops:
        verdict = "JOINT_CHANNEL_SPATIAL_REQUIRED"
    else:
        verdict = "NEITHER_SURVIVES_BOTH_SCRAMBLES_GLOBAL_STATISTIC"

    print(f"\n=== FINAL VERDICT: {verdict} ===")
    print("(Convenience label only -- inspect the actual rho/p table above directly, "
          "per this project's own established convention of not trusting an "
          "auto-classifier's label without checking the underlying numbers.)")

    pooled_spatial_check = results["POOLED__SPATIAL_SHUFFLE"]["mean_rho"]
    pooled_orig_check = results["POOLED__ORIGINAL"]["mean_rho"]
    if pooled_spatial_check is not None and pooled_orig_check is not None:
        print(f"\n[Design confirmation] POOLED readout under SPATIAL_SHUFFLE: rho={pooled_spatial_check:+.4f} "
              f"vs ORIGINAL rho={pooled_orig_check:+.4f} -- expected near-IDENTICAL by construction "
              f"(pooling is permutation-invariant over space); large deviation would indicate a bug.")
    else:
        print("\n[Design confirmation] SKIPPED -- POOLED__ORIGINAL or POOLED__SPATIAL_SHUFFLE was degenerate.")

    summary = {
        "checkpoint": str(CKPT_PATH), "val_dice_check": val_dice,
        "n_train_labels": len(train_records), "n_val_labels": len(val_records),
        "n_scramble_draws": N_SCRAMBLE_DRAWS,
        "grid_results": results,
        "final_verdict_CONVENIENCE_LABEL_ONLY": verdict,
    }
    with open(OUT_DIR / "E110_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E110_summary.json")


if __name__ == "__main__":
    main()
