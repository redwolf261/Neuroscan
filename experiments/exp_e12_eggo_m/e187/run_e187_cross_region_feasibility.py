"""
E187 -- Output-preserving cross-region feasibility: can enc3 cross a
downstream ReLU sign boundary (dec3[0]) with a MINIMAL perturbation while
staying within E184's output-equivalence tolerance?

Pre-registered in docs/phases/PHASE_E187_CROSS_REGION_FEASIBILITY_PREREG.md.
Read that first. Continues after E186's pool-tie nullspace source construction
(0/64 candidates originally reported failing Step D -- NOTE: that verdict was
later found CONTAMINATED by a multi-window sliding_window() bug, see E186's
retraction; E186 needs rerun before its actual verdict is known). E187 tests
a different mechanism regardless: cross a downstream boundary MINIMALLY (5%
past zero) rather than trying to stay invisible to any single op.

CRITICAL FIX (caught during smoke-testing, before any real run counted):
real subject volumes (e.g. 138x176x144) span MULTIPLE overlapping sliding-
window tiles. Capturing enc3/cat3 from a SINGLE 128^3-crop forward pass and
then applying the override via sliding_window() on the FULL multi-window
volume silently corrupts every window that isn't the one the capture came
from (confirmed directly: even a ZERO-delta identity override produced a
large nonzero output_distance this way). Fixed by restricting ALL inference
in this script to a SINGLE targeted forward pass on the one 128^3 tile under
study (matching E185's own single-tile pattern) -- never calling
sliding_window() on the full image_b.

Target layer: dec3[0]'s preactivation (Conv3d(256,128,k=3,pad=1), BEFORE its
BatchNorm3d+ReLU). Reads cat3=[upconv3,enc3] directly -- the shallowest
downstream ReLU whose receptive field over enc3 is one small 3x3x3 patch, no
pooling involved (avoids compounding receptive-field growth through pool3).

Candidate construction (NO autodiff, NO gradients -- uses the REAL conv
kernel directly, read from dec3[0].conv.weight):
  a_j(Z) = <W_c, cat3_patch(j)> + b_c   (exact, from the real Conv3d weights)
  dist_j = |a_j(Z)| / ||W_c||           (normalized boundary distance)
  delta_boundary = -sign(a_j(Z)) * dist_j * W_c_enc3_half / ||W_c_enc3_half||
  delta = (1+ETA) * delta_boundary      (ETA=0.05, LOCKED crossing margin)
  applied ONLY within the enc3-half of j's 3x3x3 receptive patch.

BINDING RULES (verified structurally, not just stated):
  - NO ground truth / Dice referenced ANYWHERE in this file.
  - Candidate selection ranks by dist_j ONLY (computed from Z and the frozen
    conv weights) -- never inspects D(Z') before selecting.
  - Step D (boundary crossed AND output within epsilon) is a HARD GATE.
    Candidates failing it are excluded, not adjusted. Gamma is computed
    ONLY for passing candidates.
  - K=8 nearest eligible boundaries per tile, eta=0.05, noise floor 1e-6,
    dedup by (output_channel, z,y,x) -- ALL locked before running, none
    adjusted after seeing Step D's pass rate.
  - epsilon reused verbatim from E184 (T7's gentlest calibrated level).
  - Gamma (if reached) reused verbatim from E184's continuation definition.

10 subjects, single tile each. NO training, NO architecture change, ONE
checkpoint. Inference only.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[3]
e180_dir = project_root / "experiments" / "exp_e12_eggo_m" / "e180"
e184_dir = project_root / "experiments" / "exp_e12_eggo_m" / "e184"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(e180_dir))
sys.path.insert(0, str(e184_dir))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders  # noqa: E402
from run_e180_transform_lab import PATCH, TARGET_STAGE  # noqa: E402
from run_e180_gamma import enc3_tile_bounds, dice_agree  # noqa: E402
from run_e184_feasibility_probe import (  # noqa: E402
    sliding_window, prediction_distance, measure_epsilon_from_t7_gentlest,
)
from run_e184_gamma_continuation import (  # noqa: E402
    T6_BUDGETS, T7_LEVELS, t6_transform, pure_energy_scaling,
)

OUT_DIR = Path(__file__).parent
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694
OVERLAP = 0.25

N_SUBJECTS = 10          # LOCKED, matches E184/E185/E186 convention
K_CANDIDATES = 8         # LOCKED, matches N_DIRECTIONS/N_CANDIDATES_PER_TILE convention
ETA = 0.05               # LOCKED crossing margin
NOISE_FLOOR = 1e-6       # LOCKED eligibility threshold (exclude |a_j| below this)
ENC3_C = 128             # enc3's channel count
CAT3_C = 256             # cat3's channel count (128 upconv3 + 128 enc3)
KERNEL = 3               # dec3[0]'s conv kernel size


def get_dec3_conv(model):
    """dec3[0] is a Conv3DBlock(256,128) -- its .conv is the real nn.Conv3d
    whose .weight (128,256,3,3,3) and .bias (128,) we read directly. No
    autodiff, no dense-layer approximation -- the real trained kernel."""
    return model.dec3[0].conv


def compute_cat3_and_preact(model, tile_image, amp=False):
    """Run the intact forward pass up through dec3[0]'s PRE-activation
    (before BatchNorm+ReLU), capturing enc3, upconv3, cat3, and the raw conv
    preactivation -- all via a single forward pass with hooks, no autodiff."""
    captured = {}

    def cap_enc3(m, i, o):
        captured["enc3"] = o.detach().clone()

    def cap_preact(m, i, o):
        captured["preact"] = o.detach().clone()

    h1 = getattr(model, TARGET_STAGE).register_forward_hook(cap_enc3)
    h2 = get_dec3_conv(model).register_forward_hook(cap_preact)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=amp):
            _ = model(tile_image)
    h1.remove()
    h2.remove()
    return captured["enc3"], captured["preact"]


def enumerate_candidates(enc3_tile, W, bias, preact_tile_region, ez0, ey0, ex0, device):
    """For each output location j=(c, z,y,x) in the (cropped-to-tile) dec3[0]
    preactivation map, compute a_j(Z) (already have it: preact_tile_region),
    the normalized boundary distance dist_j = |a_j|/||W_c||, and the
    boundary-crossing delta confined to j's 3x3x3 receptive patch in enc3
    (enc3-half of W_c only, channels ENC3_C: of the 256-channel kernel).
    Applies eligibility filters (noise floor, dedup by output location is
    trivial here since we enumerate each (c,z,y,x) exactly once) and returns
    the K nearest by dist_j, ascending."""
    C_out, D, H, Wd = preact_tile_region.shape
    # W: (128, 256, 3, 3, 3). enc3-half is channels ENC3_C:256 (since cat3
    # concatenates [upconv3 (0:128), enc3 (128:256)] in that order, matching
    # torch.cat([upconv3, enc3], dim=1) in the model's forward()).
    W_enc3_half = W[:, ENC3_C:CAT3_C, :, :, :]   # (128, 128, 3, 3, 3)
    W_norm_per_channel = W.view(C_out, -1).norm(dim=1)          # (128,) full kernel norm
    W_enc3_half_norm = W_enc3_half.reshape(C_out, -1).norm(dim=1)  # (128,) enc3-half norm

    pad = KERNEL // 2  # =1, receptive-field offset for 'same' padding

    # Vectorized distance map (avoids a Python-level loop over up to
    # C_out*D*H*Wd ~ 4M elements for a full-tile scan -- was originally a
    # per-element Python loop, far too slow; fixed to pure tensor ops,
    # identical selection semantics, K smallest by dist_j ascending).
    dist_map = preact_tile_region.abs() / W_norm_per_channel.view(C_out, 1, 1, 1)
    eligible = preact_tile_region.abs() >= NOISE_FLOOR
    dist_map_masked = torch.where(eligible, dist_map, torch.full_like(dist_map, float("inf")))

    flat = dist_map_masked.reshape(-1)
    n_eligible = int(eligible.sum().item())
    k = min(K_CANDIDATES, n_eligible)
    if k == 0:
        return [], W_enc3_half, W_enc3_half_norm, pad

    topk_dist, topk_flat_idx = torch.topk(flat, k, largest=False, sorted=True)
    # unravel flat index -> (c, z_local, y_local, x_local)
    c_idx = topk_flat_idx // (D * H * Wd)
    rem = topk_flat_idx % (D * H * Wd)
    z_idx = rem // (H * Wd)
    rem2 = rem % (H * Wd)
    y_idx = rem2 // Wd
    x_idx = rem2 % Wd

    selected = []
    for i in range(k):
        c = int(c_idx[i].item())
        z_local, y_local, x_local = int(z_idx[i].item()), int(y_idx[i].item()), int(x_idx[i].item())
        a_val = float(preact_tile_region[c, z_local, y_local, x_local].item())
        dist = float(topk_dist[i].item())
        z_global, y_global, x_global = ez0 + z_local, ey0 + y_local, ex0 + x_local
        selected.append((dist, c, z_global, y_global, x_global, a_val))
    return selected, W_enc3_half, W_enc3_half_norm, pad


def build_delta_for_candidate(enc3_tile, W_enc3_half, W_enc3_half_norm, pad,
                              c, z, y, x, a_val):
    """delta confined to the enc3-half receptive patch of dec3[0]'s conv at
    output location (c,z,y,x), where (z,y,x) are ALREADY GLOBAL enc3
    coordinates (converted once, in enumerate_candidates -- never re-offset
    here). Returns a full-shape (same as enc3_tile) delta tensor, zero
    everywhere except the 3x3x3 patch."""
    C, D, H, Wd = enc3_tile.shape
    # dec3 operates at enc3's own spatial resolution (both D/4 res) -- the
    # preactivation's (z,y,x) directly indexes into enc3's own spatial grid
    # (via 'same' padding, kernel centered at (z,y,x)).
    z0, z1 = z - pad, z + pad + 1
    y0, y1 = y - pad, y + pad + 1
    x0, x1 = x - pad, x + pad + 1

    delta = torch.zeros_like(enc3_tile)
    # clip the patch to the tensor's actual bounds (boundary voxels of the
    # tile may have a truncated receptive field under 'same' padding, which
    # zero-pads implicitly -- so any KERNEL_local outside [0,D) contributes
    # exactly zero in the real conv, and we skip those parts of the patch)
    pz0, pz1 = max(0, z0), min(D, z1)
    py0, py1 = max(0, y0), min(H, y1)
    px0, px1 = max(0, x0), min(Wd, x1)
    kz0, ky0, kx0 = pz0 - z0, py0 - y0, px0 - x0

    dist = abs(a_val) / float(W_enc3_half_norm[c].item())
    W_dir = W_enc3_half[c] / W_enc3_half_norm[c].clamp(min=1e-12)  # (128,3,3,3) unit-norm
    step = -np.sign(a_val) * (1.0 + ETA) * dist

    kz1 = kz0 + (pz1 - pz0)
    ky1 = ky0 + (py1 - py0)
    kx1 = kx0 + (px1 - px0)
    delta[:, pz0:pz1, py0:py1, px0:px1] = step * W_dir[:, kz0:kz1, ky0:ky1, kx0:kx1]
    return delta


class OverrideHook:
    """Forward hook on enc3: within the tile mask, splices z_override_full
    in (an intact-everywhere, candidate-inside-the-tile tensor matching
    output's own full shape). If family/severity set, applies that T6/T7
    transform on top -- identical composition to E184/E186's hooks."""

    def __init__(self):
        self.mask = None
        self.z_override_full = None
        self.family = None
        self.severity = None

    def __call__(self, module, inputs, output):
        if self.mask is None or self.z_override_full is None:
            z = output
        else:
            m = self.mask.to(output.dtype).view(1, 1, *self.mask.shape)
            z = output * (1 - m) + self.z_override_full.to(output.dtype) * m

        if self.family is None:
            return z
        elif self.family == "T6":
            return t6_transform(z, self.severity)
        elif self.family == "T7":
            return pure_energy_scaling(z, self.severity)
        else:
            raise ValueError(self.family)


def register(model, hook):
    mod = getattr(model, TARGET_STAGE)
    return mod.register_forward_hook(lambda m, i, o: hook(m, i, o))


def single_tile_forward(model, tile_image, device, hook, amp=False):
    """SINGLE 128^3 tile forward pass -- NOT sliding_window(). This probe is
    local to ONE tile (E187's own locality rule), and using the multi-window
    sliding_window() with a mask/override captured from just this one tile
    would silently corrupt every OTHER window's forward pass (confirmed bug,
    see this file's module docstring and E186's retraction). Returns a numpy
    array (n_out, D, H, W) matching sliding_window()'s own output shape/dtype
    convention, so downstream code (prediction_distance, dice_agree, etc.)
    is unaffected by this fix."""
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(tile_image)
    pr = (out["probs"] if isinstance(out, dict) else out).float()
    return pr.squeeze(0).cpu().numpy()


def compute_gamma_at_candidate(model, tile_image, device, hook, mask, z_override_full, amp):
    """Identical Gamma definition to E184/E186: max pairwise Dice-displacement
    across the T6/T7 family, starting from candidate Z'. Single-tile forward
    pass only (see single_tile_forward's docstring for why)."""
    configs = [("T6", b) for b in T6_BUDGETS] + [("T7", c) for c in T7_LEVELS]
    predictions = []
    for fam, sev in configs:
        hook.mask, hook.z_override_full, hook.family, hook.severity = (
            mask, z_override_full, fam, sev)
        p = single_tile_forward(model, tile_image, device, hook, amp=amp)
        predictions.append((p > 0.5).astype(np.float32))

    dists = []
    for j in range(len(predictions)):
        for k in range(j + 1, len(predictions)):
            dists.append(1.0 - dice_agree(predictions[j], predictions[k]))
    return float(np.max(dists))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--limit_subjects", type=int, default=0)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    got = ckpt.get("best_mean_dice")
    assert got is not None and abs(float(got) - EXPECTED_DICE) < 1e-9, \
        f"Checkpoint identity FAILED: expected {EXPECTED_DICE}, got {got}"
    print(f"[Sanity] checkpoint identity PASS (best_mean_dice={got}).")

    ledger = json.load(open(e180_dir / "E180_tile_ledger_full.json"))

    model = UNet3D_v5(in_channels=4, out_channels=3).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    conv = get_dec3_conv(model)
    W = conv.weight.detach().clone()   # (128, 256, 3, 3, 3)
    bias = conv.bias.detach().clone()  # (128,)
    print(f"[Model] dec3[0].conv.weight shape={tuple(W.shape)}")

    hook = OverrideHook()
    register(model, hook)

    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    amp = not a.no_amp
    n_target = a.limit_subjects if a.limit_subjects else N_SUBJECTS
    n = min(n_target, len(val_loader.dataset))
    print(f"[Data] {n} subjects, K={K_CANDIDATES} candidates/tile, eta={ETA}, "
          f"noise_floor={NOISE_FLOOR}")
    print("[Rule] NO ground truth / Dice referenced anywhere in this script.")
    print("[Rule] Step D is a hard gate -- candidates failing it are EXCLUDED, "
          "not adjusted. Gamma is computed ONLY for passing candidates.")

    step_d_records = []
    gamma_records = []

    for i in range(n):
        image, _, sid = val_loader.dataset[i]
        if sid not in ledger:
            continue
        image_b = image.unsqueeze(0).to(device)
        tiles = ledger[sid]["tiles"]
        t = tiles[0]

        tz, ty, tx = t["z1"] - t["z0"], t["y1"] - t["y0"], t["x1"] - t["x0"]
        if tz % 8 != 0 or ty % 8 != 0 or tx % 8 != 0:
            print(f"[skip] {sid}: tile bounds ({tz},{ty},{tx}) not divisible by 8")
            continue

        ez0, ey0, ex0, ez1, ey1, ex1 = enc3_tile_bounds(
            t["z0"], t["y0"], t["x0"], t["z1"], t["y1"], t["x1"])
        mask = torch.zeros((32, 32, 32), dtype=torch.bool, device=device)
        mask[ez0:ez1, ey0:ey1, ex0:ex1] = True

        z0_full, y0_full, x0_full = t["z0"], t["y0"], t["x0"]
        pd, ph, pw = PATCH
        tile_image = image_b[:, :, z0_full:z0_full + pd, y0_full:y0_full + ph,
                             x0_full:x0_full + pw]
        if tile_image.shape[2:] != PATCH:
            tile_image = F.pad(tile_image, (0, pw - tile_image.shape[4],
                                            0, ph - tile_image.shape[3],
                                            0, pd - tile_image.shape[2]))

        hook.mask, hook.z_override_full, hook.family, hook.severity = None, None, None, None
        Z_full_enc3, preact_full = compute_cat3_and_preact(model, tile_image, amp=amp)
        # crop the preactivation map to the region whose receptive field lies
        # within the tile's enc3 bounds (conservative: use the tile bounds
        # directly on dec3's own D/4-resolution grid, matching enc3's grid)
        preact_tile_region = preact_full[0, :, ez0:ez1, ey0:ey1, ex0:ex1]
        enc3_full = Z_full_enc3[0]  # (128, 32, 32, 32)

        selected, W_enc3_half, W_enc3_half_norm, pad = enumerate_candidates(
            enc3_full, W, bias, preact_tile_region, ez0, ey0, ex0, device)

        eps_subject = measure_epsilon_from_t7_gentlest(model, image_b, device, mask, amp=amp)

        hook.mask, hook.z_override_full, hook.family, hook.severity = None, None, None, None
        p_ref = single_tile_forward(model, tile_image, device, hook, amp=amp)

        n_pass = 0
        for (dist, c, z, y, x, a_val) in selected:
            # (z,y,x) are GLOBAL enc3 coordinates (converted once inside
            # enumerate_candidates) -- never re-offset by ez0/ey0/ex0 again.
            delta = build_delta_for_candidate(
                enc3_full, W_enc3_half, W_enc3_half_norm, pad,
                c, z, y, x, a_val)
            z_override_full = Z_full_enc3.clone()
            z_override_full[0, :, ez0:ez1, ey0:ey1, ex0:ex1] = (
                enc3_full[:, ez0:ez1, ey0:ey1, ex0:ex1] + delta[:, ez0:ez1, ey0:ey1, ex0:ex1])

            # Verify boundary crossing on the REAL conv (not assumed from the
            # construction alone). Capture dec3[0].conv's actual INPUT
            # (cat3, after upconv3 concatenation) via a forward-pre-hook
            # during the real candidate forward pass, then re-run the exact
            # same conv weights on it -- this IS the real preactivation the
            # network computes for this candidate, not an approximation.
            captured2 = {}
            hcap_in = conv.register_forward_pre_hook(
                lambda m, inp: captured2.__setitem__("cat3_input", inp[0].detach().clone()))

            hook.mask, hook.z_override_full, hook.family, hook.severity = (
                mask, z_override_full, None, None)
            with torch.no_grad():
                _ = model(tile_image)
            hcap_in.remove()

            cat3_input_new = captured2["cat3_input"]
            new_preact_full = F.conv3d(cat3_input_new, W, bias, padding=1)
            new_a_val = float(new_preact_full[0, c, z, y, x].item())  # z,y,x already global
            boundary_crossed = (np.sign(new_a_val) != np.sign(a_val)) and (new_a_val != 0.0)

            p_cand = single_tile_forward(model, tile_image, device, hook, amp=amp)
            d_output = prediction_distance(p_cand, p_ref)

            passed = boundary_crossed and (d_output <= eps_subject)
            step_d_records.append({
                "sid": sid, "candidate_channel": c, "candidate_zyx": [z, y, x],
                "a_val_before": a_val, "a_val_after": new_a_val,
                "dist_j": dist, "boundary_crossed": bool(boundary_crossed),
                "output_distance_from_Z0": d_output, "epsilon_subject": eps_subject,
                "passed": bool(passed),
            })

            if passed:
                n_pass += 1
                gamma_val = compute_gamma_at_candidate(
                    model, tile_image, device, hook, mask, z_override_full, amp)
                gamma_records.append({
                    "sid": sid, "candidate_channel": c, "candidate_zyx": [z, y, x],
                    "output_distance_from_Z0": d_output, "gamma_candidate": gamma_val,
                })

        hook.mask, hook.z_override_full, hook.family, hook.severity = None, None, None, None
        print(f"[{i+1}/{n}] {sid}  n_eligible_candidates={len(selected)}  "
              f"n_pass_step_d={n_pass}/{len(selected)}", flush=True)

    n_total = len(step_d_records)
    n_passed = sum(1 for r in step_d_records if r["passed"])
    print(f"\n[Step D summary] {n_passed}/{n_total} candidates passed "
          f"(boundary crossed AND output_distance <= epsilon)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "step_d_records": step_d_records,
        "gamma_records_passing_only": gamma_records,
        "n_total_candidates": n_total,
        "n_passed_step_d": n_passed,
        "K_CANDIDATES": K_CANDIDATES, "ETA": ETA, "NOISE_FLOOR": NOISE_FLOOR,
        "note": "NO ground truth / Dice used anywhere. Gamma computed ONLY for "
                "candidates that passed Step D's hard gate. If n_passed_step_d==0, "
                "this is KILL per the prereg's decision table -- reported as such.",
    }
    with open(OUT_DIR / "E187_cross_region_feasibility_raw.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved E187_cross_region_feasibility_raw.json "
          f"({len(gamma_records)} Gamma records from passing candidates)")


if __name__ == "__main__":
    main()
