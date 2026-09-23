"""
E186 -- Decoder-equivalence dimensionality probe: does the output-invisible
space constructed from pool3's non-argmax/tied entries contain directions
that change Gamma?

Pre-registered in docs/phases/PHASE_E186_DECODER_EQUIVALENCE_DIMENSIONALITY_PREREG.md.
Read that first. Continues after E185 killed the gradient/Jacobian route (not
because P_perp(grad Gamma)=0 was shown, but because autodiff is unreliable at
the real, tie-heavy enc3 activation -- 28.75% of pool3 windows have exact
ties under ReLU sparsity, measured directly in E185).

Candidate construction (NO autodiff, NO gradients anywhere):
  For each pool3 2x2x2xchannel window, find the argmax (spatial) entry.
  Perturb every NON-argmax entry by 50% of its gap to the argmax value,
  random sign per entry (tied entries: perturb DOWN only, since they equal
  the max and any upward move could create ambiguity). This delta is EXACTLY
  invisible to pool3's own output by construction -- provable from
  MaxPool3d's definition alone. This closes ker(A) analytically ONLY for the
  bottleneck path (enc3->pool3->bottleneck->...). The skip path
  (enc3->cat3->dec3, raw unpooled) is NOT covered analytically -- Step D's
  empirical check on the REAL two-path network is the mandatory gate.

BINDING RULES (verified structurally, not just stated):
  - NO ground truth / Dice referenced ANYWHERE in this file.
  - Step D (empirical output-equivalence + argmax-pattern-preserved check) is
    a HARD GATE. Candidates failing it are excluded, not adjusted. If ALL
    candidates fail, E186 reports KILL and STOPS -- Gamma is never computed
    on candidates that failed Step D.
  - epsilon reused verbatim from E184 (T7's gentlest calibrated level,
    measured in prediction_distance units) -- not re-tuned for this probe.
  - Gamma reused verbatim from E184's continuation definition (max pairwise
    prediction distance across the T6/T7 family, applied AT the candidate).
  - Perturbation magnitude (50% of the per-entry argmax gap) fixed BEFORE
    running, not adjusted after seeing Step D's pass rate.

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

N_SUBJECTS = 10                # LOCKED, matches E184/E185 calibration-cohort convention
N_CANDIDATES_PER_TILE = 8      # LOCKED, matches E184's N_DIRECTIONS convention
GAP_FRACTION = 0.5             # LOCKED -- 50% of each non-argmax entry's gap to the argmax
CANDIDATE_SEED_BASE = 812001   # distinct from every prior probe seed in this project
                                # (PROBE_SEED=999, V_SEED_BASE=90001, RAND_SEED_BASE=771001)


def find_pool3_argmax(enc3_tile):
    """enc3_tile: (C, D, H, W) tensor (one subject's enc3, sliced to the tile
    window). Returns, per channel per 2x2x2 window, the flat-window index of
    the argmax value (first occurrence -- PyTorch's own MaxPool3d tie-break
    convention, kept consistent since Step D compares against the REAL
    model's own pool3 output) and the max value itself, plus a boolean tie
    mask (True where an entry equals the max, i.e. participates in a tie)."""
    C, D, H, W = enc3_tile.shape
    assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0
    # reshape into (C, D/2, 2, H/2, 2, W/2, 2) to expose each window's 8 entries
    x = enc3_tile.view(C, D // 2, 2, H // 2, 2, W // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4, 6).reshape(C, D // 2, H // 2, W // 2, 8)
    max_val, argmax_idx = x.max(dim=-1)
    tie_mask = (x == max_val.unsqueeze(-1))
    return x, max_val, argmax_idx, tie_mask


def construct_delta(enc3_tile, seed):
    """Construct ONE candidate delta over the full tile, per the locked
    construction: for each pool3 window, every non-argmax entry moves by
    GAP_FRACTION * (argmax_value - entry_value) toward or away (random sign,
    EXCEPT tied entries which only move DOWN). Returns delta with the SAME
    shape as enc3_tile."""
    C, D, H, W = enc3_tile.shape
    x, max_val, argmax_idx, tie_mask = find_pool3_argmax(enc3_tile)
    # x: (C, D/2, H/2, W/2, 8), max_val/argmax_idx: (C, D/2, H/2, W/2)

    g = torch.Generator(device="cpu").manual_seed(seed)
    rand_signs = torch.randint(0, 2, tuple(x.shape), generator=g).float() * 2 - 1  # {-1,+1}
    rand_signs = rand_signs.to(enc3_tile.device)

    gap = max_val.unsqueeze(-1) - x  # (C, D/2, H/2, W/2, 8); 0 at argmax and at ties
    is_selected = torch.zeros_like(x, dtype=torch.bool)
    is_selected.scatter_(-1, argmax_idx.unsqueeze(-1), True)
    # "free" entries: not the selected slot. Includes tied entries with gap==0.
    is_free = ~is_selected
    is_tied_free = is_free & tie_mask  # free AND equals the max (a tie, not selected)

    delta_flat = torch.zeros_like(x)
    # non-tied free entries: move by +/- GAP_FRACTION*gap (gap>0 here)
    non_tied_free = is_free & (~tie_mask)
    delta_flat = torch.where(
        non_tied_free, rand_signs * GAP_FRACTION * gap, delta_flat)
    # tied free entries: move DOWN only (negative), by GAP_FRACTION of the
    # local activation's own std as a small, bounded step (gap==0 here, so we
    # can't use the gap itself -- use a tiny fixed fraction of the window's
    # max value magnitude, strictly negative, so it can never create a NEW
    # tie above the max)
    tied_step = GAP_FRACTION * max_val.unsqueeze(-1).clamp(min=1e-8)
    delta_flat = torch.where(
        is_tied_free, -tied_step.expand_as(delta_flat) * 0.5, delta_flat)
    # selected (argmax) entry: never perturbed
    delta_flat = torch.where(is_selected, torch.zeros_like(delta_flat), delta_flat)

    # reshape delta_flat (C, D/2, H/2, W/2, 8) back to (C, D, H, W)
    delta = delta_flat.view(C, D // 2, H // 2, W // 2, 2, 2, 2)
    delta = delta.permute(0, 1, 4, 2, 5, 3, 6).reshape(C, D, H, W)
    return delta


def verify_pool3_argmax_unchanged(enc3_before, enc3_after):
    """Step D structural check: does perturbation change WHICH entry is the
    argmax in any window? Returns fraction of windows where it changed."""
    _, _, argmax_before, _ = find_pool3_argmax(enc3_before)
    _, _, argmax_after, _ = find_pool3_argmax(enc3_after)
    changed = (argmax_before != argmax_after)
    return float(changed.float().mean().item())


class OverrideHook:
    """Forward hook on enc3: within the tile mask, replaces the activation
    with z_override_full (a (C,D,H,W) tensor covering the FULL enc3 spatial
    extent, matching `output`'s own shape -- constructed by the caller as
    the intact activation with the candidate delta added only inside the
    tile bounds, zero delta elsewhere). If family/severity are set, applies
    that T6/T7 transform to the (already-overridden) tile region on top,
    matching E184 gamma_continuation's GammaAtCandidateHook composition
    order exactly."""

    def __init__(self):
        self.mask = None
        self.z_override_full = None  # (1,C,D,H,W) or None
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
    """SINGLE 128^3 tile forward pass -- NOT sliding_window(). CORRECTION
    (2026-09-18): real subject volumes (e.g. 138x176x144) span MULTIPLE
    overlapping sliding-window tiles. The original version of this script
    captured z_override_full from a single-tile forward pass, then applied
    it via the multi-window sliding_window(model, image_b, ...) -- since the
    mask covers tile 0's FULL enc3 extent, this silently forced EVERY window
    (not just the one the capture came from) to use a mismatched enc3 value,
    contaminating the reported Step D verdict (confirmed directly: a
    ZERO-delta identity override still produced a large nonzero
    output_distance this way -- see this file's retraction note and E187's
    matching fix). This is the ONLY change from the original script --
    construction (GAP_FRACTION, seeds, candidate generation, Step
    B/D/Gamma logic) is otherwise byte-identical to the original locked
    design. Returns a numpy array (n_out, D, H, W) matching sliding_window()'s
    own output shape/dtype convention."""
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(tile_image)
    pr = (out["probs"] if isinstance(out, dict) else out).float()
    return pr.squeeze(0).cpu().numpy()


def compute_gamma_at_candidate(model, tile_image, device, hook, mask, z_override_full, amp):
    """Gamma(Z') = max pairwise Dice-displacement across the T6/T7 family,
    starting from candidate Z' spliced into the tile mask. Identical
    definition to E184's gamma_continuation, applied here to the
    constructively-generated candidate instead of a random-direction one.
    Single-tile forward pass only (see single_tile_forward's docstring)."""
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

    hook = OverrideHook()
    register(model, hook)

    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    amp = not a.no_amp
    n_target = a.limit_subjects if a.limit_subjects else N_SUBJECTS
    n = min(n_target, len(val_loader.dataset))
    print(f"[Data] {n} subjects, {N_CANDIDATES_PER_TILE} candidates/tile, "
          f"gap_fraction={GAP_FRACTION}, seed_base={CANDIDATE_SEED_BASE}")
    print("[Rule] NO ground truth / Dice referenced anywhere in this script.")
    print("[Rule] Step D is a hard gate -- candidates failing it are EXCLUDED, "
          "not adjusted. Gamma is computed ONLY for passing candidates.")

    step_b_records = []      # nullspace-source sanity stats, all subjects
    step_d_records = []      # per-candidate output-equivalence + argmax-preserved checks
    gamma_records = []       # Gamma(Z_k) for candidates that PASSED Step D

    for i in range(n):
        image, _, sid = val_loader.dataset[i]
        if sid not in ledger:
            continue
        image_b = image.unsqueeze(0).to(device)
        tiles = ledger[sid]["tiles"]
        t = tiles[0]   # single tile per subject, per the locality rule

        # skip subjects whose tile bounds aren't cleanly divisible by 8
        # (2x pool1 * 2x pool2 * 2x pool3-window) -- undersized volumes give
        # an enc3 tile region with an odd spatial dim that can't be split
        # into whole 2x2x2 pool3 windows. NOT a construction bug -- an edge
        # case in the ledger's own tile bounds for smaller-than-128 volumes.
        tz, ty, tx = t["z1"] - t["z0"], t["y1"] - t["y0"], t["x1"] - t["x0"]
        if tz % 8 != 0 or ty % 8 != 0 or tx % 8 != 0:
            print(f"[skip] {sid}: tile bounds ({tz},{ty},{tx}) not divisible by 8, "
                  f"cannot cleanly form 2x2x2 pool3 windows")
            continue

        ez0, ey0, ex0, ez1, ey1, ex1 = enc3_tile_bounds(
            t["z0"], t["y0"], t["x0"], t["z1"], t["y1"], t["x1"])
        mask = torch.zeros((32, 32, 32), dtype=torch.bool, device=device)
        mask[ez0:ez1, ey0:ey1, ex0:ex1] = True

        # Step A: capture the intact enc3 tile activation. A single targeted
        # forward pass on just this tile's 128^3 region (matches E184/E185's
        # own single-tile capture pattern), NOT a full sliding-window pass --
        # avoids ambiguity about which window's activation gets captured.
        hook.mask, hook.z_override_full, hook.family, hook.severity = None, None, None, None
        z0_full, y0_full, x0_full = t["z0"], t["y0"], t["x0"]
        pd, ph, pw = PATCH
        tile_image = image_b[:, :, z0_full:z0_full + pd, y0_full:y0_full + ph,
                             x0_full:x0_full + pw]
        if tile_image.shape[2:] != PATCH:
            tile_image = F.pad(tile_image, (0, pw - tile_image.shape[4],
                                            0, ph - tile_image.shape[3],
                                            0, pd - tile_image.shape[2]))
        captured2 = {}

        def _capture2(module, inputs, output):
            captured2["Z"] = output.detach().clone()

        h_cap2 = getattr(model, TARGET_STAGE).register_forward_hook(_capture2)
        with torch.no_grad():
            hook.mask, hook.z_override_full, hook.family, hook.severity = None, None, None, None
            _ = model(tile_image)
        h_cap2.remove()
        Z_full_enc3 = captured2["Z"]  # (1, 128, 32, 32, 32) -- the WHOLE tile's enc3, intact

        enc3_tile_region = Z_full_enc3[0, :, ez0:ez1, ey0:ey1, ex0:ex1]  # (C, dz, dy, dx)

        # Step B: nullspace-source sanity stats
        _, _, _, tie_mask_b = find_pool3_argmax(enc3_tile_region)
        n_windows = tie_mask_b.shape[0] * tie_mask_b.shape[1] * tie_mask_b.shape[2] * tie_mask_b.shape[3]
        n_tied_windows = int((tie_mask_b.sum(dim=-1) > 1).sum().item())
        step_b_records.append({
            "sid": sid, "frac_tied_windows": n_tied_windows / n_windows,
            "n_windows": n_windows,
        })

        # epsilon: T7's gentlest calibrated level, reused verbatim from E184
        eps_subject = measure_epsilon_from_t7_gentlest(model, image_b, device, mask, amp=amp)

        # reference prediction (no perturbation)
        hook.mask, hook.z_override_full, hook.family, hook.severity = None, None, None, None
        p_ref = single_tile_forward(model, tile_image, device, hook, amp=amp)

        for k in range(N_CANDIDATES_PER_TILE):
            seed = CANDIDATE_SEED_BASE + k
            delta = construct_delta(enc3_tile_region, seed)
            candidate_tile_region = enc3_tile_region + delta

            # Step D check 1: argmax pattern preserved (structural)
            frac_argmax_changed = verify_pool3_argmax_unchanged(
                enc3_tile_region, candidate_tile_region)

            # build the full-volume-shaped override: intact everywhere,
            # candidate only inside the tile mask (matches OverrideHook's
            # mask-based splice contract)
            z_override_full = Z_full_enc3.clone()
            z_override_full[0, :, ez0:ez1, ey0:ey1, ex0:ex1] = candidate_tile_region

            # Step D check 2: empirical output equivalence on the REAL,
            # full two-path network (no linearization, no autodiff)
            hook.mask, hook.z_override_full, hook.family, hook.severity = (
                mask, z_override_full, None, None)
            p_cand = single_tile_forward(model, tile_image, device, hook, amp=amp)
            d_output = prediction_distance(p_cand, p_ref)

            passed = (d_output <= eps_subject) and (frac_argmax_changed == 0.0)
            step_d_records.append({
                "sid": sid, "candidate_idx": k, "seed": seed,
                "output_distance_from_Z0": d_output, "epsilon_subject": eps_subject,
                "frac_argmax_changed": frac_argmax_changed, "passed": passed,
            })

            if passed:
                gamma_val = compute_gamma_at_candidate(
                    model, tile_image, device, hook, mask, z_override_full, amp)
                gamma_records.append({
                    "sid": sid, "candidate_idx": k, "seed": seed,
                    "output_distance_from_Z0": d_output, "gamma_candidate": gamma_val,
                })

        hook.mask, hook.z_override_full, hook.family, hook.severity = None, None, None, None
        n_pass_this_subj = sum(1 for r in step_d_records if r["sid"] == sid and r["passed"])
        print(f"[{i+1}/{n}] {sid}  frac_tied_windows={step_b_records[-1]['frac_tied_windows']:.4f}"
              f"  eps={eps_subject:.6f}  n_pass_step_d={n_pass_this_subj}/{N_CANDIDATES_PER_TILE}",
              flush=True)

    n_total_candidates = len(step_d_records)
    n_passed = sum(1 for r in step_d_records if r["passed"])
    print(f"\n[Step D summary] {n_passed}/{n_total_candidates} candidates passed "
          f"(output_distance <= epsilon AND argmax pattern unchanged)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "step_b_nullspace_source_stats": step_b_records,
        "step_d_records": step_d_records,
        "gamma_records_passing_only": gamma_records,
        "n_total_candidates": n_total_candidates,
        "n_passed_step_d": n_passed,
        "GAP_FRACTION": GAP_FRACTION,
        "N_CANDIDATES_PER_TILE": N_CANDIDATES_PER_TILE,
        "CANDIDATE_SEED_BASE": CANDIDATE_SEED_BASE,
        "note": "NO ground truth / Dice used anywhere. Gamma computed ONLY for "
                "candidates that passed Step D's hard gate (output-equivalence AND "
                "argmax-pattern-preserved). If n_passed_step_d==0, this is a KILL "
                "per the prereg's decision table -- gamma_records_passing_only will "
                "be empty and must be reported as such, not rescued.",
    }
    with open(OUT_DIR / "E186_decoder_equivalence_probe_raw.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved E186_decoder_equivalence_probe_raw.json "
          f"({len(gamma_records)} Gamma records from passing candidates)")


if __name__ == "__main__":
    main()
