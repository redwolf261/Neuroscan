"""
E184 -- Gamma continuation: does Gamma itself vary across the output-
equivalence class, not just output displacement from D(Z)?

Pre-registered addition in docs/phases/PHASE_E184_REPAIR_OPERATOR_DERIVATION.md
("Feasibility probe result... status corrected" section). Read that first.
Corrects the first-pass E184 result: diam of OUTPUT DISPLACEMENT from D(Z)
across directions (already shown != 0) is NOT the same claim as diam of
GAMMA across directions -- this script computes the latter, for the SAME
already-identified output-equivalent candidates, no new subjects/directions.

For each candidate Z' = Z + alpha*V_j that passed the E184 equivalence
filter (identified from E184_feasibility_probe_raw.json), apply the SAME
T6/T7 family construction already used throughout E181-E182 to Z' itself,
and measure Gamma(Z') = max pairwise Dice-displacement across that family --
the EXACT E180-E182 Gamma definition, now evaluated at Z' instead of the
original intact Z.

BINDING: no ground truth / Dice-vs-GT referenced anywhere (Gamma is
prediction-vs-prediction, per the E180-E182 confound guard, never GT). No
new subjects, no new directions, no new alpha grid, no optimizer, no new
metric definition.

NO training, NO architecture change, ONE checkpoint. Inference only.
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
e181_dir = project_root / "experiments" / "exp_e12_eggo_m" / "e181"
e184_dir = project_root / "experiments" / "exp_e12_eggo_m" / "e184"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(e180_dir))
sys.path.insert(0, str(e181_dir))
sys.path.insert(0, str(e184_dir))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders  # noqa: E402
from run_e180_transform_lab import PATCH, TARGET_STAGE  # noqa: E402
from run_e180_gamma import enc3_tile_bounds, dice_agree  # noqa: E402
from e181_transforms import pure_energy_scaling  # noqa: E402
from run_e181_b_calibration import solve_rank_matched_shape_budgeted  # noqa: E402
from e181_transforms import _svd_decompose, _reconstruct  # noqa: E402
from run_e184_feasibility_probe import (independent_random_directions,  # noqa: E402
                                        N_DIRECTIONS, ALPHA_GRID, V_SEED_BASE,
                                        measure_epsilon_from_t7_gentlest,
                                        sliding_window as e184_sliding_window,
                                        prediction_distance)

OUT_DIR = Path(__file__).parent
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694
OVERLAP = 0.25

# LOCKED, unchanged from E181/E182 -- the Gamma family the phenomenon survived on
T6_BUDGETS = [0.35, 0.50]
T7_LEVELS = [0.60, 0.80, 1.10, 1.45]


def t6_transform(x, budget, seed=0):
    orig_dtype = x.dtype
    x = x.float()
    U, S, Vh, mu, shape = _svd_decompose(x)
    sigma = S.detach().cpu().numpy().astype(np.float64)
    sigma = np.clip(sigma, 1e-12, None)
    sol = solve_rank_matched_shape_budgeted(sigma, budget, seed=seed)
    q = sol["q"]
    total_energy = float(np.sqrt((sigma ** 2).sum()))
    s = total_energy / np.sqrt((q ** 2).sum())
    sigma_new = s * q
    S_new = torch.from_numpy(sigma_new).to(S.device, S.dtype)
    out = _reconstruct(U, S_new, Vh, mu, shape)
    return out.to(orig_dtype)


class GammaAtCandidateHook:
    """Forward hook on enc3: constructs Z' = Z_intact + alpha*V_j within the
    tile mask (the SAME candidate construction as run_e184_feasibility_probe),
    then applies ONE T6/T7 family transform (fam, severity) to the WHOLE
    tile's Z' -- computing that family member's prediction, from which
    Gamma(Z') = max pairwise distance across the family is assembled
    downstream, exactly mirroring E180-E182's Gamma construction applied to
    Z' instead of the original intact Z."""

    def __init__(self):
        self.mask = None
        self.direction = None
        self.alpha = None
        self.family = None       # "T6" or "T7" or None (None = just Z', no family transform)
        self.severity = None

    def __call__(self, module, inputs, output):
        z_intact = output
        if self.mask is not None and self.direction is not None and self.alpha is not None:
            std = z_intact.std()
            perturb = (self.alpha * std) * self.direction.view(1, -1, 1, 1, 1)
            m = self.mask.to(z_intact.dtype).view(1, 1, *self.mask.shape)
            z_prime = z_intact + perturb * m
        else:
            z_prime = z_intact

        if self.family is None:
            return z_prime
        elif self.family == "T6":
            return t6_transform(z_prime, self.severity)
        elif self.family == "T7":
            return pure_energy_scaling(z_prime, self.severity)
        else:
            raise ValueError(self.family)


def register(model, hook):
    mod = getattr(model, TARGET_STAGE)
    return mod.register_forward_hook(lambda m, i, o: hook(m, i, o))


def compute_gamma_at_candidate(model, image_b, device, hook, mask, direction, alpha, amp):
    """Gamma(Z') = max pairwise Dice-displacement across the T6/T7 family,
    all evaluated starting from the SAME candidate Z' = Z_intact + alpha*V_j.
    Mirrors E180-E182's Gamma definition exactly, applied at Z' not Z."""
    configs = [("T6", b) for b in T6_BUDGETS] + [("T7", c) for c in T7_LEVELS]
    predictions = []
    for fam, sev in configs:
        hook.mask, hook.direction, hook.alpha = mask, direction, alpha
        hook.family, hook.severity = fam, sev
        p = e184_sliding_window(model, image_b, device, hook, amp=amp)
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

    # load the ALREADY-COLLECTED probe records -- no new subjects/directions
    probe_data = json.load(open(OUT_DIR / "E184_feasibility_probe_raw.json"))
    records = probe_data["records"]

    # identify which candidates PASSED the equivalence filter (recompute the
    # same filter, same epsilon per subject, already stored)
    passing = [r for r in records if r["output_distance_from_Z0"] <= r["epsilon_subject"]]
    print(f"[Data] {len(records)} total candidates, {len(passing)} passed the "
          f"equivalence filter (re-identified, not re-measured)")

    ledger = json.load(open(e180_dir / "E180_tile_ledger_full.json"))
    directions = independent_random_directions(N_DIRECTIONS, V_SEED_BASE, device)

    model = UNet3D_v5(in_channels=4, out_channels=3).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hook = GammaAtCandidateHook()
    register(model, hook)

    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    amp = not a.no_amp

    # index subjects by sid for lookup
    sid_to_idx = {}
    for idx in range(len(val_loader.dataset)):
        _, _, sid = val_loader.dataset[idx]
        if sid not in sid_to_idx:
            sid_to_idx[sid] = idx

    subjects_needed = sorted(set(r["sid"] for r in passing))
    if a.limit_subjects:
        subjects_needed = subjects_needed[:a.limit_subjects]
    print(f"[Data] computing Gamma(Z') for {len(subjects_needed)} subjects")

    results = []
    for sid in subjects_needed:
        idx = sid_to_idx[sid]
        image, _, _ = val_loader.dataset[idx]
        image_b = image.unsqueeze(0).to(device)
        tiles = ledger[sid]["tiles"]
        t = tiles[0]
        ez0, ey0, ex0, ez1, ey1, ex1 = enc3_tile_bounds(
            t["z0"], t["y0"], t["x0"], t["z1"], t["y1"], t["x1"])
        mask = torch.zeros((32, 32, 32), dtype=torch.bool, device=device)
        mask[ez0:ez1, ey0:ey1, ex0:ex1] = True

        # Gamma at Z (baseline, no candidate perturbation) -- for context
        gamma_Z = compute_gamma_at_candidate(model, image_b, device, hook, None, None, None, amp)

        sid_candidates = [r for r in passing if r["sid"] == sid]
        for r in sid_candidates:
            j = r["direction_idx"]
            alpha = r["alpha"]
            gamma_Zprime = compute_gamma_at_candidate(
                model, image_b, device, hook, mask, directions[j], alpha, amp)
            results.append({
                "sid": sid, "direction_idx": j, "alpha": alpha,
                "output_distance_from_Z0": r["output_distance_from_Z0"],
                "epsilon_subject": r["epsilon_subject"],
                "gamma_Z_baseline": gamma_Z,
                "gamma_Zprime": gamma_Zprime,
            })
        hook.mask, hook.direction, hook.alpha, hook.family, hook.severity = None, None, None, None, None
        print(f"[{sid}] gamma_Z_baseline={gamma_Z:.4f}  "
              f"n_candidates={len(sid_candidates)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "E184_gamma_continuation_raw.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nSaved E184_gamma_continuation_raw.json ({len(results)} records)")


if __name__ == "__main__":
    main()
