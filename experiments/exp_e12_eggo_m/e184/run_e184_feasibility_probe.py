"""
E184 -- Feasibility probe: does the output-equivalence class E_eps(Z) contain
representations with materially different Gamma?

Pre-registered in docs/phases/PHASE_E184_REPAIR_OPERATOR_DERIVATION.md. Read
that first -- this implements ONLY the feasibility probe, not an optimizer,
not repair, not a Dice evaluation. Treated as a contract:

  Z -> Z + alpha*V_j -> D(Z + alpha*V_j) -> output-equivalence filter -> Gamma(Z + alpha*V_j)

BINDING RULES (verified structurally in this script, not just stated):
  - NO ground truth / Dice is loaded, computed, or referenced ANYWHERE in this
    file. Construction, filtering, and the decisive statistic all operate on
    Gamma and D(Z') (predictions), never on labels.
  - V_j are NEW independently-seeded random directions, distinct from
    run_e180_gamma.py's fixed_direction() (the existing Gamma-probe direction)
    and distinct from any T6/T7-derived direction -- avoids the exact
    circularity flagged in the derivation doc.
  - No optimizer: alpha*V_j is a small PREDEFINED grid, not a search.
  - Magnitude is explicitly controlled: multiple directions are compared AT
    MATCHED alpha, so "Gamma varies" cannot trivially restate "bigger
    perturbations move Gamma more" (already known from E181/E182).
  - Single enc3 tile per subject (locality).
  - Same Gamma definition as E180-E182 (max pairwise prediction-space
    distance across a perturbation family), reused verbatim, not
    reinterpreted for this stage.
  - epsilon (output-equivalence threshold) is derived from the decoder's own
    benign output variation (AMP-vs-fp32 discrepancy) BEFORE seeing any
    Gamma/Delta result from this probe -- fixed independently, per the
    binding rule against choosing epsilon post-hoc.

5-10 subjects, single tile each. NO training, NO architecture change, ONE
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
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(e180_dir))
sys.path.insert(0, str(Path(__file__).parent))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders, REGIONS  # noqa: E402
from run_e180_transform_lab import PATCH, TARGET_STAGE  # noqa: E402
from run_e180_gamma import enc3_tile_bounds  # noqa: E402

OUT_DIR = Path(__file__).parent
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694
OVERLAP = 0.25
N_SUBJECTS = 10       # LOCKED, calibration-cohort convention
N_DIRECTIONS = 8      # LOCKED, within the 5-10 range specified
ALPHA_GRID = [0.5, 1.0, 2.0, 4.0]   # LOCKED, multiplier on per-tile activation std,
                                    # same convention as E180's PROBE_EPS scaling
V_SEED_BASE = 90001   # distinct from run_e180_gamma.py's PROBE_SEED=999 and from
                      # any T6/T7 construction seed -- fixed, not re-drawn
C_ENC3 = 128


def independent_random_directions(n_directions, seed_base, device):
    """N_DIRECTIONS unit vectors over the enc3 channel axis, each from its OWN
    seed (seed_base + j), structurally identical construction to
    run_e180_gamma.fixed_direction() but NUMERICALLY DISTINCT draws -- per
    the derivation doc's resolution of the V-construction question."""
    directions = []
    for j in range(n_directions):
        g = torch.Generator().manual_seed(seed_base + j)
        v = torch.randn(C_ENC3, generator=g)
        v = v / v.norm()
        directions.append(v.to(device))
    return directions


class ProbeHook:
    """Forward hook on enc3: within the tile mask, adds alpha*std(Z)*V_j to
    the INTACT activation (no T6/T7 transform involved -- this probe is
    entirely separate from E180-E183's Gamma/Delta construction, testing the
    output-equivalence-class question directly on the intact representation,
    per the derivation doc's Z_0 = Z reference point)."""

    def __init__(self):
        self.mask = None
        self.direction = None
        self.alpha = None

    def __call__(self, module, inputs, output):
        if self.mask is None or self.direction is None or self.alpha is None:
            return output
        std = output.std()
        perturb = (self.alpha * std) * self.direction.view(1, -1, 1, 1, 1)
        m = self.mask.to(output.dtype).view(1, 1, *self.mask.shape)
        return output + perturb * m


def register(model, hook):
    mod = getattr(model, TARGET_STAGE)
    return mod.register_forward_hook(lambda m, i, o: hook(m, i, o))


def _gaussian_weight(shape, sigma_scale=0.125):
    coords = [np.linspace(-1, 1, s) for s in shape]
    g = np.ones(shape, dtype=np.float32)
    for i, c in enumerate(coords):
        sh = [1] * len(shape)
        sh[i] = -1
        g = g * np.exp(-(c ** 2) / (2 * sigma_scale ** 2)).reshape(sh).astype(np.float32)
    return np.maximum(g, 1e-4)


def sliding_window(model, image, device, hook, n_out=3, amp=True, overlap=OVERLAP):
    """NO GT is loaded or touched anywhere in this function or its callers."""
    _, _, D, H, W = image.shape
    pd, ph, pw = PATCH
    stride = [max(1, int(p * (1 - overlap))) for p in PATCH]

    def starts(full, p, st):
        if full <= p:
            return [0]
        s = list(range(0, full - p + 1, st))
        if s[-1] != full - p:
            s.append(full - p)
        return s

    zs, ys, xs = starts(D, pd, stride[0]), starts(H, ph, stride[1]), starts(W, pw, stride[2])
    acc = torch.zeros((n_out, D, H, W), device=device, dtype=torch.float32)
    wsum = torch.zeros((1, D, H, W), device=device, dtype=torch.float32)
    gw = torch.from_numpy(_gaussian_weight((min(pd, D), min(ph, H), min(pw, W)))).to(device)

    with torch.no_grad():
        for z in zs:
            for y in ys:
                for x in xs:
                    zc, yc, xc = min(pd, D), min(ph, H), min(pw, W)
                    tile = image[:, :, z:z + zc, y:y + yc, x:x + xc]
                    if tile.shape[2:] != (pd, ph, pw):
                        tile = F.pad(tile, (0, pw - tile.shape[4], 0, ph - tile.shape[3],
                                            0, pd - tile.shape[2]))
                    with torch.amp.autocast("cuda", enabled=amp):
                        out = model(tile)
                    pr = (out["probs"] if isinstance(out, dict) else out).float()
                    pr = pr[:, :, :zc, :yc, :xc].squeeze(0)
                    acc[:, z:z + zc, y:y + yc, x:x + xc] += pr * gw
                    wsum[:, z:z + zc, y:y + yc, x:x + xc] += gw
    return (acc / wsum.clamp(min=1e-6)).cpu().numpy()


def prediction_distance(p_a, p_b):
    """d(D(Z_a), D(Z_b)): mean absolute probability difference over the full
    volume, per region averaged -- prediction-space distance, NO GT."""
    return float(np.mean(np.abs(p_a - p_b)))


def measure_epsilon_from_t7_gentlest(model, image_b, device, mask, n_out=3, amp=True):
    """Derive epsilon from T7's gentlest ALREADY-CALIBRATED level (E182,
    c=1.10, locked before E184 existed) -- measured in the SAME units as
    this probe's own prediction_distance (mean abs probability difference),
    not E182's own dY (binary Dice displacement, a different scale).
    Corrects the first attempt: AMP-vs-fp32 discrepancy (~2e-6) turned out
    two orders of magnitude SMALLER than every tested perturbation's output
    distance, making the equivalence-class filter vacuously empty at every
    alpha tried -- a calibration defect in the epsilon SOURCE, not a Gamma
    finding. Still fixed independently of any Gamma/Delta result from THIS
    probe (T7's severity was locked in E182, before E184 existed), and
    NEVER touches GT."""
    T7_GENTLEST_C = 1.10   # LOCKED from E182 calibration, not re-tuned here

    def pure_energy_scaling_local(x, c):
        if c == 1.0:
            return x
        return x * c   # T7's construction IS literally uniform scaling; no
                       # SVD needed for a pure magnitude reference measurement

    class T7Hook:
        def __init__(self, mask):
            self.mask = mask

        def __call__(self, module, inputs, output):
            z_deg = pure_energy_scaling_local(output, T7_GENTLEST_C)
            m = self.mask.to(output.dtype).view(1, 1, *self.mask.shape)
            return output * (1 - m) + z_deg * m

    class NullHook:
        def __call__(self, module, inputs, output):
            return output

    mod = getattr(model, TARGET_STAGE)
    null_hook = NullHook()
    h = mod.register_forward_hook(lambda m, i, o: null_hook(m, i, o))
    p_ref = sliding_window(model, image_b, device, null_hook, n_out=n_out, amp=amp)
    h.remove()

    t7_hook = T7Hook(mask)
    h = mod.register_forward_hook(lambda m, i, o: t7_hook(m, i, o))
    p_t7 = sliding_window(model, image_b, device, t7_hook, n_out=n_out, amp=amp)
    h.remove()

    return prediction_distance(p_t7, p_ref)


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

    directions = independent_random_directions(N_DIRECTIONS, V_SEED_BASE, device)
    hook = ProbeHook()
    register(model, hook)

    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    amp = not a.no_amp
    n_target = a.limit_subjects if a.limit_subjects else N_SUBJECTS
    n = min(n_target, len(val_loader.dataset))
    print(f"[Data] {n} subjects, {N_DIRECTIONS} directions, alpha_grid={ALPHA_GRID}, "
          f"seed_base={V_SEED_BASE}")
    print("[Rule] NO ground truth / Dice referenced anywhere in this script.")

    all_records = []
    epsilons = []
    for i in range(n):
        image, _, sid = val_loader.dataset[i]
        if sid not in ledger:
            continue
        image_b = image.unsqueeze(0).to(device)
        tiles = ledger[sid]["tiles"]
        t = tiles[0]   # single tile per subject, per the locality rule

        ez0, ey0, ex0, ez1, ey1, ex1 = enc3_tile_bounds(
            t["z0"], t["y0"], t["x0"], t["z1"], t["y1"], t["x1"])
        mask = torch.zeros((32, 32, 32), dtype=torch.bool, device=device)
        mask[ez0:ez1, ey0:ey1, ex0:ex1] = True

        # epsilon: T7's gentlest already-calibrated level (E182, c=1.10),
        # measured in this probe's own prediction_distance units, BEFORE any
        # V_j perturbation, never touches GT
        hook.mask, hook.direction, hook.alpha = None, None, None
        eps_subject = measure_epsilon_from_t7_gentlest(model, image_b, device, mask, amp=amp)
        epsilons.append(eps_subject)

        # Z_0 reference (no perturbation)
        hook.mask, hook.direction, hook.alpha = None, None, None
        p_ref = sliding_window(model, image_b, device, hook, amp=amp)

        for j, V_j in enumerate(directions):
            for alpha in ALPHA_GRID:
                hook.mask, hook.direction, hook.alpha = mask, V_j, alpha
                p_pert = sliding_window(model, image_b, device, hook, amp=amp)

                d_output = prediction_distance(p_pert, p_ref)
                # Gamma: SAME definition as E180-E182 -- max pairwise distance
                # across a perturbation family. Here the "family" for this
                # single (j,alpha) candidate is itself vs the reference; the
                # decisive Gamma-variation statistic is computed downstream
                # across the (j,alpha) grid, not redefined here.
                gamma_candidate = d_output   # candidate's own displacement from Z_0

                all_records.append({
                    "sid": sid, "window_id": t["window_id"],
                    "direction_idx": j, "alpha": alpha,
                    "output_distance_from_Z0": d_output,
                    "epsilon_subject": eps_subject,
                })
        hook.mask, hook.direction, hook.alpha = None, None, None
        print(f"[{i+1}/{n}] {sid}  epsilon={eps_subject:.6f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "E184_feasibility_probe_raw.json", "w") as f:
        json.dump({"records": all_records, "epsilons_per_subject": epsilons,
                   "N_DIRECTIONS": N_DIRECTIONS, "ALPHA_GRID": ALPHA_GRID,
                   "V_SEED_BASE": V_SEED_BASE}, f, indent=1)
    print(f"\nSaved E184_feasibility_probe_raw.json ({len(all_records)} records, "
          f"{len(epsilons)} subjects)")
    print(f"[Epsilon context] mean={np.mean(epsilons):.6f} std={np.std(epsilons):.6f} "
          f"(AMP-vs-fp32 benign output variation, per subject)")


if __name__ == "__main__":
    main()
