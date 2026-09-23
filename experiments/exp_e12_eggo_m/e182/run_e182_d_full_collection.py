"""
E182 -- full 125-subject cohort collection, asymmetric T7 energy control.

Pre-registered in docs/phases/PHASE_E182_ASYMMETRIC_ENERGY_CONTROL_PREREG.md.
Read that first. Calibration gate PASSED (E182_calibration.json, frozen to
FROZEN/): magnitude separability clean across all pairs, rank invariance
<1e-4 (corrected threshold, documented GPU SVD floor), no NaN/catastrophic.
T7 severity set LOCKED at {0.60, 0.80, 1.10, 1.45}.

LOCKED, inherited unchanged from E181-D: T6 (budgets {0.35, 0.50}), Gamma
and Delta constructions, tile selection, the fixed probe, the subject-aware
statistical framework. ONLY T7's severity set changed (from E181-D's
degenerate {0.70, 1.30} to this calibrated, magnitude-separable set).
c=1.0 remains excluded (Delta==0 by construction).

MAGNITUDE, NOT c, IS THE TREATMENT VARIABLE: every record retains BOTH the
requested c AND the measured uniform_magnitude. Test C must use the latter,
never treat c as a magnitude proxy -- that was exactly the defect E182 was
built to fix.

Reuses E180/E181's exact Gamma/Delta construction verbatim. Every T6 record
retains the FULL diagnostic (convergence, achieved shape distance, rank/
energy relative error, UV-preservation pass).

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
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(e180_dir))
sys.path.insert(0, str(e181_dir))
sys.path.insert(0, str(Path(__file__).parent))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders  # noqa: E402
from run_e180_transform_lab import PATCH, TARGET_STAGE  # noqa: E402
from run_e180_gamma import ENC3_DOWNSAMPLE, enc3_tile_bounds, fixed_direction, dice_agree  # noqa: E402
from e181_transforms import (  # noqa: E402
    _svd_decompose, _reconstruct, verify_UV_preserved_by_construction,
    pure_energy_scaling, entropy_of,
)
from run_e181_b_calibration import solve_rank_matched_shape_budgeted  # noqa: E402

OUT_DIR = Path(__file__).parent
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694
OVERLAP = 0.25

T6_BUDGETS = [0.35, 0.50]         # LOCKED, unchanged from E181-D
T7_LEVELS = [0.60, 0.80, 1.10, 1.45]   # LOCKED, per E182 calibration gate PASS

# a single fixed Stage-2b-style probe eps for the instability measurement,
# same construction as E180's fixed_direction()/PROBE_EPS -- calibrated
# separately per transform here since T6/T7 have different activation-scale
# behavior than T1/T4/T5. Fixed BEFORE this pilot runs, not tuned on results.
PROBE_EPS_T6 = 8.0    # matches E180's T4_spectral eps (same construction family: SVD-based)
PROBE_EPS_T7 = 8.0    # matches E180's T1_rank/T4_spectral eps


def t6_transform(x, budget, seed=0):
    """Same construction as run_e181_b_calibration.t6_at_budget, computed in
    fp32 regardless of input dtype (AMP precision fix from E181-B)."""
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
    uv_check = verify_UV_preserved_by_construction(U, Vh, S_new, mu, shape, out)
    diag = {
        "requested_budget": budget,
        "achieved_shape_distance": sol["achieved"],
        "converged": sol["converged"],
        "eff_rank_orig": float(np.exp(sol["H_p"])), "eff_rank_new": float(np.exp(sol["H_q"])),
        "rel_eff_rank_error": abs(np.exp(sol["H_q"]) - np.exp(sol["H_p"])) / np.exp(sol["H_p"]),
        "energy_orig": total_energy, "energy_new": float(np.sqrt((sigma_new ** 2).sum())),
        "rel_energy_error": abs(float(np.sqrt((sigma_new ** 2).sum())) - total_energy) / total_energy,
        "UV_preserved_pass": uv_check["pass"],
        "UV_reconstruction_max_diff": uv_check["reconstruction_max_diff"],
    }
    return out.to(orig_dtype), diag


def apply_uniform_t6(x, budget):
    out, _ = t6_transform(x, budget)
    return out


def apply_uniform_t7(x, c):
    return pure_energy_scaling(x, c)


class GammaDeltaHook:
    """Combined Gamma/Delta hook for T6 or T7. mode selects the transform;
    Z^deg = transform(Z_intact) always (when transform_name set). extra_mask
    (Gamma probe) or restore_mask (Delta counterfactual) are mutually
    exclusive with each other, matching E180's MaskedHook/RestoreHook split
    but combined here since T6/T7 share the same severity-parameterization
    plumbing."""

    def __init__(self, direction):
        self.transform_name = None   # "T6" or "T7" or None
        self.severity = None
        self.extra_mask = None       # Gamma: additional probe perturbation mask
        self.restore_mask = None     # Delta: restore-to-intact mask
        self.direction = direction
        self.last_t6_diag = None
        self.last_uniform_magnitude = None   # ||Z^deg - Z_intact|| / ||Z_intact||,
                                              # the MAGNITUDE COVARIATE fix -- captured
                                              # from the UNIFORM (whole-tile) degradation,
                                              # same quantity for every window in a subject

    def __call__(self, module, inputs, output):
        if self.transform_name is None:
            return output
        z_intact = output
        if self.transform_name == "T6":
            z_deg, diag = t6_transform(z_intact, self.severity)
            self.last_t6_diag = diag
        elif self.transform_name == "T7":
            z_deg = apply_uniform_t7(z_intact, self.severity)
            self.last_t6_diag = None
        else:
            raise ValueError(self.transform_name)

        if self.restore_mask is None and self.extra_mask is None:
            # this IS the uniform Z^deg construction call -- record its magnitude
            self.last_uniform_magnitude = float(
                (z_deg - z_intact).norm() / z_intact.norm().clamp(min=1e-12))

        if self.restore_mask is not None:
            m = self.restore_mask.to(z_deg.dtype).view(1, 1, *self.restore_mask.shape)
            return z_deg * (1 - m) + z_intact * m
        if self.extra_mask is not None:
            eps = PROBE_EPS_T6 if self.transform_name == "T6" else PROBE_EPS_T7
            std = z_deg.std()
            perturb = (eps * std) * self.direction.view(1, -1, 1, 1, 1)
            m = self.extra_mask.to(z_deg.dtype).view(1, 1, *self.extra_mask.shape)
            return z_deg + perturb * m
        return z_deg


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


def dice_vs_gt(pred_bin, target_bin):
    out = []
    for r in range(pred_bin.shape[0]):
        p, t = pred_bin[r], target_bin[r]
        ps, ts = p.sum(), t.sum()
        out.append(1.0 if ps == 0 and ts == 0 else float(2.0 * (p * t).sum() / (ps + ts)))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--limit_subjects", type=int, default=0)
    ap.add_argument("--limit_tiles", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--ledger_suffix", type=str, default="_full",
                    help="tile ledger to use, default the E180 full-125 ledger")
    ap.add_argument("--out_suffix", type=str, default="",
                    help="suffix for output filenames")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    got = ckpt.get("best_mean_dice")
    assert got is not None and abs(float(got) - EXPECTED_DICE) < 1e-9, \
        f"Checkpoint identity FAILED: expected {EXPECTED_DICE}, got {got}"
    print(f"[Sanity] checkpoint identity PASS (best_mean_dice={got}).")

    ledger = json.load(open(e180_dir / f"E180_tile_ledger{a.ledger_suffix}.json"))
    print(f"[Data] using E180 tile ledger ({a.ledger_suffix}): {len(ledger)} subjects")

    model = UNet3D_v5(in_channels=4, out_channels=3).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    direction = fixed_direction(device)
    hook = GammaDeltaHook(direction)
    register(model, hook)

    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    amp = not a.no_amp
    configs = [("T6", b) for b in T6_BUDGETS] + [("T7", c) for c in T7_LEVELS]

    gamma_records = []
    delta_records = []

    n_done = 0
    for i in range(a.offset, len(val_loader.dataset)):
        image, target, sid = val_loader.dataset[i]
        if sid not in ledger:
            continue
        if a.limit_subjects and n_done >= a.limit_subjects:
            break
        image_b = image.unsqueeze(0).to(device)
        tgt = target.numpy()
        tiles = ledger[sid]["tiles"]
        if a.limit_tiles:
            tiles = tiles[:a.limit_tiles]

        for transform_name, severity in configs:
            # ---- Z^deg baseline (uniform transform, no mask) ----
            hook.transform_name, hook.severity = transform_name, severity
            hook.extra_mask, hook.restore_mask = None, None
            p_deg = sliding_window(model, image_b, device, hook, amp=amp)
            deg_bin = (p_deg > 0.5).astype(np.float32)
            dice_deg = dice_vs_gt(deg_bin, tgt)
            deg_diag = hook.last_t6_diag
            # magnitude covariate: ||Z^deg - Z_intact|| / ||Z_intact||, the same
            # value for every tile of this subject/config (uniform degradation),
            # explicitly stored so T6-vs-T7 is never compared by raw severity label
            uniform_magnitude = hook.last_uniform_magnitude

            for t in tiles:
                ez0, ey0, ex0, ez1, ey1, ex1 = enc3_tile_bounds(
                    t["z0"], t["y0"], t["x0"], t["z1"], t["y1"], t["x1"])
                mask = torch.zeros((32, 32, 32), dtype=torch.bool, device=device)
                mask[ez0:ez1, ey0:ey1, ex0:ex1] = True

                # ---- Gamma: extra probe perturbation on tile i ----
                hook.transform_name, hook.severity = transform_name, severity
                hook.extra_mask, hook.restore_mask = mask, None
                p_probe = sliding_window(model, image_b, device, hook, amp=amp)
                probe_bin = (p_probe > 0.5).astype(np.float32)
                gamma_dice = 1.0 - dice_agree(probe_bin.astype(np.float32), deg_bin)

                # ---- Delta: restore tile i to intact ----
                hook.transform_name, hook.severity = transform_name, severity
                hook.extra_mask, hook.restore_mask = None, mask
                p_cf = sliding_window(model, image_b, device, hook, amp=amp)
                cf_bin = (p_cf > 0.5).astype(np.float32)
                dice_cf = dice_vs_gt(cf_bin, tgt)

                rec_common = {
                    "sid": sid, "window_id": t["window_id"],
                    "transform": transform_name, "severity": severity,
                    "uniform_magnitude": uniform_magnitude,
                }
                gamma_records.append({**rec_common, "gamma_dice": gamma_dice})
                delta_records.append({**rec_common, "dice_deg": dice_deg,
                                      "dice_cf_i": dice_cf,
                                      "delta_i_global": dice_cf - dice_deg})
                if transform_name == "T6" and deg_diag is not None:
                    # retain the FULL T6 diagnostic per record, not collapsed
                    gamma_records[-1]["t6_diag"] = deg_diag
                    delta_records[-1]["t6_diag"] = deg_diag

            hook.transform_name, hook.severity = None, None
            hook.extra_mask, hook.restore_mask = None, None
        n_done += 1
        print(f"[{sid}] done ({len(tiles)} tiles x {len(configs)} configs)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"E182_D_gamma{a.out_suffix}.json", "w") as f:
        json.dump(gamma_records, f, indent=1)
    with open(OUT_DIR / f"E182_D_delta{a.out_suffix}.json", "w") as f:
        json.dump(delta_records, f, indent=1)

    print(f"\nSaved E182_D_gamma{a.out_suffix}.json ({len(gamma_records)} records), "
          f"E182_D_delta{a.out_suffix}.json ({len(delta_records)} records)")


if __name__ == "__main__":
    main()
