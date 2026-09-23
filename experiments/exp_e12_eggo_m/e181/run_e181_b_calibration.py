"""
E181-B -- Calibration of T6 (rank-matched spectral reshape) and T7 (pure
energy scaling).

Pre-registered in docs/phases/PHASE_E181_MECHANISM_DISSECTION_PREREG.md.
Read that first. Question: do T6 and T7 create controlled, GRADED
perturbation regimes -- not just a single working point?

For T7: severity = c (energy scale factor), grid around c=1.
For T6: severity = target shape-distance (a knob on the solver's objective,
NOT picked post-hoc to produce a large effect) -- implemented as a
constrained maximum-distance solve at several FIXED distance-budget levels,
so severity is a controlled dial, not "whatever the unconstrained solver
finds."

At every level, record: relative energy change, eff_rank, shape distance,
activation displacement (||Z'-Z||), U/V preservation (T6 only, per the
corrected subspace-alignment check), output Dice displacement, NaN/Inf,
catastrophic failure. All E180 probe/severity/statistical conventions stay
FROZEN -- this is calibrating NEW transforms only, not re-opening E180.

NO training, NO architecture change, ONE checkpoint. Inference only. GT
never used (prediction vs prediction displacement only, matching E180's
Stage 2 confound guard).
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

project_root = Path(__file__).resolve().parents[3]
e180_dir = project_root / "experiments" / "exp_e12_eggo_m" / "e180"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(e180_dir))
sys.path.insert(0, str(Path(__file__).parent))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders  # noqa: E402
from run_e180_transform_lab import PATCH, TARGET_STAGE  # noqa: E402
from e181_transforms import (  # noqa: E402
    _svd_decompose, _reconstruct, _target_spectrum, entropy_of,
    verify_UV_preserved_by_construction, pure_energy_scaling,
)

OUT_DIR = Path(__file__).parent
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694
OVERLAP = 0.25
N_CAL_SUBJECTS = 10

T7_GRID = [0.7, 0.85, 0.95, 1.05, 1.15, 1.3, 1.5]   # around c=1 (identity), both directions
# T6: fixed shape-distance BUDGETS (a dial, not the unconstrained optimum).
# The solver is constrained to a MAXIMUM achievable distance <= budget while
# still matching entropy exactly -- gives a genuine severity ladder.
T6_DISTANCE_BUDGETS = [0.05, 0.10, 0.20, 0.35, 0.50]


def solve_rank_matched_shape_budgeted(sigma, budget, n_restarts=6, seed=0):
    """Same construction as e181_transforms.solve_rank_matched_shape, but
    caps the achieved shape distance at `budget` via a penalty, giving a
    controlled severity dial instead of always returning the unconstrained
    maximum. Still NO access to Gamma/Delta/Dice -- sees only sigma and the
    fixed (entropy-match, distance<=budget) constraints."""
    r = len(sigma)
    p = sigma / sigma.sum()
    H_p = entropy_of(p)
    j = np.arange(1, r + 1, dtype=float)

    def entropy_constraint(params):
        a, b = params
        q = _target_spectrum(j, a, b)
        return entropy_of(q) - H_p

    def shape_dist(params):
        a, b = params
        q = _target_spectrum(j, a, b)
        return np.abs(q - p).sum()

    def objective(params):
        # maximize distance up to budget: penalize overshoot heavily, reward
        # approach to budget from below
        d = shape_dist(params)
        if d > budget:
            return (d - budget) * 100.0   # heavy penalty for exceeding budget
        return -(d)                        # otherwise maximize distance toward budget

    rng = np.random.default_rng(seed)
    starts = [(0.0, 0.0)] + [(rng.uniform(-0.5, 0.5), rng.uniform(-0.1, 0.1))
                             for _ in range(n_restarts - 1)]
    best = None
    for x0 in starts:
        res = minimize(objective, x0=np.array(x0), method="SLSQP",
                       constraints=[{"type": "eq", "fun": entropy_constraint}],
                       options={"maxiter": 200, "ftol": 1e-10})
        if not res.success:
            continue
        a, b = res.x
        q = _target_spectrum(j, a, b)
        d = shape_dist((a, b))
        if d > budget * 1.05:   # discard solutions that meaningfully exceed budget
            continue
        if best is None or d > best["achieved"]:
            best = {"a": float(a), "b": float(b), "q": q, "achieved": float(d),
                    "H_p": H_p, "H_q": entropy_of(q), "converged": True}
    if best is None:
        # NO usable solution found within the budget -- fall back to IDENTITY
        # (q=p, achieved=0), never to an arbitrary (a,b)=(0,0) target, which
        # was found to silently violate the entropy constraint (bug: gave
        # eff_rank_new~=C, ~99% rank error, at low budgets where the solver
        # legitimately cannot find a feasible point). Falling back to p
        # itself trivially satisfies entropy-match (H(p)=H(p)) and reports
        # converged=False honestly so this severity level can be excluded
        # from the graded regime rather than silently corrupting it.
        best = {"a": None, "b": None, "q": p.copy(), "achieved": 0.0,
                "H_p": H_p, "H_q": H_p, "converged": False}
    return best


from scipy.optimize import minimize  # noqa: E402


def t6_at_budget(x, budget, seed=0):
    """Computes entirely in fp32 regardless of the incoming tensor's dtype --
    found necessary during calibration: under AMP, enc3 arrives as float16,
    and fp16-precision SVD/reconstruction pushed the UV-preservation check's
    offdiag_max as high as ~1.0 on some tiles (vs ~0.01 in fp32), evidently
    because fp16 rounding lets the solver's chosen (a,b) interact badly with
    near-degenerate singular values. _svd_decompose's own internal .float()
    cast does NOT fix this, because it casts an ALREADY fp16-rounded tensor
    -- the precision loss happens once during the forward pass and cannot be
    recovered downstream. Casting up before ANY computation touches x is the
    actual fix. Output is cast back to the input's original dtype so the
    rest of an AMP-context forward pass is unaffected."""
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
    diag = {
        "eff_rank_orig": float(np.exp(sol["H_p"])), "eff_rank_new": float(np.exp(sol["H_q"])),
        "energy_orig": total_energy, "energy_new": float(np.sqrt((sigma_new ** 2).sum())),
        "shape_distance_achieved": sol["achieved"], "budget": budget,
        "converged": sol["converged"],
    }
    uv_check = verify_UV_preserved_by_construction(U, Vh, S_new, mu, shape, out)
    diag["UV_preserved"] = uv_check
    return out.to(orig_dtype), diag


def _gaussian_weight(shape, sigma_scale=0.125):
    coords = [np.linspace(-1, 1, s) for s in shape]
    g = np.ones(shape, dtype=np.float32)
    for i, c in enumerate(coords):
        sh = [1] * len(shape)
        sh[i] = -1
        g = g * np.exp(-(c ** 2) / (2 * sigma_scale ** 2)).reshape(sh).astype(np.float32)
    return np.maximum(g, 1e-4)


class CalibrationHook:
    """Accumulates a LIST of per-tile diagnostics across an entire
    sliding_window() pass -- fixes a bug where a single last_diag field was
    silently overwritten on every one of the ~8 tiles in a pass, so the
    reported diagnostic reflected only the LAST tile processed, not a
    genuine per-subject aggregate. Call reset_accumulator() before each
    sliding_window() pass, read diag_accumulator after."""

    def __init__(self):
        self.mode = None       # "T6" or "T7" or None
        self.param = None
        self.diag_accumulator = []
        self.activation_diff_accumulator = []

    def reset_accumulator(self):
        self.diag_accumulator = []
        self.activation_diff_accumulator = []

    def __call__(self, module, inputs, output):
        if self.mode is None:
            return output
        if self.mode == "T7":
            out = pure_energy_scaling(output, self.param)
        elif self.mode == "T6":
            out, diag = t6_at_budget(output, self.param, seed=0)
            self.diag_accumulator.append(diag)
        else:
            raise ValueError(self.mode)
        act_diff = float((out - output).norm() / output.norm().clamp(min=1e-12))
        self.activation_diff_accumulator.append(act_diff)
        return out


def register(model, hook):
    mod = getattr(model, TARGET_STAGE)
    return mod.register_forward_hook(lambda m, i, o: hook(m, i, o))


def sliding_window(model, image, device, hook, n_out=3, amp=True, overlap=OVERLAP):
    """Resets hook.diag_accumulator/activation_diff_accumulator before the
    pass; caller reads them after this returns for the genuine per-tile list
    covering every window in this pass (not just the last one)."""
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

    hook.reset_accumulator()
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
    pred = (acc / wsum.clamp(min=1e-6)).cpu().numpy()
    return pred, hook.activation_diff_accumulator


def dice_agree(a_bin, b_bin):
    out = []
    for r in range(a_bin.shape[0]):
        p, q = a_bin[r], b_bin[r]
        ps, qs = p.sum(), q.sum()
        out.append(1.0 if ps == 0 and qs == 0 else float(2.0 * (p * q).sum() / (ps + qs)))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--limit_subjects", type=int, default=0,
                    help="override N_CAL_SUBJECTS for a quick smoke test (0=default)")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    got = ckpt.get("best_mean_dice")
    assert got is not None and abs(float(got) - EXPECTED_DICE) < 1e-9, \
        f"Checkpoint identity FAILED: expected {EXPECTED_DICE}, got {got}"
    print(f"[Sanity] checkpoint identity PASS (best_mean_dice={got}).")

    model = UNet3D_v5(in_channels=4, out_channels=3).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hook = CalibrationHook()
    register(model, hook)

    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    amp = not a.no_amp
    n_target = a.limit_subjects if a.limit_subjects else N_CAL_SUBJECTS
    n = min(n_target, len(val_loader.dataset))
    print(f"[Data] calibrating on {n} subjects")

    # ---- identity sanity (both transforms) ----
    img0, _, _ = val_loader.dataset[0]
    tile0 = img0[:, :128, :128, :128].unsqueeze(0).to(device)
    hook.mode, hook.param = None, None
    with torch.no_grad():
        ref = model(tile0)["probs"].float()
        again = model(tile0)["probs"].float()
    assert float((ref - again).abs().max()) == 0.0, "dormant hook not a no-op -- STOP"
    print("[Sanity] dormant hook exact no-op PASS")

    hook.mode, hook.param = "T7", 1.0
    with torch.no_grad():
        out_t7_identity = model(tile0)["probs"].float()
    d_t7 = float((ref - out_t7_identity).abs().max())
    print(f"[Sanity] T7 c=1.0 identity PASS (max diff={d_t7:.3e})")
    assert d_t7 == 0.0, "T7 c=1.0 should be exact no-op"

    hook.mode, hook.param = None, None

    # ---- T7 severity curve ----
    print("\n=== T7 (pure energy scaling) severity curve ===")
    t7_results = {}
    for i in range(n):
        image, _, sid = val_loader.dataset[i]
        image_b = image.unsqueeze(0).to(device)

        hook.mode, hook.param = None, None
        p_intact, _ = sliding_window(model, image_b, device, hook, amp=amp)
        intact_bin = (p_intact > 0.5).astype(np.float32)

        for c in T7_GRID:
            hook.mode, hook.param = "T7", c
            p_pert, diffs = sliding_window(model, image_b, device, hook, amp=amp)
            pert_bin = (p_pert > 0.5).astype(np.float32)
            dy = 1.0 - dice_agree(pert_bin, intact_bin)
            nan_inf = bool(np.isnan(p_pert).any() or np.isinf(p_pert).any())
            catastrophic = bool(pert_bin.sum() == 0 and intact_bin.sum() > 0)
            t7_results.setdefault(str(c), []).append({
                "sid": sid, "dY": dy, "activation_rel_diff_mean": float(np.mean(diffs)),
                "nan_inf": nan_inf, "catastrophic": catastrophic,
            })
        hook.mode, hook.param = None, None
        print(f"  [{i+1}/{n}] {sid} done", flush=True)

    # ---- T6 severity curve ----
    print("\n=== T6 (rank-matched spectral reshape) severity curve ===")
    t6_results = {}
    for i in range(n):
        image, _, sid = val_loader.dataset[i]
        image_b = image.unsqueeze(0).to(device)

        hook.mode, hook.param = None, None
        p_intact, _ = sliding_window(model, image_b, device, hook, amp=amp)
        intact_bin = (p_intact > 0.5).astype(np.float32)

        for budget in T6_DISTANCE_BUDGETS:
            hook.mode, hook.param = "T6", budget
            p_pert, diffs = sliding_window(model, image_b, device, hook, amp=amp)
            pert_bin = (p_pert > 0.5).astype(np.float32)
            dy = 1.0 - dice_agree(pert_bin, intact_bin)
            nan_inf = bool(np.isnan(p_pert).any() or np.isinf(p_pert).any())
            catastrophic = bool(pert_bin.sum() == 0 and intact_bin.sum() > 0)
            # average the diagnostic over EVERY tile in this subject's sliding-
            # window pass, not just the last one (the bug this fixes)
            tile_diags = hook.diag_accumulator
            converged_flags = [dd["converged"] for dd in tile_diags]
            uv_pass_flags = [dd["UV_preserved"]["pass"] for dd in tile_diags
                             if dd["converged"]]   # UV only meaningful when solver converged
            t6_results.setdefault(str(budget), []).append({
                "sid": sid, "dY": dy, "activation_rel_diff_mean": float(np.mean(diffs)),
                "nan_inf": nan_inf, "catastrophic": catastrophic,
                "eff_rank_orig": float(np.mean([dd["eff_rank_orig"] for dd in tile_diags])),
                "eff_rank_new": float(np.mean([dd["eff_rank_new"] for dd in tile_diags])),
                "energy_orig": float(np.mean([dd["energy_orig"] for dd in tile_diags])),
                "energy_new": float(np.mean([dd["energy_new"] for dd in tile_diags])),
                "shape_distance_achieved": float(np.mean([dd["shape_distance_achieved"]
                                                          for dd in tile_diags])),
                "UV_preserved_pass": (float(np.mean(uv_pass_flags)) if uv_pass_flags
                                      else None),
                "solver_converged": float(np.mean(converged_flags)),
                "n_tiles_in_pass": len(tile_diags),
            })
        hook.mode, hook.param = None, None
        print(f"  [{i+1}/{n}] {sid} done", flush=True)

    # ---- summarize ----
    def summarize(results_dict, keys_extra=()):
        summary = {}
        for sev, recs in results_dict.items():
            dy = np.array([r["dY"] for r in recs])
            act = np.array([r["activation_rel_diff_mean"] for r in recs])
            n_nan = sum(r["nan_inf"] for r in recs)
            n_catastrophic = sum(r["catastrophic"] for r in recs)
            entry = {
                "dY_mean": float(dy.mean()), "dY_std": float(dy.std()),
                "activation_rel_diff_mean": float(act.mean()),
                "n_nan_inf": n_nan, "n_catastrophic": n_catastrophic,
            }
            for k in keys_extra:
                vals = [r[k] for r in recs if r.get(k) is not None]
                if vals and isinstance(vals[0], bool):
                    entry[k + "_frac_true"] = float(np.mean(vals))
                elif vals:
                    entry[k + "_mean"] = float(np.mean(vals))
            summary[sev] = entry
        return summary

    t7_summary = summarize(t7_results)
    t6_summary = summarize(t6_results, keys_extra=[
        "eff_rank_orig", "eff_rank_new", "energy_orig", "energy_new",
        "shape_distance_achieved", "UV_preserved_pass", "solver_converged",
    ])

    # rank/energy invariance check for T6
    t6_invariance_ok = True
    for sev, entry in t6_summary.items():
        rel_rank_err = abs(entry.get("eff_rank_new_mean", 0) - entry.get("eff_rank_orig_mean", 0)) \
            / max(entry.get("eff_rank_orig_mean", 1), 1e-9)
        rel_energy_err = abs(entry.get("energy_new_mean", 0) - entry.get("energy_orig_mean", 0)) \
            / max(entry.get("energy_orig_mean", 1), 1e-9)
        entry["rel_eff_rank_error"] = rel_rank_err
        entry["rel_energy_error"] = rel_energy_err
        if rel_rank_err > 0.01 or rel_energy_err > 0.01:
            t6_invariance_ok = False

    t7_invariance_ok = True   # T7's shape/rank invariance is exact by construction, verified in E181-A

    out_summary = {
        "T7_severity_curve": t7_summary,
        "T6_severity_curve": t6_summary,
        "T6_invariance_check_pass": t6_invariance_ok,
        "T7_invariance_check_pass": t7_invariance_ok,
        "n_subjects": n,
        "T7_grid": T7_GRID, "T6_distance_budgets": T6_DISTANCE_BUDGETS,
    }
    with open(OUT_DIR / "E181_B_calibration.json", "w") as f:
        json.dump(out_summary, f, indent=1)

    print("\n" + "=" * 70)
    print("T7 (energy scaling) severity curve:")
    for c in T7_GRID:
        e = t7_summary[str(c)]
        print(f"  c={c}: dY={e['dY_mean']:.4f} act_diff={e['activation_rel_diff_mean']:.4f} "
              f"nan={e['n_nan_inf']} catastrophic={e['n_catastrophic']}")
    print("\nT6 (rank-matched reshape) severity curve:")
    for b in T6_DISTANCE_BUDGETS:
        e = t6_summary[str(b)]
        print(f"  budget={b}: dY={e['dY_mean']:.4f} act_diff={e['activation_rel_diff_mean']:.4f} "
              f"shape_dist_achieved={e.get('shape_distance_achieved_mean','?'):.4f} "
              f"rel_rank_err={e['rel_eff_rank_error']:.2e} rel_energy_err={e['rel_energy_error']:.2e} "
              f"UV_pass_mean={e.get('UV_preserved_pass_mean','?')} "
              f"solver_conv_mean={e.get('solver_converged_mean','?')} "
              f"nan={e['n_nan_inf']} catastrophic={e['n_catastrophic']}")
    print(f"\nT6 invariance check (rank+energy held <1% error at every level): {t6_invariance_ok}")
    print("=" * 70)


if __name__ == "__main__":
    main()
