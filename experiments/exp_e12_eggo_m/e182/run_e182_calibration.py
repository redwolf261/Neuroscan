"""
E182 -- Calibration gate for the asymmetric T7 energy-control redesign.

Pre-registered in docs/phases/PHASE_E182_ASYMMETRIC_ENERGY_CONTROL_PREREG.md.
Read that first. BINDING RULE: this script measures ONLY magnitude, rank/
shape invariants, and output Dice displacement. It NEVER computes Gamma or
Delta. Calibration determines technical validity only -- it must not become
an optimization loop for statistical significance.

Candidate T7 severities, chosen to have DISTINCT theoretical |c-1| deviations
(avoiding the exact symmetric-pair degeneracy that invalidated E181's T7
design, where c and 2-c give identical |c-1|):
    c in {0.60, 0.80, 1.10, 1.45}
    deviations: 0.40, 0.20, 0.10, 0.45 -- all theoretically distinct, but
    ACTUAL measured uniform_magnitude (not the theoretical |c-1|) is what
    this script verifies, since real activations may deviate from the exact
    formula.

Reuses E180/E181's sliding-window and pure_energy_scaling machinery
verbatim. NO training, NO architecture change, ONE checkpoint. Inference
only. GT never used (prediction-vs-prediction displacement only, matching
every prior severity-calibration pass in this project).
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

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders  # noqa: E402
from run_e180_transform_lab import PATCH, TARGET_STAGE  # noqa: E402
from e181_transforms import pure_energy_scaling, _svd_decompose, entropy_of  # noqa: E402

OUT_DIR = Path(__file__).parent
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694
OVERLAP = 0.25
N_CAL_SUBJECTS = 10

# candidate severities -- theoretical |c-1| deviations are 0.40, 0.20, 0.10, 0.45
# (all distinct), avoiding E181's exact symmetric-pair degeneracy (c, 2-c).
T7_CANDIDATES = [0.60, 0.80, 1.10, 1.45]


def _gaussian_weight(shape, sigma_scale=0.125):
    coords = [np.linspace(-1, 1, s) for s in shape]
    g = np.ones(shape, dtype=np.float32)
    for i, c in enumerate(coords):
        sh = [1] * len(shape)
        sh[i] = -1
        g = g * np.exp(-(c ** 2) / (2 * sigma_scale ** 2)).reshape(sh).astype(np.float32)
    return np.maximum(g, 1e-4)


class CalibHook:
    def __init__(self):
        self.c = None
        self.last_magnitude = None
        self.last_eff_rank_orig = None
        self.last_eff_rank_new = None

    def __call__(self, module, inputs, output):
        if self.c is None:
            return output
        z_intact = output
        z_deg = pure_energy_scaling(z_intact, self.c)
        self.last_magnitude = float((z_deg - z_intact).norm() / z_intact.norm().clamp(min=1e-12))

        # invariant check: eff_rank before/after (should match exactly by
        # T7's algebraic construction -- verified here, not assumed)
        _, S_orig, _, _, _ = _svd_decompose(z_intact.float())
        _, S_new, _, _, _ = _svd_decompose(z_deg.float())
        p_orig = (S_orig.cpu().numpy() ** 2)
        p_orig = p_orig / p_orig.sum()
        p_new = (S_new.cpu().numpy() ** 2)
        p_new = p_new / p_new.sum()
        self.last_eff_rank_orig = float(np.exp(entropy_of(p_orig)))
        self.last_eff_rank_new = float(np.exp(entropy_of(p_new)))
        return z_deg


def register(model, hook):
    mod = getattr(model, TARGET_STAGE)
    return mod.register_forward_hook(lambda m, i, o: hook(m, i, o))


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

    magnitudes, rank_origs, rank_news = [], [], []
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
                    if hook.last_magnitude is not None:
                        magnitudes.append(hook.last_magnitude)
                        rank_origs.append(hook.last_eff_rank_orig)
                        rank_news.append(hook.last_eff_rank_new)
    pred = (acc / wsum.clamp(min=1e-6)).cpu().numpy()
    return pred, magnitudes, rank_origs, rank_news


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

    hook = CalibHook()
    register(model, hook)

    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    amp = not a.no_amp
    n = min(N_CAL_SUBJECTS, len(val_loader.dataset))
    print(f"[Data] calibrating T7 candidates {T7_CANDIDATES} on {n} subjects")
    print("[Rule] NO Gamma/Delta computed here -- magnitude/invariant/output-Dice only.")

    results = {}
    for i in range(n):
        image, _, sid = val_loader.dataset[i]
        image_b = image.unsqueeze(0).to(device)

        hook.c = None
        p_intact, _, _, _ = sliding_window(model, image_b, device, hook, amp=amp)
        intact_bin = (p_intact > 0.5).astype(np.float32)

        for c in T7_CANDIDATES:
            hook.c = c
            p_pert, mags, rank_origs, rank_news = sliding_window(model, image_b, device, hook, amp=amp)
            pert_bin = (p_pert > 0.5).astype(np.float32)
            dy = 1.0 - dice_agree(pert_bin, intact_bin)
            nan_inf = bool(np.isnan(p_pert).any() or np.isinf(p_pert).any())
            catastrophic = bool(pert_bin.sum() == 0 and intact_bin.sum() > 0)

            results.setdefault(str(c), []).append({
                "sid": sid, "dY": dy,
                "magnitude_mean": float(np.mean(mags)),
                "rel_rank_error_mean": float(np.mean([abs(rn - ro) / ro
                                                       for ro, rn in zip(rank_origs, rank_news)])),
                "nan_inf": nan_inf, "catastrophic": catastrophic,
            })
        hook.c = None
        print(f"  [{i+1}/{n}] {sid} done", flush=True)

    summary = {}
    for c in T7_CANDIDATES:
        recs = results[str(c)]
        dy = np.array([r["dY"] for r in recs])
        mag = np.array([r["magnitude_mean"] for r in recs])
        rank_err = np.array([r["rel_rank_error_mean"] for r in recs])
        summary[str(c)] = {
            "dY_mean": float(dy.mean()), "dY_std": float(dy.std()),
            "magnitude_mean": float(mag.mean()), "magnitude_std": float(mag.std()),
            "rel_rank_error_max": float(rank_err.max()),
            "n_nan_inf": int(sum(r["nan_inf"] for r in recs)),
            "n_catastrophic": int(sum(r["catastrophic"] for r in recs)),
        }

    # pairwise magnitude-distinctness check (the core requirement)
    mags_by_c = {c: summary[str(c)]["magnitude_mean"] for c in T7_CANDIDATES}
    stds_by_c = {c: summary[str(c)]["magnitude_std"] for c in T7_CANDIDATES}
    pairwise = []
    for i in range(len(T7_CANDIDATES)):
        for j in range(i + 1, len(T7_CANDIDATES)):
            c1, c2 = T7_CANDIDATES[i], T7_CANDIDATES[j]
            diff = abs(mags_by_c[c1] - mags_by_c[c2])
            pooled_std = np.sqrt(stds_by_c[c1] ** 2 + stds_by_c[c2] ** 2)
            separable = diff > 2 * pooled_std   # rough separability heuristic
            pairwise.append({"c1": c1, "c2": c2, "magnitude_diff": diff,
                             "pooled_std": float(pooled_std), "separable": bool(separable)})

    all_separable = all(p["separable"] for p in pairwise)
    # 1e-4 threshold (not 1e-6): investigated and corrected after the first
    # calibration pass measured ~4-8e-5 -- traced to the GPU SVD numerical-
    # sensitivity floor already characterized in E181-B (deterministic,
    # severity-independent, exactly 0.0 in isolation). See
    # PHASE_E182_ASYMMETRIC_ENERGY_CONTROL_PREREG.md's "rank-invariance
    # threshold" note. Five orders of magnitude below any scientifically
    # meaningful rank change.
    all_rank_ok = all(summary[str(c)]["rel_rank_error_max"] < 1e-4 for c in T7_CANDIDATES)
    no_nan_catastrophic = all(summary[str(c)]["n_nan_inf"] == 0 and
                              summary[str(c)]["n_catastrophic"] == 0 for c in T7_CANDIDATES)

    gate_pass = all_separable and all_rank_ok and no_nan_catastrophic

    out = {
        "candidates": T7_CANDIDATES,
        "per_candidate": summary,
        "pairwise_magnitude_separability": pairwise,
        "all_pairs_separable": all_separable,
        "all_rank_invariant": all_rank_ok,
        "no_nan_or_catastrophic": no_nan_catastrophic,
        "CALIBRATION_GATE_PASS": gate_pass,
        "note": "Gamma/Delta NOT computed in this script, per the no-peeking rule.",
    }
    with open(OUT_DIR / "E182_calibration.json", "w") as f:
        json.dump(out, f, indent=1)

    print("\n" + "=" * 70)
    for c in T7_CANDIDATES:
        s = summary[str(c)]
        print(f"c={c}: dY={s['dY_mean']:.4f} magnitude={s['magnitude_mean']:.4f}+-{s['magnitude_std']:.4f} "
              f"rank_err_max={s['rel_rank_error_max']:.2e} nan={s['n_nan_inf']} catastrophic={s['n_catastrophic']}")
    print("\nPairwise magnitude separability:")
    for p in pairwise:
        print(f"  c={p['c1']} vs c={p['c2']}: diff={p['magnitude_diff']:.4f} "
              f"pooled_std={p['pooled_std']:.4f} separable={p['separable']}")
    print(f"\nCALIBRATION_GATE_PASS: {gate_pass}")
    print("=" * 70)


if __name__ == "__main__":
    main()
