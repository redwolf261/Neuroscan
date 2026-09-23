"""
E180 Stage 2b -- Probe calibration (correction, inserted after Stage 3's first
implementation revealed T1/T4/T5 are idempotent or compounding operators, so
reapplying the SAME transform as the instability probe was broken: T1 gave a
near-zero false null, T4 compounded gamma=3 twice into an uncalibrated gamma=9).

Pre-registered in docs/phases/PHASE_E180_3_4_GAMMA_PREREG.md (Stage 2b section).

The probe is now a FIXED, family-independent small perturbation:
    Z^deg_perturb_i = Z^deg + M_i * (eps * v)
where v is a fixed unit-norm random direction per channel (one seed, reused
everywhere), and eps is calibrated here per family (since each family's Z^deg
has a different activation scale) to sit in a graded whole-volume Dice-
displacement regime, exactly Stage 2's method applied to the probe itself.

NO training, NO architecture change, ONE checkpoint. Inference only. GT never
used (prediction vs prediction displacement only).
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders  # noqa: E402
from run_e180_transform_lab import lowrank_channels, spectral_reshape, local_smooth, PATCH, TARGET_STAGE  # noqa: E402

OUT_DIR = Path(__file__).parent
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694
OVERLAP = 0.25
N_CAL_SUBJECTS = 10
PROBE_SEED = 999
C_ENC3 = 128

FAMILIES = {
    "T1_rank": {"fn": lowrank_channels, "Ts": 2},
    "T4_spectral": {"fn": spectral_reshape, "Ts": 3.0},
    "T5_smooth": {"fn": local_smooth, "Ts": 1.5},
}
# eps is a MULTIPLIER on the per-tensor std of Z^deg (relative scale, so the
# probe is comparably sized across families with different activation ranges).
# Extended TWICE: first pass {0.05,...,1.0} left everything DEAD/borderline;
# second pass {0.05,...,8.0} brought T1_rank (dY=0.031) and T4_spectral
# (dY=0.068) into GRADED at eps=8.0, but T5_smooth was AMBIGUOUS_NO_GRADED_POINT
# (dY=0.018 at eps=8.0, just under the 0.02 floor, still climbing). Extending
# once more to 16.0 to see whether it crosses shortly after or genuinely
# plateaus (if it plateaus, T5_smooth is honestly dropped, not forced).
EPS_GRID = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0]
GRADED_LOW, GRADED_HIGH = 0.02, 0.30
DEAD_THRESH, SATURATED_THRESH = 0.01, 0.50


def fixed_direction(device):
    g = torch.Generator().manual_seed(PROBE_SEED)
    v = torch.randn(C_ENC3, generator=g)
    v = v / v.norm()
    return v.to(device)


def _gaussian_weight(shape, sigma_scale=0.125):
    coords = [np.linspace(-1, 1, s) for s in shape]
    g = np.ones(shape, dtype=np.float32)
    for i, c in enumerate(coords):
        sh = [1] * len(shape)
        sh[i] = -1
        g = g * np.exp(-(c ** 2) / (2 * sigma_scale ** 2)).reshape(sh).astype(np.float32)
    return np.maximum(g, 1e-4)


class ProbeHook:
    """Z^deg = T_s(Z) always. If probe_eps set, adds eps*std(Z^deg)*v within
    the WHOLE tile (uniform probe application for calibration, mirroring
    Stage 2's uniform-application convention)."""

    def __init__(self, direction):
        self.family = None
        self.probe_eps = None
        self.v = direction

    def __call__(self, module, inputs, output):
        if self.family is None:
            return output
        z_deg = FAMILIES[self.family]["fn"](output, FAMILIES[self.family]["Ts"])
        if self.probe_eps is None:
            return z_deg
        std = z_deg.std()
        perturb = (self.probe_eps * std) * self.v.view(1, -1, 1, 1, 1)
        return z_deg + perturb


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
    ap.add_argument("--only_family", type=str, default=None,
                    help="if set, only calibrate this family (reuse prior results for others "
                         "via --merge_with)")
    ap.add_argument("--merge_with", type=str, default=None,
                    help="path to a prior E180_2b_probe_calibration.json to merge non-"
                         "recomputed families from")
    a = ap.parse_args()

    families_to_run = {a.only_family: FAMILIES[a.only_family]} if a.only_family else FAMILIES

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

    v = fixed_direction(device)
    hook = ProbeHook(v)
    register(model, hook)

    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    amp = not a.no_amp
    n = min(N_CAL_SUBJECTS, len(val_loader.dataset))
    print(f"[Data] calibrating probe on {n} subjects, families={list(families_to_run)}")

    per_family_dy = {fam: {str(e): [] for e in EPS_GRID} for fam in families_to_run}

    for i in range(n):
        image, _, sid = val_loader.dataset[i]
        image_b = image.unsqueeze(0).to(device)

        for fam in families_to_run:
            hook.family, hook.probe_eps = fam, None
            p_deg = sliding_window(model, image_b, device, hook, amp=amp)
            deg_bin = (p_deg > 0.5).astype(np.float32)

            for eps in EPS_GRID:
                hook.family, hook.probe_eps = fam, eps
                p_probe = sliding_window(model, image_b, device, hook, amp=amp)
                probe_bin = (p_probe > 0.5).astype(np.float32)
                dy = 1.0 - dice_agree(probe_bin, deg_bin)
                per_family_dy[fam][str(eps)].append(dy)
        hook.family, hook.probe_eps = None, None
        print(f"[{i+1}/{n}] {sid} done", flush=True)

    results = {}
    if a.merge_with:
        prior = json.load(open(a.merge_with))
        for fam, v in prior["per_family"].items():
            if fam not in families_to_run:
                results[fam] = v
                print(f"[Merged] {fam}: {v['verdict']}  chosen_eps={v['chosen_eps']} "
                      f"(reused from {a.merge_with})")
    for fam in families_to_run:
        sev_stats = {}
        for eps in EPS_GRID:
            vals = np.array(per_family_dy[fam][str(eps)])
            sev_stats[str(eps)] = {"dY_mean": float(vals.mean()), "dY_std": float(vals.std())}
        dy_means = [sev_stats[str(e)]["dY_mean"] for e in EPS_GRID]
        all_dead = all(d <= DEAD_THRESH for d in dy_means)
        all_saturated = all(d >= SATURATED_THRESH for d in dy_means)
        graded_candidates = [(e, d) for e, d in zip(EPS_GRID, dy_means)
                             if GRADED_LOW <= d <= GRADED_HIGH]
        if all_dead:
            verdict, chosen = "DEAD", None
        elif all_saturated:
            verdict, chosen = "SATURATED", None
        elif graded_candidates:
            mid = (GRADED_LOW + GRADED_HIGH) / 2
            verdict = "GRADED"
            chosen = min(graded_candidates, key=lambda t: abs(t[1] - mid))[0]
        else:
            verdict, chosen = "AMBIGUOUS_NO_GRADED_POINT", None
        results[fam] = {"severities": sev_stats, "verdict": verdict, "chosen_eps": chosen}
        print(f"[Verdict] {fam}: {verdict}  chosen_eps={chosen}")

    any_graded = any(v["verdict"] == "GRADED" for v in results.values())
    summary = {
        "question": "Stage 2b: probe eps calibration -- fixed family-independent random "
                    "direction, scaled per family's own activation std",
        "checkpoint": str(CKPT), "checkpoint_dice_verified": float(got),
        "n_subjects": n, "probe_seed": PROBE_SEED,
        "eps_grid": EPS_GRID, "graded_band": [GRADED_LOW, GRADED_HIGH],
        "per_family": results,
        "PROGRAM_VERDICT": "PROCEED" if any_graded else "STAGE2B_DEAD",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "E180_2b_probe_calibration.json", "w") as f:
        json.dump(summary, f, indent=1)

    print("\n" + "=" * 70)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_family"}, indent=1))
    for fam, v in results.items():
        print(f"{fam}: {v['verdict']}  chosen_eps={v['chosen_eps']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
