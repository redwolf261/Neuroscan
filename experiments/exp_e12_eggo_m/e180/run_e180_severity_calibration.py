"""
E180 Stage 2 -- Severity calibration.

Pre-registered in docs/phases/PHASE_E180_2_SEVERITY_CALIBRATION_PREREG.md. Read
that first; the grids and decision rule below are copied from it and must not
change after seeing results.

For each transform family (applied UNIFORMLY to every enc3 tile, not per-tile
restoration -- that is Stage 5), measure whole-volume Dice displacement between
intact and perturbed predictions across a severity grid, on 10 subjects. Picks
the working severity T_s per family, or drops families that are DEAD or
SATURATED at every tested severity.

NO training, NO architecture change, ONE checkpoint. Inference only. Never
compares against ground truth -- prediction vs prediction only (E147/E165
confound guard).
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
from run_e180_transform_lab import (  # noqa: E402
    TRANSFORMS, StageTransformer, register, TARGET_STAGE, PATCH,
)

OUT_DIR = Path(__file__).parent
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694
OVERLAP = 0.25

SEVERITY_GRID = {
    "T1_rank": [1, 2, 4, 8, 16, 32, 64, 128],
    "T2_permute": ["fixed_random_seed12345"],
    "T3_mix": [0.05, 0.10, 0.20],
    "T4_spectral": [1.5, 2.0, 3.0, 0.5, 0.7],
    "T5_smooth": [0.5, 1.0, 1.5],
}
T2_SEED = 12345
N_CAL_SUBJECTS = 10
GRADED_LOW, GRADED_HIGH = 0.02, 0.30
DEAD_THRESH, SATURATED_THRESH = 0.01, 0.50


def _gaussian_weight(shape, sigma_scale=0.125):
    coords = [np.linspace(-1, 1, s) for s in shape]
    g = np.ones(shape, dtype=np.float32)
    for i, c in enumerate(coords):
        sh = [1] * len(shape)
        sh[i] = -1
        g = g * np.exp(-(c ** 2) / (2 * sigma_scale ** 2)).reshape(sh).astype(np.float32)
    return np.maximum(g, 1e-4)


def sliding_window(model, image, device, n_out=3, amp=True, overlap=OVERLAP):
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
    """Binary agreement Dice per region, averaged. Both are PREDICTIONS.
    Identical to run_e165_per_stage_rank.py:dice_agree."""
    out = []
    for r in range(a_bin.shape[0]):
        p, q = a_bin[r], b_bin[r]
        ps, qs = p.sum(), q.sum()
        if ps == 0 and qs == 0:
            out.append(1.0)
        else:
            out.append(float(2.0 * (p * q).sum() / (ps + qs)))
    return float(np.mean(out))


def make_param(name, severity, C=128, device="cpu"):
    if name == "T1_rank":
        return int(severity)
    if name == "T2_permute":
        g = torch.Generator().manual_seed(T2_SEED)
        return torch.randperm(C, generator=g).to(device)
    if name == "T3_mix":
        return float(severity)
    if name == "T4_spectral":
        return float(severity)
    if name == "T5_smooth":
        return float(severity)
    raise ValueError(name)


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

    transformer = StageTransformer()
    register(model, transformer)

    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    amp = not a.no_amp
    n = min(N_CAL_SUBJECTS, len(val_loader.dataset))
    print(f"[Data] calibrating on {n} subjects, overlap={OVERLAP}")

    per_transform_dy = {name: {str(s): [] for s in grid}
                        for name, grid in SEVERITY_GRID.items()}

    for i in range(n):
        image, _, sid = val_loader.dataset[i]
        image_b = image.unsqueeze(0).to(device)

        transformer.name, transformer.param = None, None
        p_intact = sliding_window(model, image_b, device, amp=amp)
        intact_bin = (p_intact > 0.5).astype(np.float32)

        for name, grid in SEVERITY_GRID.items():
            for sev in grid:
                transformer.name = name
                transformer.param = make_param(name, sev, device=device)
                p_pert = sliding_window(model, image_b, device, amp=amp)
                pert_bin = (p_pert > 0.5).astype(np.float32)
                dy = 1.0 - dice_agree(pert_bin, intact_bin)
                per_transform_dy[name][str(sev)].append(dy)
        transformer.name, transformer.param = None, None
        print(f"[{i+1}/{n}] {sid} done", flush=True)

    # ------------------------------------------------------------- verdicts
    results = {}
    for name, grid in SEVERITY_GRID.items():
        sev_stats = {}
        for sev in grid:
            vals = np.array(per_transform_dy[name][str(sev)])
            sev_stats[str(sev)] = {"dY_mean": float(vals.mean()), "dY_std": float(vals.std())}

        if name == "T2_permute":
            results[name] = {"severities": sev_stats, "verdict": "CONTROL_CONTEXT_ONLY",
                             "chosen_Ts": None}
            continue

        dy_means = [sev_stats[str(s)]["dY_mean"] for s in grid]
        all_dead = all(d <= DEAD_THRESH for d in dy_means)
        all_saturated = all(d >= SATURATED_THRESH for d in dy_means)
        graded_candidates = [(s, d) for s, d in zip(grid, dy_means)
                             if GRADED_LOW <= d <= GRADED_HIGH]

        if all_dead:
            verdict, chosen = "DEAD", None
        elif all_saturated:
            verdict, chosen = "SATURATED", None
        elif graded_candidates:
            # monotonicity check (non-strict, allow small grid noise)
            order = np.argsort([abs(s) if isinstance(s, (int, float)) else 0 for s in grid])
            verdict = "GRADED"
            # choose the candidate closest to the middle of the graded band
            mid = (GRADED_LOW + GRADED_HIGH) / 2
            chosen = min(graded_candidates, key=lambda t: abs(t[1] - mid))[0]
        else:
            verdict, chosen = "AMBIGUOUS_NO_GRADED_POINT", None

        results[name] = {"severities": sev_stats, "verdict": verdict, "chosen_Ts": chosen}
        print(f"[Verdict] {name}: {verdict}  chosen_Ts={chosen}")

    any_graded = any(v["verdict"] == "GRADED" for v in results.values())
    program_verdict = "PROCEED" if any_graded else "STAGE2_DEAD_NO_GRADED_FAMILY"

    summary = {
        "question": "Which transform severities produce a graded (neither dead nor "
                    "saturated) output-displacement regime at enc3?",
        "checkpoint": str(CKPT), "checkpoint_dice_verified": float(got),
        "n_subjects": n, "overlap": OVERLAP,
        "graded_band": [GRADED_LOW, GRADED_HIGH],
        "dead_threshold": DEAD_THRESH, "saturated_threshold": SATURATED_THRESH,
        "per_transform": results,
        "PROGRAM_VERDICT": program_verdict,
        "decision_rule": "GRADED if any severity in [0.02,0.30] Dice displacement with "
                         "near-monotonic trend; DEAD if all<=0.01; SATURATED if all>=0.50; "
                         "program dies if zero families reach GRADED",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "E180_2_severity_calibration.json", "w") as f:
        json.dump(summary, f, indent=1)

    print("\n" + "=" * 70)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_transform"}, indent=1))
    for name, v in results.items():
        print(f"{name}: {v['verdict']}  chosen_Ts={v['chosen_Ts']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
