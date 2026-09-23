"""
E180 Stage 9 -- Boundary control: is Gamma merely detecting boundary proximity?

Pre-registered in docs/phases/PHASE_E180_6_9_ANALYSIS_PREREG.md. Read that
first. Reuses E167's boundary_distance/enrichment machinery VERBATIM -- no
redefinition for E180.

Tests Delta ~ Gamma + boundary_distance (subject fixed-effects, same
framework as Stage 6/8). If Gamma's contribution collapses once boundary
proximity is controlled, that is classification C (likely a boundary
phenomenon), not explained away.

Per-tile boundary distance: the mean GT-boundary distance (E167's
scipy.ndimage distance_transform_edt of the eroded GT boundary) of the
NATIVE-RESOLUTION voxels corresponding to tile i's spatial extent, averaged
over the ET/TC/WT regions with a defined boundary (matching E167's
region-averaging convention).

Requires loading ground truth (the only stage besides Delta itself that
touches GT, and only to construct the covariate, never to construct Gamma).
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from Dataset.brats_multimodal_dataset import create_multimodal_loaders, REGIONS  # noqa: E402
from run_e180_transform_lab import PATCH  # noqa: E402
from e180_stats import subject_fixed_effects_ols, within_subject_permutation_null  # noqa: E402
from e180_strata import e167_good_subject_ids  # noqa: E402

OUT_DIR = Path(__file__).parent
FAMILIES = ["T1_rank", "T4_spectral", "T5_smooth"]
N_PERM = 1000


def boundary_distance(gt_bin):
    """Identical to run_e167_boundary_residual.py:boundary_distance."""
    g = gt_bin.astype(bool)
    if not g.any():
        return None
    inner = g & ~binary_erosion(g, iterations=1, border_value=0)
    if not inner.any():
        inner = g
    return distance_transform_edt(~inner)


def per_tile_boundary_distance(target, tiles):
    """target: (3,D,H,W) binary GT per region. Returns {window_id: mean_dist}
    -- mean GT-boundary distance over the tile's native-resolution voxels,
    averaged over regions with a defined boundary (E167's convention)."""
    dists = {}
    for r in range(target.shape[0]):
        gt = target[r].astype(bool)
        d = boundary_distance(gt)
        dists[REGIONS[r]] = d

    out = {}
    for t in tiles:
        z0, y0, x0, z1, y1, x1 = t["z0"], t["y0"], t["x0"], t["z1"], t["y1"], t["x1"]
        vals = []
        for r_name, d in dists.items():
            if d is None:
                continue
            zc1 = min(z1, d.shape[0])
            yc1 = min(y1, d.shape[1])
            xc1 = min(x1, d.shape[2])
            sub = d[z0:zc1, y0:yc1, x0:xc1]
            if sub.size > 0:
                vals.append(float(sub.mean()))
        out[t["window_id"]] = float(np.mean(vals)) if vals else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_subjects", type=int, default=125)
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()

    ledger = json.load(open(OUT_DIR / "E180_tile_ledger_full.json"))
    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    total = len(val_loader.dataset)
    n = min(a.limit_subjects, total - a.offset)
    print(f"[Data] computing GT boundary distance for {n} subjects (offset={a.offset})")

    records = []
    for idx in range(n):
        i = a.offset + idx
        _, target, sid = val_loader.dataset[i]
        if sid not in ledger:
            continue
        target_np = target.numpy()
        tiles = ledger[sid]["tiles"]
        dists = per_tile_boundary_distance(target_np, tiles)
        for wid, dist in dists.items():
            records.append({"sid": sid, "window_id": wid, "gt_boundary_distance": dist})
        print(f"[{idx+1}/{n}] {sid}  n_tiles={len(tiles)}", flush=True)

    with open(OUT_DIR / "E180_9_boundary_distance.json", "w") as f:
        json.dump(records, f, indent=1)
    print(f"[Saved] E180_9_boundary_distance.json ({len(records)} records)\n")

    # ---------------------------------------------------------- analysis
    b_by_tile = {(r["sid"], r["window_id"]): r["gt_boundary_distance"] for r in records
                if not np.isnan(r["gt_boundary_distance"])}
    gamma = json.load(open(OUT_DIR / "E180_gamma_per_subject_full.json"))["per_family_metric"]
    delta = json.load(open(OUT_DIR / "E180_delta_per_subject_full.json"))
    g_by_key = {(r["sid"], r["window_id"], r["family"]): r for r in gamma}
    d_by_key = {(r["sid"], r["window_id"], r["family"]): r for r in delta}

    all_subjects = set(ledger.keys())
    good_110 = e167_good_subject_ids() & all_subjects

    results = {"A_full_125": {}, "B_E167_110": {}}
    for stratum_name, subj_filter in [("A_full_125", all_subjects), ("B_E167_110", good_110)]:
        for fam in FAMILIES:
            keys = sorted(k for k in g_by_key if k[2] == fam and k[0] in subj_filter)
            keys = [k for k in keys if k in d_by_key and (k[0], k[1]) in b_by_tile]
            if len(keys) < 20:
                results[stratum_name][fam] = {"note": "too few tiles"}
                continue
            y = np.array([d_by_key[k]["delta_i_global"] for k in keys])
            g = np.array([g_by_key[k]["gamma_dice"] for k in keys])
            b = np.array([b_by_tile[(k[0], k[1])] for k in keys])
            sids = [k[0] for k in keys]

            X = np.column_stack([b, g])   # boundary first, gamma second
            null = within_subject_permutation_null(y, X, sids, gamma_col_idx=1,
                                                    n_perm=N_PERM, seed=0)
            fe_gamma_alone = subject_fixed_effects_ols(y, g, sids)
            fe_boundary_alone = subject_fixed_effects_ols(y, b, sids)

            classification = "C_boundary_phenomenon" if null["permutation_p"] > 0.05 else "D_survives_boundary_control"
            results[stratum_name][fam] = {
                "gamma_alone_beta": fe_gamma_alone["beta"][0], "gamma_alone_p": fe_gamma_alone["p"][0],
                "boundary_alone_beta": fe_boundary_alone["beta"][0], "boundary_alone_p": fe_boundary_alone["p"][0],
                "gamma_over_boundary_control": null,
                "n_tiles": len(y), "n_subjects": len(set(sids)),
                "CLASSIFICATION": classification,
            }
            print(f"[{stratum_name}] {fam}: gamma_alone_beta={fe_gamma_alone['beta'][0]:+.4f} "
                  f"(p={fe_gamma_alone['p'][0]:.4f})  gamma|boundary perm_p={null['permutation_p']:.4f}  "
                  f"-> {classification}")

    with open(OUT_DIR / "E180_9_boundary_control.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\nSaved E180_9_boundary_control.json")


if __name__ == "__main__":
    main()
