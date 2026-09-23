"""
E184 probe-response geometry audit: characterize the transformation of the
full 6-probe response CONFIGURATION under an output-preserving perturbation
P, not just its diameter (Gamma). NO new perturbations, NO training -- same
170 E184 candidates already verified in the commutator audit (same 9/10
subjects, BraTS-GLI-00506-000 excluded for the same reconstruction-closeness
reason established there).

Preregistered descriptors (fixed BEFORE looking at any correlation with
|Delta_Gamma|, per explicit user instruction):
  1. Centroid displacement: Delta_mu = d(mu_PZ, mu_Z),
     mu_Z = (1/6) sum_T D(T(Z))  (voxelwise mean of the 6 probability maps)
  2. Frobenius norm of pairwise-distance-matrix change:
     ||G_PZ - G_Z||_F,  G_ij = d(D(T_i Z), D(T_j Z))  (6x6 matrix)
  3. [SANITY CHECK ONLY, not a candidate descriptor] max pairwise-distance
     change = |max_ij(G_PZ) - max_ij(G_Z)| = |Gamma(PZ) - Gamma(Z)| =
     |Delta_Gamma| BY CONSTRUCTION (Gamma IS max pairwise distance across
     this same 6-member family) -- reported to validate the reconstruction
     pipeline (correlation with |Delta_Gamma| should be ~1.0), explicitly
     EXCLUDED from the real descriptor-vs-Delta_Gamma analysis.
  4. Mean per-probe response displacement: mean_T r_T,
     r_T = d(D(T(PZ)), D(T(Z)))
  5. Std of per-probe response displacement: std_T(r_T)

d = prediction_distance (mean abs probability difference), the SAME
continuous metric used throughout this project's E184/E186/E187/commutator
analyses -- NOT dice_agree/thresholded (Gamma's own literal definition uses
thresholded masks, but these are new diagnostic descriptors, not a
re-derivation of Gamma).

Controls (locked, same as the commutator audit): output displacement,
||PZ-Z|| (perturb_norm), subject (within-subject throughout), permutation
null.
"""
import sys
import json
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
from run_e184_feasibility_probe import (  # noqa: E402
    independent_random_directions, N_DIRECTIONS, V_SEED_BASE, prediction_distance,
)
from run_e184_gamma_continuation import (  # noqa: E402
    T6_BUDGETS, T7_LEVELS, t6_transform, pure_energy_scaling,
)

OUT_DIR = Path(__file__).parent
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694

TFAMILY = [("T6", b) for b in T6_BUDGETS] + [("T7", c) for c in T7_LEVELS]  # 6 members, LOCKED
EXCLUDED_SIDS = {"BraTS-GLI-00506-000"}  # same exclusion as the commutator audit, same reason


def apply_T(z, fam, sev):
    if fam == "T6":
        return t6_transform(z, sev)
    elif fam == "T7":
        return pure_energy_scaling(z, sev)
    raise ValueError(fam)


class SpliceHook:
    def __init__(self):
        self.mask = None
        self.z_override = None
        self.family = None
        self.severity = None

    def __call__(self, module, inputs, output):
        if self.mask is None or self.z_override is None:
            z = output
        else:
            m = self.mask.to(output.dtype).view(1, 1, *self.mask.shape)
            z = output * (1 - m) + self.z_override.to(output.dtype) * m
        if self.family is None:
            return z
        return apply_T(z, self.family, self.severity)


def register(model, hook):
    mod = getattr(model, TARGET_STAGE)
    return mod.register_forward_hook(lambda m, i, o: hook(m, i, o))


def single_tile_forward(model, tile_image, hook, amp=False):
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(tile_image)
    pr = (out["probs"] if isinstance(out, dict) else out).float()
    return pr.squeeze(0).cpu().numpy()  # (n_out, D, H, W) continuous probability map


def capture_Z(model, tile_image, hook):
    hook.mask, hook.z_override, hook.family, hook.severity = None, None, None, None
    captured = {}

    def cap(m, i, o):
        captured["Z"] = o.detach().clone()

    h = getattr(model, TARGET_STAGE).register_forward_hook(cap)
    with torch.no_grad():
        _ = model(tile_image)
    h.remove()
    return captured["Z"]


def probe_responses(model, tile_image, hook, mask, z_override, amp):
    """Returns the 6 continuous probability maps D(T_i(z_override)) for the
    fixed T-family, in a FIXED order matching TFAMILY -- needed to build the
    6x6 pairwise-distance matrix G and centroid mu."""
    maps = []
    for fam, sev in TFAMILY:
        hook.mask, hook.z_override, hook.family, hook.severity = mask, z_override, fam, sev
        p = single_tile_forward(model, tile_image, hook, amp=amp)
        maps.append(p)
    return maps  # list of 6 (n_out, D, H, W) arrays


def gamma_of(maps):
    """The REAL, literal Gamma definition throughout E180-E187: max pairwise
    DICE displacement (1-dice_agree) across THRESHOLDED (>0.5) masks -- NOT
    the continuous prediction_distance used for this audit's own geometry
    descriptors (centroid/G-matrix/r_T). These are deliberately DIFFERENT
    quantities: this function must be used for the reconstruction-closeness
    check against stored gamma_Zprime values; prediction_distance-based
    pairwise_matrix()/centroid() are the NEW diagnostic view, never
    conflated with this. (Bug caught during smoke-testing: an earlier
    version of this script used G.max() from the continuous-distance
    pairwise_matrix as if it were Gamma, giving gamma_Z ~0.0009 instead of
    the correct ~0.0137 -- traced to exactly this metric mismatch.)"""
    bins = [(m > 0.5).astype(np.float32) for m in maps]
    dists = []
    for i in range(len(bins)):
        for j in range(i + 1, len(bins)):
            dists.append(1.0 - dice_agree(bins[i], bins[j]))
    return float(np.max(dists))


def pairwise_matrix(maps):
    n = len(maps)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            g = prediction_distance(maps[i], maps[j])
            G[i, j] = g
            G[j, i] = g
    return G


def centroid(maps):
    return np.mean(np.stack(maps, axis=0), axis=0)  # voxelwise mean, (n_out, D, H, W)


def main():
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

    hook = SpliceHook()
    register(model, hook)

    ledger = json.load(open(e180_dir / "E180_tile_ledger_full.json"))
    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)
    sid_to_idx = {}
    for idx in range(len(val_loader.dataset)):
        _, _, sid = val_loader.dataset[idx]
        if sid not in sid_to_idx:
            sid_to_idx[sid] = idx

    directions = independent_random_directions(N_DIRECTIONS, V_SEED_BASE, device)

    e184_raw = json.load(open(e184_dir / "E184_gamma_continuation_raw.json"))
    e184_by_sid = {}
    for r in e184_raw:
        if r["sid"] in EXCLUDED_SIDS:
            continue
        e184_by_sid.setdefault(r["sid"], []).append(r)

    results = []
    for sid, recs in e184_by_sid.items():
        idx = sid_to_idx[sid]
        image, _, _ = val_loader.dataset[idx]
        image_b = image.unsqueeze(0).to(device)
        t = ledger[sid]["tiles"][0]
        pd, ph, pw = PATCH
        tile_image = image_b[:, :, t["z0"]:t["z0"] + pd, t["y0"]:t["y0"] + ph,
                             t["x0"]:t["x0"] + pw]
        if tile_image.shape[2:] != PATCH:
            tile_image = F.pad(tile_image, (0, pw - tile_image.shape[4],
                                            0, ph - tile_image.shape[3],
                                            0, pd - tile_image.shape[2]))
        ez0, ey0, ex0, ez1, ey1, ex1 = enc3_tile_bounds(
            t["z0"], t["y0"], t["x0"], t["z1"], t["y1"], t["x1"])
        mask = torch.zeros((32, 32, 32), dtype=torch.bool, device=device)
        mask[ez0:ez1, ey0:ey1, ex0:ex1] = True

        Z_full = capture_Z(model, tile_image, hook)
        maps_Z = probe_responses(model, tile_image, hook, mask, Z_full, amp=False)
        G_Z = pairwise_matrix(maps_Z)
        mu_Z = centroid(maps_Z)
        gamma_Z = gamma_of(maps_Z)  # REAL dice-based Gamma, not G_Z.max()

        for r in recs:
            j, alpha = r["direction_idx"], r["alpha"]
            Vj = directions[j]
            std_Z = Z_full.std()
            m3 = mask.view(1, 1, *mask.shape).to(Z_full.dtype)
            P_delta_full = (alpha * std_Z) * Vj.view(1, -1, 1, 1, 1) * m3
            PZ_full = Z_full + P_delta_full

            maps_PZ = probe_responses(model, tile_image, hook, mask, PZ_full, amp=False)
            G_PZ = pairwise_matrix(maps_PZ)
            mu_PZ = centroid(maps_PZ)
            gamma_PZ = gamma_of(maps_PZ)  # REAL dice-based Gamma, not G_PZ.max()

            # reconstruction closeness check (same convention/tolerance as
            # the commutator audit -- single-tile vs original multi-window+
            # AMP convention difference is expected, large mismatches are not)
            gamma_rel_err = abs(gamma_PZ - r["gamma_Zprime"]) / max(abs(r["gamma_Zprime"]), 1e-8)
            if gamma_rel_err >= 0.15:
                continue  # matches commutator audit's exclusion rule exactly

            delta_gamma = gamma_PZ - gamma_Z

            # Descriptor 1: centroid displacement
            delta_mu = prediction_distance(mu_PZ, mu_Z)

            # Descriptor 2: Frobenius norm of pairwise-matrix change
            frob_delta_G = float(np.linalg.norm(G_PZ - G_Z, ord="fro"))

            # Descriptor 3 [SANITY CHECK ONLY]: max pairwise-distance change
            max_pairwise_change = abs(gamma_PZ - gamma_Z)  # == |delta_gamma| by construction

            # Descriptors 4/5: per-probe response displacement r_T
            r_T = np.array([prediction_distance(maps_PZ[i], maps_Z[i]) for i in range(6)])
            mean_rT = float(r_T.mean())
            std_rT = float(r_T.std())

            results.append({
                "sid": sid, "direction_idx": j, "alpha": alpha,
                "gamma_Z": gamma_Z, "gamma_PZ": gamma_PZ, "delta_gamma": delta_gamma,
                "delta_mu": delta_mu, "frob_delta_G": frob_delta_G,
                "max_pairwise_change_SANITY_ONLY": max_pairwise_change,
                "mean_rT": mean_rT, "std_rT": std_rT,
                "output_distance_from_Z0": r["output_distance_from_Z0"],
                "perturb_norm": float((alpha * std_Z).item()),
                "gamma_rel_err_vs_stored": gamma_rel_err,
            })
        print(f"[Geometry] {sid}: {len(recs)} candidates processed", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "geometry_audit_raw.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nSaved geometry_audit_raw.json ({len(results)} records)")


if __name__ == "__main__":
    main()
