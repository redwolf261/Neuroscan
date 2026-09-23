"""
E183-A -- Response geometry audit: collection of raw response vectors.

Pre-registered in docs/phases/PHASE_E183_A_RESPONSE_GEOMETRY_AUDIT_PREREG.md.
Read that first. Implementation protocol locked in that doc, treated as a
contract -- not modified here.

For each tile i, each transform family (T6 at budgets {0.35,0.50}, T7 at
c in {0.60,0.80,1.10,1.45} -- the exact severities already locked in
E181/E182, NOT re-tuned), saves the RAW response vector
    r_k = D(T_k(Z_i)) - D(Z_i)
as a 3-component (ET, TC, WT) region-averaged signed probability
displacement within tile i's native-space spatial footprint -- not just the
collapsed Gamma scalar. Gamma_i = max_{k,l} d(r_k, r_l) is ALSO computed and
saved from the same raw vectors, so it can be directly compared against the
four derived geometric descriptors computed downstream from the same data.

10 subjects, same calibration-cohort convention as every prior stage. Same
frozen checkpoint, same enc3 tile definition, same locked severities. Does
NOT optimize any perturbation parameter using these statistics.

NO training, NO architecture change, ONE checkpoint. Inference only. GT
never used (prediction-vs-prediction displacement only, matching every
severity/probe measurement in this project).
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
from Dataset.brats_multimodal_dataset import create_multimodal_loaders, REGIONS  # noqa: E402
from run_e180_transform_lab import PATCH, TARGET_STAGE  # noqa: E402
from run_e180_gamma import ENC3_DOWNSAMPLE, enc3_tile_bounds, fixed_direction  # noqa: E402
from e181_transforms import pure_energy_scaling  # noqa: E402
from run_e181_b_calibration import solve_rank_matched_shape_budgeted  # noqa: E402
from e181_transforms import _svd_decompose, _reconstruct  # noqa: E402

OUT_DIR = Path(__file__).parent
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694
OVERLAP = 0.25
N_SUBJECTS = 10   # LOCKED, same calibration-cohort convention as E181-B/E182 calibration

T6_BUDGETS = [0.35, 0.50]              # LOCKED, unchanged from E181/E182
T7_LEVELS = [0.60, 0.80, 1.10, 1.45]   # LOCKED, unchanged from E182

PROBE_EPS_T6 = 8.0   # LOCKED, unchanged from E180/E181/E182
PROBE_EPS_T7 = 8.0   # LOCKED, unchanged from E180/E181/E182

FAMILIES = {}  # populated below to match E181/E182's t6_transform/pure_energy_scaling


def t6_transform(x, budget, seed=0):
    """Identical construction to E181/E182's t6_transform -- fp32 regardless
    of input dtype (AMP precision fix), entropy-matched target spectrum via
    the SAME budgeted solver, same energy/UV guarantees."""
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


class ResponseHook:
    """Forward hook on enc3: Z^deg = T_s(Z_intact) uniformly (the baseline
    for this family/severity). If probe_mask is set, ADDS the fixed E180-
    style instability probe within tile i's enc3 mask on top of Z^deg --
    exactly E180/E181/E182's Gamma construction, reused verbatim."""

    def __init__(self, direction):
        self.transform_name = None
        self.severity = None
        self.probe_mask = None
        self.direction = direction

    def __call__(self, module, inputs, output):
        if self.transform_name is None:
            return output
        z_intact = output
        if self.transform_name == "T6":
            z_deg = t6_transform(z_intact, self.severity)
        elif self.transform_name == "T7":
            z_deg = pure_energy_scaling(z_intact, self.severity)
        else:
            raise ValueError(self.transform_name)

        if self.probe_mask is None:
            return z_deg

        eps = PROBE_EPS_T6 if self.transform_name == "T6" else PROBE_EPS_T7
        std = z_deg.std()
        perturb = (eps * std) * self.direction.view(1, -1, 1, 1, 1)
        m = self.probe_mask.to(z_deg.dtype).view(1, 1, *self.probe_mask.shape)
        return z_deg + perturb * m


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


def region_mean_displacement(p_probe, p_deg, z0, y0, x0, z1, y1, x1):
    """r_k: 3-component (ET,TC,WT) region-averaged SIGNED probability
    displacement, restricted to tile i's native-space spatial footprint.
    p_probe, p_deg: (3,D,H,W) full-volume probability maps."""
    sub_probe = p_probe[:, z0:z1, y0:y1, x0:x1]
    sub_deg = p_deg[:, z0:z1, y0:y1, x0:x1]
    diff = sub_probe - sub_deg   # (3, dz, dy, dx)
    return diff.reshape(3, -1).mean(axis=1)   # (3,) per region


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--limit_subjects", type=int, default=0)
    ap.add_argument("--limit_tiles", type=int, default=0)
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

    direction = fixed_direction(device)
    hook = ResponseHook(direction)
    register(model, hook)

    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    amp = not a.no_amp
    n_target = a.limit_subjects if a.limit_subjects else N_SUBJECTS
    n = min(n_target, len(val_loader.dataset))
    configs = [("T6", b) for b in T6_BUDGETS] + [("T7", c) for c in T7_LEVELS]
    print(f"[Data] collecting response vectors for {n} subjects, configs={configs}")

    records = []
    for i in range(n):
        image, _, sid = val_loader.dataset[i]
        if sid not in ledger:
            continue
        image_b = image.unsqueeze(0).to(device)
        tiles = ledger[sid]["tiles"]
        if a.limit_tiles:
            tiles = tiles[:a.limit_tiles]

        for t in tiles:
            ez0, ey0, ex0, ez1, ey1, ex1 = enc3_tile_bounds(
                t["z0"], t["y0"], t["x0"], t["z1"], t["y1"], t["x1"])
            mask = torch.zeros((32, 32, 32), dtype=torch.bool, device=device)
            mask[ez0:ez1, ey0:ey1, ex0:ex1] = True

            response_vectors = {}
            for transform_name, severity in configs:
                hook.transform_name, hook.severity, hook.probe_mask = transform_name, severity, None
                p_deg = sliding_window(model, image_b, device, hook, amp=amp)

                hook.transform_name, hook.severity, hook.probe_mask = transform_name, severity, mask
                p_probe = sliding_window(model, image_b, device, hook, amp=amp)

                r_k = region_mean_displacement(p_probe, p_deg,
                                               t["z0"], t["y0"], t["x0"],
                                               t["z1"], t["y1"], t["x1"])
                response_vectors[f"{transform_name}_{severity}"] = r_k.tolist()

            records.append({
                "sid": sid, "window_id": t["window_id"],
                "response_vectors": response_vectors,   # {config_key: [ET,TC,WT] displacement}
                "regions": list(REGIONS),
                "config_order": [f"{tn}_{sv}" for tn, sv in configs],
            })
        hook.transform_name, hook.severity, hook.probe_mask = None, None, None
        print(f"[{i+1}/{n}] {sid}  n_tiles={len(tiles)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "E183_A_response_vectors_raw.json", "w") as f:
        json.dump(records, f, indent=1)
    print(f"\nSaved E183_A_response_vectors_raw.json ({len(records)} tile records, "
          f"{len(configs)} response vectors each)")


if __name__ == "__main__":
    main()
