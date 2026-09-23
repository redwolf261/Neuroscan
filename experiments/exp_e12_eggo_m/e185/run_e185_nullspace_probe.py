"""
E185 -- differential probe: does grad(Gamma_soft) have a component in the
decoder's output-null space?

Preregistered in docs/phases/PHASE_E185_NULLSPACE_GRADIENT_PROBE_PREREG.md.
Read that first. Single tile, single subject (extend only if ambiguous, not
a free license to keep sampling). NO training, NO optimizer, NO GT/Dice
anywhere. One backward pass for grad(Gamma_soft), one JVP for J @ v, a
finite-difference check on the JVP, and a matched-norm random-direction
control -- that is the entire computation.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[3]
e180_dir = project_root / "experiments" / "exp_e12_eggo_m" / "e180"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(e180_dir))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders  # noqa: E402
from run_e180_transform_lab import PATCH, TARGET_STAGE  # noqa: E402
from run_e180_gamma import enc3_tile_bounds  # noqa: E402

OUT_DIR = Path(__file__).parent
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694

# Smaller-than-usual input patch for THIS probe only (not E180-E184's PATCH).
# The differential JVP-vs-nullspace question needs one real tile through the
# real checkpoint, not full 128^3 resolution -- shrinking avoids the float64
# double-backward OOM on an 8GB GPU (float64 doubles memory vs the float32
# runs elsewhere in this project; double-backward keeps extra activation
# copies alive). U-Net is fully convolutional, divisible-by-8 patch sizes
# are valid; enc3 tile becomes (1,128,8,8,8) instead of (1,128,32,32,32).
PROBE_PATCH = (32, 32, 32)

T7_SEVERITIES = [0.60, 0.80, 1.10, 1.45]   # unchanged from E181/E182
TAU = 50.0
N_RANDOM_DIRS = 5
RAND_SEED_BASE = 771001   # distinct from every prior probe seed in this project
FD_H = 1e-3               # finite-difference step for JVP sanity check
FD_N_DIRS = 3             # how many directions to finite-difference-check


def pure_energy_scaling(x, c):
    if c == 1.0:
        return x
    return x * c


class SingleTileHook:
    """Forward hook on enc3: replaces the ENTIRE tile's activation with a
    supplied differentiable tensor z_override (requires_grad leaf), or T7-
    transforms it if severity is set. One hook, swappable mode, so the same
    single 128^3 tile is used for both the D(Z) forward (mode='identity')
    and each D(T7(Z,c)) forward (mode='t7')."""

    def __init__(self):
        self.z_override = None
        self.severity = None  # None = identity, else apply T7 at this severity

    def __call__(self, module, inputs, output):
        z = self.z_override if self.z_override is not None else output
        if self.severity is None:
            return z
        return pure_energy_scaling(z, self.severity)


def register(model, hook):
    mod = getattr(model, TARGET_STAGE)
    return mod.register_forward_hook(lambda m, i, o: hook(m, i, o))


def forward_single_tile(model, tile_image, hook, z_override=None, severity=None, amp=False):
    """Single 128^3 tile forward pass (no sliding window -- this probe is
    local to ONE tile, matching E184's single-tile-locality discipline).
    amp=False: autograd through AMP's autocast is workable but adds an
    unnecessary numerical-sensitivity axis (E181-B's documented GPU-SVD/AMP
    finding) for a probe whose entire point is a clean gradient; fp32 only."""
    hook.z_override, hook.severity = z_override, severity
    out = model(tile_image)
    return out["probs"] if isinstance(out, dict) else out


def compute_gamma_soft(model, tile_image, hook, z, taus=T7_SEVERITIES, tau=TAU):
    """Gamma_soft(z) = (1/tau) * logsumexp(tau * ||p_j - p_k||^2) over pairs
    j<k of T7(z, c) predictions, c in T7_SEVERITIES. z must have
    requires_grad=True for gradients to flow back through this function."""
    probs = []
    for c in taus:
        p = forward_single_tile(model, tile_image, hook, z_override=z, severity=c)
        probs.append(p)
    pairdists = []
    for j in range(len(probs)):
        for k in range(j + 1, len(probs)):
            pairdists.append(((probs[j] - probs[k]) ** 2).sum())
    pd = torch.stack(pairdists)
    gamma_soft = torch.logsumexp(tau * pd, dim=0) / tau
    return gamma_soft


def decoder_output(model, tile_image, hook, z):
    """D(z): the raw probability map with z spliced into enc3, no T7."""
    return forward_single_tile(model, tile_image, hook, z_override=z, severity=None)


def main():
    # GPU, float64. float64 at the full 128^3 tile OOM'd the 8GB card (double-
    # backward through the whole U-Net keeps multiple float64 activation
    # copies alive at once). Fix: shrink the INPUT PATCH for this probe only
    # (PROBE_PATCH below) -- the differential JVP-vs-nullspace question does
    # not require full 128^3 resolution, only a real forward/backward through
    # the real checkpoint at one tile. CPU was tried and killed by the user
    # for eating system memory; GPU + smaller patch is the correct fix.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # DIAGNOSTIC FINDING (recorded, not silently fixed): the original float32
    # run gave r=130.3 with a FAILING finite-difference sanity check (rel err
    # approx 1.0, cosine approx 0). Root-caused via nn.Conv3d isolation +
    # torch.autograd.gradgradcheck: PyTorch's own double-backward IS correct
    # (gradgradcheck passes at float64), but float32 finite differences at
    # this model's activation scale (enc3 tile norm ~1000+, 128-256 channels)
    # suffer catastrophic cancellation in (D1-D0)/h well before the true
    # Jacobian signal -- a precision artifact, not an autograd bug, not a
    # NeuroScan-code bug. Fix: run the ENTIRE probe in float64. Confirmed in
    # isolation this recovers cos~1.0 between JVP and finite differences.
    torch.set_default_dtype(torch.float64)

    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    got = ckpt.get("best_mean_dice")
    assert got is not None and abs(float(got) - EXPECTED_DICE) < 1e-9, \
        f"Checkpoint identity FAILED: expected {EXPECTED_DICE}, got {got}"
    print(f"[Sanity] checkpoint identity PASS (best_mean_dice={got}).")

    ledger = json.load(open(e180_dir / "E180_tile_ledger_full.json"))
    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    SID = "BraTS-GLI-00250-000"  # first subject in the E184 ledger, fixed not cherry-picked
    sid_to_idx = {}
    for idx in range(len(val_loader.dataset)):
        _, _, sid = val_loader.dataset[idx]
        if sid not in sid_to_idx:
            sid_to_idx[sid] = idx
        if sid == SID:
            break
    idx = sid_to_idx[SID]
    image, _, sid = val_loader.dataset[idx]
    assert sid == SID
    image_b = image.unsqueeze(0).to(device).double()

    t = ledger[SID]["tiles"][0]
    pd, ph, pw = PROBE_PATCH
    # Center the small probe patch within tile 0's full 128^3 extent, not its
    # corner -- a corner sub-block is likely background-heavy (near-zero
    # tissue signal), which would make Gamma_soft's gradient near-vacuous for
    # reasons unrelated to the null-space question being tested.
    full_extent = t["z1"] - t["z0"]
    z0 = t["z0"] + (full_extent - pd) // 2
    y0 = t["y0"] + (full_extent - ph) // 2
    x0 = t["x0"] + (full_extent - pw) // 2
    tile_image = image_b[:, :, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
    if tile_image.shape[2:] != PROBE_PATCH:
        tile_image = F.pad(tile_image, (0, pw - tile_image.shape[4],
                                        0, ph - tile_image.shape[3],
                                        0, pd - tile_image.shape[2]))
    print(f"[Data] subject={SID} tile_image shape={tuple(tile_image.shape)} (PROBE_PATCH={PROBE_PATCH})")

    model = UNet3D_v5(in_channels=4, out_channels=3).to(device).double()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hook = SingleTileHook()
    register(model, hook)

    # 1) forward once with hook in pass-through mode to get the intact enc3
    #    tile value Z (the leaf tensor we differentiate around).
    hook.z_override, hook.severity = None, None
    with torch.no_grad():
        _ = model(tile_image)
    # capture Z via a plain forward hook read (no override), then detach+leaf-ify
    captured = {}

    def _capture(module, inputs, output):
        captured["Z"] = output.detach().clone()

    h = getattr(model, TARGET_STAGE).register_forward_hook(_capture)
    with torch.no_grad():
        _ = model(tile_image)
    h.remove()
    Z = captured["Z"].requires_grad_(True)
    print(f"[Data] enc3 tile Z shape={tuple(Z.shape)}  norm={Z.norm().item():.4f}")

    # 2) grad(Gamma_soft) at Z -- ONE backward pass
    gamma_soft = compute_gamma_soft(model, tile_image, hook, Z)
    gamma_soft_val = float(gamma_soft.item())
    grad_gamma, = torch.autograd.grad(gamma_soft, Z, retain_graph=False)
    v = grad_gamma.detach()
    v_norm = float(v.norm().item())
    print(f"[Gamma_soft] value={gamma_soft_val:.6f}  ||grad||={v_norm:.6e}")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if v_norm == 0.0:
        out = {"status": "VACUOUS", "reason": "grad(Gamma_soft) is exactly zero at this point",
               "sid": SID, "gamma_soft_value": gamma_soft_val}
        json.dump(out, open(OUT_DIR / "E185_nullspace_probe_result.json", "w"), indent=1)
        print("[STOP] grad norm is zero -- probe is vacuous at this point. See saved result.")
        return

    # 3) JVP: J @ v via double-backward trick, D(Z) w.r.t. Z in direction v
    def D_of_z(z):
        return decoder_output(model, tile_image, hook, z)

    def jvp(z, direction):
        z = z.detach().requires_grad_(True)
        out = D_of_z(z)
        dummy = torch.zeros_like(out, requires_grad=True)
        g = torch.autograd.grad(out, z, grad_outputs=dummy, create_graph=True)[0]
        jv, = torch.autograd.grad(g, dummy, grad_outputs=direction)
        jv = jv.detach()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return jv

    Jv = jvp(Z, v)
    Jv_norm = float(Jv.norm().item())
    print(f"[JVP] ||J v|| = {Jv_norm:.6e}  (v = grad(Gamma_soft))")

    # 3b) finite-difference sanity check on the JVP implementation itself
    fd_checks = []
    g = torch.Generator(device="cpu")
    for i in range(FD_N_DIRS):
        g.manual_seed(RAND_SEED_BASE + 500 + i)
        d = torch.randn(Z.shape, generator=g).to(device)
        d = d / d.norm() * v_norm  # matched norm to v for comparability
        with torch.no_grad():
            D0 = D_of_z(Z)
            D1 = D_of_z(Z + FD_H * d)
        fd_est = (D1 - D0) / FD_H
        jvp_est = jvp(Z, d)
        rel_err = float((fd_est - jvp_est).norm() / (fd_est.norm() + 1e-12))
        fd_checks.append(rel_err)
    print(f"[FD check] relative errors (JVP autograd vs finite-difference): {fd_checks}")

    # 4) matched-norm random-direction control
    random_Jv_norms = []
    for i in range(N_RANDOM_DIRS):
        g.manual_seed(RAND_SEED_BASE + i)
        d = torch.randn(Z.shape, generator=g).to(device)
        d = d / d.norm() * v_norm
        Jd = jvp(Z, d)
        random_Jv_norms.append(float(Jd.norm().item()))
    mean_random_Jv = float(np.mean(random_Jv_norms))
    print(f"[Control] ||J d_rand|| per draw: {random_Jv_norms}")
    print(f"[Control] mean ||J d_rand|| = {mean_random_Jv:.6e}")

    r = Jv_norm / mean_random_Jv if mean_random_Jv > 0 else float("inf")
    if r < 0.3:
        verdict = "YES -- grad(Gamma_soft) is substantially output-null (r<0.3)"
    elif r > 0.7:
        verdict = "NO -- grad(Gamma_soft) is a generic direction, no null-space preference (r>0.7)"
    else:
        verdict = f"AMBIGUOUS -- r={r:.3f} in [0.3,0.7], no verdict forced"

    out = {
        "status": "COMPLETE",
        "sid": SID, "window_id": t["window_id"],
        "gamma_soft_value": gamma_soft_val,
        "grad_gamma_norm": v_norm,
        "Jv_norm": Jv_norm,
        "random_Jv_norms": random_Jv_norms,
        "mean_random_Jv_norm": mean_random_Jv,
        "r_ratio": r,
        "fd_check_relative_errors": fd_checks,
        "fd_check_max_rel_error": float(np.max(fd_checks)),
        "verdict": verdict,
        "note": "r = ||J @ grad(Gamma_soft)|| / mean(||J @ d_random||) at matched norm. "
                "Small r means grad(Gamma_soft) is preferentially output-null relative to a "
                "generic direction of the same size. No GT/Dice/optimizer used anywhere.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_DIR / "E185_nullspace_probe_result.json", "w"), indent=1)

    print("\n" + "=" * 70)
    print(f"r = {r:.4f}")
    print(f"VERDICT: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
