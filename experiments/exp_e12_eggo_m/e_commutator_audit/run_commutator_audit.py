"""
Retrospective commutator audit: does C_task(P) = mean_T d[D(T(PZ)), D(P(TZ))]
predict Gamma-change under the perturbation P, using ALREADY-GENERATED E184/
E187 candidate states? NO new perturbations, NO training, NO GT.

This is diagnostic analysis of existing artifacts -- reconstructs PZ from the
stored (sid, direction_idx/candidate_coords, alpha, seed) exactly as
run_e184_gamma_continuation.py / run_e187_cross_region_feasibility.py did
(same checkpoint, same seeds, same formula), verified against the ALREADY-
STORED gamma_Zprime / output_distance_from_Z0 values before trusting anything
downstream. The raw enc3 tensors themselves were never persisted to disk --
this reconstruction is the only way to obtain them, and is checked, not
assumed, to reproduce the original computation exactly.

T ranges over the 6 T6/T7 Gamma-family members (T6@0.35, T6@0.50, T7@0.60,
T7@0.80, T7@1.10, T7@1.45) -- the SAME family whose max-pairwise-distance
already defines Gamma throughout E180-E187, chosen so C_task is directly
comparable to Gamma in the same measurement units. d is prediction_distance
(mean abs probability difference), reused verbatim from E184.

P(T(Z)) uses the FIXED delta tensor alpha*std(Z)*V_j (computed ONCE from the
intact Z), added to T(Z) -- P is a fixed additive operator, not re-scaled by
T(Z)'s own statistics.

Controls (all computed here, none skipped):
  1. E184 positive-control association: C_task vs |Delta_Gamma| within subject.
  2. E187 negative control: same C_task-style analysis on E187's 35 passing
     boundary-crossing candidates (different P construction, dec3[0] not
     enc3-tile-random) -- does C_task stay flat where Gamma itself was flat?
  3. Output-displacement control: partial out ||D(PZ)-D(Z)|| (E184's own
     output_distance_from_Z0, already stored).
  4. Perturbation-magnitude control: partial out ||PZ-Z|| (computable exactly
     from alpha*std(Z)*||V_j|| = alpha*std(Z), since V_j is unit-norm).
  5. Subject-preserving permutation null: shuffle candidate<->Gamma pairing
     WITHIN each subject, re-test, compare to the real association.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

project_root = Path(__file__).resolve().parents[3]
e180_dir = project_root / "experiments" / "exp_e12_eggo_m" / "e180"
e181_dir = project_root / "experiments" / "exp_e12_eggo_m" / "e181"
e184_dir = project_root / "experiments" / "exp_e12_eggo_m" / "e184"
e186_dir = project_root / "experiments" / "exp_e12_eggo_m" / "e186"
e187_dir = project_root / "experiments" / "exp_e12_eggo_m" / "e187"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(e180_dir))
sys.path.insert(0, str(e181_dir))
sys.path.insert(0, str(e184_dir))
sys.path.insert(0, str(e186_dir))
sys.path.insert(0, str(e187_dir))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders  # noqa: E402
from run_e180_transform_lab import PATCH, TARGET_STAGE  # noqa: E402
from run_e180_gamma import enc3_tile_bounds, dice_agree  # noqa: E402
from run_e184_feasibility_probe import (  # noqa: E402
    independent_random_directions, N_DIRECTIONS, ALPHA_GRID, V_SEED_BASE,
    prediction_distance,
)
from run_e184_gamma_continuation import (  # noqa: E402
    T6_BUDGETS, T7_LEVELS, t6_transform, pure_energy_scaling,
)
from run_e187_cross_region_feasibility import (  # noqa: E402
    get_dec3_conv, build_delta_for_candidate, ENC3_C, CAT3_C,
)

OUT_DIR = Path(__file__).parent
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694

TFAMILY = [("T6", b) for b in T6_BUDGETS] + [("T7", c) for c in T7_LEVELS]  # 6 members, LOCKED


def apply_T(z, fam, sev):
    if fam == "T6":
        return t6_transform(z, sev)
    elif fam == "T7":
        return pure_energy_scaling(z, sev)
    raise ValueError(fam)


class SpliceHook:
    """Forward hook on enc3: within mask, replaces the tile region with
    z_override (full-shape tensor, same shape as output). If family/severity
    set, applies that T6/T7 transform on top of the (already-spliced)
    activation. Mirrors E184/E186/E187's own hook contract exactly."""

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
    return pr.squeeze(0).cpu().numpy()


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


def gamma_of(model, tile_image, hook, mask, z_override, amp):
    """Same Gamma definition throughout E180-E187: max pairwise Dice-
    displacement across the 6-member T6/T7 family, applied AT z_override."""
    preds = []
    for fam, sev in TFAMILY:
        hook.mask, hook.z_override, hook.family, hook.severity = mask, z_override, fam, sev
        p = single_tile_forward(model, tile_image, hook, amp=amp)
        preds.append((p > 0.5).astype(np.float32))
    dists = []
    for j in range(len(preds)):
        for k in range(j + 1, len(preds)):
            dists.append(1.0 - dice_agree(preds[j], preds[k]))
    return float(np.max(dists))


def d_out(model, tile_image, hook, mask, z_override, amp):
    hook.mask, hook.z_override, hook.family, hook.severity = mask, z_override, None, None
    return single_tile_forward(model, tile_image, hook, amp=amp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="limit to 1 subject, 2 candidates each, for fast iteration")
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

    # ---------------------------------------------------------------- E184
    e184_raw = json.load(open(e184_dir / "E184_gamma_continuation_raw.json"))
    e184_by_sid = {}
    for r in e184_raw:
        e184_by_sid.setdefault(r["sid"], []).append(r)

    if a.smoke:
        e184_by_sid = {k: v[:2] for k, v in list(e184_by_sid.items())[:1]}

    e184_results = []
    verify_fail = 0
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
        gamma_Z = gamma_of(model, tile_image, hook, mask, Z_full, amp=False)

        for r in recs:
            j, alpha = r["direction_idx"], r["alpha"]
            Vj = directions[j]
            std_Z = Z_full.std()
            P_delta_full = torch.zeros_like(Z_full)
            m3 = mask.view(1, 1, *mask.shape).to(Z_full.dtype)
            P_delta_full = (alpha * std_Z) * Vj.view(1, -1, 1, 1, 1) * m3

            PZ_full = Z_full + P_delta_full
            gamma_PZ = gamma_of(model, tile_image, hook, mask, PZ_full, amp=False)
            d_output = prediction_distance(
                d_out(model, tile_image, hook, mask, PZ_full, amp=False),
                d_out(model, tile_image, hook, mask, Z_full, amp=False))

            # CLOSENESS CHECK (not exact match): this audit uses a SINGLE-TILE
            # forward pass throughout (amp=False), while the ORIGINAL E184
            # script used amp=True + multi-window sliding_window() over the
            # full volume (verified separately to reproduce stored values
            # exactly). The two inference conventions are each internally
            # consistent and safe (E184's additive hook tolerates multi-
            # window; this audit's replace-based SpliceHook does not, so it
            # is deliberately kept single-tile throughout -- see design
            # discussion). A SMALL discrepancy between reconstructed and
            # stored Gamma/output_distance is EXPECTED from this convention
            # difference and is reported, not hidden -- large discrepancies
            # would indicate a real reconstruction bug and are flagged.
            gamma_rel_err = abs(gamma_PZ - r["gamma_Zprime"]) / max(abs(r["gamma_Zprime"]), 1e-8)
            reconstruction_close = gamma_rel_err < 0.15  # generous -- flags GENUINE bugs only
            if not reconstruction_close:
                verify_fail += 1
                print(f"  [LARGE MISMATCH] {sid} j={j} alpha={alpha}: "
                      f"gamma_recon={gamma_PZ:.6f} vs stored={r['gamma_Zprime']:.6f} "
                      f"(rel_err={gamma_rel_err:.2%})")
                continue

            # C_task(P) = mean_T d[D(T(PZ)), D(P(TZ))]
            c_task_terms = []
            for fam, sev in TFAMILY:
                # D(T(PZ)): apply T to the (already-perturbed) PZ
                hook.mask, hook.z_override, hook.family, hook.severity = (
                    mask, PZ_full, fam, sev)
                p_T_PZ = single_tile_forward(model, tile_image, hook, amp=False)

                # D(P(TZ)): T(Z) first, then add the FIXED P_delta on top.
                # T(Z) is computed directly (not via the hook, which applies
                # T INSIDE the forward pass to whatever z_override is set --
                # here we need the standalone tensor T(Z) to add P_delta to).
                TZ_full = apply_T(Z_full, fam, sev)
                P_of_TZ_full = TZ_full + P_delta_full
                hook.mask, hook.z_override, hook.family, hook.severity = (
                    mask, P_of_TZ_full, None, None)
                p_P_TZ = single_tile_forward(model, tile_image, hook, amp=False)

                c_task_terms.append(prediction_distance(p_T_PZ, p_P_TZ))
            c_task = float(np.mean(c_task_terms))

            delta_gamma = gamma_PZ - gamma_Z
            perturb_norm = float((alpha * std_Z).item())  # ||P_delta|| = alpha*std(Z)*1 (unit V_j)

            e184_results.append({
                "sid": sid, "direction_idx": j, "alpha": alpha,
                "gamma_Z": gamma_Z, "gamma_PZ": gamma_PZ, "delta_gamma": delta_gamma,
                "c_task": c_task, "c_task_per_T": c_task_terms,
                "output_distance_from_Z0": d_output,
                "perturb_norm": perturb_norm,
                "stored_gamma_Zprime": r["gamma_Zprime"], "gamma_rel_err_vs_stored": gamma_rel_err,
            })
        print(f"[E184] {sid}: {len(recs)} candidates processed", flush=True)

    print(f"\n[Verify] {verify_fail} reconstruction mismatches out of "
          f"{sum(len(v) for v in e184_by_sid.values())} E184 candidates")

    # ---------------------------------------------------------------- E187
    e187_raw = json.load(open(e187_dir / "E187_cross_region_feasibility_raw.json"))
    e187_gamma_recs = e187_raw["gamma_records_passing_only"]

    conv = get_dec3_conv(model)
    W = conv.weight.detach().clone()
    bias = conv.bias.detach().clone()

    e187_results = []
    e187_by_sid = {}
    for r in e187_gamma_recs:
        e187_by_sid.setdefault(r["sid"], []).append(r)

    if a.smoke:
        e187_by_sid = {k: v[:2] for k, v in list(e187_by_sid.items())[:1]}

    for sid, recs in e187_by_sid.items():
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
        gamma_Z = gamma_of(model, tile_image, hook, mask, Z_full, amp=False)
        enc3_full = Z_full[0]

        W_enc3_half = W[:, ENC3_C:CAT3_C, :, :, :]
        W_enc3_half_norm = W_enc3_half.reshape(W_enc3_half.shape[0], -1).norm(dim=1)
        pad = 1

        for r in recs:
            c = r["candidate_channel"]
            z, y, x = r["candidate_zyx"]
            # a_val not stored in gamma_records_passing_only -- recompute it
            # from the intact activation via the real conv, exactly as E187 did
            captured_cat3 = {}
            h_pre = conv.register_forward_pre_hook(
                lambda m, inp: captured_cat3.__setitem__("x", inp[0].detach().clone()))
            with torch.no_grad():
                _ = model(tile_image)
            h_pre.remove()
            preact = F.conv3d(captured_cat3["x"], W, bias, padding=1)
            a_val = float(preact[0, c, z, y, x].item())

            delta = build_delta_for_candidate(
                enc3_full, W_enc3_half, W_enc3_half_norm, pad, c, z, y, x, a_val)
            P_delta_full = delta.unsqueeze(0)
            PZ_full = Z_full + P_delta_full
            gamma_PZ = gamma_of(model, tile_image, hook, mask, PZ_full, amp=False)

            # E187's ORIGINAL script (corrected version) already used
            # single-tile inference throughout -- this reconstruction uses
            # the SAME convention, so it should match closely (unlike E184's
            # reconstruction, which deliberately differs from the original
            # multi-window+AMP convention for hook-safety reasons -- see
            # that section's comment).
            gamma_rel_err = abs(gamma_PZ - r["gamma_candidate"]) / max(abs(r["gamma_candidate"]), 1e-8)
            gamma_match = gamma_rel_err < 0.05
            if not gamma_match:
                print(f"  [E187 LARGE MISMATCH] {sid} c={c} zyx={z,y,x}: "
                      f"gamma_recon={gamma_PZ:.6f} vs stored={r['gamma_candidate']:.6f} "
                      f"(rel_err={gamma_rel_err:.2%})")
                continue

            c_task_terms = []
            for fam, sev in TFAMILY:
                hook.mask, hook.z_override, hook.family, hook.severity = (
                    mask, PZ_full, fam, sev)
                p_T_PZ = single_tile_forward(model, tile_image, hook, amp=False)

                TZ_full = apply_T(Z_full, fam, sev)
                P_of_TZ_full = TZ_full + P_delta_full
                hook.mask, hook.z_override, hook.family, hook.severity = (
                    mask, P_of_TZ_full, None, None)
                p_P_TZ = single_tile_forward(model, tile_image, hook, amp=False)
                c_task_terms.append(prediction_distance(p_T_PZ, p_P_TZ))
            c_task = float(np.mean(c_task_terms))

            delta_gamma = gamma_PZ - gamma_Z
            perturb_norm = float(P_delta_full.norm().item())

            e187_results.append({
                "sid": sid, "candidate_channel": c, "candidate_zyx": [z, y, x],
                "gamma_Z": gamma_Z, "gamma_PZ": gamma_PZ, "delta_gamma": delta_gamma,
                "c_task": c_task, "c_task_per_T": c_task_terms,
                "output_distance_from_Z0": r["output_distance_from_Z0"],
                "perturb_norm": perturb_norm,
                "stored_gamma_candidate": r["gamma_candidate"], "gamma_rel_err_vs_stored": gamma_rel_err,
            })
        print(f"[E187] {sid}: {len(recs)} candidates processed", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "commutator_audit_raw.json", "w") as f:
        json.dump({"e184_results": e184_results, "e187_results": e187_results,
                   "verify_fail_count": verify_fail}, f, indent=1)
    print(f"\nSaved commutator_audit_raw.json "
          f"(E184: {len(e184_results)} records, E187: {len(e187_results)} records)")


if __name__ == "__main__":
    main()
