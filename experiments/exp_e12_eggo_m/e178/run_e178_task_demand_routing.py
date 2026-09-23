"""
E178 Stage 1 -- Task-Demand-Weighted Compute Routing: single-seed, single-
budget (B=10%) feasibility pass across 5 arms + a Random sanity check.

Pre-registered in docs/phases/PHASE_E178_TASK_DEMAND_ROUTING_PREREG.md. Read
that first. This is NOT training -- the refinement operator is instantiated
with random init weights and the SAME weights are used, unrouted, across all
arms (a randomly-initialized but FIXED small residual block). This stage
tests only whether ROUTING SELECTION matters, holding the (untrained)
refinement operator fixed -- this isolates the routing principle from
"did we also train a better network," matching the prereg's explicit
"same refinement operator, different allocation" requirement. Training the
refinement operator end-to-end is deferred to Stage 2, contingent on Stage 1
passing (routing selection must matter even before the operator is any good
-- if E<=B here already, training end-to-end would only ever muddy whether
gains come from routing or from the extra trained parameters).

BINDING RULES (verified structurally, not just stated):
  - NO ground truth used anywhere in routing (D-arm's "size" signal is the
    model's OWN predicted foreground mass from the baseline forward pass,
    not GT volume).
  - The R_hat_enc1 predictor is the FROZEN pkl from E170 -- never refit here.
  - GAP CARRIED EXPLICITLY: R_hat's ridge weights were fit against PER-
    SUBJECT-aggregated E169 features vs PER-SUBJECT R*_enc1 (E165). Applying
    them to a single tile's own (non-aggregated) feature vector is an
    UNVALIDATED EXTRAPOLATION -- no per-tile R* ground truth exists anywhere
    to check this against. Stated here, not hidden.
  - Same RefinementBlock (2x Conv3DBlock(32,32), residual) in every arm.
  - Budget B=10% of tiles per subject, IDENTICAL COUNT across every arm.
  - GT/Dice used ONLY for final evaluation, never in routing.
"""
import sys
import json
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[3]
e169_dir = project_root / "experiments" / "exp_e12_eggo_m" / "e169"
e170_dir = project_root / "experiments" / "exp_e12_eggo_m" / "e170"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(e169_dir))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders  # noqa: E402
from neuroscan_3d_fixed import Conv3DBlock  # noqa: E402

OUT_DIR = Path(__file__).parent
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694
PATCH = (128, 128, 128)
STAGES = {"enc1": 32, "enc2": 64, "enc3": 128, "bottleneck": 256, "dec1": 32}
OVERLAP = 0.25
BUDGET = 0.10          # LOCKED, Stage 1 single-budget value
SEED = 0               # LOCKED, Stage 1 single-seed value
REFINE_SEED = 12345    # fixed random init for the (untrained) RefinementBlock,
                        # identical across every arm -- only used to verify
                        # ROUTING matters, not to claim any Dice improvement
                        # from a trained operator (that is Stage 2's question)


class RefinementBlock(nn.Module):
    """2x Conv3DBlock(32,32), residual. Identical instance (same weights,
    randomly initialized once) used by every arm -- isolates the routing
    SELECTION from the refinement OPERATOR itself."""

    def __init__(self, channels=32):
        super().__init__()
        self.block = nn.Sequential(
            Conv3DBlock(channels, channels),
            Conv3DBlock(channels, channels),
        )

    def forward(self, x):
        return x + self.block(x)


class StageFeaturesTiled:
    """Same per-tile feature computation as E169's StageFeatures.__call__,
    but returns the RAW per-tile dict (not subject-averaged) -- this is the
    per-tile feature vector fed to the frozen R_hat predictor."""

    def __init__(self):
        self.last = {}

    def __call__(self, name, out):
        x = out.detach().float()
        C = x.shape[1]
        m = x.reshape(C, -1)
        mu = m.mean(dim=1, keepdim=True)
        mc = m - mu
        n = mc.shape[1]
        if n > 20000:
            idx = torch.randperm(n, device=mc.device)[:20000]
            mcs = mc[:, idx]
        else:
            mcs = mc
        try:
            s = torch.linalg.svdvals(mcs)
        except Exception:
            return
        s = s.clamp(min=0)
        ev = s ** 2
        tot = ev.sum().clamp(min=1e-12)
        p = ev / tot
        eff_rank = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
        part_ratio = float((ev.sum() ** 2) / (ev ** 2).sum().clamp(min=1e-12))
        stable_rank = float(ev.sum() / ev.max().clamp(min=1e-12))
        cum = torch.cumsum(p, 0)
        top1 = float(p[0])
        top4 = float(cum[min(3, C - 1)])
        top16 = float(cum[min(15, C - 1)])
        chan_energy = m.abs().mean(dim=1)
        dead = float((chan_energy < 1e-3 * chan_energy.max().clamp(min=1e-12)).float().mean())
        self.last[name] = dict(
            eff_rank=eff_rank, eff_rank_frac=eff_rank / C,
            part_ratio=part_ratio, part_ratio_frac=part_ratio / C,
            stable_rank=stable_rank, top1=top1, top4=top4, top16=top16,
            energy_mean=float(m.abs().mean()), energy_std=float(m.std()),
            energy_max=float(m.abs().max()),
            chan_disp=float(chan_energy.std() / chan_energy.mean().clamp(min=1e-12)),
            dead_frac=dead,
        )


def ridge_apply(model, feat_dict):
    """Apply the frozen ridge weights to a per-tile feature dict. Returns
    R_hat (already exponentiated back from log2 space)."""
    x = np.array([feat_dict.get(k, 0.0) for k in model["keys"]], dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    xs = (x - model["mu"]) / model["sd"]
    log2_rhat = float(xs @ model["w"] + model["ym"])
    return float(2.0 ** log2_rhat)


def dice_per_region(pred_bin, target_bin):
    out = []
    for r in range(pred_bin.shape[0]):
        p, t = pred_bin[r], target_bin[r]
        ps, ts = p.sum(), t.sum()
        out.append(1.0 if ps == 0 and ts == 0 else float(2.0 * (p * t).sum() / (ps + ts)))
    return out


def _gaussian_weight(shape, sigma_scale=0.125):
    coords = [np.linspace(-1, 1, s) for s in shape]
    g = np.ones(shape, dtype=np.float32)
    for i, c in enumerate(coords):
        sh = [1] * len(shape)
        sh[i] = -1
        g = g * np.exp(-(c ** 2) / (2 * sigma_scale ** 2)).reshape(sh).astype(np.float32)
    return np.maximum(g, 1e-4)


def tile_starts(full, p, st):
    if full <= p:
        return [0]
    s = list(range(0, full - p + 1, st))
    if s[-1] != full - p:
        s.append(full - p)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--limit_subjects", type=int, default=0)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    got = ckpt.get("best_mean_dice")
    assert got is not None and abs(float(got) - EXPECTED_DICE) < 1e-9, \
        f"Checkpoint identity FAILED: expected {EXPECTED_DICE}, got {got}"
    print(f"[Sanity] checkpoint identity PASS (best_mean_dice={got}).")

    rhat_model = pickle.load(open(e170_dir / "rhat_enc1_predictor.pkl", "rb"))
    print(f"[Model] R_hat predictor loaded: {len(rhat_model['keys'])} features, "
          f"target={rhat_model['target_stage']}")

    model = UNet3D_v5(in_channels=4, out_channels=3).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    g = torch.Generator(device=device).manual_seed(REFINE_SEED)
    refine = RefinementBlock(channels=32).to(device)
    with torch.no_grad():
        # BUG CAUGHT during smoke-testing: only re-initializing conv weights
        # (dim>1) and zeroing every 1-D parameter left BatchNorm's weight
        # (gamma, 1-D) at ZERO -- a zero-scale BatchNorm outputs exactly
        # zero regardless of input, ReLU keeps it zero, so BOTH Conv3DBlocks
        # collapsed to producing exactly zero, making the residual block an
        # EXACT identity (verified: relative diff = 0.0 on a random test
        # input). Fixed: only zero conv BIASES (the intended "start near
        # identity but not exactly" default); leave BatchNorm's own default
        # init (weight=1, bias=0) untouched, matching standard practice.
        for name, p in refine.named_parameters():
            if "conv.weight" in name:
                nn.init.kaiming_normal_(p, generator=g)
            elif "conv.bias" in name:
                nn.init.zeros_(p)
            # bn.weight/bn.bias: leave at PyTorch's own default init
            # (weight=1, bias=0) -- do NOT touch here.
    refine.eval()

    feats = StageFeaturesTiled()
    tile_state = {"routed_mask": None, "apply_refine": False}

    def dec1_hook(module, inputs, output):
        feats("dec1", output)
        if tile_state["apply_refine"]:
            return refine(output)
        return output

    handles = []
    for name in STAGES:
        mod = getattr(model, name)
        if name == "dec1":
            handles.append(mod.register_forward_hook(dec1_hook))
        else:
            handles.append(mod.register_forward_hook(
                lambda m, i, o, nm=name: feats(nm, o)))

    _, val_loader = create_multimodal_loaders(
        root_dir=str(project_root / "Dataset" / "Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

    amp = not a.no_amp
    n_target = a.limit_subjects if a.limit_subjects else len(val_loader.dataset)
    n = min(n_target, len(val_loader.dataset))
    print(f"[Data] {n} validation subjects, budget={BUDGET:.0%}, seed={SEED}")

    ARMS = ["baseline", "constant", "uncertainty", "size", "neuroscan", "random"]
    per_subject_records = []

    rng = np.random.default_rng(SEED)

    for si in range(n):
        image, target, sid = val_loader.dataset[si]
        image_b = image.unsqueeze(0).to(device)
        tgt = target.numpy()
        _, D, H, W = image.shape
        pd, ph, pw = PATCH
        stride = [max(1, int(p * (1 - OVERLAP))) for p in PATCH]
        zs = tile_starts(D, pd, stride[0])
        ys = tile_starts(H, ph, stride[1])
        xs = tile_starts(W, pw, stride[2])
        tile_coords = [(z, y, x) for z in zs for y in ys for x in xs]
        n_tiles = len(tile_coords)
        n_route = max(1, int(round(n_tiles * BUDGET)))

        # ---- PASS 1: baseline forward, collect per-tile features + signals
        per_tile_rhat = []
        per_tile_entropy = []
        per_tile_size = []
        gw = torch.from_numpy(_gaussian_weight(
            (min(pd, D), min(ph, H), min(pw, W)))).to(device)
        acc = torch.zeros((3, D, H, W), device=device, dtype=torch.float32)
        wsum = torch.zeros((1, D, H, W), device=device, dtype=torch.float32)

        tile_state["apply_refine"] = False
        with torch.no_grad():
            for (z, y, x) in tile_coords:
                zc, yc, xc = min(pd, D), min(ph, H), min(pw, W)
                tile = image_b[:, :, z:z + zc, y:y + yc, x:x + xc]
                if tile.shape[2:] != (pd, ph, pw):
                    tile = F.pad(tile, (0, pw - tile.shape[4], 0, ph - tile.shape[3],
                                        0, pd - tile.shape[2]))
                with torch.amp.autocast("cuda", enabled=amp):
                    out = model(tile)
                pr = (out["probs"] if isinstance(out, dict) else out).float()
                pr = pr[:, :, :zc, :yc, :xc].squeeze(0)
                acc[:, z:z + zc, y:y + yc, x:x + xc] += pr * gw
                wsum[:, z:z + zc, y:y + yc, x:x + xc] += gw

                rhat_i = ridge_apply(rhat_model, feats.last)
                p = pr.clamp(1e-6, 1 - 1e-6)
                entropy_i = float(-(p * p.log() + (1 - p) * (1 - p).log()).mean().item())
                size_i = float(pr.sum().item())
                per_tile_rhat.append(rhat_i)
                per_tile_entropy.append(entropy_i)
                per_tile_size.append(size_i)

        p_baseline = (acc / wsum.clamp(min=1e-6)).cpu().numpy()
        dice_baseline = dice_per_region((p_baseline > 0.5).astype(np.float32), tgt)

        # ---- routing sets per arm
        idx_sorted_rhat = np.argsort(per_tile_rhat)[::-1]
        idx_sorted_entropy = np.argsort(per_tile_entropy)[::-1]
        idx_sorted_size = np.argsort(per_tile_size)[::-1]
        idx_constant = np.arange(n_tiles)  # deterministic, content-blind order
        idx_random = rng.permutation(n_tiles)

        routed_sets = {
            "baseline": set(),
            "constant": set(idx_constant[:n_route].tolist()),
            "uncertainty": set(idx_sorted_entropy[:n_route].tolist()),
            "size": set(idx_sorted_size[:n_route].tolist()),
            "neuroscan": set(idx_sorted_rhat[:n_route].tolist()),
            "random": set(idx_random[:n_route].tolist()),
        }

        # ---- PASS 2 per arm: re-run sliding window, refine routed tiles only
        subject_result = {"sid": sid, "n_tiles": n_tiles, "n_route": n_route,
                          "dice_baseline": dice_baseline}
        for arm in ARMS:
            routed = routed_sets[arm]
            if not routed:
                subject_result[f"dice_{arm}"] = dice_baseline
                continue
            acc2 = torch.zeros((3, D, H, W), device=device, dtype=torch.float32)
            wsum2 = torch.zeros((1, D, H, W), device=device, dtype=torch.float32)
            with torch.no_grad():
                for ti, (z, y, x) in enumerate(tile_coords):
                    zc, yc, xc = min(pd, D), min(ph, H), min(pw, W)
                    tile = image_b[:, :, z:z + zc, y:y + yc, x:x + xc]
                    if tile.shape[2:] != (pd, ph, pw):
                        tile = F.pad(tile, (0, pw - tile.shape[4], 0, ph - tile.shape[3],
                                            0, pd - tile.shape[2]))
                    tile_state["apply_refine"] = ti in routed
                    with torch.amp.autocast("cuda", enabled=amp):
                        out = model(tile)
                    pr = (out["probs"] if isinstance(out, dict) else out).float()
                    pr = pr[:, :, :zc, :yc, :xc].squeeze(0)
                    acc2[:, z:z + zc, y:y + yc, x:x + xc] += pr * gw
                    wsum2[:, z:z + zc, y:y + yc, x:x + xc] += gw
            p_arm = (acc2 / wsum2.clamp(min=1e-6)).cpu().numpy()
            subject_result[f"dice_{arm}"] = dice_per_region(
                (p_arm > 0.5).astype(np.float32), tgt)

        tile_state["apply_refine"] = False
        per_subject_records.append(subject_result)
        print(f"[{si+1}/{n}] {sid}  n_tiles={n_tiles} n_route={n_route}  "
              f"dice_baseline(mean)={np.mean(dice_baseline):.4f}  "
              f"dice_neuroscan(mean)={np.mean(subject_result['dice_neuroscan']):.4f}",
              flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "E178_stage1_raw.json", "w") as f:
        json.dump({"budget": BUDGET, "seed": SEED, "refine_seed": REFINE_SEED,
                   "records": per_subject_records}, f, indent=1)
    print(f"\nSaved E178_stage1_raw.json ({len(per_subject_records)} subjects)")


if __name__ == "__main__":
    main()
