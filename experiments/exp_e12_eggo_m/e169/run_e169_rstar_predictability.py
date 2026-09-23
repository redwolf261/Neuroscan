"""
E169 -- Is R*_{i,l} predictable from a SINGLE intact forward pass?

PRE-REGISTERED in docs/phases/PHASE_E169_RSTAR_PREDICTABILITY_PREREG.md.
Read that first; the decision rule below is copied from it and must not be
changed after seeing results.

THE QUESTION. E165 measured, per subject and stage, the minimum channel rank
R* retaining >=90% agreement with the subject's OWN undegraded prediction
(never ground truth -- the E147 confound guard). Obtaining R* requires a
truncation SWEEP: up to 9 extra full sliding-window passes per stage. Is R*
instead obtainable from ONE intact forward pass?

BINDING CONSTRAINT -- NON-CIRCULARITY. Every predictor must come from a
single intact forward pass. Anything derived from the truncation sweep (in
particular E165's stored `agree_curve`) is FORBIDDEN: a feature built from
the sweep is not cheaper than the measurement it predicts.

WHAT A PASS WOULD AND WOULD NOT MEAN. R* is already label-free, so this is a
COMPUTE question, not a supervision one. A pass shows R* is a cheap
statistic. It does NOT authorise an allocation design: "predict demand ->
allocate capacity" is closed three independent ways (E71 sign -0.290
backwards; E147 inverse -0.496 with 74/88 subjects served by rank<=4 of 256,
i.e. no scarcity; E72/E73 inert). Predicting a quantity does not create a
resource to allocate. Hence the rule below can only KILL or PARK.

PRE-REGISTERED DECISION RULE (fixed before running):
  A KILL   out-of-sample R^2 <= 0.10 at EVERY stage, or permutation p > 0.05.
           R* is obtainable only by running the sweep. Adaptive-rank branch
           closes.
  B PARK   R^2 > 0.10 raw but loses >=50% of R^2 after partialling lesion
           volume => a restatement of lesion size.
  C PARK+  R^2 > 0.10 AND retains >=50% after the volume control AND
           permutation p < 0.01. R* is a cheap statistic; still no allocation
           licence.

  The 0.10 bar is deliberately LOW because it is a KILL threshold -- it must
  be easy to clear so that failing it is decisive. It is not a success bar.

ANTICIPATED (recorded before running, so the result cannot be read as
confirmation): outcome B or C at enc1, A at the bottleneck. The bottleneck is
90% served by rank<=4 -- near-constant, little variance to predict. enc1 has
real spread (R* in {4,8,16,32}, 31% of subjects above half capacity) and is
the only stage where prediction is even well-posed.
"""
import sys
import json
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders, REGIONS  # noqa: E402

OUT_DIR = Path(__file__).parent
E165_TABLE = project_root / "experiments" / "exp_e12_eggo_m" / "e165" / "E165_per_subject.json"
E160_TABLE = project_root / "experiments" / "exp_e12_eggo_m" / "e160" / "E160_L_per_subject.json"

PATCH = (128, 128, 128)
CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E131_v5control_seed0" / "checkpoints" / "best.pth")
EXPECTED_DICE = 0.8929357248544694

STAGES = {"enc1": 32, "enc2": 64, "enc3": 128, "bottleneck": 256, "dec1": 32}
N_FOLDS = 5
SEED = 0


# --------------------------------------------------------- feature capture
class StageFeatures:
    """Accumulates per-stage activation statistics over sliding-window tiles.

    All statistics come from the INTACT forward pass. No truncation anywhere.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.acc = {k: [] for k in STAGES}

    def __call__(self, name, out):
        # out: (1,C,D,H,W) -- summarise this tile, accumulate, average later.
        x = out.detach().float()
        b, C = x.shape[0], x.shape[1]
        m = x.reshape(C, -1)                       # (C, N)
        mu = m.mean(dim=1, keepdim=True)
        mc = m - mu
        # channel covariance spectrum via economy SVD on a voxel subsample
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
        ev = (s ** 2)
        tot = ev.sum().clamp(min=1e-12)
        p = ev / tot
        # spectral descriptors
        eff_rank = float(torch.exp(-(p * (p + 1e-12).log()).sum()))   # entropy-based
        part_ratio = float((ev.sum() ** 2) / (ev ** 2).sum().clamp(min=1e-12))
        stable_rank = float(ev.sum() / ev.max().clamp(min=1e-12))
        cum = torch.cumsum(p, 0)
        top1 = float(p[0])
        top4 = float(cum[min(3, C - 1)])
        top16 = float(cum[min(15, C - 1)])
        # energy / occupancy
        chan_energy = m.abs().mean(dim=1)
        dead = float((chan_energy < 1e-3 * chan_energy.max().clamp(min=1e-12)).float().mean())
        self.acc[name].append(dict(
            eff_rank=eff_rank, eff_rank_frac=eff_rank / C,
            part_ratio=part_ratio, part_ratio_frac=part_ratio / C,
            stable_rank=stable_rank, top1=top1, top4=top4, top16=top16,
            energy_mean=float(m.abs().mean()), energy_std=float(m.std()),
            energy_max=float(m.abs().max()),
            chan_disp=float(chan_energy.std() / chan_energy.mean().clamp(min=1e-12)),
            dead_frac=dead,
        ))

    def summary(self):
        out = {}
        for name, lst in self.acc.items():
            if not lst:
                continue
            keys = lst[0].keys()
            for k in keys:
                out[f"{name}.{k}"] = float(np.mean([d[k] for d in lst]))
        return out


def _gaussian_weight(shape, sigma_scale=0.125):
    coords = [np.linspace(-1, 1, s) for s in shape]
    g = np.ones(shape, dtype=np.float32)
    for i, c in enumerate(coords):
        sh = [1] * len(shape)
        sh[i] = -1
        g = g * np.exp(-(c ** 2) / (2 * sigma_scale ** 2)).reshape(sh).astype(np.float32)
    return np.maximum(g, 1e-4)


def forward_collect(model, image, feats, device, amp=True, overlap=0.25):
    """ONE intact sliding-window pass; collects features and the prediction."""
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
    acc = torch.zeros((3, D, H, W), device=device, dtype=torch.float32)
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


# ------------------------------------------------------------- estimators
def ridge_cv(X, y, folds, alphas=(0.1, 1.0, 10.0, 100.0)):
    """Cross-fitted ridge with inner alpha selection. Returns OOS predictions."""
    pred = np.zeros_like(y, dtype=float)
    for f in range(folds.max() + 1):
        tr, te = folds != f, folds == f
        Xtr, ytr = X[tr], y[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (X[te] - mu) / sd
        ym = ytr.mean()
        # inner split for alpha
        best, best_a = np.inf, alphas[0]
        inner = np.arange(len(ytr)) % 3
        for a in alphas:
            errs = []
            for g in range(3):
                itr, ite = inner != g, inner == g
                A = Xtr_s[itr].T @ Xtr_s[itr] + a * np.eye(X.shape[1])
                w = np.linalg.solve(A, Xtr_s[itr].T @ (ytr[itr] - ytr[itr].mean()))
                p = Xtr_s[ite] @ w + ytr[itr].mean()
                errs.append(np.mean((p - ytr[ite]) ** 2))
            if np.mean(errs) < best:
                best, best_a = np.mean(errs), a
        A = Xtr_s.T @ Xtr_s + best_a * np.eye(X.shape[1])
        w = np.linalg.solve(A, Xtr_s.T @ (ytr - ym))
        pred[te] = Xte_s @ w + ym
    return pred


def r2(y, p):
    ss = np.sum((y - p) ** 2)
    st_ = np.sum((y - y.mean()) ** 2)
    return float(1 - ss / max(st_, 1e-12))


def partial_out(X, Z):
    """Residualise X on Z (with intercept)."""
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(Z1, X, rcond=None)
    return X - Z1 @ beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--features_only", action="store_true",
                    help="stop after writing features (skip modelling)")
    a = ap.parse_args()

    feat_path = OUT_DIR / "E169_features.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------- 1. features
    if feat_path.exists() and not a.limit:
        print(f"[Features] reusing {feat_path}")
        rows = json.load(open(feat_path))
    else:
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

        feats = StageFeatures()
        handles = []
        for name in STAGES:
            mod = getattr(model, name)
            handles.append(mod.register_forward_hook(
                lambda m, i, o, n=name: feats(n, o)))

        _, val_loader = create_multimodal_loaders(
            root_dir=str(project_root / "Dataset" / "Training"),
            batch_size=1, num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)

        amp = not a.no_amp
        rows = []
        n = len(val_loader.dataset)
        if a.limit:
            n = min(n, a.limit)
        for i in range(n):
            image, target, sid = val_loader.dataset[i]
            feats.reset()
            probs = forward_collect(model, image.unsqueeze(0).to(device), feats, device, amp=amp)
            d = {"sid": sid}
            d.update(feats.summary())
            # readout features (label-free: from the PREDICTION, not the target)
            pred = (probs > 0.5)
            for r, name in enumerate(REGIONS):
                d[f"pred.vol_{name}"] = float(pred[r].sum())
                d[f"pred.logvol_{name}"] = float(np.log1p(pred[r].sum()))
                d[f"pred.maxp_{name}"] = float(probs[r].max())
                d[f"pred.meanp_fg_{name}"] = float(probs[r][pred[r]].mean()) if pred[r].any() else 0.0
            rows.append(d)
            print(f"[{i+1}/{n}] {sid}", flush=True)
        for h in handles:
            h.remove()
        with open(feat_path, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"[Features] wrote {feat_path}")

    if a.features_only:
        return

    # ---------------------------------------------------- 2. targets
    e165 = {r["sid"]: r for r in json.load(open(E165_TABLE))}
    e160 = {r["sid"]: r for r in json.load(open(E160_TABLE))}
    rows = [r for r in rows if r["sid"] in e165]
    sids = [r["sid"] for r in rows]
    print(f"[Data] {len(sids)} subjects with both features and E165 targets")

    fkeys = sorted(k for k in rows[0] if k != "sid")
    X_all = np.array([[r[k] for k in fkeys] for r in rows], float)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    vol = np.array([[np.log1p(e160[s][f"size_{n}"]) for n in REGIONS] for s in sids], float)

    rng = np.random.default_rng(SEED)
    folds = rng.permutation(len(sids)) % N_FOLDS   # by SUBJECT, seeded before modelling

    results = {}
    for stage, C in STAGES.items():
        y_raw = np.array([e165[s]["stages"][stage]["R_star"] for s in sids], float)
        y = np.log2(y_raw)
        if y.std() < 1e-9:
            results[stage] = {"note": "target is constant -- prediction ill-posed",
                              "R_star_unique": sorted(set(y_raw.tolist()))}
            continue

        pred = ridge_cv(X_all, y, folds)
        r2_raw = r2(y, pred)
        rho_raw = float(stats.spearmanr(y, pred).statistic)

        # volume control: residualise BOTH features and target on log lesion volume
        Xr = partial_out(X_all, vol)
        yr = partial_out(y.reshape(-1, 1), vol).ravel()
        pred_r = ridge_cv(Xr, yr, folds)
        r2_ctl = r2(yr, pred_r)

        # permutation null on the raw model (shuffle target WITHIN folds)
        null = []
        for _ in range(1000):
            ysh = y.copy()
            for f in range(N_FOLDS):
                m = folds == f
                ysh[m] = rng.permutation(ysh[m])
            null.append(r2(ysh, ridge_cv(X_all, ysh, folds)))
        null = np.array(null)
        p_perm = float((null >= r2_raw).mean())

        retain = (r2_ctl / r2_raw) if r2_raw > 0 else 0.0
        results[stage] = {
            "C": C,
            "R_star_distribution": {str(int(v)): int((y_raw == v).sum())
                                    for v in sorted(set(y_raw.tolist()))},
            "target_std_log2": float(y.std()),
            "r2_oos": r2_raw, "spearman_oos": rho_raw,
            "r2_oos_volume_controlled": r2_ctl,
            "retained_fraction": float(retain),
            "permutation_p": p_perm,
            "permutation_null_mean": float(null.mean()),
        }

    # ---------------------------------------------------- 3. verdict
    scored = {k: v for k, v in results.items() if "r2_oos" in v}
    any_pass = any(v["r2_oos"] > 0.10 and v["permutation_p"] <= 0.05 for v in scored.values())
    if not any_pass:
        verdict = "A_KILL"
    else:
        best = max((v for v in scored.values()
                    if v["r2_oos"] > 0.10 and v["permutation_p"] <= 0.05),
                   key=lambda v: v["r2_oos"])
        if best["retained_fraction"] >= 0.50 and best["permutation_p"] < 0.01:
            verdict = "C_PARK_PLUS"
        else:
            verdict = "B_PARK_SIZE_PROXY"

    summary = {
        "question": "Is R*_{i,l} predictable from a SINGLE intact forward pass?",
        "prereg": "docs/phases/PHASE_E169_RSTAR_PREDICTABILITY_PREREG.md",
        "non_circularity": "all predictors from ONE intact forward pass; E165 agree_curve FORBIDDEN",
        "n_subjects": len(sids), "n_features": len(fkeys), "n_folds": N_FOLDS,
        "cross_fitting": "by subject, folds seeded before features were modelled",
        "per_stage": results,
        "PREREGISTERED_VERDICT": verdict,
        "decision_rule": "A_KILL if no stage has R2>0.10 with perm p<=0.05; "
                         "C_PARK_PLUS if best retains >=50% after volume control and p<0.01; "
                         "else B_PARK_SIZE_PROXY",
        "allocation_note": "A pass does NOT authorise an allocation design -- closed by E71 "
                           "(sign -0.290 backwards), E147 (inverse, no scarcity), E72/E73 (inert).",
    }

    with open(OUT_DIR / "E169_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    with open(OUT_DIR / "E169_per_stage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "C", "r2_oos", "spearman_oos", "r2_volume_controlled",
                    "retained_fraction", "permutation_p"])
        for s, v in results.items():
            if "r2_oos" in v:
                w.writerow([s, v["C"], round(v["r2_oos"], 4), round(v["spearman_oos"], 4),
                            round(v["r2_oos_volume_controlled"], 4),
                            round(v["retained_fraction"], 4), v["permutation_p"]])

    print("\n" + "=" * 70)
    print(json.dumps(summary, indent=1))
    print("=" * 70)


if __name__ == "__main__":
    main()
