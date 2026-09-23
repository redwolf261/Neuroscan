"""
E165b analysis -- threshold robustness of the E169b phenomenon, tau in
{0.80, 0.85, 0.90, 0.95, 0.99}.

PRE-REGISTERED. The ONLY change upstream was removing E165's early `break` so
the full agreement curve is stored. Everything else is frozen: same checkpoint,
same 125 subjects, same tile extraction, same truncation operator, same 77
predictor features (pred.* removed for the size-controlled model, per E169b),
same seeded by-subject folds, same volume controls, same dyadic rank grid.

THE PREDICTOR IS NOT RETUNED for the new thresholds. Ridge alpha is selected
inside each training fold exactly as in E169b -- no threshold-specific tuning.
Otherwise the robustness test silently becomes another modelling experiment.

SUCCESS CRITERION (qualitative, fixed in advance): identical R^2 is NOT
required. The question is whether R*_enc3 >> R*_dec1 survives the extended
range. If the separation collapses at 0.95/0.99, we report the phenomenon as
threshold-dependent rather than rescuing it.

CEILING DIAGNOSTIC (required): at high tau, R* can saturate against the dyadic
grid. A strong or weak R^2 could then reflect quantisation rather than signal.
So the R* DISTRIBUTION is reported at every tau: unique values, min/median/max,
and the fraction pinned at the grid ceiling C.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
E165B = HERE / "E165b_per_subject.json"
FEATS = ROOT / "experiments/exp_e12_eggo_m/e169/FROZEN/E169_features.json"
E160 = ROOT / "experiments/exp_e12_eggo_m/e160/E160_L_per_subject.json"

GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256]
TAUS = [0.80, 0.85, 0.90, 0.95, 0.99]
STAGES = ["enc1", "enc2", "enc3", "bottleneck", "dec1"]
REGIONS = ("ET", "TC", "WT")
N_FOLDS, SEED = 5, 0


def rstar_at(rec, stage, tau):
    """First dyadic rank reaching agreement >= tau. Ceiling = C if never."""
    c = rec["stages"][stage]["agree_curve"]
    C = rec["stages"][stage]["C"]
    for i, v in enumerate(c):
        if v is not None and v >= tau:
            return GRID[i], False
    return C, True                      # (value, hit_ceiling)


def partial_out(M, Z):
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    b, *_ = np.linalg.lstsq(Z1, M, rcond=None)
    return M - Z1 @ b


def ridge_oof(X, y, folds, alphas=(0.1, 1.0, 10.0, 100.0)):
    p = np.zeros_like(y)
    for f in range(folds.max() + 1):
        tr, te = folds != f, folds == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ym = y[tr].mean()
        best = (np.inf, alphas[0])
        inner = np.arange(int(tr.sum())) % 3
        for a in alphas:
            e = []
            for g in range(3):
                i1, i2 = inner != g, inner == g
                M = A[i1].T @ A[i1] + a * np.eye(X.shape[1])
                w = np.linalg.solve(M, A[i1].T @ (y[tr][i1] - y[tr][i1].mean()))
                e.append(np.mean((A[i2] @ w + y[tr][i1].mean() - y[tr][i2]) ** 2))
            if np.mean(e) < best[0]:
                best = (np.mean(e), a)
        M = A.T @ A + best[1] * np.eye(X.shape[1])
        w = np.linalg.solve(M, A.T @ (y[tr] - ym))
        p[te] = B @ w + ym
    return p


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))


def main():
    recs = {r["sid"]: r for r in json.load(open(E165B))}
    feats = [f for f in json.load(open(FEATS)) if f["sid"] in recs]
    e160 = {r["sid"]: r for r in json.load(open(E160))}
    feats = [f for f in feats if f["sid"] in e160]
    sids = [f["sid"] for f in feats]
    print(f"[Data] {len(sids)} subjects with full curves + features + volumes")

    keys = sorted(k for k in feats[0] if k != "sid")
    act = [k for k in keys if not k.startswith("pred.")]
    X = np.nan_to_num(np.array([[f[k] for k in act] for f in feats], float))
    Zg = np.array([[np.log1p(e160[s][f"size_{n}"]) for n in REGIONS] for s in sids])
    Zp = np.nan_to_num(np.array([[f[f"pred.logvol_{n}"] for n in REGIONS] for f in feats], float))
    Z = np.column_stack([Zg, Zp])                       # model D: BOTH controls

    rng = np.random.default_rng(SEED)
    folds = rng.permutation(len(sids)) % N_FOLDS

    out = {"taus": TAUS, "n_subjects": len(sids), "per_tau": {}}
    print(f"\n{'tau':>5s} " + " ".join(f"{s:>10s}" for s in STAGES))
    for tau in TAUS:
        row, dist = {}, {}
        for st in STAGES:
            vals, ceil = zip(*[rstar_at(recs[s], st, tau) for s in sids])
            v = np.array(vals, float)
            dist[st] = {
                "unique": sorted(set(int(x) for x in v)),
                "min": int(v.min()), "median": float(np.median(v)), "max": int(v.max()),
                "ceiling_frac": float(np.mean(ceil)),
                "C": recs[sids[0]]["stages"][st]["C"],
                "counts": {str(int(u)): int((v == u).sum()) for u in sorted(set(v))},
            }
            if v.std() < 1e-9:
                row[st] = float("nan")      # constant target: ill-posed
                continue
            y = np.log2(v)
            yr = partial_out(y.reshape(-1, 1), Z).ravel()
            Xr = partial_out(X, Z)
            row[st] = r2(yr, ridge_oof(Xr, yr, folds))
        out["per_tau"][str(tau)] = {"r2_D_both": row, "rstar_distribution": dist}
        print(f"{tau:5.2f} " + " ".join(
            f"{row[s]:10.4f}" if not np.isnan(row[s]) else f"{'const':>10s}" for s in STAGES))

    print("\n--- R* DISTRIBUTION (ceiling diagnostic) ---")
    for tau in TAUS:
        d = out["per_tau"][str(tau)]["rstar_distribution"]
        print(f"tau={tau:.2f}")
        for st in STAGES:
            x = d[st]
            print(f"   {st:11s} C={x['C']:3d} median={x['median']:6.1f} "
                  f"max={x['max']:3d} ceiling={100*x['ceiling_frac']:5.1f}% "
                  f"uniq={x['unique']}")

    e3 = [out["per_tau"][str(t)]["r2_D_both"]["enc3"] for t in TAUS]
    d1 = [out["per_tau"][str(t)]["r2_D_both"]["dec1"] for t in TAUS]
    survives = all((not np.isnan(a)) and (not np.isnan(b)) and a > b + 0.15
                   for a, b in zip(e3, d1))
    out["VERDICT"] = ("SURVIVES_extended_threshold_range" if survives
                      else "THRESHOLD_DEPENDENT_report_as_such")
    out["frozen_reference_tau090"] = {"enc1": 0.3929, "enc2": 0.4514, "enc3": 0.5178,
                                      "bottleneck": 0.1117, "dec1": 0.0042}
    json.dump(out, open(HERE / "E165b_threshold_sweep.json", "w"), indent=1)
    print("\nVERDICT:", out["VERDICT"])
    print("(criterion fixed in advance: enc3 exceeds dec1 by >0.15 at every tau)")


if __name__ == "__main__":
    main()
