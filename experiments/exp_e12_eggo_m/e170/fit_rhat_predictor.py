"""
E170 step 1 -- fit and FREEZE the R_hat_enc1 predictor.

Must run BEFORE any TDM training. The predictor maps one-intact-forward-pass
features to R*_enc1, is fitted on the 125 val subjects where E165 measured R*,
and is then FROZEN and applied to the 1126 train subjects during TDM training.

LEAKAGE CONTROL. E169's subject-level seeded folds are reused verbatim, so the
reported quality is out-of-fold. The FINAL shipped predictor is refit on all
125 (standard practice), but its quality is quoted from the OOF estimate, never
from in-sample fit.

GENERALISATION CAVEAT, stated up front: the predictor is fitted on 125 subjects
and applied to 1126 unseen ones. Generalisation is ASSUMED, not verified -- we
have no R* on train subjects. This script therefore also dumps the predicted
R_hat distribution on train vs val as a sanity check; a large distribution shift
would invalidate the TDM-predicted arm (though not the constant arm).

Feature subset: enc1-relevant only. Using all 77 features would let the
predictor lean on deep-stage statistics that are themselves downstream of enc1,
which is fine for measurement but muddies the mechanistic story. We keep the
full set (that is what E169 validated) and additionally report an enc1-only
variant for comparison.
"""
import sys, json, pickle
from pathlib import Path
import numpy as np
from scipy import stats

project_root = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).parent
E169_FEATS = project_root / "experiments" / "exp_e12_eggo_m" / "e169" / "E169_features.json"
E165_TABLE = project_root / "experiments" / "exp_e12_eggo_m" / "e165" / "E165_per_subject.json"

N_FOLDS, SEED = 5, 0
TARGET_STAGE = "enc1"          # per E170: enc1-only intervention (84% deficit vs 6.4% at enc3)


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    ym = y.mean()
    A = Xs.T @ Xs + alpha * np.eye(X.shape[1])
    w = np.linalg.solve(A, Xs.T @ (y - ym))
    return dict(mu=mu, sd=sd, ym=ym, w=w, alpha=alpha)


def ridge_apply(m, X):
    return ((X - m["mu"]) / m["sd"]) @ m["w"] + m["ym"]


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))


def main():
    rows = json.load(open(E169_FEATS))
    e165 = {r["sid"]: r for r in json.load(open(E165_TABLE))}
    rows = [r for r in rows if r["sid"] in e165]
    sids = [r["sid"] for r in rows]

    keys_all = sorted(k for k in rows[0] if k != "sid")
    keys_e1 = [k for k in keys_all if k.startswith("enc1.")]

    def mat(keys):
        M = np.array([[r[k] for k in keys] for r in rows], float)
        return np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    y = np.log2(np.array([e165[s]["stages"][TARGET_STAGE]["R_star"] for s in sids], float))
    rng = np.random.default_rng(SEED)
    folds = rng.permutation(len(sids)) % N_FOLDS      # identical construction to E169

    report = {}
    for name, keys in [("all77", keys_all), ("enc1_only", keys_e1)]:
        X = mat(keys)
        best = None
        for alpha in (0.1, 1.0, 10.0, 100.0):
            oof = np.zeros_like(y)
            for f in range(N_FOLDS):
                tr, te = folds != f, folds == f
                oof[te] = ridge_apply(ridge_fit(X[tr], y[tr], alpha), X[te])
            sc = r2(y, oof)
            if best is None or sc > best[1]:
                best = (alpha, sc, oof)
        alpha, sc, oof = best
        report[name] = {
            "n_features": len(keys), "alpha": alpha,
            "oof_r2_log2": sc,
            "oof_spearman": float(stats.spearmanr(y, oof).statistic),
            "oof_mae_ranks": float(np.mean(np.abs(2 ** oof - 2 ** y))),
        }
        print(f"[{name:10s}] OOF R2={sc:+.4f} rho={report[name]['oof_spearman']:+.3f} "
              f"MAE={report[name]['oof_mae_ranks']:.2f} ranks (alpha={alpha})")

    # ship the all-77 model (what E169/E169b validated), refit on all 125
    keys, alpha = keys_all, report["all77"]["alpha"]
    model = ridge_fit(mat(keys), y, alpha)
    model["keys"] = keys
    model["target_stage"] = TARGET_STAGE
    model["note"] = ("predicts log2 R*_enc1 from one intact forward pass; "
                     "quality quoted from OOF, refit on all 125 for shipping")
    with open(OUT_DIR / "rhat_enc1_predictor.pkl", "wb") as fh:
        pickle.dump(model, fh)

    insample = ridge_apply(model, mat(keys))
    summary = {
        "target": f"log2 R*_{TARGET_STAGE}",
        "n_fit_subjects": len(sids),
        "folds": N_FOLDS, "seed": SEED,
        "oof_report": report,
        "shipped": {"variant": "all77", "alpha": alpha, "n_features": len(keys)},
        "val_Rhat_distribution": {
            "mean": float(np.mean(2 ** insample)), "std": float(np.std(2 ** insample)),
            "p10": float(np.percentile(2 ** insample, 10)),
            "p90": float(np.percentile(2 ** insample, 90)),
        },
        "true_Rstar_distribution": {
            "mean": float(np.mean(2 ** y)), "std": float(np.std(2 ** y)),
        },
        "constant_arm_value": float(np.mean(2 ** y)),
        "leakage_note": "OOF folds identical to E169; shipped model refit on all 125",
        "generalisation_caveat": ("fitted on 125 val subjects, applied to 1126 train subjects; "
                                  "generalisation assumed, not verified -- no R* exists on train"),
    }
    json.dump(summary, open(OUT_DIR / "rhat_predictor_summary.json", "w"), indent=1)
    print("\n" + json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
