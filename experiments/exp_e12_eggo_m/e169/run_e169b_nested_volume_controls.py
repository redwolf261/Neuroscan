"""
E169b -- Nested volume controls: what did E169 actually measure?

E169 found R* substantially predictable from ONE intact forward pass
(enc3 R^2=0.789, enc2 0.734, enc1 0.634), surviving a GROUND-TRUTH lesion
volume control (77-97% retained). But the strongest single predictors were
enc3.energy_std and pred.logvol_ET (rho=-0.806 at enc2), and pred.vol_*
correlates with R* MORE strongly than GT size does (-0.81 vs -0.43). So the
GT control likely UNDER-REMOVES the size pathway.

THE QUESTION: is R*'s predictability mediated by the model's own estimate of
lesion extent, or does it contain information beyond lesion size?

FOUR NESTED MODELS. Everything else identical to E169 -- same 125 subjects,
same seeded by-subject folds, same 77-feature pool, same target log2 R*,
same cross-fitted ridge, same within-fold permutation null.

  A  no volume control                 raw predictability
  B  GT volumes (log1p ET/TC/WT)       survives true lesion size?
  C  PREDICTED volumes                 survives the model's own size estimate?
  D  GT + predicted volumes            anything beyond BOTH size pathways?

CRITICAL DESIGN POINT. pred.vol_* / pred.logvol_* are themselves among the 77
predictors. Residualising the TARGET on them while leaving them in the FEATURE
set would let the model route around the control. So in models C and D the
entire `pred.*` feature block is REMOVED from X, not merely residualised.
Model B keeps them (matching E169 exactly, so B reproduces E169's number and
serves as a correctness check on this script).

PRE-REGISTERED READING (fixed before running):
  If C collapses to <=0.10            => E169 was largely a model-derived
                                         lesion-size surrogate.
  If C retains roughly 0.4-0.5        => task-required rank carries information
                                         BEYOND lesion extent.
  If D also retains substantially     => the strongest form of that claim.

VERDICT STAYS PARK REGARDLESS. This determines what E169 measured; it does not
reopen the allocation branch, which is closed by E71 (sign -0.290 backwards),
E147 (inverse, no scarcity: 74/88 at rank<=4 of 256) and E72/E73 (inert).
Predicting a quantity does not create a resource to allocate.

WORDING (corrected): do NOT write "R* is a cheap statistic" -- the predictor
still requires intact model inference plus feature extraction. The demonstrated
saving is exactly:
    9-point causal truncation sweep  ->  1 intact forward pass + predictor
"""
import sys
import json
import csv
from pathlib import Path

import numpy as np
from scipy import stats

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

OUT_DIR = Path(__file__).parent
FEATS = OUT_DIR / "E169_features.json"
E165_TABLE = project_root / "experiments" / "exp_e12_eggo_m" / "e165" / "E165_per_subject.json"
E160_TABLE = project_root / "experiments" / "exp_e12_eggo_m" / "e160" / "E160_L_per_subject.json"

STAGES = {"enc1": 32, "enc2": 64, "enc3": 128, "bottleneck": 256, "dec1": 32}
REGIONS = ("ET", "TC", "WT")
N_FOLDS = 5
SEED = 0
N_PERM = 1000


def ridge_cv(X, y, folds, alphas=(0.1, 1.0, 10.0, 100.0)):
    """Cross-fitted ridge, inner alpha selection inside each training fold."""
    pred = np.zeros_like(y, dtype=float)
    for f in range(folds.max() + 1):
        tr, te = folds != f, folds == f
        Xtr, ytr = X[tr], y[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr_s, Xte_s = (Xtr - mu) / sd, (X[te] - mu) / sd
        ym = ytr.mean()
        best, best_a = np.inf, alphas[0]
        inner = np.arange(len(ytr)) % 3
        for a in alphas:
            errs = []
            for g in range(3):
                itr, ite = inner != g, inner == g
                A = Xtr_s[itr].T @ Xtr_s[itr] + a * np.eye(X.shape[1])
                w = np.linalg.solve(A, Xtr_s[itr].T @ (ytr[itr] - ytr[itr].mean()))
                errs.append(np.mean((Xtr_s[ite] @ w + ytr[itr].mean() - ytr[ite]) ** 2))
            if np.mean(errs) < best:
                best, best_a = np.mean(errs), a
        A = Xtr_s.T @ Xtr_s + best_a * np.eye(X.shape[1])
        w = np.linalg.solve(A, Xtr_s.T @ (ytr - ym))
        pred[te] = Xte_s @ w + ym
    return pred


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))


def partial_out(M, Z):
    """Residualise columns of M on Z (intercept included)."""
    if Z is None or Z.shape[1] == 0:
        return M
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(Z1, M, rcond=None)
    return M - Z1 @ beta


def main():
    rows = json.load(open(FEATS))
    e165 = {r["sid"]: r for r in json.load(open(E165_TABLE))}
    e160 = {r["sid"]: r for r in json.load(open(E160_TABLE))}
    rows = [r for r in rows if r["sid"] in e165 and r["sid"] in e160]
    sids = [r["sid"] for r in rows]
    print(f"[Data] {len(sids)} subjects")

    all_keys = sorted(k for k in rows[0] if k != "sid")
    pred_keys = [k for k in all_keys if k.startswith("pred.")]
    act_keys = [k for k in all_keys if not k.startswith("pred.")]
    print(f"[Features] {len(all_keys)} total = {len(act_keys)} activation + {len(pred_keys)} prediction-derived")

    def mat(keys):
        M = np.array([[r[k] for k in keys] for r in rows], float)
        return np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    X_all, X_act = mat(all_keys), mat(act_keys)
    Z_gt = np.array([[np.log1p(e160[s][f"size_{n}"]) for n in REGIONS] for s in sids], float)
    Z_pred = mat([f"pred.logvol_{n}" for n in REGIONS])

    # sanity: how strongly do the two size proxies agree?
    agree = [float(stats.spearmanr(Z_gt[:, i], Z_pred[:, i]).statistic) for i in range(3)]
    print(f"[Sanity] rho(GT vol, predicted vol) per region: "
          f"{dict(zip(REGIONS, [round(a,3) for a in agree]))}")

    rng = np.random.default_rng(SEED)
    folds = rng.permutation(len(sids)) % N_FOLDS      # identical construction to E169

    MODELS = {
        "A_raw":        dict(X=X_all, Z=None,
                             note="no control; all 77 features"),
        "B_gt":         dict(X=X_all, Z=Z_gt,
                             note="GT volume control; all 77 features (reproduces E169)"),
        "C_predvol":    dict(X=X_act, Z=Z_pred,
                             note="PREDICTED volume control; pred.* REMOVED from X"),
        "D_both":       dict(X=X_act, Z=np.column_stack([Z_gt, Z_pred]),
                             note="GT + predicted volume control; pred.* REMOVED from X"),
    }

    results = {}
    for stage, C in STAGES.items():
        y_raw = np.array([e165[s]["stages"][stage]["R_star"] for s in sids], float)
        y0 = np.log2(y_raw)
        if y0.std() < 1e-9:
            results[stage] = {"note": "constant target -- ill-posed"}
            continue
        entry = {"C": C}
        for mname, cfg in MODELS.items():
            X, Z = cfg["X"], cfg["Z"]
            Xc = partial_out(X, Z)
            yc = partial_out(y0.reshape(-1, 1), Z).ravel()
            pred = ridge_cv(Xc, yc, folds)
            val = r2(yc, pred)
            # within-fold permutation null
            null = []
            for _ in range(N_PERM):
                ysh = yc.copy()
                for f in range(N_FOLDS):
                    m = folds == f
                    ysh[m] = rng.permutation(ysh[m])
                null.append(r2(ysh, ridge_cv(Xc, ysh, folds)))
            null = np.array(null)
            entry[mname] = {
                "r2_oos": val,
                "spearman_oos": float(stats.spearmanr(yc, pred).statistic),
                "permutation_p": float((null >= val).mean()),
                "null_mean": float(null.mean()),
                "n_features": X.shape[1],
            }
        a = entry["A_raw"]["r2_oos"]
        entry["retained_vs_raw"] = {m: (entry[m]["r2_oos"] / a if a > 0 else 0.0)
                                    for m in MODELS}
        results[stage] = entry
        print(f"  {stage:11s} A={entry['A_raw']['r2_oos']:+.4f} "
              f"B={entry['B_gt']['r2_oos']:+.4f} "
              f"C={entry['C_predvol']['r2_oos']:+.4f} "
              f"D={entry['D_both']['r2_oos']:+.4f}", flush=True)

    scored = {k: v for k, v in results.items() if "A_raw" in v}
    bestC = max(v["C_predvol"]["r2_oos"] for v in scored.values())
    bestD = max(v["D_both"]["r2_oos"] for v in scored.values())
    if bestC <= 0.10:
        reading = "SIZE_SURROGATE: E169 was largely a model-derived lesion-size surrogate"
    elif bestC >= 0.40:
        reading = "BEYOND_SIZE: task-required rank carries information beyond lesion extent"
    else:
        reading = "PARTIAL: some signal beyond size, but substantially size-mediated"

    summary = {
        "question": "Is R*'s predictability mediated by the model's own lesion-extent estimate?",
        "n_subjects": len(sids), "n_folds": N_FOLDS, "n_permutations": N_PERM,
        "models": {k: v["note"] for k, v in MODELS.items()},
        "design_note": "pred.* features are REMOVED from X in models C and D, not merely "
                       "residualised -- otherwise the model routes around its own control.",
        "gt_vs_pred_volume_agreement": dict(zip(REGIONS, agree)),
        "per_stage": results,
        "best_C_r2": bestC, "best_D_r2": bestD,
        "READING": reading,
        "verdict": "PARK regardless -- this determines what E169 measured; it does not "
                   "reopen the allocation branch (E71 sign -0.290, E147 inverse/no scarcity, "
                   "E72/E73 inert).",
        "wording": "Do NOT say 'R* is a cheap statistic'. The demonstrated saving is: "
                   "9-point causal truncation sweep -> 1 intact forward pass + predictor.",
    }

    with open(OUT_DIR / "E169b_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    with open(OUT_DIR / "E169b_nested.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "C", "A_raw", "B_gt", "C_predvol", "D_both",
                    "retain_C", "retain_D", "p_C", "p_D"])
        for s, v in results.items():
            if "A_raw" not in v:
                continue
            w.writerow([s, v["C"], round(v["A_raw"]["r2_oos"], 4), round(v["B_gt"]["r2_oos"], 4),
                        round(v["C_predvol"]["r2_oos"], 4), round(v["D_both"]["r2_oos"], 4),
                        round(v["retained_vs_raw"]["C_predvol"], 3),
                        round(v["retained_vs_raw"]["D_both"], 3),
                        v["C_predvol"]["permutation_p"], v["D_both"]["permutation_p"]])

    print("\n" + "=" * 70)
    print(json.dumps(summary, indent=1))
    print("=" * 70)


if __name__ == "__main__":
    main()
