"""
Phase E104b: S <-> N_b Bridge Experiment.

CONTEXT: E104 found this project's existing evidential head is correctly
calibrated (S=alpha+beta is significantly lower at missed small-lesion
voxels than at background). This leaves open whether the model's
uncertainty signal S already implicitly tracks E48's causal necessity
N_b, or whether the two are distinct properties. User's explicit
requirement: subject-level analysis (not pooled voxels), Spearman +
permutation (not just Pearson), AND a multiple regression of N_b on S
plus native_size plus baseline difficulty, to check whether S retains
explanatory power beyond known confounders -- not just a raw
correlation.

DEFINITION: for each small-lesion subject (same 63-subject set as E102/
E104), define S(x) as the mean total evidence (alpha+beta) specifically
over LESION-RELEVANT voxels (the union of missed-lesion (FN) and
correctly-detected (TP) voxels -- i.e. all ground-truth-foreground
voxels), NOT the whole-volume mean (which would be dominated by
background and dilute any lesion-specific signal). N_b(x) is E48's
existing per-subject causal necessity (bottleneck ablation Dice drop).
Baseline difficulty is operationalized as the subject's OWN dice_intact
(already computed in E104's per-subject table via the E48-lineage
checkpoint's standard forward pass).

THIS IS A ZERO-TRAINING, ZERO-NEW-FORWARD-PASS ANALYSIS -- reuses
E48's existing table and E104's already-saved per-subject S values
directly, no new computation on the checkpoint needed.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

E48_TABLE_PATH = Path(__file__).parent.parent / "e48" / "E48_encoding_audit_table.json"
E104_TABLE_PATH = Path(__file__).parent / "E104_per_subject_table.json"
OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 2000


def main():
    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}

    with open(E104_TABLE_PATH) as f:
        e104_records = json.load(f)

    records = []
    for r in e104_records:
        sid = r["subject_id"]
        if sid not in e48_by_id:
            continue
        n_fn, n_tp = r["n_fn"], r["n_tp"]
        s_fn, s_tp = r["mean_S_fn"], r["mean_S_tp"]
        if n_fn + n_tp == 0:
            continue
        # Voxel-count-weighted mean over ALL lesion-relevant (FN + TP) voxels --
        # this is S(x) as specified: mean evidence over ground-truth-foreground voxels.
        s_lesion = (s_fn * n_fn + (s_tp if s_tp is not None else 0.0) * n_tp) / (n_fn + n_tp)

        records.append({
            "subject_id": sid,
            "S_lesion": s_lesion,
            "N_b": e48_by_id[sid]["drop"],
            "native_size": e48_by_id[sid]["native_size"],
            "dice_intact": e48_by_id[sid]["dice_intact"],
        })

    n = len(records)
    print(f"n subjects with valid S_lesion and N_b: {n}")

    s_lesion = np.array([r["S_lesion"] for r in records])
    n_b = np.array([r["N_b"] for r in records])
    native_size = np.array([r["native_size"] for r in records])
    dice_intact = np.array([r["dice_intact"] for r in records])

    print(f"\n=== E104b S <-> N_b Bridge: n={n} small-lesion subjects ===")
    print(f"S_lesion: mean={s_lesion.mean():.4f}, sd={s_lesion.std():.4f}, "
          f"range=[{s_lesion.min():.4f}, {s_lesion.max():.4f}]")
    print(f"N_b: mean={n_b.mean():.4f}, sd={n_b.std():.4f}")

    # ================= Spearman + permutation =================
    rho, p_param = stats.spearmanr(s_lesion, n_b)
    rng = np.random.default_rng(SEED)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_rhos[i], _ = stats.spearmanr(s_lesion, rng.permutation(n_b))
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())

    # Bootstrap CI on rho (percentile method, 2000 resamples)
    boot_rhos = np.empty(N_PERM)
    rng2 = np.random.default_rng(SEED + 1)
    for i in range(N_PERM):
        idx = rng2.integers(0, n, size=n)
        boot_rhos[i], _ = stats.spearmanr(s_lesion[idx], n_b[idx])
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

    print(f"\nSpearman(S_lesion, N_b) = {rho:+.4f}")
    print(f"Permutation p = {p_perm:.4f}")
    print(f"95% bootstrap CI on rho: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # ================= Multiple regression: N_b ~ S + size + dice_intact =================
    # Rank-based (Spearman-consistent), matching this project's own established
    # partial-correlation convention from E85/E95/E96.
    X = np.column_stack([
        stats.rankdata(s_lesion), stats.rankdata(native_size), stats.rankdata(dice_intact),
    ])
    X1 = np.column_stack([np.ones(n), X])
    y = stats.rankdata(n_b)
    beta, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
    y_hat = X1 @ beta
    resid = y - y_hat
    k = X1.shape[1]
    sigma2 = np.sum(resid ** 2) / (n - k)
    cov_beta = sigma2 * np.linalg.inv(X1.T @ X1)
    se_beta = np.sqrt(np.diag(cov_beta))
    t_stats = beta / se_beta
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

    r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)

    print(f"\n=== Multiple regression (rank-based): N_b ~ S_lesion + native_size + dice_intact ===")
    names = ["intercept", "S_lesion", "native_size", "dice_intact"]
    for name, b, p in zip(names, beta, p_vals):
        print(f"  {name:15s}: coef={b:+.4f}, p={p:.4f}")
    print(f"  R^2 = {r2:.4f}")

    s_retains_power = p_vals[1] < 0.05

    # ================= Interpretation =================
    if abs(rho) < 0.2 or p_perm >= 0.05:
        interpretation = "DISTINCT_SIGNALS"
        detail = ("S and N_b show weak/non-significant correlation -- epistemic uncertainty and causal "
                   "pathway necessity appear to be DISTINCT properties. A combined signal f(S, N_b) may "
                   "carry information neither provides alone -- worth a novelty audit before any design.")
    elif rho > 0.2 and s_retains_power:
        interpretation = "S_IMPLICIT_PROXY_FOR_NB"
        detail = ("S correlates with N_b AND retains explanatory power after controlling for native_size "
                   "and dice_intact -- the model's existing (free, already-computed) uncertainty signal is "
                   "an implicit proxy for causal necessity, beyond what size/difficulty alone would predict. "
                   "Worth investigating whether S could substitute for expensive N_b measurement in a "
                   "training-time mechanism -- pending its own novelty audit.")
    elif rho > 0.2 and not s_retains_power:
        interpretation = "CORRELATION_EXPLAINED_BY_CONFOUNDS"
        detail = ("S correlates with N_b in the raw comparison, but this is fully explained by native_size "
                   "and/or dice_intact -- S does not carry INCREMENTAL information about causal necessity "
                   "beyond what's already known from size/difficulty. Not a useful bridge on its own.")
    elif rho < -0.2 and p_perm < 0.05:
        interpretation = "NEGATIVE_RELATIONSHIP"
        detail = ("S and N_b are significantly NEGATIVELY correlated -- higher uncertainty corresponds to "
                   "LOWER causal necessity, an unexpected direction requiring its own investigation before "
                   "any design.")
    else:
        interpretation = "AMBIGUOUS"
        detail = "Result does not cleanly match a pre-declared interpretation category."

    print(f"\n=== INTERPRETATION: {interpretation} ===")
    print(detail)

    summary = {
        "n_subjects": n,
        "spearman_rho": float(rho), "permutation_p": p_perm,
        "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
        "regression_coefficients": dict(zip(names, beta.tolist())),
        "regression_p_values": dict(zip(names, p_vals.tolist())),
        "regression_r2": float(r2),
        "s_retains_explanatory_power": bool(s_retains_power),
        "interpretation": interpretation,
    }
    with open(OUT_DIR / "E104b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E104b_summary.json")


if __name__ == "__main__":
    main()
