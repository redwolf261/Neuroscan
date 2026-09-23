"""
E180 Stages 6-9 -- shared subject-aware statistics.

Pre-registered in docs/phases/PHASE_E180_6_9_ANALYSIS_PREREG.md. Read that
first. Central discipline: 868 tiles across 125 subjects are NOT independent
observations (adjacent tiles share 83-94% enc3 extent, measured directly in
Stage 5.5). Every regression here is subject-aware (fixed effects / within-
subject demeaning with cluster-robust SEs); pooled tile-level statistics are
computed only as explicitly-labelled descriptive context, never as evidence.
"""
import numpy as np
from scipy import stats


def subject_fixed_effects_ols(y, x, subject_ids):
    """y ~ x + subject dummies (no separate intercept -- dummies span it).
    Returns dict with beta_x, se (cluster-robust by subject), t, p, r2, dof,
    n, n_subjects. x may be a 1D array (single predictor) or 2D (n, k)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    subs = sorted(set(subject_ids))
    sub_idx = {s: i for i, s in enumerate(subs)}
    D = np.zeros((len(y), len(subs)))
    for i, s in enumerate(subject_ids):
        D[i, sub_idx[s]] = 1.0
    X = np.column_stack([x, D])
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    n, k = X.shape
    dof = max(n - k, 1)

    # cluster-robust (by subject) sandwich covariance for the x-coefficients
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for s in subs:
        idx = [i for i, si in enumerate(subject_ids) if si == s]
        Xg = X[idx]
        ug = resid[idx]
        score = Xg.T @ ug
        meat += np.outer(score, score)
    n_clusters = len(subs)
    cluster_correction = n_clusters / max(n_clusters - 1, 1)
    cov = cluster_correction * (XtX_inv @ meat @ XtX_inv)

    n_x = x.shape[1]
    se_x = np.sqrt(np.clip(np.diag(cov)[:n_x], 0, None))
    beta_x = beta[:n_x]
    t = np.divide(beta_x, se_x, out=np.full_like(beta_x, np.nan), where=se_x > 0)
    p = 2 * (1 - stats.t.cdf(np.abs(t), max(n_clusters - 1, 1)))

    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "beta": beta_x.tolist(), "se_cluster_robust": se_x.tolist(),
        "t": t.tolist(), "p": p.tolist(),
        "r2": r2, "n": int(n), "n_subjects": int(n_clusters), "dof": int(dof),
    }


def within_subject_demean(y, x, subject_ids):
    """Return (y - subject mean, x - subject mean) -- equivalent transform to
    the fixed-effects regression above, useful for a quick paired look."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    subs = np.array(subject_ids)
    y_dm = y.copy()
    x_dm = x.copy()
    for s in set(subject_ids):
        idx = subs == s
        y_dm[idx] = y[idx] - y[idx].mean()
        x_dm[idx] = x[idx] - x[idx].mean()
    return y_dm, x_dm


def within_subject_permutation_null(y, x_matrix, subject_ids, gamma_col_idx,
                                     n_perm=1000, seed=0):
    """Permutation null for the incremental-R2 of adding column gamma_col_idx
    to x_matrix, respecting subject structure: gamma is shuffled WITHIN each
    subject's own tiles only (preserves the subject-level marginal), never
    shuffled across the full 2604 rows as IID."""
    rng = np.random.default_rng(seed)
    x_matrix = np.asarray(x_matrix, dtype=float)
    y = np.asarray(y, dtype=float)
    subs = np.array(subject_ids)

    base_cols = [i for i in range(x_matrix.shape[1]) if i != gamma_col_idx]
    x_base = x_matrix[:, base_cols] if base_cols else np.ones((len(y), 1))
    r2_base = subject_fixed_effects_ols(y, x_base, subject_ids)["r2"]
    r2_full = subject_fixed_effects_ols(y, x_matrix, subject_ids)["r2"]
    observed_delta = r2_full - r2_base

    gamma_col = x_matrix[:, gamma_col_idx].copy()
    null_deltas = []
    for _ in range(n_perm):
        shuffled = gamma_col.copy()
        for s in set(subject_ids):
            idx = np.where(subs == s)[0]
            shuffled[idx] = rng.permutation(shuffled[idx])
        x_perm = x_matrix.copy()
        x_perm[:, gamma_col_idx] = shuffled
        r2_perm_full = subject_fixed_effects_ols(y, x_perm, subject_ids)["r2"]
        null_deltas.append(r2_perm_full - r2_base)
    null_deltas = np.array(null_deltas)
    p = float((null_deltas >= observed_delta).mean())

    return {
        "r2_base": r2_base, "r2_full": r2_full,
        "observed_delta_r2": observed_delta,
        "null_mean": float(null_deltas.mean()), "null_std": float(null_deltas.std()),
        "permutation_p": p, "n_perm": n_perm,
    }


def reduced_granularity_pairs(ledger):
    """For each subject, the tile pair with maximum enc3-center L2 distance --
    the FIXED criterion locked in Stage 1's amendment, reused verbatim."""
    def enc3_box(t):
        return (t["z0"] // 4, t["y0"] // 4, t["x0"] // 4,
                t["z1"] // 4, t["y1"] // 4, t["x1"] // 4)

    def box_center(t):
        z0, y0, x0, z1, y1, x1 = enc3_box(t)
        return np.array([(z0 + z1) / 2, (y0 + y1) / 2, (x0 + x1) / 2])

    pairs = {}
    for sid, v in ledger.items():
        tiles = v["tiles"]
        if len(tiles) < 2:
            continue
        centers = [box_center(t) for t in tiles]
        best_pair, best_dist = None, -1
        for i in range(len(tiles)):
            for j in range(i + 1, len(tiles)):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist > best_dist:
                    best_dist = dist
                    best_pair = (tiles[i]["window_id"], tiles[j]["window_id"])
        pairs[sid] = best_pair
    return pairs


def sign_concordance_test(g_by_key, d_by_key, reduced_pairs, family, metric="gamma_dice"):
    """Exact binomial sign test on the reduced-granularity pairs -- a
    robustness diagnostic, never the decisive gate (per the explicit lock)."""
    signs = []
    detail = []
    for sid, pair in reduced_pairs.items():
        k0, k1 = (sid, pair[0], family), (sid, pair[1], family)
        if k0 not in g_by_key or k1 not in g_by_key:
            continue
        if k0 not in d_by_key or k1 not in d_by_key:
            continue
        dg = g_by_key[k1][metric] - g_by_key[k0][metric]
        dd = d_by_key[k1]["delta_i_global"] - d_by_key[k0]["delta_i_global"]
        same = bool(np.sign(dg) == np.sign(dd))
        signs.append(same)
        detail.append({"sid": sid, "d_gamma": float(dg), "d_delta": float(dd), "concordant": same})
    n_concord = sum(signs)
    n_total = len(signs)
    p = stats.binomtest(n_concord, n_total, 0.5).pvalue if n_total > 0 else float("nan")
    return {"n_concordant": n_concord, "n_total": n_total, "binomial_p": float(p),
            "detail": detail}
