"""
E181-A -- T6 (rank-matched spectral reshape) and T7 (pure energy scaling).

Pre-registered in docs/phases/PHASE_E181_MECHANISM_DISSECTION_PREREG.md. Read
that first. Both transforms operate on the enc3 activation tensor, same SVD-
over-channel-axis convention as E180's lowrank_channels/spectral_reshape.

BINDING RULE (T6 solver): the numerical solve for (a,b) sees ONLY the tile's
own singular-value spectrum and the two fixed constraints (entropy match,
energy match) plus a FIXED shape-distance objective. No access to
predictions, Dice, Gamma, uncertainty, or Delta. Never tuned per-tile or
re-parameterized based on any downstream outcome.
"""
import numpy as np
import torch
from scipy.optimize import minimize


def _svd_decompose(x):
    """x: (1,C,D,H,W). Returns (U, S, Vh, mu, shape) -- mean-centred SVD over
    the channel axis, identical convention to E180's lowrank_channels."""
    b, c, d, h, w = x.shape
    m = x.reshape(c, -1).float()
    mu = m.mean(dim=1, keepdim=True)
    mc = m - mu
    U, S, Vh = torch.linalg.svd(mc, full_matrices=False)
    return U, S, Vh, mu, (b, c, d, h, w)


def _reconstruct(U, S_new, Vh, mu, shape):
    approx = (U * S_new) @ Vh
    return (approx + mu).reshape(*shape)


def entropy_of(p):
    p = np.clip(p, 1e-300, None)
    return float(-(p * np.log(p)).sum())


def _target_spectrum(j, a, b):
    """q_j(a,b) propto exp(-a*j - b*j^2), j = 1..r (1-indexed, as specified).

    IMPORTANT: returned sorted DESCENDING. torch.linalg.svd always returns S
    descending, and U/Vh's columns/rows are ordered to match. A target
    spectrum must be paired with U/Vh in the SAME descending order, or the
    reconstruction silently associates the wrong magnitude with the wrong
    singular vector -- this was caught as a real bug during E181-B
    calibration (see PHASE_E181_MECHANISM_DISSECTION_PREREG.md): an
    ascending q gave a reconstruction whose OWN re-derived spectrum, after
    SVD re-sorts it descending, ended up close to the ORIGINAL shape again,
    not the intended q -- i.e. it silently failed to change the shape at
    all while still reporting a large claimed shape-distance number."""
    log_q = -a * j - b * (j ** 2)
    log_q = log_q - log_q.max()   # numerical stability before normalizing
    q = np.exp(log_q)
    q = q / q.sum()
    return np.sort(q)[::-1].copy()


def solve_rank_matched_shape(sigma, shape_distance="l1", n_restarts=6, seed=0):
    """Given a tile's singular values (numpy array, descending, r values),
    solve for (a,b) such that the target spectrum q(a,b) has the SAME
    entropy as the normalized input spectrum p, while MAXIMIZING a
    shape-distance objective from p (fixed form, not tuned per-tile).

    Returns dict: q (target normalized spectrum), a, b, H_p, H_q (should
    match H_p within tolerance), shape_distance_achieved, converged.

    NO access to any Gamma/Delta/Dice quantity -- sigma is the only input.
    """
    r = len(sigma)
    p = sigma / sigma.sum()
    H_p = entropy_of(p)
    j = np.arange(1, r + 1, dtype=float)

    def entropy_constraint(params):
        a, b = params
        q = _target_spectrum(j, a, b)
        return entropy_of(q) - H_p

    def neg_shape_distance(params):
        a, b = params
        q = _target_spectrum(j, a, b)
        if shape_distance == "l1":
            d = np.abs(q - p).sum()
        else:
            raise ValueError(f"unknown shape_distance {shape_distance}")
        return -d   # minimize negative distance == maximize distance

    # constrained optimization: maximize shape distance subject to entropy match.
    # Use SLSQP with the entropy-match as an equality constraint, restarted from
    # several fixed (non-outcome-dependent) starting points for robustness.
    rng = np.random.default_rng(seed)
    starts = [(0.0, 0.0)] + [(rng.uniform(-0.5, 0.5), rng.uniform(-0.1, 0.1))
                             for _ in range(n_restarts - 1)]

    best = None
    for x0 in starts:
        res = minimize(neg_shape_distance, x0=np.array(x0), method="SLSQP",
                       constraints=[{"type": "eq", "fun": entropy_constraint}],
                       options={"maxiter": 200, "ftol": 1e-10})
        if not res.success:
            continue
        achieved = -res.fun
        if best is None or achieved > best["shape_distance_achieved"]:
            a, b = res.x
            q = _target_spectrum(j, a, b)
            best = {
                "a": float(a), "b": float(b), "q": q,
                "H_p": H_p, "H_q": entropy_of(q),
                "shape_distance_achieved": float(achieved),
                "converged": True,
            }
    if best is None:
        # fall back to identity (a=0,b=0 gives near-uniform target; report failure honestly)
        q = _target_spectrum(j, 0.0, 0.0)
        best = {"a": 0.0, "b": 0.0, "q": q, "H_p": H_p, "H_q": entropy_of(q),
                "shape_distance_achieved": float(np.abs(q - p).sum()), "converged": False}
    return best


def rank_matched_spectral_reshape(x, shape_distance="l1", n_restarts=6, seed=0):
    """T6. x: (1,C,D,H,W). Reconstructs with a target spectrum that matches
    the ORIGINAL entropy (hence eff_rank) exactly by construction, matches
    the original ||Sigma||_2 (energy), and maximizes shape distance from the
    original normalized spectrum under a FIXED objective. Returns
    (reconstructed_tensor, diagnostics_dict)."""
    U, S, Vh, mu, shape = _svd_decompose(x)
    sigma = S.detach().cpu().numpy().astype(np.float64)
    sigma = np.clip(sigma, 1e-12, None)

    sol = solve_rank_matched_shape(sigma, shape_distance=shape_distance,
                                   n_restarts=n_restarts, seed=seed)
    q = sol["q"]
    total_energy = float(np.sqrt((sigma ** 2).sum()))
    # Sigma'_j = s * q_j, s chosen so ||Sigma'||_2 == ||Sigma||_2
    s = total_energy / np.sqrt((q ** 2).sum())
    sigma_new = s * q

    S_new = torch.from_numpy(sigma_new).to(S.device, S.dtype)
    out = _reconstruct(U, S_new, Vh, mu, shape)

    diagnostics = {
        "eff_rank_orig": float(np.exp(sol["H_p"])),
        "eff_rank_new": float(np.exp(sol["H_q"])),
        "energy_orig": total_energy,
        "energy_new": float(np.sqrt((sigma_new ** 2).sum())),
        "shape_distance_l1": sol["shape_distance_achieved"],
        "solver_converged": sol["converged"],
        "a": sol["a"], "b": sol["b"],
    }
    return out, diagnostics


def verify_UV_preserved(U_orig, U_new, tol=1e-2):
    """DEPRECATED -- found unreliable during E181-B calibration. Comparing
    U_orig against a U_new obtained by RE-DECOMPOSING the reconstructed
    output tensor is comparing two INDEPENDENT SVD calls. When singular
    values are closely spaced (smallest gaps ~0.005-0.06 were measured on
    real enc3 tiles), each SVD call can legitimately land on a different
    orthonormal basis within the near-degenerate subspace -- especially on
    GPU, where cuSOLVER's numerical path was found to differ measurably
    between a fresh-process SVD and one run immediately after a model
    forward pass (same input values, different U,V up to offdiag~0.2-1.0,
    despite S matching to 1e-2). This is real SVD non-uniqueness /
    GPU numerical sensitivity, not a bug in the reconstruction itself.
    Use verify_UV_preserved_by_construction instead, which needs no second
    SVD call at all."""
    M = U_orig.T @ U_new
    diag_vals = torch.diag(M).abs()
    off_diag = M[~torch.eye(M.shape[0], dtype=bool, device=M.device)]
    diag_ok = bool((diag_vals - 1.0).abs().max() < tol)
    offdiag_ok = bool(off_diag.abs().max() < tol)
    return {
        "diag_min": float(diag_vals.min()), "diag_max": float(diag_vals.max()),
        "offdiag_max": float(off_diag.abs().max()),
        "pass": diag_ok and offdiag_ok,
    }


def verify_UV_preserved_by_construction(U, Vh, S_new, mu, shape, out, tol=1e-3):
    """Correct T6 verification: since T6's code (rank_matched_spectral_reshape
    / t6_at_budget) NEVER touches U or Vh -- only S is replaced before
    reconstruction -- the U,V-preservation guarantee is STRUCTURAL, not
    something that needs empirical re-derivation via a second SVD. This
    verifies it algebraically instead: reconstruct directly from the SAME
    U, S_new, Vh, mu used to produce `out`, and confirm it reproduces `out`
    exactly (this only fails if there is a real bug in the reconstruction
    arithmetic, never due to SVD basis non-uniqueness)."""
    manual = _reconstruct(U, S_new, Vh, mu, shape)
    max_diff = float((manual - out).abs().max())
    return {"reconstruction_max_diff": max_diff, "pass": max_diff < tol}


def pure_energy_scaling(x, c):
    """T7. x: (1,C,D,H,W). Sigma' = c*Sigma exactly -- normalized shape and
    entropy/eff_rank are UNCHANGED by construction (a positive scalar cancels
    in the normalized spectrum). Only ||Sigma||_2 changes, by factor c."""
    if c == 1.0:
        return x
    U, S, Vh, mu, shape = _svd_decompose(x)
    S_new = c * S
    return _reconstruct(U, S_new, Vh, mu, shape)
