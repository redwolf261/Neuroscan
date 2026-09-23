"""
Shared Lanczos/HVP infrastructure for Phase E42. Verified against direct
computation before use (see verify_lanczos_and_hvp.py, run separately).

HVP: standard Pearlmutter double-backward trick. allow_unused=True is
REQUIRED and CORRECT here, not a workaround for a bug -- aux_head3/aux_head2
architecturally do not participate in L_0 (main loss)'s forward computation
at all (verified directly: model(images) only routes through dec1/seg_head/
evidential_head/boundary_head for L_0's own outputs), so their gradient from
L_0 is exactly, correctly zero, not merely small.
"""
import torch
import numpy as np


def make_flat_params_and_hvp_fn(model, params):
    """params: list of model parameters (order fixed once per model instance)."""

    def hvp(loss_fn, v_list):
        loss = loss_fn()
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
        gv = sum(torch.sum(g * v) for g, v in zip(grads, v_list))
        hvp_list = torch.autograd.grad(gv, params, retain_graph=False, allow_unused=True)
        hvp_list = [h if h is not None else torch.zeros_like(p) for h, p in zip(hvp_list, params)]
        return hvp_list, float(loss.detach().item())

    return hvp


def flatten(tensor_list):
    return torch.cat([t.reshape(-1) for t in tensor_list])


def unflatten_like(flat, like_list):
    out = []
    i = 0
    for t in like_list:
        n = t.numel()
        out.append(flat[i:i + n].reshape(t.shape))
        i += n
    return out


def lanczos_tridiagonalize(hvp_fn, loss_fn, params, n_iter, seed, device, dtype=torch.float32):
    """Standard Lanczos algorithm for a symmetric operator (the Hessian),
    accessed only via matrix-vector products (HVPs). Returns (alphas, betas)
    -- the tridiagonal matrix's diagonal and off-diagonal entries -- from
    which the FULL eigenvalue spectrum of the Krylov-subspace projection of
    H can be recovered via a standard tridiagonal eigensolver (numpy/scipy),
    without ever forming H explicitly. Full reorthogonalization is used
    (not just the 2-term recurrence) since n_iter is small enough (<=40)
    that this is cheap and avoids the well-known Lanczos loss-of-
    orthogonality problem that otherwise corrupts eigenvalue estimates.
    """
    n_params = sum(p.numel() for p in params)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    v0 = torch.randn(n_params, generator=rng).to(device=device, dtype=dtype)
    v0 = v0 / v0.norm()

    Q = [v0]
    alphas = []
    betas = []

    v_prev = torch.zeros_like(v0)
    beta_prev = 0.0

    for i in range(n_iter):
        v_list = unflatten_like(Q[-1], params)
        hv_list, _ = hvp_fn(loss_fn, v_list)
        w = flatten(hv_list)

        alpha = float(torch.dot(w, Q[-1]).item())
        alphas.append(alpha)

        w = w - alpha * Q[-1] - beta_prev * v_prev

        # Full reorthogonalization against all previous Lanczos vectors
        # (not just the last one) -- standard fix for floating-point loss
        # of orthogonality in Lanczos, verified necessary and sufficient at
        # this iteration count in verify_lanczos_and_hvp.py.
        for q in Q:
            w = w - torch.dot(w, q) * q

        beta = float(w.norm().item())
        if beta < 1e-10:
            # Krylov subspace exhausted (invariant subspace found) -- stop early,
            # WITHOUT appending this beta (matches the "n alphas, n-1 betas"
            # tridiagonal convention exactly -- appending it here was the
            # original bug caught by this file's own verification script:
            # eigh_tridiagonal requires len(betas) == len(alphas)-1 exactly,
            # and appending a beta on every iteration (including the last,
            # where there is no (i+1)-th alpha for it to sit beside) violated
            # that by construction, not by an edge case).
            break

        if i < n_iter - 1:
            # Only append beta if there WILL be a next alpha (iteration i+1)
            # for it to pair with in the tridiagonal matrix -- the final
            # iteration's own beta (connecting to a hypothetical iteration
            # n_iter, never computed) must NOT be included.
            betas.append(beta)
            v_prev = Q[-1]
            beta_prev = beta
            Q.append(w / beta)

    return np.array(alphas), np.array(betas)


def eigs_from_tridiagonal(alphas, betas):
    """Builds the tridiagonal matrix from Lanczos coefficients and returns
    its full eigenvalue spectrum (sorted descending) via a standard,
    numerically stable tridiagonal eigensolver."""
    from scipy.linalg import eigh_tridiagonal
    if len(betas) == 0:
        return np.array(alphas)
    eigs, _ = eigh_tridiagonal(alphas, betas)
    return np.sort(eigs)[::-1]
