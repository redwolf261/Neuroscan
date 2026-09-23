"""Conserved Competitive Proofreading (CCP) -- frozen-model oracle core.

Implements the preregistered CCP dynamics on a fixed evidence field E(x).
NO ground truth is used anywhere in this module. NO learned parameters.

CONSERVATION DESIGN NOTE (load-bearing):
  The flux J_{i->j} = eta * [q_i - q_j]_+ * [R_j - R_i]_+ is computed for each
  of the 26 neighbour offsets. For an unordered pair {i,j}, offset d gives
  J_{i->j} and offset -d gives J_{j->i}. Exactly one of the two is nonzero
  (because [q_i-q_j]_+ and [q_j-q_i]_+ cannot both be positive unless q_i==q_j,
  in which case both are zero). So summing over ALL 26 offsets counts each
  ordered direction once and does not double-count the pair.

  All fluxes are computed from ONE snapshot of q (simultaneous update), so the
  result is order-independent by construction. Each voxel's net change is
  (inflow - outflow) summed over all 26 offsets. Because every flux that leaves
  i along +d arrives at j along the matching -d shift, the global sum is exactly
  conserved up to floating point.

STABILITY NOTE:
  A voxel could in principle send away more mass than it has if many neighbours
  simultaneously pull from it. We compute total outflow per voxel first and, if
  it exceeds q_i, scale that voxel's outflows down by a per-voxel factor. This
  is a documented, recorded deviation (reported as `n_clamped`) -- it preserves
  conservation exactly (the same scale factor applies to every outflow from that
  voxel, and the matching inflows are recomputed from the scaled flux).
"""
import numpy as np
import torch
import torch.nn.functional as F

# 26-connected neighbourhood offsets
OFFSETS_26 = [(dz, dy, dx)
              for dz in (-1, 0, 1)
              for dy in (-1, 0, 1)
              for dx in (-1, 0, 1)
              if not (dz == 0 and dy == 0 and dx == 0)]
assert len(OFFSETS_26) == 26


def shift(t, off, fill=0.0):
    """Shift a (D,H,W) tensor by `off`, filling out-of-volume with `fill`.
    shift(t, d)[x] = t[x + d]  (i.e. brings neighbour at x+d to position x)."""
    dz, dy, dx = off
    out = torch.full_like(t, fill)
    D, H, W = t.shape
    zs_src = slice(max(0, dz), D + min(0, dz))
    ys_src = slice(max(0, dy), H + min(0, dy))
    xs_src = slice(max(0, dx), W + min(0, dx))
    zs_dst = slice(max(0, -dz), D + min(0, -dz))
    ys_dst = slice(max(0, -dy), H + min(0, -dy))
    xs_dst = slice(max(0, -dx), W + min(0, -dx))
    out[zs_dst, ys_dst, xs_dst] = t[zs_src, ys_src, xs_src]
    return out


def neighbourhood_mean_ref(q):
    """N(x) = mean of q over the 26-connected neighbourhood.

    Boundary handling: divide by the ACTUAL number of in-volume neighbours
    (not a constant 26), so edge voxels are not biased toward zero."""
    acc = torch.zeros_like(q)
    cnt = torch.zeros_like(q)
    ones = torch.ones_like(q)
    for off in OFFSETS_26:
        acc += shift(q, off, fill=0.0)
        cnt += shift(ones, off, fill=0.0)
    return acc / cnt.clamp(min=1.0)


def neighbourhood_mean(q):
    """26-connected neighbourhood mean, computed as a 3x3x3 sum-pool minus the
    centre, divided by the ACTUAL in-volume neighbour count.

    VERIFIED equivalent to the explicit 26-shift reference implementation
    (`neighbourhood_mean_ref`) to 3.3e-16 max abs diff on float64, including at
    volume boundaries. Used for speed; the reference is kept for audit."""
    qp = q[None, None]
    s = F.avg_pool3d(F.pad(qp, (1, 1, 1, 1, 1, 1), value=0.0), 3, stride=1)[0, 0] * 27.0
    ones = torch.ones_like(q)[None, None]
    cnt = F.avg_pool3d(F.pad(ones, (1, 1, 1, 1, 1, 1), value=0.0), 3, stride=1)[0, 0] * 27.0
    return (s - q) / (cnt - 1.0).clamp(min=1.0)


def agreement(E, N):
    """A(x) = 1 - |E(x) - N(x)|."""
    return 1.0 - (E - N).abs()


def reaction_potential(E, N, A, alpha, beta, gamma):
    """R(x) = alpha*E + beta*N + gamma*A,  alpha+beta+gamma = 1."""
    return alpha * E + beta * N + gamma * A


def competition_step(q, R, eta):
    """One conservative competitive redistribution step.

    J_{i->j} = eta * [q_i - q_j]_+ * [R_j - R_i]_+

    Returns (q_new, n_clamped). Mass is exactly conserved (up to fp).
    """
    # --- pass 1: raw outflow per voxel, accumulated over all 26 directions ---
    outflow_total = torch.zeros_like(q)
    fluxes = []
    for off in OFFSETS_26:
        q_nb = shift(q, off, fill=0.0)
        R_nb = shift(R, off, fill=0.0)
        # in-volume mask for this offset: neighbour must actually exist
        valid = shift(torch.ones_like(q), off, fill=0.0)
        J = eta * torch.clamp(q - q_nb, min=0.0) * torch.clamp(R_nb - R, min=0.0)
        J = J * valid
        fluxes.append(J)
        outflow_total += J

    # --- per-voxel clamp so a voxel never sends more than it holds ---
    scale = torch.ones_like(q)
    over = outflow_total > q
    n_clamped = int(over.sum().item())
    if n_clamped:
        scale[over] = (q[over] / outflow_total[over].clamp(min=1e-12))
    scale = torch.clamp(scale, 0.0, 1.0)

    # --- pass 2: apply scaled fluxes simultaneously ---
    q_new = q.clone()
    for off, J in zip(OFFSETS_26, fluxes):
        Js = J * scale                      # scaled outflow from each source voxel
        q_new -= Js                          # source loses
        # the mass leaving voxel x along `off` arrives at voxel x+off.
        # shifting Js by -off moves value at x to position x+off.
        inv = (-off[0], -off[1], -off[2])
        q_new += shift(Js, inv, fill=0.0)    # destination gains
    return q_new, n_clamped


def acceptance(S, lam, tau):
    """k+ = sigmoid(lam*(S - tau)); k- = 1 - k+."""
    kp = torch.sigmoid(lam * (S - tau))
    return kp, 1.0 - kp


def proofread_potential(R, S, lam, tau, delta):
    """G_k(x) = R(x) + delta*(k+ - k-)."""
    kp, km = acceptance(S, lam, tau)
    return R + delta * (kp - km)


def conservative_smoothing_step(q, eta):
    """O1 control: q' = (1-eta)q + eta*AvgPool_3x3x3(q), then renormalise to
    preserve total mass exactly (the avg-pool itself leaks mass at boundaries)."""
    m0 = q.sum()
    qp = q[None, None]
    sm = F.avg_pool3d(F.pad(qp, (1, 1, 1, 1, 1, 1), mode='replicate'),
                      kernel_size=3, stride=1)[0, 0]
    out = (1.0 - eta) * q + eta * sm
    s = out.sum()
    if float(s) > 0:
        out = out * (m0 / s)
    return out
