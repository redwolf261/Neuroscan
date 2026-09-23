"""
Per-parameter gradient capture and ABO update assembly.

Phase B's gradient_utils.compute_branch_gradients (exp01_diagnostics)
returns FLATTENED, concatenated gradient vectors per parameter group --
sufficient for computing norms/cosine similarity, but not for
reconstructing per-parameter .grad tensors to feed into optimizer.step().
This module captures gradients per-parameter (preserving shape) so they
can be reassembled into a modified gradient before the real optimizer step.

Update rule (Design Step 6), exact:

  trunk param:      grad = focal_weight * g_focal[param]
                          + multiplier * evidential_weight * g_evid[param]
  seg_head param:    grad = focal_weight * g_focal[param]
                          (g_evid[param] is architecturally always exactly
                           zero here -- seg_head has no path to L_evidential)
  evidential_head:   grad = evidential_weight * g_evid[param]
                          (g_focal[param] is architecturally always exactly
                           zero here -- evidential_head has no path to L_focal)

When multiplier == 1.0 (ABO inactive / first batch), this reduces EXACTLY
to the baseline's original combined-loss gradient
(focal_weight*g_focal + evidential_weight*g_evid) for every parameter --
i.e. ABO is an identity transform of the baseline update whenever
multiplier=1, which is what makes the Step 3 "monitor mode" verification
meaningful: with multiplier forced to 1.0 for every batch, training must
reproduce Phase B/C0 numbers exactly.

focal_weight and evidential_weight here are the SAME frozen static
weights used throughout Phase A.5/B/C0 (0.5/0.5) -- they are not
modified by ABO. Only the trunk's evidential contribution gets the
additional dynamic `multiplier` factor on top of the static weight.
The heads always receive exactly what they would under the frozen
baseline, unchanged, per Step 6's explicit requirement.
"""

import torch

from gradient_utils import TRUNK_MODULE_NAMES  # noqa: E402 (sibling exp01_diagnostics module, see sys.path setup in caller)


def capture_named_grads(model, loss, retain_graph):
    """
    {param_name: grad_tensor} for every parameter (zeros for parameters with
    no path to `loss`, e.g. evidential_head's params when loss=focal_loss).

    Uses torch.autograd.grad instead of loss.backward() + reading .grad.
    Two benefits over the backward()-based approach this replaced:
      1. No .grad buffer is populated at all, so no model.zero_grad() is
         needed before or between the two extraction calls -- the two
         branches' gradients never touch .grad or each other.
      2. The returned tensors are freshly allocated by autograd.grad itself
         (never aliased to a persistent, later-overwritten .grad buffer),
         so no defensive .clone() is needed either -- removing a full
         redundant copy of every parameter's gradient that the previous
         backward()+.grad.clone() approach required.
    Net effect: materially lower peak memory and one fewer full graph
    traversal's worth of buffer bookkeeping per batch than the previous
    retain_graph=True + .grad.clone() + zero_grad() pattern.

    retain_graph must be True for the FIRST of the two calls (the graph is
    needed again for the second loss) and can be False for the second
    (nothing needs the graph after both branches are extracted).
    """
    params = list(model.parameters())
    names = [name for name, _ in model.named_parameters()]
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return {
        name: (g.detach() if g is not None else torch.zeros_like(p))
        for name, g, p in zip(names, grads, params)
    }


def group_flat_norm(named_grads, top_level_name_set):
    """L2 norm over all parameters whose top-level module name is in the given set."""
    total_sq = 0.0
    for name, g in named_grads.items():
        top = name.split(".")[0]
        if top in top_level_name_set:
            total_sq += torch.sum(g * g).item()
    return total_sq ** 0.5


def group_flat_vector(named_grads, top_level_name_set):
    """Concatenated flat vector over all parameters whose top-level name is in the given set."""
    parts = []
    for name, g in named_grads.items():
        top = name.split(".")[0]
        if top in top_level_name_set:
            parts.append(g.reshape(-1))
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def trunk_norms(focal_grads, evid_grads):
    """Convenience: raw (unweighted) trunk gradient norms for both branches."""
    trunk_set = set(TRUNK_MODULE_NAMES)
    return (
        group_flat_norm(focal_grads, trunk_set),
        group_flat_norm(evid_grads, trunk_set),
    )


def apply_abo_update(model, focal_grads, evid_grads, focal_weight, evidential_weight, multiplier):
    """
    Sets model.*.grad in place for every parameter per the update rule above.
    Does NOT call optimizer.step() -- caller does that afterward, with the
    same grad-norm clipping used throughout Phase A.5/B/C0 applied to
    these final assembled gradients (unchanged from baseline behavior).
    """
    trunk_set = set(TRUNK_MODULE_NAMES)
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        fg = focal_grads[name]
        eg = evid_grads[name]
        if top in trunk_set:
            p.grad = focal_weight * fg + multiplier * evidential_weight * eg
        elif top == "seg_head":
            p.grad = focal_weight * fg
        elif top == "evidential_head":
            p.grad = evidential_weight * eg
        else:
            raise ValueError(f"Unrecognized top-level module in param name: {name}")
