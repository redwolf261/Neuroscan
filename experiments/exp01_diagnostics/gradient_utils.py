"""
Independent per-loss gradient extraction for the two-head UNet3D.

Core idea: to measure how L_focal and L_evidential individually pull on
shared parameters, we need d(L_focal)/d(theta) and d(L_evidential)/d(theta)
as SEPARATE vectors, not the gradient of their weighted sum. Autograd only
ever gives you the gradient of whatever scalar you call .backward() on, so
getting both requires two backward passes over the same forward graph.

This module does that without touching the actual optimizer step: it takes
snapshots of .grad via two extra backward() calls (graph kept alive with
retain_graph=True), then restores the model to a clean zero-grad state
before the real combined-loss backward + optimizer.step() runs. Training
dynamics are therefore bit-for-bit identical to a diagnostics-free run;
only extra backward/copy work is added.

Parameter groups:
  trunk = everything up to and including dec1 (shared encoder-decoder)
  seg_head = model.seg_head
  evid_head = model.evidential_head
"""

import torch


TRUNK_MODULE_NAMES = [
    "enc1", "pool1", "enc2", "pool2", "enc3", "pool3",
    "bottleneck",
    "upconv3", "dec3", "upconv2", "dec2", "upconv1", "dec1",
]
HEAD_MODULE_NAMES = {
    "seg_head": "seg_head",
    "evid_head": "evidential_head",
}


def get_param_groups(model):
    """
    Returns dict: {"trunk": [params...], "seg_head": [...], "evid_head": [...]}
    Named parameters, in a stable order, for reproducible flattening.
    """
    groups = {"trunk": [], "seg_head": [], "evid_head": []}
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        if top in TRUNK_MODULE_NAMES:
            groups["trunk"].append((name, param))
        elif top == "seg_head":
            groups["seg_head"].append((name, param))
        elif top == "evidential_head":
            groups["evid_head"].append((name, param))
        else:
            raise ValueError(f"Unrecognized top-level module in param name: {name}")
    return groups


def _flatten_grad(named_params, use_grad=True):
    """Concatenate .grad (or zeros if None) for a list of (name, param) into one vector."""
    parts = []
    for _, p in named_params:
        if use_grad and p.grad is not None:
            parts.append(p.grad.detach().reshape(-1))
        else:
            parts.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def compute_branch_gradients(model, focal_loss, evidential_loss):
    """
    Runs two extra backward passes (focal, then evidential) to capture their
    individual gradients per parameter group, then zeros model.grad so the
    caller can proceed with the real combined-loss backward cleanly.

    Requires the forward graph still alive, i.e. call this BEFORE the
    combined-loss .backward() and AFTER outputs = model(...); losses computed.

    Returns:
        dict with keys "focal" and "evidential", each mapping to a dict
        {"trunk": Tensor, "seg_head": Tensor, "evid_head": Tensor} of
        flattened gradient vectors (detached, on the model's device).
    """
    groups = get_param_groups(model)
    result = {"focal": {}, "evidential": {}}

    # --- Focal gradient ---
    model.zero_grad(set_to_none=True)
    focal_loss.backward(retain_graph=True)
    for gname, named_params in groups.items():
        result["focal"][gname] = _flatten_grad(named_params)

    # --- Evidential gradient ---
    model.zero_grad(set_to_none=True)
    evidential_loss.backward(retain_graph=True)
    for gname, named_params in groups.items():
        result["evidential"][gname] = _flatten_grad(named_params)

    # Leave model in a clean state for the real backward pass that follows
    model.zero_grad(set_to_none=True)

    return result


def cosine_similarity(a, b, eps=1e-12):
    """
    Cosine similarity between two flat tensors.

    Returns None (not 0.0) when either vector has ~zero norm. This matters
    architecturally: focal_loss has NO path to evidential_head's parameters
    (and vice versa), so e.g. focal_grad[evid_head] is identically zero by
    construction, every batch. A cosine of 0.0 there would look like a
    measured near-orthogonality when it is actually a tautology -- there is
    nothing to compare. Only the trunk vectors are both genuinely nonzero
    (both losses flow through the shared trunk), so cos(trunk) is the only
    one of the three group-wise cosines that is a real measurement.
    """
    if a.numel() == 0 or b.numel() == 0:
        return None
    na, nb = torch.norm(a), torch.norm(b)
    if na < eps or nb < eps:
        return None
    return (torch.dot(a, b) / (na * nb)).item()


def grad_norm(a):
    return torch.norm(a).item()


class EMATracker:
    """Exponential moving average mean/variance per key, for gradient-norm ratio tracking."""
    def __init__(self, decay=0.9):
        self.decay = decay
        self.mean = {}
        self.var = {}

    def update(self, key, value):
        if key not in self.mean:
            self.mean[key] = value
            self.var[key] = 0.0
        else:
            delta = value - self.mean[key]
            self.mean[key] += (1 - self.decay) * delta
            self.var[key] = self.decay * self.var[key] + (1 - self.decay) * delta ** 2
        return self.mean[key], self.var[key]
