"""
Phase E34, Section 4: component-level adaptive weighting mechanism.

Design (per the explicit instruction: "Do not replace the entire baseline
loss. Implement component-level adaptive weighting as an ADDITIONAL
segmentation weighting mechanism while preserving the existing baseline
objective"):

  L_total = L_seg (UNCHANGED FocalTversky+Evidential, exactly as in A)
          + mu * L_boundary (UNCHANGED, exactly as in A)
          + lambda_cw * L_component_weighted

L_component_weighted is a per-component weighted soft-Dice-style term,
computed ONLY over foreground GT components (never touching background
voxels -- per the explicit "background must not accidentally receive
adaptive weights" requirement). Each connected GT component in the batch's
64^3 masks gets its own weight w_c (from EITHER the size-control or the
alpha_c mechanism, selected by which condition is training), and the
per-component Dice terms are combined via a WEIGHTED AVERAGE (not sum),
matching the L_R/L_S formula's own normalized-weighted-average structure:

    L_component_weighted = sum_c [ w_c * (1 - dice_c) ] / sum_c [ w_c ]

where dice_c is the standard soft Dice of the model's prediction restricted
to component c's own voxels (computed on continuous probabilities, no
thresholding, consistent with every other soft-Dice-style term in this
project's loss stack).

Component identity (native-space labeling, resized to 64^3) is PRECOMPUTED
ONCE per subject, before training starts (see precompute_labeled_64_cache),
NOT recomputed per-batch. An earlier version of this module recomputed the
native->64^3 resize inside the per-batch loss function; direct profiling
before launching the real A/S/R runs measured this at ~5s/batch (~11.7
min/epoch of pure overhead, ~6-7x slower than baseline A), traced to the
fully deterministic, subject-only-dependent scipy.ndimage.zoom call being
redone on every single batch despite always producing the identical result
for a given subject. Fixed by precomputing once. This precomputed
labeled_64 volume is guaranteed to match each training batch's own 64^3 GT
mask exactly, since both derive from the SAME underlying native seg.nii.gz
file via the SAME nearest-neighbor resize convention (BraTSDataset's own
_resize_volume, order=0 for masks) -- verified directly, not assumed (see
test_component_weighted_loss.py).

Per-component weight (w_c) is supplied externally by a WeightLookup object
(constructed once, before training, from the precomputed E34 alpha_c/size
tables), keyed by (subject_id, spatial-overlap match to precomputed
component). Since training-time components are computed fresh from the same
deterministic resize as the precomputation, and BOTH use the SAME native
component-ID volume resized with the SAME nearest-neighbor convention, the
match is exact by construction, verified by a direct consistency check
before any training run (see test_component_weighted_loss.py).
"""
import numpy as np
import torch
import scipy.ndimage as ndi


class WeightLookup:
    """Precomputed per-(subject, native_component_id) weight, for either
    the size-control (S) or alpha_c (R) condition. Constructed once from
    the E30/E31/E34-produced tables; NEVER recomputed from model outputs or
    training-time quantities -- purely a function of GT geometry, fixed
    before training starts, per the requirement that this is a supervision
    weighting mechanism, not a model-adaptive one."""

    def __init__(self, subject_component_weights):
        """subject_component_weights: dict {(subject_id, native_component_id): weight}"""
        self.weights = subject_component_weights

    def get(self, subject_id, native_component_id, default=1.0):
        return self.weights.get((subject_id, native_component_id), default)


def precompute_labeled_64_cache(native_labeled_cache, resize_nn_fn, target_shape=(64, 64, 64)):
    """
    PERFORMANCE FIX (found via direct profiling before launching the real
    A/S/R runs): the original compute_component_weighted_loss resized each
    subject's native component-ID volume to 64^3 FRESH ON EVERY BATCH,
    every epoch -- a deterministic, subject-only-dependent computation
    recomputed ~141 times/epoch per subject unnecessarily (measured at
    ~5s/batch, which would have made S/R epochs ~6-7x slower than baseline
    A, turning a ~67min run into several hours each). Since the resize
    result never changes for a given subject, precompute it ONCE here
    (called once per experiment, at the same point native_labeled_cache
    itself is built), not inside the per-batch loss function.

    Returns: dict {subject_id: labeled_64_int32_array}
    """
    labeled_64_cache = {}
    for subject_id, (native_labeled, native_shape) in native_labeled_cache.items():
        labeled_64 = resize_nn_fn(native_labeled.astype(np.float32), target_shape)
        labeled_64_cache[subject_id] = np.round(labeled_64).astype(np.int32)
    return labeled_64_cache


def compute_component_weighted_loss(probs, masks, subject_ids, weight_lookup, labeled_64_cache,
                                     device, eps=1e-6):
    """
    probs: (B, 1, D, H, W) model output probabilities
    masks: (B, 1, D, H, W) GT binary masks (the ACTUAL 64^3 training masks)
    subject_ids: list of B subject_id strings
    weight_lookup: WeightLookup instance
    labeled_64_cache: dict {subject_id: labeled_64_int32_array} -- PRECOMPUTED
        (see precompute_labeled_64_cache), not resized per-batch.

    Returns: (loss_scalar, diagnostics_dict)

    PERFORMANCE FIX #2 (found via direct profiling after fix #1 alone did
    not resolve the slowdown): the per-component Python loop issued many
    small, separate GPU operations (.to(device) transfer, boolean indexing,
    multiple .sum() reductions, and a fresh torch.tensor(w_c) allocation)
    PER COMPONENT PER BATCH -- with ~10-20 components/batch, this produced
    highly variable per-batch timing (0.16s-4.3s, confirmed via explicit
    torch.cuda.synchronize() calls to rule out an async-CUDA measurement
    artifact) from GPU kernel-launch overhead, not from any growing memory
    cost. FIX: vectorize per-component reductions using
    scatter_add-based bincount-style aggregation computed in ONE pass over
    all components in a batch item at once, with weights precomputed as a
    single per-voxel weight MAP (not per-component tensor allocations),
    eliminating the per-component Python/GPU round-trip entirely.
    """
    B = probs.shape[0]
    D, H, W = probs.shape[2], probs.shape[3], probs.shape[4]

    total_weighted_dice_loss = torch.tensor(0.0, device=device)
    total_weight = torch.tensor(0.0, device=device)
    n_components_seen = 0
    n_components_matched = 0

    for b in range(B):
        subject_id = subject_ids[b]
        if subject_id not in labeled_64_cache:
            continue  # subject has no GT components at all (shouldn't occur in this curated dataset, guarded anyway)
        labeled_64 = labeled_64_cache[subject_id]
        n_native = int(labeled_64.max())
        if n_native == 0:
            continue

        # Build per-component intersection/prediction-sum/gt-sum via a
        # SINGLE vectorized bincount pass, instead of a Python loop with one
        # GPU indexing operation per component.
        labeled_flat_np = labeled_64.reshape(-1)  # (D*H*W,), values in [0, n_native]
        labeled_flat_t = torch.from_numpy(labeled_flat_np.astype(np.int64)).to(device)

        probs_b_flat = probs[b, 0].reshape(-1)  # (D*H*W,), on device, part of the autograd graph

        # sum of probs per component id (bincount weighted by probs) --
        # ONE scatter_add call covers every component in this batch item.
        p_sum_per_comp = torch.zeros(n_native + 1, device=device, dtype=probs_b_flat.dtype)
        p_sum_per_comp.scatter_add_(0, labeled_flat_t, probs_b_flat)

        # voxel count per component id (bincount) -- computed on CPU (cheap,
        # integer-only, no autograd needed) then moved once.
        comp_sizes = np.bincount(labeled_flat_np, minlength=n_native + 1)
        comp_sizes_t = torch.from_numpy(comp_sizes.astype(np.float32)).to(device)

        # p_sum_per_comp[c] = sum of predicted probs inside component c
        # comp_sizes_t[c] = |Y_c| (voxel count, i.e. sum of GT-ones inside component c)
        # dice_c = (2*inter + 1) / (p_sum + gt_sum + 1), where inter = p_sum
        # (since every GT voxel in a component is foreground, p*y = p) and
        # gt_sum = comp_sizes_t[c] (each GT voxel contributes 1).
        dice_per_comp = (2 * p_sum_per_comp[1:] + 1.0) / (p_sum_per_comp[1:] + comp_sizes_t[1:] + 1.0)
        loss_per_comp = 1 - dice_per_comp  # (n_native,) tensor, index i = component (i+1)

        # Build the weight vector for this subject's components in ONE pass
        # (Python-level dict lookups are cheap/CPU-only, no GPU round-trip)
        weights_np = np.zeros(n_native, dtype=np.float32)
        matched_mask = np.zeros(n_native, dtype=bool)
        valid_mask = comp_sizes[1:] > 0  # exclude components that vanished at 64^3 (size 0) -- SAME guard as before, now vectorized
        for comp_id in range(1, n_native + 1):
            if comp_sizes[comp_id] == 0:
                continue
            n_components_seen += 1
            w_c = weight_lookup.get(subject_id, comp_id, default=None)
            if w_c is None:
                continue
            n_components_matched += 1
            weights_np[comp_id - 1] = w_c
            matched_mask[comp_id - 1] = True

        if not matched_mask.any():
            continue  # no matched components for this subject -- contributes nothing, not an error

        weights_t = torch.from_numpy(weights_np).to(device)
        matched_t = torch.from_numpy(matched_mask).to(device)

        weighted_loss_b = (weights_t[matched_t] * loss_per_comp[matched_t]).sum()
        weight_sum_b = weights_t[matched_t].sum()

        total_weighted_dice_loss = total_weighted_dice_loss + weighted_loss_b
        total_weight = total_weight + weight_sum_b

    if total_weight.item() < eps:
        # No components in this batch had a matched weight (should not
        # happen in normal operation, but guarded to keep total loss finite
        # rather than dividing by ~0) -- return exactly 0, contributes
        # nothing to the total loss this step, not NaN/Inf.
        return torch.tensor(0.0, device=device), {
            "n_components_seen": n_components_seen, "n_components_matched": n_components_matched,
        }

    loss = total_weighted_dice_loss / total_weight
    return loss, {"n_components_seen": n_components_seen, "n_components_matched": n_components_matched}
