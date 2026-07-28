"""
Diagnostic configuration flags.

ALL flags default to False. When every flag below is False, the training
script behaves exactly as it did before this diagnostics package existed —
no extra tensors are retained, no extra backward passes run, no extra files
are written. Each flag can also be set via an environment variable of the
same name, so diagnostics can be toggled per-run without editing code:

    set ENABLE_LOSS_DIAGNOSTICS=1
    python resume_training.py

Flags are intentionally independent and composable:
  - ENABLE_LOSS_DIAGNOSTICS      -> training_metrics.csv        (~0.1% overhead)
  - ENABLE_SLICE_DIAGNOSTICS     -> slice_metrics.csv           (~1-3% overhead)
  - ENABLE_GRADIENT_DIAGNOSTICS  -> gradient_metrics.csv        (~2% overhead)
  - ENABLE_GRADIENT_SIMILARITY   -> gradient_metrics_similarity.csv (~20-30% overhead
                                    on the batches it runs on; gated further by
                                    GRADIENT_SIMILARITY_EVERY_N_BATCHES so the
                                    *average* overhead across an epoch is much lower)

Each is a strict measurement addition: none of them feed back into the
forward pass, loss computation, or optimizer step.
"""

import os


def _bool_env(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _int_env(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


# ── Master switches (all default False) ─────────────────────────────────────
ENABLE_LOSS_DIAGNOSTICS = _bool_env("ENABLE_LOSS_DIAGNOSTICS", False)
ENABLE_SLICE_DIAGNOSTICS = _bool_env("ENABLE_SLICE_DIAGNOSTICS", False)
ENABLE_GRADIENT_DIAGNOSTICS = _bool_env("ENABLE_GRADIENT_DIAGNOSTICS", False)
ENABLE_GRADIENT_SIMILARITY = _bool_env("ENABLE_GRADIENT_SIMILARITY", False)

# ── Sampling frequency controls (reduce overhead of expensive diagnostics) ──
# Layer-wise gradient logging is verbose (one row per parameter tensor per
# logged batch) - only log every Nth batch.
LAYER_WISE_EVERY_N_BATCHES = _int_env("LAYER_WISE_EVERY_N_BATCHES", 10)

# Loss-specific gradient attribution requires separate backward passes per
# loss term - only do this every Nth batch.
LOSS_SPECIFIC_GRADIENT_EVERY_N_BATCHES = _int_env("LOSS_SPECIFIC_GRADIENT_EVERY_N_BATCHES", 5)

# Gradient cosine similarity requires 3+ separate backward passes per
# measured batch (most expensive diagnostic) - only do this every Nth batch.
GRADIENT_SIMILARITY_EVERY_N_BATCHES = _int_env("GRADIENT_SIMILARITY_EVERY_N_BATCHES", 20)


def any_enabled():
    """True if any diagnostic subsystem is active. Used for a single
    startup print so it's always obvious whether a run has diagnostics on."""
    return (ENABLE_LOSS_DIAGNOSTICS or ENABLE_SLICE_DIAGNOSTICS
            or ENABLE_GRADIENT_DIAGNOSTICS or ENABLE_GRADIENT_SIMILARITY)


def print_status():
    print("=" * 80)
    print("DIAGNOSTICS CONFIG")
    print("=" * 80)
    print(f"  ENABLE_LOSS_DIAGNOSTICS:     {ENABLE_LOSS_DIAGNOSTICS}")
    print(f"  ENABLE_SLICE_DIAGNOSTICS:    {ENABLE_SLICE_DIAGNOSTICS}")
    print(f"  ENABLE_GRADIENT_DIAGNOSTICS: {ENABLE_GRADIENT_DIAGNOSTICS}")
    print(f"  ENABLE_GRADIENT_SIMILARITY:  {ENABLE_GRADIENT_SIMILARITY}")
    if any_enabled():
        print(f"  (sampling) layer-wise every {LAYER_WISE_EVERY_N_BATCHES} batches")
        print(f"  (sampling) loss-specific grad every {LOSS_SPECIFIC_GRADIENT_EVERY_N_BATCHES} batches")
        print(f"  (sampling) gradient similarity every {GRADIENT_SIMILARITY_EVERY_N_BATCHES} batches")
    else:
        print("  All diagnostics OFF - training behavior is unmodified from baseline.")
    print("=" * 80)
