"""
Phase E66: Optimization Topology Audit (block-wise gradient interaction).

NO NEW TRAINING. NO ARCHITECTURE CHANGE. NO NEW OPTIMIZER. Pure
observational diagnostic on the network's REAL, currently-active
multi-objective loss composition, at existing saved checkpoints.

CONTEXT: after the E62-E65 causal-diagnostic branch (a legitimate but
separate research thread on WHY this particular architecture behaves as
it does) and a brief, self-corrected multimodal detour (CMRR, killed on
novelty grounds before training; the multimodal-audit follow-up itself
was then killed as scope creep before any training), the project returns
to its original stated axis: an architecture-independent optimization
mechanism for multi-objective neural networks.

CORRECTION TO THE ORIGINAL PROPOSAL'S OBJECTIVE SET: the proposal's L1-L4
(seg/evidential/consistency/pseudo) does not match what actually trains
in this project's live EGGO-M pipeline (experiments/exp_e12_eggo_m/
train_eggo_m.py, the SAME lineage as every E43-E65 checkpoint).
Consistency and pseudo-label losses exist only in the older, unrelated,
frozen final_model.py/Phase-1-3 pipeline (MAE pretraining + teacher-
student) and are not used here. The live loss composition is actually:
    seg_loss  = 0.5*FocalTversky + 0.5*EvidentialBeta
    total     = seg_loss + mu*BoundaryBCE + lambda_margin*margin (dormant, =0)

SECOND CORRECTION, found during implementation: boundary_head reads
dec1.detach() (a deliberate, documented v2 design choice -- see
neuroscan_3d_v2.py's own module docstring: "prevents boundary_head from
becoming separable on its own account"). This means boundary loss's
gradient is architecturally ZERO in encoder/bottleneck/decoder BY
CONSTRUCTION -- not a discovered phenomenon. Including it would
contaminate the block-specific-structure KILL/GO decision with a known
architectural artifact (explicit user sign-off to exclude it, matching
the project's own falsifiability/mechanism-vs-performance constraints).

THE REAL OBJECTIVE SET USED HERE (only 2 losses; margin is dormant,
boundary is architecturally excluded per the above):
    L1 = FocalTversky   (focal_fn(probs, masks))
    L2 = EvidentialBeta (evidential_fn(alpha, beta, masks))
These are the only two losses that both flow through the FULL shared
trunk (encoder->bottleneck->decoder->heads) UNDETACHED, making their
cross-block gradient interaction the only pair in this architecture
where the measurement reflects real optimization structure rather than
an architectural given.

METHOD (pre-declared, decided by explicit user sign-off before running):
  Use the 7 existing saved checkpoints from E25's DeepSup_D4only_seed0
  run (epochs 1, 5, 10, 15, 20, 25, 30 -- all present on disk, ZERO new
  training). At each checkpoint, load the weights, run ONE forward pass
  on a fixed batch of real training subjects, then compute THREE
  SEPARATE backward passes (one per loss, each with retain_graph as
  needed) to get g_1, g_2, g_3 -- the per-parameter gradient of each
  loss ALONE (not the combined training loss) at that checkpoint's
  weights. This is a pure diagnostic: no optimizer step is taken, no
  weights are ever modified.

PARAMETER BLOCKS (4, matching the proposal's own E/B/D/H notation,
decided by explicit sign-off over per-layer granularity for this first
pass):
    theta_E = enc1, enc2, enc3
    theta_B = bottleneck
    theta_D = upconv1, upconv2, upconv3, dec1, dec2, dec3
    theta_H = seg_head, evidential_head, boundary_head
  (aux_head3/aux_head2 excluded -- they are deep-supervision outputs
  trained by lambda_ds3 * aux3_loss, a 4th loss this phase does NOT
  additionally introduce, per the sign-off to use only the 3 real
  losses already active in the checkpoint's own training recipe.)

FOR EVERY (checkpoint, block) PAIR, computed on the SAME fixed batch at
every checkpoint (so cross-checkpoint comparisons are not confounded by
batch variation):
    ||g_i||                              -- per-loss gradient norm
    cos(g_i, g_j) for all 3 loss pairs    -- pairwise alignment
    r_i = ||g_i|| / sum_j ||g_j||         -- relative influence

PRE-DECLARED THREE-WAY OUTCOME (verbatim from the user's own spec):
  A. Gradients are basically homogeneous across blocks (no meaningful
     block-to-block variation in cos(g_i,g_j) beyond noise) -> KILL,
     no reason for block-specific optimization.
  B. Gradients differ by block, but the pattern is fully consistent with
     what existing global gradient-surgery methods (GradNorm/PCGrad/
     MGDA/CAGrad-style: conflict is conflict everywhere it appears, no
     structure beyond magnitude/sign) would already handle -> KILL the
     proposed novelty, search for a different abstraction.
  C. Gradients show a STABLE, MEANINGFUL block-specific role structure
     (e.g. a loss pair that is reliably cooperative in one block and
     reliably conflicting in another, consistently across checkpoints)
     -> GO, this becomes the raw material for an operator design (not
     designed in this phase).
This phase's own analysis reports the block x block x checkpoint
interaction structure and evaluates against outcomes A/B/C explicitly;
it does NOT itself design or claim to design the optimizer.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
BATCH_SIZE = 8  # matches the project's own established training batch size (configs/brats.yaml)
N_BATCHES_AVERAGED = 4  # average gradient statistics over several fixed batches, not just one, for stability

CKPT_DIR = (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs"
            / "DeepSup_D4only_seed0" / "checkpoints")
CHECKPOINT_EPOCHS = [1, 5, 10, 15, 20, 25, 30]

BLOCK_MODULES = {
    "encoder": ["enc1", "enc2", "enc3"],
    "bottleneck": ["bottleneck"],
    "decoder": ["upconv1", "upconv2", "upconv3", "dec1", "dec2", "dec3"],
    "heads": ["seg_head", "evidential_head", "boundary_head"],
}
BLOCK_NAMES = list(BLOCK_MODULES.keys())

# "heads" is EXCLUDED from the outcome (KILL/GO) decision, per explicit
# user sign-off: seg_head and evidential_head are disjoint parameter
# subsets (focal_tversky's gradient only ever touches seg_head, since
# probs=seg_head(dec1); evidential's gradient only ever touches
# evidential_head), so their flattened block vectors are GEOMETRICALLY
# FORCED to cos=0 at every checkpoint regardless of any real
# relationship -- pure linear algebra, not discovered structure. This
# is the block-composition analogue of the boundary-loss exclusion
# above (both are cases where a structural/detachment property, not an
# observed optimization phenomenon, would determine the number).
# encoder/bottleneck/decoder are exempt from this issue: seg_head and
# evidential_head both read the SAME dec1 output, so both losses'
# gradients flow back through the SAME shared encoder/bottleneck/
# decoder weights via the chain rule -- cos there reflects real
# optimization structure. "heads" is still computed and reported for
# transparency, just excluded from between_block_range/sign_flips/outcome.
OUTCOME_BLOCKS = ["encoder", "bottleneck", "decoder"]

LOSS_NAMES = ["focal_tversky", "evidential"]
LOSS_PAIRS = [("focal_tversky", "evidential")]
# boundary loss deliberately excluded: boundary_head reads dec1.detach()
# (a documented v2 design choice, see neuroscan_3d_v2.py's own module
# docstring), so its gradient is architecturally ZERO in
# encoder/bottleneck/decoder by construction -- including it would
# contaminate the block-specific-structure KILL/GO decision with a known
# architectural artifact rather than a discovered phenomenon (explicit
# user sign-off). focal_tversky and evidential are the only two losses
# that both flow through the full shared trunk (encoder->bottleneck->
# decoder->heads) UNDETACHED, making their cross-block interaction the
# only pair where the measurement is actually meaningful here.


def module_to_block(param_name):
    """Map a named parameter (e.g. 'enc2.0.conv.weight') to its block
    label, or None if it belongs to an excluded module (aux heads)."""
    top = param_name.split(".")[0]
    for block, modules in BLOCK_MODULES.items():
        if top in modules:
            return block
    return None  # aux_head3/aux_head2 -- excluded per pre-declaration


def flatten_block_grads(model, block_params):
    """block_params: dict block_name -> list of (name, param) tuples.
    Returns dict block_name -> flat 1D tensor of that block's gradients
    (concatenated across all its parameters, using .grad as populated by
    the most recent backward() call)."""
    out = {}
    for block, params in block_params.items():
        flats = []
        for name, p in params:
            if p.grad is None:
                flats.append(torch.zeros(p.numel(), device=p.device))
            else:
                flats.append(p.grad.detach().reshape(-1))
        out[block] = torch.cat(flats) if flats else torch.zeros(0)
    return out


def compute_loss_gradients(model, images, masks, focal_fn, evidential_fn, block_params):
    """Runs ONE shared forward pass, then TWO separate backward passes
    (one per loss), zeroing grads between each, to get each loss's OWN
    gradient (not the combined training gradient) at the current
    weights. Returns dict loss_name -> {block_name -> flat grad tensor}.
    No optimizer step; weights are never modified. boundary_head/
    boundary_criterion deliberately excluded -- see LOSS_NAMES comment."""
    outputs = model(images)
    probs, alpha, beta = outputs["probs"], outputs["alpha"], outputs["beta"]

    losses = {
        "focal_tversky": focal_fn(probs, masks),
        "evidential": evidential_fn(alpha, beta, masks),
    }

    grads_by_loss = {}
    for i, (loss_name, loss_val) in enumerate(losses.items()):
        model.zero_grad(set_to_none=True)
        retain = i < len(losses) - 1  # need the shared forward graph for subsequent backward calls
        loss_val.backward(retain_graph=retain)
        grads_by_loss[loss_name] = flatten_block_grads(model, block_params)
        model.zero_grad(set_to_none=True)

    return grads_by_loss, {k: float(v.item()) for k, v in losses.items()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="train", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    print(f"Training set size: {len(train_dataset)}")

    # Fixed batches (same subjects at every checkpoint, so cross-checkpoint
    # comparisons are not confounded by batch variation), sampled once,
    # deterministically, before looking at any gradient result.
    rng = np.random.default_rng(SEED)
    n_train = len(train_dataset)
    batch_indices = []
    for _ in range(N_BATCHES_AVERAGED):
        idx = rng.choice(n_train, size=BATCH_SIZE, replace=False)
        batch_indices.append(idx)

    fixed_batches = []
    for idx_arr in batch_indices:
        imgs, msks = [], []
        for i in idx_arr:
            image, mask, _ = train_dataset[int(i)]
            imgs.append(image)
            msks.append(mask)
        fixed_batches.append((torch.stack(imgs).to(device), torch.stack(msks).to(device)))
    print(f"Prepared {len(fixed_batches)} fixed batches of size {BATCH_SIZE} (same subjects reused at every checkpoint).\n")

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)

    records = []  # one entry per (checkpoint_epoch, batch_idx)

    for epoch in CHECKPOINT_EPOCHS:
        ckpt_path = CKPT_DIR / f"epoch_{epoch}.pth"
        assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = UNet3D_v3(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.train()  # match the training-time BN/dropout mode the gradients would actually see

        block_params = {b: [] for b in BLOCK_NAMES}
        for name, p in model.named_parameters():
            block = module_to_block(name)
            if block is not None:
                block_params[block].append((name, p))

        print(f"=== Checkpoint epoch {epoch} (val_dice={ckpt.get('best_val_dice')}) ===")

        for b_idx, (images, masks) in enumerate(fixed_batches):
            grads_by_loss, loss_values = compute_loss_gradients(
                model, images, masks, focal_fn, evidential_fn, block_params)

            entry = {"epoch": epoch, "batch_idx": b_idx, "loss_values": loss_values, "blocks": {}}
            for block in BLOCK_NAMES:
                block_entry = {"norms": {}, "cosines": {}}
                for loss_name in LOSS_NAMES:
                    g = grads_by_loss[loss_name][block]
                    block_entry["norms"][loss_name] = float(torch.norm(g).item())
                for (li, lj) in LOSS_PAIRS:
                    gi = grads_by_loss[li][block]
                    gj = grads_by_loss[lj][block]
                    ni, nj = torch.norm(gi), torch.norm(gj)
                    if ni > 1e-12 and nj > 1e-12:
                        cos = float((torch.dot(gi, gj) / (ni * nj)).item())
                    else:
                        cos = float("nan")
                    block_entry["cosines"][f"{li}__{lj}"] = cos
                entry["blocks"][block] = block_entry

            records.append(entry)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(OUT_DIR / "E66_gradient_topology_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} (checkpoint, batch) records.\n")

    # ================= Aggregate analysis =================
    print("=== E66 Gradient Topology Audit: aggregate results ===\n")

    # Mean cosine per (block, loss-pair), averaged over batches, reported per checkpoint
    # and also averaged ACROSS checkpoints (for the "stable structure" question).
    summary_per_checkpoint = {}
    for epoch in CHECKPOINT_EPOCHS:
        epoch_records = [r for r in records if r["epoch"] == epoch]
        summary_per_checkpoint[epoch] = {}
        for block in BLOCK_NAMES:
            summary_per_checkpoint[epoch][block] = {}
            for (li, lj) in LOSS_PAIRS:
                key = f"{li}__{lj}"
                vals = [r["blocks"][block]["cosines"][key] for r in epoch_records
                        if not np.isnan(r["blocks"][block]["cosines"][key])]
                summary_per_checkpoint[epoch][block][key] = float(np.mean(vals)) if vals else float("nan")

    print("Mean cos(g_i, g_j) per block, per checkpoint (rows=blocks, within each checkpoint):")
    for epoch in CHECKPOINT_EPOCHS:
        print(f"\n  --- epoch {epoch} ---")
        for block in BLOCK_NAMES:
            vals_str = ", ".join(f"{k}={v:+.3f}" for k, v in summary_per_checkpoint[epoch][block].items())
            print(f"    {block:10s}: {vals_str}")

    # Cross-checkpoint aggregate: mean and std of each (block, pair) cosine across all 7 checkpoints
    cross_checkpoint_stats = {}
    for block in BLOCK_NAMES:
        cross_checkpoint_stats[block] = {}
        for (li, lj) in LOSS_PAIRS:
            key = f"{li}__{lj}"
            vals = [summary_per_checkpoint[epoch][block][key] for epoch in CHECKPOINT_EPOCHS
                    if not np.isnan(summary_per_checkpoint[epoch][block][key])]
            cross_checkpoint_stats[block][key] = {
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "std": float(np.std(vals)) if vals else float("nan"),
                "min": float(np.min(vals)) if vals else float("nan"),
                "max": float(np.max(vals)) if vals else float("nan"),
            }

    print("\n\n=== Cross-checkpoint stability (mean +/- std of cos(g_i,g_j) across all 7 checkpoints) ===")
    for block in BLOCK_NAMES:
        print(f"\n  {block}:")
        for (li, lj) in LOSS_PAIRS:
            key = f"{li}__{lj}"
            s = cross_checkpoint_stats[block][key]
            print(f"    {key:30s}: mean={s['mean']:+.3f} std={s['std']:.3f} range=[{s['min']:+.3f},{s['max']:+.3f}]")

    # ================= Between-block variation (does structure differ meaningfully across blocks?) =================
    # NOTE: computed over OUTCOME_BLOCKS only (encoder/bottleneck/decoder),
    # EXCLUDING "heads" -- seg_head and evidential_head are disjoint
    # parameter subsets for these two losses (each loss's gradient only
    # ever touches its OWN head), so "heads" cos(g_i,g_j) is
    # geometrically forced to 0 regardless of any real relationship.
    # Reported separately below for transparency, not fed into the
    # outcome decision (explicit user sign-off).
    print("\n\n=== Between-block variation, per loss-pair (does conflict/cooperation differ meaningfully by block?) ===")
    print("(OUTCOME_BLOCKS only: encoder/bottleneck/decoder -- 'heads' excluded, see NOTE above/in code)")
    between_block_range = {}
    for (li, lj) in LOSS_PAIRS:
        key = f"{li}__{lj}"
        block_means = [cross_checkpoint_stats[b][key]["mean"] for b in OUTCOME_BLOCKS]
        rng_val = float(np.max(block_means) - np.min(block_means))
        between_block_range[key] = rng_val
        block_str = ", ".join(f"{b}={cross_checkpoint_stats[b][key]['mean']:+.3f}" for b in OUTCOME_BLOCKS)
        print(f"  {key:30s}: range across blocks={rng_val:.3f}  ({block_str})")
        heads_val = cross_checkpoint_stats["heads"][key]["mean"]
        print(f"    [reported, NOT in outcome] heads={heads_val:+.3f} "
              f"(geometrically forced ~0 -- disjoint seg_head/evidential_head support)")

    # ================= Sign-flip check (a pair that is cooperative in one block, conflicting in another) =================
    print("\n\n=== Sign-flip check: does any loss pair change cooperative<->conflicting sign across blocks? ===")
    print("(OUTCOME_BLOCKS only: encoder/bottleneck/decoder)")
    sign_flips = {}
    for (li, lj) in LOSS_PAIRS:
        key = f"{li}__{lj}"
        block_means = {b: cross_checkpoint_stats[b][key]["mean"] for b in OUTCOME_BLOCKS}
        signs = set(np.sign(v) for v in block_means.values() if not np.isnan(v))
        flips = len(signs) > 1
        sign_flips[key] = {"flips": bool(flips), "block_means": block_means}
        print(f"  {key:30s}: sign flip across blocks = {flips}  ({block_means})")

    # ================= Pre-declared outcome evaluation =================
    print("\n\n=== Pre-declared outcome evaluation (A / B / C) ===")
    # Heuristic thresholds, declared here (not tuned post-hoc against the
    # result): "meaningful" between-block variation = range > 0.15 in
    # mean cosine; "stable" = cross-checkpoint std < 0.15 for the blocks
    # showing the largest effect.
    MEANINGFUL_RANGE_THRESHOLD = 0.15
    STABILITY_STD_THRESHOLD = 0.15

    any_meaningful_variation = any(v > MEANINGFUL_RANGE_THRESHOLD for v in between_block_range.values())
    any_sign_flip = any(v["flips"] for v in sign_flips.values())

    stability_ok = True
    for (li, lj) in LOSS_PAIRS:
        key = f"{li}__{lj}"
        if between_block_range[key] > MEANINGFUL_RANGE_THRESHOLD:
            # check stability specifically for the block with the extreme value (OUTCOME_BLOCKS only)
            block_means = {b: cross_checkpoint_stats[b][key]["mean"] for b in OUTCOME_BLOCKS}
            extreme_block = max(block_means, key=lambda b: abs(block_means[b] - np.mean(list(block_means.values()))))
            if cross_checkpoint_stats[extreme_block][key]["std"] > STABILITY_STD_THRESHOLD:
                stability_ok = False

    if not any_meaningful_variation:
        outcome = "A_HOMOGENEOUS_KILL"
        print(f"Between-block range never exceeds {MEANINGFUL_RANGE_THRESHOLD} for any loss pair.")
        print("=== OUTCOME A: gradients are basically homogeneous across blocks. KILL -- no reason for "
              "block-specific optimization. ===")
    elif any_meaningful_variation and not any_sign_flip:
        outcome = "B_MAGNITUDE_ONLY_LIKELY_KILL"
        print("Between-block variation exists, but no loss pair flips sign (cooperative<->conflicting) "
              "across blocks -- the variation looks like magnitude/degree differences, which existing "
              "global gradient-surgery methods (GradNorm/PCGrad/MGDA/CAGrad-style) are already designed "
              "to handle via per-parameter or per-layer scaling.")
        print("=== OUTCOME B (likely): KILL the proposed block-specific-role novelty; if pursued further, "
              "a direct comparison against an existing method's actual behavior on this exact structure "
              "is required before any GO. ===")
    elif any_sign_flip and stability_ok:
        outcome = "C_STRUCTURED_GO"
        print("At least one loss pair meaningfully changes sign (cooperative in one block, conflicting in "
              "another) AND this pattern is stable across checkpoints (std below threshold in the extreme block).")
        print("=== OUTCOME C: stable, meaningful block-specific role structure found. GO -- this becomes "
              "raw material for operator design in a SEPARATE future phase (not designed here). ===")
    else:
        outcome = "C_STRUCTURED_BUT_UNSTABLE_QUALIFIED"
        print("At least one loss pair flips sign across blocks, but the pattern is NOT stable across "
              "checkpoints (std exceeds threshold) -- real structure exists but its instability over "
              "training undermines a fixed block-specific routing rule.")
        print("=== OUTCOME: QUALIFIED -- structure exists but is not stable enough for a straightforward "
              "operator; would need to characterize HOW it evolves before any design. ===")

    summary = {
        "checkpoint_epochs": CHECKPOINT_EPOCHS, "n_batches_averaged": N_BATCHES_AVERAGED,
        "loss_names": LOSS_NAMES, "block_names": BLOCK_NAMES,
        "summary_per_checkpoint": summary_per_checkpoint,
        "cross_checkpoint_stats": cross_checkpoint_stats,
        "between_block_range": between_block_range,
        "sign_flips": sign_flips,
        "thresholds": {"meaningful_range": MEANINGFUL_RANGE_THRESHOLD, "stability_std": STABILITY_STD_THRESHOLD},
        "outcome": outcome,
    }
    with open(OUT_DIR / "E66_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E66_summary.json")


if __name__ == "__main__":
    main()
