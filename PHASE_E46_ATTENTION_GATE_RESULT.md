# Phase E46 — Bottleneck-Conditioned Attention Gate (UNet3D_v5)

## Context

Following E44 (RCGW, killed on instability, see `PHASE_E44_E45_POST_PIVOT_ALGORITHMIC_ATTEMPTS.md`) and E45 (D4+D8 bottleneck auxiliary head, trained clean, +0.47pp, below target), the project redirected to a mechanistically distinct candidate: an **architecture-side** change rather than another deep-supervision loss/head variant.

## Motivation

E43 (Representation Change Audit) established that the bottleneck carries real, reproducible representation change associated with the coarse boundary problem, but found no evidence that change is spatially localized to boundary regions specifically (the interior-vs-boundary comparison was null after controlling a cell-count confound). The open question E43 left unresolved: is the bottleneck's context actually being *routed* to where the fine decoder needs it? v1/v2/v3's `enc1` skip connection (feeding `dec1`, the primary 64³ prediction path) is an unconditional concatenation — identical at every spatial location regardless of what the bottleneck "knows" about that location's difficulty.

## Architecture: `neuroscan_3d_v5.py`, `UNet3D_v5`

Extends `UNet3D_v3` (not v4 — v4's D8 head is a separate, closed line of investigation) with a single Attention-U-Net-style additive gate (`AttentionGate3D`) on the `enc1` skip:

```
g = W_g(bottleneck)              # (B, 16, 8, 8, 8)
g_up = upsample(g, size=enc1)    # (B, 16, 64, 64, 64)
x = W_x(enc1)                    # (B, 16, 64, 64, 64)
psi = sigmoid(W_psi(relu(g_up + x)))  # (B, 1, 64, 64, 64)
enc1_gated = enc1 * psi
```

`enc1_gated` replaces raw `enc1` in `cat1` (dec1's skip concatenation) only. Every other tensor in the trunk (enc2, enc3, bottleneck, dec2, dec3, aux heads, evidential/boundary heads) is computed identically to v3.

Params: 5,607,447 (v3: 5,602,822, +4,625 — negligible).

**Verified before training, not assumed:**
- Correct output shapes at all resolutions.
- **Ablation-safety**: forcing the gate to identity (`W_g=0, W_x=0, psi bias→sigmoid≈1`) reproduces v3's `probs` and `aux_probs3` outputs bit-for-bit (max abs diff 0.0).
- psi is not saturated at fresh init (range ~0.52–0.59, nonzero spatial variance) — gate starts in a live, learnable regime.

## Training (`experiments/exp_e12_eggo_m/e46/train_e46_attention_gate.py`)

No new loss term, no calibration needed — the gate is trained purely through the existing `seg_loss` and D4 deep-supervision loss (`lambda_ds3=0.9927`, reused unchanged). D2 not used, matching the project's established finding that D2 doesn't help beyond D4. Same protocol as every prior condition: seed 0, 30 epochs, AdamW, CosineAnnealingLR, mu=0.1, `validate()` unchanged.

New diagnostic: per-epoch psi (attention map) statistics logged to check the gate learns a real spatial pattern rather than collapsing toward an uninformative constant.

**Pre-declared kill condition**: dice < 0.5 after epoch 5 → immediate stop. Never triggered.

## Result

Full 30-epoch run, `AttnGate_seed0`, clean throughout, no instability.

| Epoch | val_dice |
|---|---|
| 29 (final) | **0.9102** (best) |
| 24 | 0.9087 |
| 28 | 0.9082 |

Psi diagnostic across training: mean drifted from ~0.30 (epoch 0) down to ~0.06–0.11 by the end, std staying ~0.07–0.18, min/max spanning nearly the full [0,1] range at every epoch checked — the gate learned a real, spatially-varying, non-degenerate pattern throughout, not a collapse to identity or to a trivial constant.

## Verdict vs. pre-declared criteria

- Canonical baseline (0.9063) → **+0.39pp**. Required ≥1.0pp (≥0.9163). **NOT MET.**
- D4-only (0.9096) → **+0.06pp** — a statistical tie, not a decisive secondary-check edge.
- E45's D4+D8 (0.9110) → E46 is marginally *below* (−0.08pp).
- Training stability: clean, no kill condition triggered.

**This is the third algorithmic direction in a row — one loss-reweighting (E44, killed), one new-head architecture change (E45, +0.47pp), one attention-routing architecture change (E46, +0.39pp) — landing in the same narrow +0.3–0.5pp band, all short of the +1pp bar.** The gate demonstrably learned something real (psi is non-trivial and stable), but that real, verified mechanism did not translate into a decisive Dice gain. Not reframed as a success.

## Implication for the next direction

The consistency of the ceiling across three structurally distinct mechanisms (loss schedule, new auxiliary head, attention routing) is itself a finding: it suggests the current single-seed, single-modification, incremental-change paradigm on this exact architecture/data regime may have a ceiling around +0.5pp, rather than any one idea being poorly executed. Per the project's kill/stop discipline, no further single-seed tuning of the attention-gate variant specifically. The next candidate direction should be evaluated against this pattern explicitly — either a qualitatively larger structural change, or a change to the evaluation protocol itself (e.g., multi-seed aggregation, ensembling) if genuine architectural headroom is judged to be exhausted at this scale.

## Artifacts on disk

- `neuroscan_3d_v5.py` (git-tracked)
- `experiments/exp_e12_eggo_m/e46/train_e46_attention_gate.py` (disk only, per `experiments/` gitignore convention)
- `experiments/exp_e12_eggo_m/e46/e46_full_run_log.txt` (disk only)
- `experiments/exp_e12_eggo_m/e46/runs/AttnGate_seed0/epoch_metrics.csv` (disk only, 30 rows + psi diagnostics)
- `experiments/exp_e12_eggo_m/e46/runs/AttnGate_seed0/checkpoints/` (best.pth + periodic, disk only, gitignored)
