# Phase E49 — CCABA Result (Causally-Calibrated Adaptive Bottleneck Amplification)

## Context

Following the diagnosis-driven strategy (E43 null → E47 causal null → E48 reversed causal finding: small lesions depend more on the bottleneck, ρ=−0.454, p<0.001), and a 2025–2026 literature scan confirming no prior work fits a gating mechanism's conditioning function to a measured causal-ablation dependency curve (see `PHASE_E49_CCABA_DESIGN.md`), CCABA was built and trained: `UNet3D_v6`, extending v3 with a bottleneck-amplification mechanism whose size-conditioning function `w(frac_hat)` is a fixed log-linear fit to E48's real 125-subject data, scaled by a single learnable trust parameter `alpha`.

## Training

Full 30-epoch run, `CCABA_seed0`, seed 0, identical protocol to every prior condition (AdamW, CosineAnnealingLR, mu=0.1). New calibrated auxiliary loss `lambda_frac=0.0203` (gradient-matched to a deliberately light 0.10 target ratio — value-matching would have given a 25.4× gradient blowup, caught and rejected per the E34 safeguard, see `calibrate_lambda_frac.py`).

Clean throughout: no instability, kill condition (dice<0.5 after epoch 5) never triggered. `ccaba_alpha` declined smoothly and monotonically from its 0.1 init to ~0.081 by the end (a stable, non-degenerate operating point — not collapsing to 0, not exploding). `frac_loss` converged to near-zero (4×10⁻⁵ by epoch 29), confirming the size-proxy head learned a meaningful, accurate occupancy estimate as intended for a light auxiliary signal.

## Result

| Epoch | val_dice |
|---|---|
| 28 (best) | **0.9114** |
| 29 (final) | 0.9111 |
| 25 | 0.9101 |

## Verdict vs. pre-declared criteria

- Canonical baseline (0.9063) → **+0.51pp**. Required ≥1.0pp (≥0.9163). **NOT MET.**
- D4-only (0.9096) → **+0.18pp** — real, but modest.
- vs. E45 (D4+D8, 0.9110) → **+0.04pp**, essentially tied.
- vs. E46 (attention gate, 0.9102) → **+0.12pp**, a small edge.

**CCABA is the best of the four post-pivot algorithmic attempts (E44 killed, E45 +0.47pp, E46 +0.39pp, E49 +0.51pp)** — but only marginally, and still well short of the +1.0pp bar. Not reframed as a success.

## Strategic implication (important for the paper and for next steps)

Four structurally and mechanistically distinct interventions — a loss-reweighting schedule (E44, killed), a new deep-supervision head (E45), a global attention-routing gate (E46), and a causally-derived, literature-differentiated amplification mechanism (E49) — have now all landed in the same narrow **+0.3–0.5pp band**. CCABA had the strongest diagnostic grounding and clearest novelty argument of the four, and still didn't break out of that band.

This is now strong evidence that the ceiling is **not** about the quality or novelty of any single incremental mechanism. It is more likely structural: a **single-seed, single seed's worth of training signal, on this exact 1,251-subject/FLAIR-only/64³ setup**, may simply not carry enough statistical power for any one architectural tweak to move Dice by a full point. Two candidate explanations, not yet distinguished:
1. **Seed noise**: +0.3–0.5pp may be within the noise band of a single-seed run, and a genuinely real +1pp-capable mechanism could be getting washed out or under/over-estimated by seed variance that's never been measured in this project (every condition so far is n=1 seed).
2. **A true architectural ceiling** at this scale/data regime, where no single incremental change (however well-motivated) can close the remaining gap alone, and only a combination of independently-validated mechanisms (e.g., CCABA + attention gate stacked, or multi-seed ensembling) could plausibly cross +1pp.

## Recommended next step

Before spending compute on a fifth single mechanism, this ceiling pattern itself should be investigated directly:
1. **Multi-seed variance check** on the best single mechanism found so far (CCABA) — a cheap, 2-3 extra seeds — to establish whether +0.5pp is a stable estimate or falls within noise that could occasionally exceed +1pp, or never will.
2. If seed noise is ruled out as the explanation, consider **combining CCABA with the E46 attention gate** (both independently verified, non-conflicting mechanisms targeting different aspects — amplification magnitude vs. spatial routing) as a genuinely combined, not incremental-single, next architecture.

Per this project's own kill/stop discipline: no further single-seed tuning of CCABA alone.

## Artifacts on disk

- `neuroscan_3d_v6.py` (git-tracked)
- `experiments/exp_e12_eggo_m/e49/calibrate_ccaba.py`, `calibrate_lambda_frac.py`, `train_e49_ccaba.py` (disk only, per `experiments/` gitignore convention)
- `experiments/exp_e12_eggo_m/e49/e49_full_run_log.txt` (disk only)
- `experiments/exp_e12_eggo_m/e49/runs/CCABA_seed0/epoch_metrics.csv` (disk only, 30 rows)
- `experiments/exp_e12_eggo_m/e49/runs/CCABA_seed0/checkpoints/` (disk only, gitignored)
