# Phase E50 — IECG Result (3-Seed Evaluation)

## Context

Following E49's multi-seed finding (CCABA's headline +0.51pp was largely seed noise; true mean +0.31pp, statistically tied with D4-only), the user directed a bigger, genuinely novel structural change rather than another incremental single-lever tweak. IECG (Internally-Estimated Counterfactual Gating, `UNet3D_v7`) was designed and confirmed, via a 2025-2026 literature scan, to be a genuinely new mechanism class: unlike CCABA's static externally-fit calibration curve, IECG computes a live, per-input, jointly-trained causal-sensitivity estimate via an on-the-fly bottleneck ablation replayed through the decoder every training step (see `PHASE_E50_IECG_DESIGN.md`).

Per the project's own new policy (established after E49), IECG was evaluated on 3 seeds **from the start**, not as an afterthought.

## Training

All 3 seeds (0, 1, 2) trained cleanly, no instability, no kill condition triggered, chained automatically. Training-time cost was real but manageable (~2.5x per-step compute vs. a single-decoder-pass model, measured directly before committing to the full run; total wall-clock across 3 seeds was substantially longer than any prior condition's multi-seed check).

## Result

| Seed | best_val_dice |
|---|---|
| 0 | 0.9081 |
| 1 | 0.9030 |
| 2 | 0.9084 |
| **Mean** | **0.9065** |
| Std (n=3) | 0.0030 (≈0.30pp) |
| 95% CI | [0.8989, 0.9140] |

## Verdict vs. pre-declared criteria

- Mean improvement over canonical baseline (0.9063): **+0.02pp** — statistically indistinguishable from zero.
- vs. D4-only (0.9096): **−0.31pp** — IECG underperforms D4-only on average.
- vs. CCABA's own 3-seed mean (0.9094, E49): **−0.29pp**, and IECG was lower than CCABA on all three individually seed-matched comparisons (0 vs 0, 1 vs 1, 2 vs 2), though a paired t-test at n=3 does not reach significance (p=0.11 — underpowered to confirm at this sample size, not evidence of no difference).

**Required: ≥1.0pp mean improvement with a 95% CI that does not overlap D4-only. NOT MET, and not close.**

## Honest interpretation

IECG does not work, at least not in this exact instantiation. Despite being architecturally larger, more computationally expensive (~2.5x per-step training cost), and more novel than every mechanism tried before it (genuinely differentiated from all 2025-2026 literature: no prior method trains a live causal-sensitivity gate jointly with the task, as confirmed via direct checks of the closest candidates — counterfactual MoE routing analysis and TRACE-Seg3D, both post-hoc frozen-model diagnostics), it landed at essentially zero net improvement and trended *worse* than the far simpler CCABA mechanism across all three matched seeds.

This is not spun as a partial win. A plausible explanation, offered honestly rather than as an excuse: the added machinery (a full second decoder replay per training step, a jointly-trained sensitivity head with its own loss term) may be diverting optimization capacity and gradient signal away from the primary segmentation task without a compensating benefit — bigger and more novel is not the same as more effective, and this result is direct evidence of that distinction. The mechanism's own internal consistency check (s_hat tracking s_true closely throughout training, confirmed in per-epoch logs) shows the sensitivity head DID learn to predict real causal sensitivity accurately — the mechanism worked as designed internally, it simply didn't help the actual task.

## Implication

Five mechanisms have now been tried since the strategic pivot (E44 killed, E45 +0.47pp single-seed, E46 +0.39pp single-seed, E49/CCABA +0.31pp 3-seed mean, E50/IECG +0.02pp 3-seed mean). The trend across the two properly multi-seed-evaluated conditions (CCABA, IECG) is that neither clears even D4-only's own baseline with confidence, and the more structurally ambitious mechanism (IECG) performed worse, not better, than the simpler one (CCABA). This is now reasonably strong evidence that further single-mechanism architectural modifications on this exact 1,251-subject/FLAIR-only/64³ setup are unlikely to independently clear the +1.0pp bar, regardless of novelty or structural ambition.

## Recommendation

Per the project's own kill/stop discipline: this specific mechanism (IECG) should not be iterated on further. Before attempting a sixth mechanism, the more productive use of effort is likely a candid reassessment of whether the +1.0pp bar is achievable at all within this project's current constraints (single architecture family, FLAIR-only, this dataset size), or whether the project's real contribution for a paper should center on the causal-diagnostic methodology itself (E43→E47→E48's genuine, rigorous causal chain) and the disclosed multi-seed variance correction (E49→E50), both of which are legitimate, rare-for-the-field methodological contributions independent of whether any single architectural fix crosses +1pp.

## Artifacts on disk

- `neuroscan_3d_v7.py` (git-tracked)
- `experiments/exp_e12_eggo_m/e50/calibrate_lambda_sens.py`, `train_e50_iecg.py` (disk only, per `experiments/` gitignore convention)
- `experiments/exp_e12_eggo_m/e50/e50_seed{0,1,2}_run_log.txt` (disk only)
- `experiments/exp_e12_eggo_m/e50/runs/IECG_seed{0,1,2}/epoch_metrics.csv` (disk only, 30 rows each)
