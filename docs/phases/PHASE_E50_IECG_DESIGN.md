# Phase E50 — Internally-Estimated Counterfactual Gating (IECG): Design & Literature Positioning

## Motivation

E49's multi-seed check (see `PHASE_E49_CCABA_MULTISEED_VARIANCE.md`) revealed that CCABA's headline +0.51pp was largely seed noise (true mean +0.31pp, CI spanning −0.17pp to +0.79pp, statistically tied with D4-only). All four post-pivot mechanisms tried (E44–E49) sit in a band that, once noise is accounted for, cannot be confidently distinguished from zero improvement over D4-only. The user's explicit direction: try one qualitatively bigger structural change — not incremental, not a repeat of prior published work.

CCABA's core limitation, diagnosed directly: its causal-sensitivity signal (`w(frac_hat)`) is a **fixed, externally-fit curve**, computed once from a 125-subject offline audit (E48) and then frozen for the life of the model. It cannot adapt, cannot reflect what the model itself learns during training, and is a static approximation rather than a live measurement.

## Literature scan (2025–2026)

Searched for: internal self-probing counterfactual gating, learned counterfactual sensitivity estimation for MoE/routing, self-estimated sensitivity gates, differentiable internal counterfactual modules for medical segmentation.

**Findings — every counterfactual/causal method found is a post-hoc analysis tool, not a live trainable architectural component:**
- **"When Are Experts Misrouted? Counterfactual Routing Analysis in MoE Language Models"** (May 2026) — confirmed via direct fetch: compares a trained, **frozen** router's choices against sampled equal-compute alternatives, purely as an **external diagnostic**. Explicitly does not train a differentiable causal-sensitivity component; identifies "future optimization targets," does not integrate live estimation into the model.
- **TRACE-Seg3D** (2025, 3D glioma segmentation) — confirmed via direct fetch of the abstract: "counterfactual context auditing" is a **test-time/post-hoc reliability-assessment tool**, applied *after* segmentation to flag unreliable predictions under simulated context shift. Not a trainable gating module; does not alter the segmentation model's own forward computation.
- **CausalX-Net** (checked previously for E49): do-calculus intervention framework for **modality-level attribution and spurious-correlation removal**, not for live sensitivity-based gating.
- **Optimal Ablation for Interpretability**, **causal head gating** (NLP): both are analysis/interpretability frameworks for understanding trained, frozen models, not trainable segmentation-architecture components.

**Gap confirmed**: no method found trains a module, live during the forward pass, to predict its own causal sensitivity to an internal ablation and uses that live prediction to gate computation — jointly optimized with the task loss, at both train and inference time (with the ablation itself confined to training only, since it is unneeded once the sensitivity head is trained). This is a genuinely different mechanism class from CCABA (static external curve) and from every prior counterfactual method found (post-hoc, frozen-model diagnostics).

## Mechanism: IECG

Implemented in `neuroscan_3d_v7.py`, `UNet3D_v7` (extends v3, isolating IECG's own effect from E46's attention gate or E49's CCABA):

1. **Real bottleneck** computed as always (encoder shared, unchanged from v3).
2. **On-the-fly simulated ablation** (training only): the exact same intervention E48 already causally validated (full zeroing of the bottleneck) is applied, and the **decoder is replayed a second time** from the ablated bottleneck (encoder computation is NOT duplicated — factored via `_run_decoder_from_bottleneck`, called twice from shared encoder features).
3. **Real, per-subject, per-batch sensitivity label**: `s_true = Dice(probs_intact) − Dice(probs_ablated)` — literally E48's own metric, computed fresh every training step, not from a static offline table.
4. **SensitivityPredictorHead**: reads ONLY the real bottleneck (never the ablated one), predicts `s_hat`, trained via MSE against `s_true` (stop-gradient target — `s_true` is a label, not a differentiable path through the ablated pass).
5. **Gating**: `bottleneck_amp = bottleneck * (1 + alpha·s_hat)`, same amplification form as CCABA, but `s_hat` is now a live, per-input, jointly-learned estimate rather than a fixed external curve.
6. **At inference**: the ablation branch never runs — `compute_sensitivity_target=False` skips it entirely, so IECG costs nothing extra at deployment beyond one small head (`SensitivityPredictorHead`), verified to add only 8,258 parameters vs. v3.

## Verified properties (before training)

- **Ablation-safety**: forcing `iecg_alpha=0` in inference mode reproduces v3's `probs`/`aux_probs3`/`aux_probs2` bit-for-bit (max abs diff 0.0).
- **Primary-path independence**: `probs` is identical whether or not the ablation/sensitivity-target branch runs (confirmed via direct comparison) — masks and the diagnostic branch never leak into the segmentation path.
- **Training-time cost, measured directly** (not assumed): batch=8, with the ablation branch: 1.63s/step, 5.94GB peak GPU memory (vs. 0.66s/step, 5.00GB without) — roughly 2.5× per-step time, comfortably within the 8.15GB GPU budget. Disclosed as a real training-time-only overhead; inference cost is unchanged from v3 plus one negligible head.
- **Calibration**: `lambda_sens` (the sensitivity-head's MSE loss weight) gradient-calibrated to 0.0841 via `calibrate_lambda_sens.py` — value-matching would have produced a 221× gradient blowup (rejected per the E34 safeguard, same discipline as every prior lambda calibration).

## Pre-declared evaluation plan (multi-seed from the start, per the E49 policy)

Given E49's finding that single-seed Dice claims are not reliable at this project's noise scale (~0.19pp std), IECG will be evaluated on **3 seeds from the outset**, not as an afterthought. Success requires the **mean** improvement over the canonical baseline to reach ≥1.0pp with a 95% CI that does not include D4-only's own level, not a single lucky seed.

## Follow-up (if IECG shows a real, seed-robust improvement)

Re-run E48's own bottleneck-ablation audit methodology on the trained model and compare `s_hat`'s learned pattern against the real per-subject ablation drop it was trained to predict — direct evidence of whether the live sensitivity estimate actually tracks the true causal signal, closing the loop between diagnosis and mechanism in a way no single-seed Dice number alone can support.
