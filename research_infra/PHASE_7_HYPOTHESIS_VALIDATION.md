# Phase 7: Hypothesis Validation Report

**Role**: Research Scientist, hypothesis validation. Objective is to determine whether the proposed research direction (multi-objective optimization dynamics in uncertainty-aware MRI segmentation) has enough evidence to justify becoming a semester research project. This report attempts to falsify the hypothesis before supporting it.

**Run reference**: `OptimalModel_FrozenSelector_20260727_012405`, GPU: NVIDIA GeForce RTX 5050 Laptop (8GB), CUDA 12.8, PyTorch 2.11.0+cu128. Plots archived at `research_infra/run_20260727_012405_plots/`; raw quantitative summary at `research_infra/run_20260727_012405_analysis_summary.json`.

---

## Experimental Setup

**What was run**: The baseline NeuroScan training pipeline, unmodified in architecture, loss functions, optimizer, or scheduler. The only changes applied were:
1. Two pre-existing environment/portability bugs fixed so the pipeline could execute at all on this machine (documented below, not research-scope changes).
2. Epoch counts overridden via environment variables (`MAE_EPOCHS_OVERRIDE=20`, `SEGMENTATION_EPOCHS_OVERRIDE=30`) to bound this to an exploratory run, per instruction.
3. The pre-existing, already-verified-as-non-invasive diagnostic instrumentation (Phases 3-5) enabled: `ENABLE_LOSS_DIAGNOSTICS`, `ENABLE_GRADIENT_DIAGNOSTICS`, `ENABLE_GRADIENT_SIMILARITY`, all sampled at every batch (dataset is tiny — 8 batches/epoch — so full sampling was affordable).

**Prerequisite fixes applied** (both documented in-line in `final_model.py` and in `research_infra/PHASE_1_CODEBASE_AUDIT.md`/this file; neither touches architecture, loss, optimizer, or scheduler):
- **Dataset path bug**: the original `DATA_PATH` pointed at a nonexistent nested folder (`Dataset/PediMS/PediMS`) and assumed a `{patient}/{modality}/processed/` layout; the real, downloaded PediMS dataset lives at `<repo_root>/PediMS/{patient}/{timepoint}/processed/` with modality encoded in filenames. Fixed to discover the same intended data (FLAIR primary modality, mask with Consensus.nii fallback — same selection priority as the original code).
- **Environment-compatibility bug**: `sklearn.metrics` (used only for validation-time precision/recall/F1 reporting, never in the training loop) transitively imports a scipy Cython/pythran extension that a Windows Application Control policy on this machine blocks. Replaced with a numpy-only equivalent using the identical formulas and `zero_division=0` semantics. A second instance of the same class of issue (`pandas`, used for CSV logging) was resolved by running under Python 3.14 instead of 3.12 — empirically, Python 3.14's compiled extensions were not blocked by the same policy, while Python 3.12's were, for reasons outside this project's scope.

**Actual run achieved**: MAE pretraining ran its full 20 epochs. Segmentation fine-tuning ran 29 of the planned 30 epochs before the pipeline's own pre-existing early-stopping criterion (patience=20 epochs without validation-Dice improvement — not modified) triggered. Best validation Dice recorded: 0.0060.

**Important context for interpreting everything below**: this run's segmentation loss (Dice ≈ 0.99, FocalTversky ≈ 0.99 throughout) shows essentially no improvement over 29 epochs — the model is not meaningfully learning the segmentation task in this exploratory run (see `loss_evolution.png`, `phase_comparison.png`). Likely contributors, none investigated further here since diagnosing training failure was not this report's objective: severe lesion/background class imbalance, only 20 MAE epochs and 22 training volumes, and/or the early-stopping patience being reached quickly on a genuinely hard, small-data task. This matters because **all findings below describe optimization *dynamics* in a regime where the primary task loss is nearly flat** — the gradient behavior observed is not necessarily representative of a well-converging training run, and this caveat applies to every answer that follows.

---

## Question 1: Do gradient norms remain balanced, or does one loss dominate?

**Quantified.** Mean gradient-norm contribution per loss term, isolated via separate backward passes (`grad_norm_loss_dice`, `grad_norm_loss_ft`, `grad_norm_loss_evid` in `gradient_metrics.csv`, averaged over all 232 logged batches):

| Loss | Mean isolated gradient norm | Share of total |
|---|---|---|
| Dice | 0.0269 | 30.5% |
| FocalTversky | 0.0335 | 37.9% |
| Evidential | 0.0278 | 31.5% |

**No single loss dominates the gradient magnitude** — the three-way split is close to even (30/38/32%), despite Evidential's loss *value* being ~25x smaller than Dice/FocalTversky's (0.04 vs 0.99, from `training_metrics.csv`). This is the first concrete, quantified finding: **the tiny evidential loss weight (λ=1e-3) does not translate into a proportionally tiny gradient contribution.**

By model component (encoder+CBAM vs decoder), the split is close to balanced: encoder+CBAM mean grad norm 0.0275, decoder mean grad norm 0.0304 (ratio 1.11), see `q1_decoder_to_encoder_ratio` in the analysis summary. However, this masks large batch-to-batch variance (`gradient_norms.png`): encoder+CBAM shows sporadic large spikes (up to 0.555, an order of magnitude above its typical ~0.03 baseline) while the decoder stays comparatively flat throughout. This spikiness, not a sustained imbalance, is the more notable feature of Q1's evidence — addressed further under Q6.

---

## Question 2/3: Does gradient cosine similarity remain constant, or does it evolve? Positive, near-zero, negative, stage-dependent?

**Quantified and visualized** (`gradient_similarity.png`, `gradient_similarity_heatmap.png`, `q2_q3_*` in the analysis summary).

- **Dice vs. FocalTversky**: consistently **positive**, epoch-mean values ranging 0.38–0.71 across all 29 epochs, batch-level values spiking as high as ~1.0 and rarely dipping to ~0.3-0.4 (never epoch-mean-negative). Correlation of this similarity with epoch number: **r = −0.154** (weak, and in the direction of a slight *decrease* over training, not an increase — see caveat below).
- **Dice vs. Evidential, FocalTversky vs. Evidential, Hybrid vs. Evidential**: consistently **near zero** at the epoch-mean level (typically within ±0.02 of zero) across all 29 epochs. Correlation with epoch: weakly positive (r = 0.12–0.19) but the values themselves stay tiny and noisy — a correlation coefficient here reflects a marginal statistical trend, not a visible, meaningful shift (see the heatmap: these three columns stay visually flat/white across all 29 rows).
- **Sign changes across epoch-means**: Dice-vs-FT shows **zero** sign changes over 29 epochs (always positive). The three Evidential-paired similarities show 14–17 sign changes each out of 28 possible transitions — but given their magnitude sits within noise-band of zero, this reflects **noise crossing zero, not a systematic oscillation** (see per-batch fraction-negative under Q6 for the same caution).

**Verdict for Q2/3**: Dice-vs-FocalTversky similarity is stable and positive throughout (no stage-dependence, no phase transition). Evidential's relationship to the other two is stable and near-zero (orthogonal) throughout — also no stage-dependence, no phase transition. **This is evidence against the "evolving relationship" / "phase transition" component of the hypothesis**, at least in this run.

---

## Question 4: Are some network components influenced much more strongly by certain losses?

**Quantified** (`q4_component_gradient_summary`, cross-checked against per-parameter-normalized values, `layerwise_gradients_epoch29.png`).

| Component | Mean gradient norm (raw, per-tensor) |
|---|---|
| Decoder evidential/causal heads (anatomy/pathology/noise) | **0.00672** |
| Decoder probability head | 0.00406 |
| Decoder upsample/skip | 0.00080 |
| Encoder ResBlocks (incl. Mini-Swin attention) | 0.00092 |
| CBAM | 0.00014 |
| Conv2D5Stem (2.5D fusion) | 0.00064 |
| AdaptiveSliceSelector | **no gradient recorded (see below)** |

**The evidential/causal heads carry the single largest gradient of any component in the network** — larger even than the primary probability head, and roughly 7x larger than the average encoder ResBlock. This was checked against a possible confound (fewer parameters mechanically producing different norm scales): normalizing by parameter count, `decoder.pathology_head.weight` and `decoder.anatomy_head.weight` still show 2–5x higher per-parameter gradient magnitude than `decoder.prob_head`, and roughly 60-70x higher than a typical encoder conv layer (0.0024 vs 0.0011 vs 0.00003-0.00004; full breakdown in the Bash session output archived alongside this report). **This is a real, non-artifactual finding**: the small uncertainty-estimation subnetwork receives disproportionately strong gradient signal relative to its tiny (λ=1e-3) contribution to the total loss *value*.

One important structural caveat: the evidential heads are a largely separate parameter path (branching off `decoder.causal_shared`, which itself branches off the same upsampled decoder features the probability head uses) — so this is not literally the evidential loss "stealing" gradient that would otherwise go to the segmentation head; the parameter sets are mostly disjoint. What it does show is that **the optimizer is being asked to take much larger steps (relatively) in the small auxiliary-uncertainty subnetwork than anywhere else in the model**, which is worth further investigation (see Q7).

**`AdaptiveSliceSelector` recorded zero gradient across all 232 logged batches of this real run** — independently reconfirming the finding from `PHASE_1_CODEBASE_AUDIT.md` (there verified via a standalone reimplementation and a synthetic-data end-to-end test) in a third, fully independent context: a real 29-epoch training run on the real dataset. This is not part of the multi-objective-loss hypothesis under test here, but it is directly relevant to the companion Project A assessment, and its reproducibility across three independent verifications (synthetic unit test, synthetic end-to-end test, and now this real run) makes it a very high-confidence finding.

---

## Question 5: Are there optimization phases (early/mid/late)?

**Quantified** (`q5_phase_stats`, `q5_grad_phase_stats`, `phase_comparison.png`), splitting the 29 epochs into early (1–9), mid (10–18), late (19–29) thirds.

| Phase | Mean Dice loss | Mean FocalTversky loss | Mean Evidential loss | Mean total grad norm |
|---|---|---|---|---|
| Early | 0.99596 | 0.99501 | 0.05073 | 0.04331 |
| Mid | 0.99394 | 0.99275 | 0.04389 | 0.04148 |
| Late | 0.99245 | 0.99100 | 0.04083 | 0.05193 |

Dice and FocalTversky loss decline by less than 0.4% (relative) from early to late phase — **not a meaningful phase transition, just very slow, monotonic, and small drift**, consistent with the "task loss essentially flat" observation noted in the Experimental Setup caveat. Evidential loss declines by about 20% relative (0.0507→0.0408) — proportionally the largest phase-to-phase change of any loss term, though its absolute magnitude remains small throughout. Total gradient norm is not monotonic across phases (dips in mid, rises in late) — driven by the encoder-side spikes noted under Q1/Q6, not by a systematic phase effect.

**Verdict for Q5**: No evidence of distinct optimization phases with different objectives dominating at different times. The one directional trend (Evidential loss declining faster, proportionally, than Dice/FocalTversky) is real but small in absolute terms and does not correspond to a shift in which loss dominates gradient magnitude (Q1's ~30/38/32% split does not show a comparable phase-dependent shift in `q1_per_loss_grad_norm_mean` when checked per-phase — the aggregate 29-epoch numbers already reported are not substantially different phase-to-phase; not tabulated separately here since the difference did not reach a magnitude worth a separate table).

---

## Question 6: Is there evidence of optimization inefficiency?

**Quantified** (`q6_inefficiency_signals`).

- **Coefficient of variation of total gradient norm**: 1.09 (i.e., standard deviation exceeds the mean) — driven almost entirely by the encoder-side spikes visible in `gradient_norms.png` (min 0.0254, max 0.555, a 22x range). This is the **strongest single piece of evidence for "inefficiency"-adjacent behavior** in this run: a small number of batches produce encoder gradients an order of magnitude larger than typical. Given the dataset has only 22 training volumes at batch size 3, this is at least as consistent with **ordinary small-dataset batch-composition variance** (a batch happening to contain a volume with unusually large/small lesion burden) as with a genuine optimization pathology — this run's diagnostics cannot distinguish those two explanations, and doing so would require per-sample (not just per-batch) gradient attribution, which is not currently instrumented.
- **No vanishing gradients**: zero batches recorded a total gradient norm below 1e-6.
- **No exploding gradients** in the total-norm sense: zero batches exceeded a norm of 10 (the largest observed, 0.555, is far below that threshold) — the spikes noted above are large *relative to this run's own baseline*, not large in an absolute, destabilizing sense.
- **"Negative similarity" fraction**: 43% of sampled batches showed negative Dice-vs-Evidential similarity, 49% showed negative FocalTversky-vs-Evidential similarity. **Read with caution**: as shown in `gradient_similarity.png` and the heatmap, these negative values cluster tightly around zero (typically within ±0.05, one outlier at −0.32 in epoch 1-2). A near-50% negative fraction sounds like "conflict half the time," but the actual magnitudes describe **noise scattered symmetrically around an orthogonal (≈0) relationship, not a systematic or strong opposing-gradient conflict**. Reporting the raw fraction without this context would overstate the finding.
- **No dead objectives**: all three losses (Dice, FocalTversky, Evidential) remained active and non-zero throughout, per `training_metrics.csv`.
- **One dead component, not an objective**: `AdaptiveSliceSelector`, as discussed under Q4 — a mechanism-level dead-parameter finding, not a loss-level one.

**Verdict for Q6**: The strongest, most defensible inefficiency-adjacent signal is the encoder-side gradient-norm spikiness (CV > 1), and the disproportionate gradient share captured by the evidential/causal heads (Q4). Both are real, quantified, and reproducible within this run. Neither rises to "persistent conflict" or "unstable divergence" — total gradient norm never exploded or vanished, and the task loss, while not improving much, did not diverge either.

---

## Question 7: Can the observed behavior naturally motivate an optimization improvement?

Two specific, evidence-grounded candidates emerge from the above — and one clear non-candidate.

**Candidate 1 — component-specific gradient scaling for the evidential/causal heads.** Q4's finding (disproportionate per-parameter gradient magnitude in a subnetwork contributing only λ=1e-3 to the total loss value) is concrete and reproducible. This motivates investigating **per-parameter-group learning-rate or gradient-norm scaling specific to the uncertainty-estimation branch** — distinct from, and narrower than, "dynamic loss-weighting between Dice/FocalTversky/Evidential" as commonly proposed in multi-task-learning literature, because the finding here is about gradient *magnitude in a specific subnetwork*, not about the three losses fighting for shared parameters.

**Candidate 2 — batch-level gradient-norm stabilization (e.g. clipping, or investigating batch composition).** Q1/Q6's spikiness finding (CV 1.09, 22x range in encoder gradient norm) motivates looking at whether **gradient clipping or a batch-composition-aware sampling strategy** would stabilize training — but this report cannot distinguish "genuine optimization pathology" from "ordinary variance in a 22-sample training set," so this candidate is weaker and would need a targeted follow-up (e.g., per-sample gradient logging, not just per-batch) before being worth pursuing on its own.

**Non-candidate — conflict-aware multi-loss optimization (e.g. PCGrad, GradNorm, gradient-surgery methods).** These methods are designed to resolve *persistent, systematic gradient conflict between loss terms competing for the same parameters*. This run's evidence does not support that premise: Dice-vs-FocalTversky gradients are consistently positively aligned (never epoch-mean-negative across 29 epochs), and Dice/FocalTversky-vs-Evidential gradients are consistently near-zero (orthogonal, not opposed) throughout training, with no stage-dependent drift toward conflict. **Applying a conflict-resolution method here would be solving a problem this data does not show exists.**

---

## Final Report

### 1. Was the research hypothesis supported?

**Partially supported.**

The hypothesis as stated — *"the optimization process contains measurable interactions, conflicts, stage-dependent behaviors, or imbalances that can be exploited to design a better optimization strategy"* — is a disjunction of several distinct sub-claims. The evidence splits cleanly across them:

- **"Interactions" / "imbalances"**: **Supported.** Concrete, reproducible, non-artifactual (parameter-count-normalized) finding that the evidential/causal heads carry disproportionate gradient magnitude relative to their loss weight (Q4), and that gradient-norm variance is high and encoder-side-spiky (Q1/Q6).
- **"Conflicts"**: **Not supported.** Gradient similarity between all three loss pairs stayed non-negative-on-average (Dice-FT) or near-zero/orthogonal (either-vs-Evidential) across the entire run, with no sustained negative (conflicting) relationship at any point (Q2/3, Q6).
- **"Stage-dependent behavior" / "phase transitions"**: **Not supported.** Loss composition and gradient-similarity relationships were essentially flat across early/mid/late thirds of training (Q5); the one real trend (Evidential loss declining faster proportionally) did not correspond to a shift in gradient-magnitude dominance.

Because the hypothesis explicitly offers "imbalances" as a satisfying condition on its own (via the "or" in its phrasing), and imbalance evidence is real and quantified, the hypothesis is not rejected outright — but two of its three other named phenomena (conflict, phase-dependence) are directly contradicted by this run's evidence, so "fully supported" would overstate the result.

### 2. What optimization phenomena were actually observed?

- A stable, positive gradient-alignment relationship between Dice and FocalTversky (expected, given they are both computed from the same TP/FP/FN quantities on the same predictions — `loss_dice` and `loss_focal_tversky` correlate at r=0.99 in `correlation_matrix.png`, meaning this "cooperation" finding is close to mathematically guaranteed rather than a nontrivial emergent property).
- A stable, near-zero (orthogonal) gradient relationship between the Evidential loss and the two segmentation losses.
- A pronounced, parameter-count-normalized-confirmed gradient-magnitude imbalance favoring the small evidential/causal-uncertainty heads over the primary segmentation head and the encoder.
- High batch-to-batch gradient-norm variance concentrated in the encoder, of uncertain origin (genuine dynamic vs. small-dataset batch-composition noise).
- Independent (third) confirmation that `AdaptiveSliceSelector` receives zero gradient throughout real training — a mechanism-level finding, not an optimization-dynamics one, but worth carrying forward.
- Essentially flat task-loss (Dice/FocalTversky) across all 29 epochs — the training run itself did not demonstrate meaningful convergence, which qualifies every dynamics finding above.

### 3. Are the observations reproducible?

**Partially, within the scope actually tested.** The `AdaptiveSliceSelector` zero-gradient finding is reproducible across three independent contexts (standalone module test, synthetic end-to-end test, this real run) and is the single most reproducible finding in this entire investigation. The multi-objective-loss findings above come from **one run only** — no repeated-seed or repeated-run variance analysis was performed, so whether the specific numeric values (30/38/32% gradient share, CV=1.09, etc.) are stable across different random seeds or different train/val splits is untested and should not be assumed. The qualitative pattern (Dice-FT positive, Evidential-vs-rest near-zero, evidential heads carrying outsized gradient) is strong enough within this run's own internal consistency (stable across all 29 epochs, not a one-off artifact of a single epoch) that it is unlikely to be pure noise, but formal reproducibility would require at least 2-3 additional runs with different seeds.

### 4. What evidence is strongest?

Ranked by combination of magnitude, independent corroboration, and robustness to alternative explanations:
1. Evidential/causal-head gradient-magnitude imbalance (Q4) — confirmed after controlling for parameter count, the single strongest quantitative finding specific to this hypothesis.
2. Stable positive Dice-vs-FocalTversky alignment across all 29 epochs, zero sign changes (Q2/3) — though tempered by the near-tautological relationship between the two loss formulas.
3. `AdaptiveSliceSelector` zero-gradient finding — strongest in terms of reproducibility, though outside this specific hypothesis's scope.

### 5. What evidence contradicts the hypothesis?

- No epoch-mean-negative gradient similarity between any loss pair, at any point in training (contradicts "conflict").
- No shift in which loss dominates gradient share across early/mid/late phases (contradicts "stage-dependent behavior" / "phase transition").
- No exploding or vanishing total gradient norm at any point (contradicts a strong reading of "unstable convergence").

### 6. Would these observations justify proposing a new optimization strategy?

**A narrow one, yes; a broad one, not on this evidence.** The data justifies investigating **targeted gradient/learning-rate scaling for the evidential-uncertainty subnetwork specifically** (Q4/Q7's Candidate 1) — this is a concrete, falsifiable, well-motivated next experiment. It does **not** justify a general conflict-aware multi-objective optimizer (e.g., PCGrad-style gradient surgery), because the conflict this class of method is designed to resolve was not observed.

### 7. If yes, what classes of optimization methods appear justified?

- **Per-component (not per-loss) gradient/learning-rate scaling**, applied specifically to the decoder's evidential/causal heads relative to the rest of the network — motivated directly by Q4.
- **Gradient-norm monitoring/clipping investigation**, motivated by Q1/Q6's spikiness — contingent on a follow-up that distinguishes genuine dynamics from small-dataset batch-composition noise (would require per-sample, not per-batch, gradient logging, which is not yet instrumented).

Explicitly **not** justified by this evidence: dynamic inter-loss weighting schedules, conflict-aware gradient surgery (PCGrad, GradNorm, CAGrad, or similar), or any method premised on resolving competition between Dice/FocalTversky/Evidential for shared parameters — this run found no such competition.

**No implementation of any of the above was performed, per the scope of this task.**

---

## Explicit Limitations of This Report

- Single run, single seed, no statistical replication.
- Training did not converge in any meaningful sense (Dice loss ≈ flat at 0.99 throughout) — all dynamics described are dynamics of a stalled optimization, and may not generalize to a run that actually learns the task.
- Encoder gradient-spike origin (genuine dynamic vs. batch-composition artifact) is not resolved by the current instrumentation and would require per-sample gradient attribution.
- GPU memory profiling was not instrumented for this run (no OOM occurred at batch size 3 on the 8GB RTX 5050; peak usage was not logged).
- The MAE pretraining phase's own loss/gradient dynamics were not analyzed in this report (out of scope — the hypothesis concerns the multi-objective segmentation-phase losses specifically).
