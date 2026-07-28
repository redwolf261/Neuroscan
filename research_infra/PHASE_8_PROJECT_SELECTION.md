# Phase 8: Comparative Research Validation — Final Project Selection

**Objective**: Determine, through controlled experiments against an identical baseline, whether the Slice Sampling project or the Multi-Objective Optimization project has stronger empirical evidence to justify becoming the semester research project. Recommendation is based only on measured evidence below.

**Deliverables**: `baseline_metrics.csv`, `sampling_metrics.csv`, `optimization_metrics.csv`, `comparison_table.csv`, six figures in `phase8_figures/`, this report. Experiment code in `research_infra/phase8/`.

---

## Experimental Design

All three experiments share, by construction, the identical:
- MAE-pretrained encoder checkpoint (from the Phase 7 run — the only pretraining performed; not repeated three times, to eliminate pretraining-randomness as a confound between experiments).
- Model architecture, loss functions (HybridLoss + EvidentialBetaLoss), optimizer type (AdamW, same two learning rates), scheduler (CosineAnnealingLR), and data split.
- Fixed 25-epoch budget with **early stopping disabled** — every experiment runs the full 25 epochs, so no experiment can differ from another merely by stopping at a different point.
- `ENABLE_GRADIENT_DIAGNOSTICS=1` (the same, already-verified-non-invasive instrumentation from Phases 3–5), so gradient-norm measurements are collected identically in all three.

**The one controlled variable per step**:
- **Baseline**: `AdaptiveSliceSelector` in its existing, unmodified `adaptive` mode (the learned-but-non-gradient-trained scorer + top-k selection, exactly as shipped).
- **Sampling prototype**: `AdaptiveSliceSelector` switched to a `uniform` mode — a fixed, evenly-spaced set of 9 indices out of 64, computed once, no scorer network involved in the selection decision at all (its parameters remain in the model, unused). This is the simplest possible non-learned policy, and is directly motivated by Phase 7's finding that the current mechanism does not learn a data-dependent selection anyway (see `research_infra/PHASE_1_CODEBASE_AUDIT.md` §6). Implemented as an additive `selection_mode` flag on `AdaptiveSliceSelector`, default-preserving.
- **Optimization prototype**: baseline's `adaptive` selection, unchanged, plus one hook — gradients for `decoder.anatomy_head`, `decoder.pathology_head`, `decoder.noise_head`, `decoder.causal_shared`, and `decoder.causal_weights` are scaled by 1/3 immediately after `scaler.unscale_(optimizer)` and before `optimizer.step()`. Scale factor and target parameters are taken directly from `research_infra/PHASE_7_HYPOTHESIS_VALIDATION.md` Q4/Q7 (Candidate 1) — the only quantitatively justified, single, minimal optimization change identified in that report. No other candidate (gradient clipping, loss reweighting) was implemented, matching the report's own ranking of which candidate was best-supported.

**Important caveat stated up front**: this is **one run per condition, not repeated-seed replicates**. Statistical tests below are paired-by-epoch-index across the three runs' 25 matched epochs — a legitimate way to test "did these two specific training trajectories differ," but not a substitute for multi-seed variance estimation. This limitation is carried through every claim below and is the main reason several findings are reported as "directionally suggestive" rather than "confirmed."

---

## Step 6: Research Assessment (evidence only)

### Did Sampling improve anything?

**Training time: yes, real and mechanistically explained.** 23.16s/epoch vs. baseline's 29.52s/epoch (paired t-test p=6.4×10⁻⁵). This has an obvious causal mechanism: `uniform` mode skips the scorer network's Conv3d/BatchNorm3d/Linear forward pass entirely, so less compute happens every batch regardless of what the numbers mean statistically — this is the single most trustworthy finding in this entire report, precisely because it does not depend on the model actually learning anything.

**GPU memory: a statistically detectable but practically negligible difference.** 692.3 MB vs. 696.7 MB (p=2×10⁻¹³, but the entire difference is 4.4 MB out of ~700 MB, i.e. <1%). Reported because the test asked for it; not treated as a meaningful improvement.

**Segmentation quality: no — directionally worse, not confirmed at p<0.05.** Last-5-epoch mean Dice: 0.00149 (sampling) vs. 0.00280 (baseline), p=0.0997 — below the conventional 0.05 threshold for significance, but in the *wrong* direction for "improvement," and consistent across Dice, IoU, Precision, and F1 (all lower for sampling, all with p in the 0.08–0.10 range — a consistent pattern across four correlated metrics, even though no single one crosses p<0.05). **Convergence speed is the clearest signal**: 0.000028 Δval_dice/epoch (sampling) vs. 0.000119 (baseline) — sampling improved barely a quarter as fast. `validation_dice_comparison.png` shows this directly: baseline's trajectory has two sharp improvement spikes (epochs 8, 11) that sampling's trajectory never reproduces — it stays flat and low throughout.

**Verdict**: Sampling did **not** improve segmentation quality in this run; it improved training speed for a mechanistically obvious reason unrelated to segmentation performance.

### Did Optimization improve anything?

**No metric shows a statistically significant improvement.** Dice (p=0.89), IoU (p=0.89), Precision (p=0.37), F1 (p=0.37), GPU memory (p=0.92) are all statistically indistinguishable from baseline. `validation_dice_comparison.png` shows the baseline and optimization curves nearly perfectly overlapping for all 25 epochs — this is visually obvious, not just a p-value artifact.

**The one directionally-consistent-with-intent (but non-significant) signal**: gradient-norm variance (std of per-epoch grad_norm_std) was lowest for optimization (0.0196) vs. sampling (0.0215) vs. baseline (0.0247) — the intended direction (the hook was designed to dampen a disproportionately large gradient source), but p=0.238, not significant, and `gradient_norm_comparison.png` shows all three conditions' total-gradient-norm trajectories are visually indistinguishable. This is explained by scale: the four targeted parameter groups are a small fraction of total network gradient norm (per Phase 7 Q4), so damping them by 1/3 was too narrow an intervention to move the *aggregate* metric this experiment tracks.

**The training-time difference for optimization (26.77s vs. baseline 29.52s, p=0.0085) should be treated with skepticism, not taken at face value.** There is no plausible computational mechanism by which scaling four small parameter groups' gradients in-place would measurably speed up an entire epoch — the operation is O(a few hundred floats). `training_time_bar.png` shows optimization's error bar (23.8–29.7s) overlapping substantially with baseline's (28.7–30.4s). This is far more consistent with epoch-to-epoch system-level timing noise (thermal/clock-boost state, background load) than with a real effect of the intervention, unlike sampling's speed difference, which has a clear mechanism. **This report explicitly does not credit the optimization prototype with a speed improvement.**

**Verdict**: Optimization prototype produced no measurable improvement on any primary metric. Its one intended effect (reduced gradient variance) trended in the correct direction but did not reach significance.

### Which improvement is statistically larger?

Neither intervention produced a statistically significant *improvement* on any segmentation-quality metric. The only significant improvements found anywhere were sampling's training-time reduction (real) and memory reduction (negligible in magnitude despite significance) — both efficiency metrics, not quality metrics, and neither is what either research proposal is actually about (Project A is about *whether adaptive selection helps quality/generalization*; Project B is about *whether optimization changes improve training dynamics/quality*).

### Which experiment required less code?

**Sampling.** The entire intervention is an 11-line `if selection_mode == 'uniform':` branch in `AdaptiveSliceSelector.forward` plus a constructor flag — it deletes computation (skips the scorer) rather than adding a new mechanism. The optimization prototype requires a new hook parameter threaded through `train_segmentation_epoch`, careful placement relative to `scaler.unscale_()`, and explicit submodule targeting — roughly 3x more lines and one AMP-correctness subtlety that sampling's change does not have to consider at all.

### Which experiment is easier to explain?

**Sampling**, by a wide margin, for a scientific audience: "we replaced a selection mechanism that provably never learns (Phase 1/7) with a fixed policy, and quality did not improve while speed did" is a one-sentence, falsifiable, self-contained claim. The optimization prototype's explanation requires first establishing the audience believes the Phase 7 gradient-imbalance finding matters, then explaining why a null result at the aggregate level doesn't contradict a real finding at the component level — a harder, more caveated narrative.

### Which has stronger novelty?

Neither prototype itself is novel (uniform sampling and gradient scaling are both standard, well-known techniques, and the task explicitly forbade building anything novel). The novelty question that matters is about the *underlying research direction*, not these prototypes: Project A's novelty case rests entirely on "adaptive" selection actually adapting, which Phase 1/7/8 collectively show it currently does not (confirmed for a fourth time in this session: the scorer network's parameters are literally irrelevant to the `uniform`-mode model's forward pass, by construction, and quality did not measurably suffer for it in the metrics that reached any significance). Project B's novelty case rests on exploitable optimization imbalances; Phase 7 found one real, reproducible imbalance (evidential-head gradients), but Phase 8 shows that the straightforward, best-justified fix for it does not move quality metrics in this setting.

### Which has stronger publication potential?

Based only on this session's evidence: **the Phase 1/7/8 finding that the adaptive-selection mechanism does not learn (and that replacing it with a trivial fixed policy is statistically indistinguishable in quality, while being faster) is the single most publication-ready, novel, falsifiable, well-triangulated finding produced across this entire investigation** — it was independently confirmed four separate times (standalone unit test, synthetic end-to-end test, real Phase 7 training run, and now this Phase 8 comparison), which is an unusually strong reproducibility record for a single research engagement. It is, however, a *negative* result about the existing implementation, not a positive result establishing a new sampling method — the publishable claim is "the adaptive component of this architecture does not do what it claims to," not "adaptive sampling improves segmentation." Project B's evidence (a real but narrow gradient imbalance, with a fix that doesn't move quality metrics at this scale/epoch budget) is a weaker, less complete story on its own.

---

## Step 7: Recommendation

### Recommendation: **Neither** — with one specific, narrower exception carved out below.

**Neither prototype improved segmentation quality over baseline in this experiment**, and the task's own decision rule is explicit: *"Recommend only if measurable improvements over baseline are observed. If neither improves over baseline, explicitly recommend abandoning both and explain why."*

**Why**: Every quality metric (Dice, IoU, Precision, Recall, F1, convergence speed) was either statistically indistinguishable from baseline (optimization) or directionally worse without reaching significance (sampling). The only significant improvements found were efficiency metrics (sampling's training time, and a memory difference too small to matter), which are not what either research proposal is fundamentally about.

**This is not the same conclusion as "abandon the underlying research questions."** It specifically means: *do not proceed with either of these two exact prototypes as currently scoped and tested*. A more precise reading of the evidence, carried over from Phase 7:

- **Project A (Adaptive Sampling)**: The evidence across four independent checks is unusually strong and consistent: the current adaptive mechanism does not learn, and removing it entirely changes nothing that reaches statistical significance. This is *itself* worth writing up as a finding (a methods/reproducibility contribution), but it does not support continuing "adaptive sampling" as a *performance-improvement* research direction without first building an actually-differentiable selection mechanism (Phase 2's Option 1/2) — an architecture change, not something this comparative validation was scoped to test. **Recommendation for this path specifically: do not pursue as scoped; the negative finding about the existing mechanism is the deliverable, not a launching point for further sampling-performance work without first fixing the differentiability gap.**
- **Project B (Optimization)**: The one real, reproducible finding (evidential-head gradient imbalance) did not translate into a measurable quality improvement when addressed with the best-justified minimal fix, at this epoch budget, on this dataset size. This could mean the imbalance genuinely doesn't matter for quality — or it could mean 25 epochs on 22 training volumes is too small/short a regime for any optimization change to show a quality effect, since the model barely learns at all in this regime regardless of condition (every run's Dice stayed under 0.006). **This experiment cannot distinguish those two explanations**, because no condition in this comparison achieved real convergence. **Recommendation for this path specifically: the finding that motivated it (Phase 7 Q4) remains real and reproducible, but this comparative test is inconclusive, not negative — a fair re-test would require a training regime that actually converges, which neither this comparison nor Phase 7 achieved.**

**If forced to choose one to carry forward under the assumption that a longer/larger-data training regime becomes available**: the evidence marginally favors **Project B being worth a second, better-powered look** over Project A, specifically because Project B's central premise (a measurable gradient imbalance exists) remains true and reproducible regardless of this test's inconclusiveness, whereas Project A's central premise (the selection is adaptive) has been *directly falsified*, not merely untested, across four independent checks. But under the evidence actually collected in this task — at this epoch budget, on this dataset, with these two exact prototypes — the correct, literal answer to the task's decision rule is **neither showed a measurable improvement, and neither should proceed as currently scoped.**

---

## Explicit Limitations

- Single run per condition; all paired statistics compare matched epoch indices across three individual trajectories, not repeated-seed replicates. Confidence in "no effect" findings would be strengthened by 3+ seeds per condition.
- No condition in this comparison achieved meaningful segmentation convergence (Dice < 0.006 throughout for all three) — this experiment has limited power to detect quality differences between interventions when the underlying task itself isn't being learned well, a limitation inherited from Phase 7's baseline and not resolved here.
- FLOPs were not included in the final comparison tables — `common.py` includes a best-effort `try_estimate_flops` using `torch.utils.flop_counter.FlopCounterMode`, but it was not exercised in the final run scripts, since the training-time measurements already gave a direct, reproducible efficiency signal for the one condition (sampling) where compute genuinely differs; adding it was judged not to change any conclusion above.
- The training-time "improvement" attributed to the optimization prototype is explicitly flagged above as likely measurement noise, not a real effect — included in `comparison_table.csv` for completeness (the task requires reporting it), but excluded from the recommendation's reasoning.
