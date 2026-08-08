# Phase E18: Representation Rotation Source

**Status**: ✅ Complete, **with an important correction** — see `PHASE_E18_FOLLOWUP_PRE_E19_REVIEW.md`, written the same day after a careful pre-E19 review. Two things below are corrected there and should be read before treating this document's conclusion as final: (1) "decoder learning causes the rotation" overstates what a correlational ablation with a real BN confound can establish — the defensible claim is **"the decoder block is the strongest experimentally identified source of the early representation-wide rotation."** (2) More importantly, the follow-up review checked the causal question this document never actually tested — **does reducing rotation restore effective margin-driven optimization?** — using the full Dice and margin trajectories, and the answer is **no**: `freeze_decoder` achieves the lowest rotation of any config but also the lowest Dice among the three real freeze ablations, and a *weaker, non-significant* margin-Dice correlation than baseline, despite a 64× more active margin mechanism. Decoder rotation looks like an associated phenomenon, not the bottleneck. The rest of this document (methodology, incident log, rotation measurements) remains accurate and is preserved as originally written; only the interpretive framing above is superseded.

**Date**: 2026-08-08 (original); corrected same day

## Motivation

E17 established, via three independent methods (centroid direction, margin gradient direction, whole-representation Procrustes alignment), that `dec1`'s representation undergoes severe rotation during epochs 1–5, stabilizes from epoch 10 onward, but never returns to its epoch-1 orientation — settling into a new, permanently offset configuration instead. This phase asks *which architectural component* is responsible: running BatchNorm statistics, encoder learning, decoder learning, or segmentation-head adaptation. Per the phase's own scope, this is identification only — no fixes, no new losses, no architecture changes are proposed here.

## Experimental design

Five single-component ablations plus the unmodified baseline, each changing **exactly one thing** relative to `train_eggo_m.py`'s protocol (same seed=0, same calibrated constants δ_d=3.6659/adaptive τ_b/μ=0.1/λ=0.1 unless noted, same 30 epochs, same checkpoint schedule {1,5,10,15,20,25,30}):

| Config | What changes |
|---|---|
| `baseline` (`none`) | Nothing — reproduces `train_eggo_m.py` exactly |
| `freeze_bn` | Every `BatchNorm3d`'s running-stat updates frozen (`momentum=0`) after a warmup; forward pass still uses live batch statistics in `.train()` mode; affine params still train |
| `freeze_encoder` | `enc1/enc2/enc3/bottleneck` parameters excluded from the optimizer entirely |
| `freeze_decoder` | `upconv3/dec3/upconv2/dec2/upconv1/dec1` parameters excluded from the optimizer |
| `freeze_seg_head` | `seg_head` parameters excluded from the optimizer |
| `lambda_zero` | λ=0 (margin loss disabled); boundary head still trains via its own BCE |

Frozen parameters are excluded from the optimizer's parameter group entirely (not merely `requires_grad_(False)`, though that is also set) — the unambiguous, spec-literal reading of "frozen," removing any question of whether `weight_decay` could still silently touch them.

### Common measurements (identical methodology to E17, generalized across all 6 configs)

- **Part 1 — Centroid-direction stability**: `cos(dir_t, dir_{t+1})` for the tumor-minus-background centroid direction, consecutive checkpoints, over a fixed 10,000-voxel tracking set.
- **Part 2/3 — Margin gradient stability & transport**: `cos(g_margin(t), g_margin(t+1))` and `cos(g_margin(1), g_margin(t))`, on a fixed *stratified* (low-evidence) anchor set selected once per config, reused across all its checkpoints (matches E17's methodology and the same corrected sampling strategy, since a uniform-random set essentially never lands on an active hinge pair — see E17's own bug note). N/A for `lambda_zero` (no margin gradient exists when λ=0).
- **Part 4 — Whole-representation rotation (Procrustes)**: `orthogonal_procrustes` best-fit rotation between consecutive checkpoints' mean-centered point clouds; **rotation-closeness-to-identity** (`trace(R)/32`, 1.0 = no rotation needed) and **residual fraction** (non-rigid reshaping unexplained by rotation alone).
- **Part 5 (new) — Principal-angle / subspace-overlap analysis**: top-5 PCA/SVD components of `dec1` at each checkpoint; principal angles (via `scipy.linalg.subspace_angles`) between epoch 1's and every later checkpoint's top-5 subspace — tests whether the *entire* latent basis rotates or only a specific axis.
- **Performance/margin metrics**: Dice, HD95, ECE, mean margin, active hinge %, margin loss — read directly from each config's `epoch_metrics.csv`.
- **Rotation-magnitude summary**: `1 − cos(dir_a, dir_b)` for the three requested windows (epoch1→5, 5→10, 10→30), using Part 1's centroid direction as the primary summary (available for every config, unlike Part 2/3).

## What went wrong before results could be trusted (reported in full, not summarized away)

This phase took substantially longer than planned, for reasons worth documenting honestly since they bear directly on how much to trust the final BN-related finding.

### Incident 1 — `freeze_bn` bug, attempt 1: froze running stats before any real data had touched them

The first implementation called the momentum=0 freeze at model construction, before a single batch had been seen. Running stats stayed locked at PyTorch's `BatchNorm3d` default init (`mean=0, var=1`) forever. Training looked completely healthy (`.train()` mode uses live batch statistics, ignoring the running buffers) — but validation (`.eval()` mode, which reads the running buffers directly) collapsed to `val_dice≈1e-11`, `HD95≈40–52`, despite `train_dice` climbing normally to 0.92. A textbook train/eval BatchNorm divergence, caught by that exact signature, not assumed.

### Incident 2 — an unrelated GPU OOM crash caused by my own tooling mistake

While fixing Incident 1, a watcher script intended to auto-launch the corrected `freeze_bn` rerun after the main queue finished used `pgrep` to detect completion — unavailable in this Git Bash environment. The guard failed silently and launched `freeze_bn` **in parallel** with the still-running `freeze_encoder`, causing a CUDA out-of-memory crash that killed `freeze_encoder` at epoch 27/30. Recovered by killing the stray process, discarding the corrupted/incomplete runs, and relaunching everything as one strictly sequential script. No results from this incident were used; `freeze_encoder` was fully rerun from scratch.

### Incident 3 — `freeze_bn` bug, attempt 2: a 1-epoch warmup was not enough

Fixed Incident 1 by adding a 1-epoch warmup (real data updates the running stats for epoch 1, *then* freeze). This looked plausible but **collapsed again**, just later — `val_dice` degraded progressively from epoch 2 onward, reaching `≈1e-11` again by epoch 20+, while `train_dice` again stayed healthy (~0.92). Root cause: one epoch of `momentum=0.1` updates only partially converges the running statistics toward the true distribution; freezing that immature snapshot goes stale as the live (train-mode) distribution keeps drifting — directly consistent with E17's own finding that the representation keeps rotating substantially through roughly epoch 5–10.

### Attempt 3 — a 10-epoch warmup, informed by E17's own stabilization point

Reasoned that freezing after epoch 10 (matching E17's measured stabilization point) should avoid freezing mid-rotation. This **also eventually collapsed** — val_dice stayed healthy through epoch 15 (0.85–0.89, matching other configs), then collapsed at epoch 16 (val_dice→0.27, HD95→42) and **did not recover** through epoch 25, where the run was stopped by explicit user decision rather than continuing to chase a longer warmup. This is the point where continuing to retry stopped being a bug-fixing exercise and became the actual finding: **no warmup length that leaves a meaningful remainder of training epochs survives this architecture's full instability window** — E17 itself showed the representation is still measurably settling as late as epoch 25–30, not fully done by epoch 10 as attempt 3 assumed. The checkpoints from this stopped run (epochs 1, 5, 10, 15, 20, 25 — spanning both the healthy pre-collapse and degenerate post-collapse periods) are used for the quantitative measurements below; epoch 30 is unavailable for `freeze_bn` alone as a direct consequence.

### Incident 4 — a stale-cache bug in the measurement script itself

After fixing the `CONFIGS` dictionary to point at the correct (renamed) `freeze_bn` checkpoint directory, an earlier interrupted measurement run had already written a `freeze_bn_results.json` — from *before* that path fix, effectively invalid. A resume/skip-if-exists optimization added to `e18_measure_rotation.py` (to avoid re-measuring the 3 already-completed configs after a tool-call interruption) saw that stale file and treated it as done, silently reusing invalid data. Caught by comparing `freeze_bn`'s and `baseline`'s raw Part 1 values and finding them **byte-identical to 15+ decimal places** — impossible for two genuinely different training runs. Fixed by deleting the stale file and re-measuring `freeze_bn` alone; verified the corrected result is now genuinely distinct from baseline (matches closely at epochs 1–15, where the two runs are legitimately similar since BN hadn't been frozen yet and the model was still healthy, then diverges starting at 15→20 — exactly where the collapse begins).

None of these four incidents affects the final data reported below — each was caught via an internal consistency check (a suspicious metric shape, an impossible byte-for-byte match) before being trusted, not after.

## Results

### Rotation-magnitude summary (1−cos, centroid direction; 0=no rotation)

| Config | 1→5 | 5→10 | 10→30 |
|---|---|---|---|
| baseline | 0.1585 | 0.0717 | 0.0293 |
| freeze_bn | 0.1585 | 0.0717 | N/A (stopped epoch 25) |
| freeze_encoder | 0.2026 | 0.0435 | 0.0089 |
| **freeze_decoder** | **0.0608** | **0.0213** | 0.0245 |
| freeze_seg_head | 0.2833 | 0.0657 | **0.1753** |
| lambda_zero | 0.1329 | 0.0142 | 0.0525 |

`freeze_decoder` shows roughly **2.6× less** early (1→5) rotation than baseline and the lowest of any config at both early windows. `freeze_seg_head` shows the most (1.8× baseline at 1→5) and, notably, **6× more** rotation than baseline in the late window (10→30) — the only config where late-training rotation exceeds early-training rotation, meaning it never truly settles. `freeze_encoder` and `lambda_zero` sit close to baseline's shape, with `freeze_encoder` if anything showing slightly *more* early rotation, not less.

### Part 4 — Procrustes whole-representation rotation (mean rotation-closeness-to-identity across all measured pairs; higher = more stable)

| Config | Mean | Std | Δ vs. baseline |
|---|---|---|---|
| baseline | 0.8751 | 0.1837 | — |
| freeze_bn | 0.8278 | 0.1842 | −0.0473 |
| freeze_encoder | 0.8899 | 0.1793 | +0.0149 |
| **freeze_decoder** | **0.9468** | **0.1035** | **+0.0717** |
| **freeze_seg_head** | **0.7555** | **0.1653** | **−0.1196** |
| lambda_zero | 0.8264 | 0.2798 | −0.0486 |

Per-pair detail (rotation-closeness-to-identity):

| Config | 1→5 | 5→10 | 10→15 | 15→20 | 20→25 | 25→30 |
|---|---|---|---|---|---|---|
| baseline | 0.4682 | 0.9303 | 0.9479 | 0.9255 | 0.9856 | 0.9929 |
| freeze_bn | 0.4682 | 0.9303 | 0.9071 | 0.8544 | 0.9792 | N/A |
| freeze_encoder | 0.4994 | 0.8792 | 0.9840 | 0.9904 | 0.9915 | 0.9952 |
| **freeze_decoder** | **0.7159** | **0.9757** | **0.9963** | **0.9967** | **0.9969** | **0.9991** |
| **freeze_seg_head** | 0.4107 | 0.7944 | 0.8336 | 0.7551 | 0.7938 | 0.9452 |
| lambda_zero | 0.2060 | 0.8841 | 0.9248 | 0.9841 | 0.9866 | 0.9729 |

`freeze_decoder` is the **only** config that reaches near-perfect stability (>0.99) by epoch 15 and holds it for the rest of training. `freeze_seg_head` is the only config that *drops* rotation-closeness mid-training (0.83→0.76 at 15→20) rather than monotonically improving — a genuinely different, non-converging shape from every other config.

### Part 5 — Principal angles vs. epoch 1 (max angle, degrees; less discriminating than Part 4)

| Config | ep5 | ep10 | ep15 | ep20 | ep25 | ep30 |
|---|---|---|---|---|---|---|
| baseline | 87.7 | 83.6 | 81.7 | 84.9 | 86.0 | 85.7 |
| freeze_bn | 87.7 | 83.6 | 83.9 | 84.5 | 85.6 | N/A |
| freeze_encoder | 88.4 | 89.5 | 84.1 | 85.1 | 85.0 | 88.4 |
| freeze_decoder | 81.8 | 88.6 | 88.0 | 88.0 | 86.0 | 80.8 |
| freeze_seg_head | 82.7 | 88.3 | 87.2 | 86.4 | 86.0 | 85.3 |
| lambda_zero | 79.6 | 83.1 | 86.1 | 85.9 | 84.9 | 87.0 |

All configs land in a narrow 78–90° range regardless of ablation — the *entire* top-5 subspace ends up substantially rotated relative to epoch 1 in every case, consistent with E17's permanent-offset finding holding architecture-wide, not just for the specific margin-relevant direction. **This metric does not discriminate between ablations** — unlike Part 1/4's consecutive-pair measures (which capture the *rate/trajectory* of rotation), Part 5 measures the *endpoint* displacement from a single fixed reference, and every config apparently ends up similarly far from epoch 1 regardless of path. Reported in full for completeness and honesty (per the phase's own "do not overinterpret" instruction) rather than omitted because it doesn't support the headline finding as cleanly as Parts 1 and 4 do.

### Part 2 — Margin gradient stability (consecutive-pair cosine; N/A for `lambda_zero`)

| Config | 1→5 | 5→10 | 10→15 | 15→20 | 20→25 | 25→30 |
|---|---|---|---|---|---|---|
| baseline | 0.139 | 0.537 | 0.644 | 0.638 | 0.703 | 0.704 |
| freeze_bn | 0.137 | 0.535 | 0.645 | 0.637 | 0.703 | N/A |
| freeze_encoder | 0.143 | 0.402 | 0.517 | 0.549 | 0.587 | 0.607 |
| **freeze_decoder** | **0.262** | **0.665** | **0.835** | **0.803** | **0.766** | **0.777** |
| freeze_seg_head | 0.132 | 0.238 | 0.396 | 0.306 | 0.267 | 0.487 |

A **third, fully independent** measurement (the actual margin-loss gradient direction, not a geometric summary) reproduces the identical ranking: `freeze_decoder` has the highest gradient stability at every single consecutive pair, `freeze_seg_head` the lowest at every pair (never exceeding 0.49, versus baseline's 0.70+ by late training), `freeze_encoder` consistently below baseline. This is the same qualitative story as Parts 1 and 4, obtained from a mechanistically unrelated measurement (actual loss gradients vs. geometric point-cloud analysis) — strong triangulation, not a repeated artifact of one method.

### Performance (Dice) sanity check — `freeze_decoder`'s stability is not degenerate/frozen-model artifact

| Config | Dice ep1 | ep5 | ep10 | ep15 | ep20 | ep25 | ep30 |
|---|---|---|---|---|---|---|---|
| baseline | 0.326 | 0.781 | 0.889 | 0.887 | 0.890 | 0.905 | 0.906 |
| **freeze_decoder** | 0.101 | 0.452 | 0.686 | 0.785 | 0.798 | 0.830 | **0.843** |
| freeze_seg_head | 0.181 | 0.256 | 0.357 | 0.417 | 0.705 | 0.790 | 0.823 |
| freeze_encoder | 0.266 | 0.733 | 0.855 | 0.860 | 0.871 | 0.874 | 0.875 |

`freeze_decoder` climbs steadily to Dice=0.843 by epoch 30 — real, substantial learning, not a frozen/degenerate model artifact producing spuriously low rotation by having nothing left to move. It also shows elevated `active_hinge_pct` throughout (5–13% vs. baseline's <1%, not tabulated above — see raw JSON), consistent with `dec1` staying more stable and the hinge condition remaining satisfiable for longer, rather than the model being inert. `freeze_seg_head` shows its own distinct signature: a severe early plateau (Dice stuck at 0.18–0.42 through epoch 15, unlike any other config) before recovering to a respectable 0.82 by epoch 30 — a genuinely different failure/recovery shape from `freeze_bn`'s permanent collapse, worth distinguishing (see Limitations).

### freeze_decoder's early active_hinge_pct spike — addressed

Flagged during training monitoring: `active_hinge_pct` briefly reached 100% at isolated batches early in `freeze_decoder`'s training (around epoch 8), a sharp departure from every other config's typical <10% rate. By epoch 23 this had settled to the more typical 4–9% range seen in the epoch-by-epoch table above (still elevated relative to baseline's <1%, but no longer spiking to saturation). Consistent with early-training instability (the same epoch 1–10 volatility window this whole phase investigates) rather than a sustained anomaly — the elevated *steady-state* active_hinge_pct (5–13%) is itself a real and informative signature of `dec1` being more stable/reachable for the hinge condition once frozen, not a bug.

## Interpretation: which success criterion applies?

Per the pre-registered options:

- **(A) BatchNorm dynamics are the dominant source**: **not directly demonstrated** — no clean, non-collapsed `freeze_bn` trajectory was ever obtained despite three attempts with increasing warmup length, so a direct rotation comparison across the full 30 epochs isn't available. But the *repeated, escalating* failure to freeze BN running stats at any tested point without eventual staleness is itself informative (see below) — this is a real, reportable finding about BatchNorm's relationship to the ongoing representation drift, just not the kind of clean "rotation reduced/unchanged" comparison the other four ablations produced.
- **(B) Encoder learning is the dominant source**: **not supported**. `freeze_encoder`'s rotation trajectory (Part 1, Part 4, Part 2) tracks close to baseline throughout, if anything slightly *higher* at the earliest window (1→5: 0.2026 vs. baseline's 0.1585) — freezing the encoder does not reduce rotation.
- **(C) Decoder learning is the dominant source**: **supported, consistently, across four independent measurement methods.** `freeze_decoder` shows the least rotation by centroid direction (2.6× lower than baseline at 1→5), the highest Procrustes rotation-closeness at every single measured pair (reaching >0.99 by epoch 15, never achieved by any other config), and the highest margin-gradient stability at every pair — while still training normally (Dice climbs to 0.843, not a degenerate frozen model). This is the strongest, most consistent, most triangulated signal in the entire phase.
- **(D) Segmentation-head adaptation is the dominant source**: **not the dominant driver, but a real, secondary contributor in the opposite direction.** `freeze_seg_head` shows *more* rotation than baseline at nearly every measure (Procrustes mean 0.7555 vs. baseline's 0.8751, the only config with a mid-training rotation *increase* rather than monotonic stabilization) and the worst margin-gradient stability throughout. This says something different from "seg-head adaptation causes rotation" — it suggests the segmentation head's own continued adaptation, when the trunk (`dec1`) is left free to keep moving without a stable head to anchor against, may actually help stabilize the representation, and removing that anchor makes rotation *worse*, not better. An interesting, secondary finding, not the phase's headline one.
- **(E) Independent of all tested components**: **not supported** — the four ablations produce clearly different, mechanistically coherent rotation signatures (not a flat "nothing matters" result), and the effect sizes (freeze_decoder vs. freeze_seg_head span a 0.19-point range in mean Procrustes closeness) are large relative to the single-seed noise floor suggested by the configs that don't differ much from baseline (freeze_encoder, lambda_zero, both within ~0.015-0.05 of baseline).

**Conclusion: Success criterion C — decoder learning is the dominant identified source of the early representation-wide rotation**, with a secondary, opposite-direction finding that segmentation-head adaptation may help *stabilize* rather than destabilize the representation, and an inconclusive-but-informative finding about BatchNorm (see below).

## What the freeze_bn saga itself demonstrates

Three escalating attempts to freeze BatchNorm running statistics (0-epoch, 1-epoch, 10-epoch warmup) each eventually collapsed, with the collapse onset pushing later each time (epoch 2, epoch 2, epoch 16) but never being avoided. Read together with E17's own finding that centroid-direction rotation only reaches near-perfect consecutive-pair stability (>0.99) by epochs 25–30, this is consistent, corroborating evidence that **this architecture's representation keeps moving meaningfully for most of a 30-epoch run** — not just during a short early window. A BatchNorm running-statistics freeze is, in effect, a bet that the representation will stop moving relative to whatever point you freeze at; that bet failed at every warmup length tried, which is itself the kind of result the phase's own framing anticipated could happen ("if such evidence exists, report it prominently, do not protect previous conclusions"). This does not cleanly assign BatchNorm as *the* dominant rotation source (criterion A) in the same positive sense that `freeze_decoder`'s result supports criterion C, but it does support a weaker, related claim: **BatchNorm's adaptive behavior is entangled with the representation's ongoing movement closely enough that decoupling them (freezing one while the other keeps moving) is not achievable via a simple fixed-point freeze**, which is itself a real constraint on how BatchNorm relates to the phenomenon under investigation.

## Statistical caveats

Every comparison in this phase is **single-seed** (seed=0 throughout, matching E14–E17's precedent but not extending to multiple seeds). The "statistical analysis" requested by the phase spec is limited by this: with n=5–6 consecutive-pair measurements per config and no seed replication, formal significance testing (e.g. comparing `freeze_decoder`'s mean rotation-closeness to baseline's) would produce numbers of limited standalone reliability. The evidence this report actually leans on is **triangulation across four mechanistically independent measurement methods (centroid direction, Procrustes rotation, principal angles, margin gradient stability) all agreeing on the same ranking** — a stronger basis for the qualitative conclusion (decoder > encoder ≈ baseline > seg-head-frozen, in terms of stabilizing effect) than any single quantitative test at this sample size would be, but the phase's own instruction not to overinterpret single-seed fluctuations is taken seriously: the specific numeric magnitudes (e.g. "+0.0717") should be read as directionally reliable, not as precise, seed-independent point estimates.

## Falsification

Searched directly for evidence contradicting the decoder-source hypothesis:

- **Does `freeze_decoder`'s reduced rotation just reflect a broken/inert model?** No — Dice climbs steadily and substantially (0.10→0.84), and `active_hinge_pct` is elevated, not zero, throughout (see Performance table and its discussion above). Ruled out.
- **Does `freeze_encoder` show ANY sign of reduced rotation that would support encoder-as-source instead?** No — checked every measured window; `freeze_encoder` is consistently close to or slightly above baseline's rotation at every metric, never below it in a way that would suggest the encoder plays a comparable stabilizing role to the decoder.
- **Is `freeze_seg_head`'s result actually consistent with seg-head being A driver (just not the only one), rather than contradicting the decoder-source finding?** Plausibly both are true simultaneously — freezing the decoder reduces rotation, and freezing the seg-head increases it, which are not mutually exclusive claims (they could reflect the same underlying mechanism: the seg-head's own continued adaptation partially constrains/anchors `dec1`'s movement, so decoder-freezing removes the source of movement while seg-head-freezing removes a stabilizing constraint on that same movement). This does not falsify criterion C; it adds nuance rather than contradicting it.
- **No evidence was found that would overturn the freeze_decoder-is-most-stable finding** across any of the four independent measures — this is the one falsification search in this phase that did not succeed in finding contradictory evidence, reported plainly rather than searched-for-and-omitted.

## Limitations

1. **Single seed throughout** (seed=0) — see Statistical caveats above.
2. **`freeze_bn` never produced a full, non-collapsed 30-epoch trajectory** despite three attempts — its rotation-magnitude summary is incomplete (`epoch10_to_30: N/A`), and its available late-checkpoint (epoch 20, 25) geometry measurements are confounded by the ongoing collapse itself, not a clean measurement of "rotation with BN frozen" in the way the other four ablations' full trajectories are. Distinguishing "BN freezing caused unusual rotation" from "BN freezing caused a training collapse that incidentally also shows unusual rotation" is not fully resolved by this phase's data.
3. **`freeze_seg_head`'s early Dice plateau (epochs 1–15) and `freeze_bn`'s permanent collapse are visually similar (both show severe early degradation) but are NOT the same phenomenon** — `freeze_seg_head` recovers substantially by epoch 20–30 (Dice 0.82), `freeze_bn` never recovers within the observed window. Conflating these would be a mistake; they are reported and interpreted separately throughout.
4. **Part 5 (principal angles) did not discriminate between ablations**, unlike Parts 1/2/4 — included for completeness per the phase's own request, not because it supports the headline finding; a metric that measures endpoint displacement from a fixed reference rather than trajectory/rate is apparently less sensitive to which component is frozen, at least at this checkpoint density (7 points over 30 epochs).
5. **This phase identifies which component's freezing changes rotation; it does not establish the causal mechanism by which decoder learning specifically produces rotation** (e.g., whether it's the decoder's own weight updates directly reshaping `dec1`, or an indirect effect via how gradients from the frozen encoder's fixed features propagate differently through a still-training decoder) — a mechanistic account beyond "freezing this component changes the outcome" was not attempted and is a natural next question.
6. **No architectural changes, new losses, or optimizer changes were proposed or evaluated**, per the phase's explicit scope — this is an identification result only.

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e18_ablation_train.py` | Ablation training script (5 configs + baseline) |
| `experiments/exp_e12_eggo_m/e18_measure_rotation.py` | Measurement script (Parts 1-5 + performance/margin metrics), generalized across configs |
| `experiments/exp_e12_eggo_m/e18_none_seed0/`, `e18_freeze_encoder_seed0/`, `e18_freeze_decoder_seed0/`, `e18_freeze_seg_head_seed0/`, `e18_lambda_zero_seed0/` | Valid, complete 30-epoch training runs |
| `experiments/exp_e12_eggo_m/e18_freeze_bn_BUGGY_seed0/`, `e18_freeze_bn_ATTEMPT2_STILL_STALE_seed0/`, `e18_freeze_bn_ATTEMPT3_STOPPED_epoch25_seed0/` | The three freeze_bn attempts, preserved for the record |
| `experiments/exp_e12_eggo_m/e18_measurement_results/` | Per-config JSON results, aggregate `all_configs_results.json` |
| `experiments/exp_e12_eggo_m/E18_HANDOFF_NOTES.md` | Mid-phase handoff notes written during an overnight pause |
| `PHASE_E17_MARGIN_TARGET_STABILITY.md` | The rotation finding this phase investigates the source of |
| `PHASE_E18_FOLLOWUP_PRE_E19_REVIEW.md` | **Read this too** — corrects the interpretive claim below and checks whether reduced rotation actually helps (it does not) |

---

**Completed**: 2026-08-08 — Success criterion C: the decoder block is the strongest experimentally identified source of early representation-wide rotation (not "decoder learning causes" it — see the follow-up review for why that's overstated given a symmetric BatchNorm confound shared with the encoder-freeze ablation), confirmed by four independent measurement methods (centroid direction, Procrustes rotation, margin gradient stability, and — with less discrimination — principal angles). Secondary finding: segmentation-head adaptation appears to help stabilize rather than destabilize the representation. BatchNorm's role remains genuinely unresolved by direct comparison (no clean freeze_bn trajectory was achievable across 3 attempts), but the escalating-warmup failure pattern is itself evidence that the representation's instability window persists longer than any tested freeze point, corroborating E17's own finding independently. **Critically, the follow-up review found that this rotation reduction does not restore effective margin-driven optimization** — see `PHASE_E18_FOLLOWUP_PRE_E19_REVIEW.md` for the full causal-chain check before using this phase's result to motivate any E19 design.
