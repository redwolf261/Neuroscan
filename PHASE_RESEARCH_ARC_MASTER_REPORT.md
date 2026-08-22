# NeuroScan / EGGO-M Research Report: Complete History of Hypotheses, Experiments, and Results

**Date compiled**: 2026-08-15

**Purpose**: this document is self-contained. It explains the project, the model, the data, every hypothesis tested, what was actually done to test each one, and what the result was — in enough detail that no other file needs to be opened to understand it.

---

## 1. The task and the baseline system

**Task**: automatically outline brain tumors in MRI scans. Specifically, "whole tumor" binary segmentation (tumor vs. not-tumor, ignoring tumor sub-type) on the BraTS 2023 GLI (glioma) public dataset, using only the FLAIR MRI sequence as input (one of four MRI sequences available per patient; the other three — T1, T1-contrast, T2 — are not used, to keep the model and problem simple).

**Data**: 1,251 patient scans. Each scan is a 3D volume, roughly 240×240×155 voxels at its original resolution. Every scan comes with a "ground truth" mask — a human-expert-drawn outline of the tumor — used both to train the model and to score it. The 1,251 scans are split, once, at the start of the project, into 1,126 for training and 125 held out for evaluation ("the validation set"), and that split has never changed.

**Preprocessing**: every scan, regardless of its original size, is resized down to a fixed 64×64×64 voxel cube before being shown to the model. This is a large compression (roughly 3.75× in two dimensions, 2.4× in the third) and, as later sections describe, turns out to be one of the most consequential decisions in the whole project — small tumors can shrink to just a few voxels, or vanish entirely, purely from this resizing step, before the model ever sees them.

**Model architecture**: a 3D U-Net — a standard architecture for this kind of task. Pictured simply:

```
input scan (64x64x64)
   -> shrinks through 3 "encoder" stages (each stage halves the resolution, doubles the channel count)
   -> a "bottleneck" (smallest resolution, most channels)
   -> grows back through 3 "decoder" stages (each stage doubles the resolution back up)
   -> a final 1x1x1 convolution + sigmoid produces, for every voxel, a probability "this voxel is tumor"
```

The model has two independent output "heads" that both read from the last decoder stage: one produces the tumor-probability map described above; the other produces a pair of numbers per voxel used for "evidential" uncertainty estimation (a technique where the model also learns to express how confident it is, not just what it thinks).

**Loss function** (what the model is mathematically punished for getting wrong during training): a fixed 50/50 blend of two terms — a "Focal Tversky" loss (a variant of the standard overlap-based Dice loss, tuned to weight false negatives and false positives, with an extra focusing term that emphasizes hard voxels) computed on the probability map, and an "Evidential Beta" loss computed on the uncertainty head. This 50/50 blend was itself chosen by a small ablation early in the project (testing 80/20, 50/50, and 20/80 splits) and has been frozen unchanged for the entire project since.

**Training setup, held fixed throughout almost the entire project unless explicitly noted otherwise**: 30 training epochs (full passes over the 1,126 training scans), the AdamW optimizer, a learning rate of 0.0004 that decays smoothly over training ("cosine annealing"), and processing 8 scans at a time ("batch size 8"). Hardware: a laptop with an NVIDIA RTX 5050 GPU with 8GB of memory.

**The headline number used throughout this report** is "Dice score" — the standard metric for how well a predicted tumor outline overlaps the true outline, ranging from 0 (no overlap) to 1 (perfect overlap). Two Dice numbers show up in this report and they are NOT interchangeable, which caused confusion partway through the project until it was explicitly flagged: (1) "pooled/batch Dice" — the number the training code itself reports, computed by pooling together all 8 scans in a batch before computing one Dice score for the whole batch, then averaging across batches; (2) "per-subject Dice" — computing one Dice score per individual scan, then averaging those. Per-subject Dice is the more standard, more interpretable number, but pooled Dice is what most of this project's historical results are recorded in, since that's what the training loop itself outputs. Where it matters, this report notes which one is being used.

**The reference baseline**, called "**A**" throughout the rest of this report: the system described above, trained with a fixed random seed ("seed 0") for reproducibility, achieving a **pooled Dice of 0.9063** on the 125-scan validation set. This number recurs constantly below as the number every other idea is compared against.

**The project's goal**: find a change to this system — ideally a genuinely novel algorithmic idea, not just a known trick — that improves Dice by at least 1 full percentage point (i.e., to roughly 0.916 or higher) over this 0.9063 baseline, in a way that could plausibly be written up as a research contribution (a university course/thesis project with an implicit "this should look like real research" expectation, including a literature-novelty check before any idea gets a full training run).

---

## 2. Early architecture decisions (before the main research arc)

Two housekeeping decisions were made before any "real" experimentation began:

- **The two-head architecture was necessary, not optional.** An earlier, simpler version of the model had only one output head, where the "uncertainty" numbers were computed by a fixed mathematical formula applied to the tumor-probability output, rather than being learned independently. This was recognized as a problem — it meant the "uncertainty" signal carried no real, independent information — so the model was redesigned to have two genuinely separate output heads (verified by checking that they share zero parameters and receive different, independently-computed gradients during training). The single-head version reached 90.92% Dice; it was retired, and the two-head version (0.9107 Dice, averaged over 3 random seeds, in a longer 50-epoch training run used only for this specific check) became the reference going forward.

- **The 50/50 loss-blend ablation**, mentioned above, confirmed the two loss terms should be weighted equally.

---

## 3. Idea #1 — ABO (Adaptive Boundary Optimization): NULL RESULT

**The idea**: instead of a fixed way of combining the loss terms, build a controller that automatically adjusts how much weight to put on "boundary" precision (getting the edge of the tumor exactly right) versus general region overlap, adapting during training based on how confident/uncertain the model currently is near tumor boundaries.

**What was done**: the controller was built and trained (referred to as "Experiment D" in the underlying project files), plus follow-up sweeps testing a range of its own internal hyperparameters (how aggressively it adapts, what target it aims for) to make sure a bad hyperparameter choice wasn't hiding a real effect.

**Result**: no meaningful Dice improvement over the fixed-weight baseline, and this held up robustly across the hyperparameter sweep (ruling out "just needs better tuning" as an explanation). **The idea was formally closed as a null result** — a deliberate decision to stop investing further tuning effort once the sweeps confirmed the null wasn't a hyperparameter artifact.

---

## 4. Building toward Idea #2 — literature review and margin-loss design (no training yet)

Before designing the next idea, a structured literature search and internal exploration phase happened:

- A survey of the model's own internal learned representations (the numeric "feature vectors" the network computes partway through, before the final prediction) — looking at their scale, geometric structure, and density — to find an actual, measurable weakness to design around, rather than picking an idea from theory alone.
- A comparison of five different mathematical formulations from the broader machine-learning literature for a "margin loss" — a type of loss that explicitly pushes a model's internal representations of "tumor" and "not tumor" voxels apart from each other, on top of the ordinary segmentation loss. The five candidates considered were: a contrastive/SupCon-style loss, a triplet loss, ArcFace/CosFace-style losses (popular in face recognition), and a pairwise-hinge loss based on a paper by De Brabandere et al. Each was checked against three practical requirements: does it work per-voxel (rather than per-image), does it avoid needing an extra competing normalization term, and does it work natively for a binary (tumor/not-tumor) problem rather than needing adaptation from a many-category setting.
- **The pairwise-hinge formulation was selected** as the only candidate meeting all three requirements natively.

This resulted in a locked mathematical specification for what became "EGGO-M" (the margin-loss mechanism explored extensively next) and a shared, carefully-tested piece of code (referred to internally as `compute_margin_loss`) that computes this margin loss, used unchanged by every subsequent experiment in this whole margin-loss research line.

---

## 5. Idea #2 — EGGO-M margin loss: mechanism works, Dice does NOT improve (extensively diagnosed, eventually closed)

**The idea**: add the margin loss designed above as an extra term on top of the existing segmentation loss during training. Concretely, at each training step, a set of "anchor" voxels are sampled (weighted toward voxels the model is currently most uncertain about), and the margin loss pushes the model's internal representation of each anchor voxel away from a computed decision boundary, in the direction that should make the model's eventual prediction more correct.

**What was done — an unusually long diagnostic chain, because early results were promising but final Dice never improved:**

1. **Initial pilot runs**: after some early calibration problems (the mechanism was initially inactive — contributing essentially zero gradient — due to a threshold hyperparameter needing recalibration), the mechanism was made genuinely active: measurably nonzero, correctly signed. But Dice was flat versus baseline.

2. **4-seed confirmation** (seeds 0, 1, 2, 3), same hyperparameters, no tuning, to rule out the flat result being a single-seed fluke:

   | Seed | Best Dice |
   |---|---:|
   | 0 | 0.9063 |
   | 1 | 0.9064 |
   | 2 | 0.9028 |
   | 3 | 0.9020 |

   No seed exceeded the baseline; at a matched training epoch, EGGO-M was actually statistically slightly *worse* than baseline. **This 0.9063 (seed 0) figure is where the "A = 0.9063" reference baseline used throughout the rest of this report originates.**

3. **Gradient-conflict check**: tested whether the margin loss's gradient was fighting against the segmentation loss's gradient in the shared part of the network. Ruled out — no meaningful conflict was found at the point where the two losses meet.

4. **Decoder-sensitivity check**: tested whether the later parts of the network (the decoder) were simply insensitive to the kinds of representation changes the margin loss was trying to induce. Ruled out.

5. **Margin-reachability check**: found that the margin loss's gradient direction was, in fact, correctly pointed (pushing representations the right way), but that this didn't accumulate into coherent overall training progress — individual steps were "right" but the net effect over many steps wasn't.

6. **Target-stability check**: found a likely explanation for #5 — the mathematical "target" the margin loss was optimizing toward was itself shifting during training (a "moving target" problem), which could explain why individually-correct steps failed to accumulate.

7. **Freezing ablations**: tried freezing different parts of the network (the encoder, the decoder, the final prediction layer, the batch-normalization statistics) one at a time, to see if stopping a specific kind of representation drift ("rotation," a geometric measure of how much the representation space was moving during training) would fix the accumulation problem. Freezing the decoder reduced this "rotation" the most. **This was initially treated as a promising lead — but a careful follow-up review found the opposite of what was hoped**: the decoder-frozen condition actually had the *lowest* Dice of all the conditions tested, and a *weaker* relationship between the margin loss and Dice, despite the margin loss being far more active in that condition. This was explicitly corrected in the record: representation "rotation" is *associated with* the problem, but is not *causing* it, and stopping the rotation does not fix the underlying issue.

8. **Further mechanistic digging**: additional analyses measured how gradients flow through individual layers, decomposed exactly how the key internal representation updates itself each step, and analyzed the "transport" of information through the network — none of these produced a new actionable lead.

9. **A full mathematical/implementation integrity audit**: a read-only, no-training check of the entire codebase for this margin-loss mechanism — verifying gradients numerically, checking sign conventions with an independent synthetic test, checking for accidental state mutations in batch-normalization layers, and verifying training reproducibility. **This confirmed the implementation itself was correct** — the null result was real, not a bug.

**Final verdict for the entire margin-loss line of research**: the mechanism is real, measurably active, and correctly mathematically signed — but it does not improve Dice, and none of the eight+ follow-up diagnostics above found a fixable cause. **Formally closed as a null result.**

---

## 6. Idea #3 — SC-TAM / "counterfactual objective" redesign: also NULL, but with an important bug found and fixed along the way

**The idea**: rather than another new mechanism from scratch, reframe the existing margin-loss idea more carefully — explicitly test whether the *direction* the margin loss pushes representations actually aligns with what would improve Dice, using a rigorous counterfactual-perturbation methodology (perturb the model slightly in the margin loss's suggested direction, and separately in the opposite direction, and directly measure which one helps). This led to a redesigned mechanism named "**SC-TAM**" (Scale-Consistent Task-Aligned Margin).

**A rigorous experimental protocol was locked before training**, including: hashing the model's initial random weights across every experimental condition to mathematically guarantee they all started from an identical point (not just "the same seed" but a byte-for-byte verified identical starting model); a hard "regression gate" re-checking that this new code reproduced the exact same numbers as the original, already-verified code, on every metric, before trusting any new result.

**Training results, referred to as "C6-2" and "C6-3":**

| Condition | Best Dice |
|---|---:|
| A (baseline) | 0.9063 |
| C6-2 | 0.9030 |
| C6-3 | 0.9026 |

Both slightly below baseline.

**A major sign-convention bug was found and fixed during the diagnostic process, before accepting a strange finding at face value.** While analyzing exactly which voxel-level errors the mechanism was correcting versus damaging, an initial analysis suggested a genuinely strange result (a kind of gradient-vs-geometry inconsistency worth treating with suspicion). Rather than accept this as a novel finding, a rigorous consistency check was run first — an explicit test comparing the direction of small "nudges" applied to the model against the model's own gradient, at several different nudge sizes, using a precise mathematical technique (a Jacobian-vector product check) to make sure the check itself wasn't the thing lying. **This uncovered a real bug**: every diagnostic script written from a certain point onward had been using a sign convention (essentially: "does correcting an error mean moving toward +something or −something") that was backwards relative to how the underlying mechanism's own code actually worked, and had been silently backwards for several analyses. This was fixed, and every affected analysis was rerun under the corrected convention.

**After the fix**, the corrected picture showed the mechanism was actually working well on individual voxel-level errors (98% of false-negative corrections were in the right direction, up from a seemingly terrible 2% under the buggy convention) — but Dice was still not improving overall. Follow-up work traced this further: representation shifts inside correctly-predicted voxels were sometimes larger and in a "damaging" direction than the corrective shifts happening inside actually-wrong voxels — i.e., fixing errors seemed to come with unwanted collateral damage to voxels that were already right, roughly canceling out the benefit.

**A confidence-gated version was also tried** (only apply the margin mechanism to voxels the model is currently uncertain about, using the ground truth to define a gate). An initial, simpler version of this gate failed the project's own strict acceptance criterion (it only preserved 23% of the useful, corrective signal while removing damage, when the bar had been set at retaining "most" of it). A redesigned gate, informed by diagnosing exactly why the first gate failed, passed this criterion decisively (92% corrective signal retained, 99% of damage removed) — but even this **still did not translate into a better Dice score** (0.9026, statistically the same as before).

**Verdict**: closed as a null result on the outcome metric, despite passing every internal, mechanistic validity check along the way. At this point, the project's direction was explicitly redirected by an instruction to stop refining loss-geometry mechanisms and instead find a genuinely different, structural cause of the problem.

---

## 7. Idea #4 — Deep Supervision: the only real positive result in the project so far

**The idea**: instead of another loss-geometry trick, address a more architectural hypothesis directly — maybe the later parts of the network (the "decoder," which rebuilds the full-resolution prediction from the compressed bottleneck representation) aren't getting a strong enough training signal at intermediate resolutions to properly handle small tumors. The fix tested: add two extra, temporary prediction heads that read from the decoder's intermediate stages (one at 1/4 resolution, one at 1/2 resolution) and are trained (only during training, discarded at prediction time) to match a correspondingly-downsampled version of the true tumor mask.

**Result — "Both" (both extra heads active): best Dice = 0.9091** — a +0.28 percentage-point improvement over the 0.9063 baseline. **This was the first idea in the entire project's history to beat the baseline.**

**Mechanism audit, run before doing anything further**, per an explicit instruction not to assume the obvious explanation ("it helps small tumors") without checking it directly:

- Checked whether small tumors (1-50 voxels after the 64³ resize) were now being *detected* more often. **They were not** — detection rate for that size range was essentially flat (22.0% → 20.6%, if anything very slightly worse).
- But a follow-up measurement found the real mechanism: for the small tumors that *were* already being detected by both the old and new model, the *quality* of that detection (how much of the tumor was correctly outlined, not just "was any of it found") improved substantially and specifically for small lesions — a 40% relative improvement in component-level Dice for the 1-50 voxel size range, shrinking to near-zero improvement for large tumors (>1000 voxels). A precise, size-graded, genuine mechanism: **the extra supervision improves segmentation completeness of small tumors that were already partially found, not the odds of finding a small tumor at all.**

**A scale ablation** then separated the two new heads to see which one mattered:

| Condition | Best Dice |
|---|---:|
| A (baseline) | 0.9063 |
| D2-only (1/2-resolution head only) | 0.9080 |
| Both (1/4 + 1/2) | 0.9091 |
| D4-only (1/4-resolution head only) | **0.9096** |

D4-only (the coarser of the two extra heads) was slightly *better* than using both together.

**A detailed follow-up comparison across all four conditions** (A, D4-only, D2-only, Both) found: D4-only's benefit was concentrated in a fairly small number of scans (the 10 most-improved scans out of 125 accounted for 73% of the total improvement); D4-only and "Both" were not statistically distinguishable from each other; and — importantly — **none of the six possible pairwise comparisons among A/D4-only/D2-only/Both reached standard statistical significance** at this sample size (125 scans), even though real, structured differences showed up when looking at tumor-size-stratified sub-analyses rather than the single aggregate number. This is a caution against over-reading the headline 0.9096 number as a clean, statistically bulletproof win — it's a real, reproducible-looking, mechanistically-explained effect, but a modest one relative to measurement noise at this sample size.

Also discovered during this comparison: a real discrepancy between the "pooled Dice" the training code reports and a proper "per-subject Dice" (roughly a 1.7-1.9 percentage-point systematic gap) — flagged and tracked carefully in every subsequent analysis so the two numbers were never silently mixed up.

---

## 8. Two more "does X explain the improvement" searches — both honest nulls

**Search #1 — does anything about a scan or tumor (besides size and baseline difficulty) predict which ones benefit from the deep-supervision trick?** Tested: tumor shape, internal intensity pattern, spatial location in the brain, how fragmented/scattered the tumor is, how isolated it is from other tumor pieces. **Result**: baseline difficulty and size together already explain about half the variance (a formal R² of 0.51); none of the other properties tested added anything beyond that. Reported as a clean, honest null — no attempt was made to force one of the weaker candidate explanations into a story.

**Search #2 — is the deep-supervision improvement actually about a specific tumor sub-type (e.g., the swelling/edema around a tumor, versus the tumor's solid core, versus its actively-growing rim), rather than about size per se?** This required, for the first time in the project, going back to the original, more detailed 4-category tumor labels (rather than the simplified binary tumor/not-tumor labels used everywhere else) to check subregion composition. **Initial result looked promising**: components dominated by "edema" were detected far worse than components dominated by other subregions (correlation as high as 0.80). **But this was investigated further before being trusted, and found to be a confound, not a real finding**: components dominated by "edema" turned out to have a median size of just 3 voxels (versus 2,561 voxels for components dominated by the tumor's necrotic core), so "edema-dominant" was really just a stand-in for "extremely tiny" — once tumor size was properly statistically controlled for, the apparent subregion effect collapsed to essentially nothing (the correlation dropped from 0.80 to 0.02-0.04). **Killed as a confound**, explicitly and honestly reported as such rather than left ambiguous.

---

## 9. A full project audit — the turning point toward "resolution" as the real bottleneck

At this point, a comprehensive, from-scratch audit of the entire project's actual code and data (not memory or summaries) was requested and performed, answering roughly 50 specific questions about the architecture, training setup, data pipeline, and prior experiments. The most important findings:

1. **No data augmentation exists anywhere in the training pipeline** — no random flips, rotations, noise, or intensity variation of any kind. Every experiment in the project's history has trained on the exact same 1,126 fixed scans, unaugmented, every epoch.
2. **There is no cropping or patch-based training** — every scan, however large its original resolution, is resized in a single step to a fixed 64×64×64 cube. This is a substantial compression (roughly 3.75× in two dimensions, 2.4× in the third) that had never itself been directly tested as a variable.
3. **Only one type of segmentation loss has ever actually been trained** in this project's history (the Focal-Tversky + Evidential blend described in Section 1) — several other well-known alternatives (plain Dice, plain cross-entropy, Hausdorff-distance-based losses, topology-aware losses, a loss called Lovász) have never been tried.
4. A hardware correction: the GPU actually available is more capable (8GB of memory) than had been assumed in planning (4GB), meaning more experiments were realistically affordable than previously budgeted for.
5. **The 125-scan validation set has been reused for every single model-selection and stopping decision across the entire project** — meaning, strictly speaking, it can no longer be treated as a perfectly "unseen" test set; a genuinely fresh held-out set (or a more rigorous cross-validation scheme) would be needed before reporting a final, publication-quality number.

**The synthesized conclusion of this audit**: small tumors are simultaneously (a) the dominant category of complete failures (of all fully-missed tumor pieces across the whole validation set, the overwhelming majority — 174 out of 178 — are in the smallest size category), (b) exactly the size category where the deep-supervision trick concentrates its benefit, and (c) a size category that may already be crushed down to just a few voxels by the mandatory 64³ resizing step, *before* the model or any of its training tricks ever get a chance to work with the data. This reframed the whole prior several months of loss-function and training-trick experimentation as possibly fighting a problem that was actually being caused earlier, by preprocessing.

---

## 10. Testing the resolution hypothesis directly

**Step 1 — a pure geometry check, no model involved.** For every real tumor piece in the 125 validation scans, its native (original, pre-resize) size was measured, and then it was resized down to several different target resolutions (64³, 80³, 96³, 128³, and a finer 160³ reference) to see how many voxels of it survive at each size.

**Result**: at the standard 64³ resolution, the **typical (median) tumor piece completely vanishes** — zero surviving voxels. A large fraction (about 65%) of all tumor pieces are so small at their original resolution (5 voxels or fewer) that they're likely just annotation noise rather than clinically real, and no resizing scheme will ever recover those. But for genuinely medium-small real tumor pieces (roughly 5-150 voxels at native resolution), there is a real, meaningful, resolution-dependent recovery effect: the typical voxel count present after resizing goes from 0 (at 64³) to about 1 (at 96³) to 3-23 (at 128³) — a real but modest effect, not a dramatic one.

**Step 2 — actually training at higher resolutions.** Three otherwise-identical models were trained, differing only in the target resolution: 64³ (matching the standard baseline exactly, batch size 8), 96³ (batch size reduced to 2, due to GPU memory limits), and 128³ (batch size reduced to 1, the smallest possible).

**Results**:
- 64³: Dice = 0.9038 (a fresh re-run of the standard baseline, confirming the training code reproduces the known number).
- 96³: Dice = 0.8986 — *worse*, not better.
- 128³: **invalidated**, not usable as evidence either way. Its validation Dice bounced around wildly and unpredictably across all 30 training epochs (ranging roughly 0.53 to 0.75, with no stable upward trend), which was traced to a specific, well-known technical problem: the "batch normalization" layers used throughout the network need to see multiple examples at once during training to work correctly, and at 128³ resolution the GPU could only fit one example at a time (batch size 1) — batch normalization essentially breaks under those conditions, producing erratic, meaningless results. This was correctly recognized as an artifact of the experimental setup, not a real finding about high resolution being bad, and training was explicitly stopped rather than reported as if it were a valid result.

**Conclusion at this stage**: the naive "just train at higher resolution" approach doesn't cleanly work (96³ actively hurt, and 128³ couldn't be properly tested with this hardware/architecture combination) — but the underlying geometric problem (small tumors vanishing under the mandatory resize) was now directly, quantitatively confirmed to be real, which motivated a more targeted approach rather than abandoning the resolution angle entirely.

---

## 11. Designing a targeted mechanism: the "Degradation-Trajectory Constraint" idea, killed after careful testing

**The idea**: rather than simply training at a higher fixed resolution, design a new training-loss term that explicitly measures, for every individual tumor piece, how its true (ground-truth) extent shrinks as resolution is progressively degraded from a fine reference grid down to the standard 64³, compare that against how the model's own predicted confidence shrinks along the same degradation sequence, and use any *mismatch* between the two curves as an extra training signal — the intuition being that voxels where the model's confidence disappears *faster* than the geometric ground truth does are exactly the voxels where the model is failing to make good use of the (little) information that remains.

**What happened, in detail, because two real bugs were caught along the way rather than producing a misleading result:**

- **Bug #1**: an initial statistical test seemed to show this "survival curve" had almost no relationship to tumor size (a near-zero result), which flatly contradicted a separate, much stronger correlation measurement on the same data. This contradiction was investigated rather than either number being trusted blindly, and the cause was found: the relationship between tumor size and survival isn't a smooth curve — it's closer to a coin flip (a tumor either survives resizing roughly intact, or vanishes completely to exactly zero voxels), and the specific statistical test being used (linear regression) is mathematically unable to detect that kind of on/off relationship, even though it's very real. Fixed by using an appropriate statistical measure instead.

- **Bug #2**: the model's predicted-confidence values, when divided by a small reference number (as the mathematical formula required), sometimes blew up to values in the millions, because that reference number was itself sometimes extremely close to zero (the model, evaluated at unfamiliar resolutions it was never trained on, often produces near-zero confidence everywhere for small tumors — a real, separately-interesting finding, but one that broke the specific mathematical formula being used). Fixed by explicitly excluding the roughly three-quarters of cases where this division was mathematically unreliable, being transparent that this exclusion meant the remaining, analyzable cases were mostly *large* tumors — the opposite of the small-tumor population the whole idea was meant to help.

- **The core finding, after both fixes**: for the specific small-tumor population the idea was designed to help, an apparent, seemingly-solid positive signal was found — it passed several rigorous checks (it wasn't just a restatement of tumor size, it survived removing extreme outliers, and a resampling-based confidence interval excluded zero). **But one final, critical safeguard caught something the others missed**: repeating the exact same analysis after deliberately scrambling (randomly shuffling) the model's actual predictions — so any real signal *should* disappear, since the "model" data is now meaningless noise — the scrambled version produced a correlation *just as strong* as the real one (in fact 200 random-shuffle trials produced a stronger correlation than the real result 96.5% of the time). This proved the apparent finding was a **mathematical artifact** of how the measurement was constructed (specifically, dividing by near-zero numbers on exactly the population of interest), not a real property of the model at all.

**Verdict: this specific idea was killed.** The underlying problem it was trying to address (small tumors losing information under resizing) remains real and important — but this particular mathematical mechanism for exploiting it does not work, and this was caught by a deliberate, rigorous "randomize and re-test" safeguard rather than being accepted at face value.

---

## 12. A narrower, simpler idea survives every check thrown at it: "critical resolution" (α_c)

**The idea, simplified from the failed one above**: instead of the whole complicated survival-curve-vs-model-confidence comparison, define one single, simple number per tumor piece — call it **α_c**, "the critical resolution" — literally just: at what resolution level does this specific tumor piece first completely disappear, as you progressively shrink the image? A tumor that vanishes very early (at a coarse, heavily-shrunk resolution) gets a low α_c; one that survives all the way down to the standard 64³ resolution gets a high α_c (or is marked as "never vanishes" if it's still present even then).

**Question asked**: does knowing a tumor piece's α_c tell you anything useful about how well the model will actually segment it — *beyond* what you'd already know just from the tumor's plain size? This matters because if α_c turned out to just be "a fancier way of measuring size," it wouldn't be a meaningfully new idea.

**What was done, this time avoiding the bug that killed the previous idea**: instead of dividing by the model's own shaky confidence numbers (the thing that broke before), this used a real, properly-thresholded, standard segmentation-quality score (component-level Dice, computed the normal, well-established way) as the thing being predicted, and carefully made sure the "predictor" (α_c, computed purely from geometry) never touched the model's predictions at all during its own construction — avoiding any risk of circular reasoning (using the answer to predict itself).

**Results, run through the same "randomize and re-test" safeguard that caught the previous idea's flaw, this time applied proactively rather than after the fact:**
- α_c does predict which tumors are well-segmented, beyond what tumor size alone predicts — confirmed by a statistical technique that specifically removes the portion of the relationship explainable by size alone before checking what's left (result: still a real, if modest, positive relationship).
- The random-shuffle safeguard was applied before trusting this result (learning directly from the previous idea's failure mode): 500 random-shuffle trials were run, and **none of them** produced as strong a relationship as the real, unshuffled data — the opposite of what happened with the failed prior idea.
- A resampling-based confidence interval (accounting properly for the fact that multiple tumor pieces from the same patient scan aren't independent data points) came out solidly excluding "no effect."

**Verdict: α_c survived every falsification test applied to it** — the most rigorously double-checked positive finding in the entire project. It is described as "real, but modest" — not a strong, dominant predictor, just a genuine, non-trivial, non-circular piece of information that tumor size alone doesn't capture.

**Important**: at this stage, α_c was explicitly treated as *only* a validated raw measurement — not yet a proven training algorithm. Turning "this number is informative" into "training with this number actually improves the final Dice score" is a separate, harder question, addressed next.

---

## 13. A literature check, before spending any real training time

Before committing to build and train a new algorithm around α_c, a deliberate literature-novelty search was performed — reflecting an explicit project-wide policy that no idea gets a full training run until it's been checked against recent published work, to avoid reinventing something that already exists (and to make sure a good Dice result would actually be defensible as a "novel contribution," which the project's broader goals require).

Roughly a dozen different search angles were used (covering things like "resolution-aware segmentation loss," "component-specific adaptive weighting," "object survival under downsampling," and similar), prioritizing the most recent (2026, then 2025) published work, and — critically — actually reading the mathematical details of every close-looking paper found, not just its title or summary, to judge true overlap rather than surface-level similarity.

**The closest related ideas found**, and how they differ:
- A 2026 paper proposing a similarly-named "component-adaptive" weighting scheme for small structures in brain MRI — but its actual formula weights purely by raw tumor size (a simple inverse power law), with no notion of resolution or degradation at all.
- An older (2020), well-established method that also weights training loss inversely by lesion size — again, purely size-based.
- Two other loss functions ("blob loss" and a similar "CC-Metrics" method) that give every detected tumor piece equal weight regardless of size — a different strategy again, neither size-based nor resolution-based.
- A family of methods based on a mathematical technique called "persistent homology," which does have a formally similar concept — a "critical value" at which a feature appears or disappears — but in every case found, that critical value tracks a *confidence/probability threshold*, not a *spatial resolution*. This is a real, substantive mathematical difference, not just a difference in wording, but it was flagged as the closest conceptual relative worth being aware of.

**Verdict: no existing published method was found that computes anything mathematically equivalent to α_c** (a resolution/degradation-derived vanishing point, used to weight training). The idea was judged to have real, if not certain, novelty — good enough to proceed to an actual training test, while being explicit that this was based on a focused, not exhaustive, literature search.

---

## 14. The decisive training experiment: designing it carefully, then a bug derails the first attempt

Given α_c survived every check above, a properly controlled training experiment was designed to answer the real question: does training a model *with* α_c-based extra supervision actually produce a better Dice score, and — crucially — is any such improvement something that couldn't be achieved just as easily by weighting tumors by their plain size instead? This last point matters enormously: if α_c helps, but a simple size-based weighting scheme helps by roughly the same amount, then α_c isn't really buying anything new, however mathematically interesting it is.

**Three conditions were designed to be trained and compared, all starting from mathematically identical initial random weights** (verified, not assumed, by checksumming the model's starting parameters):

- **A**: the standard, unmodified baseline (no change at all).
- **S**: the baseline, plus an extra training-loss term that gives more weight to smaller tumor pieces (using the size-weighting formula identified in the literature search above, as the fairest possible size-only comparison), carefully calibrated so its *average* weighting strength matches condition R's as closely as possible — the point being to make sure any difference between S and R is really about *what kind* of information is used (size vs. critical resolution), not just about *how much* extra emphasis small tumors get overall.
- **R**: the baseline, plus the same kind of extra training-loss term, but weighted by α_c instead of size.

Before training anything, a whole checklist of numerical safety checks was run on the new loss-weighting code and confirmed to pass: no invalid (NaN/undefined) numbers ever produced, tumor pieces that completely vanish under resizing are correctly excluded rather than accidentally creating a broken "zero-size" weighted term, background (non-tumor) voxels are never accidentally given special weight, and the overall scale of the new loss term doesn't blow up unpredictably.

**A significant, unrelated engineering problem was found and fixed before training could even start**: the new component-weighting code, as first written, was far too slow — it would have made each of the three-hour-plus training runs several times slower than necessary. This was tracked down through several rounds of careful profiling (an initial guess about the cause turned out to be wrong; the real cause — inefficient, many small individual computer-chip operations happening one-by-one for every single tumor piece in every training batch, rather than being done all at once — was found and fixed by rewriting that part of the code to process everything in a single, efficient batch operation). The rewritten version was explicitly checked to produce mathematically identical results to the slow-but-correct original, before being trusted.

**The actual training run, once launched (all three conditions, condition A first, as the standard baseline re-confirmation):**

| Condition | Best Dice |
|---|---:|
| A (baseline) | 0.9038 |
| S (size-weighted) | 0.7360 |
| R (α_c-weighted) | 0.7108 |

**Both new conditions collapsed dramatically — roughly 17-19 percentage points below baseline**, not the hoped-for improvement. Given the pre-agreed rule for this experiment ("any improvement smaller than +1 percentage point counts as a failure, and is not to be argued around"), this on its face fails outright.

**However, before accepting this as a real finding about α_c (or even about size-weighting), the sheer size of the collapse — and the fact that both new conditions failed by a similar, large amount — was treated as suspicious and investigated directly, rather than reported as a conclusion.** The investigation confirmed the model's starting weights and the very first thing computed by the ordinary, unmodified part of the loss function were mathematically identical between condition A and condition R on the very first training batch — ruling out a data-ordering or initialization bug. But the *overall size of the gradient* (the signal driving each training update) on that very first batch was roughly 75 times larger for condition R than for condition A, and this was traced specifically to the new extra loss term: it had been calibrated (before training) to have a reasonable-looking *typical value*, but nobody had separately checked that its *gradient* (its effect on training) was reasonably scaled — and because this new term is computed over just a handful of voxels per tumor piece (rather than being averaged across an entire large scan, like the ordinary loss is), its gradient turns out to be disproportionately steep, and it overwhelmed and destabilized training from the very first step.

**Current status, as of this report: this training experiment's result is considered invalid and not usable as evidence either for or against α_c or size-weighting.** The diagnosis (a miscalibrated new loss term destabilizing training) is understood and a fix (recalibrating that term based on its effect on the training gradient, not just its raw numeric size) has been proposed but **not yet carried out or re-tested** at the time of writing.

---

## 15. Where the project stands right now, overall

Out of every algorithmic idea attempted across the whole project's history, exactly one produced a real, positive, reproducible Dice improvement over the 0.9063 baseline: **Deep Supervision, at roughly +0.28 to +0.33 percentage points** (best version: 0.9096) — clearly short of the project's ≥1 percentage-point goal, though not statistically bulletproof at this sample size either.

Separately, a single specific measurement — **α_c, "critical resolution"** — has, on its own terms as a *measurement*, survived every rigorous check thrown at it (unlike several similar-looking ideas that were caught and correctly killed along the way): it carries real information about tumor difficulty beyond plain size, it isn't a mathematical artifact (confirmed via a deliberate randomize-and-retest safeguard), and a focused literature check didn't find an existing published method doing the same thing. But turning that validated measurement into an actual, working training algorithm that improves Dice is a separate, so-far-unresolved question — the first real attempt at this (Section 14) hit a genuine engineering/calibration bug rather than producing a clean answer, and that attempt has not yet been redone correctly.

**Everything else tried — the adaptive boundary controller, the margin-loss family (in two major redesigned forms), naive higher-resolution training, and the more elaborate degradation-trajectory idea — was tested carefully and closed as a genuine null result** (not abandoned prematurely, and not from lack of trying to make each one work), several of them only after unusually deep, multi-step diagnostic investigations that in a few cases caught and corrected real bugs in the diagnostic process itself before finalizing a conclusion.
