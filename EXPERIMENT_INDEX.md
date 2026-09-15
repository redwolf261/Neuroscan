# NeuroScan / EGGO-M — Complete Experiment Index

**Compiled from**: `docs/RESEARCH_KNOWLEDGE_MAP.md` and `docs/phases/PHASE_RESEARCH_ARC_MASTER_REPORT.md`

**Date**: 2026-09-13 | **Total Experiments**: 135+ phases (A–E107+)

---

## BASELINE ESTABLISHMENT (Pre-Arc)

| ID | Task | Result | Status |
|---|---|---|---|
| **Pre-arc (2D frozen baseline)** | Test original frozen HybridMiniSwin2D5_CBAM model on 3D BraTS tumors | Collapsed to near-random (0–30% Dice) — model fundamentally inadequate for 3D data | ✅ CLOSED — architecture replaced with 3D U-Net |
| **Pre-arc (3D baseline)** | Establish first 3D U-Net baseline on BraTS | Best val Dice **81.5% (epoch 4)** — clean baseline established | ✅ ADOPTED as reference |
| **Pre-arc (two-head redesign)** | Test single-head vs. two-head architecture (seg + evidential) | Two-head **0.9107 Dice** vs. single-head **0.9092** — independent heads required | ✅ ADOPTED as architecture standard |
| **Pre-arc (baseline frozen, 3-seed)** | Confirm frozen baseline across 3 random seeds | **0.9100 ± 0.0005** (seeds 0.9107/0.9095/0.9097) — highly reproducible | ✅ FROZEN as reference "A" |

---

## PHASE A–D: EARLY LOSS & GRADIENT EXPERIMENTS

| ID | Task | Result | Status | Key Finding |
|---|---|---|---|---|
| **Phase C0 (weight ablation)** | Test if changing seg/evidential loss weights (0.8/0.2 → 0.2/0.8) resolves gradient conflict | Trunk gradient ratio invariant (~0.09–0.10) across 8× weight swing; Dice flat (0.8787–0.8797) | ✅ CLOSED — falsifies "static reweighting solves conflict" hypothesis | ECE improves monotonically with evidential weight |
| **Phase B/C0/C1/D (ABO)** | Adaptive Boundary Optimization: dynamic gradient controller adjusting loss balance during training | Controller measurably shifts trunk gradient ratio ~2× (baseline 0.098 → ABO 0.19); mechanism verified working, **Dice unchanged (0.9100 ± 0.0006 vs 0.9100 ± 0.0007)** across 3 seeds | ❌ KILLED (mechanism real, outcome null) | "Delta chasing alpha" hypothesis false; undershoot explained as calibration gap |

---

## PHASE E1–E24: LITERATURE, DESIGN & MECHANISM DIAGNOSTICS

| ID | Task | Result | Status | Key Finding |
|---|---|---|---|---|
| **E1 (literature matrix)** | 45-paper survey (2023–2026): map project findings against MOO/gradient/evidential literature | Converges with literature: gradient-level fixes treat symptoms, not root cause (representation-level entanglement) | N/A (synthesis) | Early pointer to bottleneck/representation issues |
| **E1.1 (boundary vs evidence)** | Does evidence ≈ boundary-distance detector? | R² = 0.150 (boundary explains only 15% of evidence variance); incorrect voxels have **dramatically lower evidence** (mean 6.3 vs 19.8, t=10.62, p=3.4e-17) | ✅ CLOSED | Evidence encodes something richer than boundary distance |
| **E1.2 (feature-norm)** | Does evidence correlate with feature magnitude? | Pooled R² = 0.572 appears large but is Simpson's-paradox confound (tumor r=+0.907, background r=−0.680, opposite signs); controlled partial r = **+0.043 incremental** (not independent) | ✅ CLOSED (confound diagnosed, rejected as cause) | Feature-norm NOT an independent predictor |
| **E1.3 (latent geometry)** | Test separation in learned decoder representations | **AUC = 0.9995** (near-perfect); background tight (4.35), tumor **3× diffuse** (13.43); latent boundary explains evidence better than image boundary (R² 0.219 vs 0.150) | ✅ CLOSED | Sharpens target to decision-boundary margin |
| **E1.4 (density analysis)** | Does local manifold density (dec1 space) predict evidence? | Incorrect voxels **12× sparser** (ρ 2.660 vs 0.229); essentially uncorrelated with latent boundary (r = −0.0068, p=0.097) | 🟡 CLOSED (real effect, undersold by R²-framing) | Density is genuine independent signal |
| **E2 (loss-math literature)** | Audit prior work on margin/boundary geometry in embeddings + evidential uncertainty | No prior paper combines evidential head + margin loss validated by latent-vs-image boundary R² — judged genuinely novel as specific combination | N/A (literature audit) | Recommends uncertainty-weighted margin as variant |
| **E5 (EGGO design)** | Full EGGO ("Evidence-Guided Geometry Optimization"): two-term loss (separation + local-density compactification), gated by evidential uncertainty | Design locked with τ_b ≈ 0.972, τ_ρ ≈ 0.781 derived from measured E1.3/E1.4 crossover points | N/A (design) | Hyperparameters calibrated from measured data |
| **E6 (stress test)** | Full joint regression (all 4 predictors): does density matter? | Density's standardized β drops to rank 3/4 (β = −0.081) — honest downgrade from E1.4's headline | 🟡 CLOSED | Motivates E7/E8 simplification |
| **E7 (causality test)** | Temporal ordering: class-separability (latent AUC), calibration (ECE), density — which appears first during training? | Separability forms instantly (AUC 0.9945 by epoch 1) and **clearly precedes calibration**; density's ordering equivocal; boundary→evidence R² noisy/non-monotonic | ✅ CLOSED (separability precedence confirmed; partial honest answer) | Informs E8 simplification |
| **E9 (trainable boundary head)** | Replace offline boundary classifier with trainable Conv3d head on dec1 | ORIGINAL design (gradients into trunk) found WRONG during E11.5 review — makes head a second unbudgeted segmentation classifier; **corrected design detaches dec1**, head supervised but trunk-zero-gradient | N/A (design correction) | Detached design confirmed correct |
| **E10 (margin loss selection)** | Evaluate 5 candidate metric-learning formulations | Rejects SupCon (unwanted intra-class repulsion), triplet (extra sampling hyperparam), ArcFace/CosFace (many-class softmax, ill-suited); **selects De Brabandere pairwise-hinge** | N/A (selection) | Hinge margin meets all 3 requirements |
| **E11 (implementation spec)** | Full tensor-shape/gradient-flow/compute-budget specification for EGGO-v1 | Identifies NEW required fix: Û_i (uncertainty gate input) must ALSO be detached to prevent margin loss "gradient-hacking" the evidential head | N/A (implementation) | Detach points formalized |
| **E11.5 (readiness review)** | Pre-implementation PR review: does the design have fatal flaws? | Confirms dec1-detach fix; sets 6 pre-registered falsifiable failure criteria; renames margin-only design "EGGO-M" | ✅ CLOSED (readiness confirmed) | Formalizes design correctness gate |
| **E12a (smoke test)** | First code/training (2–3 epochs): v2 byte-exact to v1 when dormant; 5 of 6 pre-registered criteria pass | Mechanism verified **bit-exact** to frozen when dormant (max diff 0.0); criterion 2 (margin trend) inconclusive at 3 epochs | ✅ CLOSED | Clears path to pilot |
| **E12b/E12b.5/E12c (pilot, 30 epochs)** | Full EGGO-M pilot: mechanism active yet Dice flat | Margin **decreased** (34.29→31.17), r(margin,Dice)=−0.131 (p=0.780); mechanism works **but hypothesis NOT supported** | ❌ CLOSED (mechanism-verification null) | Triggers mechanism-failure investigation |
| **E12d (mechanism failure)** | Post-hoc: why didn't margin grow despite activation? | H1/H3/H4 supported: threshold `2×δ_d ≈ 2.0` set ~10× below real scale (0.65%→0.00% active); tau_b obsolete as head's logit magnitude grows | ✅ CLOSED (root-cause precise) | Recalibration required |
| **E12e (calibration)** | Recalibrate delta_d from measured data | **CRITICAL BUG FOUND**: `.eval()` mode gave δ_d = 0.0148 (0.0% active hinge). Corrected `.train()` mode: δ_d = **3.6659** (247× larger, ~68× BatchNorm eval/train discrepancy at fresh init) | ✅ CLOSED (bug caught, fixed) | Live EMA required for τ_b |
| **E12f (recalibrated pilot)** | Full 30-epoch pilot with E12e fixes active | Mechanism now **active throughout** (0.12–0.18% hinge plateau); Dice **still does not improve** (best 0.9063, dips −0.039 at epoch 20); margin +0.8%, r(margin,Dice)=−0.0016, p=0.997 | ✅ CLOSED ("Outcome B": mechanism active, hypothesis NOT supported) | Legitimate mechanism-verified negative result |
| **E13 (multiseed, 4 seeds)** | Seeds 1–3 identical hyperparams (no tuning): do all 4 replicat the null? | All 4 **clustered tightly** (0.9020–0.9064, mean 0.9044 ± 0.0023); EGGO-M **statistically worse than baseline** (mean Δ = −0.0037 ± 0.0020, t=−3.80, p=0.032, n=4) | ✅ CLOSED (4-seed null confirmed) | Mechanism-level negative result |
| **E13 (code audit)** | Parallel correctness audit of E12a–E12f | Forward pass/hinge/detach/balance all correct; **real LR schedule mismatch found** (T_max=50 vs actual 30 epochs, ~70× LR discrepancy at cutoff) | ✅ CLOSED (self-corrects mid-document) | Config bug traced, corrected later |
| **E14 (gradient conflict)** | Cosine similarity between L_seg and L_margin gradients on shared dec1 at every checkpoint | Stays small and **positive** (pooled +0.0938), "mildly cooperative" | ✅ CLOSED | Rules out gradient conflict as null cause |
| **E15 (decoder sensitivity)** | Causal intervention (frozen decoder): push dec1 away from opposite-class centroid, measure Dice change | Monotonic gain (0.9050→0.9443 at push=14, r=+0.9905, p=0.0001); EGGO-M's real margin growth (+0.21 units) predicts only ~+0.001 Dice at local slope | ✅ CLOSED | Reframes null as optimization shortfall |
| **E16 (margin reachability)** | Six-part decomposition: can the mechanism actually reach useful targets? | Real movement budget exists (6.47 units); 15.7–35.4% margin-aligned but net signed displacement small/negative (−0.86 to −1.59); reachability R=−0.114 | 🟡 CLOSED | Gradient correctly directed but net accumulation incoherent |
| **E17 (margin target stability)** | Four independent methods (centroid direction, gradient stability, transport, Procrustes): is the target moving? | Severe rotation epochs 1–5 (centroid cos 1→5=0.842); sharp stabilization epoch 10+ (>0.99 by 25–30); **representation never returns to epoch-1 orientation** (alignment stays flat 0.13–0.15) | ✅ CLOSED | Target instability real but time-limited |
| **E18 (representation rotation)** | Ablation (freeze BN/encoder/decoder/seg_head/lambda_zero): what causes rotation? | freeze_decoder shows **reduced rotation** (0.9468 vs baseline 0.8751, Dice 0.843); **INITIAL VERDICT: decoder is dominant source** | 🟡→❌ (REVERSED by same-day E18-followup review) | See E18-followup below |
| **E18-followup (REVERSAL)** | Same-day review of E18: decoder-frozen has **64× more margin mechanism active** yet **WEAKER margin-Dice correlation** and **LOWEST Dice** (0.8432 vs 0.9062) | freeze_decoder's margin activity (0.09972 vs baseline 0.00155) paradoxically associates with worse outcome AND weaker mechanistic coupling | ✅ CLOSED (E18 downgraded from causal to correlational; decoder stabilization is "associated phenomenon, not bottleneck") | "We stabilized a representation that isn't good enough" |
| **E19–E20 (layer-wise, dec1 update)** | Further mechanistic digging feeding E21.5's audit | — | 🟡 CLOSED (no independently actionable lead) | Feeds E21.5 audit |
| **E21 (transport analysis)** | "Margin update moves dec1 away from useful direction" — negative cosine result | Transport shows dec1 **moves away from correct direction** (negative cosine −0.24 to −0.26) | ✅ CLOSED (reinforces "not enough optimization") | Cross-checked stronger in E21.5 |
| **E21.5 (comprehensive audit)** | Read-only audit of E12–E21 (17 components: margin formula, finite-diff gradients, tau_b, AdamW, autograd, BN mutation, geometry) | All components pass integrity checks; **NO FATAL BUGS**, proceeds unchanged to E22 | ✅ CLOSED (implementation correct) | 3 C-severity interpretive issues flagged |
| **E22 (counterfactual geometry)** | Direct causal test: does descending along −∇L_margin improve Dice? | **NO**: 192/192 voxels get worse or neutral under descent; 24/24 beaten by random perturbation; pooled corr(ΔL_margin,ΔDice)=+0.2432, p=6.75e-04 (WRONG SIGN) | ✅ CLOSED (Case C: locally misaligned) | Locally minimizing margin loss is **actively harmful** |
| **E23 (algorithmic redesign)** | "Task-aligned projected margin loss" in response to E22's misalignment | Novelty audit progressively narrows claim — Round 1: rescaled pairwise-hinge for linear head (not novel); Round 2: Jacobian pullback is established differential geometry | N/A (design, self-narrowing) | Explicitly documents 3 precision corrections |
| **E24 (calibration + spec)** | Recalibrate delta_d_w=0.2553 per E12e methodology; lock experimental spec for "Gate 6" training | Verified against 2 sanity checks; condition matrix (A/B/C/E) defined; 4-level interpretation hierarchy (H1–H4) to block "Dice improved therefore mechanism worked" | N/A (design) | E's hinge-threshold bug caught & fixed pre-launch |

---

## PHASE E25: SC-TAM SIGN-CONVENTION CORRECTION CHAIN (16 sequential documents)

| Sub-Phase | Task | Result | Status | Critical Issue |
|---|---|---|---|---|
| **E25 Design** | SC-TAM (Signed Class-Conditional Task-Aligned Margin): signed loss d_ij = (z_i-z_j)^T w_hat for fg/bg pairs | Design locked from E24's finding (BG moved wrong-directed 39.1%; FG moved correctly 64.5%) | N/A | — |
| **E25 C6-2 results** | Initial training | Best Dice **0.9030** (below baseline 0.9063, −1.53pp); H1: ρ=−0.4924 (correct sign, first EGGO-family correctly-signed loss-Dice corr); 24/24 checkpoints show correct displacement direction (H2) | ✅ CLOSED | H2 favorable, Q3/Q4 conflicted |
| **E25 C6-2 Mechanism Audit** | Split H2 by confusion class: TP/FP/FN/TN; where did SC-TAM help vs. harm? | **FN moves WRONG-SIGNED 98.5%** (larger magnitude than TP's correct movement) — SC-TAM "reinforces already-correct boundary" not fixes errors | ✅ CLOSED (mechanism misfire) | Anchor-composition bias ruled out |
| **E25 C6-2 Isolation Checks** | Why does FN move wrong-signed? (under then-current framing) | Real evidential weighting reduces but doesn't eliminate FN/FP loss-share dominance; isolated single-anchor FN probes **wrong-signed 36/36** | ✅ CLOSED | Same failure at smallest scale |
| **E25 C6-2 Jacobian Localization** | Is inversion localized to parameter subset? | **NOT localized** — near-identical magnitude/sign everywhere (r=0.98 cross-pathway); all 4 confusion categories (TP/TN/FP/FN): **0% correct sign across 72/72 confusions** | ✅ CLOSED | Suspiciously clean 0% flagged as needing scrutiny |
| **E25 SIGN CONVENTION CORRECTION** | **THE PIVOT**: Root cause found — every diagnostic script E2 onward scored against `tumor→−ŵ, background→+ŵ`, **OPPOSITE** of SC-TAM's actual formula | After fix: actual correct convention is `tumor→+ŵ, background→−ŵ` — **the entire chain of "inversions" was purely a sign bug in diagnostics, not the network** | ✅ CLOSED — BUG CONFIRMED | This is THE bug the project flagged in reminders as "major self-correction" |
| **E25 Gradient-Jacobian Consistency** | Independent validation: does `cos(Δz_GD, ∇_zL) ≤ 0` hold (must for any descent step)? | **YES, 95.8–100%** at corrected convention; **ALL 48/48 voxels across 4 categories correctly signed** at 100% — NO network inversion exists | ✅ CLOSED | Entire E2–E4 "inversion" saga was the sign bug |
| **E25 Corrected Reanalysis** | Recompute E2–E4 using complement (`new = 1 − old`) | **FN now moves correctly-signed 98.0%**, FP 82.3%; **TP/TN move WRONG-SIGNED** (TP 6.1%, TN 21.1%) — errors fixed but numerically-dominant correct voxels damaged | ✅ CLOSED | Mechanism actually works on errors; collateral damage to majority |
| **E25 Representation-to-Logit** | Trace whether corrected FN/FP movement flips predictions | FN transitions 64–84%; FP **flat ~22–26%** despite growing corrective logit (unresolved FP-specific anomaly); TP shows **damaging shift 96.0%** (logit 3.73, exceeding FN's own 3.48), TN 79.5% | ✅ CLOSED | Strong support for collateral damage hypothesis |
| **E25 Population-Weighted Accounting** | Apply real transition rates to real population counts; naive extrapolation predicts outcome | Naive single-step predicts catastrophic collapse (0.86→0.64); real training held near baseline (0.9030 vs 0.9063) | ✅ CLOSED (NOT refuted) | Multi-epoch dynamics invalidate one-shot replay |
| **E25 Real Trajectory, 8-subject** | Measure real cumulative transitions from consecutive checkpoints; no extrapolation | Both A and C6-2 show healthy positive net flux late training — predicted collapse doesn't occur; on this 8-subject sample, C6-2 **above A** (0.9235 vs 0.9211), **opposite H4's direction** | 🟡 CLOSED | Unresolved: sample variance vs. real difference |
| **E25 Real Trajectory, full 125-subject** | Resolves 8-subject discrepancy at full scale | A (0.8899) **exceeds C6-2** (0.8889), matches H4. **C6-2's net flux 1.76× larger** (18,687 vs 10,614) with **better benefit/damage ratio** (1.326 vs 1.166) — C6-2 dynamics MORE favorable yet yields slightly lower Dice | ✅ CLOSED (paradox unresolved but genuine) | "More interesting, not less" |
| **E25 Spatial Error Analysis** | Compare A and C6-2 best checkpoints across all 125 subjects; which spatial category explains gap? | No single category (slice depth, boundary distance, size, fragmentation) cleanly explains gap; boundary-distance error peaks **one bin away** from boundary (1.30 at 2–4 voxels) rather than monotonically | ✅ CLOSED (non-tidy null) | "SC-TAM specifically damages boundary" rejected |
| **E25 C6-3 Gate Retrospective** | Before training C6-3, test confidence gate to filter collateral damage | Symmetric gate FAILS strict criterion (retains 23.4% signal); asymmetric ground-truth-conditioned gate **PASSES** (92.3% signal, 99.2% damage removal) | ✅ CLOSED (GO for C6-3) | C6-3 trained; scored 0.9026 (still below baseline) |
| **E25 Structural Pivot** | BUILD NEW FAILURE MODEL from A's own absolute error; small-lesion whole-component detection failure accounts for **58.9% of A's total Dice shortfall** | r(size, miss rate)=−0.601, p<0.0001; 44.0% of subjects have ≥1 fully missed component | ✅ CLOSED (pivots target to deep supervision) | Deep-supervision proposed |
| **E25 Deep-Supervision Audits (1A/1B/4-way)** | D4-only **best 0.9096** (+0.33pp); original hypothesis (improved detection) NOT supported (20.6% vs 22.0%, slightly worse); **REAL mechanism: quality improvement at small sizes** (1–50 voxel Dice 0.186→0.261, +40% relative, vanishes >1000 voxels) | ✅ CLOSED | **FIRST positive-looking lead** (but sub-threshold) |

**Summary of E25**: SC-TAM's core mechanism works as designed (FN/FP correctly-signed) — the opposite of what 3 intermediate documents concluded due to a backwards sign convention. Real puzzle: why correct mechanism + favorable net flux still yields lower Dice. Directly motivated deep supervision, project's only real Dice gain.

---

## PHASE E26–E43: POST-SC-TAM DIAGNOSTIC ARC

| ID | Task | Result | Status | Key Finding |
|---|---|---|---|---|
| **E26 (covariate search)** | Which subjects benefit from D4 beyond size+baseline-Dice? (joint R²=0.51) | CLEAN NULL after correction: surface-to-volume and intensity both traced to size reparameterizations | ✅ CLOSED (honest null) | No new covariate emerges |
| **E27 (project audit)** | Comprehensive Q&A: augmentation? patches? held-out split? GPU model? | NO augmentation, no patch/crop (whole-64³ only), 125-subject set reused (not true held-out), GPU RTX **5050** 8GB (not 2050 4GB), pooled–per-subject Dice gap 1.91pp | ✅ CLOSED | Single most important audit finding: preprocessing crushes small lesions |
| **E28 (subregion)** | Does BraTS subregion (NCR/ED/ET) composition explain missed components? | Raw ρ = 0.796/−0.687 (large) collapse to partial r = 0.02–0.04 (near-zero, all p>0.48); confound fully explained by component size | ✅ CLOSED (confound) | Edema-dominant = geometrically tiny post-resize |
| **E29 (resize survival)** | Pure geometry: do small native lesions survive 64³ resize? | **Median native component vanishes entirely at 64³** (0 voxels); 65.3% ≤5 voxels natively; 5–150 natively show **real monotonic recovery** at 96³/128³ (2–8× per doubling) | ✅ CLOSED | Motivates E30–E32, E54 resolution experiments |
| **E29 (96³/128³ training, single seed)** | Actually train at higher resolutions | 64³=0.9038, 96³=**0.8986** (worse), 128³=invalid (oscillated 0.53–0.75, BatchNorm batch=1 artifact) | 🟡 INVALIDATED (high-res doesn't cleanly help) | 128³ instability explicitly recognized as artifact |
| **E30 (DTC feasibility)** | "Degradation-Trajectory Constraint": resolution-survival curve as loss weight; Gates A/B | Gate A (near-binary threshold) **PASSES**; Gate B applies only to 25.8% (large components, opposite motivation); **qualified GO** but flagged as misleading | 🟡 (pass on wrong population) | THREE bugs in one document caught and corrected |
| **E31 (small-lesion DTC)** | DTC reformulation safe for small lesions; both original and reformulated forms | **KILLS both**: small-lesion population has 522/611 components with **zero footprint at all α**, making ratio noisy (not real); honest 89-component population: ρ=0.255 **FAILS permutation safeguard** (worse than 96.5% of shuffles) | ✅ KILLED | Division-by-vanished-footprint artifact |
| **E32 (α_c feasibility)** | Critical-resolution α_c against freshly-computed real component Dice | Full-population ρ=0.676 exceeds **all 500 permutation trials**; small-lesion (≤150 native voxel) partial Spearman ρ=+0.132 (p=0.0014), survives permutation (0/500) and bootstrap CI [0.144,0.346] | ✅ CLOSED as validated MEASUREMENT | Signal real, partial/multicollinearity artifact in naive OLS resolved by Spearman |
| **E33 (α_c novelty audit)** | Literature novelty check for α_c weighting; closest: Component-Adaptive Tversky (arXiv 2604.08015), size-inverse-power only | No exact collision; persistent homology is formal analogue but filters intensity, not spatial resolution | N/A (audit) | GO to controlled comparison |
| **E34 (Adaptive Size-Reweighting)** | Both [S] (size-based) and [R] (α_c-based) component-loss reweighting | **CATASTROPHIC FAILURE**: [S]=0.7360 (−17.03pp), [R]=0.7108 (−19.55pp); all-tumor collapse signs | ✅ KILLED | **CRITICAL BUG FOUND**: value-only calibration left gradient **~75× too large** on first batch; retroactive reconstruction; origin of mandatory "gradient-magnitude, not value" calibration policy |
| **E35 (scale-context feasibility)** | Tests Gaussian-blurred "contextual field" survival under coarsening | Exact outline 100% vanishes at D4; blurred field also does NOT survive meaningfully (>0.1 threshold = 0% for small fragments) | ✅ KILLED | Catches own spurious "100% survival" before trusting |
| **E36 (deep-supervision autopsy)** | WHY does D4-only (+0.33pp) beat D2-only/Both? Mechanistic decomposition | D4 loss **"stuck" at ~0.15** (never converges like main loss); ratio grows 0.69→2.74; gradient alignment becomes LESS aligned over training (0.26→0.84→0.25); real explanation: **boundary/partial-coverage problem** (partial-coverage voxels ~10× error rate), not size-only | ✅ CLOSED (mechanistic explanation) | D4 keeps supplying gradient where main loss converged |
| **E37 (grid-entropy feasibility)** | Per-tumor grid-ambiguity (binary entropy of coverage fraction) captures E36's story? | **KILLS**: small lesions coarsening NEVER increases entropy (0% show D4>D2); mathematically impossible (diluting small mass→0 coverage=LOW entropy) | ✅ KILLED (proven mathematically necessary) | Breaks 3-of-4 pre-declared tests intentionally per discipline |
| **E38 (orthogonal residual)** | Does D4's "orthogonal residual" (gradient NOT explained by main gradient) predict benefit? | Pre-declared I_r **FAILS** (ordering reversed); pre-declared Q_r **SURVIVES** (matches known ordering, p<0.00001, 95% pairwise dominance) | ✅ CLOSED (Q_r) / ❌ (I_r) | Alternative passed demanding test; pre-declared alternative BEFORE testing |
| **E39 (mechanism test)** | Does E38's Q_r/P4/M4 predict **new** λ-sweep (6 conditions, λ=0–2.0)? | P4/M4 **monotonically climb with λ** while Dice shows inverted-U (peaks at λ=0.25); **fundamentally different shapes**; λ=0.25 (best Dice) has LOWEST P4/M4 | ✅ CLOSED (null) | Clean, confound-checked null |
| **E40 (subspace audit)** | D4's diversity of independent optimization directions (effective rank) explains benefit? | Pooled 2/3 comparisons, not distinguishable from chance (p=0.17); "Both" condition: significant (p=0.016) but **does NOT survive permutation safeguard** (87th percentile of 500 shuffles) | ✅ KILLED | Effective-rank cross-verified against synthetic data (8 decimals) |
| **E41 (transformation error)** | Residual prediction error against mathematically-correct fractional-occupancy coarse target? | CONFIRMED fractional target used (no thresholding bug); correlation ρ=−0.002 (detected), ρ=+0.09 (rescued); permutation: real result at 23rd percentile | ✅ KILLED | Honest gap disclosure: small-fragment split untested (only 7 remained) |
| **E42 (cross-scale operator audit)** | Structural KILL: does code assume resize pipeline that doesn't exist? | **PROVES FALSE PREMISE**: three entirely separate, independently-parameterized heads (no resize anywhere) | ✅ KILLED (structural, pre-data) | — |
| **E42 (curvature, incomplete)** | Lanczos spectral analysis of E39 λ-sweep; method verification passed | Stopped mid-execution (57/72 records, no error message) | N/A (ABANDONED/INCOMPLETE) | Distinguished from retroactive writeups; genuinely never finished |
| **E43 (representation change)** | Does E36's boundary-localization hypothesis hold at representation level (forward-only, 125 scans)? | Representation changes measurably (each model>other model far more); boundary-localization hypothesis **INITIALLY showed opposite** (interior>boundary) | ✅ CLOSED (correlational null after correction) | Confound fixed inline: interior mislabels boundary-adjacent in small lesions |

---

## PHASE E44–E51: POST-PIVOT ARCHITECTURAL ARC (FIRST SINGLE-MECHANISM ATTEMPTS)

| ID | Task | Result | Status | Key Finding |
|---|---|---|---|---|
| **E44 (RCGW v1/v2)** | Relative-Convergence-Gap Weighting: dynamic weight based on relative loss convergence | v1 caused **real collapse** at epoch 6 (val Dice 0.7135 vs 0.8463); v2 (saturating transform) fixed instability but **no signal** (mean Δ=−0.35pp) | ✅ KILLED | Raw-ratio instability mechanism identified precisely |
| **E45 (D4+D8, UNet3D_v4)** | Add 8³ bottleneck auxiliary head to D4; 3-seed confirmed | Single-seed pooled **0.9110** (+0.47pp); per-subject doubly-significant 0.8986 (+1.44pp); **3-seed mean per-subject 0.8939**, just **BELOW corrected +1pp target** (0.8942) by 0.0003 (margin smaller than 1/10 std) | 🟡 NOT MET (3-seed inside noise) | First multi-seed check; established ≥3-seed mandatory policy |
| **E46 (attention gate, UNet3D_v5)** | Bottleneck-conditioned attention gate on enc1 skip; 3-seed | Pooled **0.9102** (+0.39pp); per-subject 0.8949 (+1.08pp, Wilcoxon-only); **inside noise range** | 🟡 NOT MET | Gate learned real non-degenerate spatial pattern (not collapsed) |
| **E47 (causal gate audit)** | Causal test: does attention-gate correlate with boundary-routing (the original motivation)? | Clamping ψ=1 boundary vs. matched-count interior: **NO significant difference** (drop=+0.00025, p=0.808) | ✅ CLOSED | Confirms E43's correlational null causally; routing diffuse |
| **E48 (bottleneck causal necessity)** | Causal ablation: sever bottleneck entirely, measure Dice drop vs. tumor size | Spearman(native_size, drop) = **−0.454**, p<0.001 — **SMALL lesions depend MORE on bottleneck** (opposite hypothesis direction) | ✅ PROVEN (re-verified E85/E86: ρ=−0.3834, p=1.02e-05, same sign/order) | **STILL LIVE** — unexplained anomaly spawns entire E48–E97 diagnostic chain |
| **E49 (CCABA, UNet3D_v6)** | Causally-Calibrated Adaptive Bottleneck Amplification: size-conditioning fit directly to E48's curve (R²=0.260) | Single-seed **0.9114** (+0.51pp); **3-seed mean 0.9094** (+0.31pp), CI [0.9046,0.9142] **crosses D4-only** (statistically tied) | 🟡 NOT MET (3-seed CI crosses zero) | Amplifies already-utilized pathway (explained later by E89) |
| **E50 (IECG, UNet3D_v7)** | Internally-Estimated Counterfactual Gating: live bottleneck-ablation replay during training | 3-seed: 0.9081/0.9030/0.9084, mean **0.9065** (+0.02pp), worse than CCABA on all 3 seeds despite ~2.5× cost | ✅ KILLED | Mechanism verified working; just unhelpful |
| **E51 (CCAG, UNet3D_v8)** | CCABA + attention gate combined; "last single-architecture attempt" | 3-seed mean **0.9095** (+0.32pp), statistically tied with CCABA alone (p=0.919, tightest variance any condition, std≈0.05pp) — **did NOT compose additively** | ✅ KILLED | Gate's effect too diffuse to synergize |

---

## PHASE E52–E57: FAILED CALIBRATION & CURRICULUM ATTEMPTS

| ID | Task | Result | Status | Key Finding |
|---|---|---|---|---|
| **E52 (ASR recalibration)** | ASR component-weighting re-attempt with **true gradient-magnitude match** (43× blowup corrected) | Smoke test: val Dice **0.10** epoch 1 — same **degenerate collapse as E34** despite magnitude fix | ✅ CLOSED | Direction, NOT magnitude, fights early curriculum |
| **E53 (ASR + warmup)** | ASR with curriculum warmup (delay term epochs 8–20) | Only seed 0 of 3 completed (user stopped): **0.8969**, below baseline | 🟡 INCOMPLETE (1 of 3 seeds) | Instability fixed; competitiveness unresolved |
| **E54 (A96, 96³)** | Whole-volume training at 96³ (no arch change), testing E29's unexecuted proposal; batch=2 physical only (BatchNorm sees only 2) | Single-seed pooled **0.9066** (+0.03pp, dead tie, −0.30pp vs D4-only); per-subject **0.8981** (+1.39pp, doubly-sig); **3-seed mean per-subject 0.8944**, **technically clears corrected target (0.8942) by 0.0002** (margin <1/10 std) | 🟡 NOT MET (margin too small) | Explicitly NOT treated as "confirmed" vs E45 |
| **E55 (dual-resolution, UNet3D_v9)** | Local refinement (differentiable soft-centroid crop+fusion), motivated by E54+E48 | Single-seed pooled **0.9007** (−0.56pp, below baseline, D4-only, E54); fusion_gate declined (0.356→0.234); HD95 improved (1.30, best) | ✅ CLOSED | Mechanism worked, unhelpful on outcome |
| **E56 (measurement audit)** | Pooled Dice inflates every mechanism 0.85–2.40pp vs per-subject; canonical "0.9063" was never variance-checked | Corrected baseline: **4-seed mean 0.9044** (not 0.9063), CI [0.9007,0.9080]; corrected +1pp target **0.8942** (per-subject, not pooled) | ✅ CLOSED (methodology correction) | E45/E54 re-scoring: already clear corrected target on seed 0 |
| **E57 (multi-stage bottleneck)** | [Not separately detailed in master report] | — | — | — |

---

## PHASE E58–E72: BOTTLENECK & SKIP-CONNECTION CAUSAL CHAIN (THE CORE DIAGNOSTIC ARC)

| ID | Task | Result | Status | Key Finding |
|---|---|---|---|---|
| **E58 (Stage 1, octant ablation)** | E48 follow-up: does full-ablation size-signature localize to specific bottleneck octant? | **NO**: neither octant reproduces ρ=−0.454 (octant A −0.186, B −0.081) | ✅ CLOSED | Non-localized effect |
| **E58 (Stage 2, donor substitution)** | Substitute foreign subject's bottleneck: worse than zeroing? | YES, p=0.026; replicated stronger (p<0.0001) — initially "wrong context worse than none" | 🟡→✅ | Later revealed OOD ablation artifact (E58 Stage 3) |
| **E58 (Stage 3, OOD confound)** | Mean-bottleneck ablation (in-distribution, generic) vs. foreign donor (out-of-distribution) | OOD damages far more, matching **published confound** (Li & Janson NeurIPS 2024) | ✅ CLOSED | Real-time walk-back within same file |
| **E59 (coalition, size-stratified)** | Bottleneck coalitional interaction: does full bottleneck have synergistic/redundant parts? | Population-average NOT sig (p=0.146); **size-stratified**: small synergistic (+0.00104, p=0.049; smallest quartile p=0.0003); large redundant (−0.00192, p<0.0001) | ✅ CLOSED | Small-lesion info **synergistic, coalitional** |
| **E60 (DBC feasibility)** | Does E59's small-lesion synergy reproduce with DBC's actual balanced 4-vs-4 sampling? | **NO**: small-lesion Γ p=0.530 (not significant); large-lesion redundancy DOES reproduce | ✅ KILLED (zero-cost pre-training) | Sampling-granularity mismatch OR signal fragility |
| **E61 (novelty search, mechanism)** | Literature audit on skip-correspondence fixes, 8 search rounds | Base mechanism (Dynamic U-Net DCU) occupied; offset-learning adjacent-occupied; position/semantic decoupling occupied; size-conditional variant not found | N/A (literature audit) | Base mechanism occupied; narrow calibration claim survives |
| **E61 (dimensionality audit)** | Higher-effective-rank bottleneck explains small-lesion dependence? (3 metrics tested) | **ALL FAIL**: effective rank shows REVERSE (large>small); all 7 pre-declared GO criteria failed | ✅ KILLED | "Yet another framing" to fail |
| **E62 (MaxPool position, INVALIDATED)** | MaxPool3d position-discarding causally hurts small lesions? | Initial claim: Dice drop **0.0226**, p=2e-28, size-specific ρ=−0.355 — **INVALIDATED** | ❌ INVALIDATED | **SHARED-TENSOR BUG**: pool1 reused same enc1 for both input AND skip, making pool1(x)=pool1(π(x)) (identity, mathematically impossible to harm) — effect came entirely through skip |
| **E63 (winner/runner-up competition, INVALIDATED)** | Winner/runner-up geometry explains pooling loss? Test A/B/C | **INVALIDATED**: inherits E62's shared-tensor bug; Test C signal diffuse | ❌ INVALIDATED | Same bug as E62 |
| **E64 (corrected split-intervention)** | E62/E63 corrected: fully independent tensors E_pool and E_skip | Δ_pool = **0.0 exactly** (all subjects/draws — proves pool1(x)=pool1(πx) to float precision); Δ_skip REAL and LARGE (v5=0.0226 exact match E62, v3=0.0270 LARGER without gate) | ✅ CLOSED (skip-connection effect confirmed, pooling effect = 0) | Effect relocated from pooling to skip; rules out gate as mechanism |
| **E65 (skip decomposition)** | Decompose skip-connection sensitivity into 4 properties (v3, ungated) | Translation **+0.2143** (dominant); channel permutation **+0.1009** (secondary, 4× smaller); smoothing +0.0284; local permutation +0.0270 | ✅ CLOSED | Absolute spatial correspondence is dominant factor |
| **E66 (gradient-topology)** | FocalTversky vs EvidentialBeta gradient cosine similarity at 7 checkpoints | Cooperative EVERYWHERE (encoder 0.845, bottleneck 0.813, decoder 0.463), never negative | ✅ CLOSED | Real, ordinary depth gradient; not conflict |
| **E67/E67b (novelty search, correspondence)** | 8-round search: skip-correspondence correction novelty? | **Direct collision**: Dynamic U-Net DCU (arXiv:2403.07303, 2024); offset-learning adjacent-occupied; position/semantic decoupling occupied | N/A (literature audit) | Base mechanism occupied |
| **E68 (SC-DCU design)** | Size-conditional deformable skip correction; original Sections 1–5 retracted in-place | Original claim (hard-equality offset-fidelity loss) **retracted**: synthetic-deformation and transformation-consistency both well-established | N/A (design retraction) | Extensive in-place self-correction |
| **E69 (E54 offline consistency)** | E54 existing 3 seeds: how many subjects consistently improve? | **52/125 (41.6%)** consistently improve across all 3 seeds vs 25% expected by chance (p<0.0001) — real, non-noise subject-level effect | ✅ CLOSED (consistent sub-population exists) | 1 more seed recommended, not full 7-seed replication |
| **E70 (CAS, UNet3D_v10)** | Correspondence-Aware Skip, pre-registered from E65; 3-seed test | Single-seed MM_CAS vs MM +0.30pp (CI [+0.07,+0.53] excludes 0, inside noise band); **3-seed: +0.30/+0.42/−0.01pp, mean +0.24±0.22** — **unreplicated** | ❌ KILLED (unreplicated; mechanism falsified) | Gate diagnostic opposite predicted direction ρ=−0.388 unanimous 0/125 |
| **E71 (prediction 1)** | CDCG (Causally-Distilled Capacity Gating): bottleneck predicts its own E48-style causal sensitivity | Held-out ρ=**0.701–0.87** across runs; partial ρ up to **0.904** controlling for size (stronger than raw) | ✅ CONFIRMED (strongest standing finding post-pivot) | Network carries real, recoverable, self-knowledge |
| **E71 (prediction 2)** | Does predicted sensitivity match real E48 sensitivity? Small lesions | 200-label run: FAILS (ρ=+0.395, wrong direction); **1126-label re-test: PASSES MARGINALLY** (ρ=−0.190, p=0.035, correct sign; partial ρ=0.855 strong) | ✅ CLOSED (after fixing undertraining artifact) | "Not emphatic confirmation" but marginal pass |
| **E71 (gate mechanism)** | Does bottleneck-sensitivity predict tolerance to enc1-skip suppression? | Spearman(d_i, tolerance) = **−0.290** (parametric p=0.001, perm p=0.001) — **BACKWARDS** from requirement | ✅ KILLED (gate specifically) | Bottleneck and skip are correlated markers of general difficulty, NOT substitutable |
| **E72 (FWL)** | Fragility-Weighted Loss: use E71's frozen aux-head as per-subject loss weight | v1 (aggressive) −0.75pp (sig p=0.0001, mechanism ρ=+0.21); v2 (gentle) −0.10pp (n.s. p=0.53, mechanism ρ=+0.07 gone) | ✅ KILLED (both variants) | Cost and targeting shrink together |

---

## PHASE E73–E84: SPATIAL & MECHANISM REFINEMENTS (THE FINAL DIAGNOSTIC CHAIN)

| ID | Task | Result | Status | Key Finding |
|---|---|---|---|---|
| **E73 (SDLR, UNet3D_v11)** | Self-Diagnostic Localized Refinement: reframe fragility signal from scalar to spatial map | Stage 1 probe **22.6× error/correct enrichment**, p<1e-9; Stage 3 distilled ρ=0.394–0.414; **downstream distillation collapses** (22.6× → 1.07–1.19× only) | ✅ CLOSED | Spatial structure exists but distillation loses it |
| **E74 (diagnostic reframing)** | E71–E73 kills finish; reframe: translation-sensitivity generalizes across depth? | enc1 drop **0.317**, p=2.0e-64; bottleneck **0.211**, p=1.4e-23; feature magnitude consistent positive predictor (partial ρ enc1=+0.301, bneck=+0.525); frequency REVERSES (ρ=−0.589 bneck, smoother=fragile) | ✅ STILL LIVE | Magnitude correlation established; frequency-reversal open question |
| **E75 (magnitude-direction decomposition)** | Rescale magnitude only (direction fixed), alpha∈{0.25,0.5,1.0,2.0} | enc1 **PASSES** cleanly (monotonic 0.204→0.372, p<1e-14); bottleneck **FAILS REVERSED** (0.213→0.247→0.211→0.125, p=1.7e-15 wrong) | ✅ enc1 LIVE / ❌ bottleneck CLOSED | Bottleneck reversal itself unexplained |
| **E76 (enc1 regime)** | Tight non-OOD alpha range (0.70–1.30) addressing E75's confound | Both pre-declared conditions **PASS** — D_intact within 0.0101, S(alpha) **strictly monotonic** across 7 points, local slope p=7.4e-42 | ✅ CLOSED (clears path for enc1 intervention) | Confirms magnitude→sensitivity genuine LOCAL property |
| **E77 (natural geometry)** | enc1 magnitude distribution (125 subjects): shape, lesion dependence? | **Sharply bimodal** (p10=p25=p50=1.389, jump to p90=6.158); high-mag IS lesion (interior 10.906 vs bg 2.350); pooled ρ=+0.539 but subject-level ρ=+0.132 n.s. | ✅ CLOSED | Caution: magnitude suppression would destroy lesion salience |
| **E78 (z=r·u decomposition)** | Causal decomposition into magnitude-only vs direction-only hybrids | S_full=0.3174 (100%); S_r (mag-only) 0.0223 (~7%); **S_u (dir-only) 0.2134 (~67%), 9.6× larger** (p=2.2e-55); **superadditive interaction** +0.082 | ✅ CLOSED (key causal link) | Direction carries positional structure, magnitude amplifies |
| **E79 (directional dependence)** | Is direction locally noisy (fixable by smoothing) or bound to absolute coords? | Direction locally coherent (cos θ 0.88–0.95); pre-smoothing recovers only 11.0%; local permutation causes 17.4% damage vs translation's 0.2134 — **H_B CONFIRMED** | ✅ CLOSED | Decoder uses direction as absolute-coordinate lookup key |
| **E80 (decoder readout)** | 4-arm split-forward: encoder-only, decoder-upconv1-only, coherent joint, mismatch sweep | Arm 1 (enc-only) 0.2134; Arm 2 (dec-only) **0.2615**; Arm 3 (joint) 0.4386 WORSE than Arm 1; Arm 4 (mismatch) monotonic in upconv1's absolute offset | ✅ Arms 1/2 / ❌ Arm 3/4 (later retracted by E81) | Decoder's upconv1 independently translation-sensitive |
| **E81 (realignment control, E80 correction)** | E80's confound check: does undoing coherent shift on output recover Arm 3's apparent damage? | **YES, 93.5% recovery** (0.4637→0.029, p=3.0e-84); equivariance check confirms near-equivariance (error 0.096) | ✅ CLOSED (E80 Arm 3/4 retracted; Arms 1/2 reaffirmed) | Network near-equivariant under coherent joint shift |
| **E82 (local donor)** | Local donor identity: anisotropy and distance-grading? | Anisotropic (axis range 0.090 vs mean 0.155, z-hurts-less); distance-smooth (radius 1→3: 0.0274→0.0450, all p<0.02, ratio 0.61) | ✅ CLOSED | Moderate anisotropic, distance-graded structure |
| **E83 (anisotropy confound)** | E82 follow-up: BraTS resize non-uniform (x/y 3.750mm/vox, z 2.422mm/vox) — voxel-offset ≠ physical distance across axes | Voxel-matched reproduces E82 (0.0191, p=1.8e-3); **physical-distance-matched SHRINKS ~48%** to 0.0100 (p=0.142 loses sig) | 🟡 (effect real voxel-level, ambiguous after confound) | Cross-axis "learned anisotropy" doesn't survive scrutiny; within-axis distance-grading unaffected |
| **E84 (contribution audit)** | Freezes experimentation at E83: does E48–E83 evidence license a genuinely NEW architectural operation? | **Answer: NO** — standing findings don't fall outside already-occupied deformable/attention/spatial-transformer space | ✅ CLOSED (research-design decision) | Enumerates 4 bugs caught across chain as methodological contribution |

---

## PHASE E85–E107+: FINAL NECESSITY-ALLOCATION & INFORMATION-LOSS DIAGNOSTICS

| ID | Task | Result | Status | Key Finding |
|---|---|---|---|---|
| **E85 (necessity-allocation mismatch)** | Spearman(N_b bottleneck-necessity, A_b gradient-L2-allocation) = ? | ρ = +0.0644, p=0.475 (essentially zero); **later shown to be proxy artifact by E89** | ⏳ SUPERSEDED | Measurement proxy artifact, not real property |
| **E86 (compensating-circuit)** | Superadditivity ratio (joint ablation / individual sum): >1 for compensation? | Median=0.802, mean=0.815, p=1.0 (one-sided >1) — **SUB-additive, opposite published transformer finding** | ✅ CLOSED | NO compensation confound; pathways don't substitute |
| **E87 (necessity predictor)** | Can mask-geometry+entropy features cheaply predict E48-style causal necessity? | 3 model classes (ridge/GBR/RF) all fail (mean CV R²∈[−0.01, 0.07], threshold 0.10); trees worse than linear | ✅ CLOSED | E48-necessity not cheaply predictable at n=125 |
| **E88 (temporal necessity-allocation)** | Cross-sectional NULL stable (ρ range −0.04 to +0.11, all p>0.22); temporal deltas coupled? | **Temporal ΔN_b & ΔA_b coupled** (ρ=+0.234, p=8.7e-11); A_b(t) weakly predicts worse gain (ρ=−0.096, p=0.008) | 🟡 SUPERSEDED by E89 | Q3 later killed as pooling artifact |
| **E89 (MAJOR CORRECTION)** | Functional utilization U_b (not gradient-norm A_b) vs. causal necessity N_b? | **U_b tracks N_b ALMOST PERFECTLY** (ρ up to +0.95, all p<0.001 from epoch 5+); gradient-norm G_b stays near zero (−0.04 to +0.11) at SAME checkpoints | ✅ CLOSED | **Entire E85–E88 "mismatch" premise = proxy artifact; model DOES functionally rely on bottleneck proportionally to necessity** |
| **E90 (info decodability)** | Bottleneck decodability (frozen linear probe): small vs large lesions? | Significantly **lower small lesions** (0.652 vs 0.759, perm p=0.019); rules out H3 (decoder-extraction-limited) | ✅ CLOSED | Cannot distinguish capacity-limited from upstream-loss |
| **E91 (encoder-stage)** | Decodability across stages (enc1/enc2/enc3/bottleneck)? | enc1/enc2 no deficit (p=0.846/0.642); enc3 trending (p=0.083); **bottleneck significant** (p=0.026) — **gradually BUILDING deficit** | ✅ CLOSED | Localizes to bottleneck's aggressive final compression |
| **E92 (pool vs. transform)** | Delta_pool (MaxPool3d loss alone) vs delta_transform (channel-mixing) for small/large? | **Delta_pool significant** (small=+0.127, large=+0.062, p=0.022); delta_transform NOT significant, actually improves | ✅ CLOSED | MaxPool3d **specifically** disproportionately discards small-lesion signal |
| **E93 (mechanism: WHY pooling loses small-info)** | Winner-identity bias? Rank-recovery? Activation-margin? | All 3 KILL or negligible; information appears **genuinely destroyed**, not misweighted | ✅ CLOSED (none explain) | Anomaly: proven lost, mechanism unexplained |
| **E94 (neighborhood recovery)** | Can small-lesion info at pooling be recovered from wider linear neighborhood? | **NO**: small decodability **declines with context** (r1=0.596→r4=0.551); **information genuinely unrecoverable** | ✅ CLOSED (buggy original run retested cleanly) | **E48→E94 branch FORMALLY CLOSED** per user directive |
| **E95 (causal necessity + context)** | Does causal necessity predict context-benefit? (rho(N_b, context-benefit)=?) | **ρ=+0.436**, p=3.7e-7, confound-independent of size (partial ρ=+0.437) | ✅ PASSED | Real, evidenced link; still live |
| **E96 (timing diagnostic)** | Does benefit track TIMING-specific resolution gap, or resolution generally? | Low/mid severity gaps: ρ nearly EQUAL (0.824/0.768, both p<1e-25) — **kills curriculum-TIMING, strengthens N_b→resolution-benefit** | ✅ CLOSED | Underlying finding strengthened, timing redirected |
| **E97 (NC-LCA, final attempt)** | Necessity-Conditioned Lesion-Centric Augmentation bypass around pool3 | Neither **CONSTANT** (Δ=+0.0125, p=0.076) nor **NC-conditioned** (Δ=+0.0037, p=0.301) lambda recovers small-lesion decodability | ✅ KILLED (Gate 1 failure decisive) | **E48→E97 branch PERMANENTLY CLOSED**: 6 independent attempts all failed |
| **E98 (literature verification)** | BraTS 2025 workshop paper (arXiv 2512.14937) "+0.9% Dice" claim? | Verified from full text: ranking metric gain (1.137→1.127), actual per-region Dice ~0.001, SSA task Dice unchanged | N/A (literature verification) | Rejected; not viable |
| **E99 (NC-LCA zero-training)** | Model fragility to lesion-region elastic perturbation + N_b tracking? | **Negligible fragility** (mean Dice drop 0.0038); fragility **doesn't track N_b** (ρ=+0.101, p=0.246; partial ρ=−0.0001) | ✅ KILLED | No differential-necessity signal for conditioning |
| **E100 (plain LCA smoke)** | Unconditioned LCA vs baseline (UNet3D_v5, 15 epochs) | LCA **0.8927** vs baseline **0.8977** — below 0.5pp threshold | ✅ KILLED | Augmentation-diversity lever closed |
| **E101 (CC-DiceCE literature)** | arXiv 2511.17146: specific documented BraTS failure (precision drops, recall rises) | Found & documented; mechanism explained by source paper; untested on NeuroScan model | N/A (literature verification) | Still live (tested/closed E103) |
| **E102 (smoothness-vs-data)** | Graph-cut diagnostic: smoothness/data ratio elevated at missed voxels? | Elevated at FN (4.045 vs TP 0.493, p<0.0001) BUT **doesn't correlate N_b** (ρ=0.141, p=0.262); root: **data evidence 2.5× weaker at missed voxels** | ✅ CLOSED | "Evidence-was-never-there story," not over-regularization |
| **E103 (CC-DiceCE audit)** | CC-DiceCE reimplemented exactly, 5-epoch test vs baseline | **OPPOSITE mechanism**: FEWER FP (−12.3%), MORE FN (+16.5%) vs paper's documented BraTS failure direction | ✅ KILLED | Mechanism doesn't transfer; doesn't manifest predicted failure |
| **E104 (EDL calibration)** | Evidential head's own uncertainty (S=alpha+beta): correct uncertainty ordering at missed voxels? | **CORRECT ordering**: FN=9.02 (lowest), TN=18.22 (high), TP=15.42 (mid), p<0.0001 monotonic | ✅ CLOSED as "calibration" story killed | Model's uncertainty signal is NOT broken; correctly flags missing info |
| **E104b (S vs N_b bridge)** | Raw correlation NULL; multiple regression after controlling size/baseline? | Raw ρ=−0.122 n.s.; **partial coefficient negative −0.359, p=0.0094** (R²=0.37) — high-N_b subjects show lower evidence after confound control | 🟡 PAUSED | Fragile foundation; explicit user pause instruction |
| **E105 (Hausdorff gradient)** | Normalized distortion worse for small lesions? Argmax overconfidence? | Distortion WORSE small (0.161 vs 0.090, p<0.0001); **MAJORITY (61.6%) low-confidence, NOT false-overconfidence** | ✅ KILLED (4th independent method confirming genuine evidence scarcity) | No Hausdorff pathology (would need false overconfidence) |
| **E106 (Integrated Gradients)** | IG vs local-evidence disagreement: magnitude and confound check | **Strong at FN** (ρ=−0.431) vs weak detected (ρ=−0.067), diff p<0.0001; **confound-independent** of size (p=0.728) | ✅ CLOSED (E106 clears every pre-reg correlational gate) | **FIRST candidate E98–E107 to pass ALL checks** |
| **E107 (causal do(X) test, E106 validation)** | IG-flagged region disruption vs evidence-flagged vs random | IG-flagged **much larger change** (p<0.0001 both); **FN-specificity REVERSED** (TA=0.810 vs FN=0.109, p<0.0001, replicated in logit space) | ✅ CONFIRMED (E107-B/C) / ✅ KILLED (FN-specificity, CASE_3_KILL for verified reason) | IG identifies causally load-bearing regions even in failed voxels; FN sits in flatter regime |

---

## SUMMARY TABLE: WHICH EXPERIMENTS SUCCEEDED

### Positive Results (Dice gain >+0.0pp, mechanistically explained)

| Experiment | Model | Pooled Dice | Per-Subject | Mechanism | Status |
|---|---|---|---|---|---|
| **Deep Supervision (D4-only)** | UNet3D_v3 | **0.9096** (+0.33pp) | 0.8981 (+1.39pp) | Quality improvement (not detection) in small lesions 1–50 voxels (+40% relative Dice); D4-only loss "stuck," supplies gradient after main loss converges | ✅ ONLY POSITIVE RESULT — sub-threshold (+0.33pp < +1.0pp bar) |
| **Deep Supervision (Both)** | UNet3D_v3 | 0.9091 (+0.28pp) | — | Same mechanism as D4 | Slightly worse than D4-only; not statistically distinguishable |
| **Deep Supervision (D2-only)** | UNet3D_v3 | 0.9080 (+0.17pp) | — | Same mechanism, weaker | Benefit concentrated in 10/125 scans (73% of total gain) |

### Measurements/Findings (No Dice improvement, but validated findings)

| Finding | Validation | Status | Significance |
|---|---|---|---|
| **α_c (critical resolution)** | Survived 500 permutation trials, bootstrap CI excludes zero, size-independent | ✅ VALIDATED MEASUREMENT | Real modest predictor beyond size; algorithm attempt (E34) derailed by calibration bug (not yet retried correctly) |
| **E48 bottleneck size-dependence** | Proven: ρ=−0.454 (small→high dependence); re-verified fresh ρ=−0.3834 | ✅ PROVEN ANOMALY | UNEXPLAINED mechanistically after 6 intervention attempts (E49, E50, E51, E52, E72, E97) all failed |
| **E71 Prediction 1** | Bottleneck predicts its own causal necessity; held-out ρ=0.87, partial ρ=0.904 (strongest standing finding) | ✅ CONFIRMED | Network carries real, recoverable self-knowledge; never successfully turned into algorithm |
| **E65/E78/E79 Skip correspondence** | Translation ≫ local permutation (9.6× larger); direction-based, not magnitude; absolute-coordinate-bound | ✅ PROVEN MECHANISM | Decoder uses direction as absolute-coordinate lookup; smoothing recovers only 11% |
| **E89 Functional necessity tracking** | U_b (utilization) tracks N_b ρ=+0.95; gradient-norm doesn't (ρ≈0) at same checkpoints | ✅ PROVEN CORRECTION | E85–E88's "mismatch" entirely proxy artifact; model DOES functionally allocate to necessity |

### Null Results (No Dice improvement, mechanism verified/investigated)

| Idea | Mechanism Status | Why Failed | Status |
|---|---|---|---|
| **ABO** | Mechanism active, correctly controls gradient ratio (~2×); margin loss non-conflicting | Controlling gradient ratio doesn't move outcome | ✅ Null confirmed |
| **EGGO-M margin loss (entire family)** | Mechanism mathematically correct, active in training, gradients properly signed; boundary-routing rejected as cause; margin grows inadequately | Locally minimizing margin loss is actively harmful (E22); accumulation incoherent despite local correctness | ✅ Extensively diagnosed null |
| **SC-TAM (E25 entire chain)** | Mechanism works correctly on errors (FN 98% correct-signed); real collateral damage to already-correct voxels (TP/TN ~90% wrong-signed) | Collateral damage to numerically-dominant population exactly offsets error correction gain | ✅ Mechanism-verified null |
| **E48 interventions (6 attempts)** | All 6 (CCABA, IECG, CCAG, ASR, E73 SDLR, E97 NC-LCA) pass internal mechanism checks; zero gain on Dice | Documented causally-necessary structure does NOT translate to exploitable algorithmic lever | ✅ Six-fold null confirmed |
| **CCABA, IECG, CCAG** | Amplification mechanisms work; attention gate real; compositionality tested | No scaling | ✅ Individually killed or statistically tied |
| **Higher resolution (naive)** | 96³ geometry shows recovery; 96³ training possible but unstable; 128³ breaks BatchNorm | Simple upsampling doesn't help (0.8986 worse than 0.9038) | ✅ Invalidated (batch=1 BatchNorm issue isolated) |
| **DTC (Degradation-Trajectory Constraint)** | Both original and reformulated versions mathematically well-founded | Division-by-vanished-footprint artifact at exactly the small-lesion population it targets | ✅ Killed pre-training (zero-cost gate) |
| **Grid entropy** | Geometric proof: small lesion coarsening CANNOT increase entropy (mathematically necessary impossibility) | Pre-registered gate failure; mechanism theoretically impossible | ✅ Killed pre-training |
| **FWL (Fragility-Weighted Loss)** | Fragility signal real and independent; weighting mechanism works as designed | No regime found where weighting helps without net cost; cost/targeting shrink together | ✅ Both FWL variants killed |
| **SDLR (Spatial refinement)** | Spatial structure exists (22.6× enrichment); distillation fails to preserve it (→1.07× only) | Lightweight predictor loses almost all error-localizing power; ceiling on distillation, not data-scale artifact | ✅ Refinement pilot killed |
| **CAS (Correspondence-Aware Skip)** | Mechanism real (magnitude-gated smoothing); single-seed +0.30pp inside noise range | **Unreplicated across 3 seeds** (+0.30/+0.42/−0.01pp, mean +0.24±0.22); intended mechanism falsified (gate opposite predicted direction) | ✅ Unreplicated, mechanism inverted |

---

## METHODOLOGY CONTRIBUTIONS (Bugs caught & fixed)

1. **E12e: BatchNorm train/eval discrepancy** — δ_d calibration in `.eval()` mode gave 0.0148 (0.0% hinge); corrected `.train()` mode 3.6659 (247× difference, ~68× BN discrepancy) — origin of "always calibrate in train mode" policy
2. **E25 sign-convention bug** — Every diagnostic E2 onward used backwards sign convention for target direction; entire "gradient inversion" saga = measurement artifact, not real finding
3. **E62/E63/E64 shared-tensor bug** — pool1 reused same enc1 tensor for both input AND skip connection, making MaxPool3d effect unmeasurable; bug caught by user, effect relocated to skip by E64
4. **E80/E81 coordinate-frame confound** — E80's "independent absolute binding" retracted by E81's realignment check (93.5% recovery); network near-equivariant, not independently absolute-bound
5. **E82 Dice-insensitivity bug** — Only ~10/262,144 voxels perturbed; corrected to direct probability measurement at swap sites
6. **E83 physical-spacing confound** — Voxel-matched anisotropy didn't survive physical-distance-matched control; confound introduced by project's own non-isotropic preprocessing
7. **E89 proxy-artifact** — E85–E88's entire "necessity-allocation mismatch" premise caused by gradient-norm measurement failing to capture functional utilization; CORRECTION not reversion
8. **E104b auto-classifier bug** — Script checked raw correlation threshold BEFORE checking regression significance; manual re-analysis caught correct finding (partial ρ negative, p=0.0094) hidden by script's logic

---

## DATA QUALITY / REPRODUCIBILITY NOTES

- **Pooled vs. per-subject Dice**: pooled Dice (what training code reports) inflates every mechanism 0.85–2.40pp vs per-subject Dice (standard metric)
- **Baseline recalibration (E56)**: canonical "A = 0.9063" was seed 0 of 4-seed run; true mean = 0.9044 (CI [0.9007, 0.9080]); corrected +1pp target = 0.8942 (per-subject, not pooled)
- **Validation set reuse**: 125-scan set used for every model-selection and stopping decision; strictly not a held-out test set; fresh held-out or rigorous CV needed for final publication number
- **E48 data-provenance gap**: Fresh recompute (E85/E86) on same checkpoint/code/subjects differs up to 0.209 on individual subjects from stored E48 table; aggregate finding re-verifies robustly; root cause NOT identified
- **E70 title-vs-body gap**: Single-seed Dice looks positive (+0.30pp inside noise); 3-seed unreplicated; intended mechanism (gate diagnostic) **opposite predicted direction** (ρ=−0.388 unanimous)

---

## THE BIGGER PICTURE

**Total experiments**: 135+ phases
**Positive results**: 1 (Deep Supervision, +0.33pp, sub-threshold)
**Null results (extensively diagnosed)**: 45+
**Bugs found and fixed**: 8 major (plus dozens of minor)
**Pre-training kills (zero GPU cost)**: 12+
**Validated measurements (no Dice improvement)**: 4 (α_c, E48, E71-pred1, E89-utilization)
**Unexplained anomalies (proven but mechanism unknown)**: 7 (E48 size-dependence most prominent)

**Project narrative**: shifted from "stack plausible loss/architecture tricks" → "the real problem is preprocessing and information loss" → "small-lesion info is genuinely scarce and no single mechanism recovers it — the bottleneck and skip connection are causally load-bearing but in ways we haven't successfully exploited." The failure corpus is rich; the right principle remains elusive.

---

## E156–E164 — INVENTORY CLOSED, ASSUMPTION HUNT, COLLISION PROTOCOL (2026-09-15)

| Phase | Result |
|---|---|
| **E156** | All 5 positive results audited: E70 MM +1.75pp (3-seed) is **removal of a self-imposed FLAIR-only handicap**, not a discovery; E15 +3.93pp is an **oracle** (GT centroid, rank-1 decoder); E25 D4-only +0.33pp single-seed, 0/6 pairwise significant; **E45 seed-0 +1.44pp per-subject doubly significant but 3-seed FAILED** (0.8939 vs 0.8942, seed 1 p=0.588); E54 CI includes 0. Case B. |
| **E157** | Bottleneck self-knowledge: real (held-out ρ=+0.633, partial +0.791) but principle **PUBLISHED** — arXiv 2608.14894 "predictive self-knowledge", model-space, superset of our interventions |
| **E158** | Terminal state. **Two incompatible regimes** documented: Regime 1 (FLAIR-only, binary, 64³, baseline 0.8842) vs Regime 2 (4-mod, 3-region, native res). Pre-E70 pp-claims are all against the handicapped baseline |
| **E159** | Assumption enumeration. E (dense voxelwise prediction) 🔴 MaskFormer/Mask2Former/SAM |
| **E160** | **Assumption L: SURVIVES.** Native-resolution re-test, n=125: ρ_WT = **−0.5234, p=3.8e-10**, *stronger* than E48's −0.454 ⇒ the E48→E97 chain is **NOT a preprocessing artifact**, ~50 experiments admissible. But ET/TC show no size dependence ⇒ orthogonal to the tail |
| **E161** | **Assumption K (modality informativeness): 🔴 OCCUPIED on all 6 axes.** CoReFuse-Med (2609.10261) claims "modality-quality mismatch even when spatially aligned" as its own novelty |
| **E162** | Pivot premise ("our causal measurements are less crowded") **tested and FAILED**. Structural law: *on a well-studied task, a measurement's informativeness predicts its prior-art density* |
| **E163** | Vaswani Track 2.0 charter — the question survives, BraTS as its venue does not |
| **E164** | Collision-matrix protocol specified. Targets the **inertness claim**, not a module. P(novel ∧ ≥1pp) = 3–5%; P(contradiction-branch contribution) = 35–50% |

## E165–E168 — ARCHITECTURE-NOVELTY SEARCH CLOSED, WITH A MEASURED REASON (2026-09-15)

**The claim is not "we could not improve Dice."** It is that the residual was experimentally
decomposed into information-limited and boundary-localised error, and **both routes to the
≥1pp bar were independently closed by measurement**.

| Phase | Result |
|---|---|
| **E165** | Per-stage task-relevant rank, n=125: **RANK_LIMITED**. Monotone 35× gradient — enc1 R*/C **0.613** (19.6/32) → bottleneck **0.018** (4.5/256), Wilcoxon p=2.0e-22. **Refutes the earlier "low rank is universal" claim**; E147's ~1.9% is a property of *the bottleneck*, not the network. Independently **replicates E147** at the bottleneck (4.50 vs 4.95; 90% vs 84% at ≤4) |
| **E166** | Task-equivalent transformation ($D(G(Z))\approx D(F(Z))$ rather than $G(Z)\approx F(Z)$): **🔴 OCCUPIED, verdict #6** — DCP (NeurIPS 2018), BERT-of-Theseus (EMNLP 2020), *Local to Global* (ACL 2026). Also +0.00pp by construction |
| **Route (a)** | **CLOSED** — E142: zeroing t1c collapses ET 0.8433→0.0015 while WT holds. Information absent from the input |
| **E167** | Boundary-residual decomposition, n=110: **CLOSED**. ET **93.9%** of weighted error within ≤2 voxels, **98.6%** within ≤3, **interior 1.4%**. Geometry control (enrichment = err_frac / GT_frac) is load-bearing: TC/WT enrichment ≈ **2×** confirms real concentration, not geometry |
| **Ceiling** | Menze (TMI 2015) inter-rater ET **median 77%** (rater-vs-rater 74–85%); our model **0.900** |
| **Plateau** | *nnU-Net Revisited* (MICCAI 2024): "scores on BraTS21 are **saturated**"; winning ET 0.8245 (2018) → 0.8203 (2020) |
| **E168** | Cross-domain search, 10 fields (OT, operator splitting, multigrid, adjoint, RG, predictive coding, sheaf, influence functions, compressed sensing, Lyapunov): **no defensible candidate, verdict #7**. 7 died on prior art, 3 on arithmetic |

**Established**: residual → boundary-concentrated. **Not established**: residual → incorrect
labels (needs rater-level evidence). Closure stands on the *conjunction* of the three legs, not
on any one.

**Defensible wording**: *"Under the evaluated input, annotation, and architectural regime, the
study found no experimentally supported route to the pre-registered ≥1pp improvement."* Not
"no further improvement is possible."

---

### The arithmetic that reframes the whole tail branch

From `e160/E160_L_per_subject.json`: cohort mean **0.8595**; headroom is a **15-subject tail**
(ET 0.227 / TC 0.186 / **WT 0.824** — tumour found, failure is sub-partition). Lifting ET&TC on
those 15 to 0.40 gives **+1.94pp** — but **E140's per-subject oracle (+0.0154 ET) is worth only
+0.062pp overall.** E137/E140/E141 were therefore *arithmetically incapable* of clearing 1pp
regardless of mechanism. **Run this check before any future audit.**

Also settled: the 15 failures are **not small lesions** (ET sizes up to **39,207** voxels;
9/15 above 2000), and tiny-target Dice instability does not apply (only 3/125 have ET<500).
