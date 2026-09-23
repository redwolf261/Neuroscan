# NeuroScan Reading Guide

**Purpose**: Navigate this project's 188+ phases and 27k+ lines of documentation in a structured way.

---

## 5-Minute Orientation

1. **[EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)** — This document. One page, all essential facts.
2. **[PROJECT_UNDERSTANDING.md](PROJECT_UNDERSTANDING.md)** — Comprehensive 8-section reference (goal, task, all major arcs E1–E188, findings, architecture, code, current state, lessons).

---

## By Role / Question

### "I want to understand the whole project end-to-end"
**Time**: 2–3 hours. Read in order:
1. `EXECUTIVE_SUMMARY.md` (5 min)
2. `docs/phases/PHASE_RESEARCH_ARC_MASTER_REPORT.md` (E1–E43, 15 sections, self-contained, ~30 min)
3. `docs/phases/PHASE_POST_E43_SUMMARY.md` (E44–E53, 3 sections, ~20 min)
4. `docs/RESEARCH_KNOWLEDGE_MAP.md` (E1–E107+ audit table, dense, ~1 hour skimming)
5. `docs/phases/PHASE_E188_E15_BRANCH_FROZEN.md` (E188 latest, Sept 20, ~30 min)

**Outcome**: You will understand the full hypothesis chain, all 7 mechanism families, all major closed branches, and current frontier.

### "I want to know what worked and why it didn't"
**Time**: 30 min. Read:
1. `EXECUTIVE_SUMMARY.md` → "What We Know Works" section
2. `EXECUTIVE_SUMMARY.md` → "The Failure Structure" table
3. `docs/phases/PHASE_RESEARCH_ARC_MASTER_REPORT.md` § 7 (Deep Supervision mechanism audit, E36)
4. `docs/phases/PHASE_E25_C62_CORRECTED_REANALYSIS.md` (if curious about the sign-bug correction)

**Outcome**: You understand that D4 (+0.33pp) is the only validated improvement, why E49–E51 failed (multi-seed variance), and what the project learned from each dead-end.

### "I want to focus on the latest findings (E169–E188)"
**Time**: 1–2 hours. Read:
1. Memory index: `/MEMORY.md` (top section)
2. `docs/phases/PHASE_E188_E15_BRANCH_FROZEN.md` (readout decomposition proof, Gate F)
3. `docs/phases/PHASE_E188A_LOCAL_RECOVERABILITY_MAP_PREREG.md` (Experiment A)
4. `docs/phases/PHASE_E188F_READOUT_DECOMPOSITION_PREREG.md` (Experiment F)
5. `docs/phases/PHASE_E169_RSTAR_PREDICTABILITY_PREREG.md` (if interested in the search-termination decision)

**Outcome**: You understand why the latent-steering branch is frozen, what Gate F enables, and why the project stopped searching.

### "I want to understand why specific mechanisms failed"
**See the table below.**

### "I want to replicate or extend an experiment"
**See the section below: "Code & Running Experiments."**

### "I want to understand the methodological contributions"
**Time**: 1 hour. Read:
1. `EXECUTIVE_SUMMARY.md` → "Three Methodological Contributions"
2. `docs/phases/PHASE_E56_MEASUREMENT_AUDIT.md` (pooled-vs-per-subject Dice gap discovery)
3. `docs/phases/PHASE_POST_E43_SUMMARY.md` § Part 2 (multi-seed variance, E49 headline overestimate)
4. `docs/phases/PHASE_E48_NECESSITY_ALLOCATION_MISMATCH_REFRAME.md` (causal-finding reframe)
5. `docs/phases/PHASE_E25_C62_SIGN_CONVENTION_CORRECTION.md` (bug-fixing methodology)

**Outcome**: You understand the three defensible contributions (methodology, not mechanics).

---

## Deep Dives by Topic

### Deep Supervision (D4, +0.33pp) — The Only Positive Find
**Documents** (read in order):
1. `docs/phases/PHASE_RESEARCH_ARC_MASTER_REPORT.md` § 7 (initial design & mechanism hypothesis)
2. `docs/phases/PHASE_E25_STRUCTURAL_PIVOT_1A_DETECTION_HYPOTHESIS.md` (original hypothesis: detection improves)
3. `docs/phases/PHASE_E36_DEEP_SUPERVISION_AUTOPSY.md` (mechanism kill: detection flat, quality improves)
4. `docs/phases/PHASE_E37_GRID_ENTROPY_FEASIBILITY.md` (attempted explanation: grid-entropy killed)
5. `docs/phases/PHASE_E38_ORTHOGONAL_RESIDUAL_PROBE.md` (attempted Q_r signal, survives one test)
6. `docs/phases/PHASE_E39_MECHANISM_TEST.md` (Q_r fails prediction: D4 has LOWEST residual at best Dice)
7. `docs/phases/PHASE_E40_SUBSPACE_AUDIT.md` (effective-rank hypothesis killed)
8. `docs/phases/PHASE_E41_TRANSFORMATION_ERROR_AUDIT.md` (residual-error correlation null)

**Outcome**: D4 is real, mechanism is "improves small-tumor segmentation quality" (1–50 voxel +40% Dice), but no deeper lever exploited it.

### Latent Steering (E15 → E188) — Proven Dead
**Documents** (read in order):
1. `docs/phases/PHASE_E15_EVIDENCE_FREEZE.md` (original E15: +0.0393 on retired checkpoint)
2. `docs/phases/PHASE_E172_LABEL_FREE_DIRECTION_PREREG.md` (label-free selection null; orphaned smoke_oracle_e15.py result: +0.0139 on current, non-monotonic)
3. `docs/phases/PHASE_E184_REPAIR_OPERATOR_DERIVATION.md` (E184 geometry audit: E15's effect NOT explained by geometry)
4. `docs/phases/PHASE_E188A_LOCAL_RECOVERABILITY_MAP_PREREG.md` (Exp A: 32 random directions never beat native, E15 at 100th percentile, no recoverable neighborhood)
5. `docs/phases/PHASE_E188F_READOUT_DECOMPOSITION_PREREG.md` (Exp F: readout decomposition proof, P_W accounts for full effect, ker(W) = 5.2e−5)
6. `docs/phases/PHASE_E188_E15_BRANCH_FROZEN.md` (synthesis, Gate F, record correction)

**Outcome**: E15 is readout steering in disguise; 34–55% of its energy does nothing. Gate F established as cheap architectural filter for future candidates.

### Margin Loss Family (EGGO-M, SC-TAM) — The Longest Diagnostic Chain
**Phase sequence** (14+ phases, pick depth):
- **Quick (30 min)**: `PHASE_RESEARCH_ARC_MASTER_REPORT.md` § 5–6 (EGGO-M null, SC-TAM null, sign-bug fix)
- **Complete (2 hours)**: All E12–E25 phase files in `/docs/phases/`, specifically:
  - E12a–E12f (EGGO-M pilot, calibration bugs, final null)
  - E13 (multiseed confirmation)
  - E14–E21 (diagnostics: gradient conflict, decoder sensitivity, reachability, transport, audit)
  - E22 (direction test: wrong-signed)
  - E23–E24 (SC-TAM redesign, spec)
  - E25-Steps-1–16 (sign-bug correction chain, worst→best confidence gate pass)

**Outcome**: Mechanism was real, mathematically correct, and entirely null on Dice. The E25 sign-bug fix is the project's best example of catching mistakes inline.

### Post-Pivot Architecture Search (E44–E51) — All Trailed D4
**Phase sequence**:
1. `docs/phases/PHASE_POST_E43_SUMMARY.md` § Part 1–2 (complete narrative, E44–E51 all in ~30 min)
2. Individual phase files if interested in specific mechanisms:
   - E44: `PHASE_E44_RCGW_KILLED.md` (adaptive weighting, killed twice)
   - E45: `PHASE_E45_D4_D8_BELOW_TARGET.md` (single seed +0.47pp → 3-seed mean +0.31pp)
   - E46: `PHASE_E46_ATTENTION_GATE_BELOW_TARGET.md` (single seed +0.39pp, gate learned pattern)
   - E49–E51: (CCABA, IECG, CCAG in POST_E43_SUMMARY or individual files)

**Outcome**: Multi-seed variance killed every single-seed headline. CCABA/IECG/CCAG all 3-seed-checked; none beat D4-only (p>0.05).

### Bottleneck Size-Dependency (E48) & Mechanism Attempts (E49–E60)
**Core finding + exploitation attempts**:
1. `docs/phases/PHASE_E48_NECESSITY_ALLOCATION_MISMATCH_REFRAME.md` (bottleneck size ρ=−0.454, discovery)
2. `docs/phases/PHASE_E85_E86_REVERIFICATION_AND_COMPENSATING_CIRCUIT.md` (re-verify: ρ=−0.3834, same direction)
3. `docs/phases/PHASE_POST_E43_SUMMARY.md` § Part 2 (E49 CCABA + E50 IECG attempts, both failed)
4. `docs/phases/PHASE_E58_BOTTLENECK_CHARACTERIZATION.md` (spatial non-localization; donor effect OOD confound)
5. `docs/phases/PHASE_E59_COALITIONAL_INTERACTION_AUDIT.md` (small-lesion synergy ρ=−0.498 with size)
6. `docs/phases/PHASE_E60_DBC_FEASIBILITY.md` (small-lesion synergy doesn't reproduce; killed)

**Outcome**: E48's finding is real and durable, but no mechanism to exploit it cleared the bar.

### Preprocessing & Resolution (E29–E35) — Truth & Traps
**Sequence**:
1. `docs/phases/PHASE_E29_RESOLUTION_SURVIVAL_AUDIT.md` (geometric proof: median tumor vanishes at 64³; recovery monotonic 5–150 voxel band)
2. `docs/phases/PHASE_E30_DTC_FEASIBILITY.md` (degradation-trajectory candidate, Gate A passes, B fails population test)
3. `docs/phases/PHASE_E31_SMALL_LESION_DTC.md` (DTC reformulation killed by division-by-vanished-footprint artifact)
4. `docs/phases/PHASE_E32_CRITICAL_RESOLUTION.md` (α_c definition, survives 500 permutations, modest but real)
5. `docs/phases/PHASE_E33_NOVELTY_AUDIT.md` (α_c literature check: clean novelty)
6. `docs/phases/PHASE_E34_ADAPTIVE_SIZE_REWEIGHTING_KILLED.md` (gradient-blowup collapse, calibration-bug origin)

**Outcome**: Resolution problem is real (geometric, pre-model). α_c is validated measurement, not yet trained. DTC was artifact. E34 origins the gradient-calibration safeguard.

### Representation Space Search Termination (E169–E176)
**Read for the decision**:
1. `docs/phases/PHASE_E169_RSTAR_PREDICTABILITY_PREREG.md` (stage-specific rank, durable finding R²=0.518 enc3)
2. `docs/phases/PHASE_E174_RANK_UNIFICATION.md` (enc3 R²=0.518 vs dec1 R²=0.004, size-controlled)
3. `docs/phases/PHASE_E176_METRIC_AXIS_AUDIT.md` (8 novelty axes audit table + search termination decision)

**Outcome**: All 8 axes occupied by 2025–2026 literature. No differentiated principle found. Search closed.

### Sign-Convention Bug Correction (E25 Step 5)
**Read for methodology**:
1. `docs/phases/PHASE_E25_C62_MECHANISM_AUDIT.md` (FN moves wrong-signed, hypothesis: inversion)
2. `docs/phases/PHASE_E25_C62_SIGN_CONVENTION_CORRECTION.md` (THE FIX: diagnostic scripts used backwards convention)
3. `docs/phases/PHASE_E25_C62_GRADIENT_JACOBIAN_CONSISTENCY.md` (validation: Δz·∇_zL consistency check, 100% pass)

**Outcome**: Bug caught and fixed inline. Entire narrative reversed (backwards→correct). Original code was fine; diagnostics were inverted.

---

## Code & Running Experiments

### To Understand the Codebase
1. **Architecture files** (git-tracked): `/neuroscan_3d_v*.py` (v1–v9 lineage)
   - See `v2` (baseline), `v3` (+ D4), `v5` (+ attention gate), `v6` (+ CCABA)

2. **Main experiment hub**: `/experiments/exp_e12_eggo_m/`
   - Subdirs: `e100`, `e102`, ..., `e198` (one per phase)
   - Structure: `run_*.py` (launch script) + cached `*.json` results (no re-training needed)

3. **Frozen checkpoint** (used for later audits): `E131_v5control_seed0` (UNet3D_v5)
   - Referenced in E188, E192–E198 cross-model audits

4. **Training utilities**: `/experiments/exp_e12_eggo_m/` root level has helpers for calibration, gradient analysis, metric computation

### To Replicate an Experiment
- Most results cached as JSON (e.g., `E192_Z_cache.json`, `E193_recoverability_TC.json`)
- No re-training needed for audits or re-analysis
- For full re-train: See individual phase's `run_*.py`; config templates in `/configs/`

### To Extend the Project
1. **Prerequisite**: Read `PHASE_E188_E15_BRANCH_FROZEN.md` (Gate F) — understand why prior attempts failed
2. **Next gate**: Prior-art audit on "upstream → nonlinear decoder → altered reachable output (not reducible to readout steering)"
3. **If audit clear**: Gate F is the mandatory architectural check (cost ~280 forward passes, no training)

---

## Finding Specific Information

### "Where is [phase name] documented?"
- `/docs/phases/PHASE_*.md` — grep the phase number or name
- Or check `/MEMORY.md` for linked memory files

### "What's the canonical baseline number?"
- **Pooled Dice**: 0.9063 (E1–E43 history)
- **Per-subject Dice**: 0.8872 (corrected E56)
- **Corrected +1pp target**: 0.8942 per-subject

### "Which mechanism is the best?"
- **Deep Supervision (D4)**: +0.33pp, validated, mechanism closed
- **All others**: Below D4 or tied (E45–E51, E52–E53)

### "What does Gate F mean?"
- See `PHASE_E188_E15_BRANCH_FROZEN.md` § "The Durable Output: Gate F"
- Required check: ΔDice(δ_l) ≉ ΔDice(P_W Δz) — if benefit fully reproduced by readout steering, upstream did no work

### "What were the orphaned JSON files?"
- `E151_dec3_sweep.json`, `E152_resolution_recovery.json`, `E153_responsibility.json`
- **Do not cite** — no generating script, no phase doc, definitions unrecoverable

### "What's the tightest multi-seed result?"
- E51 (CCAG): mean 0.9095, std ≈0.05pp (tightest of any 3-seed condition)
- But tied D4-only (p=0.919), so not a real win

---

## Timeline of Discovery

| Dates | Phases | Key Event |
|-------|--------|-----------|
| Aug–Nov 2024 | E1–E43 | EGGO-M null + D4 discovery |
| Nov 2024–Jan 2025 | E44–E53 | Post-pivot architecture search; multi-seed variance problem discovered |
| Jan–Feb 2025 | E54–E97 | Bottleneck diagnostics; capacity/rank arc |
| Feb–Mar 2025 | E98–E129 | Capacity/effective-rank branch (no methods clear bar) |
| Mar–Apr 2025 | E138–E153 | Readout-frontier; contrast-intervention family |
| Apr–Sep 2025 | E169–E188 | Prior-art search; latent-steering closure; Gate F established |

---

## Recommended Reading Order by Depth

### Depth 1: "Just the headlines" (15 min)
1. `EXECUTIVE_SUMMARY.md`

### Depth 2: "Understand the arc & outcome" (1–2 hours)
1. `EXECUTIVE_SUMMARY.md`
2. `docs/phases/PHASE_RESEARCH_ARC_MASTER_REPORT.md`
3. `docs/phases/PHASE_POST_E43_SUMMARY.md`

### Depth 3: "Full story with detail" (3–4 hours)
1. All of Depth 2
2. `docs/RESEARCH_KNOWLEDGE_MAP.md` (audit table, E1–E107+)
3. `docs/phases/PHASE_E188_E15_BRANCH_FROZEN.md`
4. One deep-dive by topic (e.g., D4, E15, E48)

### Depth 4: "Expert-level mastery" (8+ hours)
1. All of Depth 3
2. All individual phase files for your topic of interest
3. Code inspection: `/neuroscan_3d_v*.py` + experiment scripts
4. Memory index: `/MEMORY.md` + linked memory files

---

## File Sizes (for planning your read)

| Document | Lines | Time | Note |
|----------|-------|------|------|
| EXECUTIVE_SUMMARY.md | 250 | 10 min | Start here |
| PROJECT_UNDERSTANDING.md | 500 | 30 min | Comprehensive reference |
| PHASE_RESEARCH_ARC_MASTER_REPORT.md | 290 | 30 min | E1–E43 narrative |
| PHASE_POST_E43_SUMMARY.md | 135 | 20 min | E44–E53 narrative |
| RESEARCH_KNOWLEDGE_MAP.md | 37k tokens | 1–2 hours | E1–E107+ audit table (dense) |
| PHASE_E188_E15_BRANCH_FROZEN.md | 210 | 30 min | Latest (Sept 20), readout proof |
| Average PHASE_*.md | 300–500 lines | 30–45 min | Individual mechanism dives |

---

**Last Updated**: 2026-09-20 (by reading all major docs, not memory alone)
