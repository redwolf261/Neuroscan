# Optimization Controller — New Work (separate from the frozen baseline)

**Baseline status**: `01_source_code/models/final_model.py` and `01_source_code/diagnostics/` are **frozen** as of git tag `baseline-frozen` (commit `000c461`). They should not be edited further except to fix an actual bug in reproducing the frozen results. Every measured number in `research_infra/PHASE_7_HYPOTHESIS_VALIDATION.md`, `PHASE_8_PROJECT_SELECTION.md`, and `NEUROSCAN_MATHEMATICAL_SPECIFICATION.md` refers to that exact frozen state.

**Rule for everything in this directory**: code here **imports** the frozen baseline (`sys.path` + `import final_model as fm`, the same pattern already used in `research_infra/phase8/common.py`) and builds new functionality *alongside* it — new modules, new training loops, new optimizer/controller logic. It never edits `final_model.py` or `01_source_code/diagnostics/`.

If a genuine baseline bug is found later that must be fixed to keep reproducing results, that's a deliberate exception: fix it in `final_model.py`, note it clearly in a commit message, and re-tag (e.g. `baseline-frozen-v2`) rather than silently drifting the reference point.

## What goes here

- The Optimization Controller implementation (Sections 3.1–3.6 of `../OPTIMIZATION_CONTROLLER_DESIGN.md`), starting with the measurement-only pass recommended in that document's Section 6 before any gradient modification is attempted.
- New training-loop variants that call the controller.
- Ablations over $\gamma$, $\tau$, and the other controller hyperparameters.
- The direct static-vs-adaptive comparison against the Phase 8 gradient-scaling result, under matched conditions.

## Reference points

- Frozen baseline: git tag `baseline-frozen`
- Mathematical spec of what's frozen: `../NEUROSCAN_MATHEMATICAL_SPECIFICATION.md`
- Controller design this implements: `../OPTIMIZATION_CONTROLLER_DESIGN.md`
