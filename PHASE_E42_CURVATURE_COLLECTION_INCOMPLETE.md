# Phase E42 (curvature sub-investigation) — Incomplete, No Findings

**Note**: written during a documentation audit to close a gap between raw disk artifacts and top-level reports. Unlike `PHASE_E34_ADAPTIVE_SIZE_REWEIGHTING_KILLED.md` (a completed experiment that was simply never written up), this sub-investigation was **not completed** — the run log (`experiments/exp_e12_eggo_m/e42/e42_run_log.txt`) stops mid-execution, and no conclusions were ever drawn from it. This is disclosed here for completeness, not resurrected or interpreted after the fact.

## What was attempted

A separate, earlier sub-investigation from `PHASE_E42_CROSS_SCALE_OPERATOR_AUDIT.md` (which is a self-contained structural kill decided before any compute was spent — see that document). This one, `run_e42_curvature_collection.py`, was an "Objective Geometry Audit": estimating Hessian-vector-product / Lanczos curvature spectra (largest/smallest eigenvalue, trace estimate) of the training loss, at 3 checkpoint epochs (1, 15, 30), for the 6 conditions from E39's lambda sweep, comparing the main loss (`L0`) against the D4 auxiliary loss alone (`L4`). Method verification (`verify_lanczos_and_hvp.py`) was run and passed before the real collection: Lanczos-recovered eigenvalues matched direct eigendecomposition on a synthetic matrix to 1e-2, and HVP matched finite-difference directional derivatives on the real model to 3.4e-5 relative error.

## What actually happened

The full protocol called for 6 conditions × 3 epochs × 2 loss terms × 2 probe repeats = 72 HVP-Lanczos runs, plus a separate D2-comparison block (3 more conditions reused from E25). **The run log stops after 57 of the planned 72 primary records**, mid-way through the `lambda_2.0` condition (`epoch=30 term=L0 probe=0` is the last completed line; `probe=1`, `term=L4` for that epoch, and the entire D2-comparison block never ran). No error or crash message appears in the log — the process appears to have simply stopped, most likely intentionally interrupted (this project's other long-running phases, e.g. E39's mechanism test, IECG's cost measurement, log explicit stop/kill reasons; this one does not, consistent with a manual interruption rather than a failure).

## Why this is reported rather than silently left alone

Per this project's own standing discipline ("keep every report self-contained", "report honest nulls and misses plainly"), an incomplete investigation with real compute spent (57 real HVP-Lanczos measurements, ~2,500 GPU-seconds) should be disclosed in the project's documentation trail rather than left as an orphaned JSON file with no record of what it was or why it stopped. **No scientific conclusion is drawn from the partial data** — 57 records with no D2-comparison block and missing probe/term combinations for the highest-lambda condition are not sufficient to support any curvature-vs-lambda claim, and none is made here.

## Disposition

Not resumed, not included in the project's kill-list or surviving-findings list. If a future phase needs Hessian-curvature evidence for the lambda-sweep conditions, this script and its verification harness are reusable, but the collection would need to be re-run to completion first.

## Artifacts on disk

- `experiments/exp_e12_eggo_m/e42/run_e42_curvature_collection.py`, `lanczos_hvp.py`, `verify_lanczos_and_hvp.py`
- `experiments/exp_e12_eggo_m/e42/E42_curvature_data.json` (57 partial records)
- `experiments/exp_e12_eggo_m/e42/e42_run_log.txt` (log ending mid-run, no error)
