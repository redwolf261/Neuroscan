# NeuroScan / EGGO-M — Current Project Docs

This `docs/` folder holds the documentation for the **current, active** research line (BraTS 2023 GLI, FLAIR-only, 3D U-Net family, phases E1–E58+). It is separate from the top-level `README.md` and the `0X_*` folders at the project root, which describe an **earlier, different project iteration** (a 2.5D HybridMiniSwin MS-lesion webapp) and are left untouched.

## Layout

- **`phases/`** — every `PHASE_*.md` report from the E1–E58 arc (design docs, results, audits, kill reports). `PHASE_POST_E43_SUMMARY.md` is the best single entry point for the post-pivot arc (E44 onward); `PHASE_RESEARCH_ARC_MASTER_REPORT.md` covers the earlier E1–E27 arc narratively.
- **`setup/`** — environment/dataset setup and status docs (GPU setup, BraTS dataset info, training pipeline map, session/status summaries) from earlier in the project.

## Related, not moved

- `neuroscan_3d_v1..v9.py` (architecture lineage) — stay at the project root; every experiment script imports them by bare module name assuming that location.
- `experiments/` — per-phase training/analysis scripts and disk-only results (gitignored by convention; only the `PHASE_*.md` reports and the versioned architecture files are git-tracked).
- `configs/`, `Dataset/` — training config and dataset, unchanged.

## Project memory

The authoritative running index of findings (with links into the phase docs above) lives in Claude's own project memory (`MEMORY.md` and linked files), not in this repo — see that index for the canonical, up-to-date summary of what's been established.
