# NeuroScan

3D U-Net brain-tumor segmentation research on BraTS 2023 GLI — a 200+ phase causal-intervention study hunting for a defensible ≥1.0 percentage-point Dice improvement over a fixed baseline.

**Status**: Mechanism search closed. See [`docs/NEUROSCAN_MASTER_EXPERIMENT_TABLE.md`](docs/NEUROSCAN_MASTER_EXPERIMENT_TABLE.md) for the full closure statement and every tested experiment.

## Start here

- **[docs/EXECUTIVE_SUMMARY.md](docs/EXECUTIVE_SUMMARY.md)** — one-page orientation: the problem, what worked, why the rest failed
- **[docs/READING_GUIDE.md](docs/READING_GUIDE.md)** — structured navigation by depth/topic/role
- **[docs/PROJECT_UNDERSTANDING.md](docs/PROJECT_UNDERSTANDING.md)** — comprehensive 8-section technical reference
- **[docs/NEUROSCAN_MASTER_EXPERIMENT_TABLE.md](docs/NEUROSCAN_MASTER_EXPERIMENT_TABLE.md)** — every experiment (E1–E203+), verdicts, and citations; the project's authoritative closing artifact
- **[docs/REPORT_RECOVERABILITY_FRONTIER.md](docs/REPORT_RECOVERABILITY_FRONTIER.md)** — the O_i recoverability measurement paper draft

## Layout

```
neuroscan_3d_v*.py       Architecture lineage (v1-v16). Root-level by convention — every
                          experiment script imports these by bare module name.
Dataset/                 BraTS 2023 GLI dataset + loaders (multimodal, cached, split logic)
experiments/              Per-phase training/analysis scripts (exp_e12_eggo_m/, exp_recoverability/, ...)
docs/                     All documentation, including docs/phases/ (every PHASE_*.md report)
configs/                  Training config templates
scripts/legacy/           Early pre-pipeline NeuroScan scripts, superseded but kept for reference
research_infra/           Project 2 (separate, active): OOD cross-shift generalization research
legacy/                   An earlier, unrelated, abandoned project (2.5D MS-lesion detector).
                          Not NeuroScan. Kept for history only — see legacy/README.md.
```

## Current baseline

Frozen checkpoint `E131_v5control_seed0` (`UNet3D_v5`), 4-modality (t1c/t1n/t2f/t2w) input, 3-region (ET/TC/WT) output, 128³ patch training at native resolution:

| region | Dice |
|---|---:|
| ET | 0.8191 |
| TC | 0.8481 |
| WT | 0.9119 |
| **mean** | **0.8597** |

Target: ≥0.8697 (+1.0pp). Not cleared by any tested mechanism — see the master experiment table for the full elimination tree.

## License

MIT — see [LICENSE](LICENSE).
