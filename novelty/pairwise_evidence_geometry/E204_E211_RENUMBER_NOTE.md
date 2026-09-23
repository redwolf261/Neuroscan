# Renumbering note: E190-E197 -> E204-E211

The scripts originally written this session as E190-E197 (Hamiltonian
phase-space correction, Evidence Closure, and the HRS chain: E194 forensic
falsification, E195 crossing test, E196 endogenous bridge, E197 HRS-Lite)
collided with pre-existing phase docs `docs/phases/PHASE_E190_*` through
`PHASE_E200_*`, which document unrelated earlier experiments.

Renamed before pushing to avoid an ambiguous E-number sequence in project
history. Mapping (old -> new), files and all internal references (docstrings,
CSV read/write paths) updated accordingly. `.log` files were left unrenamed
since they are historical run output, not source, and rewriting them would
misrepresent what was actually invoked at the time.

| Old | New | Content |
|---|---|---|
| E190 | E204 | Gradient-validity gate for the SPA candidate (spatial influence map, E3 vs bottleneck) |
| E191 | E205 | Hamiltonian kinetic-energy existence gate, `K=\|\|grad h\|\|^2/(h^2+eps)` |
| E192 | E206 | Inertness check for the Hamiltonian correction `h'=h(1+alpha*K)` -- KILLED |
| E193 | E207 | "Evidence Closure" individual/collective/relational bottleneck-evidence gate -- KILLED |
| E194 | E208 | HRS forensic falsification (donor-texture splicing, Delta_E) -- PASSED |
| E195 | E209 | Delta_E -> segmentation-crossing predictive test -- PASSED |
| E196 | E210 | Endogenous representation-space hypothesis bridge (prototype vs random) -- PASSED |
| E197 | E211 | HRS-Lite: learned proposal + evaluator, 5-fold subject-grouped CV -- KILLED (proposal didn't beat random; zero-crossing anomaly needs root-causing before retry) |

Directories: E204-E206 live in `novelty/sec_pool2/` (continuing the
E185-E189 skip/pooling branch's gradient/Hamiltonian thread); E207-E211 live
in `novelty/pairwise_evidence_geometry/` (the HRS chain).
