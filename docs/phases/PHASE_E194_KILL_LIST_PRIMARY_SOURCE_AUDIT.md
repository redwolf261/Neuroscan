# E194 — Kill-list primary-source audit

**Date**: 2026-09-20
**Status**: AUDIT. No new compute beyond recomputation from stored artifacts.
**Purpose**: The E143/E144 paper intends to cite the kill-list as *evidence* that the model sits
at the information frontier. Before citing eleven experiments, verify each says what the memory
summaries claim. Memory has already been wrong twice this month (E15 non-reproduction; PediMS
"~9 patients" vs 2 files on disk).

---

## Method

For each cited kill: locate the primary artifact on disk, recompute the headline number, and
compare against the memory claim. Not "read the summary again" — recompute.

## Verified — memory is accurate

| Experiment | Memory claim | Recomputed from raw eval | Verdict |
|---|---|---|---|
| **E137** conditional readout | −0.89 pp | **−0.888 pp** (`E137_v16_seed0`, mean of 3 regions, n=125, vs `E131_v5control_seed0`) | ✅ exact |
| **E141** evidence-conditioned supervision | −0.37 pp | **−0.372 pp** (`E141_evidence_ep30`) | ✅ exact |
| **E142** t1c ablation | ET 0.8433 → 0.0015 | (prior session; artifact consistent) | ✅ |
| **E143** observer independence | ρ ∈ [0.786, 0.974] | **all six recomputed, exact to 3 dp** | ✅ exact |
| **E144** ΔR², Regime III empty | +0.5631, n=0/90 | **+0.5631 reproduced; n=0/90** | ✅ exact |

Baseline confirmed: `E131_v5control_seed0` = **0.8597** mean (ET 0.8191 / TC 0.8481 / WT 0.9119),
matching the E139/E165 figure of 0.8595–0.8597.

Additional datum recovered: `E141_shuffled_ep30` = **−0.634 pp**. The shuffled *control* scored
between the two real arms — which is itself part of why E141 was killed.

## Discrepancy 1 — E147 is mischaracterised in memory

| | |
|---|---|
| **Memory says** | "Representation Demand INVERSE — high recoverability needs LESS capacity; allocation hypothesis dead by 3 routes" |
| **Primary doc says** | *"**GREEN, provisional**... The fixed-contract hypothesis is **not killed**."* Later downgraded to **YELLOW** pending a checkpoint-invariance test — **not** killed. |

The ρ = −0.496 figure is real and correctly remembered (partial Spearman, image-only AP vs
`Rstar_self`, controlling baseline Dice, p = 8.74e-7). But the doc treats the *inverse sign* as
**supporting** a conditional-demand hypothesis, not as refuting allocation.

E147's own blocking limitation is more serious than the memory records:

> *"A repo-wide search for `self_ag`/`Rstar` across all `*.py` returns **zero hits**... **The
> generator was never committed and does not exist on disk.** So it remains UNKNOWN what
> `self_ag` actually measures."*

And: `R*` is a **6-valued ordinal** on a dyadic grid, not a continuous 1–32 range; 31/88
agreement curves are non-monotonic; checkpoint-invariance **never tested** (E129 showed
comparable quantities are run-dependent).

**Consequence**: E147 must **not** be cited as an allocation kill. It is an unresolved YELLOW
built on an unrecoverable generator. The claim "allocation is dead" rests on **E144's empty
Regime III**, which is solid and independently verified — not on E147.

## Discrepancy 2 — the E148/E150 readout gap is smaller than remembered

| | |
|---|---|
| **Memory says** | "linear probe beats trained head by **+0.0388 AP**" |
| **Recomputed from `E148_probes.json`** | **+0.0316** at dec1 (n=90; probe 0.9091 vs model 0.8775) |

And the direction reverses at other depths: **dec2 −0.0709**, **dec3 −0.0184**. The probe beats
the model at dec1 only. Memory reports a single positive figure with no mention that two of
three probe sites are *negative*.

The E150 conclusion (AP gains do not convert to Dice — "real but inert") is unaffected, and in
fact strengthened: the gap is smaller and site-specific.

## Consequence for the paper

**The core thesis is unaffected and is now better supported**, because it never depended on the
disputed items:

$$
\boxed{\text{Regime III empty } (n=0/90) \text{ — verified by recomputation — is the load-bearing fact.}}
$$

The kill-list is still usable as corroborating evidence, with three corrections:

1. **Cite E137 (−0.888 pp) and E141 (−0.372 pp) as measured**, with E141's shuffled control
   (−0.634 pp) included — it is informative, not noise.
2. **Do not cite E147 as an allocation kill.** Cite it, if at all, as an unresolved thread whose
   generator is lost. Its own doc says YELLOW.
3. **Correct the readout-gap figure to +0.0316 at dec1**, and state that dec2/dec3 are negative.

## Not yet audited (declared, not glossed)

- **E140, E178, E179, E188** — no primary phase doc read this session; E178/E179/E188 were run
  *in this session's lineage* and their docs exist (`PHASE_E178_*`, `PHASE_E179_*`,
  `PHASE_E188_*`), so they are higher-confidence than E140, which has neither doc nor artifact
  located.
- **~170 of 187 phase docs** remain unread at primary-source level. The E1–E130 material — in
  particular the E48→E97 bottleneck chain — is known from summaries only. E158 independently
  warns that a large part of that chain may have been diagnosing a preprocessing artifact
  created by the project's own 64³ resize (Regime 1: FLAIR-only, binary whole-tumour).

**Rule adopted**: no experiment gets cited in the paper unless its headline number has been
recomputed from a stored artifact, as E137/E141/E143/E144/E148 were here.
