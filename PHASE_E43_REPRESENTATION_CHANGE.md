# E43 — Representation Change Audit

**Verdict: KILL** for E36's specific boundary-localization claim, with a real, unambiguous positive finding kept and disclosed separately.

No model was trained. This document is self-contained.

---

## 1. What was measured

Four already-trained models exist for this project: an unmodified baseline, one trained with extra supervision only at the coarsest internal resolution ("D4-only"), one trained with extra supervision only at the medium resolution ("D2-only"), and one trained with both. For all 125 held-out validation scans, this phase ran ordinary forward passes through all four models (no training, no gradients) and extracted the internal decoder representation at each of the model's three internal resolutions, for each model, on the exact same scans.

For every scan, every resolution, and three separately-defined regions — the tumor's solid interior, its boundary-ambiguous cells (where the true tumor edge cuts through a coarse grid cell), and the background — three plain, unadorned quantities were computed, exactly as specified, before any derived or "clever" statistic was considered: the typical size of the representation vectors in that region, how much the representation actually changed relative to the baseline model in that same region, and whether that change pointed in a consistent direction (rather than just growing in magnitude without a coherent direction).

## 2. A real, large, and highly reproducible effect — confirmed first

Comparing the D4-only model against the D2-only model, at the two resolutions each was specifically trained to supervise: **each model changes its own supervised resolution's representation far more than the other model does.** At the coarsest resolution (the one D4 supervision targets), D4-only's representation change is nearly double D2-only's; at the medium resolution (the one D2 supervision targets), the pattern reverses, with D2-only changing more. Both differences are overwhelmingly statistically solid (each comparison uses all 125 scans, paired, and the resulting probability of this pattern occurring by chance is astronomically small).

![figure placeholder — see figures/e43_representation_change.png, left panel]

This confirms something genuinely useful and previously only inferred indirectly: coarse-resolution supervision does not just change the model's output at that resolution — it measurably, substantially, and specifically reshapes the model's own internal representation at that resolution, more than an alternative form of supervision at a different resolution does. This is a real finding, not an artifact of measurement noise, and it is kept and reported regardless of what follows.

## 3. The decisive question: is that change specifically at the coarse boundary?

This is the test that actually distinguishes between competing explanations. An earlier phase's own explanation for why coarse supervision helps rests on a specific geometric claim: that shrinking an image down turns a tumor's boundary into ambiguous, partially-covered grid cells, and that this is where the real difficulty (and therefore, plausibly, where a useful representation change) should concentrate.

The first, raw check of this gave a striking, statistically solid result — and it pointed the wrong way. The representation change at the tumor's solid interior was *larger* than at the boundary, not smaller, and this held up as a real, non-trivial pattern (only about a third of scans showed the boundary changing more than the interior, when a boundary-specific mechanism would predict the opposite most of the time).

## 4. Before accepting that at face value: a real confound was found and corrected

A surprising result pointing the wrong way from what a prior finding predicted is exactly the kind of thing this project's own accumulated practice says to check for a mundane explanation before accepting. That check found a real one: at the coarsest resolution, a "solid interior" region is often just a handful of coarse grid cells — for half the scans in this dataset, nine cells or fewer. Each of those cells corresponds to a large chunk of the original, full-resolution scan. For a small tumor, a coarse cell can be labeled "interior" by this measurement while still, in real physical terms, sitting right up against the tumor's true edge — the coarse grid is simply too blunt an instrument to correctly separate "deep interior" from "boundary-adjacent" for a small lesion.

This was checked directly, not assumed: scans with a smaller interior region showed a distinctly larger representation change there (a strong, statistically decisive relationship), and scans with a genuinely tiny interior region (three cells or fewer) showed nearly 40% more change than scans with a properly sized interior. This confirms the earlier "interior changes more" result was substantially driven by small lesions whose "interior" label was not measuring what it was supposed to measure.

![figure placeholder — see figures/e43_representation_change.png, middle panel]

## 5. The corrected comparison

Splitting scans into those with a genuinely large, properly interior region (nine or more coarse cells — large enough that the label means what it says) and those without, and re-running the interior-versus-boundary comparison separately in each group, gives a materially different and more trustworthy picture: **among scans with a real, deep interior region, there is no detectable difference between how much the interior changes and how much the boundary changes** — the two are statistically indistinguishable, splitting almost exactly down the middle (48% of scans show the boundary changing more, 52% show the interior changing more — no better than a coin flip). The earlier "interior changes more" signal is confirmed to have come entirely from the small-lesion subgroup, where the interior/boundary label itself was unreliable, not from a genuine tendency for interior cells to change more than boundary cells.

![figure placeholder — see figures/e43_representation_change.png, right panel]

## 6. What this does and does not establish

Put together, this phase establishes two separate things clearly, and they should not be blurred into one conclusion:

- **Coarse supervision demonstrably, specifically reshapes its own targeted resolution's internal representation**, more than an alternative form of supervision at a different resolution does. This is real and confirmed with strong statistical support.
- **That reshaping is not detectably concentrated at the geometrically ambiguous boundary cells specifically**, once a real measurement confound (small lesions making the interior/boundary label unreliable) is properly corrected for. Among tumors large enough for the label to be trustworthy, boundary and interior cells change by statistically indistinguishable amounts.

The second point is the one that matters for the specific mechanistic question this phase was designed to answer, and it does not support the idea that coarse supervision works *because* it specifically targets boundary-ambiguous geometry, in the way an earlier phase's explanation proposed. The representation change is real, large, and reproducible — it is just not spatially localized in the way that explanation would require.

## 7. Verdict

Per this project's own pre-declared rule for this phase: no localized, boundary-specific representation change was found, once a real confound in how "interior" was being measured was identified and corrected for. The honest, scientifically appropriate conclusion is exactly the one this phase's own instructions anticipated as an acceptable outcome: **coarse-resolution supervision demonstrably improves segmentation performance, and demonstrably, specifically reshapes its own targeted resolution's internal representation — but the precise geometric mechanism by which that reshaping translates into better segmentation cannot be identified from the representations available in these four trained models.** This is reported as the closing finding of this line of investigation, not as a prompt for a fifth representation-level phase; per the project's own stated intent, no further scalar, normalization, or loss is proposed here.

## 8. Limitations

- All four models were compared at a single checkpoint (their own best/final trained state) — this phase does not speak to whether the representation-change pattern found here was present earlier or later in training.
- The "deep interior" threshold used to correct for the small-lesion confound (nine or more coarse cells) was chosen as the natural midpoint of this dataset's own distribution, not tuned to produce a particular result — but a different threshold choice was not separately tested, and the corrected finding should be read as robust to roughly this choice, not to an arbitrarily different one.
- This phase measured the *decoder* representation only (the part of the network responsible for building the final prediction from compressed features); it does not speak to whether a similar or different pattern would be found earlier in the network, in the shared encoder.
