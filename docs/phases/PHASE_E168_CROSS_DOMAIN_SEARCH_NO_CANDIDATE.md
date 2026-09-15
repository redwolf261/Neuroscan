# E168 — Cross-domain principle search: NO DEFENSIBLE CANDIDATE (verdict #7)

**Date**: 2026-09-15
**Status**: Complete. No compute. Closes the "borrow a principle from another field" strategy.

---

## The strategy under test

After six OCCUPIED verdicts inside segmentation architecture design, the proposal was to change
the *search domain* rather than the search method:

$$\text{borrow a principle from another field} \rightarrow \text{derive its segmentation form} \rightarrow \text{test it}$$

with one added constraint learned from E166: the new computation must be able to **change the
optimum**, not merely approximate the existing network.

Ten fields were searched with the full kill-list attached, the hardware limits stated, and each
candidate required to carry its own hostile prior-art audit *plus* a specific argument for route
(b) (+2pp on already-good subjects).

## Result

$$\boxed{\textbf{No defensible novel candidate under these constraints.}}$$

**Seven fields died on prior art. Three died on arithmetic** — which is the worse outcome, since
arithmetic failures cannot be fixed by looking harder.

| Field | Verdict | Why |
|---|---|---|
| **Optimal transport / Wasserstein** | 🔴 DIRECT, on our dataset | **Generalized Wasserstein Dice Loss** is BraTS-native and already encodes the ET⊆TC⊆WT nesting as a ground metric: *"allows taking advantage of the hierarchical structure of the tumor regions labeled in BraTS."* Official implementation exists |
| **Operator splitting / PDE** | 🔴 DIRECT, twice | Liu et al. derive segmentation nets as *"two operator-splitting algorithms solving the Potts model"*; Potts relaxation is what CRFs and graph cuts optimise — both already killed |
| **Multigrid** | 🔴 | MgNet unifies multigrid and CNNs; and multigrid is a *convergence accelerator* — it changes how fast you reach a solution, not which solution is optimal. Fails the E166 constraint |
| **Adjoint / Pontryagin / optimal control** | 🔴 dead by identity | Backpropagation **is** the adjoint-state method. We already run it |
| **Renormalization group** | 🔴 not instantiable | *"there lacks a precise notion of what defines 'scale' in data or models."* Every concrete instantiation is an Ising/RBM toy; the voxel version reduces to downsampling = multi-scale fusion, killed |
| **Predictive coding / free energy** | 🔴 | Iterative inference-time error minimisation = test-time adaptation (killed); hierarchical error passing = deep supervision + top-down gates (killed) |
| **Sheaf theory** | 🟡 **apparently open**, fails arithmetic | SheafStain is 2D virtual staining; 3D cellular-sheaf gluing with a cocycle term on triple overlaps is genuinely unclaimed. **But** sliding-window inference already Gaussian-weights overlaps to suppress seams, so for +2pp on subjects at 0.90–0.94, seam disagreement would need to be ~a fifth of their entire residual. Expected +0.1–0.3pp |
| **Influence functions** | 🔴 | Reduces to difficulty reweighting (killed), and its premise — the 15 catastrophic subjects are fixable — is falsified by E142 (t1c zeroing: ET 0.8433→0.0015) |
| **Compressed sensing / RIP, spectral geometry, DDG** | 🔴 no bridge | RIP is a guarantee about a sensing matrix, not an objective; and our own E147/E165 show the bottleneck representation is *already* extremely low-rank — there is no undersampling problem. DDG's segmentation form is Discrete Morse topology loss, killed |
| **Lyapunov / ergodicity** | 🔴 | Ergodic averaging = Stochastic Weight Averaging (2018), worth ~+0.2–0.5pp; Lyapunov stability requires a neural-ODE formulation, killed |

---

## The structural reason, stated by the search itself

Route (b) demands +2pp on subjects at ET 0.900 / TC 0.938 / WT 0.924. At that level the residual
is boundary-localised (**E167: 98.6% of ET error within 3 voxels; interior 1.4%**), and the
published ET inter-rater median is **0.77**. So:

> Every candidate novel enough to be publishable is too weak to deliver 2pp, and every mechanism
> strong enough to deliver 2pp is already occupied — because the occupied ones are occupied
> *precisely because they were the reachable gains*, and they have been taken.

This is the same structural law E162 derived from the inside (*informativeness predicts prior-art
density*), reached independently from the outside. Two derivations, one phenomenon.

## Consequence

Changing the *search domain* does not escape the constraint, because the constraint is not about
where ideas come from — it is about **what is left to win on this cohort**. That is settled by
E167 and the route audits, not by the breadth of the literature search.

**Do not run an eighth search.** The E164 stop rule (3 candidates, 2 rounds, 4 sessions) was
exceeded; seven consecutive verdicts with a derived structural explanation is the result.

## Sources

Generalized Wasserstein Dice Loss (arXiv 2112.13054, 2011.01614) · Operator-splitting and DNNs
(2307.09052) · MgNet · Optimal Control Approach to Deep Learning (1803.01299) · RG for DNNs
(2510.25553) · Hybrid predictive coding · SheafStain (2606.11846) · Topology Graph Consistency
(2509.22689) · Discrete Morse topology loss (2103.09992) · Stochastic Weight Averaging
(1803.05407)
