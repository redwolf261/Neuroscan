"""
Phase E63: Local Competition Geometry Audit.

NO TRAINING. NO ARCHITECTURE CHANGE. NO LOSS CHANGE. Pure causal-
intervention diagnostic, same discipline as E47/E48/E58/E62.

CONTEXT: E62 established that permuting the spatial arrangement of
activations inside pool1's 2x2x2 cells (holding the exact 8-value
multiset fixed) causally degrades segmentation, and the effect is
small-lesion-specific (rho=-0.355, p<0.001). E63 asks a narrower
question: is E62's effect driven primarily by the geometry of the TOP
TWO competing activations (winner vs. runner-up) inside each cell,
rather than the full 8-value arrangement?

Do not assume this is true -- this phase is a diagnosis, not a design
exercise. No pooling/downsampling algorithm is implemented here
regardless of outcome.

CHECKPOINT / DATA: identical to E62 -- v5/E46 attention-gate checkpoint,
same 125-subject validation split, same preprocessing, same enc1 hook
point (immediately before pool1, tensor shape (B,32,64,64,64)).

LOCAL COMPETITION DESCRIPTOR (per cell, per channel):
  8 values x1..x8 (the 2x2x2 cell). Sorted descending: x(1)>=...>=x(8).
  w = x(1) (winner), r = x(2) (runner-up), g = w - r (competition gap).
  q1, q2 = spatial coords (in {0,1}^3) of the winner and runner-up.
  dq = q2 - q1 (signed displacement), d = ||dq||_2.

CELL SAMPLING / INTERVENTION SCALE (pre-declared, before any effect is
examined; REVISED after a first run exposed a scale confound -- see
below):

  A large pooled sample of candidate cells is drawn first (across a
  30-subject subset x all 32 channels x all 32768 pool1 cells) to
  compute FIXED GLOBAL gap-quartile cutoffs (Q1..Q4 boundaries on g),
  decided before per-subject sampling and frozen. Real post-ReLU pool1
  activations are dominated (~84% empirically checked, disclosed not
  hidden) by exact gap==0 ties -- winner and runner-up both zero, i.e.
  cells with no real competition at all. These dead/tied cells are
  excluded from the candidate population BEFORE computing quartile
  cutoffs (otherwise Q1-Q3 all collapse into the same degenerate
  zero-gap bin). Quartiles are over the ACTIVE (gap>0) population only.

  SCALE CONFOUND FOUND AND CORRECTED (disclosed, not hidden): the first
  E63 run sampled a small FIXED count of cells per subject (200, 50 per
  quartile, ~0.02% of the ~1.05M total pool1 cells) and intervened on
  only those. Even Test A (the full local derangement, identical
  construction to E62's own intervention) produced a near-zero Dice
  effect at that scale (mean drop=-0.00002, indistinguishable from
  noise) -- vs. E62's own robust 0.0226 drop when EVERY cell in the
  volume was permuted simultaneously. A 200-cell intervention could not
  separate "winner-runner geometry doesn't matter" from "perturbing
  0.02% of cells is too small a perturbation to move whole-volume Dice
  at all, regardless of mechanism" -- so the first run's result was
  reported to the user as confounded, not accepted as a clean finding.

  CORRECTED DESIGN (this version): every counterfactual test (A/B/C) is
  applied to ALL active (gap>0) cells in the subject's volume
  simultaneously -- matching E62's own whole-volume intervention scale,
  so a genuine "no effect" result cannot be confused with "too small an
  intervention to register." The gap-quartile breakdown (section 8) is
  computed by restricting Test B to ONE quartile stratum's cells at a
  time, but STILL applying it to every active cell within that quartile
  across the whole volume (not a small sample of it) -- this isolates
  the gap-quartile variable while keeping each stratum's own
  intervention strong enough to produce a measurable signal in the
  first place. Cells are NOT selected based on any Dice effect at any
  point.

THREE NESTED COUNTERFACTUAL TESTS (per sampled cell, multiset always
exactly preserved, only positions of specific values change):
  Test A (E62-local reference): full 8-value derangement of the cell
    (same construction as E62, restricted to just the sampled cells).
  Test B (winner fixed, runner-up moved): winner stays at q1; a single
    pairwise swap relocates the runner-up's value to one other
    randomly-chosen (seeded) non-winner, non-runner-up slot -- the
    value previously in that target slot moves into the runner-up's old
    slot. Exactly 2 of the 8 slots change; winner's value and slot are
    untouched; all 8 values preserved.
  Test C (top-two fixed, weak values rearranged): winner and runner-up
    stay in q1, q2 exactly; a derangement is applied to the remaining
    6 slots' values (x(3)..x(8)) only.

All three tests are evaluated by editing ONLY the sampled cells within
one subject's enc1 tensor (all other pool1 cells left exactly as the
real intact enc1), batched into a single modified enc1 -> one forward
pass per test per subject (cells are spatially disjoint 2x2x2 blocks so
this is a valid batched realization of per-cell independent
interventions).

PRE-DECLARED DECISION RULE (verbatim from spec):
  GO only if ALL:
    1. Test B produces a reproducible non-zero segmentation effect.
    2. Test B captures a substantial fraction of Test A's effect
       (F_WR = mean(Test B drop) / mean(Test A drop) large, e.g. >~0.5,
       interpreted at the population level only).
    3. The effect is strongest at small competition gaps.
    4. The subject-level effect is stronger in small-lesion cases.
    5. The effect survives subject-level permutation testing.
    6. Test B (magnitude preserved, geometry changed) remains sufficient
       to produce the effect.
    7. Test C is materially weaker than Test B.
  QUALIFIED GO: Test B and Test C both real, but Test B explains only
    part of Test A -- distributed local competition geometry, not a
    top-two mechanism.
  KILL: Test B null, no gap-dependence, no small-lesion relationship,
    Test C explains the effect equally/better, or unstable across
    subjects.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000
N_BOOT = 2000
CELLS_PER_SUBJECT = 200
CELLS_PER_QUARTILE = 50
QUARTILE_SAMPLE_POOL = 200_000  # cells drawn (across all subjects) to fix global gap-quartile cutoffs

CKPT_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs" / "AttnGate_seed0" / "checkpoints" / "best.pth"
E48_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"

OFFSETS = [(di, hi, wi) for di in (0, 1) for hi in (0, 1) for wi in (0, 1)]  # fixed 8-slot enumeration
OFFSETS_ARR = np.array(OFFSETS, dtype=np.float64)  # (8,3)


# ==================== dataset / eval utilities (identical to E62) ====================

def fractional_occupancy_64(seg_binary_native):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=(64, 64, 64), mode="area").squeeze().numpy()
    return frac


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    if denom == 0:
        return 1.0
    return float(2 * tp / denom)


def forward_from_enc1(model, enc1, device):
    """Bit-for-bit identical continuation of model.forward() from a given
    enc1 tensor (B,32,64,64,64) onward -- verified against real forward()
    in main()."""
    with torch.no_grad():
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)

        upconv3 = model.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = model.dec3(cat3)

        upconv2 = model.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = model.dec2(cat2)

        upconv1 = model.upconv1(dec2)

        gate = bottleneck
        skip = enc1  # real, un-permuted enc1 for the skip -- matches E62 convention
        g = model.attn_gate1.W_g(gate)
        g_up = F.interpolate(g, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = model.attn_gate1.W_x(skip)
        psi = torch.sigmoid(model.attn_gate1.W_psi(F.relu(g_up + x)))

        enc1_gated = enc1 * psi
        cat1 = torch.cat([upconv1, enc1_gated], dim=1)
        dec1 = model.dec1(cat1)

        probs = model.seg_head(dec1)
        return probs.squeeze(0).squeeze(0).cpu().numpy()


# ==================== competition-geometry extraction ====================

def extract_cell_descriptors(enc1_np):
    """enc1_np: (C,64,64,64). Returns per-(channel,cell) arrays:
    values (N,8) in fixed OFFSETS order, w, r, g, q1_idx, q2_idx (index
    into OFFSETS, 0..7), dq (N,3), d (N,), plus (channel, d0,h0,w0) for
    addressing. N = C * 32*32*32 = 32 * 32768 = 1,048,576."""
    C, D, H, W = enc1_np.shape
    nD, nH, nW = D // 2, H // 2, W // 2
    x = enc1_np.reshape(C, nD, 2, nH, 2, nW, 2).transpose(0, 1, 3, 5, 2, 4, 6)  # (C,nD,nH,nW,2,2,2)
    values = x.reshape(-1, 8)  # (N,8), N = C*nD*nH*nW

    order = np.argsort(-values, axis=1)  # descending
    w_idx = order[:, 0]
    r_idx = order[:, 1]
    w = np.take_along_axis(values, w_idx[:, None], axis=1).squeeze(1)
    r = np.take_along_axis(values, r_idx[:, None], axis=1).squeeze(1)
    gap = w - r

    q1 = OFFSETS_ARR[w_idx]  # (N,3)
    q2 = OFFSETS_ARR[r_idx]  # (N,3)
    dq = q2 - q1
    d = np.linalg.norm(dq, axis=1)

    # channel/cell addressing for reconstructing (c, d0, h0, w0)
    cc, dd, hh, ww = np.meshgrid(np.arange(C), np.arange(nD), np.arange(nH), np.arange(nW), indexing="ij")
    addr_c = cc.reshape(-1)
    addr_d0 = (dd.reshape(-1) * 2)
    addr_h0 = (hh.reshape(-1) * 2)
    addr_w0 = (ww.reshape(-1) * 2)

    return {
        "values": values, "w_idx": w_idx, "r_idx": r_idx, "w": w, "r": r, "gap": gap,
        "dq": dq, "d": d, "addr_c": addr_c, "addr_d0": addr_d0, "addr_h0": addr_h0, "addr_w0": addr_w0,
    }


def sample_cells_for_subject(desc, gap_cutoffs, rng, n_per_quartile=CELLS_PER_QUARTILE):
    """desc: output of extract_cell_descriptors for this subject.
    gap_cutoffs: 3 FIXED global cutoffs splitting the ACTIVE (gap>0)
    competition-gap population into 4 quartile bins (Q1 smallest real
    gap .. Q4 largest), decided globally before subject sampling.

    Real post-ReLU pool1 activations are dominated (~83% empirically) by
    exact gap==0 ties -- cells where winner and runner-up are both zero
    (or otherwise exactly tied), i.e. cells with NO real competition
    happening. These are excluded from the candidate population before
    quartiling (verified/decided by explicit user sign-off): including
    them collapses Q1-Q3 into the same degenerate zero-gap bin, making
    the gap-quartile analysis in section 8 uninformative. Restricting to
    gap>0 keeps the quartiles meaningful (comparing degrees of REAL
    ambiguous vs. clear-cut competition among cells with actual
    competing activation)."""
    gap = desc["gap"]
    active = np.where(gap > 0)[0]
    active_gap = gap[active]
    quartile_active = np.digitize(active_gap, gap_cutoffs)  # 0..3, computed against gap>0 population
    sampled_idx = []
    sampled_q = []
    for q in range(4):
        candidates = active[quartile_active == q]
        if len(candidates) == 0:
            continue
        take = min(n_per_quartile, len(candidates))
        chosen = rng.choice(candidates, size=take, replace=False)
        sampled_idx.append(chosen)
        sampled_q.append(np.full(take, q))
    return np.concatenate(sampled_idx), np.concatenate(sampled_q)


# ==================== counterfactual constructors ====================

def _derange_row(values_8, rng):
    """One row (8,) -> a true derangement of its 8 values (rejection loop,
    trivially cheap for a single row at a time, used only per sampled
    cell -- not the whole-volume path, so a simple loop is fine here)."""
    idx = np.arange(8)
    while True:
        perm = rng.permutation(8)
        if not np.any(perm == idx):
            return values_8[perm]


def build_test_A(values_8, rng):
    """Full 8-value derangement -- local analogue of E62."""
    return _derange_row(values_8, rng)


def build_test_B(values_8, w_idx, r_idx, rng):
    """Winner fixed at its slot; runner-up's value swaps with ONE randomly
    chosen other non-winner, non-runner-up slot. Exactly 2 slots change;
    all 8 values preserved."""
    out = values_8.copy()
    other_slots = [i for i in range(8) if i != w_idx and i != r_idx]
    target = other_slots[rng.integers(0, len(other_slots))]
    out[r_idx], out[target] = out[target], out[r_idx]
    return out


def build_test_C(values_8, w_idx, r_idx, rng):
    """Winner and runner-up slots untouched; derangement applied to the
    remaining 6 slots' values only."""
    out = values_8.copy()
    other_slots = [i for i in range(8) if i != w_idx and i != r_idx]
    sub = out[other_slots]
    idx6 = np.arange(6)
    while True:
        perm = rng.permutation(6)
        if not np.any(perm == idx6):
            break
    out[other_slots] = sub[perm]
    return out


# ==================== vectorized batch versions (whole-volume scale) ====================
# Reuses the E62-style fully-vectorized derangement construction (draw a
# random permutation per row via argsort-of-random-keys, then repair any
# residual fixed points with a per-row cyclic roll, iterated a few times
# -- no Python loop over cells regardless of how many rows).

def _batched_random_permutations(k, n_items, rng):
    keys = rng.random((k, n_items))
    return np.argsort(keys, axis=1)


def _derangement_batch(rows, rng):
    """rows: (M, n) array of values. Returns (M, n) array with EVERY row
    independently deranged (no value stays in its original column)."""
    M, n_items = rows.shape
    arange = np.arange(n_items)
    perm_idx = _batched_random_permutations(M, n_items, rng)
    for _ in range(n_items):
        fixed_mask = perm_idx == arange[None, :]
        if not fixed_mask.any():
            break
        rows_with_fixed = fixed_mask.any(axis=1)
        perm_idx[rows_with_fixed] = np.roll(perm_idx[rows_with_fixed], shift=1, axis=1)
    assert not (perm_idx == arange[None, :]).any(), "Derangement construction failed to converge -- STOP."
    return np.take_along_axis(rows, perm_idx, axis=1)


def build_test_A_batch(values_batch, rng):
    """values_batch: (M,8). Full 8-value derangement, every row
    independently, fully vectorized."""
    return _derangement_batch(values_batch, rng)


def build_test_B_batch(values_batch, w_idx_batch, r_idx_batch, rng):
    """values_batch: (M,8); w_idx_batch,r_idx_batch: (M,) int in [0,8).
    Winner's slot untouched; runner-up's value swaps with ONE randomly
    chosen other non-winner,non-runner-up slot, per row, fully
    vectorized (no Python loop over rows)."""
    M = values_batch.shape[0]
    out = values_batch.copy()
    # For each row, choose a random target slot from the 6 "other" slots
    # (not w_idx, not r_idx). Draw a random index in [0,6) then map to
    # the actual slot id by excluding w_idx/r_idx via a vectorized trick:
    # build all 8 slot ids per row, mask out w_idx/r_idx, pick one of the
    # remaining 6 at random.
    all_slots = np.tile(np.arange(8), (M, 1))  # (M,8)
    is_excluded = (all_slots == w_idx_batch[:, None]) | (all_slots == r_idx_batch[:, None])
    # argsort a random key, but push excluded slots to the end so the
    # first non-excluded slot after a random shuffle is a uniform choice
    # among the 6 valid ones.
    keys = rng.random((M, 8))
    keys = np.where(is_excluded, keys + 10.0, keys)  # excluded slots sort last
    order = np.argsort(keys, axis=1)
    target_slot = order[:, 0]  # first non-excluded slot in random order

    rows = np.arange(M)
    tmp = out[rows, r_idx_batch].copy()
    out[rows, r_idx_batch] = out[rows, target_slot]
    out[rows, target_slot] = tmp
    return out


def build_test_C_batch(values_batch, w_idx_batch, r_idx_batch, rng):
    """values_batch: (M,8); winner/runner-up slots untouched, derangement
    applied to the remaining 6 slots' values only, fully vectorized."""
    M = values_batch.shape[0]
    out = values_batch.copy()
    all_slots = np.tile(np.arange(8), (M, 1))  # (M,8)
    is_excluded = (all_slots == w_idx_batch[:, None]) | (all_slots == r_idx_batch[:, None])
    # Sort each row so the 6 non-excluded slots come first, in a fixed
    # (but arbitrary-per-row) order -- gives a consistent (M,6) sub-view.
    order = np.argsort(is_excluded.astype(np.int8), axis=1, kind="stable")  # False(0) before True(1)
    other_slots = order[:, :6]  # (M,6) column indices of the 6 non-top-two slots, per row

    sub = np.take_along_axis(out, other_slots, axis=1)  # (M,6)
    sub_deranged = _derangement_batch(sub, rng)
    np.put_along_axis(out, other_slots, sub_deranged, axis=1)
    return out


def apply_edits_to_enc1(enc1_np, desc, sampled_idx, edited_values):
    """enc1_np: (C,64,64,64) copy to edit in place. desc: descriptors for
    addressing. sampled_idx: indices into desc arrays for the cells being
    edited. edited_values: (len(sampled_idx), 8) new values for those
    cells, in the fixed OFFSETS order. Returns the edited array (new
    copy). Python-loop version, fine for small (~hundreds) sampled_idx;
    use apply_edits_to_enc1_vectorized for whole-volume-scale edits
    (tens/hundreds of thousands of cells)."""
    out = enc1_np.copy()
    for row, cell_idx in enumerate(sampled_idx):
        c = desc["addr_c"][cell_idx]
        d0 = desc["addr_d0"][cell_idx]
        h0 = desc["addr_h0"][cell_idx]
        w0 = desc["addr_w0"][cell_idx]
        for slot, (di, hi, wi) in enumerate(OFFSETS):
            out[c, d0 + di, h0 + hi, w0 + wi] = edited_values[row, slot]
    return out


def apply_edits_to_enc1_vectorized(enc1_np, desc, sel_idx, edited_values):
    """Vectorized equivalent of apply_edits_to_enc1, for whole-volume-scale
    edits (sel_idx can be tens/hundreds of thousands of cells). Uses
    fancy-indexing scatter instead of a Python loop over cells."""
    C, D, H, W = enc1_np.shape
    out = enc1_np.copy()
    c = desc["addr_c"][sel_idx]        # (M,)
    d0 = desc["addr_d0"][sel_idx]
    h0 = desc["addr_h0"][sel_idx]
    w0 = desc["addr_w0"][sel_idx]
    for slot, (di, hi, wi) in enumerate(OFFSETS):
        out[c, d0 + di, h0 + hi, w0 + wi] = edited_values[:, slot]
    return out


def build_edits_batch(desc, sel_idx, test_fn, rng, needs_wr=False):
    """Vectorized-ish batch construction of edited values for a whole
    population of cells at once, using the appropriate build_test_* rule
    per cell (still one row at a time internally -- test_fn is applied
    per selected cell, since each cell's derangement/swap is drawn
    independently -- but avoids re-deriving indices repeatedly)."""
    values = desc["values"]
    if needs_wr:
        w_idx_arr = desc["w_idx"]
        r_idx_arr = desc["r_idx"]
        return np.stack([test_fn(values[ci], int(w_idx_arr[ci]), int(r_idx_arr[ci]), rng) for ci in sel_idx])
    return np.stack([test_fn(values[ci], rng) for ci in sel_idx])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded E46/v5 checkpoint: {CKPT_PATH}")
    print(f"best_val_dice={ckpt.get('best_val_dice')}\n")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n = len(val_dataset)
    print(f"Validation set size: {n}", flush=True)

    # ---------------- Unit test 1: permutation correctness on synthetic cell ----------------
    synth = np.array([1., 2., 3., 4., 5., 6., 7., 8.])
    rng_unit = np.random.default_rng(12345)
    for _ in range(20):
        permA = build_test_A(synth, rng_unit)
        assert np.allclose(np.sort(permA), np.sort(synth)), "Test A: multiset changed -- STOP."
        assert not np.allclose(permA, synth), "Test A: identity permutation -- STOP."
        assert permA.max() == synth.max() and permA.sum() == synth.sum(), "Test A: order stats changed -- STOP."
    order = np.argsort(-synth)
    w_idx0, r_idx0 = int(order[0]), int(order[1])
    for _ in range(20):
        permB = build_test_B(synth, w_idx0, r_idx0, rng_unit)
        assert np.allclose(np.sort(permB), np.sort(synth)), "Test B: multiset changed -- STOP."
        assert permB[w_idx0] == synth[w_idx0], "Test B: winner slot changed -- STOP."
        n_changed = int(np.sum(~np.isclose(permB, synth)))
        assert n_changed == 2, f"Test B: expected exactly 2 slots changed, got {n_changed} -- STOP."
    for _ in range(20):
        permC = build_test_C(synth, w_idx0, r_idx0, rng_unit)
        assert np.allclose(np.sort(permC), np.sort(synth)), "Test C: multiset changed -- STOP."
        assert permC[w_idx0] == synth[w_idx0] and permC[r_idx0] == synth[r_idx0], \
            "Test C: winner/runner-up slot changed -- STOP."
    print("[Unit test 1] Test A/B/C permutation correctness (multiset, winner/runner-up fixation): PASS.\n")

    # ---------------- Unit test 1b: vectorized batch versions match the per-row semantics ----------------
    rng_batch = np.random.default_rng(54321)
    M = 500
    batch_vals = rng_batch.random((M, 8)) * 10  # random, ties possible but fine for this check
    order_b = np.argsort(-batch_vals, axis=1)
    w_idx_b = order_b[:, 0]
    r_idx_b = order_b[:, 1]

    permA_b = build_test_A_batch(batch_vals.copy(), rng_batch)
    assert np.allclose(np.sort(permA_b, axis=1), np.sort(batch_vals, axis=1)), "Batch Test A: multiset changed -- STOP."
    assert not np.any(np.all(np.isclose(permA_b, batch_vals), axis=1)), "Batch Test A: some row unchanged -- STOP."

    permB_b = build_test_B_batch(batch_vals.copy(), w_idx_b, r_idx_b, rng_batch)
    assert np.allclose(np.sort(permB_b, axis=1), np.sort(batch_vals, axis=1)), "Batch Test B: multiset changed -- STOP."
    assert np.allclose(permB_b[np.arange(M), w_idx_b], batch_vals[np.arange(M), w_idx_b]), \
        "Batch Test B: winner slot changed -- STOP."
    n_changed_per_row = np.sum(~np.isclose(permB_b, batch_vals), axis=1)
    assert np.all(n_changed_per_row == 2), "Batch Test B: expected exactly 2 slots changed per row -- STOP."

    permC_b = build_test_C_batch(batch_vals.copy(), w_idx_b, r_idx_b, rng_batch)
    assert np.allclose(np.sort(permC_b, axis=1), np.sort(batch_vals, axis=1)), "Batch Test C: multiset changed -- STOP."
    assert np.allclose(permC_b[np.arange(M), w_idx_b], batch_vals[np.arange(M), w_idx_b]) and \
        np.allclose(permC_b[np.arange(M), r_idx_b], batch_vals[np.arange(M), r_idx_b]), \
        "Batch Test C: winner/runner-up slot changed -- STOP."
    print(f"[Unit test 1b] Vectorized batch Test A/B/C correctness on {M} synthetic rows: PASS.\n")

    # ---------------- Unit test 2: identity reproduces real forward() ----------------
    image0, _, _ = val_dataset[0]
    image0_b = image0.unsqueeze(0).to(device)
    with torch.no_grad():
        real = model(image0_b)["probs"].squeeze(0).squeeze(0).cpu().numpy()
        enc1_0 = model.enc1(image0_b)
    manual = forward_from_enc1(model, enc1_0, device)
    max_diff = float(np.abs(real - manual).max())
    print(f"[Unit test 2] forward_from_enc1(real enc1) vs real forward(): max abs diff = {max_diff:.6e}")
    assert max_diff == 0.0, "Manual trunk reimplementation mismatch -- STOP."
    print("[Unit test 2] PASS.\n")

    # ---------------- Unit test 3: only intended cells change ----------------
    enc1_0_np = enc1_0.squeeze(0).cpu().numpy()
    desc0 = extract_cell_descriptors(enc1_0_np)
    rng_u3 = np.random.default_rng(777)
    one_idx = np.array([1000])
    edited_vals = np.stack([build_test_A(desc0["values"][1000], rng_u3)])
    edited_enc1 = apply_edits_to_enc1(enc1_0_np, desc0, one_idx, edited_vals)
    diff_mask = ~np.isclose(edited_enc1, enc1_0_np)
    n_diff_voxels = int(diff_mask.sum())
    c0 = desc0["addr_c"][1000]
    d0_, h0_, w0_ = desc0["addr_d0"][1000], desc0["addr_h0"][1000], desc0["addr_w0"][1000]
    expected_region = np.zeros_like(diff_mask)
    expected_region[c0, d0_:d0_ + 2, h0_:h0_ + 2, w0_:w0_ + 2] = True
    outside_changed = int((diff_mask & ~expected_region).sum())
    print(f"[Unit test 3] Single-cell edit: {n_diff_voxels} voxels changed total, "
          f"{outside_changed} changed OUTSIDE the targeted 2x2x2 cell.")
    assert outside_changed == 0, "Edit leaked outside the targeted cell -- STOP."
    assert n_diff_voxels <= 8, "Edit changed more than 8 voxels for a single cell -- STOP."
    print("[Unit test 3] PASS.\n")

    # ---------------- Unit test 3b: vectorized whole-volume apply matches the per-cell version ----------------
    many_idx = np.arange(0, 5000, 7)  # a scattered subset, cheap to check exhaustively
    rng_u3b = np.random.default_rng(4242)
    edits_many = build_test_A_batch(desc0["values"][many_idx].copy(), rng_u3b)
    out_loop = apply_edits_to_enc1(enc1_0_np, desc0, many_idx, edits_many)
    out_vec = apply_edits_to_enc1_vectorized(enc1_0_np, desc0, many_idx, edits_many)
    assert np.array_equal(out_loop, out_vec), "Vectorized apply_edits mismatch vs. per-cell loop version -- STOP."
    print(f"[Unit test 3b] apply_edits_to_enc1_vectorized matches per-cell loop version on "
          f"{len(many_idx)} cells: PASS.\n")

    # ---------------- Step: pool candidate cells to fix GLOBAL gap-quartile cutoffs ----------------
    print(f"Sampling {QUARTILE_SAMPLE_POOL} cells (pooled across subjects) to fix global gap-quartile cutoffs...")
    rng_pool = np.random.default_rng(SEED)
    subject_pick = rng_pool.choice(n, size=min(30, n), replace=False)  # sample a subset of subjects for the pool
    pooled_gaps = []
    with torch.no_grad():
        for si in subject_pick:
            image, _, _ = val_dataset[si]
            image_b = image.unsqueeze(0).to(device)
            enc1 = model.enc1(image_b).squeeze(0).cpu().numpy()
            desc = extract_cell_descriptors(enc1)
            take = min(QUARTILE_SAMPLE_POOL // len(subject_pick) + 1, len(desc["gap"]))
            idx = rng_pool.choice(len(desc["gap"]), size=take, replace=False)
            pooled_gaps.append(desc["gap"][idx])
    pooled_gaps = np.concatenate(pooled_gaps)
    frac_zero_gap = float((pooled_gaps == 0).mean())
    pooled_gaps_active = pooled_gaps[pooled_gaps > 0]
    gap_cutoffs = np.quantile(pooled_gaps_active, [0.25, 0.5, 0.75])
    print(f"Fraction of pooled cells with EXACT gap==0 (dead/tied cells, excluded from quartiling): "
          f"{frac_zero_gap:.4f}")
    print(f"Global gap-quartile cutoffs (25/50/75th pct of pooled gap, ACTIVE gap>0 cells only): "
          f"{gap_cutoffs}\n")

    # ---------------- Main audit ----------------
    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}

    records = []
    for idx in range(n):
        image, mask, subject_id = val_dataset[idx]
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_dataset.subject_dirs[idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_size = int(seg_binary_native.sum())

        mask_frac_64 = fractional_occupancy_64(seg_binary_native)
        target_bin = (mask_frac_64 > 0.5).astype(np.float32)

        with torch.no_grad():
            enc1 = model.enc1(image_b)
        enc1_np = enc1.squeeze(0).cpu().numpy()

        probs_orig = forward_from_enc1(model, enc1, device)
        dice_orig = dice_score((probs_orig >= 0.5).astype(np.float32), target_bin)

        desc = extract_cell_descriptors(enc1_np)
        subj_rng = np.random.default_rng(SEED * 100000 + idx)

        # WHOLE-ACTIVE-VOLUME scale (matches E62's own scale: every cell
        # with real competition (gap>0) is intervened on simultaneously,
        # not a small fixed-size sample -- corrects the scale confound
        # found in the first E63 run, where perturbing only 200/~1.05M
        # cells (~0.02%) produced a near-zero Dice signal even for Test A
        # regardless of mechanism, making the test uninformative).
        active_idx = np.where(desc["gap"] > 0)[0]
        n_active = len(active_idx)

        edits_A = build_test_A_batch(desc["values"][active_idx], subj_rng)
        edits_B = build_test_B_batch(desc["values"][active_idx], desc["w_idx"][active_idx],
                                      desc["r_idx"][active_idx], subj_rng)
        edits_C = build_test_C_batch(desc["values"][active_idx], desc["w_idx"][active_idx],
                                      desc["r_idx"][active_idx], subj_rng)

        enc1_A_np = apply_edits_to_enc1_vectorized(enc1_np, desc, active_idx, edits_A)
        enc1_B_np = apply_edits_to_enc1_vectorized(enc1_np, desc, active_idx, edits_B)
        enc1_C_np = apply_edits_to_enc1_vectorized(enc1_np, desc, active_idx, edits_C)

        probs_A = forward_from_enc1(model, torch.from_numpy(enc1_A_np).unsqueeze(0).to(device), device)
        probs_B = forward_from_enc1(model, torch.from_numpy(enc1_B_np).unsqueeze(0).to(device), device)
        probs_C = forward_from_enc1(model, torch.from_numpy(enc1_C_np).unsqueeze(0).to(device), device)

        dice_A = dice_score((probs_A >= 0.5).astype(np.float32), target_bin)
        dice_B = dice_score((probs_B >= 0.5).astype(np.float32), target_bin)
        dice_C = dice_score((probs_C >= 0.5).astype(np.float32), target_bin)

        drop_A = dice_orig - dice_A
        drop_B = dice_orig - dice_B
        drop_C = dice_orig - dice_C

        # Gap-quartile breakdown: apply Test B to ONLY the cells in one
        # gap quartile at a time, but STILL at whole-stratum scale (every
        # active cell in that quartile across the whole volume, not a
        # small sample of it) -- keeps each quartile's own intervention
        # strong enough to register a Dice signal, while isolating the
        # gap-quartile variable (uses the FIXED global cutoffs from the
        # pooled pre-declaration step).
        active_gap = desc["gap"][active_idx]
        quartile_of_active = np.digitize(active_gap, gap_cutoffs)  # 0..3
        quartile_drops_B = {}
        quartile_n_cells = {}
        for q in range(4):
            q_mask = quartile_of_active == q
            q_idx = active_idx[q_mask]
            if len(q_idx) == 0:
                continue
            q_edits_B = build_test_B_batch(desc["values"][q_idx], desc["w_idx"][q_idx], desc["r_idx"][q_idx], subj_rng)
            enc1_qB_np = apply_edits_to_enc1_vectorized(enc1_np, desc, q_idx, q_edits_B)
            probs_qB = forward_from_enc1(model, torch.from_numpy(enc1_qB_np).unsqueeze(0).to(device), device)
            dice_qB = dice_score((probs_qB >= 0.5).astype(np.float32), target_bin)
            quartile_drops_B[q] = dice_orig - dice_qB
            quartile_n_cells[q] = int(len(q_idx))

        e48r = e48_by_id.get(subject_id)

        records.append({
            "subject_id": subject_id, "native_size": native_size,
            "n_active_cells": int(n_active),
            "dice_orig": dice_orig, "dice_A": dice_A, "dice_B": dice_B, "dice_C": dice_C,
            "drop_A": drop_A, "drop_B": drop_B, "drop_C": drop_C,
            "quartile_drops_B": {str(k): v for k, v in quartile_drops_B.items()},
            "quartile_n_cells": {str(k): v for k, v in quartile_n_cells.items()},
            "e48_causal_drop": e48r["drop"] if e48r is not None else None,
        })

        if (idx + 1) % 10 == 0:
            print(f"  processed {idx+1}/{n} subjects", flush=True)

    with open(OUT_DIR / "E63_competition_geometry_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.\n")

    # ================= Statistics =================
    native_size = np.array([r["native_size"] for r in records], dtype=np.float64)
    drop_A = np.array([r["drop_A"] for r in records], dtype=np.float64)
    drop_B = np.array([r["drop_B"] for r in records], dtype=np.float64)
    drop_C = np.array([r["drop_C"] for r in records], dtype=np.float64)

    mean_active_cells = float(np.mean([r["n_active_cells"] for r in records]))
    print("=== E63 Local Competition Geometry Audit ===")
    print(f"n_subjects={len(records)}, mean active (gap>0) cells/subject={mean_active_cells:.0f} "
          f"(whole-active-volume scale, not a fixed sample), global gap cutoffs={gap_cutoffs.tolist()}\n")
    print(f"Mean drop_A (E62-style local derangement) = {drop_A.mean():.5f} (+/-{drop_A.std():.5f})")
    print(f"Mean drop_B (winner fixed, runner-up moved) = {drop_B.mean():.5f} (+/-{drop_B.std():.5f})")
    print(f"Mean drop_C (top-two fixed, weak rearranged) = {drop_C.mean():.5f} (+/-{drop_C.std():.5f})\n")

    def paired_tests(drop, label):
        t_stat, t_p = stats.ttest_1samp(drop, 0.0, alternative="greater")
        w_stat, w_p = stats.wilcoxon(drop, alternative="greater") if np.any(drop != 0) else (np.nan, 1.0)
        rng = np.random.default_rng(SEED)
        perm_means = np.empty(N_PERM)
        for i in range(N_PERM):
            signs = rng.choice([-1, 1], size=len(drop))
            perm_means[i] = (drop * signs).mean()
        p_perm = float((perm_means >= drop.mean()).mean())
        boot_rng = np.random.default_rng(SEED + 1)
        boot_means = np.empty(N_BOOT)
        for i in range(N_BOOT):
            bs = boot_rng.choice(drop, size=len(drop), replace=True)
            boot_means[i] = bs.mean()
        ci = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))
        print(f"  [{label}] t p={t_p:.4e}, wilcoxon p={w_p:.4e}, sign-flip perm p={p_perm:.4f}, "
              f"boot95CI={ci}")
        return {"t_p": float(t_p), "wilcoxon_p": float(w_p), "perm_p": p_perm, "bootstrap_95ci": list(ci)}

    print("Paired significance tests (H1: mean drop > 0), subject-level:")
    stats_A = paired_tests(drop_A, "Test A")
    stats_B = paired_tests(drop_B, "Test B")
    stats_C = paired_tests(drop_C, "Test C")

    eps = 1e-9
    F_WR = float(drop_B.mean() / (drop_A.mean() + eps))
    print(f"\nF_WR (fraction of Test A effect captured by Test B) = {F_WR:.4f}")

    # ---------------- Competition-gap analysis ----------------
    print("\n=== Competition-gap analysis (Test B, by quartile) ===")
    quartile_means = {}
    for q in range(4):
        vals = []
        for r in records:
            v = r["quartile_drops_B"].get(str(q))
            if v is not None:
                vals.append(v)
        if vals:
            arr = np.array(vals)
            quartile_means[q] = {"mean": float(arr.mean()), "n_subjects": len(arr)}
            print(f"  Q{q+1} (gap {'smallest' if q == 0 else 'largest' if q == 3 else 'mid'}): "
                  f"mean drop_B = {arr.mean():.5f}, n={len(arr)}")

    gap_monotonic = False
    if all(q in quartile_means for q in range(4)):
        means_by_q = [quartile_means[q]["mean"] for q in range(4)]
        # Primary prediction: g down => drop_B up, i.e. Q1 (smallest gap) > Q4 (largest gap)
        gap_monotonic = means_by_q[0] > means_by_q[-1]
        rho_gap, p_gap = stats.spearmanr([0, 1, 2, 3], means_by_q)
        print(f"  Spearman(quartile index, mean drop_B) = {rho_gap:+.4f} (p={p_gap:.4f}) "
              f"(negative = drop_B decreases as gap increases, matching prediction)")
        gap_hypothesis_supported = gap_monotonic and (rho_gap < 0)
    else:
        gap_hypothesis_supported = False
    print(f"  Gap hypothesis supported (Q1 mean > Q4 mean, monotonic decreasing trend): {gap_hypothesis_supported}")

    # ---------------- Small-lesion analysis ----------------
    print("\n=== Small-lesion analysis (subject-level drop_B vs native_size) ===")
    rho_size, p_size_param = stats.spearmanr(native_size, drop_B)
    rng2 = np.random.default_rng(SEED + 2)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng2.permutation(drop_B)
        perm_rhos[i], _ = stats.spearmanr(native_size, perm_y)
    p_size_perm = float((np.abs(perm_rhos) >= np.abs(rho_size)).mean())
    print(f"  Spearman(native_size, drop_B) = {rho_size:+.4f} (parametric p={p_size_param:.4e}, "
          f"permutation p={p_size_perm:.4f})")
    small_lesion_supported = (rho_size < 0) and (p_size_perm < 0.05)
    print(f"  Small-lesion hypothesis supported (rho<0, perm p<0.05): {small_lesion_supported}")

    # ---------------- Decision rule ----------------
    print("\n=== Step 15: Decision ===")
    criterion_1 = stats_B["t_p"] < 0.05 and stats_B["perm_p"] < 0.05 and drop_B.mean() > 0
    criterion_2 = F_WR > 0.5
    criterion_3 = gap_hypothesis_supported
    criterion_4 = small_lesion_supported
    criterion_5 = stats_B["perm_p"] < 0.05
    criterion_6 = criterion_1  # Test B by construction preserves w,r magnitudes; sufficiency = criterion 1 itself
    criterion_7 = drop_C.mean() < drop_B.mean() and stats_C["perm_p"] >= 0.05 or (drop_C.mean() < 0.5 * drop_B.mean())

    print(f"  1. Test B non-zero, reproducible effect: {criterion_1}")
    print(f"  2. Test B captures substantial fraction of Test A (F_WR>0.5): {criterion_2} (F_WR={F_WR:.4f})")
    print(f"  3. Strongest at small competition gaps: {criterion_3}")
    print(f"  4. Stronger in small-lesion cases: {criterion_4}")
    print(f"  5. Survives subject-level permutation testing: {criterion_5}")
    print(f"  6. Magnitude-preserved geometry change sufficient: {criterion_6}")
    print(f"  7. Test C materially weaker than Test B: {criterion_7}")

    all_go = all([criterion_1, criterion_2, criterion_3, criterion_4, criterion_5, criterion_6, criterion_7])
    qualified = criterion_1 and (stats_C["perm_p"] < 0.05) and not all_go

    if all_go:
        verdict = "GO"
    elif qualified:
        verdict = "QUALIFIED_GO"
    else:
        verdict = "KILL"
    print(f"\n=== VERDICT: {verdict} ===")

    summary = {
        "checkpoint": str(CKPT_PATH),
        "n_subjects": len(records), "mean_active_cells_per_subject": mean_active_cells,
        "intervention_scale": "whole_active_volume_per_subject", "global_gap_cutoffs": gap_cutoffs.tolist(),
        "mean_drop_A": float(drop_A.mean()), "mean_drop_B": float(drop_B.mean()), "mean_drop_C": float(drop_C.mean()),
        "stats_A": stats_A, "stats_B": stats_B, "stats_C": stats_C,
        "F_WR": F_WR,
        "quartile_means_drop_B": {str(k): v for k, v in quartile_means.items()},
        "gap_hypothesis_supported": bool(gap_hypothesis_supported),
        "size_dependence": {"rho": float(rho_size), "parametric_p": float(p_size_param), "permutation_p": p_size_perm},
        "small_lesion_hypothesis_supported": bool(small_lesion_supported),
        "criteria": {
            "1_test_b_nonzero": bool(criterion_1), "2_F_WR_substantial": bool(criterion_2),
            "3_gap_dependence": bool(criterion_3), "4_small_lesion": bool(criterion_4),
            "5_permutation_survives": bool(criterion_5), "6_sufficiency": bool(criterion_6),
            "7_test_c_weaker": bool(criterion_7),
        },
        "verdict": verdict,
    }
    with open(OUT_DIR / "E63_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E63_summary.json")


if __name__ == "__main__":
    main()
