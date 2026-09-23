"""
E183-A -- Response geometry analysis.

Pre-registered in docs/phases/PHASE_E183_A_RESPONSE_GEOMETRY_AUDIT_PREREG.md.
Read that first. Derives Gamma and the four candidate geometric descriptors
from the RAW response vectors (run_e183_a_response_vectors.py output), joins
with mean Delta per tile (Delta-pairing decision recorded in the prereg doc),
and runs the nested subject-FE regression: Delta ~ Gamma vs Delta ~ Gamma +
{centroid, eff_dim, coherence, cross_family_alignment}.

CPU-only, no GPU. Operates on already-collected E183-A response vectors and
the FROZEN E182 delta records (for mean Delta per tile).
"""
import sys
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).parent
e180_dir = OUT_DIR.parent / "e180"
e182_dir = OUT_DIR.parent / "e182"
sys.path.insert(0, str(e180_dir))

from e180_stats import (subject_fixed_effects_ols, within_subject_permutation_null,  # noqa: E402
                        reduced_granularity_pairs)

N_PERM = 1000
T6_CONFIGS = ["T6_0.35", "T6_0.5"]
T7_CONFIGS = ["T7_0.6", "T7_0.8", "T7_1.1", "T7_1.45"]
ALL_CONFIGS = T6_CONFIGS + T7_CONFIGS


def entropy_of(p):
    p = np.clip(p, 1e-300, None)
    return float(-(p * np.log(p)).sum())


def compute_gamma_and_descriptors(response_vectors):
    """response_vectors: dict config_key -> [ET,TC,WT] displacement.
    Returns (gamma, centroid_mag, eff_dim, coherence, cross_family_alignment)."""
    R = np.array([response_vectors[k] for k in ALL_CONFIGS])   # (6, 3)

    # Gamma: max pairwise Euclidean distance among the response vectors
    # (same "diameter" construction as E180-E182's max-pairwise-disagreement
    # Gamma, applied here to the region-displacement representation)
    dists = []
    for j in range(len(R)):
        for k in range(j + 1, len(R)):
            dists.append(np.linalg.norm(R[j] - R[k]))
    gamma = float(np.max(dists))

    # 1. centroid magnitude
    centroid = R.mean(axis=0)
    centroid_mag = float(np.linalg.norm(centroid))

    # 2. effective dimensionality of the response covariance (entropy-based,
    # same exp(H(p)) convention used throughout this project)
    cov = np.cov(R.T)   # (3,3)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)
    tot = eigvals.sum()
    if tot > 1e-12:
        p = eigvals / tot
        eff_dim = float(np.exp(entropy_of(p)))
    else:
        eff_dim = 1.0   # degenerate (all responses identical) -> minimal dimensionality

    # 3. coherence: mean pairwise cosine similarity between displacement directions
    norms = np.linalg.norm(R, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-12, 1.0, norms)
    R_unit = R / norms_safe
    cos_sims = []
    for j in range(len(R)):
        for k in range(j + 1, len(R)):
            if norms[j] < 1e-12 or norms[k] < 1e-12:
                continue
            cos_sims.append(float(np.dot(R_unit[j], R_unit[k])))
    coherence = float(np.mean(cos_sims)) if cos_sims else 0.0

    # 4. cross-family alignment: cosine similarity between T6's and T7's
    # mean response direction
    r_t6 = np.array([response_vectors[k] for k in T6_CONFIGS]).mean(axis=0)
    r_t7 = np.array([response_vectors[k] for k in T7_CONFIGS]).mean(axis=0)
    n_t6, n_t7 = np.linalg.norm(r_t6), np.linalg.norm(r_t7)
    if n_t6 < 1e-12 or n_t7 < 1e-12:
        cross_family_alignment = 0.0
    else:
        cross_family_alignment = float(np.dot(r_t6, r_t7) / (n_t6 * n_t7))

    return gamma, centroid_mag, eff_dim, coherence, cross_family_alignment


def load_mean_delta():
    """mean delta_i_global across all 6 configs, per (sid, window_id), from
    the FROZEN E182 delta records (T6 unchanged, T7 the calibrated set)."""
    delta = json.load(open(e182_dir / "FROZEN" / "E182_D_delta.json"))
    by_tile = {}
    for r in delta:
        key = (r["sid"], r["window_id"])
        by_tile.setdefault(key, []).append(r["delta_i_global"])
    return {k: float(np.mean(v)) for k, v in by_tile.items()}


def run_analysis(records, mean_delta, label):
    print(f"\n{'='*70}\n[{label}] n_tile_records={len(records)}\n{'='*70}")

    y, gam, centroid, eff_dim, coherence, cross_align, sids = [], [], [], [], [], [], []
    missing_delta = 0
    for r in records:
        key = (r["sid"], r["window_id"])
        if key not in mean_delta:
            missing_delta += 1
            continue
        g, c, ed, coh, ca = compute_gamma_and_descriptors(r["response_vectors"])
        y.append(mean_delta[key])
        gam.append(g)
        centroid.append(c)
        eff_dim.append(ed)
        coherence.append(coh)
        cross_align.append(ca)
        sids.append(r["sid"])

    if missing_delta:
        print(f"[Warning] {missing_delta} tile records had no matching Delta (skipped)")

    y = np.array(y)
    gam = np.array(gam)
    centroid = np.array(centroid)
    eff_dim = np.array(eff_dim)
    coherence = np.array(coherence)
    cross_align = np.array(cross_align)

    print(f"[Descriptor ranges] Gamma: [{gam.min():.4f},{gam.max():.4f}]  "
          f"centroid: [{centroid.min():.4f},{centroid.max():.4f}]  "
          f"eff_dim: [{eff_dim.min():.4f},{eff_dim.max():.4f}]  "
          f"coherence: [{coherence.min():.4f},{coherence.max():.4f}]  "
          f"cross_align: [{cross_align.min():.4f},{cross_align.max():.4f}]")

    fe_base = subject_fixed_effects_ols(y, gam, sids)
    print(f"\n[Baseline] Delta ~ Gamma: beta={fe_base['beta'][0]:+.4f} p={fe_base['p'][0]:.4f} "
          f"R2={fe_base['r2']:.4f}  n={fe_base['n']} n_subj={fe_base['n_subjects']}")

    X_ext = np.column_stack([gam, centroid, eff_dim, coherence, cross_align])
    fe_ext = subject_fixed_effects_ols(y, X_ext, sids)
    print(f"[Extended] Delta ~ Gamma + 4 descriptors: R2={fe_ext['r2']:.4f}  "
          f"(baseline R2={fe_base['r2']:.4f}, delta={fe_ext['r2']-fe_base['r2']:+.4f})")
    for name, beta, p in zip(["gamma", "centroid", "eff_dim", "coherence", "cross_align"],
                             fe_ext["beta"], fe_ext["p"]):
        print(f"    {name}: beta={beta:+.4f} p={p:.4f}")

    def r2_of(y_, X_, sids_):
        return subject_fixed_effects_ols(y_, X_, sids_)["r2"]

    r2_base = r2_of(y, gam.reshape(-1, 1), sids)
    r2_full = r2_of(y, X_ext, sids)
    observed_delta_r2 = r2_full - r2_base

    rng = np.random.default_rng(0)
    sids_arr = np.array(sids)
    null_deltas = []
    desc_cols = np.column_stack([centroid, eff_dim, coherence, cross_align])
    for _ in range(N_PERM):
        shuffled = desc_cols.copy()
        for s in set(sids):
            idx = np.where(sids_arr == s)[0]
            perm_idx = rng.permutation(idx)
            shuffled[idx] = desc_cols[perm_idx]
        X_perm = np.column_stack([gam, shuffled])
        r2_perm_full = r2_of(y, X_perm, sids)
        null_deltas.append(r2_perm_full - r2_base)
    null_deltas = np.array(null_deltas)
    perm_p = float((null_deltas >= observed_delta_r2).mean())
    p_display = f"< {1/N_PERM:.3f}" if perm_p == 0 else f"{perm_p:.4f}"

    print(f"\n[Joint incremental R2 of the 4 descriptors over Gamma alone] "
          f"observed={observed_delta_r2:+.4f}  null_mean={null_deltas.mean():.4f}  "
          f"permutation_p={p_display}")

    outcome = "GEOMETRY_INFORMATIVE" if perm_p < 0.05 else "GAMMA_SUFFICIENT"
    print(f"[PRE-REGISTERED READING] {outcome}")

    return {
        "n_tiles": len(y), "n_subjects": len(set(sids)),
        "descriptor_ranges": {
            "gamma": [float(gam.min()), float(gam.max())],
            "centroid": [float(centroid.min()), float(centroid.max())],
            "eff_dim": [float(eff_dim.min()), float(eff_dim.max())],
            "coherence": [float(coherence.min()), float(coherence.max())],
            "cross_align": [float(cross_align.min()), float(cross_align.max())],
        },
        "baseline_model": {"beta_gamma": fe_base["beta"][0], "p_gamma": fe_base["p"][0],
                           "r2": fe_base["r2"]},
        "extended_model": {
            "r2": fe_ext["r2"],
            "coefficients": {name: {"beta": float(b), "p": float(p)}
                             for name, b, p in zip(
                                 ["gamma", "centroid", "eff_dim", "coherence", "cross_align"],
                                 fe_ext["beta"], fe_ext["p"])},
        },
        "joint_incremental_r2_test": {
            "observed_delta_r2": float(observed_delta_r2),
            "null_mean": float(null_deltas.mean()), "null_std": float(null_deltas.std()),
            "permutation_p": perm_p, "n_perm": N_PERM,
        },
        "PRE_REGISTERED_READING": outcome,
    }


def main():
    records = json.load(open(OUT_DIR / "E183_A_response_vectors_raw.json"))
    mean_delta = load_mean_delta()
    ledger = json.load(open(e180_dir / "E180_tile_ledger_full.json"))
    print(f"[Data] {len(records)} tile records with response vectors")

    # ---- PRIMARY: full tile set. Flagged: with only 64 tiles/10 subjects and
    # heavy within-subject tile-overlap (same artifact E180's Stage 5.5 found),
    # this R2 is expected to be INFLATED and is not trustworthy standalone --
    # reported alongside, never instead of, the reduced-granularity check below. ----
    primary = run_analysis(records, mean_delta, "PRIMARY (full tile set -- R2 likely inflated "
                           "by tile non-independence, see reduced-granularity check)")

    # ---- ROBUSTNESS: reduced-granularity, per E180's established fix for
    # exactly this artifact -- max-separated tile pair per subject, fixed
    # enc3-center-distance criterion, not reselected. ----
    subjects_in_data = set(r["sid"] for r in records)
    ledger_subset = {sid: v for sid, v in ledger.items() if sid in subjects_in_data}
    reduced_pairs = reduced_granularity_pairs(ledger_subset)
    reduced_window_ids = set()
    for sid, pair in reduced_pairs.items():
        reduced_window_ids.add((sid, pair[0]))
        reduced_window_ids.add((sid, pair[1]))
    reduced_records = [r for r in records if (r["sid"], r["window_id"]) in reduced_window_ids]
    print(f"\n[Reduced-granularity] {len(reduced_records)} tiles retained "
          f"(max-separated pair per subject) out of {len(records)} total")

    robustness = run_analysis(reduced_records, mean_delta,
                              "ROBUSTNESS (reduced-granularity, max-separated pair per subject)")

    results = {"primary_full_tiles": primary, "robustness_reduced_granularity": robustness}
    with open(OUT_DIR / "E183_A_response_geometry.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\nSaved E183_A_response_geometry.json")


if __name__ == "__main__":
    main()
