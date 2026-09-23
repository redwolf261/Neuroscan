"""
Phase E124: Transition-Specific Signature Audit -- Effective Channel Rank
per Pooling Window (third targeted property, per user's explicit request
for one more property with a specific new rationale after E122/E123 both
failed the same way -- NOT a shotgun continuation).

CONTEXT: E121 found N_1(64->32)~=N_3(16->8)>>N_2(32->16). E122 (relative
lesion scale) and E123 (lesion/background separability) BOTH showed a
visually plausible match to this pattern but BOTH failed the pooled
cross-stage regression (Delta R^2 ~0.001, below the pre-registered 0.01
magnitude bar, despite "significant" p-values that are an artifact of
large pooled n). Two independent property searches converging on the same
null suggests either (a) no clean continuous property explains this, and
stage identity itself is doing real (irreducible) explanatory work, or
(b) the right property hasn't been tried yet.

NEW RATIONALE (distinct from E122's geometric-scale and E123's
separability): channel CAPACITY. enc1/enc2/enc3 have 32/64/128 channels
respectively, each stage's channel count doubling alongside its 2x spatial
compression. Raw channel count is a FIXED CONSTANT per stage (same
structural flaw as E122's q_k -- rejected as the measure for that reason).
Instead: EFFECTIVE RANK per pooling window -- how much of a window's
channel capacity is genuinely independent (non-redundant) information,
computed per SUBJECT per WINDOW (varies naturally across subjects, unlike
raw channel count). Hypothesis: stages where pooling is more destructive
(1 and 3) have LOW effective-rank-fraction relative to their channel
count (concentrated, non-redundant signal -- pooling destroys real
independent information); stage 2 has HIGH effective-rank-fraction
(redundant channels -- pooling loses comparatively little).

MEASURE: for each subject, each stage's pre-pooling tensor (C channels,
spatial res r^3), partition into non-overlapping 2x2x2 windows (8
sub-voxels each, matching MaxPool3d's own partition, E119's exact
windowing convention). For each window, form the (C x 8) matrix of
channel activations across the 8 sub-voxels, compute its covariance's
eigenvalues (C x C, but rank <=8 given only 8 samples -- use the 8x8 Gram
matrix's eigenvalues instead, mathematically equivalent nonzero spectrum,
much cheaper), and compute the PARTICIPATION RATIO:
    PR = (sum(eigenvalues))^2 / sum(eigenvalues^2)
a standard, assumption-light continuous measure of effective dimensionality
(PR=1 if all variance is in one eigenvector -- maximally redundant across
sub-voxels in a channel sense is NOT what PR measures here; PR measures
how many independent DIRECTIONS the 8 sub-voxel activation vectors span in
channel-space -- PR=8 if all 8 sub-voxels are along orthogonal/equal-
variance directions in channel space, i.e. maximally distinct sub-voxel
signatures; PR=1 if all 8 sub-voxels have nearly the same channel-space
activation pattern, i.e. maximally redundant across the window).
E_frac (per subject per stage) = mean over windows of PR / 8 (normalized
to [1/8, 1] -- low E_frac means sub-voxels within a window are highly
redundant with each other in channel-space, so pooling (which discards 7
of 8 sub-voxels) loses little; high E_frac means sub-voxels are highly
DISTINCT from each other, so pooling destroys real, non-redundant
per-sub-voxel information).

HYPOTHESIS (restated precisely): stages 1 and 3 have HIGH E_frac
(sub-voxels distinct, pooling destructive), stage 2 has LOW E_frac
(sub-voxels redundant, pooling non-destructive) -- i.e. the OPPOSITE
association direction from "channel capacity" naively suggests (more
channels != more redundancy at the sub-voxel level; this is about
SPATIAL redundancy WITHIN a window, measured through the channel
dimension, not channel redundancy itself).

CROSS-STAGE TEST: identical pooled-regression + within-subject-permutation
design as E122/E123 (for direct comparability of Delta R^2 magnitudes).

PRE-REGISTERED DECISION GATES: identical structure to E123's (Case
A/B/C/D), applied to E_frac in place of S_stage.

NO ARCHITECTURE MODIFICATION, NO TRAINING, NO NOVELTY CLAIM.
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
E121_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e121" / "E121_per_subject_table.json"

CKPT_PATH = (project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs"
             / "AttnGate_seed0" / "checkpoints" / "best.pth")
EXPECTED_VAL_DICE = 0.9101624600589275

STAGES = {"N_1": ("enc1", 64), "N_2": ("enc2", 32), "N_3": ("enc3", 16)}


def windows_from_tensor(t):
    """t: (C, R, R, R) -> (C, (R/2)^3, 8) sub-voxel values per window,
    IDENTICAL partition/order convention to E119's windows_from_enc3."""
    C = t.shape[0]
    x = t.unfold(1, 2, 2).unfold(2, 2, 2).unfold(3, 2, 2)
    n_windows = (t.shape[1] // 2) * (t.shape[2] // 2) * (t.shape[3] // 2)
    return x.contiguous().view(C, n_windows, 8)


MAX_EIGH_BATCH = 4096  # cusolver's batched syevBatched has an internal
# batch-size limit -- the full enc1 stage has 32768 windows in one call,
# which crashed with CUSOLVER_STATUS_INVALID_VALUE (a real crash, caught
# via the persistent monitor's error report, NOT a NaN issue despite the
# error text's own suggestion -- verified: the identical unit-test input,
# batch size 1, succeeded cleanly). Fixed by chunking the batched
# eigendecomposition -- same math, just processed in smaller batches to
# stay under cusolver's internal limit.


def participation_ratio_per_window(windows):
    """windows: (C, nW, 8) torch tensor. For each window, form the 8x8 Gram
    matrix of the C-dim sub-voxel vectors, compute eigenvalues, return PR
    = (sum eig)^2 / sum(eig^2) per window. Chunked batched eigvalsh (see
    MAX_EIGH_BATCH note above)."""
    v = windows.permute(1, 2, 0)  # (nW, 8, C)
    nW = v.shape[0]
    pr_chunks = []
    for start in range(0, nW, MAX_EIGH_BATCH):
        end = min(start + MAX_EIGH_BATCH, nW)
        v_chunk = v[start:end]
        gram = torch.bmm(v_chunk, v_chunk.transpose(1, 2))  # (chunk, 8, 8)
        eigvals = torch.linalg.eigvalsh(gram)  # (chunk, 8), ascending, >=0 (PSD)
        eigvals = eigvals.clamp(min=0)
        s1 = eigvals.sum(dim=1)
        s2 = (eigvals ** 2).sum(dim=1)
        pr = torch.where(s2 > 1e-12, (s1 ** 2) / (s2 + 1e-12), torch.ones_like(s1))
        pr_chunks.append(pr.cpu().numpy())
    return np.concatenate(pr_chunks)  # (nW,)


def unit_test_participation_ratio():
    # Case 1: all 8 sub-voxels identical across channels -> rank-1 Gram -> PR=1
    C = 5
    base = torch.randn(C)
    w = base.unsqueeze(1).repeat(1, 8).unsqueeze(1)  # (C,1,8)
    pr = participation_ratio_per_window(w)
    assert abs(pr[0] - 1.0) < 1e-3, f"Identical sub-voxels should give PR~1, got {pr[0]}"

    # Case 2: 8 sub-voxels orthogonal in channel space (C>=8, identity-like) -> PR~8
    C2 = 8
    w2 = torch.eye(C2).unsqueeze(1)  # (C2, 1, 8) -- each sub-voxel is a one-hot channel vector
    pr2 = participation_ratio_per_window(w2)
    assert pr2[0] > 6.5, f"Orthogonal sub-voxels should give PR close to 8, got {pr2[0]}"
    print(f"[Unit test] participation_ratio_per_window: identical->PR={pr[0]:.3f} (expect ~1), "
          f"orthogonal->PR={pr2[0]:.3f} (expect ~8). PASS.")


def fractional_occupancy(seg_binary_native, shape):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=shape, mode="area").squeeze().numpy()
    return frac


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    unit_test_participation_ratio()

    with open(E121_TABLE_PATH) as f:
        e121_records = json.load(f)
    e121_by_id = {r["subject_id"]: r for r in e121_records}
    print(f"Loaded {len(e121_records)} subjects from E121's per-subject table.")

    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    val_dice = ckpt.get("best_val_dice") or ckpt.get("val_dice")
    assert val_dice is not None and abs(float(val_dice) - EXPECTED_VAL_DICE) < 1e-6, \
        f"Checkpoint identity check FAILED: expected {EXPECTED_VAL_DICE}, got {val_dice}"
    print(f"[Sanity check] checkpoint identity PASS (val_dice={val_dice}).")

    val_ds = BraTSDataset(root_dir=str(project_root / "Dataset" / "Training"), split="val",
                          val_split=0.1, target_shape=(64, 64, 64), normalize=True)

    records = []
    print(f"\nExtracting pre-pooling activations + computing E_frac for "
          f"{len(val_ds)} subjects...", flush=True)
    for idx in range(len(val_ds)):
        image, mask, sid = val_ds[idx]
        if sid not in e121_by_id:
            continue
        image_b = image.unsqueeze(0).to(device)

        with torch.no_grad():
            enc1 = model.enc1(image_b)
            pool1 = model.pool1(enc1)
            enc2 = model.enc2(pool1)
            pool2 = model.pool2(enc2)
            enc3 = model.enc3(pool2)

        tensors = {"N_1": enc1.squeeze(0), "N_2": enc2.squeeze(0), "N_3": enc3.squeeze(0)}

        rec = dict(e121_by_id[sid])
        for stage, (attr, res) in STAGES.items():
            t = tensors[stage]
            windows = windows_from_tensor(t)  # (C, nW, 8)
            pr = participation_ratio_per_window(windows)  # (nW,)
            rec[f"E_{stage}"] = float((pr / 8.0).mean())  # normalized to [1/8, 1]

        records.append(rec)
        if len(records) % 25 == 0:
            print(f"  {len(records)}/{len(e121_records)}", flush=True)

    with open(OUT_DIR / "E124_per_subject_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nTotal subjects: {len(records)}")

    print("\n=== Raw per-stage effective-rank fraction E_stage ===")
    valid = {}
    for stage in STAGES:
        vals = np.array([r[f"E_{stage}"] for r in records])
        valid[stage] = vals
        print(f"  E_{stage}: mean={vals.mean():.4f} std={vals.std():.4f}")

    e1, e2, e3 = valid["N_1"].mean(), valid["N_2"].mean(), valid["N_3"].mean()
    visible_pattern = (abs(e1 - e3) < 0.5 * abs(e1 - e2) + 0.5 * abs(e3 - e2)) and \
                       (abs(e1 - e2) > 0.02 or abs(e3 - e2) > 0.02)
    print(f"\n  Visible pattern check (e1~=e3, both differ from e2): "
          f"e1={e1:.4f}, e2={e2:.4f}, e3={e3:.4f} -> {'YES' if visible_pattern else 'NO'}")

    print("\n=== Pooled cross-stage regression: does E_stage explain N beyond stage+size? ===")
    stage_dummy = {"N_1": (1, 0), "N_2": (0, 1), "N_3": (0, 0)}
    pooled_N, pooled_E, pooled_size, pooled_d1, pooled_d2 = [], [], [], [], []
    for r in records:
        for stage in STAGES:
            pooled_N.append(r[stage])
            pooled_E.append(r[f"E_{stage}"])
            pooled_size.append(np.log(r["native_size"] + 1))
            d1, d2 = stage_dummy[stage]
            pooled_d1.append(d1)
            pooled_d2.append(d2)
    pooled_N, pooled_E = np.array(pooled_N), np.array(pooled_E)
    pooled_size, pooled_d1, pooled_d2 = np.array(pooled_size), np.array(pooled_d1), np.array(pooled_d2)

    X_p1 = np.column_stack([np.ones(len(pooled_N)), pooled_d1, pooled_d2, pooled_size])
    beta_p1, *_ = np.linalg.lstsq(X_p1, pooled_N, rcond=None)
    resid_p1 = pooled_N - X_p1 @ beta_p1
    r2_p1 = 1 - (resid_p1 ** 2).sum() / ((pooled_N - pooled_N.mean()) ** 2).sum()

    X_p2 = np.column_stack([np.ones(len(pooled_N)), pooled_d1, pooled_d2, pooled_size, pooled_E])
    beta_p2, *_ = np.linalg.lstsq(X_p2, pooled_N, rcond=None)
    resid_p2 = pooled_N - X_p2 @ beta_p2
    r2_p2 = 1 - (resid_p2 ** 2).sum() / ((pooled_N - pooled_N.mean()) ** 2).sum()
    e_coef = beta_p2[-1]

    print(f"Pooled n={len(pooled_N)} observations ({len(records)} subjects x 3 stages)")
    print(f"Model P1 (N ~ stage + size):     R^2={r2_p1:.4f}")
    print(f"Model P2 (N ~ stage + size + E): R^2={r2_p2:.4f}, E coefficient={e_coef:+.4f}")
    print(f"Delta R^2 from adding E: {r2_p2 - r2_p1:+.4f}")

    rng = np.random.default_rng(SEED + 30)
    E_by_subject = pooled_E.reshape(len(records), 3)
    perm_coefs = np.empty(N_PERM)
    for i in range(N_PERM):
        E_perm = np.array([rng.permutation(row) for row in E_by_subject]).reshape(-1)
        X_perm = np.column_stack([np.ones(len(pooled_N)), pooled_d1, pooled_d2, pooled_size, E_perm])
        beta_perm, *_ = np.linalg.lstsq(X_perm, pooled_N, rcond=None)
        perm_coefs[i] = beta_perm[-1]
    p_perm_e = float((np.abs(perm_coefs) >= np.abs(e_coef)).mean())
    print(f"Within-subject-permutation test on E's coefficient: p={p_perm_e:.4f}")

    print("\n=== DECISION ===")
    delta_r2 = r2_p2 - r2_p1
    e_matters = (abs(delta_r2) > 0.01) and (p_perm_e < 0.05)

    if not visible_pattern and not e_matters:
        decision = "CASE_A_KILL_NO_PATTERN_NO_EXPLANATORY_POWER"
        detail = "No visible pattern in raw E_stage, and no explanatory power beyond stage+size. KILL."
    elif visible_pattern and not e_matters:
        decision = "CASE_B_KILL_REDUNDANT_WITH_STAGE_SIZE"
        detail = f"Raw E_stage shows the visible pattern, but redundant with stage+size (Delta R^2={delta_r2:.4f}, perm p={p_perm_e:.4f}). KILL."
    elif visible_pattern and e_matters:
        decision = "CASE_C_INTERESTING_EFFECTIVE_RANK_SIGNATURE"
        detail = f"E_stage shows the visible pattern AND survives the pooled regression (Delta R^2={delta_r2:+.4f}, perm p={p_perm_e:.4f}). Proceed to a full prior-art audit before any architecture work."
    else:
        decision = "CASE_D_NO_CLEAN_PROPERTY_UNEXPECTED"
        detail = "E_stage adds explanatory power but no visible raw pattern -- report honestly."

    print(f"{decision}\n{detail}")

    summary = {
        "checkpoint": str(CKPT_PATH), "val_dice_check": val_dice,
        "n_subjects": len(records),
        "E_stage_means": {s: float(valid[s].mean()) for s in STAGES},
        "E_stage_stds": {s: float(valid[s].std()) for s in STAGES},
        "visible_pattern": bool(visible_pattern),
        "pooled_r2_stage_size_only": float(r2_p1),
        "pooled_r2_with_E": float(r2_p2),
        "delta_r2_from_E": float(delta_r2),
        "E_coefficient": float(e_coef),
        "within_subject_permutation_p": p_perm_e,
        "decision": decision, "detail": detail,
    }
    with open(OUT_DIR / "E124_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E124_summary.json, E124_per_subject_table.json")


if __name__ == "__main__":
    main()
