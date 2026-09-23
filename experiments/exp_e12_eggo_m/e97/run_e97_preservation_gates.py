"""
Phase E97: Frozen-Feature Preservation Pilot -- Gate 1 (does preservation
recover the E92/E96 deficit) and Gate 2 (is the gain specifically
predicted by N_b, beyond ordinary difficulty covariates, and does
N_b-conditioning beat a constant-lambda control).

CONTEXT: E48-E96 established a causal chain (bottleneck necessity ->
detail/resolution-specific benefit, rho=0.77-0.82). Candidate mechanism:
Causal Necessity-Conditioned Compression Preservation -- a small residual
bypass R_phi(enc3) added to the pooled bottleneck, with strength
lambda(x) = f(N_b(x)), trained to preserve pre-compression information
(NOT directly optimized for segmentation). Novelty audit (see
phase_e97... memory, written after this run) found no direct collision
on the specific combination, but flagged that generic preservation/
distillation losses are themselves a crowded primitive -- so the
critical test is whether the CAUSAL conditioning variable N_b adds
anything beyond (a) a constant-strength preservation branch and (b)
ordinary difficulty covariates (size, baseline Dice, entropy).

NO TRAINING of the main segmentation network -- fully frozen (E48-E96's
exact checkpoint). Only the tiny preservation branch R_phi and a linear
readout q are trained, per subject-group (see below), matching this
project's own frozen-feature-probe convention (E90-E94).

DESIGN:
  R_phi: a single 1x1x1 Conv3d, enc3's 128 channels -> 256 channels
    (matching bottleneck's own channel count), stride 1, no spatial
    change -- deliberately minimal capacity (linear), consistent with
    this project's "linear probe" convention throughout E90-E96, so any
    recovered decodability is attributable to the INFORMATION being
    accessible via a simple readout, not to R_phi's own representational
    power.
  Z_b' = Z_b + lambda(x) * Downsample(R_phi(enc3))  -- R_phi operates at
    enc3's native 16^3 resolution, then is average-pooled (2x2x2, matching
    the pool3 spatial reduction exactly) to align with Z_b's 8^3 grid
    before the residual add.
  q: linear probe (1x1x1 Conv3d, 256->1, upsampled to 64^3), reused from
    E90-E94's exact methodology, trained on Z_b' to predict the lesion
    mask -- this IS how "decodability" is measured throughout this
    project, kept identical here for direct comparability.

THREE CONDITIONS, same R_phi architecture, differing ONLY in lambda:
  (1) BASE: lambda=0 for all subjects (equivalent to no preservation --
      reproduces E90-E92's pool3_output/bottleneck decodability numbers
      as an internal consistency check).
  (2) CONSTANT: lambda_i = lambda_const (same fixed positive value for
      ALL subjects, chosen to match the MEAN of the NC condition's
      lambda for a fair comparison) -- this is the critical control the
      user specified: isolates whether ANY preservation helps, before
      asking whether CAUSAL conditioning specifically helps.
  (3) NC (Necessity-Conditioned): lambda_i = lambda_max * sigmoid((N_b(i)
      - tau) / T) -- bounded, per user's own design, with tau/T fixed
      from the N_b distribution's own median/IQR (NOT tuned against
      validation Dice, per user's explicit requirement).

GATE 1 (does preservation recover the deficit at all):
  Delta_D_i = D_i(preservation condition) - D_i(BASE)
  PASS if Delta_D_small > 0 (permutation-significant) for EITHER the
  CONSTANT or NC condition. If neither recovers ANY decodability,
  STOP -- the preservation mechanism itself doesn't work, independent
  of how lambda is set.

GATE 2 (is the gain specifically explained by N_b, beyond difficulty,
and does NC beat CONSTANT):
  (a) rho(N_b, Delta_D_NC) with the same confound/partial-correlation
      methodology as E95/E96.
  (b) Multiple regression: Delta_D_NC ~ N_b + native_size + pred_entropy
      + baseline_dice (all available from E48/E86/E87's existing
      tables) -- does the N_b coefficient remain significant after
      controlling for these difficulty covariates? If beta_1 (N_b's
      coefficient) is not significant once covariates are included,
      the causal-specific claim is NOT supported.
  (c) THE DECISIVE TEST: Delta_D_NC vs Delta_D_CONSTANT, specifically
      for HIGH-N_b subjects (top tercile) -- does NC-conditioning
      outperform constant preservation where it's supposed to matter
      most? If NC and CONSTANT perform equivalently even for high-N_b
      subjects, causal conditioning adds nothing beyond generic
      preservation -- KILL the causal-specific claim (constant
      preservation alone might still be worth reporting separately,
      but is not this project's novel contribution).

PRE-DECLARED DECISION RULE:
  Gate 1 fails (neither CONSTANT nor NC recovers ANY decodability) ->
    KILL the whole preservation-mechanism idea.
  Gate 1 passes, Gate 2(a)/(b) show N_b's effect vanishes after
    difficulty-covariate control, OR Gate 2(c) shows NC does not
    outperform CONSTANT for high-N_b subjects ->
    KILL the CAUSAL-CONDITIONING-specific claim (report honestly; do
    not silently rebrand as "just found preservation helps").
  Both gates pass cleanly -> proceed to a 1-seed training smoke test
    (NOT this script -- a genuinely new training run, deferred).
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
N_PROBE_EPOCHS = 200
PROBE_LR = 3e-3  # matched to E94's tuned, converging LR for a similarly-sized linear layer

CKPT_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs" / "AttnGate_seed0" / "checkpoints" / "best.pth"
E48_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"
E87_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e87" / "E87_feature_table.json"


class PreservationProbe(nn.Module):
    """R_phi (enc3, 128ch @ 16^3 -> 256ch @ 16^3, then avg-pooled 2x2x2
    to 8^3 to align with bottleneck's own grid) + linear readout q
    (256ch @ 8^3 -> 1 logit, upsampled to 64^3), matching E90-E94's
    linear-probe convention. lambda is a PER-SUBJECT SCALAR passed at
    forward time (not a learned parameter), so the same trained R_phi/q
    can be evaluated under different lambda conditions without
    retraining -- but for cleanliness we train separate instances per
    condition below (BASE effectively has R_phi's contribution zeroed
    regardless of what R_phi learns, since lambda=0)."""
    def __init__(self):
        super().__init__()
        self.r_phi = nn.Conv3d(128, 256, kernel_size=1)
        self.readout = nn.Conv3d(256, 1, kernel_size=1)

    def forward(self, enc3, bottleneck, lam):
        r = self.r_phi(enc3)  # (1, 256, 16, 16, 16)
        r_pooled = F.avg_pool3d(r, kernel_size=2, stride=2)  # (1, 256, 8, 8, 8), matches pool3's spatial reduction
        z_b_prime = bottleneck + lam * r_pooled
        logits_8 = self.readout(z_b_prime)
        logits_64 = F.interpolate(logits_8, size=(64, 64, 64), mode="trilinear", align_corners=False)
        return logits_64


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


def train_and_eval(condition_name, lambdas_by_subject, enc3_by_subject, bottleneck_by_subject,
                    target_by_subject, probe_train_ids, probe_test_ids, device):
    torch.manual_seed(SEED)
    probe = PreservationProbe().to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    rng = np.random.default_rng(SEED)

    epoch_losses = []
    for epoch in range(N_PROBE_EPOCHS):
        perm = rng.permutation(len(probe_train_ids))
        epoch_loss = 0.0
        for idx in perm:
            sid = probe_train_ids[idx]
            lam = lambdas_by_subject[sid]
            target_t = torch.from_numpy(target_by_subject[sid]).unsqueeze(0).unsqueeze(0).to(device)
            logits = probe(enc3_by_subject[sid], bottleneck_by_subject[sid], lam)
            loss = F.binary_cross_entropy_with_logits(logits, target_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_losses.append(epoch_loss / len(probe_train_ids))

    tail = epoch_losses[-40:]
    tail_cv = float(np.std(tail) / (np.mean(tail) + 1e-9))
    converged = tail_cv < 0.25
    print(f"  [{condition_name}] final-tail loss cv={tail_cv:.4f} -> {'CONVERGED' if converged else 'NOT CONVERGED'}", flush=True)

    probe.eval()
    results = {}
    with torch.no_grad():
        for sid in probe_test_ids:
            lam = lambdas_by_subject[sid]
            logits = probe(enc3_by_subject[sid], bottleneck_by_subject[sid], lam)
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
            results[sid] = dice_score((probs >= 0.5).astype(np.float32), target_by_subject[sid])
    return results, converged


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}

    with open(E87_TABLE_PATH) as f:
        e87_records = json.load(f)
    e87_by_id = {r["subject_id"]: r for r in e87_records}

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    assert abs(ckpt.get("best_val_dice", 0) - 0.9101624600589275) < 1e-9, \
        "Checkpoint mismatch -- must match E48-E96's exact checkpoint."
    model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded checkpoint, val_dice={ckpt.get('best_val_dice')}", flush=True)

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    print(f"Validation set size: {len(val_dataset)}", flush=True)

    enc3_by_subject, bottleneck_by_subject, target_by_subject = {}, {}, {}
    n_b_by_subject, native_size_by_subject = {}, {}
    dice_intact_by_subject = {}

    for subject_idx in range(len(val_dataset)):
        image, mask, subject_id = val_dataset[subject_idx]
        if subject_id not in e48_by_id or subject_id not in e87_by_id:
            continue
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_dataset.subject_dirs[subject_idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        target_64 = (fractional_occupancy_64(seg_binary_native) > 0.5).astype(np.float32)

        with torch.no_grad():
            enc1 = model.enc1(image_b)
            pool1 = model.pool1(enc1)
            enc2 = model.enc2(pool1)
            pool2 = model.pool2(enc2)
            enc3 = model.enc3(pool2)
            pool3_output = model.pool3(enc3)
            bottleneck = model.bottleneck(pool3_output)
            full_probs = model(image_b)["probs"].squeeze(0).squeeze(0).cpu().numpy()
        dice_intact = dice_score((full_probs >= 0.5).astype(np.float32), target_64)

        enc3_by_subject[subject_id] = enc3.detach()
        bottleneck_by_subject[subject_id] = bottleneck.detach()
        target_by_subject[subject_id] = target_64
        n_b_by_subject[subject_id] = e48_by_id[subject_id]["drop"]
        native_size_by_subject[subject_id] = e48_by_id[subject_id]["native_size"]
        dice_intact_by_subject[subject_id] = dice_intact

        if len(enc3_by_subject) % 25 == 0:
            print(f"  extracted {len(enc3_by_subject)} subjects", flush=True)

    subject_ids = list(enc3_by_subject.keys())
    print(f"\nTotal subjects: {len(subject_ids)}", flush=True)

    sorted_indices = sorted(subject_ids, key=lambda sid: native_size_by_subject[sid])
    test_flags = [(pos % 4 == 0) for pos in range(len(sorted_indices))]
    probe_test_ids = [sorted_indices[pos] for pos, flag in enumerate(test_flags) if flag]
    probe_train_ids = [sorted_indices[pos] for pos, flag in enumerate(test_flags) if not flag]
    print(f"Probe-train: {len(probe_train_ids)}, Probe-test: {len(probe_test_ids)} "
          f"(IDENTICAL split rule to E90-E96)", flush=True)

    median_size = float(np.median([native_size_by_subject[sid] for sid in probe_test_ids]))

    # ================= lambda(N_b) mapping, fixed from N_b's own distribution =================
    n_b_all = np.array([n_b_by_subject[sid] for sid in subject_ids])
    tau = float(np.median(n_b_all))
    T = float(np.percentile(n_b_all, 75) - np.percentile(n_b_all, 25)) / 2 + 1e-6  # half-IQR
    LAMBDA_MAX = 1.0

    def lam_nc(n_b):
        return LAMBDA_MAX / (1.0 + np.exp(-(n_b - tau) / T))

    lambdas_nc = {sid: float(lam_nc(n_b_by_subject[sid])) for sid in subject_ids}
    lambda_const_value = float(np.mean(list(lambdas_nc.values())))  # match NC's mean, per user's fairness requirement
    print(f"\nlambda(N_b) params: tau={tau:.4f}, T={T:.4f}, lambda_max={LAMBDA_MAX}")
    print(f"NC lambda range: [{min(lambdas_nc.values()):.4f}, {max(lambdas_nc.values()):.4f}], "
          f"mean={lambda_const_value:.4f} (used as CONSTANT condition's fixed value)")

    lambdas_base = {sid: 0.0 for sid in subject_ids}
    lambdas_const = {sid: lambda_const_value for sid in subject_ids}

    # ================= Train and evaluate all 3 conditions =================
    condition_results = {}
    for cond_name, lambdas in [("BASE", lambdas_base), ("CONSTANT", lambdas_const), ("NC", lambdas_nc)]:
        print(f"\n=== Training condition: {cond_name} ===", flush=True)
        results, converged = train_and_eval(
            cond_name, lambdas, enc3_by_subject, bottleneck_by_subject, target_by_subject,
            probe_train_ids, probe_test_ids, device
        )
        condition_results[cond_name] = {"results": results, "converged": converged}
        small_decod = np.mean([results[sid] for sid in probe_test_ids if native_size_by_subject[sid] <= median_size])
        large_decod = np.mean([results[sid] for sid in probe_test_ids if native_size_by_subject[sid] > median_size])
        print(f"  {cond_name}: small={small_decod:.4f}, large={large_decod:.4f}")

    with open(OUT_DIR / "E97_condition_results.json", "w") as f:
        json.dump({c: r["results"] for c, r in condition_results.items()}, f, indent=2)

    # ================= Gate 1: does preservation recover ANY decodability? =================
    print(f"\n=== GATE 1: does preservation recover decodability vs BASE? ===")
    base_results = condition_results["BASE"]["results"]

    def delta_vs_base(cond_name):
        return {sid: condition_results[cond_name]["results"][sid] - base_results[sid] for sid in probe_test_ids}

    delta_const = delta_vs_base("CONSTANT")
    delta_nc = delta_vs_base("NC")

    small_ids_test = [sid for sid in probe_test_ids if native_size_by_subject[sid] <= median_size]
    large_ids_test = [sid for sid in probe_test_ids if native_size_by_subject[sid] > median_size]

    delta_const_small = np.array([delta_const[sid] for sid in small_ids_test])
    delta_nc_small = np.array([delta_nc[sid] for sid in small_ids_test])

    def sign_perm_test(x, seed):
        observed = x.mean()
        rng = np.random.default_rng(seed)
        perm_means = np.empty(N_PERM)
        for i in range(N_PERM):
            signs = rng.choice([-1, 1], size=len(x))
            perm_means[i] = (x * signs).mean()
        p = float((perm_means >= observed).mean())  # one-sided, testing >0
        return float(observed), p

    obs_const, p_const = sign_perm_test(delta_const_small, SEED)
    obs_nc, p_nc = sign_perm_test(delta_nc_small, SEED + 1)

    print(f"CONSTANT: mean Delta_D_small = {obs_const:+.4f}, one-sided permutation p={p_const:.4f}")
    print(f"NC:       mean Delta_D_small = {obs_nc:+.4f}, one-sided permutation p={p_nc:.4f}")

    gate1_pass = (obs_const > 0.01 and p_const < 0.05) or (obs_nc > 0.01 and p_nc < 0.05)
    print(f"\nGATE 1: {'PASS' if gate1_pass else 'FAIL'}")

    if not gate1_pass:
        print("\n=== FINAL DECISION: KILL -- preservation mechanism does not recover any decodability. ===")
        summary = {
            "gate1_pass": False, "gate2_evaluated": False,
            "delta_const_small_mean": obs_const, "delta_const_small_p": p_const,
            "delta_nc_small_mean": obs_nc, "delta_nc_small_p": p_nc,
            "final_decision": "KILL_GATE1",
        }
        with open(OUT_DIR / "E97_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return

    # ================= Gate 2 =================
    print(f"\n=== GATE 2: is the gain specifically explained by N_b? ===")

    n_b_test = np.array([n_b_by_subject[sid] for sid in probe_test_ids])
    delta_nc_test = np.array([delta_nc[sid] for sid in probe_test_ids])
    native_size_test = np.array([native_size_by_subject[sid] for sid in probe_test_ids])
    pred_entropy_test = np.array([e87_by_id[sid]["pred_entropy"] for sid in probe_test_ids])
    dice_intact_test = np.array([dice_intact_by_subject[sid] for sid in probe_test_ids])

    rho_nb_deltanc, p_nb_deltanc = stats.spearmanr(n_b_test, delta_nc_test)
    print(f"(a) Spearman(N_b, Delta_D_NC) = {rho_nb_deltanc:+.4f} (p={p_nb_deltanc:.4e})")

    # (b) Multiple regression on RANKS (robust, consistent with this project's Spearman-based convention)
    X = np.column_stack([
        stats.rankdata(n_b_test), stats.rankdata(native_size_test),
        stats.rankdata(pred_entropy_test), stats.rankdata(dice_intact_test),
    ])
    X1 = np.column_stack([np.ones(len(X)), X])
    y = stats.rankdata(delta_nc_test)
    beta, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
    y_hat = X1 @ beta
    resid = y - y_hat
    n, k = X1.shape
    sigma2 = np.sum(resid ** 2) / (n - k)
    cov_beta = sigma2 * np.linalg.inv(X1.T @ X1)
    se_beta = np.sqrt(np.diag(cov_beta))
    t_stats = beta / se_beta
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

    print(f"(b) Regression (rank-based) coefficients [intercept, N_b, size, entropy, dice_intact]:")
    names = ["intercept", "N_b", "native_size", "pred_entropy", "dice_intact"]
    for name, b, p in zip(names, beta, p_vals):
        print(f"    {name:15s}: coef={b:+.4f}, p={p:.4f}")
    nb_beta_significant = p_vals[1] < 0.05

    # (c) NC vs CONSTANT specifically for high-N_b subjects (top tercile)
    tercile_cut = np.percentile(n_b_test, 66.67)
    high_nb_ids = [sid for sid in probe_test_ids if n_b_by_subject[sid] >= tercile_cut]
    delta_nc_high = np.array([delta_nc[sid] for sid in high_nb_ids])
    delta_const_high = np.array([delta_const[sid] for sid in high_nb_ids])
    diff_high = delta_nc_high - delta_const_high

    obs_diff_high, p_diff_high = sign_perm_test(diff_high, SEED + 2)
    print(f"(c) High-N_b subjects (n={len(high_nb_ids)}): "
          f"mean(Delta_D_NC - Delta_D_CONSTANT) = {obs_diff_high:+.4f}, one-sided p={p_diff_high:.4f}")

    nc_beats_constant_for_high_nb = (obs_diff_high > 0.01) and (p_diff_high < 0.05)

    gate2_pass = (abs(rho_nb_deltanc) > 0.3 and p_nb_deltanc < 0.05) and nb_beta_significant and nc_beats_constant_for_high_nb

    print(f"\nGATE 2 sub-results: rho significant={abs(rho_nb_deltanc) > 0.3 and p_nb_deltanc < 0.05}, "
          f"N_b beta significant after covariates={nb_beta_significant}, "
          f"NC beats CONSTANT for high-N_b={nc_beats_constant_for_high_nb}")
    print(f"GATE 2: {'PASS' if gate2_pass else 'FAIL'}")

    final_decision = "PROCEED_TO_1_SEED_SMOKE_TEST" if gate2_pass else "KILL_CAUSAL_CONDITIONING_CLAIM"
    print(f"\n=== FINAL DECISION: {final_decision} ===")

    summary = {
        "gate1_pass": True,
        "delta_const_small_mean": obs_const, "delta_const_small_p": p_const,
        "delta_nc_small_mean": obs_nc, "delta_nc_small_p": p_nc,
        "gate2_evaluated": True,
        "rho_Nb_deltaNC": float(rho_nb_deltanc), "p_Nb_deltaNC": float(p_nb_deltanc),
        "regression_coefficients": dict(zip(names, beta.tolist())),
        "regression_p_values": dict(zip(names, p_vals.tolist())),
        "nb_beta_significant_after_covariates": bool(nb_beta_significant),
        "high_nb_n": len(high_nb_ids),
        "nc_minus_constant_high_nb_mean": obs_diff_high, "nc_minus_constant_high_nb_p": p_diff_high,
        "nc_beats_constant_for_high_nb": bool(nc_beats_constant_for_high_nb),
        "gate2_pass": bool(gate2_pass),
        "final_decision": final_decision,
    }
    with open(OUT_DIR / "E97_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E97_summary.json")


if __name__ == "__main__":
    main()
