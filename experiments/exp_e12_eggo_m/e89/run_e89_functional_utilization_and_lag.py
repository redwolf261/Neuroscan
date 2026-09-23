"""
Phase E89: Functional Utilization + Lagged Temporal Structure.

CONTEXT: E88 found that gradient-norm allocation A_b (=G_b here) is
cross-sectionally dissociated from causal necessity N_b at every
checkpoint (rho~0, stable across training), but the CHANGES in G_b and
N_b are positively correlated (rho(Delta N, Delta G)=0.234, p=8.7e-11).
It ALSO found G_b(t) weakly but significantly predicts WORSE subsequent
Dice improvement (rho=-0.096, p=0.008, POOLED across all checkpoint
pairs). User raised two essential concerns before any mechanism design:

  (1) G_b (gradient L2-norm allocation) may not be a valid measure of
      "functional allocation" at all -- its negative Q3 relationship is
      a warning sign, not just a caveat. Need a SECOND, independent
      allocation measure -- functional utilization U_b, defined via how
      much the bottleneck's own intervention changes the OUTPUT (not
      just how much gradient flows into its parameters) -- to check
      whether G_b and U_b tell the same story or diverge.

  (2) The temporal coupling rho(Delta N, Delta G) > 0 could reflect
      genuine reactive tracking (Delta N -> Delta G) OR a third-variable
      confound (some Z drives both, e.g. simply "the whole model is
      improving fast right now"). A LAGGED analysis in both directions
      is needed to establish which temporal direction the relationship
      actually has, before calling it "the optimizer reacts to
      necessity."

PRIOR-ART CHECK (done before running): CausalX-Net (Frontiers in
Medicine, 2025) performs interventional forward passes DURING training
with backprop through them (p=0.2 stochastic intervention rate,
Algorithm 1) -- this occupies "causal-intervention-in-the-training-loop"
generally. HOWEVER, verified directly (not just abstract-level): its
loss weighting is STATIC (fixed alpha/beta/gamma coefficients), with NO
periodic causal remeasurement driving ADAPTIVE weighting or gradient
scaling. That specific gap -- causal signal measured periodically and
used to dynamically steer allocation -- remains open. This experiment is
still a MEASUREMENT phase, not yet a claim to that gap; establishing
whether the phenomenon is real is a prerequisite either way.

THIS PHASE, at the same 7 checkpoints as E88 (epoch_1/5/10/15/20/25/30,
same 125 subjects, NO NEW TRAINING):

  1. Recomputes N_b(x,t) (E48-style causal ablation drop) and G_b(x,t)
     (E85/E88-style gradient L2-norm allocation) -- reused/re-verified,
     not re-derived differently, so results are directly comparable to
     E88.

  2. NEW: computes U_b(x,t) = functional utilization = the L2 norm of
     the DIFFERENCE between the model's full-precision output and its
     bottleneck-ablated output (i.e. ||probs_intact - probs_ablated||),
     normalized by the total output norm. This measures how much the
     bottleneck's CURRENT STATE actually changes the model's behavior
     for this subject -- a functional-consequence quantity, independent
     of gradient magnitude. (Note: this reuses forward passes ALREADY
     computed for N_b -- no additional forward pass cost.)

  3. Per-checkpoint (not just pooled) Q3-style test: does G_b(t) predict
     subsequent dice_intact gain, checkpoint-pair by checkpoint-pair?
     Does U_b(t) predict it too? Reported separately per transition, not
     collapsed into one pooled number this time.

  4. LAGGED cross-correlation: rho(Delta G_b(t), Delta N_b(t+1)) vs
     rho(Delta N_b(t), Delta G_b(t+1)) -- and the same pair for U_b --
     to establish which temporal direction the coupling actually has.

PRE-DECLARED INTERPRETATION:
  - If U_b and G_b show materially DIFFERENT relationships to N_b (cross-
    sectional or temporal), G_b is confirmed a poor/partial proxy for
    "functional allocation" and any future mechanism should be built on
    U_b-like signals, not raw gradient norm.
  - If Delta_N(t) -> Delta_G/U(t+1) is stronger than the reverse
    direction, that supports "optimizer reacts to necessity" (necessity
    changes lead allocation changes).
  - If Delta_G/U(t) -> Delta_N(t+1) is stronger, that suggests allocation
    changes DRIVE necessity changes (a different, also interesting,
    causal story -- e.g. more gradient reshapes what the bottleneck is
    necessary for).
  - If both lagged directions are comparably weak/null while the
    zero-lag (contemporaneous) correlation is strong, that is most
    consistent with a third-variable confound (Z drives both
    simultaneously, e.g. rapid overall learning early in training) --
    the reactive-tracking story would then NOT be well-supported by this
    data, and Case 1 from E88 should be revised toward "confound-driven
    contemporaneous co-movement" rather than "the optimizer reacts."
  - Per-checkpoint Q3 results are also decisive: if the negative
    G_b-vs-subsequent-gain relationship appears only in the pooled
    average and is inconsistent/non-significant checkpoint-by-checkpoint,
    that pooled result should be treated with much less confidence.
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

CKPT_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e46" / "runs" / "AttnGate_seed0" / "checkpoints"
CHECKPOINT_FILES = ["epoch_1.pth", "epoch_5.pth", "epoch_10.pth", "epoch_15.pth",
                     "epoch_20.pth", "epoch_25.pth", "epoch_30.pth"]


def dice_loss(probs, target_bin):
    probs_flat = probs.reshape(-1)
    target_flat = target_bin.reshape(-1)
    intersection = (probs_flat * target_flat).sum()
    denom = probs_flat.sum() + target_flat.sum()
    return 1.0 - (2.0 * intersection + 1e-6) / (denom + 1e-6)


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


def forward_with_bottleneck_ablation(model, image, ablate, device):
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)

        if ablate:
            bottleneck = torch.zeros_like(bottleneck)

        upconv3 = model.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = model.dec3(cat3)

        upconv2 = model.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = model.dec2(cat2)

        upconv1 = model.upconv1(dec2)

        gate = bottleneck
        skip = enc1
        g = model.attn_gate1.W_g(gate)
        g_up = F.interpolate(g, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = model.attn_gate1.W_x(skip)
        psi = torch.sigmoid(model.attn_gate1.W_psi(F.relu(g_up + x)))

        enc1_gated = enc1 * psi
        cat1 = torch.cat([upconv1, enc1_gated], dim=1)
        dec1 = model.dec1(cat1)

        probs = model.seg_head(dec1)
        return probs.squeeze(0).squeeze(0).cpu().numpy()


def bottleneck_param_names(model):
    return [name for name, _ in model.named_parameters() if name.startswith("bottleneck.")]


def compute_allocation(model, image_b, target_bin_t, bottleneck_names, device):
    model.zero_grad(set_to_none=True)
    out = model(image_b)
    probs = out["probs"]
    loss = dice_loss(probs, target_bin_t)
    loss.backward()

    bottleneck_sq_sum = 0.0
    total_sq_sum = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g_sq = float((p.grad ** 2).sum().item())
        total_sq_sum += g_sq
        if name in bottleneck_names:
            bottleneck_sq_sum += g_sq

    model.zero_grad(set_to_none=True)
    if total_sq_sum <= 0:
        return 0.0
    return (bottleneck_sq_sum ** 0.5) / (total_sq_sum ** 0.5)


def lagged_rho(x_t, y_t1):
    """Spearman rho + permutation p for x at time t predicting y at t+1,
    pooled across all subjects and consecutive checkpoint-pairs."""
    x_t = np.array(x_t)
    y_t1 = np.array(y_t1)
    rho, p_param = stats.spearmanr(x_t, y_t1)
    rng = np.random.default_rng(SEED)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng.permutation(y_t1)
        perm_rhos[i], _ = stats.spearmanr(x_t, perm_y)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())
    return float(rho), float(p_param), p_perm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    print(f"Validation set size: {len(val_dataset)}", flush=True)

    subjects = []
    for subject_idx in range(len(val_dataset)):
        image, mask, subject_id = val_dataset[subject_idx]
        subject_dir = val_dataset.subject_dirs[subject_idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        mask_frac_64 = fractional_occupancy_64(seg_binary_native)
        target_bin = (mask_frac_64 > 0.5).astype(np.float32)
        subjects.append({"subject_id": subject_id, "image": image, "target_bin": target_bin})
    print(f"Pre-loaded {len(subjects)} subjects.", flush=True)

    all_records = {}
    for ckpt_file in CHECKPOINT_FILES:
        ckpt_path = CKPT_DIR / ckpt_file
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = UNet3D_v5(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(ckpt["model_state"])
        bottleneck_names = bottleneck_param_names(model)

        print(f"\nProcessing checkpoint {ckpt_file}...", flush=True)

        records = []
        for s in subjects:
            image_b = s["image"].unsqueeze(0).to(device)
            target_bin = s["target_bin"]
            target_bin_t = torch.from_numpy(target_bin).unsqueeze(0).unsqueeze(0).to(device)

            model.eval()
            probs_intact = forward_with_bottleneck_ablation(model, image_b, ablate=False, device=device)
            probs_ablated = forward_with_bottleneck_ablation(model, image_b, ablate=True, device=device)
            dice_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin)
            dice_ablated = dice_score((probs_ablated >= 0.5).astype(np.float32), target_bin)
            n_b = dice_intact - dice_ablated

            # U_b: functional utilization, reusing the SAME two forward
            # passes already computed for N_b -- no extra forward cost.
            output_diff = np.linalg.norm(probs_intact - probs_ablated)
            output_norm = np.linalg.norm(probs_intact) + 1e-9
            u_b = float(output_diff / output_norm)

            model.train()
            g_b = compute_allocation(model, image_b, target_bin_t, bottleneck_names, device)

            records.append({
                "subject_id": s["subject_id"], "N_b": n_b, "G_b": g_b, "U_b": u_b,
                "dice_intact": dice_intact,
            })

        all_records[ckpt_file] = records
        rho_ng, _ = stats.spearmanr([r["N_b"] for r in records], [r["G_b"] for r in records])
        rho_nu, _ = stats.spearmanr([r["N_b"] for r in records], [r["U_b"] for r in records])
        print(f"  {ckpt_file}: rho(N_b,G_b)={rho_ng:+.4f}, rho(N_b,U_b)={rho_nu:+.4f}")

    with open(OUT_DIR / "E89_temporal_table.json", "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nSaved temporal table.", flush=True)

    by_subject = {}
    for ckpt_file in CHECKPOINT_FILES:
        for r in all_records[ckpt_file]:
            by_subject.setdefault(r["subject_id"], {})[ckpt_file] = r

    # ================= Cross-sectional: N_b vs G_b vs U_b, per checkpoint =================
    print("\n=== Cross-sectional comparison: G_b vs U_b as necessity proxies ===")
    cross_sectional = {}
    for ckpt_file in CHECKPOINT_FILES:
        recs = all_records[ckpt_file]
        rho_ng, p_ng = stats.spearmanr([r["N_b"] for r in recs], [r["G_b"] for r in recs])
        rho_nu, p_nu = stats.spearmanr([r["N_b"] for r in recs], [r["U_b"] for r in recs])
        cross_sectional[ckpt_file] = {
            "rho_N_G": float(rho_ng), "p_N_G": float(p_ng),
            "rho_N_U": float(rho_nu), "p_N_U": float(p_nu),
        }
        print(f"  {ckpt_file}: rho(N,G)={rho_ng:+.4f} (p={p_ng:.3f})  "
              f"rho(N,U)={rho_nu:+.4f} (p={p_nu:.3f})")

    # ================= Per-checkpoint-pair Q3 (NOT pooled) =================
    print("\n=== Per-checkpoint-pair Q3: does G_b(t) / U_b(t) predict subsequent dice gain? ===")
    per_pair_q3 = {}
    for i in range(len(CHECKPOINT_FILES) - 1):
        ck_t, ck_t1 = CHECKPOINT_FILES[i], CHECKPOINT_FILES[i + 1]
        g_t, u_t, gain = [], [], []
        for sid, per_ckpt in by_subject.items():
            if ck_t in per_ckpt and ck_t1 in per_ckpt:
                g_t.append(per_ckpt[ck_t]["G_b"])
                u_t.append(per_ckpt[ck_t]["U_b"])
                gain.append(per_ckpt[ck_t1]["dice_intact"] - per_ckpt[ck_t]["dice_intact"])
        rho_g, p_g = stats.spearmanr(g_t, gain)
        rho_u, p_u = stats.spearmanr(u_t, gain)
        transition = f"{ck_t}->{ck_t1}"
        per_pair_q3[transition] = {
            "rho_G_vs_gain": float(rho_g), "p_G_vs_gain": float(p_g),
            "rho_U_vs_gain": float(rho_u), "p_U_vs_gain": float(p_u),
        }
        print(f"  {transition}: rho(G,gain)={rho_g:+.4f} (p={p_g:.3f})  "
              f"rho(U,gain)={rho_u:+.4f} (p={p_u:.3f})")

    n_significant_negative_G = sum(
        1 for v in per_pair_q3.values() if v["rho_G_vs_gain"] < 0 and v["p_G_vs_gain"] < 0.05
    )
    n_transitions = len(per_pair_q3)
    print(f"\nG_b-vs-gain: significant NEGATIVE in {n_significant_negative_G}/{n_transitions} transitions "
          f"(pooled E88 result was rho=-0.096, p=0.008 -- checking if that holds per-transition or was pooled-only)")

    # ================= Lagged cross-correlation =================
    print("\n=== Lagged temporal structure: does Delta_N lead or lag Delta_G / Delta_U? ===")
    dN_t, dG_t1, dU_t1 = [], [], []   # Delta_N(t) -> (t+1) deltas
    dG_t, dN_t1_a = [], []            # Delta_G(t) -> Delta_N(t+1)
    dU_t, dN_t1_b = [], []            # Delta_U(t) -> Delta_N(t+1)

    for i in range(len(CHECKPOINT_FILES) - 2):
        ck0, ck1, ck2 = CHECKPOINT_FILES[i], CHECKPOINT_FILES[i + 1], CHECKPOINT_FILES[i + 2]
        for sid, per_ckpt in by_subject.items():
            if ck0 in per_ckpt and ck1 in per_ckpt and ck2 in per_ckpt:
                delta_n_01 = per_ckpt[ck1]["N_b"] - per_ckpt[ck0]["N_b"]
                delta_g_01 = per_ckpt[ck1]["G_b"] - per_ckpt[ck0]["G_b"]
                delta_u_01 = per_ckpt[ck1]["U_b"] - per_ckpt[ck0]["U_b"]
                delta_n_12 = per_ckpt[ck2]["N_b"] - per_ckpt[ck1]["N_b"]
                delta_g_12 = per_ckpt[ck2]["G_b"] - per_ckpt[ck1]["G_b"]
                delta_u_12 = per_ckpt[ck2]["U_b"] - per_ckpt[ck1]["U_b"]

                # Direction A: Delta_N(t) predicts Delta_G/U(t+1) -- "necessity change leads"
                dN_t.append(delta_n_01)
                dG_t1.append(delta_g_12)
                dU_t1.append(delta_u_12)

                # Direction B: Delta_G(t) predicts Delta_N(t+1) -- "allocation change leads"
                dG_t.append(delta_g_01)
                dN_t1_a.append(delta_n_12)

                dU_t.append(delta_u_01)
                dN_t1_b.append(delta_n_12)

    rho_NleadsG, p_param_NleadsG, p_perm_NleadsG = lagged_rho(dN_t, dG_t1)
    rho_NleadsU, p_param_NleadsU, p_perm_NleadsU = lagged_rho(dN_t, dU_t1)
    rho_GleadsN, p_param_GleadsN, p_perm_GleadsN = lagged_rho(dG_t, dN_t1_a)
    rho_UleadsN, p_param_UleadsN, p_perm_UleadsN = lagged_rho(dU_t, dN_t1_b)

    print(f"n_triplets = {len(dN_t)}")
    print(f"Delta_N(t) -> Delta_G(t+1) [necessity leads]: rho={rho_NleadsG:+.4f}, "
          f"p_param={p_param_NleadsG:.4f}, p_perm={p_perm_NleadsG:.4f}")
    print(f"Delta_G(t) -> Delta_N(t+1) [allocation leads]: rho={rho_GleadsN:+.4f}, "
          f"p_param={p_param_GleadsN:.4f}, p_perm={p_perm_GleadsN:.4f}")
    print(f"Delta_N(t) -> Delta_U(t+1) [necessity leads, functional]: rho={rho_NleadsU:+.4f}, "
          f"p_param={p_param_NleadsU:.4f}, p_perm={p_perm_NleadsU:.4f}")
    print(f"Delta_U(t) -> Delta_N(t+1) [functional-util leads]: rho={rho_UleadsN:+.4f}, "
          f"p_param={p_param_UleadsN:.4f}, p_perm={p_perm_UleadsN:.4f}")

    # ================= Interpretation =================
    print("\n=== INTERPRETATION ===")
    g_u_diverge = None
    cross_sectional_G_vals = [v["rho_N_G"] for v in cross_sectional.values()]
    cross_sectional_U_vals = [v["rho_N_U"] for v in cross_sectional.values()]
    max_abs_diff = max(abs(g - u) for g, u in zip(cross_sectional_G_vals, cross_sectional_U_vals))
    print(f"Max |rho(N,G) - rho(N,U)| across checkpoints = {max_abs_diff:.4f} "
          f"({'MATERIALLY DIFFERENT -- G_b and U_b tell different stories' if max_abs_diff > 0.15 else 'similar, G_b and U_b broadly agree'})")

    lead_direction = "UNCLEAR"
    if abs(rho_NleadsG) > abs(rho_GleadsN) + 0.05 and p_perm_NleadsG < 0.05:
        lead_direction = "NECESSITY_LEADS_ALLOCATION"
    elif abs(rho_GleadsN) > abs(rho_NleadsG) + 0.05 and p_perm_GleadsN < 0.05:
        lead_direction = "ALLOCATION_LEADS_NECESSITY"
    elif p_perm_NleadsG >= 0.05 and p_perm_GleadsN >= 0.05:
        lead_direction = "NEITHER_LAG_SIGNIFICANT_LIKELY_CONFOUND"

    print(f"Lag-direction verdict: {lead_direction}")

    summary = {
        "cross_sectional_per_checkpoint": cross_sectional,
        "per_transition_q3": per_pair_q3,
        "n_transitions_significant_negative_G": n_significant_negative_G,
        "n_transitions_total": n_transitions,
        "max_abs_diff_rho_N_G_vs_rho_N_U": float(max_abs_diff),
        "lagged": {
            "n_triplets": len(dN_t),
            "necessity_leads_allocation_G": {"rho": rho_NleadsG, "p_param": p_param_NleadsG, "p_perm": p_perm_NleadsG},
            "allocation_leads_necessity_G": {"rho": rho_GleadsN, "p_param": p_param_GleadsN, "p_perm": p_perm_GleadsN},
            "necessity_leads_utilization_U": {"rho": rho_NleadsU, "p_param": p_param_NleadsU, "p_perm": p_perm_NleadsU},
            "utilization_leads_necessity_U": {"rho": rho_UleadsN, "p_param": p_param_UleadsN, "p_perm": p_perm_UleadsN},
        },
        "lead_direction_verdict": lead_direction,
    }
    with open(OUT_DIR / "E89_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E89_summary.json")


if __name__ == "__main__":
    main()
