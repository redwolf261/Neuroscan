"""
Phase E129: PHASE-2 CAUSAL COMPARISON -- does extra capacity change the
causal vulnerability itself, or only (marginally) the Dice?

CONTEXT: E128 killed capacity as a Track A lineage on Dice grounds
(v12 +0.139pp over an AMP-matched v5 control; gate was +0.5pp). But the
plan correctly separates two different questions:
    (1) does Dice improve?                      -- answered: essentially no
    (2) does the CAUSAL VULNERABILITY change?   -- THIS PHASE
Outcome B (Dice flat AND vulnerability unchanged) strengthens the case for
a mechanism-level intervention. Outcome C (vulnerability changes anyway)
would mean capacity moved the mechanism without moving the score -- a
different and more interesting story that would need its own follow-up.

CRITICAL METHODOLOGICAL POINT: the entire E48-E127 causal chain was
measured on the fp32 canonical checkpoint. BOTH E128 arms are AMP, and
E128 measured that AMP alone is worth +0.110pp of Dice. Comparing v12-AMP
against the HISTORICAL fp32 causal numbers would let precision contaminate
the causal comparison exactly as it nearly contaminated the Dice one. This
phase therefore compares
    v12-AMP   vs   v5-AMP-control
both from E128 -- same precision, protocol, seed, epochs -- so encoder
width is the only difference. Historical fp32 numbers are printed for
reference ONLY and are never used as a control.

MEASUREMENTS (all reuse this project's own established constructions):
  N_1, N_2, N_3 -- per-stage causal necessity via ablation, IDENTICAL to
    E121's CORRECTED construction: zero that stage's OWN OUTPUT tensor and
    let everything downstream recompute for real. For stage 3 this means
    zeroing the BOTTLENECK tensor (post model.bottleneck conv), NOT pool3's
    raw output -- E121 found the latter is an out-of-distribution
    intervention that inflated N_3 to 0.88 and flipped its size
    correlation. That bug is NOT repeated here.
  effective rank -- IDENTICAL to E124/E126: participation ratio
    PR = (sum eig)^2 / sum(eig^2) of each 2x2x2 pool3 window's 8x8
    sub-voxel Gram matrix in channel space, normalized by 8, averaged over
    windows. Chunked eigendecomposition (MAX_EIGH_BATCH=4096) because
    cusolver's batched syevBatched crashes on a 32768-window batch -- a
    real failure found in E124, not a hypothetical.
  lesion-size dependence -- Spearman(N_k, native_size) plus small/large
    median-split stratification with a permutation test (E121 convention).
  Dice-error dependence -- Spearman(N_k, 1 - dice_intact).

NOTE ON enc3 WIDTH (measured, not assumed): v5's enc3 is 128ch, v12's is
256ch. PR is computed from the 8x8 Gram of the eight sub-voxel vectors,
whose nonzero spectrum has rank <= 8 regardless of channel count, so PR is
bounded to [1,8] for BOTH arms -- same scale, hence comparable at all.
HOWEVER, PR is NOT width-neutral in EXPECTATION: on i.i.d. random data,
measured before this run, 128ch gives PR mean 7.489 and 256ch gives 7.739.
More channels make 8 random vectors likelier to look mutually independent.
=> A raw eff_rank DIFFERENCE between arms is therefore partly a width
artifact and must NOT be read as "v12 preserves more information". The
width-robust quantity is the WITHIN-ARM coupling rho(eff_rank, N_3),
computed across subjects inside a single arm, which no constant width
offset can affect. The raw means are reported for completeness with this
caveat attached, and the verdict logic keys on the coupling and the
small-vs-large gap, never on the raw eff_rank delta.

WHAT WOULD CHANGE THE CONCLUSION: if v12's N_3 small-vs-large gap or its
effective-rank/N_3 coupling is materially WEAKER than v5's, capacity
partially fixed the mechanism despite not moving Dice (outcome C). If both
are statistically indistinguishable, outcome B holds.

NO TRAINING. NO ARCHITECTURE CHANGE. Read-only on two frozen checkpoints.
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
from neuroscan_3d_v12 import UNet3D_v12  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000
MAX_EIGH_BATCH = 4096

E128_RUNS = project_root / "experiments" / "exp_e12_eggo_m" / "e128" / "runs"
ARMS = {
    "v5_amp_control": ("v5", E128_RUNS / "Control_v5amp_seed0" / "checkpoints" / "best.pth"),
    "v12_capacity": ("v12", E128_RUNS / "Capacity_v12_seed0" / "checkpoints" / "best.pth"),
}

# Historical fp32 reference values -- PRINTED ONLY, never used as a control.
FP32_REF = {
    "best_val_dice": 0.9101624600589275,
    "N_1_mean": 0.6525, "N_2_mean": 0.6143, "N_3_mean": 0.2717,
    "eff_rank_mean": 0.1912,
}

# Measured PR on i.i.d. random data (see docstring): width artifact baseline.
RANDOM_PR_BASELINE = {"128ch": 7.489 / 8.0, "256ch": 7.739 / 8.0}


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    den = pred_bin.sum() + target_bin.sum()
    return 1.0 if den == 0 else float(2 * tp / den)


def frac_occ(seg, shape):
    t = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=shape, mode="area").squeeze().numpy()


def forward_stage_ablation(model, image, stage, device):
    """stage in {None, 'pool1', 'pool2', 'bottleneck'}. IDENTICAL to E121's
    corrected construction -- see module docstring."""
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1)
        if stage == "pool1":
            pool1 = torch.zeros_like(pool1)

        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        if stage == "pool2":
            pool2 = torch.zeros_like(pool2)

        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)
        if stage == "bottleneck":
            bottleneck = torch.zeros_like(bottleneck)

        up3 = model.upconv3(bottleneck)
        dec3 = model.dec3(torch.cat([up3, enc3], dim=1))
        up2 = model.upconv2(dec3)
        dec2 = model.dec2(torch.cat([up2, enc2], dim=1))
        up1 = model.upconv1(dec2)

        g = model.attn_gate1.W_g(bottleneck)
        g_up = F.interpolate(g, size=enc1.shape[2:], mode="trilinear", align_corners=False)
        x = model.attn_gate1.W_x(enc1)
        psi = torch.sigmoid(model.attn_gate1.W_psi(F.relu(g_up + x)))

        dec1 = model.dec1(torch.cat([up1, enc1 * psi], dim=1))
        probs = model.seg_head(dec1)
    return probs.squeeze(0).squeeze(0).cpu().numpy(), enc3.squeeze(0)


def windows_from_tensor(t):
    C = t.shape[0]
    x = t.unfold(1, 2, 2).unfold(2, 2, 2).unfold(3, 2, 2)
    nW = (t.shape[1] // 2) * (t.shape[2] // 2) * (t.shape[3] // 2)
    return x.contiguous().view(C, nW, 8)


def participation_ratio(windows):
    v = windows.permute(1, 2, 0)
    out = []
    for s in range(0, v.shape[0], MAX_EIGH_BATCH):
        vc = v[s:s + MAX_EIGH_BATCH]
        gram = torch.bmm(vc, vc.transpose(1, 2))
        ev = torch.linalg.eigvalsh(gram).clamp(min=0)
        s1 = ev.sum(1)
        s2 = (ev ** 2).sum(1)
        out.append(torch.where(s2 > 1e-12, s1 ** 2 / (s2 + 1e-12), torch.ones_like(s1)).cpu().numpy())
    return np.concatenate(out)


def perm_corr(x, y, seed):
    rho, _ = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    pr = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(N_PERM)])
    return float(rho), float((np.abs(pr) >= abs(rho)).mean())


def perm_diff(a, b, seed):
    obs = a.mean() - b.mean()
    comb = np.concatenate([a, b])
    n = len(a)
    rng = np.random.default_rng(seed)
    pd = np.empty(N_PERM)
    for i in range(N_PERM):
        p = rng.permutation(len(comb))
        pd[i] = comb[p[:n]].mean() - comb[p[n:]].mean()
    return float(obs), float((np.abs(pd) >= abs(obs)).mean())


def partial_corr(x, y, ctrl, seed):
    X = np.column_stack([np.ones(len(ctrl)), ctrl])
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    rho, _ = stats.spearmanr(rx, ry)
    rng = np.random.default_rng(seed)
    pr = np.array([stats.spearmanr(rx, rng.permutation(ry))[0] for _ in range(N_PERM)])
    return float(rho), float((np.abs(pr) >= abs(rho)).mean())


def measure(arm_name, arch, ckpt_path, val_ds, device):
    print("\n" + "=" * 68)
    print("ARM: {}  (arch={})".format(arm_name, arch))
    print("=" * 68, flush=True)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    assert ckpt.get("arch") == arch, "checkpoint arch mismatch: {} != {}".format(ckpt.get("arch"), arch)
    model = (UNet3D_v5 if arch == "v5" else UNet3D_v12)(1, 1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("loaded best_val_dice={:.6f} epoch={}".format(ckpt["best_val_dice"], ckpt["epoch"]))

    with torch.no_grad():
        img0, _, _ = val_ds[0]
        ib = img0.unsqueeze(0).to(device)
        ro = model(ib)
        rp = ro["probs"] if isinstance(ro, dict) else ro
        mp, _ = forward_stage_ablation(model, ib, None, device)
        md = float(np.abs(rp.squeeze(0).squeeze(0).cpu().numpy() - mp).max())
    assert md < 1e-4, "manual trunk mismatch {}".format(md)
    print("[sanity] manual trunk vs forward(): {:.2e} PASS".format(md))

    recs = []
    for idx in range(len(val_ds)):
        image, _, sid = val_ds[idx]
        ib = image.unsqueeze(0).to(device)
        sd = Path(val_ds.subject_dirs[idx]) / "{}-seg.nii.gz".format(sid)
        seg = (nib.load(str(sd)).get_fdata().astype(np.float32) > 0).astype(np.float32)
        size = int(seg.sum())
        tgt = (frac_occ(seg, (64, 64, 64)) > 0.5).astype(np.float32)

        p_i, enc3 = forward_stage_ablation(model, ib, None, device)
        d_i = dice_score((p_i >= 0.5).astype(np.float32), tgt)
        r = {"subject_id": sid, "native_size": size, "dice_intact": d_i}
        for k, st in [("N_1", "pool1"), ("N_2", "pool2"), ("N_3", "bottleneck")]:
            p_a, _ = forward_stage_ablation(model, ib, st, device)
            r[k] = d_i - dice_score((p_a >= 0.5).astype(np.float32), tgt)
        r["eff_rank"] = float((participation_ratio(windows_from_tensor(enc3)) / 8.0).mean())
        recs.append(r)
        if (idx + 1) % 25 == 0:
            print("  {}/{}".format(idx + 1, len(val_ds)), flush=True)

    size = np.array([r["native_size"] for r in recs])
    derr = 1.0 - np.array([r["dice_intact"] for r in recs])
    er = np.array([r["eff_rank"] for r in recs])
    med = float(np.median(size))
    sm = size <= med

    res = {"arm": arm_name, "arch": arch, "best_val_dice": float(ckpt["best_val_dice"]),
           "n": len(recs), "eff_rank_mean": float(er.mean()), "eff_rank_std": float(er.std()),
           "stages": {}}

    print("\neffective rank: {:.4f} +/- {:.4f}".format(er.mean(), er.std()))
    print("{:6} {:>8} {:>8} {:>8} {:>8} {:>7}  {:>9} {:>7}  {:>8} {:>7}".format(
        "stage", "mean", "small", "large", "s-l", "p", "rho_size", "p", "rho_err", "p"))
    for k in ["N_1", "N_2", "N_3"]:
        v = np.array([r[k] for r in recs])
        d, pd_ = perm_diff(v[sm], v[~sm], SEED)
        rs, ps = perm_corr(v, size, SEED + 1)
        re_, pe = perm_corr(v, derr, SEED + 2)
        print("{:6} {:8.4f} {:8.4f} {:8.4f} {:+8.4f} {:7.4f}  {:+9.4f} {:7.4f}  {:+8.4f} {:7.4f}".format(
            k, v.mean(), v[sm].mean(), v[~sm].mean(), d, pd_, rs, ps, re_, pe))
        res["stages"][k] = {
            "mean": float(v.mean()), "std": float(v.std()),
            "small_mean": float(v[sm].mean()), "large_mean": float(v[~sm].mean()),
            "small_minus_large": d, "small_large_perm_p": pd_,
            "rho_size": rs, "rho_size_perm_p": ps,
            "rho_dice_err": re_, "rho_dice_err_perm_p": pe,
        }

    n3 = np.array([r["N_3"] for r in recs])
    rho_m, p_m = perm_corr(er, n3, SEED + 3)
    rho_p, p_p = partial_corr(er, n3, np.log(size + 1), SEED + 4)
    print("\nE126 coupling  Spearman(eff_rank, N_3) = {:+.4f} (p={:.4f})".format(rho_m, p_m))
    print("               partial | log(size)      = {:+.4f} (p={:.4f})".format(rho_p, p_p))
    res["eff_rank_vs_N3"] = {"marginal_rho": rho_m, "marginal_p": p_m,
                             "partial_rho_given_size": rho_p, "partial_p": p_p}

    with open(OUT_DIR / "E129_{}_table.json".format(arm_name), "w") as f:
        json.dump(recs, f, indent=2)
    return res, recs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    val_ds = BraTSDataset(root_dir=str(project_root / "Dataset" / "Training"), split="val",
                          val_split=0.1, target_shape=(64, 64, 64), normalize=True)

    out, tables = {}, {}
    for arm, (arch, ck) in ARMS.items():
        out[arm], tables[arm] = measure(arm, arch, ck, val_ds, device)

    a, b = out["v5_amp_control"], out["v12_capacity"]
    print("\n\n" + "=" * 68)
    print("PHASE-2 COMPARISON  (v12 vs AMP-matched v5 control)")
    print("=" * 68)
    print("{:34} {:>10} {:>10} {:>11}".format("quantity", "v5_ctrl", "v12", "delta"))
    print("{:34} {:10.5f} {:10.5f} {:+10.3f}pp".format(
        "best_val_dice", a["best_val_dice"], b["best_val_dice"],
        (b["best_val_dice"] - a["best_val_dice"]) * 100))
    print("{:34} {:10.4f} {:10.4f} {:+11.4f}".format(
        "effective rank (mean)", a["eff_rank_mean"], b["eff_rank_mean"],
        b["eff_rank_mean"] - a["eff_rank_mean"]))
    print("{:34} {:10.4f} {:10.4f} {:+11.4f}   <- width artifact, NOT a result".format(
        "  [random-data PR baseline]", RANDOM_PR_BASELINE["128ch"], RANDOM_PR_BASELINE["256ch"],
        RANDOM_PR_BASELINE["256ch"] - RANDOM_PR_BASELINE["128ch"]))
    for k in ["N_1", "N_2", "N_3"]:
        for f_, lab in [("mean", k + " mean"), ("small_minus_large", k + " small-large gap")]:
            print("{:34} {:10.4f} {:10.4f} {:+11.4f}".format(
                lab, a["stages"][k][f_], b["stages"][k][f_],
                b["stages"][k][f_] - a["stages"][k][f_]))
    print("{:34} {:10.4f} {:10.4f} {:+11.4f}".format(
        "rho(eff_rank,N_3) partial|size",
        a["eff_rank_vs_N3"]["partial_rho_given_size"],
        b["eff_rank_vs_N3"]["partial_rho_given_size"],
        b["eff_rank_vs_N3"]["partial_rho_given_size"] - a["eff_rank_vs_N3"]["partial_rho_given_size"]))

    n3a = np.array([r["N_3"] for r in tables["v5_amp_control"]])
    n3b = np.array([r["N_3"] for r in tables["v12_capacity"]])
    t_p = float(stats.ttest_rel(n3b, n3a).pvalue)
    w_p = float(stats.wilcoxon(n3b, n3a).pvalue)
    print("\npaired N_3 shift (v12 - v5): {:+.4f}  t p={:.3e}  wilcoxon p={:.3e}".format(
        (n3b - n3a).mean(), t_p, w_p))

    print("\n[reference only, NOT a control] fp32 canonical: dice={:.5f} N_3={:.4f} eff_rank={:.4f}".format(
        FP32_REF["best_val_dice"], FP32_REF["N_3_mean"], FP32_REF["eff_rank_mean"]))

    gap_a = a["stages"]["N_3"]["small_minus_large"]
    gap_b = b["stages"]["N_3"]["small_minus_large"]
    rho_a = a["eff_rank_vs_N3"]["partial_rho_given_size"]
    rho_b = b["eff_rank_vs_N3"]["partial_rho_given_size"]
    vuln_intact = (gap_b > 0 and b["stages"]["N_3"]["small_large_perm_p"] < 0.05) and (abs(rho_b) > 0.5)

    print("\n=== VERDICT ===")
    if vuln_intact and abs(gap_b - gap_a) < 0.05 and abs(rho_b - rho_a) < 0.2:
        verdict = "OUTCOME_B_VULNERABILITY_UNCHANGED"
        detail = ("Dice essentially flat AND the causal vulnerability is intact and "
                  "statistically indistinguishable from the control: the N_3 small-lesion gap "
                  "and the effective-rank/N_3 coupling both survive 3x capacity. The deficit is "
                  "NOT a capacity problem -- this strengthens the case for a mechanism-level "
                  "intervention and supports freezing the architecture/capacity axis.")
    elif not vuln_intact:
        verdict = "OUTCOME_C_VULNERABILITY_ALTERED"
        detail = ("The causal vulnerability is materially weaker in v12 even though Dice barely "
                  "moved -- capacity changed the mechanism without changing the score. This is a "
                  "DIFFERENT and more interesting result than E128's Dice null alone; it needs "
                  "its own follow-up before the architecture axis is frozen.")
    else:
        verdict = "MIXED_PARTIAL_SHIFT"
        detail = ("Vulnerability is intact by the pre-declared thresholds but one or more "
                  "signatures shifted non-trivially. Report the raw numbers; do not force a "
                  "clean B-vs-C label.")
    print(verdict)
    print(detail)

    out["comparison"] = {
        "paired_N3_shift": float((n3b - n3a).mean()), "paired_t_p": t_p, "wilcoxon_p": w_p,
        "verdict": verdict, "detail": detail,
        "fp32_reference_NOT_A_CONTROL": FP32_REF,
        "random_pr_width_baseline": RANDOM_PR_BASELINE,
    }
    with open(OUT_DIR / "E129_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved E129_summary.json + per-arm tables")


if __name__ == "__main__":
    main()
