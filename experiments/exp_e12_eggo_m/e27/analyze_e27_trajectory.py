"""
Phase E27, Sections 10-12: early predictability of scale preference.

Uses E27_trajectory_table.json (periodic checkpoints: epochs 1,5,10,15,20,25,30
for A/D4/D2/Both) to build EARLY features (from checkpoints <= EARLY_CUTOFF_EPOCH
only) and test whether they predict the FINAL scale preference label
(computed from best.pth, i.e. E27_component_full_table.json's scale_preference
field). No training. Strict no-future-leakage: early features are built
exclusively from checkpoint_epoch <= cutoff records.
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score

BASE = Path(__file__).parent
EARLY_CUTOFF_EPOCH = 10  # fixed BEFORE this analysis was written, matches analyze_e27_scale_preference.py

with open(BASE / "E27_trajectory_table.json") as f:
    traj = json.load(f)
with open(BASE / "E27_component_full_table.json") as f:
    final = json.load(f)
print(f"Loaded {len(traj)} trajectory records, {len(final)} final component records")

# ---- index final labels by (subject_idx, component_id) ----
final_by_key = {}
for r in final:
    key = (r["subject_idx"], r["component_id"])
    final_by_key[key] = r

# ---- restrict trajectory to early epochs only (no leakage) ----
early_traj = [t for t in traj if t["checkpoint_epoch"] <= EARLY_CUTOFF_EPOCH]
print(f"Early-window records (epoch <= {EARLY_CUTOFF_EPOCH}): {len(early_traj)}")

# organize: {(subject_idx, component_id): {condition: [(epoch, dice, coverage, precision, detected), ...]}}
by_component = defaultdict(lambda: defaultdict(list))
for t in early_traj:
    key = (t["subject_idx"], t["component_id"])
    by_component[key][t["condition"]].append(
        (t["checkpoint_epoch"], t["dice"], t["coverage"], t["precision"], t["detected"])
    )
for key in by_component:
    for cond in by_component[key]:
        by_component[key][cond].sort(key=lambda x: x[0])

# ---- build early features per component ----
feature_rows = []
for key, cond_data in by_component.items():
    if key not in final_by_key:
        continue
    final_row = final_by_key[key]
    if final_row["scale_preference"] == "Neutral":
        continue  # Section 12: classification among clearly non-neutral components only

    row = {"subject_idx": key[0], "component_id": key[1], "gt_size": final_row["gt_size"]}
    ok = True
    for cond in ("A", "D4", "D2"):
        pts = cond_data.get(cond, [])
        if len(pts) < 2:
            ok = False
            break
        epochs = np.array([p[0] for p in pts])
        dices = np.array([p[1] for p in pts])
        covs = np.array([p[2] for p in pts])
        detected_flags = np.array([p[4] for p in pts])

        # slope of dice over early epochs (simple linear fit)
        slope = np.polyfit(epochs, dices, 1)[0] if len(epochs) >= 2 else 0.0
        recent_mean_improve = float(np.mean(np.diff(dices))) if len(dices) >= 2 else 0.0
        stagnation = int(np.sum(np.abs(np.diff(dices)) < 1e-3))
        n_regressions = int(np.sum(np.diff(dices) < -1e-3))
        stability = float(np.std(dices))
        last_dice = float(dices[-1])
        last_cov = float(covs[-1])
        detect_rate_early = float(detected_flags.mean())

        row[f"{cond}_slope"] = slope
        row[f"{cond}_recent_mean_improve"] = recent_mean_improve
        row[f"{cond}_stagnation"] = stagnation
        row[f"{cond}_n_regressions"] = n_regressions
        row[f"{cond}_stability"] = stability
        row[f"{cond}_last_dice"] = last_dice
        row[f"{cond}_last_cov"] = last_cov
        row[f"{cond}_detect_rate_early"] = detect_rate_early
    if not ok:
        continue

    # early analogue of S_c: (D4_last_dice - A_last_dice) - (D2_last_dice - A_last_dice)
    row["early_S_dice"] = (row["D4_last_dice"] - row["A_last_dice"]) - (row["D2_last_dice"] - row["A_last_dice"])
    row["early_slope_diff"] = row["D4_slope"] - row["D2_slope"]

    row["label"] = 1 if final_row["scale_preference"] == "D4-preferred" else 0  # 1=D4, 0=D2
    row["final_S_dice"] = final_row["S_dice"]
    feature_rows.append(row)

print(f"\nComponents with valid early features AND non-neutral final label: {len(feature_rows)}")

if len(feature_rows) < 20:
    print("\n*** INSUFFICIENT DATA for a meaningful cross-validated predictability analysis ***")
    print("Reporting this limitation explicitly, per Section 10's instruction not to invent an analysis.")
    result = {"status": "insufficient_data", "n_valid_components": len(feature_rows)}
    with open(BASE / "E27_trajectory_analysis_results.json", "w") as f:
        json.dump(result, f, indent=2)
    raise SystemExit(0)

import pandas as pd
df = pd.DataFrame(feature_rows)
print(df[["label", "early_S_dice", "early_slope_diff"]].describe())

# ---- Spearman: early_S_dice vs final S_dice ----
rho, p = stats.spearmanr(df["early_S_dice"], df["final_S_dice"])
print(f"\nSpearman(early_S_dice, final_S_dice) = {rho:+.3f}, p={p:.4f}")

feature_cols = [
    "A_slope", "D4_slope", "D2_slope", "early_slope_diff",
    "A_recent_mean_improve", "D4_recent_mean_improve", "D2_recent_mean_improve",
    "A_stagnation", "D4_stagnation", "D2_stagnation",
    "A_n_regressions", "D4_n_regressions", "D2_n_regressions",
    "A_stability", "D4_stability", "D2_stability",
    "A_last_dice", "D4_last_dice", "D2_last_dice",
    "A_last_cov", "D4_last_cov", "D2_last_cov",
    "gt_size", "early_S_dice",
]
X = df[feature_cols].values
y = df["label"].values
groups = df["subject_idx"].values

n_pos, n_neg = int(y.sum()), int((1 - y).sum())
print(f"\nClass balance: D4-preferred={n_pos}, D2-preferred={n_neg}")

# ---- cross-validated logistic regression, grouped by subject ----
n_splits = min(5, min(n_pos, n_neg)) if min(n_pos, n_neg) >= 2 else 0
results_cv = {}
if n_splits >= 2:
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    aurocs, bal_accs, f1s = [], [], []
    for train_idx, test_idx in sgkf.split(X, y, groups):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(Xtr_s, ytr)
        proba = clf.predict_proba(Xte_s)[:, 1]
        pred = (proba >= 0.5).astype(int)
        if len(set(yte)) > 1:
            aurocs.append(roc_auc_score(yte, proba))
        bal_accs.append(balanced_accuracy_score(yte, pred))
        f1s.append(f1_score(yte, pred, zero_division=0))
    results_cv = {
        "n_splits": n_splits,
        "auroc_mean": float(np.mean(aurocs)) if aurocs else None,
        "auroc_std": float(np.std(aurocs)) if aurocs else None,
        "balanced_acc_mean": float(np.mean(bal_accs)),
        "balanced_acc_std": float(np.std(bal_accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
    }
    print(f"\nFull-feature logistic regression (grouped {n_splits}-fold CV):")
    print(f"  AUROC: {results_cv['auroc_mean']:.3f} +/- {results_cv['auroc_std']:.3f}" if results_cv["auroc_mean"] else "  AUROC: n/a (single class in some fold)")
    print(f"  Balanced accuracy: {results_cv['balanced_acc_mean']:.3f} +/- {results_cv['balanced_acc_std']:.3f}")
    print(f"  F1: {results_cv['f1_mean']:.3f} +/- {results_cv['f1_std']:.3f}")
else:
    print("\nToo few samples in minority class for cross-validation -- skipping CV logistic regression")

# ---- trivial baselines ----
print("\n--- Trivial baselines (same CV splits conceptually, computed directly) ---")
always_d4_bal_acc = balanced_accuracy_score(y, np.ones_like(y))
always_d2_bal_acc = balanced_accuracy_score(y, np.zeros_like(y))
print(f"  Always-D4 balanced accuracy: {always_d4_bal_acc:.3f}")
print(f"  Always-D2 balanced accuracy: {always_d2_bal_acc:.3f}")

size_only_pred = (df["gt_size"] < df["gt_size"].median()).astype(int)  # arbitrary direction, just a baseline
size_auroc = roc_auc_score(y, df["gt_size"]) if len(set(y)) > 1 else None
print(f"  Size-only AUROC (raw size as score): {size_auroc:.3f}" if size_auroc else "  Size-only AUROC: n/a")

baseline_dice_auroc = roc_auc_score(y, -df["A_last_dice"]) if len(set(y)) > 1 else None
print(f"  Baseline-Dice-only AUROC (A_last_dice, sign chosen post-hoc for best baseline direction -- report both directions): "
      f"{baseline_dice_auroc:.3f}" if baseline_dice_auroc else "n/a")
baseline_dice_auroc_alt = roc_auc_score(y, df["A_last_dice"]) if len(set(y)) > 1 else None
print(f"  Baseline-Dice-only AUROC (other direction): {baseline_dice_auroc_alt:.3f}" if baseline_dice_auroc_alt else "n/a")

trivial_baselines = {
    "always_D4_balanced_acc": always_d4_bal_acc,
    "always_D2_balanced_acc": always_d2_bal_acc,
    "size_only_auroc": size_auroc,
    "baseline_dice_only_auroc": max(baseline_dice_auroc, baseline_dice_auroc_alt) if baseline_dice_auroc else None,
}

result = {
    "status": "ok",
    "n_valid_components": len(feature_rows),
    "n_D4_preferred": n_pos, "n_D2_preferred": n_neg,
    "early_cutoff_epoch": EARLY_CUTOFF_EPOCH,
    "spearman_early_S_vs_final_S": {"rho": rho, "p": p},
    "logistic_regression_cv": results_cv,
    "trivial_baselines": trivial_baselines,
}
with open(BASE / "E27_trajectory_analysis_results.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"\nSaved to {BASE / 'E27_trajectory_analysis_results.json'}")
