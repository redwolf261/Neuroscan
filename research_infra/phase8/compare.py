"""
Phase 8 Step 4/5: statistical comparison + publication figures across
baseline_metrics.csv, sampling_metrics.csv, optimization_metrics.csv.

Produces:
  research_infra/comparison_table.csv
  research_infra/phase8_figures/{training_loss_comparison,validation_dice_comparison,
      gradient_norm_comparison,training_time_bar,memory_usage_bar,performance_radar}.png

All comparisons are epoch-index-paired (same epoch number across the three runs,
same shared MAE-pretrained encoder start, same deterministic data split) - NOT
repeated-seed replicates. This is stated explicitly in the output and in
PHASE_8_PROJECT_SELECTION.md; it is a real limitation, not hidden here.
"""
import os
import csv

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESEARCH_INFRA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(RESEARCH_INFRA, "phase8_figures")
os.makedirs(FIG_DIR, exist_ok=True)

RUNS = {
    'baseline': os.path.join(RESEARCH_INFRA, 'baseline_metrics.csv'),
    'sampling': os.path.join(RESEARCH_INFRA, 'sampling_metrics.csv'),
    'optimization': os.path.join(RESEARCH_INFRA, 'optimization_metrics.csv'),
}

dfs = {name: pd.read_csv(path) for name, path in RUNS.items()}
for name, df in dfs.items():
    print(f"[compare] {name}: {len(df)} epochs")

LAST_N = 5  # "converged" snapshot window - last 5 of 25 epochs


def last_n_mean(df, col, n=LAST_N):
    return df[col].tail(n).mean()


def paired_test(colname, a_name, b_name):
    """Paired t-test across matched epoch indices (see module docstring caveat)."""
    a = dfs[a_name][colname].values
    b = dfs[b_name][colname].values
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if np.allclose(a, b):
        return 0.0, 1.0
    try:
        t_stat, p_val = stats.ttest_rel(a, b)
        return float(t_stat), float(p_val)
    except Exception as e:
        print(f"[compare] paired test failed for {colname} ({a_name} vs {b_name}): {e}")
        return float('nan'), float('nan')


def convergence_rate(df):
    """Simple, explicit operationalization: (val_dice[last] - val_dice[first]) / n_epochs.
    Positive = improving, negative = degrading, near-zero = flat/stalled."""
    vals = df['val_dice'].values
    return (vals[-1] - vals[0]) / len(vals) if len(vals) > 1 else float('nan')


# ---------------------------------------------------------------------------
# comparison_table.csv
# ---------------------------------------------------------------------------
metrics_spec = [
    ('Dice (val, last-5-epoch mean)', 'val_dice'),
    ('IoU (val, last-5-epoch mean)', 'val_iou'),
    ('Precision (val, last-5-epoch mean)', 'val_precision'),
    ('Recall (val, last-5-epoch mean)', 'val_recall'),
    ('F1 (val, last-5-epoch mean)', 'val_f1'),
    ('Training Time (sec/epoch, mean excl. first warmup epoch)', 'epoch_time_sec'),
    ('GPU Memory (MB peak, mean excl. first warmup epoch)', 'gpu_memory_mb_peak'),
    ('Gradient Norm Variance (std of per-epoch grad_norm_std)', 'grad_norm_std'),
    ('Validation Dice (best single epoch)', 'val_dice'),
]

rows = []
for label, col in metrics_spec:
    if label.startswith('Training Time') or label.startswith('GPU Memory'):
        # exclude epoch 1 (CUDA/cudnn warmup dominates timing, not representative)
        vals = {name: df[col].iloc[1:].mean() for name, df in dfs.items()}
    elif label.startswith('Gradient Norm Variance'):
        vals = {name: df['grad_norm_std'].std() for name, df in dfs.items()}
    elif label.endswith('(best single epoch)'):
        vals = {name: df[col].max() for name, df in dfs.items()}
    else:
        vals = {name: last_n_mean(df, col) for name, df in dfs.items()}

    t_samp, p_samp = paired_test(col, 'baseline', 'sampling') if col in dfs['baseline'].columns else (float('nan'), float('nan'))
    t_opt, p_opt = paired_test(col, 'baseline', 'optimization') if col in dfs['baseline'].columns else (float('nan'), float('nan'))

    rows.append({
        'metric': label,
        'baseline': vals['baseline'],
        'sampling_prototype': vals['sampling'],
        'optimization_prototype': vals['optimization'],
        'sampling_vs_baseline_pvalue': p_samp,
        'optimization_vs_baseline_pvalue': p_opt,
    })

# Convergence speed (not a per-column mean, computed specially)
conv = {name: convergence_rate(df) for name, df in dfs.items()}
rows.append({
    'metric': 'Convergence Speed (delta_val_dice per epoch, first-to-last)',
    'baseline': conv['baseline'],
    'sampling_prototype': conv['sampling'],
    'optimization_prototype': conv['optimization'],
    'sampling_vs_baseline_pvalue': '',
    'optimization_vs_baseline_pvalue': '',
})

comparison_path = os.path.join(RESEARCH_INFRA, 'comparison_table.csv')
with open(comparison_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'metric', 'baseline', 'sampling_prototype', 'optimization_prototype',
        'sampling_vs_baseline_pvalue', 'optimization_vs_baseline_pvalue',
    ])
    writer.writeheader()
    writer.writerows(rows)
print(f"[compare] wrote {comparison_path}")
for r in rows:
    print(f"  {r['metric']:55s} base={r['baseline']:.5f}  samp={r['sampling_prototype']:.5f}  "
          f"opt={r['optimization_prototype']:.5f}  p(samp)={r['sampling_vs_baseline_pvalue']}  "
          f"p(opt)={r['optimization_vs_baseline_pvalue']}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
COLORS = {'baseline': 'tab:blue', 'sampling': 'tab:orange', 'optimization': 'tab:green'}
LABELS = {'baseline': 'Baseline', 'sampling': 'Sampling Prototype (uniform slices)',
          'optimization': 'Optimization Prototype (evidential-head grad scaling)'}


def fig_training_loss_comparison():
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, df in dfs.items():
        ax.plot(df['epoch'], df['train_loss'], label=LABELS[name], color=COLORS[name], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (Hybrid + Evidential)")
    ax.set_title("Training Loss Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "training_loss_comparison.png"), dpi=200)
    plt.close(fig)


def fig_validation_dice_comparison():
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, df in dfs.items():
        ax.plot(df['epoch'], df['val_dice'], label=LABELS[name], color=COLORS[name],
                linewidth=1.5, marker='o', markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Dice")
    ax.set_title("Validation Dice Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "validation_dice_comparison.png"), dpi=200)
    plt.close(fig)


def fig_gradient_norm_comparison():
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, df in dfs.items():
        ax.plot(df['epoch'], df['grad_norm_mean'], label=LABELS[name], color=COLORS[name], linewidth=1.5)
        ax.fill_between(df['epoch'], df['grad_norm_mean'] - df['grad_norm_std'],
                         df['grad_norm_mean'] + df['grad_norm_std'], color=COLORS[name], alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean total gradient norm per epoch (±1 std band)")
    ax.set_title("Gradient Norm Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "gradient_norm_comparison.png"), dpi=200)
    plt.close(fig)


def fig_training_time_bar():
    fig, ax = plt.subplots(figsize=(7, 6))
    means = [dfs[n]['epoch_time_sec'].iloc[1:].mean() for n in ['baseline', 'sampling', 'optimization']]
    stds = [dfs[n]['epoch_time_sec'].iloc[1:].std() for n in ['baseline', 'sampling', 'optimization']]
    names = ['Baseline', 'Sampling', 'Optimization']
    ax.bar(names, means, yerr=stds, capsize=6,
           color=[COLORS['baseline'], COLORS['sampling'], COLORS['optimization']])
    ax.set_ylabel("Mean training time per epoch (sec, excl. warmup epoch 1)")
    ax.set_title("Training Time Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "training_time_bar.png"), dpi=200)
    plt.close(fig)


def fig_memory_usage_bar():
    fig, ax = plt.subplots(figsize=(7, 6))
    means = [dfs[n]['gpu_memory_mb_peak'].iloc[1:].mean() for n in ['baseline', 'sampling', 'optimization']]
    stds = [dfs[n]['gpu_memory_mb_peak'].iloc[1:].std() for n in ['baseline', 'sampling', 'optimization']]
    names = ['Baseline', 'Sampling', 'Optimization']
    ax.bar(names, means, yerr=stds, capsize=6,
           color=[COLORS['baseline'], COLORS['sampling'], COLORS['optimization']])
    ax.set_ylabel("Mean peak GPU memory per epoch (MB, excl. warmup epoch 1)")
    ax.set_title("GPU Memory Usage Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "memory_usage_bar.png"), dpi=200)
    plt.close(fig)


def fig_performance_radar():
    # Dice/IoU/Precision/Recall/F1 already share a natural [0,1] scale - plot raw values
    # directly rather than per-metric-normalizing, which would visually distort metrics
    # (e.g. Recall) where all three runs happen to sit close together near 1.0.
    radar_metrics = ['val_dice', 'val_iou', 'val_precision', 'val_recall', 'val_f1']
    radar_labels = ['Dice', 'IoU', 'Precision', 'Recall', 'F1']

    raw = {name: [last_n_mean(dfs[name], m) for m in radar_metrics] for name in dfs}

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 8), subplot_kw=dict(polar=True))
    for name in ['baseline', 'sampling', 'optimization']:
        vals = raw[name] + raw[name][:1]
        ax.plot(angles, vals, label=LABELS[name], color=COLORS[name], linewidth=1.8)
        ax.fill(angles, vals, color=COLORS[name], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels)
    ax.set_ylim(0, 1.0)
    ax.set_title("Performance Radar (raw values, last-5-epoch mean; shared [0,1] scale)", pad=40)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "performance_radar.png"), dpi=200)
    plt.close(fig)


fig_training_loss_comparison()
fig_validation_dice_comparison()
fig_gradient_norm_comparison()
fig_training_time_bar()
fig_memory_usage_bar()
fig_performance_radar()

print(f"\n[compare] All figures written to {FIG_DIR}")
