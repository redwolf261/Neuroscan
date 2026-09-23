"""
Phase 5: Visualization of diagnostic CSVs produced by the loggers in this package.

This script only plots data that already exists in the CSVs - it adds no
interpretation, no thresholds, no pass/fail judgements. Each function reads
one CSV and produces one or more PNG figures. A CSV that doesn't exist for a
given run (because its diagnostic flag was off) is silently skipped.

Usage:
    python visualize.py --log-dir <SEGMENTATION_DIR> --out-dir <where to save PNGs>

Requires: pandas, matplotlib (already project dependencies).
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _read_csv(log_dir, filename):
    path = os.path.join(log_dir, filename)
    if not os.path.exists(path):
        print(f"[visualize] skip {filename} (not found in {log_dir})")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"[visualize] skip {filename} (empty)")
        return None
    df['global_step'] = range(len(df))
    return df


def plot_training_curves(log_dir, out_dir):
    """training_metrics.csv -> loss_evolution.png, dice_metric.png"""
    df = _read_csv(log_dir, "training_metrics.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for col, label in [
        ('loss_dice', 'Dice'), ('loss_focal_tversky', 'FocalTversky'),
        ('loss_hybrid', 'Hybrid (0.5*Dice+0.5*FT)'),
        ('loss_evidential', 'Evidential'), ('loss_total', 'Total'),
    ]:
        if col in df.columns:
            ax.plot(df['global_step'], df[col], label=label, linewidth=1.2)
    ax.set_xlabel("batch (global step)")
    ax.set_ylabel("loss value")
    ax.set_title("Loss component evolution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_evolution.png"), dpi=150)
    plt.close(fig)

    if 'dice_metric' in df.columns and df['dice_metric'].notna().any():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['global_step'], df['dice_metric'], color='tab:green', linewidth=1.2)
        ax.set_xlabel("batch (global step)")
        ax.set_ylabel("Dice metric (thresholded)")
        ax.set_title("Segmentation Dice metric over training")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "dice_metric.png"), dpi=150)
        plt.close(fig)

    print(f"[visualize] wrote loss_evolution.png (+ dice_metric.png) to {out_dir}")


def plot_gradient_norms(log_dir, out_dir):
    """gradient_metrics.csv -> gradient_norms.png, gradient_loss_attribution.png"""
    df = _read_csv(log_dir, "gradient_metrics.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for col, label in [
        ('total_grad_norm', 'Total'), ('encoder_grad_norm', 'Encoder+CBAM'),
        ('decoder_grad_norm', 'Decoder'), ('cbam_grad_norm', 'CBAM only'),
    ]:
        if col in df.columns:
            ax.plot(df['global_step'], df[col], label=label, linewidth=1.2)
    ax.set_xlabel("batch (global step)")
    ax.set_ylabel("L2 gradient norm (unscaled)")
    ax.set_title("Gradient norms by model component")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gradient_norms.png"), dpi=150)
    plt.close(fig)

    attribution_cols = ['grad_norm_loss_dice', 'grad_norm_loss_ft', 'grad_norm_loss_evid']
    present = [c for c in attribution_cols if c in df.columns and df[c].notna().any()]
    if present:
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in present:
            sub = df[df[col].notna()]
            ax.plot(sub['global_step'], sub[col], marker='o', markersize=3,
                    label=col.replace('grad_norm_loss_', ''), linewidth=1.0)
        ax.set_xlabel("batch (global step, sampled)")
        ax.set_ylabel("L2 gradient norm contributed by this loss alone")
        ax.set_title("Per-loss gradient attribution (isolated backward passes)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "gradient_loss_attribution.png"), dpi=150)
        plt.close(fig)

    print(f"[visualize] wrote gradient_norms.png (+ gradient_loss_attribution.png) to {out_dir}")


def plot_layerwise_gradients(log_dir, out_dir, epoch=None):
    """gradient_metrics_layerwise.csv -> layerwise_gradients_epoch<N>.png (latest logged epoch by default)"""
    df = _read_csv(log_dir, "gradient_metrics_layerwise.csv")
    if df is None:
        return

    target_epoch = epoch if epoch is not None else df['epoch'].max()
    sub = df[df['epoch'] == target_epoch]
    # last logged batch within that epoch, for a single clean snapshot
    last_batch = sub['iteration'].max()
    sub = sub[sub['iteration'] == last_batch].sort_values('grad_norm', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, 0.18 * len(sub))))
    ax.barh(sub['layer_name'], sub['grad_norm'])
    ax.set_xlabel("L2 gradient norm")
    ax.set_title(f"Layer-wise gradient norms (epoch {target_epoch}, batch {last_batch})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"layerwise_gradients_epoch{target_epoch}.png"), dpi=150)
    plt.close(fig)
    print(f"[visualize] wrote layerwise_gradients_epoch{target_epoch}.png to {out_dir} "
          f"({len(sub)} of the model's parameter tensors had a non-None gradient at this step)")


def plot_gradient_similarity(log_dir, out_dir):
    """gradient_metrics_similarity.csv -> gradient_similarity.png"""
    df = _read_csv(log_dir, "gradient_metrics_similarity.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for col, label in [
        ('cosine_sim_dice_vs_ft', 'Dice vs FocalTversky'),
        ('cosine_sim_dice_vs_evid', 'Dice vs Evidential'),
        ('cosine_sim_ft_vs_evid', 'FocalTversky vs Evidential'),
        ('cosine_sim_hybrid_vs_evid', 'Hybrid vs Evidential'),
    ]:
        if col in df.columns:
            ax.plot(df['global_step'], df[col], marker='o', markersize=3, label=label, linewidth=1.0)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel("sampled batch (global step)")
    ax.set_ylabel("cosine similarity between gradient vectors")
    ax.set_title("Gradient similarity between loss terms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gradient_similarity.png"), dpi=150)
    plt.close(fig)
    print(f"[visualize] wrote gradient_similarity.png to {out_dir}")


def plot_slice_diagnostics(log_dir, out_dir):
    """
    slice_metrics.csv -> slice_selection_histogram.png, fusion_weight_distribution.png,
                          center_index_selected_rate.png
    """
    df = _read_csv(log_dir, "slice_metrics.csv")
    if df is None:
        return

    # Histogram: which of the 64 resampled slice indices get selected, across all logged batches
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['selected_slice_index'], bins=64, range=(0, 64))
    if 'supervised_center_index' in df.columns and df['supervised_center_index'].notna().any():
        center = df['supervised_center_index'].dropna().iloc[0]
        ax.axvline(center, color='red', linestyle='--', label=f'supervised center index ({int(center)})')
        ax.legend()
    ax.set_xlabel("resampled slice index (0-63)")
    ax.set_ylabel("times selected (across all logged batches)")
    ax.set_title("Slice selection frequency histogram")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "slice_selection_histogram.png"), dpi=150)
    plt.close(fig)

    if 'fusion_attention_weight' in df.columns and df['fusion_attention_weight'].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['fusion_attention_weight'].dropna(), bins=40)
        ax.set_xlabel("Conv2D5Stem fusion attention weight")
        ax.set_ylabel("count")
        ax.set_title("Distribution of learned per-slice fusion weights")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fusion_weight_distribution.png"), dpi=150)
        plt.close(fig)

    if 'center_index_was_selected' in df.columns:
        by_epoch = df.groupby('epoch')['center_index_was_selected'].apply(
            lambda s: (s.astype(str) == 'True').mean())
        if len(by_epoch) > 0:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(by_epoch.index, by_epoch.values, marker='o')
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("epoch")
            ax.set_ylabel("fraction of samples where the supervised\ncenter index was among the selected slices")
            ax.set_title("Selector/supervision correspondence rate over training")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "center_index_selected_rate.png"), dpi=150)
            plt.close(fig)

    print(f"[visualize] wrote slice_selection_histogram.png "
          f"(+ fusion_weight_distribution.png, center_index_selected_rate.png) to {out_dir}")


def plot_similarity_heatmap(log_dir, out_dir):
    """gradient_metrics_similarity.csv -> gradient_similarity_heatmap.png
    Rows = epoch (mean over sampled batches that epoch), columns = loss pair.
    """
    df = _read_csv(log_dir, "gradient_metrics_similarity.csv")
    if df is None:
        return

    sim_cols = ['cosine_sim_dice_vs_ft', 'cosine_sim_dice_vs_evid',
                'cosine_sim_ft_vs_evid', 'cosine_sim_hybrid_vs_evid']
    present = [c for c in sim_cols if c in df.columns]
    by_epoch = df.groupby('epoch')[present].mean()

    fig, ax = plt.subplots(figsize=(6, max(4, 0.35 * len(by_epoch))))
    data = by_epoch.values
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(by_epoch.index)))
    ax.set_yticklabels(by_epoch.index)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([c.replace('cosine_sim_', '') for c in present], rotation=30, ha='right')
    ax.set_ylabel("epoch")
    ax.set_title("Gradient cosine similarity by epoch (mean of sampled batches)")
    fig.colorbar(im, ax=ax, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gradient_similarity_heatmap.png"), dpi=150)
    plt.close(fig)
    print(f"[visualize] wrote gradient_similarity_heatmap.png to {out_dir}")


def plot_correlation_matrix(log_dir, out_dir):
    """
    training_metrics.csv + gradient_metrics.csv -> correlation_matrix.png
    Correlates loss components against gradient-norm components across batches
    (inner-joined on epoch+iteration). Purely descriptive - Pearson correlation,
    no causal claim.
    """
    train_df = _read_csv(log_dir, "training_metrics.csv")
    grad_df = _read_csv(log_dir, "gradient_metrics.csv")
    if train_df is None or grad_df is None:
        print("[visualize] skip correlation_matrix.png (need both training_metrics.csv and gradient_metrics.csv)")
        return

    merged = pd.merge(
        train_df[['epoch', 'iteration', 'loss_dice', 'loss_focal_tversky', 'loss_evidential', 'loss_total']],
        grad_df[['epoch', 'iteration', 'total_grad_norm', 'encoder_grad_norm', 'decoder_grad_norm', 'cbam_grad_norm']],
        on=['epoch', 'iteration'], how='inner',
    )
    cols = ['loss_dice', 'loss_focal_tversky', 'loss_evidential', 'loss_total',
            'total_grad_norm', 'encoder_grad_norm', 'decoder_grad_norm', 'cbam_grad_norm']
    cols = [c for c in cols if c in merged.columns and merged[c].notna().any()]
    if len(merged) < 3 or len(cols) < 2:
        print("[visualize] skip correlation_matrix.png (insufficient joined rows/columns)")
        return

    corr = merged[cols].corr()

    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 2, 1.1 * len(cols) + 2))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha='center', va='center', fontsize=8)
    ax.set_title("Correlation matrix: loss components vs gradient norms\n(across all logged batches)")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "correlation_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"[visualize] wrote correlation_matrix.png to {out_dir}")


def plot_phase_comparison(log_dir, out_dir):
    """
    training_metrics.csv -> phase_comparison.png
    Splits the run into early/mid/late thirds (by epoch) and bar-charts mean
    loss components per phase - a direct visual for "are there optimization phases."
    """
    df = _read_csv(log_dir, "training_metrics.csv")
    if df is None:
        return
    epochs = sorted(df['epoch'].unique())
    if len(epochs) < 3:
        print("[visualize] skip phase_comparison.png (need >=3 distinct epochs)")
        return
    third = max(1, len(epochs) // 3)
    phase_map = {}
    for e in epochs[:third]:
        phase_map[e] = 'early'
    for e in epochs[third:2 * third]:
        phase_map[e] = 'mid'
    for e in epochs[2 * third:]:
        phase_map[e] = 'late'
    df = df.copy()
    df['phase'] = df['epoch'].map(phase_map)

    cols = ['loss_dice', 'loss_focal_tversky', 'loss_evidential']
    grouped = df.groupby('phase')[cols].mean().reindex(['early', 'mid', 'late'])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(cols))
    width = 0.25
    for i, phase in enumerate(['early', 'mid', 'late']):
        if phase in grouped.index:
            ax.bar(x + (i - 1) * width, grouped.loc[phase, cols].values, width, label=phase)
    ax.set_xticks(x)
    ax.set_xticklabels(['Dice', 'FocalTversky', 'Evidential'])
    ax.set_ylabel("mean loss value")
    ax.set_title("Loss composition by training phase (early/mid/late thirds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "phase_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"[visualize] wrote phase_comparison.png to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True,
                         help="Directory containing the diagnostic CSVs (e.g. SEGMENTATION_DIR)")
    parser.add_argument("--out-dir", default=None,
                         help="Where to write PNGs (default: <log-dir>/diagnostic_plots)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.log_dir, "diagnostic_plots")
    os.makedirs(out_dir, exist_ok=True)

    plot_training_curves(args.log_dir, out_dir)
    plot_gradient_norms(args.log_dir, out_dir)
    plot_layerwise_gradients(args.log_dir, out_dir)
    plot_gradient_similarity(args.log_dir, out_dir)
    plot_similarity_heatmap(args.log_dir, out_dir)
    plot_correlation_matrix(args.log_dir, out_dir)
    plot_phase_comparison(args.log_dir, out_dir)
    plot_slice_diagnostics(args.log_dir, out_dir)

    print(f"\nAll available plots written to: {out_dir}")


if __name__ == "__main__":
    main()
