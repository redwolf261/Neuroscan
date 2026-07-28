"""
Quantitative analysis of a single training run's diagnostic CSVs, produced to
answer the seven hypothesis-validation questions in
research_infra/PHASE_7_HYPOTHESIS_VALIDATION.md.

This script only computes and reports numbers - it does not decide whether the
hypothesis is supported. That judgement is made in the report, informed by
this script's output.

Usage:
    python analyze_run.py --log-dir <SEGMENTATION_DIR>
"""
import os
import argparse
import json

import numpy as np
import pandas as pd


# Rough mapping from named-parameter prefixes to architectural components, based on
# research_infra/PHASE_1_CODEBASE_AUDIT.md's read of final_model.py's module structure.
COMPONENT_PATTERNS = [
    ("AdaptiveSliceSelector", ("encoder.slice_selector.",)),
    ("Conv2D5Stem (2.5D fusion)", ("encoder.stem.",)),
    ("Encoder ResBlocks (incl. Mini-Swin attn)", ("encoder.stages.",)),
    ("CBAM", ("cbam.",)),
    ("Decoder upsample/skip", ("decoder.up_blocks.", "decoder.skip_convs.")),
    ("Decoder probability head", ("decoder.prob_head.",)),
    ("Decoder evidential/causal heads", ("decoder.causal_shared.", "decoder.anatomy_head.",
                                         "decoder.pathology_head.", "decoder.noise_head.",
                                         "decoder.causal_weights",)),
]


def classify_layer(name):
    for label, prefixes in COMPONENT_PATTERNS:
        if any(name.startswith(p) for p in prefixes):
            return label
    return "other"


def load(log_dir, name):
    path = os.path.join(log_dir, name)
    if not os.path.exists(path):
        print(f"[analyze] MISSING: {name}")
        return None
    df = pd.read_csv(path)
    print(f"[analyze] loaded {name}: {len(df)} rows, epochs {df['epoch'].min()}-{df['epoch'].max()}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    out_json = args.out_json or os.path.join(args.log_dir, "analysis_summary.json")
    results = {}

    train_df = load(args.log_dir, "training_metrics.csv")
    grad_df = load(args.log_dir, "gradient_metrics.csv")
    layer_df = load(args.log_dir, "gradient_metrics_layerwise.csv")
    sim_df = load(args.log_dir, "gradient_metrics_similarity.csv")

    # ---------------------------------------------------------------
    # Q1: gradient balance - total across full run, and by component
    # ---------------------------------------------------------------
    if grad_df is not None:
        comp_cols = ['encoder_grad_norm', 'decoder_grad_norm', 'cbam_grad_norm']
        means = grad_df[comp_cols].mean().to_dict()
        stds = grad_df[comp_cols].std().to_dict()
        ratio_decoder_encoder = means['decoder_grad_norm'] / means['encoder_grad_norm'] if means['encoder_grad_norm'] else float('nan')
        results['q1_component_grad_norm_mean'] = means
        results['q1_component_grad_norm_std'] = stds
        results['q1_decoder_to_encoder_ratio'] = ratio_decoder_encoder

        loss_cols = ['grad_norm_loss_dice', 'grad_norm_loss_ft', 'grad_norm_loss_evid']
        avail = [c for c in loss_cols if grad_df[c].notna().any()]
        if avail:
            loss_means = grad_df[avail].mean().to_dict()
            results['q1_per_loss_grad_norm_mean'] = loss_means
            total = sum(loss_means.values())
            results['q1_per_loss_grad_norm_share'] = {k: v / total for k, v in loss_means.items()} if total else {}

    # ---------------------------------------------------------------
    # Q2/Q3: does gradient similarity change over training? correlate with epoch
    # ---------------------------------------------------------------
    if sim_df is not None:
        sim_cols = ['cosine_sim_dice_vs_ft', 'cosine_sim_dice_vs_evid',
                    'cosine_sim_ft_vs_evid', 'cosine_sim_hybrid_vs_evid']
        by_epoch = sim_df.groupby('epoch')[sim_cols].mean()
        results['q2_q3_similarity_by_epoch'] = by_epoch.to_dict(orient='index')

        corr_with_epoch = {}
        for c in sim_cols:
            if sim_df[c].notna().sum() > 2:
                corr_with_epoch[c] = float(np.corrcoef(sim_df['epoch'], sim_df[c])[0, 1])
        results['q2_q3_correlation_with_epoch'] = corr_with_epoch

        # sign-change detection per column (potential phase transition)
        sign_changes = {}
        for c in sim_cols:
            vals = by_epoch[c].dropna().values
            signs = np.sign(vals)
            changes = np.sum(np.diff(signs) != 0)
            sign_changes[c] = int(changes)
        results['q2_q3_sign_changes_across_epochs'] = sign_changes

    # ---------------------------------------------------------------
    # Q4: per-component gradient share, aggregated by architectural block
    # ---------------------------------------------------------------
    if layer_df is not None:
        layer_df = layer_df.copy()
        layer_df['component'] = layer_df['layer_name'].apply(classify_layer)
        comp_summary = layer_df.groupby('component')['grad_norm'].agg(['mean', 'std', 'count'])
        results['q4_component_gradient_summary'] = comp_summary.to_dict(orient='index')

        # which components are present at all vs entirely absent (zero gradient - dead component)
        all_components = {label for label, _ in COMPONENT_PATTERNS}
        present_components = set(layer_df['component'].unique())
        results['q4_components_with_zero_recorded_gradient'] = sorted(all_components - present_components)

    # ---------------------------------------------------------------
    # Q5: phase detection - split into thirds by epoch, compare loss/gradient means
    # ---------------------------------------------------------------
    if train_df is not None:
        epochs = sorted(train_df['epoch'].unique())
        n = len(epochs)
        if n >= 3:
            third = max(1, n // 3)
            early_epochs = epochs[:third]
            mid_epochs = epochs[third:2 * third]
            late_epochs = epochs[2 * third:]

            phase_stats = {}
            for phase_name, phase_epochs in [('early', early_epochs), ('mid', mid_epochs), ('late', late_epochs)]:
                sub = train_df[train_df['epoch'].isin(phase_epochs)]
                phase_stats[phase_name] = {
                    'epochs': [int(e) for e in phase_epochs],
                    'mean_loss_dice': float(sub['loss_dice'].mean()),
                    'mean_loss_focal_tversky': float(sub['loss_focal_tversky'].mean()),
                    'mean_loss_evidential': float(sub['loss_evidential'].mean()),
                    'mean_loss_total': float(sub['loss_total'].mean()),
                }
            results['q5_phase_stats'] = phase_stats

        if grad_df is not None:
            grad_epochs = sorted(grad_df['epoch'].unique())
            ng = len(grad_epochs)
            if ng >= 3:
                third_g = max(1, ng // 3)
                phases_g = {
                    'early': grad_epochs[:third_g],
                    'mid': grad_epochs[third_g:2 * third_g],
                    'late': grad_epochs[2 * third_g:],
                }
                grad_phase_stats = {}
                for name, ep_list in phases_g.items():
                    sub = grad_df[grad_df['epoch'].isin(ep_list)]
                    grad_phase_stats[name] = {
                        'mean_total_grad_norm': float(sub['total_grad_norm'].mean()),
                        'mean_encoder_grad_norm': float(sub['encoder_grad_norm'].mean()),
                        'mean_decoder_grad_norm': float(sub['decoder_grad_norm'].mean()),
                    }
                results['q5_grad_phase_stats'] = grad_phase_stats

    # ---------------------------------------------------------------
    # Q6: inefficiency signals - coefficient of variation, negative similarity fraction,
    #     vanishing/exploding gradient checks
    # ---------------------------------------------------------------
    inefficiency = {}
    if grad_df is not None:
        cv = (grad_df['total_grad_norm'].std() / grad_df['total_grad_norm'].mean()
              if grad_df['total_grad_norm'].mean() else float('nan'))
        inefficiency['total_grad_norm_coefficient_of_variation'] = float(cv)
        inefficiency['max_total_grad_norm'] = float(grad_df['total_grad_norm'].max())
        inefficiency['min_total_grad_norm'] = float(grad_df['total_grad_norm'].min())
        inefficiency['any_grad_norm_near_zero (<1e-6)'] = bool((grad_df['total_grad_norm'] < 1e-6).any())
        inefficiency['any_grad_norm_very_large (>10)'] = bool((grad_df['total_grad_norm'] > 10).any())
    if sim_df is not None:
        neg_fraction = {}
        for c in ['cosine_sim_dice_vs_ft', 'cosine_sim_dice_vs_evid', 'cosine_sim_ft_vs_evid']:
            vals = sim_df[c].dropna()
            neg_fraction[c] = float((vals < 0).mean()) if len(vals) else float('nan')
        inefficiency['fraction_of_batches_with_negative_similarity'] = neg_fraction
    results['q6_inefficiency_signals'] = inefficiency

    # ---------------------------------------------------------------
    # Write results
    # ---------------------------------------------------------------
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[analyze] wrote {out_json}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
