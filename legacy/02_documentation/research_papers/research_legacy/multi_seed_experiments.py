"""
Multi-Seed Experiment Framework
Run experiments with multiple random seeds for statistical validation
"""

import torch
import numpy as np
import random
import scipy.stats as stats
import json
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# Standard seeds for reproducibility
STANDARD_SEEDS = [42, 123, 456, 789, 1024]


def set_all_seeds(seed):
    """
    Set all random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ Set all random seeds to {seed}")


def train_with_seed(model_fn, data, seed, config):
    """
    Train model with specific random seed
    
    Args:
        model_fn: Function that returns initialized model
        data: Training/validation data
        seed: Random seed
        config: Training configuration dict
    
    Returns:
        Dict with validation metrics
    """
    
    print(f"\n{'='*100}")
    print(f"TRAINING WITH SEED={seed}")
    print(f"{'='*100}")
    
    # Set all seeds
    set_all_seeds(seed)
    
    # Initialize model with this seed
    model = model_fn()
    
    # TODO: Implement actual training
    # This is a placeholder showing the structure
    
    print(f"\nConfiguration:")
    print(f"  Epochs: {config.get('epochs', 100)}")
    print(f"  Learning rate: {config.get('lr', 1e-4)}")
    print(f"  Batch size: {config.get('batch_size', 3)}")
    print(f"  Optimizer: {config.get('optimizer', 'AdamW')}")
    
    print(f"\n⚠️  Placeholder: Implement actual training here")
    print(f"You would:")
    print(f"  1. Create dataloaders with seed")
    print(f"  2. Initialize model, optimizer, scheduler")
    print(f"  3. Train for {config.get('epochs', 100)} epochs")
    print(f"  4. Evaluate on validation set")
    print(f"  5. Return validation metrics")
    
    # Dummy results (replace with actual training)
    # Add some variance to simulate real training
    base_dice = 0.8350
    variance = np.random.normal(0, 0.008)
    
    results = {
        'seed': seed,
        'dice': base_dice + variance,
        'precision': 0.7750 + np.random.normal(0, 0.010),
        'recall': 0.9150 + np.random.normal(0, 0.008),
        'f1': 0.8400 + variance,
        'specificity': 0.9950 + np.random.normal(0, 0.0002),
        'hd95': 8.50 + np.random.normal(0, 0.5),
        'params': 34.20,  # Million
        'flops': 3.51,     # GFLOPs
        'convergence_epoch': np.random.randint(25, 35),
        'training_time_minutes': np.random.uniform(180, 220)
    }
    
    print(f"\nResults (seed={seed}):")
    print(f"  Dice: {results['dice']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    print(f"  Converged at epoch: {results['convergence_epoch']}")
    
    return results


def run_multi_seed_experiment(model_fn, data, seeds=STANDARD_SEEDS, config=None):
    """
    Run experiment with multiple seeds
    
    Args:
        model_fn: Function that returns initialized model
        data: Training/validation data
        seeds: List of random seeds
        config: Training configuration
    
    Returns:
        List of result dicts, one per seed
    """
    
    if config is None:
        config = {
            'epochs': 100,
            'lr': 1e-4,
            'batch_size': 3,
            'optimizer': 'AdamW',
            'weight_decay': 0.01
        }
    
    print(f"\n{'='*100}")
    print(f"MULTI-SEED EXPERIMENT")
    print(f"{'='*100}")
    print(f"Number of seeds: {len(seeds)}")
    print(f"Seeds: {seeds}")
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    results = []
    
    for seed in seeds:
        result = train_with_seed(model_fn, data, seed, config)
        results.append(result)
    
    return results


def compute_statistics(values):
    """
    Compute statistical measures for a list of values
    
    Args:
        values: List or array of values
    
    Returns:
        Dict with statistical measures
    """
    values = np.array(values)
    n = len(values)
    
    stats_dict = {
        'mean': np.mean(values),
        'std': np.std(values, ddof=1),  # Sample std
        'stderr': stats.sem(values),
        'median': np.median(values),
        'min': np.min(values),
        'max': np.max(values),
        'n': n
    }
    
    # 95% confidence interval
    if n > 1:
        ci = stats.t.interval(0.95, n-1, loc=stats_dict['mean'], scale=stats_dict['stderr'])
        stats_dict['ci_95_lower'] = ci[0]
        stats_dict['ci_95_upper'] = ci[1]
    else:
        stats_dict['ci_95_lower'] = stats_dict['mean']
        stats_dict['ci_95_upper'] = stats_dict['mean']
    
    return stats_dict


def paired_t_test(baseline_values, proposed_values):
    """
    Perform paired t-test
    
    Args:
        baseline_values: Array of baseline metric values
        proposed_values: Array of proposed metric values
    
    Returns:
        Dict with test results
    """
    baseline_values = np.array(baseline_values)
    proposed_values = np.array(proposed_values)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(proposed_values, baseline_values)
    
    # Effect size (Cohen's d for paired samples)
    diff = proposed_values - baseline_values
    d = np.mean(diff) / np.std(diff, ddof=1)
    
    # Determine significance
    if p_value < 0.001:
        significance = "***"
        conclusion = "Highly significant"
    elif p_value < 0.01:
        significance = "**"
        conclusion = "Very significant"
    elif p_value < 0.05:
        significance = "*"
        conclusion = "Significant"
    else:
        significance = "ns"
        conclusion = "Not significant"
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significance': significance,
        'conclusion': conclusion,
        'cohens_d': d,
        'mean_improvement': np.mean(diff),
        'std_improvement': np.std(diff, ddof=1)
    }


def wilcoxon_test(baseline_values, proposed_values):
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
    
    Args:
        baseline_values: Array of baseline metric values
        proposed_values: Array of proposed metric values
    
    Returns:
        Dict with test results
    """
    baseline_values = np.array(baseline_values)
    proposed_values = np.array(proposed_values)
    
    # Wilcoxon signed-rank test
    w_stat, p_value = stats.wilcoxon(proposed_values, baseline_values)
    
    # Determine significance
    if p_value < 0.001:
        significance = "***"
        conclusion = "Highly significant"
    elif p_value < 0.01:
        significance = "**"
        conclusion = "Very significant"
    elif p_value < 0.05:
        significance = "*"
        conclusion = "Significant"
    else:
        significance = "ns"
        conclusion = "Not significant"
    
    return {
        'w_statistic': w_stat,
        'p_value': p_value,
        'significance': significance,
        'conclusion': conclusion
    }


def aggregate_results(baseline_results, proposed_results):
    """
    Aggregate results from multiple seeds
    
    Args:
        baseline_results: List of result dicts for baseline
        proposed_results: List of result dicts for proposed method
    
    Returns:
        Dict with aggregated statistics and comparisons
    """
    
    metrics = ['dice', 'precision', 'recall', 'f1', 'specificity', 'hd95']
    
    aggregated = {
        'baseline': {},
        'proposed': {},
        'comparison': {}
    }
    
    # Aggregate baseline
    for metric in metrics:
        if metric in baseline_results[0]:
            values = [r[metric] for r in baseline_results]
            aggregated['baseline'][metric] = compute_statistics(values)
    
    # Aggregate proposed
    for metric in metrics:
        if metric in proposed_results[0]:
            values = [r[metric] for r in proposed_results]
            aggregated['proposed'][metric] = compute_statistics(values)
    
    # Statistical comparison
    for metric in metrics:
        if metric in baseline_results[0] and metric in proposed_results[0]:
            baseline_values = [r[metric] for r in baseline_results]
            proposed_values = [r[metric] for r in proposed_results]
            
            # For HD95 (lower is better), flip comparison
            if metric == 'hd95':
                ttest = paired_t_test(proposed_values, baseline_values)  # Flip
            else:
                ttest = paired_t_test(baseline_values, proposed_values)
            
            wilcoxon = wilcoxon_test(baseline_values, proposed_values)
            
            aggregated['comparison'][metric] = {
                'paired_t_test': ttest,
                'wilcoxon_test': wilcoxon,
                'improvement_pct': ((aggregated['proposed'][metric]['mean'] - 
                                   aggregated['baseline'][metric]['mean']) / 
                                   aggregated['baseline'][metric]['mean'] * 100)
            }
    
    return aggregated


def print_results(baseline_results, proposed_results, aggregated):
    """Print formatted multi-seed experiment results"""
    
    print(f"\n{'='*100}")
    print(f"MULTI-SEED EXPERIMENT RESULTS")
    print(f"{'='*100}")
    
    # Per-seed results
    print(f"\nBASELINE RESULTS (per seed):")
    print(f"{'-'*100}")
    print(f"{'Seed':<8} {'Dice':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'Specificity':<12} {'HD95':<10}")
    print(f"{'-'*100}")
    for r in baseline_results:
        print(f"{r['seed']:<8} {r['dice']:<10.4f} {r['precision']:<12.4f} {r['recall']:<10.4f} "
              f"{r['f1']:<10.4f} {r['specificity']:<12.4f} {r['hd95']:<10.2f}")
    print(f"{'-'*100}\n")
    
    print(f"PROPOSED RESULTS (per seed):")
    print(f"{'-'*100}")
    print(f"{'Seed':<8} {'Dice':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'Specificity':<12} {'HD95':<10}")
    print(f"{'-'*100}")
    for r in proposed_results:
        print(f"{r['seed']:<8} {r['dice']:<10.4f} {r['precision']:<12.4f} {r['recall']:<10.4f} "
              f"{r['f1']:<10.4f} {r['specificity']:<12.4f} {r['hd95']:<10.2f}")
    print(f"{'-'*100}\n")
    
    # Aggregated statistics
    print(f"AGGREGATED STATISTICS (Mean ± Std [95% CI]):")
    print(f"{'-'*100}")
    print(f"{'Metric':<15} {'Baseline':<35} {'Proposed':<35} {'Improvement':<15}")
    print(f"{'-'*100}")
    
    metrics = ['dice', 'precision', 'recall', 'f1', 'specificity', 'hd95']
    for metric in metrics:
        if metric in aggregated['baseline']:
            b = aggregated['baseline'][metric]
            p = aggregated['proposed'][metric]
            imp = aggregated['comparison'][metric]['improvement_pct']
            
            baseline_str = f"{b['mean']:.4f} ± {b['std']:.4f} [{b['ci_95_lower']:.4f}, {b['ci_95_upper']:.4f}]"
            proposed_str = f"{p['mean']:.4f} ± {p['std']:.4f} [{p['ci_95_lower']:.4f}, {p['ci_95_upper']:.4f}]"
            imp_str = f"{imp:+.2f}%"
            
            print(f"{metric.capitalize():<15} {baseline_str:<35} {proposed_str:<35} {imp_str:<15}")
    
    print(f"{'-'*100}\n")
    
    # Statistical tests
    print(f"STATISTICAL SIGNIFICANCE TESTS:")
    print(f"{'-'*100}")
    print(f"{'Metric':<15} {'t-statistic':<15} {'p-value':<15} {'Significance':<15} {'Cohen d':<15}")
    print(f"{'-'*100}")
    
    for metric in metrics:
        if metric in aggregated['comparison']:
            comp = aggregated['comparison'][metric]['paired_t_test']
            
            print(f"{metric.capitalize():<15} {comp['t_statistic']:<15.4f} {comp['p_value']:<15.6f} "
                  f"{comp['significance']:<15} {comp['cohens_d']:<15.4f}")
    
    print(f"{'-'*100}\n")
    
    # Interpretation
    print(f"INTERPRETATION:")
    print(f"{'-'*100}")
    
    dice_test = aggregated['comparison']['dice']['paired_t_test']
    print(f"Dice Score Comparison:")
    print(f"  Baseline: {aggregated['baseline']['dice']['mean']:.4f} ± {aggregated['baseline']['dice']['std']:.4f}")
    print(f"  Proposed: {aggregated['proposed']['dice']['mean']:.4f} ± {aggregated['proposed']['dice']['std']:.4f}")
    print(f"  Improvement: {aggregated['comparison']['dice']['improvement_pct']:+.2f}%")
    print(f"  t-statistic: {dice_test['t_statistic']:.4f}")
    print(f"  p-value: {dice_test['p_value']:.6f} {dice_test['significance']}")
    print(f"  Effect size (Cohen's d): {dice_test['cohens_d']:.4f}", end="")
    
    # Interpret effect size
    d = abs(dice_test['cohens_d'])
    if d < 0.2:
        effect = "negligible"
    elif d < 0.5:
        effect = "small"
    elif d < 0.8:
        effect = "medium"
    else:
        effect = "large"
    print(f" ({effect} effect)")
    
    print(f"\n  Conclusion: {dice_test['conclusion']}")
    if dice_test['p_value'] < 0.05:
        print(f"  ✓ Proposed method significantly outperforms baseline (p < 0.05)")
    else:
        print(f"  ✗ No significant difference between methods")
    
    print(f"{'-'*100}\n")


def save_results(baseline_results, proposed_results, aggregated, output_path):
    """Save multi-seed experiment results"""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results = {
        'baseline_results': convert_numpy(baseline_results),
        'proposed_results': convert_numpy(proposed_results),
        'aggregated': convert_numpy(aggregated),
        'seeds': STANDARD_SEEDS
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved results to: {output_path}")
    
    # Save summary text
    summary_path = output_path.parent / 'multi_seed_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("MULTI-SEED EXPERIMENT SUMMARY\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Number of seeds: {len(STANDARD_SEEDS)}\n")
        f.write(f"Seeds: {STANDARD_SEEDS}\n\n")
        
        f.write("AGGREGATED RESULTS (Mean ± Std [95% CI]):\n")
        f.write("-"*100 + "\n")
        
        metrics = ['dice', 'precision', 'recall', 'f1', 'specificity', 'hd95']
        for metric in metrics:
            if metric in aggregated['baseline']:
                b = aggregated['baseline'][metric]
                p = aggregated['proposed'][metric]
                
                f.write(f"\n{metric.upper()}:\n")
                f.write(f"  Baseline: {b['mean']:.4f} ± {b['std']:.4f} [{b['ci_95_lower']:.4f}, {b['ci_95_upper']:.4f}]\n")
                f.write(f"  Proposed: {p['mean']:.4f} ± {p['std']:.4f} [{p['ci_95_lower']:.4f}, {p['ci_95_upper']:.4f}]\n")
                
                comp = aggregated['comparison'][metric]['paired_t_test']
                f.write(f"  t-test: t={comp['t_statistic']:.4f}, p={comp['p_value']:.6f} {comp['significance']}\n")
                f.write(f"  Effect size: Cohen's d = {comp['cohens_d']:.4f}\n")
                f.write(f"  Improvement: {aggregated['comparison'][metric]['improvement_pct']:+.2f}%\n")
    
    print(f"✓ Saved summary to: {summary_path}")


def main():
    """Run multi-seed experiment demo"""
    
    print("="*100)
    print("MULTI-SEED EXPERIMENT FRAMEWORK")
    print("="*100)
    
    # Placeholder model function
    def create_baseline_model():
        print("Creating baseline model (2.5D without CSRF, no MAE)...")
        # return baseline_model
        return None
    
    def create_proposed_model():
        print("Creating proposed model (2.5D + CSRF + MAE)...")
        # return proposed_model
        return None
    
    # Placeholder data
    data = None
    
    config = {
        'epochs': 100,
        'lr': 1e-4,
        'batch_size': 3,
        'optimizer': 'AdamW',
        'weight_decay': 0.01
    }
    
    print("\n⚠️  Running with placeholder/dummy results")
    print("Replace create_baseline_model() and create_proposed_model() with actual implementations")
    
    # Run experiments
    print(f"\n{'='*100}")
    print("BASELINE: 2.5D without CSRF, no MAE")
    print(f"{'='*100}")
    baseline_results = run_multi_seed_experiment(
        create_baseline_model,
        data,
        seeds=STANDARD_SEEDS,
        config=config
    )
    
    print(f"\n{'='*100}")
    print("PROPOSED: 2.5D + CSRF + MAE")
    print(f"{'='*100}")
    proposed_results = run_multi_seed_experiment(
        create_proposed_model,
        data,
        seeds=STANDARD_SEEDS,
        config=config
    )
    
    # Aggregate and compare
    aggregated = aggregate_results(baseline_results, proposed_results)
    
    # Print results
    print_results(baseline_results, proposed_results, aggregated)
    
    # Save results
    save_results(
        baseline_results,
        proposed_results,
        aggregated,
        output_path='research/multi_seed_results.json'
    )
    
    print(f"\n{'='*100}")
    print("MULTI-SEED EXPERIMENT COMPLETE")
    print(f"{'='*100}")
    print("\nNEXT STEPS:")
    print("1. Implement actual model creation functions")
    print("2. Load real training/validation data")
    print("3. Run full training for all seeds (will take several hours per seed)")
    print("4. Analyze results and significance tests")
    print("="*100)


if __name__ == "__main__":
    main()
