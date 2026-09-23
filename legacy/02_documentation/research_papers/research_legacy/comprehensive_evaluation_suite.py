"""
Comprehensive Evaluation Suite for Rigorous Paper Submission
==============================================================

This suite includes:
1. Repeated runs with multiple seeds (5+ runs) + statistical tests
2. Strong baseline comparisons (nnU-Net, SwinUNETR, 3D U-Net, MC-Dropout, Ensembles)
3. Factorial ablation study (all component combinations)
4. Calibration metrics (ECE, Brier, reliability diagrams)
5. Computational profiling (FLOPs, params, memory, latency)

Author: Research Team
Date: November 9, 2025
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import time
from scipy import stats
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from final_model import (
    HybridMiniSwin2D5_CBAM,
    HybridMiniSwin2D5_ResNetEncoder,
    LightweightDecoder,
    CBAM_Module,
    AdaptiveSliceSelector
)

# ============================================================================
# 1. MULTIPLE SEED EXPERIMENTS
# ============================================================================

class MultiSeedExperiment:
    """
    Run experiments with multiple random seeds and compute statistical significance
    """
    
    def __init__(self, output_dir, num_seeds=5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.num_seeds = num_seeds
        self.results = []
    
    def run_experiment(self, model_config, dataset, seed):
        """Run single experiment with given seed"""
        # Set all random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Initialize model with config
        model = self._build_model(model_config, seed)
        
        # Train and evaluate
        metrics = self._train_and_evaluate(model, dataset, seed)
        
        return metrics
    
    def run_multiple_seeds(self, model_config, dataset, model_name):
        """Run experiment for multiple seeds"""
        print(f"\n{'='*80}")
        print(f"Running {model_name} with {self.num_seeds} different seeds")
        print(f"{'='*80}")
        
        all_results = []
        
        for seed in range(self.num_seeds):
            print(f"\nSeed {seed+1}/{self.num_seeds}: {seed}")
            metrics = self.run_experiment(model_config, dataset, seed)
            metrics['seed'] = seed
            metrics['model'] = model_name
            all_results.append(metrics)
        
        # Compute statistics
        stats_summary = self._compute_statistics(all_results, model_name)
        
        # Save results
        self._save_results(all_results, stats_summary, model_name)
        
        return all_results, stats_summary
    
    def _build_model(self, config, seed):
        """Build model from config"""
        # Placeholder - implement based on config
        pass
    
    def _train_and_evaluate(self, model, dataset, seed):
        """Train and evaluate model"""
        # Placeholder - implement training loop
        pass
    
    def _compute_statistics(self, results, model_name):
        """Compute mean, std, and confidence intervals"""
        metrics = ['dice', 'precision', 'recall', 'f1', 'iou']
        
        stats = {
            'model': model_name,
            'num_runs': len(results)
        }
        
        for metric in metrics:
            values = [r[metric] for r in results]
            stats[f'{metric}_mean'] = np.mean(values)
            stats[f'{metric}_std'] = np.std(values)
            stats[f'{metric}_min'] = np.min(values)
            stats[f'{metric}_max'] = np.max(values)
            
            # 95% confidence interval
            ci = stats.t.interval(0.95, len(values)-1, 
                                  loc=np.mean(values), 
                                  scale=stats.sem(values))
            stats[f'{metric}_ci_lower'] = ci[0]
            stats[f'{metric}_ci_upper'] = ci[1]
        
        return stats
    
    def compare_models(self, results_a, results_b, model_a_name, model_b_name):
        """Perform statistical comparison between two models"""
        print(f"\n{'='*80}")
        print(f"Statistical Comparison: {model_a_name} vs {model_b_name}")
        print(f"{'='*80}")
        
        metrics = ['dice', 'precision', 'recall', 'f1']
        comparison = {}
        
        for metric in metrics:
            values_a = [r[metric] for r in results_a]
            values_b = [r[metric] for r in results_b]
            
            # Paired t-test
            t_stat, t_pval = stats.ttest_rel(values_a, values_b)
            
            # Wilcoxon signed-rank test (non-parametric)
            w_stat, w_pval = stats.wilcoxon(values_a, values_b)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.std(values_a)**2 + np.std(values_b)**2) / 2)
            cohens_d = (np.mean(values_a) - np.mean(values_b)) / pooled_std
            
            comparison[metric] = {
                'mean_a': np.mean(values_a),
                'mean_b': np.mean(values_b),
                'diff': np.mean(values_a) - np.mean(values_b),
                't_statistic': t_stat,
                't_pvalue': t_pval,
                'wilcoxon_statistic': w_stat,
                'wilcoxon_pvalue': w_pval,
                'cohens_d': cohens_d,
                'significant_t': t_pval < 0.05,
                'significant_w': w_pval < 0.05
            }
            
            # Print results
            print(f"\n{metric.upper()}:")
            print(f"  {model_a_name}: {np.mean(values_a):.4f} ± {np.std(values_a):.4f}")
            print(f"  {model_b_name}: {np.mean(values_b):.4f} ± {np.std(values_b):.4f}")
            print(f"  Difference: {comparison[metric]['diff']:.4f}")
            print(f"  Paired t-test: p={t_pval:.4f} {'***' if t_pval < 0.001 else '**' if t_pval < 0.01 else '*' if t_pval < 0.05 else 'ns'}")
            print(f"  Wilcoxon test: p={w_pval:.4f} {'***' if w_pval < 0.001 else '**' if w_pval < 0.01 else '*' if w_pval < 0.05 else 'ns'}")
            print(f"  Cohen's d: {cohens_d:.4f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})")
        
        # Save comparison
        self._save_comparison(comparison, model_a_name, model_b_name)
        
        return comparison
    
    def _save_results(self, results, stats, model_name):
        """Save results to CSV and JSON"""
        # Individual runs
        df = pd.DataFrame(results)
        csv_path = self.output_dir / f'{model_name}_individual_runs.csv'
        df.to_csv(csv_path, index=False)
        
        # Statistics summary
        stats_path = self.output_dir / f'{model_name}_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✅ Results saved:")
        print(f"   Individual runs: {csv_path}")
        print(f"   Statistics: {stats_path}")
    
    def _save_comparison(self, comparison, model_a, model_b):
        """Save statistical comparison"""
        comp_path = self.output_dir / f'comparison_{model_a}_vs_{model_b}.json'
        with open(comp_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n✅ Comparison saved: {comp_path}")


# ============================================================================
# 2. BASELINE MODELS
# ============================================================================

class BaselineModels:
    """
    Strong baseline implementations:
    - nnU-Net style architecture
    - SwinUNETR (if feasible)
    - 3D U-Net
    - MC-Dropout uncertainty
    - Deep Ensembles
    """
    
    @staticmethod
    def create_3d_unet(in_channels=1, num_classes=1, base_features=32):
        """Standard 3D U-Net baseline"""
        
        class UNet3D(nn.Module):
            def __init__(self, in_ch, out_ch, features):
                super().__init__()
                
                # Encoder
                self.enc1 = self._conv_block(in_ch, features)
                self.enc2 = self._conv_block(features, features*2)
                self.enc3 = self._conv_block(features*2, features*4)
                self.enc4 = self._conv_block(features*4, features*8)
                
                self.pool = nn.MaxPool3d(2)
                
                # Bottleneck
                self.bottleneck = self._conv_block(features*8, features*16)
                
                # Decoder
                self.up4 = nn.ConvTranspose3d(features*16, features*8, 2, stride=2)
                self.dec4 = self._conv_block(features*16, features*8)
                
                self.up3 = nn.ConvTranspose3d(features*8, features*4, 2, stride=2)
                self.dec3 = self._conv_block(features*8, features*4)
                
                self.up2 = nn.ConvTranspose3d(features*4, features*2, 2, stride=2)
                self.dec2 = self._conv_block(features*4, features*2)
                
                self.up1 = nn.ConvTranspose3d(features*2, features, 2, stride=2)
                self.dec1 = self._conv_block(features*2, features)
                
                # Output
                self.out = nn.Conv3d(features, out_ch, 1)
            
            def _conv_block(self, in_ch, out_ch):
                return nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                # Encoder
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                e4 = self.enc4(self.pool(e3))
                
                # Bottleneck
                b = self.bottleneck(self.pool(e4))
                
                # Decoder
                d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
                d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
                d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
                d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
                
                return self.out(d1)
        
        return UNet3D(in_channels, num_classes, base_features)
    
    @staticmethod
    def create_mc_dropout_model(base_model, dropout_rate=0.5):
        """MC-Dropout wrapper for uncertainty estimation"""
        
        class MCDropoutModel(nn.Module):
            def __init__(self, model, p=0.5):
                super().__init__()
                self.model = model
                self.dropout = nn.Dropout3d(p=p)
            
            def forward(self, x, num_samples=10):
                """
                Forward pass with MC-Dropout
                Returns: mean prediction and uncertainty (std)
                """
                self.train()  # Enable dropout at test time
                
                predictions = []
                for _ in range(num_samples):
                    pred = self.model(x)
                    predictions.append(torch.sigmoid(pred))
                
                predictions = torch.stack(predictions)
                
                mean_pred = predictions.mean(dim=0)
                std_pred = predictions.std(dim=0)
                
                return {
                    'probs': mean_pred,
                    'uncertainty': std_pred,
                    'all_predictions': predictions
                }
        
        return MCDropoutModel(base_model, dropout_rate)
    
    @staticmethod
    def create_deep_ensemble(model_configs, num_models=3):
        """Deep Ensemble for uncertainty estimation"""
        
        class DeepEnsemble(nn.Module):
            def __init__(self, models):
                super().__init__()
                self.models = nn.ModuleList(models)
            
            def forward(self, x):
                """
                Forward pass through ensemble
                Returns: mean prediction and uncertainty
                """
                predictions = []
                
                for model in self.models:
                    with torch.no_grad():
                        pred = model(x)
                        predictions.append(torch.sigmoid(pred))
                
                predictions = torch.stack(predictions)
                
                mean_pred = predictions.mean(dim=0)
                std_pred = predictions.std(dim=0)
                
                return {
                    'probs': mean_pred,
                    'uncertainty': std_pred,
                    'all_predictions': predictions
                }
        
        # Create ensemble of independently initialized models
        models = [model_configs[i] for i in range(num_models)]
        
        return DeepEnsemble(models)


# ============================================================================
# 3. FACTORIAL ABLATION STUDY
# ============================================================================

class FactorialAblation:
    """
    Systematic ablation of all model components
    
    Factors:
    - Adaptive selector: {on, off}
    - MAE pretrain: {on, off}
    - Attention: {CBAM, CSRF, none}
    - Evidential head: {on, off}
    - Self-correction: {on, off}
    
    Total: 2 × 2 × 3 × 2 × 2 = 48 configurations
    """
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Define all factor levels
        self.factors = {
            'adaptive_selector': [True, False],
            'mae_pretrain': [True, False],
            'attention': ['cbam', 'csrf', 'none'],
            'evidential': [True, False],
            'self_correction': [True, False]
        }
        
        self.results = []
    
    def generate_all_configs(self):
        """Generate all combinations of factors"""
        from itertools import product
        
        configs = []
        
        for adaptive, mae, attention, evidential, self_corr in product(
            self.factors['adaptive_selector'],
            self.factors['mae_pretrain'],
            self.factors['attention'],
            self.factors['evidential'],
            self.factors['self_correction']
        ):
            config = {
                'adaptive_selector': adaptive,
                'mae_pretrain': mae,
                'attention': attention,
                'evidential': evidential,
                'self_correction': self_corr
            }
            configs.append(config)
        
        print(f"Generated {len(configs)} configurations for factorial ablation")
        return configs
    
    def run_ablation(self, dataset, num_seeds=3):
        """Run full factorial ablation"""
        configs = self.generate_all_configs()
        
        print(f"\n{'='*80}")
        print(f"FACTORIAL ABLATION STUDY")
        print(f"Total configurations: {len(configs)}")
        print(f"Seeds per config: {num_seeds}")
        print(f"Total experiments: {len(configs) * num_seeds}")
        print(f"{'='*80}\n")
        
        for idx, config in enumerate(tqdm(configs, desc="Running ablations")):
            print(f"\nConfiguration {idx+1}/{len(configs)}")
            print(f"  {config}")
            
            # Run multiple seeds for this config
            config_results = []
            for seed in range(num_seeds):
                metrics = self._run_config(config, dataset, seed)
                metrics.update(config)
                metrics['seed'] = seed
                config_results.append(metrics)
            
            # Compute statistics for this config
            config_stats = self._compute_config_stats(config_results)
            self.results.append(config_stats)
        
        # Analyze results
        self._analyze_factorial_effects()
        
        return self.results
    
    def _run_config(self, config, dataset, seed):
        """Run single configuration"""
        # Placeholder - implement training with specific config
        pass
    
    def _compute_config_stats(self, results):
        """Compute statistics for a configuration"""
        stats = {}
        
        # Copy config
        stats.update(results[0])
        stats.pop('seed', None)
        
        # Compute mean and std for metrics
        metrics = ['dice', 'precision', 'recall', 'f1']
        for metric in metrics:
            values = [r[metric] for r in results]
            stats[f'{metric}_mean'] = np.mean(values)
            stats[f'{metric}_std'] = np.std(values)
        
        return stats
    
    def _analyze_factorial_effects(self):
        """Analyze main effects and interactions"""
        print(f"\n{'='*80}")
        print("FACTORIAL ANALYSIS")
        print(f"{'='*80}\n")
        
        df = pd.DataFrame(self.results)
        
        # Main effects
        print("MAIN EFFECTS (Impact on Dice Score):")
        print("-" * 60)
        
        for factor in self.factors.keys():
            if factor == 'attention':
                # Categorical with 3 levels
                grouped = df.groupby(factor)['dice_mean'].mean()
                print(f"\n{factor}:")
                for level, value in grouped.items():
                    print(f"  {level}: {value:.4f}")
            else:
                # Binary factors
                on_mean = df[df[factor] == True]['dice_mean'].mean()
                off_mean = df[df[factor] == False]['dice_mean'].mean()
                effect = on_mean - off_mean
                print(f"\n{factor}:")
                print(f"  ON:  {on_mean:.4f}")
                print(f"  OFF: {off_mean:.4f}")
                print(f"  Effect: {effect:+.4f}")
        
        # Save results
        results_path = self.output_dir / 'factorial_ablation_results.csv'
        df.to_csv(results_path, index=False)
        
        print(f"\n✅ Factorial ablation results saved: {results_path}")


# ============================================================================
# 4. CALIBRATION METRICS
# ============================================================================

class CalibrationAnalysis:
    """
    Comprehensive calibration evaluation:
    - Expected Calibration Error (ECE)
    - Adaptive ECE
    - Brier Score
    - Reliability Diagrams
    - Precision@Confidence
    """
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def compute_ece(self, probs, targets, n_bins=10):
        """
        Compute Expected Calibration Error
        
        ECE = Σ (n_b / n) * |acc_b - conf_b|
        """
        probs_flat = probs.flatten()
        targets_flat = targets.flatten()
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        bin_metrics = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (probs_flat >= bin_lower) & (probs_flat < bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = targets_flat[in_bin].mean()
                avg_confidence_in_bin = probs_flat[in_bin].mean()
                
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
                
                bin_metrics.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'accuracy': accuracy_in_bin,
                    'confidence': avg_confidence_in_bin,
                    'proportion': prop_in_bin
                })
        
        return ece, bin_metrics
    
    def compute_adaptive_ece(self, probs, targets, n_bins=10):
        """
        Adaptive ECE with equal-mass bins
        """
        probs_flat = probs.flatten()
        targets_flat = targets.flatten()
        
        # Create equal-mass bins
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_boundaries = np.percentile(probs_flat, quantiles * 100)
        
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (probs_flat >= bin_boundaries[i]) & (probs_flat < bin_boundaries[i+1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = targets_flat[in_bin].mean()
                avg_confidence_in_bin = probs_flat[in_bin].mean()
                
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
        
        return ece
    
    def compute_brier_score(self, probs, targets):
        """
        Compute Brier Score
        BS = (1/n) Σ (p_i - y_i)²
        """
        return np.mean((probs - targets) ** 2)
    
    def plot_reliability_diagram(self, probs, targets, n_bins=10, save_path=None):
        """
        Plot reliability diagram (calibration curve)
        """
        _, bin_metrics = self.compute_ece(probs, targets, n_bins)
        
        if not bin_metrics:
            return
        
        bin_centers = [(m['bin_lower'] + m['bin_upper']) / 2 for m in bin_metrics]
        accuracies = [m['accuracy'] for m in bin_metrics]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot reliability curve
        ax.plot(bin_centers, accuracies, 'o-', label='Model', linewidth=2, markersize=8)
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.5)
        
        ax.set_xlabel('Confidence', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax.set_title('Reliability Diagram', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def precision_at_confidence(self, probs, preds, targets, confidence_thresholds=[0.5, 0.7, 0.9]):
        """
        Compute precision at different confidence thresholds
        """
        results = []
        
        for threshold in confidence_thresholds:
            # Only consider predictions with confidence >= threshold
            high_conf_mask = probs >= threshold
            
            if high_conf_mask.sum() == 0:
                continue
            
            high_conf_preds = preds[high_conf_mask]
            high_conf_targets = targets[high_conf_mask]
            
            tp = ((high_conf_preds == 1) & (high_conf_targets == 1)).sum()
            fp = ((high_conf_preds == 1) & (high_conf_targets == 0)).sum()
            
            precision = tp / (tp + fp + 1e-8)
            coverage = high_conf_mask.sum() / len(probs)
            
            results.append({
                'confidence_threshold': threshold,
                'precision': precision.item() if hasattr(precision, 'item') else precision,
                'coverage': coverage.item() if hasattr(coverage, 'item') else coverage,
                'num_samples': high_conf_mask.sum().item()
            })
        
        return results


# ============================================================================
# 5. COMPUTATIONAL PROFILING
# ============================================================================

class ComputationalProfiler:
    """
    Profile model computational requirements:
    - Parameters
    - FLOPs
    - GPU Memory
    - Training time
    - Inference latency
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
    
    def count_parameters(self, model):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_params_M': total_params / 1e6,
            'trainable_params_M': trainable_params / 1e6
        }
    
    def estimate_flops(self, model, input_shape=(1, 1, 64, 64, 64)):
        """
        Estimate FLOPs using thop library
        """
        try:
            from thop import profile, clever_format
            
            input_tensor = torch.randn(input_shape).to(self.device)
            model = model.to(self.device)
            
            flops, params = profile(model, inputs=(input_tensor,), verbose=False)
            flops, params = clever_format([flops, params], "%.3f")
            
            return {
                'flops': flops,
                'params_thop': params
            }
        except ImportError:
            print("⚠️ thop not installed. Run: pip install thop")
            return {'flops': 'N/A', 'params_thop': 'N/A'}
    
    def measure_memory(self, model, input_shape=(1, 1, 64, 64, 64)):
        """Measure peak GPU memory usage"""
        if not torch.cuda.is_available():
            return {'peak_memory_MB': 'N/A', 'allocated_memory_MB': 'N/A'}
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        model = model.to('cuda')
        input_tensor = torch.randn(input_shape).to('cuda')
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_tensor)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
        allocated_memory = torch.cuda.memory_allocated() / 1024**2
        
        return {
            'peak_memory_MB': peak_memory,
            'allocated_memory_MB': allocated_memory
        }
    
    def measure_inference_latency(self, model, input_shape=(1, 1, 64, 64, 64), num_runs=100):
        """Measure inference latency"""
        model = model.to(self.device)
        model.eval()
        
        input_tensor = torch.randn(input_shape).to(self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_tensor)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Measure
        times = []
        for _ in range(num_runs):
            start = time.time()
            
            with torch.no_grad():
                _ = model(input_tensor)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.time()
            times.append(end - start)
        
        return {
            'mean_latency_ms': np.mean(times) * 1000,
            'std_latency_ms': np.std(times) * 1000,
            'min_latency_ms': np.min(times) * 1000,
            'max_latency_ms': np.max(times) * 1000,
            'throughput_fps': 1.0 / np.mean(times)
        }
    
    def profile_model(self, model, model_name, input_shape=(1, 1, 64, 64, 64)):
        """Complete profiling of a model"""
        print(f"\n{'='*80}")
        print(f"Profiling: {model_name}")
        print(f"{'='*80}\n")
        
        profile_results = {'model': model_name}
        
        # Parameters
        param_stats = self.count_parameters(model)
        profile_results.update(param_stats)
        print(f"Parameters: {param_stats['total_params_M']:.2f}M ({param_stats['trainable_params_M']:.2f}M trainable)")
        
        # FLOPs
        flop_stats = self.estimate_flops(model, input_shape)
        profile_results.update(flop_stats)
        print(f"FLOPs: {flop_stats['flops']}")
        
        # Memory
        memory_stats = self.measure_memory(model, input_shape)
        profile_results.update(memory_stats)
        print(f"Peak Memory: {memory_stats.get('peak_memory_MB', 'N/A')} MB")
        
        # Latency
        latency_stats = self.measure_inference_latency(model, input_shape)
        profile_results.update(latency_stats)
        print(f"Inference Latency: {latency_stats['mean_latency_ms']:.2f} ± {latency_stats['std_latency_ms']:.2f} ms")
        print(f"Throughput: {latency_stats['throughput_fps']:.2f} FPS")
        
        self.results[model_name] = profile_results
        
        return profile_results
    
    def compare_models(self, models_dict, input_shape=(1, 1, 64, 64, 64)):
        """Profile and compare multiple models"""
        all_results = []
        
        for model_name, model in models_dict.items():
            results = self.profile_model(model, model_name, input_shape)
            all_results.append(results)
        
        # Create comparison table
        df = pd.DataFrame(all_results)
        
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}\n")
        print(df.to_string(index=False))
        
        return df


# ============================================================================
# MAIN EXECUTION SCRIPT
# ============================================================================

def main():
    """
    Run comprehensive evaluation suite
    """
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║       COMPREHENSIVE EVALUATION SUITE FOR PAPER SUBMISSION           ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    This suite will run:
    1. Multiple seed experiments (5 runs per model)
    2. Statistical significance tests
    3. Strong baseline comparisons
    4. Factorial ablation study (48 configurations)
    5. Calibration analysis (ECE, Brier, reliability)
    6. Computational profiling
    
    Estimated time: 24-48 hours depending on hardware
    """)
    
    # Setup
    BASE_DIR = Path(r"C:\Users\HP\EDI")
    OUTPUT_DIR = BASE_DIR / "comprehensive_evaluation_results"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 1. Multi-seed experiments
    print("\n" + "="*80)
    print("STAGE 1: MULTI-SEED EXPERIMENTS")
    print("="*80)
    
    seed_exp = MultiSeedExperiment(OUTPUT_DIR / "multi_seed", num_seeds=5)
    
    # TODO: Run experiments for final model and baselines
    
    # 2. Factorial ablation
    print("\n" + "="*80)
    print("STAGE 2: FACTORIAL ABLATION")
    print("="*80)
    
    ablation = FactorialAblation(OUTPUT_DIR / "factorial_ablation")
    
    # TODO: Run factorial ablation
    
    # 3. Calibration analysis
    print("\n" + "="*80)
    print("STAGE 3: CALIBRATION ANALYSIS")
    print("="*80)
    
    calibration = CalibrationAnalysis(OUTPUT_DIR / "calibration")
    
    # TODO: Run calibration analysis
    
    # 4. Computational profiling
    print("\n" + "="*80)
    print("STAGE 4: COMPUTATIONAL PROFILING")
    print("="*80)
    
    profiler = ComputationalProfiler(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # TODO: Profile all models
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                    EVALUATION COMPLETE                               ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    Results saved to: {OUTPUT_DIR}
    """)


if __name__ == "__main__":
    main()
