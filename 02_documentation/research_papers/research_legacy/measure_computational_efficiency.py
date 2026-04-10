"""
Computational Efficiency Comparison: Trial Model vs Final Model
================================================================

This script measures and compares computational efficiency metrics between
trial.py model and final_model.py, including:
- Parameters count
- FLOPs (Floating Point Operations)
- Model size
- Inference time (CPU & GPU)
- Memory consumption
- Throughput

Author: Research Team
Date: November 2025
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

# Import models
sys.path.append(str(Path(__file__).parent.parent))
from trials.trial import HybridMiniSwin3D  # Trial model
from final_model import HybridMiniSwin2D5_CSRF  # Final model

# Try to import thop for FLOPs calculation
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: thop not installed. FLOPs calculation will be skipped.")
    print("   Install with: pip install thop")
    THOP_AVAILABLE = False

# Colors
TRIAL_COLOR = '#e74c3c'
FINAL_COLOR = '#27ae60'
IMPROVEMENT_COLOR = '#3498db'

# Output directory
OUTPUT_DIR = Path("paper_figures/computational_efficiency")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 80)
print("COMPUTATIONAL EFFICIENCY ANALYSIS: TRIAL vs FINAL MODEL")
print("=" * 80)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("=" * 80)

# ============================================================================
# 1. COUNT PARAMETERS
# ============================================================================

def count_parameters(model, name="Model"):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{name} Parameters:")
    print(f"  Total:      {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable:  {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'total_millions': total_params / 1e6,
        'trainable_millions': trainable_params / 1e6
    }

def get_model_size(model, name="Model"):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    print(f"\n{name} Size:")
    print(f"  Parameters: {param_size/1024/1024:.2f} MB")
    print(f"  Buffers:    {buffer_size/1024/1024:.2f} MB")
    print(f"  Total:      {size_mb:.2f} MB")
    
    return {
        'param_size_mb': param_size / 1024 / 1024,
        'buffer_size_mb': buffer_size / 1024 / 1024,
        'total_size_mb': size_mb
    }

# Initialize models
print("\n" + "=" * 80)
print("INITIALIZING MODELS")
print("=" * 80)

print("\n1. Trial Model (HybridMiniSwin3D)...")
trial_model = HybridMiniSwin3D(in_channels=1, embed_dim=96, num_heads=4, depth=3)
trial_model.eval()

print("\n2. Final Model (HybridMiniSwin2D5_CSRF)...")
final_model = HybridMiniSwin2D5_CSRF()
final_model.eval()

# Count parameters
print("\n" + "=" * 80)
print("PARAMETER ANALYSIS")
print("=" * 80)

trial_params = count_parameters(trial_model, "Trial Model")
final_params = count_parameters(final_model, "Final Model")

print(f"\nParameter Ratio: {final_params['total_millions'] / trial_params['total_millions']:.2f}×")

# Model sizes
print("\n" + "=" * 80)
print("MODEL SIZE ANALYSIS")
print("=" * 80)

trial_size = get_model_size(trial_model, "Trial Model")
final_size = get_model_size(final_model, "Final Model")

print(f"\nSize Ratio: {final_size['total_size_mb'] / trial_size['total_size_mb']:.2f}×")

# ============================================================================
# 2. FLOPS CALCULATION
# ============================================================================

print("\n" + "=" * 80)
print("FLOPS ANALYSIS")
print("=" * 80)

# Input size (typical for your dataset)
input_size = (1, 1, 64, 64, 64)  # (batch, channels, depth, height, width)
dummy_input = torch.randn(input_size)

trial_flops_data = {}
final_flops_data = {}

if THOP_AVAILABLE:
    try:
        print("\n1. Trial Model FLOPs...")
        trial_flops, trial_params_check = profile(trial_model, inputs=(dummy_input,), verbose=False)
        trial_flops_formatted, trial_params_formatted = clever_format([trial_flops, trial_params_check], "%.3f")
        print(f"  FLOPs: {trial_flops_formatted}")
        print(f"  Params: {trial_params_formatted}")
        trial_flops_data = {
            'flops': trial_flops,
            'flops_gflops': trial_flops / 1e9,
            'flops_formatted': trial_flops_formatted
        }
    except Exception as e:
        print(f"  ⚠️  Error calculating trial model FLOPs: {e}")
    
    try:
        print("\n2. Final Model FLOPs...")
        final_flops, final_params_check = profile(final_model, inputs=(dummy_input,), verbose=False)
        final_flops_formatted, final_params_formatted = clever_format([final_flops, final_params_check], "%.3f")
        print(f"  FLOPs: {final_flops_formatted}")
        print(f"  Params: {final_params_formatted}")
        final_flops_data = {
            'flops': final_flops,
            'flops_gflops': final_flops / 1e9,
            'flops_formatted': final_flops_formatted
        }
        
        if trial_flops_data:
            print(f"\nFLOPs Ratio: {final_flops / trial_flops:.2f}×")
    except Exception as e:
        print(f"  ⚠️  Error calculating final model FLOPs: {e}")
else:
    print("⚠️  Skipping FLOPs calculation (thop not installed)")

# ============================================================================
# 3. INFERENCE TIME BENCHMARKING
# ============================================================================

def benchmark_inference(model, device, input_size, num_warmup=10, num_runs=100, name="Model"):
    """Benchmark inference time"""
    model = model.to(device)
    model.eval()
    
    # Warmup
    dummy_input = torch.randn(input_size).to(device)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Actual benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    print(f"\n{name} on {device.type.upper()}:")
    print(f"  Mean:   {times.mean():.2f} ms")
    print(f"  Std:    {times.std():.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Min:    {times.min():.2f} ms")
    print(f"  Max:    {times.max():.2f} ms")
    print(f"  Throughput: {1000 / times.mean():.2f} vol/sec")
    
    return {
        'mean_ms': times.mean(),
        'std_ms': times.std(),
        'median_ms': np.median(times),
        'min_ms': times.min(),
        'max_ms': times.max(),
        'throughput': 1000 / times.mean()
    }

print("\n" + "=" * 80)
print("INFERENCE TIME BENCHMARKING")
print("=" * 80)

# CPU benchmarking
print("\nCPU Benchmarking (100 runs)...")
trial_cpu_times = benchmark_inference(trial_model, torch.device('cpu'), input_size, 
                                     num_warmup=5, num_runs=100, name="Trial Model")
final_cpu_times = benchmark_inference(final_model, torch.device('cpu'), input_size, 
                                     num_warmup=5, num_runs=100, name="Final Model")

print(f"\nCPU Speed Ratio: {final_cpu_times['mean_ms'] / trial_cpu_times['mean_ms']:.2f}×")

# GPU benchmarking (if available)
trial_gpu_times = {}
final_gpu_times = {}

if torch.cuda.is_available():
    print("\nGPU Benchmarking (100 runs)...")
    trial_gpu_times = benchmark_inference(trial_model, DEVICE, input_size, 
                                         num_warmup=10, num_runs=100, name="Trial Model")
    final_gpu_times = benchmark_inference(final_model, DEVICE, input_size, 
                                         num_warmup=10, num_runs=100, name="Final Model")
    
    print(f"\nGPU Speed Ratio: {final_gpu_times['mean_ms'] / trial_gpu_times['mean_ms']:.2f}×")
    print(f"Trial GPU Speedup vs CPU: {trial_cpu_times['mean_ms'] / trial_gpu_times['mean_ms']:.2f}×")
    print(f"Final GPU Speedup vs CPU: {final_cpu_times['mean_ms'] / final_gpu_times['mean_ms']:.2f}×")

# ============================================================================
# 4. MEMORY CONSUMPTION
# ============================================================================

def measure_memory(model, device, input_size, name="Model"):
    """Measure GPU memory consumption"""
    if device.type != 'cuda':
        print(f"⚠️  GPU not available, skipping memory measurement for {name}")
        return {}
    
    model = model.to(device)
    model.eval()
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    dummy_input = torch.randn(input_size).to(device)
    
    memory_before = torch.cuda.memory_allocated(device) / 1024 / 1024  # MB
    
    with torch.no_grad():
        output = model(dummy_input)
    
    memory_after = torch.cuda.memory_allocated(device) / 1024 / 1024  # MB
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024  # MB
    
    print(f"\n{name} GPU Memory:")
    print(f"  Before inference: {memory_before:.2f} MB")
    print(f"  After inference:  {memory_after:.2f} MB")
    print(f"  Peak memory:      {peak_memory:.2f} MB")
    print(f"  Memory delta:     {memory_after - memory_before:.2f} MB")
    
    return {
        'before_mb': memory_before,
        'after_mb': memory_after,
        'peak_mb': peak_memory,
        'delta_mb': memory_after - memory_before
    }

if torch.cuda.is_available():
    print("\n" + "=" * 80)
    print("GPU MEMORY CONSUMPTION")
    print("=" * 80)
    
    trial_memory = measure_memory(trial_model, DEVICE, input_size, "Trial Model")
    final_memory = measure_memory(final_model, DEVICE, input_size, "Final Model")
    
    if trial_memory and final_memory:
        print(f"\nMemory Ratio: {final_memory['peak_mb'] / trial_memory['peak_mb']:.2f}×")

# ============================================================================
# 5. COMPILE RESULTS
# ============================================================================

results = {
    'trial_model': {
        'parameters': trial_params,
        'model_size': trial_size,
        'flops': trial_flops_data,
        'cpu_inference': trial_cpu_times,
        'gpu_inference': trial_gpu_times if torch.cuda.is_available() else {},
        'gpu_memory': trial_memory if torch.cuda.is_available() else {}
    },
    'final_model': {
        'parameters': final_params,
        'model_size': final_size,
        'flops': final_flops_data,
        'cpu_inference': final_cpu_times,
        'gpu_inference': final_gpu_times if torch.cuda.is_available() else {},
        'gpu_memory': final_memory if torch.cuda.is_available() else {}
    },
    'ratios': {
        'parameters_ratio': final_params['total_millions'] / trial_params['total_millions'],
        'size_ratio': final_size['total_size_mb'] / trial_size['total_size_mb'],
        'cpu_time_ratio': final_cpu_times['mean_ms'] / trial_cpu_times['mean_ms'],
    }
}

if THOP_AVAILABLE and trial_flops_data and final_flops_data:
    results['ratios']['flops_ratio'] = final_flops / trial_flops

if torch.cuda.is_available() and trial_gpu_times and final_gpu_times:
    results['ratios']['gpu_time_ratio'] = final_gpu_times['mean_ms'] / trial_gpu_times['mean_ms']
    results['ratios']['gpu_memory_ratio'] = final_memory['peak_mb'] / trial_memory['peak_mb']

# Save results
results_path = OUTPUT_DIR / "computational_efficiency_results.json"
with open(results_path, 'w') as f:
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    json.dump(results, f, indent=2, default=convert)

print(f"\n✅ Results saved to: {results_path}")

print("\n" + "=" * 80)
print("COMPUTATIONAL EFFICIENCY ANALYSIS COMPLETE")
print("=" * 80)
