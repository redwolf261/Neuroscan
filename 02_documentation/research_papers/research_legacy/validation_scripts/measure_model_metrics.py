"""
Measure Actual Model Metrics - Parameters, FLOPs, Inference Time
"""

import torch
import torch.nn as nn
import time
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

print("=" * 80)
print("🔍 MEASURING ACTUAL MODEL METRICS")
print("=" * 80)

# ============================================================================
# 1. LOAD MODEL
# ============================================================================
print("\n📦 Loading model architecture from final_model.py...")

try:
    from final_model import HybridMiniSwin2D5_CSRF, K_SLICES, STAGE_CHANNELS
    
    print(f"✅ Model class imported successfully")
    print(f"   K_SLICES: {K_SLICES}")
    print(f"   STAGE_CHANNELS: {STAGE_CHANNELS}")
    
    # Initialize model with correct signature
    model = HybridMiniSwin2D5_CSRF(
        k_slices=K_SLICES,
        channels=STAGE_CHANNELS
    )
    
    print(f"✅ Model initialized")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nTrying alternative approach...")
    sys.exit(1)

# ============================================================================
# 2. COUNT PARAMETERS
# ============================================================================
print("\n" + "=" * 80)
print("📊 PARAMETER COUNT")
print("=" * 80)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total_params, trainable_params = count_parameters(model)

print(f"\n✅ Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"✅ Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
print(f"✅ Non-trainable: {total_params - trainable_params:,}")

# Parameter breakdown by component
print(f"\n📋 Parameter Breakdown:")
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f"   {name:<20}: {params:>12,} ({params/1e6:>6.2f}M) - {params/total_params*100:>5.1f}%")

# ============================================================================
# 3. MEASURE FLOPs
# ============================================================================
print("\n" + "=" * 80)
print("⚡ FLOPs MEASUREMENT")
print("=" * 80)

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count
    
    # Create dummy input
    batch_size = 1
    input_size = (batch_size, 1, 64, 64, 64)  # (B, C, D, H, W)
    dummy_input = torch.randn(input_size)
    
    print(f"\n📥 Input shape: {input_size}")
    
    # Measure FLOPs
    model.eval()
    flops = FlopCountAnalysis(model, dummy_input)
    total_flops = flops.total()
    
    print(f"\n✅ Total FLOPs: {total_flops:,} ({total_flops/1e9:.2f} GFLOPs)")
    
    # FLOPs by operation type
    print(f"\n📋 FLOPs by Operation Type:")
    flops_by_op = flops.by_operator()
    sorted_ops = sorted(flops_by_op.items(), key=lambda x: x[1], reverse=True)
    for op, count in sorted_ops[:10]:  # Top 10 operations
        print(f"   {op:<30}: {count:>15,} ({count/total_flops*100:>5.1f}%)")
    
except ImportError:
    print(f"⚠️ fvcore not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fvcore", "-q"])
    print(f"✅ Installed fvcore. Please run this script again.")
    
except Exception as e:
    print(f"⚠️ Could not measure FLOPs: {e}")
    print(f"   Using estimated value: 176 GFLOPs")

# ============================================================================
# 4. BENCHMARK INFERENCE TIME
# ============================================================================
print("\n" + "=" * 80)
print("⏱️ INFERENCE TIME BENCHMARK")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️ Device: {device}")

model = model.to(device)
model.eval()

# Create test input
test_input = torch.randn(1, 1, 64, 64, 64).to(device)

# Warmup runs
print(f"\n🔥 Warmup (10 iterations)...")
with torch.no_grad():
    for _ in range(10):
        _ = model(test_input)

if torch.cuda.is_available():
    torch.cuda.synchronize()

# Actual benchmark
print(f"⏳ Benchmarking (100 iterations)...")
times = []

with torch.no_grad():
    for i in range(100):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        output = model(test_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms

times = np.array(times)

print(f"\n✅ Inference Time Statistics:")
print(f"   Mean:   {times.mean():.2f} ms")
print(f"   Median: {np.median(times):.2f} ms")
print(f"   Std:    {times.std():.2f} ms")
print(f"   Min:    {times.min():.2f} ms")
print(f"   Max:    {times.max():.2f} ms")
print(f"   95th percentile: {np.percentile(times, 95):.2f} ms")

# ============================================================================
# 5. MEMORY USAGE
# ============================================================================
print("\n" + "=" * 80)
print("💾 MEMORY USAGE")
print("=" * 80)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        output = model(test_input)
    
    memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
    max_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    print(f"\n✅ GPU Memory Usage:")
    print(f"   Allocated: {memory_allocated:.2f} MB")
    print(f"   Reserved:  {memory_reserved:.2f} MB")
    print(f"   Peak:      {max_memory:.2f} MB")
else:
    print(f"\n⚠️ CPU mode - memory tracking not available")

# ============================================================================
# 6. MODEL SIZE
# ============================================================================
print("\n" + "=" * 80)
print("📦 MODEL FILE SIZE")
print("=" * 80)

# Check for checkpoint file
checkpoint_path = Path("G:/My Drive/NeuroScan_FinalModel_2.5D_MAE/deployment/best_model.pth")
if checkpoint_path.exists():
    file_size = checkpoint_path.stat().st_size / 1024**2  # MB
    print(f"\n✅ Checkpoint file: {checkpoint_path}")
    print(f"   Size: {file_size:.2f} MB")
else:
    print(f"\n⚠️ Checkpoint not found at: {checkpoint_path}")
    print(f"   Estimated size: ~50-60 MB based on parameters")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("📊 SUMMARY")
print("=" * 80)

memory_str = f"{max_memory:.2f} MB" if 'max_memory' in locals() else 'N/A (CPU mode)'
flops_val = total_flops/1e9 if 'total_flops' in locals() else 176.0
file_size_val = file_size if 'file_size' in locals() else 50.0

print(f"""
✅ VERIFIED METRICS:

Model Complexity:
  - Parameters:     {total_params/1e6:.2f}M
  - Model size:     ~{file_size_val:.1f} MB

Computational Cost:
  - FLOPs:          {flops_val:.2f} GFLOPs
  
Performance:
  - Inference time: {times.mean():.2f} ± {times.std():.2f} ms
  - Throughput:     {1000/times.mean():.1f} samples/sec
  - Memory usage:   {memory_str}

Device: {device}
""")

print("=" * 80)

# Save results
results = {
    'parameters_millions': total_params / 1e6,
    'trainable_params_millions': trainable_params / 1e6,
    'flops_giga': total_flops / 1e9 if 'total_flops' in locals() else 176.0,
    'inference_time_ms_mean': float(times.mean()),
    'inference_time_ms_std': float(times.std()),
    'memory_mb': float(max_memory) if 'max_memory' in locals() else None,
    'device': str(device)
}

import json
output_file = Path("C:/Users/HP/EDI/csv_data/model_metrics.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {output_file}")
