"""
Inference Benchmark Suite
Measures latency, throughput, and preprocessing time on exact hardware
"""

import torch
import time
import numpy as np
import platform
import psutil
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from final_model import HybridMiniSwin2D5_CSRF


def get_hardware_info():
    """Get exact hardware specifications"""
    info = {
        'os': platform.system(),
        'os_version': platform.version(),
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(logical=False),
        'cpu_threads': psutil.cpu_count(logical=True),
        'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
    }
    
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        info['cuda_version'] = torch.version.cuda
    else:
        info['gpu_name'] = 'N/A'
        info['gpu_memory_gb'] = 'N/A'
        info['cuda_version'] = 'N/A'
    
    info['pytorch_version'] = torch.__version__
    
    return info


def benchmark_inference(model, input_shape=(1, 1, 64, 64, 64), num_runs=50, warmup=10, device='cuda'):
    """
    Benchmark inference latency and throughput
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (B, C, D, H, W)
        num_runs: Number of inference runs
        warmup: Number of warmup runs
        device: 'cuda' or 'cpu'
    
    Returns:
        dict with mean, std, median, min, max latency in ms
    """
    model.to(device)
    model.eval()
    
    # Create dummy input
    dummy = torch.randn(*input_shape).to(device)
    
    print(f"\n{'='*100}")
    print(f"BENCHMARKING ON {device.upper()}")
    print(f"{'='*100}")
    print(f"Input shape: {input_shape}")
    print(f"Warmup runs: {warmup}")
    print(f"Benchmark runs: {num_runs}")
    
    # Warmup
    print(f"\nWarming up...")
    with torch.no_grad():
        for i in range(warmup):
            _ = model(dummy)
            if device == 'cuda':
                torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running benchmark...")
    times = []
    with torch.no_grad():
        for i in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            output = model(dummy)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            times.append(latency_ms)
            
            # Print progress every 10 runs
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_runs} runs completed")
    
    times = np.array(times)
    
    results = {
        'device': device,
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'median_ms': np.median(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'throughput_images_per_sec': 1000.0 / np.mean(times)  # Images/sec
    }
    
    return results, times


def benchmark_preprocessing(data_path, num_samples=30):
    """
    Benchmark preprocessing pipeline time
    
    Args:
        data_path: Path to sample data
        num_samples: Number of samples to process
    
    Returns:
        dict with preprocessing times
    """
    import nibabel as nib
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
        Spacingd, NormalizeIntensityd, Resized
    )
    
    # Define preprocessing pipeline (same as training)
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        Resized(keys=["image"], spatial_size=(64, 64, 64), mode="trilinear"),
    ])
    
    print(f"\n{'='*100}")
    print(f"BENCHMARKING PREPROCESSING")
    print(f"{'='*100}")
    print(f"Number of samples: {num_samples}")
    
    # Find sample files
    sample_files = list(Path(data_path).rglob("*FLAIR.nii.gz"))[:num_samples]
    
    if len(sample_files) == 0:
        print(f"⚠️  No FLAIR files found in {data_path}")
        return None
    
    print(f"Found {len(sample_files)} sample files")
    
    times = []
    for i, file_path in enumerate(sample_files):
        data_dict = {"image": str(file_path)}
        
        start = time.perf_counter()
        processed = transforms(data_dict)
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        times.append(latency_ms)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(sample_files)} samples processed")
    
    times = np.array(times)
    
    results = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'median_ms': np.median(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
    }
    
    return results


def benchmark_batch_sizes(model, device='cuda', max_batch_size=8):
    """
    Benchmark different batch sizes
    
    Args:
        model: PyTorch model
        device: 'cuda' or 'cpu'
        max_batch_size: Maximum batch size to test
    
    Returns:
        dict with results per batch size
    """
    model.to(device)
    model.eval()
    
    print(f"\n{'='*100}")
    print(f"BATCH SIZE SWEEP")
    print(f"{'='*100}")
    
    batch_results = {}
    
    for batch_size in [1, 2, 4, 8]:
        if batch_size > max_batch_size:
            break
        
        input_shape = (batch_size, 1, 64, 64, 64)
        
        try:
            print(f"\nTesting batch size: {batch_size}")
            results, _ = benchmark_inference(
                model, 
                input_shape=input_shape,
                num_runs=30,
                warmup=5,
                device=device
            )
            batch_results[batch_size] = results
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ⚠️  Batch size {batch_size} exceeds GPU memory")
                break
            else:
                raise e
    
    return batch_results


def print_results(hardware_info, gpu_results, cpu_results, preproc_results, batch_results):
    """Print formatted benchmark results"""
    
    print(f"\n{'='*100}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*100}")
    
    # Hardware info
    print(f"\n{'HARDWARE SPECIFICATIONS':^100}")
    print(f"{'='*100}")
    print(f"  OS:              {hardware_info['os']} {hardware_info['os_version']}")
    print(f"  Processor:       {hardware_info['processor']}")
    print(f"  CPU Cores:       {hardware_info['cpu_count']} physical, {hardware_info['cpu_threads']} threads")
    print(f"  RAM:             {hardware_info['ram_gb']} GB")
    print(f"  GPU:             {hardware_info['gpu_name']}")
    print(f"  GPU Memory:      {hardware_info['gpu_memory_gb']} GB")
    print(f"  CUDA Version:    {hardware_info['cuda_version']}")
    print(f"  PyTorch Version: {hardware_info['pytorch_version']}")
    
    # GPU inference
    if gpu_results:
        print(f"\n{'GPU INFERENCE (RTX 2050)':^100}")
        print(f"{'='*100}")
        print(f"  Input Shape:     (1, 1, 64, 64, 64)")
        print(f"  Mean Latency:    {gpu_results['mean_ms']:.2f} ± {gpu_results['std_ms']:.2f} ms")
        print(f"  Median Latency:  {gpu_results['median_ms']:.2f} ms")
        print(f"  Min Latency:     {gpu_results['min_ms']:.2f} ms")
        print(f"  Max Latency:     {gpu_results['max_ms']:.2f} ms")
        print(f"  Throughput:      {gpu_results['throughput_images_per_sec']:.2f} images/sec")
    
    # CPU inference
    if cpu_results:
        print(f"\n{'CPU INFERENCE':^100}")
        print(f"{'='*100}")
        print(f"  Input Shape:     (1, 1, 64, 64, 64)")
        print(f"  Mean Latency:    {cpu_results['mean_ms']:.2f} ± {cpu_results['std_ms']:.2f} ms")
        print(f"  Median Latency:  {cpu_results['median_ms']:.2f} ms")
        print(f"  Min Latency:     {cpu_results['min_ms']:.2f} ms")
        print(f"  Max Latency:     {cpu_results['max_ms']:.2f} ms")
        print(f"  Throughput:      {cpu_results['throughput_images_per_sec']:.2f} images/sec")
    
    # Preprocessing
    if preproc_results:
        print(f"\n{'PREPROCESSING PIPELINE':^100}")
        print(f"{'='*100}")
        print(f"  Mean Time:       {preproc_results['mean_ms']:.2f} ± {preproc_results['std_ms']:.2f} ms")
        print(f"  Median Time:     {preproc_results['median_ms']:.2f} ms")
        print(f"  Min Time:        {preproc_results['min_ms']:.2f} ms")
        print(f"  Max Time:        {preproc_results['max_ms']:.2f} ms")
    
    # Total pipeline time
    if gpu_results and preproc_results:
        total_gpu = gpu_results['mean_ms'] + preproc_results['mean_ms']
        print(f"\n{'TOTAL END-TO-END PIPELINE':^100}")
        print(f"{'='*100}")
        print(f"  Preprocessing:   {preproc_results['mean_ms']:.2f} ms")
        print(f"  Inference (GPU): {gpu_results['mean_ms']:.2f} ms")
        print(f"  Total (GPU):     {total_gpu:.2f} ms ({1000.0/total_gpu:.2f} images/sec)")
        
        if cpu_results:
            total_cpu = cpu_results['mean_ms'] + preproc_results['mean_ms']
            print(f"  Total (CPU):     {total_cpu:.2f} ms ({1000.0/total_cpu:.2f} images/sec)")
    
    # Batch size results
    if batch_results:
        print(f"\n{'BATCH SIZE SWEEP (GPU)':^100}")
        print(f"{'='*100}")
        print(f"  {'Batch Size':<12} {'Mean (ms)':<15} {'Throughput (img/s)':<20} {'Memory Efficient':<20}")
        print(f"  {'-'*12} {'-'*15} {'-'*20} {'-'*20}")
        for bs, res in batch_results.items():
            throughput = bs * res['throughput_images_per_sec']
            memory_eff = throughput / (hardware_info['gpu_memory_gb'] * 1024)  # img/s per MB
            print(f"  {bs:<12} {res['mean_ms']:<15.2f} {throughput:<20.2f} {memory_eff:<20.4f}")
    
    print(f"\n{'='*100}")


def save_results(output_dir, hardware_info, gpu_results, cpu_results, preproc_results, batch_results, gpu_times, cpu_times):
    """Save benchmark results to files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    import json
    
    results = {
        'hardware': hardware_info,
        'gpu_inference': gpu_results,
        'cpu_inference': cpu_results,
        'preprocessing': preproc_results,
        'batch_sizes': batch_results,
    }
    
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save raw timing data
    if gpu_times is not None:
        np.save(output_dir / 'gpu_inference_times.npy', gpu_times)
    if cpu_times is not None:
        np.save(output_dir / 'cpu_inference_times.npy', cpu_times)
    
    # Save summary text
    with open(output_dir / 'benchmark_summary.txt', 'w') as f:
        f.write("="*100 + "\n")
        f.write("INFERENCE BENCHMARK RESULTS\n")
        f.write("="*100 + "\n\n")
        
        f.write("HARDWARE SPECIFICATIONS\n")
        f.write("-"*100 + "\n")
        for key, value in hardware_info.items():
            f.write(f"{key:20s}: {value}\n")
        
        f.write("\n")
        f.write("GPU INFERENCE (RTX 2050)\n")
        f.write("-"*100 + "\n")
        if gpu_results:
            f.write(f"Mean Latency:    {gpu_results['mean_ms']:.2f} ± {gpu_results['std_ms']:.2f} ms\n")
            f.write(f"Median Latency:  {gpu_results['median_ms']:.2f} ms\n")
            f.write(f"Throughput:      {gpu_results['throughput_images_per_sec']:.2f} images/sec\n")
        
        f.write("\n")
        f.write("CPU INFERENCE\n")
        f.write("-"*100 + "\n")
        if cpu_results:
            f.write(f"Mean Latency:    {cpu_results['mean_ms']:.2f} ± {cpu_results['std_ms']:.2f} ms\n")
            f.write(f"Median Latency:  {cpu_results['median_ms']:.2f} ms\n")
            f.write(f"Throughput:      {cpu_results['throughput_images_per_sec']:.2f} images/sec\n")
        
        f.write("\n")
        f.write("PREPROCESSING PIPELINE\n")
        f.write("-"*100 + "\n")
        if preproc_results:
            f.write(f"Mean Time:       {preproc_results['mean_ms']:.2f} ± {preproc_results['std_ms']:.2f} ms\n")
            f.write(f"Median Time:     {preproc_results['median_ms']:.2f} ms\n")
    
    print(f"\n✓ Saved results to: {output_dir}")


def main():
    """Run complete inference benchmark suite"""
    
    print("="*100)
    print("INFERENCE BENCHMARK SUITE")
    print("="*100)
    
    # Get hardware info
    print("\nDetecting hardware...")
    hardware_info = get_hardware_info()
    
    # Load model
    print("\nLoading model...")
    model = HybridMiniSwin2D5_CSRF(
        k_slices=5,
        channels=[32, 64, 128, 256, 512]
    )
    
    # Load checkpoint
    checkpoint_path = Path("seg_resume.pth")
    if checkpoint_path.exists():
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Checkpoint loaded")
    else:
        print("⚠️  No checkpoint found, using random weights")
    
    # Benchmark GPU inference
    gpu_results = None
    gpu_times = None
    if torch.cuda.is_available():
        print("\n" + "="*100)
        print("Starting GPU benchmark...")
        gpu_results, gpu_times = benchmark_inference(
            model,
            input_shape=(1, 1, 64, 64, 64),
            num_runs=50,
            warmup=10,
            device='cuda'
        )
    else:
        print("\n⚠️  CUDA not available, skipping GPU benchmark")
    
    # Benchmark CPU inference
    print("\n" + "="*100)
    print("Starting CPU benchmark...")
    cpu_results, cpu_times = benchmark_inference(
        model,
        input_shape=(1, 1, 64, 64, 64),
        num_runs=50,
        warmup=10,
        device='cpu'
    )
    
    # Benchmark preprocessing
    data_path = r"G:\My Drive\Dataset\PediMS\PediMS"
    preproc_results = None
    if Path(data_path).exists():
        preproc_results = benchmark_preprocessing(data_path, num_samples=30)
    else:
        print(f"\n⚠️  Data path not found: {data_path}")
        print("Skipping preprocessing benchmark")
    
    # Benchmark batch sizes (GPU only)
    batch_results = None
    if torch.cuda.is_available():
        batch_results = benchmark_batch_sizes(model, device='cuda', max_batch_size=8)
    
    # Print results
    print_results(hardware_info, gpu_results, cpu_results, preproc_results, batch_results)
    
    # Save results
    save_results(
        output_dir='research',
        hardware_info=hardware_info,
        gpu_results=gpu_results,
        cpu_results=cpu_results,
        preproc_results=preproc_results,
        batch_results=batch_results,
        gpu_times=gpu_times,
        cpu_times=cpu_times
    )
    
    print("\n" + "="*100)
    print("BENCHMARK COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
