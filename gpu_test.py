"""
GPU Test Script for RTX 5050

Diagnoses:
1. GPU detection
2. CUDA compute capability compatibility
3. PyTorch CUDA support
4. GPU memory
5. Training speed on GPU
"""

import torch
import sys

print("="*70)
print("NeuroScan GPU Test")
print("="*70)
print()

# Test 1: GPU Detection
print("Test 1: GPU Detection")
print("-" * 70)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Test 2: Simple tensor operation
    print("Test 2: Simple GPU Operation")
    print("-" * 70)
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✓ Matrix multiplication on GPU works")
        print()
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

    # Test 3: Training speed benchmark
    print("Test 3: Training Speed Benchmark")
    print("-" * 70)
    import time

    model = torch.nn.Linear(1024, 1024).cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    batch_size = 32
    num_batches = 100

    print(f"Running {num_batches} batches of size {batch_size}...")

    start = time.time()
    for i in range(num_batches):
        x = torch.randn(batch_size, 1024).cuda()
        y = torch.randn(batch_size, 1024).cuda()

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    elapsed = time.time() - start
    time_per_batch = elapsed / num_batches * 1000  # ms

    print(f"Time per batch: {time_per_batch:.2f} ms")
    print(f"Total time: {elapsed:.2f} seconds")
    print()

    print("="*70)
    print("✓ GPU is working correctly")
    print("="*70)
else:
    print("✗ CUDA not available")
    print()
    print("Troubleshooting:")
    print("  1. Check NVIDIA drivers: nvidia-smi")
    print("  2. Check CUDA is installed")
    print("  3. Try different PyTorch build:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    sys.exit(1)
