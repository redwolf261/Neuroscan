"""
Empirical Computational Profiling for NeuroScan:
Measures real parameters, VRAM usage, and timing benchmarks across:
- UNet3D-v5 Trunk
- 8-fold Test-Time Augmentation (TTA)
- MRD dense linear readout
- 27-feature extraction & A2 meta-classifier
"""

import sys, time, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'novelty' / 'pairwise_evidence_geometry'))

from neuroscan_3d_v5 import UNet3D_v5

def profile_parameters():
    print("=" * 60)
    print("1. PARAMETER PROFILE")
    print("=" * 60)
    
    model = UNet3D_v5(4, 3)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Module-level breakdown
    submodules = {
        "Encoder 1": model.enc1,
        "Encoder 2": model.enc2,
        "Encoder 3": model.enc3,
        "Bottleneck": model.bottleneck,
        "Attention Gate 1": model.attn_gate1,
        "Decoder 3": (model.upconv3, model.dec3),
        "Decoder 2": (model.upconv2, model.dec2),
        "Decoder 1 (D1)": (model.upconv1, model.dec1),
        "Segmentation Head": model.seg_head,
        "Evidential Head": model.evidential_head,
        "Boundary Head": model.boundary_head,
        "Aux Heads (2 & 3)": (model.aux_head2, model.aux_head3)
    }
    
    print(f"UNet3D-v5 Total Parameters: {total_params:,} ({total_params/1e6:.3f}M)")
    for name, mods in submodules.items():
        if isinstance(mods, tuple):
            p_cnt = sum(sum(p.numel() for p in m.parameters()) for m in mods)
        else:
            p_cnt = sum(p.numel() for p in mods.parameters())
        print(f"  - {name:<22}: {p_cnt:>10,} ({p_cnt/1e6:.3f}M, {100*p_cnt/total_params:5.1f}%)")
        
    # MRD & A2 Parameter count
    mrd_params_per_seed = 32 + 1  # 32 weights + 1 bias
    mrd_total_5_seeds = mrd_params_per_seed * 5
    a2_params = 27 + 1  # 27 linear weights + 1 intercept
    
    print("\nLightweight Readout Parameters:")
    print(f"  - MRD Linear Readout (1 seed) : {mrd_params_per_seed} parameters (32 weights + 1 bias)")
    print(f"  - MRD Multi-Seed (5 seeds)   : {mrd_total_5_seeds} parameters ({mrd_total_5_seeds/1e6:.6f}M)")
    print(f"  - A2 Meta-Classifier (LogReg) : {a2_params} parameters (27 weights + 1 bias)")
    print(f"  - Total Non-Backbone Params   : {mrd_total_5_seeds + a2_params} parameters (0.00019M, ~0.003% of trunk)")
    
    return total_params, mrd_total_5_seeds, a2_params

def profile_vram_and_latency():
    print("\n" + "=" * 60)
    print("2. VRAM AND LATENCY BENCHMARKS")
    print("=" * 60)
    
    cuda_avail = torch.cuda.is_available()
    dev = torch.device('cuda' if cuda_avail else 'cpu')
    print(f"Execution Device: {dev}")
    if cuda_avail:
        dev_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Hardware: {dev_name} ({total_vram_gb:.2f} GB VRAM)")
    
    patch_size = (1, 4, 128, 128, 128)
    model = UNet3D_v5(4, 3).to(dev).eval()
    
    # 2A. Model weights footprint
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    weight_mb = weight_bytes / (1024**2)
    print(f"Model static weights footprint: {weight_mb:.2f} MB")
    
    # Dummy input
    x = torch.randn(patch_size, device=dev)
    
    # Warmup
    print("Warming up GPU / CPU...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
            if cuda_avail:
                torch.cuda.synchronize()
                
    # 2B. Single Forward Pass Latency & VRAM
    if cuda_avail:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    n_runs = 20
    times_single = []
    
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            out = model(x)
            if cuda_avail:
                torch.cuda.synchronize()
            times_single.append(time.perf_counter() - t0)
            
    mean_t_single = np.mean(times_single)
    std_t_single = np.std(times_single)
    min_t_single = np.min(times_single)
    
    inf_peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if cuda_avail else 0.0
    print(f"Single Forward Pass (128^3 patch):")
    print(f"  - Latency  : {mean_t_single*1000:.2f} ms ± {std_t_single*1000:.2f} ms (min: {min_t_single*1000:.2f} ms)")
    print(f"  - Peak VRAM: {inf_peak_vram_mb:.2f} MB")
    
    # 2C. 8-Fold TTA Latency
    times_tta = []
    # 8 dihedral transformations (canonical + 7 spatial flips)
    flips = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]
    
    if cuda_avail:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
    with torch.no_grad():
        for _ in range(10):
            t0 = time.perf_counter()
            accum = torch.zeros((1, 3, 128, 128, 128), device=dev)
            for f in flips:
                x_f = torch.flip(x, f) if f else x
                out_f = model(x_f)
                p_f = out_f['probs']
                if f:
                    p_f = torch.flip(p_f, f)
                accum += p_f
            accum /= len(flips)
            if cuda_avail:
                torch.cuda.synchronize()
            times_tta.append(time.perf_counter() - t0)
            
    mean_t_tta = np.mean(times_tta)
    std_t_tta = np.std(times_tta)
    tta_peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if cuda_avail else 0.0
    print(f"8-Fold TTA Inference (128^3 patch):")
    print(f"  - Latency  : {mean_t_tta*1000:.2f} ms ({mean_t_tta:.3f} s) ± {std_t_tta*1000:.2f} ms")
    print(f"  - Peak VRAM: {tta_peak_vram_mb:.2f} MB")
    
    # 2D. MRD Dense Readout Latency
    # MRD evaluates w_R^T D1 + b_R on dec1 tensor (32, 128, 128, 128)
    dec1 = torch.randn((1, 32, 128, 128, 128), device=dev)
    w_mrd = torch.randn((5, 32, 1, 1, 1), device=dev)  # 5 seeds
    b_mrd = torch.randn(5, device=dev)
    
    times_mrd = []
    with torch.no_grad():
        for _ in range(30):
            t0 = time.perf_counter()
            # 5 seeds 1x1x1 conv or matmul
            r_maps = []
            for s in range(5):
                z = F.conv3d(dec1, w_mrd[s:s+1], b_mrd[s:s+1])
                r = torch.sigmoid(z)
                r_maps.append(r)
            if cuda_avail:
                torch.cuda.synchronize()
            times_mrd.append(time.perf_counter() - t0)
            
    mean_t_mrd = np.mean(times_mrd)
    print(f"MRD Readout (5-Seed Ensemble on D1):")
    print(f"  - Latency  : {mean_t_mrd*1000:.2f} ms ({mean_t_mrd*1000:.3f} ms)")
    
    # 2E. A2 Feature Extraction & Classifier Latency
    # Simulate realistic candidate components: typically 2 to 10 candidates per scan
    times_a2 = []
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    # Mock trained model with 27 features
    clf.coef_ = np.random.randn(1, 27)
    clf.intercept_ = np.array([0.0])
    clf.classes_ = np.array([0, 1])
    
    # Simulate feature matrix for 6 candidate components
    feats = np.random.randn(6, 27)
    for _ in range(100):
        t0 = time.perf_counter()
        _ = clf.predict_proba(feats)[:, 1]
        times_a2.append(time.perf_counter() - t0)
        
    mean_t_a2 = np.mean(times_a2)
    print(f"A2 Classifier Inference (6 candidates, 27 features):")
    print(f"  - Latency  : {mean_t_a2*1000:.3f} ms ({mean_t_a2*1e6:.1f} µs)")
    
    # 2F. Training VRAM Benchmark (Forward + Backward on Batch size 1)
    if cuda_avail:
        model.train()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        x_tr = torch.randn(patch_size, device=dev, requires_grad=True)
        target = torch.randint(0, 2, (1, 3, 128, 128, 128), device=dev).float()
        
        optimizer.zero_grad()
        out_tr = model(x_tr)
        loss = F.binary_cross_entropy_with_logits(out_tr['probs'], target)
        loss.backward()
        optimizer.step()
        
        train_peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"\nTraining Peak VRAM (Batch 1, 128^3 patch, Adam optimizer):")
        print(f"  - Peak VRAM: {train_peak_vram_mb:.2f} MB ({train_peak_vram_mb/1024:.2f} GB)")
    else:
        train_peak_vram_mb = 0.0
        
    return {
        'mean_t_single': mean_t_single,
        'mean_t_tta': mean_t_tta,
        'mean_t_mrd': mean_t_mrd,
        'mean_t_a2': mean_t_a2,
        'weight_mb': weight_mb,
        'inf_peak_vram_mb': inf_peak_vram_mb,
        'tta_peak_vram_mb': tta_peak_vram_mb,
        'train_peak_vram_mb': train_peak_vram_mb
    }

if __name__ == '__main__':
    profile_parameters()
    profile_vram_and_latency()
