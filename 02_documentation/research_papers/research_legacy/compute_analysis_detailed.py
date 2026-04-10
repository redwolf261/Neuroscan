"""
Layerwise Parameter and FLOP Analysis Tool
==========================================
Generates detailed tables showing:
- Layer name, output shape, kernel size, Cin, Cout
- FLOPs (with formulas)
- Parameters
- Cumulative totals

Usage:
    python research/compute_analysis_detailed.py
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from final_model import HybridMiniSwin2D5_CSRF

def count_conv2d_flops(in_channels, out_channels, kernel_size, output_shape):
    """
    FLOPs for Conv2d = 2 * Cin * Cout * K * K * H_out * W_out
    (2 = multiply-add counts as 2 ops)
    """
    if isinstance(kernel_size, tuple):
        k_h, k_w = kernel_size
    else:
        k_h = k_w = kernel_size
    
    h_out, w_out = output_shape
    flops = 2 * in_channels * out_channels * k_h * k_w * h_out * w_out
    return flops

def count_batchnorm2d_flops(num_features, output_shape):
    """
    BatchNorm2D FLOPs = 2 * num_features * H * W
    (normalize + scale/shift)
    """
    h_out, w_out = output_shape
    flops = 2 * num_features * h_out * w_out
    return flops

def count_linear_flops(in_features, out_features, batch_size=1):
    """
    Linear FLOPs = 2 * in_features * out_features * batch_size
    """
    flops = 2 * in_features * out_features * batch_size
    return flops

def count_attention_flops(seq_len, embed_dim, num_heads):
    """
    Multi-Head Self-Attention FLOPs:
    - QKV projection: 3 * (2 * seq_len * embed_dim * embed_dim)
    - Attention matrix: 2 * num_heads * seq_len * seq_len * (embed_dim / num_heads)
    - Attention-Value: 2 * num_heads * seq_len * seq_len * (embed_dim / num_heads)
    - Output projection: 2 * seq_len * embed_dim * embed_dim
    """
    head_dim = embed_dim // num_heads
    
    # QKV projection
    qkv_flops = 3 * (2 * seq_len * embed_dim * embed_dim)
    
    # Attention scores (Q @ K^T)
    attn_flops = 2 * num_heads * seq_len * seq_len * head_dim
    
    # Attention @ V
    attn_v_flops = 2 * num_heads * seq_len * seq_len * head_dim
    
    # Output projection
    out_proj_flops = 2 * seq_len * embed_dim * embed_dim
    
    total_flops = qkv_flops + attn_flops + attn_v_flops + out_proj_flops
    return total_flops

def analyze_model_detailed(model, input_shape=(1, 1, 64, 64, 64)):
    """
    Perform detailed layer-by-layer analysis
    """
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Track all layers
    layer_stats = []
    total_params = 0
    total_flops = 0
    
    print("="*100)
    print("LAYERWISE PARAMETER AND FLOP ANALYSIS")
    print("="*100)
    print(f"Input shape: {input_shape}")
    print()
    
    # Analyze stem (2.5D Conv)
    print("=" * 100)
    print("STEM - 2.5D Convolutional Stem")
    print("=" * 100)
    
    stem = model.encoder.stem
    
    # Stem processes k=5 slices
    # slice_conv: Conv2d(1, 32, kernel=3)
    slice_conv = stem.slice_conv[0]  # Conv2d layer
    cin = slice_conv.in_channels
    cout = slice_conv.out_channels
    kernel = slice_conv.kernel_size[0]
    
    # Output shape after stem (assuming input 64x64x64, center 5 slices)
    h_out, w_out = 64, 64  # Same spatial size (padding=1)
    
    params_conv = cin * cout * kernel * kernel + cout  # weights + bias
    flops_conv = count_conv2d_flops(cin, cout, kernel, (h_out, w_out)) * 5  # 5 slices
    
    # BatchNorm
    bn = stem.slice_conv[1]
    params_bn = 2 * cout  # gamma, beta
    flops_bn = count_batchnorm2d_flops(cout, (h_out, w_out)) * 5
    
    # Slice attention (small FC layers)
    attn_conv1 = stem.slice_attention[1]  # Conv2d(32, 8)
    attn_conv2 = stem.slice_attention[3]  # Conv2d(8, 1)
    params_attn = (cout * (cout // 4) * 1 * 1 + (cout // 4)) + ((cout // 4) * 1 * 1 * 1 + 1)
    flops_attn = (count_conv2d_flops(cout, cout // 4, 1, (1, 1)) + 
                  count_conv2d_flops(cout // 4, 1, 1, (1, 1))) * 5
    
    stem_params = params_conv + params_bn + params_attn
    stem_flops = flops_conv + flops_bn + flops_attn
    
    layer_stats.append({
        'Layer': 'Stem.SliceConv',
        'Type': 'Conv2d',
        'Output Shape': f'(B, 32, 64, 64)',
        'Kernel': '3x3',
        'Cin': cin,
        'Cout': cout,
        'Params': params_conv,
        'FLOPs': flops_conv,
        'Formula': f'2*{cin}*{cout}*{kernel}²*64*64*5'
    })
    
    layer_stats.append({
        'Layer': 'Stem.BatchNorm',
        'Type': 'BN2d',
        'Output Shape': f'(B, 32, 64, 64)',
        'Kernel': '-',
        'Cin': cout,
        'Cout': cout,
        'Params': params_bn,
        'FLOPs': flops_bn,
        'Formula': f'2*{cout}*64*64*5'
    })
    
    layer_stats.append({
        'Layer': 'Stem.SliceAttention',
        'Type': 'SE-Attn',
        'Output Shape': f'(B, 32, 64, 64)',
        'Kernel': '1x1',
        'Cin': cout,
        'Cout': 1,
        'Params': params_attn,
        'FLOPs': flops_attn,
        'Formula': f'SE-reduction 32→8→1'
    })
    
    total_params += stem_params
    total_flops += stem_flops
    
    # Analyze encoder stages
    print("\n" + "=" * 100)
    print("ENCODER STAGES")
    print("=" * 100)
    
    channels = [32, 64, 128, 256, 512]
    spatial_sizes = [64, 32, 16, 8, 4]  # After each stage
    
    for stage_idx, stage in enumerate(model.encoder.stages):
        cin = channels[stage_idx]
        cout = channels[stage_idx + 1]
        h = spatial_sizes[stage_idx]
        w = spatial_sizes[stage_idx]
        h_out = spatial_sizes[stage_idx + 1]
        w_out = spatial_sizes[stage_idx + 1]
        
        print(f"\nStage {stage_idx + 1}: {cin} → {cout} channels, {h}x{w} → {h_out}x{w_out}")
        
        # Each stage has 4 ResBlocks
        for block_idx, block in enumerate(stage):
            stride = 2 if block_idx == 0 else 1
            current_cin = cin if block_idx == 0 else cout
            current_h = h if block_idx == 0 else h_out
            current_w = w if block_idx == 0 else w_out
            next_h = h_out if block_idx == 0 else h_out
            next_w = w_out if block_idx == 0 else w_out
            
            # Conv1
            conv1_params = current_cin * cout * 3 * 3
            conv1_flops = count_conv2d_flops(current_cin, cout, 3, (next_h, next_w))
            
            # BN1
            bn1_params = 2 * cout
            bn1_flops = count_batchnorm2d_flops(cout, (next_h, next_w))
            
            # Conv2
            conv2_params = cout * cout * 3 * 3
            conv2_flops = count_conv2d_flops(cout, cout, 3, (next_h, next_w))
            
            # BN2
            bn2_params = 2 * cout
            bn2_flops = count_batchnorm2d_flops(cout, (next_h, next_w))
            
            # Downsample (if stride=2)
            downsample_params = 0
            downsample_flops = 0
            if stride == 2:
                downsample_params = current_cin * cout * 1 * 1 + 2 * cout  # conv + bn
                downsample_flops = count_conv2d_flops(current_cin, cout, 1, (next_h, next_w))
                downsample_flops += count_batchnorm2d_flops(cout, (next_h, next_w))
            
            # Mini-Swin Attention (approximate)
            # Window size = 4, num windows = (h_out/4) * (w_out/4)
            num_windows = (next_h // 4) * (next_w // 4) if next_h >= 4 else 1
            tokens_per_window = 4 * 4
            attn_params = cout * cout * 3 + cout * cout  # QKV + proj
            attn_flops = count_attention_flops(tokens_per_window, cout, 4) * num_windows
            
            block_params = conv1_params + bn1_params + conv2_params + bn2_params + downsample_params + attn_params
            block_flops = conv1_flops + bn1_flops + conv2_flops + bn2_flops + downsample_flops + attn_flops
            
            layer_stats.append({
                'Layer': f'Stage{stage_idx+1}.Block{block_idx+1}',
                'Type': 'ResBlock+Attn',
                'Output Shape': f'(B, {cout}, {next_h}, {next_w})',
                'Kernel': '3x3',
                'Cin': current_cin,
                'Cout': cout,
                'Params': block_params,
                'FLOPs': block_flops,
                'Formula': f'Conv+BN+Attn, stride={stride}'
            })
            
            total_params += block_params
            total_flops += block_flops
    
    # CSRF Module
    print("\n" + "=" * 100)
    print("CSRF MODULE")
    print("=" * 100)
    
    csrf_channels = 512
    k_slices = 5
    reduction = 4
    
    # Alpha parameter: per-channel learnable weights
    csrf_alpha_params = csrf_channels
    
    # SE attention: fc1 + fc2
    csrf_se_params = (k_slices * csrf_channels) * (k_slices * csrf_channels // reduction) + \
                     (k_slices * csrf_channels // reduction) * (k_slices * csrf_channels)
    csrf_se_flops = count_linear_flops(k_slices * csrf_channels, k_slices * csrf_channels // reduction, 1) + \
                    count_linear_flops(k_slices * csrf_channels // reduction, k_slices * csrf_channels, 1)
    
    # Residual computation (element-wise ops, not counted in params)
    csrf_residual_flops = k_slices * csrf_channels * 4 * 4 * 3  # diff ops
    
    csrf_params = csrf_alpha_params + csrf_se_params
    csrf_flops = csrf_se_flops + csrf_residual_flops
    
    layer_stats.append({
        'Layer': 'CSRF',
        'Type': 'Cross-Slice Fusion',
        'Output Shape': f'(B, 512, 4, 4)',
        'Kernel': '-',
        'Cin': 512,
        'Cout': 512,
        'Params': csrf_params,
        'FLOPs': csrf_flops,
        'Formula': f'SE({k_slices}*512→{k_slices}*512//{reduction}→{k_slices}*512) + Residuals'
    })
    
    total_params += csrf_params
    total_flops += csrf_flops
    
    # Decoder
    print("\n" + "=" * 100)
    print("DECODER")
    print("=" * 100)
    
    decoder_channels = [512, 256, 128, 64, 32]
    decoder_spatial = [4, 8, 16, 32, 64]
    
    for i in range(len(decoder_channels) - 1):
        cin = decoder_channels[i]
        cout = decoder_channels[i + 1]
        h_in = decoder_spatial[i]
        w_in = decoder_spatial[i]
        h_out = decoder_spatial[i + 1]
        w_out = decoder_spatial[i + 1]
        
        # Upsample (bilinear - no params, only FLOPs for interpolation)
        upsample_flops = cin * h_out * w_out * 4  # bilinear interpolation
        
        # Conv
        conv_params = cin * cout * 3 * 3
        conv_flops = count_conv2d_flops(cin, cout, 3, (h_out, w_out))
        
        # BN
        bn_params = 2 * cout
        bn_flops = count_batchnorm2d_flops(cout, (h_out, w_out))
        
        # Skip conv (1x1)
        skip_params = cout * cout * 1 * 1 + 2 * cout
        skip_flops = count_conv2d_flops(cout, cout, 1, (h_out, w_out)) + count_batchnorm2d_flops(cout, (h_out, w_out))
        
        block_params = conv_params + bn_params + skip_params
        block_flops = upsample_flops + conv_flops + bn_flops + skip_flops
        
        layer_stats.append({
            'Layer': f'Decoder.Up{i+1}',
            'Type': 'Up+Conv',
            'Output Shape': f'(B, {cout}, {h_out}, {w_out})',
            'Kernel': '3x3',
            'Cin': cin,
            'Cout': cout,
            'Params': block_params,
            'FLOPs': block_flops,
            'Formula': f'Upsample+Conv+Skip'
        })
        
        total_params += block_params
        total_flops += block_flops
    
    # Final output layer
    final_params = 32 * 1 * 1 * 1 + 1
    final_flops = count_conv2d_flops(32, 1, 1, (64, 64))
    
    layer_stats.append({
        'Layer': 'Final',
        'Type': 'Conv2d+Sigmoid',
        'Output Shape': f'(B, 1, 64, 64)',
        'Kernel': '1x1',
        'Cin': 32,
        'Cout': 1,
        'Params': final_params,
        'FLOPs': final_flops,
        'Formula': '2*32*1*1²*64*64'
    })
    
    total_params += final_params
    total_flops += final_flops
    
    # Create DataFrame
    df = pd.DataFrame(layer_stats)
    
    # Add cumulative columns
    df['Cumulative Params'] = df['Params'].cumsum()
    df['Cumulative FLOPs'] = df['FLOPs'].cumsum()
    
    return df, total_params, total_flops

def main():
    print("\n" + "="*100)
    print("INITIALIZING MODEL")
    print("="*100)
    
    # Create model
    model = HybridMiniSwin2D5_CSRF(k_slices=5, channels=[32, 64, 128, 256, 512])
    
    # Analyze
    df, total_params, total_flops = analyze_model_detailed(model)
    
    # Display results
    print("\n" + "="*100)
    print("DETAILED LAYERWISE ANALYSIS")
    print("="*100)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(df.to_string(index=False))
    
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Total FLOPs: {total_flops:,} ({total_flops / 1e9:.2f}G)")
    print()
    
    # Save to CSV
    output_path = Path(__file__).parent / 'layerwise_compute_analysis.csv'
    df.to_csv(output_path, index=False)
    print(f"✓ Saved detailed analysis to: {output_path}")
    
    # Save summary
    summary_path = Path(__file__).parent / 'compute_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("COMPUTE ANALYSIS SUMMARY\n")
        f.write("="*100 + "\n\n")
        f.write(f"Model: HybridMiniSwin2D5_CSRF\n")
        f.write(f"Input Shape: (B, 1, 64, 64, 64)\n")
        f.write(f"K-slices (2.5D): 5\n\n")
        f.write(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)\n")
        f.write(f"Total FLOPs: {total_flops:,} ({total_flops / 1e9:.2f}GFLOPs)\n\n")
        f.write("Breakdown by Component:\n")
        f.write("-" * 50 + "\n")
        
        # Calculate per-component
        stem_params = df[df['Layer'].str.contains('Stem')]['Params'].sum()
        encoder_params = df[df['Layer'].str.contains('Stage')]['Params'].sum()
        csrf_params = df[df['Layer'] == 'CSRF']['Params'].sum()
        decoder_params = df[df['Layer'].str.contains('Decoder|Final')]['Params'].sum()
        
        stem_flops = df[df['Layer'].str.contains('Stem')]['FLOPs'].sum()
        encoder_flops = df[df['Layer'].str.contains('Stage')]['FLOPs'].sum()
        csrf_flops = df[df['Layer'] == 'CSRF']['FLOPs'].sum()
        decoder_flops = df[df['Layer'].str.contains('Decoder|Final')]['FLOPs'].sum()
        
        f.write(f"Stem:    {stem_params:>12,} params ({stem_params/total_params*100:5.2f}%)  |  {stem_flops:>15,} FLOPs ({stem_flops/total_flops*100:5.2f}%)\n")
        f.write(f"Encoder: {encoder_params:>12,} params ({encoder_params/total_params*100:5.2f}%)  |  {encoder_flops:>15,} FLOPs ({encoder_flops/total_flops*100:5.2f}%)\n")
        f.write(f"CSRF:    {csrf_params:>12,} params ({csrf_params/total_params*100:5.2f}%)  |  {csrf_flops:>15,} FLOPs ({csrf_flops/total_flops*100:5.2f}%)\n")
        f.write(f"Decoder: {decoder_params:>12,} params ({decoder_params/total_params*100:5.2f}%)  |  {decoder_flops:>15,} FLOPs ({decoder_flops/total_flops*100:5.2f}%)\n")
        f.write("-" * 50 + "\n")
        f.write(f"TOTAL:   {total_params:>12,} params (100.00%)  |  {total_flops:>15,} FLOPs (100.00%)\n")
    
    print(f"✓ Saved summary to: {summary_path}")
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)

if __name__ == '__main__':
    main()
