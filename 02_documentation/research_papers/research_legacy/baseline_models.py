"""
Minimal Experiment Battery - Baseline Models
Implements capacity-matched 2D and 3D baselines for fair comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================================
# 2D BASELINE: U-Net (Capacity-Matched)
# ===========================================================================================

class DoubleConv2D(nn.Module):
    """Double convolution block for 2D U-Net"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet2D_Baseline(nn.Module):
    """
    2D U-Net baseline (capacity-matched to ~34M params)
    Processes central slice only
    """
    def __init__(self, in_channels=1, out_channels=1, features=[56, 112, 224, 448, 896]):
        super().__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        in_ch = in_channels
        for feature in features:
            self.encoder.append(DoubleConv2D(in_ch, feature))
            in_ch = feature
        
        # Decoder
        for feature in reversed(features[:-1]):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv2D(feature * 2, feature))
        
        self.bottleneck = DoubleConv2D(features[-2], features[-1])
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, D, H, W) - 3D volume
        Returns:
            (B, 1, H, W) - 2D segmentation of central slice
        """
        # Extract central slice
        B, C, D, H, W = x.shape
        central_idx = D // 2
        x = x[:, :, central_idx, :, :]  # (B, 1, H, W)
        
        skip_connections = []
        
        # Encoder
        for encode in self.encoder[:-1]:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.encoder[-1](x)
        
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Upsample
            skip = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat([skip, x], dim=1)
            x = self.decoder[idx + 1](x)  # Double conv
        
        return torch.sigmoid(self.final_conv(x))


# ===========================================================================================
# 3D BASELINE: 3D U-Net (Capacity-Matched)
# ===========================================================================================

class DoubleConv3D(nn.Module):
    """Double convolution block for 3D U-Net"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet3D_Baseline(nn.Module):
    """
    3D U-Net baseline (capacity-matched to ~34M params)
    Processes full 3D volume but outputs central slice
    """
    def __init__(self, in_channels=1, out_channels=1, features=[72, 144, 288, 432]):
        super().__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Encoder (reduced features to match parameter count)
        in_ch = in_channels
        for feature in features:
            self.encoder.append(DoubleConv3D(in_ch, feature))
            in_ch = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv3D(features[-1], features[-1])
        
        # Decoder
        for i, feature in enumerate(reversed(features[:-1])):
            # Upsample from higher to current feature level
            in_channels = features[len(features) - 1 - i]  # Current bottleneck/decoder output
            self.decoder.append(
                nn.ConvTranspose3d(in_channels, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv3D(feature * 2, feature))
        
        # 3D to 2D projection (extract central slice)
        self.slice_projection = nn.Conv3d(features[0], out_channels, kernel_size=(1, 1, 1))
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, D, H, W) - 3D volume
        Returns:
            (B, 1, H, W) - 2D segmentation of central slice
        """
        skip_connections = []
        
        # Encoder
        for encode in self.encoder[:-1]:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.encoder[-1](x)
        
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Upsample
            skip = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
            
            x = torch.cat([skip, x], dim=1)
            x = self.decoder[idx + 1](x)  # Double conv
        
        # Extract central slice
        B, C, D, H, W = x.shape
        central_idx = D // 2
        x = x[:, :, central_idx:central_idx+1, :, :]  # Keep D dimension
        
        # Project to output
        x = self.slice_projection(x)
        x = x.squeeze(2)  # Remove D dimension: (B, C, H, W)
        
        return torch.sigmoid(x)


# ===========================================================================================
# PARAMETER COUNTING UTILITIES
# ===========================================================================================

def count_parameters(model):
    """Count trainable parameters in a model"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


def count_parameters_detailed(model):
    """Count parameters with layer breakdown"""
    table = []
    total = 0
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.append({
            'name': name,
            'shape': list(parameter.shape),
            'params': params
        })
        total += params
    
    return table, total


def test_model_shapes():
    """Test baseline model shapes and parameter counts"""
    
    print("="*100)
    print("BASELINE MODEL TESTING")
    print("="*100)
    
    batch_size = 2
    input_shape = (batch_size, 1, 64, 64, 64)
    x = torch.randn(input_shape)
    
    # Test 2D U-Net
    print("\n2D U-Net Baseline:")
    print("-"*100)
    model_2d = UNet2D_Baseline()  # Use default capacity-matched features
    output_2d = model_2d(x)
    params_2d = count_parameters(model_2d)
    
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(output_2d.shape)}")
    print(f"Parameters:   {params_2d:,} ({params_2d/1e6:.2f}M)")
    print(f"Expected:     (B, 1, H, W)")
    print(f"✓ Shape correct: {output_2d.shape == (batch_size, 1, 64, 64)}")
    
    # Test 3D U-Net
    print("\n3D U-Net Baseline:")
    print("-"*100)
    model_3d = UNet3D_Baseline()  # Use default capacity-matched features
    output_3d = model_3d(x)
    params_3d = count_parameters(model_3d)
    
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(output_3d.shape)}")
    print(f"Parameters:   {params_3d:,} ({params_3d/1e6:.2f}M)")
    print(f"Expected:     (B, 1, H, W)")
    print(f"✓ Shape correct: {output_3d.shape == (batch_size, 1, 64, 64)}")
    
    # Target comparison
    print("\n"+"="*100)
    print("CAPACITY MATCHING")
    print("="*100)
    target_params = 34_200_000
    print(f"Target (Proposed):  {target_params:,} ({target_params/1e6:.2f}M)")
    print(f"2D U-Net:           {params_2d:,} ({params_2d/1e6:.2f}M) - Ratio: {params_2d/target_params:.2f}x")
    print(f"3D U-Net:           {params_3d:,} ({params_3d/1e6:.2f}M) - Ratio: {params_3d/target_params:.2f}x")
    
    # Recommendations
    print("\n"+"="*100)
    print("CAPACITY ADJUSTMENT RECOMMENDATIONS")
    print("="*100)
    
    if params_2d > target_params * 1.1:
        print(f"2D U-Net: TOO LARGE - Reduce features or depth")
        reduction_factor = (target_params / params_2d) ** 0.5
        new_features = [int(f * reduction_factor) for f in [64, 128, 256, 512, 1024]]
        print(f"  Suggested features: {new_features}")
    elif params_2d < target_params * 0.9:
        print(f"2D U-Net: TOO SMALL - Increase features or depth")
        increase_factor = (target_params / params_2d) ** 0.5
        new_features = [int(f * increase_factor) for f in [64, 128, 256, 512, 1024]]
        print(f"  Suggested features: {new_features}")
    else:
        print(f"2D U-Net: ✓ WELL MATCHED (within 10%)")
    
    if params_3d > target_params * 1.1:
        print(f"\n3D U-Net: TOO LARGE - Reduce features or depth")
        reduction_factor = (target_params / params_3d) ** 0.5
        new_features = [int(f * reduction_factor) for f in [32, 64, 128, 256]]
        print(f"  Suggested features: {new_features}")
    elif params_3d < target_params * 0.9:
        print(f"\n3D U-Net: TOO SMALL - Increase features or depth")
        increase_factor = (target_params / params_3d) ** 0.5
        new_features = [int(f * increase_factor) for f in [32, 64, 128, 256]]
        print(f"  Suggested features: {new_features}")
    else:
        print(f"\n3D U-Net: ✓ WELL MATCHED (within 10%)")
    
    print("\n"+"="*100)


if __name__ == "__main__":
    test_model_shapes()
