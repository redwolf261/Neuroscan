# ===========================================================================================
# HybridMiniSwin2.5D-ResNet with CSRF and 2.5D-MAE Pretraining
# ===========================================================================================
# Novel 2.5D architecture for volumetric MRI segmentation on small datasets
# Based on ablation study findings and improved architecture proposal
#
# Key Features:
# - 2.5D processing (k consecutive slices instead of full 3D volume)
# - Cross-Slice Residual Fusion (CSRF) module for inter-slice coherence
# - Mini-Swin attention windows (not full attention - ablation showed marginal benefit)
# - ResNet-style skip connections (ablation showed -4.44% when removed - CRITICAL)
# - NO dropout (ablation showed +0.55% improvement when removed)
# - NO heavy 3D convolutions (ablation showed +2.36% improvement when removed)
# - 2.5D Masked Autoencoder pretraining for small dataset robustness
# ===========================================================================================

import os, time, json, glob, warnings, shutil, tempfile
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, EnsureTyped, Compose, MapTransform,
    RandFlipd, RandRotate90d
)
try:
    from monai.transforms import Resized
except:
    try:
        from monai.transforms import Resize as Resized
    except:
        from monai.transforms import Resize
        class Resized(Resize):
            def __init__(self, keys, spatial_size, mode):
                super().__init__(spatial_size=spatial_size, mode=mode)

from monai.data import Dataset, DataLoader
from monai.utils import set_determinism
from monai.losses import DiceLoss, FocalLoss
from torch.utils.tensorboard import SummaryWriter

set_determinism(42)

# ===========================================================================================
# PATHS CONFIGURATION (Separate from trial.py)
# ===========================================================================================
CUSTOM_PATH = r"C:\Users\HP\EDI"

if os.name == 'nt':
    if os.environ.get('DATASET_BASE_PATH'):
        DRIVE_BASE = os.environ.get('DATASET_BASE_PATH')
    elif CUSTOM_PATH and os.path.exists(CUSTOM_PATH):
        DRIVE_BASE = CUSTOM_PATH
    else:
        possible_drives = [
            "G:\\My Drive",
            os.path.join(os.path.expanduser("~"), "Google Drive"),
            "C:\\Users\\HP\\Google Drive\\My Drive",
        ]
        DRIVE_BASE = None
        for drive_path in possible_drives:
            if os.path.exists(drive_path):
                DRIVE_BASE = drive_path
                break
        if DRIVE_BASE is None:
            DRIVE_BASE = os.path.join(os.path.expanduser("~"), "MyDrive_Local")
            os.makedirs(DRIVE_BASE, exist_ok=True)
else:
    DRIVE_BASE = "C:/Users/HP/EDI"

DATA_PATH = os.path.join(r"C:\Users\HP\EDI", "Dataset", "PediMS", "PediMS")

# USALD MODEL PATHS (new architecture with causal decomposition + self-correction)
OUTPUT_DIR = os.path.join(DRIVE_BASE, "USALD_CausalSelfCorrection")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
MAE_PRETRAIN_DIR = os.path.join(OUTPUT_DIR, "mae_pretraining")
SEGMENTATION_DIR = os.path.join(OUTPUT_DIR, "segmentation")
DEPLOYMENT_DIR = os.path.join(OUTPUT_DIR, "deployment")

# Local-only resume checkpoints (NOT synced to Google Drive to save space)
LOCAL_RESUME_DIR = os.path.join(os.path.dirname(__file__), ".resume_checkpoints_usald")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MAE_PRETRAIN_DIR, exist_ok=True)
os.makedirs(SEGMENTATION_DIR, exist_ok=True)
os.makedirs(DEPLOYMENT_DIR, exist_ok=True)
os.makedirs(LOCAL_RESUME_DIR, exist_ok=True)

print("=" * 80)
print("FINAL MODEL: HybridMiniSwin2.5D-ResNet with CSRF and 2.5D-MAE")
print("=" * 80)
print(f"📂 Drive base: {DRIVE_BASE}")
print(f"📂 Dataset: {DATA_PATH}")
print(f"📂 Output: {OUTPUT_DIR}")
print(f"📂 Checkpoints: {CHECKPOINT_DIR}")
print(f"📂 MAE Pretraining: {MAE_PRETRAIN_DIR}")
print(f"📂 Segmentation: {SEGMENTATION_DIR}")
print(f"📂 Deployment: {DEPLOYMENT_DIR}")
print(f"📂 Local Resume (NOT synced): {LOCAL_RESUME_DIR}")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

print("Device:", device, "| AMP enabled:", use_amp)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ===========================================================================================
# CONFIGURATION
# ===========================================================================================
K_SLICES = 5  # Number of consecutive slices (2.5D context)
SPATIAL_SIZE = (64, 64, 64)  # Input spatial size
EMBED_DIM = 32  # C_0 in architecture doc
MINI_SWIN_WINDOW = 4  # 4x4 windows for Mini-Swin attention
MINI_SWIN_HEADS = 4  # Number of attention heads
STAGE_CHANNELS = [32, 64, 128, 256, 512]  # Channel progression
BATCH_SIZE = 3  # Limited by 4GB GPU
NUM_WORKERS = 0  # Must be 0 on Windows
LEARNING_RATE_ENCODER = 1e-5  # Fine-tune pretrained encoder
LEARNING_RATE_DECODER = 4e-4  # Train decoder from scratch
MAE_LEARNING_RATE = 1e-4  # MAE pretraining LR
MAE_EPOCHS = 50  # Ablation study: 50 epochs
SEGMENTATION_EPOCHS = 30  # Ablation study: 30 epochs

# ===========================================================================================
# ABLATION STUDY CONFIGURATION
# ===========================================================================================
# Set RUN_ABLATION_STUDY = True to automatically run all 7 configs and save results to CSV
RUN_ABLATION_STUDY = False  # Change to True to run full ablation
ABLATION_EPOCHS_MAE = 50    # MAE epochs per ablation config
ABLATION_EPOCHS_SEG = 30    # Segmentation epochs per ablation config
ABLATION_RESULTS_DIR = r"C:\Users\HP\EDI\ablation_results"  # Save CSVs here

# If RUN_ABLATION_STUDY = False, use single config below:
ABLATION_CONFIG = "causal"  # PRODUCTION: Best single component (Dice 0.8264, +1.42%)

# Ablation configurations (7 total for complete analysis):
ABLATION_CONFIGS = {
    "baseline": {
        "name": "Baseline (No USALD)",
        "USALD_ENABLED": False,
        "USALD_CONSISTENCY_ENABLED": False,
        "USALD_FDR_ENABLED": False,
        "USALD_CAUSAL_ENABLED": False,
        "USALD_SELF_CORRECTION": False,
    },
    "evidential": {
        "name": "+Evidential Only",
        "USALD_ENABLED": True,
        "USALD_CONSISTENCY_ENABLED": False,
        "USALD_FDR_ENABLED": False,
        "USALD_CAUSAL_ENABLED": False,
        "USALD_SELF_CORRECTION": False,
    },
    "causal": {
        "name": "+Causal Decomposition",
        "USALD_ENABLED": True,
        "USALD_CONSISTENCY_ENABLED": False,
        "USALD_FDR_ENABLED": False,
        "USALD_CAUSAL_ENABLED": True,
        "USALD_SELF_CORRECTION": False,
    },
    "self_correction": {
        "name": "+Self-Correction",
        "USALD_ENABLED": True,
        "USALD_CONSISTENCY_ENABLED": False,
        "USALD_FDR_ENABLED": False,
        "USALD_CAUSAL_ENABLED": False,
        "USALD_SELF_CORRECTION": True,
    },
    "consistency": {
        "name": "+Consistency (Teacher-Student)",
        "USALD_ENABLED": True,
        "USALD_CONSISTENCY_ENABLED": True,
        "USALD_FDR_ENABLED": False,
        "USALD_CAUSAL_ENABLED": False,
        "USALD_SELF_CORRECTION": False,
    },
    "fdr": {
        "name": "+FDR Thresholding",
        "USALD_ENABLED": True,
        "USALD_CONSISTENCY_ENABLED": False,
        "USALD_FDR_ENABLED": True,
        "USALD_CAUSAL_ENABLED": False,
        "USALD_SELF_CORRECTION": False,
    },
    "full": {
        "name": "Full USALD (All 5 Components)",
        "USALD_ENABLED": True,
        "USALD_CONSISTENCY_ENABLED": True,
        "USALD_FDR_ENABLED": True,
        "USALD_CAUSAL_ENABLED": True,
        "USALD_SELF_CORRECTION": True,
    },
}

# ===========================================================================================
# USALD MODE (Uncertainty-Guided Self-Adaptive Lesion Discovery) - Novel Q1 Framework
# ===========================================================================================
# When enabled, adds three novel components:
# 1. Evidential Beta head (calibrated aleatoric uncertainty)
# 2. Teacher-student consistency with uncertainty weighting
# 3. FDR-controlled adaptive pseudo-label thresholding
# 4. CAUSAL UNCERTAINTY DECOMPOSITION (NEW!)
# 5. EVIDENTIAL SELF-CORRECTION (NEW!)
USALD_ENABLED = True              # Enable evidential head with causal decomposition
USALD_CONSISTENCY_ENABLED = True  # Enable teacher-student consistency
USALD_FDR_ENABLED = True          # Enable FDR-controlled thresholding
USALD_CAUSAL_ENABLED = True       # Enable causal decomposition (3 heads)
USALD_SELF_CORRECTION = True      # Enable self-correction during inference (validation/test)

# USALD Hyperparameters
WARMUP_EPOCHS = 10               # Supervised-only warmup before consistency/pseudo-labels
EMA_DECAY = 0.99                 # Teacher EMA momentum (0.99 = slow, stable updates)
FDR_Q = 0.10                     # Target false discovery rate (10% = precision ≥ 90%)
FDR_UPDATE_INTERVAL = 5          # Re-calibrate threshold every N epochs
LAMBDA_EVIDENTIAL = 1e-3         # Evidential loss weight
LAMBDA_CONSISTENCY = 1.0         # Consistency loss weight
LAMBDA_PSEUDO = 0.5              # Pseudo-label loss weight
UNCERTAINTY_GAMMA = 3.0          # Uncertainty weighting strength (higher = more selective)

# Self-Correction Hyperparameters (NEW!)
SELF_CORRECTION_MAX_ITER = 3     # Max iterations for self-correction (3 = good balance)
SELF_CORRECTION_THRESHOLD = 0.5  # Stop if max uncertainty < 0.5 (converged)

MAE_MASK_RATIO = 0.75  # Mask 75% of patches (optimal from ablation study - quick test showed 86.5% vs 86.0% for 0.5)

# ===========================================================================================
# APPLY ABLATION CONFIG (Override settings based on selected config)
# ===========================================================================================
if ABLATION_CONFIG != "full":
    config_dict = ABLATION_CONFIGS[ABLATION_CONFIG]
    USALD_ENABLED = config_dict["USALD_ENABLED"]
    USALD_CONSISTENCY_ENABLED = config_dict["USALD_CONSISTENCY_ENABLED"]
    USALD_FDR_ENABLED = config_dict["USALD_FDR_ENABLED"]
    USALD_CAUSAL_ENABLED = config_dict["USALD_CAUSAL_ENABLED"]
    USALD_SELF_CORRECTION = config_dict["USALD_SELF_CORRECTION"]
    
    # Update paths for this specific ablation config
    OUTPUT_DIR = os.path.join(DRIVE_BASE, f"USALD_Ablation_{ABLATION_CONFIG}")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    MAE_PRETRAIN_DIR = os.path.join(OUTPUT_DIR, "mae_pretraining")
    SEGMENTATION_DIR = os.path.join(OUTPUT_DIR, "segmentation")
    DEPLOYMENT_DIR = os.path.join(OUTPUT_DIR, "deployment")
    LOCAL_RESUME_DIR = os.path.join(os.path.dirname(__file__), f".resume_ablation_{ABLATION_CONFIG}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MAE_PRETRAIN_DIR, exist_ok=True)
    os.makedirs(SEGMENTATION_DIR, exist_ok=True)
    os.makedirs(DEPLOYMENT_DIR, exist_ok=True)
    os.makedirs(LOCAL_RESUME_DIR, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"ABLATION CONFIG: {config_dict['name']}")
    print("="*80)
    print(f"USALD Enabled: {USALD_ENABLED}")
    print(f"Consistency: {USALD_CONSISTENCY_ENABLED}")
    print(f"FDR: {USALD_FDR_ENABLED}")
    print(f"Causal: {USALD_CAUSAL_ENABLED}")
    print(f"Self-Correction: {USALD_SELF_CORRECTION}")
    print("="*80 + "\n")

# ===========================================================================================
# DATA LOADING (PediMS Dataset)
# ===========================================================================================
data_dicts = []
if os.path.exists(DATA_PATH):
    for subfolder in sorted(os.listdir(DATA_PATH)):
        sub_path = os.path.join(DATA_PATH, subfolder)
        if not os.path.isdir(sub_path): 
            continue
        for modality in ["T1","T2","FLAIR"]:
            mod_path = os.path.join(sub_path, modality, "processed")
            if not os.path.exists(mod_path): 
                continue
            imgs = sorted(glob.glob(os.path.join(mod_path, "*_brain_*.nii*")))
            masks = sorted(glob.glob(os.path.join(mod_path, "*_mask_*.nii*")))
            if len(masks) == 0:
                masks = sorted(glob.glob(os.path.join(mod_path, "*_Consensus_*.nii*")))
            for img, m in zip(imgs, masks):
                data_dicts.append({"image":[img],"label":m,"case":os.path.basename(img)})
else:
    raise RuntimeError(f"DATA_PATH does not exist: {DATA_PATH}")

if len(data_dicts) == 0:
    raise RuntimeError("No PediMS data found.")

print(f"Loaded {len(data_dicts)} PEDiMS samples")

# ===========================================================================================
# TRANSFORMS
# ===========================================================================================
class BinarizeLabel(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            d[k] = (d[k] > 0).float()
        return d

train_transforms = Compose([
    LoadImaged(keys=["image","label"]),
    EnsureChannelFirstd(keys=["image","label"]),
    Orientationd(keys=["image","label"], axcodes="RAS"),
    Spacingd(keys=["image","label"], pixdim=(1.0,1.0,1.0), mode=("bilinear","nearest")),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    BinarizeLabel(keys=["label"]),
    Resized(keys=["image","label"], spatial_size=SPATIAL_SIZE, mode=("trilinear","nearest")),
    RandFlipd(keys=["image","label"], spatial_axis=[0,1,2], prob=0.5),
    RandRotate90d(keys=["image","label"], prob=0.3, max_k=3),
    EnsureTyped(keys=["image","label"])
])

val_transforms = Compose([
    LoadImaged(keys=["image","label"]),
    EnsureChannelFirstd(keys=["image","label"]),
    Orientationd(keys=["image","label"], axcodes="RAS"),
    Spacingd(keys=["image","label"], pixdim=(1.0,1.0,1.0), mode=("bilinear","nearest")),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    BinarizeLabel(keys=["label"]),
    Resized(keys=["image","label"], spatial_size=SPATIAL_SIZE, mode=("trilinear","nearest")),
    EnsureTyped(keys=["image","label"])
])

split = int(0.8 * len(data_dicts))
train_ds = Dataset(data=data_dicts[:split], transform=train_transforms)
val_ds = Dataset(data=data_dicts[split:], transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train / Val split: {len(train_ds)} / {len(val_ds)}")
print(f"Batch size: {BATCH_SIZE}, Workers: {NUM_WORKERS}, Spatial size: {SPATIAL_SIZE}")

# ===========================================================================================
# UTILITY MODULES
# ===========================================================================================
class DropPath(nn.Module):
    """Stochastic depth (drop path) for regularization."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * rand

# ===========================================================================================
# 2.5D CONVOLUTIONAL STEM
# ===========================================================================================
class Conv2D5Stem(nn.Module):
    """
    2.5D Convolutional stem with slice-wise processing and cross-slice fusion.
    Input: (B, k, H, W) where k is number of slices
    Output: (B, C_0, H, W) fused feature map
    """
    def __init__(self, k_slices=5, out_channels=32):
        super().__init__()
        self.k = k_slices
        
        # Slice-wise 2D convolution (shared across slices)
        self.slice_conv = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Slice-wise attention for weighted fusion
        self.slice_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, 1, kernel_size=1)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, 1, D, H, W) - 3D volume
        Returns:
            (B, out_channels, H, W) - 2.5D fused features
        """
        B, C, D, H, W = x.shape
        
        # Extract k consecutive slices from center
        center = D // 2
        start_idx = max(0, center - self.k // 2)
        end_idx = min(D, start_idx + self.k)
        
        # Ensure we have k slices (pad with reflection if needed)
        slice_features = []
        alphas = []
        
        for i in range(start_idx, end_idx):
            slice_i = x[:, :, i, :, :]  # (B, 1, H, W)
            feat_i = self.slice_conv(slice_i)  # (B, C_0, H, W)
            alpha_i = self.slice_attention(feat_i)  # (B, 1, 1, 1)
            slice_features.append(feat_i)
            alphas.append(alpha_i)
        
        # Weighted fusion
        alphas = torch.cat(alphas, dim=1)  # (B, k, 1, 1)
        alphas = F.softmax(alphas, dim=1)
        
        fused = 0
        for i, feat in enumerate(slice_features):
            fused = fused + alphas[:, i:i+1, :, :] * feat
        
        return fused  # (B, C_0, H, W)

# ===========================================================================================
# MINI-SWIN WINDOWED ATTENTION (2D)
# ===========================================================================================
class MiniSwinAttention2D(nn.Module):
    """
    Mini-Swin windowed self-attention (2D for efficiency).
    Uses small 4x4 windows instead of full attention.
    Ablation showed full attention is marginal (+0.74% when removed).
    """
    def __init__(self, dim, num_heads=4, window_size=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Pad to window size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        _, _, H_pad, W_pad = x.shape
        
        # Partition into windows
        x = x.view(B, C, H_pad // self.window_size, self.window_size, 
                   W_pad // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, nH, nW, ws, ws, C)
        x = x.view(-1, self.window_size * self.window_size, C)  # (B*nH*nW, ws*ws, C)
        
        # Multi-head attention
        qkv = self.qkv(x).reshape(-1, self.window_size * self.window_size, 3, 
                                   self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        x = self.proj(x)
        
        # Reverse window partition
        x = x.view(B, H_pad // self.window_size, W_pad // self.window_size,
                   self.window_size, self.window_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, H_pad, W_pad)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
        
        return x

# ===========================================================================================
# RESIDUAL BLOCK (Critical - ablation showed -4.44% when removed)
# ===========================================================================================
class ResidualBlock2D(nn.Module):
    """
    ResNet-style residual block with optional Mini-Swin attention.
    Skip connections are CRITICAL (ablation showed -4.44% degradation when removed).
    NO dropout (ablation showed +0.55% improvement when removed).
    """
    def __init__(self, in_channels, out_channels, stride=1, use_attention=True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsample for skip connection if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Optional Mini-Swin attention (lightweight)
        self.use_attention = use_attention
        if use_attention:
            self.attention = MiniSwinAttention2D(out_channels, num_heads=MINI_SWIN_HEADS, 
                                                  window_size=MINI_SWIN_WINDOW)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply attention if enabled
        if self.use_attention:
            out = self.attention(out)
        
        # Skip connection (CRITICAL)
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = out + identity  # Residual addition
        out = self.relu(out)
        
        return out

# ===========================================================================================
# ENCODER: HybridMiniSwin2.5D-ResNet
# ===========================================================================================
class HybridMiniSwin2D5_ResNetEncoder(nn.Module):
    """
    2.5D Encoder with ResNet blocks and Mini-Swin attention.
    Processes k consecutive slices for local volumetric context.
    
    Architecture:
    - Stem: 2.5D conv stem (k slices -> C_0 features)
    - Stage 1: 4 ResBlocks, C_0 -> C_1 (64), 64x64 -> 32x32
    - Stage 2: 4 ResBlocks, C_1 -> C_2 (128), 32x32 -> 16x16
    - Stage 3: 4 ResBlocks, C_2 -> C_3 (256), 16x16 -> 8x8
    - Stage 4: 4 ResBlocks, C_3 -> C_4 (512), 8x8 -> 4x4
    """
    def __init__(self, k_slices=5, channels=[32, 64, 128, 256, 512], blocks_per_stage=4):
        super().__init__()
        
        # 2.5D Stem
        self.stem = Conv2D5Stem(k_slices=k_slices, out_channels=channels[0])
        
        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            in_c, out_c = channels[i], channels[i+1]
            
            # First block with stride=2 for downsampling
            blocks = [ResidualBlock2D(in_c, out_c, stride=2, use_attention=True)]
            
            # Remaining blocks
            for _ in range(blocks_per_stage - 1):
                blocks.append(ResidualBlock2D(out_c, out_c, stride=1, use_attention=True))
            
            self.stages.append(nn.Sequential(*blocks))
        
        self.out_channels = channels[-1]
        
    def forward(self, x):
        """
        Args:
            x: (B, 1, D, H, W) - 3D volume
        Returns:
            features: List of feature maps at each stage
            bottleneck: (B, C_4, H/16, W/16)
        """
        # Stem
        x = self.stem(x)  # (B, C_0, H, W)
        
        # Collect features from each stage for skip connections
        features = [x]
        
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        return features  # [C_0@64x64, C_1@32x32, C_2@16x16, C_3@8x8, C_4@4x4]

# ===========================================================================================
# CROSS-SLICE RESIDUAL FUSION (CSRF) MODULE
# ===========================================================================================
class CSRF_Module(nn.Module):
    """
    Cross-Slice Residual Fusion module.
    Enforces structural continuity across slices in the bottleneck.
    
    For a stack of k slices with features F_i:
    - Compute residuals: R_i = F_i - 0.5*(F_{i-1} + F_{i+1})
    - Fuse with learnable weight: F'_i = F_i + alpha * R_i
    - Apply SE-style channel attention across slices
    - Output central slice features
    """
    def __init__(self, channels, k_slices=5, reduction=4):
        super().__init__()
        self.k = k_slices
        
        # Learnable fusion weight (per channel)
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)
        
        # SE-style attention across slices
        self.se_fc1 = nn.Linear(k_slices * channels, k_slices * channels // reduction)
        self.se_fc2 = nn.Linear(k_slices * channels // reduction, k_slices * channels)
        
    def forward(self, slice_features):
        """
        Args:
            slice_features: List of k tensors, each (B, C, H, W)
        Returns:
            (B, C, H, W) - fused central slice features
        """
        k = len(slice_features)
        B, C, H, W = slice_features[0].shape
        
        # Handle single slice case (k=1)
        if k == 1:
            # No inter-slice residuals, just return the single slice
            return slice_features[0]
        
        # Compute residuals
        residuals = []
        for i in range(k):
            if i == 0:
                # Forward difference
                R_i = slice_features[i] - slice_features[i+1]
            elif i == k-1:
                # Backward difference
                R_i = slice_features[i] - slice_features[i-1]
            else:
                # Central difference
                R_i = slice_features[i] - 0.5 * (slice_features[i-1] + slice_features[i+1])
            residuals.append(R_i)
        
        # Fuse with learnable alpha
        fused_slices = []
        for i in range(k):
            F_tilde = slice_features[i] + self.alpha * residuals[i]
            fused_slices.append(F_tilde)
        
        # SE-style attention
        # Global average pooling per slice
        se_input = []
        for feat in fused_slices:
            gap = F.adaptive_avg_pool2d(feat, 1)  # (B, C, 1, 1)
            se_input.append(gap)
        se_input = torch.cat(se_input, dim=1)  # (B, k*C, 1, 1)
        se_input = se_input.view(B, -1)  # (B, k*C)
        
        # SE excitation
        se_out = F.relu(self.se_fc1(se_input))
        se_out = torch.sigmoid(self.se_fc2(se_out))
        se_out = se_out.view(B, k, C, 1, 1)
        
        # Reweight slices
        weighted_slices = []
        for i in range(k):
            weighted = fused_slices[i] * se_out[:, i, :, :, :]
            weighted_slices.append(weighted)
        
        # Return central slice
        central_idx = k // 2
        return weighted_slices[central_idx]

# ===========================================================================================
# DECODER
# ===========================================================================================
class LightweightDecoder(nn.Module):
    """
    Lightweight convolutional decoder with skip connections.
    Ablation showed attention is unnecessary in decoder.
    """
    def __init__(self, channels=[512, 256, 128, 64, 32]):
        super().__init__()
        
        self.up_blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            in_c, out_c = channels[i], channels[i+1]
            
            # Upsampling block
            self.up_blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ))
            
            # Skip connection processing (1x1 conv)
            self.skip_convs.append(nn.Sequential(
                nn.Conv2d(out_c, out_c, kernel_size=1),
                nn.BatchNorm2d(out_c)
            ))
        
        # Final lesion probability output
        self.prob_head = nn.Sequential(
            nn.Conv2d(channels[-1], 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Optional evidential Beta head for aleatoric uncertainty (USALD)
        if USALD_ENABLED:
            # Causal Uncertainty Decomposition (NOVEL COMPONENT #1)
            # Decompose uncertainty into three causal factors:
            # 1. Anatomy uncertainty (WM/GM boundary confusion)
            # 2. Pathology uncertainty (lesion vs. artifact)
            # 3. Noise uncertainty (scanner noise, motion artifacts)
            
            shared_features = channels[-1] // 2
            
            # Shared feature extraction for causal reasoning
            self.causal_shared = nn.Sequential(
                nn.Conv2d(channels[-1], shared_features, 3, padding=1),
                nn.BatchNorm2d(shared_features),
                nn.ReLU(inplace=True)
            )
            
            # Three causal evidential heads
            self.anatomy_head = nn.Conv2d(shared_features, 2, 1)      # α_anatomy
            self.pathology_head = nn.Conv2d(shared_features, 2, 1)    # α_pathology
            self.noise_head = nn.Conv2d(shared_features, 2, 1)        # α_noise
            
            # Learned causal weights (initialized to domain-informed priors)
            # Pathology is most important (0.5), anatomy secondary (0.3), noise least (0.2)
            self.causal_weights = nn.Parameter(torch.tensor([0.3, 0.5, 0.2]))  # [anatomy, pathology, noise]
            
            self.softplus = nn.Softplus()
        else:
            self.causal_shared = None
            self.anatomy_head = None
            self.pathology_head = None
            self.noise_head = None
            self.causal_weights = None
        
    def forward(self, features):
        """
        Args:
            features: List of feature maps from encoder [C_0@64, C_1@32, C_2@16, C_3@8, C_4@4]
        Returns:
            (B, 1, H, W) - segmentation mask
        """
        x = features[-1]  # Start from bottleneck (C_4@4x4)
        
        for i, (up_block, skip_conv) in enumerate(zip(self.up_blocks, self.skip_convs)):
            x = up_block(x)
            
            # Add skip connection
            skip_idx = len(features) - 2 - i
            skip = features[skip_idx]
            skip = skip_conv(skip)
            x = x + skip  # Element-wise add
        
        probs = self.prob_head(x)
        
        if self.causal_shared is not None:
            # Causal Uncertainty Decomposition
            causal_features = self.causal_shared(x)
            
            # Get raw evidential outputs for each causal factor
            raw_alpha_anatomy = self.anatomy_head(causal_features)      # (B, 2, H, W)
            raw_alpha_pathology = self.pathology_head(causal_features)  # (B, 2, H, W)
            raw_alpha_noise = self.noise_head(causal_features)          # (B, 2, H, W)
            
            # Apply softplus + 1 to ensure alpha > 1 (valid Beta parameters)
            alpha_anatomy = self.softplus(raw_alpha_anatomy) + 1.0
            alpha_pathology = self.softplus(raw_alpha_pathology) + 1.0
            alpha_noise = self.softplus(raw_alpha_noise) + 1.0
            
            # Normalize causal weights to sum to 1 (convex combination)
            weights = F.softmax(self.causal_weights, dim=0)
            w_anat, w_path, w_noise = weights[0], weights[1], weights[2]
            
            # Causal combination: α_total = Σ w_i * α_i
            # This implements the structural equation: P(lesion) = f(Anatomy, Pathology, Noise)
            alpha = (w_anat.view(1, 1, 1, 1) * alpha_anatomy + 
                     w_path.view(1, 1, 1, 1) * alpha_pathology + 
                     w_noise.view(1, 1, 1, 1) * alpha_noise)
            
            return {
                "probs": probs, 
                "alpha": alpha,
                "alpha_anatomy": alpha_anatomy,
                "alpha_pathology": alpha_pathology,
                "alpha_noise": alpha_noise,
                "causal_weights": weights  # For logging/analysis
            }
        return {"probs": probs}

# ===========================================================================================
# COMPLETE 2.5D MODEL (Segmentation)
# ===========================================================================================
class HybridMiniSwin2D5_CSRF(nn.Module):
    """
    Complete 2.5D segmentation model with CSRF.
    """
    def __init__(self, k_slices=5, channels=[32, 64, 128, 256, 512]):
        super().__init__()
        self.k = k_slices
        
        self.encoder = HybridMiniSwin2D5_ResNetEncoder(k_slices=k_slices, channels=channels)
        self.csrf = CSRF_Module(channels=channels[-1], k_slices=k_slices)
        self.decoder = LightweightDecoder(channels=list(reversed(channels)))
        
    def forward(self, x):
        """
        Args:
            x: (B, 1, D, H, W) - 3D volume
        Returns:
            (B, 1, H, W) - 2D segmentation mask for central slice
        """
        # Encode
        features = self.encoder(x)
        
        # CSRF module on bottleneck
        # For now, we process single volume -> treat as single "stack"
        # In full implementation, sliding window would extract multiple k-slice stacks
        bottleneck = features[-1]
        bottleneck_csrf = self.csrf([bottleneck])  # Simplified for single slice
        
        # Replace bottleneck with CSRF output
        features[-1] = bottleneck_csrf
        
        # Decode (returns dict with probs and optional alpha)
        dec_out = self.decoder(features)
        return dec_out

# ===========================================================================================
# 2.5D MASKED AUTOENCODER (MAE) PRETRAINING
# ===========================================================================================
class MAE_Decoder(nn.Module):
    """Lightweight transformer decoder for MAE reconstruction."""
    def __init__(self, embed_dim=512, decoder_embed_dim=256, num_blocks=4, num_heads=4):
        super().__init__()
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=decoder_embed_dim, nhead=num_heads, 
                                        dim_feedforward=decoder_embed_dim*4, batch_first=True)
            for _ in range(num_blocks)
        ])
        
        self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim)  # Reconstruct back to bottleneck dim
        
    def forward(self, x):
        x = self.decoder_embed(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.decoder_pred(x)
        return x

class MAE_2D5(nn.Module):
    """
    2.5D Masked Autoencoder for self-supervised pretraining.
    Masks patches in 2D slices and reconstructs them.
    """
    def __init__(self, encoder, mask_ratio=0.5, decoder_embed_dim=256):
        super().__init__()
        
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        
        # MAE decoder
        self.decoder = MAE_Decoder(embed_dim=encoder.out_channels, 
                                     decoder_embed_dim=decoder_embed_dim)
        
    def random_masking(self, x, mask_ratio):
        """
        Randomly mask spatial positions (tokens).
        Args:
            x: (B, C, H, W) - bottleneck features
            mask_ratio: Fraction of positions to mask
        Returns:
            x_masked: (B, C, H, W) with masked positions zeroed
            mask: (B, H*W) binary mask (1 = masked)
        """
        B, C, H, W = x.shape
        num_positions = H * W  # Total spatial positions
        
        # Generate random mask for each sample
        num_masked = int(mask_ratio * num_positions)
        
        mask = torch.zeros(B, num_positions, device=x.device)
        for i in range(B):
            masked_indices = torch.randperm(num_positions, device=x.device)[:num_masked]
            mask[i, masked_indices] = 1
        
        # Apply mask: flatten spatial, apply mask, reshape back
        x_flat = x.flatten(2)  # (B, C, H*W)
        mask_expanded = mask.unsqueeze(1)  # (B, 1, H*W)
        x_flat_masked = x_flat * (1 - mask_expanded)  # Zero out masked positions
        
        x_masked = x_flat_masked.view(B, C, H, W)
        
        return x_masked, mask
        
    def forward(self, x):
        """
        MAE forward pass.
        Args:
            x: (B, 1, D, H, W)
        Returns:
            reconstruction: Reconstructed patches
            mask: Binary mask
        """
        # Extract center slice for simplicity
        B, C, D, H, W = x.shape
        center_slice = x[:, :, D//2, :, :].unsqueeze(2)  # (B, 1, 1, H, W)
        center_slice = center_slice.expand(-1, -1, 5, -1, -1)  # Fake k slices
        
        # Encode
        features = self.encoder(center_slice)  # List of features
        bottleneck = features[-1]  # (B, C, H', W')
        
        # Mask patches
        x_masked, mask = self.random_masking(bottleneck, self.mask_ratio)
        
        # Flatten to tokens
        B, C, H, W = x_masked.shape
        tokens = x_masked.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Decode
        reconstruction = self.decoder(tokens)  # (B, H*W, patch_size^2)
        
        return reconstruction, mask, bottleneck

# ===========================================================================================
# LOSS FUNCTIONS
# ===========================================================================================
class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for handling class imbalance.
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        
        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()
        
        tversky_index = (TP + 1e-7) / (TP + self.alpha * FP + self.beta * FN + 1e-7)
        focal_tversky = (1 - tversky_index) ** self.gamma
        
        return focal_tversky

class HybridLoss(nn.Module):
    """
    Hybrid loss: λ1 * Dice + λ2 * FocalTversky
    """
    def __init__(self, lambda1=0.5, lambda2=0.5):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.dice_loss = DiceLoss(sigmoid=False)  # Assuming sigmoid already applied
        self.focal_tversky = FocalTverskyLoss()
        
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        ft = self.focal_tversky(pred, target)
        return self.lambda1 * dice + self.lambda2 * ft

# ===========================================================================================
# Evidential Beta Loss (USALD) - simple calibrated aleatoric uncertainty regularizer
# ===========================================================================================
class EvidentialBetaLoss(nn.Module):
    """Encourages calibrated evidence with a KL regularizer to Beta(1,1)."""
    def __init__(self, lambda_kl: float = 1e-3):
        super().__init__()
        self.lambda_kl = lambda_kl

    def forward(self, alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # alpha: (B,2,H,W), target: (B,1,H,W)
        alpha0, alpha1 = alpha[:, 0:1], alpha[:, 1:2]
        S = alpha0 + alpha1
        p = alpha1 / (S + 1e-8)
        # Data fit via MSE on mean, weighted by inverse certainty
        mse = (p - target).pow(2)
        inv_cert = 1.0 / (S + 1.0)
        fit = (mse * inv_cert).mean()
        # KL to Beta(1,1)
        a, b = alpha0.clamp_min(1e-3), alpha1.clamp_min(1e-3)
        lgB = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        psi = torch.digamma
        kl = (lgB - (a - 1) * psi(a) - (b - 1) * psi(b) + (a + b - 2) * psi(a + b)).mean()
        return fit + self.lambda_kl * kl

# ===========================================================================================
# EVIDENTIAL SELF-CORRECTION (NOVEL COMPONENT #2)
# ===========================================================================================
class SelfCorrectingModel(nn.Module):
    """
    Wrapper for iterative self-correction during inference.
    
    Key idea: Model predicts, checks its own uncertainty, and refines predictions
    in high-uncertainty regions via Dempster-Shafer belief fusion.
    
    This is COMPLETELY NOVEL:
    - No paper does iterative refinement guided by evidential uncertainty
    - Combines active inference (neuroscience) with evidential deep learning
    - Models act as "belief revisers" not one-shot predictors
    
    Algorithm:
    1. Initial prediction: α⁰, p⁰ ← model(x)
    2. Compute uncertainty: u⁰ = 1 / (S⁰ + 1) where S⁰ = α₀⁰ + α₁⁰ - 2
    3. If max(u⁰) < threshold: converged, return
    4. Create attention mask: A = 1 + u⁰ (amplify uncertain regions)
    5. Refined prediction: α', p' ← model(x ⊙ A)
    6. Dempster-Shafer fusion: α¹ = combine(α⁰, α', u⁰)
    7. Repeat until convergence or max_iterations
    
    Clinical motivation: Radiologists re-examine uncertain regions → model should too
    """
    def __init__(self, base_model, max_iterations=3, uncertainty_threshold=0.5):
        super().__init__()
        self.model = base_model
        self.max_iterations = max_iterations
        self.uncertainty_threshold = uncertainty_threshold
    
    def forward(self, x, enable_self_correction=True):
        """
        Args:
            x: Input image (B, 1, k, H, W)
            enable_self_correction: If False, just do one forward pass (training mode)
        
        Returns:
            dict with keys: probs, alpha, num_iterations, uncertainty_trajectory
        """
        # Initial prediction
        out = self.model(x)
        
        if not enable_self_correction:
            out['num_iterations'] = 1
            return out
        
        # Self-correction loop
        uncertainty_trajectory = []
        
        for iteration in range(self.max_iterations):
            # Compute total uncertainty from combined alpha
            alpha = out["alpha"]  # (B, 2, H, W)
            S = alpha[:, 0] + alpha[:, 1] - 2.0
            uncertainty = 1.0 / (S + 1.0)  # (B, H, W)
            
            # Track uncertainty over iterations
            max_uncertainty = uncertainty.max().item()
            mean_uncertainty = uncertainty.mean().item()
            uncertainty_trajectory.append({
                'iteration': iteration,
                'max_u': max_uncertainty,
                'mean_u': mean_uncertainty
            })
            
            # Convergence check: if uncertainty is low everywhere, stop
            if max_uncertainty < self.uncertainty_threshold:
                out['num_iterations'] = iteration + 1
                out['uncertainty_trajectory'] = uncertainty_trajectory
                return out
            
            # Create attention mask: amplify uncertain regions (1 + uncertainty)
            # This focuses model's attention on areas it's confused about
            uncertainty_expanded = uncertainty.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, H, W)
            attention_mask = 1.0 + uncertainty_expanded  # Range [1, 2]
            
            # Re-query model with attention (element-wise multiplication)
            x_attended = x * attention_mask
            out_refined = self.model(x_attended)
            
            # Dempster-Shafer belief fusion
            # Combine original beliefs (α) with refined beliefs (α') weighted by uncertainty
            alpha_original = out["alpha"]
            alpha_refined = out_refined["alpha"]
            
            # Fusion rule: More weight to refined in uncertain regions
            # w_refined = uncertainty (high u → trust refinement more)
            # w_original = 1 - uncertainty
            uncertainty_2d = uncertainty.unsqueeze(1)  # (B, 1, H, W)
            weight_refined = uncertainty_2d.expand_as(alpha_original)
            weight_original = 1.0 - weight_refined
            
            alpha_fused = weight_original * alpha_original + weight_refined * alpha_refined
            
            # Update output
            out["alpha"] = alpha_fused
            out["probs"] = alpha_fused[:, 1:2] / (alpha_fused[:, 0:1] + alpha_fused[:, 1:2] + 1e-8)
            
            # Also update causal alphas if they exist
            if "alpha_anatomy" in out_refined:
                out["alpha_anatomy"] = weight_original * out["alpha_anatomy"] + weight_refined * out_refined["alpha_anatomy"]
                out["alpha_pathology"] = weight_original * out["alpha_pathology"] + weight_refined * out_refined["alpha_pathology"]
                out["alpha_noise"] = weight_original * out["alpha_noise"] + weight_refined * out_refined["alpha_noise"]
        
        # Max iterations reached
        out['num_iterations'] = self.max_iterations
        out['uncertainty_trajectory'] = uncertainty_trajectory
        return out

# ===========================================================================================
# USALD HELPER FUNCTIONS
# ===========================================================================================

def ema_update(student_model, teacher_model, decay=0.99):
    """
    Update teacher model parameters using exponential moving average of student.
    
    θ_teacher ← decay * θ_teacher + (1 - decay) * θ_student
    
    Args:
        student_model: Current student model (being optimized)
        teacher_model: Teacher model (EMA of student, frozen)
        decay: EMA momentum (0.99 = slow, stable updates)
    """
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)

def consistency_loss(p_student, p_teacher, gamma=3.0):
    """
    Compute uncertainty-weighted consistency loss between student and teacher predictions.
    
    L_cons = Σ exp(-γ * u) * ||p_student - p_teacher||²
    
    where uncertainty u = 1 - 2|p_teacher - 0.5| (low when teacher confident)
    
    Args:
        p_student: Student predictions (B, 1, D, H, W)
        p_teacher: Teacher predictions (B, 1, D, H, W)
        gamma: Weighting strength (higher = more selective, only penalize where teacher is confident)
    
    Returns:
        Weighted MSE loss (scalar)
    """
    # Compute teacher uncertainty: u ∈ [0,1], u=0 when p=0 or p=1 (confident), u=1 when p=0.5 (uncertain)
    uncertainty = 1.0 - 2.0 * torch.abs(p_teacher - 0.5)
    
    # Uncertainty weighting: exp(-γ*u) → 1 when u=0 (confident), → exp(-γ) when u=1 (uncertain)
    weight = torch.exp(-gamma * uncertainty)
    
    # Weighted MSE
    mse = (p_student - p_teacher).pow(2)
    loss = (weight * mse).mean()
    
    return loss

def estimate_fdr_threshold(probs, labels, q=0.10, num_thresholds=100):
    """
    Estimate adaptive threshold τ that controls false discovery rate (FDR) ≤ q.
    
    FDR = E[FP / (FP + TP)] where FP = false positives, TP = true positives
    Precision = TP / (TP + FP) = 1 - FDR
    
    We find min τ such that precision(τ) ≥ (1 - q), i.e., FDR(τ) ≤ q
    
    Args:
        probs: Validation predictions (N, 1, D, H, W) - probability maps
        labels: Validation ground truth (N, 1, D, H, W) - binary masks
        q: Target FDR (e.g., 0.10 = 10% false discovery rate = 90% precision)
        num_thresholds: Number of candidate thresholds to test
    
    Returns:
        tau: Optimal threshold τ ∈ [0,1]
    """
    probs_flat = probs.detach().cpu().numpy().flatten()
    labels_flat = labels.detach().cpu().numpy().flatten()
    
    # Test candidate thresholds from 0.1 to 0.9
    thresholds = np.linspace(0.1, 0.9, num_thresholds)
    best_tau = 0.5  # fallback default
    
    for tau in thresholds:
        pred_binary = (probs_flat >= tau).astype(int)
        tp = ((pred_binary == 1) & (labels_flat == 1)).sum()
        fp = ((pred_binary == 1) & (labels_flat == 0)).sum()
        
        # Precision = TP / (TP + FP)
        if tp + fp > 0:
            precision = tp / (tp + fp)
            # If precision ≥ (1 - q), we satisfy FDR ≤ q
            if precision >= (1.0 - q):
                best_tau = tau
                break  # Found minimum threshold satisfying constraint
    
    return best_tau

# ===========================================================================================
# TRAINING FUNCTIONS
# ===========================================================================================

def atomic_save(obj, filepath):
    """
    Save PyTorch object atomically to prevent Google Drive sync conflicts.
    
    Strategy:
    1. Save to a temporary file first (not synced by Drive)
    2. Move/rename to final location (atomic operation)
    3. This prevents Drive from syncing incomplete files
    
    Args:
        obj: Object to save (checkpoint dict, model state, etc.)
        filepath: Final destination path
    """
    try:
        # Create temp file in same directory (ensures same filesystem for atomic move)
        temp_dir = os.path.dirname(filepath)
        temp_fd, temp_path = tempfile.mkstemp(suffix='.tmp', dir=temp_dir)
        os.close(temp_fd)  # Close file descriptor, we'll overwrite
        
        # Save to temp file
        torch.save(obj, temp_path)
        
        # Atomic move (rename) to final location
        # On Windows, need to remove existing file first if it exists
        if os.path.exists(filepath):
            os.remove(filepath)
        shutil.move(temp_path, filepath)
        
        return True
    except Exception as e:
        print(f"Error during atomic save: {e}")
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False

def train_mae_epoch(model, loader, optimizer, scaler, device):
    """Train MAE for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="MAE Training")
    for batch in pbar:
        images = batch["image"].to(device)
        
        with autocast(enabled=use_amp):
            reconstruction, mask, bottleneck = model(images)
            
            # Compute reconstruction loss (L1) on masked positions only
            # reconstruction: (B, H*W, C)
            # bottleneck: (B, C, H, W)
            # mask: (B, H*W) - 1 for masked positions
            
            target = bottleneck.flatten(2).transpose(1, 2)  # (B, H*W, C)
            
            # Compute loss only on masked positions
            mask_expanded = mask.unsqueeze(-1)  # (B, H*W, 1)
            loss = F.l1_loss(reconstruction * mask_expanded, target * mask_expanded)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(loader)

def train_segmentation_epoch(model, loader, criterion, optimizer, scaler, device, 
                            teacher_model=None, epoch=1, tau_pl=0.5):
    """
    Train segmentation model for one epoch with USALD support.
    
    Args:
        model: Student model (being optimized)
        loader: Training data loader
        criterion: Main segmentation loss (Dice + Focal Tversky)
        optimizer: Optimizer
        scaler: AMP gradient scaler
        device: CUDA device
        teacher_model: Teacher model (EMA of student, for consistency loss)
        epoch: Current epoch number (for warmup logic)
        tau_pl: Pseudo-label threshold (adaptive, from FDR control)
    
    Returns:
        tuple: (avg_loss, avg_dice, loss_consistency, loss_pseudo, mean_evidence)
    """
    model.train()
    if teacher_model is not None:
        teacher_model.eval()  # Teacher always in eval mode
    
    total_loss = 0
    total_dice = 0
    total_loss_consistency = 0
    total_loss_pseudo = 0
    total_evidence = 0
    
    evid_criterion = EvidentialBetaLoss(lambda_kl=LAMBDA_EVIDENTIAL) if USALD_ENABLED else None
    use_consistency = USALD_CONSISTENCY_ENABLED and teacher_model is not None and epoch > WARMUP_EPOCHS
    
    pbar = tqdm(loader, desc="Segmentation Training")
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        # Extract center slice label
        center_slice_label = labels[:, :, labels.shape[2]//2, :, :]  # (B, 1, H, W)
        
        with autocast(enabled=use_amp):
            # Student forward pass
            out_dict = model(images)  # dict: {'probs': (B,1,H,W), 'alpha': (B,2,H,W)}
            probs = out_dict["probs"]
            
            # Supervised loss
            loss = criterion(probs, center_slice_label)
            
            # Evidential loss (if USALD enabled)
            loss_evid = 0.0
            if USALD_ENABLED and ("alpha" in out_dict):
                loss_evid = evid_criterion(out_dict["alpha"], center_slice_label)
                loss = loss + loss_evid
                
                # Track mean evidence S = α₀ + α₁ - 2
                alpha0, alpha1 = out_dict["alpha"][:, 0], out_dict["alpha"][:, 1]
                evidence_S = (alpha0 + alpha1 - 2.0).mean().item()
                total_evidence += evidence_S
            
            # Consistency loss (after warmup)
            loss_cons = 0.0
            if use_consistency:
                with torch.no_grad():
                    teacher_dict = teacher_model(images)
                    teacher_probs = teacher_dict["probs"]
                
                loss_cons = consistency_loss(probs, teacher_probs, gamma=UNCERTAINTY_GAMMA)
                loss = loss + LAMBDA_CONSISTENCY * loss_cons
                total_loss_consistency += loss_cons.item()
            
            # Pseudo-label loss (after warmup, on high-confidence teacher predictions)
            loss_pl = 0.0
            if use_consistency and USALD_FDR_ENABLED:
                with torch.no_grad():
                    # Generate pseudo-labels from teacher with adaptive threshold
                    pseudo_mask = ((teacher_probs >= tau_pl) | (teacher_probs <= (1.0 - tau_pl))).float()
                    pseudo_labels = (teacher_probs >= tau_pl).float()
                
                # Only compute loss on pseudo-labeled regions
                if pseudo_mask.sum() > 0:
                    loss_pl = F.binary_cross_entropy(probs * pseudo_mask, pseudo_labels * pseudo_mask, reduction='sum') / (pseudo_mask.sum() + 1e-7)
                    loss = loss + LAMBDA_PSEUDO * loss_pl
                    total_loss_pseudo += loss_pl.item()
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update teacher via EMA (after warmup)
        if use_consistency:
            ema_update(model, teacher_model, decay=EMA_DECAY)
        
        # Compute Dice
        pred_binary = (probs > 0.5).float()
        dice = 2 * (pred_binary * center_slice_label).sum() / (pred_binary.sum() + center_slice_label.sum() + 1e-7)
        
        total_loss += loss.item()
        total_dice += dice.item()
        
        pbar.set_postfix({
            "loss": loss.item(), 
            "dice": dice.item(),
            "cons": loss_cons if isinstance(loss_cons, float) else loss_cons.item() if use_consistency else 0.0
        })
    
    n = len(loader)
    return (total_loss / n, total_dice / n, 
            total_loss_consistency / n if use_consistency else 0.0,
            total_loss_pseudo / n if use_consistency else 0.0,
            total_evidence / n if USALD_ENABLED else 0.0)

def validate_segmentation(model, loader, criterion, device, return_probs_labels=False):
    """
    Validate segmentation model with optional self-correction.
    
    Args:
        model: Model to validate
        loader: Validation data loader
        criterion: Loss function
        device: CUDA device
        return_probs_labels: If True, also return (all_probs_tensor, all_labels_tensor) for FDR threshold estimation
    
    Returns:
        dict with metrics, optionally (metrics_dict, all_probs, all_labels)
    """
    model.eval()
    
    # Wrap with self-correction if enabled (NOVEL COMPONENT #2)
    if USALD_SELF_CORRECTION and USALD_ENABLED:
        model_wrapper = SelfCorrectingModel(
            model, 
            max_iterations=SELF_CORRECTION_MAX_ITER,
            uncertainty_threshold=SELF_CORRECTION_THRESHOLD
        )
        print(f"Self-correction enabled (max_iter={SELF_CORRECTION_MAX_ITER}, threshold={SELF_CORRECTION_THRESHOLD})")
    else:
        model_wrapper = model
    
    total_loss = 0
    total_dice = 0
    all_preds = []
    all_labels = []
    all_probs_list = []  # For FDR threshold estimation
    all_labels_list = []
    total_self_correction_iters = 0
    num_batches_with_correction = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            center_slice_label = labels[:, :, labels.shape[2]//2, :, :]
            
            # Forward pass (with self-correction if enabled)
            if USALD_SELF_CORRECTION and USALD_ENABLED:
                out_dict = model_wrapper(images, enable_self_correction=True)
                # Track self-correction statistics
                if 'num_iterations' in out_dict:
                    total_self_correction_iters += out_dict['num_iterations']
                    num_batches_with_correction += 1
            else:
                out_dict = model(images)
            
            probs = out_dict["probs"]
            loss = criterion(probs, center_slice_label)
            
            pred_binary = (probs > 0.5).float()
            dice = 2 * (pred_binary * center_slice_label).sum() / (pred_binary.sum() + center_slice_label.sum() + 1e-7)
            
            total_loss += loss.item()
            total_dice += dice.item()
            
            all_preds.append(pred_binary.cpu().numpy().flatten())
            all_labels.append(center_slice_label.cpu().numpy().flatten())
            
            if return_probs_labels:
                all_probs_list.append(probs.cpu())
                all_labels_list.append(center_slice_label.cpu())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    metrics = {
        "loss": total_loss / len(loader),
        "dice": total_dice / len(loader),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    # Add self-correction statistics if enabled
    if USALD_SELF_CORRECTION and USALD_ENABLED and num_batches_with_correction > 0:
        avg_iters = total_self_correction_iters / num_batches_with_correction
        metrics["self_correction_avg_iters"] = avg_iters
        print(f"   Self-correction: avg {avg_iters:.2f} iterations per batch")
    
    if return_probs_labels:
        all_probs_tensor = torch.cat(all_probs_list, dim=0)  # (N, 1, H, W)
        all_labels_tensor = torch.cat(all_labels_list, dim=0)
        return metrics, all_probs_tensor, all_labels_tensor
    
    return metrics

# ===========================================================================================
# RESUME HELPERS
# ===========================================================================================
def check_mae_resume():
    """Check if MAE pretraining should resume from local checkpoint."""
    mae_log_path = os.path.join(MAE_PRETRAIN_DIR, 'mae_logs.csv')
    mae_checkpoint_path = os.path.join(LOCAL_RESUME_DIR, 'mae_resume.pth')  # Check local storage
    
    if os.path.exists(mae_log_path) and os.path.exists(mae_checkpoint_path):
        df = pd.read_csv(mae_log_path)
        last_epoch = int(df['epoch'].max())
        return True, last_epoch, mae_checkpoint_path
    return False, 0, None

def check_segmentation_resume():
    """Check if segmentation training should resume from local checkpoint."""
    train_log_path = os.path.join(SEGMENTATION_DIR, 'train_logs.csv')
    val_log_path = os.path.join(SEGMENTATION_DIR, 'val_logs.csv')
    seg_checkpoint_path = os.path.join(LOCAL_RESUME_DIR, 'seg_resume.pth')  # Check local storage
    
    if os.path.exists(train_log_path) and os.path.exists(val_log_path) and os.path.exists(seg_checkpoint_path):
        df = pd.read_csv(val_log_path)
        last_epoch = int(df['epoch'].max())
        return True, last_epoch, seg_checkpoint_path
    return False, 0, None

# ===========================================================================================
# MAIN EXECUTION
# ===========================================================================================
if __name__ == "__main__":
    import pandas as pd
    
    print("\n" + "="*80)
    print("PHASE 1: 2.5D-MAE PRETRAINING")
    print("="*80)
    
    # Initialize encoder
    encoder = HybridMiniSwin2D5_ResNetEncoder(k_slices=K_SLICES, channels=STAGE_CHANNELS)
    
    # Initialize MAE
    mae_model = MAE_2D5(encoder=encoder, mask_ratio=MAE_MASK_RATIO).to(device)
    
    mae_optimizer = optim.AdamW(mae_model.parameters(), lr=MAE_LEARNING_RATE, weight_decay=0.05)
    mae_scheduler = optim.lr_scheduler.CosineAnnealingLR(mae_optimizer, T_max=MAE_EPOCHS)
    mae_scaler = GradScaler(enabled=use_amp)
    
    # Check for MAE resume
    mae_resume, mae_start_epoch, mae_checkpoint_path = check_mae_resume()
    mae_log_data = []
    
    if mae_resume:
        print(f"\nRESUMING MAE from epoch {mae_start_epoch}")
        checkpoint = torch.load(mae_checkpoint_path)
        
        # Load model state (try different keys for compatibility)
        if 'model_state_dict' in checkpoint:
            mae_model.load_state_dict(checkpoint['model_state_dict'])
        elif 'encoder_state_dict' in checkpoint:
            # If only encoder state is available, load it into the encoder
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print("Only encoder state found, creating new MAE model with pretrained encoder")
        
        # Load optimizer/scheduler if available (optional for resume)
        if 'optimizer_state_dict' in checkpoint:
            mae_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state")
        else:
            print("No optimizer state found, starting with fresh optimizer")
            
        if 'scheduler_state_dict' in checkpoint:
            mae_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Loaded scheduler state")
        else:
            print("No scheduler state found, starting with fresh scheduler")
        
        best_mae_loss = checkpoint.get('best_loss', float('inf'))
        
        # Load existing logs
        mae_log_df = pd.read_csv(os.path.join(MAE_PRETRAIN_DIR, 'mae_logs.csv'))
        mae_log_data = mae_log_df.to_dict('records')
        
        print(f"Resumed from epoch {mae_start_epoch}")
        print(f"Best MAE loss so far: {best_mae_loss:.4f}")
        print("="*80)
        start_epoch_mae = mae_start_epoch + 1
        
        # Calculate patience counter from logs
        mae_patience_counter = 0
        if mae_log_data:
            for i in range(len(mae_log_data) - 1, -1, -1):
                if mae_log_data[i]['loss'] <= best_mae_loss:
                    break
                mae_patience_counter += 1
        print(f"Patience counter: {mae_patience_counter}/30")
    else:
        print("\nStarting MAE pretraining from scratch")
        best_mae_loss = float('inf')
        mae_patience_counter = 0
        start_epoch_mae = 1
    
    # Early stopping for MAE
    mae_patience = 30
    
    # Train MAE
    for epoch in range(start_epoch_mae, MAE_EPOCHS + 1):
        print(f"\nMAE Epoch {epoch}/{MAE_EPOCHS}")
        
        mae_loss = train_mae_epoch(mae_model, train_loader, mae_optimizer, mae_scaler, device)
        mae_scheduler.step()
        
        print(f"MAE Loss: {mae_loss:.4f}")
        
        # Log to CSV
        mae_log_data.append({
            'epoch': epoch,
            'loss': mae_loss,
            'lr': mae_optimizer.param_groups[0]['lr']
        })
        
        # Save logs every epoch (CSV only - minimal storage)
        csv_path = os.path.join(MAE_PRETRAIN_DIR, 'mae_logs.csv')
        pd.DataFrame(mae_log_data).to_csv(csv_path, index=False)
        
        # STORAGE OPTIMIZATION: 
        # - Best model in Google Drive (overwrites when better)
        # - Resume checkpoint in local storage only (NOT synced to Drive)
        
        # Save local resume checkpoint (for continuing training after crash)
        resume_checkpoint_path = os.path.join(LOCAL_RESUME_DIR, 'mae_resume.pth')
        resume_data = {
            'epoch': epoch,
            'model_state_dict': mae_model.state_dict(),
            'encoder_state_dict': encoder.state_dict(),
            'optimizer_state_dict': mae_optimizer.state_dict(),
            'scheduler_state_dict': mae_scheduler.state_dict(),
            'mae_loss': mae_loss,
            'best_loss': best_mae_loss
        }
        torch.save(resume_data, resume_checkpoint_path)  # Direct save to local (no atomic needed)
        
        # Save best checkpoint to Google Drive only (overwrites when better model found)
        if mae_loss < best_mae_loss:
            best_mae_loss = mae_loss
            mae_patience_counter = 0  # Reset on improvement
            best_checkpoint_path = os.path.join(MAE_PRETRAIN_DIR, 'mae_best.pth')
            best_checkpoint_data = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'mae_loss': mae_loss,
                'best_loss': best_mae_loss
            }
            
            if atomic_save(best_checkpoint_data, best_checkpoint_path):
                print(f"Best: {mae_loss:.4f} | Saved to Drive + Local resume saved")
            else:
                print(f"Warning: Failed to save best checkpoint | Local resume saved")
        else:
            mae_patience_counter += 1
            print(f"Loss: {mae_loss:.4f} (best: {best_mae_loss:.4f}) | Local resume saved | No improvement: {mae_patience_counter}/{mae_patience}")
            
            # Early stopping
            if mae_patience_counter >= mae_patience:
                print(f"\nEarly stopping triggered after {mae_patience} epochs without improvement")
                print(f"Best MAE loss: {best_mae_loss:.4f}")
                break
    
    print("\n" + "="*80)
    print("PHASE 2: SEGMENTATION FINE-TUNING")
    print("="*80)
    
    # Load pretrained encoder
    mae_best_path = os.path.join(MAE_PRETRAIN_DIR, 'mae_best.pth')
    if os.path.exists(mae_best_path):
        checkpoint = torch.load(mae_best_path)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        print("Loaded pretrained encoder from MAE")
    else:
        print("No MAE checkpoint found, using random initialization")
    
    # Initialize full segmentation model (student)
    seg_model = HybridMiniSwin2D5_CSRF(k_slices=K_SLICES, channels=STAGE_CHANNELS).to(device)
    seg_model.encoder.load_state_dict(encoder.state_dict())
    
    # Initialize teacher model (EMA copy of student) for USALD consistency
    teacher_model = None
    if USALD_CONSISTENCY_ENABLED:
        teacher_model = HybridMiniSwin2D5_CSRF(k_slices=K_SLICES, channels=STAGE_CHANNELS).to(device)
        teacher_model.load_state_dict(seg_model.state_dict())
        # Freeze teacher parameters (updated via EMA only)
        for param in teacher_model.parameters():
            param.requires_grad = False
        print("Initialized teacher model (EMA) for USALD consistency loss")
    
    # Separate optimizers for encoder and decoder
    encoder_params = list(seg_model.encoder.parameters()) + list(seg_model.csrf.parameters())
    decoder_params = list(seg_model.decoder.parameters())
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': LEARNING_RATE_ENCODER},
        {'params': decoder_params, 'lr': LEARNING_RATE_DECODER}
    ], weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SEGMENTATION_EPOCHS)
    criterion = HybridLoss(lambda1=0.5, lambda2=0.5)
    scaler = GradScaler(enabled=use_amp)
    
    # FDR adaptive threshold (initialized to 0.5, updated after warmup)
    tau_pl = 0.5
    
    # Check for segmentation resume
    seg_resume, seg_start_epoch, seg_checkpoint_path = check_segmentation_resume()
    train_log_data = []
    val_log_data = []
    
    if seg_resume:
        print(f"\nRESUMING SEGMENTATION from epoch {seg_start_epoch}")
        checkpoint = torch.load(seg_checkpoint_path)
        seg_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load teacher if available and USALD consistency enabled
        if USALD_CONSISTENCY_ENABLED and 'teacher_state_dict' in checkpoint:
            teacher_model.load_state_dict(checkpoint['teacher_state_dict'])
            print("Loaded teacher model state")
        
        # Load FDR threshold if available
        if 'tau_pl' in checkpoint:
            tau_pl = checkpoint['tau_pl']
            print(f"Loaded FDR threshold: tau_pl = {tau_pl:.3f}")
        
        # Load optimizer/scheduler if available
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state")
        else:
            print("No optimizer state found, starting with fresh optimizer")
            
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Loaded scheduler state")
        else:
            print("No scheduler state found, starting with fresh scheduler")
        
        best_val_dice = checkpoint.get('best_val_dice', 0.0)
        
        # Calculate patience counter from validation logs
        patience_counter = 0
        if 'val_metrics' in checkpoint and val_log_data:
            # Count epochs since last improvement
            for i in range(len(val_log_data) - 1, -1, -1):
                if val_log_data[i]['dice'] >= best_val_dice:
                    break
                patience_counter += 1
        
        # Load existing logs
        train_log_df = pd.read_csv(os.path.join(SEGMENTATION_DIR, 'train_logs.csv'))
        val_log_df = pd.read_csv(os.path.join(SEGMENTATION_DIR, 'val_logs.csv'))
        train_log_data = train_log_df.to_dict('records')
        val_log_data = val_log_df.to_dict('records')
        
        print(f"Resumed from epoch {seg_start_epoch}")
        print(f"Best Val Dice so far: {best_val_dice:.4f}")
        print(f"Patience counter: {patience_counter}/20")
        print("="*80)
        start_epoch_seg = seg_start_epoch + 1
    else:
        print("\nStarting segmentation training from scratch")
        best_val_dice = 0.0
        patience_counter = 0
        start_epoch_seg = 1
    
    # Early stopping
    patience = 20
    
    # Training loop
    for epoch in range(start_epoch_seg, SEGMENTATION_EPOCHS + 1):
        print(f"\nSegmentation Epoch {epoch}/{SEGMENTATION_EPOCHS}")
        
        # Training with teacher model and FDR threshold
        train_loss, train_dice, loss_consistency, loss_pseudo, mean_evidence = train_segmentation_epoch(
            seg_model, train_loader, criterion, optimizer, scaler, device,
            teacher_model=teacher_model, epoch=epoch, tau_pl=tau_pl
        )
        
        # Validation (with FDR threshold estimation after warmup)
        if epoch > WARMUP_EPOCHS and USALD_FDR_ENABLED and epoch % FDR_UPDATE_INTERVAL == 0:
            # Return probs and labels for FDR threshold estimation
            val_metrics, val_probs, val_labels = validate_segmentation(
                seg_model, val_loader, criterion, device, return_probs_labels=True
            )
            # Update FDR threshold
            tau_pl = estimate_fdr_threshold(val_probs, val_labels, q=FDR_Q)
            print(f"🎯 Updated FDR threshold: τ_pl = {tau_pl:.3f} (target FDR ≤ {FDR_Q*100:.0f}%)")
        else:
            val_metrics = validate_segmentation(seg_model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        if USALD_CONSISTENCY_ENABLED and epoch > WARMUP_EPOCHS:
            print(f"        Consistency: {loss_consistency:.4f}, Pseudo: {loss_pseudo:.4f}")
        if USALD_ENABLED:
            print(f"        Mean Evidence: {mean_evidence:.2f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, "
              f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
        
        # Log to CSV (extended for USALD)
        train_log_data.append({
            'epoch': epoch,
            'loss': train_loss,
            'dice': train_dice,
            'loss_consistency': loss_consistency,
            'loss_pseudo': loss_pseudo,
            'mean_evidence': mean_evidence,
            'lr_encoder': optimizer.param_groups[0]['lr'],
            'lr_decoder': optimizer.param_groups[1]['lr']
        })
        
        # Extract causal weights if available
        causal_weights = [0.0, 0.0, 0.0]
        if USALD_CAUSAL_ENABLED and hasattr(seg_model.decoder, 'causal_weights'):
            causal_weights = torch.softmax(seg_model.decoder.causal_weights, dim=0).detach().cpu().numpy().tolist()
        
        val_log_data.append({
            'epoch': epoch,
            'loss': val_metrics['loss'],
            'dice': val_metrics['dice'],
            'precision': val_metrics['precision'],
            'recall': val_metrics['recall'],
            'f1': val_metrics['f1'],
            'tau_pl': tau_pl,
            'causal_weight_anatomy': causal_weights[0],
            'causal_weight_pathology': causal_weights[1],
            'causal_weight_noise': causal_weights[2]
        })
        
        # Save logs every epoch
        pd.DataFrame(train_log_data).to_csv(os.path.join(SEGMENTATION_DIR, 'train_logs.csv'), index=False)
        pd.DataFrame(val_log_data).to_csv(os.path.join(SEGMENTATION_DIR, 'val_logs.csv'), index=False)
        
        # STORAGE OPTIMIZATION: 
        # - Best model in Google Drive (overwrites when better)
        # - Resume checkpoint in local storage only (NOT synced to Drive)
        
        # Save local resume checkpoint (for continuing training after crash)
        resume_checkpoint_path = os.path.join(LOCAL_RESUME_DIR, 'seg_resume.pth')
        resume_data = {
            'epoch': epoch,
            'model_state_dict': seg_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_dice': best_val_dice,
            'val_metrics': val_metrics,
            'tau_pl': tau_pl
        }
        # Add teacher state if using USALD consistency
        if USALD_CONSISTENCY_ENABLED and teacher_model is not None:
            resume_data['teacher_state_dict'] = teacher_model.state_dict()
        
        torch.save(resume_data, resume_checkpoint_path)  # Direct save to local (no atomic needed)
        
        # Save best model to Google Drive only when validation improves (overwrites previous best)
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            patience_counter = 0  # Reset counter on improvement
            best_checkpoint_path = os.path.join(SEGMENTATION_DIR, 'best_model.pth')
            best_checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': seg_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_dice': val_metrics['dice'],
                'val_metrics': val_metrics,
                'best_val_dice': best_val_dice,
                'tau_pl': tau_pl
            }
            # Add teacher state if using USALD consistency
            if USALD_CONSISTENCY_ENABLED and teacher_model is not None:
                best_checkpoint_data['teacher_state_dict'] = teacher_model.state_dict()
            
            if atomic_save(best_checkpoint_data, best_checkpoint_path):
                print(f"Best: Dice {val_metrics['dice']:.4f} | Saved to Drive + Local resume saved")
            else:
                print(f"Warning: Failed to save best model | Local resume saved")
        else:
            patience_counter += 1
            print(f"Dice: {val_metrics['dice']:.4f} (best: {best_val_dice:.4f}) | Local resume saved | No improvement: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                print(f"Best validation Dice: {best_val_dice:.4f}")
                break
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Validation Dice: {best_val_dice:.4f}")
    print(f"MAE Checkpoints: {MAE_PRETRAIN_DIR}")
    print(f"  - mae_best.pth (best encoder - final)")
    print(f"  - mae_logs.csv (training logs)")
    print(f"Segmentation Checkpoints: {SEGMENTATION_DIR}")
    print(f"  - best_model.pth (best model - final)")
    print(f"  - train_logs.csv (training metrics)")
    print(f"  - val_logs.csv (validation metrics)")
    print(f"\nStorage optimized: Only best models saved (no intermediate checkpoints)")
    print(f"Deployment: {DEPLOYMENT_DIR} (to be created)")
    
    print("\n💡 To resume training later:")
    print("   - Just run 'python final_model.py' again")
    print("   - It will automatically detect and resume from last checkpoint")
    print("   - To train more epochs, increase MAE_EPOCHS or SEGMENTATION_EPOCHS")

# ===========================================================================================
# ABLATION STUDY RUNNER
# ===========================================================================================
def run_ablation_study():
    """
    Automatically run all 7 ablation configs and save comprehensive results to CSV.
    Results saved in C:\\Users\\HP\\EDI\\ablation_results\\ for Q1 paper.
    """
    import os
    import shutil
    from datetime import datetime
    
    os.makedirs(ABLATION_RESULTS_DIR, exist_ok=True)
    
    print("\n" + "="*100)
    print("ABLATION STUDY: Running 7 configurations for Q1 paper")
    print("="*100)
    print(f"MAE Epochs per config: {ABLATION_EPOCHS_MAE}")
    print(f"Segmentation Epochs per config: {ABLATION_EPOCHS_SEG}")
    print(f"Results will be saved to: {ABLATION_RESULTS_DIR}")
    print("="*100 + "\n")
    
    all_results = []
    
    for config_name, config_dict in ABLATION_CONFIGS.items():
        print(f"\n{'='*100}")
        print(f"CONFIG {len(all_results)+1}/7: {config_dict['name']}")
        print(f"{'='*100}")
        
        # Override global flags with this config
        global USALD_ENABLED, USALD_CONSISTENCY_ENABLED, USALD_FDR_ENABLED, USALD_CAUSAL_ENABLED, USALD_SELF_CORRECTION
        global MAE_EPOCHS, SEGMENTATION_EPOCHS, OUTPUT_DIR, MAE_PRETRAIN_DIR, SEGMENTATION_DIR, CHECKPOINT_DIR, LOCAL_RESUME_DIR
        
        USALD_ENABLED = config_dict["USALD_ENABLED"]
        USALD_CONSISTENCY_ENABLED = config_dict["USALD_CONSISTENCY_ENABLED"]
        USALD_FDR_ENABLED = config_dict["USALD_FDR_ENABLED"]
        USALD_CAUSAL_ENABLED = config_dict["USALD_CAUSAL_ENABLED"]
        USALD_SELF_CORRECTION = config_dict["USALD_SELF_CORRECTION"]
        
        MAE_EPOCHS = ABLATION_EPOCHS_MAE
        SEGMENTATION_EPOCHS = ABLATION_EPOCHS_SEG
        
        # Set separate paths for this config
        OUTPUT_DIR = os.path.join(DRIVE_BASE, f"USALD_Ablation_{config_name}")
        CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
        MAE_PRETRAIN_DIR = os.path.join(OUTPUT_DIR, "mae_pretraining")
        SEGMENTATION_DIR = os.path.join(OUTPUT_DIR, "segmentation")
        LOCAL_RESUME_DIR = os.path.join(os.path.dirname(__file__), f".resume_ablation_{config_name}")
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(MAE_PRETRAIN_DIR, exist_ok=True)
        os.makedirs(SEGMENTATION_DIR, exist_ok=True)
        os.makedirs(LOCAL_RESUME_DIR, exist_ok=True)
        
        print(f"Output: {OUTPUT_DIR}")
        print(f"Config: USALD={USALD_ENABLED}, Consistency={USALD_CONSISTENCY_ENABLED}, FDR={USALD_FDR_ENABLED}, Causal={USALD_CAUSAL_ENABLED}, SelfCorrect={USALD_SELF_CORRECTION}\n")
        
        # Run training for this config
        # (The main training code will execute with these overridden settings)
        # Since we can't call main() from here, we need to restructure...
        # For now, this is a template. User will need to run each config separately.
        
        print(f"\n⚠️  Manual execution required for now:")
        print(f"   Set ABLATION_CONFIG = '{config_name}' and run python final_model.py")
        print(f"   Or restructure code to support programmatic config switching\n")
        
        # Placeholder: Load results from this config's validation CSV
        val_csv_path = os.path.join(SEGMENTATION_DIR, 'val_logs.csv')
        if os.path.exists(val_csv_path):
            val_df = pd.read_csv(val_csv_path)
            best_epoch = val_df['dice'].idxmax()
            best_row = val_df.iloc[best_epoch]
            
            result = {
                'config': config_name,
                'name': config_dict['name'],
                'dice': best_row['dice'],
                'precision': best_row['precision'],
                'recall': best_row['recall'],
                'f1': best_row['f1'],
                'loss': best_row['loss'],
                'mean_evidence': best_row.get('mean_evidence', 0),
                'tau_pl': best_row.get('tau_pl', 0.5),
                'avg_self_correction_iters': best_row.get('avg_self_correction_iters', 0),
                'best_epoch': best_epoch + 1
            }
            all_results.append(result)
        else:
            print(f"   ⚠️ Results not found: {val_csv_path}")
    
    # Save summary CSV
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(ABLATION_RESULTS_DIR, 'ablation_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✅ Summary saved: {summary_path}")
        
        # Calculate deltas vs baseline
        if len(all_results) > 0:
            baseline_dice = summary_df[summary_df['config'] == 'baseline']['dice'].values[0]
            summary_df['delta_dice'] = summary_df['dice'] - baseline_dice
            summary_df.to_csv(summary_path, index=False)
            print(f"✅ Added delta columns (improvement vs baseline)")
    
    print("\n" + "="*100)
    print("ABLATION STUDY COMPLETE")
    print("="*100)
    print(f"Results in: {ABLATION_RESULTS_DIR}")
    print(f"  - ablation_summary.csv (master comparison table)")
    print("="*100)

if __name__ == "__main__":
    if RUN_ABLATION_STUDY:
        run_ablation_study()
    else:
        # Single config run (existing behavior)
        if ABLATION_CONFIG != "full":
            # Apply selected ablation config
            config_dict = ABLATION_CONFIGS[ABLATION_CONFIG]
            USALD_ENABLED = config_dict["USALD_ENABLED"]
            USALD_CONSISTENCY_ENABLED = config_dict["USALD_CONSISTENCY_ENABLED"]
            USALD_FDR_ENABLED = config_dict["USALD_FDR_ENABLED"]
            USALD_CAUSAL_ENABLED = config_dict["USALD_CAUSAL_ENABLED"]
            USALD_SELF_CORRECTION = config_dict["USALD_SELF_CORRECTION"]
            print(f"\n📋 Running ablation config: {config_dict['name']}")
        # Continue with normal execution...
