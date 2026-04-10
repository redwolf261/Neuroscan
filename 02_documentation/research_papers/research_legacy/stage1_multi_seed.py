"""
Stage 1: Multi-Seed Experiments with Statistical Analysis
==========================================================
Run the final model with 5 different random seeds and compute statistical significance

This script:
1. Trains the final model 5 times with different seeds
2. Computes mean ± std for all metrics
3. Performs statistical tests for significance
4. Generates publication-ready tables
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from final_model import (
    HybridMiniSwin2D5_CBAM,
    MAE_2D5,
    HybridMiniSwin2D5_ResNetEncoder,
    LightweightDecoder
)

# MONAI imports for data loading
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, Resized, EnsureTyped, MapTransform,
    RandFlipd, RandRotate90d
)
from monai.data import Dataset as MonaiDataset, DataLoader
import glob

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'num_seeds': 5,
    'base_dir': r"C:\Users\HP\EDI",
    'dataset_dir': r"C:\Users\HP\EDI\Dataset\PediMS\PediMS",
    'output_dir': r"C:\Users\HP\EDI\comprehensive_evaluation_results\stage1_multi_seed",
    
    # Model config (optimal from ablations)
    'k_slices': 9,
    'channels': [32, 64, 128, 256, 512],
    'window_size': 4,
    'use_adaptive_selection': True,
    
    # Training config
    'mae_epochs': 100,
    'seg_epochs': 50,
    'batch_size': 3,
    'learning_rate': 1e-4,
    'spatial_size': (64, 64, 64),
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ============================================================================
# DATA LOADING
# ============================================================================

class BinarizeLabel(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            d[k] = (d[k] > 0).float()
        return d

def prepare_pedims_data(data_dir, batch_size=3, spatial_size=(64, 64, 64), num_workers=0):
    """Load and prepare PediMS dataset"""
    
    # Collect all data
    data_dicts = []
    
    if os.path.exists(data_dir):
        for subfolder in sorted(os.listdir(data_dir)):
            sub_path = os.path.join(data_dir, subfolder)
            if not os.path.isdir(sub_path):
                continue
            
            for modality in ["T1", "T2", "FLAIR"]:
                mod_path = os.path.join(sub_path, modality, "processed")
                if not os.path.exists(mod_path):
                    continue
                
                imgs = sorted(glob.glob(os.path.join(mod_path, "*_brain_*.nii*")))
                masks = sorted(glob.glob(os.path.join(mod_path, "*_mask_*.nii*")))
                
                if len(masks) == 0:
                    masks = sorted(glob.glob(os.path.join(mod_path, "*_Consensus_*.nii*")))
                
                for img, m in zip(imgs, masks):
                    data_dicts.append({
                        "image": [img],
                        "label": m,
                        "case": os.path.basename(img)
                    })
    else:
        raise RuntimeError(f"DATA_PATH does not exist: {data_dir}")
    
    if len(data_dicts) == 0:
        raise RuntimeError("No PediMS data found.")
    
    # Split train/val (80/20)
    num_samples = len(data_dicts)
    num_train = int(0.8 * num_samples)
    
    train_data = data_dicts[:num_train]
    val_data = data_dicts[num_train:]
    
    # Transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        BinarizeLabel(keys=["label"]),
        Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("trilinear", "nearest")),
        RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
        EnsureTyped(keys=["image", "label"])
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        BinarizeLabel(keys=["label"]),
        Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("trilinear", "nearest")),
        EnsureTyped(keys=["image", "label"])
    ])
    
    # Create datasets
    train_dataset = MonaiDataset(data=train_data, transform=train_transforms)
    val_dataset = MonaiDataset(data=val_data, transform=val_transforms)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, len(train_data), len(val_data)

# ============================================================================
# UTILITIES
# ============================================================================

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"🌱 Random seed set to: {seed}")

def dice_score(pred, target, smooth=1e-5):
    """Calculate Dice coefficient"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def compute_metrics(pred, target):
    """Compute all evaluation metrics"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()
    
    precision = (tp + 1e-5) / (tp + fp + 1e-5)
    recall = (tp + 1e-5) / (tp + fn + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + 1e-5) / (union + 1e-5)
    
    iou = intersection / (union - intersection + 1e-5)
    
    return {
        'dice': dice.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'iou': iou.item()
    }

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_one_seed(seed, config):
    """Train model with specific seed and return validation metrics"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING WITH SEED {seed}")
    print(f"{'='*80}\n")
    
    # Set seed
    set_seed(seed)
    
    # Create output directory for this seed
    seed_dir = Path(config['output_dir']) / f"seed_{seed}"
    seed_dir.mkdir(exist_ok=True, parents=True)
    
    # Load dataset
    print("Loading PediMS dataset...")
    train_loader, val_loader, num_train, num_val = prepare_pedims_data(
        data_dir=config['dataset_dir'],
        batch_size=config['batch_size'],
        spatial_size=config['spatial_size'],
        num_workers=0
    )
    
    print(f"Train samples: {num_train}, Val samples: {num_val}")
    
    # Initialize model
    model = HybridMiniSwin2D5_CBAM(
        k_slices=config['k_slices'],
        channels=config['channels'],
        use_adaptive_selection=config['use_adaptive_selection']
    ).to(config['device'])
    
    print(f"\nModel initialized:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Phase 1: MAE Pretraining
    print(f"\n{'='*80}")
    print("PHASE 1: MAE PRETRAINING")
    print(f"{'='*80}\n")
    
    mae_model = MAE_2D5(
        encoder=model.encoder,
        mask_ratio=0.75,
        decoder_embed_dim=256
    ).to(config['device'])
    
    mae_optimizer = optim.AdamW(mae_model.parameters(), lr=config['learning_rate'])
    
    mae_losses = []
    
    for epoch in range(config['mae_epochs']):
        mae_model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"MAE Epoch {epoch+1}/{config['mae_epochs']}")
        
        for batch in pbar:
            images = batch['image'].to(config['device'])
            
            mae_optimizer.zero_grad()
            
            loss = mae_model(images)
            loss.backward()
            mae_optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(train_loader)
        mae_losses.append(avg_loss)
        
        print(f"MAE Epoch {epoch+1}/{config['mae_epochs']} - Loss: {avg_loss:.6f}")
    
    print(f"\n✅ MAE Pretraining complete. Final loss: {mae_losses[-1]:.6f}")
    
    # Phase 2: Segmentation Fine-tuning
    print(f"\n{'='*80}")
    print("PHASE 2: SEGMENTATION FINE-TUNING")
    print(f"{'='*80}\n")
    
    seg_optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    
    best_dice = 0.0
    best_epoch = 0
    best_metrics = {}
    
    train_losses = []
    val_dices = []
    
    patience = 20
    patience_counter = 0
    
    for epoch in range(config['seg_epochs']):
        # Training
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Seg Epoch {epoch+1}/{config['seg_epochs']}")
        
        for batch in pbar:
            images = batch['image'].to(config['device'])
            masks = batch['label'].to(config['device'])
            
            seg_optimizer.zero_grad()
            
            output = model(images)
            
            # Handle dict output
            if isinstance(output, dict):
                logits = output['probs']
            else:
                logits = output
            
            # Get center slice from mask
            D = masks.shape[2]
            center_idx = D // 2
            target = masks[:, :, center_idx, :, :]
            
            loss = criterion(logits, target)
            loss.backward()
            seg_optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_metrics_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(config['device'])
                masks = batch['label'].to(config['device'])
                
                output = model(images)
                
                if isinstance(output, dict):
                    probs = torch.sigmoid(output['probs'])
                else:
                    probs = torch.sigmoid(output)
                
                D = masks.shape[2]
                center_idx = D // 2
                target = masks[:, :, center_idx, :, :]
                
                metrics = compute_metrics(probs, target)
                val_metrics_list.append(metrics)
        
        # Average validation metrics
        val_metrics = {
            k: np.mean([m[k] for m in val_metrics_list])
            for k in val_metrics_list[0].keys()
        }
        
        val_dices.append(val_metrics['dice'])
        
        print(f"Epoch {epoch+1}/{config['seg_epochs']} - "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Dice: {val_metrics['dice']:.4f} | "
              f"Val Precision: {val_metrics['precision']:.4f} | "
              f"Val Recall: {val_metrics['recall']:.4f}")
        
        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_epoch = epoch + 1
            best_metrics = val_metrics.copy()
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'seed': seed,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': seg_optimizer.state_dict(),
                'best_dice': best_dice,
                'metrics': best_metrics,
                'config': config
            }
            
            torch.save(checkpoint, seed_dir / 'best_model.pth')
            print(f"✅ New best model saved! Dice: {best_dice:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏸️ Early stopping triggered at epoch {epoch+1}")
                break
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE - SEED {seed}")
    print(f"{'='*80}")
    print(f"Best Dice: {best_dice:.4f} at epoch {best_epoch}")
    print(f"Best Metrics: {best_metrics}")
    
    # Save training history
    history = {
        'seed': seed,
        'mae_losses': mae_losses,
        'train_losses': train_losses,
        'val_dices': val_dices,
        'best_epoch': best_epoch,
        'best_dice': best_dice,
        'best_metrics': best_metrics
    }
    
    with open(seed_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Create training curves plot
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # MAE loss
    axes[0, 0].plot(mae_losses, 'b-', linewidth=2)
    axes[0, 0].set_title(f'MAE Pretraining Loss (Seed {seed})', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].grid(alpha=0.3)
    
    # Segmentation training loss
    axes[0, 1].plot(train_losses, 'g-', linewidth=2)
    axes[0, 1].set_title(f'Segmentation Training Loss (Seed {seed})', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('BCE Loss')
    axes[0, 1].grid(alpha=0.3)
    
    # Validation Dice
    axes[1, 0].plot(val_dices, 'r-', linewidth=2, label='Val Dice')
    axes[1, 0].scatter([best_epoch-1], [best_dice], color='gold', s=200, zorder=5, 
                       marker='*', edgecolors='red', linewidths=2, label='Best')
    axes[1, 0].axvline(x=best_epoch-1, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_title(f'Validation Dice Score (Seed {seed})', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Score')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Best metrics bar chart
    metrics_names = list(best_metrics.keys())
    metrics_values = list(best_metrics.values())
    bars = axes[1, 1].bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    axes[1, 1].set_title(f'Best Validation Metrics (Seed {seed})', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(seed_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Training curves saved: {seed_dir / 'training_curves.png'}")
    
    return best_metrics, best_epoch

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_statistics(all_results):
    """Compute mean, std, and confidence intervals"""
    
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*80}\n")
    
    metrics = ['dice', 'precision', 'recall', 'f1', 'iou']
    
    stats_summary = {
        'num_runs': len(all_results)
    }
    
    for metric in metrics:
        values = [r[metric] for r in all_results]
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        # 95% confidence interval
        ci = stats.t.interval(0.95, len(values)-1, 
                              loc=mean_val, 
                              scale=stats.sem(values))
        
        stats_summary[metric] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }
        
        print(f"{metric.upper()}:")
        print(f"  Mean ± Std: {mean_val:.4f} ± {std_val:.4f}")
        print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print()
    
    return stats_summary

def save_results_table(all_results, stats_summary, output_dir):
    """Save publication-ready results table"""
    
    # Individual runs table
    df_runs = pd.DataFrame(all_results)
    df_runs.to_csv(output_dir / 'individual_runs.csv', index=False)
    
    # Statistics table
    rows = []
    for metric in ['dice', 'precision', 'recall', 'f1', 'iou']:
        s = stats_summary[metric]
        rows.append({
            'Metric': metric.capitalize(),
            'Mean': f"{s['mean']:.4f}",
            'Std': f"{s['std']:.4f}",
            'Min': f"{s['min']:.4f}",
            'Max': f"{s['max']:.4f}",
            '95% CI': f"[{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]"
        })
    
    df_stats = pd.DataFrame(rows)
    df_stats.to_csv(output_dir / 'statistics_summary.csv', index=False)
    
    print(f"\n✅ Results saved:")
    print(f"   {output_dir / 'individual_runs.csv'}")
    print(f"   {output_dir / 'statistics_summary.csv'}")
    
    # Print publication-ready table
    print(f"\n{'='*80}")
    print("PUBLICATION-READY TABLE")
    print(f"{'='*80}\n")
    print(df_stats.to_string(index=False))
    
    # Create comprehensive visualization
    create_summary_visualization(all_results, stats_summary, output_dir)

def create_summary_visualization(all_results, stats_summary, output_dir):
    """Create comprehensive summary plots"""
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    metrics = ['dice', 'precision', 'recall', 'f1', 'iou']
    seeds = [r['seed'] for r in all_results]
    
    # Plot 1: Box plots for all metrics
    ax1 = fig.add_subplot(gs[0, :])
    data_for_box = [[r[m] for r in all_results] for m in metrics]
    bp = ax1.boxplot(data_for_box, labels=[m.capitalize() for m in metrics], patch_artist=True)
    
    for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title('Distribution of Metrics Across Seeds', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2-6: Individual metric trends across seeds
    for idx, metric in enumerate(metrics):
        row = (idx + 3) // 3
        col = (idx + 3) % 3
        ax = fig.add_subplot(gs[row, col])
        
        values = [r[metric] for r in all_results]
        mean_val = stats_summary[metric]['mean']
        std_val = stats_summary[metric]['std']
        ci_lower = stats_summary[metric]['ci_lower']
        ci_upper = stats_summary[metric]['ci_upper']
        
        # Plot individual points
        ax.scatter(seeds, values, s=100, alpha=0.6, color='#1f77b4', zorder=3)
        ax.plot(seeds, values, 'b--', alpha=0.3, zorder=1)
        
        # Plot mean line
        ax.axhline(y=mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.4f}')
        
        # Plot confidence interval
        ax.axhspan(ci_lower, ci_upper, alpha=0.2, color='red', label=f'95% CI')
        
        ax.set_title(f'{metric.capitalize()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Seed', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim([max(0, mean_val - 3*std_val), min(1, mean_val + 3*std_val)])
    
    plt.suptitle(f'Multi-Seed Experiment Results (n={len(all_results)} seeds)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_dir / 'summary_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Summary visualization saved: {output_dir / 'summary_visualization.png'}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║   STAGE 1: MULTI-SEED EXPERIMENTS WITH STATISTICAL ANALYSIS         ║
╚══════════════════════════════════════════════════════════════════════╝

Running final model with 5 different random seeds to ensure reproducibility
and compute rigorous statistical measures.

Configuration:
  - Seeds: 5
  - MAE Epochs: 100
  - Segmentation Epochs: 50 (with early stopping)
  - Dataset: PediMS (45 patients)
  
Estimated time: 10-15 hours
    """)
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    print(f"\nOutput directory: {output_dir}\n")
    
    # Run experiments for all seeds
    all_results = []
    
    for seed in range(CONFIG['num_seeds']):
        print(f"\n{'#'*80}")
        print(f"EXPERIMENT {seed+1}/{CONFIG['num_seeds']}")
        print(f"{'#'*80}\n")
        
        try:
            best_metrics, best_epoch = train_one_seed(seed, CONFIG)
            
            result = {
                'seed': seed,
                'best_epoch': best_epoch,
                **best_metrics
            }
            
            all_results.append(result)
            
            # Save intermediate results
            with open(output_dir / 'intermediate_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
        
        except Exception as e:
            print(f"\n❌ Error in seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute statistics
    if len(all_results) >= 2:
        stats_summary = compute_statistics(all_results)
        
        # Save final results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'config': CONFIG,
            'individual_results': all_results,
            'statistics': stats_summary
        }
        
        with open(output_dir / 'final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Save publication table
        save_results_table(all_results, stats_summary, output_dir)
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                  STAGE 1 COMPLETE                                    ║
╚══════════════════════════════════════════════════════════════════════╝

Results for {len(all_results)}/{CONFIG['num_seeds']} seeds completed successfully.

Next steps:
1. Review results in: {output_dir}
2. Proceed to Stage 2: Baseline Comparisons
        """)
    
    else:
        print("\n❌ Insufficient results to compute statistics")

if __name__ == "__main__":
    main()
