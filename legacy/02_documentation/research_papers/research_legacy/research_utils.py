"""
Shared Utilities for Research Experiments
==========================================

Common functions used across hyperparameter_sensitivity.py, csrf_variants_analysis.py, etc.
Extracted from final_model.py to ensure consistency with production code.

Author: Research Team
Date: November 2025
"""

import os
import torch
import tempfile
import shutil
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# ===========================================================================================
# ATOMIC SAVE (Prevents Google Drive Sync Corruption)
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
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create temp file in same directory (ensures same filesystem for atomic move)
        temp_dir = os.path.dirname(filepath)
        os.makedirs(temp_dir, exist_ok=True)
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
        print(f"⚠️ Error during atomic save: {e}")
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False

# ===========================================================================================
# COMPREHENSIVE METRICS COMPUTATION
# ===========================================================================================

def compute_metrics(preds, labels):
    """
    Compute comprehensive segmentation metrics.
    
    Args:
        preds: Binary predictions (numpy array, flattened)
        labels: Ground truth labels (numpy array, flattened)
    
    Returns:
        dict: Dictionary containing dice, precision, recall, f1, specificity
    """
    # Ensure inputs are flattened
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    
    # Compute intersection and union for Dice
    intersection = (preds_flat * labels_flat).sum()
    union = preds_flat.sum() + labels_flat.sum()
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    
    # Compute precision, recall, f1 using sklearn
    precision = precision_score(labels_flat, preds_flat, zero_division=0)
    recall = recall_score(labels_flat, preds_flat, zero_division=0)
    f1 = f1_score(labels_flat, preds_flat, zero_division=0)
    
    # Compute specificity (true negative rate)
    tn = ((1 - preds_flat) * (1 - labels_flat)).sum()
    fp = (preds_flat * (1 - labels_flat)).sum()
    specificity = tn / (tn + fp + 1e-7)
    
    return {
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity)
    }

# ===========================================================================================
# VALIDATION WITH COMPREHENSIVE METRICS
# ===========================================================================================

def validate_with_metrics(model, loader, criterion, device):
    """
    Validate model and compute comprehensive metrics.
    
    Args:
        model: PyTorch model
        loader: DataLoader for validation set
        criterion: Loss function
        device: Device (cuda/cpu)
    
    Returns:
        dict: Dictionary with loss, dice, precision, recall, f1, specificity
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Get center slice for 2.5D models
            center_slice_label = labels[:, :, labels.shape[2]//2, :, :]
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, center_slice_label)
            
            # Binary predictions
            pred_binary = (outputs > 0.5).float()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Collect predictions and labels for metrics
            all_preds.append(pred_binary.cpu().numpy().flatten())
            all_labels.append(center_slice_label.cpu().numpy().flatten())
    
    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Compute comprehensive metrics
    metrics = compute_metrics(all_preds, all_labels)
    
    # Add loss to metrics
    metrics['loss'] = total_loss / len(loader)
    
    return metrics

# ===========================================================================================
# PATH UTILITIES
# ===========================================================================================

def get_google_drive_base():
    """
    Get Google Drive base path (cross-platform).
    
    Returns:
        str: Path to Google Drive base directory
    """
    if os.name == 'nt':  # Windows
        possible_paths = [
            r"C:\Users\HP\EDI",
            os.path.join(os.path.expanduser("~"), "Google Drive"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        # Fallback
        return os.path.join(os.path.expanduser("~"), "MyDrive_Local")
    else:  # Linux/Colab
        return "C:/Users/HP/EDI"

def create_experiment_directories(base_name, gdrive_dir_name):
    """
    Create directory structure for experiments.
    
    Args:
        base_name: Base name for local directory (e.g., "hyperparam_sensitivity")
        gdrive_dir_name: Google Drive directory name (e.g., "Research_HyperparamSensitivity")
    
    Returns:
        tuple: (local_dir, gdrive_dir) paths
    """
    # Local directory (in research folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_dir = os.path.join(script_dir, f"{base_name}_results")
    os.makedirs(local_dir, exist_ok=True)
    
    # Google Drive directory (separate from main model)
    # Structure: C:\Users\HP\EDI\NeuroScan_Research\{gdrive_dir_name}
    gdrive_base = get_google_drive_base()
    gdrive_research = os.path.join(gdrive_base, "NeuroScan_Research")
    gdrive_dir = os.path.join(gdrive_research, gdrive_dir_name)
    os.makedirs(gdrive_dir, exist_ok=True)
    
    return local_dir, gdrive_dir

# ===========================================================================================
# LOGGING HELPERS
# ===========================================================================================

def format_time(seconds):
    """Format seconds to human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def print_metrics(metrics, prefix=""):
    """Pretty print metrics dictionary."""
    if prefix:
        print(f"{prefix}:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Dice: {metrics['dice']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
