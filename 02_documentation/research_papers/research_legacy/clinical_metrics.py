"""
Clinical Metrics Suite
Comprehensive clinical evaluation metrics for MS lesion segmentation
"""

import numpy as np
import torch
from scipy.ndimage import label as connected_components
from scipy.spatial.distance import directed_hausdorff
import warnings
warnings.filterwarnings('ignore')


# ===========================================================================================
# LESION-WISE DETECTION METRICS
# ===========================================================================================

def match_lesions(pred_lesions, gt_lesions, iou_threshold=0.1):
    """
    Match predicted lesions to ground truth lesions based on IoU
    
    Args:
        pred_lesions: Labeled prediction (each lesion has unique ID)
        gt_lesions: Labeled ground truth (each lesion has unique ID)
        iou_threshold: Minimum IoU to consider a match
    
    Returns:
        List of (gt_id, pred_id, iou) tuples for matched lesions
    """
    num_gt = pred_lesions.max()
    num_pred = gt_lesions.max()
    
    matches = []
    matched_pred = set()
    
    for gt_id in range(1, num_gt + 1):
        gt_mask = (gt_lesions == gt_id)
        best_iou = 0
        best_pred_id = None
        
        for pred_id in range(1, num_pred + 1):
            if pred_id in matched_pred:
                continue
            
            pred_mask = (pred_lesions == pred_id)
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            
            if union == 0:
                continue
            
            iou = intersection / union
            
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_pred_id = pred_id
        
        if best_pred_id is not None:
            matches.append((gt_id, best_pred_id, best_iou))
            matched_pred.add(best_pred_id)
    
    return matches


def lesion_wise_detection(pred_mask, gt_mask, iou_threshold=0.1, min_lesion_size=3):
    """
    Compute lesion-wise detection metrics
    
    Args:
        pred_mask: Binary prediction mask (numpy array)
        gt_mask: Binary ground truth mask (numpy array)
        iou_threshold: Minimum IoU for matching
        min_lesion_size: Minimum lesion size in voxels
    
    Returns:
        Dict with lesion-wise metrics
    """
    # Connected component labeling
    gt_labeled, num_gt = connected_components(gt_mask)
    pred_labeled, num_pred = connected_components(pred_mask)
    
    # Filter small lesions
    gt_sizes = np.bincount(gt_labeled.ravel())
    pred_sizes = np.bincount(pred_labeled.ravel())
    
    # Remove small GT lesions
    small_gt = np.where(gt_sizes < min_lesion_size)[0]
    for lesion_id in small_gt:
        if lesion_id > 0:  # Skip background
            gt_labeled[gt_labeled == lesion_id] = 0
    
    # Remove small predicted lesions
    small_pred = np.where(pred_sizes < min_lesion_size)[0]
    for lesion_id in small_pred:
        if lesion_id > 0:
            pred_labeled[pred_labeled == lesion_id] = 0
    
    # Relabel after filtering
    gt_labeled, num_gt = connected_components(gt_labeled > 0)
    pred_labeled, num_pred = connected_components(pred_labeled > 0)
    
    # Match lesions
    matches = match_lesions(pred_labeled, gt_labeled, iou_threshold)
    
    tp = len(matches)
    fp = num_pred - tp
    fn = num_gt - tp
    
    # Metrics
    lesion_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    lesion_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    lesion_f1 = 2 * lesion_recall * lesion_precision / (lesion_recall + lesion_precision) if (lesion_recall + lesion_precision) > 0 else 0
    
    return {
        'num_gt_lesions': num_gt,
        'num_pred_lesions': num_pred,
        'lesion_tp': tp,
        'lesion_fp': fp,
        'lesion_fn': fn,
        'lesion_recall': lesion_recall,
        'lesion_precision': lesion_precision,
        'lesion_f1': lesion_f1,
        'lesion_fnr': fn / num_gt if num_gt > 0 else 0
    }


def small_lesion_sensitivity(pred_mask, gt_mask, size_threshold=10, min_size=3):
    """
    Compute sensitivity specifically for small lesions
    
    Args:
        pred_mask: Binary prediction mask
        gt_mask: Binary ground truth mask
        size_threshold: Maximum size (voxels) to consider as "small"
        min_size: Minimum lesion size to consider
    
    Returns:
        Dict with small lesion metrics
    """
    # Label GT lesions
    gt_labeled, num_gt = connected_components(gt_mask)
    
    # Find small lesions
    small_lesion_ids = []
    for lesion_id in range(1, num_gt + 1):
        size = (gt_labeled == lesion_id).sum()
        if min_size <= size <= size_threshold:
            small_lesion_ids.append(lesion_id)
    
    if len(small_lesion_ids) == 0:
        return {
            'num_small_lesions': 0,
            'num_detected_small': 0,
            'small_lesion_sensitivity': None,
            'small_lesion_fnr': None
        }
    
    # Check detection
    detected = 0
    for lesion_id in small_lesion_ids:
        lesion_mask = (gt_labeled == lesion_id)
        overlap = np.logical_and(pred_mask, lesion_mask).sum()
        if overlap > 0:
            detected += 1
    
    sensitivity = detected / len(small_lesion_ids)
    fnr = 1.0 - sensitivity
    
    return {
        'num_small_lesions': len(small_lesion_ids),
        'num_detected_small': detected,
        'small_lesion_sensitivity': sensitivity,
        'small_lesion_fnr': fnr
    }


# ===========================================================================================
# FALSE NEGATIVE RATES
# ===========================================================================================

def compute_false_negative_rates(pred_mask, gt_mask):
    """
    Compute voxel-wise and lesion-wise false negative rates
    
    Args:
        pred_mask: Binary prediction mask
        gt_mask: Binary ground truth mask
    
    Returns:
        Dict with FNR metrics
    """
    # Voxel-wise FNR
    gt_positive = gt_mask.sum()
    false_negatives = np.logical_and(gt_mask, ~pred_mask).sum()
    fnr_voxel = false_negatives / gt_positive if gt_positive > 0 else 0
    
    # Lesion-wise FNR
    lesion_metrics = lesion_wise_detection(pred_mask, gt_mask)
    fnr_lesion = lesion_metrics['lesion_fnr']
    
    return {
        'fnr_voxel': fnr_voxel,
        'fnr_lesion': fnr_lesion,
        'false_negatives': int(false_negatives),
        'gt_positive_voxels': int(gt_positive)
    }


# ===========================================================================================
# CALIBRATION ANALYSIS
# ===========================================================================================

def compute_ece(pred_probs, gt_mask, n_bins=10):
    """
    Compute Expected Calibration Error (ECE)
    
    Args:
        pred_probs: Predicted probabilities (numpy array, 0-1)
        gt_mask: Binary ground truth mask
        n_bins: Number of bins for calibration
    
    Returns:
        Dict with ECE and bin statistics
    """
    # Flatten arrays
    pred_probs_flat = pred_probs.ravel()
    gt_flat = gt_mask.ravel().astype(float)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_stats = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = np.logical_and(pred_probs_flat >= bin_lower, pred_probs_flat < bin_upper)
        
        if bin_upper == 1.0:  # Include upper boundary for last bin
            in_bin = np.logical_and(pred_probs_flat >= bin_lower, pred_probs_flat <= bin_upper)
        
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = gt_flat[in_bin].mean()
            avg_confidence_in_bin = pred_probs_flat[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_stats.append({
                'bin_lower': float(bin_lower),
                'bin_upper': float(bin_upper),
                'confidence': float(avg_confidence_in_bin),
                'accuracy': float(accuracy_in_bin),
                'proportion': float(prop_in_bin),
                'count': int(in_bin.sum())
            })
        else:
            bin_stats.append({
                'bin_lower': float(bin_lower),
                'bin_upper': float(bin_upper),
                'confidence': 0.0,
                'accuracy': 0.0,
                'proportion': 0.0,
                'count': 0
            })
    
    # Calibration interpretation
    if ece < 0.05:
        interpretation = "Excellent calibration"
    elif ece < 0.10:
        interpretation = "Good calibration"
    elif ece < 0.15:
        interpretation = "Moderate calibration"
    else:
        interpretation = "Poor calibration"
    
    return {
        'ece': float(ece),
        'n_bins': n_bins,
        'interpretation': interpretation,
        'bin_statistics': bin_stats
    }


def reliability_diagram_data(pred_probs, gt_mask, n_bins=10):
    """
    Generate data for reliability diagram (calibration plot)
    
    Args:
        pred_probs: Predicted probabilities
        gt_mask: Binary ground truth
        n_bins: Number of bins
    
    Returns:
        Dict with bin confidences and accuracies for plotting
    """
    ece_result = compute_ece(pred_probs, gt_mask, n_bins)
    
    confidences = [b['confidence'] for b in ece_result['bin_statistics'] if b['count'] > 0]
    accuracies = [b['accuracy'] for b in ece_result['bin_statistics'] if b['count'] > 0]
    counts = [b['count'] for b in ece_result['bin_statistics'] if b['count'] > 0]
    
    return {
        'confidences': confidences,
        'accuracies': accuracies,
        'counts': counts,
        'ece': ece_result['ece']
    }


# ===========================================================================================
# UNCERTAINTY ESTIMATION
# ===========================================================================================

def monte_carlo_dropout_uncertainty(model, input_tensor, num_samples=30, dropout_rate=0.1):
    """
    Estimate prediction uncertainty using Monte Carlo Dropout
    
    Args:
        model: PyTorch model with dropout layers
        input_tensor: Input tensor (B, C, D, H, W)
        num_samples: Number of MC samples
        dropout_rate: Dropout probability
    
    Returns:
        Dict with mean prediction, std (uncertainty), and entropy
    """
    model.train()  # Enable dropout
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(input_tensor)
            predictions.append(pred.cpu().numpy())
    
    predictions = np.array(predictions)  # (num_samples, B, C, H, W)
    
    # Mean prediction
    mean_pred = predictions.mean(axis=0)
    
    # Standard deviation (uncertainty)
    std_pred = predictions.std(axis=0)
    
    # Predictive entropy
    # H = -sum(p * log(p))
    mean_prob = mean_pred
    epsilon = 1e-10
    entropy = -mean_prob * np.log(mean_prob + epsilon) - (1 - mean_prob) * np.log(1 - mean_prob + epsilon)
    
    # Uncertainty statistics
    uncertainty_stats = {
        'mean_uncertainty': float(std_pred.mean()),
        'max_uncertainty': float(std_pred.max()),
        'min_uncertainty': float(std_pred.min()),
        'high_uncertainty_percent': float((std_pred > 0.2).sum() / std_pred.size * 100),
        'low_uncertainty_percent': float((std_pred < 0.05).sum() / std_pred.size * 100)
    }
    
    return {
        'mean_prediction': mean_pred,
        'uncertainty_map': std_pred,
        'entropy_map': entropy,
        'statistics': uncertainty_stats
    }


# ===========================================================================================
# COMPREHENSIVE CLINICAL EVALUATION
# ===========================================================================================

def comprehensive_clinical_evaluation(pred_mask, gt_mask, pred_probs=None, model=None, input_tensor=None):
    """
    Run complete clinical metrics suite
    
    Args:
        pred_mask: Binary prediction mask (numpy array)
        gt_mask: Binary ground truth mask (numpy array)
        pred_probs: Predicted probabilities (optional, for calibration)
        model: PyTorch model (optional, for uncertainty)
        input_tensor: Input tensor (optional, for uncertainty)
    
    Returns:
        Dict with all clinical metrics
    """
    
    print("="*100)
    print("COMPREHENSIVE CLINICAL EVALUATION")
    print("="*100)
    
    results = {}
    
    # 1. Lesion-wise detection
    print("\n1. Computing lesion-wise detection metrics...")
    lesion_metrics = lesion_wise_detection(pred_mask, gt_mask)
    results['lesion_detection'] = lesion_metrics
    
    print(f"   Total GT lesions: {lesion_metrics['num_gt_lesions']}")
    print(f"   Detected (TP): {lesion_metrics['lesion_tp']}")
    print(f"   False Positives: {lesion_metrics['lesion_fp']}")
    print(f"   False Negatives: {lesion_metrics['lesion_fn']}")
    print(f"   Lesion Recall: {lesion_metrics['lesion_recall']:.4f}")
    print(f"   Lesion Precision: {lesion_metrics['lesion_precision']:.4f}")
    print(f"   Lesion F1: {lesion_metrics['lesion_f1']:.4f}")
    
    # 2. Small lesion sensitivity
    print("\n2. Computing small lesion sensitivity...")
    small_lesion_metrics = small_lesion_sensitivity(pred_mask, gt_mask, size_threshold=10)
    results['small_lesion'] = small_lesion_metrics
    
    if small_lesion_metrics['small_lesion_sensitivity'] is not None:
        print(f"   Small lesions (<10 voxels): {small_lesion_metrics['num_small_lesions']}")
        print(f"   Detected: {small_lesion_metrics['num_detected_small']}")
        print(f"   Sensitivity: {small_lesion_metrics['small_lesion_sensitivity']:.4f}")
    else:
        print(f"   No small lesions found in ground truth")
    
    # 3. False negative rates
    print("\n3. Computing false negative rates...")
    fnr_metrics = compute_false_negative_rates(pred_mask, gt_mask)
    results['false_negative_rates'] = fnr_metrics
    
    print(f"   FNR (voxel-wise): {fnr_metrics['fnr_voxel']:.4f}")
    print(f"   FNR (lesion-wise): {fnr_metrics['fnr_lesion']:.4f}")
    
    # 4. Calibration analysis
    if pred_probs is not None:
        print("\n4. Computing calibration metrics...")
        ece_metrics = compute_ece(pred_probs, gt_mask, n_bins=10)
        results['calibration'] = ece_metrics
        
        print(f"   ECE: {ece_metrics['ece']:.4f}")
        print(f"   Interpretation: {ece_metrics['interpretation']}")
    else:
        print("\n4. Skipping calibration (no probabilities provided)")
        results['calibration'] = None
    
    # 5. Uncertainty estimation
    if model is not None and input_tensor is not None:
        print("\n5. Computing uncertainty estimates...")
        try:
            uncertainty_metrics = monte_carlo_dropout_uncertainty(model, input_tensor, num_samples=30)
            results['uncertainty'] = uncertainty_metrics['statistics']
            
            print(f"   Mean uncertainty: {uncertainty_metrics['statistics']['mean_uncertainty']:.4f}")
            print(f"   High uncertainty regions (>0.2): {uncertainty_metrics['statistics']['high_uncertainty_percent']:.2f}%")
            print(f"   Low uncertainty regions (<0.05): {uncertainty_metrics['statistics']['low_uncertainty_percent']:.2f}%")
        except Exception as e:
            print(f"   Error computing uncertainty: {e}")
            results['uncertainty'] = None
    else:
        print("\n5. Skipping uncertainty (no model provided)")
        results['uncertainty'] = None
    
    print("\n" + "="*100)
    print("CLINICAL EVALUATION COMPLETE")
    print("="*100)
    
    return results


def save_clinical_metrics(results, output_path='research/clinical_metrics.json'):
    """Save clinical metrics to JSON file"""
    import json
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved clinical metrics to: {output_path}")


def demo_clinical_metrics():
    """Demo with synthetic data"""
    
    print("="*100)
    print("CLINICAL METRICS SUITE - DEMO")
    print("="*100)
    print("\nGenerating synthetic data for demonstration...")
    
    # Create synthetic 3D data
    shape = (128, 128, 64)
    
    # Ground truth: 10 lesions of various sizes
    gt_mask = np.zeros(shape, dtype=bool)
    
    # Add lesions
    np.random.seed(42)
    for i in range(10):
        center = (np.random.randint(20, 108), np.random.randint(20, 108), np.random.randint(10, 54))
        radius = np.random.randint(3, 8)
        
        # Create spherical lesion
        y, x, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2
        gt_mask |= mask
    
    # Prediction: 90% sensitivity, some false positives
    pred_mask = gt_mask.copy()
    
    # Add some false negatives (miss 10% of voxels)
    false_neg_mask = np.random.rand(*shape) < 0.1
    pred_mask = pred_mask & ~false_neg_mask
    
    # Add some false positives
    false_pos_mask = np.random.rand(*shape) < 0.02
    pred_mask = pred_mask | false_pos_mask
    
    # Generate predicted probabilities
    pred_probs = pred_mask.astype(float)
    pred_probs += np.random.rand(*shape) * 0.1  # Add noise
    pred_probs = np.clip(pred_probs, 0, 1)
    
    print(f"Ground truth: {gt_mask.sum()} positive voxels")
    print(f"Prediction: {pred_mask.sum()} positive voxels")
    
    # Run comprehensive evaluation
    results = comprehensive_clinical_evaluation(
        pred_mask=pred_mask,
        gt_mask=gt_mask,
        pred_probs=pred_probs,
        model=None,  # Skip uncertainty for demo
        input_tensor=None
    )
    
    # Save results
    save_clinical_metrics(results, 'research/clinical_metrics_demo.json')
    
    return results


if __name__ == "__main__":
    demo_clinical_metrics()
