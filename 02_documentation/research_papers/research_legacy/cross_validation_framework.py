"""
5-Fold Patient-Wise Cross-Validation Framework
Implements stratified patient-level splits with statistical analysis
"""

import json
import numpy as np
import scipy.stats as stats
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_patient_id(case_path):
    """Extract patient ID from file path"""
    case_str = str(case_path)
    # Extract patient ID (e.g., P1, P2, P03, etc.)
    parts = Path(case_str).parts
    for part in parts:
        if part.startswith('P') and part[1:].isdigit():
            return part
    return None


def compute_lesion_load(patient_slices):
    """
    Compute lesion load for a patient
    This is a placeholder - you would load actual masks and compute
    """
    # In real implementation, load ground truth and compute total lesion volume
    # For now, return a dummy value
    return len(patient_slices)  # Number of slices as proxy


def create_patient_wise_folds(data_dicts, n_folds=5, stratify=True, random_seed=42):
    """
    Create patient-wise stratified folds
    
    Args:
        data_dicts: List of data dictionaries with 'image' and 'label' keys
        n_folds: Number of folds
        stratify: Whether to stratify by lesion load
        random_seed: Random seed for reproducibility
    
    Returns:
        List of (train_data, val_data) tuples for each fold
    """
    
    # Group data by patient ID
    patients = {}
    for item in data_dicts:
        pid = extract_patient_id(item.get('case', item['image']))
        if pid is None:
            print(f"Warning: Could not extract patient ID from {item['image']}")
            continue
        
        if pid not in patients:
            patients[pid] = []
        patients[pid].append(item)
    
    patient_ids = sorted(list(patients.keys()))
    print(f"\nFound {len(patient_ids)} unique patients: {patient_ids}")
    
    # Compute lesion loads for stratification
    if stratify:
        lesion_loads = [compute_lesion_load(patients[pid]) for pid in patient_ids]
        
        # Bin into tertiles for stratification
        if len(set(lesion_loads)) > 1:
            bins = np.percentile(lesion_loads, [33, 67])
            strata = np.digitize(lesion_loads, bins)
            print(f"Stratifying by lesion load (tertiles): {np.bincount(strata)}")
            
            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
            splits = list(kfold.split(patient_ids, strata))
        else:
            print("Warning: All patients have same lesion load, using standard KFold")
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
            splits = list(kfold.split(patient_ids))
    else:
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        splits = list(kfold.split(patient_ids))
    
    # Generate folds
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_pids = [patient_ids[i] for i in train_idx]
        val_pids = [patient_ids[i] for i in val_idx]
        
        train_data = []
        for pid in train_pids:
            train_data.extend(patients[pid])
        
        val_data = []
        for pid in val_pids:
            val_data.extend(patients[pid])
        
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train patients ({len(train_pids)}): {train_pids}")
        print(f"  Val patients ({len(val_pids)}): {val_pids}")
        print(f"  Train samples: {len(train_data)}")
        print(f"  Val samples: {len(val_data)}")
        
        folds.append({
            'fold': fold_idx + 1,
            'train_patients': train_pids,
            'val_patients': val_pids,
            'train_data': train_data,
            'val_data': val_data
        })
    
    return folds


def compute_confidence_interval(values, confidence=0.95):
    """
    Compute confidence interval using t-distribution
    
    Args:
        values: Array of values
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        (mean, (lower_bound, upper_bound))
    """
    n = len(values)
    mean = np.mean(values)
    std_err = stats.sem(values)
    
    if n > 1:
        ci = stats.t.interval(confidence, n-1, loc=mean, scale=std_err)
    else:
        ci = (mean, mean)
    
    return mean, ci


def aggregate_cv_results(fold_results):
    """
    Aggregate cross-validation results across folds
    
    Args:
        fold_results: List of dicts with metrics per fold
    
    Returns:
        Dict with aggregated statistics
    """
    
    metrics = ['dice', 'precision', 'recall', 'f1', 'specificity', 'hd95']
    
    aggregated = {}
    
    for metric in metrics:
        if metric in fold_results[0]:
            values = [r[metric] for r in fold_results]
            mean, ci = compute_confidence_interval(values)
            
            aggregated[metric] = {
                'values': values,
                'mean': mean,
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'ci_95': ci
            }
    
    return aggregated


def print_cv_results(fold_results, aggregated):
    """Print formatted cross-validation results"""
    
    print("\n" + "="*100)
    print("5-FOLD PATIENT-WISE CROSS-VALIDATION RESULTS")
    print("="*100)
    
    # Per-fold results
    print("\nPER-FOLD RESULTS:")
    print("-"*100)
    print(f"{'Fold':<6} {'Dice':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'Specificity':<12} {'HD95':<10}")
    print("-"*100)
    
    for result in fold_results:
        fold = result['fold']
        dice = result.get('dice', 0)
        prec = result.get('precision', 0)
        rec = result.get('recall', 0)
        f1 = result.get('f1', 0)
        spec = result.get('specificity', 0)
        hd95 = result.get('hd95', 0)
        
        print(f"{fold:<6} {dice:<10.4f} {prec:<12.4f} {rec:<10.4f} {f1:<10.4f} {spec:<12.4f} {hd95:<10.2f}")
    
    print("-"*100)
    
    # Aggregated results
    print("\nAGGREGATED RESULTS (Mean ± Std [95% CI]):")
    print("-"*100)
    
    for metric_name, stats_dict in aggregated.items():
        mean = stats_dict['mean']
        std = stats_dict['std']
        ci_lower, ci_upper = stats_dict['ci_95']
        
        print(f"{metric_name.capitalize():<15}: {mean:.4f} ± {std:.4f}  [95% CI: {ci_lower:.4f}, {ci_upper:.4f}]")
    
    print("-"*100)
    print()


def save_cv_results(fold_results, aggregated, output_path):
    """Save cross-validation results to JSON"""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results = {
        'fold_results': fold_results,
        'aggregated': {
            metric: {
                'mean': convert_numpy(stats['mean']),
                'std': convert_numpy(stats['std']),
                'median': convert_numpy(stats['median']),
                'min': convert_numpy(stats['min']),
                'max': convert_numpy(stats['max']),
                'ci_95_lower': convert_numpy(stats['ci_95'][0]),
                'ci_95_upper': convert_numpy(stats['ci_95'][1]),
                'values': [convert_numpy(v) for v in stats['values']]
            }
            for metric, stats in aggregated.items()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved cross-validation results to: {output_path}")
    
    # Also save summary text
    summary_path = output_path.parent / 'cv_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("5-FOLD PATIENT-WISE CROSS-VALIDATION SUMMARY\n")
        f.write("="*100 + "\n\n")
        
        f.write("AGGREGATED RESULTS (Mean ± Std [95% CI]):\n")
        f.write("-"*100 + "\n")
        
        for metric_name, stats_dict in aggregated.items():
            mean = stats_dict['mean']
            std = stats_dict['std']
            ci_lower, ci_upper = stats_dict['ci_95']
            f.write(f"{metric_name.capitalize():<15}: {mean:.4f} ± {std:.4f}  [95% CI: {ci_lower:.4f}, {ci_upper:.4f}]\n")
        
        f.write("\n")
        f.write("PER-FOLD RESULTS:\n")
        f.write("-"*100 + "\n")
        
        for result in fold_results:
            f.write(f"\nFold {result['fold']}:\n")
            for metric in ['dice', 'precision', 'recall', 'f1', 'specificity', 'hd95']:
                if metric in result:
                    f.write(f"  {metric.capitalize():<15}: {result[metric]:.4f}\n")
    
    print(f"✓ Saved summary to: {summary_path}")


def train_fold(fold_data, fold_idx, config):
    """
    Train model on one fold
    
    This is a placeholder - you would implement actual training here
    
    Args:
        fold_data: Dict with 'train_data', 'val_data', etc.
        fold_idx: Fold number
        config: Training configuration
    
    Returns:
        Dict with validation metrics
    """
    
    print(f"\n{'='*100}")
    print(f"TRAINING FOLD {fold_idx}")
    print(f"{'='*100}")
    
    train_data = fold_data['train_data']
    val_data = fold_data['val_data']
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Train patients: {fold_data['train_patients']}")
    print(f"Val patients: {fold_data['val_patients']}")
    
    # TODO: Implement actual training
    # For now, return dummy metrics
    
    print("\n⚠️  This is a placeholder. Implement actual training in train_fold()")
    print("You would:")
    print("  1. Create dataloaders from train_data and val_data")
    print("  2. Initialize model with random seed")
    print("  3. Train for N epochs")
    print("  4. Evaluate on validation set")
    print("  5. Return validation metrics")
    
    # Dummy results
    results = {
        'fold': fold_idx,
        'dice': np.random.uniform(0.82, 0.85),
        'precision': np.random.uniform(0.75, 0.80),
        'recall': np.random.uniform(0.90, 0.93),
        'f1': np.random.uniform(0.82, 0.85),
        'specificity': np.random.uniform(0.994, 0.996),
        'hd95': np.random.uniform(7.5, 9.5),
    }
    
    return results


def main():
    """Run 5-fold cross-validation"""
    
    print("="*100)
    print("5-FOLD PATIENT-WISE CROSS-VALIDATION")
    print("="*100)
    
    # Load data paths
    # This is a placeholder - replace with actual data loading
    print("\n⚠️  Placeholder: Load your data dictionaries here")
    print("Expected format:")
    print("  data_dicts = [")
    print("    {'image': 'path/to/P1/FLAIR.nii.gz', 'label': 'path/to/P1/GT.nii', 'case': 'P1'},")
    print("    {'image': 'path/to/P2/FLAIR.nii.gz', 'label': 'path/to/P2/GT.nii', 'case': 'P2'},")
    print("    ...")
    print("  ]")
    
    # Create dummy data for demonstration
    data_dicts = []
    for i in range(1, 10):  # 9 patients
        patient_id = f"P{i}"
        data_dicts.append({
            'image': f'C:/Users/HP/EDI/Dataset/PediMS/PediMS/{patient_id}/T2/processed/FLAIR.nii.gz',
            'label': f'C:/Users/HP/EDI/Dataset/PediMS/PediMS/{patient_id}/T2/processed/GT.nii',
            'case': patient_id
        })
    
    # Create folds
    folds = create_patient_wise_folds(
        data_dicts,
        n_folds=5,
        stratify=True,
        random_seed=42
    )
    
    # Train each fold
    fold_results = []
    for fold_data in folds:
        results = train_fold(
            fold_data,
            fold_idx=fold_data['fold'],
            config={'epochs': 100, 'lr': 1e-4}
        )
        fold_results.append(results)
    
    # Aggregate results
    aggregated = aggregate_cv_results(fold_results)
    
    # Print results
    print_cv_results(fold_results, aggregated)
    
    # Save results
    save_cv_results(
        fold_results,
        aggregated,
        output_path='research/cv_results.json'
    )
    
    print("\n" + "="*100)
    print("CROSS-VALIDATION COMPLETE")
    print("="*100)
    print("\nNEXT STEPS:")
    print("1. Implement actual training in train_fold() function")
    print("2. Load real data paths in main()")
    print("3. Configure training parameters (epochs, lr, etc.)")
    print("4. Run full cross-validation (will take several hours)")
    print("="*100)


if __name__ == "__main__":
    main()
